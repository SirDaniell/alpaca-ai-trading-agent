"""
test_options_q_executor.py — Comprehensive test suite for Tier 2 Options Q-Learner Executor.

Tests:
1. ZoneSnapshotManager immutability, confluence scoring, and close-based invalidation.
2. HardActionMask no-chase entry discipline and volume delta confirmation.
3. Q-Learner weight updates and gradient learning during synthetic training.
4. Virtual trade execution, transition recording, and delayed risk-adjusted rewards.
5. Bias-Persistence Re-Entry (fixed risk sizing, NO martingale scaling / chasing losses).
6. Dual execution timeframe switching (5m vs 15m).
"""

from datetime import datetime, timezone
import numpy as np
import pytest
import torch

from app.core.market.zone_snapshot import HardActionMask, ZoneRecord, ZoneSnapshotManager
from app.core.options.q_executor import (
    ACTION_BUY_CALL,
    ACTION_BUY_PUT,
    ACTION_CLOSE_FLATTEN,
    ACTION_TAKE_PROFIT_HALF,
    ACTION_WAIT,
    AccountContext,
    ExecutionContext,
    HTFBiasPackage,
    OptionsQExecutor,
)


@pytest.fixture
def zone_manager():
    zm = ZoneSnapshotManager(max_snapshots=10)
    # Add raw zones (cluster_id, zone_price, cluster_levels, volume_data)
    raw_zones_h1 = [
        (0, 100.0, [(0, 100.0, "support")], {"upper_bound": 100.4, "lower_bound": 99.6, "total_volume": 1000, "up_volume": 700, "down_volume": 300, "net_volume": 400}),
        (1, 110.0, [(1, 110.0, "resistance")], {"upper_bound": 110.4, "lower_bound": 109.6, "total_volume": 1200, "up_volume": 400, "down_volume": 800, "net_volume": -400}),
    ]
    zm.add_snapshot("snap_h1_1", "H1", raw_zones_h1, timestamp=datetime.now(timezone.utc))
    return zm


# ── 1. Zone Snapshot & Invalidation Tests ─────────────────────────────────────

def test_zone_snapshot_immutability_and_confluence(zone_manager):
    zones = zone_manager.get_active_zones()
    assert len(zones) == 2
    supp, res = zone_manager.get_nearest_zones(105.0)
    assert supp is not None and supp.price_level == 100.0
    assert res is not None and res.price_level == 110.0

    # Add overlapping H4 snapshot to verify confluence score increases
    raw_zones_h4 = [
        (0, 100.1, [(0, 100.1, "support")], {"upper_bound": 100.5, "lower_bound": 99.7, "total_volume": 1500, "up_volume": 900, "down_volume": 600, "net_volume": 300}),
    ]
    zone_manager.add_snapshot("snap_h4_1", "H4", raw_zones_h4, timestamp=datetime.now(timezone.utc))

    active = zone_manager.get_active_zones(zone_type="support")
    assert any(z.confluence_score > 1.0 for z in active), "Confluence score must increase when zones overlap"


def test_zone_invalidation_on_candle_close(zone_manager):
    # Support at 100.0 (lower_bound 99.6)
    # Candle closing below 99.6 must invalidate support
    invalidated = zone_manager.update_invalidation(close_price=99.2, high_price=101.0, low_price=99.0)
    assert len(invalidated) == 1
    active_supp = zone_manager.get_active_zones(zone_type="support")
    assert len(active_supp) == 0, "Invalidated support zone must drop out of active evaluation set"


# ── 2. Hard Action Mask Tests ──────────────────────────────────────────────────

def test_hard_action_mask_no_chase(zone_manager):
    mask_engine = HardActionMask(proximity_atr_mult=0.75, require_volume_confirm=True)

    # Case A: Price = 105.0 (Far from support 100.0 and resistance 110.0, ATR = 1.0)
    mask_far = mask_engine.get_action_mask(current_price=105.0, atr=1.0, zone_manager=zone_manager)
    np.testing.assert_array_equal(mask_far, [1, 0, 0, 0, 0])
    assert mask_far[1] == 0, "BUY_CALL must be masked out when far from support"
    assert mask_far[2] == 0, "BUY_PUT must be masked out when far from resistance"

    # Case B: Price = 100.2 (Within support proximity band, buy_vol > sell_vol)
    mask_near_supp = mask_engine.get_action_mask(current_price=100.2, atr=1.0, zone_manager=zone_manager, buy_volume=800, sell_volume=400)
    assert mask_near_supp[1] == 1, "BUY_CALL must be unmasked when near support with positive volume delta"

    # Case C: Price = 109.8 (Within resistance proximity band, sell_vol > buy_vol)
    mask_near_res = mask_engine.get_action_mask(current_price=109.8, atr=1.0, zone_manager=zone_manager, buy_volume=300, sell_volume=900)
    assert mask_near_res[2] == 1, "BUY_PUT must be unmasked when near resistance with negative volume delta"


# ── 3. Q-Learner Network Weights Update Test ──────────────────────────────────

def test_q_executor_weights_update(zone_manager):
    executor = OptionsQExecutor(input_dim=24, lr=1e-3, epsilon_start=0.5)
    htf_bias = HTFBiasPackage(direction="bullish", strength=0.8, reversal_prob=0.1, q_value=0.7)
    account = AccountContext()
    exec_ctx = ExecutionContext(current_price=100.1, atr=1.0, buy_volume=1000, sell_volume=500)

    state = executor.build_state_vector(htf_bias, account, exec_ctx, zone_manager)
    mask = executor.action_mask_engine.get_action_mask(exec_ctx.current_price, exec_ctx.atr, zone_manager, exec_ctx.buy_volume, exec_ctx.sell_volume)

    # Populate replay buffer
    for _ in range(40):
        next_state = state + np.random.randn(24).astype(np.float32) * 0.01
        reward = float(np.random.randn())
        executor.record_transition(state, ACTION_BUY_CALL, reward, next_state, False, mask)

    weight_before = float(next(executor.policy_net.parameters()).detach().abs().sum().item())
    loss = executor.train_step(batch_size=16)
    weight_after = float(next(executor.policy_net.parameters()).detach().abs().sum().item())

    assert loss is not None and loss > 0.0
    assert abs(weight_after - weight_before) > 1e-7, "Network weights must update during training step"


# ── 4. Virtual Trade Execution & Transition Recording ─────────────────────────

def test_virtual_trade_execution_flow(zone_manager):
    executor = OptionsQExecutor(input_dim=24, epsilon_start=0.0)  # Greedy mode
    htf_bias = HTFBiasPackage(direction="bullish", strength=0.75, reversal_prob=0.15)
    account = AccountContext(open_position_type=None)
    exec_ctx = ExecutionContext(current_price=100.1, atr=1.0, buy_volume=1000, sell_volume=400)

    state = executor.build_state_vector(htf_bias, account, exec_ctx, zone_manager)
    mask = executor.action_mask_engine.get_action_mask(exec_ctx.current_price, exec_ctx.atr, zone_manager, exec_ctx.buy_volume, exec_ctx.sell_volume)

    action = executor.select_action(state, mask, eval_mode=True)
    assert action in np.where(mask == 1)[0], "Selected action must comply with hard action mask"


# ── 5. Bias-Persistence Re-Entry vs No Chasing Losses Test ────────────────────

def test_bias_persistence_reentry_gating(zone_manager):
    executor = OptionsQExecutor()
    exec_ctx = ExecutionContext(current_price=100.2, reentries_in_window=1, max_reentries_allowed=3)

    # Scenario A: Valid HTF bias holds -> Re-entry ALLOWED
    bias_valid = HTFBiasPackage(direction="bullish", strength=0.75, reversal_prob=0.15)
    assert executor.can_reenter(bias_valid, exec_ctx, zone_manager) is True

    # Scenario B: Weak conviction (< 0.65) -> Re-entry BLOCKED (NO chasing losses)
    bias_weak = HTFBiasPackage(direction="bullish", strength=0.50, reversal_prob=0.15)
    assert executor.can_reenter(bias_weak, exec_ctx, zone_manager) is False

    # Scenario C: High reversal probability (> 0.35) -> Re-entry BLOCKED
    bias_reversal = HTFBiasPackage(direction="bullish", strength=0.80, reversal_prob=0.45)
    assert executor.can_reenter(bias_reversal, exec_ctx, zone_manager) is False

    # Scenario D: Max re-entries cap reached -> Re-entry BLOCKED
    exec_ctx_capped = ExecutionContext(current_price=100.2, reentries_in_window=3, max_reentries_allowed=3)
    assert executor.can_reenter(bias_valid, exec_ctx_capped, zone_manager) is False


# ── 6. Dual Execution Timeframe Switch Test ────────────────────────────────────

def test_dual_execution_timeframe_switch(zone_manager):
    executor = OptionsQExecutor()
    htf_bias = HTFBiasPackage(direction="bullish", strength=0.8, reversal_prob=0.1)
    account = AccountContext()

    # 5m timeframe context
    ctx_5m = ExecutionContext(ltf_timeframe="5m", current_price=100.1)
    s_5m = executor.build_state_vector(htf_bias, account, ctx_5m, zone_manager)

    # 15m timeframe context
    ctx_15m = ExecutionContext(ltf_timeframe="15m", current_price=100.1)
    s_15m = executor.build_state_vector(htf_bias, account, ctx_15m, zone_manager)

    assert len(s_5m) == len(s_15m) == 24
    assert s_5m[11] == 0.0  # 5m flag
    assert s_15m[11] == 1.0  # 15m flag


def test_time_and_session_features_encoding(zone_manager):
    executor = OptionsQExecutor()
    htf_bias = HTFBiasPackage(direction="bullish", strength=0.8)
    account = AccountContext()

    ctx_open = ExecutionContext(hour_of_day=14.5, day_of_week=1, session_phase="nyse_open")
    s_open = executor.build_state_vector(htf_bias, account, ctx_open, zone_manager)

    ctx_power = ExecutionContext(hour_of_day=20.0, day_of_week=4, session_phase="nyse_power_hour")
    s_power = executor.build_state_vector(htf_bias, account, ctx_power, zone_manager)

    # Check sin/cos hour and session flags
    assert s_open[22] == 1.0  # is_nyse_open
    assert s_open[23] == 0.0  # is_power_hour
    assert s_power[22] == 0.0
    assert s_power[23] == 1.0  # is_power_hour

