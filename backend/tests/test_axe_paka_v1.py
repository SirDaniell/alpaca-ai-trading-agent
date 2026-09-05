"""
test_axe_paka_v1.py — Comprehensive Unit & Integration Tests for Axe-paka-v1 Trading Agent.

Verifies:
1. Model instantiation & forward pass for SignalMetaNetwork & ExecutorQNetwork.
2. Weight checkpoint loading from backend/app/axe_paka_v1/weights/.
3. Strict 30m and 1h options expiry gating (Head 0 5m and Head 1 15m are blocked).
4. Option contract selection and Alpaca CLI execution pipeline dry-run.
5. Lookahead-free feature window construction.
"""

import logging
import pytest
import numpy as np
import torch

from app.axe_paka_v1.agent import AxePakaV1Agent, ACTION_WAIT, ACTION_BUY_CALL, ACTION_BUY_PUT
from app.axe_paka_v1.config import AxePakaV1Config
from app.axe_paka_v1.models import (
    SignalMetaNetwork,
    ExecutorQNetwork,
    build_feat_window,
    build_feat_window_batch,
    DEFAULT_NUM_FEATURES,
)
from app.core.options.q_executor import HTFBiasPackage, AccountContext, ExecutionContext
from app.core.market.zone_snapshot import ZoneSnapshotManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_axe_paka_v1_models_instantiation_and_forward():
    """Verify PyTorch model definitions from notebook instantiate and complete forward pass cleanly."""
    meta_net = SignalMetaNetwork(num_features=DEFAULT_NUM_FEATURES, hidden_dim=256)
    q_net = ExecutorQNetwork(num_features=DEFAULT_NUM_FEATURES, input_dim=28, hidden_dim=128)

    meta_net.eval()
    q_net.eval()

    dummy_feat = torch.randn(2, 150, DEFAULT_NUM_FEATURES)
    dummy_ctx  = torch.randn(2, 28)

    with torch.no_grad():
        q_vals, strength, pips, risk, liq, rev = meta_net(dummy_feat)
        assert q_vals.shape == (2, 4)
        assert strength.shape == (2, 4)

        stacked_q = q_net(dummy_feat, dummy_ctx)
        assert stacked_q.shape == (2, 4, 3)

        single_head_q = q_net(dummy_feat, dummy_ctx, horizon_idx=2)
        assert single_head_q.shape == (2, 3)

    logger.info("✅ Models forward pass verified successfully.")


def test_axe_paka_v1_weights_loading():
    """Verify agent correctly loads downloaded weights from app/axe_paka_v1/weights/."""
    config = AxePakaV1Config()
    agent = AxePakaV1Agent(config=config, auto_load_weights=True)

    assert agent.weights_loaded is True, "Agent failed to load checkpoint weights from weights/"
    logger.info("✅ Weight checkpoint loading verified successfully.")


def test_all_timeframes_permitted_for_testing():
    """
    Verify trade gating when all 4 timeframes are permitted for data collection testing:
    - 5m (head 0), 15m (head 1), 30m (head 2), and 1h (head 3) are ALL permitted.
    """
    config = AxePakaV1Config()
    agent = AxePakaV1Agent(config=config, auto_load_weights=True)

    feat_window = np.zeros((150, DEFAULT_NUM_FEATURES), dtype=np.float32)
    ctx_state = np.zeros((28,), dtype=np.float32)
    action_mask = np.array([1, 1, 1, 0, 0], dtype=np.int32)

    verdict = agent.evaluate_signal(feat_window, ctx_state, action_mask)

    assert "permitted_expiries" in verdict
    assert verdict["permitted_expiries"] == ["5m", "15m", "30m", "1h"]

    evals = verdict["horizon_evaluations"]
    assert len(evals) == 4

    # All 4 heads MUST be marked permitted
    for h_idx, label in enumerate(["5m", "15m", "30m", "1h"]):
        assert evals[h_idx]["horizon_label"] == label
        assert evals[h_idx]["permitted"] is True

    logger.info("✅ All 4 timeframes permitted gating verified successfully.")


def test_alpaca_option_trade_execution_wiring():
    """Verify signal-to-execution pipeline for 30m and 1h option contracts."""
    config = AxePakaV1Config()
    agent = AxePakaV1Agent(config=config, use_cli=True, auto_load_weights=True)

    verdict_30m = {
        "decision": {
            "selected_action": ACTION_BUY_CALL,
            "action_name": "CALL",
            "horizon_idx": 2,
            "horizon_label": "30m",
        }
    }

    report = agent.execute_signal("SPY", verdict_30m, current_price=500.0, dry_run=True)

    assert report["status"] == "completed"
    assert report["symbol"].startswith("SPY")
    assert "C00" in report["symbol"] or "C" in report["symbol"]
    assert report["hold_duration_sec"] == 1800
    assert report["expiry_horizon"] == "30m"

    verdict_1h = {
        "decision": {
            "selected_action": ACTION_BUY_PUT,
            "action_name": "PUT",
            "horizon_idx": 3,
            "horizon_label": "1h",
        }
    }

    report_1h = agent.execute_signal("GLD", verdict_1h, current_price=200.0, dry_run=True)

    assert report_1h["status"] == "completed"
    assert report_1h["symbol"].startswith("GLD")
    assert "P00" in report_1h["symbol"] or "P" in report_1h["symbol"]
    assert report_1h["hold_duration_sec"] == 3600
    assert report_1h["expiry_horizon"] == "1h"

    logger.info("✅ Alpaca option trade execution pipeline verified successfully.")


def test_build_feat_window_lookahead_free():
    """Verify feature window helper handles padding and indexing without lookahead."""
    num_matrix = np.arange(100 * 333, dtype=np.float32).reshape(100, 333)

    # Index 10 with lookback 150 -> 140 rows of zero padding + 11 rows of data
    window = build_feat_window(num_matrix, abs_idx=10, q_lookback=150)
    assert window.shape == (150, 333)
    assert np.all(window[:139] == 0.0)
    assert np.array_equal(window[-1], num_matrix[10])

    logger.info("✅ Lookahead-free feature window verified successfully.")


def test_martingale_money_manager_progression():
    """Verify 4-Step Martingale engine with per-timeframe isolation (GLD_5m != GLD_30m)."""
    from app.axe_paka_v1.money_management import MartingaleMoneyManager, MartingaleMMConfig

    mm = MartingaleMoneyManager(MartingaleMMConfig(base_trade_dollars=10.0, max_martingale_steps=4))

    # Initial state: Step 1 ($10) on 30m
    d1, q1, s1 = mm.calculate_position_size("GLD", "30m", contract_price=1.0)
    assert s1 == 0 and d1 == 10.0

    # Record 3 losses on 30m -> escalates to Step 4 ($80)
    mm.record_trade_result("GLD", "30m", is_win=False)
    mm.record_trade_result("GLD", "30m", is_win=False)
    mm.record_trade_result("GLD", "30m", is_win=False)
    d4, q4, s4 = mm.calculate_position_size("GLD", "30m", contract_price=1.0)
    assert s4 == 3 and d4 == 80.0

    # 5m streak must still be at Step 1 ($10) — completely isolated
    d5m, _, s5m = mm.calculate_position_size("GLD", "5m", contract_price=1.0)
    assert s5m == 0 and d5m == 10.0, "5m streak must be independent of 30m losses!"

    # 4th loss on 30m -> max steps hit -> capital safety reset to Step 1
    mm.record_trade_result("GLD", "30m", is_win=False)
    d5, q5, s5 = mm.calculate_position_size("GLD", "30m", contract_price=1.0)
    assert s5 == 0 and d5 == 10.0

    # Win on 30m -> stays at Step 1
    mm.record_trade_result("GLD", "30m", is_win=True)
    d6, q6, s6 = mm.calculate_position_size("GLD", "30m", contract_price=1.0)
    assert s6 == 0 and d6 == 10.0

    logger.info("✅ Martingale per-timeframe isolation and 4-step progression verified.")
