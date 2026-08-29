"""
inspect_q_training_data.py — Inspection & Audit Tool for Options Q-Learner Training Data.

Prints detailed statistical distributions, sample state vectors, action masks,
and replay buffer transition tuples constructed during real-market pretraining.
"""

from __future__ import annotations

import logging
import pprint
import numpy as np
import pandas as pd

from app.core.ml.real_data_pipeline import RealTrainConfig, fetch_real_candles, align_multi_timeframe_datasets, _htf_start_for
from app.core.options.q_executor import OptionsQExecutor, HTFBiasPackage, AccountContext, ExecutionContext, ACTION_WAIT, EXECUTOR_STATE_DIM
from app.core.market.zone_snapshot import ZoneSnapshotManager, HardActionMask
from app.core.ml.signal_meta_learner import OnlineSignalMetaLearner, SIGNAL_META_FEATURE_COUNT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("InspectQData")

FEATURE_NAMES = [
    "00: dir_flag",
    "01: htf_strength",
    "02: reversal_prob",
    "03: htf_q_value",
    "04: expected_mfe_norm",
    "05: expected_mae_norm",
    "06: daily_drawdown_pct",
    "07: open_pos_type_flag",
    "08: open_pos_pnl_pct",
    "09: win_streak_norm",
    "10: loss_streak_norm",
    "11: ltf_tf_flag",
    "12: atr_ratio",
    "13: supp_dist",
    "14: res_dist",
    "15: supp_vol_ratio",
    "16: res_vol_ratio",
    "17: vol_delta_ratio",
    "18: reentries_norm",
    "19: sin_hour",
    "20: cos_hour",
    "21: dow_norm",
    "22: is_nyse_open",
    "23: is_power_hour",
]

ACTION_NAMES = {
    0: "WAIT",
    1: "BUY_CALL",
    2: "BUY_PUT",
    3: "TAKE_PROFIT_HALF",
    4: "CLOSE_FLATTEN",
}


def inspect_symbol_q_data(symbol: str = "GLD"):
    print(f"\n==================================================================")
    print(f"       INSPECTING Q-LEARNER TRAINING DATA FOR SYMBOL: {symbol}")
    print(f"==================================================================")

    config = RealTrainConfig()
    
    # 1. Fetch & Align real market data
    print("\n[Step 1] Fetching candles from Alpaca...")
    ltf_df = fetch_real_candles(symbol, timeframe="5m", limit=1000)
    if ltf_df.empty:
        print(f"Error: Could not fetch 5m candles for {symbol}")
        return

    ltf_anchor = ltf_df["timestamp"].min().to_pydatetime()
    tf_dfs = {"5m": ltf_df}

    for tf in ["15m", "1h", "4h", "1d"]:
        start_str = _htf_start_for(ltf_anchor, tf, lookback_bars=48)
        df_htf = fetch_real_candles(symbol, timeframe=tf, limit=1200, start=start_str)
        if not df_htf.empty:
            tf_dfs[tf] = df_htf

    aligned_df = align_multi_timeframe_datasets(tf_dfs, primary_tf="5m")
    aligned_df = aligned_df[aligned_df["timestamp"] >= pd.Timestamp(ltf_anchor)].reset_index(drop=True)
    
    print(f"\n[Step 2] Aligned Multi-Timeframe DataFrame Summary:")
    print(f"  - Total Aligned Rows: {len(aligned_df)}")
    print(f"  - Time Range: {aligned_df['timestamp'].min()} --> {aligned_df['timestamp'].max()}")
    print(f"  - Columns ({len(aligned_df.columns)}): {list(aligned_df.columns[:10])} ...")

    # 2. Initialize Models & State Components
    meta_learner = OnlineSignalMetaLearner(input_dim=SIGNAL_META_FEATURE_COUNT)
    q_executor = OptionsQExecutor(device="cpu")
    zone_manager = ZoneSnapshotManager(max_snapshots=20)
    account = AccountContext()

    # Populate S&R zones
    close_col = "close_5m" if "close_5m" in aligned_df.columns else "close"
    high_col = "high_5m" if "high_5m" in aligned_df.columns else "high"
    low_col = "low_5m" if "low_5m" in aligned_df.columns else "low"
    vol_col = "volume_5m" if "volume_5m" in aligned_df.columns else "volume"

    low_min = float(aligned_df[low_col].min())
    high_max = float(aligned_df[high_col].max())
    mid_p = (low_min + high_max) / 2.0
    
    raw_zones = [
        (1, low_min, [(0, low_min, "support")], {"upper_bound": low_min * 1.005, "lower_bound": low_min * 0.995, "total_volume": 50000.0, "net_volume": 10000.0}),
        (2, mid_p * 0.99, [(0, mid_p * 0.99, "support")], {"upper_bound": mid_p * 0.995, "lower_bound": mid_p * 0.985, "total_volume": 40000.0, "net_volume": 5000.0}),
        (3, high_max, [(0, high_max, "resistance")], {"upper_bound": high_max * 1.005, "lower_bound": high_max * 0.995, "total_volume": 60000.0, "net_volume": -12000.0}),
        (4, mid_p * 1.01, [(0, mid_p * 1.01, "resistance")], {"upper_bound": mid_p * 1.015, "lower_bound": mid_p * 1.005, "total_volume": 45000.0, "net_volume": -8000.0}),
    ]
    zone_manager.add_snapshot("snap_inspect", "1h", raw_zones)

    def _make_exec_ctx(price: float, row: pd.Series) -> ExecutionContext:
        ts = row.get("timestamp", pd.Timestamp.now(tz="UTC"))
        hour_f = ts.hour + ts.minute / 60.0 if isinstance(ts, pd.Timestamp) else 14.5
        dow = ts.dayofweek if isinstance(ts, pd.Timestamp) else 1
        return ExecutionContext(
            symbol=symbol,
            current_price=price,
            atr=float(row.get("atr_5m", price * 0.005)),
            buy_volume=float(row.get(vol_col, 1000.0)),
            sell_volume=float(row.get(vol_col, 800.0)) * 0.85,
            hour_of_day=hour_f,
            day_of_week=dow,
            session_phase="nyse_open" if 13 <= ts.hour < 17 else "off_hours",
        )

    print("\n[Step 3] Generating 50 Q-Learner Training Transitions & Buffer Samples...")
    states_collected = []
    actions_collected = []
    rewards_collected = []

    for i in range(min(50, len(aligned_df) - 2)):
        row = aligned_df.iloc[i]
        next_row = aligned_df.iloc[i + 1]

        cur_price = float(row[close_col])
        next_price = float(next_row[close_col])
        fwd_move_pct = (next_price - cur_price) / (cur_price + 1e-8)

        pred = meta_learner.predict({"close": cur_price, "volume": float(row[vol_col])})
        meta_score = float(pred.get("signal_strength", 0.5))

        htf_bias = HTFBiasPackage(
            direction="bullish" if meta_score > 0.5 else "bearish",
            strength=meta_score,
            reversal_prob=float(pred.get("reversal_prob", 0.2)),
            q_value=float(pred.get("q_value", 0.5)),
            expected_mfe_pips=float(pred.get("expected_mfe_pips", 50.0)),
            expected_mae_pips=float(pred.get("expected_mae_pips", 15.0)),
        )

        exec_ctx = _make_exec_ctx(cur_price, row)
        state_vec = q_executor.build_state_vector(htf_bias, account, exec_ctx, zone_manager)
        
        mask_engine = HardActionMask()
        action_mask = mask_engine.get_action_mask(
            current_price=cur_price,
            atr=exec_ctx.atr,
            zone_manager=zone_manager,
            buy_volume=exec_ctx.buy_volume,
            sell_volume=exec_ctx.sell_volume,
            has_open_position=False,
        )

        action = q_executor.select_action(state_vec, action_mask, eval_mode=False)
        reward = q_executor.calculate_executor_reward(
            action=action,
            action_mask=action_mask,
            pnl_pct=fwd_move_pct,
            max_drawdown_exposed=0.01,
            forward_move_pct=fwd_move_pct,
            htf_bias=htf_bias,
        )

        next_state_vec = q_executor.build_state_vector(htf_bias, account, exec_ctx, zone_manager)
        q_executor.record_transition(state_vec, action, reward, next_state_vec, False, action_mask)

        states_collected.append(state_vec)
        actions_collected.append(action)
        rewards_collected.append(reward)

    states_arr = np.array(states_collected)
    
    print("\n[Step 4] State Vector Breakdown (24 Features across 50 samples):")
    print(f"{'Index & Feature Name':<28} | {'Min':<8} | {'Max':<8} | {'Mean':<8} | {'Std':<8}")
    print("-" * 72)
    for idx, fname in enumerate(FEATURE_NAMES):
        col_vals = states_arr[:, idx]
        print(f"{fname:<28} | {col_vals.min():<8.4f} | {col_vals.max():<8.4f} | {col_vals.mean():<8.4f} | {col_vals.std():<8.4f}")

    print("\n[Step 5] Action & Reward Distribution:")
    action_counts = pd.Series([ACTION_NAMES[a] for a in actions_collected]).value_counts()
    for act_name, cnt in action_counts.items():
        print(f"  - Action '{act_name}': {cnt} times ({cnt/len(actions_collected)*100:.1f}%)")

    print(f"  - Reward Min  : {min(rewards_collected):.6f}")
    print(f"  - Reward Max  : {max(rewards_collected):.6f}")
    print(f"  - Reward Mean : {np.mean(rewards_collected):.6f}")

    print("\n[Step 6] Sample Transition Record #0 from Replay Buffer:")
    sample_trans = q_executor.replay_buffer[0]
    sample_s, sample_a, sample_r, sample_ns, sample_d, sample_m = sample_trans

    print(f"  - State Vector (first 10 dims)  : {np.round(sample_s[:10], 4).tolist()}")
    print(f"  - Action Index & Name           : {sample_a} ({ACTION_NAMES[sample_a]})")
    print(f"  - Calculated Shaped Reward      : {sample_r:.6f}")
    print(f"  - Next State Vector (first 10)  : {np.round(sample_ns[:10], 4).tolist()}")
    print(f"  - Done Flag                     : {sample_d}")
    print(f"  - Hard Action Mask              : {sample_m.tolist()} (WAIT=1, CALL={sample_m[1]}, PUT={sample_m[2]})")
    print("\n==================================================================\n")


if __name__ == "__main__":
    inspect_symbol_q_data("GLD")
    inspect_symbol_q_data("BTC/USD")
