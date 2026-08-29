"""
evaluate_option_expiries.py — Benchmark Tool to Find Optimal Option Expiry Horizon on 5m Execution.

Evaluates out-of-sample directional win rates, streaks, and performance across 4 option expiry horizons:
1. 5m  Expiry (N = 1 bar ahead)
2. 15m Expiry (N = 3 bars ahead)
3. 30m Expiry (N = 6 bars ahead)
4. 1h  Expiry (N = 12 bars ahead)
"""

from __future__ import annotations

import logging
import pprint
import numpy as np
import pandas as pd

from app.core.ml.real_data_pipeline import (
    RealTrainConfig,
    fetch_real_candles,
    align_multi_timeframe_datasets,
    _htf_start_for,
)
from app.core.options.q_executor import OptionsQExecutor, HTFBiasPackage, AccountContext, ExecutionContext
from app.core.market.zone_snapshot import ZoneSnapshotManager, HardActionMask
from app.core.ml.signal_meta_learner import OnlineSignalMetaLearner, SIGNAL_META_FEATURE_COUNT

def _make_exec_ctx(symbol: str, price: float, row: dict) -> ExecutionContext:
    ts = row.get("timestamp", pd.Timestamp.now(tz="UTC"))
    if isinstance(ts, pd.Timestamp) and ts.tzinfo is not None:
        hour_f = ts.hour + ts.minute / 60.0
        dow = ts.dayofweek
    else:
        hour_f, dow = 14.5, 1
    phase = "nyse_open" if 13 <= getattr(ts, "hour", 14) < 17 else "off_hours"
    vol = float(row.get("volume_5m", row.get("volume", 1000.0)))
    return ExecutionContext(
        symbol=symbol,
        current_price=price,
        atr=float(row.get("atr_5m", price * 0.005)),
        buy_volume=vol,
        sell_volume=vol * 0.85,
        hour_of_day=hour_f,
        day_of_week=dow,
        session_phase=phase,
    )

def _row_to_feature_dict(row_data: pd.Series, close_col: str, vol_col: str) -> dict:
    f_dict = {"close": float(row_data.get(close_col, 0.0)), "volume": float(row_data.get(vol_col, 0.0))}
    for k, v in row_data.items():
        if k not in ("timestamp", close_col, vol_col) and isinstance(v, (int, float, np.number)):
            f_dict[str(k)] = float(v)
    return f_dict


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EvalExpiries")

EXPIRY_HORIZONS = {
    "5m (1 bar)": 1,
    "15m (3 bars)": 3,
    "30m (6 bars)": 6,
    "1h (12 bars)": 12,
}


def evaluate_expiries_for_symbol(symbol: str = "GLD"):
    print(f"\n==================================================================================")
    print(f"      EVALUATING OPTION EXPIRY HORIZONS FOR SYMBOL: {symbol} (5m Execution)")
    print(f"==================================================================================")

    # 1. Fetch & Align real market data with expanded dataset (30,000 rows)
    logger.info("Fetching real market bars for %s (30,000 bars limit)...", symbol)
    ltf_df = fetch_real_candles(symbol, timeframe="5m", limit=30000)
    if ltf_df.empty:
        print(f"Error: Could not fetch 5m candles for {symbol}")
        return

    ltf_anchor = ltf_df["timestamp"].min().to_pydatetime()
    tf_dfs = {"5m": ltf_df}

    for tf in ["15m", "1h", "4h", "1d"]:
        start_str = _htf_start_for(ltf_anchor, tf, lookback_bars=1000)
        df_htf = fetch_real_candles(symbol, timeframe=tf, limit=30000, start=start_str)
        if not df_htf.empty:
            tf_dfs[tf] = df_htf


    aligned_df = align_multi_timeframe_datasets(tf_dfs, primary_tf="5m")
    aligned_df = aligned_df[aligned_df["timestamp"] >= pd.Timestamp(ltf_anchor)].reset_index(drop=True)

    close_col = "close_5m" if "close_5m" in aligned_df.columns else "close"
    high_col = "high_5m" if "high_5m" in aligned_df.columns else "high"
    low_col = "low_5m" if "low_5m" in aligned_df.columns else "low"
    vol_col = "volume_5m" if "volume_5m" in aligned_df.columns else "volume"

    aligned_df["forward_move_24"] = aligned_df[close_col].shift(-24) - aligned_df[close_col]
    aligned_df.dropna(subset=["forward_move_24"], inplace=True)

    # 80/20 Train / Test Split
    split_idx = int(len(aligned_df) * 0.8)
    train_df = aligned_df.iloc[:split_idx].reset_index(drop=True)
    test_df = aligned_df.iloc[split_idx:].reset_index(drop=True)

    print(f"Dataset Split -> Total Rows: {len(aligned_df)} | Train Rows: {len(train_df)} | Holdout Test Rows: {len(test_df)}")

    # 2. Train Two-Tier Ensemble Learners once on Train Set
    meta_learner = OnlineSignalMetaLearner(input_dim=SIGNAL_META_FEATURE_COUNT)
    q_executor = OptionsQExecutor(device="cpu")
    zone_manager = ZoneSnapshotManager(max_snapshots=20)
    account = AccountContext()

    def _update_dynamic_zones(df_slice: pd.DataFrame):
        if len(df_slice) < 20:
            return
        recent = df_slice.tail(1000)
        supp_level = float(recent[low_col].quantile(0.20))
        res_level = float(recent[high_col].quantile(0.80))

        mid_level = (supp_level + res_level) / 2.0
        raw_z = [
            (1, supp_level, [(0, supp_level, "support")], {"upper_bound": supp_level * 1.004, "lower_bound": supp_level * 0.996, "total_volume": 50000.0, "net_volume": 10000.0}),
            (2, res_level, [(0, res_level, "resistance")], {"upper_bound": res_level * 1.004, "lower_bound": res_level * 0.996, "total_volume": 60000.0, "net_volume": -12000.0}),
            (3, mid_level, [(0, mid_level, "support" if mid_level < float(df_slice[close_col].iloc[-1]) else "resistance")], {"upper_bound": mid_level * 1.002, "lower_bound": mid_level * 0.998, "total_volume": 35000.0, "net_volume": 2000.0}),
        ]
        zone_manager.add_snapshot(f"snap_{len(zone_manager.history)}", "15m", raw_z)

    _update_dynamic_zones(aligned_df)

    print("\n[Phase 1] Training Meta-Learner (150 gradient steps)...")
    for step in range(150):
        batch_indices = np.random.choice(len(train_df) - 25, size=min(64, len(train_df) - 25), replace=False)
        for idx in batch_indices:
            row = train_df.iloc[idx]
            fut_highs = train_df[high_col].iloc[idx+1:idx+25].values
            fut_lows = train_df[low_col].iloc[idx+1:idx+25].values
            fut_closes = train_df[close_col].iloc[idx+1:idx+25].values
            f_dict = {"close": float(row[close_col]), "volume": float(row[vol_col])}
            meta_learner.record_experience(
                feature_dict=f_dict,
                signal_id=f"sig_{idx}",
                symbol=symbol,
                direction="bullish" if row["forward_move_24"] > 0 else "bearish",
                entry_price=float(row[close_col]),
                future_highs=fut_highs,
                future_lows=fut_lows,
                future_closes=fut_closes,
            )
        meta_learner.train_step(batch_size=64)

    print("\n[Phase 2] Training Dual-Branch Ensemble Options Q-Learner (300 gradient steps)...")
    for _ in range(128):
        idx = np.random.randint(0, len(train_df) - 2)
        row_pf = train_df.iloc[idx]
        next_row_pf = train_df.iloc[idx + 1]
        cp = float(row_pf[close_col])
        np_ = float(next_row_pf[close_col])
        fwd_pf = (np_ - cp) / (cp + 1e-8)
        f_dict_pf = _row_to_feature_dict(row_pf, close_col, vol_col)
        pred_pf = meta_learner.predict(f_dict_pf)
        ms_pf = float(pred_pf.get("signal_strength", 0.5))
        bias_pf = HTFBiasPackage(
            direction="bullish" if ms_pf > 0.5 else "bearish",
            strength=ms_pf,
            reversal_prob=float(pred_pf.get("reversal_prob", 0.2)),
            q_value=float(pred_pf.get("q_value", 0.5)),
            expected_mfe_pips=float(pred_pf.get("expected_mfe_pips", 50.0)),
            expected_mae_pips=float(pred_pf.get("expected_mae_pips", 15.0)),
        )
        ctx_pf = _make_exec_ctx(symbol, cp, dict(row_pf))
        st_pf = q_executor.build_state_vector(bias_pf, account, ctx_pf, zone_manager)
        mask_pf = np.ones(5, dtype=np.int32)
        act_pf = q_executor.select_action(st_pf, mask_pf)
        rew_pf = q_executor.calculate_executor_reward(
            action=act_pf, action_mask=mask_pf, pnl_pct=fwd_pf, max_drawdown_exposed=0.01, forward_move_pct=fwd_pf, htf_bias=bias_pf,
        )
        nst_pf = q_executor.build_state_vector(bias_pf, account, ctx_pf, zone_manager)
        q_executor.record_transition(st_pf, act_pf, rew_pf, nst_pf, False, mask_pf)

    for step in range(300):
        idx = np.random.randint(0, len(train_df) - 2)
        row = train_df.iloc[idx]
        next_row = train_df.iloc[idx + 1]
        cur_price = float(row[close_col])
        next_price = float(next_row[close_col])
        fwd_move_pct = (next_price - cur_price) / (cur_price + 1e-8)
        f_dict = _row_to_feature_dict(row, close_col, vol_col)
        pred = meta_learner.predict(f_dict)
        meta_score = float(pred.get("signal_strength", 0.5))
        htf_bias = HTFBiasPackage(
            direction="bullish" if meta_score > 0.5 else "bearish",
            strength=meta_score,
            reversal_prob=float(pred.get("reversal_prob", 0.2)),
            q_value=float(pred.get("q_value", 0.5)),
            expected_mfe_pips=float(pred.get("expected_mfe_pips", 50.0)),
            expected_mae_pips=float(pred.get("expected_mae_pips", 15.0)),
        )
        exec_ctx = _make_exec_ctx(symbol, cur_price, dict(row))
        state = q_executor.build_state_vector(htf_bias, account, exec_ctx, zone_manager)
        mask_engine = HardActionMask()
        action_mask = mask_engine.get_action_mask(
            current_price=cur_price, atr=exec_ctx.atr, zone_manager=zone_manager, buy_volume=exec_ctx.buy_volume, sell_volume=exec_ctx.sell_volume, has_open_position=False,
        )
        action = q_executor.select_action(state, action_mask)
        reward = q_executor.calculate_executor_reward(
            action=action, action_mask=action_mask, pnl_pct=fwd_move_pct, max_drawdown_exposed=0.01, forward_move_pct=fwd_move_pct, htf_bias=htf_bias,
        )
        next_state = q_executor.build_state_vector(htf_bias, account, exec_ctx, zone_manager)
        q_executor.record_transition(state, action, reward, next_state, False, action_mask)
        q_executor.train_step(batch_size=64)

    # 3. Evaluate Out-of-Sample Test Window across all 4 Expiry Horizons
    print("\n[Phase 3] Evaluating Out-of-Sample Performance across Expiry Horizons:")
    print(f"{'Expiry Horizon':<18} | {'Trades':<8} | {'Wins':<6} | {'Losses':<8} | {'Waits':<8} | {'Win Rate %':<10} | {'Max Streak [W, L]':<18}")
    print("-" * 92)

    results_table = {}

    for exp_label, lookahead_bars in EXPIRY_HORIZONS.items():
        wins, losses, waits = 0, 0, 0
        cur_w_streak, cur_l_streak = 0, 0
        max_w_streak, max_l_streak = 0, 0
        open_trade_until_idx = -1

        for idx in range(len(test_df) - lookahead_bars):
            row = test_df.iloc[idx]
            expiry_row = test_df.iloc[idx + lookahead_bars]

            cur_price = float(row[close_col])
            expiry_price = float(expiry_row[close_col])

            has_open_position = (idx < open_trade_until_idx)

            if idx % 50 == 0:
                _update_dynamic_zones(test_df.iloc[max(0, idx - 150): idx + 1])

            f_dict = _row_to_feature_dict(row, close_col, vol_col)
            pred = meta_learner.predict(f_dict)
            meta_score = float(pred.get("signal_strength", 0.5))

            htf_bias = HTFBiasPackage(
                direction="bullish" if meta_score > 0.5 else "bearish",
                strength=meta_score,
                reversal_prob=float(pred.get("reversal_prob", 0.2)),
                q_value=float(pred.get("q_value", 0.5)),
                expected_mfe_pips=float(pred.get("expected_mfe_pips", 50.0)),
                expected_mae_pips=float(pred.get("expected_mae_pips", 15.0)),
            )

            account.open_position_type = "CALL" if has_open_position else None
            exec_ctx = _make_exec_ctx(symbol, cur_price, dict(row))
            state = q_executor.build_state_vector(htf_bias, account, exec_ctx, zone_manager)
            mask_engine = HardActionMask()
            action_mask = mask_engine.get_action_mask(
                current_price=cur_price, atr=exec_ctx.atr, zone_manager=zone_manager, buy_volume=exec_ctx.buy_volume, sell_volume=exec_ctx.sell_volume, has_open_position=has_open_position,
            )
            if not has_open_position and action_mask[1] == 0 and action_mask[2] == 0 and htf_bias.strength >= 0.50:
                if htf_bias.direction == "bullish":
                    action_mask[1] = 1
                else:
                    action_mask[2] = 1

            action = q_executor.select_action(state, action_mask, eval_mode=True)

            if action == 1 and not has_open_position:  # BUY_CALL
                open_trade_until_idx = idx + lookahead_bars
                if expiry_price > cur_price:
                    wins += 1
                    cur_w_streak += 1
                    cur_l_streak = 0
                else:
                    losses += 1
                    cur_l_streak += 1
                    cur_w_streak = 0
            elif action == 2 and not has_open_position:  # BUY_PUT
                open_trade_until_idx = idx + lookahead_bars
                if expiry_price < cur_price:
                    wins += 1
                    cur_w_streak += 1
                    cur_l_streak = 0
                else:
                    losses += 1
                    cur_l_streak += 1
                    cur_w_streak = 0
            else:
                waits += 1

            max_w_streak = max(max_w_streak, cur_w_streak)
            max_l_streak = max(max_l_streak, cur_l_streak)


        total_trades = wins + losses
        win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0

        print(f"{exp_label:<18} | {total_trades:<8} | {wins:<6} | {losses:<8} | {waits:<8} | {win_rate:<10.2f} | W:{max_w_streak} / L:{max_l_streak}")

        results_table[exp_label] = {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "waits": waits,
            "win_rate_pct": round(win_rate, 2),
            "max_win_streak": max_w_streak,
            "max_loss_streak": max_l_streak,
        }

    print("\n==================================================================================\n")
    return results_table


if __name__ == "__main__":
    evaluate_expiries_for_symbol("GLD")
    evaluate_expiries_for_symbol("BTC/USD")
