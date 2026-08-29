"""
real_data_pipeline.py — Multi-Timeframe Real Market Data Alignment & Two-Tier Training Pipeline.

Alignment guarantee (no look-forward bias):
  - The LTF (5m) is the master timeline. Its bars define every training step.
  - HTF bars are timestamped at their OPEN by Alpaca. A 1H bar opening at 14:00
    does not close until 14:59. To prevent a 5m bar at 14:05 from seeing that
    incomplete 1H bar, each HTF's timestamps are shifted forward by exactly
    one HTF interval before `merge_asof(direction='backward')`. This means
    the 14:00 H1 bar only becomes visible to the LTF index at 15:00 (after close).
  - HTF fetch windows start earlier than the LTF start by (lookback_bars × htf_interval)
    so the context window is fully populated from bar 1 of the training window.
  - All fetches are co-anchored: LTF start is determined first from actual API
    data, then each HTF computes its own extended start from that anchor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from app.utils.alpaca_client import AlpacaClient
from app.core.analysis.support_resistance import detect_snr_levels_sequential, create_clustered_zones_sequential
from app.core.market.zone_snapshot import ZoneSnapshotManager, HardActionMask
from app.core.ml.signal_meta_learner import OnlineSignalMetaLearner, SIGNAL_META_FEATURE_COUNT
from app.core.options.q_executor import OptionsQExecutor, HTFBiasPackage, AccountContext, ExecutionContext, ACTION_WAIT

logger = logging.getLogger(__name__)


@dataclass
class RealTrainConfig:
    symbols: Tuple[str, ...] = ("GLD", "BTC/USD")
    primary_tf: str = "5m"
    timeframes: Tuple[str, ...] = ("5m", "15m", "1h", "4h", "1d")
    target_rows_per_tf: int = 40000
    meta_train_steps: int = 100
    q_train_steps: int = 100
    batch_size: int = 32
    device: str = "cpu"
    # How many closed HTF bars to guarantee before the first LTF training bar.
    # This fills the context window so the learner never trains on padding.
    htf_context_lookback_bars: int = 1000



# Interval seconds per timeframe label
_TF_INTERVAL_SECS: Dict[str, int] = {
    "5m":  300,
    "15m": 900,
    "1h":  3600,
    "4h":  14400,
    "1d":  86400,
}


def _htf_start_for(ltf_start: datetime, htf_tf: str, lookback_bars: int) -> str:
    """
    Compute the ISO-8601 start string for an HTF fetch so that *at minimum*
    `lookback_bars` fully-closed HTF bars exist before `ltf_start`.

    We add a 1.5× safety multiplier to absorb weekends / market-closed gaps.
    """
    interval_s = _TF_INTERVAL_SECS.get(htf_tf, 3600)
    padding = timedelta(seconds=interval_s * lookback_bars * 1.5)
    htf_start = ltf_start - padding
    return htf_start.strftime("%Y-%m-%dT%H:%M:%SZ")


def _bars_to_df(raw_bars: List[Dict], symbol: str, timeframe: str) -> pd.DataFrame:
    """Convert a list of Alpaca bar dicts to a normalised DataFrame."""
    records = []
    for b in raw_bars:
        raw_t = b.get("t") or b.get("timestamp")
        ts = pd.to_datetime(raw_t).tz_convert("UTC") if raw_t else pd.Timestamp.now(tz="UTC")
        records.append({
            "timestamp": ts,
            "open":   float(b.get("o", b.get("open",  0))),
            "high":   float(b.get("h", b.get("high",  0))),
            "low":    float(b.get("l", b.get("low",   0))),
            "close":  float(b.get("c", b.get("close", 0))),
            "volume": float(b.get("v", b.get("volume",0))),
        })
    df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    logger.info("[DataFetch] Normalised %d bars for %s (%s)", len(df), symbol, timeframe)
    return df


def fetch_real_candles(
    symbol: str,
    timeframe: str,
    limit: int = 10000,
    start: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch real market candles via Alpaca Data API (primary), with a
    synthetic generator as last-resort fallback.

    - Equity/ETF (e.g. GLD) → Alpaca v2 stocks/bars with IEX feed + start date.
    - Crypto (e.g. BTC/USD) → Alpaca v1beta3 crypto/us/bars + start date.
    - Synthetic candles are only generated if both Alpaca paths return zero bars.
    - Each timeframe uses a unique seed offset so timestamps never collide.

    Normalises output columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
    """
    client = AlpacaClient()

    logger.info("[DataFetch] Requesting %d bars for %s (%s) via Alpaca (start=%s)...", limit, symbol, timeframe, start or "auto")
    raw_bars = client.get_bars(symbol, timeframe=timeframe, limit=limit, start=start)

    if raw_bars:
        return _bars_to_df(raw_bars, symbol, timeframe)

    # ── Synthetic last-resort fallback ────────────────────────────────────────
    # Use a unique seed per timeframe so each HTF gets distinct timestamps and
    # doesn't collide with the 5m primary index after merge_asof.
    tf_seed_offset = {"5m": 0, "15m": 1, "1h": 2, "4h": 3, "1d": 4}
    seed = 42 + tf_seed_offset.get(timeframe, 0)

    logger.warning(
        "[DataFetch] Alpaca returned 0 bars for %s (%s) — using synthetic fallback (seed=%d).",
        symbol, timeframe, seed,
    )
    from app.core.data.synthetic_data_generator import SyntheticDataGenerator
    gen = SyntheticDataGenerator(seed=seed)
    start_p = 2500.0 if ("GLD" in symbol or "XAU" in symbol) else 65000.0
    interval_s = _TF_INTERVAL_SECS.get(timeframe, 300)
    synth_candles = gen.generate_session(
        symbol=symbol,
        num_candles=min(limit, 1000),
        start_price=start_p,
        interval_seconds=interval_s,
    )
    records = [
        {
            "timestamp": pd.to_datetime(b.time, unit="s", utc=True),
            "open":   float(b.open),
            "high":   float(b.high),
            "low":    float(b.low),
            "close":  float(b.close),
            "volume": float(b.volume),
        }
        for b in synth_candles
    ]
    df_synth = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    logger.info(
        "[DataFetch] Synthetic fallback: generated %d bars for %s (%s)",
        len(df_synth), symbol, timeframe,
    )
    return df_synth




def align_multi_timeframe_datasets(
    tf_dfs: Dict[str, pd.DataFrame],
    primary_tf: str = "5m",
) -> pd.DataFrame:
    """
    Align multi-timeframe candle data onto the primary (lowest) timeframe index
    with a strict no-lookahead guarantee.

    The key insight: Alpaca timestamps every bar at its *open*. A 1H bar that
    opens at 14:00 does not close until 14:59:59. If we naïvely merge_asof, a
    5m bar at 14:05 sees the still-open 1H bar — this is look-forward bias.

    Fix: before merging, we shift each HTF's timestamp forward by exactly one
    HTF interval. The 14:00 1H bar becomes 15:00, meaning it only becomes
    visible to the LTF index after the bar has closed. For the primary TF we
    keep timestamps as-is (the live bar is the current context, not a future one).
    """
    if primary_tf not in tf_dfs or tf_dfs[primary_tf].empty:
        raise ValueError(f"Primary timeframe {primary_tf} DataFrame is missing or empty")

    base_df = tf_dfs[primary_tf].sort_values("timestamp").copy()
    base_df.columns = [
        f"{c}_{primary_tf}" if c != "timestamp" else "timestamp"
        for c in base_df.columns
    ]

    aligned = base_df
    for tf, df in tf_dfs.items():
        if tf == primary_tf or df.empty:
            continue

        interval_s = _TF_INTERVAL_SECS.get(tf, 3600)
        df_sorted = df.sort_values("timestamp").copy()

        # ── Anti-lookahead shift ──────────────────────────────────────────────
        # Shift each HTF bar's label to the timestamp at which it *closes*
        # (open_time + 1 interval). After this shift, merge_asof(backward)
        # will only match a bar to LTF rows that are ≥ bar_close_time, i.e.
        # the bar is fully settled before we use its data.
        df_sorted["timestamp"] = df_sorted["timestamp"] + pd.Timedelta(seconds=interval_s)

        df_sorted.columns = [
            f"{c}_{tf}" if c != "timestamp" else "timestamp"
            for c in df_sorted.columns
        ]

        aligned = pd.merge_asof(
            aligned,
            df_sorted,
            on="timestamp",
            direction="backward",
        )

    # Forward-fill only; never backward-fill (that would introduce future info)
    aligned.ffill(inplace=True)
    logger.info(
        "[Alignment] Aligned MTF dataset: %d LTF rows, HTF timestamps shifted +1 interval (no lookahead).",
        len(aligned),
    )
    return aligned


def train_sequential_two_tier(
    symbol: str,
    aligned_df: pd.DataFrame,
    config: RealTrainConfig
) -> Dict[str, Any]:
    """
    Sequential Two-Tier Training Pipeline:
    Phase 1: Pretrain Tier 1 Meta-Learner on forward-looking 24-bar target metrics.
    Phase 2: Pretrain Tier 2 Options Q-Learner on LTF context using Tier 1 bias inputs.
    """
    logger.info("=== STARTING SEQUENTIAL TWO-TIER TRAINING FOR %s ===", symbol)

    # 1. Initialize Models
    meta_learner = OnlineSignalMetaLearner(input_dim=SIGNAL_META_FEATURE_COUNT)
    q_executor = OptionsQExecutor(device=config.device)
    zone_manager = ZoneSnapshotManager(max_snapshots=20)

    # Calculate indicators & forward 24-bar target metrics
    close_col = "close_5m" if "close_5m" in aligned_df.columns else "close"
    high_col = "high_5m" if "high_5m" in aligned_df.columns else "high"
    low_col = "low_5m" if "low_5m" in aligned_df.columns else "low"
    vol_col = "volume_5m" if "volume_5m" in aligned_df.columns else "volume"

    def _update_dynamic_zones(df_slice: pd.DataFrame):
        """Update zone_manager with rolling Support/Resistance zones."""
        if len(df_slice) < 20:
            return
        recent = df_slice.tail(150)
        supp_level = float(recent[low_col].quantile(0.20))
        res_level = float(recent[high_col].quantile(0.80))
        mid_level = (supp_level + res_level) / 2.0
        raw_z = [
            (1, supp_level, [(0, supp_level, "support")], {"upper_bound": supp_level * 1.004, "lower_bound": supp_level * 0.996, "total_volume": 50000.0, "net_volume": 10000.0}),
            (2, res_level, [(0, res_level, "resistance")], {"upper_bound": res_level * 1.004, "lower_bound": res_level * 0.996, "total_volume": 60000.0, "net_volume": -12000.0}),
            (3, mid_level, [(0, mid_level, "support" if mid_level < float(df_slice[close_col].iloc[-1]) else "resistance")], {"upper_bound": mid_level * 1.002, "lower_bound": mid_level * 0.998, "total_volume": 35000.0, "net_volume": 2000.0}),
        ]
        zone_manager.add_snapshot(f"snap_{len(zone_manager.history)}", "15m", raw_z)

    # Initial S&R zone population
    _update_dynamic_zones(aligned_df)

    aligned_df["forward_move_24"] = aligned_df[close_col].shift(-24) - aligned_df[close_col]
    aligned_df["forward_mfe_24"] = aligned_df[high_col].rolling(24).max().shift(-24) - aligned_df[close_col]
    aligned_df["forward_mae_24"] = aligned_df[close_col] - aligned_df[low_col].rolling(24).min().shift(-24)
    aligned_df.dropna(subset=["forward_move_24"], inplace=True)


    meta_losses = []
    q_losses = []


    # ── 80/20 Chronological Train/Test Split ──────────────────────────────────
    split_idx = int(len(aligned_df) * 0.8)


    train_df = aligned_df.iloc[:split_idx].reset_index(drop=True)
    test_df = aligned_df.iloc[split_idx:].reset_index(drop=True)
    logger.info("[Split] Dataset split into %d Train rows (80%%) and %d Test rows (20%%)", len(train_df), len(test_df))

    # Phase 1: Train Tier 1 Meta-Learner
    logger.info("--> Phase 1: Training Tier 1 Meta-Learner (%d steps on Train set)...", config.meta_train_steps)
    for step in range(config.meta_train_steps):
        batch_indices = np.random.choice(len(train_df) - 25, size=min(config.batch_size, len(train_df) - 25), replace=False)
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

        step_loss = meta_learner.train_step(batch_size=config.batch_size)
        if step_loss is not None:
            meta_losses.append(step_loss)

    # Phase 2: Train Tier 2 Options Q-Learner
    logger.info("--> Phase 2: Training Tier 2 Options Q-Learner (%d steps on Train set)...", config.q_train_steps)
    account = AccountContext()

    def _make_exec_ctx(price: float, row: "pd.Series") -> ExecutionContext:
        """Build ExecutionContext from an aligned row's timestamp-derived features."""
        ts = row.get("timestamp", pd.Timestamp.now(tz="UTC"))
        if isinstance(ts, pd.Timestamp) and ts.tzinfo is not None:
            hour_f = ts.hour + ts.minute / 60.0
            dow = ts.dayofweek
        else:
            hour_f, dow = 14.5, 1
        if 13 <= ts.hour < 17:
            phase = "nyse_open"
        elif 9 <= ts.hour < 13:
            phase = "london_open"
        elif 21 <= ts.hour or ts.hour < 3:
            phase = "asian_open"
        else:
            phase = "off_hours"
        return ExecutionContext(
            symbol=symbol,
            current_price=price,
            atr=float(row.get(f"atr_{config.primary_tf}", 5.0)),
            buy_volume=float(row.get(vol_col, 1000.0)),
            sell_volume=float(row.get(vol_col, 800.0)) * 0.85,
            hour_of_day=hour_f,
            day_of_week=dow,
            session_phase=phase,
        )

    def _row_to_feature_dict(row_data: "pd.Series") -> Dict[str, Any]:
        """Convert a row Series into a feature dictionary for Meta-Learner prediction."""
        f_dict = {"close": float(row_data.get(close_col, 0.0)), "volume": float(row_data.get(vol_col, 0.0))}
        for k, v in row_data.items():
            if k not in ("timestamp", close_col, vol_col) and isinstance(v, (int, float, np.number)):
                f_dict[str(k)] = float(v)
        return f_dict

    # ── Pre-fill replay buffer from train set so train_step has ≥ batch_size samples ───
    prefill_count = max(config.batch_size * 2, 64)
    logger.info("[Q-Pretraining] Pre-filling replay buffer with %d transitions using Meta-Learner trade suggestions...", prefill_count)
    for _ in range(prefill_count):
        idx = np.random.randint(0, len(train_df) - 2)
        row_pf = train_df.iloc[idx]
        next_row_pf = train_df.iloc[idx + 1]
        cp = float(row_pf[close_col])
        np_ = float(next_row_pf[close_col])
        fwd_pf = (np_ - cp) / (cp + 1e-8)
        f_dict_pf = _row_to_feature_dict(row_pf)
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
        ctx_pf = _make_exec_ctx(cp, dict(row_pf))
        st_pf = q_executor.build_state_vector(bias_pf, account, ctx_pf, zone_manager)
        mask_pf = np.ones(5, dtype=np.int32)
        act_pf = q_executor.select_action(st_pf, mask_pf)
        rew_pf = q_executor.calculate_executor_reward(
            action=act_pf, action_mask=mask_pf,
            pnl_pct=fwd_pf, max_drawdown_exposed=0.01,
            forward_move_pct=fwd_pf, htf_bias=bias_pf,
        )
        nst_pf = q_executor.build_state_vector(bias_pf, account, ctx_pf, zone_manager)
        q_executor.record_transition(st_pf, act_pf, rew_pf, nst_pf, False, mask_pf)

    # ── Training loop on train set ───────────────────────────────────────────────
    for step in range(config.q_train_steps):
        idx = np.random.randint(0, len(train_df) - 2)
        row = train_df.iloc[idx]
        next_row = train_df.iloc[idx + 1]

        cur_price = float(row[close_col])
        next_price = float(next_row[close_col])
        fwd_move_pct = (next_price - cur_price) / (cur_price + 1e-8)

        f_dict = _row_to_feature_dict(row)
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

        exec_ctx = _make_exec_ctx(cur_price, dict(row))
        state = q_executor.build_state_vector(htf_bias, account, exec_ctx, zone_manager)
        action_mask = np.ones(5, dtype=np.int32)
        action = q_executor.select_action(state, action_mask)

        reward = q_executor.calculate_executor_reward(
            action=action,
            action_mask=action_mask,
            pnl_pct=fwd_move_pct,
            max_drawdown_exposed=0.01,
            forward_move_pct=fwd_move_pct,
            htf_bias=htf_bias,
        )

        next_state = q_executor.build_state_vector(htf_bias, account, exec_ctx, zone_manager)
        q_executor.record_transition(state, action, reward, next_state, False, action_mask)

        q_loss = q_executor.train_step(batch_size=config.batch_size)
        if q_loss is not None:
            q_losses.append(q_loss)

    # ── Phase 3: Out-of-Sample Test Window Evaluation (20% Holdout) ───────────
    logger.info("--> Phase 3: Evaluating Q-Learner Directional Accuracy on Test Window (%d rows)...", len(test_df))
    test_account = AccountContext()
    wins, losses, waits = 0, 0, 0
    cur_win_streak, cur_loss_streak = 0, 0
    max_win_streak, max_loss_streak = 0, 0

    for idx in range(len(test_df) - 1):
        row = test_df.iloc[idx]
        next_row = test_df.iloc[idx + 1]

        cur_price = float(row[close_col])
        next_price = float(next_row[close_col])

        if idx % 50 == 0:
            _update_dynamic_zones(test_df.iloc[max(0, idx - 150): idx + 1])

        f_dict = _row_to_feature_dict(row)
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

        exec_ctx = _make_exec_ctx(cur_price, dict(row))
        state = q_executor.build_state_vector(htf_bias, test_account, exec_ctx, zone_manager)
        
        mask_engine = HardActionMask()
        action_mask = mask_engine.get_action_mask(
            current_price=cur_price,
            atr=exec_ctx.atr,
            zone_manager=zone_manager,
            buy_volume=exec_ctx.buy_volume,
            sell_volume=exec_ctx.sell_volume,
            has_open_position=False,
        )
        # If HardActionMask returns all 0 except WAIT, unmask entries when Meta strength >= 0.50
        if action_mask[1] == 0 and action_mask[2] == 0 and htf_bias.strength >= 0.50:
            if htf_bias.direction == "bullish":
                action_mask[1] = 1
            else:
                action_mask[2] = 1

        # Select action greedily in eval_mode (no random exploration)
        action = q_executor.select_action(state, action_mask, eval_mode=True)


        if action == 1:  # ACTION_BUY_CALL (Bullish)
            if next_price > cur_price:
                wins += 1
                cur_win_streak += 1
                cur_loss_streak = 0
            else:
                losses += 1
                cur_loss_streak += 1
                cur_win_streak = 0
        elif action == 2:  # ACTION_BUY_PUT (Bearish)
            if next_price < cur_price:
                wins += 1
                cur_win_streak += 1
                cur_loss_streak = 0
            else:
                losses += 1
                cur_loss_streak += 1
                cur_win_streak = 0
        else:
            waits += 1

        max_win_streak = max(max_win_streak, cur_win_streak)
        max_loss_streak = max(max_loss_streak, cur_loss_streak)
        test_account.win_streak = cur_win_streak
        test_account.loss_streak = cur_loss_streak

    total_trades = wins + losses
    win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0

    test_metrics = {
        "test_rows": len(test_df),
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "waits": waits,
        "win_rate_pct": round(win_rate, 2),
        "max_consecutive_wins": max_win_streak,
        "max_consecutive_losses": max_loss_streak,
        "final_win_streak": cur_win_streak,
        "final_loss_streak": cur_loss_streak,
    }

    logger.info(
        "[TestEval] %s Results -> Trades: %d (Wins: %d, Losses: %d, Waits: %d) | Win Rate: %.2f%% | Max Streak [W: %d, L: %d]",
        symbol, total_trades, wins, losses, waits, win_rate, max_win_streak, max_loss_streak,
    )

    logger.info("=== TWO-TIER TRAINING & EVALUATION COMPLETE FOR %s ===", symbol)
    return {
        "symbol": symbol,
        "total_aligned_rows": len(aligned_df),
        "train_rows": len(train_df),
        "meta_final_loss": meta_losses[-1] if meta_losses else None,
        "q_final_loss": q_losses[-1] if q_losses else None,
        "test_metrics": test_metrics,
    }



def run_real_data_pipeline(config: RealTrainConfig = RealTrainConfig()) -> Dict[str, Any]:
    """
    Execute the complete real-market pretraining pipeline.

    Fetch strategy (no-lookahead guarantee):
    1. Fetch the LTF (primary) data first using its natural look-back window.
    2. Record the LTF's earliest actual timestamp as `ltf_anchor`.
    3. For every HTF, compute an *extended* start date:
           htf_start = ltf_anchor − (htf_context_lookback_bars × htf_interval × 1.5)
       This ensures at least `htf_context_lookback_bars` fully-closed HTF bars
       exist before the first LTF training row, so the learner's context
       window is never empty or padded at the start of training.
    4. Align with anti-lookahead HTF timestamp shifting (+1 interval).
    """
    results = {}
    for sym in config.symbols:
        logger.info("[Pipeline] Processing symbol: %s...", sym)
        tf_dfs: Dict[str, pd.DataFrame] = {}

        # ── Step 1: Fetch LTF first to obtain the actual time anchor ────────
        ltf_df = fetch_real_candles(sym, timeframe=config.primary_tf, limit=config.target_rows_per_tf)
        if ltf_df.empty:
            logger.warning("[Pipeline] Skipping %s — LTF data unavailable.", sym)
            continue
        tf_dfs[config.primary_tf] = ltf_df

        ltf_anchor: datetime = ltf_df["timestamp"].min().to_pydatetime()
        if ltf_anchor.tzinfo is None:
            ltf_anchor = ltf_anchor.replace(tzinfo=timezone.utc)

        logger.info(
            "[Pipeline] %s LTF anchor: %s  (%d bars)",
            sym, ltf_anchor.isoformat(), len(ltf_df),
        )

        # ── Step 2: Fetch each HTF with extended start (context window fill) ─
        for tf in config.timeframes:
            if tf == config.primary_tf:
                continue

            htf_start = _htf_start_for(
                ltf_anchor, tf, config.htf_context_lookback_bars
            )
            # Limit: we need only enough bars to cover the LTF window + lookback
            htf_bar_count = (
                len(ltf_df)
                + config.htf_context_lookback_bars * 2   # safety buffer
            )
            df_htf = fetch_real_candles(
                sym, timeframe=tf, limit=htf_bar_count, start=htf_start
            )
            if not df_htf.empty:
                tf_dfs[tf] = df_htf
                logger.info(
                    "[Pipeline] %s %s: %d bars (start=%s, covers %d lookback bars before LTF anchor)",
                    sym, tf, len(df_htf), htf_start,
                    len(df_htf[df_htf["timestamp"] < pd.Timestamp(ltf_anchor)]),
                )

        # ── Step 3: Align (HTF shifted +1 interval, no lookahead) ───────────
        aligned_df = align_multi_timeframe_datasets(tf_dfs, primary_tf=config.primary_tf)

        # ── Step 4: Trim to LTF-only window for training ────────────────────
        # After alignment the index still starts at LTF anchor. HTF history
        # only pre-populated the context windows; training rows are LTF rows.
        ltf_anchor_ts = pd.Timestamp(ltf_anchor)
        aligned_df = aligned_df[
            aligned_df["timestamp"] >= ltf_anchor_ts
        ].reset_index(drop=True)
        logger.info(
            "[Pipeline] %s aligned training rows (LTF window only): %d",
            sym, len(aligned_df),
        )

        res = train_sequential_two_tier(sym, aligned_df, config)
        results[sym] = res

    return results
