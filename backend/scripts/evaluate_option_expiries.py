"""
evaluate_option_expiries.py — Benchmark Tool to Find Optimal Option Expiry Horizon on 5m Execution.

Evaluates out-of-sample directional win rates, streaks, and performance across 4 option expiry horizons:
1. 5m  Expiry (N = 1 bar ahead)
2. 15m Expiry (N = 3 bars ahead)
3. 30m Expiry (N = 6 bars ahead)
4. 1h  Expiry (N = 12 bars ahead)
"""

from __future__ import annotations

from datetime import datetime
import sys
import random
import logging
import pprint
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from app.core.ml.real_data_pipeline import (
    RealTrainConfig,
    fetch_real_candles,
    align_multi_timeframe_datasets,
    _htf_start_for,
)
from app.core.options.q_executor import OptionsQExecutor, HTFBiasPackage, AccountContext, ExecutionContext, EXECUTOR_STATE_DIM
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
    vol = float(row.get("volume", row.get("volume_5m", 1000.0)))
    c_price = float(row.get("close", price))
    o_price = float(row.get("open", price))
    if c_price >= o_price:
        buy_vol = vol
        sell_vol = 0.0
    else:
        buy_vol = 0.0
        sell_vol = vol
    return ExecutionContext(
        symbol=symbol,
        current_price=price,
        atr=float(row.get("atr_5m", float(row.get("atr", price * 0.005)))),
        buy_volume=buy_vol,
        sell_volume=sell_vol,
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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("EvalExpiries")

EXPIRY_HORIZONS = {
    "5m (1 bar)": 1,
    "15m (3 bars)": 3,
    "30m (6 bars)": 6,
    "1h (12 bars)": 12,
}

# Maps model-output short labels (HORIZON_LABELS) → EXPIRY_HORIZONS keys
HORIZON_LABEL_MAP = {
    "5m": "5m (1 bar)",
    "15m": "15m (3 bars)",
    "30m": "30m (6 bars)",
    "1h": "1h (12 bars)",
}


def compute_full_context_features(aligned_df: pd.DataFrame) -> pd.DataFrame:
    """Compute and populate all 21 CONTEXT_FEATURE_KEYS (MTF RSI, DXY divergence, SNR distances, confluence)."""
    df = aligned_df.copy()
    close_col = "close_5m" if "close_5m" in df.columns else "close"
    high_col = "high_5m" if "high_5m" in df.columns else "high"
    low_col = "low_5m" if "low_5m" in df.columns else "low"

    asset_close = df[close_col].values.astype(np.float64)

    # 1. Synthetic DXY / Market Benchmark Relative Baseline
    ema_200 = pd.Series(asset_close).ewm(span=200, adjust=False).mean().values
    dxy_synth = (2.0 * ema_200 - asset_close)
    df["dxy_close"] = dxy_synth.astype(np.float32)

    # 2. Fast & Slow Divergence Scales
    fast_w, slow_w = 5, 14
    asset_fast_pct = pd.Series(asset_close).pct_change(fast_w).fillna(0.0).values
    asset_slow_pct = pd.Series(asset_close).pct_change(slow_w).fillna(0.0).values
    dxy_fast_pct   = pd.Series(dxy_synth).pct_change(fast_w).fillna(0.0).values
    dxy_slow_pct   = pd.Series(dxy_synth).pct_change(slow_w).fillna(0.0).values

    def _norm(arr):
        std = float(np.std(arr)) + 1e-8
        return np.clip(arr / (2.0 * std), -1.0, 1.0).astype(np.float32)

    asset_fast_norm = _norm(asset_fast_pct)
    asset_slow_norm = _norm(asset_slow_pct)
    dxy_fast_norm   = _norm(dxy_fast_pct)
    dxy_slow_norm   = _norm(dxy_slow_pct)

    df["asset_slow_norm"] = asset_slow_norm
    df["dxy_slow_norm"]   = dxy_slow_norm
    df["asset_fast_norm"] = asset_fast_norm
    df["dxy_fast_norm"]   = dxy_fast_norm

    slow_diff = asset_slow_norm - dxy_slow_norm
    fast_diff = asset_fast_norm - dxy_fast_norm
    df["slow_diff"] = slow_diff
    df["fast_diff"] = fast_diff

    df["regime_strong_asset"] = ((slow_diff >= 0) & (fast_diff >= 0)).astype(np.float32)
    df["regime_weak_asset"]   = ((slow_diff >= 0) & (fast_diff < 0)).astype(np.float32)
    df["regime_weak_dxy"]     = ((slow_diff < 0)  & (fast_diff >= 0)).astype(np.float32)
    df["regime_strong_dxy"]   = ((slow_diff < 0)  & (fast_diff < 0)).astype(np.float32)

    # 3. MTF RSI (5m, 15m, 1h)
    def _rsi(s, w=14):
        delta = s.diff()
        gain = delta.where(delta > 0, 0.0).rolling(w).mean()
        loss = (-delta.where(delta < 0, 0.0).abs()).rolling(w).mean()
        rs = gain / (loss + 1e-8)
        return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0).values

    rsi_5m  = _rsi(pd.Series(asset_close), 14)
    rsi_15m = _rsi(pd.Series(df["close_15m"] if "close_15m" in df.columns else asset_close), 14)
    rsi_1h  = _rsi(pd.Series(df["close_1h"] if "close_1h" in df.columns else asset_close), 14)

    mtf_rsi_asset = (0.5 * rsi_5m + 0.3 * rsi_15m + 0.2 * rsi_1h).astype(np.float32)
    dxy_rsi_5m    = _rsi(pd.Series(dxy_synth), 14).astype(np.float32)

    df["mtf_rsi_asset"] = mtf_rsi_asset
    df["mtf_rsi_dxy"]   = dxy_rsi_5m
    df["mtf_rsi_diff"]  = (mtf_rsi_asset - dxy_rsi_5m).astype(np.float32)

    # 4. Cross Signals
    rsi_series = pd.Series(rsi_5m)
    dxy_rsi_series = pd.Series(dxy_rsi_5m)
    fast_diff_series = pd.Series(fast_diff)

    cross_index_sig = np.where((rsi_series > 50) & (rsi_series.shift(1) <= 50), 1.0,
                        np.where((rsi_series < 50) & (rsi_series.shift(1) >= 50), -1.0, 0.0))
    cross_dxy_sig   = np.where((dxy_rsi_series > 50) & (dxy_rsi_series.shift(1) <= 50), 1.0,
                        np.where((dxy_rsi_series < 50) & (dxy_rsi_series.shift(1) >= 50), -1.0, 0.0))
    cross_index_dxy = np.where((mtf_rsi_asset > dxy_rsi_5m) & (pd.Series(mtf_rsi_asset).shift(1) <= pd.Series(dxy_rsi_5m).shift(1)), 1.0,
                        np.where((mtf_rsi_asset < dxy_rsi_5m) & (pd.Series(mtf_rsi_asset).shift(1) >= pd.Series(dxy_rsi_5m).shift(1)), -1.0, 0.0))
    cross_dxy_sym   = np.where((fast_diff_series > 0) & (fast_diff_series.shift(1) <= 0), 1.0,
                        np.where((fast_diff_series < 0) & (fast_diff_series.shift(1) >= 0), -1.0, 0.0))

    df["cross_index_signal"] = cross_index_sig.astype(np.float32)
    df["cross_dxy_signal"]   = cross_dxy_sig.astype(np.float32)
    df["cross_index_dxy"]    = cross_index_dxy.astype(np.float32)
    df["cross_dxy_symbol"]   = cross_dxy_sym.astype(np.float32)

    # 5. Dynamic SNR distances and MTF SNR confluence
    atr = (df[high_col] - df[low_col]).rolling(14).mean().fillna(df[close_col] * 0.005).values
    supp_q = df[low_col].rolling(100, min_periods=10).quantile(0.20).fillna(df[low_col]).values
    res_q  = df[high_col].rolling(100, min_periods=10).quantile(0.80).fillna(df[high_col]).values

    df["snr_dist_support"]    = np.clip((asset_close - supp_q) / (atr + 1e-8), 0.0, 10.0).astype(np.float32)
    df["snr_dist_resistance"] = np.clip((res_q - asset_close) / (atr + 1e-8), 0.0, 10.0).astype(np.float32)

    low_15m = df["low_15m"].values if "low_15m" in df.columns else df[low_col].values
    high_15m = df["high_15m"].values if "high_15m" in df.columns else df[high_col].values
    dist_supp_15m = np.abs(asset_close - low_15m) / (atr + 1e-8)
    dist_res_15m  = np.abs(high_15m - asset_close) / (atr + 1e-8)

    confluence = ((df["snr_dist_support"].values <= 1.0).astype(int) +
                  (df["snr_dist_resistance"].values <= 1.0).astype(int) +
                  (dist_supp_15m <= 1.0).astype(int) +
                  (dist_res_15m <= 1.0).astype(int))
    df["mtf_snr_confluence"] = confluence.astype(np.float32)

    return df


def evaluate_expiries_for_symbol(symbol: str = "GLD", limit: int = 40000, framework: str = "keras"):
    print(f"\n==================================================================================")
    print(f"      EVALUATING OPTION EXPIRY HORIZONS FOR SYMBOL: {symbol} (5m Execution | Framework: {framework.upper()})")
    print(f"==================================================================================")

    # 1. Fetch & Align real market data with expanded dataset
    logger.info("Fetching real market bars for %s (%d bars limit)...", symbol, limit)
    ltf_df = fetch_real_candles(symbol, timeframe="5m", limit=limit)
    if ltf_df.empty:
        print(f"Error: Could not fetch 5m candles for {symbol}")
        return

    ltf_anchor = ltf_df["timestamp"].min().to_pydatetime()
    tf_dfs = {"5m": ltf_df}

    import signal as _signal

    def _fetch_with_timeout(symbol, tf, limit, start, timeout_sec=90):
        """Fetch HTF candles with a hard timeout to prevent indefinite hangs."""
        result = [None]
        def _handler(signum, frame):
            raise TimeoutError(f"[DataFetch] Timed out fetching {tf} bars for {symbol}")
        prev = _signal.signal(_signal.SIGALRM, _handler)
        _signal.alarm(timeout_sec)
        try:
            result[0] = fetch_real_candles(symbol, timeframe=tf, limit=limit, start=start)
        except TimeoutError as e:
            logger.warning("%s — skipping timeframe.", e)
            result[0] = None
        finally:
            _signal.alarm(0)
            _signal.signal(_signal.SIGALRM, prev)
        return result[0]

    for tf in ["15m", "1h", "4h", "1d"]:
        start_str = _htf_start_for(ltf_anchor, tf, lookback_bars=1000)
        df_htf = _fetch_with_timeout(symbol, tf, limit, start_str)
        if df_htf is not None and not df_htf.empty:
            tf_dfs[tf] = df_htf

    aligned_df = align_multi_timeframe_datasets(tf_dfs, primary_tf="5m")
    aligned_df = aligned_df[aligned_df["timestamp"] >= pd.Timestamp(ltf_anchor)].reset_index(drop=True)

    # ── Compute & Inject All 21 CONTEXT_FEATURE_KEYS (MTF RSI, DXY Divergence, SNR Confluence) ──
    aligned_df = compute_full_context_features(aligned_df)

    close_col = "close_5m" if "close_5m" in aligned_df.columns else "close"
    open_col  = "open_5m"  if "open_5m"  in aligned_df.columns else "open"
    high_col  = "high_5m"  if "high_5m"  in aligned_df.columns else "high"
    low_col   = "low_5m"   if "low_5m"   in aligned_df.columns else "low"
    vol_col   = "volume_5m" if "volume_5m" in aligned_df.columns else "volume"

    # ── 2. Run 200+ Technical Indicators & Log Enrichment ──
    logger.info("[TI Enrichment] Enriching aligned dataset (%d rows) with 200+ Technical Indicators...", len(aligned_df))
    try:
        from app.core.analysis.technical_indicators import TechnicalIndicators, IndicatorConfig
        from app.core.ml.ti_meta_features import TI_NUMERIC_FEATURE_KEYS, CONTEXT_FEATURE_KEYS

        ti_calc = TechnicalIndicators(IndicatorConfig())
        ta_input = aligned_df.rename(columns={
            open_col: "Open", high_col: "High", low_col: "Low", close_col: "Close", vol_col: "Volume"
        })
        ti_enriched = ti_calc.calculate_all_indicators(ta_input, mode="training")

        ti_added_count = 0
        for col in TI_NUMERIC_FEATURE_KEYS:
            if col in ti_enriched.columns:
                aligned_df[col] = ti_enriched[col].astype(np.float32)
                ti_added_count += 1
            else:
                aligned_df[col] = 0.0

        logger.info(
            "✅ [TI Enrichment] Complete! Enriched dataset with %d Technical Indicators + %d Context Features (%d total columns).",
            ti_added_count, len(CONTEXT_FEATURE_KEYS), len(aligned_df.columns)
        )
    except Exception as ti_err:
        logger.warning("⚠ [TI Enrichment] Warning during TI calculation: %s", ti_err)

    # ── 3. Multi-Horizon Targets & Directional Bias Labeling (5m, 15m, 30m, 1h) ──
    aligned_df["forward_move_1"]  = aligned_df[close_col].shift(-1) - aligned_df[close_col]   # 5m
    aligned_df["forward_move_3"]  = aligned_df[close_col].shift(-3) - aligned_df[close_col]   # 15m
    aligned_df["forward_move_6"]  = aligned_df[close_col].shift(-6) - aligned_df[close_col]   # 30m
    aligned_df["forward_move_12"] = aligned_df[close_col].shift(-12) - aligned_df[close_col]  # 1h

    aligned_df["target_dir_5m"]  = (aligned_df["forward_move_1"]  > 0).astype(int)
    aligned_df["target_dir_15m"] = (aligned_df["forward_move_3"]  > 0).astype(int)
    aligned_df["target_dir_30m"] = (aligned_df["forward_move_6"]  > 0).astype(int)
    aligned_df["target_dir_1h"]  = (aligned_df["forward_move_12"] > 0).astype(int)

    aligned_df.dropna(subset=["forward_move_12"], inplace=True)

    # 70 / 15 / 15 Train / Validation / Test Split (Strict chronological order)
    n_total = len(aligned_df)
    train_idx = int(n_total * 0.70)
    val_idx = int(n_total * 0.85)

    train_df = aligned_df.iloc[:train_idx].reset_index(drop=True)
    val_df   = aligned_df.iloc[train_idx:val_idx].reset_index(drop=True)
    test_df  = aligned_df.iloc[val_idx:].reset_index(drop=True)

    print(f"Dataset Split -> Total: {n_total} | Train (70%): {len(train_df)} | Val (15%): {len(val_df)} | Test (15%): {len(test_df)}")

    # ── 4. Train-Only Scaler Fitting (Preventing Data Leakage) ──
    try:
        from app.core.ml.signal_meta_learner import FeatureScaler
        scaler = FeatureScaler()
        # Collect numeric feature columns
        num_feature_cols = [c for c in train_df.columns if c not in ("timestamp", "Time") and np.issubdtype(train_df[c].dtype, np.number)]
        scaler.fit(train_df[num_feature_cols].values)
        meta_learner.scaler = scaler
        logger.info("✅ [Scaler] FeatureScaler fitted EXCLUSIVELY on Train split (%d rows, %d feature columns).", len(train_df), len(num_feature_cols))
    except Exception as scaler_err:
        logger.warning("⚠ [Scaler] Notice during scaler fitting: %s", scaler_err)

    # ── 5. Automatic Dataset Export & Zipping for Kaggle (Train, Val, Test, Full) ──
    try:
        import os
        import zipfile
        os.makedirs("data", exist_ok=True)

        csv_train_path = "data/train_40k.csv"
        csv_val_path   = "data/val_40k.csv"
        csv_test_path  = "data/test_40k.csv"
        csv_full_path  = "data/full_40k.csv"
        zip_export_path = "data/axe_meta_dataset.zip"

        train_df.to_csv(csv_train_path, index=False)
        val_df.to_csv(csv_val_path, index=False)
        test_df.to_csv(csv_test_path, index=False)
        aligned_df.to_csv(csv_full_path, index=False)

        with zipfile.ZipFile(zip_export_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(csv_train_path, arcname="train_40k.csv")
            zipf.write(csv_val_path,   arcname="val_40k.csv")
            zipf.write(csv_test_path,  arcname="test_40k.csv")
            zipf.write(csv_full_path,  arcname="full_40k.csv")

        zip_mb = os.path.getsize(zip_export_path) / (1024 * 1024)
        print(f"📦 [DataExport] Exported & zipped all enriched splits (train/val/test/full) to: {zip_export_path} ({zip_mb:.2f} MB)")
    except Exception as exc:
        print(f"⚠ [DataExport] Dataset zip export warning: {exc}")

    # ── Training Data Diagnostics ──────────────────────────────────────────────
    bullish_count = int((train_df["forward_move_12"] > 0).sum())
    bearish_count = int((train_df["forward_move_12"] <= 0).sum())
    bull_pct = 100.0 * bullish_count / len(train_df)
    print(f"\n[TrainData] Direction Split (12-bar): Bullish={bullish_count} ({bull_pct:.1f}%) | Bearish={bearish_count} ({100-bull_pct:.1f}%)")

    # DXY / context feature coverage audit
    dxy_cols = [c for c in aligned_df.columns if "dxy" in c.lower() or "dollar" in c.lower()]
    context_cols_present = [c for c in aligned_df.columns if any(k in c for k in ["cross_", "regime_", "mtf_rsi", "snr_dist", "asset_slow", "asset_fast"])]
    print(f"[TrainData] DXY columns in dataset ({len(dxy_cols)}): {dxy_cols if dxy_cols else '⚠ NONE — DXY not aligned into dataset'}")
    print(f"[TrainData] Context feature columns ({len(context_cols_present)}): {context_cols_present[:8]}{'...' if len(context_cols_present) > 8 else ''}")

    # Forward move distribution summary for each horizon
    for h_bars, h_label in zip([1, 3, 6, 12], ["5m", "15m", "30m", "1h"]):
        fwd = aligned_df[close_col].shift(-h_bars) - aligned_df[close_col]
        fwd_bull = int((fwd > 0).sum())
        fwd_bear = int((fwd <= 0).sum())
        fwd_mean = float(fwd.mean())
        fwd_std = float(fwd.std())
        print(f"[TrainData] Horizon {h_label:>4} ({h_bars:>2} bars): Bull={fwd_bull:>5} ({100*fwd_bull/(fwd_bull+fwd_bear):.1f}%) | Bear={fwd_bear:>5} | mean={fwd_mean:+.4f} | std={fwd_std:.4f}")

    print(f"[TrainData] Feature vector sample size (SIGNAL_META_FEATURE_COUNT): {SIGNAL_META_FEATURE_COUNT}")
    sample_f = _row_to_feature_dict(train_df.iloc[0], close_col, vol_col)
    print(f"[TrainData] Feature dict keys from first training row ({len(sample_f)} keys): {list(sample_f.keys())[:10]}... ({'DXY present' if any('dxy' in k.lower() for k in sample_f) else '⚠ DXY MISSING from feature_dict'})")
    print()

    # 2. Train Two-Tier Ensemble Learners once on Train Set
    if framework == "keras":
        from app.core.ml.keras_signal_meta_learner import KerasSignalMetaLearner
        from app.core.options.keras_trade_executor import KerasTradeExecutor
        meta_learner = KerasSignalMetaLearner(num_features=238, lookback_bars=48, replay_capacity=20000)
        q_executor   = KerasTradeExecutor(seq_len=1, n_features=EXECUTOR_STATE_DIM)
    else:
        meta_learner = OnlineSignalMetaLearner(input_dim=SIGNAL_META_FEATURE_COUNT, replay_capacity=20000)
        q_executor = OptionsQExecutor(device="cpu")

    zone_manager = ZoneSnapshotManager(max_snapshots=20)
    account = AccountContext()

    from app.core.analysis.support_resistance import (
        detect_snr_levels_sequential,
        create_clustered_zones_sequential,
    )

    def update_real_snr_snapshot(
        df_full: pd.DataFrame,
        up_to_idx: int,
        zm: ZoneSnapshotManager,
        timeframe: str = "15m",
        lookback_period: int = 500,
    ):
        """
        CRITICAL (NO LOOKAHEAD LEAKAGE):
        Detect real S&R levels and volume profiles using ONLY historical price data up to `up_to_idx`.
        """
        if up_to_idx < 20:
            return

        df_slice = df_full.iloc[max(0, up_to_idx - lookback_period): up_to_idx + 1].copy()
        df_slice = df_slice.rename(columns={
            high_col: "High", low_col: "Low", close_col: "Close", open_col: "Open", vol_col: "Volume"
        })

        if "High" not in df_slice.columns or "Low" not in df_slice.columns:
            return

        levels = detect_snr_levels_sequential(
            price_data=df_slice,
            up_to_index=len(df_slice) - 1,
            lookback_period=min(lookback_period, len(df_slice) - 1),
        )

        if not levels:
            return

        raw_zones = create_clustered_zones_sequential(
            levels=levels,
            price_data_slice=df_slice,
            n_clusters=min(8, max(3, len(levels))),
        )

        if raw_zones:
            ts = df_slice.index[-1] if hasattr(df_slice.index[-1], "to_pydatetime") else None
            zm.add_snapshot(
                snapshot_id=f"snap_{up_to_idx}",
                timeframe=timeframe,
                zones_raw=raw_zones,
                timestamp=ts if isinstance(ts, datetime) else None,
            )

    # ── Phase 1: Meta-Learner Training (100% Full-Set Systematic Epoch Sweep) ──
    META_EPOCHS = 50
    total_train_bars = len(train_df) - 13
    total_expected_steps = (total_train_bars // 64) * META_EPOCHS
    print(f"\n[Phase 1] Training Meta-Learner (100% Full-Set Sweep: {META_EPOCHS} Epochs across {total_train_bars} bars)...")

    _meta_loss_acc = 0.0
    _meta_loss_q_acc = 0.0
    _meta_loss_str_acc = 0.0
    _meta_loss_pips_acc = 0.0
    _meta_loss_risk_acc = 0.0
    _meta_loss_liq_acc = 0.0
    _meta_loss_rev_acc = 0.0
    _meta_loss_aux1_acc = 0.0
    _meta_loss_aux2_acc = 0.0
    global_step = 0
    LOG_EVERY = 50

    best_meta_loss = float("inf")
    best_meta_weights = None

    for ep in range(META_EPOCHS):
        # Epoch 1 runs sequentially (warm-up); subsequent epochs are shuffled.
        # Each sample's 1,000-bar context window is still causally built internally,
        # so shuffling presentation order is safe and breaks gradient autocorrelation.
        epoch_indices = list(range(total_train_bars))
        if ep > 0:
            random.shuffle(epoch_indices)

        for idx in epoch_indices:
            row = train_df.iloc[idx]
            fut_highs = train_df[high_col].iloc[idx+1:idx+13].values
            fut_lows = train_df[low_col].iloc[idx+1:idx+13].values
            fut_closes = train_df[close_col].iloc[idx+1:idx+13].values

            f_dict = _row_to_feature_dict(row, close_col, vol_col)
            meta_learner.record_experience(
                feature_dict=f_dict,
                signal_id=f"sig_{ep}_{idx}",
                symbol=symbol,
                direction="bullish" if row["forward_move_12"] > 0 else "bearish",
                entry_price=float(row[close_col]),
                future_highs=fut_highs,
                future_lows=fut_lows,
                future_closes=fut_closes,
            )

            # Perform gradient update step every 64 bars
            if (idx + 1) % 64 == 0:
                global_step += 1
                train_metrics = meta_learner.train_step(batch_size=64)
                if train_metrics.get("loss", 0) > 0:
                    _meta_loss_acc += train_metrics["loss"]
                    _meta_loss_q_acc += train_metrics.get("loss_q", 0)
                    _meta_loss_str_acc += train_metrics.get("loss_strength", 0)
                    _meta_loss_pips_acc += train_metrics.get("loss_pips", 0)
                    _meta_loss_risk_acc += train_metrics.get("loss_risk", 0)
                    _meta_loss_liq_acc += train_metrics.get("loss_liquidity", 0)
                    _meta_loss_rev_acc += train_metrics.get("loss_reversal", 0)
                    _meta_loss_aux1_acc += train_metrics.get("loss_aux1", 0)
                    _meta_loss_aux2_acc += train_metrics.get("loss_aux2", 0)

                if global_step % LOG_EVERY == 0:
                    n_samples = max(1, LOG_EVERY)
                    avg_loss = _meta_loss_acc / n_samples
                    avg_q = _meta_loss_q_acc / n_samples
                    avg_str = _meta_loss_str_acc / n_samples
                    avg_pips = _meta_loss_pips_acc / n_samples
                    avg_risk = _meta_loss_risk_acc / n_samples
                    avg_liq = _meta_loss_liq_acc / n_samples
                    avg_rev = _meta_loss_rev_acc / n_samples
                    avg_aux1 = _meta_loss_aux1_acc / n_samples
                    avg_aux2 = _meta_loss_aux2_acc / n_samples
                    buf = train_metrics.get("buffer_size", "?")

                    # Best Meta Weights Checkpoint Tracking
                    # Warmup guard: skip the first 500 steps so cold-start noise is never
                    # incorrectly saved as the "best" weights.
                    if global_step >= 500 and avg_loss < best_meta_loss:
                        best_meta_loss = avg_loss
                        if hasattr(meta_learner, "get_weights"):
                            best_meta_weights = meta_learner.get_weights()
                        elif hasattr(meta_learner, "network"):
                            best_meta_weights = {k: v.cpu().clone() for k, v in meta_learner.network.state_dict().items()}

                    cur_lr = meta_learner.optimizer.param_groups[0]["lr"] if hasattr(meta_learner, "optimizer") else 0.0
                    print(
                        f"  [Meta Epoch {ep+1:>2}/{META_EPOCHS} | Step {global_step:>5}/{total_expected_steps}] total={avg_loss:.4e} (Best={best_meta_loss:.4e}) | "
                        f"q={avg_q:.4e} | str={avg_str:.4e} | pips={avg_pips:.4e} | "
                        f"risk={avg_risk:.4e} | rev={avg_rev:.4e} | lr={cur_lr:.2e} | buf={buf}"
                    )
                    _meta_loss_acc = _meta_loss_q_acc = _meta_loss_str_acc = 0.0
                    _meta_loss_pips_acc = _meta_loss_risk_acc = _meta_loss_liq_acc = 0.0
                    _meta_loss_rev_acc = _meta_loss_aux1_acc = _meta_loss_aux2_acc = 0.0

    # Restore best Meta-Learner weights before buffer generation
    if best_meta_weights is not None:
        if hasattr(meta_learner, "set_weights"):
            meta_learner.set_weights(best_meta_weights)
        elif hasattr(meta_learner, "network"):
            meta_learner.network.load_state_dict(best_meta_weights)
        print(f"\n[Phase 1 Complete] Restored best Meta-Learner checkpoint (loss = {best_meta_loss:.4e}).")

    # ── Phase 1b: Meta-Learner Inference Pass → Q-Learner Buffer Generation ──
    # The meta-learner makes sequential passes over train_df to construct quality transitions.
    # Adaptive quality gating ensures exactly Q_BUFFER_TARGET (10,000) high-conviction memories are collected.
    Q_BUFFER_TARGET = 10_000
    print(f"\n[Phase 1b] Meta-Learner inference pass → generating target {Q_BUFFER_TARGET} Q-Learner transitions from train_df...")

    q_buf_filled = 0
    q_buf_attempts = 0
    zone_manager_gen = ZoneSnapshotManager()

    # Multi-pass threshold schedule
    threshold_schedule = [
        (0.60, 0.003),   # Pass 1: High conviction (strength >= 0.60 or move >= 30 pips)
        (0.55, 0.0015),  # Pass 2: Medium conviction (strength >= 0.55 or move >= 15 pips)
        (0.50, 0.0005),  # Pass 3: Moderate conviction
        (0.00, 0.0000),  # Pass 4: Full dataset sampling baseline
    ]

    for pass_idx, (min_ms, min_fwd) in enumerate(threshold_schedule):
        if q_buf_filled >= Q_BUFFER_TARGET:
            break
        print(f"  [Phase 1b Pass {pass_idx+1}] Sampling with min_ms={min_ms}, min_fwd={min_fwd} (Current Buffer: {q_buf_filled}/{Q_BUFFER_TARGET})...")

        for idx in range(len(train_df) - 13):
            if q_buf_filled >= Q_BUFFER_TARGET:
                break
            q_buf_attempts += 1

            row = train_df.iloc[idx]
            cp = float(row[close_col])
            
            # Evaluate max forward move across multi-horizon window (1 to 12 bars)
            max_fwd_cp = max([float(train_df.iloc[idx + k][close_col]) for k in (1, 3, 6, 12)])
            min_fwd_cp = min([float(train_df.iloc[idx + k][close_col]) for k in (1, 3, 6, 12)])
            fwd_pct = (max_fwd_cp - cp) / (cp + 1e-8) if abs(max_fwd_cp - cp) > abs(min_fwd_cp - cp) else (min_fwd_cp - cp) / (cp + 1e-8)

            if idx == 0 or idx % 15 == 0 or len(zone_manager_gen.get_active_zones()) == 0:
                update_real_snr_snapshot(train_df, idx, zone_manager_gen)
            zone_manager_gen.update_invalidation(cp, float(row[high_col]), float(row[low_col]))

            f_dict = _row_to_feature_dict(row, close_col, vol_col)
            pred = meta_learner.predict(f_dict)
            ms = float(pred.get("signal_strength", 0.5))

            if ms < min_ms and abs(fwd_pct) < min_fwd:
                continue

            bias = HTFBiasPackage(
                direction="bullish" if ms > 0.5 else "bearish",
                strength=ms,
                reversal_prob=float(pred.get("reversal_prob", 0.2)),
                q_value=float(pred.get("q_value", 0.5)),
                expected_mfe_pips=float(pred.get("expected_mfe_pips", 50.0)),
                expected_mae_pips=float(pred.get("expected_mae_pips", 15.0)),
                horizon_strengths=pred.get("horizon_strengths", [0.5, 0.5, 0.5, 0.5]),
                optimal_horizon_idx=int(pred.get("optimal_horizon_idx", 2)),
                recommended_expiry=str(pred.get("recommended_expiry", "30m")),
            )
            ctx = _make_exec_ctx(symbol, cp, dict(row))
            state = q_executor.build_state_vector(bias, account, ctx, zone_manager_gen)
            mask_engine = HardActionMask()
            action_mask = mask_engine.get_action_mask(
                current_price=cp, atr=ctx.atr, zone_manager=zone_manager_gen,
                buy_volume=ctx.buy_volume, sell_volume=ctx.sell_volume, has_open_position=False,
            )

            if action_mask[1] == 1 and bias.direction == "bullish" and fwd_pct >= 0.0015:
                oracle_action = 1
            elif action_mask[2] == 1 and bias.direction == "bearish" and fwd_pct <= -0.0015:
                oracle_action = 2
            else:
                oracle_action = 0

            reward = q_executor.calculate_executor_reward(
                action=oracle_action, action_mask=action_mask, pnl_pct=fwd_pct,
                max_drawdown_exposed=0.005, forward_move_pct=fwd_pct, htf_bias=bias,
            )

            next_row = train_df.iloc[idx + 1]
            next_cp = float(next_row[close_col])
            next_ctx = _make_exec_ctx(symbol, next_cp, dict(next_row))
            next_state = q_executor.build_state_vector(bias, account, next_ctx, zone_manager_gen)

            q_executor.record_transition(state, oracle_action, reward, next_state, False, action_mask)
            q_buf_filled += 1

            if q_buf_filled % 2000 == 0:
                print(f"  [Phase 1b] Buffer: {q_buf_filled}/{Q_BUFFER_TARGET} transitions | Scanned: {q_buf_attempts} bars")

    print(f"  [Phase 1b] Complete: {q_buf_filled} transitions generated from {q_buf_attempts} scanned bars.")

    # ── Phase 2: Q-Learner Training with Episode Checks & Performance Threshold Early Stopping ──
    MAX_Q_TRAIN_STEPS = 2000
    EVAL_EVERY_STEPS = 150
    TARGET_WIN_RATE_THRESHOLD = 60.0  # Stop training early if validation episode win rate hits 60%

    print(f"\n[Phase 2] Training Q-Learner (Episodes + Early Stopping Threshold at {TARGET_WIN_RATE_THRESHOLD}% Win Rate)...")

    _q_loss_acc = 0.0
    action_dist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    best_eval_wr = 0.0
    best_q_weights = None
    episode_counter = 0

    for step in range(MAX_Q_TRAIN_STEPS):
        q_loss = q_executor.train_step(batch_size=128)
        if q_loss is not None:
            _q_loss_acc += float(q_loss)

        # Sample live bar for action distribution tracking
        idx = np.random.randint(0, len(train_df) - 2)
        row = train_df.iloc[idx]
        cur_price = float(row[close_col])
        exec_ctx = _make_exec_ctx(symbol, cur_price, dict(row))
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
            horizon_strengths=pred.get("horizon_strengths", [0.5, 0.5, 0.5, 0.5]),
            optimal_horizon_idx=int(pred.get("optimal_horizon_idx", 2)),
            recommended_expiry=str(pred.get("recommended_expiry", "30m")),
        )
        state = q_executor.build_state_vector(htf_bias, account, exec_ctx, zone_manager)
        mask_engine = HardActionMask()
        action_mask = mask_engine.get_action_mask(
            current_price=cur_price, atr=exec_ctx.atr, zone_manager=zone_manager,
            buy_volume=exec_ctx.buy_volume, sell_volume=exec_ctx.sell_volume, has_open_position=False,
        )
        action = q_executor.select_action(state, action_mask)
        action_dist[action] = action_dist.get(action, 0) + 1

        # ── Episode Evaluation Checkpoint ─────────────────────────────────────
        if (step + 1) % EVAL_EVERY_STEPS == 0:
            episode_counter += 1
            avg_q_loss = _q_loss_acc / EVAL_EVERY_STEPS
            act_names = {0: "WAIT", 1: "CALL", 2: "PUT", 3: "TP_HALF", 4: "CLOSE"}
            dist_str = " | ".join(f"{act_names.get(a, a)}={c}" for a, c in sorted(action_dist.items()) if c > 0)

            # Fast 300-bar Episode Evaluation on Validation Slice
            val_wins, val_total = 0, 0
            val_indices = np.random.choice(len(train_df) - 6, size=min(300, len(train_df) - 6), replace=False)
            for v_idx in val_indices:
                v_row = train_df.iloc[v_idx]
                v_exp_row = train_df.iloc[v_idx + 3]
                v_cp = float(v_row[close_col])
                v_exp_cp = float(v_exp_row[close_col])

                v_fdict = _row_to_feature_dict(v_row, close_col, vol_col)
                v_pred = meta_learner.predict(v_fdict)
                v_ms = float(v_pred.get("signal_strength", 0.5))
                v_bias = HTFBiasPackage(
                    direction="bullish" if v_ms > 0.5 else "bearish", strength=v_ms,
                    reversal_prob=float(v_pred.get("reversal_prob", 0.2)), q_value=float(v_pred.get("q_value", 0.5)),
                    expected_mfe_pips=float(v_pred.get("expected_mfe_pips", 50.0)), expected_mae_pips=float(v_pred.get("expected_mae_pips", 15.0)),
                    horizon_strengths=v_pred.get("horizon_strengths", [0.5, 0.5, 0.5, 0.5]),
                    optimal_horizon_idx=int(v_pred.get("optimal_horizon_idx", 2)), recommended_expiry=str(v_pred.get("recommended_expiry", "30m")),
                )
                v_ctx = _make_exec_ctx(symbol, v_cp, dict(v_row))
                v_st = q_executor.build_state_vector(v_bias, account, v_ctx, zone_manager)
                v_mask = mask_engine.get_action_mask(
                    current_price=v_cp, atr=v_ctx.atr, zone_manager=zone_manager,
                    buy_volume=v_ctx.buy_volume, sell_volume=v_ctx.sell_volume, has_open_position=False,
                )
                v_act = q_executor.select_action(v_st, v_mask, eval_mode=True)
                if v_act == 1:
                    val_total += 1
                    if v_exp_cp > v_cp: val_wins += 1
                elif v_act == 2:
                    val_total += 1
                    if v_exp_cp < v_cp: val_wins += 1

            val_wr = (val_wins / val_total * 100.0) if val_total > 0 else 0.0

            # Save best Q-Learner weights when win rate improves
            if val_wr > best_eval_wr:
                best_eval_wr = val_wr
                if hasattr(q_executor, "get_weights"):
                    best_q_weights = q_executor.get_weights()
                elif hasattr(q_executor, "policy_net"):
                    best_q_weights = {k: v.cpu().clone() for k, v in q_executor.policy_net.state_dict().items()}

            print(
                f"  [Q Episode {episode_counter:>2} | Step {step+1:>4}/{MAX_Q_TRAIN_STEPS}] "
                f"avg_loss={avg_q_loss:.4e} | Val Trades={val_total} | Val WinRate={val_wr:.1f}% (Best={best_eval_wr:.1f}%) | actions[{dist_str}]"
            )

            _q_loss_acc = 0.0
            action_dist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

            # Early Stopping Check: If validation win rate hits target threshold
            if val_wr >= TARGET_WIN_RATE_THRESHOLD and step >= 450:
                print(f"  [Q Early Stopping] Target performance threshold reached ({val_wr:.1f}% >= {TARGET_WIN_RATE_THRESHOLD}%) at step {step+1}. Stopping Q-training.")
                break

    # Restore best Q-Learner weights before out-of-sample evaluation
    if best_q_weights is not None:
        if hasattr(q_executor, "set_weights"):
            q_executor.set_weights(best_q_weights)
        elif hasattr(q_executor, "policy_net"):
            q_executor.policy_net.load_state_dict(best_q_weights)
        print(f"\n[Phase 2 Complete] Restored best Q-Learner checkpoint (Best Val WinRate = {best_eval_wr:.1f}%).")


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

        # Learning curve tracking
        trade_outcomes: list[int] = []      # 1=win, 0=loss, in chronological order
        win_streaks: list[int] = []         # all win streak lengths
        loss_streaks: list[int] = []        # all loss streak lengths
        _cur_ws, _cur_ls = 0, 0

        for idx in range(len(test_df) - lookahead_bars):
            row = test_df.iloc[idx]
            expiry_row = test_df.iloc[idx + lookahead_bars]

            cur_price = float(row[close_col])
            expiry_price = float(expiry_row[close_col])

            has_open_position = (idx < open_trade_until_idx)

            if idx == 0 or idx % 15 == 0 or len(zone_manager.get_active_zones()) == 0:
                update_real_snr_snapshot(test_df, idx, zone_manager)
            zone_manager.update_invalidation(cur_price, float(row[high_col]), float(row[low_col]))

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
                horizon_strengths=pred.get("horizon_strengths", [0.5, 0.5, 0.5, 0.5]),
                optimal_horizon_idx=int(pred.get("optimal_horizon_idx", 2)),
                recommended_expiry=str(pred.get("recommended_expiry", "30m")),
            )

            account.open_position_type = "CALL" if has_open_position else None
            exec_ctx = _make_exec_ctx(symbol, cur_price, dict(row))
            state = q_executor.build_state_vector(htf_bias, account, exec_ctx, zone_manager)
            mask_engine = HardActionMask()
            action_mask = mask_engine.get_action_mask(
                current_price=cur_price, atr=exec_ctx.atr, zone_manager=zone_manager, buy_volume=exec_ctx.buy_volume, sell_volume=exec_ctx.sell_volume, has_open_position=has_open_position,
            )
            # HardActionMask is authoritative — WAIT when zone/volume says no.
            # Never force-unmask based on HTF bias strength.

            action = q_executor.select_action(state, action_mask, eval_mode=True)

            outcome = None
            if action == 1 and not has_open_position:  # BUY_CALL
                open_trade_until_idx = idx + lookahead_bars
                outcome = 1 if expiry_price > cur_price else 0
            elif action == 2 and not has_open_position:  # BUY_PUT
                open_trade_until_idx = idx + lookahead_bars
                outcome = 1 if expiry_price < cur_price else 0
            else:
                waits += 1

            if outcome is not None:
                trade_outcomes.append(outcome)
                if outcome == 1:
                    wins += 1
                    cur_w_streak += 1; cur_l_streak = 0
                    _cur_ws += 1
                    if _cur_ls > 0:
                        loss_streaks.append(_cur_ls)
                        _cur_ls = 0
                else:
                    losses += 1
                    cur_l_streak += 1; cur_w_streak = 0
                    _cur_ls += 1
                    if _cur_ws > 0:
                        win_streaks.append(_cur_ws)
                        _cur_ws = 0

            max_w_streak = max(max_w_streak, cur_w_streak)
            max_l_streak = max(max_l_streak, cur_l_streak)

        # Flush any open streaks at end
        if _cur_ws > 0: win_streaks.append(_cur_ws)
        if _cur_ls > 0: loss_streaks.append(_cur_ls)

        total_trades = wins + losses
        win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0

        # ── Quartile Learning Curve ──────────────────────────────────────────
        q_size = max(1, total_trades // 4)
        quartile_rates = []
        for q in range(4):
            seg = trade_outcomes[q * q_size: (q + 1) * q_size]
            qr = (sum(seg) / len(seg) * 100.0) if seg else 0.0
            quartile_rates.append(round(qr, 1))

        # ── Rolling 20-Trade Win Rate (min/mean/max) ─────────────────────────
        window = 20
        rolling_wrs = []
        for i in range(len(trade_outcomes) - window + 1):
            seg = trade_outcomes[i: i + window]
            rolling_wrs.append(sum(seg) / window * 100.0)
        rolling_min = round(min(rolling_wrs), 1) if rolling_wrs else 0.0
        rolling_mean = round(sum(rolling_wrs) / len(rolling_wrs), 1) if rolling_wrs else 0.0
        rolling_max = round(max(rolling_wrs), 1) if rolling_wrs else 0.0

        # ── Streak Distribution Summary ──────────────────────────────────────
        avg_win_streak = round(sum(win_streaks) / len(win_streaks), 2) if win_streaks else 0.0
        avg_loss_streak = round(sum(loss_streaks) / len(loss_streaks), 2) if loss_streaks else 0.0
        streak_ratio = round(avg_win_streak / avg_loss_streak, 2) if avg_loss_streak > 0 else float("inf")

        # Win rate in last 25% vs first 25% (drift signal)
        late_wr = quartile_rates[3] if len(quartile_rates) == 4 else 0.0
        early_wr = quartile_rates[0] if len(quartile_rates) == 4 else 0.0
        drift = round(late_wr - early_wr, 1)
        drift_str = f"+{drift}%" if drift >= 0 else f"{drift}%"

        print(f"{exp_label:<18} | {total_trades:<8} | {wins:<6} | {losses:<8} | {waits:<8} | {win_rate:<10.2f} | W:{max_w_streak} / L:{max_l_streak}")
        print(f"  {'':>18}   Quartile WR [Q1→Q4]: {quartile_rates}  |  Late vs Early Drift: {drift_str}")
        print(f"  {'':>18}   Rolling-20 WR: min={rolling_min}% avg={rolling_mean}% max={rolling_max}%")
        print(f"  {'':>18}   Avg Streak: W={avg_win_streak} bars / L={avg_loss_streak} bars  (streak ratio {streak_ratio}x)")
        print()

        results_table[exp_label] = {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "waits": waits,
            "win_rate_pct": round(win_rate, 2),
            "max_win_streak": max_w_streak,
            "max_loss_streak": max_l_streak,
            "quartile_win_rates": quartile_rates,
            "rolling20_min": rolling_min,
            "rolling20_mean": rolling_mean,
            "rolling20_max": rolling_max,
            "avg_win_streak": avg_win_streak,
            "avg_loss_streak": avg_loss_streak,
            "streak_ratio": streak_ratio,
            "late_vs_early_drift_pct": drift,
        }

    # 4. Collective Multi-Horizon Concurrent Portfolio Simulation
    print("\n[Phase 4] Evaluating Collective Multi-Horizon Concurrent Portfolio Performance:")
    print("Policy: Max 1 active trade per horizon concurrently from shared account.")
    print("-" * 92)

    active_horizon_until = {exp: -1 for exp in EXPIRY_HORIZONS}
    portfolio_outcomes: list[int] = []
    portfolio_win_streaks: list[int] = []
    portfolio_loss_streaks: list[int] = []
    horizon_trade_counts = {exp: {"wins": 0, "losses": 0, "total": 0} for exp in EXPIRY_HORIZONS}

    cur_p_w_streak, cur_p_l_streak = 0, 0
    max_p_w_streak, max_p_l_streak = 0, 0
    _p_cur_ws, _p_cur_ls = 0, 0
    recommended_matches = 0

    max_lookahead = max(EXPIRY_HORIZONS.values())

    for idx in range(len(test_df) - max_lookahead):
        row = test_df.iloc[idx]
        cur_price = float(row[close_col])

        if idx == 0 or idx % 15 == 0 or len(zone_manager.get_active_zones()) == 0:
            update_real_snr_snapshot(test_df, idx, zone_manager)
        zone_manager.update_invalidation(cur_price, float(row[high_col]), float(row[low_col]))

        f_dict = _row_to_feature_dict(row, close_col, vol_col)
        pred = meta_learner.predict(f_dict)
        meta_score = float(pred.get("signal_strength", 0.5))
        rec_expiry = str(pred.get("recommended_expiry", "30m"))
        h_strengths = pred.get("horizon_strengths", [0.5, 0.5, 0.5, 0.5])

        htf_bias = HTFBiasPackage(
            direction="bullish" if meta_score > 0.5 else "bearish",
            strength=meta_score,
            reversal_prob=float(pred.get("reversal_prob", 0.2)),
            q_value=float(pred.get("q_value", 0.5)),
            expected_mfe_pips=float(pred.get("expected_mfe_pips", 50.0)),
            expected_mae_pips=float(pred.get("expected_mae_pips", 15.0)),
            horizon_strengths=h_strengths,
            optimal_horizon_idx=int(pred.get("optimal_horizon_idx", 2)),
            recommended_expiry=rec_expiry,
        )

        exec_ctx = _make_exec_ctx(symbol, cur_price, dict(row))

        # Map recommended expiry to full EXPIRY_HORIZONS key
        rec_expiry_full = HORIZON_LABEL_MAP.get(rec_expiry, None)

        # Check entry per horizon if free — no strength threshold gate here;
        # let the Q-Learner decide WAIT vs CALL/PUT per horizon independently
        for h_idx, (exp_label, lookahead_bars) in enumerate(EXPIRY_HORIZONS.items()):
            if idx < active_horizon_until[exp_label]:
                continue  # Horizon slot is occupied by active trade

            expiry_row = test_df.iloc[idx + lookahead_bars]
            expiry_price = float(expiry_row[close_col])

            account.open_position_type = None  # Free for this horizon
            state = q_executor.build_state_vector(htf_bias, account, exec_ctx, zone_manager)
            mask_engine = HardActionMask()
            action_mask = mask_engine.get_action_mask(
                current_price=cur_price, atr=exec_ctx.atr, zone_manager=zone_manager, buy_volume=exec_ctx.buy_volume, sell_volume=exec_ctx.sell_volume, has_open_position=False,
            )
            # HardActionMask is authoritative — WAIT is a valid outcome.
            # Do NOT override the mask based on HTF bias here; that defeats
            # the no-chase discipline and inflates trade counts artificially.

            action = q_executor.select_action(state, action_mask, eval_mode=True)

            outcome = None
            if action == 1:  # BUY_CALL
                outcome = 1 if expiry_price > cur_price else 0
            elif action == 2:  # BUY_PUT
                outcome = 1 if expiry_price < cur_price else 0

            if outcome is not None:
                active_horizon_until[exp_label] = idx + lookahead_bars
                horizon_trade_counts[exp_label]["total"] += 1
                if exp_label == rec_expiry_full:
                    recommended_matches += 1

                portfolio_outcomes.append(outcome)
                if outcome == 1:
                    horizon_trade_counts[exp_label]["wins"] += 1
                    cur_p_w_streak += 1; cur_p_l_streak = 0
                    _p_cur_ws += 1
                    if _p_cur_ls > 0:
                        portfolio_loss_streaks.append(_p_cur_ls)
                        _p_cur_ls = 0
                else:
                    horizon_trade_counts[exp_label]["losses"] += 1
                    cur_p_l_streak += 1; cur_p_w_streak = 0
                    _p_cur_ls += 1
                    if _p_cur_ws > 0:
                        portfolio_win_streaks.append(_p_cur_ws)
                        _p_cur_ws = 0

                max_p_w_streak = max(max_p_w_streak, cur_p_w_streak)
                max_p_l_streak = max(max_p_l_streak, cur_p_l_streak)

    if _p_cur_ws > 0: portfolio_win_streaks.append(_p_cur_ws)
    if _p_cur_ls > 0: portfolio_loss_streaks.append(_p_cur_ls)

    total_p_trades = len(portfolio_outcomes)
    p_wins = sum(portfolio_outcomes)
    p_losses = total_p_trades - p_wins
    p_win_rate = (p_wins / total_p_trades * 100.0) if total_p_trades > 0 else 0.0

    pq_size = max(1, total_p_trades // 4)
    p_quartiles = []
    for q in range(4):
        seg = portfolio_outcomes[q * pq_size: (q + 1) * pq_size]
        qr = (sum(seg) / len(seg) * 100.0) if seg else 0.0
        p_quartiles.append(round(qr, 1))

    p_late_wr = p_quartiles[3] if len(p_quartiles) == 4 else 0.0
    p_early_wr = p_quartiles[0] if len(p_quartiles) == 4 else 0.0
    p_drift = round(p_late_wr - p_early_wr, 1)
    p_drift_str = f"+{p_drift}%" if p_drift >= 0 else f"{p_drift}%"

    p_avg_ws = round(sum(portfolio_win_streaks) / len(portfolio_win_streaks), 2) if portfolio_win_streaks else 0.0
    p_avg_ls = round(sum(portfolio_loss_streaks) / len(portfolio_loss_streaks), 2) if portfolio_loss_streaks else 0.0
    p_streak_ratio = round(p_avg_ws / p_avg_ls, 2) if p_avg_ls > 0 else float("inf")

    rec_alignment_pct = round(recommended_matches / total_p_trades * 100.0, 1) if total_p_trades > 0 else 0.0

    print(f"COLLECTIVE PORTFOLIO  | Trades: {total_p_trades:<5} | Wins: {p_wins:<5} | Losses: {p_losses:<5} | Win Rate: {p_win_rate:.2f}% | Max Streaks [W:{max_p_w_streak}, L:{max_p_l_streak}]")
    print(f"  Quartile WR [Q1→Q4]: {p_quartiles}  |  Late vs Early Drift: {p_drift_str}")
    print(f"  Avg Streaks: W={p_avg_ws} / L={p_avg_ls}  (Streak Ratio: {p_streak_ratio}x)")
    print(f"  Recommended Expiry Alignment Rate: {rec_alignment_pct}% ({recommended_matches}/{total_p_trades})")
    print("  Per-Horizon Portfolio Contribution:")
    for exp_label, counts in horizon_trade_counts.items():
        h_tot = counts["total"]
        h_wr = (counts["wins"] / h_tot * 100.0) if h_tot > 0 else 0.0
        print(f"    - {exp_label:<5}: {h_tot:<5} trades | Win Rate: {h_wr:.2f}% (Wins: {counts['wins']}, Losses: {counts['losses']})")

    results_table["COLLECTIVE_PORTFOLIO"] = {
        "total_trades": total_p_trades,
        "wins": p_wins,
        "losses": p_losses,
        "win_rate_pct": round(p_win_rate, 2),
        "max_win_streak": max_p_w_streak,
        "max_loss_streak": max_l_streak,
        "quartile_win_rates": p_quartiles,
        "late_vs_early_drift_pct": p_drift,
        "avg_win_streak": p_avg_ws,
        "avg_loss_streak": p_avg_ls,
        "streak_ratio": p_streak_ratio,
        "recommended_expiry_alignment_pct": rec_alignment_pct,
        "per_horizon_breakdown": horizon_trade_counts,
    }

    print("\n==================================================================================\n")
    return results_table



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Option Expiries Benchmark")
    parser.add_argument("--limit", type=int, default=40000, help="Candle limit per timeframe")
    parser.add_argument("--framework", type=str, choices=["keras", "pytorch"], default="keras", help="Model framework (keras or pytorch)")
    parser.add_argument(
        "--symbols", type=str,
        default="GLD,SPY,QQQ,TLT,SLV,GDX,USO,EEM,XLF,XLE",
        help=(
            "Comma-separated list of equity/ETF symbols priced in USD (left-of-dollar). "
            "BTC/USD is excluded — Alpaca does not offer options on crypto. "
            "All symbols here are USD-denominated equities/ETFs with active Alpaca options chains, "
            "so the DXY synthetic inversion signal is directionally consistent."
        )
    )
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.replace(" ", ",").split(",") if s.strip()]

    all_results = {}
    for sym in symbols:
        print(f"\n{'='*80}")
        print(f"  SYMBOL: {sym} | Framework: {args.framework.upper()} | Limit: {args.limit}")
        print(f"{'='*80}")
        try:
            result = evaluate_expiries_for_symbol(sym, limit=args.limit, framework=args.framework)
            all_results[sym] = result
        except Exception as exc:
            logger.error("[%s] Evaluation failed: %s", sym, exc, exc_info=True)
            all_results[sym] = {"error": str(exc)}

    # ── Cross-symbol summary ───────────────────────────────────────────────────
    print("\n" + "="*80)
    print("  CROSS-SYMBOL EDGE SUMMARY")
    print("="*80)
    print(f"{'Symbol':<8} {'Trades':>7} {'Win Rate':>10} {'Best Horizon':<20} {'Alignment':>10}")
    print("-"*60)
    for sym, res in all_results.items():
        if "error" in res:
            print(f"{sym:<8}  ERROR: {res['error'][:50]}")
            continue
        port = res.get("COLLECTIVE_PORTFOLIO", {})
        best_h, best_wr = "N/A", 0.0
        for k, v in res.items():
            if k == "COLLECTIVE_PORTFOLIO":
                continue
            wr = v.get("win_rate_pct", 0.0)
            if wr > best_wr:
                best_wr, best_h = wr, k
        print(
            f"{sym:<8} {port.get('total_trades', 0):>7} "
            f"{port.get('win_rate_pct', 0.0):>9.2f}% "
            f"{best_h:<20} "
            f"{port.get('recommended_expiry_alignment_pct', 0.0):>9.1f}%"
        )
    print("="*80 + "\n")
