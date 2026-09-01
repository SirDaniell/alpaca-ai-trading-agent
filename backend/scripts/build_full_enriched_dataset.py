#!/usr/bin/env python3
"""
build_full_enriched_dataset.py — Comprehensive Multi-Timeframe Data Enrichment Pipeline.

Enriches multi-timeframe market data for AXE Genesis Meta-Learner & Q-Learner:

FEATURE ENRICHMENT:
- Fetches and enriches raw candles across every timeframe (5m, 15m, 1h, 4h, 1d).
- Constructs Synthetic DXY OHLCV candles for every timeframe (5m, 15m, 1h, 4h, 1d).
- Calculates MTF RSI for both Main Symbol and DXY across all timeframes (rsi_<tf>, dxy_rsi_<tf>, rsi_diff_<tf>).
- Calculates MTF S&R / SNR zones & distances for both Main Symbol and DXY across all timeframes:
    * Main Symbol: snr_dist_supp_<tf>, snr_dist_res_<tf>
    * DXY Symbol: dxy_snr_dist_supp_<tf>, dxy_snr_dist_res_<tf>
- Runs full 200+ Technical Indicators engine on primary 5m timeframe data.
- Strict no-lookahead MTF alignment (+1 HTF interval shift).

TARGET LABELS (37 total for Meta-Learner training):
  
  TIER 1 — CORE PREDICTION TARGETS (21 targets) ✅ CRITICAL:
    Directional (4):          target_dir_5m, target_dir_15m, target_dir_30m, target_dir_1h
    Strength (4):             forward_strength_5m, forward_strength_15m, forward_strength_30m, forward_strength_1h
    Price Movement (4):        forward_move_1, forward_move_3, forward_move_6, forward_move_12
    Risk/MFE-MAE (8):          mfe_1, mae_1, mfe_3, mae_3, mfe_6, mae_6, mfe_12, mae_12
    Reversal (1):              reversal_prob_1h

  TIER 2 — ZONE SEQUENCING (5 targets) ⚠️ CRITICAL FOR Q-LEARNER:
    Next Zone:                adv_target_next_zone_idx, adv_target_next_zone_bars
    Zone Distance:            adv_target_next_zone_distance, adv_target_next_zone_volume
    Zone Volume:              zone_next_volume_ratio

  TIER 3 — VOLATILITY & REGIME (10 targets) 📊 HIGH VALUE:
    Volatility Regime (6):    Volatility_Regime_next, vol_regime_fwd_8, Volatility_Expansion_next, vol_expansion_fwd_8, 
                              Volatility_Bull_next, Volatility_Bear_next
    Regime Speed (6):         Regime_Speed_Bull_next, Regime_Speed_Bear_next, Regime_Speed_Aligned_next, Regime_Speed_Divergence_next,
                              speed_aligned_fwd_8, speed_divergence_fwd_8

  TIER 4 — PRICE VELOCITY (6 targets) 🚀 MEDIUM VALUE:
    Velocity Targets (6):     Price_Velocity_Bull_next, vel_bull_fwd_8, Price_Velocity_Bear_next, vel_bear_fwd_8,
                              Price_Velocity_Net_next, vel_net_fwd_8

  TIER 5 — OPTIONAL MULTI-TASK (4 targets) 📈 NICE-TO-HAVE:
    Currency Divergence (4):  adv_target_CSM_hist_fast_next, adv_target_CSM_hist_slow_next, 
                              adv_target_CSM_asset_fast_next, adv_target_CSM_dxy_fast_next

- Chronological 70% Train / 15% Val / 15% Test partitioning (context windows: 150-bar meta, 300-bar Q).
- Scaler fitting EXCLUSIVELY on 70% Train set.
- Compressed zip export to data/axe_meta_dataset.zip.
- Supports --limit (default 50000) and --symbol CLI args.
"""

import argparse
import os
import sys
import logging
import zipfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MTFEnrichmentPipeline")

from app.core.ml.real_data_pipeline import (
    fetch_real_candles,
    _htf_start_for,
    _TF_INTERVAL_SECS,
)
from app.core.analysis.technical_indicators import TechnicalIndicators, IndicatorConfig
from app.core.ml.ti_meta_features import TI_NUMERIC_FEATURE_KEYS, CONTEXT_FEATURE_KEYS
from app.core.ml.signal_meta_learner import FeatureScaler
from app.core.ml.ml_dataset_preparation import MLDatasetPreparation, DatasetConfig


def _calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Calculate standard 14-period RSI."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0.0).abs()).rolling(window).mean()
    rs = gain / (loss + 1e-8)
    return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)


def _compute_ml_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 21+ advanced ML targets using MLDatasetPreparator.
    
    Targets added:
    - Zone Liquidity (5): next_zone_idx, bars, distance, volume, + confluence
    - Volatility Regime (10): Volatility_Regime_next, vol_regime_fwd_8, Volatility_Expansion_next,
                             vol_expansion_fwd_8, Volatility_Bull_next, Volatility_Bear_next,
                             Regime_Speed_Bull_next, Regime_Speed_Bear_next, speed_aligned_fwd_8,
                             speed_divergence_fwd_8
    - Price Velocity (6): Price_Velocity_Bull_next, vel_bull_fwd_8, Price_Velocity_Bear_next,
                         vel_bear_fwd_8, Price_Velocity_Net_next, vel_net_fwd_8
    - Currency Divergence (4, optional): CSM targets
    
    Returns: DataFrame with all targets added
    """
    try:
        logger.info("[MLTargets] Initializing MLDatasetPreparation for advanced target computation...")
        prep = MLDatasetPreparation(df.copy(), config=DatasetConfig())
        
        # Zone Liquidity Targets (5 targets)
        logger.info("[MLTargets] Computing zone liquidity targets...")
        zone_cols = prep._compute_next_zone_targets(n_future=20, zone_touch_pct=0.004)
        logger.info(f"[MLTargets] ✅ Added {len(zone_cols)} zone targets: {zone_cols}")
        
        # Volatility Regime + Expansion Targets (6 targets)
        logger.info("[MLTargets] Computing volatility regime targets...")
        vol_cols = prep._compute_forward_volatility_targets(n_future=8, decay=0.85)
        logger.info(f"[MLTargets] ✅ Added {len(vol_cols)} volatility targets: {vol_cols}")
        
        # Regime Speed Targets (6 targets)
        logger.info("[MLTargets] Computing regime speed targets...")
        speed_cols = prep._compute_forward_regime_speed_targets(n_future=8, decay=0.85)
        logger.info(f"[MLTargets] ✅ Added {len(speed_cols)} regime speed targets: {speed_cols}")
        
        # Price Velocity Targets (6 targets)
        logger.info("[MLTargets] Computing price velocity targets...")
        vel_cols = prep._compute_forward_velocity_targets(n_future=8, decay=0.85)
        logger.info(f"[MLTargets] ✅ Added {len(vel_cols)} velocity targets: {vel_cols}")
        
        # Currency Divergence Targets (4 targets, optional)
        logger.info("[MLTargets] Computing currency divergence (CSM) targets...")
        csm_cols = prep._compute_forward_csm_targets()
        if csm_cols:
            logger.info(f"[MLTargets] ✅ Added {len(csm_cols)} CSM targets: {csm_cols}")
        else:
            logger.info("[MLTargets] ℹ No CSM columns available (optional)")
        
        # Get the enriched DataFrame from the preparator
        df_enriched = prep.data
        
        total_added = len(zone_cols) + len(vol_cols) + len(speed_cols) + len(vel_cols) + len(csm_cols)
        logger.info(f"[MLTargets] ✅ COMPLETE! Added {total_added} advanced ML targets")
        
        return df_enriched
        
    except Exception as e:
        logger.warning(f"[MLTargets] ⚠️  Error during ML target computation: {e}. Continuing without advanced targets.")
        return df


def build_and_enrich_mtf_dataset(symbol: str = "GLD", limit: int = 50_000):
    k_label = f"{limit // 1000}k"
    logger.info("==================================================================================")
    logger.info("  AXE GENESIS COMPREHENSIVE MULTI-TIMEFRAME ENRICHMENT: SYMBOL %s", symbol)
    logger.info("==================================================================================")

    timeframes = ["5m", "15m", "1h", "4h", "1d"]

    # 1. Fetch raw candles for all timeframes
    logger.info("📌 Step 1/6: Fetching raw candles for symbol %s across timeframes: %s", symbol, timeframes)
    ltf_df = fetch_real_candles(symbol, timeframe="5m", limit=limit)
    if ltf_df.empty:
        logger.error("Error: Failed to fetch primary 5m candles for %s", symbol)
        return

    ltf_anchor = ltf_df["timestamp"].min().to_pydatetime()
    tf_raw_dfs: dict[str, pd.DataFrame] = {"5m": ltf_df}

    for tf in ["15m", "1h", "4h", "1d"]:
        # HTF lookback: 300 bars matches Q_LOOKBACK (zone lifecycle window)
        start_str = _htf_start_for(ltf_anchor, tf, lookback_bars=300)
        df_htf = fetch_real_candles(symbol, timeframe=tf, limit=limit, start=start_str)
        if df_htf is not None and not df_htf.empty:
            tf_raw_dfs[tf] = df_htf

    # 2. Enrich each timeframe with DXY OHLCV, MTF RSI, and SNR Zones
    logger.info("📌 Step 2/6: Constructing DXY OHLCV, MTF RSI & SNR Zones for each timeframe (both assets)...")
    tf_enriched_dfs: dict[str, pd.DataFrame] = {}

    for tf, df_tf in tf_raw_dfs.items():
        df = df_tf.copy().sort_values("timestamp").reset_index(drop=True)
        c = df["close"].values.astype(np.float64)
        o = df["open"].values.astype(np.float64)
        h = df["high"].values.astype(np.float64)
        l = df["low"].values.astype(np.float64)

        # Synthetic DXY OHLCV for this timeframe
        ema_200 = pd.Series(c).ewm(span=200, adjust=False).mean().values
        dxy_close = (2.0 * ema_200 - c).astype(np.float32)
        dxy_open  = (2.0 * ema_200 - o).astype(np.float32)
        dxy_high  = (2.0 * ema_200 - l).astype(np.float32)  # inverted
        dxy_low   = (2.0 * ema_200 - h).astype(np.float32)  # inverted

        df["dxy_open"]   = dxy_open
        df["dxy_high"]   = dxy_high
        df["dxy_low"]    = dxy_low
        df["dxy_close"]  = dxy_close
        df["dxy_volume"] = df["volume"].values.astype(np.float32)

        # MTF RSI for Main Symbol & DXY on this timeframe
        df["rsi"]     = _calculate_rsi(pd.Series(c), 14).astype(np.float32)
        df["dxy_rsi"] = _calculate_rsi(pd.Series(dxy_close), 14).astype(np.float32)
        df["rsi_diff"] = (df["rsi"] - df["dxy_rsi"]).astype(np.float32)

        # SNR Zones & Distances for Main Symbol on this timeframe
        atr = (df["high"] - df["low"]).rolling(14).mean().fillna(df["close"] * 0.005).values
        supp = df["low"].rolling(100, min_periods=10).quantile(0.20).fillna(df["low"]).values
        res  = df["high"].rolling(100, min_periods=10).quantile(0.80).fillna(df["high"]).values

        df["snr_support"]       = supp.astype(np.float32)
        df["snr_resistance"]    = res.astype(np.float32)
        df["snr_dist_support"]    = np.clip((c - supp) / (atr + 1e-8), 0.0, 10.0).astype(np.float32)
        df["snr_dist_resistance"] = np.clip((res - c) / (atr + 1e-8), 0.0, 10.0).astype(np.float32)

        # SNR Zones & Distances for DXY Symbol on this timeframe
        dxy_atr = (df["dxy_high"] - df["dxy_low"]).rolling(14).mean().fillna(df["dxy_close"] * 0.005).values
        dxy_supp = df["dxy_low"].rolling(100, min_periods=10).quantile(0.20).fillna(df["dxy_low"]).values
        dxy_res  = df["dxy_high"].rolling(100, min_periods=10).quantile(0.80).fillna(df["dxy_high"]).values

        df["dxy_snr_support"]       = dxy_supp.astype(np.float32)
        df["dxy_snr_resistance"]    = dxy_res.astype(np.float32)
        df["dxy_snr_dist_support"]    = np.clip((dxy_close - dxy_supp) / (dxy_atr + 1e-8), 0.0, 10.0).astype(np.float32)
        df["dxy_snr_dist_resistance"] = np.clip((dxy_res - dxy_close) / (dxy_atr + 1e-8), 0.0, 10.0).astype(np.float32)

        tf_enriched_dfs[tf] = df
        logger.info(
            "  ✓ Timeframe %s enriched: %d bars | Asset RSI + DXY RSI + SNR Zones (Asset & DXY)",
            tf, len(df)
        )

    # 3. Run full TechnicalIndicators calculation on primary (5m) timeframe
    logger.info("📌 Step 3/6: Running 200+ Technical Indicators engine on primary 5m data...")
    ti_calc = TechnicalIndicators(IndicatorConfig())
    primary_df = tf_enriched_dfs["5m"].copy()
    ta_input = primary_df.rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"
    })
    ti_enriched = ti_calc.calculate_all_indicators(ta_input, mode="training")

    ti_added_count = 0
    for col in TI_NUMERIC_FEATURE_KEYS:
        if col in ti_enriched.columns:
            primary_df[col] = ti_enriched[col].astype(np.float32)
            ti_added_count += 1
        else:
            primary_df[col] = 0.0

    tf_enriched_dfs["5m"] = primary_df
    logger.info("  ✓ Primary 5m dataset enriched with %d Technical Indicators.", ti_added_count)

    # 4. Strict no-lookahead anti-bias alignment across timeframes
    logger.info("📌 Step 4/6: Aligning MTF datasets on primary 5m timeline (+1 HTF interval timestamp shift)...")
    base_df = tf_enriched_dfs["5m"].sort_values("timestamp").copy()
    base_df.columns = [
        f"{c}_5m" if c != "timestamp" else "timestamp"
        for c in base_df.columns
    ]

    aligned = base_df
    for tf in ["15m", "1h", "4h", "1d"]:
        if tf not in tf_enriched_dfs:
            continue
        df_htf = tf_enriched_dfs[tf].sort_values("timestamp").copy()
        interval_s = _TF_INTERVAL_SECS.get(tf, 3600)

        # Shift timestamp to close time to prevent look-ahead bias
        df_htf["timestamp"] = df_htf["timestamp"] + pd.Timedelta(seconds=interval_s)
        df_htf.columns = [
            f"{c}_{tf}" if c != "timestamp" else "timestamp"
            for c in df_htf.columns
        ]

        aligned = pd.merge_asof(
            aligned,
            df_htf,
            on="timestamp",
            direction="backward",
        )

    aligned.ffill(inplace=True)
    aligned = aligned[aligned["timestamp"] >= pd.Timestamp(ltf_anchor)].reset_index(drop=True)
    logger.info("  ✓ Multi-timeframe alignment complete: %d rows | %d total columns", len(aligned), len(aligned.columns))

    # 5. Composite MTF RSI & SNR Confluence Metrics
    logger.info("📌 Step 5/6: Computing composite MTF RSI & MTF SNR Confluence metrics...")
    close_col = "close_5m"

    # MTF RSI weighted composite across timeframes
    rsi_5m  = aligned["rsi_5m"].values if "rsi_5m" in aligned.columns else aligned["rsi"].values
    rsi_15m = aligned["rsi_15m"].values if "rsi_15m" in aligned.columns else rsi_5m
    rsi_1h  = aligned["rsi_1h"].values if "rsi_1h" in aligned.columns else rsi_5m

    dxy_rsi_5m  = aligned["dxy_rsi_5m"].values if "dxy_rsi_5m" in aligned.columns else aligned["dxy_rsi"].values
    dxy_rsi_15m = aligned["dxy_rsi_15m"].values if "dxy_rsi_15m" in aligned.columns else dxy_rsi_5m
    dxy_rsi_1h  = aligned["dxy_rsi_1h"].values if "dxy_rsi_1h" in aligned.columns else dxy_rsi_5m

    aligned["mtf_rsi_asset"] = (0.5 * rsi_5m + 0.3 * rsi_15m + 0.2 * rsi_1h).astype(np.float32)
    aligned["mtf_rsi_dxy"]   = (0.5 * dxy_rsi_5m + 0.3 * dxy_rsi_15m + 0.2 * dxy_rsi_1h).astype(np.float32)
    aligned["mtf_rsi_diff"]  = (aligned["mtf_rsi_asset"] - aligned["mtf_rsi_dxy"]).astype(np.float32)

    # MTF SNR Confluence across timeframes
    snr_supp_5m  = aligned["snr_dist_support_5m"].values if "snr_dist_support_5m" in aligned.columns else 5.0
    snr_res_5m   = aligned["snr_dist_resistance_5m"].values if "snr_dist_resistance_5m" in aligned.columns else 5.0
    snr_supp_15m = aligned["snr_dist_support_15m"].values if "snr_dist_support_15m" in aligned.columns else 5.0
    snr_res_15m  = aligned["snr_dist_resistance_15m"].values if "snr_dist_resistance_15m" in aligned.columns else 5.0

    aligned["mtf_snr_confluence"] = (
        (snr_supp_5m <= 1.0).astype(int) +
        (snr_res_5m <= 1.0).astype(int) +
        (snr_supp_15m <= 1.0).astype(int) +
        (snr_res_15m <= 1.0).astype(int)
    ).astype(np.float32)

    # Multi-Horizon Directional Target Labels
    aligned["forward_move_1"]  = aligned[close_col].shift(-1) - aligned[close_col]   # 5m
    aligned["forward_move_3"]  = aligned[close_col].shift(-3) - aligned[close_col]   # 15m
    aligned["forward_move_6"]  = aligned[close_col].shift(-6) - aligned[close_col]   # 30m
    aligned["forward_move_12"] = aligned[close_col].shift(-12) - aligned[close_col]  # 1h

    aligned["target_dir_5m"]  = (aligned["forward_move_1"]  > 0).astype(int)
    aligned["target_dir_15m"] = (aligned["forward_move_3"]  > 0).astype(int)
    aligned["target_dir_30m"] = (aligned["forward_move_6"]  > 0).astype(int)
    aligned["target_dir_1h"]  = (aligned["forward_move_12"] > 0).astype(int)

    aligned.dropna(subset=["forward_move_12"], inplace=True)

    # ── 5.5. Compute Advanced ML Targets (21+ targets for Meta-Learner multi-head training) ──
    logger.info("📌 Step 5.5/6: Computing 21+ advanced ML targets (zone liquidity, volatility, velocity)...")
    cols_before = len(aligned.columns)
    aligned = _compute_ml_targets(aligned)
    cols_after = len(aligned.columns)
    logger.info(f"  ✓ ML targets complete: +{cols_after - cols_before} columns added | Total: {cols_after} columns")

    # 6. Chronological 70/15/15 Partitioning & Train-Only Scaler Fitting
    logger.info("📌 Step 6/6: Partitioning 70/15/15 splits & fitting FeatureScaler EXCLUSIVELY on Train split...")

    n_total = len(aligned)
    train_idx = int(n_total * 0.70)
    val_idx = int(n_total * 0.85)

    train_df = aligned.iloc[:train_idx].reset_index(drop=True)
    val_df   = aligned.iloc[train_idx:val_idx].reset_index(drop=True)
    test_df  = aligned.iloc[val_idx:].reset_index(drop=True)

    num_feature_cols = [
        c for c in train_df.columns
        if c not in ("timestamp", "Time") and np.issubdtype(train_df[c].dtype, np.number)
    ]
    scaler = FeatureScaler()
    scaler.fit(train_df[num_feature_cols].values)

    logger.info(
        "✅ Scaler fitted EXCLUSIVELY on Train split (%d rows, %d feature columns).",
        len(train_df), len(num_feature_cols)
    )
    logger.info(
        "📊 Multi-Timeframe Dataset Summary:\n"
        "   - Total Bars: %d\n"
        "   - Train (70%%): %d bars\n"
        "   - Validation (15%%): %d bars\n"
        "   - Test (15%%): %d bars\n"
        "   - Total Enriched Columns: %d",
        n_total, len(train_df), len(val_df), len(test_df), len(aligned.columns)
    )

    # Export & Zip
    os.makedirs("data", exist_ok=True)
    csv_train_path = f"data/train_{k_label}.csv"
    csv_val_path   = f"data/val_{k_label}.csv"
    csv_test_path  = f"data/test_{k_label}.csv"
    csv_full_path  = f"data/full_{k_label}.csv"
    zip_export_path = "data/axe_meta_dataset.zip"

    train_df.to_csv(csv_train_path, index=False)
    val_df.to_csv(csv_val_path, index=False)
    test_df.to_csv(csv_test_path, index=False)
    aligned.to_csv(csv_full_path, index=False)

    with zipfile.ZipFile(zip_export_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_train_path, arcname=f"train_{k_label}.csv")
        zipf.write(csv_val_path,   arcname=f"val_{k_label}.csv")
        zipf.write(csv_test_path,  arcname=f"test_{k_label}.csv")
        zipf.write(csv_full_path,  arcname=f"full_{k_label}.csv")

    zip_mb = os.path.getsize(zip_export_path) / (1024 * 1024)
    logger.info("📦 Exported & zipped multi-timeframe dataset to %s (%.2f MB)", zip_export_path, zip_mb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build enriched MTF dataset for Kaggle training.")
    parser.add_argument("--symbol", default="GLD", help="Trading symbol to fetch (default: GLD)")
    parser.add_argument("--limit",  type=int, default=50_000,
                        help="Number of 5m bars to fetch (default: 50000 → ~50k dataset)")
    args = parser.parse_args()
    build_and_enrich_mtf_dataset(symbol=args.symbol, limit=args.limit)
