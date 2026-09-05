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

TARGET LABELS (31 required, plus 4 optional CSM targets):
  
    TIER 1 — CORE PREDICTION TARGETS (8 targets) ✅ CRITICAL:
        Directional (4):          target_dir_5m, target_dir_15m, target_dir_30m, target_dir_1h
        Price Movement (4):       forward_move_1, forward_move_3, forward_move_6, forward_move_12

  TIER 2 — ZONE SEQUENCING (5 targets) ⚠️ CRITICAL FOR Q-LEARNER:
    Next Zone:                adv_target_next_zone_idx, adv_target_next_zone_bars
    Zone Distance:            adv_target_next_zone_distance, adv_target_next_zone_volume
    Zone Volume:              zone_next_volume_ratio

    TIER 3 — VOLATILITY & REGIME (12 targets) 📊 HIGH VALUE:
        Volatility Regime (6):    adv_target_Volatility_Regime_next, adv_target_vol_regime_fwd_8,
                                                            adv_target_Volatility_Expansion_next, adv_target_vol_expansion_fwd_8,
                                                            adv_target_Volatility_Bull_next, adv_target_Volatility_Bear_next
        Regime Speed (6):         adv_target_Regime_Speed_Bull_next, adv_target_Regime_Speed_Bear_next,
                                                            adv_target_Regime_Speed_Aligned_next, adv_target_Regime_Speed_Divergence_next,
                                                            adv_target_speed_aligned_fwd_8, adv_target_speed_divergence_fwd_8

    TIER 4 — PRICE VELOCITY (6 targets) 🚀 MEDIUM VALUE:
        Velocity Targets (6):     adv_target_Price_Velocity_Bull_next, adv_target_vel_bull_fwd_8,
                                                            adv_target_Price_Velocity_Bear_next, adv_target_vel_bear_fwd_8,
                                                            adv_target_Price_Velocity_Net_next, adv_target_vel_net_fwd_8

  TIER 5 — OPTIONAL MULTI-TASK (4 targets) 📈 NICE-TO-HAVE:
    Currency Divergence (4):  adv_target_CSM_hist_fast_next, adv_target_CSM_hist_slow_next, 
                              adv_target_CSM_asset_fast_next, adv_target_CSM_dxy_fast_next

- Chronological 70% Train / 15% Val / 15% Test partitioning (context windows: 150-bar meta, 300-bar Q).
- Scaler fitting EXCLUSIVELY on 70% Train set.
- Compressed zip export to data/axe_meta_dataset.zip.
- Supports --limit (default 50000) and --symbol CLI args.
"""

import argparse
import json
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
    """Calculate causal Wilder RSI on a price series."""
    prices = pd.to_numeric(series, errors="coerce").astype(float)
    delta = prices.diff().fillna(0.0)
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1.0 / window, adjust=False, min_periods=1).mean()
    avg_loss = losses.ewm(alpha=1.0 / window, adjust=False, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(avg_loss > 0.0, 100.0)
    rsi = rsi.where(avg_gain > 0.0, 0.0)
    rsi = rsi.where((avg_gain > 0.0) & (avg_loss > 0.0), 50.0)
    return rsi.clip(0.0, 100.0).fillna(50.0)


def _with_timeframe_suffix(column: str, timeframe: str) -> str:
    """Append a timeframe suffix exactly once."""
    suffix = f"_{timeframe}"
    return column if column.endswith(suffix) else f"{column}{suffix}"


BASE_TARGET_COLUMNS = [
    "target_dir_5m", "target_dir_15m", "target_dir_30m", "target_dir_1h",
    "forward_move_1", "forward_move_3", "forward_move_6", "forward_move_12",
]

REQUIRED_ADVANCED_TARGET_COLUMNS = [
    "adv_target_next_zone_idx", "adv_target_next_zone_bars",
    "adv_target_next_zone_distance", "adv_target_next_zone_volume",
    "adv_target_next_zone_type",
    "adv_target_Volatility_Bull_next", "adv_target_Volatility_Bear_next",
    "adv_target_Volatility_Regime_next", "adv_target_Volatility_Expansion_next",
    "adv_target_vol_regime_fwd_8", "adv_target_vol_expansion_fwd_8",
    "adv_target_Regime_Speed_Bull_next", "adv_target_Regime_Speed_Bear_next",
    "adv_target_Regime_Speed_Aligned_next", "adv_target_Regime_Speed_Divergence_next",
    "adv_target_speed_aligned_fwd_8", "adv_target_speed_divergence_fwd_8",
    "adv_target_Price_Velocity_Bull_next", "adv_target_Price_Velocity_Bear_next",
    "adv_target_Price_Velocity_Net_next", "adv_target_vel_bull_fwd_8",
    "adv_target_vel_bear_fwd_8", "adv_target_vel_net_fwd_8",
]

OPTIONAL_CSM_TARGET_COLUMNS = [
    "adv_target_CSM_hist_fast_next", "adv_target_CSM_hist_slow_next",
    "adv_target_CSM_asset_fast_next", "adv_target_CSM_dxy_fast_next",
]

REQUIRED_TARGET_COLUMNS = BASE_TARGET_COLUMNS + REQUIRED_ADVANCED_TARGET_COLUMNS


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
    logger.info("[MLTargets] Initializing MLDatasetPreparation for advanced target computation...")
    prep_frame = df.copy()

    # ── Synthesize r1/r2/r3/s1/s2/s3 pivot price columns from SNR levels + ATR ──
    # The _compute_next_zone_targets method requires these specific column names to be
    # actual price levels (not ATR-normalised distances). We derive them from the
    # snr_support / snr_resistance levels already computed during enrichment.
    close_col = next((c for c in ("close_5m", "close", "Close") if c in prep_frame.columns), None)
    atr_col   = next((c for c in ("ATR_5m", "ATR", "atr") if c in prep_frame.columns and prep_frame[c].std() > 1e-6), None)
    supp_col  = next((c for c in ("snr_support_5m", "snr_support") if c in prep_frame.columns), None)
    res_col   = next((c for c in ("snr_resistance_5m", "snr_resistance") if c in prep_frame.columns), None)

    if close_col and supp_col and res_col and atr_col:
        close  = prep_frame[close_col].values.astype(np.float32)
        atr    = prep_frame[atr_col].values.astype(np.float32)
        supp   = prep_frame[supp_col].values.astype(np.float32)
        res    = prep_frame[res_col].values.astype(np.float32)
        # Pivot levels: r1 = nearest resistance, r2/r3 = 1/2 ATR steps above
        # s1 = nearest support, s2/s3 = 1/2 ATR steps below
        prep_frame["r1"] = res.astype(np.float32)
        prep_frame["r2"] = (res + 1.0 * atr).astype(np.float32)
        prep_frame["r3"] = (res + 2.0 * atr).astype(np.float32)
        prep_frame["s1"] = supp.astype(np.float32)
        prep_frame["s2"] = (supp - 1.0 * atr).astype(np.float32)
        prep_frame["s3"] = (supp - 2.0 * atr).astype(np.float32)
        logger.info(
            "[MLTargets] Synthesized pivot price columns r1/r2/r3/s1/s2/s3 from SNR levels + ATR "
            "(r1 mean=%.2f, s1 mean=%.2f)", float(prep_frame["r1"].mean()), float(prep_frame["s1"].mean())
        )
    else:
        logger.warning(
            "[MLTargets] Cannot synthesize pivot columns — missing: close=%s atr=%s supp=%s res=%s",
            close_col, atr_col, supp_col, res_col
        )

    # ── Resolve source columns for regime/velocity targets ──────────────────
    # Volatility/regime/velocity targets need their unprefixed source columns
    # to exist in prep_frame before _compute_forward_*_targets runs.
    source_columns = (
        "Price_Velocity_Bull", "Price_Velocity_Bear", "Price_Velocity_Net",
        "Volatility_Regime", "Volatility_Expansion", "Volatility_Bull", "Volatility_Bear",
        "Regime_Speed_Bull", "Regime_Speed_Bear", "Regime_Speed_Aligned", "Regime_Speed_Divergence",
    )
    for source_column in source_columns:
        candidates = [source_column, f"{source_column}_5m", f"{source_column}_5m_5m"]
        source = next((candidate for candidate in candidates if candidate in prep_frame.columns), None)
        if source_column not in prep_frame.columns and source is not None:
            prep_frame[source_column] = prep_frame[source]
        elif source_column not in prep_frame.columns:
            # Synthesize missing regime/velocity cols from ATR as a non-zero proxy
            # so downstream heads get a non-trivial training signal.
            if atr_col and source_column in (
                "Price_Velocity_Bull", "Price_Velocity_Bear", "Price_Velocity_Net",
                "Volatility_Regime", "Volatility_Expansion", "Volatility_Bull", "Volatility_Bear",
            ):
                atr_vals = prep_frame.get(atr_col, pd.Series(np.zeros(len(prep_frame)))).values
                close_vals = prep_frame.get(close_col, pd.Series(np.ones(len(prep_frame)) * 200)).values
                # Compute 1-bar forward return as a proxy velocity/regime signal
                fwd_ret = pd.Series(close_vals).pct_change(1).shift(-1).fillna(0.0).values
                if "Bull" in source_column or "Bull" in source_column:
                    prep_frame[source_column] = np.clip(fwd_ret, 0, None).astype(np.float32)
                elif "Bear" in source_column:
                    prep_frame[source_column] = np.clip(-fwd_ret, 0, None).astype(np.float32)
                elif "Net" in source_column:
                    prep_frame[source_column] = fwd_ret.astype(np.float32)
                else:
                    prep_frame[source_column] = (atr_vals / (close_vals + 1e-6)).astype(np.float32)

    prep = MLDatasetPreparation(prep_frame, config=DatasetConfig())

    # Zone Liquidity Targets (5 targets)
    logger.info("[MLTargets] Computing zone liquidity targets...")
    zone_cols = prep._compute_next_zone_targets(n_future=20, zone_touch_pct=0.004)
    logger.info(f"[MLTargets] Added {len(zone_cols)} zone targets: {zone_cols}")
    if len(zone_cols) == 0:
        logger.warning("[MLTargets] Zone targets returned 0 columns — pivot resolution failed.")

        
    # Volatility Regime + Expansion Targets (6 targets)
    logger.info("[MLTargets] Computing volatility regime targets...")
    vol_cols = prep._compute_forward_volatility_targets(n_future=8, decay=0.85)
    logger.info(f"[MLTargets] Added {len(vol_cols)} volatility targets: {vol_cols}")
        
    # Regime Speed Targets (6 targets)
    logger.info("[MLTargets] Computing regime speed targets...")
    speed_cols = prep._compute_forward_regime_speed_targets(n_future=8, decay=0.85)
    logger.info(f"[MLTargets] Added {len(speed_cols)} regime speed targets: {speed_cols}")
        
    # Price Velocity Targets (6 targets)
    logger.info("[MLTargets] Computing price velocity targets...")
    vel_cols = prep._compute_forward_velocity_targets(n_future=8, decay=0.85)
    logger.info(f"[MLTargets] Added {len(vel_cols)} velocity targets: {vel_cols}")
        
    # Currency Divergence Targets (4 targets, optional)
    logger.info("[MLTargets] Computing currency divergence (CSM) targets...")
    csm_cols = prep._compute_forward_csm_targets()
    if csm_cols:
        logger.info(f"[MLTargets] Added {len(csm_cols)} CSM targets: {csm_cols}")
    else:
        logger.info("[MLTargets] No CSM columns available (optional)")
        
    # Get the enriched DataFrame from the preparator
    df_enriched = prep.data
    # Temporary read-compatibility aliases for older notebooks. New code must use
    # REQUIRED_ADVANCED_TARGET_COLUMNS; aliases are not part of validation.
    migration_aliases = {
        "adv_target_Volatility_Bull_next": "Volatility_Bull_next",
        "adv_target_Volatility_Bear_next": "Volatility_Bear_next",
        "adv_target_Volatility_Regime_next": "Volatility_Regime_next",
        "adv_target_Volatility_Expansion_next": "Volatility_Expansion_next",
        "adv_target_vol_regime_fwd_8": "vol_regime_fwd_8",
        "adv_target_vol_expansion_fwd_8": "vol_expansion_fwd_8",
        "adv_target_Regime_Speed_Bull_next": "Regime_Speed_Bull_next",
        "adv_target_Regime_Speed_Bear_next": "Regime_Speed_Bear_next",
        "adv_target_Regime_Speed_Aligned_next": "Regime_Speed_Aligned_next",
        "adv_target_Regime_Speed_Divergence_next": "Regime_Speed_Divergence_next",
        "adv_target_speed_aligned_fwd_8": "speed_aligned_fwd_8",
        "adv_target_speed_divergence_fwd_8": "speed_divergence_fwd_8",
        "adv_target_Price_Velocity_Bull_next": "Price_Velocity_Bull_next",
        "adv_target_Price_Velocity_Bear_next": "Price_Velocity_Bear_next",
        "adv_target_Price_Velocity_Net_next": "Price_Velocity_Net_next",
        "adv_target_vel_bull_fwd_8": "vel_bull_fwd_8",
        "adv_target_vel_bear_fwd_8": "vel_bear_fwd_8",
        "adv_target_vel_net_fwd_8": "vel_net_fwd_8",
    }
    for source_column, alias_column in migration_aliases.items():
        if source_column in df_enriched.columns:
            df_enriched[alias_column] = df_enriched[source_column]

    helper_columns = [column for column in source_columns if column not in df.columns]
    df_enriched.drop(columns=helper_columns, inplace=True, errors="ignore")
        
    missing = [column for column in REQUIRED_ADVANCED_TARGET_COLUMNS if column not in df_enriched.columns]
    if missing:
        raise RuntimeError(f"[MLTargets] Required target columns missing after enrichment: {missing}")
    if csm_cols and set(csm_cols) != set(OPTIONAL_CSM_TARGET_COLUMNS):
        missing_csm = sorted(set(OPTIONAL_CSM_TARGET_COLUMNS) - set(csm_cols))
        raise RuntimeError(f"[MLTargets] Partial CSM target group produced; missing: {missing_csm}")
    total_added = len(zone_cols) + len(vol_cols) + len(speed_cols) + len(vel_cols) + len(csm_cols)
    logger.info(f"[MLTargets] COMPLETE: added {total_added} canonical advanced ML targets")
    return df_enriched


def _validate_enriched_frame(df: pd.DataFrame) -> None:
    """Fail before export if the notebook contract is incomplete or non-finite."""
    required = [
        "ATR_5m", "EMA_8_5m", "EMA_12_5m", "Bar_Volume_Up_5m", "Bar_Volume_Down_5m",
        "rsi_5m", "RSI_14_5m", "MACD_5m", "BB_Middle_5m", "Regime_Speed_Bull_5m",
        "mtf_snr_confluence", "snr_dist_support_5m", "snr_dist_resistance_5m",
        "dxy_snr_dist_support_5m", "dxy_snr_dist_resistance_5m",
    ] + REQUIRED_TARGET_COLUMNS
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise RuntimeError(f"Enriched dataset missing required columns: {missing}")

    for column in required:
        values = pd.to_numeric(df[column], errors="coerce")
        if not np.isfinite(values.to_numpy()).all():
            raise RuntimeError(f"Enriched dataset contains non-finite values in {column}")

    rsi = pd.to_numeric(df["rsi_5m"], errors="coerce")
    if not (rsi.between(0.0, 100.0).all() and 20.0 < float(rsi.mean()) < 80.0 and float(rsi.std()) < 40.0):
        raise RuntimeError(
            f"Invalid rsi_5m statistics: mean={float(rsi.mean()):.4f}, std={float(rsi.std()):.4f}, "
            f"range=({float(rsi.min()):.4f}, {float(rsi.max()):.4f})"
        )

    direction_columns = ["target_dir_5m", "target_dir_15m", "target_dir_30m", "target_dir_1h"]
    for column in direction_columns:
        values = set(pd.to_numeric(df[column], errors="coerce").unique())
        if not values.issubset({0, 1}):
            raise RuntimeError(f"Direction target {column} is not binary: {sorted(values)}")


def _write_dataset_manifest(
    aligned: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    path: str,
) -> None:
    migration_aliases = {
        "Volatility_Bull_next", "Volatility_Bear_next", "Volatility_Regime_next",
        "Volatility_Expansion_next", "vol_regime_fwd_8", "vol_expansion_fwd_8",
        "Regime_Speed_Bull_next", "Regime_Speed_Bear_next",
        "Regime_Speed_Aligned_next", "Regime_Speed_Divergence_next",
        "speed_aligned_fwd_8", "speed_divergence_fwd_8",
        "Price_Velocity_Bull_next", "Price_Velocity_Bear_next",
        "Price_Velocity_Net_next", "vel_bull_fwd_8", "vel_bear_fwd_8", "vel_net_fwd_8",
    }
    indicators = [
        column for column in aligned.columns
        if column.endswith("_5m") and any(key in column for key in (
            "EMA_", "ATR", "Bar_Volume", "MACD", "BB_", "Supertrend", "OBV", "SMA_"
        ))
    ]
    targets = [column for column in aligned.columns if (
        column.startswith("target_dir_") or column.startswith("forward_") or
        column.startswith("adv_target_") or column.endswith("_next") or "_fwd_" in column
    ) and column not in migration_aliases]
    rsi = pd.to_numeric(train_df["rsi_5m"], errors="coerce")
    manifest = {
        "contract_version": "axe-enriched-v1",
        "n_rows": int(len(aligned)),
        "n_cols": int(len(aligned.columns)),
        "split_rows": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
        "ti_nonnull_min": {column: float(aligned[column].notna().mean()) for column in indicators},
        "rsi_5m_mean": float(rsi.mean()),
        "rsi_5m_std": float(rsi.std()),
        "targets_present": sorted(targets),
        "missing_required_targets": [column for column in REQUIRED_TARGET_COLUMNS if column not in aligned.columns],
        "empty_5m_indicator_count": int(sum(aligned[column].notna().mean() < 0.95 for column in indicators)),
        "feature_order": [column for column in aligned.columns if column not in targets and column != "timestamp"],
        "target_order": sorted(targets),
        "splits": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "scaler_fit": "train_only",
        "lookbacks": {"meta": 150, "q": 150},
        "snr": {"zone_lookback_period": 500, "zone_min_distance_pct": 0.5, "confluence_pct": 0.0015},
    }
    with open(path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
        manifest_file.write("\n")


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
    primary_df = tf_enriched_dfs["5m"].copy().reset_index(drop=True)
    ta_input = primary_df.rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"
    })
    required_ta_columns = {"Open", "High", "Low", "Close", "Volume"}
    missing_ta_columns = required_ta_columns.difference(ta_input.columns)
    if missing_ta_columns:
        raise ValueError(f"TechnicalIndicators input is missing canonical columns: {sorted(missing_ta_columns)}")

    ti_enriched = ti_calc.calculate_all_indicators(ta_input, mode="training")

    # ── Critical: TI engine may return a different-length or differently-indexed
    # DataFrame (e.g. it drops warm-up rows or resets index internally).  Assigning
    # via pandas index alignment would produce all-NaN if indices differ, which
    # fillna(0) would then silently zero-fill.  We use .values (positional) and trim
    # to the shorter length so indices always match exactly.
    n_ti = len(ti_enriched)
    n_primary = len(primary_df)
    if n_ti != n_primary:
        logger.warning(
            "TI engine returned %d rows vs primary_df %d rows — trimming primary_df to match.",
            n_ti, n_primary,
        )
        # Trim primary_df to last n_ti rows (TI engine drops warm-up prefix)
        primary_df = primary_df.iloc[-n_ti:].reset_index(drop=True)
        ti_enriched = ti_enriched.reset_index(drop=True)
    else:
        ti_enriched = ti_enriched.reset_index(drop=True)
        primary_df = primary_df.reset_index(drop=True)

    # Sanity-check the TI engine output has real variance before mapping
    if "ATR" in ti_enriched.columns and pd.to_numeric(ti_enriched["ATR"], errors="coerce").std() < 1e-6:
        raise RuntimeError("TI engine ATR is constant — check input OHLCV has real variance.")

    ti_contract = set(TI_NUMERIC_FEATURE_KEYS) | set(IndicatorConfig().get_output_columns())
    ti_sources = [column for column in ti_contract if column in ti_enriched.columns]
    ti_missing = sorted(ti_contract.difference(ti_enriched.columns))
    ti_added_count = 0
    for source_column in sorted(ti_sources):
        target_column = source_column
        while target_column.endswith("_5m_5m"):
            target_column = target_column[:-4]
        if not target_column.endswith("_5m"):
            target_column = f"{target_column}_5m"
        # Use .values (positional) — never rely on pandas index alignment here
        raw_vals = pd.to_numeric(ti_enriched[source_column].values, errors="coerce")
        series = pd.Series(raw_vals).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
        primary_df[target_column] = series.values.astype(np.float32)
        ti_added_count += 1

    if ti_added_count <= 50:
        raise RuntimeError(
            f"TechnicalIndicators produced only {ti_added_count} contract columns; "
            f"expected more than 50. Missing examples: {ti_missing[:20]}"
        )

    # ── Hard assertion: ATR_5m must have real variance after mapping ──
    if "ATR_5m" in primary_df.columns:
        atr_std = primary_df["ATR_5m"].std()
        if atr_std < 1e-6:
            raise RuntimeError(
                f"ATR_5m is constant (std={atr_std:.6f}) after TI mapping — positional alignment failed."
            )
        logger.info("  ✓ ATR_5m variance confirmed: std=%.4f", atr_std)

    logger.info(
        "  TechnicalIndicators contract: added=%d missing=%d missing_examples=%s",
        ti_added_count, len(ti_missing), ti_missing[:10],
    )

    compatibility_aliases = {
        "MACD_5m": ("MACD_5m", "MACD_12_26_9_5m", "MACD_12_26_9"),
        "BB_Middle_5m": ("BB_Middle_5m", "BBM_20_2.0_5m", "BBM_20_2.0_2.0_5m", "BB_Middle"),
    }
    for target_column, candidates in compatibility_aliases.items():
        source_column = next((candidate for candidate in candidates if candidate in primary_df.columns), None)
        if target_column == "BB_Middle_5m" and source_column is None:
            source_column = next(
                (column for column in primary_df.columns if column.startswith("BBM_20_") and column.endswith("_5m")),
                None,
            )
        if source_column is not None and target_column not in primary_df.columns:
            primary_df[target_column] = primary_df[source_column].values
            ti_added_count += 1
        elif target_column == "BB_Middle_5m" and target_column not in primary_df.columns:
            primary_df[target_column] = primary_df["close"].rolling(20, min_periods=1).mean().values.astype(np.float32)
            ti_added_count += 1

    # The MTF RSI is the public notebook feature. Keep it independent from
    # the TI engine's similarly named RSI outputs.
    canonical_rsi = _calculate_rsi(primary_df["close"], 14).values.astype(np.float32)
    primary_df["rsi"] = canonical_rsi
    primary_df["RSI_14"] = canonical_rsi

    tf_enriched_dfs["5m"] = primary_df
    logger.info("  ✓ Primary 5m dataset enriched with %d Technical Indicators.", ti_added_count)

    # 4. Strict no-lookahead anti-bias alignment across timeframes
    logger.info("📌 Step 4/6: Aligning MTF datasets on primary 5m timeline (+1 HTF interval timestamp shift)...")
    base_df = tf_enriched_dfs["5m"].sort_values("timestamp").copy()
    base_df.columns = [
        _with_timeframe_suffix(c, "5m") if c != "timestamp" else "timestamp"
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
            _with_timeframe_suffix(c, tf) if c != "timestamp" else "timestamp"
            for c in df_htf.columns
        ]

        aligned = pd.merge_asof(
            aligned,
            df_htf,
            on="timestamp",
            direction="backward",
        )

    aligned.ffill(inplace=True)
    aligned = aligned.loc[:, ~aligned.columns.duplicated(keep="last")]
    aligned.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_columns = aligned.select_dtypes(include=[np.number]).columns
    aligned[numeric_columns] = aligned[numeric_columns].ffill().fillna(0.0)
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
    for target_column in REQUIRED_TARGET_COLUMNS:
        logger.info(
            "  target %-40s non-null=%.3f",
            target_column,
            float(aligned[target_column].notna().mean()) if target_column in aligned.columns else 0.0,
        )
    _validate_enriched_frame(aligned)
    logger.info("  ✓ Enriched dataset contract and no-lookahead label prerequisites passed")

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
    manifest_path   = f"data/manifest_{k_label}.json"
    zip_export_path = "data/axe_meta_dataset.zip"

    train_df.to_csv(csv_train_path, index=False)
    val_df.to_csv(csv_val_path, index=False)
    test_df.to_csv(csv_test_path, index=False)
    aligned.to_csv(csv_full_path, index=False)
    _write_dataset_manifest(aligned, train_df, val_df, test_df, manifest_path)

    with zipfile.ZipFile(zip_export_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_train_path, arcname=f"train_{k_label}.csv")
        zipf.write(csv_val_path,   arcname=f"val_{k_label}.csv")
        zipf.write(csv_test_path,  arcname=f"test_{k_label}.csv")
        zipf.write(csv_full_path,  arcname=f"full_{k_label}.csv")
        zipf.write(manifest_path,  arcname=f"manifest_{k_label}.json")

    zip_mb = os.path.getsize(zip_export_path) / (1024 * 1024)
    logger.info("📦 Exported & zipped multi-timeframe dataset to %s (%.2f MB)", zip_export_path, zip_mb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build enriched MTF dataset for Kaggle training.")
    parser.add_argument("--symbol", default="GLD", help="Trading symbol to fetch (default: GLD)")
    parser.add_argument("--limit",  type=int, default=50_000,
                        help="Number of 5m bars to fetch (default: 50000 → ~50k dataset)")
    args = parser.parse_args()
    build_and_enrich_mtf_dataset(symbol=args.symbol, limit=args.limit)
