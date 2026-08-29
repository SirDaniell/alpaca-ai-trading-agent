"""
MetricFeatureEngineer
=====================
Calculates the 4 metric-card feature columns (volatility, speed, direction, regime)
plus their rolling statistics and momentum derivatives.

Called from: analyze_technical_impl() in processing_handlers.py
             AFTER TechnicalIndicators.calculate_all_indicators()

Produced columns (all prefixed with ``metric_``):
  Base targets (same formulas as StockHeader.tsx):
    metric_volatility          – (High-Low)/Close * 100
    metric_speed               – abs((Close-Open)/Open) * 100
    metric_direction           – 1=bullish, 0=bearish
    metric_regime              – 1=volatile (abs change > threshold), 0=stable

  Rolling statistics (window = volatility_window, default 20):
    metric_volatility_ma{N}    – rolling mean of volatility
    metric_volatility_std{N}   – rolling std  of volatility
    metric_speed_ma{N}         – rolling mean of speed
    metric_speed_std{N}        – rolling std  of speed
    metric_bullish_ratio{N}    – fraction of bullish candles in window
    metric_volatile_ratio{N}   – fraction of volatile candles in window

  Momentum (short_window vs long_window):
    metric_volatility_momentum     – direction: +1 increasing, -1 decreasing, 0 stable
    metric_volatility_momentum_str – strength 0-100
    metric_speed_momentum          – direction
    metric_speed_momentum_str      – strength 0-100

These columns flow through the full pipeline:
  TechnicalAnalysis → SNRAnalysis → MLPreparation → ModelTraining
and are available as features OR targets for metric-card model training.

Design notes
------------
* Column names are case-normalised: the class tries 'Close'/'Open'/'High'/'Low'
  first (standard after normalise_dataframe_columns) then lowercase fallbacks.
* All operations are vectorised (no Python loops over rows).
* Returns a copy of the input DataFrame – never mutates in place.
* Mirrors advancedStockMetrics.ts momentum logic exactly so training targets
  match what the frontend displays.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class MetricFeatureConfig:
    """
    Configuration for metric feature generation.

    Added to TechnicalConfig as an optional sub-config.
    Exposed in TechnicalAnalysisStepPanel as new toggle switches under
    "Advanced Features → Metric Card Features".
    """

    enable_metric_features: bool = True

    # Rolling window sizes
    volatility_window: int = 20      # bars for rolling vol/speed stats
    speed_window: int = 20           # kept separate in case user wants different windows

    # Regime threshold: abs(close-open)/open > threshold → volatile
    # Matches StockHeader.tsx: Math.abs(changePercent) > 2 ? "Volatile" : "Stable"
    regime_threshold: float = 2.0    # percent

    # Momentum: short_window mean vs long_window mean
    # Mirrors advancedStockMetrics.ts: recentMomentum (last 3) vs olderMomentum
    momentum_short_window: int = 5
    momentum_long_window: int = 20

    # Whether to include sequence-level features (reserved for future use)
    include_sequence_features: bool = True
    sequence_length: int = 60


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MetricFeatureEngineer:
    """
    Adds metric-card feature columns to an OHLCV DataFrame.

    Usage (from processing_handlers.py)::

        from app.core.ml.metric_feature_engineer import MetricFeatureEngineer, MetricFeatureConfig

        mfe = MetricFeatureEngineer(MetricFeatureConfig())
        result_df = mfe.calculate_all_metric_features(result_df, reporter=reporter)
    """

    # Column name candidates (capitalised first, lowercase fallback)
    _COL_CLOSE = ("Close", "close")
    _COL_OPEN  = ("Open",  "open")
    _COL_HIGH  = ("High",  "high")
    _COL_LOW   = ("Low",   "low")

    def __init__(self, config: MetricFeatureConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def calculate_all_metric_features(
        self,
        df: pd.DataFrame,
        reporter=None,
    ) -> pd.DataFrame:
        """
        Add all metric feature columns to *df* and return the enriched copy.

        Args:
            df:       Input OHLCV DataFrame (already has technical indicators).
            reporter: Optional ProgressReporter for WebSocket updates.

        Returns:
            New DataFrame with ``metric_*`` columns appended.
        """
        if not self.config.enable_metric_features:
            return df

        if df is None or df.empty:
            logger.warning("[MetricFeatureEngineer] Empty DataFrame – skipping")
            return df

        # Validate required columns exist
        missing = self._check_required_columns(df)
        if missing:
            logger.warning(
                f"[MetricFeatureEngineer] Missing OHLCV columns {missing} – skipping metric features"
            )
            return df

        try:
            result = df.copy()
            result = self._calculate_base_metrics(result)
            result = self._calculate_rolling_stats(result)
            result = self._calculate_momentum_features(result)

            added = [c for c in result.columns if c.startswith("metric_")]
            logger.info(
                f"[MetricFeatureEngineer] Added {len(added)} metric columns to DataFrame "
                f"({len(result)} rows)"
            )
            return result

        except Exception as exc:
            logger.error(
                f"[MetricFeatureEngineer] Failed to calculate metric features: {exc}",
                exc_info=True,
            )
            # Return original df unchanged so the pipeline can continue
            return df

    # ------------------------------------------------------------------
    # Private helpers – column resolution
    # ------------------------------------------------------------------

    def _col(self, df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
        """Return the first candidate column name that exists in *df*."""
        for name in candidates:
            if name in df.columns:
                return name
        raise KeyError(f"None of {candidates} found in DataFrame columns")

    def _check_required_columns(self, df: pd.DataFrame) -> list[str]:
        """Return list of missing required column groups."""
        missing = []
        for group in (self._COL_CLOSE, self._COL_OPEN, self._COL_HIGH, self._COL_LOW):
            if not any(c in df.columns for c in group):
                missing.append(group[0])
        return missing

    # ------------------------------------------------------------------
    # Step 1 – Base metric columns
    # ------------------------------------------------------------------

    def _calculate_base_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        4 base metric columns.  Exact same formulas as StockHeader.tsx so that
        training targets match what the frontend displays.

        StockHeader.tsx lines 214-217::

            const volatility = (((high - low) / price) * 100).toFixed(2)
            const priceSpeed = Math.abs(changePercent).toFixed(2)
            const direction  = isPositive ? "Bullish" : "Bearish"
            const regime     = Math.abs(changePercent) > 2 ? "Volatile" : "Stable"
        """
        close = df[self._col(df, self._COL_CLOSE)]
        open_ = df[self._col(df, self._COL_OPEN)]
        high  = df[self._col(df, self._COL_HIGH)]
        low   = df[self._col(df, self._COL_LOW)]

        # Volatility: (High - Low) / Close * 100
        df["metric_volatility"] = ((high - low) / close.replace(0, np.nan)) * 100

        # Speed: abs((Close - Open) / Open) * 100
        df["metric_speed"] = ((close - open_).abs() / open_.replace(0, np.nan)) * 100

        # Direction: 1 = bullish (close > open), 0 = bearish
        df["metric_direction"] = (close > open_).astype(int)

        # Regime: 1 = volatile (abs change > threshold %), 0 = stable
        change_pct = ((close - open_) / open_.replace(0, np.nan)).abs() * 100
        df["metric_regime"] = (change_pct > self.config.regime_threshold).astype(int)

        return df

    # ------------------------------------------------------------------
    # Step 2 – Rolling statistics
    # ------------------------------------------------------------------

    def _calculate_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling statistics for volatility and speed over the configured window.
        Also computes bullish_ratio and volatile_ratio for the same window.
        """
        w = self.config.volatility_window  # single window for all rolling stats

        vol   = df["metric_volatility"]
        speed = df["metric_speed"]
        bull  = df["metric_direction"]
        reg   = df["metric_regime"]

        df[f"metric_volatility_ma{w}"]  = vol.rolling(w, min_periods=1).mean()
        df[f"metric_volatility_std{w}"] = vol.rolling(w, min_periods=2).std().fillna(0)
        df[f"metric_speed_ma{w}"]       = speed.rolling(w, min_periods=1).mean()
        df[f"metric_speed_std{w}"]      = speed.rolling(w, min_periods=2).std().fillna(0)
        df[f"metric_bullish_ratio{w}"]  = bull.rolling(w, min_periods=1).mean()
        df[f"metric_volatile_ratio{w}"] = reg.rolling(w, min_periods=1).mean()

        return df

    # ------------------------------------------------------------------
    # Step 3 – Momentum features
    # ------------------------------------------------------------------

    def _calculate_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Momentum direction and strength for volatility and speed.

        Optimisation (loop unroll + threading):
          The original 2-iteration ``for metric in ('volatility', 'speed')``
          loop is fully unrolled.  Both computations are dispatched to a
          two-thread pool so they execute concurrently; the results are
          written back to the DataFrame after both futures resolve.

        Mirrors advancedStockMetrics.ts::

            if (recentAvgStrength > olderAvgStrength * 1.2) momentum = 'Increasing'
            else if (recentAvgStrength < olderAvgStrength * 0.8) momentum = 'Decreasing'
            else momentum = 'Stable'

        We encode: Increasing = +1, Decreasing = -1, Stable = 0.
        Strength = clipped deviation from 1.0, scaled to 0-100.
        """
        s = self.config.momentum_short_window
        l = self.config.momentum_long_window

        def _compute_momentum(
            series: pd.Series,
        ) -> Tuple[pd.Series, pd.Series]:
            """Return (direction, strength) for one metric series."""
            short_ma = series.rolling(s, min_periods=1).mean()
            long_ma  = series.rolling(l, min_periods=1).mean()
            ratio    = short_ma / long_ma.replace(0, np.nan)

            direction = pd.Series(
                np.where(ratio > 1.2, 1.0, np.where(ratio < 0.8, -1.0, 0.0)),
                index=series.index,
            )
            strength = ((ratio - 1.0).abs().clip(0, 1) * 100).fillna(0)
            return direction, strength

        # ── Unrolled: dispatch volatility and speed concurrently ──────────
        with ThreadPoolExecutor(max_workers=2) as pool:
            vol_future = pool.submit(_compute_momentum, df["metric_volatility"])
            spd_future = pool.submit(_compute_momentum, df["metric_speed"])

            # Unrolled assignments (no loop) — volatility
            vol_dir, vol_str = vol_future.result()
            df["metric_volatility_momentum"]     = vol_dir
            df["metric_volatility_momentum_str"] = vol_str

            # Unrolled assignments — speed
            spd_dir, spd_str = spd_future.result()
            df["metric_speed_momentum"]     = spd_dir
            df["metric_speed_momentum_str"] = spd_str

        return df
