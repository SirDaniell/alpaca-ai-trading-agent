"""Full TechnicalIndicators contract for the signal meta-learner.

The original product computes 200+ causal TI columns. Previous competition
code truncated that to a 16-key or 32-key subset. This module keeps the full
numeric TI output plus a frozen 48-bar decision window.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from app.core.analysis.technical_indicators import IndicatorConfig, TechnicalIndicators

SIGNAL_META_LOOKBACK_BARS = 48
SIGNAL_META_FEATURE_CONTRACT_VERSION = "signal-meta-ti-seq-48-v1"

_SKIP_COLUMNS = {
    "Open", "High", "Low", "Close", "Volume", "Time", "time_index",
    "session", "Doji_Type", "HA_Candle",
}

CONTEXT_FEATURE_KEYS = (
    "dxy_close",
    "asset_slow_norm",
    "dxy_slow_norm",
    "asset_fast_norm",
    "dxy_fast_norm",
    "slow_diff",
    "fast_diff",
    "regime_strong_asset",
    "regime_weak_asset",
    "regime_weak_dxy",
    "regime_strong_dxy",
    "snr_dist_support",
    "snr_dist_resistance",
    "mtf_snr_confluence",
    "mtf_rsi_asset",
    "mtf_rsi_dxy",
    "mtf_rsi_diff",
    "cross_index_signal",
    "cross_dxy_signal",
    "cross_index_dxy",
    "cross_dxy_symbol",
)


def _ti_numeric_contract() -> Tuple[str, ...]:
    return tuple(sorted(
        column for column in IndicatorConfig().get_output_columns()
        if column not in _SKIP_COLUMNS
    ))


TI_NUMERIC_FEATURE_KEYS = _ti_numeric_contract()
TI_META_FEATURE_KEYS = TI_NUMERIC_FEATURE_KEYS  # full contract, not the old 32-key subset
DECISION_FEATURE_KEYS = TI_NUMERIC_FEATURE_KEYS + CONTEXT_FEATURE_KEYS
DECISION_FEATURE_COUNT = len(DECISION_FEATURE_KEYS)
DECISION_WINDOW_DIM = DECISION_FEATURE_COUNT * SIGNAL_META_LOOKBACK_BARS


def calculate_ti_features(bars: list[Dict[str, Any]]) -> pd.DataFrame:
    """Run the causal TechnicalIndicators pipeline on OHLCV bars."""
    if not bars:
        return pd.DataFrame()

    frame = pd.DataFrame(bars).rename(columns={
        "time": "Time",
        "timestamp": "Time",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })
    required = ["Open", "High", "Low", "Close"]
    if any(column not in frame.columns for column in required):
        return pd.DataFrame()

    frame["Volume"] = pd.to_numeric(frame.get("Volume", 1.0), errors="coerce").fillna(1.0)
    columns = required + ["Volume"] + (["Time"] if "Time" in frame.columns else [])
    return TechnicalIndicators(IndicatorConfig()).calculate_all_indicators(
        frame[columns],
        mode="inference",
    )


def align_ti_numeric_frame(enriched: pd.DataFrame, row_count: int) -> pd.DataFrame:
    """Force the full TI column contract, filling missing/non-finite values with 0."""
    aligned = pd.DataFrame(index=range(row_count), columns=list(TI_NUMERIC_FEATURE_KEYS), dtype=np.float32)
    if enriched is None or enriched.empty:
        return aligned.fillna(0.0)

    source = enriched.reset_index(drop=True)
    n = min(len(source), row_count)
    for key in TI_NUMERIC_FEATURE_KEYS:
        if key not in source.columns:
            continue
        values = pd.to_numeric(source[key], errors="coerce").to_numpy()[:n]
        values = np.where(np.isfinite(values), values, 0.0)
        aligned.iloc[:n, aligned.columns.get_loc(key)] = values.astype(np.float32)
    return aligned.fillna(0.0)


def build_ti_feature_snapshot(enriched: pd.DataFrame, row_index: int) -> Dict[str, float]:
    aligned = align_ti_numeric_frame(enriched, max(row_index + 1, len(enriched) if enriched is not None else 0))
    if aligned.empty or row_index < 0 or row_index >= len(aligned):
        return {key: 0.0 for key in TI_NUMERIC_FEATURE_KEYS}
    row = aligned.iloc[row_index]
    return {key: float(row[key]) for key in TI_NUMERIC_FEATURE_KEYS}


def window_matrix(
    feature_rows: np.ndarray,
    index: int,
    lookback: int = SIGNAL_META_LOOKBACK_BARS,
) -> np.ndarray:
    """Last `lookback` rows ending at `index`, zero-padded on the left."""
    n_features = feature_rows.shape[1] if feature_rows.ndim == 2 else 0
    window = np.zeros((lookback, n_features), dtype=np.float32)
    if feature_rows.size == 0 or index < 0:
        return window
    start = index - lookback + 1
    if start >= 0:
        window[:] = feature_rows[start:index + 1]
    else:
        take = index + 1
        window[lookback - take:] = feature_rows[:take]
    return window


def flatten_window(window: np.ndarray) -> np.ndarray:
    return np.asarray(window, dtype=np.float32).reshape(-1)
