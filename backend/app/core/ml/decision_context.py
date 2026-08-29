"""Assemble the last-48-bar decision tensor: TI + DXY miniseries + MTF SNR + crosses."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from app.core.analysis.support_resistance import detect_snr_levels_sequential
from app.core.market.divergence_scale import (
    build_unified_divergence_scale,
    detect_dxy_symbol_cross_signals,
)
from app.core.market.mtf_rsi import calculate_mtf_rsi, calculate_wilder_rsi, detect_mtf_rsi_cross_signals
from app.core.ml.ti_meta_features import (
    CONTEXT_FEATURE_KEYS,
    DECISION_FEATURE_KEYS,
    SIGNAL_META_LOOKBACK_BARS,
    align_ti_numeric_frame,
    calculate_ti_features,
    flatten_window,
    window_matrix,
)

SLOW_WINDOW = 14
FAST_WINDOW = 5
CONFLUENCE_PCT = 0.0015
TF_AGGREGATE = {"H1": 1, "H4": 4, "D1": 24}


def resample_ohlcv(candles: Sequence[Dict[str, Any]], group_size: int) -> List[Dict[str, Any]]:
    if group_size <= 1:
        return list(candles)
    out: List[Dict[str, Any]] = []
    for start in range(0, len(candles) - group_size + 1, group_size):
        chunk = candles[start:start + group_size]
        out.append({
            "time": chunk[-1]["time"],
            "open": chunk[0]["open"],
            "high": max(c["high"] for c in chunk),
            "low": min(c["low"] for c in chunk),
            "close": chunk[-1]["close"],
            "volume": sum(int(c.get("volume") or 0) for c in chunk),
        })
    return out


def _ffill_join(asset_times: Sequence[int], series_times: Sequence[int], series_values: Sequence[float]) -> List[Optional[float]]:
    joined: List[Optional[float]] = []
    j = 0
    last: Optional[float] = None
    n = len(series_times)
    for t in asset_times:
        while j < n and int(series_times[j]) <= int(t):
            val = series_values[j]
            if val is not None and np.isfinite(val):
                last = float(val)
            j += 1
        joined.append(last)
    return joined


def _cross_series(length: int, signals: List[Dict[str, Any]], source: Optional[str] = None) -> np.ndarray:
    out = np.zeros(length, dtype=np.float32)
    for signal in signals:
        if source and signal.get("source") != source:
            continue
        idx = int(signal.get("barIndex") or signal.get("crossBar") or -1)
        if 0 <= idx < length:
            out[idx] = 1.0 if signal.get("type") == "bull" else -1.0
    return out


def _snr_distances(candles: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(candles)
    dist_s = np.zeros(n, dtype=np.float32)
    dist_r = np.zeros(n, dtype=np.float32)
    confluence = np.zeros(n, dtype=np.float32)
    if n < 10:
        return dist_s, dist_r, confluence

    frame = pd.DataFrame(candles).rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume",
    })
    lookback = min(80, n)
    h4 = resample_ohlcv(candles, 4)
    d1 = resample_ohlcv(candles, 24)

    def zones_for(bars: List[Dict[str, Any]]) -> List[Tuple[float, str]]:
        if len(bars) < 10:
            return []
        tf_frame = pd.DataFrame(bars).rename(columns={
            "open": "Open", "high": "High", "low": "Low", "close": "Close",
        })
        raw = detect_snr_levels_sequential(tf_frame, len(tf_frame) - 1, min(lookback, len(tf_frame) - 1))
        return [(float(level[1]), str(level[2])) for level in raw if len(level) >= 3]

    h1_zones = zones_for(candles)
    h4_zones = zones_for(h4)
    d1_zones = zones_for(d1)
    all_zones = h1_zones + h4_zones + d1_zones

    for i in range(n):
        price = float(candles[i]["close"])
        atr = max(float(candles[i]["high"]) - float(candles[i]["low"]), 1e-8)
        supports = [z[0] for z in all_zones if z[1] == "support"]
        resists = [z[0] for z in all_zones if z[1] == "resistance"]
        if supports:
            nearest_s = min(supports, key=lambda z: abs(price - z))
            dist_s[i] = (price - nearest_s) / atr
        if resists:
            nearest_r = min(resists, key=lambda z: abs(price - z))
            dist_r[i] = (nearest_r - price) / atr
        hits = 0
        for price_a, _kind in h1_zones:
            for price_b, _ in h4_zones + d1_zones:
                if abs(price_a - price_b) / max(price_a, 1e-8) <= CONFLUENCE_PCT:
                    hits += 1
        confluence[i] = float(hits)
    return dist_s, dist_r, confluence


def build_decision_feature_matrix(
    candles: List[Dict[str, Any]],
    dxy_candles: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Fetch-path equivalent: OHLCV → full TI → MTF SNR → DXY/miniseries crosses.

    Returns (T, F) matrix in DECISION_FEATURE_KEYS order and the TI frame.
    """
    n = len(candles)
    ti_frame = calculate_ti_features(candles)
    aligned_ti = align_ti_numeric_frame(ti_frame, n)

    asset_close = [float(c["close"]) for c in candles]
    times = [int(c["time"]) for c in candles]
    dxy_close: List[Optional[float]] = [None] * n
    if dxy_candles:
        dxy_close = _ffill_join(
            times,
            [int(c["time"]) for c in dxy_candles],
            [float(c["close"]) for c in dxy_candles],
        )

    dxy_values = [v if v is not None else 0.0 for v in dxy_close]
    slow = build_unified_divergence_scale(
        asset_close, {"Dollar": dxy_values}, ["Dollar"], rolling_window=SLOW_WINDOW,
    )
    fast = build_unified_divergence_scale(
        asset_close, {"Dollar": dxy_values}, ["Dollar"], rolling_window=FAST_WINDOW,
    )
    asset_slow = list(slow["asset_norm"])
    dxy_slow = list(slow["indices_norm"].get("Dollar", [0.0] * n))
    asset_fast = list(fast["asset_norm"])
    dxy_fast = list(fast["indices_norm"].get("Dollar", [0.0] * n))
    # align_trailing may shorten; pad left
    def _pad(values: List[float]) -> List[float]:
        if len(values) >= n:
            return values[-n:]
        return [0.0] * (n - len(values)) + list(values)

    asset_slow, dxy_slow, asset_fast, dxy_fast = map(_pad, (asset_slow, dxy_slow, asset_fast, dxy_fast))

    dxy_symbol_crosses = detect_dxy_symbol_cross_signals(
        dxy_fast, asset_fast, dxy_slow, asset_slow, times,
    )

    tf_map = {
        "H1": candles,
        "H4": resample_ohlcv(candles, TF_AGGREGATE["H4"]),
        "D1": resample_ohlcv(candles, TF_AGGREGATE["D1"]),
    }
    asset_mtf = calculate_mtf_rsi("H1", tf_map)
    dxy_tf_map = tf_map
    if dxy_candles:
        dxy_tf_map = {
            "H1": dxy_candles,
            "H4": resample_ohlcv(dxy_candles, 4),
            "D1": resample_ohlcv(dxy_candles, 24),
        }
    dxy_mtf = calculate_mtf_rsi("H1", dxy_tf_map)

    asset_rsi = asset_mtf.get("weighted_rsi") or calculate_wilder_rsi(candles, period=7)
    dxy_rsi_points = dxy_mtf.get("weighted_rsi") or []
    if not dxy_rsi_points and dxy_candles:
        dxy_rsi_points = calculate_wilder_rsi(dxy_candles, period=7)

    dxy_rsi_aligned = _ffill_join(
        times,
        [int(p.get("time") or 0) for p in dxy_rsi_points],
        [p.get("value") if p.get("value") is not None else float("nan") for p in dxy_rsi_points],
    )
    asset_rsi_vals = [p.get("value") for p in asset_rsi]
    if len(asset_rsi_vals) != n:
        asset_rsi_vals = _ffill_join(times, [int(p.get("time") or 0) for p in asset_rsi], asset_rsi_vals)

    index_pts = [{"time": times[i], "value": asset_rsi_vals[i]} for i in range(n)]
    dxy_pts = [{"time": times[i], "value": dxy_rsi_aligned[i]} for i in range(n)]
    mtf_crosses = detect_mtf_rsi_cross_signals(index_pts, dxy_pts)

    dist_s, dist_r, confluence = _snr_distances(candles)
    cross_index_signal = _cross_series(n, mtf_crosses, "index-signal")
    cross_dxy_signal = _cross_series(n, mtf_crosses, "dxy-signal")
    cross_index_dxy = _cross_series(n, mtf_crosses, "index-dxy")
    cross_dxy_symbol = _cross_series(n, dxy_symbol_crosses, "dxy-symbol")

    context = pd.DataFrame(index=range(n))
    context["dxy_close"] = dxy_values
    context["asset_slow_norm"] = asset_slow
    context["dxy_slow_norm"] = dxy_slow
    context["asset_fast_norm"] = asset_fast
    context["dxy_fast_norm"] = dxy_fast
    slow_diff = np.array(asset_slow, dtype=np.float64) - np.array(dxy_slow, dtype=np.float64)
    fast_diff = np.array(asset_fast, dtype=np.float64) - np.array(dxy_fast, dtype=np.float64)
    context["slow_diff"] = slow_diff
    context["fast_diff"] = fast_diff
    context["regime_strong_asset"] = ((slow_diff >= 0) & (fast_diff >= 0)).astype(np.float32)
    context["regime_weak_asset"] = ((slow_diff >= 0) & (fast_diff < 0)).astype(np.float32)
    context["regime_weak_dxy"] = ((slow_diff < 0) & (fast_diff >= 0)).astype(np.float32)
    context["regime_strong_dxy"] = ((slow_diff < 0) & (fast_diff < 0)).astype(np.float32)
    context["snr_dist_support"] = dist_s
    context["snr_dist_resistance"] = dist_r
    context["mtf_snr_confluence"] = confluence
    context["mtf_rsi_asset"] = [0.0 if v is None else float(v) for v in asset_rsi_vals]
    context["mtf_rsi_dxy"] = [0.0 if v is None else float(v) for v in dxy_rsi_aligned]
    context["mtf_rsi_diff"] = context["mtf_rsi_asset"] - context["mtf_rsi_dxy"]
    context["cross_index_signal"] = cross_index_signal
    context["cross_dxy_signal"] = cross_dxy_signal
    context["cross_index_dxy"] = cross_index_dxy
    context["cross_dxy_symbol"] = cross_dxy_symbol
    for key in CONTEXT_FEATURE_KEYS:
        context[key] = pd.to_numeric(context[key], errors="coerce").fillna(0.0)

    combined = pd.concat([aligned_ti.reset_index(drop=True), context[list(CONTEXT_FEATURE_KEYS)]], axis=1)
    combined = combined.reindex(columns=list(DECISION_FEATURE_KEYS)).fillna(0.0)
    matrix = combined.to_numpy(dtype=np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix, ti_frame


def decision_vector_at(matrix: np.ndarray, index: int, lookback: int = SIGNAL_META_LOOKBACK_BARS) -> np.ndarray:
    return flatten_window(window_matrix(matrix, index, lookback=lookback))


def last_bar_snapshot(matrix: np.ndarray, index: int) -> Dict[str, float]:
    if matrix.size == 0 or index < 0 or index >= len(matrix):
        return {key: 0.0 for key in DECISION_FEATURE_KEYS}
    row = matrix[index]
    return {key: float(row[i]) for i, key in enumerate(DECISION_FEATURE_KEYS)}


def infer_direction_from_row(snapshot: Dict[str, float]) -> str:
    rsi = snapshot.get("mtf_rsi_asset") or snapshot.get("RSI_14") or 50.0
    slow_diff = snapshot.get("slow_diff", 0.0)
    if snapshot.get("cross_index_dxy", 0.0) > 0 or snapshot.get("cross_dxy_symbol", 0.0) > 0:
        return "bullish"
    if snapshot.get("cross_index_dxy", 0.0) < 0 or snapshot.get("cross_dxy_symbol", 0.0) < 0:
        return "bearish"
    if rsi < 40 or slow_diff > 0:
        return "bullish"
    return "bearish"
