from __future__ import annotations

from typing import Dict, List, Optional, Tuple


DIVERGENCE_MAX_BARS = 3000
ROLLING_PCT_WINDOW = 14
ZSCORE_CLAMP = 3


def _is_valid_price(value: object) -> bool:
    try:
        num = float(value)
        return num == num and num > 0
    except (TypeError, ValueError):
        return False


def _is_finite_number(value: object) -> bool:
    try:
        num = float(value)
        return num == num and num is not None
    except (TypeError, ValueError):
        return False


def align_trailing(series: List[Optional[float]], length: int) -> List[Optional[float]]:
    cleaned = [value if _is_valid_price(value) else None for value in series]
    if len(cleaned) >= length:
        return cleaned[-length:]
    return [None] * (length - len(cleaned)) + cleaned


def rolling_pct_change(series: List[Optional[float]], lookback: int) -> List[Optional[float]]:
    values: List[Optional[float]] = []
    for index, value in enumerate(series):
        if not _is_valid_price(value):
            values.append(None)
            continue
        if index < lookback:
            values.append(None)
            continue
        ref = series[index - lookback]
        if not _is_valid_price(ref):
            values.append(None)
            continue
        values.append(((float(value) - float(ref)) / float(ref)) * 100.0)
    return values


def rolling_z_score(series: List[Optional[float]], lookback: int) -> List[Optional[float]]:
    min_samples = max(2, min(lookback, 5))
    output: List[Optional[float]] = []

    for index, value in enumerate(series):
        if not _is_finite_number(value):
            output.append(None)
            continue
        start = max(0, index - lookback)
        window = [float(item) for item in series[start : index + 1] if _is_finite_number(item)]
        if len(window) < min_samples:
            output.append(None)
            continue
        mean = sum(window) / len(window)
        variance = sum((item - mean) ** 2 for item in window) / (len(window) - 1)
        std_dev = variance ** 0.5
        if std_dev == 0:
            output.append(0.0)
            continue
        output.append((float(value) - mean) / std_dev)
    return output


def detect_dxy_symbol_cross_signals(
    short_dxy: List[float],
    short_symbol: List[float],
    long_dxy: List[float],
    long_symbol: List[float],
    timestamps: List[int],
) -> List[Dict[str, object]]:
    """Port of MiniSeries/StockChart detectDxySymbolCrossSignals.

    Bull: DXY drops below the symbol on both fast and slow diffs.
    Bear: DXY rises above the symbol on both diffs.
    """
    length = min(len(short_dxy), len(short_symbol), len(long_dxy), len(long_symbol), len(timestamps))
    if length < 4:
        return []
    signals: List[Dict[str, object]] = []
    for cross_bar in range(1, length):
        prev_short = short_dxy[cross_bar - 1] - short_symbol[cross_bar - 1]
        curr_short = short_dxy[cross_bar] - short_symbol[cross_bar]
        prev_long = long_dxy[cross_bar - 1] - long_symbol[cross_bar - 1]
        curr_long = long_dxy[cross_bar] - long_symbol[cross_bar]
        if not all(v == v for v in (prev_short, curr_short, prev_long, curr_long)):
            continue
        signal_type = None
        entered_bull = prev_short >= 0 or prev_long >= 0
        entered_bear = prev_short <= 0 or prev_long <= 0
        if entered_bull and curr_short < 0 and curr_long < 0:
            signal_type = "bull"
        elif entered_bear and curr_short > 0 and curr_long > 0:
            signal_type = "bear"
        if signal_type:
            signals.append({
                "time": timestamps[cross_bar],
                "barIndex": cross_bar,
                "crossBar": cross_bar,
                "type": signal_type,
                "source": "dxy-symbol",
                "provisional": cross_bar == length - 1,
            })
    return signals


def build_unified_divergence_scale(
    asset: List[Optional[float]],
    indices: Dict[str, List[Optional[float]]],
    scale_ids: List[str],
    max_bars: int = DIVERGENCE_MAX_BARS,
    rolling_window: int = ROLLING_PCT_WINDOW,
) -> Dict[str, object]:
    if len(asset) < 2:
        return {
            "length": 0,
            "asset_aligned": [],
            "asset_pct": [],
            "asset_norm": [],
            "indices_aligned": {},
            "indices_pct": {},
            "indices_norm": {},
            "pct_min": 0.0,
            "pct_max": 0.0,
        }

    length = min(max_bars, len(asset))
    asset_aligned = align_trailing(asset, length)
    asset_pct = rolling_pct_change(asset_aligned, rolling_window)
    indices_aligned: Dict[str, List[Optional[float]]] = {}
    indices_pct: Dict[str, List[Optional[float]]] = {}
    indices_norm: Dict[str, List[float]] = {}

    for scale_id in scale_ids:
        raw = indices.get(scale_id, [])
        aligned = align_trailing(raw, length)
        pct = rolling_pct_change(aligned, rolling_window)
        indices_aligned[scale_id] = aligned
        indices_pct[scale_id] = pct
        norm = rolling_z_score(pct, rolling_window)
        indices_norm[scale_id] = [value if value is not None else 0.0 for value in norm]

    asset_norm = rolling_z_score(asset_pct, rolling_window)
    asset_norm_values = [value if value is not None else 0.0 for value in asset_norm]

    return {
        "length": length,
        "asset_aligned": asset_aligned,
        "asset_pct": asset_pct,
        "asset_norm": asset_norm_values,
        "indices_aligned": indices_aligned,
        "indices_pct": indices_pct,
        "indices_norm": indices_norm,
        "pct_min": min(value for value in asset_norm_values if value != 0.0) if any(value != 0.0 for value in asset_norm_values) else 0.0,
        "pct_max": max(value for value in asset_norm_values if value != 0.0) if any(value != 0.0 for value in asset_norm_values) else 0.0,
    }
