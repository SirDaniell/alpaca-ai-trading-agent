from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

MTF_RSI_TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]
MTF_RSI_PERIOD = 7
MTF_RSI_SIGNAL_PERIOD = 14
MTF_RSI_WEIGHTS = [0.4, 0.6, 0.7, 0.4]


def _normalize_timeframe(timeframe: str) -> str:
    raw = (timeframe or "").upper()
    mapping = {
        "1M": "M1",
        "5M": "M5",
        "15M": "M15",
        "30M": "M30",
        "1H": "H1",
        "4H": "H4",
        "1D": "D1",
    }
    tf = mapping.get(raw, raw)
    return tf if tf in MTF_RSI_TIMEFRAMES else "H1"


def _as_number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        f = float(value)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _get_time(candle: Dict[str, Any]) -> int:
    raw = candle.get("time") if candle.get("time") is not None else candle.get("timestamp")
    t = _as_number(raw, 0.0)
    return int(t // 1000) if t > 1e11 else int(t)


def _rsi_from_averages(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0.0:
        return 50.0 if avg_gain == 0.0 else 100.0
    if avg_gain == 0.0:
        return 0.0
    return 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))


def calculate_wilder_rsi_detailed(
    data: List[Dict[str, Any]],
    period: int = MTF_RSI_PERIOD,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Returns (points, states) where state contains internal Wilder parameters."""
    if period < 1 or not data:
        return [], []

    points: List[Dict[str, Any]] = [{"time": _get_time(c), "value": None} for c in data]
    states: List[Dict[str, Any]] = [
        {
            "avg_gain": 0.0,
            "avg_loss": 0.0,
            "close": _as_number(c.get("close")),
            "time": _get_time(c),
            "rsi": None,
        }
        for c in data
    ]

    if len(data) <= period:
        return points, states

    gain_sum = 0.0
    loss_sum = 0.0
    for index in range(1, period + 1):
        change = _as_number(data[index].get("close")) - _as_number(data[index - 1].get("close"))
        gain_sum += max(change, 0.0)
        loss_sum += max(-change, 0.0)

    avg_gain = gain_sum / period
    avg_loss = loss_sum / period
    rsi = _rsi_from_averages(avg_gain, avg_loss)

    points[period] = {"time": _get_time(data[period]), "value": float(rsi)}
    states[period] = {
        "avg_gain": avg_gain,
        "avg_loss": avg_loss,
        "close": _as_number(data[period].get("close")),
        "time": _get_time(data[period]),
        "rsi": rsi,
    }

    for index in range(period + 1, len(data)):
        change = _as_number(data[index].get("close")) - _as_number(data[index - 1].get("close"))
        avg_gain = ((avg_gain * (period - 1)) + max(change, 0.0)) / period
        avg_loss = ((avg_loss * (period - 1)) + max(-change, 0.0)) / period
        rsi = _rsi_from_averages(avg_gain, avg_loss)

        t = _get_time(data[index])
        points[index] = {"time": t, "value": float(rsi)}
        states[index] = {
            "avg_gain": avg_gain,
            "avg_loss": avg_loss,
            "close": _as_number(data[index].get("close")),
            "time": t,
            "rsi": rsi,
        }

    return points, states


def calculate_wilder_rsi(data: List[Dict[str, Any]], period: int = MTF_RSI_PERIOD) -> List[Dict[str, Any]]:
    points, _ = calculate_wilder_rsi_detailed(data, period=period)
    return points


def calculate_moving_average(values: List[Dict[str, Any]], period: int = MTF_RSI_SIGNAL_PERIOD) -> List[Dict[str, Any]]:
    if period < 1:
        return [{"time": item.get("time"), "value": None} for item in values]

    window: List[float] = []
    total = 0.0
    out: List[Dict[str, Any]] = []
    for point in values:
        val = point.get("value")
        if val is None or not math.isfinite(float(val)):
            window.clear()
            total = 0.0
            out.append({"time": point.get("time"), "value": None})
            continue
        numeric = float(val)
        window.append(numeric)
        total += numeric
        if len(window) > period:
            total -= window.pop(0)
        out.append({"time": point.get("time"), "value": (total / len(window)) if len(window) == period else None})
    return out


def _latest_state_at_or_before(states: List[Dict[str, Any]], target_time: int) -> Optional[Dict[str, Any]]:
    low, high = 0, len(states) - 1
    match = None
    while low <= high:
        mid = (low + high) // 2
        if states[mid]["time"] <= target_time:
            match = states[mid]
            low = mid + 1
        else:
            high = mid - 1
    return match


def _latest_state_before(states: List[Dict[str, Any]], target_time: int) -> Optional[Dict[str, Any]]:
    low, high = 0, len(states) - 1
    match = None
    while low <= high:
        mid = (low + high) // 2
        if states[mid]["time"] < target_time:
            match = states[mid]
            low = mid + 1
        else:
            high = mid - 1
    return match


def calculate_mtf_rsi(
    timeframe: str,
    all_timeframes: Dict[str, List[Dict[str, Any]]],
    period: int = MTF_RSI_PERIOD,
) -> Dict[str, Any]:
    tf = _normalize_timeframe(timeframe)
    base_data = all_timeframes.get(tf, [])
    if not base_data:
        return {
            "timeframe": tf,
            "weighted_rsi": [],
            "component_rsi": {},
            "delta": [],
            "signal_delta": [],
            "signal_lines": {"index": [], "dxy": []},
            "cross_signals": [],
            "divergences": [],
        }

    start_idx = MTF_RSI_TIMEFRAMES.index(tf) if tf in MTF_RSI_TIMEFRAMES else 4
    included_tfs = MTF_RSI_TIMEFRAMES[start_idx:]

    detailed_htf: Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = {}
    component_rsi: Dict[str, List[Dict[str, Any]]] = {}

    for t in included_tfs:
        tf_candles = all_timeframes.get(t)
        if tf_candles:
            pts, states = calculate_wilder_rsi_detailed(tf_candles, period=period)
            detailed_htf[t] = (pts, states)
            component_rsi[t] = pts

    weighted_rsi: List[Dict[str, Any]] = []
    for candle in base_data:
        candle_time = _get_time(candle)
        candle_close = _as_number(candle.get("close"))
        weighted_sum = 0.0
        total_weight = 0.0

        for idx, t in enumerate(included_tfs):
            if t not in detailed_htf:
                continue
            _pts, states = detailed_htf[t]
            if not states:
                continue

            weight = MTF_RSI_WEIGHTS[min(idx, len(MTF_RSI_WEIGHTS) - 1)]
            rsi_val: Optional[float] = None

            if t == tf:
                state = _latest_state_at_or_before(states, candle_time)
                if state and state["time"] == candle_time:
                    rsi_val = state["rsi"]
            else:
                state = _latest_state_before(states, candle_time)
                if state and state["rsi"] is not None:
                    change = candle_close - state["close"]
                    prov_gain = ((state["avg_gain"] * (period - 1)) + max(change, 0.0)) / period
                    prov_loss = ((state["avg_loss"] * (period - 1)) + max(-change, 0.0)) / period
                    rsi_val = _rsi_from_averages(prov_gain, prov_loss)

            if rsi_val is not None and math.isfinite(rsi_val):
                weighted_sum += rsi_val * weight
                total_weight += weight

        final_val = (weighted_sum / total_weight) if total_weight > 0 else None
        weighted_rsi.append({"time": candle_time, "value": final_val})

    signal_line = calculate_moving_average(weighted_rsi, period=MTF_RSI_SIGNAL_PERIOD)

    return {
        "timeframe": tf,
        "weighted_rsi": weighted_rsi,
        "component_rsi": component_rsi,
        "delta": weighted_rsi,
        "signal_delta": weighted_rsi,
        "signal_lines": {"index": signal_line, "dxy": signal_line},
        "cross_signals": [],
        "divergences": [],
    }


def _detect_pair_crosses(
    first: List[Dict[str, Any]],
    second: List[Dict[str, Any]],
    source: str,
) -> List[Dict[str, Any]]:
    signals: List[Dict[str, Any]] = []
    length = min(len(first), len(second))
    for idx in range(1, length):
        prev_a = first[idx - 1].get("value")
        prev_b = second[idx - 1].get("value")
        curr_a = first[idx].get("value")
        curr_b = second[idx].get("value")
        if None in (prev_a, prev_b, curr_a, curr_b):
            continue
        if prev_a < prev_b and curr_a >= curr_b:
            signals.append({
                "time": first[idx].get("time"),
                "barIndex": idx,
                "crossBar": idx,
                "type": "bull",
                "source": source,
            })
        if prev_a > prev_b and curr_a <= curr_b:
            signals.append({
                "time": first[idx].get("time"),
                "barIndex": idx,
                "crossBar": idx,
                "type": "bear",
                "source": source,
            })
    return signals


def detect_mtf_rsi_cross_signals(
    index_rsi: List[Dict[str, Any]],
    dxy_rsi: List[Dict[str, Any]],
    signal_period: int = MTF_RSI_SIGNAL_PERIOD,
) -> List[Dict[str, Any]]:
    index_signal = calculate_moving_average(index_rsi, period=signal_period)
    dxy_signal = calculate_moving_average(dxy_rsi, period=signal_period)
    signals = [
        *_detect_pair_crosses(index_rsi, index_signal, "index-signal"),
        *_detect_pair_crosses(dxy_rsi, dxy_signal, "dxy-signal"),
        *_detect_pair_crosses(index_rsi, dxy_rsi, "index-dxy"),
    ]
    signals.sort(key=lambda item: (int(item.get("barIndex") or 0), str(item.get("source"))))
    return signals
