"""Causal OHLCV → SIGNAL_META_FEATURE_KEYS snapshots for the meta-learner."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.core.market.mtf_rsi import calculate_wilder_rsi
from app.core.ml.signal_meta_learner import SIGNAL_META_FEATURE_KEYS


def _finite(value: float, default: float = 0.0) -> float:
    if value is None or not math.isfinite(value):
        return default
    return float(value)


def _rsi_series(candles: List[Dict[str, Any]], period: int = 14) -> List[Optional[float]]:
    points = calculate_wilder_rsi(candles, period=period)
    return [p.get("value") for p in points]


def build_signal_meta_feature_frame(candles: List[Dict[str, Any]]) -> pd.DataFrame:
    """Return one row per candle with the 16-key meta-learner contract."""
    n = len(candles)
    empty = pd.DataFrame({key: np.zeros(n, dtype=np.float32) for key in SIGNAL_META_FEATURE_KEYS})
    if n == 0:
        return empty

    closes = np.array([float(c["close"]) for c in candles], dtype=np.float64)
    highs = np.array([float(c["high"]) for c in candles], dtype=np.float64)
    lows = np.array([float(c["low"]) for c in candles], dtype=np.float64)
    volumes = np.array([float(c.get("volume", 1.0)) for c in candles], dtype=np.float64)
    times = [int(c.get("time") or 0) for c in candles]

    rsi_raw = _rsi_series(candles, period=14)
    rsi = np.array([50.0 if v is None else float(v) for v in rsi_raw], dtype=np.float64)
    rsi_norm = rsi / 100.0

    rsi_ma = pd.Series(rsi_norm).rolling(14, min_periods=1).mean().to_numpy()
    close_s = pd.Series(closes)
    sma20 = close_s.rolling(20, min_periods=1).mean()
    sma50 = close_s.rolling(50, min_periods=1).mean()
    vol_ma = pd.Series(volumes).rolling(20, min_periods=1).mean()
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - np.roll(closes, 1)), np.abs(lows - np.roll(closes, 1))))
    tr[0] = highs[0] - lows[0]
    atr = pd.Series(tr).rolling(14, min_periods=1).mean().to_numpy()
    ret = close_s.pct_change().fillna(0.0)
    vol_short = ret.rolling(10, min_periods=1).std().fillna(0.0).to_numpy()
    vol_long = ret.rolling(40, min_periods=1).std().replace(0, np.nan).fillna(1e-8).to_numpy()
    roll_high = close_s.rolling(20, min_periods=1).max().to_numpy()
    roll_low = close_s.rolling(20, min_periods=1).min().to_numpy()
    rng = np.maximum(roll_high - roll_low, 1e-8)

    bounce = np.zeros(n, dtype=np.float64)
    breakout = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - 19)
        window_low = lows[start:i + 1]
        window_high = highs[start:i + 1]
        if len(window_low) == 0:
            continue
        lo = window_low.min()
        hi = window_high.max()
        bounce[i] = float(np.sum(np.abs(lows[start:i + 1] - lo) <= 0.15 * (hi - lo + 1e-8)))
        breakout[i] = float(np.sum(closes[start:i + 1] >= hi * 0.995) + np.sum(closes[start:i + 1] <= lo * 1.005))

    hour_sin = np.zeros(n, dtype=np.float64)
    hour_cos = np.zeros(n, dtype=np.float64)
    dow = np.zeros(n, dtype=np.float64)
    for i, ts in enumerate(times):
        if ts <= 0:
            continue
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour_sin[i] = math.sin(2 * math.pi * dt.hour / 24.0)
        hour_cos[i] = math.cos(2 * math.pi * dt.hour / 24.0)
        dow[i] = dt.weekday() / 6.0

    frame = pd.DataFrame({
        "rsi": rsi_norm,
        "rsi_ma_diff": rsi_norm - rsi_ma,
        "dxy_rsi_diff": np.zeros(n),
        "csm_strength": ((closes - sma20.to_numpy()) / np.maximum(sma20.to_numpy(), 1e-8)),
        "atr_norm": atr / np.maximum(closes, 1e-8),
        "volatility_surge": vol_short / vol_long,
        "momentum_delta": (closes - np.roll(closes, 5)) / np.maximum(closes, 1e-8),
        "volume_ratio": volumes / np.maximum(vol_ma.to_numpy(), 1e-8),
        "snr_distance": (closes - roll_low) / rng,
        "bounce_count": bounce / 20.0,
        "breakout_count": breakout / 20.0,
        "trend_alignment": np.tanh((sma20 - sma50).to_numpy() / np.maximum(closes, 1e-8) * 50.0),
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "day_of_week": dow,
        "timeframe_scale": np.ones(n),
    })
    frame.iloc[0, frame.columns.get_loc("momentum_delta")] = 0.0
    for key in SIGNAL_META_FEATURE_KEYS:
        frame[key] = frame[key].map(lambda v: _finite(float(v)))
    return frame[list(SIGNAL_META_FEATURE_KEYS)]


def snapshot_at(frame: pd.DataFrame, index: int) -> Dict[str, float]:
    if frame.empty or index < 0 or index >= len(frame):
        return {key: 0.0 for key in SIGNAL_META_FEATURE_KEYS}
    row = frame.iloc[index]
    return {key: _finite(float(row[key])) for key in SIGNAL_META_FEATURE_KEYS}


def infer_direction(frame: pd.DataFrame, index: int) -> str:
    """RSI + trend alignment → a synthetic trade direction the learner can fit."""
    rsi = float(frame.iloc[index]["rsi"]) if index < len(frame) else 0.5
    trend = float(frame.iloc[index]["trend_alignment"]) if index < len(frame) else 0.0
    if rsi < 0.35 or (rsi < 0.5 and trend > 0):
        return "bullish"
    if rsi > 0.65 or (rsi > 0.5 and trend < 0):
        return "bearish"
    return "bullish" if trend >= 0 else "bearish"
