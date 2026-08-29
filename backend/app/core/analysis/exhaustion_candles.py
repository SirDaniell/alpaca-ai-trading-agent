"""Closed-candle exhaustion detection using OHLCV and existing SNR zones."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExhaustionConfig:
    volume_window: int = 20
    volume_zscore_threshold: float = 2.0
    snr_proximity_pct: float = 0.004
    rejection_ratio_threshold: float = 0.30
    minimum_score: float = 0.60
    h1_weight: float = 0.70
    h4_weight: float = 0.30
    rectangle_bars: int = 3


@dataclass(frozen=True)
class ExhaustionEvent:
    symbol: str
    source_timeframe: str
    event_candle_time: Any
    direction: str
    score: float
    event_bull_volume: float
    event_bear_volume: float
    rolling_bull_volume: float
    rolling_bear_volume: float
    event_volume: float
    rolling_volume: float
    volume_ratio: float
    volume_zscore: float
    body_ratio: float
    rejection_ratio: float
    snr_distance_pct: float
    mtf_snr_confluence: bool
    confirmed_close: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    for candidate in (name, name.lower(), name.upper()):
        if candidate in frame.columns:
            return pd.to_numeric(frame[candidate], errors="coerce")
    raise ValueError(f"Missing OHLCV column: {name}")


def _zone_prices(zones: Iterable[Any]) -> list[float]:
    prices: list[float] = []
    for zone in zones:
        try:
            price = float(zone[1] if isinstance(zone, (tuple, list)) else zone.get("price"))
        except (AttributeError, IndexError, TypeError, ValueError):
            continue
        if np.isfinite(price) and price > 0:
            prices.append(price)
    return prices


def _nearest_zone_distance(price: float, zones: Iterable[Any]) -> float:
    prices = _zone_prices(zones)
    if not prices or not np.isfinite(price) or price <= 0:
        return float("inf")
    return min(abs(price - level) / price for level in prices)


def detect_exhaustion_candles(
    data: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    snr_zones: Sequence[Any] = (),
    mtf_snr_zones: Sequence[Any] = (),
    config: ExhaustionConfig | None = None,
) -> list[ExhaustionEvent]:
    """Return confirmed exhaustion events without using future rows.

    The row at ``index`` is eligible only when it is a completed candle and at
    least ``volume_window`` rows precede it. The event row is never included in
    its rolling volume baseline.
    """
    cfg = config or ExhaustionConfig()
    if cfg.volume_window < 2:
        raise ValueError("volume_window must be at least 2")

    open_price = _column(data, "Open")
    high = _column(data, "High")
    low = _column(data, "Low")
    close = _column(data, "Close")
    volume = _column(data, "Volume")

    events: list[ExhaustionEvent] = []
    for index in range(cfg.volume_window, len(data)):
        previous_volume = volume.iloc[index - cfg.volume_window:index]
        event_volume = float(volume.iloc[index])
        if not np.isfinite(event_volume) or previous_volume.empty:
            continue

        baseline_mean = float(previous_volume.mean())
        baseline_std = float(previous_volume.std(ddof=0))
        volume_zscore = (event_volume - baseline_mean) / baseline_std if baseline_std > 0 else 0.0
        volume_ratio = event_volume / baseline_mean if baseline_mean > 0 else 0.0
        # Z-score is preferred, but ratio keeps detection useful for broker feeds
        # whose rolling volume variance is zero or artificially compressed.
        if (
            volume_zscore < cfg.volume_zscore_threshold
            and volume_ratio < 2.0
        ):
            continue

        candle_range = float(high.iloc[index] - low.iloc[index])
        if not np.isfinite(candle_range) or candle_range <= 0:
            continue
        body = abs(float(close.iloc[index] - open_price.iloc[index]))
        body_ratio = min(1.0, body / candle_range)
        upper_wick = float(high.iloc[index] - max(open_price.iloc[index], close.iloc[index]))
        lower_wick = float(min(open_price.iloc[index], close.iloc[index]) - low.iloc[index])
        rejection_ratio = max(upper_wick, lower_wick) / candle_range
        if rejection_ratio < cfg.rejection_ratio_threshold:
            continue

        close_location = (float(close.iloc[index]) - float(low.iloc[index])) / candle_range
        bullish = lower_wick > upper_wick and close_location >= 0.50
        bearish = upper_wick >= lower_wick and close_location <= 0.50
        if not bullish and not bearish:
            continue

        price = float(close.iloc[index])
        snr_distance_pct = _nearest_zone_distance(price, snr_zones)
        mtf_distance_pct = _nearest_zone_distance(price, mtf_snr_zones)
        if min(snr_distance_pct, mtf_distance_pct) > cfg.snr_proximity_pct:
            continue

        confluence = (
            np.isfinite(snr_distance_pct)
            and np.isfinite(mtf_distance_pct)
            and abs(snr_distance_pct - mtf_distance_pct) <= cfg.snr_proximity_pct
        )
        volume_score = min(
            1.0,
            max(
                volume_zscore / (cfg.volume_zscore_threshold * 2.0),
                volume_ratio / 4.0,
            ),
        )
        rejection_score = min(1.0, rejection_ratio)
        zone_score = 1.0 if confluence else 0.75
        score = min(1.0, 0.45 * volume_score + 0.30 * rejection_score + 0.25 * zone_score)
        if score < cfg.minimum_score:
            continue

        event_bull_volume = event_volume if bullish else 0.0
        event_bear_volume = event_volume if bearish else 0.0
        previous_bull = previous_volume.where(close.iloc[index - cfg.volume_window:index] >= open_price.iloc[index - cfg.volume_window:index], 0.0)
        previous_bear = previous_volume.where(close.iloc[index - cfg.volume_window:index] < open_price.iloc[index - cfg.volume_window:index], 0.0)
        events.append(ExhaustionEvent(
            symbol=symbol,
            source_timeframe=timeframe,
            event_candle_time=data.index[index],
            direction="bullish" if bullish else "bearish",
            score=score,
            event_bull_volume=event_bull_volume,
            event_bear_volume=event_bear_volume,
            rolling_bull_volume=float(previous_bull.sum()),
            rolling_bear_volume=float(previous_bear.sum()),
            event_volume=event_volume,
            rolling_volume=float(previous_volume.sum()),
            volume_ratio=volume_ratio,
            volume_zscore=volume_zscore,
            body_ratio=body_ratio,
            rejection_ratio=rejection_ratio,
            snr_distance_pct=min(snr_distance_pct, mtf_distance_pct),
            mtf_snr_confluence=confluence,
        ))

    return events


def combine_h1_h4_exhaustion(
    h1_events: Sequence[ExhaustionEvent],
    h4_events: Sequence[ExhaustionEvent],
    *,
    config: ExhaustionConfig | None = None,
) -> list[dict[str, Any]]:
    """Combine aligned events while preserving valid H1-only signals."""
    cfg = config or ExhaustionConfig()
    combined: list[dict[str, Any]] = []

    def event_timestamp(value: Any) -> pd.Timestamp:
        return pd.Timestamp(value)

    # H1 and H4 events are emitted on their own candle opens, so exact timestamp
    # equality would miss a valid H4 event for three of the four H1 bars inside it.
    h4_by_direction = {
        event.direction: sorted(h4_events, key=lambda item: event_timestamp(item.event_candle_time))
        for event in h4_events
    }

    for h1 in h1_events:
        h1_time = event_timestamp(h1.event_candle_time)
        h4 = next(
            (
                candidate
                for candidate in reversed(h4_by_direction.get(h1.direction, []))
                if event_timestamp(candidate.event_candle_time) <= h1_time
                and h1_time - event_timestamp(candidate.event_candle_time) <= pd.Timedelta(hours=4)
            ),
            None,
        )
        score = cfg.h1_weight * h1.score + (cfg.h4_weight * h4.score if h4 else 0.0)
        item = h1.to_dict()
        item.update({
            "composite_score": score,
            "h1_score": h1.score,
            "h4_score": h4.score if h4 else None,
            "h4_event": h4.to_dict() if h4 else None,
        })
        combined.append(item)

    h1_times = [(event.direction, event_timestamp(event.event_candle_time)) for event in h1_events]
    for h4 in h4_events:
        h4_time = event_timestamp(h4.event_candle_time)
        has_h1_match = any(
            direction == h4.direction
            and h1_time >= h4_time
            and h1_time - h4_time <= pd.Timedelta(hours=4)
            for direction, h1_time in h1_times
        )
        if not has_h1_match:
            item = h4.to_dict()
            item.update({"composite_score": h4.score, "h1_score": None, "h4_score": h4.score, "h4_event": None})
            combined.append(item)

    return combined
