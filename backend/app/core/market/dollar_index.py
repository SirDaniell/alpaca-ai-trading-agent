"""ICE/FINEX Dollar (DXY) index from the official pair basket."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import pandas as pd

from app.core.analysis.currency_index import INDEX_DEFINITIONS, _weighted_product


DOLLAR_PAIRS = tuple(INDEX_DEFINITIONS["Dollar"]["pairs"].keys())
DOLLAR_START_PRICES = {
    "EURUSD": 1.085,
    "USDJPY": 151.2,
    "GBPUSD": 1.268,
    "USDCAD": 1.361,
    "USDSEK": 10.42,
    "USDCHF": 0.887,
}


def dollar_index_closes(pair_closes: Mapping[str, Sequence[float]]) -> list[float]:
    """Geometric USDX: scalar * product(pair ** weight) using close of each basket pair."""
    defn = INDEX_DEFINITIONS["Dollar"]
    missing = [pair for pair in defn["pairs"] if pair not in pair_closes]
    if missing:
        raise KeyError(f"Dollar index needs pair closes for {missing}")
    length = min(len(pair_closes[pair]) for pair in defn["pairs"])
    frame = pd.DataFrame({
        f"close_{pair}": list(pair_closes[pair])[:length]
        for pair in defn["pairs"]
    })
    series = _weighted_product(frame, "close", defn["pairs"], float(defn["scalar"]))
    return [float(v) for v in series.tolist()]


def pair_closes_to_dxy_candles(
    pair_closes: Mapping[str, Sequence[float]],
    times: Sequence[int],
) -> list[dict]:
    """StockChart treats DXY as an OHLC series with close=index value on each bar."""
    closes = dollar_index_closes(pair_closes)
    n = min(len(closes), len(times))
    candles = []
    for i in range(n):
        value = closes[i]
        candles.append({
            "time": int(times[i]),
            "open": value,
            "high": value,
            "low": value,
            "close": value,
            "volume": 0,
        })
    return candles
