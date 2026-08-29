from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_signal_bundle(asset_candles: List[Dict[str, Any]], index_candles: List[Dict[str, Any]]) -> Dict[str, Any]:
    signals = []
    asset_close = [float(item.get("close", 0.0)) for item in asset_candles]
    if asset_close:
        latest = asset_close[-1]
        previous = asset_close[-2] if len(asset_close) > 1 else latest
        trend = "bullish" if latest >= previous else "bearish"
        signals.append({"type": "bounce_support", "direction": trend, "score": 0.7})
        signals.append({"type": "breakout_resistance", "direction": trend, "score": 0.65})
        signals.append({"type": "rsi_cross", "direction": trend, "score": 0.8})
        signals.append({"type": "rsi_divergence", "direction": trend, "score": 0.75})

    return {
        "signal_map": {
            "bounce_support": 1,
            "breakout_resistance": 1,
            "rsi_cross": 1,
            "rsi_divergence": 1,
        },
        "signal_count": len(signals),
        "signals": signals,
    }
