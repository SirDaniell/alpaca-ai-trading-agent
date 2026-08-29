from typing import Optional, Sequence

import numpy as np

from app.core.ml.signal_meta_learner import (
    ForwardMoveRewardCalculator,
    ForwardMoveStats,
    SIGNAL_META_HORIZON_BARS,
)


def resolve_forward_outcome(
    *,
    calculator: ForwardMoveRewardCalculator,
    signal_id: str,
    symbol: str,
    direction: str,
    entry_price: float,
    future_highs: Sequence[float],
    future_lows: Sequence[float],
    future_closes: Sequence[float],
    atr_pips: Optional[float] = None,
) -> Optional[ForwardMoveStats]:
    """Return a resolved label only when a complete forward horizon exists."""
    if len(future_highs) < SIGNAL_META_HORIZON_BARS:
        return None
    if len(future_lows) < SIGNAL_META_HORIZON_BARS:
        return None
    if len(future_closes) < SIGNAL_META_HORIZON_BARS:
        return None

    return calculator.calculate(
        signal_id=signal_id,
        symbol=symbol,
        direction=direction,
        entry_price=entry_price,
        future_highs=np.asarray(future_highs[:SIGNAL_META_HORIZON_BARS], dtype=np.float64),
        future_lows=np.asarray(future_lows[:SIGNAL_META_HORIZON_BARS], dtype=np.float64),
        future_closes=np.asarray(future_closes[:SIGNAL_META_HORIZON_BARS], dtype=np.float64),
        atr_pips=atr_pips,
    )


def resolve_causal_outcome(
    *,
    direction: str,
    entry_price: float,
    symbol: str,
    forward_bars: Sequence[dict],
    horizon_bars: int = 24,
    signal_id: str = "sig",
    atr_pips: Optional[float] = None,
) -> Optional[dict]:
    """Helper converting forward bar dictionaries into causal outcome statistics."""
    if len(forward_bars) < horizon_bars:
        return None

    highs = [float(b.get("high") if "high" in b else b.get("High", 0.0)) for b in forward_bars[:horizon_bars]]
    lows = [float(b.get("low") if "low" in b else b.get("Low", 0.0)) for b in forward_bars[:horizon_bars]]
    closes = [float(b.get("close") if "close" in b else b.get("Close", 0.0)) for b in forward_bars[:horizon_bars]]

    calc = ForwardMoveRewardCalculator(lookforward_bars=horizon_bars)
    stats = resolve_forward_outcome(
        calculator=calc,
        signal_id=signal_id,
        symbol=symbol,
        direction=direction,
        entry_price=entry_price,
        future_highs=highs,
        future_lows=lows,
        future_closes=closes,
        atr_pips=atr_pips,
    )
    if stats is None:
        return None

    res_bar_idx = horizon_bars - 1
    res_time = forward_bars[res_bar_idx].get("timestamp") or forward_bars[res_bar_idx].get("time")

    return {
        "mfe_pips": stats.mfe_pips,
        "mae_pips": stats.mae_pips,
        "net_pips": stats.net_pips_24h,
        "reversal_bar": stats.reversal_bar,
        "reward": stats.reward,
        "signal_strength": stats.signal_strength,
        "resolution_time": res_time,
    }

