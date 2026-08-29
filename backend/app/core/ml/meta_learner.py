from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class ForwardMoveStats:
    signal_id: str
    symbol: str
    direction: str
    entry_price: float
    lookforward_bars: int = 24
    mfe_pips: float = 0.0
    mae_pips: float = 0.0
    net_pips_24h: float = 0.0
    reversal_bar: int = -1
    reward: float = 0.0
    signal_strength: float = 0.5


class ForwardMoveRewardCalculator:
    def __init__(self, lookforward_bars: int = 24, pip_scale: Optional[float] = 1.0, alpha_mae_penalty: float = 1.5):
        self.lookforward_bars = lookforward_bars
        self.pip_scale = pip_scale or 1.0
        self.alpha_mae_penalty = alpha_mae_penalty

    def calculate(
        self,
        signal_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        future_highs: np.ndarray,
        future_lows: np.ndarray,
        future_closes: np.ndarray,
        atr_pips: Optional[float] = None,
    ) -> ForwardMoveStats:
        n = min(len(future_closes), self.lookforward_bars)
        if n == 0 or entry_price <= 0:
            return ForwardMoveStats(signal_id, symbol, direction, entry_price)

        atr = atr_pips if atr_pips is not None and atr_pips > 0 else 10.0
        is_bullish = direction.lower() == "bullish"
        highs = future_highs[:n]
        lows = future_lows[:n]
        closes = future_closes[:n]

        if is_bullish:
            fav_diffs = (highs - entry_price) * self.pip_scale
            adv_diffs = (entry_price - lows) * self.pip_scale
            net_diff = (closes[-1] - entry_price) * self.pip_scale
        else:
            fav_diffs = (entry_price - lows) * self.pip_scale
            adv_diffs = (highs - entry_price) * self.pip_scale
            net_diff = (entry_price - closes[-1]) * self.pip_scale

        mfe = float(np.max(fav_diffs)) if len(fav_diffs) > 0 else 0.0
        mae = float(np.max(adv_diffs)) if len(adv_diffs) > 0 else 0.0
        net_24h = float(net_diff)

        reversal_threshold = max(0.5 * mfe, 1.5 * atr)
        reversal_indices = np.where(adv_diffs > reversal_threshold)[0]
        reversal_bar = int(reversal_indices[0]) if len(reversal_indices) > 0 else -1

        raw_reward = (mfe - self.alpha_mae_penalty * mae) / max(atr, 1.0)
        reward = float(math.tanh(raw_reward / 3.0))
        clipped = max(-60.0, min(60.0, raw_reward))
        signal_strength = float(1.0 / (1.0 + math.exp(-clipped)))

        return ForwardMoveStats(
            signal_id=signal_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            lookforward_bars=n,
            mfe_pips=mfe,
            mae_pips=mae,
            net_pips_24h=net_24h,
            reversal_bar=reversal_bar,
            reward=reward,
            signal_strength=signal_strength,
        )
