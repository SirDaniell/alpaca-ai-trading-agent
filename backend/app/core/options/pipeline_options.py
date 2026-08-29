"""
pipeline_options.py — Pipeline configuration for Options Trading on Equity/ETF Underlyings.

Directives:
- Extract instrument-agnostic core signal engine.
- Operates on equity/ETF underlyings (SPY, QQQ, AAPL, NVDA, TSLA).
- Decouples DXY basket (which remains in pipeline_fx.py).
- Interfaces with Tier 1 Meta-Learner (HTF bias) and Tier 2 Q-Learner Executor (LTF order entry & zone discipline).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class OptionsPipelineConfig:
    """Pipeline settings for Options Trading on Alpaca."""
    symbols: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "AAPL", "NVDA", "TSLA"])
    htf_timeframes: List[str] = field(default_factory=lambda: ["H1", "H4", "D1"])
    ltf_execution_timeframes: List[str] = field(default_factory=lambda: ["5m", "15m"])
    active_ltf_timeframe: str = "5m"

    # Zone & Action Mask Parameters
    proximity_atr_mult: float = 0.75
    require_volume_confirm: bool = True
    max_reentries_per_window: int = 3

    # Risk Gate Limits (Non-Bypassable)
    max_daily_drawdown_pct: float = 0.03   # 3% max daily account loss
    max_position_risk_pct: float = 0.02    # 2% max risk per trade
    max_concurrent_option_positions: int = 4

    # Meta-Learner Conviction Thresholds
    min_bias_conviction: float = 0.65
    max_reversal_suppression: float = 0.35

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbols": self.symbols,
            "htf_timeframes": self.htf_timeframes,
            "ltf_execution_timeframes": self.ltf_execution_timeframes,
            "active_ltf_timeframe": self.active_ltf_timeframe,
            "proximity_atr_mult": self.proximity_atr_mult,
            "require_volume_confirm": self.require_volume_confirm,
            "max_reentries_per_window": self.max_reentries_per_window,
            "max_daily_drawdown_pct": self.max_daily_drawdown_pct,
            "max_position_risk_pct": self.max_position_risk_pct,
            "max_concurrent_option_positions": self.max_concurrent_option_positions,
            "min_bias_conviction": self.min_bias_conviction,
            "max_reversal_suppression": self.max_reversal_suppression,
        }
