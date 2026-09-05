"""
money_management.py — Martingale Money Management (MM) Engine for Axe-paka-v1.

Implements a 4-Step Capped Martingale Position Sizing Engine:
- Base Allocation: $10 per trade baseline
- Multiplier: 2.0x after each loss
- Max Steps: 4 steps ($10 -> $20 -> $40 -> $80)
- Reset Trigger: Resets back to Step 0 ($10) upon ANY WIN or after reaching max 4 steps.
- Capital Protection: Prevents unbounded exponential drawdown.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MartingaleMMConfig:
    base_trade_dollars: float = 10.0       # Starting trade size ($10)
    martingale_multiplier: float = 2.0    # 2x step progression
    max_martingale_steps: int = 4         # Max 4 steps (0, 1, 2, 3)
    max_position_dollars: float = 160.0   # Hard ceiling cap ($160)


class MartingaleMoneyManager:
    """
    Stateful Martingale Money Management Engine for Axe-paka-v1 option execution.
    Tracks consecutive loss streaks per symbol/account and calculates target trade size/contracts.
    """

    def __init__(self, config: Optional[MartingaleMMConfig] = None):
        self.config = config or MartingaleMMConfig()
        self.consecutive_losses: Dict[str, int] = {}
        self.total_trades: Dict[str, int] = {}
        self.total_wins: Dict[str, int] = {}

    def _key(self, symbol: str, timeframe: str) -> str:
        """Unique state key per symbol + timeframe horizon (e.g. 'GLD_30m')."""
        return f"{symbol.upper()}_{timeframe}"

    def get_step(self, symbol: str = "GLD", timeframe: str = "5m") -> int:
        """Get current Martingale step (0 to max_steps-1) for this symbol+timeframe."""
        loss_streak = self.consecutive_losses.get(self._key(symbol, timeframe), 0)
        return min(loss_streak, self.config.max_martingale_steps - 1)

    def calculate_position_size(
        self,
        symbol: str = "GLD",
        timeframe: str = "5m",
        contract_price: float = 1.0,
    ) -> Tuple[float, int, int]:
        """
        Calculate target position dollar allocation and contract quantity
        for a specific symbol + timeframe combination.

        Returns:
            (target_dollar_allocation, contract_qty, current_step)
        """
        step = self.get_step(symbol, timeframe)
        dollar_size = self.config.base_trade_dollars * (self.config.martingale_multiplier ** step)
        dollar_size = min(dollar_size, self.config.max_position_dollars)

        # Options contracts multiplier is 100 shares per contract
        cost_per_contract = max(contract_price * 100.0, 1.0)
        contract_qty = max(1, int(np_round_qty(dollar_size, cost_per_contract)))

        logger.info(
            "💰 [MartingaleMM] %s | Step: %d/%d | Streak: %d losses | Target: $%.2f | Qty: %d contracts",
            self._key(symbol, timeframe), step + 1, self.config.max_martingale_steps,
            self.consecutive_losses.get(self._key(symbol, timeframe), 0),
            dollar_size, contract_qty
        )
        return dollar_size, contract_qty, step

    def record_trade_result(self, symbol: str, timeframe: str, is_win: bool) -> None:
        """Update Martingale state after trade exit for a specific symbol+timeframe."""
        key = self._key(symbol, timeframe)
        self.total_trades[key] = self.total_trades.get(key, 0) + 1

        if is_win:
            self.total_wins[key] = self.total_wins.get(key, 0) + 1
            prev_streak = self.consecutive_losses.get(key, 0)
            self.consecutive_losses[key] = 0
            logger.info(
                "🎉 [MartingaleMM] WIN on %s! Reset streak %d -> 0 (next: $%.2f)",
                key, prev_streak, self.config.base_trade_dollars
            )
        else:
            current_streak = self.consecutive_losses.get(key, 0) + 1
            if current_streak >= self.config.max_martingale_steps:
                logger.warning(
                    "⚠️ [MartingaleMM] Max 4-step streak reached on %s (%d losses) — capital safety reset.",
                    key, current_streak
                )
                self.consecutive_losses[key] = 0
            else:
                self.consecutive_losses[key] = current_streak
                next_dollars = self.config.base_trade_dollars * (self.config.martingale_multiplier ** current_streak)
                logger.info(
                    "📉 [MartingaleMM] LOSS on %s — streak -> %d (next allocation: $%.2f)",
                    key, current_streak, next_dollars
                )

    def get_stats(self, symbol: str = "GLD", timeframe: str = "5m") -> Dict[str, float]:
        """Return performance & streak stats for a specific symbol+timeframe."""
        key = self._key(symbol, timeframe)
        trades = self.total_trades.get(key, 0)
        wins = self.total_wins.get(key, 0)
        win_rate = (wins / max(trades, 1)) * 100.0
        return {
            "key": key,
            "total_trades": trades,
            "wins": wins,
            "losses": trades - wins,
            "win_rate_pct": win_rate,
            "current_step": self.get_step(symbol, timeframe) + 1,
            "consecutive_losses": self.consecutive_losses.get(key, 0),
        }


def np_round_qty(dollar_size: float, cost_per_contract: float) -> int:
    """Helper to calculate contract quantity."""
    if cost_per_contract <= dollar_size:
        return max(1, int(round(dollar_size / cost_per_contract)))
    # Default minimum 1 contract for options baseline
    return 1
