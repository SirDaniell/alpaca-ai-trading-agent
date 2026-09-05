"""
config.py — Configuration parameters for Axe-paka-v1 Agent.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class AxePakaV1Config:
    agent_name: str = "Axe-paka-v1"
    version: str = "1.0.0"

    # Context & Window Settings
    meta_lookback_bars: int = 150
    q_lookback_bars: int = 150
    ctx_dim: int = 28
    num_horizons: int = 4

    # Horizon mapping & Permitted Expiries
    # All 4 heads: 0="5m", 1="15m", 2="30m", 3="1h"
    # User instruction: Allow ALL timeframes (5m, 15m, 30m, 1h) to trade for testing data collection
    horizon_labels: Tuple[str, ...] = ("5m", "15m", "30m", "1h")
    permitted_horizon_indices: Tuple[int, ...] = (0, 1, 2, 3)
    permitted_horizon_labels: Tuple[str, ...] = ("5m", "15m", "30m", "1h")

    # Expiry holding window timers in seconds for Alpaca options tracking
    # 5m = 300s, 15m = 900s, 30m = 1800s, 1h = 3600s
    holding_seconds_map: Dict[int, int] = field(
        default_factory=lambda: {0: 300, 1: 900, 2: 1800, 3: 3600}
    )

    # Decision Gates (relaxed to 0.00 for full testing data collection)
    confidence_threshold: float = 0.00
    horizon_margin: float = 0.00
    eval_epsilon: float = 0.0

    # 4-Step Martingale Money Management System ($10 starting baseline, max 4 steps)
    base_trade_dollars: float = 10.0
    max_martingale_steps: int = 4
    martingale_multiplier: float = 2.0
    max_position_dollars: float = 160.0

    # Risk Controls
    max_daily_drawdown_pct: float = 0.03  # 3% max daily drawdown
    max_concurrent_option_positions: int = 4
    max_risk_per_trade_pct: float = 0.02

    # Default Target Symbols (GLD is primary since model was trained on Gold dataset)
    primary_symbol: str = "GLD"
    symbols: Tuple[str, ...] = ("GLD", "SPY", "QQQ")

    # Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    weights_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "weights")
    fallback_weights_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "checkpoints")

    meta_weights_name: str = "meta_learner_best.pt"
    meta_weights_last_name: str = "meta_learner_last.pt"
    q_weights_name: str = "q_executor_best.pt"
    q_weights_last_name: str = "q_executor_last.pt"
