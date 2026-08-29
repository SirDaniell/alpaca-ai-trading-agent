"""
Synthetic OHLCV candle generator for testing and validation.

Generates realistic market-like price series with configurable:
  - Trend direction and strength
  - Volatility (intra-candle and inter-candle)
  - Volume patterns
  - Gaps and outliers (optional)

All output is deterministic (seed-based) for reproducibility.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional
import random
import math

logger = logging.getLogger(__name__)


@dataclass
class Candle:
    """OHLCV candle."""
    time: int  # unix timestamp
    open: float
    high: float
    low: float
    close: float
    volume: int


class SyntheticDataGenerator:
    """
    Generate synthetic OHLCV data for testing and validation.
    
    Example:
        >>> gen = SyntheticDataGenerator(seed=42)
        >>> candles = gen.generate_session(symbol="AAPL", num_candles=500)
        >>> len(candles)
        500
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        logger.info(f"SyntheticDataGenerator initialized with seed={seed}")
    
    def generate_session(
        self,
        symbol: str,
        num_candles: int = 500,
        start_price: float = 100.0,
        volatility: float = 0.02,
        trend: float = 0.001,
        start_time: Optional[datetime] = None,
        interval_seconds: int = 3600,  # 1 hour candles
        base_volume: int = 1000000,
    ) -> List[Candle]:
        """
        Generate a series of synthetic candles.
        
        Args:
            symbol: Stock/asset symbol (e.g., "AAPL")
            num_candles: Number of candles to generate
            start_price: Starting close price
            volatility: Intra-candle volatility (std dev as % of price)
            trend: Drift per candle (as % of price)
            start_time: Candle start time (default: now - num_candles * interval)
            interval_seconds: Seconds per candle (default: 3600 = 1H)
            base_volume: Base volume per candle (varies with volatility)
        
        Returns:
            List of Candle objects
        """
        random.seed(self.seed)
        if start_time is None:
            start_time = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc) - timedelta(seconds=num_candles * interval_seconds)
        
        logger.info(
            f"Generating {num_candles} candles for {symbol} "
            f"(start_price={start_price}, volatility={volatility}, trend={trend})"
        )
        
        candles = []
        current_price = start_price
        current_time = start_time
        
        for i in range(num_candles):
            # Open = previous close
            open_price = current_price
            
            # Calculate intra-candle movement
            # 1. Random walk with volatility
            intra_volatility = random.gauss(0, volatility * current_price)
            
            # 2. Trend component
            trend_component = trend * current_price
            
            # 3. High and low (random walk within range)
            high_low_range = abs(intra_volatility) * random.uniform(0.5, 1.5)
            high_price = open_price + max(0, intra_volatility + trend_component + high_low_range)
            low_price = open_price + min(0, intra_volatility + trend_component - high_low_range)
            
            # Close = high/low within range, with trend bias
            close_price = open_price + intra_volatility + trend_component
            close_price = max(low_price, min(high_price, close_price))
            
            # Ensure high >= open, close; low <= open, close
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Volume: inversely correlated with volatility in real markets
            vol_factor = 1.0 + abs(intra_volatility) / (volatility * current_price)
            volume = int(base_volume * vol_factor * random.uniform(0.8, 1.2))
            
            # Add occasional spikes
            if random.random() < 0.05:  # 5% chance of volume spike
                volume = int(volume * random.uniform(1.5, 3.0))
            
            timestamp = int(current_time.timestamp())
            candle = Candle(
                time=timestamp,
                open=round(open_price, 2),
                high=round(high_price, 2),
                low=round(low_price, 2),
                close=round(close_price, 2),
                volume=volume,
            )
            candles.append(candle)
            
            # Update for next iteration
            current_price = close_price
            current_time += timedelta(seconds=interval_seconds)
        
        logger.info(
            f"Generated {len(candles)} candles for {symbol}. "
            f"Price range: {min(c.low for c in candles):.2f} - {max(c.high for c in candles):.2f}"
        )
        
        return candles
    
    def generate_multi_symbol(
        self,
        symbols: List[str],
        num_candles: int = 500,
        start_prices: Optional[dict] = None,
        volatilities: Optional[dict] = None,
        trends: Optional[dict] = None,
        **kwargs
    ) -> dict:
        """
        Generate synthetic data for multiple symbols.
        
        Args:
            symbols: List of symbols (e.g., ["AAPL", "GOOGL", "TSLA"])
            num_candles: Number of candles per symbol
            start_prices: {symbol: price} (default: 100 for all)
            volatilities: {symbol: vol} (default: 0.02 for all)
            trends: {symbol: trend} (default: 0.001 for all)
            **kwargs: Other args passed to generate_session
        
        Returns:
            {symbol: [candles]}
        """
        start_prices = start_prices or {s: 100.0 for s in symbols}
        volatilities = volatilities or {s: 0.02 for s in symbols}
        trends = trends or {s: 0.001 for s in symbols}
        
        result = {}
        for symbol in symbols:
            result[symbol] = self.generate_session(
                symbol=symbol,
                num_candles=num_candles,
                start_price=start_prices.get(symbol, 100.0),
                volatility=volatilities.get(symbol, 0.02),
                trend=trends.get(symbol, 0.001),
                **kwargs
            )
        
        logger.info(f"Generated data for {len(symbols)} symbols")
        return result
    
    def add_pattern(
        self,
        candles: List[Candle],
        pattern_type: str,
        position: Optional[int] = None,
        intensity: float = 1.0,
    ) -> List[Candle]:
        """
        Inject a known pattern into candle series.
        
        Args:
            candles: Existing candles
            pattern_type: "bearish_divergence", "bullish_divergence", "gap", "spike"
            position: Index to apply pattern (default: random)
            intensity: Strength of pattern (0.0 - 2.0)
        
        Returns:
            Modified candle list
        """
        if position is None:
            position = random.randint(50, len(candles) - 50)
        
        logger.info(f"Injecting {pattern_type} pattern at index {position} (intensity={intensity})")
        
        modified = [c for c in candles]  # shallow copy
        
        if pattern_type == "bullish_divergence":
            # Lower low, higher close (pattern for reversal)
            idx = position
            if idx > 0 and idx < len(modified):
                prev = modified[idx - 1]
                modified[idx] = Candle(
                    time=modified[idx].time,
                    open=modified[idx].open,
                    high=prev.high * 0.95,  # Lower high
                    low=prev.low * 1.05,    # Higher low (divergence)
                    close=prev.close * (1.02 * intensity),  # Stronger close
                    volume=int(modified[idx].volume * 1.5),
                )
        
        elif pattern_type == "bearish_divergence":
            # Higher high, lower close
            idx = position
            if idx > 0 and idx < len(modified):
                prev = modified[idx - 1]
                modified[idx] = Candle(
                    time=modified[idx].time,
                    open=modified[idx].open,
                    high=prev.high * (1.02 * intensity),  # Higher high
                    low=prev.low * 0.95,    # Lower low (divergence)
                    close=prev.close * 0.98,  # Weaker close
                    volume=int(modified[idx].volume * 1.5),
                )
        
        elif pattern_type == "gap":
            # Gap up or down
            idx = position
            if idx > 0 and idx < len(modified):
                prev = modified[idx - 1]
                gap = prev.close * 0.03 * intensity * random.choice([-1, 1])
                modified[idx] = Candle(
                    time=modified[idx].time,
                    open=prev.close + gap,
                    high=prev.high + gap,
                    low=prev.low + gap,
                    close=prev.close + gap,
                    volume=int(modified[idx].volume * 2),
                )
        
        elif pattern_type == "spike":
            # Spike up or down within candle
            idx = position
            if idx > 0 and idx < len(modified):
                direction = random.choice([-1, 1])
                spike_mag = modified[idx].close * 0.05 * intensity
                modified[idx] = Candle(
                    time=modified[idx].time,
                    open=modified[idx].open,
                    high=max(modified[idx].high, modified[idx].close + spike_mag * abs(direction)),
                    low=min(modified[idx].low, modified[idx].close - spike_mag * abs(direction)),
                    close=modified[idx].close,
                    volume=int(modified[idx].volume * 3),
                )
        
        return modified
    
    def to_dict_list(self, candles: List[Candle]) -> List[dict]:
        """Convert candles to list of dicts (for JSON serialization)."""
        return [
            {
                "time": c.time,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
            for c in candles
        ]


# ============================================================================
# Demo/Test
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Generate basic session
    gen = SyntheticDataGenerator(seed=42)
    candles = gen.generate_session("AAPL", num_candles=100, start_price=150.0)
    print(f"\nGenerated {len(candles)} candles")
    print(f"First candle: {candles[0]}")
    print(f"Last candle: {candles[-1]}")
    
    # Generate multi-symbol
    multi = gen.generate_multi_symbol(
        symbols=["AAPL", "GOOGL", "TSLA"],
        num_candles=50,
    )
    print(f"\nMulti-symbol data: {list(multi.keys())}")
    
    # Inject pattern
    patterned = gen.add_pattern(candles, "bullish_divergence", intensity=2.0)
    print(f"\nAfter pattern injection: {len(patterned)} candles")
