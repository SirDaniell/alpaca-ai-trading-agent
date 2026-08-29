"""
Autonomous agent loop.
Runs on a schedule (cron / APScheduler) independent of any client UI.
This is the core of the hackathon submission — it must:
  1. Pull market data
  2. Make a trading decision (options strategy)
  3. Execute via Alpaca Trading API
  4. Log the decision + result
"""

import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from app.utils.alpaca_client import AlpacaClient
from app.core.market.mtf_rsi import calculate_mtf_rsi, calculate_wilder_rsi
from app.core.market.divergence_scale import build_unified_divergence_scale
from app.core.market.signal_events import build_signal_bundle
from app.db.connection import SessionLocal
from app.db.models import SignalOutcome

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SYMBOLS = os.getenv("MARKET_SYMBOLS", "AAPL,TSLA,GOOGL,MSFT").split(",")
TIMEFRAME = os.getenv("MARKET_TIMEFRAME", "1min")
LOOKBACK = int(os.getenv("MARKET_LOOKBACK_BARS", "500"))


def fetch_market_data(client: AlpacaClient, symbol: str) -> dict:
    """Fetch OHLCV bars from Alpaca."""
    bars = client.get_bars(symbol, timeframe=TIMEFRAME, limit=LOOKBACK)
    if not bars:
        logger.warning(f"No bars fetched for {symbol}")
        return None
    
    # Convert Alpaca format to internal format
    candles = []
    for bar in bars:
        candles.append({
            "time": int(datetime.fromisoformat(bar["t"].replace("Z", "+00:00")).timestamp()),
            "open": float(bar["o"]),
            "high": float(bar["h"]),
            "low": float(bar["l"]),
            "close": float(bar["c"]),
            "volume": int(bar["v"]),
        })
    
    return sorted(candles, key=lambda x: x["time"])


def generate_signals(candles: list) -> dict:
    """Generate MTF RSI and signal bundle from candles."""
    if len(candles) < 30:
        return None
    
    # Calculate Wilder RSI for each timeframe simulation
    rsi_result = calculate_wilder_rsi(candles, period=14)
    
    # Build signal bundle (bounce/breakout detection)
    signal_bundle = build_signal_bundle(candles, [])
    
    # Build divergence scale
    try:
        divergence = build_unified_divergence_scale(candles)
    except Exception as e:
        logger.warning(f"Divergence scale calculation failed: {e}")
        divergence = {}
    
    return {
        "rsi": rsi_result,
        "signals": signal_bundle,
        "divergence": divergence,
    }


def log_signal_outcome(db: Session, symbol: str, signal_data: dict, entry_price: float):
    """Persist signal outcome to database."""
    try:
        outcome = SignalOutcome(
            signal_id=f"{symbol}_{datetime.utcnow().isoformat()}",
            symbol=symbol,
            source_timeframe="1min",
            direction="UNKNOWN",  # TODO: infer from signal_data
            signal_type="MTF_RSI_DIVERGENCE",
            entry_time=datetime.utcnow(),
            entry_price=entry_price,
            feature_contract_version="v1.0",
            feature_names=list(signal_data.keys()),
            feature_values=signal_data,
        )
        db.add(outcome)
        db.commit()
        logger.info(f"Signal outcome logged for {symbol}")
    except Exception as e:
        logger.error(f"Failed to log signal outcome: {e}")
        db.rollback()


def run_cycle():
    """Execute one cycle: fetch data, generate signals, log outcomes."""
    if not SYMBOLS:
        logger.warning("No symbols configured")
        return
    
    client = AlpacaClient()
    db = SessionLocal()
    
    # Check if market is open
    if not client.is_market_open():
        logger.info("Market is closed, skipping cycle")
        db.close()
        return
    
    account = client.get_account()
    if not account:
        logger.error("Failed to connect to Alpaca API")
        db.close()
        return
    
    logger.info(f"Account connected. Portfolio value: ${account.get('portfolio_value', 0)}")
    
    # Process each symbol
    for symbol in SYMBOLS:
        try:
            logger.info(f"Processing {symbol}...")
            candles = fetch_market_data(client, symbol)
            if not candles:
                continue
            
            signals = generate_signals(candles)
            if not signals:
                logger.warning(f"No signals generated for {symbol}")
                continue
            
            # Get current price
            current_price = candles[-1]["close"] if candles else None
            
            # Log the signal outcome
            log_signal_outcome(db, symbol, signals, current_price)
            
            logger.info(f"✓ {symbol} signal generated: {signals.get('signals', {})}")
        
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    db.close()
    logger.info("Cycle complete")


if __name__ == "__main__":
    run_cycle()
