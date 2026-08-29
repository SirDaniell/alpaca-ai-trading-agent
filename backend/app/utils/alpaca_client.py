import os
import logging
from typing import Optional, Dict, List
from dotenv import load_dotenv
import requests

load_dotenv()

logger = logging.getLogger(__name__)

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")


class AlpacaClient:
    """Wrapper around Alpaca Trading API for options and equity trading."""

    def __init__(self):
        self.api_key = ALPACA_API_KEY
        self.secret_key = ALPACA_SECRET_KEY
        self.base_url = ALPACA_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            "APCA-API-KEY-ID": self.api_key,
        })

    def get_account(self) -> Optional[Dict]:
        """Get current account info."""
        try:
            response = self.session.get(f"{self.base_url}/v2/account")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch account: {e}")
            return None

    def get_positions(self) -> List[Dict]:
        """Get current open positions."""
        try:
            response = self.session.get(f"{self.base_url}/v2/positions")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return []

    def get_portfolio_value(self) -> Optional[float]:
        """Get total portfolio value."""
        account = self.get_account()
        if account:
            return float(account.get("portfolio_value", 0))
        return None

    def place_order(self, symbol: str, qty: int, side: str, order_type: str = "market") -> Optional[Dict]:
        """Place a market order."""
        try:
            payload = {
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "type": order_type,
                "time_in_force": "day",
            }
            response = self.session.post(f"{self.base_url}/v2/orders", json=payload)
            response.raise_for_status()
            logger.info(f"Order placed: {symbol} {qty} {side}")
            return response.json()
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None

    def get_bars(self, symbol: str, timeframe: str = "1min", limit: int = 500) -> List[Dict]:
        """Get historical bars for a symbol."""
        try:
            params = {
                "timeframe": timeframe,
                "limit": limit,
            }
            response = self.session.get(
                f"{self.base_url}/v2/stocks/{symbol}/bars",
                params=params
            )
            response.raise_for_status()
            data = response.json()
            return data.get("bars", []) if isinstance(data, dict) else data
        except Exception as e:
            logger.error(f"Failed to fetch bars for {symbol}: {e}")
            return []

    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        try:
            response = self.session.get(f"{self.base_url}/v2/clock")
            response.raise_for_status()
            clock = response.json()
            return clock.get("is_open", False)
        except Exception as e:
            logger.error(f"Failed to check market status: {e}")
            return False
