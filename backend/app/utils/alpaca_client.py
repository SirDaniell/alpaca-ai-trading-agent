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
            "APCA-API-KEY-ID": self.api_key or "",
            "APCA-API-SECRET-KEY": self.secret_key or "",
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

    def get_bars(
        self,
        symbol: str,
        timeframe: str = "5Min",
        limit: int = 500,
        start: Optional[str] = None,
    ) -> List[Dict]:
        """Get historical bars for a stock/ETF or Crypto symbol using Alpaca Data API.

        Paginates automatically until `limit` bars are collected or the API
        has no more data.  For stocks the Alpaca v2 data endpoint requires a
        `start` date to return intraday bars; if one is not supplied we default
        to 90 days ago so the call always returns real data.

        Args:
            symbol:    Ticker string.  Crypto uses the slash form, e.g. "BTC/USD".
            timeframe: Alpaca timeframe string or short alias (5m, 15m, 1h, 4h, 1d).
            limit:     Maximum number of bars to return across all pages.
            start:     ISO-8601 UTC start datetime string, e.g. "2026-01-01T00:00:00Z".
                       Defaults to 90 days ago for stocks, 7 days ago for crypto.
        """
        import datetime as _dt
        data_base_url = "https://data.alpaca.markets"
        tf_map = {"5m": "5Min", "15m": "15Min", "1h": "1Hour", "4h": "4Hour", "1d": "1Day"}
        norm_tf = tf_map.get(timeframe.lower(), timeframe)

        is_crypto = "/" in symbol or symbol.upper().startswith("BTC")

        # Determine default look-back window (scaled proportionally with limit)
        if start is None:
            if is_crypto:
                default_days = max(7, int(limit / 288) + 1)
            else:
                # 5m bars: ~78 bars per day for equities. Scale start window up to 2+ years for 30k limit
                default_days = max(90, int(limit / 50) + 1)
            start = (
                _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=default_days)
            ).strftime("%Y-%m-%dT00:00:00Z")


        all_bars: List[Dict] = []
        page_size = min(limit, 1000)      # Alpaca max per page

        try:
            if is_crypto:
                url = f"{data_base_url}/v1beta3/crypto/us/bars"
                next_token: Optional[str] = None
                while len(all_bars) < limit:
                    params: Dict = {
                        "symbols": symbol,
                        "timeframe": norm_tf,
                        "limit": min(page_size, limit - len(all_bars)),
                        "start": start,
                    }
                    if next_token:
                        params["page_token"] = next_token

                    resp = self.session.get(url, params=params)
                    resp.raise_for_status()
                    data = resp.json()
                    page_bars = (data.get("bars") or {}).get(symbol, []) or []
                    all_bars.extend(page_bars)

                    next_token = data.get("next_page_token")
                    if not next_token or not page_bars:
                        break

            else:
                # Equity / ETF
                url = f"{data_base_url}/v2/stocks/{symbol}/bars"
                next_token = None
                while len(all_bars) < limit:
                    params = {
                        "timeframe": norm_tf,
                        "limit": min(page_size, limit - len(all_bars)),
                        "feed": "iex",
                        "start": start,
                    }
                    if next_token:
                        params["page_token"] = next_token

                    resp = self.session.get(url, params=params)
                    resp.raise_for_status()
                    data = resp.json()
                    page_bars = data.get("bars") or []
                    all_bars.extend(page_bars)

                    next_token = data.get("next_page_token")
                    if not next_token or not page_bars:
                        break

        except Exception as e:
            logger.error(f"Failed to fetch bars for {symbol}: {e}")

        logger.info(
            "[AlpacaClient] Fetched %d bars for %s (%s, start=%s)",
            len(all_bars), symbol, norm_tf, start,
        )
        return all_bars[:limit]


    def place_option_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        type: str = "market",
        time_in_force: str = "day"
    ) -> Optional[Dict]:
        """Place an option order on Alpaca API (using asset_class='option')."""
        try:
            payload = {
                "symbol": symbol,
                "qty": str(qty),
                "side": side,
                "type": type,
                "time_in_force": time_in_force,
                "asset_class": "option",
            }
            response = self.session.post(f"{self.base_url}/v2/orders", json=payload)
            response.raise_for_status()
            logger.info(f"Option order placed: {symbol} {qty} {side}")
            return response.json()
        except Exception as e:
            logger.error(f"Failed to place option order for {symbol}: {e}")
            return None

    def get_option_contracts(self, underlying_symbol: str, expiration_date_gte: Optional[str] = None) -> List[Dict]:
        """Fetch option contracts for an underlying ticker."""
        try:
            params = {
                "underlying_symbols": underlying_symbol,
                "status": "active",
            }
            if expiration_date_gte:
                params["expiration_date_gte"] = expiration_date_gte

            response = self.session.get(f"{self.base_url}/v2/options/contracts", params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("option_contracts", []) if isinstance(data, dict) else data
        except Exception as e:
            logger.error(f"Failed to fetch option contracts for {underlying_symbol}: {e}")
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

