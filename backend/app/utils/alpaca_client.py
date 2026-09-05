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

    def get_closed_orders(self, limit: int = 50) -> List[Dict]:
        """Fetch closed/filled orders directly from Alpaca API."""
        try:
            response = self.session.get(f"{self.base_url}/v2/orders", params={"status": "closed", "limit": limit})
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch closed orders: {e}")
            return []

    def get_portfolio_value(self) -> Optional[float]:
        """Get total portfolio value."""
        account = self.get_account()
        if account:
            return float(account.get("portfolio_value", 0))
        return None

    def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        time_in_force: Optional[str] = None,
    ) -> Optional[Dict]:
        """Place a market or limit order (supports equities and crypto)."""
        try:
            is_crypto = "/" in symbol or symbol.upper().startswith("BTC") or symbol.upper().startswith("ETH")
            tif = time_in_force or ("gtc" if is_crypto else "day")
            payload = {
                "symbol": symbol,
                "qty": str(qty) if isinstance(qty, float) else qty,
                "side": side.lower(),
                "type": order_type.lower(),
                "time_in_force": tif,
            }
            response = self.session.post(f"{self.base_url}/v2/orders", json=payload)
            response.raise_for_status()
            logger.info(f"Order placed: {symbol} {qty} {side} (tif={tif})")
            return response.json()
        except Exception as e:
            logger.error(f"Failed to place order for {symbol}: {e}")
            return None

    def get_order(self, order_id: str) -> Optional[Dict]:
        """Fetch order details by order ID."""
        try:
            response = self.session.get(f"{self.base_url}/v2/orders/{order_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch order {order_id}: {e}")
            return None

    def close_position(self, symbol: str) -> Optional[Dict]:
        """Close an open position by symbol (e.g. BTCUSD or SPY)."""
        clean_sym = symbol.replace("/", "")
        try:
            response = self.session.delete(f"{self.base_url}/v2/positions/{clean_sym}")
            response.raise_for_status()
            logger.info(f"Position closed for {symbol}")
            return response.json()
        except Exception as e:
            logger.error(f"Failed to close position for {symbol}: {e}")
            return None

    def get_portfolio_history(
        self,
        period: str = "1D",
        timeframe: str = "5Min",
        pnl_reset: str = "per_day",
    ) -> Optional[Dict]:
        """Fetch historical portfolio equity and PnL curve."""
        try:
            params = {
                "period": period,
                "timeframe": timeframe,
                "pnl_reset": pnl_reset,
            }
            response = self.session.get(f"{self.base_url}/v2/account/portfolio/history", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch portfolio history ({period}): {e}")
            return None

    def get_multi_period_pnl(self, session_start_equity: Optional[float] = None) -> Dict[str, float]:
        """
        Fetch multi-period PnL performance metrics in strict order:
        1. Session PnL ($ and %)
        2. Day PnL ($ and %)
        3. Week PnL ($ and %)
        4. Month PnL ($ and %)
        """
        account = self.get_account()
        if not account:
            return {
                "session_pnl": 0.0, "session_pnl_pct": 0.0,
                "day_pnl": 0.0, "day_pnl_pct": 0.0,
                "week_pnl": 0.0, "week_pnl_pct": 0.0,
                "month_pnl": 0.0, "month_pnl_pct": 0.0,
                "current_equity": 0.0,
            }

        curr_equity = float(account.get("equity", 0.0))
        last_equity = float(account.get("last_equity", curr_equity))

        # 1. Session PnL
        if session_start_equity and session_start_equity > 0:
            session_pnl = curr_equity - session_start_equity
            session_pnl_pct = (session_pnl / session_start_equity) * 100.0
        else:
            session_pnl = curr_equity - last_equity
            session_pnl_pct = (session_pnl / max(last_equity, 1e-6)) * 100.0

        # 2. Day PnL (1D)
        hist_1d = self.get_portfolio_history(period="1D", timeframe="5Min")
        if hist_1d and hist_1d.get("profit_loss"):
            valid_pnl = [p for p in hist_1d["profit_loss"] if p is not None]
            valid_pct = [p for p in hist_1d["profit_loss_pct"] if p is not None]
            day_pnl = float(valid_pnl[-1]) if valid_pnl else (curr_equity - last_equity)
            day_pnl_pct = (float(valid_pct[-1]) * 100.0) if valid_pct else (day_pnl / max(last_equity, 1e-6) * 100.0)
        else:
            day_pnl = curr_equity - last_equity
            day_pnl_pct = (day_pnl / max(last_equity, 1e-6)) * 100.0

        # 3. Week PnL (1W)
        hist_1w = self.get_portfolio_history(period="1W", timeframe="1D")
        if hist_1w and hist_1w.get("profit_loss"):
            valid_pnl = [p for p in hist_1w["profit_loss"] if p is not None]
            valid_pct = [p for p in hist_1w["profit_loss_pct"] if p is not None]
            week_pnl = float(valid_pnl[-1]) if valid_pnl else day_pnl
            week_pnl_pct = (float(valid_pct[-1]) * 100.0) if valid_pct else day_pnl_pct
        else:
            week_pnl = day_pnl
            week_pnl_pct = day_pnl_pct

        # 4. Month PnL (1M)
        hist_1m = self.get_portfolio_history(period="1M", timeframe="1D")
        if hist_1m and hist_1m.get("profit_loss"):
            valid_pnl = [p for p in hist_1m["profit_loss"] if p is not None]
            valid_pct = [p for p in hist_1m["profit_loss_pct"] if p is not None]
            month_pnl = float(valid_pnl[-1]) if valid_pnl else week_pnl
            month_pnl_pct = (float(valid_pct[-1]) * 100.0) if valid_pct else week_pnl_pct
        else:
            month_pnl = week_pnl
            month_pnl_pct = week_pnl_pct

        return {
            "session_pnl": float(session_pnl),
            "session_pnl_pct": float(session_pnl_pct),
            "day_pnl": float(day_pnl),
            "day_pnl_pct": float(day_pnl_pct),
            "week_pnl": float(week_pnl),
            "week_pnl_pct": float(week_pnl_pct),
            "month_pnl": float(month_pnl),
            "month_pnl_pct": float(month_pnl_pct),
            "current_equity": float(curr_equity),
        }

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
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
    ) -> Optional[Dict]:
        """
        Place an option order via Alpaca /v2/orders.

        Per Alpaca docs (https://docs.alpaca.markets/us/docs/options-orders):
        - symbol: OCC option symbol e.g. 'GLD260911C00200000'
        - qty: whole number string
        - side: 'buy' or 'sell'
        - type: 'market' or 'limit'
        - time_in_force: 'day' or 'gtc'
        - NO asset_class field (it is NOT a valid param and causes rejection)
        """
        try:
            payload: Dict = {
                "symbol": symbol,
                "qty": str(int(qty)),
                "side": side.lower(),
                "type": type.lower(),
                "time_in_force": time_in_force,
            }
            if type.lower() == "limit" and limit_price is not None:
                payload["limit_price"] = str(limit_price)
            response = self.session.post(f"{self.base_url}/v2/orders", json=payload)
            if not response.ok:
                logger.error(f"Option order rejected [{response.status_code}]: {response.text}")
            response.raise_for_status()
            order = response.json()
            logger.info(f"✅ Option order placed: {symbol} qty={qty} side={side} | id={order.get('id')}")
            return order
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


    def get_orders(self, status: str = "all", limit: int = 50) -> List[Dict]:
        """Fetch orders by status ('open', 'closed', 'all')."""
        try:
            response = self.session.get(f"{self.base_url}/v2/orders", params={"status": status, "limit": limit})
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch orders (status={status}): {e}")
            return []

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

