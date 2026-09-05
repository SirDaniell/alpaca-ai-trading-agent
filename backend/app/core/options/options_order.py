"""
options_order.py — OCC Option Symbol Formatter and ATM Contract Selector.

Fetches real active contracts from Alpaca /v2/options/contracts and picks
the nearest tradable ATM strike for the given underlying and direction.
Falls back to a locally-constructed OCC symbol only if the API is unavailable.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")


def build_occ_symbol(
    underlying: str,
    expiration_date: datetime,
    option_type: str,
    strike_price: float,
) -> str:
    """Format underlying ticker, expiration, type, and strike into standard OCC symbol."""
    clean_root = underlying.upper()
    date_str = expiration_date.strftime("%y%m%d")
    type_char = "C" if option_type.upper().startswith("C") else "P"
    strike_int = int(round(strike_price * 1000))
    strike_str = f"{strike_int:08d}"
    return f"{clean_root}{date_str}{type_char}{strike_str}"


def select_target_option_contract(
    underlying: str,
    current_price: float,
    option_type: str,
    days_to_expiration: int = 7,
) -> Dict[str, Any]:
    """
    Select the nearest tradable ATM option contract for the given underlying
    by querying the Alpaca /v2/options/contracts endpoint.

    Falls back to a locally-constructed OCC symbol if the API call fails.
    """
    opt_type_norm = "call" if option_type.upper().startswith("C") else "put"

    # Try fetching real contracts from Alpaca
    try:
        session = requests.Session()
        session.headers.update({
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        })
        min_exp = (datetime.now(timezone.utc) + timedelta(days=2)).strftime("%Y-%m-%d")
        max_exp = (datetime.now(timezone.utc) + timedelta(days=days_to_expiration + 14)).strftime("%Y-%m-%d")

        resp = session.get(
            f"{ALPACA_BASE_URL}/v2/options/contracts",
            params={
                "underlying_symbols": underlying.upper(),
                "type": opt_type_norm,
                "status": "active",
                "expiration_date_gte": min_exp,
                "expiration_date_lte": max_exp,
                "limit": 100,
            },
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        contracts = data.get("option_contracts", [])

        # Filter: tradable only, valid strike price
        tradable = [
            c for c in contracts
            if c.get("tradable") is True and c.get("strike_price") and c.get("symbol")
        ]

        if tradable:
            # Pick nearest ATM strike to current price
            best = min(tradable, key=lambda c: abs(float(c["strike_price"]) - current_price))
            occ_symbol = best["symbol"]
            strike = float(best["strike_price"])
            exp_date = best.get("expiration_date", "unknown")
            close_price = best.get("close_price")

            logger.info(
                "📋 [ContractSelector] Real ATM contract for %s %s: %s (strike=$%.2f, exp=%s, last_close=%s)",
                underlying, opt_type_norm.upper(), occ_symbol, strike, exp_date, close_price
            )
            return {
                "underlying": underlying,
                "occ_symbol": occ_symbol,
                "option_type": opt_type_norm.upper(),
                "strike_price": strike,
                "expiration_date": exp_date,
                "close_price": close_price,
                "source": "alpaca_api",
            }

        logger.warning(
            "⚠ No tradable %s %s contracts found — falling back to OCC builder.",
            underlying, opt_type_norm
        )

    except Exception as e:
        logger.warning("⚠ Alpaca contract fetch failed for %s: %s — falling back to OCC builder.", underlying, e)

    # Fallback: construct OCC symbol locally
    exp_date_dt = datetime.now(timezone.utc) + timedelta(days=days_to_expiration)
    days_ahead = 4 - exp_date_dt.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    target_exp = exp_date_dt + timedelta(days=days_ahead)

    interval = 1.0 if current_price < 50 else (5.0 if current_price < 500 else 10.0)
    strike = round(current_price / interval) * interval
    occ_ticker = build_occ_symbol(underlying, target_exp, option_type, strike)

    logger.warning(
        "⚠ [ContractSelector] Fallback OCC symbol: %s (strike=$%.2f, exp=%s)",
        occ_ticker, strike, target_exp.strftime("%Y-%m-%d")
    )
    return {
        "underlying": underlying,
        "occ_symbol": occ_ticker,
        "option_type": option_type.upper(),
        "strike_price": strike,
        "expiration_date": target_exp.strftime("%Y-%m-%d"),
        "close_price": None,
        "source": "fallback_occ_builder",
    }
