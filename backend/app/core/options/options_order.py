"""
options_order.py — OCC Option Symbol Formatter and Contract Selector.

Formats standard OCC Option tickers according to OCC specs:
Format: {Root 6 chars}{YYMMDD}{C/P}{Strike Price * 1000 formatted as 8 digits}
Example: SPY260918C00500000 (SPY $500 Call expiring 2026-09-18)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional


def build_occ_symbol(
    underlying: str,
    expiration_date: datetime,
    option_type: str,  # "call" or "put" (or "C"/"P")
    strike_price: float,
) -> str:
    """
    Format underlying ticker, expiration, type, and strike into standard OCC symbol.
    """
    root = underlying.upper().ljust(6)  # Pad root to 6 characters if needed or trim
    if len(root) > 6:
        root = root[:6]

    date_str = expiration_date.strftime("%y%m%d")
    type_char = "C" if option_type.upper().startswith("C") else "P"
    strike_int = int(round(strike_price * 1000))
    strike_str = f"{strike_int:08d}"

    # Standard OCC ticker format: SPY260918C00500000
    # Note: Alpaca API strips trailing spaces from root
    clean_root = underlying.upper()
    return f"{clean_root}{date_str}{type_char}{strike_str}"


def select_target_option_contract(
    underlying: str,
    current_price: float,
    option_type: str,
    days_to_expiration: int = 7,
) -> Dict[str, Any]:
    """
    Select target option contract parameters (strike, expiration, OCC symbol) for trading.
    Finds nearest at-the-money (ATM) strike price rounded to standard option intervals.
    """
    exp_date = datetime.now(timezone.utc) + timedelta(days=days_to_expiration)
    # Standard Friday expiration adjustment if needed
    days_ahead = 4 - exp_date.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    target_exp = exp_date + timedelta(days=days_ahead)

    # Strike selection (ATM rounded to 1.0 or 5.0 interval depending on price level)
    interval = 1.0 if current_price < 200 else 5.0
    strike = round(current_price / interval) * interval

    occ_ticker = build_occ_symbol(underlying, target_exp, option_type, strike)

    return {
        "underlying": underlying,
        "occ_symbol": occ_ticker,
        "option_type": option_type.upper(),
        "strike_price": strike,
        "expiration_date": target_exp.strftime("%Y-%m-%d"),
        "target_days_to_exp": days_to_expiration,
    }
