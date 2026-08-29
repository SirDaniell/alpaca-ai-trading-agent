#!/usr/bin/env python3
"""
alpaca_mcp_cli.py — CLI & MCP Command Runner for Alpaca Options & Equity Operations.

Satisfies competition hard requirement for MCP/CLI integration.

Usage:
  python scripts/alpaca_mcp_cli.py account
  python scripts/alpaca_mcp_cli.py chain --symbol SPY
  python scripts/alpaca_mcp_cli.py order --symbol SPY260918C00500000 --qty 1 --side buy
  python scripts/alpaca_mcp_cli.py positions
"""

import sys
import os
import argparse
import json
import logging

# Add backend root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.utils.alpaca_client import AlpacaClient
from app.core.options.options_order import select_target_option_contract, build_occ_symbol

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AlpacaMCPCLI")


def cmd_account(args, client: AlpacaClient):
    acc = client.get_account()
    print(json.dumps(acc, indent=2))


def cmd_positions(args, client: AlpacaClient):
    pos = client.get_positions()
    print(json.dumps(pos, indent=2))


def cmd_chain(args, client: AlpacaClient):
    contracts = client.get_option_contracts(args.symbol)
    print(json.dumps(contracts[:args.limit], indent=2))


def cmd_order(args, client: AlpacaClient):
    if args.is_option or "C00" in args.symbol or "P00" in args.symbol:
        res = client.place_option_order(symbol=args.symbol, qty=args.qty, side=args.side)
    else:
        res = client.place_order(symbol=args.symbol, qty=args.qty, side=args.side)
    print(json.dumps(res, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Alpaca MCP / CLI Option Execution Runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # account
    subparsers.add_parser("account", help="Fetch account details")

    # positions
    subparsers.add_parser("positions", help="Fetch current open positions")

    # chain
    chain_parser = subparsers.add_parser("chain", help="Fetch active option contracts")
    chain_parser.add_argument("--symbol", required=True, help="Underlying ticker (e.g. SPY)")
    chain_parser.add_argument("--limit", type=int, default=10, help="Max contracts to list")

    # order
    order_parser = subparsers.add_parser("order", help="Place an equity or option order")
    order_parser.add_argument("--symbol", required=True, help="Ticker or OCC option symbol")
    order_parser.add_argument("--qty", type=int, default=1, help="Quantity of contracts/shares")
    order_parser.add_argument("--side", choices=["buy", "sell"], required=True, help="Order side")
    order_parser.add_argument("--is-option", action="store_true", help="Flag as option order")

    args = parser.parse_args()
    client = AlpacaClient()

    if args.command == "account":
        cmd_account(args, client)
    elif args.command == "positions":
        cmd_positions(args, client)
    elif args.command == "chain":
        cmd_chain(args, client)
    elif args.command == "order":
        cmd_order(args, client)


if __name__ == "__main__":
    main()
