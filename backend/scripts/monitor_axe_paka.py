#!/usr/bin/env python3
"""
monitor_axe_paka.py — Production monitoring runner for Axe-paka-v1 agent.

Runs the agent loop on a 5-minute cycle (aligned to candle close),
printing a clear live dashboard after each cycle.
Options-only execution — compliant with Alpaca AI Trading Agents Hackathon rules.
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timezone

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.utils.alpaca_client import AlpacaClient
from app.agent.loop import run_cycle

# Ensure root logger has FileHandler attached even if basicConfig was previously initialized
log_file_path = backend_dir / "data" / "axe_paka_monitor.log"
log_file_path.parent.mkdir(parents=True, exist_ok=True)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_file_path, mode="a")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
root_logger.addHandler(file_handler)

logger = logging.getLogger("AxePakaMonitor")

CYCLE_INTERVAL_SECONDS = 300  # 5-minute candle cadence


def print_dashboard(client: AlpacaClient, cycle_num: int):
    """Print a concise live dashboard after each cycle to stdout and log file."""
    account = client.get_account()
    positions = client.get_positions()
    orders = client.get_orders(status="all", limit=20)

    equity = float(account.get("equity", 0)) if account else 0
    last_equity = float(account.get("last_equity", equity)) if account else equity
    day_pnl = equity - last_equity
    options_level = account.get("options_trading_level", "?") if account else "?"

    option_orders = [o for o in orders if o.get("asset_class") == "us_option"]
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_option_orders = [o for o in option_orders if o.get("submitted_at", "")[:10] == today]

    dash_lines = [
        "=" * 70,
        f"  AXE-PAKA-V1 LIVE MONITOR  |  Cycle #{cycle_num}  |  {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}",
        "=" * 70,
        f"  💰 Equity:          ${equity:,.2f}",
        f"  📈 Day PnL:         ${day_pnl:+,.2f}",
        f"  🎯 Options Level:   {options_level}",
        f"  📋 Open Positions:  {len(positions)}",
    ]
    for p in positions:
        sym = p.get("symbol", "?")
        qty = p.get("qty", 0)
        unr = float(p.get("unrealized_pl", 0))
        cls = p.get("asset_class", "?")
        dash_lines.append(f"    → {sym:30s}  qty={qty}  unrealized_pl=${unr:+.2f}  [{cls}]")

    dash_lines.append(f"  📦 Today's Option Orders: {len(today_option_orders)}")
    for o in today_option_orders[-5:]:
        sym = o.get("symbol", "?")
        side = o.get("side", "?")
        status = o.get("status", "?")
        ts = o.get("submitted_at", "")[:19]
        dash_lines.append(f"    → [{ts}] {sym:30s}  {side:4s}  status={status}")
    dash_lines.append("=" * 70)

    dash_text = "\n".join(dash_lines)
    logger.info("\n" + dash_text)


def main():
    client = AlpacaClient()
    cycle_num = 0

    logger.info("=" * 60)
    logger.info("  AXE-PAKA-V1 AGENT — LIVE PAPER MONITORING STARTED")
    logger.info("  Options-only | GLD primary | 4-step Martingale MM ($10 base)")
    logger.info("  Cycle interval: %ds | Hard Martingale cap: $80 max, reset at 4 losses", CYCLE_INTERVAL_SECONDS)
    logger.info("=" * 60)

    while True:
        cycle_num += 1
        t_start = time.time()

        try:
            logger.info("\n🔄 Starting cycle #%d...", cycle_num)
            run_cycle()
        except Exception as e:
            logger.error("❌ Cycle #%d error: %s", cycle_num, e)

        print_dashboard(client, cycle_num)

        elapsed = time.time() - t_start
        sleep_time = max(5, CYCLE_INTERVAL_SECONDS - elapsed)
        logger.info("⏳ Next cycle in %.0fs...", sleep_time)
        time.sleep(sleep_time)


if __name__ == "__main__":
    main()
