"""
alpaca_execution_engine.py — Alpaca Live Trade Execution & Non-Blocking Position Monitor.

Position monitoring runs in a daemon background thread so the main agent loop
continues scanning and evaluating new signals every 5 minutes uninterrupted.

Martingale result recording happens when the position thread exits (win/loss on close).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime, timezone

from app.utils.alpaca_client import AlpacaClient
from app.utils.alpaca_cli_wrapper import AlpacaCLIWrapper

logger = logging.getLogger(__name__)


class AlpacaExecutionEngine:
    """
    Live Trade Execution Engine for Alpaca Trading API.
    Entry order is placed synchronously; position monitoring runs in a background
    daemon thread so the main agent loop is NEVER blocked.
    """

    def __init__(self, use_cli: bool = True):
        self.cli_wrapper = AlpacaCLIWrapper(use_cli_if_available=use_cli)
        self.client = self.cli_wrapper.rest_client

        account = self.client.get_account()
        self.session_start_equity = float(account.get("equity", 100000.0)) if account else 100000.0
        self.session_start_time = datetime.now(timezone.utc)
        self.trade_history: List[Dict[str, Any]] = []
        self._monitor_threads: List[threading.Thread] = []

    # ── Performance Summary ──────────────────────────────────────────────────

    def get_performance_summary(self) -> Dict[str, Any]:
        pnl_metrics = self.client.get_multi_period_pnl(session_start_equity=self.session_start_equity)
        total = len(self.trade_history)
        wins = sum(1 for t in self.trade_history if t.get("realized_pnl", 0) > 0)
        losses = total - wins
        return {
            "session_start_time": self.session_start_time.isoformat(),
            "session_start_equity": self.session_start_equity,
            "current_equity": pnl_metrics["current_equity"],
            "session_pnl": pnl_metrics["session_pnl"],
            "session_pnl_pct": pnl_metrics["session_pnl_pct"],
            "day_pnl": pnl_metrics["day_pnl"],
            "day_pnl_pct": pnl_metrics["day_pnl_pct"],
            "week_pnl": pnl_metrics["week_pnl"],
            "week_pnl_pct": pnl_metrics["week_pnl_pct"],
            "month_pnl": pnl_metrics["month_pnl"],
            "month_pnl_pct": pnl_metrics["month_pnl_pct"],
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": (wins / max(total, 1)) * 100.0,
        }

    # ── Background Position Monitor ──────────────────────────────────────────

    def _monitor_and_close(
        self,
        symbol: str,
        entry_price: float,
        expiry_seconds: int,
        stop_loss_pct: Optional[float],
        take_profit_pct: Optional[float],
        poll_interval: float,
        on_close_callback: Optional[Callable[[Dict[str, Any]], None]],
    ) -> None:
        """
        Runs in a daemon background thread.
        Polls position every poll_interval seconds, exits on SL/TP/expiry,
        then calls on_close_callback with the trade result.
        """
        t_start = time.time()
        exit_reason = "EXPIRY_REACHED"
        exit_price = entry_price
        unrealized_pl = 0.0
        unrealized_plpc = 0.0

        logger.info("🧵 [Monitor:%s] Background thread started (expiry=%ds, SL=%s, TP=%s)",
                    symbol, expiry_seconds, stop_loss_pct, take_profit_pct)

        while (time.time() - t_start) < expiry_seconds:
            time.sleep(poll_interval)
            elapsed = time.time() - t_start

            try:
                positions = self.client.get_positions()
                matched = next((p for p in positions if p.get("symbol") == symbol), None)

                if not matched:
                    logger.info("🧵 [Monitor:%s] Position closed by exchange at %.1fs", symbol, elapsed)
                    exit_reason = "CLOSED_BY_EXCHANGE"
                    break

                curr_price = float(matched.get("current_price", entry_price))
                unrealized_pl = float(matched.get("unrealized_pl", 0.0))
                unrealized_plpc = float(matched.get("unrealized_plpc", 0.0)) * 100.0
                exit_price = curr_price

                # Reduce log noise: only log every 30s
                if int(elapsed) % 30 < poll_interval:
                    logger.info("⏱️ [Monitor:%s] %.0fs/%.0fs | $%.4f | PnL $%.2f (%.2f%%)",
                                symbol, elapsed, expiry_seconds, curr_price, unrealized_pl, unrealized_plpc)

                if take_profit_pct and unrealized_plpc >= take_profit_pct:
                    exit_reason = "TAKE_PROFIT"
                    logger.info("🎯 [Monitor:%s] TP hit %.2f%% >= %.2f%%", symbol, unrealized_plpc, take_profit_pct)
                    break

                if stop_loss_pct and unrealized_plpc <= stop_loss_pct:
                    exit_reason = "STOP_LOSS"
                    logger.info("🛑 [Monitor:%s] SL hit %.2f%% <= %.2f%%", symbol, unrealized_plpc, stop_loss_pct)
                    break

            except Exception as poll_err:
                logger.warning("🧵 [Monitor:%s] Poll error: %s", symbol, poll_err)

        # Close position
        logger.info("🏁 [Monitor:%s] Closing position (reason: %s)...", symbol, exit_reason)
        try:
            self.client.close_position(symbol)
            # Brief wait for fill
            for _ in range(5):
                time.sleep(1.5)
                positions = self.client.get_positions()
                if not any(p.get("symbol") == symbol for p in positions):
                    break
        except Exception as close_err:
            logger.error("🧵 [Monitor:%s] Close error: %s", symbol, close_err)

        hold_duration = round(time.time() - t_start, 2)
        trade_record = {
            "status": "completed",
            "symbol": symbol,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "realized_pnl": unrealized_pl,
            "realized_pnl_pct": unrealized_plpc,
            "hold_duration_sec": hold_duration,
            "exit_reason": exit_reason,
            "end_time": datetime.now(timezone.utc).isoformat(),
        }
        self.trade_history.append(trade_record)

        logger.info(
            "✅ [Monitor:%s] Trade closed | PnL: $%.2f (%.2f%%) | Hold: %.0fs | Reason: %s",
            symbol, unrealized_pl, unrealized_plpc, hold_duration, exit_reason
        )

        if on_close_callback:
            try:
                on_close_callback(trade_record)
            except Exception as cb_err:
                logger.warning("🧵 [Monitor:%s] Callback error: %s", symbol, cb_err)

    # ── Main Entry Point ─────────────────────────────────────────────────────

    def execute_trade_with_expiry(
        self,
        symbol: str,
        qty: float,
        side: str = "buy",
        asset_type: str = "crypto",
        expiry_seconds: int = 60,
        poll_interval: float = 10.0,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        dry_run: bool = False,
        on_close_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Submit entry order SYNCHRONOUSLY, then launch a non-blocking background
        thread to monitor the position and close it on SL/TP/expiry.

        Returns immediately after entry fill — does NOT block the agent loop.
        """
        symbol_clean = symbol.upper()
        side = side.lower()
        is_option = (asset_type == "option") or ("C00" in symbol_clean or "P00" in symbol_clean)

        logger.info(
            "🚀 [ExecutionEngine] Submitting %s order: %s %s (qty=%s, expiry=%ds, dry_run=%s)",
            asset_type.upper(), side.upper(), symbol_clean, qty, expiry_seconds, dry_run,
        )

        start_time_iso = datetime.now(timezone.utc).isoformat()

        # ── Step 1: Submit Entry Order ────────────────────────────────────────
        order_res = self.cli_wrapper.submit_order(
            symbol=symbol_clean,
            side=side,
            qty=qty if not is_option else int(qty),
            order_type="market",
            is_option=is_option,
            dry_run=dry_run,
        )

        if dry_run or (isinstance(order_res, dict) and order_res.get("status") == "dry_run"):
            logger.info("🧪 Dry-run order simulated.")
            return {
                "status": "submitted",
                "mode": "dry_run",
                "symbol": symbol_clean,
                "side": side,
                "qty": qty,
                "entry_price": 100.0,
                "exit_price": None,
                "hold_duration_sec": expiry_seconds,
                "exit_reason": "DRY_RUN",
            }

        # ── Step 2: Confirm Fill ──────────────────────────────────────────────
        order_id = order_res.get("id") if isinstance(order_res, dict) else None
        entry_price = 0.0

        for _ in range(5):
            time.sleep(1.5)
            positions = self.client.get_positions()
            matched = next((p for p in positions if p.get("symbol") == symbol_clean), None)
            if matched:
                entry_price = float(matched.get("avg_entry_price", 0.0))
                break

        if entry_price == 0.0 and order_id:
            order_details = self.client.get_order(order_id)
            if order_details and order_details.get("filled_avg_price"):
                entry_price = float(order_details["filled_avg_price"])

        logger.info("✅ Trade Filled: %s at entry $%.4f", symbol_clean, entry_price)

        # ── Step 3: Launch NON-BLOCKING background monitor thread ─────────────
        monitor_thread = threading.Thread(
            target=self._monitor_and_close,
            args=(symbol_clean, entry_price, expiry_seconds, stop_loss_pct, take_profit_pct, poll_interval, on_close_callback),
            daemon=True,  # Dies cleanly if main process exits
            name=f"monitor-{symbol_clean}",
        )
        monitor_thread.start()
        self._monitor_threads.append(monitor_thread)

        logger.info(
            "🧵 Position monitor launched in background thread for %s (expiry=%ds) — agent loop continues freely.",
            symbol_clean, expiry_seconds,
        )

        return {
            "status": "submitted",
            "mode": "live",
            "symbol": symbol_clean,
            "side": side,
            "qty": qty,
            "entry_price": entry_price,
            "exit_price": None,  # Will be set when monitor thread completes
            "expiry_seconds": expiry_seconds,
            "start_time": start_time_iso,
            "monitor_thread": monitor_thread.name,
        }
