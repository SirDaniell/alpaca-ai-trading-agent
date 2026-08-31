"""
alpaca_execution_engine.py — Alpaca Live Trade Execution & Multi-Period PnL Tracking Engine.

Provides automated trade execution, live position monitoring, fast expiry tracking (e.g. 1m holding windows),
and structured multi-period performance metrics (Session, Day, Week, Month PnL).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, List
from datetime import datetime, timezone

from app.utils.alpaca_client import AlpacaClient
from app.utils.alpaca_cli_wrapper import AlpacaCLIWrapper

logger = logging.getLogger(__name__)


class AlpacaExecutionEngine:
    """
    Live Trade Execution Engine for Alpaca Trading API & CLI.
    Manages order placement, position tracking, automated expiry exits, and multi-period PnL reporting.
    """

    def __init__(self, use_cli: bool = True):
        self.cli_wrapper = AlpacaCLIWrapper(use_cli_if_available=use_cli)
        self.client = self.cli_wrapper.rest_client
        
        # Initialize Session Metrics
        account = self.client.get_account()
        self.session_start_equity = float(account.get("equity", 100000.0)) if account else 100000.0
        self.session_start_time = datetime.now(timezone.utc)
        self.trade_history: List[Dict[str, Any]] = []

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Fetch structured account performance metrics in exact priority order:
        1. Session PnL ($ and %)
        2. Day PnL ($ and %)
        3. Week PnL ($ and %)
        4. Month PnL ($ and %)
        """
        pnl_metrics = self.client.get_multi_period_pnl(session_start_equity=self.session_start_equity)
        
        total_trades = len(self.trade_history)
        wins = sum(1 for t in self.trade_history if t.get("realized_pnl", 0) > 0)
        losses = sum(1 for t in self.trade_history if t.get("realized_pnl", 0) < 0)
        win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0

        return {
            "session_start_time": self.session_start_time.isoformat(),
            "session_start_equity": self.session_start_equity,
            "current_equity": pnl_metrics["current_equity"],
            # Hierarchy: Session -> Day -> Week -> Month
            "session_pnl": pnl_metrics["session_pnl"],
            "session_pnl_pct": pnl_metrics["session_pnl_pct"],
            "day_pnl": pnl_metrics["day_pnl"],
            "day_pnl_pct": pnl_metrics["day_pnl_pct"],
            "week_pnl": pnl_metrics["week_pnl"],
            "week_pnl_pct": pnl_metrics["week_pnl_pct"],
            "month_pnl": pnl_metrics["month_pnl"],
            "month_pnl_pct": pnl_metrics["month_pnl_pct"],
            # Trade Statistics
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": win_rate,
        }

    def execute_trade_with_expiry(
        self,
        symbol: str,
        qty: float,
        side: str = "buy",
        asset_type: str = "crypto",
        expiry_seconds: int = 60,
        poll_interval: float = 2.0,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute a trade live, track its position and holding time until expiry (e.g. 60s for 1m test),
        and close the position upon completion, returning a full trade report.
        """
        symbol_clean = symbol.upper()
        side = side.lower()
        is_option = (asset_type == "option") or ("C00" in symbol_clean or "P00" in symbol_clean)

        logger.info(
            "🚀 [ExecutionEngine] Submitting %s order: %s %s (qty=%s, expiry=%ds, dry_run=%s)",
            asset_type.upper(), side.upper(), symbol_clean, qty, expiry_seconds, dry_run
        )

        t_start = time.time()
        start_time_iso = datetime.now(timezone.utc).isoformat()

        # Step 1: Submit Entry Order via CLI / REST
        order_res = self.cli_wrapper.submit_order(
            symbol=symbol_clean,
            side=side,
            qty=qty if not is_option else int(qty),
            order_type="market",
            is_option=is_option,
            dry_run=dry_run,
        )

        if dry_run or (isinstance(order_res, dict) and order_res.get("status") == "dry_run"):
            logger.info("🧪 Dry-run order simulated successfully.")
            return {
                "status": "completed",
                "mode": "dry_run",
                "symbol": symbol_clean,
                "side": side,
                "qty": qty,
                "entry_price": 100.0,
                "exit_price": 101.5,
                "realized_pnl": 1.5,
                "realized_pnl_pct": 1.5,
                "hold_duration_sec": expiry_seconds,
                "exit_reason": "EXPIRY_REACHED",
                "performance_metrics": self.get_performance_summary(),
            }

        # Step 2: Confirm Order Fill & Position Setup
        order_id = order_res.get("id") if isinstance(order_res, dict) else None
        entry_price = 0.0
        
        # Wait up to 10 seconds for order fill confirmation
        for _ in range(5):
            time.sleep(1.5)
            positions = self.client.get_positions()
            matched_pos = next((p for p in positions if p.get("symbol") == symbol_clean.replace("/", "")), None)
            if matched_pos:
                entry_price = float(matched_pos.get("avg_entry_price", 0.0))
                break
        
        if entry_price == 0.0 and order_id:
            order_details = self.client.get_order(order_id)
            if order_details and order_details.get("filled_avg_price"):
                entry_price = float(order_details["filled_avg_price"])

        logger.info("✅ Trade Filled: %s at entry price $%.4f", symbol_clean, entry_price)

        # Step 3: Monitor Position Until Expiry or SL/TP Trigger
        exit_reason = "EXPIRY_REACHED"
        exit_price = entry_price
        unrealized_pl = 0.0
        unrealized_plpc = 0.0

        while (time.time() - t_start) < expiry_seconds:
            time.sleep(poll_interval)
            elapsed = time.time() - t_start
            
            positions = self.client.get_positions()
            matched_pos = next((p for p in positions if p.get("symbol") == symbol_clean.replace("/", "")), None)
            
            if matched_pos:
                curr_price = float(matched_pos.get("current_price", entry_price))
                unrealized_pl = float(matched_pos.get("unrealized_pl", 0.0))
                unrealized_plpc = float(matched_pos.get("unrealized_plpc", 0.0)) * 100.0
                exit_price = curr_price

                logger.info(
                    "⏱️ [Position Monitor %s] Elapsed: %.1fs/ %ds | Price: $%.4f | Unr PnL: $%.2f (%.2f%%)",
                    symbol_clean, elapsed, expiry_seconds, curr_price, unrealized_pl, unrealized_plpc
                )

                # Check SL / TP
                if take_profit_pct and unrealized_plpc >= take_profit_pct:
                    exit_reason = "TAKE_PROFIT"
                    logger.info("🎯 Take-profit threshold reached (%.2f%% >= %.2f%%)", unrealized_plpc, take_profit_pct)
                    break
                if stop_loss_pct and unrealized_plpc <= stop_loss_pct:
                    exit_reason = "STOP_LOSS"
                    logger.info("🛑 Stop-loss threshold reached (%.2f%% <= %.2f%%)", unrealized_plpc, stop_loss_pct)
                    break
            else:
                logger.info("⚠️ Position no longer found in open positions; assuming closed by exchange.")
                break

        # Step 4: Execute Position Exit / Close Order
        logger.info("🏁 Closing position for %s (Reason: %s)...", symbol_clean, exit_reason)
        close_res = self.client.close_position(symbol_clean)
        
        # Wait up to 10 seconds for full position liquidation confirmation
        for _ in range(5):
            time.sleep(1.5)
            positions = self.client.get_positions()
            if not any(p.get("symbol") == symbol_clean.replace("/", "") for p in positions):
                break

        hold_duration = round(time.time() - t_start, 2)
        realized_pnl = unrealized_pl
        realized_pnl_pct = unrealized_plpc

        trade_record = {
            "status": "completed",
            "symbol": symbol_clean,
            "side": side,
            "qty": qty,
            "asset_type": asset_type,
            "start_time": start_time_iso,
            "end_time": datetime.now(timezone.utc).isoformat(),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "realized_pnl": realized_pnl,
            "realized_pnl_pct": realized_pnl_pct,
            "hold_duration_sec": hold_duration,
            "exit_reason": exit_reason,
            "close_order_response": close_res,
        }

        self.trade_history.append(trade_record)
        performance_summary = self.get_performance_summary()
        trade_record["performance_metrics"] = performance_summary

        logger.info(
            "✅ Trade Executed & Completed: %s | PnL: $%.2f (%.2f%%) | Hold Duration: %.1fs | Exit Reason: %s",
            symbol_clean, realized_pnl, realized_pnl_pct, hold_duration, exit_reason
        )

        return trade_record
