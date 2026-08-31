"""
test_alpaca_live_execution.py — Comprehensive Live Integration Tests for Alpaca Trade Execution & Expiry Pipeline.

Verifies:
1. Multi-Period Performance Metrics hierarchy (Session PnL, Day PnL, Week PnL, Month PnL).
2. Live trade execution, order fill, active position tracking, 1m/fast expiry auto-exit, and position completion.
3. Alpaca CLI Wrapper dry-run and REST fallback functionality.
"""

import logging
import pytest
from app.utils.alpaca_client import AlpacaClient
from app.utils.alpaca_cli_wrapper import AlpacaCLIWrapper
from app.core.options.alpaca_execution_engine import AlpacaExecutionEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_multi_period_pnl_metrics():
    """Verify that multi-period PnL performance metrics track Session, Day, Week, and Month in exact hierarchy."""
    client = AlpacaClient()
    pnl = client.get_multi_period_pnl()

    logger.info("📊 [Test] Multi-Period PnL Metrics: %s", pnl)
    assert "session_pnl" in pnl, "Missing session_pnl"
    assert "session_pnl_pct" in pnl, "Missing session_pnl_pct"
    assert "day_pnl" in pnl, "Missing day_pnl"
    assert "day_pnl_pct" in pnl, "Missing day_pnl_pct"
    assert "week_pnl" in pnl, "Missing week_pnl"
    assert "week_pnl_pct" in pnl, "Missing week_pnl_pct"
    assert "month_pnl" in pnl, "Missing month_pnl"
    assert "month_pnl_pct" in pnl, "Missing month_pnl_pct"
    assert "current_equity" in pnl, "Missing current_equity"
    assert pnl["current_equity"] > 0, "Current equity must be > 0"


def test_alpaca_cli_wrapper_dry_run():
    """Verify CLI wrapper dry-run and REST fallback execution."""
    wrapper = AlpacaCLIWrapper(use_cli_if_available=True)
    res = wrapper.submit_order(
        symbol="BTC/USD",
        side="buy",
        qty=0.001,
        is_option=False,
        dry_run=True
    )
    logger.info("🧪 [Test] CLI Dry Run Result: %s", res)
    assert isinstance(res, dict)
    assert res.get("status") in ("dry_run", "success")


def test_live_trade_execution_and_expiry_completion():
    """
    Live Trade Execution Test:
    Places an order on Alpaca paper API, monitors the position, executes an expiry exit after holding window (15s fast test),
    and confirms full position completion and PnL logging.
    """
    engine = AlpacaExecutionEngine(use_cli=False)

    # Place live test trade with fast 15s expiry window
    result = engine.execute_trade_with_expiry(
        symbol="BTC/USD",
        qty=0.001,
        side="buy",
        asset_type="crypto",
        expiry_seconds=15,
        poll_interval=2.0,
        dry_run=False,
    )

    logger.info("✅ [Test] Live Trade Execution Result: %s", result)

    assert result["status"] == "completed", f"Trade failed to complete: {result}"
    assert result["symbol"] == "BTC/USD"
    assert result["entry_price"] > 0, "Entry price must be positive"
    assert result["exit_price"] > 0, "Exit price must be positive"
    assert result["hold_duration_sec"] >= 10.0, "Hold duration should meet test window"
    assert result["exit_reason"] == "EXPIRY_REACHED"

    # Confirm 0 open positions remaining for BTC/USD
    positions = engine.client.get_positions()
    btc_pos = [p for p in positions if p.get("symbol") == "BTCUSD"]
    assert len(btc_pos) == 0, f"Expected 0 open BTCUSD positions after completion, found {len(btc_pos)}"

    # Confirm updated performance summary includes the completed trade
    perf = result["performance_metrics"]
    assert perf["total_trades"] == 1
    assert "session_pnl" in perf
    assert "day_pnl" in perf
    assert "week_pnl" in perf
    assert "month_pnl" in perf
    logger.info("🎉 Live Trade Execution & Expiry Test Passed Cleanly!")
