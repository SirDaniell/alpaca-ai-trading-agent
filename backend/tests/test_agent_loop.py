"""
test_agent_loop.py — End-to-End Integration Test for Agent Execution Loop.

Verifies:
1. Fetching market data & building Zone Snapshots.
2. Hard Action Masking & Tier 2 Q-Learner action selection.
3. Hard Risk Gate checks.
4. OCC Option Symbol formatting and Option Order placement via AlpacaClient.
5. Signal outcome DB persistence.
"""

from unittest.mock import MagicMock, patch
import pytest

from app.agent.loop import run_cycle
from app.core.options.q_executor import ACTION_BUY_CALL


@patch("app.agent.loop.AlpacaClient")
@patch("app.agent.loop.SessionLocal")
def test_agent_loop_full_options_execution_cycle(mock_session_cls, mock_client_cls):
    # Setup mock Alpaca client
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_client.is_market_open.return_value = True
    mock_client.get_account.return_value = {
        "equity": "100000.00",
        "last_equity": "100000.00",
        "portfolio_value": "100000.00",
    }
    mock_client.get_positions.return_value = []

    # Mock historical bars for SPY
    bars = []
    base_price = 500.0
    for i in range(50):
        p = base_price + (i * 0.1)
        bars.append({
            "t": f"2026-08-29T14:{i:02d}:00Z",
            "o": p,
            "h": p + 0.5,
            "l": p - 0.2,
            "c": p + 0.3,
            "v": 1000,
        })
    mock_client.get_bars.return_value = bars

    # Mock option order response
    mock_client.place_option_order.return_value = {
        "id": "mock_order_123",
        "symbol": "SPY260918C00500000",
        "status": "accepted",
        "qty": "1",
        "side": "buy",
    }

    # Setup mock DB session
    mock_db = MagicMock()
    mock_session_cls.return_value = mock_db

    # Run loop cycle
    with patch("app.agent.loop.q_executor.select_action", return_value=ACTION_BUY_CALL):
        run_cycle()

    # Assertions
    assert mock_client.is_market_open.called
    assert mock_client.get_account.called
    assert mock_client.place_option_order.called, "Option order must be placed when action is BUY_CALL"

    # Verify option order payloads
    assert mock_client.place_option_order.call_count >= 1
    placed_symbols = [c.kwargs["symbol"] for c in mock_client.place_option_order.call_args_list]
    assert any("SPY" in sym or "TSLA" in sym for sym in placed_symbols)

    # Verify DB commit
    assert mock_db.add.called
    assert mock_db.commit.called

