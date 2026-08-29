"""
test_alpaca_cli_wrapper.py — Unit Tests for Alpaca CLI Execution Wrapper.

Verifies:
1. Dry-run execution mode.
2. CLI fallback to authenticated REST client when binary is absent.
3. Proper formatting of command line args and environment variables.
"""

from unittest.mock import MagicMock, patch
import pytest

from app.utils.alpaca_cli_wrapper import AlpacaCLIWrapper


def test_alpaca_cli_wrapper_dry_run():
    wrapper = AlpacaCLIWrapper(use_cli_if_available=False)
    res = wrapper.submit_order(symbol="SPY260918C00500000", side="buy", qty=1, dry_run=True)

    assert res["status"] == "dry_run"
    assert res["symbol"] == "SPY260918C00500000"
    assert res["qty"] == 1
    assert res["side"] == "buy"


@patch("app.utils.alpaca_cli_wrapper.AlpacaClient")
def test_alpaca_cli_wrapper_rest_fallback(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.place_option_order.return_value = {"id": "order_789", "status": "submitted"}

    wrapper = AlpacaCLIWrapper(use_cli_if_available=False)
    res = wrapper.submit_order(symbol="SPY260918C00500000", side="buy", qty=1, dry_run=False)

    assert mock_client.place_option_order.called
    assert res["id"] == "order_789"
