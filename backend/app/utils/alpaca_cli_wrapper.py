"""
alpaca_cli_wrapper.py — Alpaca CLI Execution Wrapper & Fallback Service.

Fulfills competition requirement for Alpaca Trading CLI integration.
Executes options/equity orders via official `alpaca` CLI command line runner
with `--dry-run` test support, falling back to authenticated REST client when binary is absent.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from app.utils.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)


class AlpacaCLIWrapper:
    """
    Wrapper for Alpaca Trading CLI and REST API fallback.
    """

    def __init__(self, use_cli_if_available: bool = True):
        self.rest_client = AlpacaClient()
        self.cli_binary = shutil.which("alpaca")
        self.use_cli = use_cli_if_available and (self.cli_binary is not None)

        if self.use_cli:
            logger.info("[AlpacaCLI] Using native Alpaca CLI binary at: %s", self.cli_binary)
        else:
            logger.info("[AlpacaCLI] Alpaca CLI binary not found in PATH; using authenticated REST fallback.")

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: int = 1,
        order_type: str = "market",
        is_option: bool = True,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Submit an equity or option order via Alpaca CLI (or REST fallback).
        Support --dry-run for safety testing.
        """
        side = side.lower()
        if self.use_cli and self.cli_binary:
            cmd = [
                self.cli_binary,
                "order",
                "submit",
                "--symbol", symbol,
                "--side", side,
                "--qty", str(qty),
                "--type", order_type,
            ]
            if dry_run:
                cmd.append("--dry-run")

            env = os.environ.copy()
            env["ALPACA_API_KEY"] = self.rest_client.api_key or ""
            env["ALPACA_SECRET_KEY"] = self.rest_client.secret_key or ""

            try:
                logger.info("[AlpacaCLI] Invoking CLI: %s", " ".join(cmd))
                res = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
                output = res.stdout.strip()
                try:
                    return json.loads(output)
                except json.JSONDecodeError:
                    return {"status": "success", "raw_output": output, "command": " ".join(cmd)}
            except subprocess.CalledProcessError as e:
                logger.error("[AlpacaCLI] CLI order submission failed: %s", e.stderr)
                return {"status": "error", "error": e.stderr, "command": " ".join(cmd)}

        # Fallback to direct REST API client
        logger.info("[AlpacaCLI] Executing order via Alpaca REST Client (dry_run=%s)", dry_run)
        if dry_run:
            return {
                "status": "dry_run",
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "asset_class": "option" if is_option else "us_equity",
                "message": "Dry-run execution success (no API order placed).",
            }

        if is_option or "C00" in symbol or "P00" in symbol:
            res = self.rest_client.place_option_order(symbol=symbol, qty=qty, side=side, type=order_type)
        else:
            res = self.rest_client.place_order(symbol=symbol, qty=qty, side=side, order_type=order_type)

        return res or {"status": "error", "message": "REST call returned None"}

    def get_account(self) -> Optional[Dict[str, Any]]:
        """Fetch account state."""
        return self.rest_client.get_account()

    def get_positions(self) -> List[Dict[str, Any]]:
        """Fetch active positions."""
        return self.rest_client.get_positions()
