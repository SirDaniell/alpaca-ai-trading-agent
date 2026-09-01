#!/usr/bin/env python3
"""Wrapper to run evaluate_option_expiries.py with proper sys.path."""

import sys
import os

# Add backend app to path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

# Parse command line for limit
limit = 5000
framework = "keras"
for arg in sys.argv[1:]:
    if arg.startswith("--limit"):
        limit = int(arg.split("=")[1])
    elif arg.startswith("--framework"):
        framework = arg.split("=")[1]

print(f"[Wrapper] Running evaluate_option_expiries with limit={limit}, framework={framework}")
print(f"[Wrapper] sys.path[0] = {backend_dir}")

# Import and run
from scripts.evaluate_option_expiries import evaluate_expiries_for_symbol

evaluate_expiries_for_symbol(symbol="GLD", limit=limit, framework=framework)
