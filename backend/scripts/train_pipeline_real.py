#!/usr/bin/env python3
"""
train_pipeline_real.py — Train Tier 1 Meta-Learner and Tier 2 Q-Learner on Real Market Data.

Alternates between Gold (GLD) and Bitcoin (BTC/USD).
Performs backward `merge_asof` index alignment across timeframes (5m, 15m, 1h, 4h, 1d) to eliminate lookahead bias.
Executes sequential pretraining: Tier 1 Meta-Learner first -> Tier 2 Options Q-Learner second.
"""

import sys
import os
import argparse
import json
import logging
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.ml.real_data_pipeline import RealTrainConfig, run_real_data_pipeline
from app.db.connection import init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TrainPipelineReal")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-steps", type=int, default=300, help="Meta-Learner training steps per symbol")
    parser.add_argument("--q-steps", type=int, default=300, help="Options Q-Learner training steps per symbol")
    parser.add_argument("--target-rows", type=int, default=40000, help="Number of candles/rows per timeframe")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    init_db()
    config = RealTrainConfig(
        symbols=("GLD", "BTC/USD"),
        target_rows_per_tf=args.target_rows,
        meta_train_steps=args.meta_steps,
        q_train_steps=args.q_steps,
        batch_size=args.batch_size,
    )

    logger.info("Starting real data pretraining pipeline for Gold (GLD) & Bitcoin (BTC/USD)...")
    res = run_real_data_pipeline(config)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
