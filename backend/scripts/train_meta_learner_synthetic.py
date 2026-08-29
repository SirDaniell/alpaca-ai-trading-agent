"""Train the signal meta-learner on synthetic OHLCV and store the session in the DB."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.ml.synthetic_meta_trainer import (  # noqa: E402
    SyntheticTrainConfig,
    result_to_dict,
    train_from_synthetic,
)
from app.db.connection import SessionLocal, init_db  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-candles", type=int, default=400, help="How many synthetic bars to generate and learn from")
    parser.add_argument("--train-steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--scope", default="synthetic-default")
    parser.add_argument("--no-persist", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    init_db()
    db = SessionLocal()
    try:
        result = train_from_synthetic(
            SyntheticTrainConfig(
                symbol=args.symbol,
                num_candles=args.num_candles,
                seed=args.seed,
                train_steps=args.train_steps,
                batch_size=args.batch_size,
                persist=not args.no_persist,
                scope=args.scope,
            ),
            db=db,
        )
        print(json.dumps(result_to_dict(result), indent=2))
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
