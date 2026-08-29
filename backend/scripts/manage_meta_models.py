#!/usr/bin/env python3
"""
manage_meta_models.py — CLI for Meta-Learner Model Registry.

Commands:
    list        List stored checkpoints (optionally filter by --symbol / --scope)
    evaluate    Show eval_metrics for a specific checkpoint
    set-active  Hot-swap the active model for a symbol
    pretrain    Train a new model for a symbol and store it

Examples:
    # List all models
    python scripts/manage_meta_models.py list

    # List models for one symbol
    python scripts/manage_meta_models.py list --symbol AAPL

    # Evaluate a specific checkpoint
    python scripts/manage_meta_models.py evaluate --checkpoint-id <uuid>

    # Switch active model
    python scripts/manage_meta_models.py set-active --symbol AAPL --checkpoint-id <uuid>

    # Pretrain a new symbol-specific model
    python scripts/manage_meta_models.py pretrain --symbol EURUSD --scope eurusd-base-v1 --num-candles 500 --train-steps 30
"""

from __future__ import annotations

import argparse
import json
import sys
import os

# Allow running from backend/ or from backend/scripts/
_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from app.db.connection import SessionLocal, init_db
from app.core.ml.meta_learner_registry import MetaLearnerModelRegistry


def cmd_list(args: argparse.Namespace) -> None:
    init_db()
    db = SessionLocal()
    try:
        registry = MetaLearnerModelRegistry()
        entries = registry.list_models(
            db,
            symbol=args.symbol or None,
            scope=args.scope or None,
            limit=args.limit,
        )
        if not entries:
            print("No checkpoints found.")
            return

        # Table header
        col = "{:<38} {:<10} {:<22} {:<8} {:<8} {:<10} {:<10} {:<10}"
        print(col.format("checkpoint_id", "symbol", "scope", "active", "steps", "win_rate", "avg_reward", "mfe/mae"))
        print("-" * 120)
        for e in entries:
            print(col.format(
                e.checkpoint_id,
                e.symbol,
                e.scope[:22],
                "✓" if e.is_active else "",
                str(e.total_steps),
                f"{e.win_rate:.2%}" if e.win_rate is not None else "—",
                f"{e.avg_reward:.4f}" if e.avg_reward is not None else "—",
                f"{e.mfe_mae_ratio:.2f}" if e.mfe_mae_ratio is not None else "—",
            ))
        print(f"\n{len(entries)} checkpoint(s) shown.")
    finally:
        db.close()


def cmd_evaluate(args: argparse.Namespace) -> None:
    if not args.checkpoint_id:
        print("Error: --checkpoint-id is required.", file=sys.stderr)
        sys.exit(1)
    init_db()
    db = SessionLocal()
    try:
        registry = MetaLearnerModelRegistry()
        result = registry.evaluate(args.checkpoint_id, db)
        if result is None:
            print(f"Checkpoint {args.checkpoint_id} not found.")
        else:
            print(json.dumps(result, indent=2, default=str))
    finally:
        db.close()


def cmd_set_active(args: argparse.Namespace) -> None:
    if not args.symbol or not args.checkpoint_id:
        print("Error: --symbol and --checkpoint-id are required.", file=sys.stderr)
        sys.exit(1)
    init_db()
    db = SessionLocal()
    try:
        registry = MetaLearnerModelRegistry()
        registry.set_active(args.symbol, args.checkpoint_id, db)
        print(f"✓ Active model for {args.symbol} → {args.checkpoint_id}")
    finally:
        db.close()


def cmd_pretrain(args: argparse.Namespace) -> None:
    if not args.symbol:
        print("Error: --symbol is required.", file=sys.stderr)
        sys.exit(1)
    init_db()
    db = SessionLocal()
    try:
        registry = MetaLearnerModelRegistry()
        overrides = {}
        if args.num_candles:
            overrides["num_candles"] = args.num_candles
        if args.train_steps:
            overrides["train_steps"] = args.train_steps
        if args.batch_size:
            overrides["batch_size"] = args.batch_size
        if args.seed is not None:
            overrides["seed"] = args.seed
        if args.train_ratio:
            overrides["train_ratio"] = args.train_ratio
        if args.val_ratio:
            overrides["val_ratio"] = args.val_ratio
        if args.notes:
            overrides["notes"] = args.notes

        print(f"Pretraining {args.symbol} / scope={args.scope} ...")
        result = registry.pretrain(
            symbol=args.symbol,
            scope=args.scope or f"{args.symbol.lower()}-base-v1",
            db=db,
            config_overrides=overrides,
        )
        print(json.dumps({
            "run_id": result.run_id,
            "symbol": result.symbol,
            "checkpoint_id": result.checkpoint_id,
            "experiences_recorded": result.experiences_recorded,
            "final_loss": result.final_loss,
            "weights_changed": result.weights_changed,
            "eval_metrics": result.metrics.get("eval_metrics"),
            "scaler_fitted": result.metrics.get("scaler_fitted"),
        }, indent=2))
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="manage_meta_models",
        description="Meta-Learner Model Registry CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = sub.add_parser("list", help="List stored checkpoints")
    p_list.add_argument("--symbol", default="")
    p_list.add_argument("--scope", default="")
    p_list.add_argument("--limit", type=int, default=20)

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Show eval_metrics for a checkpoint")
    p_eval.add_argument("--checkpoint-id", required=True)

    # set-active
    p_swap = sub.add_parser("set-active", help="Switch active model for a symbol")
    p_swap.add_argument("--symbol", required=True)
    p_swap.add_argument("--checkpoint-id", required=True)

    # pretrain
    p_train = sub.add_parser("pretrain", help="Pretrain a new model for a symbol")
    p_train.add_argument("--symbol", required=True)
    p_train.add_argument("--scope", default="")
    p_train.add_argument("--num-candles", type=int, default=400)
    p_train.add_argument("--train-steps", type=int, default=40)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--seed", type=int, default=None)
    p_train.add_argument("--train-ratio", type=float, default=0.70)
    p_train.add_argument("--val-ratio", type=float, default=0.15)
    p_train.add_argument("--notes", default="")

    args = parser.parse_args()

    dispatch = {
        "list": cmd_list,
        "evaluate": cmd_evaluate,
        "set-active": cmd_set_active,
        "pretrain": cmd_pretrain,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
