"""
meta_learner_registry.py — Model versioning and hot-swap registry for OnlineSignalMetaLearner.

Answers:
- List all stored model versions (per symbol, per scope)
- Evaluate checkpoint performance from stored eval_metrics
- Switch the active model for a symbol at runtime (hot-swap)
- Pretrain a fresh model for any symbol and store it
- Forward inference to the currently active model

Usage pattern (service layer):
    registry = MetaLearnerModelRegistry()
    registry.pretrain(symbol="EURUSD", scope="eurusd-base-v1", db=db)
    registry.set_active(symbol="EURUSD", checkpoint_id="<uuid>", db=db)
    pred = registry.score_live(symbol="EURUSD", features=window, direction="bullish")
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.core.ml.signal_meta_learner import SIGNAL_META_FEATURE_COUNT, OnlineSignalMetaLearner
from app.core.ml.synthetic_meta_trainer import SyntheticTrainConfig, SyntheticTrainResult, train_from_synthetic
from app.db.models import LearnerCheckpoint

logger = logging.getLogger(__name__)


@dataclass
class ModelEntry:
    """Summary of a stored checkpoint suitable for listing / comparison."""
    checkpoint_id: str
    scope: str
    symbol: str
    arch_version: str
    checkpoint_version: str
    is_active: bool
    total_steps: int
    buffer_size: int
    scaler_fitted: bool
    eval_metrics: Optional[Dict[str, Any]]
    notes: Optional[str]
    created_at: str

    # Convenience accessors into eval_metrics
    @property
    def win_rate(self) -> Optional[float]:
        return (self.eval_metrics or {}).get("win_rate")

    @property
    def avg_reward(self) -> Optional[float]:
        return (self.eval_metrics or {}).get("avg_reward")

    @property
    def mfe_mae_ratio(self) -> Optional[float]:
        return (self.eval_metrics or {}).get("avg_mfe_mae_ratio")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "scope": self.scope,
            "symbol": self.symbol,
            "arch_version": self.arch_version,
            "is_active": self.is_active,
            "total_steps": self.total_steps,
            "buffer_size": self.buffer_size,
            "scaler_fitted": self.scaler_fitted,
            "win_rate": self.win_rate,
            "avg_reward": self.avg_reward,
            "mfe_mae_ratio": self.mfe_mae_ratio,
            "eval_metrics": self.eval_metrics,
            "notes": self.notes,
            "created_at": self.created_at,
        }


class MetaLearnerModelRegistry:
    """
    Manages versioned OnlineSignalMetaLearner models per (symbol, scope).

    - One ACTIVE model per symbol is hot-swappable at runtime.
    - All versions persist in the DB; old ones are deactivated but not deleted.
    - The FeatureScaler is bundled inside each checkpoint — switching models
      atomically switches the scaler too, so no normalisation mismatch at inference.

    Q-Learner compatibility:
        The same registry pattern applies 1:1 to a future Q-Learner. The
        checkpoint format will differ (epsilon, q_table) but listing, evaluation,
        and hot-swap follow the same API.
    """

    def __init__(self) -> None:
        # Active learner per symbol: {symbol -> OnlineSignalMetaLearner}
        self._active: Dict[str, OnlineSignalMetaLearner] = {}

    # ── Pretrain ─────────────────────────────────────────────────────────────

    def pretrain(
        self,
        symbol: str,
        scope: str,
        db: Session,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> SyntheticTrainResult:
        """
        Pretrain a fresh model for a given symbol and store it in the DB.

        Uses the chronological train/val/test split:
            TRAIN (70%) → fit scaler + fit model
            VAL   (15%) → eval_metrics (win_rate, avg_reward, mfe_mae_ratio)
            TEST  (15%) → held-out (most recent data, just before today)

        The new checkpoint is set as `is_active=True` and the previous active
        checkpoint for the same (scope, symbol) is deactivated.

        Args:
            symbol: Instrument ticker, e.g. "AAPL", "EURUSD"
            scope: Training intent identifier, e.g. "bootstrap-v1", "eurusd-live"
            db: SQLAlchemy session (must be open; caller commits or registry commits internally)
            config_overrides: Optional dict of SyntheticTrainConfig field overrides

        Returns:
            SyntheticTrainResult with checkpoint_id, eval_metrics, and run metrics
        """
        cfg_kwargs = {"symbol": symbol, "scope": scope, "persist": True}
        if config_overrides:
            cfg_kwargs.update(config_overrides)
        config = SyntheticTrainConfig(**cfg_kwargs)
        result = train_from_synthetic(config=config, db=db)

        # Hot-load the new checkpoint so inference is available immediately
        if result.checkpoint_id:
            self._load_checkpoint_into_active(symbol, result.checkpoint_id, db)
            logger.info(
                "[Registry] Pretrained and activated checkpoint %s for symbol=%s scope=%s",
                result.checkpoint_id, symbol, scope,
            )
        return result

    # ── List ─────────────────────────────────────────────────────────────────

    def list_models(
        self,
        db: Session,
        symbol: Optional[str] = None,
        scope: Optional[str] = None,
        limit: int = 50,
    ) -> List[ModelEntry]:
        """
        List stored checkpoints sorted by creation date (newest first).

        Args:
            symbol: Filter by instrument ticker (optional)
            scope: Filter by training scope (optional)
            limit: Max results to return
        """
        q = db.query(LearnerCheckpoint)
        if symbol:
            q = q.filter(LearnerCheckpoint.symbol == symbol)
        if scope:
            q = q.filter(LearnerCheckpoint.scope == scope)
        rows = q.order_by(LearnerCheckpoint.created_at.desc()).limit(limit).all()

        entries = []
        for row in rows:
            counters = row.training_counters or {}
            entries.append(ModelEntry(
                checkpoint_id=row.checkpoint_id,
                scope=row.scope,
                symbol=row.symbol or "unknown",
                arch_version=getattr(row, "arch_version", "v1.0") or "v1.0",
                checkpoint_version=row.checkpoint_version or "v1.0",
                is_active=bool(getattr(row, "is_active", False)),
                total_steps=int(counters.get("total_steps", 0)),
                buffer_size=int(counters.get("buffer_size", 0)),
                scaler_fitted=bool(counters.get("scaler_fitted", False)),
                eval_metrics=getattr(row, "eval_metrics", None),
                notes=getattr(row, "notes", None),
                created_at=row.created_at.isoformat() if row.created_at else "",
            ))
        return entries

    # ── Evaluate ─────────────────────────────────────────────────────────────

    def evaluate(self, checkpoint_id: str, db: Session) -> Optional[Dict[str, Any]]:
        """
        Return the stored eval_metrics for a checkpoint.

        Eval metrics are computed at training time on the validation slice
        (15% most-recent data within the training window). They reflect:
            - win_rate: fraction of val samples with positive reward
            - avg_reward: mean RL reward across val samples
            - avg_mfe_mae_ratio: mean MFE/MAE (>= 2.0 = good risk/reward)
            - avg_reversal_prob: mean reversal probability predicted

        For deeper re-evaluation on new data, use pretrain() with fresh candles.
        """
        row = db.query(LearnerCheckpoint).filter_by(checkpoint_id=checkpoint_id).first()
        if row is None:
            logger.warning("[Registry] Checkpoint %s not found.", checkpoint_id)
            return None
        return {
            "checkpoint_id": checkpoint_id,
            "symbol": getattr(row, "symbol", "unknown"),
            "scope": row.scope,
            "is_active": bool(getattr(row, "is_active", False)),
            "eval_metrics": getattr(row, "eval_metrics", None),
            "training_counters": row.training_counters,
            "created_at": row.created_at.isoformat() if row.created_at else "",
        }

    # ── Switch active model ───────────────────────────────────────────────────

    def set_active(self, symbol: str, checkpoint_id: str, db: Session) -> None:
        """
        Hot-swap the active model for a symbol.

        1. Deactivates all other checkpoints for the same symbol in DB.
        2. Marks the target checkpoint as is_active=True.
        3. Loads it into the in-memory active map — next predict() uses it.

        The FeatureScaler inside the checkpoint is restored atomically with the
        network weights, so normalisation is always consistent.
        """
        # Update DB flags
        db.query(LearnerCheckpoint).filter(
            LearnerCheckpoint.symbol == symbol,
            LearnerCheckpoint.is_active == True,  # noqa: E712
        ).update({"is_active": False})

        row = db.query(LearnerCheckpoint).filter_by(checkpoint_id=checkpoint_id).first()
        if row is None:
            raise ValueError(f"Checkpoint {checkpoint_id} not found.")
        row.is_active = True
        db.commit()

        # Load into in-memory active store
        self._load_checkpoint_into_active(symbol, checkpoint_id, db)
        logger.info("[Registry] Switched active model for symbol=%s → checkpoint=%s", symbol, checkpoint_id)

    def _load_checkpoint_into_active(self, symbol: str, checkpoint_id: str, db: Session) -> None:
        row = db.query(LearnerCheckpoint).filter_by(checkpoint_id=checkpoint_id).first()
        if row is None:
            raise ValueError(f"Checkpoint {checkpoint_id} not found.")
        b64 = (row.payload or {}).get("checkpoint_b64")
        if not b64:
            raise ValueError(f"Checkpoint {checkpoint_id} has no payload.")
        learner = OnlineSignalMetaLearner()
        learner.import_checkpoint(base64.b64decode(b64))
        self._active[symbol] = learner

    # ── Inference ─────────────────────────────────────────────────────────────

    def get_active(self, symbol: str) -> Optional[OnlineSignalMetaLearner]:
        """Return the in-memory active learner for a symbol, or None if not loaded."""
        return self._active.get(symbol)

    def ensure_active_loaded(self, symbol: str, db: Session) -> OnlineSignalMetaLearner:
        """
        Return the active learner, loading from DB if not in memory.
        Raises RuntimeError if no active checkpoint exists for this symbol.
        """
        if symbol in self._active:
            return self._active[symbol]

        row = (
            db.query(LearnerCheckpoint)
            .filter(LearnerCheckpoint.symbol == symbol, LearnerCheckpoint.is_active == True)  # noqa: E712
            .order_by(LearnerCheckpoint.created_at.desc())
            .first()
        )
        if row is None:
            raise RuntimeError(
                f"No active checkpoint for symbol={symbol}. "
                f"Run pretrain() or set_active() first."
            )
        self._load_checkpoint_into_active(symbol, row.checkpoint_id, db)
        return self._active[symbol]

    def score_live(
        self,
        symbol: str,
        features: Any,
        direction: str,
        db: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """
        Score a live feature window using the active model for the symbol.

        Args:
            symbol: Instrument ticker
            features: 48-bar window dict, flat list, or np.ndarray
            direction: "bullish" or "bearish"
            db: Optional session — only needed if model not yet in memory

        Returns:
            Full prediction dict (signal_strength, expected_pips, reversal_prob, etc.)
        """
        learner = self._active.get(symbol)
        if learner is None and db is not None:
            learner = self.ensure_active_loaded(symbol, db)
        if learner is None:
            raise RuntimeError(f"No active model loaded for symbol={symbol}.")
        return learner.predict(features, direction=direction)


# Module-level singleton for import by API routers
meta_learner_registry = MetaLearnerModelRegistry()
