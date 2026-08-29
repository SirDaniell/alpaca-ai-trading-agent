"""
Processing Utils  — Orchestration helpers and StrategyFactory.

Import order (acyclic):

    processing_types          (pure primitives, no app imports)
         ↑
    processing_strategies     (concrete strategy classes + worker functions)
         ↑
    processing_utils          (StrategyFactory, orchestration helpers)   ← YOU ARE HERE
         ↑
    processing_handlers       (imports StrategyFactory from processing_strategies)

Changes from the old layout
───────────────────────────
• ProcessingStrategy enum      → processing_types.py
• ProcessingContext dataclass   → processing_types.py
• ProcessingStrategyBase ABC   → processing_types.py
• StrategyFactory              MOVED FROM processing_utils → processing_strategies
  (it needs the concrete classes, so it must live below them in the hierarchy)

All three names are re-exported here so existing callers that do
  from app.core.processing.processing_utils import ProcessingContext, ...
continue to work without any change.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

# ── stdlib / third-party ──────────────────────────────────────────────────────
from sqlalchemy import insert, select, desc, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# ── app-internal (non-circular) ──────────────────────────────────────────────
from app.core.config import ProcessingConfig
from app.core.processing.tasks import TaskStore, TaskCancelledException
from app.core.services.multiprocessing_config import init_spawn_method
from app.api.routes.data.database import AsyncPostgresSessionLocal
from app.database.models import ChunkCheckpoint
from app.core.data.session_data_loader import (
    store_session_step_result,
    set_as_current_data,
)
from app.core.services.decompress_cache import get_cache as get_decompress_cache
from app.core.data.serializers import serialize_data, to_serializable
from app.core.ml.ml_dataset_preparation import MLDatasetPreparation, DatasetConfig
from app.core.ml.ml_validation import validate_ml_data
from app.core.processing.progress_reporter import ProgressReporter

# ── pure primitives (leaf module — no app imports) ───────────────────────────
from app.core.processing.processing_types import (
    ProcessingStrategy,
    ProcessingContext,
    ProcessingStrategyBase,
)

# ── StrategyFactory lives in processing_strategies (it needs the concrete
#    classes).  We import it here and re-export so callers don't break.
from app.core.processing.processing_strategies import StrategyFactory

# ── Public re-exports (backward-compat) ──────────────────────────────────────
__all__ = [
    "ProcessingStrategy",
    "ProcessingContext",
    "ProcessingStrategyBase",
    "StrategyFactory",
]