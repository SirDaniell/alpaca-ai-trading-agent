"""
Processing Types - Pure Primitive Definitions

Leaf module with zero intra-app imports.  Defines only:
  • ProcessingStrategy  – enum
  • ProcessingContext   – dataclass
  • ProcessingStrategyBase – ABC

Import order is now acyclic:

    processing_types          (no app imports)
         ↑
    processing_strategies     (imports types + handlers)
         ↑
    processing_handlers       (imports types + processing_strategies.StrategyFactory)
         ↑
    processing_utils          (imports types + processing_strategies for StrategyFactory)

Any module that previously imported ProcessingContext / ProcessingStrategy /
ProcessingStrategyBase from processing_utils or processing_strategies should
import them from here instead.
"""

from __future__ import annotations

import io
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd


# ============================================================================
# ENUM
# ============================================================================

class ProcessingStrategy(str, Enum):
    """Processing strategies based on data size."""
    SEQUENTIAL = "sequential"           # ≤ threshold_sequential_max rows
    PARALLEL_CHUNKING = "parallel_chunking"  # threshold_sequential_max < rows ≤ threshold_parallel_max
    SLICE_STREAMING = "slice_streaming"      # > threshold_parallel_max rows


# ============================================================================
# DATACLASS
# ============================================================================

@dataclass
class ProcessingContext:
    """Context passed to all handlers for consistent execution."""

    task_id: str
    session_id: str
    analysis_type: str
    config: Any  # Analysis-specific config (TechnicalConfig, SNRConfig, etc.)
    task_store: Optional[Any] = None
    connection_manager: Optional[Any] = None
    processing_config: Optional[Any] = None   # ProcessingConfig – typed as Any to stay import-free
    user_id: str = "unknown"

    # Slice context (for slice_streaming strategy)
    slice_num: int = 0
    total_slices: int = 1
    slice_start: int = 0
    slice_end: int = 0
    original_slice_start: int = 0   # Original boundary for aggregation
    original_slice_end: int = 0     # Original boundary for aggregation
    total_dataset_rows: int = 0
    global_offset: int = 0
    global_scaler: Optional[Any] = None  # Externally fitted scaler for SLICE_STREAMING


# ============================================================================
# ABSTRACT BASE CLASS
# ============================================================================

class ProcessingStrategyBase(ABC):
    """
    Abstract base class for processing strategies.

    All strategies implement the same interface but differ in execution approach:
    - Sequential:        Single-threaded processing
    - ParallelChunking:  Multi-process chunking with overlap
    - SliceStreaming:    Memory-efficient slice-by-slice processing with checkpoints
    """

    def __init__(
        self,
        context: ProcessingContext,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.context = context
        self.logger = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def execute(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Execute processing strategy.

        Args:
            df:       Input DataFrame
            **kwargs: Additional parameters (fit_scaler, split_type, etc.)

        Returns:
            Result dictionary with keys:
                result_df – Enriched DataFrame
                metadata  – Strategy metadata
                [analysis-specific keys]
        """

    # ------------------------------------------------------------------
    # Shared helpers (used by every concrete strategy)
    # ------------------------------------------------------------------

    async def _send_progress(self, progress: float, message: str, **kwargs) -> None:
        """Send progress update via WebSocket."""
        if not self.context.connection_manager:
            return
        try:
            await self.context.connection_manager.send_progress_update(
                self.context.task_id,
                {
                    "type": "progress",
                    "progress": progress,
                    "message": message,
                    "stage": kwargs.get("stage", "processing"),
                    **kwargs,
                },
            )
        except Exception as exc:
            self.logger.warning(f"Progress update failed: {exc}")

    def _accumulate_metadata(
        self, accumulator: Dict[str, Any], result: Dict[str, Any]
    ) -> None:
        """Accumulate metadata from a chunk/slice result into *accumulator*."""
        for key, value in result.items():
            if key in ("chunk_id", "global_start_idx", "error", "metadata", "result_df"):
                continue
            if value is None:
                continue

            if isinstance(value, list):
                accumulator.setdefault(key, []).extend(value)
            elif isinstance(value, dict):
                if key not in accumulator:
                    accumulator[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        accumulator[key][sub_key] = accumulator[key].get(sub_key, 0) + sub_value
                    elif isinstance(sub_value, list):
                        accumulator[key].setdefault(sub_key, []).extend(sub_value)
                    elif isinstance(sub_value, np.ndarray):
                        accumulator[key].setdefault(sub_key, []).append(sub_value)
                    else:
                        accumulator[key][sub_key] = sub_value
            elif isinstance(value, (int, float)):
                accumulator.setdefault(key, []).append(value)
            else:
                accumulator[key] = value

    def _apply_accumulated_metadata(
        self, result: Dict[str, Any], accumulator: Dict[str, Any]
    ) -> None:
        """Finalise accumulated metadata into *result*."""
        for key, value in accumulator.items():
            if isinstance(value, list) and key in (
                "g_start", "g_end", "total_signals", "rows_processed"
            ):
                if key == "g_start":
                    result[key] = min(value)
                elif key == "g_end":
                    result[key] = max(value)
                else:
                    result[key] = sum(value)
            elif isinstance(value, dict) and key == "ml_dataset":
                merged_ml: Dict[str, Any] = {}
                for sub_key, sub_val in value.items():
                    if (
                        isinstance(sub_val, list)
                        and sub_val
                        and isinstance(sub_val[0], np.ndarray)
                    ):
                        merged_ml[sub_key] = np.concatenate(sub_val, axis=0)
                    else:
                        merged_ml[sub_key] = sub_val
                result[key] = merged_ml
            else:
                result[key] = value
                if key == "signals" and isinstance(value, list):
                    result["total_signals"] = len(value)

    def _merge_ml_split_dicts(
        self, split_dicts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Merge split payloads (e.g. from multiple slices) by concatenation."""
        if not split_dicts:
            return {}

        merged: Dict[str, Any] = {}
        all_keys: set = set()
        for d in split_dicts:
            if isinstance(d, dict):
                all_keys.update(d.keys())

        for key in all_keys:
            values = [d[key] for d in split_dicts if isinstance(d, dict) and key in d]
            if not values:
                continue
            first = values[0]
            if isinstance(first, np.ndarray):
                merged[key] = np.concatenate(values, axis=0)
            elif isinstance(first, list):
                combined: list = []
                for v in values:
                    if isinstance(v, list):
                        combined.extend(v)
                merged[key] = combined
            elif isinstance(first, dict):
                merged[key] = self._merge_ml_split_dicts(values)  # type: ignore[arg-type]
            else:
                merged[key] = first

        return merged

    # ------------------------------------------------------------------
    # Scaler serialisation helpers (needed in worker-spawning strategies)
    # ------------------------------------------------------------------

    def _serialize_scaler(self, scaler: Any) -> Optional[bytes]:
        """Serialise *scaler* to bytes via joblib for multiprocessing."""
        if scaler is None:
            return None
        try:
            buf = io.BytesIO()
            joblib.dump(scaler, buf)
            return buf.getvalue()
        except Exception as exc:
            self.logger.warning(f"Failed to serialise scaler: {exc}")
            return None

    def _deserialize_scaler(self, scaler_bytes: Optional[bytes]) -> Any:
        """Deserialise a joblib-serialised scaler from *scaler_bytes*."""
        if scaler_bytes is None:
            return None
        try:
            buf = io.BytesIO(scaler_bytes)
            return joblib.load(buf)
        except Exception as exc:
            self.logger.warning(f"Failed to deserialise scaler: {exc}")
            return None