"""
UNIFIED ProcessingManager (v3) - Complete Modular Architecture

This is the definitive implementation of ProcessingManager that:
✅ Uses uniform handlers for all analysis types
✅ Auto-selects strategies (Sequential/Parallel/Streaming)
✅ Integrates with AnalysisManager TIER system
✅ Supports checkpointing and recovery
✅ Provides WebSocket progress updates
✅ Registers handlers at module load

Key Flow:
    analysis.py endpoint
        ↓
    ProcessingManager(analysis_type="snr", config=SNRConfig(...))
        ↓
    execute(df) → StrategyFactory.determine_strategy()
        ↓
    StrategyFactory.create_strategy() → SequentialStrategy|ParallelStrategy|SliceStreamingStrategy
        ↓
    strategy.execute(df) → HandlerRegistry.get() → handler(df, context)
        ↓
    Result → TIER 1 cache + DB storage
        ↓
    AnalysisManager receives result, updates TIER 0

Architecture Guarantees:
- All analysis types use same code path
- New processors added via HandlerRegistry.register()
- No repetitive code across different analyses
- Single source of truth for strategy selection
- Unified progress/checkpoint/recovery
"""

import logging
import asyncio
import pandas as pd
from typing import Dict, Optional, Any, Tuple
from datetime import datetime
import gc

from sqlalchemy import insert, select, desc

from app.core.config import ProcessingConfig
from app.core.processing.tasks import TaskStore, TaskCancelledException
from app.core.services.multiprocessing_config import init_spawn_method
from app.api.routes.data.database import AsyncPostgresSessionLocal
from app.database.models import ChunkCheckpoint
from app.core.data.session_data_loader import store_session_step_result

# CRITICAL: Import the refactored components
from app.core.processing.processing_strategies import (
    ProcessingStrategy,
    ProcessingContext,
    StrategyFactory,
    HandlerRegistry,
)
from app.core.processing.processing_handlers import (
    analyze_technical_impl,
    analyze_snr_impl,
    analyze_astronomical_impl,
    analyze_ml_prep_impl,
    analyze_model_training_impl,
    analyze_model_build_impl,
)

# Import configs
from app.core.config import (
    TechnicalConfig,
    SNRConfig,
    AstronomicalConfig,
    ProcessingConfig,
)

logger = logging.getLogger(__name__)


# ============================================================================
# TIER 1 INTERMEDIATE RESULTS CACHE
# ============================================================================

class CachedStepData:
    """Wrapper for cached step data with expiration tracking."""
    
    def __init__(self, data: Any, ttl_seconds: int = 1800):
        """Initialize cached data with TTL (default 30 min)."""
        self.data = data
        self.created_at = datetime.now(timezone.utc)
        self.ttl_seconds = ttl_seconds
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        age = (datetime.now(timezone.utc) - self.created_at).total_seconds()
        return age > self.ttl_seconds
    
    def get_age_minutes(self) -> float:
        """Get age of cached data in minutes."""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds() / 60


class IntermediateResultsCache:
    """
    TIER 1 Cache: Intermediate analysis results between steps
    
    - 30-minute TTL (auto-expire to prevent stale data)
    - Per-task scope: {(task_id, step_name): CachedStepData}
    - Thread-safe (Python dict is atomic for basic operations)
    - Falls back to database if cache miss
    
    Usage:
        IntermediateResultsCache.store(task_id, "technical_analysis", df)
        cached_df = IntermediateResultsCache.retrieve(task_id, "technical_analysis")
    """
    
    _cache: Dict[tuple, CachedStepData] = {}
    
    @classmethod
    def store(cls, task_id: str, step_name: str, data: Any, ttl_seconds: int = 1800) -> None:
        """Cache intermediate result for next step."""
        key = (task_id, step_name)
        cls._cache[key] = CachedStepData(data, ttl_seconds)
        logger.info(f"[Cache] Stored {step_name} for task {task_id} (TTL: {ttl_seconds}s)")
    
    @classmethod
    def retrieve(cls, task_id: str, step_name: str) -> Optional[Any]:
        """Retrieve cached result, return None if expired or missing."""
        key = (task_id, step_name)
        if key not in cls._cache:
            return None
        
        cached = cls._cache[key]
        if cached.is_expired():
            del cls._cache[key]
            logger.info(f"[Cache] Expired: {step_name} (age: {cached.get_age_minutes():.1f} min)")
            return None
        
        age_min = cached.get_age_minutes()
        logger.info(f"[Cache] Retrieved {step_name} (age: {age_min:.1f} min)")
        return cached.data
    
    @classmethod
    def clear(cls, task_id: str) -> None:
        """Clear all cache entries for a task."""
        keys_to_delete = [k for k in cls._cache.keys() if k[0] == task_id]
        for key in keys_to_delete:
            del cls._cache[key]
        logger.info(f"[Cache] Cleared {len(keys_to_delete)} entries for task {task_id}")
    
    @classmethod
    def cleanup_expired(cls) -> int:
        """Remove expired entries, return count removed."""
        keys_to_delete = [k for k, v in cls._cache.items() if v.is_expired()]
        for key in keys_to_delete:
            del cls._cache[key]
        return len(keys_to_delete)


# ============================================================================
# REGISTER ALL HANDLERS AT MODULE LOAD
# ============================================================================

try:
    HandlerRegistry.register("technical", analyze_technical_impl, TechnicalConfig)
    HandlerRegistry.register("snr", analyze_snr_impl, SNRConfig)
    HandlerRegistry.register("astronomical", analyze_astronomical_impl, AstronomicalConfig)
    HandlerRegistry.register("ml_preparation", analyze_ml_prep_impl, dict)
    HandlerRegistry.register("model_training", analyze_model_training_impl, dict)
    HandlerRegistry.register("model_build", analyze_model_build_impl, dict)
    logger.info("✅ All processing handlers registered successfully")
except Exception as reg_err:
    logger.error(f"❌ Failed to register handlers: {reg_err}", exc_info=True)
    raise


# ============================================================================
# PROCESSING MANAGER (UNIFIED)
# ============================================================================

class ProcessingManager:
    """
    Unified orchestrator for all analysis types.
    
    Responsibilities:
    1. ✅ Strategy auto-selection based on data size and analysis type
    2. ✅ Context building (ProcessingContext)
    3. ✅ Execution delegation to selected strategy
    4. ✅ Result storage (TIER 1 cache + DB)
    5. ✅ Progress tracking (task_store + WebSocket)
    6. ✅ Checkpoint & recovery for large datasets
    
    Usage Pattern:
        pm = ProcessingManager(
            session_id=session_id,
            task_id=task_id,
            analysis_type="snr",
            config=SNRConfig(...),
            task_store=task_store,
            connection_manager=manager,
        )
        result = await pm.execute(df)
    """

    def __init__(
        self,
        session_id: str,
        task_id: str,
        analysis_type: str,
        config: Any,
        task_store: Optional[TaskStore] = None,
        connection_manager: Optional[Any] = None,
        processing_config: Optional[ProcessingConfig] = None,
        step_name: Optional[str] = None,
    ):
        """Initialize ProcessingManager."""
        init_spawn_method()

        self.session_id = session_id
        self.task_id = task_id
        self.analysis_type = analysis_type
        self.config = config
        self.task_store = task_store
        self.connection_manager = connection_manager
        self.processing_config = processing_config or ProcessingConfig()
        self.step_name = step_name or f"{analysis_type}_analysis"
        self.logger = logging.getLogger(__name__)

        # Progress tracking
        self.total_rows = 0
        self.rows_processed = 0
        self.current_stage = "initializing"

    async def execute(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Main entry point: Execute analysis with automatic strategy selection.
        
        Args:
            df: Input DataFrame
            **kwargs: Additional parameters (user_id, etc.)
            
        Returns:
            Result dict with enriched data and metadata
            
        Raises:
            TaskCancelledException: If task is cancelled
            ValueError: If handler not registered
        """
        try:
            n_rows = len(df)
            self.total_rows = n_rows
            self.rows_processed = 0

            # ✅ STEP 1: Auto-select strategy based on size + analysis type
            strategy_type = StrategyFactory.determine_strategy(
                n_rows,
                self.processing_config,
                self.analysis_type
            )

            self.logger.info(
                f"[PM] Executing {self.analysis_type} with {n_rows} rows "
                f"using {strategy_type.value} strategy"
            )

            # ✅ STEP 2: Send initial progress
            await self._send_progress_update(0, f"Starting {self.analysis_type} analysis...")

            # ✅ STEP 3: Execute with selected strategy
            if strategy_type == ProcessingStrategy.SLICE_STREAMING:
                result = await self._execute_with_slicing(df, strategy_type, **kwargs)
            else:
                result = await self._execute_with_strategy(strategy_type, df, **kwargs)

            # ✅ STEP 4: Cache result for next step (TIER 1)
            if "result_df" in result:
                IntermediateResultsCache.store(
                    self.task_id,
                    self.step_name,
                    result["result_df"],
                    ttl_seconds=1800  # 30 min
                )
                self.logger.info(f"[PM] Cached {self.step_name} result to TIER 1")

            # ✅ STEP 5: Persist to database (TIER 3)
            await self._persist_to_database(result)

            # ✅ STEP 6: Send completion message
            await self._send_progress_update(100, f"{self.analysis_type} completed")

            return result

        except TaskCancelledException:
            self.logger.warning(f"[PM] Task {self.task_id} was cancelled")
            raise
        except Exception as err:
            self.logger.error(f"[PM] Execution failed: {err}", exc_info=True)
            await self._send_error(str(err))
            raise

    async def _execute_with_strategy(
        self,
        strategy_type: ProcessingStrategy,
        df: pd.DataFrame,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute with Sequential or Parallel Chunking strategy.
        
        Args:
            strategy_type: Strategy to use
            df: Input DataFrame
            **kwargs: Additional parameters
            
        Returns:
            Result dictionary
        """
        # Build processing context
        context = ProcessingContext(
            task_id=self.task_id,
            session_id=self.session_id,
            analysis_type=self.analysis_type,
            config=self.config,
            task_store=self.task_store,
            connection_manager=self.connection_manager,
            processing_config=self.processing_config,
        )

        # Create strategy instance
        strategy = StrategyFactory.create_strategy(
            strategy_type,
            context,
            self.logger
        )

        # Execute strategy
        result = await strategy.execute(df)

        return result

    async def _execute_with_slicing(
        self,
        df: pd.DataFrame,
        strategy_type: ProcessingStrategy,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute with Slice Streaming strategy for large datasets.
        
        Handles:
        - Slice creation with overlap
        - Per-slice processing with checkpoints
        - Result aggregation
        - Recovery from crashes
        
        Args:
            df: Input DataFrame
            strategy_type: SLICE_STREAMING
            **kwargs: Additional parameters
            
        Returns:
            Aggregated result dictionary
        """
        self.logger.info(
            f"[PM] Slice streaming: {len(df)} rows in chunks of {self.processing_config.slice_size}"
        )

        # Create slices with overlap
        slices = self._create_slices(df)
        total_slices = len(slices)

        # Check for checkpoint recovery
        resume_from, is_recovery = await self._resume_from_checkpoint()
        start_slice_idx = resume_from if resume_from is not None else 0

        if is_recovery and self.connection_manager:
            await self._send_progress_update(
                int((start_slice_idx / total_slices) * 100),
                f"Recovered from checkpoint. Resuming from slice {start_slice_idx + 1}/{total_slices}"
            )

        # Process each slice
        all_results = []
        for slice_idx, slice_info in enumerate(slices[start_slice_idx:], start=start_slice_idx):
            try:
                # Check for cancellation
                if self.task_store and self.task_store.is_cancelled(self.task_id):
                    raise TaskCancelledException(f"Task {self.task_id} cancelled")

                slice_start, slice_end = slice_info["start"], slice_info["end"]
                slice_df = df.iloc[slice_start:slice_end].copy()

                self.logger.info(f"[PM] Processing slice {slice_idx + 1}/{total_slices} (rows {slice_start}-{slice_end})")

                # Save pending checkpoint BEFORE processing
                await self._save_pending_checkpoint(slice_idx)

                # Build context for this slice
                context = ProcessingContext(
                    task_id=self.task_id,
                    session_id=self.session_id,
                    analysis_type=self.analysis_type,
                    config=self.config,
                    task_store=self.task_store,
                    connection_manager=self.connection_manager,
                    processing_config=self.processing_config,
                    # Slice context
                    slice_num=slice_idx,
                    total_slices=total_slices,
                    slice_start=slice_start,
                    slice_end=slice_end,
                    total_dataset_rows=len(df),
                    global_offset=slice_start,
                )

                # Determine strategy for this slice
                slice_strategy_type = self._get_slice_processing_strategy(len(slice_df))

                # Execute slice with appropriate strategy
                strategy = StrategyFactory.create_strategy(
                    slice_strategy_type,
                    context,
                    self.logger
                )
                slice_result = await strategy.execute(slice_df)

                # Store slice results
                all_results.append(slice_result)

                # Save completed checkpoint AFTER results stored
                progress_pct = int(((slice_idx + 1) / total_slices) * 100)
                await self._save_completed_checkpoint(slice_idx, progress_pct)

                # Report progress
                await self._send_progress_update(
                    progress_pct,
                    f"Processed slice {slice_idx + 1}/{total_slices}"
                )

            except Exception as slice_err:
                self.logger.error(f"[PM] Slice {slice_idx} failed: {slice_err}", exc_info=True)
                # Don't fail entire task, log and continue
                # (or raise depending on error severity)
                raise

        # Aggregate results from all slices
        return self._aggregate_slice_results(all_results, df)

    def _create_slices(self, df: pd.DataFrame) -> list:
        """Create slice boundaries with overlap."""
        total_rows = len(df)
        slice_size = self.processing_config.slice_size
        overlap_factor = self.processing_config.slice_overlap_factor
        min_overlap = int(slice_size * overlap_factor)

        slices = []
        slice_idx = 0

        while slice_idx < total_rows:
            slice_start = max(0, slice_idx - min_overlap)
            slice_end = min(total_rows, slice_idx + slice_size)

            slices.append({
                "slice_num": len(slices),
                "start": slice_start,
                "end": slice_end,
                "size": slice_end - slice_start,
            })

            slice_idx = slice_end

        self.logger.info(f"[PM] Created {len(slices)} slices (size: {slice_size}, overlap: {overlap_factor})")
        return slices

    def _get_slice_processing_strategy(self, n_rows: int) -> ProcessingStrategy:
        """Determine strategy for individual slice processing."""
        if n_rows < self.processing_config.threshold_sequential_max:
            return ProcessingStrategy.SEQUENTIAL
        else:
            return ProcessingStrategy.PARALLEL_CHUNKING

    async def _save_pending_checkpoint(self, slice_num: int) -> None:
        """Mark slice as pending BEFORE processing."""
        try:
            async with AsyncPostgresSessionLocal() as db:
                stmt = (
                    insert(ChunkCheckpoint)
                    .values(
                        task_id=self.task_id,
                        session_id=self.session_id,
                        last_chunk_id=slice_num,
                        total_chunks=999,  # Will be updated when known
                        progress_pct=0,
                        status="processing",
                        created_at=datetime.now(timezone.utc),
                    )
                    .on_conflict_do_update(
                        index_elements=["task_id", "session_id"],
                        set_=dict(
                            last_chunk_id=slice_num,
                            status="processing",
                            created_at=datetime.now(timezone.utc),
                        )
                    )
                )
                await db.execute(stmt)
                await db.commit()
        except Exception as err:
            self.logger.warning(f"[PM] Failed to save pending checkpoint: {err}")

    async def _save_completed_checkpoint(self, slice_num: int, progress_pct: int) -> None:
        """Mark slice as successfully completed AFTER results persisted."""
        try:
            async with AsyncPostgresSessionLocal() as db:
                stmt = (
                    insert(ChunkCheckpoint)
                    .values(
                        task_id=self.task_id,
                        session_id=self.session_id,
                        last_chunk_id=slice_num,
                        total_chunks=999,
                        progress_pct=progress_pct,
                        status="completed",
                        created_at=datetime.now(timezone.utc),
                    )
                    .on_conflict_do_update(
                        index_elements=["task_id", "session_id"],
                        set_=dict(
                            last_chunk_id=slice_num,
                            progress_pct=progress_pct,
                            status="completed",
                            created_at=datetime.now(timezone.utc),
                        )
                    )
                )
                await db.execute(stmt)
                await db.commit()
        except Exception as err:
            self.logger.warning(f"[PM] Failed to save checkpoint: {err}")

    async def _resume_from_checkpoint(self) -> tuple:
        """Resume from last checkpoint with recovery detection."""
        try:
            async with AsyncPostgresSessionLocal() as db:
                stmt = (
                    select(ChunkCheckpoint)
                    .where(ChunkCheckpoint.task_id == self.task_id)
                    .order_by(desc(ChunkCheckpoint.created_at))
                    .limit(1)
                )
                result = await db.execute(stmt)
                ckpt = result.scalars().first()

                if ckpt:
                    # Check if last checkpoint was "processing" (incomplete)
                    if ckpt.status == "processing":
                        # Recovery needed
                        resume_from = ckpt.last_chunk_id
                        logger.info(f"[PM] Recovery detected for task {self.task_id}, resuming from slice {resume_from}")
                        return resume_from, True
                    else:
                        # Already completed
                        return None, False

                return None, False

        except Exception as err:
            self.logger.warning(f"[PM] Failed to resume from checkpoint: {err}")
            return None, False

    async def _persist_to_database(self, result: Dict[str, Any]) -> None:
        """Persist result to database (TIER 3)."""
        try:
            if "result_df" in result:
                df_to_store = result["result_df"]
            else:
                # For non-dataframe results (ML splits, etc.)
                return

            await store_session_step_result(
                session_id=self.session_id,
                step_name=self.step_name,
                result=df_to_store,
                serialization_format="pickle",  # For DataFrames with numpy
            )

            self.logger.info(f"[PM] Persisted {self.step_name} to database")

        except Exception as err:
            self.logger.error(f"[PM] Database persistence failed: {err}", exc_info=True)
            # Don't fail the entire pipeline, just log

    def _aggregate_slice_results(self, all_results: list, original_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Aggregate results from all slices into a single result.
        
        Strategy:
        - Concatenate result_df from all slices (with overlap removal)
        - Merge metadata
        - Aggregate signals/zones if present
        """
        if not all_results:
            return {"result_df": original_df, "metadata": {"slices": 0}}

        # Combine DataFrames (removing overlap)
        all_dfs = [r.get("result_df", pd.DataFrame()) for r in all_results]
        combined_df = pd.concat(all_dfs, ignore_index=True).drop_duplicates(subset=None, keep="first")

        # Merge metadata
        aggregated_metadata = {
            "strategy": "slice_streaming",
            "analysis_type": self.analysis_type,
            "slices": len(all_results),
            "total_rows_processed": len(combined_df),
        }

        # Aggregate signals if present
        aggregated_signals = []
        for result in all_results:
            if "signals" in result:
                aggregated_signals.extend(result.get("signals", []))

        # Build final result
        final_result = {
            "result_df": combined_df,
            "metadata": aggregated_metadata,
        }

        # Include other result keys
        for key in ["signals", "zones", "enriched_df", "ml_dataset", "signal_counts", "train", "validation", "test"]:
            if key in all_results[0]:
                if key == "signals":
                    final_result[key] = aggregated_signals
                else:
                    # For other keys, just use first result (or aggregate as needed)
                    final_result[key] = all_results[0][key]

        return final_result

    async def _send_progress_update(self, progress: int, message: str) -> None:
        """Send progress update via task_store and WebSocket."""
        try:
            if self.task_store:
                self.task_store.update_task(
                    self.task_id,
                    progress=progress,
                    message=message,
                    stage=self.current_stage,
                )

            if self.connection_manager:
                await self.connection_manager.send_progress_update(
                    self.task_id,
                    {
                        "type": "progress",
                        "progress": progress,
                        "message": message,
                        "stage": self.current_stage,
                    }
                )
        except Exception as err:
            self.logger.warning(f"[PM] Progress update failed: {err}")

    async def _send_error(self, error_message: str) -> None:
        """Send error message via WebSocket."""
        try:
            if self.connection_manager:
                await self.connection_manager.send_progress_update(
                    self.task_id,
                    {
                        "type": "error",
                        "status": "failed",
                        "message": error_message,
                    }
                )
        except Exception as err:
            self.logger.warning(f"[PM] Error notification failed: {err}")


# ============================================================================
# EXPORT FOR BACKWARD COMPATIBILITY
# ============================================================================

# Ensure old imports still work
__all__ = [
    "ProcessingManager",
    "ProcessingContext",
    "ProcessingStrategy",
    "StrategyFactory",
    "HandlerRegistry",
    "IntermediateResultsCache",
    "CachedStepData",
]
