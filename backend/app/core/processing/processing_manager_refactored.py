"""
ProcessingManager (Refactored) - Unified Orchestrator for All Analysis Types

This refactored version uses modular strategies and handlers to process:
- Technical Analysis
- SNR Analysis
- Astronomical Analysis
- ML Dataset Preparation
- Model Building
- Model Training

Key Improvements:
1. Modular Strategies: Reusable Sequential/Parallel/SliceStreaming implementations
2. Handler Registry: Easy to add new analysis types without modifying core logic
3. Uniform Data Flow: All analysis types follow the same pipeline
4. Consistent Storage: Automatic DB persistence and cache management
5. Progress Tracking: Built-in WebSocket and task_store integration

Architecture:
    Route (analysis.py, model_builder.py)
        ↓
    ProcessingManager.__init__() [Config + Context]
        ↓
    ProcessingManager.execute(df)
        ├─ StrategyFactory.determine_strategy() [Auto-select based on size]
        ├─ StrategyFactory.create_strategy() [Instantiate strategy]
        └─ strategy.execute(df)
            ├─ HandlerRegistry.get(analysis_type) [Get handler]
            └─ handler(df, context) [Execute analysis]
                ↓
    Result (enriched_df + metadata)
        ↓
    AnalysisManager (Storage + Cache)
        ├─ store_session_step_result() [PostgreSQL]
        ├─ cache_session_data() [Memory cache]
        └─ self.current_data = enriched_df [TIER 0 pointer]
"""

import gc
import logging
from typing import Dict, Optional, Any, Set
from datetime import datetime

import pandas as pd

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
from app.core.config import (
    TechnicalConfig,
    SNRConfig,
    AstronomicalConfig,
    ProcessingConfig,
)
from app.core.processing.tasks import TaskStore, TaskCancelledException
from app.core.services.multiprocessing_config import init_spawn_method
from app.api.routes.data.database import AsyncPostgresSessionLocal
from app.database.models import ChunkCheckpoint
from app.core.data.session_data_loader import store_session_step_result
from sqlalchemy.dialects.postgresql import insert as pg_insert


logger = logging.getLogger(__name__)


# ============================================================================
# REGISTER ALL HANDLERS AT MODULE LOAD
# ============================================================================

try:
    HandlerRegistry.register("technical", analyze_technical_impl, TechnicalConfig)
    HandlerRegistry.register("snr", analyze_snr_impl, SNRConfig)
    HandlerRegistry.register("astronomical", analyze_astronomical_impl, AstronomicalConfig)
    HandlerRegistry.register("ml_preparation", analyze_ml_prep_impl, dict)  # Uses DatasetConfig
    HandlerRegistry.register("model_training", analyze_model_training_impl, dict)
    HandlerRegistry.register("model_build", analyze_model_build_impl, dict)
    logger.info("✅ All processing handlers registered")
except Exception as reg_err:
    logger.error(f"Failed to register handlers: {reg_err}")


# ============================================================================
# PROCESSING MANAGER (REFACTORED)
# ============================================================================

class ProcessingManager:
    """
    Unified orchestrator for all analysis types with intelligent strategy selection.
    
    Responsibilities:
    1. Strategy Selection: Auto-select based on data size
    2. Context Building: Create ProcessingContext with all required parameters
    3. Execution: Delegate to strategy → handler
    4. Storage: Persist results to DB and cache
    5. Progress: Track and broadcast via WebSocket
    
    Usage:
        # For SNR Analysis
        pm = ProcessingManager(
            session_id=session_id,
            task_id=task_id,
            analysis_type="snr",
            config=SNRConfig(...),
            task_store=task_store,
            connection_manager=manager,
        )
        result = await pm.execute(df)
        
        # For ML Preparation
        pm = ProcessingManager(
            session_id=session_id,
            task_id=task_id,
            analysis_type="ml_preparation",
            config=DatasetConfig(...),
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
        """
        Initialize ProcessingManager.
        
        Args:
            session_id: Session UUID
            task_id: Task UUID
            analysis_type: Analysis type identifier (e.g., "technical", "snr", "ml_preparation")
            config: Analysis-specific configuration object
            task_store: Task store for progress tracking
            connection_manager: WebSocket manager for progress updates
            processing_config: Processing configuration (thresholds, slice size, etc.)
            step_name: Optional step name for DB storage (defaults to f"{analysis_type}_analysis")
        """
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
        Main entry point. Auto-selects strategy and executes processing.
        
        Args:
            df: Input DataFrame
            **kwargs: Additional parameters (user_id, etc.)
            
        Returns:
            Result dictionary with enriched data and metadata
        """
        n_rows = len(df)
        self.total_rows = n_rows
        self.rows_processed = 0

        # Determine strategy based on data size
        strategy_type = StrategyFactory.determine_strategy(
            n_rows, self.processing_config, self.analysis_type
        )

        await self._send_initial_progress_message(strategy_type)

        # Execute with selected strategy
        if strategy_type == ProcessingStrategy.SLICE_STREAMING:
            return await self._execute_with_slicing(df, strategy_type, **kwargs)
        else:
            return await self._execute_with_strategy(strategy_type, df, **kwargs)

    async def _send_initial_progress_message(self, strategy_type: ProcessingStrategy):
        """Send initial progress message to user."""
        if not self.connection_manager:
            return
        
        try:
            message = self._get_strategy_user_message(strategy_type, "starting", {})
            await self.connection_manager.send_progress_update(
                self.task_id,
                {
                    "type": "progress",
                    "progress": 5,
                    "stage": "starting",
                    "message": message
                }
            )
        except Exception as ws_err:
            self.logger.warning(f"Initial progress update failed: {ws_err}")

    async def _execute_with_strategy(
        self,
        strategy_type: ProcessingStrategy,
        df: pd.DataFrame,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute with sequential or parallel chunking strategy.
        
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

        # Send completion message
        if self.connection_manager:
            try:
                message = self._get_strategy_user_message(strategy_type, "complete", {})
                await self.connection_manager.send_progress_update(
                    self.task_id,
                    {
                        "type": "progress",
                        "progress": 100,
                        "stage": "complete",
                        "message": message
                    }
                )
            except Exception as ws_err:
                self.logger.warning(f"Completion message failed: {ws_err}")

        return result

    async def _execute_with_slicing(
        self,
        df: pd.DataFrame,
        strategy_type: ProcessingStrategy,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute with slice streaming strategy for large datasets.
        
        Handles:
        1. Slice creation with overlap
        2. Per-slice processing with checkpoint management
        3. Result aggregation
        4. Progress tracking
        
        Args:
            df: Input DataFrame
            strategy_type: SLICE_STREAMING
            **kwargs: Additional parameters
            
        Returns:
            Aggregated result dictionary
        """
        self.logger.info(
            f"[SliceStreaming] Starting {len(df)} rows in slices of {self.processing_config.slice_size}"
        )

        # Send slicing initialization message
        if self.connection_manager:
            try:
                slices_count = max(1, -(-len(df) // self.processing_config.slice_size))
                message = self._get_strategy_user_message(
                    strategy_type, "starting", {"slices": slices_count}
                )
                await self.connection_manager.send_progress_update(
                    self.task_id,
                    {
                        "type": "progress",
                        "progress": 5,
                        "stage": "slicing_init",
                        "message": message
                    }
                )
            except Exception as ws_err:
                self.logger.warning(f"Slicing init message failed: {ws_err}")

        # Create slices
        slices = self._create_slices(df)
        total_slices = len(slices)

        # Initialize result containers
        all_results = []

        # Get resume point
        resume_from, is_recovery = await self._resume_from_checkpoint()
        start_slice_idx = resume_from if resume_from is not None else 0

        # Notify user of recovery
        if is_recovery and self.connection_manager:
            try:
                message = f"Recovering from previous crash. Resuming from slice {start_slice_idx + 1}/{total_slices}"
                await self.connection_manager.send_progress_update(
                    self.task_id,
                    {
                        "type": "recovery",
                        "recovery": True,
                        "resume_from": start_slice_idx,
                        "total_slices": total_slices,
                        "message": message,
                    }
                )
            except Exception as ws_err:
                self.logger.warning(f"Failed to send recovery notification: {ws_err}")

        # Process each slice
        for slice_info in slices[start_slice_idx:]:
            slice_num, start, end, overlap_start = slice_info

            try:
                # Check for cancellation
                if self.task_store and hasattr(self.task_store, "is_cancelled"):
                    if self.task_store.is_cancelled(self.task_id):
                        raise TaskCancelledException(f"Task cancelled at slice {slice_num}")

                # Mark as pending
                await self._save_pending_checkpoint(slice_num)

                # Notify user processing started
                if self.connection_manager:
                    try:
                        message = self._get_strategy_user_message(
                            ProcessingStrategy.SLICE_STREAMING,
                            "slice_starting",
                            {"slice_num": slice_num + 1, "total_slices": total_slices}
                        )
                        rows_per_slice = len(df) // total_slices
                        rows_processed = slice_num * rows_per_slice
                        progress_pct = int((rows_processed / len(df)) * 100) if len(df) > 0 else 0

                        await self.connection_manager.send_progress_update(
                            self.task_id,
                            {
                                "type": "progress",
                                "progress": progress_pct,
                                "stage": "slice_processing",
                                "slice": f"{slice_num + 1}/{total_slices}",
                                "message": message,
                            },
                        )
                    except Exception as ws_err:
                        self.logger.warning(f"Failed to send processing notification: {ws_err}")

                # Process slice
                df_slice = df.iloc[overlap_start:end].copy()

                # Build context for this slice
                context = ProcessingContext(
                    task_id=self.task_id,
                    session_id=self.session_id,
                    analysis_type=self.analysis_type,
                    config=self.config,
                    task_store=self.task_store,
                    connection_manager=self.connection_manager,
                    processing_config=self.processing_config,
                    slice_num=slice_num,
                    total_slices=total_slices,
                    slice_start=start,
                    slice_end=end,
                    total_dataset_rows=len(df),
                    global_offset=overlap_start,
                )

                # Determine strategy for this slice
                slice_strategy_type = self._get_slice_processing_strategy(len(df_slice))

                # Create strategy instance
                strategy = StrategyFactory.create_strategy(
                    slice_strategy_type,
                    context,
                    self.logger
                )

                # Execute strategy
                results = await strategy.execute(df_slice)

                # Store slice results
                await self._store_slice_results(slice_num, results, (start, end), all_results)

                # Mark as completed
                rows_processed_after_this_slice = (slice_num + 1) * rows_per_slice
                progress_pct = int((rows_processed_after_this_slice / len(df)) * 100) if len(df) > 0 else 0
                progress_pct = min(progress_pct, 99)
                await self._save_completed_checkpoint(slice_num, progress_pct)

                # Report progress
                await self._report_progress(slice_num, total_slices, len(all_results))

                # Cleanup
                del df_slice, results
                gc.collect()

            except Exception as slice_err:
                error_str = str(slice_err)
                self.logger.error(
                    f"[SliceProcessing] Error processing slice {slice_num}: {error_str}",
                    exc_info=True
                )

                # Save error state
                try:
                    async with AsyncPostgresSessionLocal() as db:
                        stmt_err = pg_insert(ChunkCheckpoint).values(
                            task_id=self.task_id,
                            session_id=self.session_id,
                            step_name=self.analysis_type,
                            last_failure_reason=error_str[:500],
                        ).on_conflict_do_update(
                            index_elements=["task_id"],
                            set_=dict(last_failure_reason=error_str[:500]),
                        )
                        await db.execute(stmt_err)
                        await db.commit()
                except Exception as state_err:
                    self.logger.warning(f"Failed to save error state: {state_err}")

                # Continue to next slice
                continue

        # Store chunked metadata
        if self.session_id:
            try:
                async with AsyncPostgresSessionLocal() as db:
                    metadata = {
                        "_is_chunked": True,
                        "total_rows": len(df),
                        "chunk_size": self.processing_config.slice_size,
                        "num_chunks": total_slices,
                        "step_name": self.step_name,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    await store_session_step_result(
                        session_id=self.session_id,
                        step_name=f"{self.step_name}_header",
                        data=metadata,
                        db=db,
                        is_compressed=False
                    )
            except Exception as meta_err:
                self.logger.warning(f"Failed to store chunked metadata: {meta_err}")

        # Aggregate results
        return self._aggregate_slice_results(all_results, df)

    def _create_slices(self, df: pd.DataFrame) -> list:
        """Create slice boundaries with overlap."""
        total_rows = len(df)
        slice_size = self.processing_config.slice_size

        # Calculate minimum overlap
        min_overlap = int(slice_size * self.processing_config.slice_overlap_factor)

        total_slices = max(1, -(-total_rows // slice_size))
        slices = []

        for slice_idx in range(total_slices):
            start = slice_idx * slice_size
            end = min(start + slice_size, total_rows)
            overlap_start = max(0, start - min_overlap)
            slices.append((slice_idx, start, end, overlap_start))

        self.logger.info(f"[SliceStreaming] Created {len(slices)} slices")
        return slices

    def _get_slice_processing_strategy(self, n_rows: int) -> ProcessingStrategy:
        """Determine strategy for processing individual slices."""
        if n_rows < self.processing_config.threshold_sequential_max:
            return ProcessingStrategy.SEQUENTIAL
        else:
            return ProcessingStrategy.PARALLEL_CHUNKING

    async def _store_slice_results(
        self,
        slice_num: int,
        results: Dict[str, Any],
        slice_boundaries: tuple,
        all_results: list,
    ):
        """Store and aggregate slice results."""
        slice_start, slice_end = slice_boundaries

        # Aggregate results
        all_results.append(results)

        # Store to DB
        try:
            async with AsyncPostgresSessionLocal() as db:
                await store_session_step_result(
                    session_id=self.session_id,
                    step_name=f"{self.step_name}_{slice_num}",
                    data={
                        "slice_num": slice_num,
                        "slice_start": slice_start,
                        "slice_end": slice_end,
                        "metadata": results.get("metadata", {}),
                    },
                    db=db,
                    force_pickle=True
                )
        except Exception as db_err:
            self.logger.warning(f"Could not persist slice {slice_num}: {db_err}")

    async def _save_pending_checkpoint(self, slice_num: int):
        """Mark slice as pending BEFORE processing."""
        try:
            async with AsyncPostgresSessionLocal() as db:
                stmt = pg_insert(ChunkCheckpoint).values(
                    task_id=self.task_id,
                    session_id=self.session_id,
                    step_name=self.analysis_type,
                    last_pending_chunk_id=slice_num,
                    processing_state="pending",
                    failure_count=0,
                    last_failure_time=datetime.now(timezone.utc),
                ).on_conflict_do_update(
                    index_elements=["task_id"],
                    set_={
                        "last_pending_chunk_id": slice_num,
                        "processing_state": "pending",
                        "last_failure_time": datetime.now(timezone.utc),
                    },
                )
                await db.execute(stmt)
                await db.commit()
        except Exception as ckpt_err:
            self.logger.error(f"Failed to mark slice {slice_num} pending: {ckpt_err}")

    async def _save_completed_checkpoint(self, slice_num: int, progress_pct: int):
        """Mark slice as successfully completed AFTER results persisted."""
        try:
            async with AsyncPostgresSessionLocal() as db:
                stmt = pg_insert(ChunkCheckpoint).values(
                    task_id=self.task_id,
                    session_id=self.session_id,
                    step_name=self.analysis_type,
                    last_successful_chunk_id=slice_num,
                    processing_state="completed",
                    failure_count=0,
                    last_pending_chunk_id=None,
                    progress_percentage=progress_pct,
                    last_checkpoint_time=datetime.now(timezone.utc),
                ).on_conflict_do_update(
                    index_elements=["task_id"],
                    set_={
                        "last_successful_chunk_id": slice_num,
                        "processing_state": "completed",
                        "failure_count": 0,
                        "last_pending_chunk_id": None,
                        "progress_percentage": progress_pct,
                        "last_checkpoint_time": datetime.now(timezone.utc),
                    },
                )
                await db.execute(stmt)
                await db.commit()
        except Exception as ckpt_err:
            self.logger.error(f"Failed to mark slice {slice_num} completed: {ckpt_err}")

    async def _resume_from_checkpoint(self) -> tuple:
        """Resume from last checkpoint with recovery detection."""
        try:
            from sqlalchemy import select, and_
            async with AsyncPostgresSessionLocal() as db:
                stmt_res = (
                    select(ChunkCheckpoint)
                    .where(
                        and_(
                            ChunkCheckpoint.task_id == self.task_id,
                            ChunkCheckpoint.session_id == self.session_id,
                        )
                    )
                    .order_by(ChunkCheckpoint.last_checkpoint_time.desc())
                    .limit(1)
                )
                result = await db.execute(stmt_res)
                checkpoint = result.scalar_one_or_none()

            if not checkpoint:
                return None, False

            is_recovery = checkpoint.processing_state == "pending"

            if is_recovery:
                self.logger.warning(
                    f"RECOVERY MODE: Task {self.task_id} crashed during slice {checkpoint.last_pending_chunk_id}."
                )

                if checkpoint.failure_count >= 3:
                    self.logger.error(
                        f"Slice {checkpoint.last_pending_chunk_id} has failed {checkpoint.failure_count} times. Skipping."
                    )
                    resume_from = checkpoint.last_pending_chunk_id + 1
                else:
                    resume_from = checkpoint.last_pending_chunk_id
            else:
                resume_from = checkpoint.last_successful_chunk_id + 1 if checkpoint.last_successful_chunk_id is not None else 0

            return resume_from, is_recovery
        except Exception as resume_err:
            self.logger.warning(f"Could not resume from checkpoint: {resume_err}")
            return None, False

    async def _report_progress(self, current_slice: int, total_slices: int, result_count: int):
        """Report progress via WebSocket."""
        if not self.connection_manager:
            return

        progress_pct = int((current_slice + 1) / total_slices * 100)

        user_message = self._get_strategy_user_message(
            ProcessingStrategy.SLICE_STREAMING,
            "slice_complete",
            {
                "slice_num": current_slice + 1,
                "total_slices": total_slices,
                "results": result_count,
            },
        )

        try:
            await self.connection_manager.send_progress_update(
                self.task_id,
                {
                    "type": "progress",
                    "progress": progress_pct,
                    "stage": "slice_complete",
                    "slice": f"{current_slice + 1}/{total_slices}",
                    "message": user_message,
                },
            )
        except Exception as ws_err:
            self.logger.warning(f"Progress update failed: {ws_err}")

    def _aggregate_slice_results(self, all_results: list, original_df: pd.DataFrame) -> Dict[str, Any]:
        """Aggregate results from all slices."""
        if not all_results:
            return {
                "success": False,
                "metadata": {
                    "strategy": ProcessingStrategy.SLICE_STREAMING.value,
                    "total_rows": len(original_df),
                    "analysis_type": self.analysis_type,
                },
            }

        # Merge results based on analysis type
        # For now, return first result (handler-specific aggregation needed)
        return {
            "success": True,
            "slices_processed": len(all_results),
            "metadata": {
                "strategy": ProcessingStrategy.SLICE_STREAMING.value,
                "total_rows": len(original_df),
                "analysis_type": self.analysis_type,
            },
        }

    def _get_strategy_user_message(self, strategy: ProcessingStrategy, stage: str, details: Dict) -> str:
        """Generate user-friendly messages based on strategy and stage."""
        if strategy == ProcessingStrategy.SEQUENTIAL:
            if stage == "starting":
                return f"Processing {self.total_rows} rows sequentially"
            elif stage == "complete":
                return f"Sequential analysis complete"

        elif strategy == ProcessingStrategy.PARALLEL_CHUNKING:
            if stage == "starting":
                return f"Distributing data across processors"
            elif stage == "complete":
                return f"Parallel processing complete"

        elif strategy == ProcessingStrategy.SLICE_STREAMING:
            if stage == "starting":
                slices = details.get("slices", "?")
                return f"Processing {self.total_rows} rows in {slices} memory-efficient slices"
            elif stage == "slice_starting":
                slice_num = details.get("slice_num", "?")
                total_slices = details.get("total_slices", "?")
                return f"Processing slice {slice_num}/{total_slices}"
            elif stage == "slice_complete":
                slice_num = details.get("slice_num", "?")
                return f"Slice {slice_num} complete"
            elif stage == "complete":
                return f"Large dataset analysis complete"

        return f"Processing {self.analysis_type} data"
