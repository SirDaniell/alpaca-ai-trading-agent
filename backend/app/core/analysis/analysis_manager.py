"""
AnalysisManager: Unified orchestrator for data fetching and analysis pipeline.

Consolidates the entire workflow:
  1. Data Fetch (MT5/Database/CSV) → Extract dates
  2. Session Creation (with complete metadata)
  3. Analysis Steps (Technical/SNR/Astronomical)
  4. Progress tracking + WebSocket updates

Usage:
    manager = AnalysisManager(
        session_id='uuid',
        task_id='task-uuid',
        config=config,
        task_store=task_store,
        connection_manager=ws_manager,
    )
    result = await manager.execute(request)

"""

from app.core.data.serializers import to_serializable, serialize_data, deserialize_data
import gc
import logging
import uuid
import hashlib
import base64
import time
import math
import asyncio
import platform
import os
import sys
from datetime import datetime, timezone
import tensorflow as tf
import numpy as np
import psutil
from typing import Dict, Optional, Any, List, Tuple, Union, Mapping
from dataclasses import dataclass, asdict
from collections import deque
import pandas as pd
from app.core.ml.ml_dataset_preparation import MLDatasetPreparation
from app.core.ml.model_registry import ModelRegistry
from app.core.ml.persistent_model_store import PersistentModelStore
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.core.services.data_fetcher import DataFetcher
from app.core.processing.processing_manager import ProcessingManager, AnalysisType, IntermediateResultsCache
from app.core.data.session_data_loader import set_as_current_data, store_session_step_result
from app.core.processing.tasks import TaskStore
from app.core.services.websocket_manager import ConnectionManager
from app.database.models import (
    DataSession, 
    SessionStepResult,
    CompiledModel,
    TrainingRecord,
    ModelSelectionHint,
    TrainedModelForAnalysis,
    MLDatasetChunk,
    ModelPredictions,
)
from app.api.routes.data.database import AsyncPostgresSessionLocal
from app.services.mt5_service import MT5Service
from app.database.models import MLDatasetChunk
from app.database.connection import DbConfig
from app.core.services import data_utils
from app.core.config import (
    TechnicalConfig, 
    SNRConfig, 
    AstronomicalConfig, 
    MLDatasetConfig,
    ModelBuildConfig,
    ModelTrainingConfig,
    ProcessingConfig
)


logger = logging.getLogger(__name__)

from app.core.processing.progress_reporter import ProgressReporter, ThrottlingStrategy
from app.core.services.decompress_cache import get_cache as get_decompress_cache
import app.core.processing.processing_manager as processing_manager_module
from app.core.data.session_dataset_registry import SessionDatasetRegistry
from app.core.data.session_data_loader import set_as_current_data
from app.api.routes.data.database import postgres_engine

from app.core.analysis.analysis_manager_utils import (ErrorCategory, 
                                                      ErrorContext, 
                                                      DataSourceRequest, 
                                                      AnalysisRequest, 
                                                      SessionDataCache, 
                                                      SessionMetadataBuilder)
                                         


# ============================================================================
# ANALYSIS MANAGER
# ============================================================================

from app.core.analysis.analysis_manager_analysis_mixin import AnalysisExecutionMixin
from app.core.analysis.analysis_manager_ml_mixin import MLPipelineMixin
from app.core.analysis.analysis_manager_training_mixin import TrainingEngineMixin


class AnalysisManager(AnalysisExecutionMixin, MLPipelineMixin, TrainingEngineMixin):
    """
    Unified orchestrator for data fetch + analysis pipeline.
    
    Handles:
    - Data fetching from any source with date extraction
    - Session creation with complete metadata
    - ProcessingManager delegation for analysis
    - Progress tracking via WebSocket
    - Memory cleanup
    """

    def _normalise_step_config(self, config: Any) -> Dict[str, Any]:
        if config is None:
            return {}
        if isinstance(config, dict):
            return to_serializable(config)
        if hasattr(config, "model_dump"):
            return to_serializable(config.model_dump())
        if hasattr(config, "dict"):
            return to_serializable(config.dict())
        if hasattr(config, "__dict__"):
            return to_serializable({
                key: value
                for key, value in vars(config).items()
                if not key.startswith("_")
            })
        return {"value": to_serializable(config)}

    async def _persist_step_config(
        self,
        session_id: str,
        step_key: str,
        config: Any,
        task_id: str,
    ) -> None:
        config_payload = self._normalise_step_config(config)
        # Unwrap nested 'config' or 'step_config' keys if present
        if isinstance(config_payload, dict) and len(config_payload) == 1:
            for k in ("config", "step_config", "parameters", "step_configs"):
                if k in config_payload and isinstance(config_payload[k], dict):
                    config_payload = config_payload[k]
                    break

        if not isinstance(config_payload, dict) or not config_payload:
            logger.debug("[AM] Skipping empty config snapshot for %s/%s", session_id, step_key)
            return

        payload = {
            "step_key": step_key,
            "task_id": task_id,
            "stored_at": datetime.utcnow().isoformat(),
            **config_payload,
        }
        try:
            async with AsyncPostgresSessionLocal() as db:
                await store_session_step_result(
                    session_id=session_id,
                    step_name=f"{step_key}_config",
                    data=payload,
                    db=db,
                    is_compressed=False,
                    force_pickle=False,
                )
            logger.info("[AM] Stored config snapshot (%d keys) for %s/%s", len(config_payload), session_id, step_key)
        except Exception as error:
            logger.warning("[AM] Could not store config snapshot for %s: %s", step_key, error)
    
    # ────────────────────────────────────────────────────────────────
    # CLASS-LEVEL CACHE (Session-persistent, single source of truth)
    # ────────────────────────────────────────────────────────────────
    # Format: {session_id: SessionDataCache}
    _session_cache: Dict[str, SessionDataCache] = {}
    _cache_locks: Dict[str, asyncio.Lock] = {}
    # Session-scoped TIER 0 pointers shared by manager instances in this process.
    _session_data_pointers: Dict[str, pd.DataFrame] = {}
    
    def __init__(
        self,
        task_store: Optional[TaskStore] = None,
        connection_manager: Optional[ConnectionManager] = None,
        mt5_service: Optional[MT5Service] = None,
        db_config: Optional[DbConfig] = None,
    ):
        """
        Initialize AnalysisManager as a global singleton.
        
        - task_store: Progress tracking (shared)
        - connection_manager: WebSocket manager (shared)
        - mt5_service: MT5 connection (shared)
        - db_config: Database config (shared)
        - current_data: Tracks current DataFrame during processing (per-call, reset each execution)
        
        ❌ NOT per-instance (per-call parameters):
        - session_id: Passed to each method call
        - task_id: Passed to each method call
        
        This allows one AnalysisManager instance to handle unlimited
        concurrent sessions with different session/task IDs.
        
        Args:
            task_store: TaskStore singleton for progress tracking
            connection_manager: WebSocket manager for progress updates
            mt5_service: MT5Service for fetching MT5 data
            db_config: Database config for DB queries
        """
        self.task_store = task_store
        self.connection_manager = connection_manager
        self.user_id = "anonymous"
        self.logger = logger
        
        # ✅ NEW: Track current DataFrame during processing
        # This pointer is updated during each analysis step but NOT stored per-session
        # It enables internal helper methods to access the active DataFrame without passing it everywhere
        self.current_data: Optional[pd.DataFrame] = None
        self.current_session_id: Optional[str] = None  # Tracks which session owns current_data
        
        # ✅ NEW: Active Analysis Tracking (for self-probing)
        # Tracks currently running analyses to prevent duplicates and provide status
        self.active_analyses: Dict[str, Dict[str, Any]] = {}  # {task_id: analysis_info}
        self.analysis_lock = asyncio.Lock()  # Protects active_analyses dict
        
        # ✅ NEW: Per-session busy flags (FAST REJECTION at entry point)
        # Format: {(session_id, analysis_type): True/False}
        # This provides O(1) lookup for quick rejection before expensive operations
        self.session_busy: Dict[Tuple[str, str], bool] = {}
        self.busy_lock = asyncio.Lock()  # Protects session_busy dict
        
        # TIER 0b: ML split pointers (NEW - for training pipeline)
        # Set by execute_ml_preparation() when ML splits are generated
        # Used by execute_model_training() to load training/validation/test data (ZERO LATENCY)
        # Cleared by cascade_clear_data_pointers() after ML prep creates splits
        self.ml_train: Optional[pd.DataFrame] = None
        self.ml_validation: Optional[pd.DataFrame] = None
        self.ml_test: Optional[pd.DataFrame] = None
        self.ml_session_id: Optional[str] = None  # Tracks which session owns ML pointers
        self.ml_dataset_name: Optional[str] = None  # ✅ NEW: Tracks which dataset these pointers correspond to
        
        # TIER 0c: Built model instance (CACHED - for training pipeline)
        # Set by execute_model_build() when model is compiled
        # Used by execute_model_training() to access the built model (ZERO DISK I/O)
        # Cleared after training completes or when new build happens
        self.model = None  # In-memory model instance (TensorFlow model)
        self.model_id: Optional[str] = None  # Model ID for persistence
        self.model_session_id: Optional[str] = None  # Session this model was built for
        
        # ✅ SNR Unprocessed Dataset (for direct ML prep)
        self.unprocessed_dataset: Optional[List[Dict]] = None
        self.unprocessed_session_id: Optional[str] = None
        
        # Initialize data fetcher (shared)
        self.data_fetcher = DataFetcher(
            mt5_service=mt5_service,
            db_config=db_config,
        )
        
        # Initialize persistent model store (shared)
        self.persistent_store = PersistentModelStore()
        
        logger.info(f"✅ AnalysisManager singleton initialized (shared infrastructure)")
    
    def generate_session_id(self) -> str:
        """Generate a unique session ID (UUID)."""
        return str(uuid.uuid4())

    async def build_inference_feature_window(
        self,
        feature_window: pd.DataFrame,
        supporting_ohlcv: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        model_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ) -> Tuple[Any, float]:
        """
        Helper to build an inference-ready feature tensor using the project's
        canonical InferenceFeaturePipeline while allowing AnalysisManager to
        centralise any future session/step-config lookups or prefetch logic.

        This is intentionally a thin facade for now — it delegates to
        InferenceFeaturePipeline.get_or_create and .build_feature_window but
        gives a single callsite for the rest of the codebase to use.

        Args:
            feature_window: OHLCV DataFrame (full window, not sliced)
            supporting_ohlcv: Optional dict of supporting pairs OHLCV
            model_id: Optional model identifier (for logging)
            dataset_name: Optional dataset name used by the pipeline cache

        Returns:
            Tuple of (tensor, reference_close)
        """
        from app.core.ml.inference_feature_pipeline import InferenceFeaturePipeline

        if feature_window is None or (hasattr(feature_window, 'empty') and feature_window.empty):
            raise ValueError("feature_window must be a non-empty DataFrame")

        pipeline_key = f"{model_id}:{dataset_name or 'default'}"
        try:
            pipeline = await InferenceFeaturePipeline.get_or_create(
                dataset_name=(dataset_name or "default"),
                feature_map_path=None,
                scaler_path=None,
            )
        except Exception:
            # Best-effort: fall back to directly creating the pipeline without cache
            pipeline = await InferenceFeaturePipeline.get_or_create(
                dataset_name=(dataset_name or "default"),
                feature_map_path=None,
                scaler_path=None,
            )

        logger.info(
            "[AM] Building inference feature window (model=%s dataset=%s rows=%s cols=%s supporting=%s)",
            model_id,
            dataset_name,
            getattr(feature_window, 'shape', None),
            list(feature_window.columns)[:8] if hasattr(feature_window, 'columns') else None,
            list((supporting_ohlcv or {}).keys())[:8],
        )

        tensor, reference_close = await pipeline.build_feature_window(feature_window, supporting_ohlcv or {})
        logger.info("[AM] Inference feature window built: tensor_shape=%s", getattr(tensor, 'shape', None))
        return tensor, reference_close
    
    async def execute(
        self,
        request: AnalysisRequest,
        session_id: str,
        task_id: str,
    ) -> Dict[str, Any]:
        """
        Execute complete pipeline: fetch data → create session → run analysis.
        
        ✅ session_id and task_id are PER-CALL (not instance state)
        This allows the singleton AnalysisManager to handle unlimited concurrent requests.
        
        Args:
            request: AnalysisRequest with data source + optional analysis steps
            session_id: Per-call session UUID
            task_id: Per-call task UUID
        Args:
            request: AnalysisRequest with data source + optional analysis steps
            session_id: Per-call session UUID
            task_id: Per-call task UUID
            
        Returns:
            Result dict with session_id, preview, analysis results
        """
        
        # Initialize memory monitoring
        process = psutil.Process()
        mem_start = process.memory_info().rss / 1024 / 1024
        
        try:
            user_id = request.user_id or "anonymous"
            
            # Initialize ProgressReporter
            reporter = ProgressReporter(
                task_id=task_id,
                connection_manager=self.connection_manager,
                user_id=user_id
            )
            
            # ─────────────────────────────────────────────────────────
            # STEP 1: Fetch Data (with dates extracted)
            # ─────────────────────────────────────────────────────────
            logger.info(f"📥 Step 1: Fetching data (task {task_id[:8]}...)...")
            
            # Send initial progress
            await reporter.report_async(
                progress=5,
                message="Initializing",
                message2=f"Preparing to fetch data from {request.data_source.source.upper()}..."
            )

            fetched_df, start_date, end_date, record_count = await self._fetch_data(
                request.data_source,
                reporter=reporter
            )
            
            if process:
                mem_after_fetch = process.memory_info().rss / 1024 / 1024
                logger.info(
                    f"[Pipeline] After fetch: {mem_after_fetch:.1f} MB "
                    f"(+{mem_after_fetch - mem_start:.1f} MB)"
                )
            
            # ─────────────────────────────────────────────────────────
            # STEP 2: Check for duplicates BEFORE creating session (if not skipped)
            # ─────────────────────────────────────────────────────────
            existing_session = None
            if not request.skip_duplicate_check:
                logger.info("🔍 Step 2a: Checking for duplicate sessions...")
                existing_session = await self._check_duplicate_session(
                    symbol=request.data_source.symbol,
                    timeframe=request.data_source.timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    source=request.data_source.source
                )
            
            session_metadata = None
            analysis_results = {}
            
            if existing_session:
                logger.warning(
                    f"⚠️ Duplicate session found: {existing_session['session_id']} "
                    f"({existing_session['record_count']} records). Returning existing session."
                )
                
                # If we are NOT skipping the check, return early so frontend can show conflict dialog
                if not request.skip_duplicate_check:
                    df_records = fetched_df.to_dict(orient='records')
                    return {
                        'session_id': existing_session['session_id'],
                        'task_id': task_id,
                        'is_duplicate': True,
                        'metadata': existing_session,
                        'preview': {
                            'head': df_records[:5] if len(df_records) > 0 else [],
                            'tail': df_records[-5:] if len(df_records) > 0 else [],
                            'record_count': record_count,
                            'columns': list(fetched_df.columns),
                            'start_date': start_date,
                            'end_date': end_date,
                        }
                    }
                
                # If we ARE skipping (user chose "create new"), continue to session creation
                # Note: We don't set session_metadata yet, it will be built in the 'else' block
                pass
            
            if not existing_session or request.skip_duplicate_check:
                # ─────────────────────────────────────────────────────────
                # STEP 2b: Build metadata and create session with data
                # ─────────────────────────────────────────────────────────
                logger.info("💾 Step 2b: Building metadata and creating new session...")
                session_metadata = SessionMetadataBuilder.build(
                    source=request.data_source.source,
                    symbol=request.data_source.symbol,
                    timeframe=request.data_source.timeframe,
                    start_date=start_date,  # ✅ NOW HAVE DATES
                    end_date=end_date,      # ✅ NOW HAVE DATES
                    record_count=record_count,
                    session_id=session_id,  # ✅ FROM PARAMETER
                    name=getattr(request, 'session_name', None),  # Optional session name
                    description=getattr(request, 'session_description', None),  # Optional description
                )
                
                # Create session AND store data in single atomic transaction
                await self._create_session_and_store_data_atomic(
                    session_metadata,
                    fetched_df,
                    session_id=session_id,
                    task_id=task_id,
                    user_id=user_id,
                )
                
                if process:
                    mem_after_store = process.memory_info().rss / 1024 / 1024
                    logger.info(
                        f"[Pipeline] After storage: {mem_after_store:.1f} MB "
                        f"(+{mem_after_store - mem_start:.1f} MB total)"
                    )
            
            # ─────────────────────────────────────────────────────────
            # STEP 3: Run Analysis Steps (optional)
            # ─────────────────────────────────────────────────────────
            analysis_steps = request.analysis_steps or []
            if analysis_steps:
                logger.info(f"🔬 Step 3: Running analysis steps: {analysis_steps}")
                await self._run_analysis_steps(
                    fetched_df,
                    analysis_steps,
                    request.analysis_configs or {},
                    session_id=session_id,
                    task_id=task_id,
                    user_id=user_id,  # ✅ Pass user_id for progress tracking
                )
                
                if process:
                    mem_after_analysis = process.memory_info().rss / 1024 / 1024
                    logger.info(
                        f"[Pipeline] After analysis: {mem_after_analysis:.1f} MB "
                        f"(+{mem_after_analysis - mem_start:.1f} MB total)"
                    )
                
                # ─────────────────────────────────────────────────────────
                # STEP 3b: Retrieve cached results from analysis steps
                # ─────────────────────────────────────────────────────────
                logger.info(f"📊 Step 3b: Retrieving cached analysis results...")
                analysis_results = await self._retrieve_analysis_cache(
                    analysis_steps,
                    task_id=task_id,
                )
            
            # ─────────────────────────────────────────────────────────
            # Return Results
            # ─────────────────────────────────────────────────────────
            df_records = fetched_df.to_dict(orient='records')
            
            # Use session_id from metadata (handles duplicates correctly)
            final_session_id = session_metadata.get('session_id') if session_metadata else session_id
            is_duplicate = existing_session is not None
            
            result = {
                'session_id': final_session_id,
                'task_id': task_id,
                'is_duplicate': is_duplicate,
                'metadata': session_metadata,
                'preview': {
                    'head': df_records[:5] if len(df_records) > 0 else [],
                    'tail': df_records[-5:] if len(df_records) > 0 else [],
                    'record_count': record_count,
                    'columns': list(fetched_df.columns),
                    'start_date': start_date,
                    'end_date': end_date,
                },
                'analysis_results': analysis_results,
            }
            
            # 🧹 CLEANUP: Delete large intermediate data structures before return
            del fetched_df
            del df_records
            del analysis_results
            if session_metadata:
                del session_metadata
            if existing_session:
                del existing_session
            gc.collect()
            
            # 🧹 Periodic maintenance: cleanup expired cache entries
            AnalysisManager.cleanup_expired_cache()
            
            if process and mem_start is not None:
                mem_end = process.memory_info().rss / 1024 / 1024
                logger.info(
                    f"[Pipeline] Completed. Total memory delta: +{mem_end - mem_start:.1f} MB "
                    f"(start: {mem_start:.1f} MB → end: {mem_end:.1f} MB)"
                )
            
            logger.info(f"✅ Pipeline completed: {final_session_id} (task {task_id[:8]}...) - Duplicate: {is_duplicate}")

            # 🏁 FINAL STEP: Send 100% completion message to frontend
            if reporter:
                await reporter.report_async(
                    progress=100,
                    type="complete",
                    message="Workload Complete",
                    message2=f"Session {final_session_id[:8]} created successfully with {record_count:,} records.",
                    stage="end"
                )
            
            return result
            
        except Exception as e:
            # 🧹 CLEANUP: Even on error, clear memory
            if 'fetched_df' in locals(): del fetched_df
            if 'df_records' in locals(): del df_records
            gc.collect()
            logger.error(f"❌ Pipeline error: {e}", exc_info=True)
            raise
    
        
    async def run_analysis_step(
        self,
        analysis_type: Union[AnalysisType, str],
        session_id: str,
        task_id: str,
        pm: ProcessingManager,
        request_data: Optional[List[Dict[str, Any]]] = None,
        user_id: str = "anonymous"
    ) -> Dict[str, Any]:
        """
        Unified orchestrator for a single analysis step.
        
        ✅ REFACTORED: Uses integrated helpers (no lazy import)
        
        Handles:
        1. 3-tier data loading (Request -> Cache -> DB)
        2. Column normalization via self.standardize_dataframe_columns()
        3. Delegation to integrated execute_*_analysis() methods
        4. Updates self.current_data for state tracking
        
        Args:
            analysis_type: Type of analysis (TECHNICAL, SNR, ASTRONOMICAL)
            session_id: Session UUID
            task_id: Task UUID
            pm: Externally configured ProcessingManager instance (Worker)
            request_data: Optional inline data from request
            user_id: User identifier for progress tracking
            
        Returns:
            Result dictionary from analysis
        """
        try:
            if isinstance(analysis_type, str):
                normalized = analysis_type.strip().lower()
                alias_map = {
                    "ml_preparation": AnalysisType.ML_DATASET_PREPARATION,
                    "ml_dataset_preparation": AnalysisType.ML_DATASET_PREPARATION,
                    "technical": AnalysisType.TECHNICAL,
                    "technical_analysis": AnalysisType.TECHNICAL,
                    "snr": AnalysisType.SNR,
                    "snr_analysis": AnalysisType.SNR,
                    "astronomical": AnalysisType.ASTRONOMICAL,
                    "astronomical_analysis": AnalysisType.ASTRONOMICAL,
                    "model_build": AnalysisType.MODEL_BUILD,
                    "model_training": AnalysisType.MODEL_TRAINING,
                    "currency_indices": AnalysisType.CURRENCY_INDICES,
                }
                analysis_type = alias_map.get(normalized)
                if not analysis_type:
                    analysis_type = AnalysisType(normalized)

            logger.info(f"🚀 Unified Orchestrator: Starting {analysis_type.value} for task {task_id[:8]}...")
            step_config_for_contract = request_data if request_data is not None else getattr(pm, "config", None)
            await self._persist_step_config(
                session_id=session_id,
                step_key=analysis_type.value,
                config=step_config_for_contract,
                task_id=task_id,
            )
            
            # ─────────────────────────────────────────────────────────
            # NOTE: Registration now happens atomically in route handler
            # via check_and_register_analysis() to prevent race conditions
            # ─────────────────────────────────────────────────────────
            
            # ─────────────────────────────────────────────────────────
            # 1. LOAD DATA (4-TIER HIERARCHY with proper exclude_step)
            # 🔴 BUG FIX #3: Ensure exclude_step is used (prevent direct API bypass)
            # ✅ SKIP DATA LOADING FOR ML MODEL STEPS (they use configs, not raw data)
            # ─────────────────────────────────────────────────────────
            
            # ML steps don't need raw OHLC data - they work with configs or load their own data internally
            ml_steps_skip_data = {AnalysisType.MODEL_BUILD, AnalysisType.MODEL_TRAINING, AnalysisType.ML_DATASET_PREPARATION}
            
            if analysis_type not in ml_steps_skip_data:
                # Build exclude_step: match the actual stored step_name in session_step_results.
                # Most steps store as "{analysis_type.value}_analysis" (e.g. "technical_analysis"),
                # but currency_indices stores as "currency_indices" (no _analysis suffix).
                # Using the wrong suffix means the current step's previous output is never
                # excluded from its own input load — harmless but confusing.
                _exclude_step_map = {
                    "currency_indices": "currency_indices",          # stored without _analysis suffix
                    "footprint_ingestion": "footprint_ingestion",    # stored without _analysis suffix
                }
                _exclude_step = _exclude_step_map.get(
                    analysis_type.value,
                    f"{analysis_type.value}_analysis"   # default: add _analysis suffix
                )
                df, source = await self._load_data_4_tier(
                    session_id=session_id,
                    task_id=task_id,
                    request_data=request_data,
                    exclude_step=_exclude_step,  # ✅ Correctly maps to stored step_name
                    data_type="analysis"  # TIER 0a: Analysis pointer
                )
                
                # ─────────────────────────────────────────────────────────
                # 2. NORMALIZE & PREPARE (using integrated method)
                # ─────────────────────────────────────────────────────────
                df = data_utils.normalize_dataframe_columns(df)
                
                # Determine if OHLC is required for this step
                # ✅ RELAXATION: Astronomical analysis does not strictly need OHLC to merge features
                require_ohlc = analysis_type not in [AnalysisType.ASTRONOMICAL]
                df = self.prepare_dataframe(df, require_ohlc=require_ohlc)
                
                if len(df) == 0:
                    raise ValueError(f"No data available for {analysis_type.value} analysis")
            else:
                # For ML steps, create empty dataframe (not used by these steps)
                df = pd.DataFrame()
                logger.info(f"✅ Skipping OHLC data loading for ML step: {analysis_type.value}")
            
            # ─────────────────────────────────────────────────────────
            # 4. EXECUTE (DIRECT DELEGATION WITH INTEGRATED METHODS)
            # ─────────────────────────────────────────────────────────
            if analysis_type == AnalysisType.TECHNICAL:
                result = await self.execute_technical_analysis(
                    df=df,
                    pm=pm,
                    session_id=session_id,
                    task_id=task_id,
                    user_id=user_id,
                )
            elif analysis_type == AnalysisType.SNR:
                result = await self.execute_snr_analysis(
                    df=df,
                    pm=pm,
                    session_id=session_id,
                    task_id=task_id,
                    user_id=user_id,
                )
            elif analysis_type == AnalysisType.ASTRONOMICAL:
                result = await self.execute_astronomical_analysis(
                    df=df,
                    pm=pm,
                    session_id=session_id,
                    task_id=task_id,
                    user_id=user_id,
                )
            elif analysis_type == AnalysisType.CURRENCY_INDICES:
                # Use the config that was already parsed into the PM
                # (pm.config is the CurrencyIndexConfig with calculate_ti_for_indices populated)
                config = pm.config
                result = await self.execute_currency_indices_analysis(
                    df=df,
                    pm=pm,
                    session_id=session_id,
                    task_id=task_id,
                    config=config,
                    user_id=user_id,
                )
            elif analysis_type == AnalysisType.ML_DATASET_PREPARATION:
                # ML dataset preparation - uses PM for large dataset handling
                result = await self.execute_ml_preparation(
                    session_id=session_id,
                    task_id=task_id,
                    pm=pm,
                    request_data=request_data,
                    user_id=user_id
                )
            elif analysis_type == AnalysisType.MODEL_BUILD:
                # Model build with PM for optimization
                model_config = request_data or {}
                result = await self.execute_model_build_with_pm(
                    session_id=session_id,
                    task_id=task_id,
                    pm=pm,
                    model_config=model_config,
                    user_id=user_id
                )
            elif analysis_type == AnalysisType.MODEL_TRAINING:
                # Model training with PM for data streaming
                train_config = request_data or {}
                result = await self.execute_model_training_with_pm(
                    session_id=session_id,
                    task_id=task_id,
                    pm=pm,
                    train_config=train_config,
                    user_id=user_id
                )
            else:
                raise ValueError(f"Unsupported analysis type: {analysis_type}")
            
            # ─────────────────────────────────────────────────────────
            # 5. CHECK RESULT STATUS BEFORE PROCEEDING
            # ─────────────────────────────────────────────────────────
            # If the analysis failed, we should not report success
            if isinstance(result, dict) and result.get("status") == "error":
                error_message = result.get("message", "Analysis failed")
                logger.error(f"❌ Analysis failed: {error_message}")
                
                # Send error completion message
                if self.connection_manager:
                    try:
                        type_name = analysis_type.value if hasattr(analysis_type, 'value') else str(analysis_type)
                        error_payload = {
                            "type": "error",
                            "task_id": task_id,
                            "progress": 0,
                            "status": "error",
                            "message": error_message,
                            "user_id": user_id,
                        }
                        await self.connection_manager.send_progress_update(task_id, error_payload, user_id=user_id)
                        logger.info(f"📤 [WS] Sent type='error' for task {task_id[:8]} (user={user_id})")
                    except Exception as ws_err:
                        logger.warning(f"⚠️ Failed to send error WebSocket for task {task_id[:8]}: {ws_err}")
                
                # Unregister and raise error
                self.unregister_analysis(analysis_type.value, task_id)
                raise RuntimeError(error_message)
            
            # ─────────────────────────────────────────────────────────
            # 6. PREPARE API RESPONSE (Strip DataFrames)
            # ─────────────────────────────────────────────────────────
            # Prevent FastAPI JSON serialization errors by removing pandas DataFrame objects
            # The data itself is already securely saved to the database and cached.
            if isinstance(result, dict):
                keys_to_remove = []
                for k, v in result.items():
                    if isinstance(v, pd.DataFrame):
                        keys_to_remove.append(k)
                for k in keys_to_remove:
                    df_to_strip = result.pop(k)
                    result[f"{k}_preview"] = {
                        "rows": len(df_to_strip),
                        "columns_count": len(df_to_strip.columns),
                    }
            
            logger.info(f"✅ Unified Orchestrator: {analysis_type.value} completed for task {task_id[:8]}")
            
            # Store result for finally block (completion broadcast)
            final_result = result
            final_status = 'completed'
            final_error = None
            
        except Exception as e:
            type_name = analysis_type.value if hasattr(analysis_type, 'value') else str(analysis_type)
            logger.error(f"❌ Unified Orchestrator failed for {type_name}: {e}", exc_info=True)
            
            # Prepare error result for finally block
            final_result = {"status": "error", "message": str(e)}
            final_status = 'error'
            final_error = e
        
        finally:
            # ─────────────────────────────────────────────────────────
            # ALWAYS BROADCAST COMPLETION (success OR error)
            # This ensures frontend always receives notification
            # ─────────────────────────────────────────────────────────
            type_name = analysis_type.value if hasattr(analysis_type, 'value') else str(analysis_type)
            
            try:
                if self.connection_manager:
                    completion_payload = {
                        "type": "complete",
                        "task_id": task_id,
                        "progress": 100,
                        "status": final_status,  # 'completed' or 'error'
                        "message": f"{type_name} analysis {final_status}",
                        "user_id": user_id,
                    }
                    
                    # Preserve any non-DataFrame scalar metadata from result (if successful)
                    # 🔒 PERF FIX: Strip massive data from WS broadcast to prevent main-thread hangs
                    if final_status == 'completed' and final_result:
                        STRIP_KEYS = ('result_records', 'ml_dataset', 'data', 'records', 'result_df')
                        for key, val in (final_result or {}).items():
                            if key in STRIP_KEYS:
                                continue
                            if isinstance(val, (int, float, str, bool, list, dict)) and not key.endswith("_preview"):
                                completion_payload[key] = val
                    elif final_error:
                        # Add error details if available
                        completion_payload["error"] = str(final_error)
                    
                    # ✅ WS-FIX: Ensure entire completion payload is serializable
                    completion_payload = to_serializable(completion_payload)
                    await self.connection_manager.send_progress_update(task_id, completion_payload, user_id=user_id)
                    logger.info(f"📤 [WS] Sent type='complete' (status={final_status}) for task {task_id[:8]} (user={user_id})")
            except Exception as ws_err:
                if "NOT found in active connections" in str(ws_err):
                    logger.debug(f"ℹ️ User {user_id} disconnected before completion message could be sent.")
                else:
                    logger.warning(f"⚠️ Failed to send completion WebSocket for task {task_id[:8]}: {ws_err}")
            
            # ─────────────────────────────────────────────────────────
            # ALWAYS UNREGISTER (success OR error)
            # ─────────────────────────────────────────────────────────
            try:
                await self.unregister_analysis(task_id, final_status=final_status)
                logger.info(f"✅ Unregistered analysis: {type_name} (status={final_status}, task={task_id[:8]})")
            except Exception as unreg_err:
                logger.warning(f"⚠️ Failed to unregister analysis {task_id[:8]}: {unreg_err}")
            
            # Re-raise if there was an error
            if final_error:
                raise final_error
        
        return final_result

    # ════════════════════════════════════════════════════════════════════
    # SELF-PROBING & ANALYSIS STATE MANAGEMENT (NEW)
    # ════════════════════════════════════════════════════════════════════
    
    async def register_active_analysis(
        self,
        task_id: str,
        session_id: str,
        analysis_type: str,
        user_id: str,
    ) -> bool:
        """
        Register a new analysis as active to prevent duplicates.
        
        Returns:
            True if registration successful (analysis can proceed)
            False if duplicate detected (analysis should be rejected)
        """
        async with self.analysis_lock:
            # Check if task is already running
            if task_id in self.active_analyses:
                existing = self.active_analyses[task_id]
                logger.warning(
                    f"🚫 [AnalysisManager] Duplicate analysis detected: task_id={task_id}, "
                    f"existing={existing['analysis_type']}, new={analysis_type}"
                )
                return False
            
            # Register new analysis
            self.active_analyses[task_id] = {
                'task_id': task_id,
                'session_id': session_id,
                'analysis_type': analysis_type,
                'user_id': user_id,
                'started_at': datetime.utcnow(),
                'status': 'running',
                'progress': 0,
                'last_update': datetime.utcnow(),
            }
            
            logger.info(
                f"✅ [AnalysisManager] Registered active analysis: {analysis_type} "
                f"(task={task_id[:8]}, session={session_id[:8]})"
            )
            logger.debug(f"🔵 [AnalysisManager] Active analyses count: {len(self.active_analyses)}")
            return True
    
    async def update_analysis_progress(
        self,
        task_id: str,
        progress: float,
        status: str = 'running',
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update progress for an active analysis."""
        async with self.analysis_lock:
            if task_id in self.active_analyses:
                self.active_analyses[task_id].update({
                    'progress': progress,
                    'status': status,
                    'last_update': datetime.utcnow(),
                    **(additional_info or {})
                })
    
    async def unregister_analysis(
        self,
        task_id: str,
        final_status: str = 'completed',
    ) -> None:
        """Unregister analysis when it completes or fails."""
        async with self.analysis_lock:
            if task_id in self.active_analyses:
                analysis_info = self.active_analyses.pop(task_id)
                duration = datetime.utcnow() - analysis_info['started_at']
                logger.info(
                    f"🏁 [AnalysisManager] Unregistered analysis: {analysis_info['analysis_type']} "
                    f"(task={task_id[:8]}, duration={duration.total_seconds():.1f}s, status={final_status})"
                )
                logger.debug(f"🔵 [AnalysisManager] Remaining active analyses: {len(self.active_analyses)}")
            else:
                logger.warning(f"⚠️ [AnalysisManager] Attempted to unregister non-existent task: {task_id[:8]}")
    
    async def cancel_analysis(
        self,
        task_id: str,
        user_id: str = "anonymous"
    ) -> Dict[str, Any]:
        """
        Cancel a running analysis task.
        
        Cancellation flow:
        1. Check if task exists and is running
        2. Mark task as cancelled in task_store (ProcessingManager checks this)
        3. Send cancellation message via WebSocket
        4. Unregister from active_analyses
        
        Args:
            task_id: Task ID to cancel
            user_id: User requesting cancellation
            
        Returns:
            Dict with success status and message
        """
        try:
            # Check if task exists
            async with self.analysis_lock:
                if task_id not in self.active_analyses:
                    return {
                        'success': False,
                        'error': 'Task not found or already completed',
                        'task_id': task_id
                    }
                
                analysis_info = self.active_analyses[task_id]
                analysis_type = analysis_info['analysis_type']
            
            logger.info(
                f"🛑 [AnalysisManager] Cancelling analysis: {analysis_type} "
                f"(task={task_id[:8]}, user={user_id})"
            )
            
            # Mark task as cancelled in task_store (ProcessingManager will check this)
            if self.task_store:
                self.task_store.cancel_task(task_id)
                logger.info(f"✅ [AnalysisManager] Marked task {task_id[:8]} as cancelled in task_store")
            
            # Send cancellation message via WebSocket
            if self.connection_manager:
                try:
                    await self.connection_manager.send_progress_update(
                        task_id,
                        {
                            "type": "error",
                            "progress": 0,
                            "status": "cancelled",
                            "message": "Analysis cancelled by user",
                            "user_id": user_id,
                        },
                        user_id=user_id
                    )
                    logger.info(f"📤 [AnalysisManager] Sent cancellation message for task {task_id[:8]}")
                except Exception as ws_err:
                    logger.warning(f"⚠️ [AnalysisManager] Failed to send cancellation message: {ws_err}")
            
            # Unregister from active analyses
            await self.unregister_analysis(task_id, final_status='cancelled')
            
            return {
                'success': True,
                'message': f'Analysis {analysis_type} cancelled successfully',
                'task_id': task_id
            }
            
        except Exception as e:
            logger.error(f"❌ [AnalysisManager] Failed to cancel task {task_id[:8]}: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'task_id': task_id
            }
    
    async def probe_analysis_state(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Probe the current state of an analysis.
        
        Returns:
            Analysis state dict if found, None if not active
        """
        async with self.analysis_lock:
            if task_id in self.active_analyses:
                analysis_info = self.active_analyses[task_id].copy()
                # Add computed fields
                analysis_info['duration_seconds'] = (
                    datetime.utcnow() - analysis_info['started_at']
                ).total_seconds()
                analysis_info['is_stale'] = (
                    datetime.utcnow() - analysis_info['last_update']
                ).total_seconds() > 60  # No update in 60s = stale
                return analysis_info
            return None
    
    async def get_all_active_analyses(self) -> Dict[str, Dict[str, Any]]:
        """Get all currently active analyses."""
        async with self.analysis_lock:
            result = {}
            for task_id, info in self.active_analyses.items():
                analysis_info = info.copy()
                analysis_info['duration_seconds'] = (
                    datetime.utcnow() - analysis_info['started_at']
                ).total_seconds()
                analysis_info['is_stale'] = (
                    datetime.utcnow() - analysis_info['last_update']
                ).total_seconds() > 60
                result[task_id] = analysis_info
            return result
    
    async def cleanup_stale_analyses(self, max_age_seconds: int = 3600) -> List[str]:
        """
        Clean up analyses that have been running too long without updates.
        
        Returns:
            List of cleaned up task IDs
        """
        cleaned_up = []
        async with self.analysis_lock:
            now = datetime.utcnow()
            stale_tasks = []
            
            for task_id, info in self.active_analyses.items():
                age = (now - info['started_at']).total_seconds()
                last_update_age = (now - info['last_update']).total_seconds()
                
                # Clean up if too old or no updates for too long
                if age > max_age_seconds or last_update_age > 300:  # 5 min no update
                    stale_tasks.append(task_id)
            
            for task_id in stale_tasks:
                analysis_info = self.active_analyses.pop(task_id)
                cleaned_up.append(task_id)
                logger.warning(
                    f"🧹 Cleaned up stale analysis: {analysis_info['analysis_type']} "
                    f"(task={task_id[:8]}, age={age:.1f}s)"
                )
        
        return cleaned_up
    
    async def is_analysis_duplicate(
        self,
        session_id: str,
        analysis_type: str,
        task_id: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if an analysis is already running for the same session/type.
        
        Returns:
            (is_duplicate, existing_task_id)
        """
        async with self.analysis_lock:
            for existing_task_id, info in self.active_analyses.items():
                if (info['session_id'] == session_id and 
                    info['analysis_type'] == analysis_type and
                    info['status'] == 'running'):
                    
                    # If task_id provided, check if it's the same task (not duplicate)
                    if task_id and existing_task_id == task_id:
                        continue
                    
                    return True, existing_task_id
            
            return False, None
    
    async def check_and_register_analysis(
        self,
        task_id: str,
        session_id: str,
        analysis_type: str,
        user_id: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Atomically check for duplicates AND register if not duplicate.
        
        This prevents race conditions where two requests check for duplicates
        simultaneously, both see no duplicate, then both try to register.
        
        Returns:
            (success, existing_task_id_if_duplicate)
        """
        async with self.analysis_lock:
            # Check for existing analysis
            for existing_task_id, info in self.active_analyses.items():
                if (info['session_id'] == session_id and 
                    info['analysis_type'] == analysis_type and
                    info['status'] == 'running'):
                    
                    # If same task_id, not a duplicate
                    if existing_task_id == task_id:
                        continue
                    
                    logger.warning(
                        f"🚫 [AnalysisManager] Duplicate analysis detected (atomic check): "
                        f"task_id={task_id[:8]}, existing={existing_task_id[:8]}, "
                        f"type={analysis_type}"
                    )
                    return False, existing_task_id
            
            # No duplicate found - register immediately (still holding lock)
            self.active_analyses[task_id] = {
                'task_id': task_id,
                'session_id': session_id,
                'analysis_type': analysis_type,
                'user_id': user_id,
                'started_at': datetime.utcnow(),
                'status': 'running',
                'progress': 0,
                'last_update': datetime.utcnow(),
            }
            
            logger.info(
                f"✅ [AnalysisManager] Registered active analysis (atomic): {analysis_type} "
                f"(task={task_id[:8]}, session={session_id[:8]})"
            )
            logger.debug(f"🔵 [AnalysisManager] Active analyses count: {len(self.active_analyses)}")
            return True, None
    
    async def set_session_busy(
        self,
        session_id: str,
        analysis_type: str,
    ) -> bool:
        """
        Set busy flag for a session+analysis_type combination.
        
        This provides O(1) fast rejection at entry point before expensive operations.
        
        Returns:
            True if flag was set (not already busy)
            False if already busy (reject request)
        """
        async with self.busy_lock:
            key = (session_id, analysis_type)
            
            if self.session_busy.get(key, False):
                logger.warning(
                    f"🚫 [BusyFlag] Session already busy: {analysis_type} "
                    f"(session={session_id[:8]})"
                )
                return False
            
            self.session_busy[key] = True
            logger.debug(
                f"🔒 [BusyFlag] Set busy: {analysis_type} (session={session_id[:8]})"
            )
            return True
    
    async def clear_session_busy(
        self,
        session_id: str,
        analysis_type: str,
    ) -> None:
        """
        Clear busy flag for a session+analysis_type combination.
        
        Should be called in finally block to ensure cleanup even on error.
        """
        async with self.busy_lock:
            key = (session_id, analysis_type)
            if key in self.session_busy:
                del self.session_busy[key]
                logger.debug(
                    f"🔓 [BusyFlag] Cleared busy: {analysis_type} (session={session_id[:8]})"
                )
    
    async def is_session_busy(
        self,
        session_id: str,
        analysis_type: str,
    ) -> bool:
        """
        Check if session is busy (read-only, no lock needed for simple check).
        
        Returns:
            True if busy, False if available
        """
        key = (session_id, analysis_type)
        return self.session_busy.get(key, False)
    
    # validate_ml_config → defined in MLPipelineMixin (full version)
    # MRO: AnalysisManager(AnalysisExecutionMixin, MLPipelineMixin, TrainingEngineMixin)
    # mixin2 version (66 lines) supersedes the old 37-line stub that was here.

    
    def classify_error(self, exception: Exception, step: str) -> ErrorContext:
        """
        Classify exception for retry decision and frontend notification.
        
        Returns ErrorContext with:
        - category: ErrorCategory enum
        - retryable: bool (safe to retry?)
        - backoff_seconds: float (exponential backoff)
        """
        error_msg = str(exception)
        
        logger.debug(f"🔍 [AnalysisManager.classify_error] Classifying error in {step}: {error_msg[:100]}")
        
        # Network errors - RETRYABLE (except WebSocket timeout during analysis)
        if any(keyword in error_msg.lower() for keyword in 
               ['timeout', 'connection reset', 'connection refused', 'connection timed out']):
            
            # SPECIAL CASE: WebSocket timeout during analysis is NOT retryable
            # This happens when analysis completes successfully but WebSocket times out
            # Retrying would create duplicate analysis processes
            if 'websocket timeout' in error_msg.lower() and any(analysis_step in step.lower() for analysis_step in 
                   ['snr_analysis', 'technical_analysis', 'astronomical_analysis', 'analysis']):
                
                logger.warning(
                    f"⚠️ [AnalysisManager] WebSocket timeout during {step} - treating as non-retryable "
                    f"to prevent duplicate analysis"
                )
                
                # Add context about active analyses (non-async check)
                analysis_context = ""
                try:
                    # Get count of active analyses without async call
                    active_count = len(self.active_analyses)
                    if active_count > 0:
                        analysis_context = f" (Active analyses: {active_count})"
                        logger.debug(f"🔵 [AnalysisManager] Active analyses during timeout: {list(self.active_analyses.keys())}")
                except Exception:
                    pass  # Don't fail error classification due to probing issues
                
                return ErrorContext(
                    ErrorCategory.VALIDATION_FAILED,  # Treat as non-retryable
                    f"WebSocket timeout during {step} (analysis likely completed){analysis_context}: {error_msg}",
                    step,
                    exception,
                    retry_count=0,
                )
            
            # Regular network timeouts are retryable
            logger.info(f"🔄 [AnalysisManager] Network timeout in {step} - retryable")
            return ErrorContext(
                ErrorCategory.NETWORK_TIMEOUT,
                f"Network timeout in {step}: {error_msg}",
                step,
                exception,
                retry_count=0,
            )
        
        if 'temporarily unavailable' in error_msg.lower():
            return ErrorContext(
                ErrorCategory.TEMPORARY_UNAVAILABLE,
                f"Temporary service unavailability in {step}",
                step,
                exception,
                retry_count=0,
            )
        
        # Lock/contention - RETRYABLE
        if any(keyword in error_msg.lower() for keyword in ['lock', 'deadlock', 'contention']):
            return ErrorContext(
                ErrorCategory.LOCK_CONTENTION,
                f"Lock contention in {step}: {error_msg}",
                step,
                exception,
                retry_count=0,
            )
        
        # Rate limiting - RETRYABLE  
        if any(keyword in error_msg.lower() for keyword in ['rate limit', '429', 'too many requests']):
            return ErrorContext(
                ErrorCategory.RATE_LIMITED,
                f"Rate limited in {step}",
                step,
                exception,
                retry_count=0,
            )
        
        # Resource exhaustion - RETRYABLE
        if any(keyword in error_msg.lower() for keyword in ['memory', 'cpu', 'resource', 'exhausted']):
            return ErrorContext(
                ErrorCategory.RESOURCE_EXHAUSTED,
                f"Resource exhaustion in {step}: {error_msg}",
                step,
                exception,
                retry_count=0,
            )
        
        # Validation errors - NOT RETRYABLE
        if any(keyword in error_msg.lower() for keyword in ['validation', 'schema', 'invalid']):
            return ErrorContext(
                ErrorCategory.VALIDATION_FAILED,
                f"Data validation failed in {step}: {error_msg}",
                step,
                exception,
                retry_count=0,
            )
        
        # Format errors - NOT RETRYABLE
        if any(keyword in error_msg.lower() for keyword in ['format', 'malformed', 'parse']):
            return ErrorContext(
                ErrorCategory.INVALID_DATA_FORMAT,
                f"Invalid data format in {step}: {error_msg}",
                step,
                exception,
                retry_count=0,
            )
        
        # Missing fields - NOT RETRYABLE
        if any(keyword in error_msg.lower() for keyword in ['missing', 'required field', 'column']):
            return ErrorContext(
                ErrorCategory.MISSING_REQUIRED_FIELD,
                f"Missing required field in {step}: {error_msg}",
                step,
                exception,
                retry_count=0,
            )
        
        # Corruption - NOT RETRYABLE
        if 'checksum' in error_msg.lower() or 'corrupted' in error_msg.lower():
            return ErrorContext(
                ErrorCategory.DATA_CORRUPTION,
                f"Data corruption detected in {step}: {error_msg}",
                step,
                exception,
                retry_count=0,
            )
        
        # Default
        return ErrorContext(
            ErrorCategory.UNKNOWN,
            f"Unknown error in {step}: {error_msg}",
            step,
            exception,
            retry_count=0,
        )
    
    async def verify_step_result(
        self,
        result: Any,
        step_name: str,
        expected_type: str = 'dataframe',
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify step output matches frontend expectations BEFORE sending completion.
        
        Args:
            result: The output from analysis step
            step_name: Name of the step ('technical_analysis', 'snr_analysis', etc.)
            expected_type: 'dataframe', 'dict', 'pickle_str'
        
        Returns:
            (is_valid, error_message)
            - is_valid: True if result matches expectations
            - error_message: None if valid, error string if invalid
        """
        try:
            if result is None:
                return False, f"Step {step_name} returned None"
            
            if expected_type == 'dataframe':
                if not isinstance(result, pd.DataFrame):
                    return False, f"Expected DataFrame, got {type(result).__name__}"
                
                if len(result) == 0:
                    return False, f"Step {step_name} returned empty DataFrame"
                
                # Check for required OHLCV columns
                required_cols = {'open', 'high', 'low', 'close', 'volume'}
                if not required_cols.issubset(set(result.columns)):
                    missing = required_cols - set(result.columns)
                    return False, f"Missing required columns: {missing}"
                
                # Check for NaN rows (critical for analysis)
                if result.isnull().any().any():
                    null_count = result.isnull().sum().sum()
                    return False, f"DataFrame contains {null_count} NULL values"
            
            elif expected_type == 'pickle_str':
                if not isinstance(result, str):
                    return False, f"Expected pickle string, got {type(result).__name__}"
                
                if len(result) == 0:
                    return False, "Pickle string is empty"
                
                # Try to decode and unpickle to verify integrity
                try:
                    _ = deserialize_data(result)
                except Exception as e:
                    return False, f"Pickle deserialization failed: {e}"
            
            elif expected_type == 'dict':
                if not isinstance(result, dict):
                    return False, f"Expected dict, got {type(result).__name__}"
                
                if len(result) == 0:
                    return False, "Result dict is empty"
            
            return True, None
            
        except Exception as e:
            logger.error(f"Verification error in {step_name}: {e}")
            return False, f"Verification raised exception: {e}"
    
    async def send_error_to_frontend(
        self,
        error_ctx: ErrorContext,
        task_id: str,
        session_id: str,
    ) -> None:
        """
        Notify frontend of error through progress update.
        
        Uses dual-message format:
        - message_1: User-friendly action ("Retrying", "Error", "Waiting")
        - message_2: Technical details for debugging
        """
        try:
            progress_payload = {
                'type': 'progress',
                'progress': 0,  # Reset progress on error
                'message_1': f"⚠️ {error_ctx.category.value.replace('_', ' ').title()}",
                'message_2': error_ctx.message,
                'error_info': error_ctx.to_dict(),
                'stage': error_ctx.step,
                'retryable': error_ctx.retryable,
            }
            
            if self.connection_manager:
                await self.connection_manager.send_progress_update(task_id, progress_payload)
            
            if self.task_store:
                self.task_store.update_task(task_id, **progress_payload)
            
            logger.warning(f"🔔 Frontend notified of {error_ctx.category.value}: {error_ctx.message}")
            
        except Exception as e:
            logger.error(f"Failed to send error to frontend: {e}")
    
    async def execute_with_error_handling(
        self,
        request: AnalysisRequest,
        session_id: str,
        task_id: str,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """
        Execute pipeline with comprehensive error handling and retry logic.
        
        This wrapper:
        1. Checks for duplicate analyses using self-probing
        2. Registers analysis as active
        3. Attempts execution (retrying on recoverable errors)
        4. Verifies output before completion
        5. Notifies frontend of recoverable errors (allows retry messaging)
        6. Catches fatal errors and notifies without retry
        7. Unregisters analysis when complete
        
        Args:
            request: AnalysisRequest
            session_id: Per-call session ID
            task_id: Per-call task ID
            max_retries: Max retry attempts (default 3)
        
        Returns:
            Result dict (same as execute()) or error response
        """
        # ─────────────────────────────────────────────────────────
        # STEP 1: Self-probe to check for duplicates
        # ─────────────────────────────────────────────────────────
        analysis_type = 'pipeline'  # Default for full pipeline
        if request.analysis_steps:
            analysis_type = f"pipeline_{'+'.join(request.analysis_steps)}"
        
        user_id = request.user_id or "anonymous"
        
        # Check for existing analysis
        is_duplicate, existing_task_id = await self.is_analysis_duplicate(
            session_id=session_id,
            analysis_type=analysis_type,
            task_id=task_id,
        )
        
        if is_duplicate:
            logger.warning(
                f"🚫 Duplicate analysis rejected: {analysis_type} "
                f"(session={session_id[:8]}, existing_task={existing_task_id[:8]})"
            )
            
            # Return existing analysis state
            existing_state = await self.probe_analysis_state(existing_task_id)
            return {
                'error': True,
                'error_category': 'duplicate_analysis',
                'message': f'Analysis already running (task: {existing_task_id[:8]})',
                'retryable': False,
                'existing_task_id': existing_task_id,
                'existing_state': existing_state,
            }
        
        # Register this analysis as active
        registration_success = await self.register_active_analysis(
            task_id=task_id,
            session_id=session_id,
            analysis_type=analysis_type,
            user_id=user_id,
        )
        
        if not registration_success:
            return {
                'error': True,
                'error_category': 'registration_failed',
                'message': 'Failed to register analysis (possible race condition)',
                'retryable': True,
            }
        
        # ─────────────────────────────────────────────────────────
        # STEP 2: Execute with retry logic
        # ─────────────────────────────────────────────────────────
        retry_count = 0
        last_error: Optional[ErrorContext] = None
        
        try:
            while retry_count <= max_retries:
                try:
                    # Update progress
                    await self.update_analysis_progress(
                        task_id=task_id,
                        progress=0,
                        status='running',
                        additional_info={'retry_count': retry_count}
                    )
                    
                    # Execute pipeline
                    result = await self.execute(request, session_id, task_id)
                    
                    # Verify result before returning
                    is_valid, error_msg = await self.verify_step_result(
                        result,
                        step_name='pipeline_output',
                        expected_type='dict',
                    )
                    
                    if not is_valid:
                        raise ValueError(f"Output verification failed: {error_msg}")
                    
                    # Mark as completed
                    await self.update_analysis_progress(
                        task_id=task_id,
                        progress=100,
                        status='completed'
                    )
                    
                    logger.info(f"✅ Pipeline succeeded (attempt {retry_count + 1})")
                    return result
                    
                except Exception as e:
                    last_error = self.classify_error(e, analysis_type)
                    logger.warning(f"⚠️ Pipeline attempt {retry_count + 1} failed: {last_error.message}")
                    
                    # Update progress with error
                    await self.update_analysis_progress(
                        task_id=task_id,
                        progress=0,
                        status='error',
                        additional_info={
                            'error_category': last_error.category.value,
                            'error_message': last_error.message,
                            'retry_count': retry_count
                        }
                    )
                    
                    # Notify frontend of error
                    await self.send_error_to_frontend(last_error, task_id, session_id)
                    
                    if not last_error.retryable:
                        # Fatal error - don't retry
                        logger.error(f"❌ Fatal error (non-retryable): {last_error.message}")
                        return {
                            'error': True,
                            'error_category': last_error.category.value,
                            'message': last_error.message,
                            'retryable': False,
                        }
                    
                    if retry_count >= max_retries:
                        # Out of retries
                        logger.error(f"❌ Max retries ({max_retries}) exceeded")
                        return {
                            'error': True,
                            'error_category': last_error.category.value,
                            'message': f"{last_error.message} (max retries exceeded)",
                            'retryable': True,  # User can retry manually
                            'retry_count': retry_count,
                        }
                    
                    # Wait before retry with exponential backoff
                    retry_count += 1
                    backoff = last_error.backoff_seconds
                    logger.info(f"⏳ Retrying in {backoff:.1f}s (attempt {retry_count}/{max_retries})...")
                    
                    # Update progress for retry
                    await self.update_analysis_progress(
                        task_id=task_id,
                        progress=0,
                        status='retrying',
                        additional_info={'retry_count': retry_count, 'backoff_seconds': backoff}
                    )
                    
                    # Notify frontend of retry attempt
                    retry_payload = {
                        'type': 'progress',
                        'progress': 0,
                        'message_1': f"🔄 Retrying ({retry_count}/{max_retries})",
                        'message_2': last_error.message,
                        'stage': 'retry',
                    }
                    if self.connection_manager:
                        await self.connection_manager.send_progress_update(task_id, retry_payload)
                    
                    await asyncio.sleep(backoff)
            
            # Should not reach here, but handle just in case
            return {
                'error': True,
                'message': 'Pipeline execution failed',
                'retryable': True,
            }
            
        finally:
            # ─────────────────────────────────────────────────────────
            # STEP 3: Always unregister analysis when done
            # ─────────────────────────────────────────────────────────
            final_status = 'completed' if not last_error else 'failed'
            await self.unregister_analysis(task_id, final_status)
    
    # ════════════════════════════════════════════════════════════════════
    # MEMORY MANAGEMENT & POINTER CASCADING (NEW)
    # ════════════════════════════════════════════════════════════════════
    
    def cascade_clear_data_pointers(self, session_id: str, reason: str = "") -> None:
        """
        Cascade memory cleanup: When new pointer created, delete old data.
        
        Ensures memory efficiency:
        1. current_data → enriched_data: Delete current_data, keep enriched
        2. enriched_data → ml_data: Delete enriched_data, keep ml_data
        3. ml_data training done: Delete ml_data, model persisted
        
        This follows the pattern: new_pointer_created → old_pointer_deleted
        
        Args:
            session_id: Session being cleaned
            reason: Reason for cleanup (logged for debugging)
        """
        if self.current_session_id == session_id and self.current_data is not None:
            mem_before = self.current_data.memory_usage(deep=True).sum() / (1024 * 1024)
            self.current_data = None
            self._session_data_pointers.pop(session_id, None)
            logger.info(f"🗑️ Cleared current_data ({mem_before:.1f} MB) - {reason}")
        
        if self.ml_session_id == session_id:
            mem_before = 0
            if self.ml_train is not None:
                mem_before += self.ml_train.memory_usage(deep=True).sum() / (1024 * 1024)
            if self.ml_validation is not None:
                mem_before += self.ml_validation.memory_usage(deep=True).sum() / (1024 * 1024)
            if self.ml_test is not None:
                mem_before += self.ml_test.memory_usage(deep=True).sum() / (1024 * 1024)
            
            self.ml_train = None
            self.ml_validation = None
            self.ml_test = None
            
            if mem_before > 0:
                logger.info(f"🗑️ Cleared ML data pointers ({mem_before:.1f} MB) - {reason}")
        
        if self.model_session_id == session_id and self.model is not None:
            self.model = None
            self.model_id = None
            logger.info(f"🗑️ Cleared model instance - {reason}")
    
    def cleanup_function_locals(self, step_name: str) -> None:
        """
        Force cleanup of unwanted variables at function end.
        
        Python's garbage collection sometimes delays cleanup of large objects.
        This explicitly triggers cleanup to prevent memory accumulation.
        
        Call at end of each analysis step function.
        """
        gc.collect()
        logger.debug(f"🧹 Cleaned up locals for {step_name}")
    
    async def set_current_data(
        self,
        df: pd.DataFrame,
        session_id: str,
        step_name: str,
    ) -> None:
        """
        Update current_data pointer with new DataFrame.
        
        Cascading logic:
        - If different session, clear old session's data
        - Store new DataFrame as current_data
        - Log memory usage for debugging
        """
        if self.current_session_id and self.current_session_id != session_id:
            logger.debug(f"Session change detected: {self.current_session_id} → {session_id}")
            self.cascade_clear_data_pointers(self.current_session_id, f"New session from {step_name}")
        
        self.current_data = df
        self.current_session_id = session_id
        
        mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        logger.info(f"📍 Set current_data ({len(df)} rows, {df.shape[1]} cols, {mem_mb:.1f} MB) from {step_name}")
    
    async def set_ml_data_pointers(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        session_id: str,
        ml_dataset_name: str = None,  # ✅ NEW: Track which dataset these pointers correspond to
    ) -> None:
        """
        Update ML data pointers after ML dataset preparation.
        
        🔴 CRITICAL BUG FIX: Ensures pointers contain FULLY MERGED splits, not slices.
        
        Validation (prevents corruption):
        1. Verify each split is complete (DataFrame or numpy array)
        2. Total rows/sequences = sum(train + val + test) - sanity check
        3. Log row counts to audit trail
        
        Cascading logic:
        - Clear old current_data (now replaced by enriched ML data)
        - Store new ML split pointers  
        - Log memory usage + row verification
        
        This is called by execute_ml_preparation() after splits are created.
        The pointers allow execute_model_training() to access splits (ZERO LATENCY at training start).
        
        Args:
            train_df: Complete training split (DataFrame or numpy array)
            val_df: Complete validation split (DataFrame or numpy array)
            test_df: Complete test split (DataFrame or numpy array)
            session_id: Session ID for pointer ownership
            ml_dataset_name: ✅ NEW: Dataset name/ID to track which dataset these pointers correspond to
        """
        # ✅ UPDATED: Accept both DataFrames (for raw data) and numpy arrays (for ML sequences)
        # Validate that splits are not empty
        if train_df is None or (hasattr(train_df, '__len__') and len(train_df) == 0):
            raise ValueError(f"❌ train_df must not be empty, got {type(train_df)}, len={len(train_df) if hasattr(train_df, '__len__') else '?'}")
        
        if val_df is None or (hasattr(val_df, '__len__') and len(val_df) == 0):
            raise ValueError(f"❌ val_df must not be empty, got {type(val_df)}, len={len(val_df) if hasattr(val_df, '__len__') else '?'}")
        
        if test_df is None or (hasattr(test_df, '__len__') and len(test_df) == 0):
            raise ValueError(f"❌ test_df must not be empty, got {type(test_df)}, len={len(test_df) if hasattr(test_df, '__len__') else '?'}")
        
        # Calculate total for sanity check
        total_split_rows = len(train_df) + len(val_df) + len(test_df)
        
        # 🧹 CLEANUP: Remove OLD ML pointers before setting new ones
        if self.ml_session_id != session_id:
            if self.ml_train is not None:
                del self.ml_train
            if self.ml_validation is not None:
                del self.ml_validation
            if self.ml_test is not None:
                del self.ml_test
            gc.collect()

        # Clear old enriched data - now replaced by ML splits
        if self.current_session_id == session_id and self.current_data is not None:
            mem_before = self.current_data.memory_usage(deep=True).sum() / (1024 * 1024)
            self.current_data = None
            logger.info(f"🗑️ Cleared current_data ({mem_before:.1f} MB) - Replaced by ML splits")
        
        # ✅ STORE POINTERS (accept both DataFrames and numpy arrays)
        # For DataFrames: copy to prevent external mutations
        # For numpy arrays: store directly (already immutable in practice)
        if isinstance(train_df, pd.DataFrame):
            self.ml_train = train_df.copy()
        else:
            self.ml_train = train_df
            
        if isinstance(val_df, pd.DataFrame):
            self.ml_validation = val_df.copy()
        else:
            self.ml_validation = val_df
            
        if isinstance(test_df, pd.DataFrame):
            self.ml_test = test_df.copy()
        else:
            self.ml_test = test_df
            
        self.ml_session_id = session_id
        self.ml_dataset_name = ml_dataset_name  # ✅ NEW: Track dataset name for validation during training
        
        # Calculate memory usage based on data type
        if isinstance(train_df, pd.DataFrame):
            # DataFrames: use memory_usage()
            train_mem = train_df.memory_usage(deep=True).sum() / (1024 * 1024)
            val_mem = val_df.memory_usage(deep=True).sum() / (1024 * 1024)
            test_mem = test_df.memory_usage(deep=True).sum() / (1024 * 1024)
            total_mem = train_mem + val_mem + test_mem
            data_type = "DataFrames"
        elif isinstance(train_df, dict):
            # Dictionary with numpy arrays: sum nbytes of all arrays
            def calc_dict_mem(d):
                mem = 0
                for v in d.values():
                    if isinstance(v, np.ndarray):
                        mem += v.nbytes
                    elif isinstance(v, dict):
                        mem += calc_dict_mem(v)
                return mem
            
            train_mem = calc_dict_mem(train_df) / (1024 * 1024)
            val_mem = calc_dict_mem(val_df) / (1024 * 1024)
            test_mem = calc_dict_mem(test_df) / (1024 * 1024)
            total_mem = train_mem + val_mem + test_mem
            
            # Get sequence shape for logging
            seq_shape = train_df.get('sequences', np.array([])).shape if 'sequences' in train_df else 'unknown'
            data_type = f"numpy sequence dicts (sequences shape: {seq_shape})"
        else:
            # Pure numpy arrays
            train_mem = train_df.nbytes / (1024 * 1024)
            val_mem = val_df.nbytes / (1024 * 1024)
            test_mem = test_df.nbytes / (1024 * 1024)
            total_mem = train_mem + val_mem + test_mem
            data_type = f"numpy arrays (shape: {train_df.shape})"
        
        # ✅ FIX: Extract actual sequence counts from dicts/arrays properly
        def get_seq_count(data):
            if isinstance(data, dict) and 'sequences' in data:
                return data['sequences'].shape[0]  # Dict with 'sequences' key
            elif isinstance(data, np.ndarray):
                return data.shape[0]  # Direct numpy array
            elif isinstance(data, (list, dict)):
                return len(data)
            return 0
        
        train_seq_count = get_seq_count(train_df)
        val_seq_count = get_seq_count(val_df)
        test_seq_count = get_seq_count(test_df)
        total_seq_count = train_seq_count + val_seq_count + test_seq_count
        
        logger.info(
            f"✅ Set ML data pointers ({data_type}): \n"
            f"  ├─ train: {train_seq_count} sequences ({train_mem:.1f}MB) - COMPLETE ✓\n"
            f"  ├─ val: {val_seq_count} sequences ({val_mem:.1f}MB) - COMPLETE ✓\n"
            f"  ├─ test: {test_seq_count} sequences ({test_mem:.1f}MB) - COMPLETE ✓\n"
            f"  ├─ total sequences: {total_seq_count}\n"
            f"  ├─ total memory: {total_mem:.1f}MB\n"
            f"  └─ session_id: {session_id[:8]}..."
        )
    
    async def get_current_data(self, session_id: str) -> Optional[pd.DataFrame]:
        """
        Retrieve current_data with session validation.
        
        Returns None if wrong session_id (safety check).
        """
        if self.current_session_id != session_id:
            logger.warning(
                f"⚠️ Session mismatch: current_session={self.current_session_id}, "
                f"requested={session_id}"
            )
            return None
        
        if self.current_data is None:
            logger.warning(f"⚠️ current_data is None for session {session_id}")
            return None
        
        return self.current_data
    
    async def get_ml_data_pointers(
        self,
        session_id: str,
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Retrieve ML data pointers with session validation.
        
        Returns (train_df, val_df, test_df) or (None, None, None) if mismatched.
        """
        if self.ml_session_id != session_id:
            logger.warning(
                f"⚠️ ML session mismatch: ml_session={self.ml_session_id}, "
                f"requested={session_id}"
            )
            return None, None, None
        
        return self.ml_train, self.ml_validation, self.ml_test
    
    async def _fetch_data(
        self,
        request: DataSourceRequest,
        reporter: Optional[ProgressReporter] = None,
    ) -> tuple:
        """
        Fetch data from source using DataFetcher.
        
        Returns:
            (df, start_date_iso, end_date_iso, record_count)
        """
        try:
            # Build kwargs for specific source
            kwargs = {
                'symbol': request.symbol,
                'timeframe': request.timeframe,
            }
            
            if request.source == 'mt5':
                kwargs.update({
                    'count': request.count,
                    'date_from': request.date_from,
                    'date_to': request.date_to,
                    'method': request.method,
                })
            
            elif request.source == 'database':
                kwargs.update({
                    'date_from': request.date_from,
                    'date_to': request.date_to,
                    'limit': request.limit,
                    'page': request.page,
                    'min_close': request.min_close,
                    'max_close': request.max_close,
                    'min_volume': request.min_volume,
                    'max_volume': request.max_volume,
                })
            
            elif request.source == 'csv':
                kwargs['df'] = request.df
            
            # Delegate to DataFetcher
            df, start_iso, end_iso, count = await self.data_fetcher.fetch(
                source=request.source,
                reporter=reporter,
                **kwargs
            )
            
            logger.info(
                f"✅ Data fetched: {count} rows from {request.source}, "
                f"{start_iso} to {end_iso}"
            )
            
            return df, start_iso, end_iso, count
            
        except Exception as e:
            logger.error(f"❌ Data fetch failed: {e}")
            # 🧹 CLEANUP: Ensure df is deleted if it was partially created
            if 'df' in locals() and df is not None:
                del df
                gc.collect()
            raise
    
    async def _check_duplicate_session(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        source: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Check if a session with identical metadata already exists.
        
        Prevents duplicate sessions from being created when user re-fetches
        the same symbol/timeframe/date range.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Session start date (ISO)
            end_date: Session end date (ISO)
            source: Data source ('mt5', 'database', 'csv')
            
        Returns:
            Existing session dict if found, else None
        """
        try:
            async with AsyncPostgresSessionLocal() as db:
                # Query for existing session with same parameters
                result = await db.execute(
                    DataSession.__table__.select().where(
                        (DataSession.symbol == symbol) &
                        (DataSession.timeframe == timeframe) &
                        (DataSession.start_date == start_date) &
                        (DataSession.end_date == end_date) &
                        (DataSession.data_source == source) &
                        (DataSession.status == 'active')
                    ).limit(1)
                )
                
                row = result.first()
                if row:
                    logger.info(f"Found duplicate session: {row.session_id}")
                    return {
                        'session_id': row.session_id,
                        'symbol': row.symbol,
                        'timeframe': row.timeframe,
                        'start_date': row.start_date,
                        'end_date': row.end_date,
                        'record_count': row.record_count,
                        'created_at': row.created_at,
                    }
                
                return None
                
        except Exception as e:
            logger.warning(f"Duplicate check failed (non-blocking): {e}")
            # If check fails, proceed with session creation (fail-open)
            return None
    
    async def _create_session_and_store_data_atomic(
        self,
        metadata: Dict[str, Any],
        df: pd.DataFrame,
        session_id: str,
        task_id: str,
        user_id: str = "anonymous",
    ) -> None:
        """
        Create DataSession and store raw data in a SINGLE ATOMIC TRANSACTION.
        
        
        Process:
        1. Serialize data OFF-LOCK (CPU work, 500-800ms) ← PARALLELIZABLE
        2. Acquire lock (DB connection)
        3. INSERT session + pre-serialized data (< 100ms) ← FAST
        4. Release lock
        
        Benefits:
        - Lock held for only ~10ms instead of 550ms
        - Concurrent users serialize in parallel
        - Same atomic guarantees (both-or-nothing)
        - Prevents cascade failures under load
        
        Args:
            metadata: Complete session metadata dict
            df: Raw OHLCV DataFrame
            session_id: Per-call session UUID
            task_id: Per-call task UUID
        """
        try:
            # ─────────────────────────────────────────────────────────
            # STEP 1: SERIALIZE DATA OUTSIDE LOCK (CPU WORK)
            # ─────────────────────────────────────────────────────────
            logger.info(f"📦 Serializing data (outside lock)... rows={len(df)}")
            
            # 🔴 CRITICAL FIX: Send progress update during serialization
            # This prevents WebSocket timeout during long serialization operations
            if self.connection_manager:
                try:
                    await self.connection_manager.send_progress_update(
                        task_id,
                        {
                            "type": "progress",
                            "progress": 90,
                            "message": "Serializing",
                            "message2": f"Preparing {len(df):,} records for storage (Step 2/3)",  # ✅ Top-level
                            "stage": "serializing",
                            "status": "running",
                            "user_id": user_id or "anonymous",
                            "data": {
                                "rows": len(df)
                            }
                        },
                        user_id=user_id or "anonymous"
                    )
                    logger.info(f"📤 [WS] Sent serialization progress for task {task_id[:8]}")
                except Exception as ws_err:
                    logger.debug(f"⚠️ Failed to send serialization progress: {ws_err}")
            
            start_serialize = datetime.utcnow()
            
            # Convert to records for serialization
            df_records = df.to_dict(orient='records')
            record_count = len(df_records)  # ✅ Save count before potential deletion
                       
          
            # Frontend expects uniform pickle format across ALL steps
            force_pickle = metadata.get('force_pickle', False)
            step_name = 'data_source'
            
            # P4 FIX: Add 'data_source' to pickle-forced steps for consistency
            if force_pickle or step_name in ['data_source', 'technical_analysis', 'snr_analysis', 'astronomical_analysis']:
                use_pickle = True
            else:
                # Try JSONB first for raw data (usually works)
                use_pickle = False
            
            # Pre-serialize the data
            if use_pickle:
                logger.debug(f"  Using pickle format for '{step_name}'")
                serialized_data = serialize_data(df_records, compress=True, numpy_safe=False)
                
          
                serialized_hash = None
                try:
                    serialized_bytes = base64.b64decode(serialized_data)
                    serialized_hash = hashlib.sha256(serialized_bytes).hexdigest()
                    logger.debug(f"  Pickle hash: {serialized_hash[:8]}...")
                    
                    # 🧹 CLEANUP: Delete intermediate bytes immediately
                    del serialized_bytes
                    gc.collect()
                except Exception as hash_err:
                    logger.warning(f"  Failed to compute hash: {hash_err}")
            else:
                
                serialized_data = None
                serialized_hash = None
                logger.debug(f"  Using JSONB format for '{step_name}'")
            
            serialize_time = (datetime.utcnow() - start_serialize).total_seconds()
            logger.info(f"✅ Data serialized in {serialize_time:.2f}s (outside lock)")
            
            # Send progress update after serialization completes
            if self.connection_manager:
                try:
                    await self.connection_manager.send_progress_update(
                        task_id,
                        {
                            "type": "progress",
                            "progress": 95,
                            "message": "Storing",
                            "message2": f"Serialization complete ({serialize_time:.1f}s), saving to database (Step 3/3)", # ✅ Top-level
                            "stage": "storing",
                            "status": "running",
                            "user_id": user_id or "anonymous",
                            "data": {
                                "serialize_time_seconds": serialize_time
                            }
                        },
                        user_id=user_id or "anonymous"
                    )
                    logger.info(f"📤 [WS] Sent storage progress for task {task_id[:8]}")
                except Exception as ws_err:
                    logger.debug(f"⚠️ Failed to send storage progress: {ws_err}")
            
            # ─────────────────────────────────────────────────────────
            # STEP 2: ACQUIRE LOCK AND INSERT (FAST ONLY)
            # ─────────────────────────────────────────────────────────
            logger.info("🔒 Acquiring lock for atomic insert...")
            start_lock = datetime.utcnow()
            
            async with AsyncPostgresSessionLocal() as db:
                # Create session record
                session = DataSession(
                    session_id=metadata['session_id'],
                    name=metadata.get('name'),
                    description=metadata.get('description'),
                    data_source=metadata['data_source'],
                    symbol=metadata['symbol'],
                    timeframe=metadata['timeframe'],
                    start_date=metadata['start_date'],
                    end_date=metadata['end_date'],
                    record_count=metadata['record_count'],
                    data_checksum=metadata.get('data_checksum'),
                    status=metadata['status'],
                    notes=metadata.get('notes'),
                    created_at=datetime.utcnow(),
                )
                db.add(session)
                await db.flush()  # ✅ FLUSH first so session_id exists in DB for FK constraint
                
                # Store pre-serialized data (no CPU work inside lock!)
                await store_session_step_result(
                    session_id=session_id,  # ✅ FROM PARAMETER
                    step_name='data_source',
                    data=df_records if not use_pickle else None,  # Only pass if JSONB
                    db=db,
                    pre_serialized_data=serialized_data if use_pickle else None,
                    pre_serialized_hash=serialized_hash,
                    force_pickle=use_pickle,
                )
                
                # 🧹 MEMORY FIX: Delete serialized_data immediately after DB insert
                if serialized_data:
                    serialized_size_mb = len(serialized_data) / (1024 * 1024)
                    del serialized_data
                    gc.collect()
                    logger.debug(f"  🗑️ Deleted serialized_data after DB insert ({serialized_size_mb:.1f} MB freed)")
                
                # Commit both inserts atomically
                await db.commit()
                
                lock_time = (datetime.utcnow() - start_lock).total_seconds()
                logger.info(
                    f"✅ Atomic insert complete in {lock_time*1000:.0f}ms: "
                    f"session_id={metadata['session_id'][:8]}..., "
                    f"rows={record_count}"
                )
                
            # ─────────────────────────────────────────────────────────
            # STEP 3: UPDATE POINTERS AND CACHE (Outside DB Lock)
            # ─────────────────────────────────────────────────────────
            
            # DB pointer update
            async with AsyncPostgresSessionLocal() as db_update:
                await set_as_current_data(session_id, db_update)
                
            # TIER 0 Memory Pointer
            await self.set_current_data(df, session_id, 'data_source')
            
            # TIER 2 Cache
            await self.cache_session_data(session_id, df_records, 'data_source')
            
            # 🧹 CLEANUP: Delete df_records after storage and caching complete
            if 'df_records' in locals():
                df_records_size_mb_final = sys.getsizeof(df_records) / (1024 * 1024) if 'df_records' in locals() else 0
                del df_records
                gc.collect()
                logger.debug(f"  🗑️ Deleted df_records after caching ({df_records_size_mb_final:.1f} MB freed)")
            
            
        except Exception as e:
            logger.error(f"❌ Atomic operation failed: {e}")
            # 🧹 CLEANUP: Even on error
            if 'serialized_bytes' in locals(): del serialized_bytes
            if 'serialized_data' in locals(): del serialized_data
            if 'df_records' in locals(): del df_records
            gc.collect()
            raise

    
    async def _create_session_in_db(
        self,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Create DataSession record in database with complete metadata.
        
        Args:
            metadata: Complete session metadata dict
        """
        try:
            async with AsyncPostgresSessionLocal() as db:
                session = DataSession(
                    session_id=metadata['session_id'],
                    data_source=metadata['data_source'],
                    symbol=metadata['symbol'],
                    timeframe=metadata['timeframe'],
                    start_date=metadata['start_date'],  # ✅ NOW SET
                    end_date=metadata['end_date'],      # ✅ NOW SET
                    record_count=metadata['record_count'],
                    data_checksum=metadata.get('data_checksum'),
                    status=metadata['status'],
                    notes=metadata.get('notes'),
                    created_at=datetime.utcnow(),
                )
                
                db.add(session)
                await db.commit()
                
                logger.info(f"✅ Session created: {metadata['session_id']}")
                
        except Exception as e:
            logger.error(f"❌ Session creation failed: {e}")
            raise
    
    async def _store_data_source_result(
        self,
        df: pd.DataFrame,
        session_id: str,
    ) -> None:
        """
        Store raw data as 'data_source' step result.
        
        Args:
            df: Raw OHLCV DataFrame
            session_id: Per-call session UUID
        """
        try:
            df_records = df.to_dict(orient='records')
            
            async with AsyncPostgresSessionLocal() as db:
                await store_session_step_result(
                    session_id=session_id,  # ✅ FROM PARAMETER
                    step_name='data_source',
                    data=df_records,
                    db=db,
                )
            
            logger.info(
                f"✅ Data source result stored: {len(df_records)} rows"
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to store data source result: {e}")
            raise
    
    async def _run_analysis_steps(
        self,
        df: pd.DataFrame,
        steps: List[str],
        configs: Dict[str, Any],
        session_id: str,
        task_id: str,
        user_id: Optional[str] = "anonymous",
    ) -> None:
        """
        Run optional analysis steps via ProcessingManager.
        
        
        Args:
            df: Raw data DataFrame
            steps: Analysis step names ('technical', 'snr', 'astronomical')
            configs: Config dict for each step
            session_id: Session UUID
            task_id: Task UUID
            user_id: User identifier for progress tracking
        """
        try:
            for step in steps:
                logger.info(f"🔬 Running {step}...")
                
                # Map step name to AnalysisType and Config class
                analysis_type_map = {
                    'technical': AnalysisType.TECHNICAL,
                    'snr': AnalysisType.SNR,
                    'astronomical': AnalysisType.ASTRONOMICAL,
                }
                
                config_class_map = {
                    'technical': TechnicalConfig,
                    'snr': SNRConfig,
                    'astronomical': AstronomicalConfig,
                }
                
                analysis_type = analysis_type_map.get(step)
                config_class = config_class_map.get(step)
                
                if not analysis_type:
                    logger.warning(f"Unknown analysis type: {step}")
                    continue
                
                # ─────────────────────────────────────────────────────────
                # Get step config dict and convert to proper Config object
                # ─────────────────────────────────────────────────────────
                step_config_dict = configs.get(step, {})
                
                try:
                    # ✅ FIX: Convert dict to proper Config object type
                    step_config = config_class(**step_config_dict) if step_config_dict else config_class()
                    logger.debug(f"  Built {config_class.__name__} for {step}")
                except TypeError as config_err:
                    logger.warning(f"Failed to build config for {step}: {config_err}, using defaults")
                    step_config = config_class()  # Use defaults
                
                # Create ProcessingManager for this step
                pm = ProcessingManager(
                    session_id=session_id,  # ✅ FROM PARAMETER
                    task_id=task_id,        # ✅ FROM PARAMETER
                    analysis_type=analysis_type,
                    config=step_config,  # ✅ NOW PROPER CONFIG OBJECT
                    task_store=self.task_store,
                    connection_manager=self.connection_manager,
                    processing_config=ProcessingConfig(),
                    user_id=user_id,  # ✅ Pass user_id for progress routing
                )
                
                # Extract dataset_name from config if available (for ML dataset preparation)
                execute_kwargs = {}
                if hasattr(step_config, 'dataset_name') and step_config.dataset_name:
                    execute_kwargs['dataset_name'] = step_config.dataset_name
                
                # Execute
                result = await pm.execute(df, **execute_kwargs)
                
                logger.info(f"✅ {step} completed")
                
        except Exception as e:
            logger.error(f"❌ Analysis step failed: {e}")
            raise
    
    async def _retrieve_analysis_cache(
        self,
        analysis_steps: List[str],
        task_id: str,
    ) -> Dict[str, Any]:
        """
        Retrieve cached results from analysis steps (top-level cache coordinator).
        
        AnalysisManager is at the top of the pyramid and coordinates cache access
        across all ProcessingManager steps. This ensures consistent cache retrieval.
        
        Args:
            analysis_steps: List of step names that were executed
            task_id: Per-call task UUID
            
        Returns:
            Dict mapping step names to their results
        """
        results = {}
        
        for step in analysis_steps:
            try:
                # Map step name to cache key
                cached_data = IntermediateResultsCache.retrieve(
                    task_id=task_id,  # ✅ FROM PARAMETER
                    step_name=f"{step}_analysis"
                )
                
                if cached_data is not None:
                    results[step] = {
                        'status': 'completed',
                        'cached': True,
                        'data': cached_data,
                    }
                    logger.info(f"✅ Retrieved cached {step} results from task {task_id[:8]}...")
                else:
                    logger.warning(f"⚠️ No cache found for {step} (task {task_id[:8]}...)")
                    results[step] = {
                        'status': 'completed',
                        'cached': False,
                    }
            except Exception as e:
                logger.warning(f"Cache retrieval failed for {step}: {e}")
                results[step] = {'status': 'error', 'error': str(e)}
        
        return results

    def cleanup(self):
        """
        ✅ INSTANCE CLEANUP (shared singleton resources)
        
        Note: session_id/task_id data is NOT stored as instance state,
        so there's nothing per-call to clean up.
        Only shared infrastructure (DataFetcher) is cleaned.
        """
        try:
            # Cleanup shared DataFetcher
            if self.data_fetcher:
                self.data_fetcher.cleanup()
                self.data_fetcher = None
            
            logger.debug(f"Singleton AnalysisManager cleanup complete")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    # ────────────────────────────────────────────────────────────────
    # SESSION CACHE METHODS (Class-level, session-persistent)
    # ────────────────────────────────────────────────────────────────
    
    @classmethod
    async def cache_session_data(
        cls,
        session_id: str,
        data: List[Dict],
        source_step: str,
        ttl_seconds: int = 1800
    ):
        """
        Store data in AnalysisManager cache (session-scoped, persistent).
        
        ✅ TIER 2: Cached data for next step (no database round-trip).
        
        🔴 BUG FIX #1: Use atomic setdefault to prevent race condition on lock creation
        
        Args:
            session_id: Session UUID
            data: List of dicts (session data records)
            source_step: Step that produced this data ('data_source', 'technical_analysis', etc.)
            ttl_seconds: Time-to-live in seconds (default 30 min)
        """
        # 🔴 BUG FIX #1: ATOMIC lock creation (prevents two threads creating different locks)
        lock = cls._cache_locks.setdefault(session_id, asyncio.Lock())
        
        async with lock:
            cls._session_cache[session_id] = SessionDataCache(
                data=data,
                source_step=source_step,
                ttl_seconds=ttl_seconds
            )
            
            logger.info(
                f"📌 AnalysisManager cached {len(data)} rows "
                f"from {source_step} (TTL: {ttl_seconds}s)"
            )
    
    @classmethod
    async def get_cached_data(
        cls,
        session_id: str,
        task_id: str = "unknown"
    ) -> Optional[List[Dict]]:
        """
        Retrieve data from AnalysisManager cache (TIER 2).
        Returns None if not cached or TTL expired.
        
        ✅ TIER 2: Fast retrieval (milliseconds, no DB fetch).
        
        🔴 BUG FIX #1: Use atomic setdefault to prevent race condition on lock creation
        
        Args:
            session_id: Session UUID
            task_id: Task UUID (for logging)
            
        Returns:
            List of dicts if cached and valid, None otherwise
        """
        # 🔴 BUG FIX #1: ATOMIC lock creation (prevents two threads creating different locks)
        lock = cls._cache_locks.setdefault(session_id, asyncio.Lock())
        
        # ALL operations inside lock (get + TTL check + pop)
        async with lock:
            cache_entry = cls._session_cache.get(session_id)
            
            if not cache_entry:
                logger.debug(f"Cache miss for session {session_id[:8]}... (task {task_id[:8]}...)")
                return None
            
            # Check TTL (still inside lock)
            elapsed = cache_entry.get_age_seconds()
            if cache_entry.is_expired():
                logger.info(
                    f"Cache expired for session {session_id[:8]}... "
                    f"({elapsed:.0f}s > {cache_entry.ttl_seconds}s)"
                )
                cls._session_cache.pop(session_id, None)
                return None
            
            logger.info(
                f"✅ Cache HIT for session {session_id[:8]}...: "
                f"Retrieved {cache_entry.n_rows} rows from {cache_entry.source_step} "
                f"(elapsed: {elapsed:.1f}s)"
            )
            
            return cache_entry.data
    
    @classmethod
    def clear_cache(cls, session_id: str):
        """
        Clear cache for a session (e.g., when user resets analysis).
        Also cleans up the associated lock to prevent memory leaks (BUG FIX #1).
        Forces garbage collection to free memory.
        
        Args:
            session_id: Session UUID
        """
        cls._session_cache.pop(session_id, None)
        cls._session_data_pointers.pop(session_id, None)
        cls._cache_locks.pop(session_id, None)  # BUG FIX #1: Clean up orphaned lock
        gc.collect()  # 🧹 Force cleanup to prevent memory bloat
        logger.info(f"Cache cleared for session {session_id[:8]}...")

    async def reinitialize_session(self, session_id: str, task_id: str) -> pd.DataFrame:
        """
        Force re-load of session data into memory (TIER 0 cache).
        Used when a user selects an existing session in the frontend.
        
        Args:
            session_id: Session ID to load
            task_id: Active task ID for progress reporting
            
        Returns:
            The loaded DataFrame
        """
        # Ensure we have an initialization lock for this session
        init_lock = self._cache_locks.setdefault(f"init_{session_id}", asyncio.Lock())
        
        async with init_lock:
            logger.info(f"🔄 [AnalysisManager] Re-initializing session {session_id[:8]}...")
            
            # ✅ FIX: Check TIER 0a first — if this session was recently loaded, skip the DB entirely.
            # Without this guard, every call triggers a full TIER 3 DB fetch even when data is warm.
            if self.current_data is not None and self.current_session_id == session_id:
                logger.info(
                    f"⚡ [AnalysisManager] Session {session_id[:8]} already warm in TIER 0a "
                    f"({len(self.current_data)} rows) — skipping re-initialization"
                )
                return self.current_data.copy()

            # The instance-local pointer above is insufficient when requests use
            # different AnalysisManager instances. Reuse the session pointer that
            # was populated by the previous initializer after waiting on init_lock.
            session_data = self._session_data_pointers.get(session_id)
            if session_data is not None:
                self.current_data = session_data
                self.current_session_id = session_id
                logger.info(
                    f"⚡ [AnalysisManager] Session {session_id[:8]} already warm in "
                    f"shared TIER 0 ({len(session_data)} rows) — skipping re-initialization"
                )
                return session_data.copy()
            
            df, source = await self._load_data_4_tier(
                session_id=session_id,
                task_id=task_id,
                data_type="analysis"
            )
            
            # ✅ FIX: Populate TIER 0a pointer so the next call returns instantly
            # without falling through to TIER 2/3. This is the primary guard against
            # the re-initialization loop seen in session c6d3e641.
            self.current_data = df
            self.current_session_id = session_id
            self._session_data_pointers[session_id] = df
            
            logger.info(f"✅ [AnalysisManager] Session {session_id[:8]} re-initialized from {source} ({len(df)} rows)")
            return df
    
    @classmethod
    def cleanup_expired_cache(cls) -> int:
        """
        Remove all expired cache entries and return count of entries removed.
        Called periodically to prevent long-running sessions from accumulating stale data.
        
        Returns:
            Number of cache entries removed
        """
        expired_sessions = []
        
        for session_id, cache_entry in list(cls._session_cache.items()):
            # ✅ FIX: Use cached_at (not created_at) - SessionDataCache attribute
            if cache_entry.is_expired():
                expired_sessions.append(session_id)
                cls._session_cache.pop(session_id, None)
        
        # Also clean up locks for expired sessions
        for session_id in expired_sessions:
            cls._cache_locks.pop(session_id, None)
        
        if expired_sessions:
            gc.collect()
            logger.info(f"🧹 Cleaned up {len(expired_sessions)} expired cache entries")
        
        return len(expired_sessions)
    
    @classmethod
    async def get_cache_stats(cls) -> Dict[str, Any]:
        """
        Monitor cache health and statistics.
        
        Returns:
            Dict with cache stats (total sessions, per-session details)
        """
        now = time.time()
        stats = {
            "total_sessions_cached": len(cls._session_cache),
            "sessions": {}
        }
        
        for session_id, entry in cls._session_cache.items():
            stats["sessions"][session_id[:8] + "..."] = {
                "rows": entry.n_rows,
                "source_step": entry.source_step,
                "age_seconds": now - entry.cached_at,
                "ttl_seconds": entry.ttl_seconds,
                "expired": entry.is_expired()
            }
        
        stats["total_locks"] = len(cls._cache_locks)
        return stats
    
    @classmethod
    async def cleanup_cache_maintenance(cls) -> Dict[str, Any]:
        """
        🔧 MAINTENANCE: Clean up expired cache entries and orphaned locks (BUG FIX #1 & #4)
        """
        now = time.time()
        removed_entries = 0
        removed_locks = 0
        
        # Clean up expired cache entries
        expired_sessions = [
            sid for sid, entry in cls._session_cache.items()
            if entry.is_expired()
        ]
        
        for session_id in expired_sessions:
            try:
                cls._session_cache.pop(session_id, None)
                removed_entries += 1
                logger.debug(f"Cache cleanup: Removed expired entry for {session_id[:8]}...")
            except Exception as e:
                logger.warning(f"Failed to cleanup cache entry {session_id[:8]}...: {e}")
        
        # Clean up orphaned locks (sessions no longer in cache)
        orphaned_locks = []
        for session_id in list(cls._cache_locks.keys()):
            if session_id not in cls._session_cache:
                orphaned_locks.append(session_id)
        
        for session_id in orphaned_locks:
            try:
                cls._cache_locks.pop(session_id, None)
                removed_locks += 1
                logger.debug(f"Cache cleanup: Removed orphaned lock for {session_id[:8]}...")
            except Exception as e:
                logger.warning(f"Failed to cleanup lock {session_id[:8]}...: {e}")
        
        stats = {
            "removed_cache_entries": removed_entries,
            "removed_locks": removed_locks,
            "remaining_cache_entries": len(cls._session_cache),
            "remaining_locks": len(cls._cache_locks),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if removed_entries > 0 or removed_locks > 0:
            logger.info(
                f"✅ Cache maintenance complete: "
                f"removed {removed_entries} expired entries, {removed_locks} orphaned locks"
            )
        
        return stats

    # ────────────────────────────────────────────────────────────────
    # UNDO STEP - Session Version Control
    # ────────────────────────────────────────────────────────────────
    
    async def undo_step(
        self,
        session_id: str,
        task_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Undo the current analysis step by:
        1. Finding the most enriched step (is_current_data=TRUE)
        2. Deleting it from the database
        3. Restoring the previous step as current_data
        
        ✅ ARCHITECTURE: Implements version control for analysis pipeline
        
        Args:
            session_id: Session UUID
            task_id: Task UUID for logging
            
        Returns:
            Result dict with:
            - status: 'success' or 'error'
            - previous_step: Name of the step that was restored
            - data_rows: Number of rows in restored data
            - message: Human-readable status
            
        Raises:
            HTTPException: If no step to undo or restoration fails
        """
        try:
            logger.info(f"🔙 Undo: Finding current step for session {session_id[:8]}...")
            
            # ─────────────────────────────────────────────────────────
            # STEP 1: Find current step (is_current_data=TRUE)
            # ─────────────────────────────────────────────────────────
            async with AsyncPostgresSessionLocal() as db:
                # Query for the step marked as current
                stmt = select(SessionStepResult).where(
                    and_(
                        SessionStepResult.session_id == session_id,
                        SessionStepResult.is_current_data == True
                    )
                )
                result = await db.execute(stmt)
                current_step = result.scalar_one_or_none()
                
                if not current_step:
                    logger.warning(f"No current step found for session {session_id[:8]}...")
                    return {
                        "status": "error",
                        "message": "No current step to undo (are you at the beginning?)",
                        "current_step": None,
                        "previous_step": None
                    }
                
                current_step_name = current_step.step_name
                logger.info(f"🔍 Current step: {current_step_name}")
                
                # ─────────────────────────────────────────────────────────
                # STEP 2: Delete current step (soft or hard delete)
                # ─────────────────────────────────────────────────────────
                await db.delete(current_step)
                await db.flush()  # Flush before commit to ensure deletion is registered
                logger.info(f"🗑️  Deleted step: {current_step_name}")
                
                # ─────────────────────────────────────────────────────────
                # STEP 3: Walk priority chain backwards to find previous step
                # ─────────────────────────────────────────────────────────
                # Priority order (most enriched to least enriched)
                step_priority = [
                    'astronomical_analysis',
                    'snr_analysis',
                    'technical_analysis',
                    'currency_indices',
                    'data_source'
                ]
                
                # Find current step's index
                try:
                    current_idx = step_priority.index(current_step_name)
                except ValueError:
                    # Current step is not in priority list (edge case - custom step)
                    logger.warning(f"Current step {current_step_name} not in priority list, searching backwards by timestamp")
                    current_idx = 0  # Default to earliest
                
                # Walk backwards in priority to find previous step
                previous_step = None
                previous_step_name = None
                
                for i in range(current_idx + 1, len(step_priority)):
                    prev_name = step_priority[i]
                    logger.debug(f"  Checking for previous step: {prev_name}")
                    
                    stmt_prev = select(SessionStepResult).where(
                        and_(
                            SessionStepResult.session_id == session_id,
                            SessionStepResult.step_name == prev_name
                        )
                    )
                    result_prev = await db.execute(stmt_prev)
                    potential_prev = result_prev.scalar_one_or_none()
                    
                    if potential_prev:
                        previous_step = potential_prev
                        previous_step_name = prev_name
                        logger.info(f"✅ Found previous step: {prev_name}")
                        break
                
                if not previous_step:
                    logger.error(f"No previous step found for session {session_id[:8]}... after deleting {current_step_name}")
                    # Commit the deletion even though we couldn't restore
                    await db.commit()
                    
                    return {
                        "status": "partial_success",
                        "message": f"Deleted {current_step_name}, but no previous step found",
                        "current_step": current_step_name,
                        "previous_step": None,
                        "data_rows": 0
                    }
                
                # ─────────────────────────────────────────────────────────
                # STEP 4: Mark previous step as current_data
                # ─────────────────────────────────────────────────────────
                from app.core.data.session_data_loader import set_as_current_data
                
                await set_as_current_data(
                    session_id=session_id,
                    db=db,
                    task_id=task_id
                )
                
                # Commit atomically
                await db.commit()
                
                logger.info(f"✅ Restored {previous_step_name} as current_data")
                
                # ─────────────────────────────────────────────────────────
                # STEP 5: Calculate row count for response
                # ─────────────────────────────────────────────────────────
                try:
                    data_rows = 0
                    if previous_step.result_data or previous_step.result_data_v2:
                        # Deserialize to count rows
                        if previous_step.result_data_v2:
                            # JSONB format
                            raw_data = previous_step.result_data_v2
                        else:
                            # Pickle format
                            raw_data = deserialize_data(previous_step.result_data, previous_step.is_compressed)
                        
                        # Extract data array
                        if isinstance(raw_data, dict) and 'data' in raw_data:
                            data_rows = len(raw_data['data'])
                        elif isinstance(raw_data, list):
                            data_rows = len(raw_data)
                        else:
                            data_rows = 1
                except Exception as e:
                    logger.warning(f"Could not calculate row count: {e}")
                    data_rows = 0
                
                # ─────────────────────────────────────────────────────────
                # STEP 6: Clear cache to force fresh load on next request
                # ─────────────────────────────────────────────────────────
                self.clear_cache(session_id)
                logger.info(f"Cache cleared for session {session_id[:8]}...")
                
                return {
                    "status": "success",
                    "message": f"Undone {current_step_name}, restored {previous_step_name}",
                    "current_step": current_step_name,
                    "previous_step": previous_step_name,
                    "data_rows": data_rows,
                    "restored_at": datetime.utcnow().isoformat()
                }
        
        except Exception as e:
            logger.error(f"❌ Undo failed: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Undo failed: {str(e)}",
                "error_details": str(e)
            }

    
    def _preserve_nan_values(self, data: Union[List[Dict], Dict]) -> Union[List[Dict], Dict]:
        """
        
        Converts np.nan to float('nan') which survives serialization/deserialization.
        This prevents technical indicator calculations from breaking due to None values.
        
        Args:
            data: List of dicts or a single dict with potential NaN values
            
        Returns:
            Same structure with NaN values explicitly preserved
        """
        
        
        # Guard: handle single dict
        if isinstance(data, dict):
            for key, value in data.items():
                try:
                    if pd.isna(value) and not isinstance(value, str):
                        data[key] = float('nan')
                except (TypeError, ValueError):
                    pass
            return data

        # Guard: handle list of dicts
        if not isinstance(data, list):
            logger.warning(f"_preserve_nan_values received non-list/non-dict ({type(data).__name__}), skipping NaN pass")
            return data
        
        for record in data:
            if not isinstance(record, dict):
                continue  # skip malformed records rather than crashing
            for key, value in record.items():
                try:
                    # If value is NaN, preserve as float('nan') instead of None
                    if pd.isna(value) and not isinstance(value, str):
                        record[key] = float('nan')
                except (TypeError, ValueError):
                    pass  # pd.isna raises on non-scalars (arrays) \u2014 skip safely
        
        return data
    
    # ────────────────────────────────────────────────────────────────
    # DATAFRAME PREPARATION HELPERS (INTEGRATED FROM analysis_route_helpers)
    # ────────────────────────────────────────────────────────────────
    
    def standardize_dataframe_columns(self, df: pd.DataFrame, require_ohlc: bool = True) -> pd.DataFrame:
        """
        Standardize OHLC column names to consistent format.
        ⚠️ OPTIMIZED: Early exit if already standardized (DataFetcher already normalizes)
        
        Args:
            df: Input DataFrame
            require_ohlc: If True, raises ValueError if OHLC columns are missing.
                          Set to False for steps like Astronomical that only need timestamps.
        """
        # ✅ OPTIMIZATION: Early exit if already standardized
        required_ohlc = {'Open', 'High', 'Low', 'Close'}
        actual_cols = set(df.columns)
        
        if required_ohlc.issubset(actual_cols):
            logger.debug(f"⚡ Columns already standardized, skipping standardization (0ms)")
            return df
        
        STANDARD_MAPPING = {
            'o': 'Open', 'open': 'Open', 'OPEN': 'Open',
            'h': 'High', 'high': 'High', 'HIGH': 'High',
            'l': 'Low', 'low': 'Low', 'LOW': 'Low',
            'c': 'Close', 'close': 'Close', 'CLOSE': 'Close',
            'v': 'Volume', 'vol': 'Volume', 'volume': 'Volume', 'VOLUME': 'Volume',
            'tick_volume': 'TickVolume', 'tick_vol': 'TickVolume', 'tickvolume': 'TickVolume',
            't': 'Time', 'time': 'Time', 'timestamp': 'Time',
        }
        
        lowercase_actual_cols = {c.lower(): c for c in df.columns}
        rename_map = {}
        
        for norm_key, standard_name in STANDARD_MAPPING.items():
            if norm_key in lowercase_actual_cols:
                actual_col = lowercase_actual_cols[norm_key]
                rename_map[actual_col] = standard_name
        
        df_renamed = df.rename(columns=rename_map)
        actual_cols = set(df_renamed.columns)
        missing = required_ohlc - actual_cols
        
        if missing and require_ohlc:
            raise ValueError(f"Missing required OHLC columns: {missing}")
        elif missing:
            logger.debug(f"ℹ️ Skipping OHLC requirement check (require_ohlc=False). Missing: {missing}")
        
        logger.debug(f"✅ Standardized columns: {rename_map}")
        return df_renamed
    
    def ensure_volume_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure proper volume handling for technical indicators.
        
        Rules:
        1. If 'Volume' exists with values → use it as-is
        2. Else if 'TickVolume' exists → use it
        3. Else → create synthetic volume = 1.0
        """
        if 'Volume' in df.columns and (df['Volume'] > 0).any():
            logger.debug("Using existing Volume column")
            return df
        
        elif 'TickVolume' in df.columns:
            logger.debug("Using TickVolume as Volume")
            df['Volume'] = df['TickVolume']
            return df
        
        else:
            logger.warning("Creating synthetic volume (all 1.0)")
            df['Volume'] = 1.0
            return df
    
    def prepare_dataframe(self, df: pd.DataFrame, config: Optional[Any] = None, require_ohlc: bool = True) -> pd.DataFrame:
        """
        Prepare DataFrame for analysis (column standardization + volume handling).
        """
        df = self.standardize_dataframe_columns(df, require_ohlc=require_ohlc)
        df = self.ensure_volume_column(df)
        return df
    
    # ────────────────────────────────────────────────────────────────
    # ANALYSIS EXECUTION HELPERS (INTEGRATED FROM analysis_route_helpers)
    # ────────────────────────────────────────────────────────────────
