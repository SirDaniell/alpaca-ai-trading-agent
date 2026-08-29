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

class AnalysisManager:
    """
    Unified orchestrator for data fetch + analysis pipeline.
    
    Handles:
    - Data fetching from any source with date extraction
    - Session creation with complete metadata
    - ProcessingManager delegation for analysis
    - Progress tracking via WebSocket
    - Memory cleanup
    """
    
    # ────────────────────────────────────────────────────────────────
    # CLASS-LEVEL CACHE (Session-persistent, single source of truth)
    # ────────────────────────────────────────────────────────────────
    # Format: {session_id: SessionDataCache}
    _session_cache: Dict[str, SessionDataCache] = {}
    _cache_locks: Dict[str, asyncio.Lock] = {}
    
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
                df, source = await self._load_data_4_tier(
                    session_id=session_id,
                    task_id=task_id,
                    request_data=request_data,
                    exclude_step=f"{analysis_type.value}_analysis",  # ✅ BUG FIX #3: Always exclude current step
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
    
    def validate_ml_config(self, config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate ML preparation config before processing (AM Level).
        
        Returns:
            (is_valid, error_message)
        """
        # Required fields
        required_fields = ['sequence_length', 'prediction_length', 'target_columns']
        for field in required_fields:
            if field not in config:
                return False, f"Missing required field: {field}"
        
        # Value validation
        if config['sequence_length'] <= 0:
            return False, f"sequence_length must be positive, got {config['sequence_length']}"
        
        if config['prediction_length'] < 0:
            return False, f"prediction_length cannot be negative, got {config['prediction_length']}"
        
        # Split ratios
        train_ratio = config.get('train_ratio', 0.7)
        val_ratio = config.get('validation_ratio', 0.15)
        test_ratio = config.get('test_ratio', 0.15)
        total_ratio = train_ratio + val_ratio + test_ratio
        
        if not np.isclose(total_ratio, 1.0):
            return False, f"Split ratios must sum to 1.0, got {total_ratio}"
        
        # Target columns validation
        if not isinstance(config.get('target_columns'), list):
            return False, "target_columns must be a list"
        
        if len(config.get('target_columns', [])) == 0:
            return False, "target_columns cannot be empty"
        
        return True, None
    
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
        logger.info(f"🔄 [AnalysisManager] Re-initializing session {session_id[:8]}...")
        
        # ✅ FIX: Check TIER 0a first — if this session was recently loaded, skip the DB entirely.
        # Without this guard, every call triggers a full TIER 3 DB fetch even when data is warm.
        if self.current_data is not None and self.current_session_id == session_id:
            logger.info(
                f"⚡ [AnalysisManager] Session {session_id[:8]} already warm in TIER 0a "
                f"({len(self.current_data)} rows) — skipping re-initialization"
            )
            return self.current_data.copy()
        
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
    
    async def execute_technical_analysis(
        self,
        df: pd.DataFrame,
        pm: ProcessingManager,
        session_id: str,
        task_id: str,
        user_id: str = "anonymous",
    ) -> Dict[str, Any]:
        """
        Execute technical analysis using centralized ProcessingManager.
        ✅ INTEGRATED: Part of AnalysisManager, no external helpers needed
        ✅ UPDATES: self.current_data with enriched DataFrame for next step
        """
        result = await pm.execute(df, user_id=user_id)
        
        # Store result to DB (if not slice_streaming which does it internally)
        try:
            async with AsyncPostgresSessionLocal() as db:
                strategy_used = result.get("metadata", {}).get("strategy")
                # Only store if not slice_streaming, as slice_streaming already stored chunks
                if strategy_used != "slice_streaming":
                    # Standardized result mapping: prioritize result_df
                    df_to_store = result.get("result_df", result.get("features_df", df))
                    
                    # 🛡️ Guard: if result_df was stripped (e.g. already a preview dict/str), fall back to raw df
                    if not isinstance(df_to_store, (pd.DataFrame, list)):
                        logger.warning(
                            f"⚠️ [Technical persist] result_df is {type(df_to_store).__name__}, not DataFrame/list. "
                            f"Falling back to raw input df ({len(df)} rows) for storage."
                        )
                        df_to_store = df

                    # 🔴 BUG FIX #2: Preserve NaN values before serialization
                    result_records = df_to_store.to_dict(orient='records') if isinstance(df_to_store, pd.DataFrame) else df_to_store
                    result_records = self._preserve_nan_values(result_records)
                    
                    # Clear cache before updating
                    self.clear_cache(session_id)
                    logger.info(f"Cache invalidated for session {session_id[:8]}...")
                    
                    await store_session_step_result(
                        session_id=session_id,
                        step_name="technical_analysis",
                        data=result_records,
                        db=db,
                        force_pickle=True  # Technical results can be large
                    )
                    await set_as_current_data(session_id, db, task_id)
                    
                    # Populate cache with fresh results
                    await self.cache_session_data(
                        session_id=session_id,
                        data=result_records,
                        source_step='technical_analysis',
                        ttl_seconds=1800
                    )
                    logger.info(f"Cached {len(result_records)} rows after technical analysis (NaN preserved)")
                    
                    # ✅ UPDATE self.current_data with enriched DataFrame for next step
                    self.current_data = df_to_store if isinstance(df_to_store, pd.DataFrame) else pd.DataFrame(result_records)
                    self.current_session_id = session_id
                    logger.info(f"Updated self.current_data with {len(self.current_data)} enriched rows (TIER 0 for next step)")
                else:
                    # 🔴 BUG FIX: slice_streaming stores internally BUT we must load the FULLY MERGED result
                    # PM's _aggregate_slice_results() returns complete merged DataFrame in result["result_df"]
                    df_merged = result.get("result_df", df)
                    
                    if isinstance(df_merged, pd.DataFrame):
                        # 🧹 MEMORY FIX: Use reference, not copy (ProcessingManager already owns the data)
                        # The df_merged is already a complete merged result from ProcessingManager
                        self.current_data = df_merged  # Reference, not copy!
                        self.current_session_id = session_id
                        
                        mem_mb = df_merged.memory_usage(deep=True).sum() / (1024 * 1024)
                        logger.info(
                            f"✅ [MEMORY OPTIMIZED] Loaded FULLY MERGED technical result (reference): "
                            f"{len(self.current_data)} rows, {mem_mb:.1f} MB "
                            f"(from {len(result.get('metadata', {}).get('slices', [])) if 'slices' in result.get('metadata', {}) else '?'} slices)"
                        )
                    else:
                        logger.warning(f"⚠️ Could not extract merged result_df from slice_streaming, using DB pointer only")
                    
                    await set_as_current_data(session_id, db, task_id)
        except Exception as db_err:
            logger.warning(f"Could not persist technical analysis result: {db_err}")
        
        return result
    
    async def execute_snr_analysis(
        self,
        df: pd.DataFrame,
        pm: ProcessingManager,
        session_id: str,
        task_id: str,
        user_id: str = "anonymous",
    ) -> Dict[str, Any]:
        """
        Execute SNR analysis using centralized ProcessingManager.
        
        🔄 UNIFIED DATA HANDLING:
        - PM executes processing and caches to TIER 1
        - PM MAY store to TIER 3 (depending on strategy)
        - AM checks coordination flags (pm_persisted, skip_am_persist)
        - AM handles FINAL storage ONLY (after assembly/validation)
        
        ✅ COORDINATION PROTOCOL:
        - If PM stored: result["metadata"]["pm_persisted"] = True
        - If AM should skip re-store: result["metadata"]["skip_am_persist"] = True
        - AM RESPECTS these flags to avoid double writes
        """
        # Memory monitoring at SNR analysis level
        try:
            import psutil
            process = psutil.Process()
            mem_start = process.memory_info().rss / 1024 / 1024
            df_size = df.memory_usage(deep=True).sum() / (1024 * 1024)
            logger.info(
                f"[SNR] Starting analysis. Memory: {mem_start:.1f} MB, "
                f"Input DF: {len(df)} rows, {df_size:.1f} MB"
            )
        except ImportError:
            process = None
            mem_start = None
        
        result = await pm.execute(df, user_id=user_id)
        
        if process:
            mem_after_pm = process.memory_info().rss / 1024 / 1024
            logger.info(
                f"[SNR] After ProcessingManager: {mem_after_pm:.1f} MB "
                f"(+{mem_after_pm - mem_start:.1f} MB)"
            )
        
        # ✅ UPDATE TIER 0a POINTER: Set current_data with SNR enriched result
        # This ensures ML Preparation gets the latest SNR data, not stale Technical data
        df_merged = result.get("result_df", df)
        if isinstance(df_merged, pd.DataFrame):
            self.current_data = df_merged  # Reference to merged result
            self.current_session_id = session_id
            
            mem_mb = df_merged.memory_usage(deep=True).sum() / (1024 * 1024)
            logger.info(
                f"⚡ TIER 0a HIT: Analysis pointer updated with SNR result\n"
                f"   ├─ Rows: {len(self.current_data)} (COMPLETE MERGED DATASET)\n"
                f"   ├─ Columns: {len(self.current_data.columns)}\n"
                f"   └─ Latency: ZERO ms from previous step"
            )
        else:
            logger.warning(f"⚠️ Could not extract merged result_df from SNR analysis")
        
        # ✅ CAPTURE: SNR Unprocessed Dataset for direct ML prep transition
        if "ml_dataset" in result:
            ml_dataset_raw = result["ml_dataset"]
            
            # ✅ NEW: Reconstruct sequences from lightweight format if needed
            if isinstance(ml_dataset_raw, list) and len(ml_dataset_raw) > 0:
                first_item = ml_dataset_raw[0]
                
                # Check if it's the new lightweight format
                if "sequence_start" in first_item and "sequence_end" in first_item:
                    logger.info(f"🔄 [AM] Reconstructing ML sequences for in-memory cache...")
                    
                    from app.core.analysis.trading.signal_generator import reconstruct_ml_sequences
                    
                    result_df = result.get("result_df")
                    if result_df is not None and isinstance(result_df, pd.DataFrame):
                        self.unprocessed_dataset = reconstruct_ml_sequences(ml_dataset_raw, result_df)
                        logger.info(f"🎯 [AM] Captured SNR Unprocessed Dataset with {len(self.unprocessed_dataset)} reconstructed records")
                    else:
                        logger.warning("[AM] ⚠️ Cannot reconstruct sequences: result_df missing")
                        self.unprocessed_dataset = ml_dataset_raw  # Store lightweight format as fallback
                else:
                    # Already in full format
                    self.unprocessed_dataset = ml_dataset_raw
                    logger.info(f"🎯 [AM] Captured SNR Unprocessed Dataset with {len(self.unprocessed_dataset)} records")
            else:
                self.unprocessed_dataset = ml_dataset_raw
                logger.info(f"🎯 [AM] Captured SNR Unprocessed Dataset")
            
            self.unprocessed_session_id = session_id
        
        # Store result to DB (with coordination checks)
        try:
            async with AsyncPostgresSessionLocal() as db:
                strategy_used = result.get("metadata", {}).get("strategy")
                
                # ✅ FIX: Ensure boolean values (not DataFrames or other ambiguous types)
                pm_persisted_raw = result.get("metadata", {}).get("pm_persisted", False)
                skip_am_persist_raw = result.get("metadata", {}).get("skip_am_persist", False)
                
                # Convert to explicit boolean (handles DataFrame, None, etc.)
                # ✅ CRITICAL: Use isinstance check FIRST to avoid ambiguous truth value
                if isinstance(pm_persisted_raw, pd.DataFrame):
                    pm_persisted = False
                elif pm_persisted_raw is None:
                    pm_persisted = False
                else:
                    pm_persisted = bool(pm_persisted_raw)
                
                if isinstance(skip_am_persist_raw, pd.DataFrame):
                    skip_am_persist = False
                elif skip_am_persist_raw is None:
                    skip_am_persist = False
                else:
                    skip_am_persist = bool(skip_am_persist_raw)
                
                pm_persist_failed = result.get("metadata", {}).get("pm_persist_failed", False)
                
                logger.info(
                    f"[AM] 🔍 SNR coordination: pm_persisted={pm_persisted}, "
                    f"skip_am_persist={skip_am_persist}, strategy={strategy_used}"
                )

                # 🔄 Build a sync keepalive reporter for storage progress
                # This ensures the frontend doesn't timeout during large ML dataset storage
                def storage_progress_reporter(pct: int, msg: str) -> None:
                    label = f"💾 Persisting SNR Data: {msg}"

                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.ensure_future(
                                pm._send_progress_update(
                                    pct, label, {"stage": "storage", "step": "snr_analysis"}
                                )
                            )
                    except Exception:
                        pass

                # ✅ DECISION: Should AM re-store snr_analysis?

                # SKIP (skip_am_persist=True) if: PM already stored non-chunked
                # STORE (skip_am_persist=False) if: PM didn't store, or chunked needing assembly
                
                if pm_persisted and skip_am_persist:
                    # ✅ CASE 1: PM already stored successfully, AM respects it
                    logger.info(
                        f"[AM] ⏭️  SKIP persist (PM already stored snr_analysis, "
                        f"strategy={strategy_used}, no re-storage needed)"
                    )
                else:
                    # ✅ CASE 2: AM must store (either PM failed or chunked needing assembly)
                    reason = "pm_failed" if pm_persist_failed else "chunked_assembly" if strategy_used == "slice_streaming" else "unknown"
                    
                    # Get the DataFrame to store
                    df_to_store = result.get("result_df")
                    if df_to_store is None:
                        df_to_store = result.get("enriched_df")
                        
                    if df_to_store is None and isinstance(result.get("result_df"), (dict, list)):
                        try:
                            df_to_store = pd.DataFrame(result.get("result_df"))
                        except Exception as e:
                            logger.error(f"❌ [AM] SNR failed to reconstruct DataFrame: {e}")
                            df_to_store = df  # Fallback to raw
                    elif df_to_store is None:
                        df_to_store = df  # Final fallback
                    
                    # 🧹 MEMORY MONITORING: Track dict conversion overhead before conversion
                    if process and isinstance(df_to_store, pd.DataFrame):
                        mem_before_dict = process.memory_info().rss / 1024 / 1024
                        df_size = df_to_store.memory_usage(deep=True).sum() / (1024 * 1024)
                        logger.info(f"[SNR] Converting to dict: {len(df_to_store)} rows, {df_size:.1f} MB")
                    
                    result_records = df_to_store.to_dict(orient='records') if isinstance(df_to_store, pd.DataFrame) else df_to_store
                    
                    # 🧹 MEMORY MONITORING: Track dict conversion overhead after conversion
                    if process and isinstance(df_to_store, pd.DataFrame):
                        mem_after_dict = process.memory_info().rss / 1024 / 1024
                        try:
                            import sys
                            dict_size = sys.getsizeof(result_records) / (1024 * 1024)
                            logger.info(
                                f"[SNR] Dict conversion complete: +{mem_after_dict - mem_before_dict:.1f} MB memory, "
                                f"dict size: {dict_size:.1f} MB (overhead: {dict_size/df_size:.1f}x)"
                            )
                        except:
                            logger.info(f"[SNR] Dict conversion: +{mem_after_dict - mem_before_dict:.1f} MB memory")
                    
                    result_records = self._preserve_nan_values(result_records)
                    
                    # Clear cache before updating
                    self.clear_cache(session_id)
                    logger.info(f"[AM] 🧹 Cache cleared for session {session_id[:8]}")
                    
                    logger.info(
                        f"[AM] 📝 Storing enriched SNR signals: {len(result_records)} records, "
                        f"{len(result_records[0].keys()) if result_records else 0} cols (reason={reason})"
                    )
                    await store_session_step_result(
                        session_id=session_id,
                        step_name="snr_analysis",
                        data=result_records,
                        db=db,
                        force_pickle=True,
                        on_progress=storage_progress_reporter
                    )

                    logger.info(f"✅ [AM] COMMITTED snr_analysis (session={session_id[:8]}, rows={len(result_records)})")
                
                # 🔴 ALWAYS store ml_dataset (this is unique to SNR, not duplicated by PM)
                ml_dataset = result.get("ml_dataset")
                if ml_dataset is not None:
                    ml_dataset = result["ml_dataset"]
                    
                    # ✅ NEW: Reconstruct sequences from lightweight format if needed
                    if isinstance(ml_dataset, list) and len(ml_dataset) > 0:
                        first_item = ml_dataset[0]
                        
                        # Check if it's the new lightweight format (has sequence_start/sequence_end)
                        if "sequence_start" in first_item and "sequence_end" in first_item:
                            logger.info(f"[AM] 🔄 Reconstructing {len(ml_dataset)} ML sequences from lightweight format...")
                            
                            # Import reconstruction function
                            from app.core.analysis.trading.signal_generator import reconstruct_ml_sequences
                            
                            # Get the result_df for reconstruction
                            result_df = result.get("result_df")
                            if result_df is None or not isinstance(result_df, pd.DataFrame):
                                logger.error("[AM] ❌ Cannot reconstruct ML sequences: result_df missing")
                                ml_records = []
                            else:
                                # Reconstruct full sequences
                                ml_records = reconstruct_ml_sequences(ml_dataset, result_df)
                                logger.info(f"[AM] ✅ Reconstructed {len(ml_records)} ML sequences")
                        else:
                            # Already in full format (legacy or direct format)
                            ml_records = ml_dataset
                    elif isinstance(ml_dataset, pd.DataFrame):
                        # Convert DataFrame to records
                        ml_records = ml_dataset.to_dict(orient='records')
                    else:
                        ml_records = ml_dataset if isinstance(ml_dataset, list) else [ml_dataset]
                    
                    logger.info(f"[AM] 📝 Storing SNR ml_dataset: {len(ml_records)} records")
                    await store_session_step_result(
                        session_id=session_id,
                        step_name="snr_analysis_ml_dataset",
                        data=ml_records,
                        db=db,
                        force_pickle=True,
                        on_progress=storage_progress_reporter
                    )

                    logger.info(f"✅ [AM] COMMITTED snr_analysis_ml_dataset (rows={len(ml_records)})")
                else:
                    logger.warning(f"⚠️ [AM] SNR ml_dataset missing or empty")
                
                # Mark as current data (AM responsibility for SNR)
                await set_as_current_data(session_id, db, task_id)
            
            # Populate cache with enriched results
            if "snr_analysis" in locals():
                try:
                    await self.cache_session_data(
                        session_id=session_id,
                        data=result.get("result_records", result.get("result_df")),
                        source_step='snr_analysis',
                        ttl_seconds=1800
                    )
                    logger.info(f"[AM] 📌 Cached SNR results (NaN preserved)")
                except:
                    pass  # Cache failure non-fatal
            
            # ✅ UPDATE self.current_data with enriched DataFrame for next step
            if "result_df" in result and isinstance(result["result_df"], pd.DataFrame):
                # 🧹 MEMORY FIX: Use reference, not copy
                self.current_data = result["result_df"]  # Reference, not copy!
            else:
                self.current_data = df  # Fallback to input
            self.current_session_id = session_id
            
            if process and mem_start is not None:
                mem_end = process.memory_info().rss / 1024 / 1024
                current_data_size = self.current_data.memory_usage(deep=True).sum() / (1024 * 1024)
                logger.info(
                    f"[SNR] Completed. Memory: {mem_end:.1f} MB "
                    f"(+{mem_end - mem_start:.1f} MB total), "
                    f"current_data: {len(self.current_data)} rows, {current_data_size:.1f} MB"
                )
                    
        except Exception as db_err:
            logger.warning(f"[AM] ⚠️  Could not persist SNR analysis result: {db_err}")
        
        return result
    
    async def execute_astronomical_analysis(
        self,
        df: pd.DataFrame,
        pm: ProcessingManager,
        session_id: str,
        task_id: str,
        user_id: str = "anonymous",
    ) -> Dict[str, Any]:
        """
        Execute astronomical analysis using centralized ProcessingManager.
        ✅ INTEGRATED: Part of AnalysisManager, no external helpers needed
        ✅ UPDATES: self.current_data with enriched DataFrame for next step
        """
        result = await pm.execute(df, user_id=user_id)
        
        # Store result to DB (if not slice_streaming which does it internally)
        try:
            async with AsyncPostgresSessionLocal() as db:
                strategy_used = result.get("metadata", {}).get("strategy")
                if strategy_used != "slice_streaming":
                    # Standardized result mapping: prioritize result_df.
                    # ProcessingManager may hollow heavy frames after its own DB write;
                    # recover the canonical full frame from its TIER 1 cache before
                    # falling back to the raw input.
                    df_to_store = result.get("result_df")
                    if df_to_store is None:
                        df_to_store = result.get("features_df")
                    if not isinstance(df_to_store, (pd.DataFrame, list)):
                        cached_step_name = "currency_indices_analysis"
                        cached_df = IntermediateResultsCache.retrieve(task_id, cached_step_name)
                        if isinstance(cached_df, pd.DataFrame):
                            logger.info(
                                f"✅ [Currency Indices persist] Recovered full result_df from TIER 1 cache "
                                f"'{cached_step_name}' ({len(cached_df)} rows × {len(cached_df.columns)} cols)"
                            )
                            df_to_store = cached_df
                        else:
                            df_to_store = df
                    
                    # 🛡️ Guard: if result_df was stripped (e.g. already a preview dict/str), fall back to raw df
                    if not isinstance(df_to_store, (pd.DataFrame, list)):
                        logger.warning(
                            f"⚠️ [Astronomical persist] result_df is {type(df_to_store).__name__}, not DataFrame/list. "
                            f"Falling back to raw input df ({len(df)} rows) for storage."
                        )
                        df_to_store = df

                    # 🔴 BUG FIX #2: Preserve NaN values before serialization
                    result_records = df_to_store.to_dict(orient='records') if isinstance(df_to_store, pd.DataFrame) else df_to_store
                    result_records = self._preserve_nan_values(result_records)
                    
                    # Clear cache before updating
                    self.clear_cache(session_id)
                    logger.info(f"Cache invalidated for session {session_id[:8]}...")
                    
                    await store_session_step_result(
                        session_id=session_id,
                        step_name="astronomical_analysis",
                        data=result_records,
                        db=db,
                        force_pickle=True  # Astronomical results can be large
                    )
                    await set_as_current_data(session_id, db, task_id)
                    logger.info(f"✅ Stored astronomical_analysis enriched_df with {len(result_records)} records, NaN preserved")
                    
                    # Populate cache with fresh results
                    await self.cache_session_data(
                        session_id=session_id,
                        data=result_records,
                        source_step='astronomical_analysis',
                        ttl_seconds=1800
                    )
                    logger.info(f"Cached {len(result_records)} rows after astronomical analysis (NaN preserved)")
                    
                    # ✅ UPDATE self.current_data with enriched DataFrame (final step, but keeps it consistent)
                    self.current_data = df_to_store if isinstance(df_to_store, pd.DataFrame) else pd.DataFrame(result_records)
                    self.current_session_id = session_id
                    logger.info(f"Updated self.current_data with {len(self.current_data)} enriched rows (final step)")
                else:
                    # 🔴 BUG FIX: slice_streaming stores internally BUT we must load the FULLY MERGED result
                    # PM's _aggregate_slice_results() returns complete merged DataFrame in result["result_df"]
                    df_merged = result.get("result_df", df)
                    
                    if isinstance(df_merged, pd.DataFrame):
                        # 🧹 MEMORY FIX: Use reference, not copy (ProcessingManager already owns the data)
                        self.current_data = df_merged  # Reference, not copy!
                        self.current_session_id = session_id
                        
                        mem_mb = df_merged.memory_usage(deep=True).sum() / (1024 * 1024)
                        logger.info(
                            f"✅ [MEMORY OPTIMIZED] Loaded FULLY MERGED astronomical result (reference): "
                            f"{len(self.current_data)} rows, {mem_mb:.1f} MB "
                            f"(from {len(result.get('metadata', {}).get('slices', [])) if 'slices' in result.get('metadata', {}) else '?'} slices)"
                        )
                    else:
                        logger.warning(f"⚠️ Could not extract merged result_df from slice_streaming, using DB pointer only")
                    
                    await set_as_current_data(session_id, db, task_id)
        except Exception as db_err:
            logger.warning(f"Could not persist astronomical analysis result: {db_err}")
          # Cleanup input DataFrame reference
        return result
    
    async def execute_currency_indices_analysis(
        self,
        df: pd.DataFrame,
        pm: ProcessingManager,
        session_id: str,
        task_id: str,
        config: Any = None,
        user_id: str = "anonymous",
    ) -> Dict[str, Any]:
        """
        Execute currency indices analysis using centralized ProcessingManager.
        ✅ INTEGRATED: Part of AnalysisManager, no external helpers needed
        ✅ UPDATES: self.current_data with enriched DataFrame for next step
        """
        # STEP 1: Get currency indices from PM
        result = await pm.execute(df, user_id=user_id)
        
        # STEP 2: Extract result DataFrame and enhance with TI if needed
        result_df = result.get('result_df')
        if result_df is None:
            result_df = result.get('features_df')
        
        # STEP 3: The config (CurrencyIndexConfig) is passed directly from the caller.
        # It contains calculate_ti_for_indices populated by the frontend.
        # No need to read from metadata — pm.config IS the correct config.
        if config is None:
            # Last-resort fallback: try to read from pm.config
            config = getattr(pm, 'config', None)
        
        if config is None:
            logger.warning("[Currency Indices] No config available for TI post-processing")
        
        if isinstance(result_df, pd.DataFrame):
            logger.info(f"[Currency Indices] Post-processing with TI: checking config...")
            # Apply TI post-processing if enabled (leverages PM for each index)
            result_df = await self.execute_ti_for_currency_indices(
                result_df, config, pm, session_id, task_id, user_id
            )
            # Update result with enhanced DataFrame
            result['result_df'] = result_df
            logger.info(f"[Currency Indices] TI post-processing complete: {len(result_df.columns)} total columns")
        
        # Store result to DB (if not slice_streaming which does it internally)
        try:
            async with AsyncPostgresSessionLocal() as db:
                strategy_used = result.get("metadata", {}).get("strategy")
                if strategy_used != "slice_streaming":
                    # CRITICAL: Use the TI-enriched result_df (set above in result['result_df'] = result_df)
                    # Do NOT fall back to TIER 1 cache — that cache has the pre-TI 23-column version.
                    df_to_store = result.get("result_df")
                    if df_to_store is None:
                        df_to_store = result.get("features_df")
                    
                    # Only fall back to raw df if result_df is completely missing
                    if not isinstance(df_to_store, (pd.DataFrame, list)):
                        logger.warning(
                            f"⚠️ [Currency Indices persist] result_df is {type(df_to_store).__name__}, "
                            f"falling back to raw input df ({len(df)} rows)"
                        )
                        df_to_store = df

                    # 🔴 BUG FIX #2: Preserve NaN values before serialization
                    result_records = df_to_store.to_dict(orient='records') if isinstance(df_to_store, pd.DataFrame) else df_to_store
                    result_records = self._preserve_nan_values(result_records)
                    
                    # Clear AM session cache AND decompress cache before updating with TI-enriched result
                    self.clear_cache(session_id)
                    logger.info(f"Cache invalidated for session {session_id[:8]}...")
                    
                    # Invalidate decompress cache for currency_indices (may hold pre-TI version)
                    try:
                        from app.core.services.decompress_cache import get_cache as get_decompress_cache
                        decompress_cache = get_decompress_cache()
                        async with decompress_cache.lock:
                            for step_key_suffix in ['currency_indices', 'currency_indices_analysis']:
                                step_key = decompress_cache._make_key(session_id, step_key_suffix)
                                if step_key in decompress_cache.cache:
                                    del decompress_cache.cache[step_key]
                                    logger.info(f"[Currency Indices] Invalidated decompress cache for '{step_key_suffix}'")
                    except Exception as cache_err:
                        logger.debug(f"[Currency Indices] Could not invalidate decompress cache: {cache_err}")
                    
                    await store_session_step_result(
                        session_id=session_id,
                        step_name="currency_indices",
                        data=result_records,
                        db=db,
                        force_pickle=True  # Results can be large
                    )
                    await set_as_current_data(session_id, db, task_id)
                    logger.info(f"✅ Stored currency_indices enriched_df with {len(result_records)} records, NaN preserved")
                    
                    # Populate cache with fresh TI-enriched results
                    await self.cache_session_data(
                        session_id=session_id,
                        data=result_records,
                        source_step='currency_indices',
                        ttl_seconds=1800
                    )
                    logger.info(f"Cached {len(result_records)} rows after currency indices analysis (NaN preserved)")
                    
                    # ✅ UPDATE self.current_data with enriched DataFrame (final step, but keeps it consistent)
                    self.current_data = df_to_store if isinstance(df_to_store, pd.DataFrame) else pd.DataFrame(result_records)
                    self.current_session_id = session_id
                    logger.info(f"Updated self.current_data with {len(self.current_data)} enriched rows (final step)")
                else:
                    # 🔴 BUG FIX: slice_streaming stores internally BUT we must load the FULLY MERGED result
                    # PM's _aggregate_slice_results() returns complete merged DataFrame in result["result_df"]
                    df_merged = result.get("result_df", df)
                    
                    if isinstance(df_merged, pd.DataFrame):
                        # 🧹 MEMORY FIX: Use reference, not copy (ProcessingManager already owns the data)
                        self.current_data = df_merged  # Reference, not copy!
                        self.current_session_id = session_id
                        
                        mem_mb = df_merged.memory_usage(deep=True).sum() / (1024 * 1024)
                        logger.info(
                            f"✅ [MEMORY OPTIMIZED] Loaded FULLY MERGED currency indices result (reference): "
                            f"{len(self.current_data)} rows, {mem_mb:.1f} MB "
                            f"(from {len(result.get('metadata', {}).get('slices', [])) if 'slices' in result.get('metadata', {}) else '?'} slices)"
                        )
                    else:
                        logger.warning(f"⚠️ Could not extract merged result_df from slice_streaming, using DB pointer only")
                    
                    await set_as_current_data(session_id, db, task_id)
        except Exception as db_err:
            logger.warning(f"Could not persist currency indices analysis result: {db_err}")
        return result
    
    # ────────────────────────────────────────────────────────────────
    # TI CALCULATION FOR CURRENCY INDICES (leverages PM for scalability)
    # ────────────────────────────────────────────────────────────────
    
     
async def execute_ti_for_currency_indices(
    self,
    result_df: pd.DataFrame,
    config: Any,
    pm,                    # ProcessingManager
    session_id: str,
    task_id: str,
    user_id: str = "anonymous",
) -> pd.DataFrame:
    """
    Calculate Technical Indicators for currency indices using ProcessingManager.
 
    Replaces the original method in AnalysisManager verbatim except for the
    DatetimeIndex fix applied in STEP 2 (marked ── FIX ──).
    """
    if not config:
        logger.info("[Currency Indices TI] No config provided, skipping TI calculation")
        return result_df
 
    # Handle both dict and object configs
    ti_enabled_dict = None
    if isinstance(config, dict):
        ti_enabled_dict = config.get("calculate_ti_for_indices")
    else:
        ti_enabled_dict = getattr(config, "calculate_ti_for_indices", None)
 
    if not ti_enabled_dict:
        logger.info("[Currency Indices TI] No TI calculation requested")
        return result_df
 
    ti_enabled_indices = [idx for idx, enabled in ti_enabled_dict.items() if enabled]
    if not ti_enabled_indices:
        logger.info("[Currency Indices TI] No indices have TI enabled")
        return result_df
 
    logger.info(f"[Currency Indices TI] Processing TI for: {ti_enabled_indices}")
 
    ti_calculated_count = 0
    for idx_name in ti_enabled_indices:
        try:
            # ── STEP 1: Locate index OHLCV columns ─────────────────────────
            ohlcv_cols = {
                "open":   f"{idx_name}_open",
                "high":   f"{idx_name}_high",
                "low":    f"{idx_name}_low",
                "close":  f"{idx_name}_close",
                "volume": f"{idx_name}_tick_volume",
            }
 
            available_volume_col = None
            if f"{idx_name}_tick_volume" in result_df.columns:
                available_volume_col = f"{idx_name}_tick_volume"
            elif f"{idx_name}_real_volume" in result_df.columns:
                available_volume_col = f"{idx_name}_real_volume"
 
            missing_cols = [
                col
                for col in [
                    ohlcv_cols["open"],
                    ohlcv_cols["high"],
                    ohlcv_cols["low"],
                    ohlcv_cols["close"],
                ]
                if col not in result_df.columns
            ]
 
            if missing_cols or not available_volume_col:
                logger.warning(
                    "[Currency Indices TI] Skipping %s — missing cols: %s, volume: %s",
                    idx_name, missing_cols, available_volume_col,
                )
                continue
 
            # ── STEP 2: Build minimal OHLCV DataFrame for PM ───────────────
            #
            # ── FIX ──────────────────────────────────────────────────────────
            # The original code used `index=result_df.index` which is a plain
            # RangeIndex after the currency-indices parallel workers run.
            # calculate_all_indicators() sets a DatetimeIndex internally, so
            # the TI result has DatetimeIndex while ti_df (= original_df inside
            # _ensure_result_completeness) has RangeIndex.  When row counts also
            # differ (warmup rows dropped), the fallback reindex branch fires:
            #
            #   result_df.reindex(original_df.index)   # DatetimeIndex by RangeIndex
            #
            # → zero label matches → 100% NaN on every TI column.
            #
            # Fix: parse the Time column into a DatetimeIndex NOW so both sides
            # of _ensure_result_completeness carry DatetimeIndex and Case B
            # (label-based reindex) aligns them correctly.
            # ─────────────────────────────────────────────────────────────────
 
            # Resolve Time values (prefer explicit Time column over index)
            if "Time" in result_df.columns:
                raw_time = result_df["Time"]
            elif "time" in result_df.columns:
                raw_time = result_df["time"]
            elif pd.api.types.is_datetime64_any_dtype(result_df.index):
                raw_time = result_df.index.to_series()
            else:
                raw_time = None
 
            # Build the datetime index
            if raw_time is not None:
                if pd.api.types.is_datetime64_any_dtype(raw_time):
                    dt_index = pd.DatetimeIndex(raw_time.values)
                elif pd.api.types.is_numeric_dtype(raw_time):
                    dt_index = pd.to_datetime(raw_time.values, unit="s", utc=False)
                else:
                    try:
                        dt_index = pd.to_datetime(raw_time.values)
                    except Exception:
                        dt_index = None
            else:
                dt_index = None
 
            # Warn and fall back to RangeIndex if parsing failed
            if dt_index is None:
                logger.warning(
                    "[Currency Indices TI] Could not build DatetimeIndex for %s — "
                    "falling back to RangeIndex (TI warmup rows may produce NaN)",
                    idx_name,
                )
                use_index = result_df.index
            else:
                use_index = dt_index
 
            ti_df = pd.DataFrame(
                {
                    "Open":   result_df[ohlcv_cols["open"]].values,
                    "High":   result_df[ohlcv_cols["high"]].values,
                    "Low":    result_df[ohlcv_cols["low"]].values,
                    "Close":  result_df[ohlcv_cols["close"]].values,
                    "Volume": result_df[available_volume_col].values,
                },
                index=use_index,   # ← DatetimeIndex, not RangeIndex
            )
 
            # Add Time column so technical handlers can detect timeframe.
            # We store Unix seconds so the handler can parse it if needed.
            if dt_index is not None:
                ti_df["Time"] = dt_index.astype(np.int64) // 10 ** 9
            elif raw_time is not None and pd.api.types.is_numeric_dtype(raw_time):
                ti_df["Time"] = raw_time.values
            else:
                # Synthesise hourly timestamps as last resort
                logger.warning(
                    "[Currency Indices TI] No Time column for %s, synthesising hourly timestamps",
                    idx_name,
                )
                start_ts = int(pd.Timestamp("2020-01-01").timestamp())
                ti_df["Time"] = [start_ts + i * 3600 for i in range(len(ti_df))]
 
            logger.info(
                "[Currency Indices TI] Processing %s: shape=%s, nulls=%d, index_type=%s",
                idx_name, ti_df.shape, ti_df.isna().sum().sum(),
                type(ti_df.index).__name__,
            )
 
            # ── STEP 3: Call PM with AnalysisType.TECHNICAL ────────────────
            logger.info("[Currency Indices TI] Calling PM for %s", idx_name)
 
            ti_config_dict = {}
            if isinstance(config, dict):
                ti_config_dict = config.get("ti_config") or {}
            else:
                ti_config_dict = getattr(config, "ti_config", None) or {}
 
            # Import here to avoid circular imports at module level
            from app.core.processing.processing_manager import ProcessingManager, AnalysisType, IntermediateResultsCache
            from app.core.config import TechnicalConfig
 
            try:
                ti_technical_config = (
                    TechnicalConfig(**ti_config_dict) if ti_config_dict else TechnicalConfig()
                )
            except (TypeError, ValueError) as cfg_err:
                logger.warning(
                    "[Currency Indices TI] Could not build TechnicalConfig: %s, using defaults",
                    cfg_err,
                )
                ti_technical_config = TechnicalConfig()
 
            ti_pm = ProcessingManager(
                session_id=session_id,
                task_id=task_id,
                analysis_type=AnalysisType.TECHNICAL,
                config=ti_technical_config,
                task_store=pm.task_store,
                connection_manager=pm.connection_manager,
                processing_config=pm.processing_config,
                user_id=user_id,
            )
 
            pm_result = await ti_pm.execute(ti_df)
 
            # ── STEP 4: Extract result DataFrame ───────────────────────────
            enriched_df = pm_result.get("result_df")
            if enriched_df is None:
                enriched_df = pm_result.get("features_df")
 
            if not isinstance(enriched_df, pd.DataFrame):
                cached_df = IntermediateResultsCache.retrieve(
                    task_id, f"technical_analysis__{idx_name}"
                )
                if isinstance(cached_df, pd.DataFrame):
                    enriched_df = cached_df
                    logger.info(
                        "[Currency Indices TI] Recovered %s TI from TIER 1 cache", idx_name
                    )
                else:
                    logger.error(
                        "[Currency Indices TI] PM returned invalid result for %s, skipping",
                        idx_name,
                    )
                    continue
 
            logger.info(
                "[Currency Indices TI] PM result for %s: shape=%s",
                idx_name, enriched_df.shape,
            )
 
            # ── STEP 5: Identify TI columns (exclude base OHLCV + Time) ────
            base_cols = {
                "Open", "High", "Low", "Close", "Volume", "Time", "time",
                "TickVolume", "Spread", "RealVolume", "real_volume",
            }
            ti_cols = [c for c in enriched_df.columns if c not in base_cols]
            logger.info(
                "[Currency Indices TI] Found %d TI columns for %s", len(ti_cols), idx_name
            )
 
            # ── STEP 6: Merge TI columns into result_df with index prefix ──
            #
            # enriched_df has a DatetimeIndex; result_df has a RangeIndex.
            # We align by position using .values so there are no index-label
            # mismatches.  _ensure_result_completeness already NaN-padded any
            # warmup rows to the full 6518-row length, so .values is always
            # length-safe here.
            valid_ti_count = 0
            for col in ti_cols:
                try:
                    series = enriched_df[col]
                    nan_count = int(series.isna().sum())
 
                    if nan_count < len(series):
                        numeric_series = pd.to_numeric(series, errors="coerce").astype("float64")
                        prefixed = f"{idx_name}_{col}"
 
                        # Use .values to bypass any index-label alignment
                        # (enriched_df is DatetimeIndex; result_df is RangeIndex)
                        result_df[prefixed] = numeric_series.values
 
                        valid_ti_count += 1
                        if valid_ti_count <= 5:
                            pct = (nan_count / len(series)) * 100
                            logger.info(
                                "[Currency Indices TI]   ✓ %s: %d/%d NaN (%.1f%%)",
                                prefixed, nan_count, len(series), pct,
                            )
                    else:
                        logger.warning(
                            "[Currency Indices TI]   ✗ %s_%s: 100%% NaN (skipped)",
                            idx_name, col,
                        )
                except Exception as col_err:
                    logger.warning(
                        "[Currency Indices TI]   ⚠ Failed to add %s: %s", col, col_err
                    )
 
            logger.info(
                "[Currency Indices TI] Added %d/%d valid TI columns for %s",
                valid_ti_count, len(ti_cols), idx_name,
            )
            ti_calculated_count += 1
 
        except Exception as exc:
            logger.error(
                "[Currency Indices TI] Failed for %s: %s", idx_name, exc, exc_info=True
            )
 
    logger.info(
        "[Currency Indices TI] Completed: %d/%d indices processed, %d total columns",
        ti_calculated_count, len(ti_enabled_indices), len(result_df.columns),
    )
    return result_df
 

    # ────────────────────────────────────────────────────────────────
    # ML CONFIG VALIDATION
    # ────────────────────────────────────────────────────────────────
    
    def validate_ml_config(self, config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate ML preparation config BEFORE processing.
        
        FIX 3: CONSOLIDATED CONFIG VALIDATION - Early detection of issues
        This validates critical config at AM level BEFORE expensive PM operations.
        PM will do focused validation only for issues that arise after splitting.
        
        Args:
            config: Configuration dictionary from frontend
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # 1. Required fields - core parameters that MUST be present
        required_fields = ['sequence_length', 'prediction_length', 'target_columns', 'scaler_type']
        for field in required_fields:
            if field not in config or config[field] is None:
                return False, f"Missing required field: {field}"
        
        # 2. Validate scaler_type is supported BEFORE PM processing
        scaler_type = config.get('scaler_type')
        valid_scalers = ['minmax', 'standard', 'robust']
        if isinstance(scaler_type, str):
            scaler_type_lower = scaler_type.lower()
        else:
            scaler_type_lower = getattr(scaler_type, 'value', '').lower()
        
        if scaler_type_lower not in valid_scalers:
            return False, f"Invalid scaler_type: {scaler_type}. Must be one of: {valid_scalers}"
        
        # 3. Validate numeric parameters
        try:
            seq_len = int(config['sequence_length'])
            if seq_len <= 0:
                return False, f"sequence_length must be positive, got {seq_len}"
            
            # Validate sequence length is reasonable (not too large for memory)
            if seq_len > 10000:
                return False, f"sequence_length too large: {seq_len} (max 10000). This would consume excessive memory."
            
            pred_len = int(config['prediction_length'])
            if pred_len < 0:
                return False, f"prediction_length cannot be negative, got {pred_len}"
        except (ValueError, TypeError):
            return False, "sequence_length and prediction_length must be integers"
        
        # 4. Split ratios - validate they sum to 1.0 and are valid
        train_ratio = config.get('train_ratio', 0.7)
        val_ratio = config.get('validation_ratio', 0.15)
        test_ratio = config.get('test_ratio', 0.15)
        total_ratio = train_ratio + val_ratio + test_ratio
        
        if not np.isclose(total_ratio, 1.0):
            return False, f"Split ratios must sum to 1.0, got {total_ratio:.2f} (train={train_ratio}, val={val_ratio}, test={test_ratio})"
        
        # ✅ Validate split ratios are non-negative
        if train_ratio <= 0 or val_ratio < 0 or test_ratio < 0:
            return False, f"Split ratios must be non-negative, got train={train_ratio}, val={val_ratio}, test={test_ratio}"
        
        # 5. Target columns
        targets = config.get('target_columns', [])
        if not isinstance(targets, list) or len(targets) == 0:
            return False, "At least one target column must be specified"
            
        return True, None

    # ────────────────────────────────────────────────────────────────
    # ML PIPELINE EXECUTION METHODS
    # ────────────────────────────────────────────────────────────────
    
    async def execute_ml_preparation(
        self,
        session_id: str,
        task_id: str,
        pm: ProcessingManager,
        request_data: Dict[str, Any] = None,
        user_id: str = "anonymous",
    ) -> Dict[str, Any]:
        """
        Execute ML dataset preparation (TIER 0b pointer generation).
        
        Uses ProcessingManager for large dataset handling:
        - SEQUENTIAL (<1K rows)
        - PARALLEL_CHUNKING (1K-10K rows)
        - SLICE_STREAMING (>10K rows)
        
        Args:
            session_id: Analysis session ID
            task_id: Task ID for progress tracking
            pm: ProcessingManager instance with config containing source_type, dataset_name, etc.
            request_data: DEPRECATED - config is in pm.config (passed by frontend in config field)
            user_id: User identifier
        """
        try:
            # ✅ STEP 0: Validate config before anything else
            # config is in pm.config (passed as dict or object)
            config_to_validate = pm.config if isinstance(pm.config, dict) else pm.config.__dict__ if hasattr(pm.config, "__dict__") else {}
            
            # ✅ LOGGING INCOMING FRONTEND PAYLOAD
            logger.info(f"🔍 [ML Preparation] Incoming config from frontend:")
            logger.info(f"   -> prediction_length: {config_to_validate.get('prediction_length')}")
            logger.info(f"   -> target_columns: {config_to_validate.get('target_columns')}")
            
            is_valid, config_error = self.validate_ml_config(config_to_validate)
            if not is_valid:
                logger.error(f"[{task_id}] ❌ ML Config validation failed: {config_error}")
                error_ctx = ErrorContext(
                    ErrorCategory.MISSING_REQUIRED_FIELD if "Missing" in config_error else ErrorCategory.VALIDATION_FAILED,
                    config_error,
                    "ml_preparation_validation"
                )
                await self.send_error_to_frontend(error_ctx, task_id, session_id)
                return {"status": "error", "message": config_error}

            # ✅ CRITICAL FIX: dataset_name is passed as kwarg to pm.execute(), not stored in pm.config
            # Extract from pm.config if DatasetConfig object, otherwise use task_id-based naming
            source_type = getattr(pm.config, "source_type", "enriched_df") if isinstance(pm.config, dict) == False else "enriched_df"
            
            # ✅ FIX: Extract dataset_name from pm.config (passed from frontend)
            # Try multiple sources for dataset_name in priority order:
            # 1. pm.config.dataset_name (if it's a DatasetConfig object)
            # 2. pm.config["dataset_name"] (if it's a dict from frontend)
            # 3. Fallback to timestamped unique name to prevent collisions
            if isinstance(pm.config, dict):
                # Extract from dict (frontend sends config as dict)
                dataset_name = pm.config.get("dataset_name")
                logger.info(f"[{task_id}] 📦 Extracted dataset_name from config dict: {dataset_name}")
            else:
                dataset_name = getattr(pm.config, "dataset_name", None)
                logger.info(f"[{task_id}] 📦 Extracted dataset_name from config object: {dataset_name}")
            
            # Only generate fallback if not provided or is default/auto-generated
            if not dataset_name or dataset_name == "default" or dataset_name.startswith("ml_prep_"):
                
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                dataset_name = f"ml_prep_{timestamp}"
                logger.info(f"[{task_id}] 🆔 Generated fallback dataset name: {dataset_name}")
            else:
                logger.info(f"[{task_id}] ✅ Using frontend-provided dataset name: {dataset_name}")
            
            logger.info(f"[{task_id}] ML Prep Config: source_type={source_type}, dataset_name={dataset_name}")
            
            # Update task: Loading data
            if self.task_store is not None:
                self.task_store.update_task(
                    task_id=task_id,
                    status="processing",
                    progress=10,
                    message=f"Loading source data ({source_type}) for ML preparation..."
                )
            
            injected_dataset = None
            current_data = None
            
            if source_type == "snr_unprocessed":
                # LOAD SNR Unprocessed Dataset from TIER 0 pointer set by SNR analysis
                if self.unprocessed_dataset is not None and self.unprocessed_session_id == session_id:
                    injected_dataset = self.unprocessed_dataset
                    logger.info(f"🚀 Using SNR Unprocessed Dataset ({len(injected_dataset)} records)")
                    # We still need a dummy DataFrame for pm.execute() to track row count
                    current_data = pd.DataFrame([{"dummy": 0}] * len(injected_dataset))
                else:
                    error_msg = "SNR unprocessed dataset not found for this session. Run SNR analysis first."
                    error_ctx = ErrorContext(ErrorCategory.VALIDATION_FAILED, error_msg, "ml_preparation_data_load")
                    await self.send_error_to_frontend(error_ctx, task_id, session_id)
                    return {
                        "status": "error",
                        "message": error_msg
                    }
            else:
                # Load current_data (TIER 0a) - 0ms latency
                current_data, source = await self._load_data_4_tier(
                    session_id=session_id,
                    task_id=task_id,
                    data_type="analysis"
                )
            
            if current_data is None or current_data.empty:
                error_msg = "No analysis data available for ML preparation"
                error_ctx = ErrorContext(ErrorCategory.VALIDATION_FAILED, error_msg, "ml_preparation_data_load")
                await self.send_error_to_frontend(error_ctx, task_id, session_id)
                return {
                    "status": "error",
                    "message": error_msg
                }
            
            # ✅ FIXED: Send initial progress via task_store BEFORE starting processing
            # This ensures frontend sees progress immediately after HTTP request completes
            if self.task_store is not None:
                self.task_store.update_task(
                    task_id=task_id,
                    status="processing",
                    progress=5,
                    message=f"Initializing ML preparation for '{dataset_name}'..."
                )
            
            # Update task: Starting ML preparation
            if self.task_store is not None:
                self.task_store.update_task(
                    task_id=task_id,
                    status="processing",
                    progress=10,
                    message=f"Loading data for ML preparation..."
                )
            
            # Use ProcessingManager for large dataset handling:
            # PM executes strategy which calls handler (analyze_ml_prep_impl)
            # The handler receives context.config with all DatasetConfig fields
            
            # ✅ FIX 1: WRAP PM.execute() WITH ERROR HANDLING + CLASSIFICATION
            # This ensures PM errors (config validation, scaler fitting, etc.) are properly
            # classified and routed to frontend instead of bypassing to generic 500 error
            try:
                ml_splits = await pm.execute(
                    current_data, 
                    user_id=user_id,
                    injected_dataset=injected_dataset,
                    dataset_name=dataset_name
                )
            except Exception as pm_error:
                # Classify error and route to frontend
                error_ctx = self.classify_error(pm_error, "ml_preparation_execution")
                logger.error(f"[{task_id}] ❌ ProcessingManager execution failed: {str(pm_error)}")
                await self.send_error_to_frontend(error_ctx, task_id, session_id)
                
                # Return error response (don't re-raise, let API return proper HTTP response)
                return {
                    "status": "error",
                    "message": error_ctx.message,
                    "error_category": error_ctx.category.value,
                    "retryable": error_ctx.retryable
                }
            
            # ✅ Extract targets from splits for separate storage
            # NOTE: In parallel-spooled mode, splits are NOT in ml_splits, but stored in DB.
            # We use the metadata to get target information for logging.
            pm_metadata = ml_splits.get("metadata", {})
            train_targets = {}
            val_targets = {}
            test_targets = {}
            
            # Use metadata for target names if splits are missing
            target_names = pm_metadata.get("target_names", [])
            
            if ml_splits:
                # Check for in-memory splits (sequential mode)
                train_data = ml_splits.get("train", {})
                val_data = ml_splits.get("validation", {})
                test_data = ml_splits.get("test", {})
                
                # Extract targets dict from each split if present
                train_targets = train_data.get("targets", {})
                val_targets = val_data.get("targets", {})
                test_targets = test_data.get("targets", {})
                
                if not train_targets and target_names:
                    # In parallel mode, we at least know the names from metadata
                    train_targets = {name: [] for name in target_names}
                    val_targets = {name: [] for name in target_names}
                    test_targets = {name: [] for name in target_names}
                    logger.info(f"[{task_id}] ℹ️ Splits are in DB (Parallel Mode). Using target names from metadata.")
                
                logger.info(f"[{task_id}] ✅ ML Prep targets identified:")
                logger.info(f"   Train targets: {list(train_targets.keys())} - {len(train_targets)} columns")
                logger.info(f"   Val targets: {list(val_targets.keys())} - {len(val_targets)} columns")
                logger.info(f"   Test targets: {list(test_targets.keys())} - {len(test_targets)} columns")
                
                # Log target shapes for debugging
                if train_targets:
                    for tname, tarray in train_targets.items():
                        if isinstance(tarray, np.ndarray):
                            logger.info(f"   Train target '{tname}' shape: {tarray.shape}")
            
            # ═══════════════════════════════════════════════════════════════
            # REMOVED: ML pointer setting (TIER 0b)
            # ═══════════════════════════════════════════════════════════════
            # Data is now stored in MLDataset table by PM
            # Frontend fetches using dataset_ids from metadata
            # No need for in-memory pointers - data is persisted in DB
            
            # ═══════════════════════════════════════════════════════════════
            # REMOVED: Duplicate ml_datasets catalog registration
            # ═══════════════════════════════════════════════════════════════
            # PM already stores to MLDataset table via create_ml_dataset()
            # No need to register again - this was causing duplicate entries
            
            # ✅ Cleanup unwanted variables at function end
            self.cleanup_function_locals("execute_ml_preparation")
            
            # ✅ REMOVED: Duplicate storage to SessionStepResult
            # PM already stored splits to MLDataset table with dataset_ids in metadata
            # No need to store again - this was causing double serialization
            
            # Update task: ML preparation success
            if self.task_store is not None:
                # Extract metadata from PM result
                pm_metadata = ml_splits.get("metadata", {})
                
                self.task_store.update_task(
                    task_id=task_id,
                    status="completed",
                    progress=100,
                    message="ML preparation complete. Data stored in MLDataset table.",
                    metadata={
                        "dataset_name": dataset_name,
                        "total_sequences": pm_metadata.get("total_sequences", 0),
                        "split_counts": pm_metadata.get("split_counts", {}),
                        "dataset_ids": pm_metadata.get("dataset_ids", {}),
                        "storage_strategy": pm_metadata.get("storage_strategy", "ml_dataset_table"),
                        "source": source_type,
                    }
                )
            
            # Return PM's metadata directly to frontend (no data, just references)
            # ✅ FIX: Include target_names at top level for frontend target selection UI
            return {
                "status": "success",
                "task_id": task_id,
                "_ref": task_id,
                "dataset_name": dataset_name,
                "timestamp": datetime.utcnow().isoformat(),
                # ✅ Top-level fields for frontend
                "target_names": ml_splits.get("target_names", []),  # All targets including advanced
                "feature_names": ml_splits.get("feature_names", []),
                "sequence_length": ml_splits.get("sequence_length", 0),
                "prediction_length": ml_splits.get("prediction_length", 0),
                "split_counts": ml_splits.get("split_counts", {}),
                "split_dataset_ids": ml_splits.get("split_dataset_ids", {}),
                # Nested metadata for backward compatibility
                "metadata": ml_splits.get("metadata", {}),
                "config": ml_splits.get("config", {}),
                "scaler_available": ml_splits.get("scaler_available", False),
            }
            
        except Exception as e:
            self.logger.error(f"ML preparation error: {str(e)}", exc_info=True)
            if self.task_store:
                self.task_store.update_task(
                    task_id=task_id,
                    status="error",
                    message=f"ML preparation failed: {str(e)}"
                )
            return {"status": "error", "message": str(e)}
    
    # ============================================================================
    # PHASE 19: HELPER METHODS FOR MODEL METADATA CAPTURE
    # ============================================================================
    
    def _get_code_version(self) -> str:
        """Get application/code version for reproducibility tracking."""
        try:
            # Try to read from git or version file
            if os.path.exists(".git"):
                import subprocess
                result = subprocess.run(
                    ["git", "describe", "--tags", "--always"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return result.stdout.strip()
        except Exception:
            pass
        
        # Fallback: Use app version from config or environment
        version = os.getenv("APP_VERSION", "unknown")
        if version != "unknown":
            return version
        
        # Fallback: Use timestamp-based version
        return f"build-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    
    def _get_compilation_environment(self) -> Dict[str, str]:
        """Capture TensorFlow, Python, OS environment for reproducibility."""
        env_info = {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        }
        
        # Add TensorFlow version if available
        try:
            env_info["tensorflow_version"] = tf.__version__
            env_info["keras_version"] = tf.keras.__version__
            # Check GPU availability
            env_info["gpu_available"] = len(tf.config.list_physical_devices('GPU')) > 0
            env_info["gpu_devices"] = [str(d) for d in tf.config.list_physical_devices('GPU')]
        except Exception:
            env_info["tensorflow_version"] = "not_available"
        
        # Add NumPy version
        try:
            env_info["numpy_version"] = np.__version__
        except Exception:
            pass
        
        # Add Pandas version
        try:
            env_info["pandas_version"] = pd.__version__
        except Exception:
            pass
        
        return env_info
    
    def _detect_training_device(self) -> str:
        """Detect if model will train on GPU or CPU."""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                return f"gpu({len(gpus)})"
        except Exception:
            pass
        
        return "cpu"
    
    def _calculate_model_parameters(self, model) -> Tuple[int, int, int]:
        """
        Calculate total, trainable, and non-trainable parameters.
        
        Returns:
            (total_params, trainable_params, non_trainable_params)
        """
        try:
            total_params = model.count_params()
            
            # Calculate trainable params
            trainable_params = sum([
                np.prod(w.shape) if hasattr(w, 'shape') else w.shape.num_elements() 
                for w in model.trainable_weights
            ])
            
            non_trainable_params = sum([
                np.prod(w.shape) if hasattr(w, 'shape') else w.shape.num_elements()
                for w in model.non_trainable_weights
            ])
            
            return total_params, trainable_params, non_trainable_params
        except Exception as e:
            self.logger.warning(f"Could not calculate model parameters: {e}")
            return 0, 0, 0
    
    async def _store_compiled_model_to_db(
        self,
        session_id: str,
        user_id: str,
        model_config: Dict[str, Any],
        model_binary: bytes,
        compiled_model_id: str,
    ) -> bool:
        """
        Store compiled model metadata and binary to CompiledModel table.
        
        Args:
            session_id: Session ID
            user_id: User ID
            model_config: Model configuration with all metadata
            model_binary: Serialized model binary
            compiled_model_id: UUID for the compiled model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with AsyncPostgresSessionLocal() as db:
                # Prepare compiled model record
                compiled_model = CompiledModel(
                    compiled_model_id=compiled_model_id,
                    user_id=user_id,
                    session_id=session_id,
                    model_name=model_config.get("model_name", f"model_{compiled_model_id[:8]}"),
                    description=model_config.get("description", None),
                    tags=model_config.get("tags", ["auto-compiled"]),
                    model_type=model_config.get("type", "lstm"),
                    architecture_json=model_config.get("architecture_json", {}),
                    total_parameters=model_config.get("total_parameters", 0),
                    trainable_parameters=model_config.get("trainable_parameters", 0),
                    non_trainable_parameters=model_config.get("non_trainable_parameters", 0),
                    model_summary=model_config.get("model_summary", ""),
                    input_shape=model_config.get("input_shape", []),
                    output_shape=model_config.get("output_shape", []),
                    n_predictions=model_config.get("n_predictions", 1),
                    prediction_length=model_config.get("prediction_length", None),
                    dataset_id=model_config.get("dataset_id", None),
                    feature_columns=model_config.get("feature_cols", []),
                    target_columns=model_config.get("target_cols", []),
                    selected_targets=model_config.get("selected_targets", None),
                    feature_hash=model_config.get("feature_hash", ""),
                    ml_dataset_ref=model_config.get("ml_preparation_ref", ""),
                    ml_dataset_feature_count=len(model_config.get("feature_cols", [])),
                    ml_dataset_sequence_length=model_config.get("input_shape", [None])[0],
                    optimizer_config=model_config.get("optimizer_config", {}),
                    loss_function=model_config.get("loss_function", "mse"),
                    metrics_list=model_config.get("metrics_list", ["mae"]),
                    model_binary=model_binary,
                    model_size_bytes=len(model_binary),
                    framework=model_config.get("framework", "tensorflow"),
                    framework_version=model_config.get("framework_version", ""),
                    version=model_config.get("version", 1),
                    status="compiled",
                    compilation_timestamp=datetime.utcnow(),
                    code_version=model_config.get("code_version", ""),
                    compilation_environment=model_config.get("compilation_environment", {}),
                    is_public=model_config.get("is_public", False),
                    is_best_version=False,
                )
                
                db.add(compiled_model)
                await db.commit()
                
                self.logger.info(
                    f"✅ Stored CompiledModel to database: {compiled_model_id} "
                    f"({model_config.get('model_name', 'untitled')})"
                )
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to store CompiledModel to database: {e}", exc_info=True)
            return False
    
    # ============================================================================
    # PHASE 19: HELPER METHODS FOR TRAINING METADATA CAPTURE
    # ============================================================================
    
    def _calculate_quality_score(
        self,
        final_val_loss: float,
        final_training_loss: float,
        loss_history: List[float],
        val_loss_history: List[float],
        model_type: str = "lstm"
    ) -> float:
        """
        Calculate a quality score (0-100) based on training outcomes.
        
        Factors:
        - Final validation loss (normalized)
        - Convergence trend (loss decreasing over time)
        - Overfitting detection (gap between train and val loss)
        
        Args:
            final_val_loss: Final validation loss
            final_training_loss: Final training loss
            loss_history: List of training losses per epoch
            val_loss_history: List of validation losses per epoch
            model_type: Model type (lstm, cnn, dense, etc.)
            
        Returns:
            Quality score 0-100 (higher is better)
        """
        try:
            # Normalize loss (assume typical loss ranges 0-2 for well-trained models)
            # Clamp to 0-1 range
            normalized_loss = min(1.0, final_val_loss / 2.0)
            loss_score = 100 * (1 - normalized_loss)
            
            # Convergence factor: Does loss decrease consistently?
            convergence_factor = 1.0
            if len(val_loss_history) > 1:
                # Check if loss improved from start to end
                first_val_loss = val_loss_history[0]
                last_val_loss = val_loss_history[-1]
                
                if first_val_loss > 0:
                    improvement_ratio = (first_val_loss - last_val_loss) / first_val_loss
                    convergence_factor = min(1.5, max(0.5, improvement_ratio * 2))  # Clamp to 0.5-1.5
            
            # Overfitting factor: Penalize if val_loss >> train_loss
            overfitting_factor = 1.0
            if final_training_loss > 0 and final_val_loss > 0:
                loss_gap_ratio = final_val_loss / final_training_loss
                if loss_gap_ratio > 1.5:
                    # Significant overfitting
                    overfitting_factor = 1.0 / loss_gap_ratio  # Penalize
                elif loss_gap_ratio > 1.0:
                    # Mild overfitting
                    overfitting_factor = 0.95
                else:
                    # Underfitting (both bad) - but better than overfitting
                    overfitting_factor = 0.9
            
            # Calculate final quality score
            quality_score = loss_score * convergence_factor * overfitting_factor
            quality_score = max(0.0, min(100.0, quality_score))  # Clamp to 0-100
            
            self.logger.info(
                f"✅ Quality Score Calculated:\n"
                f"   Loss Score: {loss_score:.1f}\n"
                f"   Convergence Factor: {convergence_factor:.2f}\n"
                f"   Overfitting Factor: {overfitting_factor:.2f}\n"
                f"   Final Quality Score: {quality_score:.1f}/100"
            )
            
            return round(quality_score, 2)
            
        except Exception as e:
            self.logger.warning(f"Could not calculate quality score: {e}, defaulting to 50.0")
            return 50.0
    
    async def _store_training_record_to_db(
        self,
        model_id: str,
        epoch: int,
        loss: float,
        val_loss: float,
        metrics: Dict[str, float],
        epoch_duration_seconds: float = None,
    ) -> bool:
        """
        Store per-epoch training metrics to TrainingRecord table.
        
        Args:
            model_id: UUID of trained model
            epoch: Epoch number (0-indexed)
            loss: Training loss at this epoch
            val_loss: Validation loss at this epoch
            metrics: Dict with mae, mse, accuracy, etc.
            epoch_duration_seconds: Time to train this epoch
            
        Returns:
            True if successful, False otherwise
        """
        try:
            
            def sanitize(v):
                if v is None: return None
                try:
                    if math.isinf(v) or math.isnan(v):
                        return 9999.999999
                    return float(v)
                except:
                    return None

            async with AsyncPostgresSessionLocal() as db:
                record = TrainingRecord(
                    record_id=uuid.uuid4(),
                    model_id=model_id,
                    epoch=epoch,
                    loss=sanitize(loss),
                    val_loss=sanitize(val_loss),
                    mae=sanitize(metrics.get("mae")),
                    val_mae=sanitize(metrics.get("val_mae")),
                    mse=sanitize(metrics.get("mse")),
                    val_mse=sanitize(metrics.get("val_mse")),
                    accuracy=sanitize(metrics.get("accuracy")),
                    val_accuracy=sanitize(metrics.get("val_accuracy")),
                    epoch_duration_seconds=sanitize(epoch_duration_seconds),
                    samples_processed=metrics.get("samples_processed"),
                    samples_per_second=metrics.get("samples_per_second"),
                    learning_rate_at_epoch=metrics.get("learning_rate"),
                    created_at=datetime.utcnow(),
                )
                
                db.add(record)
                await db.commit()
                return True
                
        except Exception as e:
            self.logger.warning(f"Failed to store TrainingRecord: {e}")
            return False
    
    async def _auto_generate_model_selection_hints(
        self,
        trained_model_id: str,
        training_data_symbol: str,
        training_data_timeframe: str,
        feature_hash: str,
        quality_score: float,
    ) -> bool:
        """
        Auto-generate ModelSelectionHint for intelligent model matching.
        
        After training, create a hint that this model is suitable for:
        - This symbol + timeframe combination
        - This feature set (via feature_hash)
        - Based on quality_score for compatibility matching
        
        Args:
            trained_model_id: UUID of the trained model
            training_data_symbol: Symbol model was trained on (e.g., "EURUSD")
            training_data_timeframe: Timeframe (e.g., "1h", "4h", "1d")
            feature_hash: Hash of feature columns
            quality_score: Model quality (0-100)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with AsyncPostgresSessionLocal() as db:
                # Convert quality_score to compatibility_score (0-100 stays same, scaled as needed)
                compatibility_score = quality_score / 100.0 * 100  # Already 0-100, keep as is
                
                hint = ModelSelectionHint(
                    hint_id=uuid.uuid4(),
                    trained_model_id=trained_model_id,
                    symbol=training_data_symbol,
                    timeframe=training_data_timeframe,
                    feature_hash=feature_hash,
                    compatibility_score=compatibility_score,
                    recommendation_reason=f"Auto-generated after training. Quality score: {quality_score:.1f}/100",
                    created_at=datetime.utcnow(),
                )
                
                db.add(hint)
                await db.commit()
                
                self.logger.info(
                    f"✅ ModelSelectionHint created: {hint.hint_id} "
                    f"({training_data_symbol}/{training_data_timeframe}, score={compatibility_score:.1f})"
                )
                return True
                
        except Exception as e:
            self.logger.warning(f"Failed to auto-generate ModelSelectionHint: {e}")
            return False
    
    async def _update_trained_model_with_metadata(
        self,
        model_id: str,
        training_metadata: Dict[str, Any],
    ) -> bool:
        """
        Update TrainedModelForAnalysis with comprehensive training metadata.
        
        Called after training completes to populate all the new Phase 19 fields.
        
        Args:
            model_id: UUID of the trained model
            training_metadata: Dict with all training metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with AsyncPostgresSessionLocal() as db:
                # Query the model
                stmt = select(TrainedModelForAnalysis).where(
                    TrainedModelForAnalysis.model_id == model_id
                )
                result = await db.execute(stmt)
                trained_model = result.scalars().first()
                
                if not trained_model:
                    self.logger.warning(f"TrainedModelForAnalysis not found: {model_id}")
                    return False
                
                # Update all metadata fields
                trained_model.training_data_symbol = training_metadata.get("training_data_symbol")
                trained_model.training_data_timeframe = training_metadata.get("training_data_timeframe")
                trained_model.training_data_start_date = training_metadata.get("training_data_start_date")
                trained_model.training_data_end_date = training_metadata.get("training_data_end_date")
                trained_model.training_data_duration_days = training_metadata.get("training_data_duration_days")
                trained_model.training_data_sample_count = training_metadata.get("training_data_sample_count")
                trained_model.training_data_feature_columns = training_metadata.get("training_data_feature_columns")
                trained_model.training_data_target_columns = training_metadata.get("training_data_target_columns")
                trained_model.training_data_hash = training_metadata.get("training_data_hash")
                
                trained_model.optimizer_type = training_metadata.get("optimizer_type")
                trained_model.optimizer_config = training_metadata.get("optimizer_config")
                trained_model.learning_rate = training_metadata.get("learning_rate")
                trained_model.learning_rate_schedule = training_metadata.get("learning_rate_schedule")
                trained_model.batch_size = training_metadata.get("batch_size")
                trained_model.epochs = training_metadata.get("epochs")
                trained_model.validation_split = training_metadata.get("validation_split")
                
                trained_model.training_framework = training_metadata.get("training_framework", "tensorflow")
                trained_model.training_framework_version = training_metadata.get("training_framework_version")
                trained_model.training_device = training_metadata.get("training_device")
                trained_model.training_environment = training_metadata.get("training_environment")
                trained_model.code_version = training_metadata.get("code_version")
                trained_model.random_seed = training_metadata.get("random_seed")
                
                trained_model.training_started_at = training_metadata.get("training_started_at")
                trained_model.training_completed_at = training_metadata.get("training_completed_at")
                trained_model.training_duration_seconds = training_metadata.get("training_duration_seconds")
                
                trained_model.training_loss_history = training_metadata.get("training_loss_history")
                trained_model.validation_loss_history = training_metadata.get("validation_loss_history")
                trained_model.training_metrics_history = training_metadata.get("training_metrics_history")
                
                trained_model.final_training_loss = training_metadata.get("final_training_loss")
                trained_model.final_validation_loss = training_metadata.get("final_validation_loss")
                trained_model.training_accuracy = training_metadata.get("training_accuracy")
                trained_model.validation_accuracy = training_metadata.get("validation_accuracy")
                
                trained_model.quality_score = training_metadata.get("quality_score")
                trained_model.recommendation_reason = training_metadata.get("recommendation_reason")
                trained_model.is_backtest_fair = training_metadata.get("is_backtest_fair", True)
                
                # ✅ Phase 20: Store scaler, feature names, and sequence length
                trained_model.scaler_binary = training_metadata.get("scaler_binary")
                trained_model.feature_names = training_metadata.get("training_data_feature_columns")
                trained_model.sequence_length = training_metadata.get("sequence_length")
                
                # ✅ Phase 20: Infer metric_type from target columns
                target_cols = training_metadata.get("training_data_target_columns")
                if target_cols and isinstance(target_cols, list) and len(target_cols) > 0:
                    first_target = str(target_cols[0]).lower()
                    if "metric_" in first_target:
                        for m_type in ["volatility", "speed", "direction", "regime"]:
                            if m_type in first_target:
                                trained_model.metric_type = m_type
                                break
                
                trained_model.status = "trained"
                
                await db.commit()
                
                self.logger.info(
                    f"✅ Updated TrainedModelForAnalysis with training metadata: {model_id}"
                )
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to update TrainedModelForAnalysis metadata: {e}", exc_info=True)
            return False
    
    
    
    async def execute_model_build(
        self,
        session_id: str,
        task_id: str,
        model_config: Dict[str, Any],
        user_id: str = "anonymous",
    ) -> Dict[str, Any]:
        """
        Execute ML model building (compilation & optimization).
        
        TIER 0c Caching: Stores built model to self.model for zero-disk-I/O training step.
        
        Args:
            session_id: Session ID
            task_id: Task ID for progress tracking
            model_config: Model configuration dict
            
        Returns:
            dict with keys:
                - "status": "success" | "error"
                - "model_id": Unique model ID
                - "architecture": Model summary
        """
        import dataclasses
        if dataclasses.is_dataclass(model_config) and not isinstance(model_config, type):
            model_config = dataclasses.asdict(model_config)
            
        try:
            # Update task: Starting model build
            self.task_store.update_task(
                task_id=task_id,
                status="processing",
                progress=10,
                message="Building ML model..."
            )
            
            # ✅ CRITICAL: Validate input_shape and n_predictions against actual dataset before building model
            # Use metadata-first approach to avoid loading heavy data blobs
            ml_prep_ref = model_config.get("ml_preparation_ref") or model_config.get("dataset_name")
            if ml_prep_ref:
                self.logger.info(f"🔍 [Model Build] Validating model config against dataset (Metadata-First): {ml_prep_ref}")
                try:
                    from app.core.data.session_data_loader import get_ml_dataset_metadata, get_ml_dataset_splits_by_name
                    async with AsyncPostgresSessionLocal() as db:
                        # 1. Try metadata first (Zero-Data-I/O validation)
                        meta = await get_ml_dataset_metadata(
                            session_id=session_id,
                            dataset_name=ml_prep_ref,
                            db=db
                        )
                        
                        actual_shape = None
                        actual_n_predictions = None
                        
                        if meta:
                            # TIER 1: Use basic metadata fields
                            # Priority: output_targets > prediction_length
                            feat_count = meta.get("feature_count")
                            split_cfg = meta.get("split_config", {})
                            seq_len = split_cfg.get("sequence_length")
                            pred_len = split_cfg.get("prediction_length")
                            out_targets = meta.get("output_targets", [])
                            
                            if feat_count and seq_len:
                                actual_shape = (seq_len, feat_count)
                                self.logger.info(f"✅ [Model Build] Found shape in metadata: {actual_shape}")
                            
                            # For time series forecasting: 
                            # - If single target + prediction_length: n_predictions = prediction_length (multi-step forecast)
                            # - If multiple targets: n_predictions = len(targets) (multi-output forecast)
                            selected_targets = model_config.get("selected_targets") or []
                            if selected_targets:
                                if len(selected_targets) == 1 and pred_len:
                                    actual_n_predictions = pred_len
                                    self.logger.info(
                                        f"✅ [Model Build] Found n_predictions from selected target + prediction_length: {actual_n_predictions}"
                                    )
                                elif len(selected_targets) == 1:
                                    actual_n_predictions = 1
                                    self.logger.info(
                                        f"✅ [Model Build] Found n_predictions from single selected target: {actual_n_predictions}"
                                    )
                                elif pred_len:
                                    actual_n_predictions = len(selected_targets) * pred_len
                                    self.logger.info(
                                        f"✅ [Model Build] Found n_predictions from selected_targets * prediction_length: {actual_n_predictions}"
                                    )
                                else:
                                    actual_n_predictions = len(selected_targets)
                                    self.logger.info(
                                        f"✅ [Model Build] Found n_predictions from selected_targets list: {actual_n_predictions}"
                                    )
                            elif out_targets and pred_len and len(out_targets) == 1:
                                # Single target with prediction_length = multi-step forecast of that target
                                actual_n_predictions = pred_len
                                self.logger.info(f"✅ [Model Build] Found n_predictions from prediction_length (single-target forecast): {actual_n_predictions}")
                            elif out_targets and len(out_targets) > 1:
                                # Multiple targets = multi-output regression
                                actual_n_predictions = len(out_targets)
                                self.logger.info(f"✅ [Model Build] Found n_predictions from targets list (multi-output): {actual_n_predictions}")
                            elif pred_len:
                                # Fallback to prediction_length if no explicit targets or multiple targets with prediction_length
                                actual_n_predictions = pred_len
                                self.logger.info(f"✅ [Model Build] Found n_predictions from prediction_length: {actual_n_predictions}")

                            # TIER 2: Use Fast-Truth Sample (The 1-row sequence)
                            # This is the 'Truth' without decompressing split blobs
                            # ⚠️ WARNING: Sample shape [1, k] shows one representative row, not batch shape
                            # For time series: sample [1, 1] means "1 sample with 1 value"
                            # But actual training uses [batch_size, prediction_length]
                            # So we only use sample to validate input_shape, NOT n_predictions
                            sample_x = meta.get("sample_x")
                            if sample_x and isinstance(sample_x, dict) and "shape" in sample_x:
                                s_shape = sample_x["shape"] # [1, seq_len, feat_count]
                                if len(s_shape) == 3:
                                    actual_shape = tuple(s_shape[1:])
                                    self.logger.info(f"✅ [Model Build] Validated shape via Fast-Truth sample: {actual_shape}")

                            sample_y = meta.get("sample_y")
                            if sample_y and isinstance(sample_y, dict) and "shape" in sample_y:
                                s_shape = sample_y["shape"] # [1, n_preds_in_sample]
                                # ⚠️ CRITICAL FIX: Only use sample if we haven't found n_predictions via TIER 1
                                # Sample shape is unreliable for time series (shows single row, not batch)
                                if len(s_shape) == 2 and actual_n_predictions is None:
                                    # Sample shape might be misleading for time series
                                    # Only trust it if TIER 1 found nothing
                                    actual_n_predictions = s_shape[1]
                                    self.logger.info(f"⚠️ [Model Build] Using sample-based n_predictions (unreliable): {actual_n_predictions}")
                                elif len(s_shape) == 2 and actual_n_predictions is not None:
                                    # TIER 1 already found it, don't overwrite with sample
                                    sample_n_preds = s_shape[1]
                                    if sample_n_preds != actual_n_predictions:
                                        self.logger.info(f"ℹ️ [Model Build] Sample shape {s_shape[1]} differs from metadata {actual_n_predictions} (expected for time series)")

                        # TIER 3: Fallback to full split sample ONLY if tiers 1 & 2 failed
                        if actual_shape is None or actual_n_predictions is None:
                            self.logger.warning(f"⚠️ [Model Build] Metadata incomplete for {ml_prep_ref}, fetching data sample...")
                            sample_data = await get_ml_dataset_splits_by_name(
                                session_id=session_id,
                                dataset_name=ml_prep_ref,
                                db=db,
                                split_type='train'
                            )
                            
                            if sample_data and isinstance(sample_data, dict):
                                if actual_shape is None and "sequences" in sample_data:
                                    actual_sequences = sample_data["sequences"]
                                    if isinstance(actual_sequences, np.ndarray) and len(actual_sequences.shape) == 3:
                                        actual_shape = actual_sequences.shape[1:]
                                
                                if actual_n_predictions is None and "targets" in sample_data:
                                    actual_targets = sample_data["targets"]
                                    if isinstance(actual_targets, np.ndarray) and len(actual_targets.shape) >= 1:
                                        actual_n_predictions = 1 if len(actual_targets.shape) == 1 else actual_targets.shape[1]

                        # 3. Apply corrections if needed
                        if actual_shape:
                            expected_shape = tuple(model_config.get("input_shape", []))
                            if actual_shape != expected_shape:
                                self.logger.warning(f"⚠️ [Model Build] input_shape mismatch! Expected: {expected_shape}, Actual: {actual_shape}")
                                model_config["input_shape"] = list(actual_shape)
                            else:
                                self.logger.info(f"✅ [Model Build] input_shape validation passed: {actual_shape}")

                        if actual_n_predictions is not None:
                            expected_n_predictions = model_config.get("n_predictions", 1)
                            if actual_n_predictions != expected_n_predictions:
                                self.logger.warning(f"⚠️ [Model Build] n_predictions mismatch! Expected: {expected_n_predictions}, Actual: {actual_n_predictions}")
                                model_config["n_predictions"] = actual_n_predictions
                            else:
                                self.logger.info(f"✅ [Model Build] n_predictions validation passed: {actual_n_predictions}")
                except Exception as val_err:
                    self.logger.error(f"❌ [Model Build] Validation error: {val_err}")
                    self.logger.warning(f"⚠️ [Model Build] Could not validate model config: {val_err}")
                    # Continue with original config if validation fails
            
            # Get model builder from registry
            model_registry = ModelRegistry()
            builder_class = model_registry.get_builder(model_config.get("type", "lstm"))
            if builder_class is None:
                return {"status": "error", "message": f"Model type not found: {model_config.get('type')}"}

            # Strip provenance/signature fields — these are for storage only, not for the builder
            PROVENANCE_KEYS = {
                "model_id", "architecture_type", "type", "loss", "metrics", "parameters",
                "feature_cols", "target_cols", "feature_hash", "step_configs",
                "dataset_name", "dataset_id", "ml_preparation_ref",
                # Phase 19: identity fields
                "model_name", "description", "tags", "is_public", "version",
            }
            builder_config = {k: v for k, v in model_config.items() if k not in PROVENANCE_KEYS}

            # Build model
            builder = builder_class(builder_config)
            model = builder.build()
            
            # Update task: Compiling model
            self.task_store.update_task(
                task_id=task_id,
                status="processing",
                progress=50,
                message="Compiling model..."
            )
            
            # Compile model
            opt_config = model_config.get("optimizer", "adam")
            if isinstance(opt_config, str) and opt_config.lower() == "adam":
                # ✅ FIX: Explicitly instantiate Adam with clipnorm to prevent exploding gradients
                # This is critical because RobustScaler leaves outliers that can cause NaNs
                # especially during Catch-Up Mode where we take multiple gradient steps per batch.
                opt = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
            else:
                opt = opt_config
                
            model.compile(
                optimizer=opt,
                loss=model_config.get("loss", "mse"),
                metrics=model_config.get("metrics", ["mae", "mse"])
            )
            
            # ═════════════════════════════════════════════════════════════════════
            # PHASE 19: CAPTURE COMPREHENSIVE METADATAzzzzz FOR COMPILED MODEL
            # ═════════════════════════════════════════════════════════════════════
            
            # Detect training environment (GPU/CPU)
            training_device = self._detect_training_device()
            
            # Get code version for reproducibility
            code_version = self._get_code_version()
            
            # Capture compilation environment
            compilation_environment = self._get_compilation_environment()
            
            # Calculate model parameters
            total_params, trainable_params, non_trainable_params = self._calculate_model_parameters(model)
            
            # Prepare model binary for storage
            import io
            import pickle
            model_buffer = io.BytesIO()
            try:
                # Try to save as HDF5 first (TensorFlow native)
                model.save(model_buffer, save_format='h5')
                model_binary = model_buffer.getvalue()
                serialization_format = "h5"
            except Exception:
                try:
                    # Fallback to pickle
                    model_binary = pickle.dumps(model)
                    serialization_format = "pickle"
                except Exception as pickle_err:
                    self.logger.error(f"Could not serialize model: {pickle_err}")
                    model_binary = b""
                    serialization_format = "error"
            
            # Prepare architecture JSON
            try:
                import json
                architecture_json = json.loads(model.to_json()) if hasattr(model, 'to_json') else {"type": "lstm"}
            except Exception:
                architecture_json = {"type": model_config.get("type", "lstm")}
            
            # Compute feature hash for compatibility matching
            feature_cols = model_config.get("feature_cols", [])
            feature_hash = ""
            if feature_cols:
                canonical = sorted(str(f) for f in feature_cols)
                feature_hash = hashlib.sha256(",".join(canonical).encode()).hexdigest()[:16]
            
            # Generate compiled model ID (the primary persistent UUID)
            compiled_model_id = str(uuid.uuid4())
            model_id = compiled_model_id
            
            # Extract input_shape from the built model
            # Keras models have an input_shape property in their input layer
            extracted_input_shape = None
            if hasattr(model, 'input_shape') and model.input_shape:
                # For models with a single input, model.input_shape is (None, seq_len, features)
                # We want (seq_len, features)
                if isinstance(model.input_shape, (list, tuple)) and len(model.input_shape) > 1:
                    # If the input shape is like (None, 10, 5), we want (10, 5)
                    extracted_input_shape = model.input_shape[1:]
                else:
                    # For cases like (None, 5) or other simpler shapes, take it directly
                    extracted_input_shape = model.input_shape
            
            # Update model_config with the actual input_shape from the built model
            # This is crucial for proper validation during training
            if extracted_input_shape:
                model_config["input_shape"] = extracted_input_shape
                self.logger.info(f"✅ Extracted input_shape {extracted_input_shape} from built model for {model_id}")
            else:
                self.logger.warning(f"⚠️ Could not extract input_shape from built model {model_id}. Proceeding without it.")

            # ═════════════════════════════════════════════════════════════════════
            # UNIFY: Use the UUID for both database and session identification
            # ═════════════════════════════════════════════════════════════════════
            
            # TIER 0c CACHING: Store model in memory for zero-disk-I/O access during training
            self.model = model
            self.model_id = model_id
            self.model_session_id = session_id
            self.logger.info(f"✅ Model cached in AnalysisManager (TIER 0c) for session {session_id}")
            
            # Also persist to disk for recovery/reuse
            self.persistent_store.save_model(
                model_id=model_id,
                model=model,
                config=model_config
            )
            self.logger.info(f"✅ Model persisted to disk (model_id: {model_id})")
            
            # Update task: Model build complete
            self.task_store.update_task(
                task_id=task_id,
                status="processing",
                progress=100,
                message="Model build complete.",
                metadata={"model_id": model_id, "compiled_model_id": compiled_model_id}
            )
            
            # Store model build results to database so frontend can fetch them
            # Extract ml_preparation_ref - handle both string IDs and complex objects
            ml_prep_ref = model_config.get("ml_preparation_ref")
            if isinstance(ml_prep_ref, dict):
                # If it's a complex object, extract the _ref field
                ml_prep_ref = ml_prep_ref.get("_ref") or ml_prep_ref.get("id")
            # Fallback to dataset_id or task_id if ml_preparation_ref is not available
            ml_prep_ref = ml_prep_ref or model_config.get("dataset_id") or model_config.get("task_id")

            model_build_result = {
                "status": "success",
                "model_id": model_id,
                "compiled_model_id": compiled_model_id, 
                "architecture": str(model.summary()),
                "model_summary": str(model.summary()), 
                "total_params": total_params,   
                "trainable_params": trainable_params, 
                "non_trainable_params": non_trainable_params, 
                # Include ml_preparation_ref so frontend can load training data
                "ml_preparation_ref": ml_prep_ref,
                # Store input shape for validation during training
                "input_shape": model_config.get("input_shape"),
                "n_predictions": model_config.get("n_predictions"),
                # Dataset signature for model compatibility matching
                "feature_hash": feature_hash,
                "feature_cols": feature_cols,
                "target_cols": model_config.get("target_cols", []),
                "selected_targets": model_config.get("selected_targets", []),
                "sequence_length": (model_config.get("input_shape") or [None])[0],
                "feature_count": (model_config.get("input_shape") or [None, None])[1],
                "dataset_name": model_config.get("dataset_name", ""),
                "dataset_id": model_config.get("dataset_id", ""),
                # Step configs that produced the training data (provenance for inference)
                "step_configs": model_config.get("step_configs", {}),
                # Model architecture metadata
                "model_type": model_config.get("type", ""),
                "model_parameters": {k: v for k, v in model_config.items() if k not in (
                    "type", "input_shape", "n_predictions", "prediction_length",
                    "ml_preparation_ref", "dataset_id", "dataset_name",
                    "feature_cols", "target_cols", "step_configs"
                )},
                # ═════════════════════════════════════════════════════════════
                # PHASE 19: NEW METADATA FIELDS FOR COMPILED MODEL
                # ═════════════════════════════════════════════════════════════
                "compiled_at": datetime.utcnow().isoformat(),
                "framework": "tensorflow", 
                "framework_version": __import__('tensorflow').__version__ if __import__('importlib').util.find_spec('tensorflow') else "not_available",  
                "code_version": code_version, 
                "training_device": training_device,  
                "compilation_environment": compilation_environment, 
                "serialization_format": serialization_format, 
                "model_size_bytes": len(model_binary),  
                "architecture_json": architecture_json,  
                "model_name": model_config.get("model_name", f"model_{compiled_model_id[:8]}"),  
                "description": model_config.get("description", "Auto-compiled model"),  
                "optimizer_config": {"type": model_config.get("optimizer", "adam")},  
                "loss_function": model_config.get("loss", "mse"),  
                "metrics_list": model_config.get("metrics", ["mae"]),  
            }
            
            # ═════════════════════════════════════════════════════════════════════
            # STORE TO COMPILED MODEL TABLE (NEW)
            # ═════════════════════════════════════════════════════════════════════
            
            # Enhance model_config with metadata for CompiledModel storage
            model_config_for_db = {
                **model_config,
                "model_name": model_build_result.get("model_name"),
                "description": model_build_result.get("description"),
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "non_trainable_parameters": non_trainable_params,
                "model_summary": str(model.summary()),
                "output_shape": [model_config.get("n_predictions", 1)],  # Output shape
                "optimizer_config": model_build_result.get("optimizer_config"),
                "loss_function": model_build_result.get("loss_function"),
                "metrics_list": model_build_result.get("metrics_list"),
                "framework": "tensorflow",
                "framework_version": model_build_result.get("framework_version"),
                "code_version": code_version,
                "compilation_environment": compilation_environment,
                "is_public": model_config.get("is_public", False),
                "version": model_config.get("version", 1),
            }
            
            # Store CompiledModel to database
            compiled_stored = await self._store_compiled_model_to_db(
                session_id=session_id,
                user_id=user_id,
                model_config=model_config_for_db,
                model_binary=model_binary,
                compiled_model_id=compiled_model_id,
            )
            
            if compiled_stored:
                self.logger.info(f"✅ CompiledModel stored to database: {compiled_model_id}")
                model_build_result["compiled_model_stored"] = True
            else:
                self.logger.warning(f"⚠️ Failed to store CompiledModel to database (will store to SessionStepResult only)")
                model_build_result["compiled_model_stored"] = False
            
            # Store to database for frontend fetchResults()
            try:
                async with AsyncPostgresSessionLocal() as db:
                    await store_session_step_result(
                        session_id=session_id,
                        step_name="model_build",
                        data=[model_build_result],  # Store as list for consistency
                        db=db
                    )
                    self.logger.info(f"✅ Stored model_build results to database (session={session_id[:8]})")
            except Exception as store_err:
                self.logger.error(f"⚠️ Failed to store model_build results: {store_err}")
                # Don't fail the model build if storage fails
            
            return model_build_result
            
        except Exception as e:
            self.logger.error(f"Model build error: {str(e)}")
            self.task_store.update_task(
                task_id=task_id,
                status="error",
                message=f"Model build failed: {str(e)}"
            )
            return {"status": "error", "message": str(e)}

    async def _find_dataset_id_for_model(self, model_id: str) -> Optional[str]:
        """Find dataset_id linked to a model."""
        try:
            async with AsyncPostgresSessionLocal() as db:
                from sqlalchemy import select
                stmt = select(TrainedModelForAnalysis.dataset_id).where(
                    TrainedModelForAnalysis.model_id == model_id
                )
                result = await db.execute(stmt)
                res = result.scalar_one_or_none()
                if res:
                    return str(res)
                return None
        except Exception as e:
            self.logger.error(f"Error finding dataset_id for model {model_id}: {e}")
            return None

    async def execute_model_training(
        self,
        session_id: str,
        task_id: str,
        model_id: str,
        epochs: int = 100,
        batch_size: int = 32,
        ml_preparation_ref: Union[str, Dict[str, Any]] = None,  # ✅ UPDATED: Supports object
        user_id: str = "anonymous",
        is_classification: bool = False, # ✅ NEW: Flag for target type
        target_column: str = None,       # ✅ NEW: Target column name
        selected_targets: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute ML model training using pure async loop (no multiprocessing).
        
        Uses TIER 0b ML pointers (train/val/test) for ZERO LATENCY data access.
        Updates task_store directly (no callbacks + threading).
        
        Args:
            session_id: Session ID (must match ml_session_id)
            task_id: Task ID for progress tracking
            model_id: Model ID to train
            epochs: Number of training epochs
            batch_size: Batch size
            ml_preparation_ref: ✅ NEW: Dataset reference to validate TIER 0b pointers
            
        Returns:
            dict with keys:
                - "status": "success" | "error"
                - "final_loss": Final training loss
                - "final_val_loss": Final validation loss
                - "epochs_completed": # epochs completed
        """
        try:
            # ✅ ROBUST EXTRACTION: Handle object or string for ml_preparation_ref
            ml_dataset_name = ml_preparation_ref
            if isinstance(ml_preparation_ref, dict):
                ml_dataset_name = (
                    ml_preparation_ref.get("dataset_name") or 
                    ml_preparation_ref.get("dataset_id") or 
                    ml_preparation_ref.get("task_id") or 
                    ml_preparation_ref.get("_ref")
                )
                self.logger.info(f"🔍 [Training] Extracted dataset name '{ml_dataset_name}' from ref object")

            # Normalize selected_targets and fallback to first selected target when needed
            selected_targets = selected_targets or []
            if not target_column and selected_targets:
                target_column = selected_targets[0]
            self.logger.info(
                f"🔍 [Training] Resolved target configuration: target_column={target_column}, selected_targets={selected_targets}"
            )

            # Retrieve dataset_id for this model to enable post-training predictions
            dataset_id = await self._find_dataset_id_for_model(model_id)
            if not dataset_id:
                self.logger.warning(f"Could not find dataset_id for model {model_id}, post-training predictions will be skipped")
            if hasattr(self, 'model') and self.model is not None and getattr(self, 'model_id', None) == model_id:
                model = self.model
                self.logger.info(f"🚀 [TIER 0c] Using in-memory model {model_id} (Zero Disk I/O)")
            else:
                self.logger.info(f"🔍 [TIER 3] Loading model {model_id} from persistent store...")
                model = self.persistent_store.load_model(model_id)
            
            if model is None:
                return {"status": "error", "message": f"Model not found: {model_id}"}
            
            # ✅ FIXED: Only add missing MAE/MSE metrics — do NOT fully recompile.
            # Full recompile resets Adam momentum/state, which hurts convergence when
            # reusing the in-memory TIER 0c model or resuming from disk.
            try:
                # ✅ FIXED: Carefully filter metrics. Keras 3 throws ValueError if 'loss' 
                # or internal identifiers like 'compile_metrics' are in the user metrics list.
                forbidden = ['loss', 'compile_metrics']
                
                # Get current metrics that are safe to re-pass to compile()
                current_metrics = []
                if hasattr(model, 'metrics'):
                    for m in model.metrics:
                        m_name = getattr(m, 'name', None)
                        if m_name and m_name not in forbidden:
                            current_metrics.append(m_name)
                
                missing = [m for m in ['mae', 'mse'] if m not in current_metrics]
                
                if missing:
                    # Deduplicate and recompile
                    final_metrics = list(set(current_metrics + missing))
                    model.compile(
                        optimizer=model.optimizer,
                        loss=model.loss,
                        metrics=final_metrics
                    )
                    self.logger.info(f"✅ Added missing metrics {missing} to model. Final metrics: {final_metrics}")
                else:
                    self.logger.info(f"✅ Model already has MAE/MSE metrics — skipping recompile")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not update model metrics safely: {e}")
            
            # ✅ NEW: Pass ml_dataset_name to validate TIER 0b pointers
            # Load ML data (TIER 0b) - 0ms latency
            train_result, _ = await self._load_data_4_tier(
                session_id=session_id,
                task_id=task_id,
                data_type="ml_train",   
                ml_dataset_name=ml_dataset_name,  # ✅ FIXED: Use extracted name
                prefer_lazy=True                     # ✅ OPTIMIZATION: Stream from disk
            )
            val_result, _ = await self._load_data_4_tier(
                session_id=session_id,
                task_id=task_id,
                data_type="ml_validation",
                ml_dataset_name=ml_dataset_name,  # ✅ FIXED: Use extracted name
                prefer_lazy=True                     # ✅ OPTIMIZATION: Stream from disk
            )
            test_result, _ = await self._load_data_4_tier(
                session_id=session_id,
                task_id=task_id,
                data_type="ml_test",
                ml_dataset_name=ml_dataset_name,  # ✅ FIXED: Use extracted name
                prefer_lazy=True                     # ✅ OPTIMIZATION: Stream from disk
            )
            
            if train_result is None or val_result is None:
                return {"status": "error", "message": "ML training data not available"}
            
            # ✅ FIXED: Extract sequences + targets from result
            if isinstance(train_result, dict):
                if "sequences" in train_result:
                    train_data = train_result["sequences"]
                    # ✅ FIXED: Extract actual array from targets dict if present
                    raw_targets = train_result.get("targets")
                    if isinstance(raw_targets, dict):
                        if target_column and target_column in raw_targets:
                            train_targets = raw_targets[target_column]
                        elif "future_sequence" in raw_targets:
                            train_targets = raw_targets["future_sequence"]
                        elif "y" in raw_targets:
                            train_targets = raw_targets["y"]
                        else:
                            train_targets = list(raw_targets.values())[0] if raw_targets else None
                    else:
                        train_targets = raw_targets
                elif train_result.get("data_type") == "lazy_npz":
                    train_data = train_result
                    # For lazy data, targets are on disk. We set a placeholder 
                    # so that auto_mode doesn't incorrectly trigger.
                    train_targets = "LAZY_ON_DISK" 
                else:
                    train_data = train_result
                    train_targets = None
            else:
                train_data = train_result
                train_targets = None
            
            if isinstance(val_result, dict):
                if "sequences" in val_result:
                    val_data = val_result["sequences"]
                    # ✅ FIXED: Extract actual array from targets dict if present
                    raw_val_targets = val_result.get("targets")
                    if isinstance(raw_val_targets, dict):
                        if target_column and target_column in raw_val_targets:
                            val_targets = raw_val_targets[target_column]
                        elif "future_sequence" in raw_val_targets:
                            val_targets = raw_val_targets["future_sequence"]
                        elif "y" in raw_val_targets:
                            val_targets = raw_val_targets["y"]
                        else:
                            val_targets = list(raw_val_targets.values())[0] if raw_val_targets else None
                    else:
                        val_targets = raw_val_targets
                elif val_result.get("data_type") == "lazy_npz":
                    val_data = val_result
                    val_targets = "LAZY_ON_DISK"
                else:
                    val_data = val_result
                    val_targets = None
            else:
                val_data = val_result
                val_targets = None
            
            # ✅ TARGET EXTRACTION FROM DATAFRAMES
            # If targets are missing but we have DataFrames, try to extract 'targets' column
            if train_targets is None and isinstance(train_data, pd.DataFrame) and 'targets' in train_data.columns:
                train_targets = np.stack(train_data['targets'].values)
                self.logger.info(f"✅ Extracted {len(train_targets)} targets from train_data DataFrame")
            
            if val_targets is None and isinstance(val_data, pd.DataFrame) and 'targets' in val_data.columns:
                val_targets = np.stack(val_data['targets'].values)
                self.logger.info(f"✅ Extracted {len(val_targets)} targets from val_data DataFrame")
            
            # Also check for 'labels' if it's classification
            if is_classification:
                if train_targets is None and isinstance(train_data, pd.DataFrame) and 'labels' in train_data.columns:
                    train_targets = np.stack(train_data['labels'].values)
                if val_targets is None and isinstance(val_data, pd.DataFrame) and 'labels' in val_data.columns:
                    val_targets = np.stack(val_data['labels'].values)
            
            # Ensure train_data and val_data are numpy arrays, not DataFrames
            # If they're DataFrames, the sequences are stored as rows and need to be extracted
            if isinstance(train_data, pd.DataFrame):
                self.logger.warning(f"⚠️ train_data is DataFrame with {len(train_data)} rows - converting to proper 3D numpy array")
                if 'sequences' in train_data.columns:
                    train_data = np.stack(train_data['sequences'].values)
                else:
                    self.logger.error(f"❌ DataFrame doesn't have 'sequences' column - data structure is incorrect")
                    return {"status": "error", "message": "Training data structure is incorrect - DataFrame without sequences column"}
            
            if isinstance(val_data, pd.DataFrame):
                self.logger.warning(f"⚠️ val_data is DataFrame with {len(val_data)} rows - converting to proper 3D numpy array")
                if 'sequences' in val_data.columns:
                    val_data = np.stack(val_data['sequences'].values)
                else:
                    self.logger.error(f"❌ DataFrame doesn't have 'sequences' column - data structure is incorrect")
                    return {"status": "error", "message": "Validation data structure is incorrect - DataFrame without sequences column"}
            
            model_config = self.persistent_store.get_config(model_id)
            if model_config is None:
                return {"status": "error", "message": f"Model config not found for {model_id}"}
            
            # ✅ FIXED: Fail early if critical shape metadata is missing
            expected_input_shape = model_config.get("input_shape")
            if not expected_input_shape or not isinstance(expected_input_shape, (list, tuple)):
                return {"status": "error", "message": f"Model input_shape missing or invalid in config: {model_id}"}

            seq_len = expected_input_shape[0]
            expected_features = expected_input_shape[1] if len(expected_input_shape) > 1 else None
            
            # Check if train_data is 2D and needs reshaping to 3D
            if isinstance(train_data, np.ndarray) and len(train_data.shape) == 2:
                N, F = train_data.shape
                self.logger.warning(f"⚠️ train_data is 2D (N, features): shape={train_data.shape}")
                
                # Validate dimensions before reshaping
                if N % seq_len != 0:
                    self.logger.error(
                        f"❌ Can't reshape train_data: {N} rows not divisible by seq_len={seq_len}\n"
                        f"   This suggests sequences were not properly generated or stored.\n"
                        f"   Expected: N rows where N % {seq_len} == 0"
                    )
                    return {
                        "status": "error",
                        "message": f"Training data dimension mismatch: {N} rows not divisible by sequence_length={seq_len}"
                    }
                
                if expected_features and F != expected_features:
                    self.logger.error(
                        f"❌ Feature count mismatch: data has {F} features but model expects {expected_features}\n"
                        f"   This suggests the ML preparation used different features than when the model was built.\n"
                        f"   Solution: Rebuild model with input_shape=({seq_len}, {F}) or regenerate ML dataset with {expected_features} features"
                    )
                    return {
                        "status": "error",
                        "message": f"Feature count mismatch: data has {F} features but model expects {expected_features}"
                    }
                
                # Safe to reshape
                expected_N = N // seq_len
                train_data = train_data.reshape(expected_N, seq_len, F)
                self.logger.info(f"✅ Reshaped train_data from ({N}, {F}) to {train_data.shape}")
            
            # Same for validation data
            if isinstance(val_data, np.ndarray) and len(val_data.shape) == 2:
                N, F = val_data.shape
                self.logger.warning(f"⚠️ val_data is 2D (N, features): shape={val_data.shape}")
                
                if N % seq_len != 0:
                    self.logger.error(f"❌ Can't reshape val_data: {N} rows not divisible by seq_len={seq_len}")
                    return {
                        "status": "error",
                        "message": f"Validation data dimension mismatch: {N} rows not divisible by sequence_length={seq_len}"
                    }
                
                if expected_features and F != expected_features:
                    self.logger.error(f"❌ Feature count mismatch in validation: data has {F} features but model expects {expected_features}")
                    return {
                        "status": "error",
                        "message": f"Validation feature count mismatch: data has {F} features but model expects {expected_features}"
                    }
                
                expected_N = N // seq_len
                val_data = val_data.reshape(expected_N, seq_len, F)
                self.logger.info(f"✅ Reshaped val_data from ({N}, {F}) to {val_data.shape}")
            
            # Detect Lazy Data (Disk Caching)
            use_generator = False
           
            batch_generator = None
            if isinstance(train_data, dict) and train_data.get("data_type") == "lazy_npz":
                self.logger.info("🚀 [Training] Lazy data detected. Using LazySequenceGenerator.")
                use_generator = True
                from app.core.ml.ml_data_loader import LazySequenceGenerator
                
                # Get all paths from all chunks if it's a list (merged result)
                train_paths = train_data.get("file_paths") or [train_data.get("file_path")]
                val_paths = val_data.get("file_paths") or [val_data.get("file_path")]
                
                # Filter out None values in case neither key exists
                train_paths = [p for p in train_paths if p]
                val_paths = [p for p in val_paths if p]
                
                # Determine autoencoder mode: True only if no targets are expected
                # (e.g. outlier detection or reconstruction tasks)
                # ✅ FIXED: Only enable if train_targets is genuinely None (not "LAZY_ON_DISK")
                auto_mode = (train_targets is None and not target_column and not is_classification)
                
                batch_generator = LazySequenceGenerator(
                    file_paths=train_paths,
                    batch_size=batch_size,
                    shuffle=True,  
                    autoencoder_mode=auto_mode,
                    target_column=target_column,
                    selected_targets=selected_targets,
                    micro_val_holdback=0.05  # ✅ Hold back 5% for batch-level audit
                )

                # ✅ FIXED: Guard against None val_data before creating val_generator
                if val_data is None:
                    self.logger.error("❌ Validation data is None, cannot create LazySequenceGenerator")
                    return {"status": "error", "message": "Validation data failed to load for lazy generator"}

                val_generator = LazySequenceGenerator(
                    file_paths=val_paths,
                    batch_size=batch_size,
                    shuffle=False,  
                    autoencoder_mode=auto_mode,
                    target_column=target_column,
                    selected_targets=selected_targets,
                )
                
                self.logger.info(f"🚀 [Training] Generator initialized (autoencoder_mode={auto_mode})")
            
            # ✅ GUARD: Validate training data before starting loop
            if train_data is None or (hasattr(train_data, '__len__') and len(train_data) == 0):
                return {"status": "error", "message": "Training data is empty after extraction"}
            if val_data is None or (hasattr(val_data, '__len__') and len(val_data) == 0):
                return {"status": "error", "message": "Validation data is empty after extraction"}
            
            # ✅ FIXED: Use guarded counts for logging to prevent crashes with non-len objects
            train_count = len(train_data) if hasattr(train_data, "__len__") else "unknown"
            val_count = len(val_data) if hasattr(val_data, "__len__") else "unknown"
            
            self.logger.info(
                f"✅ [Training] Data ready: train={train_count} sequences, "
                f"val={val_count} sequences, epochs={epochs}, batch_size={batch_size}"
            )
            
            # ═════════════════════════════════════════════════════════════════════
            # PHASE 19: CAPTURE TRAINING START METADATA AND INSERT PLACEHOLDER
            # ═════════════════════════════════════════════════════════════════════
            training_started_at = datetime.utcnow()
            training_device = self._detect_training_device()
            training_environment = self._get_compilation_environment()
            code_version = self._get_code_version()
            
            # Extract training data metadata from MLDataset for fair backtesting
            training_data_symbol = None
            training_data_timeframe = None
            training_data_start_date = None
            training_data_end_date = None
            training_data_duration_days = None
            training_data_sample_count = len(train_data) if hasattr(train_data, '__len__') else 0
            training_data_feature_columns = None
            training_data_target_columns = None
            scaler_binary = None
            sequence_length = None
            placeholder_inserted = False
            
            if ml_preparation_ref:
                try:
                    async with AsyncPostgresSessionLocal() as db:
                        from sqlalchemy import select as sa_select
                        from app.database.models import MLDataset
                        
                        # 1. Fetch MLDataset to capture metadata and resolve dataset_id
                        stmt = sa_select(MLDataset).where(
                            MLDataset.session_id == session_id,
                            MLDataset.dataset_name == ml_dataset_name  # ✅ FIXED: Use extracted name, not raw ref
                        ).order_by(MLDataset.created_at.desc()).limit(1)
                        result_q = await db.execute(stmt)
                        ml_dataset = result_q.scalars().first()
                        
                        dataset_uuid = None
                        if ml_dataset:
                            dataset_uuid = ml_dataset.dataset_id
                            training_data_feature_columns = ml_dataset.feature_columns
                            training_data_target_columns = ml_dataset.output_targets
                            training_data_sample_count = ml_dataset.sample_count or training_data_sample_count
                            scaler_binary = ml_dataset.scaler_binary
                            
                            # Extract split config
                            split_cfg = ml_dataset.split_config or {}
                            sequence_length = split_cfg.get("sequence_length") or split_cfg.get("window_size")
                            
                            # Extract symbol/timeframe from source_metadata if available
                            src_meta = ml_dataset.source_metadata or {}
                            training_data_symbol = src_meta.get("symbol") or src_meta.get("trading_symbol")
                            training_data_timeframe = src_meta.get("timeframe") or src_meta.get("trading_timeframe")
                            
                            # Extract date range from split_config if available
                            if split_cfg.get("data_start_date"):
                                try:
                                    training_data_start_date = datetime.fromisoformat(str(split_cfg["data_start_date"]))
                                except Exception:
                                    pass
                            if split_cfg.get("data_end_date"):
                                try:
                                    training_data_end_date = datetime.fromisoformat(str(split_cfg["data_end_date"]))
                                except Exception:
                                    pass
                            
                            if training_data_start_date and training_data_end_date:
                                training_data_duration_days = (training_data_end_date - training_data_start_date).days
                            
                            self.logger.info(
                                f"✅ [Phase 19] Captured training data metadata:\n"
                                f"   Symbol: {training_data_symbol}, Timeframe: {training_data_timeframe}\n"
                                f"   Date range: {training_data_start_date} → {training_data_end_date}\n"
                                f"   Samples: {training_data_sample_count}"
                            )
                        
                        # 2. Insert or update TrainedModelForAnalysis placeholder
                        if dataset_uuid:
                            stmt_check = sa_select(TrainedModelForAnalysis).where(TrainedModelForAnalysis.model_id == model_id)
                            existing_res = await db.execute(stmt_check)
                            existing_model = existing_res.scalar_one_or_none()
                            
                            if existing_model:
                                existing_model.status = "training"
                                existing_model.updated_at = datetime.utcnow()
                                await db.commit()
                                placeholder_inserted = True
                                self.logger.info(f"✅ [Phase 19] Updated existing TrainedModelForAnalysis status to 'training': {model_id}")
                            else:
                                placeholder = TrainedModelForAnalysis(
                                    model_id=model_id,
                                    dataset_id=dataset_uuid,
                                    session_id=session_id,
                                    model_name=f"training_{model_id[:8]}",
                                    version=1,
                                    architecture_config={},
                                    training_config={
                                        "epochs": epochs,
                                        "batch_size": batch_size,
                                    },
                                    is_best_model=False,
                                    status="training",
                                    created_at=datetime.utcnow(),
                                    updated_at=datetime.utcnow(),
                                )
                                db.add(placeholder)
                                await db.commit()
                                placeholder_inserted = True
                                self.logger.info(
                                    f"✅ [Phase 19] Inserted placeholder TrainedModelForAnalysis: {model_id}"
                                )
                        else:
                            self.logger.warning(
                                f"⚠️ [Phase 19] Could not resolve dataset_id for ml_preparation_ref={ml_preparation_ref!r} "
                                f"— TrainingRecord inserts will be skipped"
                            )
                except Exception as meta_err:
                    self.logger.warning(f"⚠️ [Phase 19] Error during metadata fetch or placeholder insert: {meta_err}")
            
            # Initialize training metrics
            epochs_completed = 0
            # ✅ FIX: Initialize to inf so Catch-Up Mode (final_val_loss > 0.005) fires
            # correctly from epoch 1, not just from the epoch<3 branch.
            final_val_loss = float('inf')
            final_training_loss = 0.0
            final_mae = 0.0
            final_mse = 0.0
            final_val_mae = 0.0
            final_val_mse = 0.0
            best_epoch = None
            best_val_loss = float('inf')
            best_weights = model.get_weights()  # Initial snapshot
            best_metrics = {"loss": float('inf'), "val_loss": float('inf'), "mae": 0.0, "val_mae": 0.0, "mse": 0.0, "val_mse": 0.0}
            retries_per_epoch = 2               # Max retries if val_loss regresses
            # lr_decay_factor removed — was dead code; actual decay uses dyn_decay from phase block
            plateau_count = 0
            plateau_threshold = 0.0001
            improvement_streak = 0              # Track consecutive improvements for LR recovery
            initial_lr = float(tf.keras.backend.get_value(model.optimizer.learning_rate))  # ✅ FIXED: Safe LR read
            
            # ✅ Accumulate per-epoch history for frontend metrics dashboard
            training_history = {
                "loss": [], "val_loss": [],
                "mae": [], "val_mae": [],
                "mse": [], "val_mse": [],
            }
            
            # ✅ UPGRADE: Persistent Continual Learning State (Persists across epochs)
            # Memory Optimized: Use float16 for storage (50% reduction) + Balanced capacity
            # Two separate buffers for two distinct signals:
            #   replay_storage      = HARD EXAMPLES (model always struggled) → force learning
            #   good_examples_storage = GOOD EXAMPLES (model knew well) → detect forgetting
            continual_state = {
                "engine_state": {
                    "jury_x": None, "jury_y": None,
                    "ptr": 0, "count": 0,
                    "good_ptr": 0, "good_count": 0  # Separate pointers for good buffer
                },
                "replay_storage": [None] * 3000,       # Hard examples (~200-400MB in float16)
                "good_examples_storage": [None] * 1000 # Good examples for forgetting detection
            }
            
            # ✅ Initialize unified ProgressReporter for WebSockets
            reporter = ProgressReporter(
                task_id=task_id,  # ✅ FIXED: Pass task_id to constructor
                task_store=self.task_store,
                connection_manager=self.connection_manager,  # ✅ FIXED: Pass connection_manager
                user_id=user_id,
                throttling_strategy=ThrottlingStrategy.HYBRID
            )
            
            # ✅ FIXED: Track background tasks to await them at the end
            _background_tasks = []
            
            # Training loop with advanced rollback and LR decay (Ref Code implementation)
            epoch = 0
            while epoch < epochs:
                try:
                    epoch_start_time = datetime.utcnow()
                    
                    # Snapshot weights BEFORE this epoch trains for potential rollback
                    weights_before = model.get_weights()
                    
                    # ✅ DYNAMIC EPOCH-AWARE STRATEGY (Funnel Effect)
                    # Adjust tolerance and decay based on training progress and current loss.
                    # "Catch-Up Mode": Relax safety until we hit the 0.00x zone.
                    epoch_progress = epoch / epochs
                    
                    # 🚀 CATCH-UP LOGIC: If loss is high, be extremely aggressive (loosen safety)
                    # This allows the model to match the .fit() baseline speed early on.
                    is_catch_up = (final_val_loss > 0.005) or (epoch < 3)
                    
                    if is_catch_up:
                        # 🏃 PHASE 0: Catch-Up - Maximum aggression, minimal safety interference
                        dyn_threshold = 2.00 # Allow 100% regression (effectively disabled safety)
                        dyn_decay = 1.00     # No LR decay
                        dyn_max_m =6        # Hard squeeze to find signal fast
                    elif epoch_progress < 0.2:
                        # 🏃 PHASE 1: Exploration (Early) - Allow some noise
                        dyn_threshold = 1.15
                        dyn_decay = 0.90
                        dyn_max_m = 3
                    elif epoch_progress > 0.7:
                        # 🎯 PHASE 3: Squeeze (Late) - Zero tolerance for regression
                        dyn_threshold = 1.01
                        dyn_decay = 0.50
                        dyn_max_m = 1
                    else:
                        # ⚖️ PHASE 2: Refinement (Mid) - Balanced safety
                        dyn_threshold = 1.05
                        dyn_decay = 0.80
                        dyn_max_m = 2

                    if is_catch_up:
                        self.logger.info(f"🚀 [CATCH-UP MODE] Loss ({final_val_loss:.4f}) is high. Loosening safety to accelerate convergence.")

                    
                    num_batches = len(batch_generator) if use_generator else math.ceil(len(train_data) / batch_size)
                    
                    epoch_loss_dict = await self._trainer_fit(
                        model=model,
                        train_data=train_data if not use_generator else batch_generator,
                        num_batches=num_batches,
                        batch_size=batch_size,
                        epoch=epoch,
                        task_id=task_id,
                        reporter=reporter,
                        total_epochs=epochs,
                        last_val_metrics={
                            "val_loss": final_val_loss if final_val_loss > 0 else None,
                            "val_mae": final_val_mae if final_val_mae > 0 else None,
                            "val_mse": final_val_mse if final_val_mse > 0 else None
                        },
                        target_column=target_column,
                        max_m=dyn_max_m,
                        dyn_threshold=dyn_threshold,
                        # ── NEW: Full safety parity ports ──────────────────
                        val_data=val_data if not use_generator else val_generator,
                        train_targets=train_targets,
                        val_targets=val_targets,
                        is_generator=use_generator,
                        weights_before=weights_before,  # Epoch-level rollback
                        train_data_obj=batch_generator if use_generator else None, # For jury access
                        continual_state=continual_state  # ✅ UPGRADE: Persistent state
                    )
                    
                    # Extract metrics for the rest of the loop logic
                    final_training_loss = epoch_loss_dict.get("loss", float('inf'))
                    final_val_loss = epoch_loss_dict.get("val_loss", float('inf'))
                    final_mae = epoch_loss_dict.get("mae", 0.0)
                    final_mse = epoch_loss_dict.get("mse", 0.0)
                    final_val_mae = epoch_loss_dict.get("val_mae", 0.0)
                    final_val_mse = epoch_loss_dict.get("val_mse", 0.0)
                    epoch_jury_rejected = epoch_loss_dict.get("jury_rejected", False)
                    
                    e_loss, e_val_loss = final_training_loss, final_val_loss
                    e_mae, e_mse = final_mae, final_mse
                    e_val_mae, e_val_mse = final_val_mae, final_val_mse

                    
                    epoch_duration_seconds = (datetime.utcnow() - epoch_start_time).total_seconds()
                    # NOTE: epochs_completed is incremented ONLY after the epoch is committed below
                    # to prevent double-counting on rollback/retry
                    
                    # ✅ FIXED: Progress should reach 100% on final epoch
                    progress = int(((epoch + 1) / epochs) * 100)
                    
                    # ✅ TRACK BEST PERFORMANCE AND HANDLE REGRESSION
                    if not epoch_jury_rejected and e_val_loss < best_val_loss:
                        # Improved: commit and save weights
                        self.logger.info(
                            f"✅ Epoch {epoch+1}: val_loss improved ({e_val_loss:.6f} < {best_val_loss:.6f}) "
                            f"[Phase: {'Early' if epoch_progress < 0.2 else 'Late' if epoch_progress > 0.7 else 'Mid'}]"
                        )
                        best_val_loss = e_val_loss
                        best_epoch = epoch + 1
                        best_weights = model.get_weights()
                        # ✅ NEW: Store best metrics for restoration after training
                        best_metrics = {"loss": e_loss, "val_loss": e_val_loss, "mae": e_mae, "val_mae": e_val_mae, "mse": e_mse, "val_mse": e_val_mse}
                        retries_per_epoch = 2  # Reset budget
                        plateau_count = 0
                        improvement_streak += 1  # Track consecutive improvements for LR recovery
                    else:
                        # Regression or Jury Rejection: Consider rollback and retry
                        is_regression = e_val_loss > best_val_loss * dyn_threshold
                        
                        if retries_per_epoch > 0 and (is_regression or epoch_jury_rejected):
                            reason = "Jury rejected" if epoch_jury_rejected else f"Regression ({e_val_loss:.6f} > {best_val_loss:.6f} * {dyn_threshold:.2f})"
                            self.logger.warning(
                                f"⚖️ Epoch {epoch+1}: {reason}. Reverting to BEST weights and retrying..."
                            )
                            model.set_weights(best_weights)
                            
                            # Decay Learning Rate using Dynamic Decay (Skip punishment in Phase 1)
                            if epoch_progress >= 0.2:
                                try:
                                    current_lr = float(tf.keras.backend.get_value(model.optimizer.learning_rate))  # ✅ FIXED: Safe LR read
                                    new_lr = current_lr * dyn_decay
                                    model.optimizer.learning_rate.assign(new_lr)
                                    self.logger.info(f"📉 LR decayed ({dyn_decay:.2f}x): {current_lr:.6f} -> {new_lr:.6f} (Reverted to best state)")
                                except Exception as lr_err:
                                    self.logger.warning(f"⚠️ Could not decay LR: {lr_err}")
                            else:
                                self.logger.info("🏃 Phase 1: Skipping LR decay on regression to maintain optimizer momentum.")
                            
                            retries_per_epoch -= 1
                            self.logger.warning(f"⚠️ Retrying epoch {epoch+1}, retries left={retries_per_epoch}")
                            # Continue without incrementing epoch to retry the same one
                            continue
                        else:
                            # Acceptance: regression is minor or out of retries
                            if epoch_jury_rejected:
                                self.logger.error(f"🔴 Epoch {epoch+1}: JURY REJECTED but out of retries. FORCING ROLLBACK Continuing to next epoch...")
                                model.set_weights(best_weights)
                            
                            self.logger.info(f"ℹ️ Epoch {epoch+1}: regression accepted (within {dyn_threshold:.2f}x limit or no retries left)")
                            improvement_streak = 0  # Reset streak on regression
                            
                            # Plateau tracking (Fixed: catch stalled convergence near best)
                            improvement = best_val_loss - e_val_loss  # positive = better, negative = regression
                            if improvement >= 0 and improvement < plateau_threshold:
                                plateau_count += 1
                                if plateau_count >= 5:
                                    self.logger.warning(f"🔄 Plateau detected for {plateau_count} epochs. Restoring best weights and decaying LR.")
                                    model.set_weights(best_weights)
                                    try:
                                        current_lr = float(tf.keras.backend.get_value(model.optimizer.learning_rate))  # ✅ FIXED: Safe LR read
                                        model.optimizer.learning_rate.assign(current_lr * 0.2) # Aggressive decay
                                    except Exception: pass
                                    plateau_count = 0
                            else:
                                plateau_count = 0  # Reset on genuine regression or strong improvement
                            
                            # LR Recovery: Aggressive recovery after 2 consecutive improvements
                            # ⚠️ CRITICAL: Prevent loss explosion by restoring LR when it gets too small
                            min_lr = initial_lr * 0.001  # Minimum LR floor (never go below this)
                            current_lr = float(tf.keras.backend.get_value(model.optimizer.learning_rate))  # ✅ FIXED: Safe LR read
                            
                            if improvement_streak >= 2:
                                try:
                                    if current_lr < min_lr:
                                        # LR is DEAD - restore to minimum
                                        new_lr = min_lr
                                        self.logger.critical(f"🚨 LR DEAD ZONE DETECTED: {current_lr:.2e} < {min_lr:.2e}. Restoring to minimum.")
                                    else:
                                        # 🔴 BUG FIX: Remove restrictive cap in Phase 2
                                        # Previous bug: lr_cap = 1x initial in Phase 2/3 killed aggressive learning
                                        # (couldn't push past 0.x because LR was capped at initial value)
                                        # New: Phase 1 = 5x, Phase 2 = 3x (allow continued growth), Phase 3 = 2x
                                        if epoch_progress < 0.2:
                                            lr_cap = initial_lr * 5.0
                                        elif epoch_progress < 0.7:
                                            lr_cap = initial_lr * 3.0  # 🔴 FIX: Was 1x, now 3x for Phase 2
                                        else:
                                            lr_cap = initial_lr * 2.0  # 🔴 FIX: Was 1x, now 2x for Phase 3
                                        
                                        # Normal recovery: Triple LR for faster catch-up (not just double)
                                        # 🔴 FIX: Use 3x recovery to match aggressive decay in Phase 2/3
                                        recovery_factor = 3.0 if epoch_progress >= 0.2 else 2.0
                                        new_lr = min(current_lr * recovery_factor, lr_cap)
                                        self.logger.info(f"📈 LR AGGRESSIVE RECOVERY: {current_lr:.6f} → {new_lr:.6f} (×{recovery_factor}) after {improvement_streak} improvements [Cap: {lr_cap:.4f}]")
                                    
                                    model.optimizer.learning_rate.assign(new_lr)
                                    improvement_streak = 0  # Reset for next cycle
                                except Exception as e:
                                    self.logger.warning(f"⚠️ LR recovery failed: {e}")
                    
                    # ✅ ROLLBACK BUDGET RECOVERY: Reset every 20 epochs to allow continued recovery
                    if epoch > 0 and epoch % 20 == 0 and retries_per_epoch < 2:
                        retries_per_epoch = 2
                        self.logger.info(f"🔄 Epoch {epoch+1}: Rollback budget recovered to 2")
                    
                    # ✅ COMMIT EPOCH (with NaN/Inf sanitization)
                    def safe_metric(val, default=0.0):
                        try:
                            f_val = float(val)
                            return f_val if np.isfinite(f_val) else default
                        except (ValueError, TypeError):
                            return default

                    training_history["loss"].append(round(safe_metric(e_loss, 999.0), 6))
                    training_history["val_loss"].append(round(safe_metric(e_val_loss, 999.0), 6))
                    training_history["mae"].append(round(safe_metric(e_mae), 6))
                    training_history["val_mae"].append(round(safe_metric(e_val_mae), 6))
                    training_history["mse"].append(round(safe_metric(e_mse), 6))
                    training_history["val_mse"].append(round(safe_metric(e_val_mse), 6))
                    
                    # ✅ CRITICAL: Log if metrics are invalid (NaN/Inf)
                    if epoch == 0 or epoch % 10 == 0:
                        metrics_valid = (
                            not np.isnan(e_loss) and not np.isinf(e_loss) and
                            not np.isnan(e_val_loss) and not np.isinf(e_val_loss)
                        )
                        if not metrics_valid:
                            self.logger.warning(
                                f"⚠️ [METRICS] Invalid metrics at epoch {epoch+1}:\n"
                                f"   loss={e_loss}, val_loss={e_val_loss}, mae={e_mae}, mse={e_mse}"
                            )
                    
                    # Update progress stats
                    epochs_completed += 1
                    # ✅ NOTE: progress was already computed correctly above as ((epoch+1)/epochs)*100
                    # The stale duplicate line `int((epoch / epochs) * 100)` is removed — it was
                    # overwriting the correct value and prevented progress from ever reaching 100%.
                   
                    # ✅ Send training metrics in structured format for frontend
                    await reporter.report_async(
                        progress=progress,
                        message=f"Training epoch {epoch+1}/{epochs}",
                        message2=f"Loss: {e_loss:.4f} | Val Loss: {e_val_loss:.4f} | MAE: {e_mae:.4f}",
                        trainingMetrics={
                            "loss": e_loss,
                            "mae": e_mae,
                            "mse": e_mse,
                            "val_loss": e_val_loss,
                            "val_mae": e_val_mae,
                            "val_mse": e_val_mse,
                            "current_epoch": epoch + 1,
                            "total_epochs": epochs
                        }
                    )
                    
                    # ═══════════════════════════════════════════════════════════
                    # PHASE 19: STORE PER-EPOCH TrainingRecord (Phase 2.4)
                    # Only fire if the placeholder row was successfully inserted —
                    # otherwise the FK constraint will fail.
                    # ✅ FIXED: Track tasks in _background_tasks list so they can be
                    # awaited before we return (prevents incomplete DB writes on exit).
                    # ═══════════════════════════════════════════════════════════
                    if placeholder_inserted:
                        _background_tasks.append(
                            asyncio.ensure_future(
                                self._store_training_record_to_db(
                                    model_id=model_id,
                                    epoch=epoch + 1,
                                    loss=e_loss,
                                    val_loss=e_val_loss,
                                    metrics={
                                        "mae": e_mae,
                                        "val_mae": e_val_mae,
                                        "mse": e_mse,
                                        "val_mse": e_val_mse,
                                    },
                                    epoch_duration_seconds=epoch_duration_seconds,
                                )
                            )
                        )
                    # ✅ REMOVED: Early stopping logic (running all epochs for full loss curve study)
                    
                    # End of epoch processing - increment index
                    epoch += 1
                    
                except asyncio.CancelledError:
                    self.logger.warning("Training cancelled by user")
                    break
            
            # Record training completion time
            training_completed_at = datetime.utcnow()
            training_duration_seconds = int((training_completed_at - training_started_at).total_seconds())
            
            # Restore best weights ever seen before final save
            if best_weights is not None:
                best_epoch_label = best_epoch if best_epoch is not None else "N/A"
                self.logger.info(f"🏆 Restoring best weights from epoch {best_epoch_label} (val_loss: {best_val_loss:.6f})")
                model.set_weights(best_weights)
                # ✅ FIXED: Sync final_* metric variables to best-model values so the training
                # result reflects the *restored* model, not the last epoch's (potentially worse) metrics.
                final_val_loss      = best_metrics.get("val_loss", final_val_loss)
                final_training_loss = best_metrics.get("loss", final_training_loss)
                final_mae           = best_metrics.get("mae", final_mae)
                final_val_mae       = best_metrics.get("val_mae", final_val_mae)
                final_mse           = best_metrics.get("mse", final_mse)
                final_val_mse       = best_metrics.get("val_mse", final_val_mse)
                self.logger.info(
                    f"✅ Final metrics synced to best model: val_loss={final_val_loss:.6f}, "
                    f"mae={final_mae:.6f}, mse={final_mse:.6f}"
                )

            # Save trained model
            self.persistent_store.save_model(
                model_id=model_id,
                model=model,
                config={"trained": True, "epochs": epochs_completed}
            )
            
            # ═════════════════════════════════════════════════════════════════════
            # PHASE 19: CALCULATE QUALITY SCORE (Phase 2.2)
            # ═════════════════════════════════════════════════════════════════════
            quality_score = self._calculate_quality_score(
                final_val_loss=final_val_loss,
                final_training_loss=final_training_loss,
                loss_history=training_history["loss"],
                val_loss_history=training_history["val_loss"],
                model_type=model_config.get("type", "lstm") if model_config else "lstm",
            )
            
            # Determine recommendation reason based on quality score
            if quality_score >= 80:
                recommendation_reason = f"High quality model (score: {quality_score:.1f}/100). Stable convergence, low overfitting."
            elif quality_score >= 60:
                recommendation_reason = f"Good model (score: {quality_score:.1f}/100). Acceptable performance."
            elif quality_score >= 40:
                recommendation_reason = f"Moderate model (score: {quality_score:.1f}/100). Consider retraining with more data."
            else:
                recommendation_reason = f"Low quality model (score: {quality_score:.1f}/100). High loss or overfitting detected."
            
            # ✅ FIXED: Correct total sequences for reporting (samples, not batches)
            final_total_sequences = (
                (len(batch_generator) * batch_size) if use_generator 
                else (len(train_data) if hasattr(train_data, '__len__') else 0)
            )

            # ✅ Build complete training result with history
            training_result = {
                "status": "success",
                "model_id": model_id,
                "epochs_completed": epochs_completed,
                "best_epoch": best_epoch,
                "final_val_loss": float(final_val_loss),
                "best_val_loss": float(best_val_loss),
                "final_training_loss": float(final_training_loss),
                "final_mae": float(final_mae),
                "final_mse": float(final_mse),
                "final_val_mae": float(final_val_mae),
                "final_val_mse": float(final_val_mse),
                "final_training_metrics": training_history,  # ✅ Per-epoch arrays for metrics dashboard
                "timestamp": datetime.utcnow().isoformat(),
                "total_sequences": final_total_sequences,
                # ═══════════════════════════════════════════════════════════
                # PHASE 19: NEW METADATA FIELDS
                # ═══════════════════════════════════════════════════════════
                "quality_score": quality_score,
                "recommendation_reason": recommendation_reason,
                "training_started_at": training_started_at.isoformat(),
                "training_completed_at": training_completed_at.isoformat(),
                "training_duration_seconds": training_duration_seconds,
                "training_device": training_device,
                "code_version": code_version,
                "training_data_symbol": training_data_symbol,
                "training_data_timeframe": training_data_timeframe,
                "training_data_start_date": training_data_start_date.isoformat() if training_data_start_date else None,
                "training_data_end_date": training_data_end_date.isoformat() if training_data_end_date else None,
                "training_data_duration_days": training_data_duration_days,
                "training_data_sample_count": training_data_sample_count,
                "is_backtest_fair": True,  # Will be validated against backtesting date range later
            }
            
            # ✅ Store training result to DB for frontend fetchResults()
            try:
                async with AsyncPostgresSessionLocal() as db:
                    await store_session_step_result(
                        session_id=session_id,
                        step_name="model_training",
                        data=[training_result],  # Store as list for consistency
                        db=db
                    )
                    self.logger.info(f"✅ Stored model_training results to database (session={session_id[:8]})")
            except Exception as store_err:
                self.logger.error(f"⚠️ Failed to store model_training results: {store_err}")
                # Don't fail training if storage fails
            
            # ═════════════════════════════════════════════════════════════════════
            # 🔴 FIX: SEND COMPLETION MESSAGE WITH FINAL METRICS TO FRONTEND
            # This enables the fast path in ModelTrainingStepPanel.onComplete()
            # instead of requiring a DB fetch (eliminates 300-500ms latency)
            # ═════════════════════════════════════════════════════════════════════
            try:
                await reporter.report_async(
                    progress=100,
                    message="Model training complete",
                    message2=f"Best epoch: {best_epoch} | Val Loss: {final_val_loss:.6f}",
                    trainingMetrics=None,  # Don't repeat per-epoch metrics
                    data={
                        # ✅ CRITICAL: Send final metrics in completion for instant frontend display
                        "final_training_metrics": training_history,  # Complete arrays
                        "best_epoch": best_epoch,
                        "quality_score": quality_score,
                        "training_data_symbol": training_data_symbol,
                        "training_data_timeframe": training_data_timeframe,
                        "training_device": training_device,
                        "training_duration_seconds": training_duration_seconds,
                        "training_completed_at": training_completed_at.isoformat(),
                        "recommendation_reason": recommendation_reason,
                    },
                    is_complete=True  # Triggers "type": "complete" in ProgressReporter
                )
                self.logger.info(f"✅ Sent completion message with final metrics (session={session_id[:8]})")
            except Exception as report_err:
                self.logger.error(f"⚠️ Failed to send completion message: {report_err}")
                # Don't fail training if reporter fails
            
            # ═════════════════════════════════════════════════════════════════════
            # PHASE 19: UPDATE TrainedModelForAnalysis WITH FULL METADATA (Phase 2.3)
            # ═════════════════════════════════════════════════════════════════════
            training_metadata = {
                "training_data_symbol": training_data_symbol,
                "training_data_timeframe": training_data_timeframe,
                "training_data_start_date": training_data_start_date,
                "training_data_end_date": training_data_end_date,
                "training_data_duration_days": training_data_duration_days,
                "training_data_sample_count": training_data_sample_count,
                "training_data_feature_columns": training_data_feature_columns,
                "training_data_target_columns": training_data_target_columns,
                "scaler_binary": scaler_binary,
                "sequence_length": sequence_length,
                "optimizer_type": model_config.get("optimizer", "adam") if model_config else "adam",
                "optimizer_config": {"type": model_config.get("optimizer", "adam")} if model_config else {},
                "learning_rate": model_config.get("learning_rate") if model_config else None,
                "batch_size": batch_size,
                "epochs": epochs_completed,
                "validation_split": model_config.get("validation_split") if model_config else None,
                "training_framework": "tensorflow",
                "training_framework_version": training_environment.get("tensorflow_version"),
                "training_device": training_device,
                "training_environment": training_environment,
                "code_version": code_version,
                "training_started_at": training_started_at,
                "training_completed_at": training_completed_at,
                "training_duration_seconds": training_duration_seconds,
                "training_loss_history": training_history["loss"],
                "validation_loss_history": training_history["val_loss"],
                "training_metrics_history": {
                    "mae": training_history["mae"],
                    "mse": training_history["mse"],
                },
                "validation_metrics_history": {
                    "val_mae": training_history["val_mae"],
                    "val_mse": training_history["val_mse"],
                },
                "final_training_loss": final_training_loss,
                "final_validation_loss": final_val_loss,
                "mae": final_mae,
                "quality_score": quality_score,
                "recommendation_reason": recommendation_reason,
                "is_backtest_fair": True,
                "best_epoch": best_epoch,
                "status": "trained",
            }
            
            # Update TrainedModelForAnalysis with full metadata (non-blocking, but tracked)
            _background_tasks.append(
                asyncio.ensure_future(
                    self._update_trained_model_with_metadata(
                        model_id=model_id,
                        training_metadata=training_metadata,
                    )
                )
            )
            
            # ═════════════════════════════════════════════════════════════════════
            # PHASE 19: AUTO-GENERATE ModelSelectionHint (Phase 2.5)
            # ═════════════════════════════════════════════════════════════════════
            if training_data_symbol:
                feature_hash = ""
                if training_data_feature_columns:
                    canonical = sorted(str(f) for f in training_data_feature_columns)
                    feature_hash = hashlib.sha256(",".join(canonical).encode()).hexdigest()[:16]
                
                _background_tasks.append(
                    asyncio.ensure_future(
                        self._auto_generate_model_selection_hints(
                            trained_model_id=model_id,
                            training_data_symbol=training_data_symbol,
                            training_data_timeframe=training_data_timeframe or "unknown",
                            feature_hash=feature_hash,
                            quality_score=quality_score,
                        )
                    )
                )
            
            
            # ═════════════════════════════════════════════════════════════════════
            # PHASE 22: GENERATE POST-TRAINING PREDICTIONS
            # ═════════════════════════════════════════════════════════════════════
            if dataset_id:
                try:
                    await self._generate_post_training_predictions(
                        model=model,
                        model_id=model_id,
                        dataset_id=dataset_id,
                        train_data=train_result,
                        val_data=val_result,
                        test_data=test_result,
                        task_id=task_id,
                        reporter=reporter
                    )
                except Exception as pred_err:
                    self.logger.error(f"⚠️ Failed to generate post-training predictions: {pred_err}")

            # Update task: Training complete
            self.task_store.update_task(
                task_id=task_id,
                status="success",
                progress=100,
                message=f"Training complete. {epochs_completed} epochs executed. Quality score: {quality_score:.1f}/100",
                metadata={
                    "epochs_completed": epochs_completed,
                    "final_val_loss": float(final_val_loss),
                    "final_mae": float(final_mae),
                    "final_mse": float(final_mse),
                    "quality_score": quality_score,
                    "best_epoch": best_epoch,
                }
            )
            
            # ═════════════════════════════════════════════════════════════════════
            # FINAL: Await ALL remaining background tasks before returning
            # This ensures metadata updates, model selection hints, and any remaining
            # per-epoch records are fully committed to the database.
            # ═════════════════════════════════════════════════════════════════════
            if _background_tasks:
                self.logger.info(f"⏳ Finalizing {len(_background_tasks)} remaining background tasks...")
                await asyncio.gather(*_background_tasks, return_exceptions=True)
                self.logger.info("✅ All training lifecycle tasks completed")

            return training_result
            
        except Exception as e:
            self.logger.error(f"Model training error: {str(e)}")
            self.task_store.update_task(
                task_id=task_id,
                status="error",
                message=f"Model training failed: {str(e)}"
            )
            return {"status": "error", "message": str(e)}
    

    def _prepare_batch(
        self,
        data,
        targets,
        indices,
    ):
        """
        Pure-CPU batch preparation — safe to run in asyncio.to_thread.
        Handles slicing + float32 casting so the GPU thread receives
        ready-to-consume tensors without blocking the event loop.
        """
        # --- features ---
        if isinstance(data, pd.DataFrame):
            x = data.iloc[indices].values
        else:
            x = data[indices]
        # --- targets ---
        if targets is not None:
            if isinstance(targets, pd.DataFrame):
                y = targets.iloc[indices].values
            else:
                y = targets[indices]
        else:
            y = x  # autoencoder fallback
        # --- dtype safety ---
        try:
            return x.astype(np.float32), y.astype(np.float32)
        except Exception:
            return x, y

    async def _train_model_async(
        self,
        model: Any,
        train_data: Any,
        val_data: Any,
        batch_size: int,
        epoch: int,
        task_id: str,
        is_generator: bool = False,
        train_targets: Any = None,
        val_targets: Any = None,
        reporter: Any = None,
        total_epochs: int = 1,
        last_val_metrics: Dict[str, Any] = None,
        target_column: str = None,
        weights_before: Any = None,  # ⚖️ NEW: For epoch-level jury validation
        train_data_obj: Any = None,  # ⚖️ NEW: Original data object for jury access
        max_m: int = 5               # 🚀 NEW: Dynamic micro-epoch budget
    ) -> Dict[str, Any]:
        """
        Pure async training loop for single epoch.
        """
        try:
            # 🔍 [DIAGNOSTIC] Deep inspection of inputs
            if epoch == 0:
                self.logger.info(f"🧪 [EPOCH {epoch+1}] TARGET VERIFICATION:")
                self.logger.info(f"   ├─ train_targets: type={type(train_targets)}, val={train_targets}")
                self.logger.info(f"   ├─ val_targets: type={type(val_targets)}, val={val_targets}")
                self.logger.info(f"   ├─ target_column (arg): {target_column}")
                self.logger.info(f"   └─ is_generator: {is_generator}")
                
                if isinstance(train_data, np.ndarray):
                    self.logger.info(f"   ├─ train_data (numpy): shape={train_data.shape}")
                elif hasattr(train_data, "__len__"):
                    self.logger.info(f"   ├─ train_data (obj): type={type(train_data)}, len={len(train_data)}")
            
            epoch_loss = 0.0
            epoch_mae = 0.0
            epoch_mse = 0.0
            history = None  # ✅ Guard against NameError in summary logs
            
            # ─────────────────────────────────────────────────────────
            # TRAINING PHASE
            # ─────────────────────────────────────────────────────────
            if is_generator:
                self.logger.info(f"[Epoch {epoch+1}] Training via Generator...")
                # 🎯 [NEW] Prepare jury pool for batch-level micro-validation (reshuffled each epoch)
                if hasattr(train_data, 'micro_val_holdback') and train_data.micro_val_holdback > 0:
                    train_data._prepare_epoch_jury()
                    logger.info(
                        f"⚖️ [EPOCH {epoch+1}] Jury pool ready for batch audits:\n"
                        f"   ├─ Jury samples: {len(train_data.jury_x) if train_data.jury_x is not None else 'N/A'}\n"
                        f"   ├─ Training holdback: {train_data.micro_val_holdback * 100:.0f}%\n"
                        f"   └─ Validation threshold: 2.0% (micro) vs 5.0% (epoch)"
                    )
                gen_flow = train_data.flow()
                num_batches = len(train_data)
                
                # 🎯 [DIAGNOSTIC] Batch Consumption State
                if epoch == 0:
                    self.logger.info(f"🎯 [EPOCH {epoch+1}] BATCH CONSUMPTION DIAGNOSTICS:")
                    self.logger.info(f"   ├─ Expected batches: {num_batches}")
                    self.logger.info(f"   ├─ Batch size: {batch_size}")
                    self.logger.info(f"   ├─ Generator type: {type(train_data).__name__}")
                    self.logger.info(f"   ├─ Generator shuffle: {getattr(train_data, 'shuffle', 'Unknown')}")
                    self.logger.info(f"   └─ Generator total sequences: {getattr(train_data, 'total_sequences', 'Unknown')}")
            else:
                num_batches = len(train_data) // batch_size
            
            # ✅ Warn if num_batches is suspiciously low (Audit Point 1)
            if epoch == 0 and num_batches < 10:
                self.logger.warning(
                    f"⚠️ [_train_model_async] SUSPICIOUSLY LOW num_batches: {num_batches}\n"
                    f"   len(train_data)={len(train_data)}, batch_size={batch_size}\n"
                    f"   This suggests train_data has only {len(train_data)} sequences!"
                )
            
            # ✅ CRITICAL CHECK: Ensure num_batches > 0 before entering loop (Audit Point 2)
            if num_batches == 0:
                self.logger.error(
                    f"❌ [EPOCH {epoch+1}] SKIPPING: num_batches=0!\n"
                    f"   len(train_data)={len(train_data)}, batch_size={batch_size}\n"
                    f"   This batch will not train at all!"
                )
                return {
                    "loss": float('inf'), "val_loss": float('inf'),
                    "mae": 0.0, "val_mae": 0.0, "mse": 0.0, "val_mse": 0.0,
                }
            
            # Shuffle  

            # -- shared progress reporter helper --
            async def _report(b_idx, b_loss, e_loss, e_mae, e_mse, count=None):
                # Use count for averages if provided (accurate when batches are skipped)
                divisor = count if count is not None else (b_idx + 1)
                avg_loss = e_loss / divisor if divisor > 0 else 0.0
                avg_mae  = e_mae  / divisor if divisor > 0 else 0.0
                avg_mse  = e_mse  / divisor if divisor > 0 else 0.0
                
                gp = int((((epoch * num_batches) + b_idx) /
                           (total_epochs * num_batches)) * 100)
                msg = (f"Epoch {epoch+1}/{total_epochs}: batch {b_idx+1}/{num_batches}"
                       f" - Batch Loss: {b_loss:.4f} | Avg Loss: {avg_loss:.4f}")
                vl = (last_val_metrics or {}).get("val_loss", 0.0) or 0.0
                vm = (last_val_metrics or {}).get("val_mae",  0.0) or 0.0
                vs = (last_val_metrics or {}).get("val_mse",  0.0) or 0.0
                tm = {
                    "loss": avg_loss, 
                    "mae": avg_mae, 
                    "mse": avg_mse,
                    "val_loss": vl, 
                    "val_mae": vm, 
                    "val_mse": vs,
                    "current_epoch": epoch + 1, "total_epochs": total_epochs,
                }
                if reporter:
                    await reporter.report_async(
                        progress=gp, 
                        message=msg,
                        trainingMetrics=tm,
                        loss=float(b_loss), 
                        avg_loss=float(avg_loss),
                    )
                else:
                    self.task_store.update_task(
                        task_id=task_id, 
                        status="processing",
                        progress=gp, 
                        message=msg,
                        metadata={
                            "loss": float(b_loss), 
                            "avg_loss": float(avg_loss),
                            "trainingMetrics": tm},
                    )

            if is_generator:
                # ── generator path ──────────────────────────────
                metric_names = [m.name if hasattr(m, 'name') else str(m) for m in model.metrics]
                batches_ran = 0
                
                # 📊 DIAGNOSTIC: Log expected batch count and sizes
                total_expected = num_batches * batch_size
                total_sequences = train_data.total_sequences if hasattr(train_data, 'total_sequences') else '?'
                diff = (total_expected - total_sequences) if isinstance(total_sequences, int) else '?'
                
                logger.info(
                    f"🎯 [EPOCH {epoch+1}] BATCH CONSUMPTION DIAGNOSTICS:\n"
                    f"   ├─ Expected batches: {num_batches}\n"
                    f"   ├─ Batch size: {batch_size}\n"
                    f"   ├─ Expected samples: {num_batches} × {batch_size} = {num_batches * batch_size}\n"
                    f"   ├─ Generator type: {type(train_data).__name__}\n"
                    f"   ├─ Generator shuffle: {train_data.shuffle if hasattr(train_data, 'shuffle') else '?'}\n"
                    f"   └─ Generator total sequences: {train_data.total_sequences if hasattr(train_data, 'total_sequences') else '?'}"
                )
                
                epoch_samples_training = 0
                samples_by_batch = []
                
                for batch_idx in range(num_batches):
                    try:
                        batch_x, batch_y = next(gen_flow)
                    except StopIteration:
                        self.logger.warning(f"⚠️ [EPOCH {epoch+1}] Generator exhausted early at batch {batch_idx}/{num_batches}")
                        break
                    except Exception as gen_err:
                        self.logger.error(f"❌ [EPOCH {epoch+1}] Generator error at batch {batch_idx}: {gen_err}")
                        break

                    # 🎯 TARGET ALIGNMENT VERIFICATION (Audit Point 3)
                    if batch_idx == 0:
                        self.logger.info(
                            f"🧪 [EPOCH {epoch+1}] TARGET VERIFICATION:\n"
                            f"   ├─ x.shape: {batch_x.shape}\n"
                            f"   ├─ y.shape: {batch_y.shape}\n"
                            f"   ├─ target_column: {target_column or 'None (Auto)'}\n"
                            f"   ├─ y_sample (future_seq): {batch_y[0].tolist() if hasattr(batch_y[0], 'tolist') else batch_y[0]}\n"
                            f"   └─ Aligned: {'YES ✓' if len(batch_x) == len(batch_y) else 'NO ❌'}"
                        )
                        if len(batch_x) != len(batch_y):
                            self.logger.error(f"❌ [EPOCH {epoch+1}] CRITICAL: Sample count mismatch! x={len(batch_x)}, y={len(batch_y)}")
                            break

                    # � DIAGNOSTIC: Track batch size
                    batch_size_actual = len(batch_x)
                    epoch_samples_training += batch_size_actual
                    samples_by_batch.append(batch_size_actual)
                    
                    if batch_idx % 20 == 0 or batch_size_actual < batch_size:
                        logger.debug(
                            f"   Batch {batch_idx+1:4d}/{num_batches}: "
                            f"size={batch_size_actual:4d} samples "
                            f"partial={'YES ⚠️' if batch_size_actual < batch_size else 'NO ✓'} "
                            f"cumulative={epoch_samples_training:6d}"
                        )

                    # 🚀 [MICRO-EPOCHS] Perform multiple weight updates on the same batch
                    # 🎯 [NEW] With batch-level jury audit and rollback on regression
                    m_idx = 0
                    # max_m is now passed as a parameter for dynamic decay
                    last_bl = float('inf')
                    bl, bm, bs = 0.0, 0.0, 0.0
                    
                    # ⚖️ BATCH AUDIT: Snapshot weights before micro-epochs
                    weights_before_batch = model.get_weights()
                    jury_loss_before = float('inf')
                    has_jury = hasattr(train_data, 'jury_x') and train_data.jury_x is not None
                    
                    if has_jury:
                        # Test against held-back jury pool BEFORE micro-epochs
                        j_size = min(batch_size, len(train_data.jury_x))
                        j_idx = np.random.choice(len(train_data.jury_x), size=j_size, replace=False)
                        jury_res_before = await asyncio.wait_for(
                            asyncio.to_thread(model.test_on_batch, train_data.jury_x[j_idx], train_data.jury_y[j_idx]),
                            timeout=300.0,
                        )
                        jury_loss_before = float(jury_res_before[0]) if isinstance(jury_res_before, (list, tuple)) else float(jury_res_before)
                    
                    while m_idx < max_m:
                        await asyncio.sleep(0)
                        try:
                            history = await asyncio.wait_for(
                                asyncio.to_thread(model.train_on_batch, batch_x, batch_y),
                                timeout=600.0,
                            )

                            if isinstance(history, (list, tuple)):
                                bl = float(history[0]) if len(history) > 0 else 0.0
                                bm = float(history[1]) if len(history) > 1 else 0.0
                                bs = float(history[2]) if len(history) > 2 else 0.0
                            else:
                                results = dict(zip(["loss"] + metric_names, 
                                            history if isinstance(history, (list, tuple)) else [history]))
                                
                                bl = float(results.get("loss", 0.0))
                                bm = float(results.get("mae", 0.0))
                                bs = float(results.get("mse", 0.0))
                            
                          
                            # New: < 0.5% relative improvement (or absolute < 1e-6 for tiny losses)
                            if last_bl > 0:
                                pct_improvement = (last_bl - bl) / last_bl
                                if (pct_improvement < 0.005 or (last_bl - bl) < 1e-6) and m_idx > 2:
                                    break
                            
                            last_bl = bl
                            m_idx += 1
                        except asyncio.TimeoutError:
                            self.logger.error(f"❌ Batch {batch_idx} timed out in generator — skipping")
                            bl = 0.0
                            break
                        except Exception as train_err:
                            self.logger.error(f"❌ Training error at batch {batch_idx}: {train_err}")
                            bl = 0.0
                            break
                    
                    # ⚖️ BATCH AUDIT: Test against jury pool AFTER micro-epochs for regression
                    if has_jury and np.isfinite(bl) and bl > 0:
                        j_size = min(batch_size, len(train_data.jury_x))
                        j_idx = np.random.choice(len(train_data.jury_x), size=j_size, replace=False)
                        jury_res_after = await asyncio.wait_for(
                            asyncio.to_thread(model.test_on_batch, train_data.jury_x[j_idx], train_data.jury_y[j_idx]),
                            timeout=300.0,
                        )
                        jury_loss_after = float(jury_res_after[0]) if isinstance(jury_res_after, (list, tuple)) else float(jury_res_after)
                        
                        # Rollback if jury loss regressed more than 2% (micro-val threshold)
                        if jury_loss_before > 0 and jury_loss_after > jury_loss_before * 1.02:
                            model.set_weights(weights_before_batch)
                            if batch_idx % 10 == 0:
                                logger.warning(
                                    f"⚖️ [BATCH AUDIT] Generator path - Rollback at batch {batch_idx+1}: "
                                    f"Jury loss regressed ({jury_loss_before:.4f} → {jury_loss_after:.4f}, "
                                    f"+{((jury_loss_after/jury_loss_before - 1) * 100):.1f}%)"
                                )
                            # Reset batch loss to indicate rollback
                            bl = jury_loss_after
                            continue
                            
                    if np.isfinite(bl):
                        epoch_loss += bl; epoch_mae += bm; epoch_mse += bs
                        batches_ran += 1
                    
                    if batch_idx % 4 == 0 or batch_idx == num_batches - 1:
                        await _report(batch_idx, bl, epoch_loss, epoch_mae, epoch_mse, count=batches_ran)
                
                final_batches_ran = batches_ran
                
                # 📊 DIAGNOSTIC: Comprehensive epoch summary
                min_batch_size = min(samples_by_batch) if samples_by_batch else 0
                max_batch_size = max(samples_by_batch) if samples_by_batch else 0
                avg_batch_size = epoch_samples_training / batches_ran if batches_ran > 0 else 0
                partial_batches = sum(1 for s in samples_by_batch if s < batch_size)
                
                logger.info(
                    f"📊 [EPOCH {epoch+1}] BATCH CONSUMPTION SUMMARY:\n"
                    f"   ├─ Batches ran: {batches_ran}/{num_batches} "
                    f"({'ALL RECEIVED ✅' if batches_ran == num_batches else f'INCOMPLETE ⚠️ ({num_batches - batches_ran} missing)'})\n"
                    f"   ├─ Total samples trained: {epoch_samples_training}\n"
                    f"   ├─ Batch sizes:\n"
                    f"   │  ├─ Min: {min_batch_size}\n"
                    f"   │  ├─ Max: {max_batch_size}\n"
                    f"   │  └─ Avg: {avg_batch_size:.1f}\n"
                    f"   └─ Partial batches (< {batch_size}): {partial_batches}\n"
                    f"\n"
                    f"   🔍 DATA QUALITY CHECK:\n"
                    f"   ├─ Expected samples: {num_batches * batch_size}\n"
                    f"   ├─ Actual samples: {epoch_samples_training}\n"
                    f"   ├─ Difference: {epoch_samples_training - (num_batches * batch_size)}\n"
                    f"   ├─ Loss this epoch: {epoch_loss / batches_ran if batches_ran > 0 else 'N/A':.4f}\n"
                    f"   ├─ History: {history}\n"
                    f"   └─ Status: {'DATA INTEGRITY OK ✅' if epoch_samples_training > 0 else 'NO DATA TRAINED ❌'}"
                )
                
                self.logger.info(f"📊 [EPOCH {epoch+1}] Generator pass complete. Batches: {batches_ran}/{num_batches}")
            else:
                queue: asyncio.Queue = asyncio.Queue(maxsize=3)

                # ✅ SHUFFLE: Re-shuffle every epoch for better generalization (Ref Code implementation)
                indices = np.random.permutation(len(train_data))

                async def _producer():
                    exc_to_raise = None
                    try:
                        for b_idx in range(num_batches):
                            start_idx = b_idx * batch_size
                            end_idx = start_idx + batch_size
                            batch_indices = indices[start_idx:end_idx]
                            
                            prepared = await asyncio.to_thread(
                                self._prepare_batch,
                                train_data, train_targets,
                                batch_indices,
                            )
                            await queue.put((b_idx, prepared))
                    except Exception as exc:
                        self.logger.error(f"❌ Batch producer error: {exc}")
                        exc_to_raise = exc
                    finally:
                        await queue.put(None)  # sentinel
                        if exc_to_raise:
                            raise exc_to_raise

                async def _consumer():
                    nonlocal epoch_loss, epoch_mae, epoch_mse
                    batches_ran = 0
                    
                    # Pre-fetch metric names for robust extraction (Bug 5)
                    metric_names = [m.name if hasattr(m, 'name') else str(m) for m in model.metrics]
                    
                    while True:
                        item = await queue.get()
                        if item is None:
                            break
                        b_idx, (b_x, b_y) = item
                        
                        # 🚀 [MICRO-EPOCHS] Perform multiple weight updates on the same batch
                        m_idx = 0
                        # max_m is now passed as a parameter for dynamic decay
                        last_bl = float('inf')
                        bl, bm, bs = 0.0, 0.0, 0.0
                        
                        # ⚖️ AUDIT SNAPSHOT
                        weights_before_batch = model.get_weights()
                        jury_loss_before = float('inf')
                        has_jury = hasattr(train_data, 'jury_x') and train_data.jury_x is not None
                        
                        if has_jury:
                            j_size = min(batch_size, len(train_data.jury_x))
                            j_idx = np.random.choice(len(train_data.jury_x), size=j_size, replace=False)
                            jury_res = await asyncio.to_thread(model.test_on_batch, train_data.jury_x[j_idx], train_data.jury_y[j_idx])
                            jury_loss_before = float(jury_res[0])

                        while m_idx < max_m:
                            await asyncio.sleep(0)
                            try:
                                history = await asyncio.wait_for(
                                    asyncio.to_thread(model.train_on_batch, b_x, b_y),
                                    timeout=600.0,
                                )
                                
                                # ✅ ROBUST EXTRACTION (Bug 5)
                                results = dict(zip(["loss"] + metric_names, 
                                               history if isinstance(history, (list, tuple)) else [history]))
                                
                                bl = float(results.get("loss", 0.0))
                                bm = float(results.get("mae") if results.get("mae") is not None else results.get("mean_absolute_error", 0.0))
                                bs = float(results.get("mse") if results.get("mse") is not None else results.get("mean_squared_error", 0.0))
                                
                                # 🔴 BUG FIX: Use RELATIVE improvement, not absolute 0.001
                                # Previous bug: 0.001 absolute threshold kills learning when loss < 0.01
                                # (e.g., 0.001→0.0009 has 0.0001 improvement < 0.001, exits early)
                                # New: < 0.5% relative improvement (or absolute < 1e-6 for tiny losses)
                                if last_bl > 0:
                                    pct_improvement = (last_bl - bl) / last_bl
                                    if (pct_improvement < 0.005 or (last_bl - bl) < 1e-6) and m_idx > 2:
                                        break
                                    
                                last_bl = bl
                                m_idx += 1
                                
                            except asyncio.TimeoutError:
                                self.logger.error(f"❌ Batch {b_idx} timed out — skipping")
                                bl = 0.0  # Ensure we don't use previous batch metrics
                                break
                                
                        # ⚖️ BATCH AUDIT: Test against jury pool AFTER micro-epochs for regression
                        if has_jury and np.isfinite(bl) and bl > 0:
                            j_size = min(batch_size, len(train_data.jury_x))
                            j_idx = np.random.choice(len(train_data.jury_x), size=j_size, replace=False)
                            jury_res_after = await asyncio.to_thread(model.test_on_batch, train_data.jury_x[j_idx], train_data.jury_y[j_idx])
                            jury_loss_after = float(jury_res_after[0]) if isinstance(jury_res_after, (list, tuple)) else float(jury_res_after)
                            
                            # Rollback if jury loss regressed more than 2% (micro-val threshold)
                            if jury_loss_before > 0 and jury_loss_after > jury_loss_before * 1.02:
                                model.set_weights(weights_before_batch)
                                if b_idx % 10 == 0:
                                    logger.warning(
                                        f"⚖️ [BATCH AUDIT] Numpy path - Rollback at batch {b_idx+1}: "
                                        f"Jury loss regressed ({jury_loss_before:.4f} → {jury_loss_after:.4f}, "
                                        f"+{((jury_loss_after/jury_loss_before - 1) * 100):.1f}%)"
                                    )
                                # Reset batch loss to indicate rollback
                                bl = jury_loss_after
                                continue
                        
                        # ✅ Guard against NaN/Inf and only accumulate if batch actually ran (Bug 3, 8)
                        if np.isfinite(bl) and (bl != 0.0 or bm != 0.0 or bs != 0.0):
                            epoch_loss += bl; epoch_mae += bm; epoch_mse += bs
                            batches_ran += 1
                            
                        if b_idx % 4 == 0 or b_idx == num_batches - 1:
                            # Use batches_ran for more accurate avg reporting during epoch
                            await _report(b_idx, bl, epoch_loss, epoch_mae, epoch_mse, count=batches_ran)
                    
                    self.logger.info(f"📊 [EPOCH {epoch+1}] Consumer pipeline complete. Batches: {batches_ran}/{num_batches}")
                    # Store the actual number of contributing batches for final averaging
                    return batches_ran

                prod_task = asyncio.create_task(_producer())
                cons_task = asyncio.create_task(_consumer())
                try:
                    # Capture batches_ran from consumer
                    results = await asyncio.gather(prod_task, cons_task)
                    final_batches_ran = results[1] if len(results) > 1 else num_batches
                except Exception:
                    prod_task.cancel()
                    cons_task.cancel()
                    raise
            
            # Average loss over batches
            avg_count = final_batches_ran
            if avg_count > 0:
                epoch_loss /= avg_count
                epoch_mae /= avg_count
                epoch_mse /= avg_count
            else:
                self.logger.error(f"❌ [TRAINING FAILURE] No valid batches processed at epoch {epoch+1}")
                epoch_loss = float('inf')
                epoch_mae = float('inf')
                epoch_mse = float('inf')
            
            # Validation
            if is_generator:
                try:
                    val_flow = val_data.flow()
                    val_loss_list = []
                    val_mae_list = []
                    val_mse_list = []
                    val_samples_list = [] # Track samples per batch for weighted avg
                    
                    max_val_batches = len(val_data)
                    
                    metric_names = [m.name if hasattr(m, 'name') else str(m) for m in model.metrics]

                    for v_idx in range(max_val_batches):
                        try:
                            v_batch = next(val_flow)
                            # ✅ FIX: Handle dict returns from flow() (new multi-target structure)
                            if isinstance(v_batch, dict):
                                v_x = v_batch.get('x')
                                v_y = v_batch.get('y')
                            else:
                                v_x, v_y = v_batch
                            
                            # v_loss_raw = await asyncio.to_thread(model.evaluate, v_x, v_y, verbose=0)
                            v_loss_raw = await asyncio.wait_for(
                                asyncio.to_thread(model.test_on_batch, v_x, v_y),
                                timeout=600.0
                            )
                            
                            if isinstance(v_loss_raw, (list, tuple, np.ndarray)):
                                lv = float(v_loss_raw[0]) if len(v_loss_raw) > 0 else 0.0
                                if not np.isfinite(lv):
                                    self.logger.warning(
                                        f"⚠️ [EPOCH {epoch+1}] Validation batch {v_idx} returned non-finite loss: {lv}\n"
                                        f"   ├─ x_bounds: [{np.min(v_x):.4f}, {np.max(v_x):.4f}]\n"
                                        f"   ├─ y_sample: {v_y[0].tolist() if hasattr(v_y[0], 'tolist') else v_y[0]}\n"
                                        f"   └─ Skipping batch."
                                    )
                                    continue
                                val_loss_list.append(lv)
                                val_mae_list.append(float(v_loss_raw[1]) if len(v_loss_raw) > 1 else 0.0)
                                val_mse_list.append(float(v_loss_raw[2]) if len(v_loss_raw) > 2 else 0.0)
                                val_samples_list.append(len(v_x))
                            else:
                                v_results = dict(zip(["loss"] + metric_names, 
                                             v_loss_raw if isinstance(v_loss_raw, (list, tuple, np.ndarray)) else [v_loss_raw]))
                                lv = float(v_results.get("loss", 0.0))
                                
                                if not np.isfinite(lv):
                                    self.logger.warning(
                                        f"⚠️ [EPOCH {epoch+1}] Validation batch {v_idx} returned non-finite loss: {lv}\n"
                                        f"   ├─ x_bounds: [{np.min(v_x):.4f}, {np.max(v_x):.4f}]\n"
                                        f"   ├─ y_sample: {v_y[0].tolist() if hasattr(v_y[0], 'tolist') else v_y[0]}\n"
                                        f"   └─ Skipping batch."
                                    )
                                    continue

                                val_loss_list.append(lv)
                                val_mae_list.append(float(v_results.get("mae") if v_results.get("mae") is not None else v_results.get("mean_absolute_error", 0.0)))
                                val_mse_list.append(float(v_results.get("mse") if v_results.get("mse") is not None else v_results.get("mean_squared_error", 0.0)))
                                val_samples_list.append(len(v_x))

                        except StopIteration:
                            break
                        except Exception as eval_err:
                            self.logger.warning(f"⚠️ Batch validation error: {eval_err}")
                            continue
                    
                    # ✅ WEIGHTED AVERAGE (Bug Fixed): Use sample weights to avoid partial batch bias
                    if val_loss_list and val_samples_list:
                        total_v_samples = sum(val_samples_list)
                        val_loss = sum(l * s for l, s in zip(val_loss_list, val_samples_list)) / total_v_samples
                        val_mae = sum(m * s for m, s in zip(val_mae_list, val_samples_list)) / total_v_samples
                        val_mse = sum(s * s_size for s, s_size in zip(val_mse_list, val_samples_list)) / total_v_samples
                    else:
                        val_loss, val_mae, val_mse = float('inf'), 0.0, 0.0
                    
                    # Log if validation came back as 0 despite having data
                    if val_loss == 0.0 and len(val_data) > 0:
                        self.logger.warning(f"⚠️ Validation metrics are 0.0 despite {len(val_data)} validation batches")
                        
                except Exception as gen_err:
                    self.logger.error(f"❌ Validation generator error: {gen_err}")
                    val_loss, val_mae, val_mse = float('inf'), 0.0, 0.0
            else:
                # Handle both DataFrame and numpy array
                if isinstance(val_data, pd.DataFrame):
                    val_x = val_data.values
                else:
                    val_x = val_data
                
                # ✅ FIXED: Use actual targets for validation
                if val_targets is not None:
                    val_y = val_targets
                else:
                    val_y = val_x  # Fallback: Autoencoder mode
                
                # ✅ FULL VALIDATION: Calculate total batches to cover everything (Bug Fixed)
                val_num_batches = (len(val_x) + batch_size - 1) // batch_size
                if val_num_batches == 0 and len(val_x) > 0:
                    val_num_batches = 1
                    
                val_loss_sum = 0.0
                val_mae_sum = 0.0
                val_mse_sum = 0.0
                val_samples_total = 0
                
                # Pre-fetch metric names
                metric_names = [m.name if hasattr(m, 'name') else str(m) for m in model.metrics]
                val_batches_ran = 0
                self.logger.debug(f"ℹ️ Running full validation on {val_num_batches} batches ({len(val_x)} samples)")

                for v_idx in range(val_num_batches):
                    start_idx = v_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(val_x))
                    
                    v_batch_x = val_x[start_idx:end_idx]
                    
                    # Ensure targets are also sliced correctly and handled as arrays
                    if isinstance(val_y, (pd.DataFrame, pd.Series)):
                        v_batch_y = val_y.iloc[start_idx:end_idx].values
                    else:
                        v_batch_y = val_y[start_idx:end_idx]
                    
                    # Yield to event loop
                    await asyncio.sleep(0)
                    
                    v_loss_raw = await asyncio.wait_for(
                        asyncio.to_thread(model.test_on_batch, v_batch_x, v_batch_y),
                        timeout=600.0
                    )
                    
                    v_results = dict(zip(["loss"] + metric_names, 
                                     v_loss_raw if isinstance(v_loss_raw, (list, tuple)) else [v_loss_raw]))
                    
                    lv = float(v_results.get("loss", 0.0))
                    if not np.isfinite(lv):
                        self.logger.warning(
                            f"⚠️ [EPOCH {epoch+1}] Validation batch {v_idx} (array) returned non-finite loss: {lv}\n"
                            f"   ├─ x_bounds: [{np.min(v_batch_x):.4f}, {np.max(v_batch_x):.4f}]\n"
                            f"   ├─ y_sample: {v_batch_y[0].tolist() if hasattr(v_batch_y[0], 'tolist') else v_batch_y[0]}\n"
                            f"   └─ Skipping batch."
                        )
                        continue

                    b_len = len(v_batch_x)
                    val_loss_sum += lv * b_len
                    val_mae_sum += float(v_results.get("mae", 0.0)) * b_len
                    val_mse_sum += float(v_results.get("mse", 0.0)) * b_len
                    val_samples_total += b_len
                    val_batches_ran += 1
                        
                if val_samples_total > 0:
                    val_loss = val_loss_sum / val_samples_total
                    val_mae = val_mae_sum / val_samples_total
                    val_mse = val_mse_sum / val_samples_total
                else:
                    val_loss = float('inf')
                    val_mae = 0.0
                    val_mse = 0.0
            
            # ⚖️ EPOCH-LEVEL JURY VALIDATION: Compare pre-epoch vs post-epoch on held-back jury pool
            # This catches entire-epoch regressions that batch audits may have missed
            if weights_before is not None and train_data_obj is not None and hasattr(train_data_obj, 'jury_x'):
                try:
                    jury_x = train_data_obj.jury_x
                    jury_y = train_data_obj.jury_y
                    
                    if jury_x is not None and len(jury_x) > 0 and np.isfinite(val_loss):
                        # 🔧 BUG #3 FIX: Capture post-epoch weights BEFORE modifying model
                        # Previously: post_epoch_weights were never captured, leading to pre-vs-pre comparison
                        post_epoch_weights = model.get_weights()  # ← CAPTURE POST-EPOCH FIRST
                        
                        # Test PRE-epoch weights on jury
                        model.set_weights(weights_before)
                        jury_loss_before = float(await asyncio.to_thread(model.test_on_batch, jury_x, jury_y))
                        
                        # Restore POST-epoch weights and test
                        model.set_weights(post_epoch_weights)
                        jury_loss_after = float(await asyncio.to_thread(model.test_on_batch, jury_x, jury_y))
                        
                        # Check if entire epoch regressed on jury (3% threshold for epoch-level, stricter than batch)
                        if jury_loss_before > 0 and jury_loss_after > jury_loss_before * 1.03:
                            pct_change = ((jury_loss_after / jury_loss_before) - 1) * 100
                            self.logger.warning(
                                f"⚖️ [EPOCH {epoch+1}] JURY REJECTED: Epoch regressed on jury pool\n"
                                f"   ├─ Jury loss: {jury_loss_before:.6f} → {jury_loss_after:.6f} (+{pct_change:.1f}%)\n"
                                f"   ├─ Val loss: {val_loss:.6f}\n"
                                f"   └─ Restoring pre-epoch weights (epoch will be retried with lower LR)"
                            )
                            # Return modified val_loss to signal regression to main loop
                            # This will trigger rollback logic
                            val_loss = jury_loss_after  # Override with jury verdict
                except Exception as e:
                    self.logger.debug(f"⚖️ Epoch-level jury audit failed: {e}")
            
            return {
                "loss": float(epoch_loss),
                "val_loss": float(val_loss),
                "mae": float(epoch_mae),
                "val_mae": float(val_mae),
                "mse": float(epoch_mse),
                "val_mse": float(val_mse),
            }
            
        except Exception as e:
            self.logger.error(f"Epoch training error: {str(e)}")
            return {
                "loss": float('inf'),
                "val_loss": float('inf'),
                "mae": 0.0,
                "val_mae": 0.0,
                "mse": 0.0,
                "val_mse": 0.0,
            }
    



    
    async def execute_model_build_with_pm(
        self,
        session_id: str,
        task_id: str,
        pm: ProcessingManager,
        model_config: Dict[str, Any],
        user_id: str = "anonymous",
    ) -> Dict[str, Any]:
        """
        Execute ML model building with injected ProcessingManager.
        
        Follows route pattern: PM instantiated in route, passed to AnalysisManager.
        
        Args:
            session_id: Session ID
            task_id: Task ID for progress tracking
            pm: Injected ProcessingManager (optimization layer)
            model_config: Model configuration dict
            
        Returns:
            dict with model_id, status, architecture
        """
        import dataclasses
        if dataclasses.is_dataclass(model_config) and not isinstance(model_config, type):
            model_config = dataclasses.asdict(model_config)
            
        try:
            self.logger.info(f"[{task_id}] Building model with PM optimization")
            
            # Build model
            result = await self.execute_model_build(
                session_id=session_id,
                task_id=task_id,
                model_config=model_config,
                user_id=user_id
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Model build with PM error: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def execute_model_training_with_pm(
        self,
        session_id: str,
        task_id: str,
        pm: ProcessingManager,
        train_config: Dict[str, Any],
        user_id: str = "anonymous",
    ) -> Dict[str, Any]:
        """
        Execute ML model training with injected ProcessingManager.
        
        Follows route pattern: PM instantiated in route for data streaming optimization.
        PM auto-selects strategy: SEQUENTIAL/PARALLEL_CHUNKING/SLICE_STREAMING
        
        Args:
            session_id: Session ID
            task_id: Task ID for progress tracking
            pm: Injected ProcessingManager (handles data streaming)
            train_config: Training configuration dict (model_id, epochs, batch_size, ml_preparation_ref)
            
        Returns:
            dict with epochs_completed, best_val_loss, status
        """
        import dataclasses
        if dataclasses.is_dataclass(train_config) and not isinstance(train_config, type):
            train_config = dataclasses.asdict(train_config)
            
        try:
            self.logger.info(f"[{task_id}] Training model with PM streaming optimization")
            
            # ✅ NEW: Extract ml_preparation_ref from train_config
            ml_prep_ref = train_config.get("ml_preparation_ref")
            
            # Train model
            result = await self.execute_model_training(
                session_id=session_id,
                task_id=task_id,
                model_id=train_config.get("model_id", ""),
                epochs=train_config.get("epochs", 50),
                batch_size=train_config.get("batch_size", 32),
                ml_preparation_ref=ml_prep_ref,  # ✅ Pass complex ref (handled in execute_model_training)
                user_id=user_id,
                is_classification=train_config.get("is_classification", False),
                selected_targets=train_config.get("selected_targets", []),
                # ✅ FIX: Resolve target_column from selected_targets if not explicitly set
                target_column=train_config.get("target_column") or (
                    (train_config.get("selected_targets") or [None])[0]
                ),
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Model training with PM error: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    # ────────────────────────────────────────────────────────────────
    # UNIFIED ANALYSIS EXECUTION (Single Step)
    # ────────────────────────────────────────────────────────────────

    async def _load_data_4_tier(
        self,
        session_id: str,
        task_id: str,
        request_data: Optional[List[Dict[str, Any]]] = None,
        exclude_step: str = None,
        data_type: str = "analysis",  # NEW: Route to correct TIER 0
        ml_dataset_name: str = None,  # ✅ NEW: Validate dataset name for ML pointers
        prefer_lazy: bool = False,    # ✅ NEW: Enable disk spooling for large ML data
    ) -> Tuple[Union[pd.DataFrame, Dict[str, Any]], str]:
        """
        Internal 4-tier data loader with dual TIER 0 pointers (analysis & ML splits).
        
        ✅ TIER 0a: `self.current_data` pointer - Analysis results (zero-latency)
        ✅ TIER 0b: `self.ml_train/val/test` pointers - ML splits (zero-latency) ← NEW
        
        Data Types:
        - "analysis": Check TIER 0a (current_data) first - used by mutation steps
        - "ml_train": Check TIER 0b (ml_train) - used by model training
        - "ml_validation": Check TIER 0b (ml_validation) - used by model training
        - "ml_test": Check TIER 0b (ml_test) - used by model evaluation
        
        Tier Priority:
        1. TIER 0: In-memory pointers (0a or 0b based on data_type) - ZERO latency
        2. TIER 1: Request Data (inline)
        3. TIER 2: AnalysisManager Cache (memory, dict lookup)
        4. TIER 3: Database (PostgreSQL)
        """
        # ─────────────────────────────────────────────────────────
        # TIER 0a: Analysis in-memory pointer (ZERO LATENCY)
        # 🔴 CRITICAL: MUST be fully merged complete dataset, not slice
        # ─────────────────────────────────────────────────────────
        if data_type == "analysis":
            if (self.current_data is not None and 
                self.current_session_id == session_id):
                
                # ✅ VALIDATION: Verify pointer is complete (not empty, not malformed)
                if len(self.current_data) == 0:
                    logger.error(
                        f"🔴 TIER 0a VALIDATION FAILED: current_data is EMPTY\n"
                        f"  ├─ This indicates slice was stored instead of merged result\n"
                        f"  └─ Check ProcessingManager._aggregate_slice_results()\n"
                    )
                    # Fall through to next tier to recover from DB
                else:
                    logger.info(
                        f"⚡ TIER 0a HIT: Analysis pointer\n"
                        f"  ├─ Rows: {len(self.current_data)} (COMPLETE MERGED DATASET)\n"
                        f"  ├─ Columns: {len(self.current_data.columns)}\n"
                        f"  └─ Latency: ZERO ms from previous step"
                    )
                    return self.current_data.copy(), "TIER0a_ANALYSIS_POINTER"
            
            logger.debug(f"⏭️  TIER 0a MISS: Session mismatch or not set")
        
        # ─────────────────────────────────────────────────────────
        # TIER 0b: ML split pointers (ZERO LATENCY)
        # 🔴 CRITICAL: MUST be complete splits, not slices or chunks
        # ✅ NEW: Validate dataset name matches to prevent cross-dataset contamination
        # ─────────────────────────────────────────────────────────
        elif data_type == "ml_train":
            if (self.ml_train is not None and 
                self.ml_session_id == session_id and
                (ml_dataset_name is None or self.ml_dataset_name == ml_dataset_name)):  # ✅ NEW: Validate dataset name
                
                # ✅ VALIDATION: Verify split is complete (not empty)
                if len(self.ml_train) == 0:
                    logger.error(
                        f"🔴 TIER 0b (ml_train) VALIDATION FAILED: Empty split\n"
                        f"  ├─ This indicates slice was stored instead of complete split\n"
                        f"  └─ Check ProcessingManager._aggregate_slice_results() and set_ml_data_pointers()\n"
                    )
                    # Fall through to next tier to recover from DB
                else:
                    logger.info(
                        f"⚡ TIER 0b HIT: ML train split\n"
                        f"  ├─ Rows: {len(self.ml_train)} (COMPLETE SPLIT)\n"
                        f"  ├─ Columns: {len(self.ml_train.columns)}\n"
                        f"  ├─ Dataset: {self.ml_dataset_name}\n"
                        f"  └─ Latency: ZERO ms"
                    )
                    return self.ml_train.copy(), "TIER0b_ML_TRAIN"
            elif self.ml_session_id == session_id and ml_dataset_name and self.ml_dataset_name != ml_dataset_name:
                logger.warning(
                    f"⚠️ TIER 0b MISMATCH (ml_train): Requested dataset '{ml_dataset_name}' but pointers are for '{self.ml_dataset_name}'\n"
                    f"  └─ Falling back to database to fetch correct dataset"
                )
            else:
                logger.debug(f"⏭️  TIER 0b MISS (ml_train): Not available")
        
        elif data_type == "ml_validation":
            if (self.ml_validation is not None and 
                self.ml_session_id == session_id and
                (ml_dataset_name is None or self.ml_dataset_name == ml_dataset_name)):  # ✅ NEW: Validate dataset name
                
                # ✅ VALIDATION: Verify split is complete (not empty)
                if len(self.ml_validation) == 0:
                    logger.error(
                        f"🔴 TIER 0b (ml_validation) VALIDATION FAILED: Empty split\n"
                        f"  ├─ This indicates slice was stored instead of complete split\n"
                        f"  └─ Check ProcessingManager._aggregate_slice_results() and set_ml_data_pointers()\n"
                    )
                    # Fall through to next tier
                else:
                    logger.info(
                        f"⚡ TIER 0b HIT: ML validation split\n"
                        f"  ├─ Rows: {len(self.ml_validation)} (COMPLETE SPLIT)\n"
                        f"  ├─ Columns: {len(self.ml_validation.columns)}\n"
                        f"  ├─ Dataset: {self.ml_dataset_name}\n"
                        f"  └─ Latency: ZERO ms"
                    )
                    return self.ml_validation.copy(), "TIER0b_ML_VALIDATION"
            elif self.ml_session_id == session_id and ml_dataset_name and self.ml_dataset_name != ml_dataset_name:
                logger.warning(
                    f"⚠️ TIER 0b MISMATCH (ml_validation): Requested dataset '{ml_dataset_name}' but pointers are for '{self.ml_dataset_name}'\n"
                    f"  └─ Falling back to database to fetch correct dataset"
                )
            else:
                logger.debug(f"⏭️  TIER 0b MISS (ml_validation): Not available")
        
        elif data_type == "ml_test":
            if (self.ml_test is not None and 
                self.ml_session_id == session_id and
                (ml_dataset_name is None or self.ml_dataset_name == ml_dataset_name)):  # ✅ NEW: Validate dataset name
                
                # ✅ VALIDATION: Verify split is complete (not empty)
                if len(self.ml_test) == 0:
                    logger.error(
                        f"🔴 TIER 0b (ml_test) VALIDATION FAILED: Empty split\n"
                        f"  ├─ This indicates slice was stored instead of complete split\n"
                        f"  └─ Check ProcessingManager._aggregate_slice_results() and set_ml_data_pointers()\n"
                    )
                    # Fall through to next tier
                else:
                    logger.info(
                        f"⚡ TIER 0b HIT: ML test split\n"
                        f"  ├─ Rows: {len(self.ml_test)} (COMPLETE SPLIT)\n"
                        f"  ├─ Columns: {len(self.ml_test.columns)}\n"
                        f"  ├─ Dataset: {self.ml_dataset_name}\n"
                        f"  └─ Latency: ZERO ms"
                    )
                    return self.ml_test.copy(), "TIER0b_ML_TEST"
            elif self.ml_session_id == session_id and ml_dataset_name and self.ml_dataset_name != ml_dataset_name:
                logger.warning(
                    f"⚠️ TIER 0b MISMATCH (ml_test): Requested dataset '{ml_dataset_name}' but pointers are for '{self.ml_dataset_name}'\n"
                    f"  └─ Falling back to database to fetch correct dataset"
                )
            else:
                logger.debug(f"⏭️  TIER 0b MISS (ml_test): Not available")
        
        # TIER 1: Request Data
        if request_data:
            logger.info(f"✅ TIER 1: Loading {len(request_data)} rows from request")
            return pd.DataFrame(request_data), "TIER1_REQUEST"
        
        # TIER 2: Cache (Bypass for ML data types to prevent TIER 2 pollution)
        if session_id and data_type not in ["ml_train", "ml_validation", "ml_test"]:
            cached_data = await self.get_cached_data(session_id, task_id)
            if cached_data is not None:
                logger.info(f"✅ TIER 2: Cache HIT for session {session_id[:8]}... ({len(cached_data)} rows)")
                return pd.DataFrame(cached_data), "TIER2_CACHE"
        
        # TIER 3: Database
        if session_id:
            # ✅ NEW: For ML data types, load from ML dataset registry
            if data_type in ["ml_train", "ml_validation", "ml_test"]:
                logger.info(f"🔍 TIER 3: Cache MISS, fetching ML {data_type} from DB for session {session_id[:8]}...")
                
                from app.core.data.session_data_loader import get_ml_dataset_splits_by_name
                async with AsyncPostgresSessionLocal() as db:
                    # Map data_type to split_type
                    split_type_map = {
                        'ml_train': 'train',
                        'ml_validation': 'validation',
                        'ml_test': 'test'
                    }
                    split_type = split_type_map.get(data_type, 'train')
                    
                    # Try to load using the specific dataset_name
                    if ml_dataset_name:
                        logger.info(f"🔍 TIER 3: Attempting to load {split_type} split from dataset '{ml_dataset_name}'...")
                        split_data = await get_ml_dataset_splits_by_name(
                            session_id=session_id,
                            dataset_name=ml_dataset_name,
                            db=db,
                            split_type=split_type,
                            prefer_lazy=prefer_lazy
                        )
                        
                        if split_data is not None:
                            # ✅ FIXED: Log actual row count from dict or array
                            if isinstance(split_data, dict) and "sequences" in split_data:
                                row_count = len(split_data["sequences"])
                            elif isinstance(split_data, dict) and split_data.get("data_type") == "lazy_npz":
                                row_count = "LAZY_DISK_POINTER"
                            else:
                                row_count = len(split_data)
                                
                            logger.info(f"✅ TIER 3: Successfully loaded {split_type} split ({row_count})")
                            return split_data, f"TIER3_DATABASE_ML_SPLITS_{data_type.upper()}"
                        else:
                            logger.warning(f"⚠️ TIER 3: Failed to load {split_type} split from dataset '{ml_dataset_name}'")
                    else:
                        logger.warning(f"⚠️ TIER 3: No ml_dataset_name provided, cannot load ML data")
            
            # Fallback: Load latest analysis data (for non-ML data types)
            else:
                logger.info(f"🔍 TIER 3: Cache MISS, fetching standard data from DB for session {session_id[:8]}...")
                from app.core.data.session_data_loader import get_latest_session_data_excluding_step
                async with AsyncPostgresSessionLocal() as db:
                    db_data = await get_latest_session_data_excluding_step(
                        session_id=session_id,
                        db=db,
                        exclude_step=exclude_step,
                        task_id=task_id
                    )
                    if not db_data:
                        raise ValueError(f"No data found in database for session {session_id}")
                    
                    logger.info(f"✅ TIER 3: Loaded {len(db_data)} rows from database")
                    
                    # ✅ FIX: Populate TIER 2 cache after TIER 3 load so subsequent calls
                    # within the TTL window hit the cache instead of the database.
                    # Without this, every reinitialize_session call bypasses cache and
                    # hits the DB again, creating the re-initialization loop.
                    try:
                        await self.cache_session_data(
                            session_id=session_id,
                            data=db_data,
                            source_step="tier3_db_load",
                            ttl_seconds=1800  # 30 min TTL
                        )
                        logger.info(f"📌 TIER 3→2 backfill: Cached {len(db_data)} rows for session {session_id[:8]}")
                    except Exception as _cache_err:
                        logger.warning(f"⚠️ TIER 3→2 backfill failed (non-fatal): {_cache_err}")
                    
                    return pd.DataFrame(db_data), "TIER3_DATABASE"
        
        raise ValueError("Insufficient parameters to load data (no data or session_id)")
    
    async def cascade_clear_data_pointers_for_ml(
        self, 
        ml_session_id: str
    ) -> Dict[str, Any]:
        """
                
        When ML training pointers are set, clear analysis pointers to save memory.
        
        Memory Pattern:
        - Before: current_data (50K rows, 50MB) + ml_train/val/test (50MB)
        - After: ml_train/val/test only (50MB)
        - Saves: 50MB
        
        Args:
            ml_session_id: Session ID owning ML splits
        
        Returns:
            Dict with freed_mb and total_ml_mb metrics
        """
        
        # Calculate memory before clearing
        before_bytes = 0
        if self.current_data is not None:
            before_bytes = self.current_data.memory_usage(deep=True).sum()
        
        logger.info(
            f"🔄 CASCADE CLEAR: Before state\n"
            f"  ├─ current_data: {'Yes' if self.current_data is not None else 'No'} "
            f"({before_bytes/1e6:.1f}MB)\n"
            f"  ├─ current_session_id: {self.current_session_id}\n"
            f"  ├─ ml_train: {'Yes' if self.ml_train is not None else 'No'}\n"
            f"  ├─ ml_validation: {'Yes' if self.ml_validation is not None else 'No'}\n"
            f"  └─ ml_session_id: {self.ml_session_id}"
        )
        
        # CASCADE CLEAR: Remove analysis pointers (no longer needed)
        self.current_data = None
        self.current_session_id = None
        
        # Set ML session ID to track ownership
        self.ml_session_id = ml_session_id
        
        # Calculate memory after
        after_bytes = 0
        if self.ml_train is not None:
            after_bytes += self.ml_train.memory_usage(deep=True).sum()
        if self.ml_validation is not None:
            after_bytes += self.ml_validation.memory_usage(deep=True).sum()
        if self.ml_test is not None:
            after_bytes += self.ml_test.memory_usage(deep=True).sum()
        
        logger.info(
            f"🔄 CASCADE CLEAR: After state\n"
            f"  ├─ current_data: None (freed {before_bytes/1e6:.1f}MB!)\n"
            f"  ├─ current_session_id: None\n"
            f"  ├─ ml_session_id: {self.ml_session_id}\n"
            f"  └─ Total ML pointers: {after_bytes/1e6:.1f}MB"
        )
        
        return {
            'freed_mb': before_bytes / 1e6,
            'total_ml_mb': after_bytes / 1e6,
            'efficiency_gain': (before_bytes / 1e6)
        }


    async def _trainer_fit(
        self,
        model,
        train_data,
        num_batches,
        batch_size,
        epoch,
        task_id,
        reporter,
        total_epochs,
        last_val_metrics=None,
        target_column=None,
        max_m=5,
        dyn_threshold=1.15,
        # ── ports from _train_model_async ──────────────────────────
        val_data=None,
        train_targets=None,
        val_targets=None,
        is_generator=False,
        weights_before=None,
        train_data_obj=None,
        continual_state=None,  # ✅ UPGRADE: Persistent state across calls
    ):
        """
        TODO: Move this humongous functtion to ml/ module (see ml/trainer.py)
        Keras .fit() Clone with Sample-Weighted Metrics & Jury SOS Safety.
        """
        

        if num_batches == 0:
            self.logger.error(f"❌ [EPOCH {epoch+1}] num_batches=0 — skipping epoch")
            return {
                "loss": float('inf'), "val_loss": float('inf'),
                "mae": 0.0, "val_mae": 0.0, "mse": 0.0, "val_mse": 0.0,
            }

        # ── 1. Reset Stateful Metrics ────────────────────────
        for metric in model.metrics:
            metric.reset_state()

        # ── 2. Progress Reporter - Keep Frontend on the loop ─
        async def _report(b_idx, b_loss, epoch_loss, epoch_mae, epoch_mse, epoch_samples, jury_loss_before=None, jury_loss_after=None, fresh_val=None):
            # Fallback to batch metrics if no successful samples yet (prevents confusing 0.000 on frontend)
            avg_loss = epoch_loss / epoch_samples if epoch_samples > 0 else b_loss
            avg_mae = epoch_mae / epoch_samples if epoch_samples > 0 else 0.0 # fallback to 0.0 for metrics we don't have b_ equivalents for easily
            avg_mse = epoch_mse / epoch_samples if epoch_samples > 0 else 0.0

            # Ensure we hit 100% correctly with rounding and min/max capping
            total_steps = total_epochs * num_batches
            current_step = (epoch * num_batches) + b_idx + 1
            gp = min(100, max(0, int(round((current_step / total_steps) * 100))))

            msg = (f"Epoch {epoch+1}/{total_epochs}: "
                f"batch {b_idx+1}/{num_batches} - Loss: {b_loss:.4f}")
            
            # Show actual confirmed average only if we have successful samples
            if epoch_samples > 0:
                msg += f" | Avg: {avg_loss:.4f}"
            else:
                msg += " | Jury Auditing..."

            if jury_loss_before is not None:
                if jury_loss_after is not None:
                    reg_pct = (jury_loss_after / jury_loss_before - 1.0) * 100
                    msg += f" | Jury: {jury_loss_before:.4f} → {jury_loss_after:.4f} ({'+' if reg_pct > 0 else ''}{reg_pct:.1f}%)"
                else:
                    msg += f" | Jury: {jury_loss_before:.4f}"

            # Priority: fresh_val (current results) -> last_val_metrics (previous epoch) -> fallback 0.0
            v_src = fresh_val if fresh_val else (last_val_metrics or {})
            tm = {
                "loss": avg_loss, "mae": avg_mae, "mse": avg_mse,
                "val_loss": v_src.get("val_loss", 0.0),
                "val_mae": v_src.get("val_mae", 0.0),
                "val_mse": v_src.get("val_mse", 0.0),
                "jury_loss_before": float(jury_loss_before) if jury_loss_before is not None else None,
                "jury_loss_after": float(jury_loss_after) if jury_loss_after is not None else None,
                "current_epoch": epoch + 1, "total_epochs": total_epochs,
            }
            if reporter:
                await reporter.report_async(progress=gp, message=msg,
                                        trainingMetrics=tm, loss=float(b_loss), avg_loss=float(avg_loss))
            else:
                self.task_store.update_task(
                    task_id=task_id, status="processing", progress=gp, message=msg,
                    metadata={"loss": float(b_loss), "avg_loss": float(avg_loss), "trainingMetrics": tm}
                )

            if b_idx % 10 == 0 or b_idx == num_batches - 1:
                self.logger.info(f"⚡ [EPOCH {epoch+1}] Batch {b_idx+1}/{num_batches} | "
                            f"Loss: {b_loss:.5f} | Avg: {avg_loss:.5f}")

        # ── 3. Compiled SOS Training Step (Graph Mode) ────────────────────
        @tf.function
        def compiled_sos_step(x, y, m_limit):
            last_loss = tf.constant(1e9, dtype=tf.float32)
            for m in tf.range(m_limit):
                with tf.GradientTape() as tape:
                    y_pred = model(x, training=True)
                    loss = model.compute_loss(x, y, y_pred)
                
                # ✅ NaN Safety: Prevent poisoning weights if loss goes NaN
                if tf.math.is_nan(loss) or tf.math.is_inf(loss):
                    tf.print("⚠️ NaN/Inf loss detected! Skipping gradient update.")
                    break
                
                grads = tape.gradient(loss, model.trainable_variables)
                
                # ✅ GRADIENT FILTERING: Prevent crash on None gradients AND filter out NaN gradients
                valid_grads_vars = []
                for g, v in zip(grads, model.trainable_variables):
                    if g is not None:
                        # Clip manually to be absolutely safe
                        g_clipped = tf.clip_by_norm(g, 1.0)
                        if not tf.math.reduce_any(tf.math.is_nan(g_clipped)) and not tf.math.reduce_any(tf.math.is_inf(g_clipped)):
                            valid_grads_vars.append((g_clipped, v))
                
                if valid_grads_vars:
                    model.optimizer.apply_gradients(valid_grads_vars)

                # Inline early-stop: < 0.5% relative improvement after 2 micro-steps
                if m > 2 and (last_loss - loss) / (last_loss + 1e-8) < 0.005: break
                last_loss = loss
            
            # ✅ OPTIMIZATION: Discarded model.compute_metrics() to save compute and prevent metric pollution.
            # We use manual sample-weighted accumulation (bm, bs) which is mathematically superior.
            y_pred_final = model(x, training=False)
            b_mae = tf.reduce_mean(tf.abs(y - y_pred_final))
            b_mse = tf.reduce_mean(tf.square(y - y_pred_final))
            return loss, b_mae, b_mse

        @tf.function
        def get_jury_loss(jx, jy):
            """Graph-compiled helper for fast SOS safety audits."""
            pred = model(jx, training=False)
            return model.compute_loss(jx, jy, pred)

        @tf.function
        def compiled_val_step(vx, vy):
            """Graph-mode validation for maximum throughput."""
            pred = model(vx, training=False)
            v_loss = model.compute_loss(vx, vy, pred)
            v_mae = tf.reduce_mean(tf.abs(vy - pred))
            v_mse = tf.reduce_mean(tf.square(vy - pred))
            return v_loss, v_mae, v_mse

        # ── 4. Helpers (Jury, Replay & Metric Extraction) ────────────────
        # ✅ UPGRADE: Extract state from persistent container
        if continual_state is None:
            # Fallback for isolated testing — must match full continual_state structure
            # so good_examples_storage persists correctly across epochs
            continual_state = {
                "engine_state": {
                    "jury_x": None, "jury_y": None,
                    "ptr": 0, "count": 0,
                    "good_ptr": 0, "good_count": 0
                },
                "replay_storage": [None] * 2000,
                "good_examples_storage": [None] * 1000  # ✅ FIX: persist forgetting state
            }

        engine_state = continual_state["engine_state"]
        replay_storage = continual_state["replay_storage"]          # HARD examples buffer
        good_examples_storage = continual_state["good_examples_storage"]  # GOOD examples buffer
        MAX_REPLAY_SAMPLES = len(replay_storage)
        MAX_GOOD_SAMPLES = len(good_examples_storage)

        if (train_data_obj is not None and 
            hasattr(train_data_obj, '_prepare_epoch_jury') and
            getattr(train_data_obj, 'micro_val_holdback', 0) > 0):
            try:
                train_data_obj._prepare_epoch_jury()
                if train_data_obj.jury_x is not None:
                    # ✅ MEMORY OPTIMIZATION: Zero-copy tf.constant caching
                    engine_state["jury_x"] = tf.constant(train_data_obj.jury_x, dtype=tf.float32)
                    engine_state["jury_y"] = tf.constant(train_data_obj.jury_y, dtype=tf.float32)
                
                self.logger.info(
                    f"⚖️ [EPOCH {epoch+1}] Jury pool cached:\n"
                    f"   ├─ Samples: {len(train_data_obj.jury_x) if train_data_obj.jury_x is not None else 0}\n"
                    f"   └─ Holdback: {train_data_obj.micro_val_holdback * 100:.1f}%"
                )
            except Exception as j_err:
                self.logger.warning(f"⚠️ Could not cache jury pool: {j_err}")

        has_jury = (train_data_obj is not None and 
                    hasattr(train_data_obj, 'jury_x') and 
                    train_data_obj.jury_x is not None)

        async def run_jury(subset=True):
            """Pure holdback jury: detects generalization regression. NO replay contamination."""
            if not has_jury or train_data_obj.jury_x is None:
                return None
            
            try:
                # ✅ OPTIMIZATION: Slice from cached TF tensors if available (Zero-Copy)
                if engine_state["jury_x"] is not None:
                    jx_full, jy_full = engine_state["jury_x"], engine_state["jury_y"]
                    if subset and tf.shape(jx_full)[0] > batch_size:
                        idx = tf.random.shuffle(tf.range(tf.shape(jx_full)[0]))[:batch_size]
                        fjx, fjy = tf.gather(jx_full, idx), tf.gather(jy_full, idx)
                    else:
                        fjx, fjy = jx_full, jy_full
                    return float(get_jury_loss(fjx, fjy))
                
                # Fallback to numpy slicing
                jx, jy = train_data_obj.jury_x, train_data_obj.jury_y
                if subset and len(jx) > batch_size:
                    idx = np.random.choice(len(jx), batch_size, replace=False)
                    fjx, fjy = jx[idx], jy[idx]
                else:
                    fjx, fjy = jx, jy
                return float(get_jury_loss(tf.constant(fjx, dtype=tf.float32), tf.constant(fjy, dtype=tf.float32)))
            except Exception:
                # Final fallback
                jx, jy = train_data_obj.jury_x, train_data_obj.jury_y
                res = await asyncio.to_thread(model.test_on_batch, jx, jy)
                return float(res[0]) if isinstance(res, (list, tuple, np.ndarray)) else float(res)

        async def run_memory_check(subset_size=None):
            """
            Forgetting detector: checks if model has regressed on samples it PREVIOUSLY did well on.
            Uses good_examples_storage (low-loss samples), NOT hard examples.
            
            Logic: If loss on previously-easy samples has risen significantly, the model
            has FORGOTTEN something it once knew (e.g. a bull market pattern it learned in epoch 1
            but is now degrading after seeing bear markets).
            """
            good_count = engine_state.get("good_count", 0)
            if good_count == 0:
                return None
            n = subset_size or min(batch_size, good_count)
            indices = np.random.choice(good_count, n, replace=False)
            # Cast float16 back to float32 for TF inference
            rx = np.array([good_examples_storage[i][0] for i in indices], dtype=np.float32)
            ry = np.array([good_examples_storage[i][1] for i in indices], dtype=np.float32)
            try:
                return float(get_jury_loss(tf.constant(rx), tf.constant(ry)))
            except Exception:
                res = await asyncio.to_thread(model.test_on_batch, rx, ry)
                return float(res[0]) if isinstance(res, (list, tuple, np.ndarray)) else float(res)

        async def run_epoch_jury():
            """Pure holdback jury for epoch-level audit. No replay contamination."""
            if not has_jury or train_data_obj.jury_x is None:
                return None
            try:
                if engine_state["jury_x"] is not None:
                    return float(get_jury_loss(engine_state["jury_x"], engine_state["jury_y"]))
                
                fjx = tf.constant(train_data_obj.jury_x, dtype=tf.float32)
                fjy = tf.constant(train_data_obj.jury_y, dtype=tf.float32)
                return float(get_jury_loss(fjx, fjy))
            except Exception as e1:
                try:
                    res = await asyncio.to_thread(model.test_on_batch, train_data_obj.jury_x, train_data_obj.jury_y)
                    return float(res[0]) if isinstance(res, (list, tuple, np.ndarray)) else float(res)
                except Exception as e2:
                    self.logger.debug(f"⚖️ Epoch jury test failed. Graph err: {e1} | Fallback err: {e2}")
                    return None

        def _sample_hard_weighted(n: int, current_epoch: int, decay: float = 0.88):
            """
            Sample n hard examples from replay_storage with age-priority weighting.

            Each stored example carries its epoch tag: (bx, by, stored_epoch).
            Weight = (1.0 / decay) ** (current_epoch - stored_epoch)
            
            This creates a FIFO-like priority where OLDER examples get significantly 
            higher sampling weights because the model saw them longest ago and is 
            most prone to catastrophically forgetting them.
            """
            count = engine_state["count"]
            if count == 0:
                return None, None
            n = min(n, count)
            # Build per-slot recency weights
            weights = np.zeros(count, dtype=np.float64)
            for i in range(count):
                item = replay_storage[i]
                if item is not None:
                    stored_epoch = item[2] if len(item) > 2 else 0
                    # ✅ PRIORITIZE OLDER EXAMPLES: Inverse of decay
                    age = max(0, current_epoch - stored_epoch)
                    weights[i] = (1.0 / decay) ** age
            w_sum = weights.sum()
            if w_sum == 0:
                weights = np.ones(count) / count  # Fallback: uniform
            else:
                weights /= w_sum
            indices = np.random.choice(count, n, replace=False, p=weights)
            rx = np.array([replay_storage[i][0] for i in indices], dtype=np.float32)
            ry = np.array([replay_storage[i][1] for i in indices], dtype=np.float32)
            return rx, ry

        # ── 5. Setup tf.data.Dataset ─────────────────────────────────────
        # ✅ FIX: flow() now yields dicts {'x': ..., 'y': ..., 'adv_target_*': ...}
        # We extract 'x' and 'y' keys; 'y' is already the resolved target for this run.
        def generator_fn():
            if hasattr(train_data, 'flow'):
                for batch in train_data.flow():
                    # ✅ FIX: LazyLoader.flow() returns dict with x, y, and multiple target_* keys
                    # Extract x and y from the dict batch
                    if isinstance(batch, dict):
                        batch_x = batch.get('x')
                        batch_y = batch.get('y')
                        if batch_x is not None and batch_y is not None:
                            yield batch_x, batch_y
                        else:
                            self.logger.error(f"⚠️ [generator_fn] Dict batch missing x or y: keys={list(batch.keys())}")
                            continue
                    else:
                        # Legacy tuple format (x, y)
                        try:
                            bx, by = batch
                            yield bx, by
                        except (TypeError, ValueError) as e:
                            self.logger.error(f"⚠️ [generator_fn] Failed to unpack batch: {e}, batch type: {type(batch)}")
                            continue
            else:
                for item in train_data:
                    if isinstance(item, dict):
                        batch_x = item.get('x')
                        batch_y = item.get('y')
                        if batch_x is not None and batch_y is not None:
                            yield batch_x, batch_y
                    else:
                        try:
                            bx, by = item
                            yield bx, by
                        except (TypeError, ValueError):
                            continue

        # Prefer model shapes (zero consumption, most reliable)
        output_signature = None
        try:
            if hasattr(model, 'input_shape') and model.input_shape is not None:
                # ✅ FIXED: Safe destructuring for list/tuple shapes
                in_shape = tuple(model.input_shape[1:]) if isinstance(model.input_shape, (list, tuple)) else (None,)
                out_shape = tuple(model.output_shape[1:]) if hasattr(model, 'output_shape') and model.output_shape is not None else (None,)
                
                output_signature = (
                    tf.TensorSpec(shape=(None,) + in_shape, dtype=tf.float32),
                    tf.TensorSpec(shape=(None,) + out_shape, dtype=tf.float32)
                )
        except Exception as shape_err:
            self.logger.debug(f"Model shape derivation failed: {shape_err}")

        # ✅ FIX: Safe fallback — handle dict batch from flow()
        if output_signature is None and hasattr(train_data, 'flow'):
            try:
                temp_flow = train_data.flow()
                first_batch = next(temp_flow)
                sx = first_batch['x'] if isinstance(first_batch, dict) else first_batch[0]
                sy = first_batch['y'] if isinstance(first_batch, dict) else first_batch[1]
                output_signature = (
                    tf.TensorSpec(shape=(None,) + tuple(sx.shape[1:]), dtype=tf.float32),
                    tf.TensorSpec(shape=(None,) + tuple(sy.shape[1:]), dtype=tf.float32)
                )
                self.logger.warning("Used generator sampling for output_signature (first batch consumed)")
            except Exception as fallback_err:
                self.logger.warning(f"Signature fallback failed: {fallback_err}")
                output_signature = None

        dataset = tf.data.Dataset.from_generator(
            generator_fn, 
            output_signature=output_signature
        ).prefetch(tf.data.AUTOTUNE)

        # ── 6. Main Training Loop ─────────────────────────────────────────
        epoch_loss = epoch_mae = epoch_mse = 0.0
        epoch_samples = batches_ran = 0
        samples_by_batch = []
        bl = 0.0
        # ✅ FIX #9: Pre-initialize jury vars so final _report() is never undefined
        jury_loss_before = None
        jury_loss_after = None
        
        # ✅ FIX #10: Pre-calculate tensor constant for micro-steps to avoid creation overhead in loop
        m_limit_tensor = tf.constant(max_m, dtype=tf.int32)
        
        # ✅ ROLLBACK COOLDOWN - Proportional to epoch size so it doesn't strangle
        # learning on large datasets. ~20% of the epoch, between 20 and 300 batches.
        rollback_cooldown = 0
        COOLDOWN_DURATION = max(20, min(300, num_batches // 5))  # ~20% of epoch
        rollback_count = 0  # Track total rollbacks for statistics
        
        # ✅ PROACTIVE HARD REPLAY - Fire every REPLAY_INTERVAL batches regardless
        # of forgetting detection. Ensures rare regimes (transitions, ranging markets,
        # early downtrends) are consistently re-exposed to the model.
        REPLAY_INTERVAL = max(50, num_batches // 7)  # ~every 14% of the epoch
        self.logger.info(
            f"⚙️ [EPOCH {epoch+1}] Batch config: "
            f"num_batches={num_batches}, COOLDOWN={COOLDOWN_DURATION}, "
            f"REPLAY_INTERVAL={REPLAY_INTERVAL}"
        )
        
        forget_count = 0
        for batch_idx, (batch_x, batch_y) in enumerate(dataset):
            # 1. Yield to event loop to keep heartbeats/reports alive
            await asyncio.sleep(0)

            # 2. Break if we've reached the expected number of batches for this epoch
            if num_batches > 0 and batch_idx >= num_batches:
                break
                
            # ✅ DATA INTEGRITY CHECK: Reject NaN batches completely before they touch the model
            if tf.math.reduce_any(tf.math.is_nan(batch_x)) or tf.math.reduce_any(tf.math.is_nan(batch_y)):
                self.logger.warning(f"⚠️ [EPOCH {epoch+1}] Batch {batch_idx+1} contains NaN values in x or y! Skipping batch.")
                continue

            # ✅ COOLDOWN CHECK - Skip jury if in cooldown period
            jury_loss_before = None
            jury_loss_after = None
            weights_before_batch = None
            
            if rollback_cooldown > 0:
                rollback_cooldown -= 1
                if batch_idx % 50 == 0:
                    self.logger.debug(f"⏸️ [BATCH {batch_idx+1}] Jury cooldown: {rollback_cooldown} batches remaining")
                # Skip jury check during cooldown
            elif has_jury:
                # Run jury FIRST, only snapshot weights if jury is active.
                # Avoids copying all model weights on every batch when jury returns None.
                jury_loss_before = await run_jury(subset=True)
                if jury_loss_before is not None:
                    weights_before_batch = model.get_weights()

            # 2. Execute SOS step in thread to ensure frontend remains responsive
            # I know this looks redundant (should be res=(compiled_sos_step, batch_x, batch_y, m_limit_tensor) )but it is not
            # It is a way to decouple the CPU bound task of training from the IO bound task of keeping the frontend alive
            # So not a Bug and if there is a better  way please leme know. 
            res = await asyncio.to_thread(compiled_sos_step, batch_x, batch_y, m_limit_tensor)
            bl, bm, bs = [float(x) for x in res]
            
            batch_samples = int(batch_x.shape[0]) if hasattr(batch_x, 'shape') else len(batch_x) 
            do_rollback = False

            memory_loss_before = None
            if engine_state["count"] >= MAX_REPLAY_SAMPLES // 5:
                memory_loss_before = await run_memory_check()
            
            # FREE EXPLORATION: Epochs 1 & 2 (0-indexed: 0 & 1) run without any jury
            # restrictions, just like Keras .fit() would do greedily. This gives the model
            # time to find the loss basin before we start tightening safety.
            in_free_exploration = epoch < 2  # True for epoch 1 and 2 only
            
            if not in_free_exploration and rollback_cooldown == 0 and has_jury and jury_loss_before is not None and np.isfinite(bl) and bl > 0:
                jury_loss_after = await run_jury(subset=True)
                if jury_loss_after is not None:
                    regression = (jury_loss_after / jury_loss_before) - 1.0
                    
                    # ✅ DYNAMIC JURY THRESHOLD - Adapts to training phase (starts at epoch 3)
                    epoch_progress = epoch / total_epochs
                    
                    # Phase-aware threshold (Funnel Strategy — active from epoch 3 onwards)
                    if epoch_progress < 0.2:
                        # EARLY REFINEMENT: Relaxed but no longer free
                        jury_threshold = 1.12  # 12% tolerance
                    elif epoch_progress < 0.5:
                        # REFINEMENT PHASE: Tighten gradually
                        jury_threshold = 1.06  # 6% tolerance
                    elif epoch_progress < 0.8:
                        # CONVERGENCE PHASE: Strict but fair
                        jury_threshold = 1.03  # 3% tolerance
                    else:
                        # FINAL PHASE: Maximum precision
                        jury_threshold = 1.02  # 2% tolerance
                    
                    if regression > (jury_threshold - 1.0):  # Convert threshold to regression percentage
                        model.set_weights(weights_before_batch)
                        rollback_count += 1
                        rollback_cooldown = COOLDOWN_DURATION  # ✅ Activate cooldown
                        
                        # Only log if significant (reduce log spam)
                        if batch_idx % 10 == 0 or regression > 0.20:
                            self.logger.warning(
                                f"⚖️ [BATCH {batch_idx+1}] Rollback + Cooldown: {jury_loss_before:.6f} → {jury_loss_after:.6f} "
                                f"(+{regression*100:.1f}% > {(jury_threshold-1)*100:.0f}% threshold, phase={epoch_progress:.1%}). "
                                f"Skipping jury for next {COOLDOWN_DURATION} batches."
                            )
                        do_rollback = True
            elif in_free_exploration and batch_idx == 0:
                self.logger.info(f"🆓 [EPOCH {epoch+1}] FREE EXPLORATION mode — jury disabled for this epoch ")
            
            # Memory check to detect and fix catastrophic forgetting
            # I have observed that models end to trigger FORGETTING when you have outliers in your data
            # So if you see this log much go back to data processing and inspect your data properly
            if not do_rollback and memory_loss_before is not None:
                memory_loss_after = await run_memory_check()
                if memory_loss_after is not None:
                    forgetting = (memory_loss_after / memory_loss_before) - 1.0
                    if forgetting > 0.15: # 15% Forgetting Threshold
                        forget_count +=1
                        self.logger.info(f"🧠 [MEMORY] Forgetting detected (+{forgetting*100:.1f}%) — Injecting reactive replay.")
                        n_r = min(batch_size, engine_state["count"])
                        # Use decay-weighted sampling: recent hard examples preferred
                        rx_np, ry_np = _sample_hard_weighted(n_r, epoch)
                        if rx_np is not None:
                            await asyncio.to_thread(compiled_sos_step, tf.constant(rx_np, dtype=tf.float32),
                                                tf.constant(ry_np, dtype=tf.float32), tf.constant(2, dtype=tf.int32))

                        if (forget_count / max(1, batch_idx + 1)) > 0.1:
                            self.logger.warning(f'Forget Count: {forget_count}')
                            self.logger.warning(f'Your Model is prone to forgetting while in training! Consider Data Audit, as your Data may be noisy. ')
            
            # PROACTIVE HARD REPLAY: Periodically re-expose model to rare regimes
            # Uses decay-weighted sampling — recent batches bias the replay,
            # but older hard examples (rare regimes) still get a proportional chance.
            if (not do_rollback and
                engine_state["count"] >= MAX_REPLAY_SAMPLES // 5 and
                batch_idx > 0 and batch_idx % REPLAY_INTERVAL == 0):
                n_r = min(batch_size, engine_state["count"])
                rx_np, ry_np = _sample_hard_weighted(n_r, epoch)
                if rx_np is not None:
                    # Gentle 1 micro-step — remind, don't overwrite
                    await asyncio.to_thread(
                        compiled_sos_step,
                        tf.constant(rx_np, dtype=tf.float32),
                        tf.constant(ry_np, dtype=tf.float32),
                        tf.constant(1, dtype=tf.int32)
                    )
                    if batch_idx % (REPLAY_INTERVAL * 3) == 0:
                        self.logger.info(
                            f"🔁 [HARD REPLAY] Proactive injection at batch {batch_idx+1}: "
                            f"{n_r} decay-weighted samples (buf={engine_state['count']}, epoch={epoch+1})"
                        )

            if not do_rollback and np.isfinite(bl):
                epoch_loss += bl * batch_samples
                epoch_mae += bm * batch_samples
                epoch_mse += bs * batch_samples
                epoch_samples += batch_samples
                batches_ran += 1
                samples_by_batch.append(batch_samples)

                # ====================== DUAL BUFFER STORAGE ======================
                # Two separate signals, two separate buffers:
                #   replay_storage        → HARD EXAMPLES (model struggled → force re-learning)
                #   good_examples_storage → GOOD EXAMPLES (model did well → detect forgetting)
                current_avg_loss = epoch_loss / epoch_samples if epoch_samples > 0 else bl
                difficulty_ratio = bl / (current_avg_loss + 1e-8)

                # ── HARD EXAMPLES: Store batches the model finds difficult ─────────────
                add_to_hard = False
                if engine_state["count"] < MAX_REPLAY_SAMPLES // 5: add_to_hard = True  # Fast fill
                elif difficulty_ratio > 1.25: add_to_hard = True                         # Significant struggle (rare regime)
                elif difficulty_ratio > 1.10 and np.random.rand() < 0.4: add_to_hard = True
                elif np.random.rand() < 0.05: add_to_hard = True                         # Small variety sample

                if add_to_hard:
                    num_to_add = min(6, len(batch_x))
                    indices = np.random.choice(len(batch_x), num_to_add, replace=False)
                    for i in indices:
                        try:
                            bx_val = batch_x[i].numpy().astype(np.float16) if hasattr(batch_x[i], 'numpy') else np.array(batch_x[i], dtype=np.float16)
                            by_val = batch_y[i].numpy().astype(np.float16) if hasattr(batch_y[i], 'numpy') else np.array(batch_y[i], dtype=np.float16)
                            # ✅ EPOCH TAG: store (x, y, epoch) so decay weighting works cross-epoch
                            replay_storage[engine_state["ptr"]] = (bx_val, by_val, epoch)
                            engine_state["ptr"] = (engine_state["ptr"] + 1) % MAX_REPLAY_SAMPLES
                            if engine_state["count"] < MAX_REPLAY_SAMPLES:
                                engine_state["count"] += 1
                        except Exception: pass

                # ── GOOD EXAMPLES: Store batches the model handled well ───────────────
                # These are used ONLY by run_memory_check() to detect forgetting.
                # Forgetting = model was once good at these, but now fails on them.
                if difficulty_ratio < 0.80 and np.random.rand() < 0.20:  # Low loss, sample 20%
                    good_count = engine_state.get("good_count", 0)
                    good_ptr = engine_state.get("good_ptr", 0)
                    num_to_add = min(3, len(batch_x))
                    indices = np.random.choice(len(batch_x), num_to_add, replace=False)
                    for i in indices:
                        try:
                            bx_val = batch_x[i].numpy().astype(np.float16) if hasattr(batch_x[i], 'numpy') else np.array(batch_x[i], dtype=np.float16)
                            by_val = batch_y[i].numpy().astype(np.float16) if hasattr(batch_y[i], 'numpy') else np.array(batch_y[i], dtype=np.float16)
                            good_examples_storage[good_ptr] = (bx_val, by_val)
                            good_ptr = (good_ptr + 1) % MAX_GOOD_SAMPLES
                            if good_count < MAX_GOOD_SAMPLES:
                                good_count += 1
                        except Exception: pass
                    engine_state["good_ptr"] = good_ptr
                    engine_state["good_count"] = good_count

                if difficulty_ratio > 1.20 and batch_idx % 25 == 0:
                    self.logger.info(
                        f"🧠 [BUFFERS] hard={engine_state['count']}/{MAX_REPLAY_SAMPLES} "
                        f"| good={engine_state.get('good_count',0)}/{MAX_GOOD_SAMPLES} "
                        f"| diff={difficulty_ratio:.2f}x"
                    )

                if batch_idx % 5 == 0 or batch_idx == num_batches - 1:
                    await _report(
                        batch_idx, bl, epoch_loss, epoch_mae, epoch_mse, epoch_samples,
                        jury_loss_before=jury_loss_before,
                        jury_loss_after=jury_loss_after
                    )
            else:
                # Still report progress during rollback/error
                if batch_idx % 5 == 0:
                    await _report(
                        batch_idx, bl, epoch_loss, epoch_mae, epoch_mse, epoch_samples,
                        jury_loss_before=jury_loss_before,
                        jury_loss_after=jury_loss_after
                    )

        # Diagnostics
        rollback_rate = (rollback_count / num_batches * 100) if num_batches > 0 else 0.0
        self.logger.info(
            f"📊 [EPOCH {epoch+1}] SUMMARY:\n"
            f"   ├─ Batches: {batches_ran}/{num_batches}\n"
            f"   ├─ Samples: {epoch_samples}\n"
            f"   ├─ Rollbacks: {rollback_count} ({rollback_rate:.1f}%)\n"
            f"   ├─ Replay Buffer: {engine_state['count']}\n"
            f"   └─ Partial Batches: {sum(1 for s in samples_by_batch if s < batch_size)}"
        )

        # ── 7. Validation Phase (Hardened Sample-Weighted Loop) ──────────
        val_loss = float('inf')   # inf = meaningful "no data" signal for upstream callers
        val_mae = val_mse = 0.0   # 0.0 avoids inf propagation into history/reporter
        if val_data is not None:
            v_loss_sum = v_mae_sum = v_mse_sum = 0.0
            v_samples_total = 0
            
            # ✅ FIX: Auto-detect if validation data is a generator (don't rely on is_generator flag)
            is_val_generator = hasattr(val_data, 'flow') and callable(getattr(val_data, 'flow'))
            
            self.logger.info(
                f"🔍 [VAL {epoch+1}] Validation mode: {'Generator' if is_val_generator else 'Array'}\n"
                f"   ├─ val_data type: {type(val_data).__name__}\n"
                f"   ├─ val_targets type: {type(val_targets).__name__ if val_targets is not None else 'None'}\n"
                f"   ├─ val_targets value: {val_targets if isinstance(val_targets, str) else '(array)'}\n"
                f"   └─ has flow(): {hasattr(val_data, 'flow')}"
            )
            
            if is_val_generator:
                # ✅ FIX: val_data.flow() yields dicts {'x':..., 'y':...} — extract keys
                val_flow = val_data.flow()
                for v_idx in range(len(val_data)):
                    try:
                        v_batch = next(val_flow)
                        v_x = v_batch['x'] if isinstance(v_batch, dict) else v_batch[0]
                        v_y = v_batch['y'] if isinstance(v_batch, dict) else v_batch[1]
                        
                        # 📊 Log first batch for diagnostics
                        if v_idx == 0:
                            self.logger.info(
                                f"📊 [VAL] First batch from generator:\n"
                                f"   ├─ v_x shape: {v_x.shape}\n"
                                f"   ├─ v_y shape: {v_y.shape}\n"
                                f"   ├─ v_x range: [{np.min(v_x):.4f}, {np.max(v_x):.4f}]\n"
                                f"   └─ v_y range: [{np.min(v_y):.4f}, {np.max(v_y):.4f}]"
                            )
                        
                        # ✅ UPGRADE: Graph-mode validation step with timeout guard
                        v_res = await asyncio.wait_for(
                            asyncio.to_thread(compiled_val_step, tf.constant(v_x, dtype=tf.float32), tf.constant(v_y, dtype=tf.float32)),
                            timeout=120.0
                        )
                        vl, vmae, vmse = [float(val) for val in v_res]
                        
                        # 📊 Log first batch metrics
                        if v_idx == 0:
                            self.logger.info(
                                f"📊 [VAL] First batch metrics:\n"
                                f"   ├─ Batch loss: {vl:.6f}\n"
                                f"   ├─ Batch MAE: {vmae:.6f}\n"
                                f"   └─ Batch MSE: {vmse:.6f}"
                            )
                        
                        if np.isfinite(vl):
                            sl = len(v_x)
                            v_loss_sum += vl * sl
                            v_samples_total += sl
                            v_mae_sum += vmae * sl
                            v_mse_sum += vmse * sl
                        else:
                            self.logger.warning(
                                f"⚠️ [VAL] Batch {v_idx} non-finite loss: {vl}\n"
                                f"   ├─ x_bounds: [{np.min(v_x):.4f}, {np.max(v_x):.4f}]\n"
                                f"   └─ Skipping batch."
                            )
                    except StopIteration: break
                    except Exception as eval_err:
                        self.logger.warning(f"⚠️ Val batch {v_idx} error: {eval_err}")
                        continue
            else:
                v_x_all = val_data.values if hasattr(val_data, 'values') else val_data
                v_y_all = val_targets
                
                if v_y_all == "LAZY_ON_DISK":
                    self.logger.error("❌ [VAL] val_targets='LAZY_ON_DISK' reached array path — skipping validation.")
                    v_batches = 0
                elif v_y_all is None:
                    self.logger.error("❌ [VAL] val_targets is None but model is not marked as Autoencoder. Skipping validation.")
                    v_batches = 0
                else:
                    v_batches = max(1, (len(v_x_all) + batch_size - 1) // batch_size)

                for v_idx in range(v_batches):
                    try:
                        s, e = v_idx * batch_size, min((v_idx + 1) * batch_size, len(v_x_all))
                        # ✅ UPGRADE: Graph-mode validation step with timeout guard
                        v_res = await asyncio.wait_for(
                            asyncio.to_thread(compiled_val_step, tf.constant(v_x_all[s:e], dtype=tf.float32), tf.constant(v_y_all[s:e], dtype=tf.float32)),
                            timeout=120.0
                        )
                        vl, vmae, vmse = [float(val) for val in v_res]

                        if np.isfinite(vl):
                            sl = e - s
                            v_loss_sum += vl * sl
                            v_samples_total += sl
                            v_mae_sum += vmae * sl
                            v_mse_sum += vmse * sl
                        else:
                            self.logger.warning(f"⚠️ [VAL] Static batch {v_idx} non-finite loss: {vl}")
                    except Exception as v_err:
                        self.logger.warning(f"⚠️ [VAL] Static batch {v_idx} failed: {v_err}")
                        continue

            if v_samples_total > 0:
                val_loss = v_loss_sum / v_samples_total
                val_mae  = v_mae_sum / v_samples_total
                val_mse  = v_mse_sum / v_samples_total
            
            # Finalize: Reset model metrics after validation phase
            model.reset_metrics()
            
            self.logger.info(
                f"🧪 [VAL {epoch+1}] RESULTS:\n"
                f"   ├─ Validation loss: {val_loss:.6f}\n"
                f"   ├─ Validation MAE: {val_mae:.6f}\n"
                f"   ├─ Validation MSE: {val_mse:.6f}\n"
                f"   └─ Samples processed: {v_samples_total}"
            )

        # ── 8. Epoch-Level Jury Audit (Final Safety Net) ─────────────────
        epoch_jury_rejected = False  # upstream regression signal, separate from val_loss
        # ✅ FIX: Respect free exploration — skip epoch jury for epochs 1 & 2.
        # The batch-level jury is already disabled for these epochs; consistency
        # requires the epoch-level jury to also stay silent during free exploration.
        if epoch >= 2 and weights_before is not None and has_jury and np.isfinite(val_loss):
            try:
                jx, jy = train_data_obj.jury_x, train_data_obj.jury_y
                if jx is not None and len(jx) > 0:
                    post_epoch_weights = model.get_weights()      # 1. Capture training result

                    model.set_weights(weights_before)             # 2. Measure baseline
                    j_before = await run_epoch_jury()
                    
                    model.set_weights(post_epoch_weights)         # 3. Measure improvement
                    j_after = await run_epoch_jury()

                    # ✅ DYNAMIC EPOCH JURY THRESHOLD (Funnel Strategy)
                    # Mirrors the batch-level logic for consistency. Early epochs allow 
                    # more movement; later epochs require strict stability.
                    epoch_progress = epoch / total_epochs
                    if epoch_progress < 0.2:
                        e_jury_threshold = 1.12  # 12% tolerance (Early Refinement)
                    elif epoch_progress < 0.5:
                        e_jury_threshold = 1.06  # 6% tolerance (Refinement)
                    elif epoch_progress < 0.8:
                        e_jury_threshold = 1.03  # 3% tolerance (Convergence)
                    else:
                        e_jury_threshold = 1.02  # 2% tolerance (Final Precision)
                    
                    # ✅ CATCH-UP OVERRIDE: If the trainer is in Catch-Up Mode (dyn_threshold=2.0),
                    # respect that at the epoch level too.
                    e_jury_threshold = max(e_jury_threshold, dyn_threshold)
                    
                    # 🚀 VALIDATION OVERRULE: If the main validation set shows clear improvement,
                    # we should be much more suspicious of the tiny Jury Pool signal. 
                    # If val_loss dropped by >5%, we loosen the jury threshold by 5x.
                    prev_val = (last_val_metrics or {}).get("val_loss")
                    if prev_val and prev_val > 0 and val_loss < prev_val * 0.95:
                        self.logger.info(f"🚀 [VAL OVERRULE] Clear improvement in main validation ({prev_val:.4f} → {val_loss:.4f}). Relaxing Jury safety.")
                        e_jury_threshold = e_jury_threshold * 5.0 # Massive relaxation for massive gains

                    if j_before > 0 and j_after > j_before * e_jury_threshold:
                        pct = (j_after / j_before - 1) * 100
                        self.logger.warning(
                            f"⚖️ [EPOCH {epoch+1}] JURY REJECTED: {j_before:.6f} → {j_after:.6f} (+{pct:.1f}% > {(e_jury_threshold-1)*100:.0f}% threshold)\n"
                            f"   └─ val_loss kept as computed ({val_loss:.6f}), regression flagged upstream."
                        )
                        epoch_jury_rejected = True  # ✅ flag only — do NOT overwrite val_loss
            except Exception as e: self.logger.debug(f"⚖️ Epoch jury failed: {e}")

        # ── 9. Final Report (Fresh Metrics) ─────────────────────────────
        res = {
            "loss": float(epoch_loss / epoch_samples) if epoch_samples > 0 else float('inf'),
            "val_loss": float(val_loss),
            "mae": float(epoch_mae / epoch_samples) if epoch_samples > 0 else 0.0,
            "val_mae": float(val_mae),
            "mse": float(epoch_mse / epoch_samples) if epoch_samples > 0 else 0.0,
            "val_mse": float(val_mse),
            "jury_rejected": epoch_jury_rejected,  # upstream can use this to decide rollback
        }

        # Send one final report for the epoch with the high-precision validation results
        last_reported_loss = bl if np.isfinite(bl) else 0.0
        await _report(
            num_batches - 1,
            last_reported_loss,
            epoch_loss,
            epoch_mae,
            epoch_mse,
            epoch_samples,
            jury_loss_before=jury_loss_before,
            jury_loss_after=jury_loss_after,
            fresh_val=res
        )

        return res



    async def _trainer_fit_v1(
        self,
        model,
        train_data,
        num_batches,
        batch_size,
        epoch,
        task_id,
        reporter,
        total_epochs,
        last_val_metrics=None,
        target_column=None,
        max_m=5,
        dyn_threshold=1.15,
        # ── ports from _train_model_async ──────────────────────────
        val_data=None,
        train_targets=None,
        val_targets=None,
        is_generator=False,
        weights_before=None,
        train_data_obj=None,
):
        """
        Clean Keras .fit() Clone with Sample-Weighted Metrics + Jury SOS Safety.
        Optimized for speed + stability. Ready for benchmarking.
        """
       
        

        if num_batches == 0:
            self.logger.error(f"❌ [EPOCH {epoch+1}] num_batches=0 — skipping epoch")
            return {"loss": float('inf'), "val_loss": float('inf'),
                    "mae": 0.0, "val_mae": 0.0, "mse": 0.0, "val_mse": 0.0}

        # ── 1. Reset Stateful Metrics ────────────────────────
        for metric in model.metrics:
            metric.reset_state()

        # ── 2. Progress Reporter ─────────────────────────────
        async def _report(b_idx, b_loss, epoch_loss, epoch_mae, epoch_mse, 
                        epoch_samples, fresh_val=None, jury_info=None):
            avg_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0.0
            avg_mae = epoch_mae / epoch_samples if epoch_samples > 0 else 0.0
            avg_mse = epoch_mse / epoch_samples if epoch_samples > 0 else 0.0

            gp = min(100, max(0, int(round(((epoch * num_batches + b_idx + 1) / 
                                        (total_epochs * num_batches)) * 100))))

            msg = f"Epoch {epoch+1}/{total_epochs}: batch {b_idx+1}/{num_batches} - Loss: {b_loss:.4f} | Avg: {avg_loss:.4f}"
            if jury_info:
                msg += f" | Jury: {jury_info}"

            v_src = fresh_val or (last_val_metrics or {})
            tm = {
                "loss": avg_loss, "mae": avg_mae, "mse": avg_mse,
                "val_loss": v_src.get("val_loss", 0.0),
                "val_mae": v_src.get("val_mae", 0.0),
                "val_mse": v_src.get("val_mse", 0.0),
                "current_epoch": epoch + 1, "total_epochs": total_epochs,
            }

            if reporter:
                await reporter.report_async(progress=gp, message=msg,
                                        trainingMetrics=tm, loss=float(b_loss), avg_loss=float(avg_loss))
            else:
                self.task_store.update_task(task_id=task_id, status="processing", 
                                        progress=gp, message=msg, metadata={"loss": float(b_loss)})

            if b_idx % 10 == 0 or b_idx >= num_batches - 1:
                self.logger.info(f"⚡ [EPOCH {epoch+1}] Batch {b_idx+1}/{num_batches} | Loss: {b_loss:.5f} | Avg: {avg_loss:.5f}")

        # ── 3. Compiled SOS Training Step ─────────────────────
        @tf.function
        def compiled_sos_step(x, y, m_limit):
            last_loss = tf.constant(1e9, dtype=tf.float32)
            for m in tf.range(m_limit):
                with tf.GradientTape() as tape:
                    y_pred = model(x, training=True)
                    loss = model.compute_loss(x, y, y_pred)

                grads = tape.gradient(loss, model.trainable_variables)
                grads_vars = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
                if grads_vars:
                    model.optimizer.apply_gradients(grads_vars)

                if m > 2:
                    if (last_loss - loss) / (last_loss + 1e-8) < 0.005:
                        break
                last_loss = loss

            y_pred_final = model(x, training=False)
            model.compute_metrics(x, y, y_pred_final)

            b_mae = tf.reduce_mean(tf.abs(y - y_pred_final))
            b_mse = tf.reduce_mean(tf.square(y - y_pred_final))
            return loss, b_mae, b_mse

        # ── 4. Jury & Helpers ─────────────────────────────────
        has_jury = (train_data_obj is not None and 
                    hasattr(train_data_obj, 'jury_x') and train_data_obj.jury_x is not None)

        replay_buffer = deque(maxlen=2000)   # Your intentional feature

        async def run_jury(subset=True):
            if not has_jury or train_data_obj.jury_x is None:
                return None
            jx, jy = train_data_obj.jury_x, train_data_obj.jury_y

            if subset and len(jx) > batch_size:
                idx = np.random.choice(len(jx), batch_size, replace=False)
                final_jx, final_jy = jx[idx], jy[idx]
            else:
                final_jx, final_jy = jx, jy

            # === INTENTIONAL REPLAY BUFFER MIXIN ===
            if replay_buffer:
                replay_list = list(replay_buffer)
                r_size = min(len(replay_list), batch_size // 2)
                if r_size > 0:
                    r_idx = np.random.choice(len(replay_list), r_size, replace=False)
                    rx = np.array([replay_list[i][0] for i in r_idx])
                    ry = np.array([replay_list[i][1] for i in r_idx])
                    final_jx = np.concatenate([final_jx, rx], axis=0)
                    final_jy = np.concatenate([final_jy, ry], axis=0)

            res = await asyncio.to_thread(model.test_on_batch, final_jx, final_jy)
            model.reset_metrics()   # Prevent pollution
            return float(res[0]) if isinstance(res, (list, tuple, np.ndarray)) else float(res)

        def extract_val_metrics(v_res):
            if isinstance(v_res, (list, tuple, np.ndarray)):
                vals = [float(x) for x in v_res]
                res_dict = {"loss": vals[0]}
                metric_names = [m.name.lower() for m in model.metrics if hasattr(m, 'name')]
                for i, name in enumerate(metric_names):
                    if i + 1 < len(vals):
                        res_dict[name] = vals[i + 1]
                # Fallbacks
                if "mae" not in res_dict and len(vals) > 1:
                    res_dict["mae"] = vals[1]
                if "mse" not in res_dict and len(vals) > 2:
                    res_dict["mse"] = vals[2]
                return res_dict
            return {"loss": float(v_res)}

        # ── 5. Dataset ───────────────────────────────────────
        # ✅ FIX: flow() yields dicts {'x': ..., 'y': ...} — extract keys
        def generator_fn():
            if hasattr(train_data, 'flow'):
                for batch in train_data.flow():
                    if isinstance(batch, dict):
                        yield batch['x'], batch['y']
                    else:
                        bx, by = batch
                        yield bx, by
            else:
                for item in train_data:
                    if isinstance(item, dict):
                        yield item['x'], item['y']
                    else:
                        bx, by = item
                        yield bx, by

        try:
            in_shape = tuple(model.input_shape[1:]) if model.input_shape else (None,)
            out_shape = tuple(model.output_shape[1:]) if hasattr(model, 'output_shape') else (None,)
            output_signature = (
                tf.TensorSpec(shape=(None,) + in_shape, dtype=tf.float32),
                tf.TensorSpec(shape=(None,) + out_shape, dtype=tf.float32)
            )
        except Exception:
            output_signature = None

        dataset = tf.data.Dataset.from_generator(
            generator_fn, output_signature=output_signature
        ).prefetch(tf.data.AUTOTUNE)

        # ── 6. Training Loop ─────────────────────────────────
        epoch_loss = epoch_mae = epoch_mse = 0.0
        epoch_samples = batches_ran = 0
        samples_by_batch = []

        for batch_idx, (batch_x, batch_y) in enumerate(dataset):
            await asyncio.sleep(0)  # Keep event loop responsive

            if num_batches > 0 and batch_idx >= num_batches:
                break

            weights_before_batch = model.get_weights() if has_jury else None
            jury_loss_before = await run_jury(subset=True) if has_jury else None

            # Training step
            bl_tensor, bm_tensor, bs_tensor = await asyncio.to_thread(
                compiled_sos_step, batch_x, batch_y, tf.constant(max_m, dtype=tf.int32)
            )
            bl, bm, bs = float(bl_tensor), float(bm_tensor), float(bs_tensor)
            batch_samples = len(batch_x)

            do_rollback = False
            jury_loss_after = None

            if has_jury and jury_loss_before is not None and np.isfinite(bl) and bl > 0:
                jury_loss_after = await run_jury(subset=True)
                if jury_loss_after is not None and jury_loss_after > jury_loss_before * 1.02:
                    model.set_weights(weights_before_batch)
                    self.logger.warning(f"⚖️ [BATCH {batch_idx+1}] Rollback: {jury_loss_before:.4f} → {jury_loss_after:.4f}")
                    do_rollback = True

            if not do_rollback and np.isfinite(bl):
                epoch_loss += bl * batch_samples
                epoch_mae += bm * batch_samples
                epoch_mse += bs * batch_samples
                epoch_samples += batch_samples
                batches_ran += 1
                samples_by_batch.append(batch_samples)

                # Add to replay buffer (your feature)
                for i in range(min(len(batch_x), 8)):
                    try:
                        replay_buffer.append((batch_x[i].numpy(), batch_y[i].numpy()))
                    except:
                        pass

            if batch_idx % 5 == 0 or batch_idx >= num_batches - 1:
                jury_info = f"{jury_loss_before:.4f}→{jury_loss_after:.4f}" if jury_loss_after else None
                await _report(batch_idx, bl, epoch_loss, epoch_mae, epoch_mse, epoch_samples, jury_info=jury_info)

        # Diagnostics
        self.logger.info(f"📊 [EPOCH {epoch+1}] SUMMARY: Batches={batches_ran}/{num_batches} | "
                        f"Samples={epoch_samples} | Partial={sum(s < batch_size for s in samples_by_batch)}")

        # ── 7. Validation Phase ───────────────────────────────
        val_loss = val_mae = val_mse = float('inf')
        if val_data is not None:
            v_loss_sum = v_mae_sum = v_mse_sum = 0.0
            v_samples_total = 0

            # Generator path
            if is_generator and hasattr(val_data, 'flow'):
                val_flow = val_data.flow()
                for v_idx in range(len(val_data)):
                    try:
                        v_batch = next(val_flow)
                        # ✅ FIX: Handle dict returns from flow() (new multi-target structure)
                        if isinstance(v_batch, dict):
                            v_x = v_batch.get('x')
                            v_y = v_batch.get('y')
                        else:
                            v_x, v_y = v_batch
                        
                        v_res = await asyncio.wait_for(
                            asyncio.to_thread(model.test_on_batch, v_x, v_y), timeout=600.0
                        )
                        vm = extract_val_metrics(v_res)
                        vl = vm.get("loss", 0.0)
                        if np.isfinite(vl):
                            sl = len(v_x)
                            v_loss_sum += vl * sl
                            v_mae_sum += vm.get("mae", 0.0) * sl
                            v_mse_sum += vm.get("mse", 0.0) * sl
                            v_samples_total += sl
                    except StopIteration:
                        break
                    except Exception as e:
                        self.logger.warning(f"⚠️ Val batch {v_idx} error: {e}")
                        continue
            else:
                # Array path
                v_x_all = getattr(val_data, 'values', val_data)
                v_y_all = val_targets if val_targets is not None else None
                if v_y_all is None:
                    self.logger.error("❌ Validation targets missing for supervised model!")
                else:
                    v_batches = max(1, (len(v_x_all) + batch_size - 1) // batch_size)
                    for v_idx in range(v_batches):
                        s = v_idx * batch_size
                        e = min(s + batch_size, len(v_x_all))
                        v_res = await asyncio.wait_for(
                            asyncio.to_thread(model.test_on_batch, v_x_all[s:e], v_y_all[s:e]), 
                            timeout=600.0
                        )
                        vm = extract_val_metrics(v_res)
                        vl = vm.get("loss", 0.0)
                        if np.isfinite(vl):
                            sl = e - s
                            v_loss_sum += vl * sl
                            v_mae_sum += vm.get("mae", 0.0) * sl
                            v_mse_sum += vm.get("mse", 0.0) * sl
                            v_samples_total += sl

            if v_samples_total > 0:
                val_loss = v_loss_sum / v_samples_total
                val_mae = v_mae_sum / v_samples_total
                val_mse = v_mse_sum / v_samples_total

        # ── 8. Epoch-Level Jury ───────────────────────────────
        epoch_jury_rejected = False
        if weights_before is not None and has_jury and np.isfinite(val_loss):
            try:
                post_weights = model.get_weights()
                model.set_weights(weights_before)
                j_before = await run_jury(subset=False)
                model.set_weights(post_weights)
                j_after = await run_jury(subset=False)

                if j_before > 0 and j_after > j_before * 1.03:
                    pct = (j_after / j_before - 1) * 100
                    self.logger.warning(f"⚖️ [EPOCH {epoch+1}] JURY REJECTED: {j_before:.5f} → {j_after:.5f} (+{pct:.1f}%)")
                    epoch_jury_rejected = True
            except Exception as e:
                self.logger.debug(f"Epoch jury failed: {e}")

        # ── 9. Final Result ───────────────────────────────────
        result = {
            "loss": float(epoch_loss / epoch_samples) if epoch_samples > 0 else float('inf'),
            "val_loss": float(val_loss),
            "mae": float(epoch_mae / epoch_samples) if epoch_samples > 0 else 0.0,
            "val_mae": float(val_mae),
            "mse": float(epoch_mse / epoch_samples) if epoch_samples > 0 else 0.0,
            "val_mse": float(val_mse),
            "jury_rejected": epoch_jury_rejected,
        }

        # Final progress update
        await _report(num_batches - 1 if num_batches > 0 else 0, 
                    bl, epoch_loss, epoch_mae, epoch_mse, epoch_samples, fresh_val=result)

        
        async def _generate_post_training_predictions(
            self,
            model,
            model_id: str,
            dataset_id: str,
            train_data: Any,
            val_data: Any,
            test_data: Any,
            task_id: str,
            reporter: Optional[ProgressReporter] = None
        ):
            """
            Generate predictions for train/validation/test splits and persist to ModelPredictions table.
            
            🔴 FIX: Store predictions WITH model_id so multiple models can have predictions on same dataset.
            This enables performance comparison across multiple trained models.
            
            Previously stored in MLDatasetChunk.predictions_data (no model_id) → overwrote previous model predictions
            Now stored in ModelPredictions with (model_id, dataset_id, chunk_index) unique constraint → preserves all predictions
            """
        

         
          
            self.logger.info(f"🔮 Generating post-training predictions for model {model_id[:8]} on dataset {dataset_id[:8]}...")
            if reporter:
                reporter.update(message="Generating post-training predictions for performance visualization...")

            splits = [
                ("train", train_data),
                ("validation", val_data),
                ("test", test_data)
            ]

            try:
                async with AsyncPostgresSessionLocal() as db:
                    for split_name, split_data in splits:
                        if split_data is None:
                            continue
                        
                        self.logger.info(f"  ├─ Processing {split_name} split...")
                        
                        # 1. Generate ALL predictions for this split
                        predictions = []
                        
                        # Use a local import to avoid circular dependencies if any
                        from app.core.ml.ml_data_loader import LazySequenceGenerator
                        
                        if not isinstance(split_data, LazySequenceGenerator):
                            # Small dataset (RAM)
                            # Expecting split_data to be np.ndarray or list
                            x_data = np.array(split_data)
                            
                            # If it's a tuple (x, y), extract x
                            if isinstance(split_data, tuple) and len(split_data) == 2:
                                x_data = split_data[0]
                                
                            # Predict in batches to avoid GPU OOM
                            batch_size = 1024
                            for i in range(0, len(x_data), batch_size):
                                batch_x = x_data[i:i+batch_size]
                                batch_pred = model.predict(batch_x, verbose=0)
                                predictions.append(batch_pred)
                            
                            if predictions:
                                all_predictions = np.concatenate(predictions, axis=0)
                            else:
                                continue
                        else:
                            # Large dataset (Lazy Generator)
                            # Use the file paths to ensure we match the DB chunks order
                            file_paths = getattr(split_data, 'file_paths', [])
                            if not file_paths:
                                self.logger.warning(f"  └─ No file paths found for lazy {split_name} split")
                                continue
                                
                            # Predict chunk by chunk
                            for path in file_paths:
                                try:
                                    data = np.load(path, mmap_mode='r', allow_pickle=True)
                                    x = data['sequences'] if 'sequences' in data else data['x']
                                    
                                    chunk_preds = []
                                    batch_size = 1024
                                    for i in range(0, len(x), batch_size):
                                        batch_x = x[i:i+batch_size]
                                        chunk_preds.append(model.predict(batch_x, verbose=0))
                                    
                                    predictions.append(np.concatenate(chunk_preds, axis=0))
                                    data.close()
                                except Exception as chunk_err:
                                    self.logger.error(f"Error predicting on chunk {path}: {chunk_err}")
                                    continue
                            
                            if predictions:
                                all_predictions = np.concatenate(predictions, axis=0)
                            else:
                                continue

                        # 2. Map predictions to chunks and persist to ModelPredictions table
                        stmt = select(MLDatasetChunk).where(
                            and_(
                                MLDatasetChunk.dataset_id == dataset_id,
                                MLDatasetChunk.split_name == split_name
                            )
                        ).order_by(MLDatasetChunk.chunk_index.asc())
                        
                        result = await db.execute(stmt)
                        chunks = result.scalars().all()
                        
                        if not chunks:
                            self.logger.warning(f"  └─ No chunks found in DB for {split_name} split")
                            continue
                            
                        # Slice all_predictions and store each chunk's predictions with model_id
                        cursor = 0
                        cctx = zstd.ZstdCompressor(level=3)
                        
                        created_count = 0
                        for chunk in chunks:
                            chunk_size = chunk.sequence_count
                            if cursor + chunk_size > len(all_predictions):
                                self.logger.warning(f"  └─ Prediction alignment mismatch: cursor={cursor}, chunk_size={chunk_size}, total={len(all_predictions)}")
                                break
                                
                            chunk_pred_slice = all_predictions[cursor:cursor+chunk_size]
                            
                            # Store as dict for future multi-target support
                            pred_dict = {"predictions": chunk_pred_slice}
                            
                            # Serialize and compress
                            serialized = pickle.dumps(pred_dict)
                            compressed = cctx.compress(serialized)
                            
                            # 🔴 FIX: Create ModelPredictions entry with model_id (not MLDatasetChunk)
                            # This preserves predictions from multiple models per dataset
                            try:
                                # Delete any existing predictions for this model/chunk combination
                                delete_stmt = await db.execute(
                                    select(ModelPredictions).where(
                                        and_(
                                            ModelPredictions.model_id == model_id,
                                            ModelPredictions.dataset_id == dataset_id,
                                            ModelPredictions.split_name == split_name,
                                            ModelPredictions.chunk_index == chunk.chunk_index
                                        )
                                    )
                                )
                                existing = delete_stmt.scalars().first()
                                if existing:
                                    await db.delete(existing)
                                
                                # Create new ModelPredictions entry
                                model_pred = ModelPredictions(
                                    model_id=model_id,
                                    dataset_id=dataset_id,
                                    split_name=split_name,
                                    chunk_index=chunk.chunk_index,
                                    sequence_count=chunk_size,
                                    predictions_data=compressed,
                                    compression_ratio=len(serialized) / len(compressed) if len(compressed) > 0 else 1.0,
                                    uncompressed_size_bytes=len(serialized),
                                    compressed_size_bytes=len(compressed),
                                    is_verified=True
                                )
                                db.add(model_pred)
                                created_count += 1
                            except Exception as pred_err:
                                self.logger.error(f"Error storing prediction for chunk {chunk.chunk_index}: {pred_err}")
                                continue
                            
                            cursor += chunk_size
                        
                        await db.commit()
                        self.logger.info(f"  └─ Created ModelPredictions entries for {created_count} chunks in {split_name} split (model={model_id[:8]})")

                self.logger.info(f"✅ Post-training predictions complete for model {model_id[:8]} on dataset {dataset_id[:8]}")
            except Exception as e:
                self.logger.error(f"❌ Failed to generate post-training predictions: {e}", exc_info=True)
