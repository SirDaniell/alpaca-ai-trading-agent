from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import asyncio
import os
import sys
import platform
import math
import uuid
import hashlib
import numpy as np
import pandas as pd
import tensorflow as tf
from sqlalchemy import select
from app.core.processing.processing_manager import ProcessingManager, AnalysisType, IntermediateResultsCache
from app.core.config import TechnicalConfig, MLDatasetConfig, ModelBuildConfig, ModelTrainingConfig
from app.core.processing.progress_reporter import ProgressReporter, ThrottlingStrategy
from app.api.routes.data.database import AsyncPostgresSessionLocal
from app.core.data.session_data_loader import set_as_current_data, store_session_step_result
from app.core.ml.model_registry import ModelRegistry, get_registry
from app.core.ml.persistent_model_store import PersistentModelStore
from app.database.models import CompiledModel, TrainingRecord, ModelSelectionHint, TrainedModelForAnalysis, MLDatasetChunk, ModelPredictions, SessionStepResult
from app.core.analysis.analysis_manager_utils import ErrorCategory, ErrorContext
from app.core.ml.output_spec import is_multi_output_model
from datetime import datetime
logger = logging.getLogger(__name__)




class MLPipelineMixin:
    """Mixin: ML preparation, model build, model training infrastructure."""

    async def _collect_step_configs(
        self,
        session_id: str,
        ml_prep_config: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        mutation_steps = {
            "technical_analysis",
            "currency_indices",
            "astronomical_analysis",
            "snr_analysis",
            "ml_dataset_preparation",
        }
        step_configs: Dict[str, Any] = {}

        try:
            async with AsyncPostgresSessionLocal() as db:
                stmt = select(SessionStepResult).where(
                    SessionStepResult.session_id == session_id,
                    SessionStepResult.step_name.in_([f"{step}_config" for step in mutation_steps]),
                )
                result = await db.execute(stmt)
                rows = result.scalars().all()
        except Exception as error:
            logger.warning("[ML Preparation] Could not collect stored step configs: %s", error)
            rows = []

        for row in rows:
            step_name = str(row.step_name).removesuffix("_config")
            payload = None
            try:
                if row.is_using_jsonb and row.result_data_v2 is not None:
                    payload = row.result_data_v2
                elif row.result_data:
                    payload = deserialize_data(row.result_data, row.is_compressed)
            except Exception as error:
                logger.warning("[ML Preparation] Could not deserialize config snapshot %s: %s", row.step_name, error)
                continue

            if isinstance(payload, dict):
                config_payload = payload.get("config", payload.get("configSnapshot", payload))
                if isinstance(config_payload, dict) and config_payload:
                    step_configs[step_name] = config_payload

        if ml_prep_config:
            step_configs["ml_dataset_preparation"] = ml_prep_config

        logger.info(
            "[ML Preparation] Collected step config contract keys: %s",
            sorted(step_configs.keys()),
        )
        return step_configs

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
        
        # Validate split ratios are non-negative
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
            # STEP 0: Validate config before anything else
            # config is in pm.config (passed as dict or object)
            config_to_validate = pm.config if isinstance(pm.config, dict) else pm.config.__dict__ if hasattr(pm.config, "__dict__") else {}
            
            # LOGGING INCOMING FRONTEND PAYLOAD
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

            # CRITICAL dataset_name is passed as kwarg to pm.execute(), not stored in pm.config
            # Extract from pm.config if DatasetConfig object, otherwise use task_id-based naming
            source_type = getattr(pm.config, "source_type", "enriched_df") if isinstance(pm.config, dict) == False else "enriched_df"
            
            # Extract dataset_name from pm.config (passed from frontend)
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
            ml_prep_config_snapshot = self._normalise_step_config(config_to_validate) if hasattr(self, "_normalise_step_config") else dict(config_to_validate)
            step_configs = await self._collect_step_configs(
                session_id=session_id,
                ml_prep_config=ml_prep_config_snapshot,
            )
            
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
            
            # Send initial progress via task_store BEFORE starting processing
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
            
            # FIX 1: WRAP PM.execute() WITH ERROR HANDLING + CLASSIFICATION
            # This ensures PM errors (config validation, scaler fitting, etc.) are properly
            # classified and routed to frontend instead of bypassing to generic 500 error
            try:
                ml_splits = await pm.execute(
                    current_data, 
                    user_id=user_id,
                    injected_dataset=injected_dataset,
                    dataset_name=dataset_name,
                    step_configs=step_configs,
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
            
            # Extract targets from splits for separate storage
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
            
            # Cleanup unwanted variables at function end
            self.cleanup_function_locals("execute_ml_preparation")
            
            # REMOVED: Duplicate storage to SessionStepResult
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
            # Include target_names at top level for frontend target selection UI
            return {
                "status": "success",
                "task_id": task_id,
                "_ref": task_id,
                "dataset_name": dataset_name,
                "timestamp": datetime.utcnow().isoformat(),
                # Top-level fields for frontend
                "target_names": ml_splits.get("target_names", []),  # All targets including advanced
                "feature_names": ml_splits.get("feature_names", []),
                "sequence_length": ml_splits.get("sequence_length", 0),
                "prediction_length": ml_splits.get("prediction_length", 0),
                "split_counts": ml_splits.get("split_counts", {}),
                "split_dataset_ids": ml_splits.get("split_dataset_ids", {}),
                # Nested metadata for backward compatibility
                "metadata": ml_splits.get("metadata", {}),
                "config": ml_splits.get("config", {}),
                "step_configs": ml_splits.get("step_configs", step_configs),
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

    async def _load_compiled_model_from_db(
        self,
        model_id: str,
    ) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        """
        Load compiled (or trained) model binary and config from CompiledModel table.

        Returns:
            (model, config_dict) on success, (None, None) on failure.
        """
        try:
            import io
            import pickle

            async with AsyncPostgresSessionLocal() as db:
                stmt = select(CompiledModel).where(
                    CompiledModel.compiled_model_id == model_id
                )
                result = await db.execute(stmt)
                record = result.scalars().first()

            if not record:
                self.logger.warning(f"[DB Loader] No CompiledModel found for {model_id}")
                return None, None

            model_binary = record.model_binary
            if not model_binary:
                self.logger.warning(f"[DB Loader] CompiledModel {model_id} has empty binary")
                return None, None

            # Deserialise — try H5 first, fall back to pickle
            model_buffer = io.BytesIO(model_binary)
            try:
                model = tf.keras.models.load_model(model_buffer)
                self.logger.info(f"✅ [DB Loader] Loaded model {model_id} from DB (H5)")
            except Exception:
                model = pickle.loads(model_binary)
                self.logger.info(f"✅ [DB Loader] Loaded model {model_id} from DB (pickle)")

            # Reconstruct config dict from stored columns so training has everything it needs
            config = {
                "type":              record.model_type,
                "input_shape":       record.input_shape,
                "n_predictions":     record.n_predictions,
                "prediction_length": record.prediction_length,
                "optimizer":         (record.optimizer_config or {}).get("type", "adam"),
                "optimizer_config":  record.optimizer_config or {},
                "loss":              record.loss_function,
                "metrics":           record.metrics_list or ["mae", "mse"],
                "feature_cols":      record.feature_columns or [],
                "target_cols":       record.target_columns or [],
                "selected_targets":  record.selected_targets or [],
                "feature_hash":      record.feature_hash,
                "dataset_id":        str(record.dataset_id) if record.dataset_id else None,
                "ml_preparation_ref": record.ml_dataset_ref,
                "model_name":        record.model_name,
                "version":           record.version,
                "framework":         record.framework,
                "framework_version": record.framework_version,
                "architecture_json": record.architecture_json or {},
            }

            return model, config

        except Exception as e:
            self.logger.error(f"[DB Loader] Failed to load model {model_id}: {e}", exc_info=True)
            return None, None

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
                
                # Phase 20: Store scaler, feature names, and sequence length
                trained_model.scaler_binary = training_metadata.get("scaler_binary")
                trained_model.feature_names = training_metadata.get("training_data_feature_columns")
                trained_model.sequence_length = training_metadata.get("sequence_length")

                # Phase 28: Serving contract snapshot for inference parity
                trained_model.scaler_config = training_metadata.get("scaler_config")
                trained_model.step_configs = training_metadata.get("step_configs")
                trained_model.output_transform_spec = training_metadata.get("output_transform_spec")
                trained_model.serving_contract = training_metadata.get("serving_contract")
                
                # Phase 20: Infer metric_type from target columns
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
            
            # Validate input_shape and n_predictions against actual dataset before building model
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
                                # ⚠️ CRITICAL Only use sample if we haven't found n_predictions via TIER 1
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
            
            # Get model builder from registry (use singleton to avoid re-running discovery on every call)
            model_registry = get_registry()
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
                # Explicitly instantiate Adam with clipnorm to prevent exploding gradients
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
        ml_preparation_ref: Union[str, Dict[str, Any]] = None,  # UPDATED: Supports object
        user_id: str = "anonymous",
        is_classification: bool = False, # NEW: Flag for target type
        target_column: str = None,       # NEW: Target column name
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
            # ROBUST EXTRACTION: Handle object or string for ml_preparation_ref
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
            _db_model_config = None  # will be populated on TIER 0c miss
            if hasattr(self, 'model') and self.model is not None and getattr(self, 'model_id', None) == model_id:
                model = self.model
                self.logger.info(f"🚀 [TIER 0c] Using in-memory model {model_id} (Zero Disk I/O)")
            else:
                self.logger.info(f"🔍 [TIER 3] Loading model {model_id} from DB (CompiledModel table)...")
                model, _db_model_config = await self._load_compiled_model_from_db(model_id)
            
            if model is None:
                return {"status": "error", "message": f"Model not found: {model_id}"}
            
            # Only add missing MAE/MSE metrics — do NOT fully recompile.
            # Full recompile resets Adam momentum/state, which hurts convergence when
            # reusing the in-memory TIER 0c model or resuming from disk.
            try:
                # Carefully filter metrics. Keras 3 throws ValueError if 'loss' 
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
            
            # NEW: Pass ml_dataset_name to validate TIER 0b pointers
            # Load ML data (TIER 0b) - 0ms latency
            train_result, _ = await self._load_data_4_tier(
                session_id=session_id,
                task_id=task_id,
                data_type="ml_train",   
                ml_dataset_name=ml_dataset_name,  # Use extracted name
                prefer_lazy=True                     # OPTIMIZATION: Stream from disk
            )
            val_result, _ = await self._load_data_4_tier(
                session_id=session_id,
                task_id=task_id,
                data_type="ml_validation",
                ml_dataset_name=ml_dataset_name,  # Use extracted name
                prefer_lazy=True                     # OPTIMIZATION: Stream from disk
            )
            test_result, _ = await self._load_data_4_tier(
                session_id=session_id,
                task_id=task_id,
                data_type="ml_test",
                ml_dataset_name=ml_dataset_name,  # Use extracted name
                prefer_lazy=True                     # OPTIMIZATION: Stream from disk
            )
            
            if train_result is None or val_result is None:
                return {"status": "error", "message": "ML training data not available"}
            
            # Extract sequences + targets from result
            if isinstance(train_result, dict):
                if "sequences" in train_result:
                    train_data = train_result["sequences"]
                    # Extract actual array from targets dict if present
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
                    # Extract actual array from targets dict if present
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
            
            # TARGET EXTRACTION FROM DATAFRAMES
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
            
            # Load model config — use the one fetched alongside the model on TIER 0c miss,
            # or fetch fresh from DB if we came via the TIER 0c hit path.
            if _db_model_config is not None:
                model_config = _db_model_config
                self.logger.info(f"✅ Using model config fetched from DB alongside model binary")
            else:
                self.logger.info(f"🔍 Fetching model config from DB (CompiledModel table) for {model_id}")
                _, model_config = await self._load_compiled_model_from_db(model_id)
            if model_config is None:
                return {"status": "error", "message": f"Model config not found in DB for {model_id}"}
            
            # Fail early if critical shape metadata is missing
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
                # Only enable if train_targets is genuinely None (not "LAZY_ON_DISK")
                auto_mode = (train_targets is None and not target_column and not is_classification)
                
                batch_generator = LazySequenceGenerator(
                    file_paths=train_paths,
                    batch_size=batch_size,
                    shuffle=True,  
                    autoencoder_mode=auto_mode,
                    target_column=target_column,
                    selected_targets=selected_targets,
                    micro_val_holdback=0.05  # Hold back 5% for batch-level audit
                )

                # Guard against None val_data before creating val_generator
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
            
            # GUARD: Validate training data before starting loop
            if train_data is None or (hasattr(train_data, '__len__') and len(train_data) == 0):
                return {"status": "error", "message": "Training data is empty after extraction"}
            if val_data is None or (hasattr(val_data, '__len__') and len(val_data) == 0):
                return {"status": "error", "message": "Validation data is empty after extraction"}
            
            # Use guarded counts for logging to prevent crashes with non-len objects
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
            ml_dataset_scaling_config = None
            ml_dataset_split_config = None
            ml_dataset_step_configs = None
            
            if ml_preparation_ref:
                try:
                    async with AsyncPostgresSessionLocal() as db:
                        from sqlalchemy import select as sa_select
                        from app.database.models import MLDataset, DataSession
                        
                        # 0. Fetch DataSession for authoritative symbol/timeframe provenance
                        session_stmt = sa_select(DataSession).where(DataSession.session_id == session_id)
                        session_result = await db.execute(session_stmt)
                        data_session = session_result.scalar_one_or_none()
                        if data_session:
                            training_data_symbol = data_session.symbol
                            training_data_timeframe = data_session.timeframe
                            try:
                                if data_session.start_date:
                                    training_data_start_date = datetime.fromisoformat(
                                        str(data_session.start_date).replace("Z", "+00:00")
                                    )
                                if data_session.end_date:
                                    training_data_end_date = datetime.fromisoformat(
                                        str(data_session.end_date).replace("Z", "+00:00")
                                    )
                            except Exception:
                                pass
                        
                        # 1. Fetch MLDataset to capture metadata and resolve dataset_id
                        stmt = sa_select(MLDataset).where(
                            MLDataset.session_id == session_id,
                            MLDataset.dataset_name == ml_dataset_name  # Use extracted name, not raw ref
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
                            ml_dataset_scaling_config = ml_dataset.scaling_config or {}
                            ml_dataset_split_config = ml_dataset.split_config or {}
                            
                            # Extract split config
                            split_cfg = ml_dataset.split_config or {}
                            sequence_length = split_cfg.get("sequence_length") or split_cfg.get("window_size")
                            
                            # Extract symbol/timeframe — prefer source_metadata, fall back to DataSession
                            src_meta = ml_dataset.source_metadata or {}
                            if isinstance(src_meta, str):
                                try:
                                    import json as _json
                                    src_meta = _json.loads(src_meta)
                                except Exception:
                                    src_meta = {}
                            ml_dataset_step_configs = src_meta.get("step_configs") or {}
                            if not training_data_symbol:
                                training_data_symbol = src_meta.get("symbol") or src_meta.get("trading_symbol")
                            if not training_data_timeframe:
                                training_data_timeframe = src_meta.get("timeframe") or src_meta.get("trading_timeframe")
                            
                            # Extract date range from source_metadata or split_config
                            if not training_data_start_date and src_meta.get("start_date"):
                                try:
                                    training_data_start_date = datetime.fromisoformat(str(src_meta["start_date"]).replace("Z", "+00:00"))
                                except Exception:
                                    pass
                            if not training_data_end_date and src_meta.get("end_date"):
                                try:
                                    training_data_end_date = datetime.fromisoformat(str(src_meta["end_date"]).replace("Z", "+00:00"))
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
            # Initialize to inf so Catch-Up Mode (final_val_loss > 0.005) fires
            # correctly from epoch 1, not just from the epoch<3 branch.
            final_val_loss = float('inf')
            final_training_loss = 0.0
            final_mae = 0.0
            final_mse = 0.0
            final_val_mae = 0.0
            final_val_mse = 0.0
            best_epoch = None
            best_core_val_loss = float('inf')         # Bug 2 fix: renamed from best_val_loss; holds CORE val loss only
            best_weights = model.get_weights()  # Initial snapshot
            best_metrics = {"loss": float('inf'), "val_loss": float('inf'), "core_val_loss": float('inf'), "mae": 0.0, "val_mae": 0.0, "mse": 0.0, "val_mse": 0.0}
            retries_per_epoch = 2               # Max retries if val_loss regresses
            # lr_decay_factor removed — was dead code; actual decay uses dyn_decay from phase block
            plateau_count = 0
            plateau_threshold = 0.0001
            improvement_streak = 0              # Track consecutive improvements for LR recovery
            initial_lr = float(tf.keras.backend.get_value(model.optimizer.learning_rate))  # Safe LR read
            
            # Accumulate per-epoch history for frontend metrics dashboard
            training_history = {
                "loss": [], "val_loss": [],
                "mae": [], "val_mae": [],
                "mse": [], "val_mse": [],
            }
            
            # UPGRADE: Persistent Continual Learning State (Persists across epochs)
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
            
            # Initialize unified ProgressReporter for WebSockets
            reporter = ProgressReporter(
                task_id=task_id,  # Pass task_id to constructor
                task_store=self.task_store,
                connection_manager=self.connection_manager,  # Pass connection_manager
                user_id=user_id,
                throttling_strategy=ThrottlingStrategy.HYBRID
            )
            
            # Track background tasks to await them at the end
            _background_tasks = []
            
            # Training loop with advanced rollback and LR decay (Ref Code implementation)
            epoch = 0
            while epoch < epochs:
                try:
                    epoch_start_time = datetime.utcnow()
                    
                    # Snapshot weights BEFORE this epoch trains for potential rollback
                    weights_before = model.get_weights()
                    
                    # DYNAMIC EPOCH-AWARE STRATEGY (Funnel Effect)
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
                        continual_state=continual_state  # UPGRADE: Persistent state
                    )
                    
                    # Extract metrics for the rest of the loop logic
                    final_training_loss = epoch_loss_dict.get("loss", float('inf'))
                    final_val_loss = epoch_loss_dict.get("val_loss", float('inf'))
                    # NEW: core loss (drives checkpoint / rollback decisions)
                    final_core_val_loss = epoch_loss_dict.get("core_val_loss", final_val_loss)
                    final_core_loss = epoch_loss_dict.get("core_loss", final_training_loss)
                    final_mae = epoch_loss_dict.get("mae", 0.0)
                    final_mse = epoch_loss_dict.get("mse", 0.0)
                    final_val_mae = epoch_loss_dict.get("val_mae", 0.0)
                    final_val_mse = epoch_loss_dict.get("val_mse", 0.0)
                    epoch_jury_rejected = epoch_loss_dict.get("jury_rejected", False)

                    e_loss, e_val_loss = final_training_loss, final_val_loss
                    e_core_val_loss = final_core_val_loss
                    e_mae, e_mse = final_mae, final_mse
                    e_val_mae, e_val_mse = final_val_mae, final_val_mse

                    
                    epoch_duration_seconds = (datetime.utcnow() - epoch_start_time).total_seconds()
                    # NOTE: epochs_completed is incremented ONLY after the epoch is committed below
                    # to prevent double-counting on rollback/retry
                    
                    # Progress should reach 100% on final epoch
                    progress = int(((epoch + 1) / epochs) * 100)
                    
                    # TRACK BEST PERFORMANCE AND HANDLE REGRESSION
                    # ── Checkpoint selection uses CORE val loss ──────────────────────
                    # core_val_loss = weighted loss over tradeable-prediction heads only.
                    # full val_loss (e_val_loss) keeps flowing into history / DB / reporter.
                    if not epoch_jury_rejected and e_core_val_loss < best_core_val_loss:
                        # Improved: commit and save weights
                        self.logger.info(
                            f"✅ Epoch {epoch+1}: core_val_loss improved ({e_core_val_loss:.6f} < {best_core_val_loss:.6f}) "
                            f"[full_val_loss={e_val_loss:.6f}] "
                            f"[Phase: {'Early' if epoch_progress < 0.2 else 'Late' if epoch_progress > 0.7 else 'Mid'}]"
                        )
                        best_core_val_loss = e_core_val_loss  # stored as core
                        best_epoch = epoch + 1
                        best_weights = model.get_weights()
                        best_metrics = {
                            "loss": e_loss, "val_loss": e_val_loss,
                            "core_val_loss": e_core_val_loss,
                            "mae": e_mae, "val_mae": e_val_mae, "mse": e_mse, "val_mse": e_val_mse,
                        }
                        retries_per_epoch = 2  # Reset budget
                        plateau_count = 0
                        improvement_streak += 1
                    else:
                        # Regression or Jury Rejection: Consider rollback and retry
                        # ── Regression check also uses core val loss ──────────────
                        is_regression = e_core_val_loss > best_core_val_loss * dyn_threshold
                        
                        if retries_per_epoch > 0 and (is_regression or epoch_jury_rejected):
                            reason = "Jury rejected" if epoch_jury_rejected else (
                                f"Regression (core: {e_core_val_loss:.6f} > {best_core_val_loss:.6f} * {dyn_threshold:.2f})"
                                f" [full_val_loss={e_val_loss:.6f}]"
                            )
                            
                            if is_multi_output_model(model):
                                self.logger.warning(
                                    f"⚖️ Epoch {epoch+1}: {reason}. Skipping global rollback for multi-head model to protect good branches..."
                                )
                            else:
                                self.logger.warning(
                                    f"⚖️ Epoch {epoch+1}: {reason}. Reverting to BEST weights and retrying..."
                                )
                                model.set_weights(best_weights)
                            
                            # Decay Learning Rate using Dynamic Decay (Skip punishment in Phase 1)
                            if epoch_progress >= 0.2:
                                try:
                                    current_lr = float(tf.keras.backend.get_value(model.optimizer.learning_rate))  # Safe LR read
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
                                if is_multi_output_model(model):
                                    self.logger.error(f"🔴 Epoch {epoch+1}: JURY REJECTED but out of retries. Skipping global rollback for multi-head model. Continuing to next epoch...")
                                else:
                                    self.logger.error(f"🔴 Epoch {epoch+1}: JURY REJECTED but out of retries. FORCING ROLLBACK Continuing to next epoch...")
                                    model.set_weights(best_weights)
                            
                            self.logger.info(f"ℹ️ Epoch {epoch+1}: regression accepted (within {dyn_threshold:.2f}x limit or no retries left)")
                            improvement_streak = 0  # Reset streak on regression
                            
                            # Plateau tracking (Fixed: catch stalled convergence near best)
                            # Bug 2 fix: compare core-vs-core, not core-minus-full
                            improvement = best_core_val_loss - e_core_val_loss  # positive = better, negative = regression
                            if improvement >= 0 and improvement < plateau_threshold:
                                plateau_count += 1
                                if plateau_count >= 5:
                                    if is_multi_output_model(model):
                                        self.logger.warning(f"🔄 Plateau detected for {plateau_count} epochs. Skipping global weight restore for multi-head model. Decaying LR.")
                                    else:
                                        self.logger.warning(f"🔄 Plateau detected for {plateau_count} epochs. Restoring best weights and decaying LR.")
                                        model.set_weights(best_weights)
                                    try:
                                        current_lr = float(tf.keras.backend.get_value(model.optimizer.learning_rate))  # Safe LR read
                                        model.optimizer.learning_rate.assign(current_lr * 0.2) # Aggressive decay
                                    except Exception: pass
                                    plateau_count = 0
                            else:
                                plateau_count = 0  # Reset on genuine regression or strong improvement
                            
                            # LR Recovery: Aggressive recovery after 2 consecutive improvements
                            # ⚠️ CRITICAL: Prevent loss explosion by restoring LR when it gets too small
                            min_lr = initial_lr * 0.001  # Minimum LR floor (never go below this)
                            current_lr = float(tf.keras.backend.get_value(model.optimizer.learning_rate))  # Safe LR read
                            
                            if improvement_streak >= 2:
                                try:
                                    if current_lr < min_lr:
                                        # LR is DEAD - restore to minimum
                                        new_lr = min_lr
                                        self.logger.critical(f"🚨 LR DEAD ZONE DETECTED: {current_lr:.2e} < {min_lr:.2e}. Restoring to minimum.")
                                    else:
                                        # Remove restrictive cap in Phase 2
                                        # Previous bug: lr_cap = 1x initial in Phase 2/3 killed aggressive learning
                                        # (couldn't push past 0.x because LR was capped at initial value)
                                        # New: Phase 1 = 5x, Phase 2 = 3x (allow continued growth), Phase 3 = 2x
                                        if epoch_progress < 0.2:
                                            lr_cap = initial_lr * 5.0
                                        elif epoch_progress < 0.7:
                                            lr_cap = initial_lr * 3.0  # Was 1x, now 3x for Phase 2
                                        else:
                                            lr_cap = initial_lr * 2.0  # Was 1x, now 2x for Phase 3
                                        
                                        # Normal recovery: Triple LR for faster catch-up (not just double)
                                        # Use 3x recovery to match aggressive decay in Phase 2/3
                                        recovery_factor = 3.0 if epoch_progress >= 0.2 else 2.0
                                        new_lr = min(current_lr * recovery_factor, lr_cap)
                                        self.logger.info(f"📈 LR AGGRESSIVE RECOVERY: {current_lr:.6f} → {new_lr:.6f} (×{recovery_factor}) after {improvement_streak} improvements [Cap: {lr_cap:.4f}]")
                                    
                                    model.optimizer.learning_rate.assign(new_lr)
                                    improvement_streak = 0  # Reset for next cycle
                                except Exception as e:
                                    self.logger.warning(f"⚠️ LR recovery failed: {e}")
                    
                    # ROLLBACK BUDGET RECOVERY: Reset every 20 epochs to allow continued recovery
                    if epoch > 0 and epoch % 20 == 0 and retries_per_epoch < 2:
                        retries_per_epoch = 2
                        self.logger.info(f"🔄 Epoch {epoch+1}: Rollback budget recovered to 2")
                    
                    # COMMIT EPOCH (with NaN/Inf sanitization)
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
                    
                    # Log if metrics are invalid (NaN/Inf)
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
                    # NOTE: progress was already computed correctly above as ((epoch+1)/epochs)*100
                    # The stale duplicate line `int((epoch / epochs) * 100)` is removed — it was
                    # overwriting the correct value and prevented progress from ever reaching 100%.
                   
                    # Send training metrics in structured format for frontend
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
                    # Track tasks in _background_tasks list so they can be
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
                    # REMOVED: Early stopping logic (running all epochs for full loss curve study)
                    
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
                self.logger.info(f"🏆 Restoring best weights from epoch {best_epoch_label} (core_val_loss: {best_core_val_loss:.6f}, full_val_loss: {best_metrics.get('val_loss', float('inf')):.6f})")
                model.set_weights(best_weights)
                # Sync final_* metric variables to best-model values so the training
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

            # Persist trained weights back to CompiledModel table (replaces disk save)
            try:
                import io as _io
                trained_buffer = _io.BytesIO()
                model.save(trained_buffer, save_format='h5')
                trained_binary = trained_buffer.getvalue()

                async with AsyncPostgresSessionLocal() as db:
                    stmt = select(CompiledModel).where(
                        CompiledModel.compiled_model_id == model_id
                    )
                    result = await db.execute(stmt)
                    compiled_record = result.scalars().first()
                    if compiled_record:
                        compiled_record.model_binary = trained_binary
                        compiled_record.model_size_bytes = len(trained_binary)
                        compiled_record.status = "trained"
                        await db.commit()
                        self.logger.info(
                            f"✅ Updated CompiledModel binary in DB with trained weights: {model_id} "
                            f"({len(trained_binary) / 1e6:.1f} MB)"
                        )
                    else:
                        self.logger.warning(
                            f"⚠️ CompiledModel record not found for trained weight update: {model_id}"
                        )
            except Exception as _save_err:
                self.logger.error(
                    f"Failed to update trained model binary in DB: {_save_err}", exc_info=True
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
            
            # Correct total sequences for reporting (samples, not batches)
            final_total_sequences = (
                (len(batch_generator) * batch_size) if use_generator 
                else (len(train_data) if hasattr(train_data, '__len__') else 0)
            )

            # Build complete training result with history
            training_result = {
                "status": "success",
                "model_id": model_id,
                "epochs_completed": epochs_completed,
                "best_epoch": best_epoch,
                "final_val_loss": float(final_val_loss),
                # Bug 2 fix: expose both core and full val loss for the best epoch
                # ("show more, not less" — don't collapse to one ambiguous number)
                "best_core_val_loss": float(best_core_val_loss),   # drove checkpoint selection
                "best_full_val_loss": float(best_metrics.get("val_loss", float('inf'))),  # full-agg at same epoch
                "final_training_loss": float(final_training_loss),
                "final_mae": float(final_mae),
                "final_mse": float(final_mse),
                "final_val_mae": float(final_val_mae),
                "final_val_mse": float(final_val_mse),
                "final_training_metrics": training_history,  # Per-epoch arrays for metrics dashboard
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
            
            # Store training result to DB for frontend fetchResults()
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
            # SEND COMPLETION MESSAGE WITH FINAL METRICS TO FRONTEND
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
                        # Send final metrics in completion for instant frontend display
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
            serving_contract_snapshot = None
            output_transform_spec_snapshot = None
            if ml_dataset_scaling_config and training_data_feature_columns:
                try:
                    from app.core.ml.serving_contract import (
                        build_session_provenance,
                        build_serving_contract,
                    )
                    provenance = build_session_provenance(
                        {
                            "session_id": session_id,
                            "symbol": training_data_symbol,
                            "timeframe": training_data_timeframe,
                            "start_date": training_data_start_date.isoformat() if training_data_start_date else None,
                            "end_date": training_data_end_date.isoformat() if training_data_end_date else None,
                        },
                        dataset_name=ml_dataset_name or "",
                        dataset_id=str(dataset_id) if dataset_id else "",
                        step_configs=ml_dataset_step_configs or {},
                    )
                    serving_contract_snapshot = build_serving_contract(
                        model_id=str(model_id),
                        provenance=provenance,
                        scaling_config=ml_dataset_scaling_config,
                        split_config=ml_dataset_split_config or {},
                        feature_columns=list(training_data_feature_columns or []),
                        target_names=list(training_data_target_columns or []),
                        step_configs=ml_dataset_step_configs or {},
                    )
                    output_transform_spec_snapshot = (
                        serving_contract_snapshot.get("output_contract", {}).get("output_transform_spec")
                        or ml_dataset_scaling_config.get("output_transform_spec")
                    )
                except Exception as contract_err:
                    self.logger.warning("[Training] Could not build serving contract snapshot: %s", contract_err)

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
                "scaler_config": ml_dataset_scaling_config,
                "step_configs": ml_dataset_step_configs,
                "output_transform_spec": output_transform_spec_snapshot,
                "serving_contract": serving_contract_snapshot,
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
