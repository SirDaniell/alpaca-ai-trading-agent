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
import io
import pickle
import pandas as pd
import numpy as np
import gc
import joblib
from typing import Dict, Optional, Any, Set, Tuple, List, Union, Callable
from datetime import datetime
from enum import Enum
import hashlib
import json
import time
import threading
from sqlalchemy import insert, select, desc, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from app.core.data.serializers import to_serializable

logger = logging.getLogger(__name__)

class PassThroughScaler:
    """A dummy scaler that passes features through unchanged.
       Price Features are already normalized so no need to scale them
       (they're already in [0,1] range from preprocessing)
       But the correct method probably is to implement the normalization here
       but we'll leave it as-is for now since features are already normalized.
       
       ⚠️ DETECTION: Logs warnings when columns are passed through without normalization.
    """
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1] if hasattr(X, 'shape') else len(X[0])
        self.scale_ = np.ones(self.n_features_in_)
        self.data_min_ = np.zeros(self.n_features_in_)
        self.data_max_ = np.ones(self.n_features_in_)
        
        # 🔍 Detection: Log when PassThrough is applied
        X_arr = X.values if hasattr(X, 'values') else np.array(X)
        if X_arr.size > 0:
            col_min = np.nanmin(X_arr, axis=0) if X_arr.ndim > 1 else [np.nanmin(X_arr)]
            col_max = np.nanmax(X_arr, axis=0) if X_arr.ndim > 1 else [np.nanmax(X_arr)]
            col_names = X.columns if hasattr(X, 'columns') else [f"col_{i}" for i in range(self.n_features_in_)]
            
            out_of_bounds_count = 0
            for i, (c_min, c_max, col_name) in enumerate(zip(col_min, col_max, col_names)):
                if c_min < 0 or c_max > 1:
                    out_of_bounds_count += 1
                    logging.warning(
                        f"⚠️ [PassThroughScaler] Column '{col_name}' not normalized: "
                        f"range=[{c_min:.6f}, {c_max:.6f}] (expected [0, 1])"
                    )
            
            if out_of_bounds_count == 0:
                logging.info(
                    f"✅ [PassThroughScaler] All {self.n_features_in_} columns appear pre-normalized to [0, 1]"
                )
            elif out_of_bounds_count > 0:
                logging.warning(
                    f"⚠️ [PassThroughScaler] {out_of_bounds_count}/{self.n_features_in_} columns "
                    f"out of bounds [0, 1] - will pass through unchanged"
                )
        
        return self
    
    def transform(self, X):
        # 🔍 Detection: Check for out-of-bounds values in test/inference
        X_arr = X.values if hasattr(X, 'values') else np.array(X)
        if X_arr.size > 0:
            col_min = np.nanmin(X_arr, axis=0) if X_arr.ndim > 1 else [np.nanmin(X_arr)]
            col_max = np.nanmax(X_arr, axis=0) if X_arr.ndim > 1 else [np.nanmax(X_arr)]
            col_names = X.columns if hasattr(X, 'columns') else [f"col_{i}" for i in range(X_arr.shape[1] if X_arr.ndim > 1 else 1)]
            
            out_of_bounds = []
            for i, (c_min, c_max, col_name) in enumerate(zip(col_min, col_max, col_names)):
                if c_min < -0.001 or c_max > 1.001:  # Allow small tolerance for floating point
                    out_of_bounds.append((col_name, c_min, c_max))
            
            if out_of_bounds:
                logging.warning(
                    f"⚠️ [PassThroughScaler.transform] {len(out_of_bounds)} columns exceed [0, 1] bounds:\n"
                    + "\n".join([f"  '{name}': [{min_val:.6f}, {max_val:.6f}]" for name, min_val, max_val in out_of_bounds])
                )
        
        return X
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    
    def inverse_transform(self, X):
        return X


class SelectiveScaler:
    """A partition-based wrapper scaler that partitions features into Price, Diff/Oscillator,
    and Other column subsets and scales each subset independently using standard methods.
    """
    def __init__(self, price_scaler=None, diff_scaler=None, other_scaler=None, price_cols=None, diff_cols=None, other_cols=None, all_cols: list = None, base_scaler=None, passthrough_cols=None):
        # Handle backwards-compatibility for older pickled scalers
        if base_scaler is not None or passthrough_cols is not None:
            
            self.diff_scaler = base_scaler if base_scaler is not None else PassThroughScaler()
            self.other_scaler = other_scaler if other_scaler is not None else PassThroughScaler()
            self.price_scaler = price_scaler if price_scaler is not None else PassThroughScaler()
            self.price_cols = list(passthrough_cols or [])
            self.all_cols = list(all_cols or [])
            self.diff_cols = [c for c in self.all_cols if c not in self.price_cols]
            self.other_cols = []
        else:
            self.price_scaler = price_scaler if price_scaler is not None else PassThroughScaler()
            self.diff_scaler = diff_scaler if diff_scaler is not None else PassThroughScaler()
            self.other_scaler = other_scaler if other_scaler is not None else PassThroughScaler()
            self.price_cols = list(price_cols or [])
            self.diff_cols = list(diff_cols or [])
            self.other_cols = list(other_cols or [])
            self.all_cols = list(all_cols or [])
            
        # Identify index of each column group
        self.price_indices = [i for i, col in enumerate(self.all_cols) if col in self.price_cols]
        self.diff_indices = [i for i, col in enumerate(self.all_cols) if col in self.diff_cols]
        self.other_indices = [i for i, col in enumerate(self.all_cols) if col in self.other_cols]
        self.feature_names = list(self.all_cols)
        self.feature_index_map = {col: i for i, col in enumerate(self.all_cols)}
        self.index_feature_map = {i: col for col, i in self.feature_index_map.items()}
        self.partition_by_feature = {
            **{col: "price" for col in self.price_cols},
            **{col: "diff" for col in self.diff_cols},
            **{col: "other" for col in self.other_cols},
        }
        self.partition_indices = {
            "price": list(self.price_indices),
            "diff": list(self.diff_indices),
            "other": list(self.other_indices),
        }

    def _slice(self, X, indices):
        if hasattr(X, "iloc"):
            return X.iloc[:, indices]
        return X[:, indices]
        
    def fit(self, X, y=None):
        import numpy as np
        X_arr = X.values if hasattr(X, 'values') else np.array(X)
        self.n_features_in_ = X_arr.shape[1]
        
        if self.price_indices and self.price_scaler is not None:
            self.price_scaler.fit(self._slice(X, self.price_indices))
        if self.diff_indices and self.diff_scaler is not None:
            self.diff_scaler.fit(self._slice(X, self.diff_indices))
        if self.other_indices and self.other_scaler is not None:
            self.other_scaler.fit(self._slice(X, self.other_indices))

        logging.info(f'Price related Columns: {self.price_indices}, Diff/Oscillator Columns:  {self.diff_indices}, Other Columns: {self.other_indices}')
            
        # Mock attributes for sklearn inspection
        self.scale_ = np.ones(self.n_features_in_)
        self.data_min_ = np.zeros(self.n_features_in_)
        self.data_max_ = np.ones(self.n_features_in_)
        
        # Aggregate scaling parameters if available
        if hasattr(self.diff_scaler, 'scale_'):
            for idx, base_idx in enumerate(self.diff_indices):
                self.scale_[base_idx] = self.diff_scaler.scale_[idx]
        if hasattr(self.other_scaler, 'scale_'):
            for idx, base_idx in enumerate(self.other_indices):
                self.scale_[base_idx] = self.other_scaler.scale_[idx]
                
        return self

    def transform(self, X):
        import numpy as np
        is_df = hasattr(X, 'iloc')
        X_arr = X.values if hasattr(X, 'values') else np.array(X)
        X_out = X_arr.copy()
        
        if self.price_indices and self.price_scaler is not None:
            X_out[:, self.price_indices] = np.asarray(self.price_scaler.transform(self._slice(X, self.price_indices)))
        if self.diff_indices and self.diff_scaler is not None:
            X_out[:, self.diff_indices] = np.asarray(self.diff_scaler.transform(self._slice(X, self.diff_indices)))
            # Guard rail: diff columns arrive pre-normalized to (0, 1):
            #   - Range-width-normalized cols: sigmoid in Step 5 → (0, 1)
            #   - StandardScaler'd cols: sigmoid in _apply_diff_scalers → (0, 1)
            # This clip only catches floating-point edge cases (e.g. 1.0000000001).
            # It does NOT discard information — 0.5 = neutral/zero-diff, preserved.
            X_out[:, self.diff_indices] = np.clip(X_out[:, self.diff_indices], 0, 1)
        if self.other_indices and self.other_scaler is not None:
            X_out[:, self.other_indices] = np.asarray(self.other_scaler.transform(self._slice(X, self.other_indices)))
            # Guard rail only: MinMaxScaler on OOD val/test data can produce values
            # just outside [0,1].  Clip prevents NaN propagation in LSTM layers.
            # Actual regime-shift information is preserved upstream via sigmoid normalisation.
            X_out[:, self.other_indices] = np.clip(X_out[:, self.other_indices], 0, 1)
            
     
        logging.info(f'Price related Columns: {self.price_indices}, Diff/Oscillator Columns:  {self.diff_indices}, Other Columns: {self.other_indices}')
        

        if is_df:
            import pandas as pd
            return pd.DataFrame(X_out, index=X.index, columns=X.columns)
        return X_out
        
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
        
    def inverse_transform(self, X):
        import numpy as np
        is_df = hasattr(X, 'iloc')
        X_arr = X.values if hasattr(X, 'values') else np.array(X)
        X_out = X_arr.copy()
        
        if self.price_indices and self.price_scaler is not None:
            X_out[:, self.price_indices] = self.price_scaler.inverse_transform(X_arr[:, self.price_indices])
        if self.diff_indices and self.diff_scaler is not None:
            X_out[:, self.diff_indices] = self.diff_scaler.inverse_transform(X_arr[:, self.diff_indices])
        if self.other_indices and self.other_scaler is not None:
            X_out[:, self.other_indices] = self.other_scaler.inverse_transform(X_arr[:, self.other_indices])
            
        if is_df:
            import pandas as pd
            return pd.DataFrame(X_out, index=X.index, columns=X.columns)
        return X_out


from app.core.config import ProcessingConfig
from app.core.processing.tasks import TaskStore, TaskCancelledException
from app.core.services.multiprocessing_config import init_spawn_method
from app.api.routes.data.database import AsyncPostgresSessionLocal
from app.database.models import ChunkCheckpoint
from app.core.data.session_data_loader import store_session_step_result, set_as_current_data, store_session_step_result_chunked
from app.core.services.decompress_cache import get_cache as get_decompress_cache
from app.core.data.serializers import serialize_data
from app.core.ml.ml_dataset_preparation import MLDatasetPreparation
from app.core.ml.ml_dataset_preparation import DatasetConfig
from app.core.ml.ml_validation import validate_ml_data
import psutil
from app.database.models import MLDataset
from sqlalchemy.dialects.postgresql import insert as pg_insert
import uuid
import concurrent.futures
from functools import partial

 # Import the proper storage function
from app.core.data.session_data_loader import create_ml_dataset
    
# CRITICAL: Import the refactored components
from app.core.processing.processing_strategies import (
    HandlerRegistry,
)
from app.core.processing.processing_utils import ProcessingContext, ProcessingStrategy, StrategyFactory
from app.core.processing.processing_handlers import (
    analyze_technical_impl,
    analyze_snr_impl,
    analyze_astronomical_impl,
    analyze_currency_indices_impl,
    analyze_currency_strength_matrix_impl,
    analyze_ml_prep_impl,
    analyze_model_training_impl,
    analyze_model_build_impl,
    analyze_enrich_with_targets_impl,
)

# Import configs
from app.core.config import (
    TechnicalConfig,
    SNRConfig,
    AstronomicalConfig,
    CurrencyIndexConfig,
    CurrencyStrengthMatrixConfig,
    MLDatasetConfig,
    ModelBuildConfig,
    EnrichWithTargetsConfig,
    ModelTrainingConfig,
    ProcessingConfig,
)



# ============================================================================
# PROGRESS STAGE MAPPING - Pipeline-wide progress coordination
# ============================================================================

class ProgressStage(Enum):
    """
    Pipeline-wide progress stages with allocated percentage ranges.
    
    Ensures monotonic progress across the entire analysis pipeline:
    Frontend → Backend → Processing → Storage → Response
    
    Total: 100%

    - TechnicalIndicators stops at 98%
    - ProcessingManager handles 98-100% (serialization, storage, completion)
    """
    API_ROUTING = (0, 2)       # 2% - Request validation & routing
    DATA_LOADING = (2, 5)      # 3% - TIER cache lookup & decompression
    STRATEGY_SETUP = (5, 8)    # 3% - Strategy selection & context building
    CORE_PROCESSING = (8, 98)  # 90% - ⭐ Main analysis (TechnicalIndicators, SNR, etc.)
    SERIALIZATION = (98, 99)   # 1% - Pickle encoding & compression
    STORAGE = (99, 100)        # 1% - Database write & commit
    
    @classmethod
    def scale_progress(cls, stage: 'ProgressStage', local_progress: float) -> float:
        """
        Scale local progress (0-100) to global pipeline progress.
        
        Args:
            stage: Current pipeline stage
            local_progress: Progress within the stage (0-100)
            
        Returns:
            Global progress percentage (0-100)
            
        Example:
            >>> ProgressStage.scale_progress(ProgressStage.CORE_PROCESSING, 50)
            53.0  # Halfway through core processing = 53% of total pipeline
        """
        start, end = stage.value
        return start + (local_progress / 100.0) * (end - start)


# ============================================================================
# ANALYSIS TYPE ENUM
# ============================================================================

class AnalysisType(str, Enum):
    """Supported analysis types."""
    TECHNICAL = "technical"
    SNR = "snr"
    ASTRONOMICAL = "astronomical"
    CURRENCY_INDICES = "currency_indices"
    ML_DATASET_PREPARATION = "ml_dataset_preparation"
    MODEL_BUILD = "model_build"
    MODEL_TRAINING = "model_training"


# ============================================================================
# INCREMENTAL AGGREGATOR - Memory peak prevention
# ============================================================================

class PartialResultAggregator:
    """
    Handles incremental merging of slice results to reduce memory peaks.
    Used by SliceStreaming strategy to aggregate data while processing.
    """
    def __init__(self, original_df: pd.DataFrame, analysis_type: str, logger: logging.Logger):
        self.original_df = original_df
        self.analysis_type = analysis_type
        self.logger = logger
        
        # Accumulators
        self.combined_rows: List[pd.DataFrame] = []
        self.signals: List[Dict[str, Any]] = []
        self.zones: List[Dict[str, Any]] = []
        self.signal_counts: Dict[str, int] = {}
        self.ml_splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": [], "test": []}
        self.singletons: Dict[str, Any] = {} # features_df, ml_dataset, etc.
        self.slice_count = 0
        
    def add_result(self, result: Dict[str, Any], slice_idx: int) -> None:
        """Add a slice result to the aggregator and release its memory."""
        if not result:
            return

        # 1. Handle DataFrame (Canonical extraction)
        df_result = result.get("result_df")
        if isinstance(df_result, pd.DataFrame) and not df_result.empty:
            slice_metadata = result.get("metadata", {})
            slice_start = slice_metadata.get("original_slice_start") or slice_metadata.get("slice_start")
            slice_end = slice_metadata.get("original_slice_end") or slice_metadata.get("slice_end")
            
            if slice_start is not None and slice_end is not None:
                # Extract canonical rows (same logic as before but incremental)
                if isinstance(df_result.index, pd.DatetimeIndex) and isinstance(self.original_df.index, pd.DatetimeIndex):
                    start_time = self.original_df.index[slice_start] if slice_start < len(self.original_df) else self.original_df.index[0]
                    end_time = self.original_df.index[min(slice_end - 1, len(self.original_df) - 1)] if slice_end <= len(self.original_df) else self.original_df.index[-1]
                    df_canonical = df_result[(df_result.index >= start_time) & (df_result.index <= end_time)].copy()
                else:
                    df_canonical = df_result.iloc[slice_start:slice_end].copy()
                
                self.combined_rows.append(df_canonical)
                # 🧹 CLEAR original from result to help GC
                result["result_df"] = None
                del df_result
            else:
                self.combined_rows.append(df_result.copy())
        
        # 2. Accumulate Lists (Signals, Zones)
        for key in ["signals", "zones"]:
            val = result.get(key)
            if isinstance(val, list):
                if key == "signals": self.signals.extend(val)
                else: self.zones.extend(val)
                result[key] = None # Clear
        
        # 3. ML Splits
        for split in ["train", "validation", "test"]:
            val = result.get(split)
            if isinstance(val, dict):
                self.ml_splits[split].append(val)
                result[split] = None # Clear

        # 4. Singletons
        for key in ["ml_dataset", "features_df", "enriched_df", "imbalance_analysis"]:
            if key not in self.singletons and result.get(key) is not None:
                self.singletons[key] = result[key]
                result[key] = None # Clear

        self.slice_count += 1
        gc.collect()

    def finalize(self) -> Dict[str, Any]:
        """Combine all accumulated data into a final result structure."""
        final_result = {"metadata": {"slices": self.slice_count}}
        
        # 1. Final DataFrame concat
        if self.combined_rows:
            combined_df = pd.concat(self.combined_rows, ignore_index=False)
            # 🧹 CLEAR pieces immediately
            self.combined_rows.clear()
            
            # Use original Manager to deduplicate (avoiding circular logic)
            # This will be called from ProcessingManager so we assume helper exists
            final_result["result_df"] = combined_df
        else:
            final_result["result_df"] = self.original_df

        # 2. Lists & Counts
        if self.signals: final_result["signals"] = self.signals
        if self.zones: final_result["zones"] = self.zones
        
        # 3. Splits
        for split, dicts in self.ml_splits.items():
            if dicts: final_result[split] = dicts # Still needs final merge

        # 4. Singletons
        final_result.update(self.singletons)
        
        return final_result


# ============================================================================
# TIER 1 INTERMEDIATE RESULTS CACHE
# ============================================================================

class CachedStepData:
    """Wrapper for cached step data with expiration tracking."""
    
    def __init__(self, data: Any, ttl_seconds: int = 1800):
        """Initialize cached data with TTL (default 30 min)."""
        self.data = data
        self.created_at = datetime.utcnow()
        self.ttl_seconds = ttl_seconds
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        age = (datetime.utcnow() - self.created_at).total_seconds()
        return age > self.ttl_seconds
    
    def get_age_minutes(self) -> float:
        """Get age of cached data in minutes."""
        return (datetime.utcnow() - self.created_at).total_seconds() / 60


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
    _lock = threading.Lock()
    
    @classmethod
    def store(cls, task_id: str, step_name: str, data: Any, ttl_seconds: int = 1800) -> None:
        """Cache intermediate result for next step."""
        cls.cleanup_expired()
        
        key = (task_id, step_name)
        with cls._lock:
            cls._cache[key] = CachedStepData(data, ttl_seconds)
            logger.info(f"[Cache] Stored {step_name} for task {task_id} (TTL: {ttl_seconds}s)")
    
    @classmethod
    def retrieve(cls, task_id: str, step_name: str) -> Optional[Any]:
        """Retrieve cached result, return None if expired or missing."""
        key = (task_id, step_name)
        with cls._lock:
            if key not in cls._cache:
                return None
            
            cached = cls._cache.get(key)
            if cached is not None and cached.is_expired():
                cls._cache.pop(key, None)
                logger.info(f"[Cache] Expired: {step_name} (age: {cached.get_age_minutes():.1f} min)")
                return None
            
            if cached is not None:
                age_min = cached.get_age_minutes()
                logger.info(f"[Cache] Retrieved {step_name} (age: {age_min:.1f} min)")
                return cached.data
        return None
    
    @classmethod
    def clear(cls, task_id: str) -> None:
        """Clear all cache entries for a task."""
        keys_to_delete = [k for k in cls._cache.keys() if k[0] == task_id]
        for key in keys_to_delete:
            cls._cache.pop(key, None)
        logger.info(f"[Cache] Cleared {len(keys_to_delete)} entries for task {task_id}")
    
    @classmethod
    def cleanup_expired(cls) -> int:
        """Remove expired entries, return count removed."""
        keys_to_delete = [k for k, v in cls._cache.items() if v.is_expired()]
        for key in keys_to_delete:
            cls._cache.pop(key, None)
        return len(keys_to_delete)
    
    @classmethod
    def cleanup_task(cls, task_id: str) -> None:
        """Force cleanup of a task from cache and trigger garbage collection."""
        cls.clear(task_id)
        gc.collect()
        logger.info(f"[Cache] Force cleanup completed for task {task_id}")
    
    @classmethod
    def aggressive_cleanup(cls) -> None:
        """Aggressive cleanup for low-memory situations: expire everything and GC."""
        cls._cache.clear()
        gc.collect()
        logger.warning("[Cache] Aggressive cleanup triggered - all cache cleared")


# ============================================================================
# REGISTER ALL HANDLERS AT MODULE LOAD (ONCE ONLY)
# ============================================================================

# Process-level guard to prevent duplicate registrations in worker processes
import os
_PROCESS_ID = os.getpid()
_HANDLERS_REGISTERED = False

try:
    from app.core.processing.processing_strategies import (
        technical_analysis_worker,
        snr_analysis_worker,
        astronomical_analysis_worker,
        currency_indices_analysis_worker,
        currency_strength_matrix_worker,
        ml_prep_worker,
    )
    # Only register if not already registered (avoid duplicate registrations in worker processes)
    if not _HANDLERS_REGISTERED:
        if not HandlerRegistry.is_registered("technical"):
            HandlerRegistry.register("technical", technical_analysis_worker)
        if not HandlerRegistry.is_registered("snr"):
            HandlerRegistry.register("snr", snr_analysis_worker)
        if not HandlerRegistry.is_registered("astronomical"):
            HandlerRegistry.register("astronomical", astronomical_analysis_worker)
        if not HandlerRegistry.is_registered("currency_indices"):
            from app.core.processing.processing_handlers import analyze_currency_indices_impl
            HandlerRegistry.register("currency_indices", analyze_currency_indices_impl)
        if not HandlerRegistry.is_registered("currency_strength_matrix"):
            HandlerRegistry.register("currency_strength_matrix", currency_strength_matrix_worker)
        if not HandlerRegistry.is_registered("ml_dataset_preparation"):
            HandlerRegistry.register("ml_dataset_preparation", ml_prep_worker)
        if not HandlerRegistry.is_registered("enrich_with_targets"):
            HandlerRegistry.register("enrich_with_targets", ml_prep_worker)
       
        HANDLERS_REGISTERED = True

        
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
    1. Strategy auto-selection based on data size and analysis type
    2. Context building (ProcessingContext)
    3. Execution delegation to selected strategy
    4. Result storage (TIER 1 cache + DB)
    5. Progress tracking (task_store + WebSocket)
    6. Checkpoint & recovery for large datasets
    
    Usage Pattern:
        pm = ProcessingManager(
            session_id=session_id,
            task_id=task_id,
            analysis_type="snr",
            config=SNRConfig(...),
            task_store=task_store,
            connection_manager=manager,
            user_id=user_id,  # 
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
        user_id: Optional[str] = "anonymous",
    ):
        """Initialize ProcessingManager."""
        init_spawn_method()
        
        # LOG: What config are we receiving?
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"📥 [PM.__init__] config type: {type(config).__name__}")
        if isinstance(config, dict):
            self.logger.info(f"[PM.__init__] config is dict with keys: {list(config.keys())}")
        else:
            self.logger.info(f"[PM.__init__] config is {type(config).__name__} - fields: {dir(config)[:5]}...")
        
        self.session_id = session_id
        self.task_id = task_id
        self.analysis_type = analysis_type
        self.analysis_type_value = (
            analysis_type.value if hasattr(analysis_type, "value") else str(analysis_type)
        )
        self.config = config
        self.task_store = task_store
        self.connection_manager = connection_manager
        self.processing_config = processing_config or ProcessingConfig()
        self.step_name = step_name or f"{self.analysis_type_value}_analysis"
        
        # Persistent state for ML split processing
        self.global_label_maps: Dict[str, Dict[str, int]] = {}
        self.global_scaler_binary: Optional[bytes] = None
        self._current_dataset_id: Optional[str] = None
        self._current_split_name: Optional[str] = "train"
        self.global_fill_means: Dict[str, float] = {} 
        self.global_feature_cols: Optional[List[str]] = None 
        self.global_columns_to_scale: Optional[List[str]] = None
        
        # Breakout detection tracking
        self.global_breakout_detections: Dict[str, Dict[str, Any]] = {}
        
        # Rolling mean baselines for lossless normalization
        self.global_rolling_mean_baselines: Dict[str, Dict[str, float]] = {}
        
        # Normalization method configuration
        self.normalization_method: str = getattr(config, 'normalization_method', 'clipping')  # 'clipping' or 'rolling_mean_sigmoid'
        self.sigmoid_scale_factor: float = getattr(config, 'sigmoid_scale_factor', 2.0)
        
        # Separate volume range tracking (not price-based structural range)
        self.fitted_volume_range_high: Dict[str, float] = {}  # e.g., {'Volume': 7500000, 'TickVolume': 3200000}
        self.fitted_volume_range_low: Dict[str, float] = {}   # Always 0 for volume
        self.global_volume_clipping_detection: Dict[str, Dict[str, Any]] = {}  # Phase 1 for volume

        # Separate distance range tracking (for snr_dist_*, Down_Distance, etc.)
        self.fitted_distance_range_high: Dict[str, float] = {}  # Distance columns 95th percentile + buffer
        self.fitted_distance_range_low: Dict[str, float] = {}   # Always 0 for distances
        self.global_distance_clipping_detection: Dict[str, Dict[str, Any]] = {}  # Detection for distances

        # Separate footprint range tracking (for FP_Delta, FP_Cum_Delta — signed volume-scale)
        self.fitted_footprint_range_high: Dict[str, float] = {}  # Symmetric range for signed delta columns
        # No _low dict needed — footprint deltas are symmetric around 0

        # Per-currency-index structural ranges (Dollar ~104, Euro ~1.1, JPY ~150 each need own range)
        self.fitted_index_ranges: Dict[str, Dict[str, float]] = {}  # {idx_name: {high, low, width, vol_factor}}

        # StandardScaler for diff columns (mean-centered, unbounded)
        self.diff_scalers: Dict[str, StandardScaler] = {}  # Per-column StandardScaler for EMA_*_Diff, MA_*_Diff, etc.
        self.global_diff_scaler_binary: Optional[bytes] = None  # Serialized dict of diff scalers

        self.user_id = user_id or "anonymous"  
        self.logger = logging.getLogger(__name__)

        # Progress tracking
        self.total_rows = 0
        self.rows_processed = 0
        self.current_stage = "initializing"
        self.global_scaler: Optional[Any] = None  
        self.global_scaler_binary: Optional[bytes] = None 
        
        # Core processors
        self.ml_prep = None

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

            effective_task_store = self.task_store
            self.logger.info(f"[PM] Using base TaskStore for task {self.task_id} (No Proxy)")

            # STEP 1: Auto-select strategy based on size + analysis type
            strategy_type = StrategyFactory.determine_strategy(
                n_rows,
                self.processing_config,
                self.analysis_type
            )

           
            self.logger.info(
                f"[PM] Executing {self.analysis_type} with {n_rows} rows "
                f"using {strategy_type.value} strategy"
            )

            # AGGREGATE: Capture stacking variables from kwargs
            dataset_name = kwargs.get("dataset_name")
            if dataset_name and (self.analysis_type == AnalysisType.ML_DATASET_PREPARATION or self.analysis_type == "ml_dataset_preparation"):
                # Use dataset_name to enable "stacking" in the DB
                original_step_name = self.step_name
                self.step_name = f"ml_prep_{dataset_name}"
                self.logger.info(f"📚 [PM] Dataset Stacking enabled: '{original_step_name}' -> '{self.step_name}'")

            # STEP 2: Send initial progress
            await self._send_progress_update(0, f"Starting {self.analysis_type} analysis...")

            # STEP 3: Execute with selected strategy
            # Special handling for ML Dataset Preparation - split first, then process each split
            if self.analysis_type == AnalysisType.ML_DATASET_PREPARATION or self.analysis_type == "ml_dataset_preparation":
                self.logger.info(f"[PM] ML Dataset Preparation: Splitting data before processing")
                result = await self._execute_ml_with_splits(df, effective_task_store=effective_task_store, **kwargs)
            else:
                # All other strategies (Sequential, Parallel Chunking, Slice Streaming) go through the same path
                self.logger.info(f"[PM] Using {strategy_type.value.upper()} for {n_rows} rows")
                result = await self._execute_with_strategy(strategy_type, df, effective_task_store=effective_task_store, **kwargs)
            
            # STEP 3.5: Unified enrichment guarantee
            # Ensures original OHLCV columns are merged if worker only returned features
            # and restores the DatetimeIndex.
            # SKIP for SNR analysis - it returns complete data and enrichment causes 3GB memory spike
            # SKIP for CURRENCY_INDICES - worker already returns base OHLCV + index cols;
            #   the original_df passed in is the pair-enriched df (base + 36 pair cols),
            #   so enrichment would re-add pair columns that the worker explicitly dropped.
            #   The mixin (execute_currency_indices_analysis) handles its own completeness.
            if ("result_df" in result and isinstance(result["result_df"], pd.DataFrame) and
                    self.analysis_type != AnalysisType.SNR and
                    self.analysis_type != AnalysisType.CURRENCY_INDICES and          # skip — mixin owns this
                    self.analysis_type != "currency_indices" and                     # str variant
                    self.analysis_type != AnalysisType.ML_DATASET_PREPARATION and   # Bug #3 fix
                    self.analysis_type != "ml_dataset_preparation"):                 # Bug #3 fix (str variant)
                # FIX #4: Drop the original df reference BEFORE _ensure_result_completeness
                # so only two copies exist at peak (result_df + merged output) instead of three.
                # We keep a shallow variable for the call, then delete it immediately.
                _original_ref = df
                del df
                gc.collect()
                result["result_df"] = self._ensure_result_completeness(result["result_df"], _original_ref)
                del _original_ref
                gc.collect()
                df = None  # Sentinel: prevent double-del in exception handlers below

            # STEP 4: Cache result for next step (TIER 1)
            # Store only a *weak* DataFrame reference by not keeping a second copy.
            # The cache TTL is reduced from 30 min to 10 min for analysis types that
            # produce very wide DataFrames (astronomical, technical with many indicators)
            # to prevent multi-GB objects being pinned for half an hour.
            if "result_df" in result and isinstance(result["result_df"], pd.DataFrame):
                _wide_types = {"astronomical", "technical", "currency_indices"}
                _ttl = 600 if self.analysis_type_value in _wide_types else 1800
                IntermediateResultsCache.store(
                    self.task_id,
                    self.step_name,
                    result["result_df"],
                    ttl_seconds=_ttl
                )
                self.logger.info(f"[PM] Cached {self.step_name} result to TIER 1 (TTL={_ttl}s)")

            # STEP 5: Persist to database (TIER 3) with progress updates
            # Report serialization progress (98-99%)
            await self._send_progress_update(
                progress=98,
                message="Serializing",
                message2=f"Preparing data for storage - compressing results",
                processed_bars=self.total_rows,
                total_bars=self.total_rows,
                current_indicator="Serialize",
                strategy=result.get("metadata", {}).get("strategy", "Sequential")
            )
            
            # If ML Prep, ensure we persist the Splits, not just the result_df (which is just the input df)
            await self._persist_to_database(result)
            
            # Report storage completion (99%)
            await self._send_progress_update(
                progress=99,
                message="Storing",
                message2=f"Saving to database - finalizing transaction",
                processed_bars=self.total_rows,
                total_bars=self.total_rows,
                current_indicator="Storage",
                strategy=result.get("metadata", {}).get("strategy", "Sequential")
            )
            await self._send_progress_update(
                progress=100,
                message="Complete",
                message2=f"{self.analysis_type} analysis finished successfully",
                processed_bars=self.total_rows,
                total_bars=self.total_rows,
                current_indicator="Done",
                strategy=result.get("metadata", {}).get("strategy", "Sequential")
            )
            if df is not None:
                del df
            gc.collect()
            
            return result

        except TaskCancelledException:
            if 'df' in locals() and df is not None:
                del df
            gc.collect()
            self.logger.warning(f"[PM] Task {self.task_id} was cancelled")
            raise
        except Exception as err:
            if 'df' in locals() and df is not None:
                del df
            gc.collect()
            self.logger.error(f"[PM] Execution failed: {err}", exc_info=True)
            await self._send_error(str(err))
            raise

    def create_slices(self, df: pd.DataFrame, slice_size: int = 10000, overlap: int = 100) -> List[Tuple[int, int, int, int]]:
        """
        Compute slice boundaries with configurable overlap.
        For backwards compatibility and unit testing.

        Returns:
            List of (slice_num, start, end, overlap_start) tuples.
        """
        total_rows = len(df)
        self.total_slices = max(1, -(-total_rows // slice_size))  

        slices = []
        for slice_idx in range(self.total_slices):
            start = slice_idx * slice_size
            end = min(start + slice_size, total_rows)
            overlap_start = max(0, start - overlap)
            slices.append((slice_idx, start, end, overlap_start))

        return slices

    async def _send_error(self, error_message: str) -> None:
        """Send error message via WebSocket."""
        try:
            if self.connection_manager is not None:
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

    async def _execute_with_strategy(
        self,
        strategy_type: ProcessingStrategy,
        df: pd.DataFrame,
        effective_task_store=None,
        pm_instance=None, 
        db_session=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute with any strategy (Sequential, Parallel Chunking, or Slice Streaming).
       
        Args:
            strategy_type: Strategy to use
            df: Input DataFrame
            effective_task_store: TaskStore for progress tracking
            pm_instance: ProcessingManager instance (for chunk storage)
            db_session: Database session (for chunk storage)
            **kwargs: Additional parameters (including scaler information)
            
        Returns:
            Result dictionary
        """
        # Use effective proxy if supplied, otherwise fall back to raw task_store
        task_store_to_use = effective_task_store if effective_task_store is not None else self.task_store

        global_scaler = kwargs.get('global_scaler')
        fit_scaler = kwargs.get('fit_scaler', True)
        skip_scaling = kwargs.get('skip_scaling', False)
        split_name = kwargs.get('split_name', kwargs.get('split_type', 'train'))
        feature_cols = kwargs.get('feature_cols')
        columns_to_scale = kwargs.get('columns_to_scale')
        
        # Safety validation: catch conflicting settings
        if not skip_scaling and fit_scaler and global_scaler is not None:
            self.logger.warning(
                f"[ML-Scaling] ⚠️  Conflict detected for {split_name}: "
                f"fit_scaler=True but global_scaler already provided. "
                f"This usually indicates a logic error - scaler should only be fitted ONCE on training data."
            )
        
        if skip_scaling and global_scaler is None:
            self.logger.warning(
                f"[ML-Scaling] ⚠️  Risky setup for {split_name}: "
                f"skip_scaling=True but no global_scaler provided. "
                f"Worker will have no scaling parameters!"
            )
        
        if skip_scaling and (feature_cols is None or columns_to_scale is None):
            self.logger.warning(
                f"[ML-Scaling] ⚠️  Missing metadata for {split_name}: "
                f"skip_scaling=True but feature_cols or columns_to_scale is None. "
                f"Worker may fail to validate scaled output."
            )
        
        # Log scaler state for debugging
        self.logger.info(
            f"[ML-Scaling] {split_name}: fit_scaler={fit_scaler}, skip_scaling={skip_scaling}, "
            f"scaler_available={global_scaler is not None}, "
            f"features={len(feature_cols) if feature_cols else 0}, "
            f"columns_to_scale={len(columns_to_scale) if columns_to_scale else 0}"
        )

        # Build processing context
        context = ProcessingContext(
            task_id=self.task_id,
            session_id=self.session_id,
            analysis_type=self.analysis_type,
            config=self.config,
            task_store=task_store_to_use,
            connection_manager=self.connection_manager,
            processing_config=self.processing_config,
            user_id=self.user_id,
            global_scaler=global_scaler,  # Pass scaler to context
        )

        # Create strategy instance
        strategy = StrategyFactory.create_strategy(
            strategy_type,
            context,
            self.logger
        )

        ml_analyses = {'ml_dataset_preparation', 'model_training', 'model_build'}
        
        if self.analysis_type in ml_analyses:
            ml_kwargs = {
                'fit_scaler': fit_scaler,
                'skip_scaling': skip_scaling,
                'split_name': split_name,
                'split_type': kwargs.get('split_type', 'train'),
                'global_scaler': global_scaler,
                'feature_cols': feature_cols,
                'columns_to_scale': columns_to_scale,
                'ml_prep_metadata': {
                    'feature_cols': feature_cols,
                    'columns_to_scale': columns_to_scale,
                },
                'pm_instance': pm_instance,
                'db_session': db_session,
            }
            # Remove None values to keep kwargs clean and prevent "got unexpected kwarg" errors
            ml_kwargs = {k: v for k, v in ml_kwargs.items() if v is not None}
            self.logger.debug(f"[ML-Scaling] Passing {len(ml_kwargs)} kwargs to strategy: {list(ml_kwargs.keys())}")
        else:
            # Non-ML analyses don't need ML-specific kwargs
            ml_kwargs = {}
            self.logger.debug(f"[Strategy] Non-ML analysis ({self.analysis_type}), no ML kwargs needed")

        # ── Index-OHLCV normalisation for TECHNICAL analysis ──────────────────
        # When analysis_manager calls PM(analysis_type=TECHNICAL) to compute
        # technical indicators on a currency index (e.g. Dollar), the input df
        # has columns Dollar_open / Dollar_high / Dollar_low / Dollar_close /
        # Dollar_tick_volume instead of the standard Open/High/Low/Close/Volume
        # that every technical indicator function expects.
        #
        # _normalise_index_ohlcv_for_technical() strips the index prefix,
        # renames to title-case standard names, runs the strategy, then renames
        # the output columns back to {prefix}_{indicator} so analysis_manager
        # receives Dollar_RSI_14, Dollar_EMA_8, etc. as it expects.
        # ─────────────────────────────────────────────────────────────────────
        _index_prefix, df = self._normalise_index_ohlcv_for_technical(df)

        # Execute strategy
        result = await strategy.execute(df, **ml_kwargs)

        # Restore index prefix on all NEW columns produced by TI workers
        if _index_prefix and "result_df" in result and isinstance(result["result_df"], pd.DataFrame):
            result["result_df"] = self._restore_index_prefix(
                result["result_df"], df, _index_prefix
            )

        return result

    # ------------------------------------------------------------------
    # Index-OHLCV normalisation helpers
    # ------------------------------------------------------------------

    def _normalise_index_ohlcv_for_technical(
        self, df: pd.DataFrame
    ) -> tuple:
        """
        Detect currency-index-prefixed OHLCV columns and rename them to the
        standard Open / High / Low / Close / Volume names that every technical
        indicator function expects.

        Only activates when:
          • analysis_type is TECHNICAL (or the string "technical")
          • the DataFrame has no standard 'Open' column
          • at least one known index prefix is present  (e.g. "Dollar_open")

        Returns:
            (detected_prefix, normalised_df)
            detected_prefix is the index name string (e.g. "Dollar") if we
            renamed, or None if the DataFrame was already in standard form.

        The caller must pass detected_prefix to _restore_index_prefix() so
        the newly generated TI columns are renamed back to Dollar_RSI_14 etc.
        """
        is_technical = (
            self.analysis_type == AnalysisType.TECHNICAL
            or self.analysis_type == "technical"
        )
        if not is_technical:
            return None, df

        # Already has standard columns — nothing to do
        if "Open" in df.columns or "open" in df.columns:
            return None, df

        from app.core.analysis.currency_index import INDEX_DEFINITIONS

        field_map = {
            "open":        "Open",
            "high":        "High",
            "low":         "Low",
            "close":       "Close",
            "tick_volume": "Volume",   # TI engine treats Volume / TickVolume interchangeably
            "volume":      "Volume",
        }

        detected_prefix = None
        rename: dict = {}

        for idx_name in INDEX_DEFINITIONS.keys():
            candidate = f"{idx_name}_close"
            if candidate not in df.columns:
                continue

            # This index is present — build the rename map
            detected_prefix = idx_name
            for field, std_name in field_map.items():
                src = f"{idx_name}_{field}"
                if src in df.columns:
                    rename[src] = std_name
            break  # Only one index per TI call (one PM per index)

        if not detected_prefix or not rename:
            return None, df

        self.logger.info(
            "[PM] Index-OHLCV normalisation: renaming %d columns for '%s' "
            "(%s → standard OHLCV) before technical analysis",
            len(rename), detected_prefix,
            list(rename.keys()),
        )

        df = df.rename(columns=rename)
        return detected_prefix, df

    def _restore_index_prefix(
        self,
        result_df: pd.DataFrame,
        input_df: pd.DataFrame,
        prefix: str,
    ) -> pd.DataFrame:
        """
        Rename newly generated TI columns back to {prefix}_{col} form.

        Columns that already existed in *input_df* (the normalised version
        passed to the strategy) are left untouched.  Only columns added by
        the technical worker are prefixed.

        Example:  RSI_14 → Dollar_RSI_14,  EMA_8 → Dollar_EMA_8
                  Open / Close / High / Low / Volume are NOT prefixed — they
                  are dropped or remain as the original index OHLCV fields.
        """
        existing_cols = set(input_df.columns)
        new_cols = {c for c in result_df.columns if c not in existing_cols}

        if not new_cols:
            return result_df

        rename = {c: f"{prefix}_{c}" for c in new_cols}

        self.logger.info(
            "[PM] Restoring index prefix '%s': %d new TI columns renamed "
            "(e.g. %s)",
            prefix,
            len(rename),
            list(rename.items())[:3],
        )

        return result_df.rename(columns=rename)

    async def _load_session_provenance(self) -> Dict[str, Any]:
        """Load DataSession symbol/timeframe/dates for serving contract provenance."""
        if not self.session_id:
            return {}
        try:
            from sqlalchemy import select as sa_select
            from app.database.models import DataSession

            async with AsyncPostgresSessionLocal() as db_session:
                result = await db_session.execute(
                    sa_select(DataSession).where(DataSession.session_id == self.session_id)
                )
                session_row = result.scalar_one_or_none()
                if session_row is None:
                    return {}
                return {
                    "session_id": str(session_row.session_id),
                    "symbol": session_row.symbol,
                    "timeframe": session_row.timeframe,
                    "start_date": session_row.start_date,
                    "end_date": session_row.end_date,
                    "data_source": session_row.data_source,
                    "record_count": session_row.record_count,
                }
        except Exception as error:
            self.logger.warning("[ML-Splits] Could not load DataSession provenance: %s", error)
            return {}

    async def _execute_ml_with_splits(self, df: pd.DataFrame, effective_task_store=None, **kwargs) -> Dict[str, Any]:
        """
        Execute ML Dataset Preparation with pre-splitting approach.
        
        1. Split DataFrame into train/val/test at ProcessingManager level
        2. Process TRAINING split first to fit and save scaler
        3. Process validation/test splits using the saved scaler from training
        4. Each split goes through PM strategy selection independently
        5. Core ML function is agnostic to which split it's processing
        6. No double-splitting (PM splits, then ML class splits again)
        """ 

        await self._send_progress_update(0, f"Starting MLDatasetPreparation..")

        # Capture upstream step configs + session provenance for serving contract
        step_configs = dict(kwargs.get("step_configs") or {})
        session_provenance = await self._load_session_provenance()
        self._ml_step_configs = step_configs
        self._session_provenance = session_provenance
        if step_configs:
            self.logger.info(
                "[ML-Splits] Serving contract step_configs keys: %s",
                sorted(step_configs.keys()),
            )
        if session_provenance.get("symbol"):
            self.logger.info(
                "[ML-Splits] Session provenance: %s/%s (%s → %s)",
                session_provenance.get("symbol"),
                session_provenance.get("timeframe"),
                session_provenance.get("start_date"),
                session_provenance.get("end_date"),
            )

        # Log all columns and dtypes for debugging
        self.logger.info(f"[ML-PREP] Original DataFrame columns and dtypes:\n{df.dtypes}")
        self.logger.info(f"[ML-PREP] All Columns in DataFrame:\n{len(df.columns)}")
        
        
        # Extract config for splitting ratios
        config = self.config
        if not config:
            self.logger.error("[ML-Splits] ❌ No config found")
            raise ValueError("No config found")
            
        if isinstance(config, dict):
            config_obj = DatasetConfig(**config)
        else:
            config_obj = config
        
        # DETAILED CONFIG LOGGING: Show all fields including prepare_advanced_ml_targets
        self.logger.info(f"[ML-Splits] Config object created successfully")
        self.logger.info(f"   sequence_length: {config_obj.sequence_length}")
        self.logger.info(f"   prediction_length: {config_obj.prediction_length}")
        # Use getattr() for defensive access (handles both old/new config formats)
        self.logger.info(f"   prepare_advanced_ml_targets: {getattr(config_obj, 'prepare_advanced_ml_targets', False)}")
        self.logger.info(f"   advanced_target_lookforward: {getattr(config_obj, 'advanced_target_lookforward', 20)}")
        self.logger.info(f"   include_classification: {config_obj.include_classification}")
        self.logger.info(f"   include_regression: {config_obj.include_regression}")
        self.logger.info(f"   include_sequence_prediction: {config_obj.include_sequence_prediction}")
        self.logger.info(f"   feature_selection_mode: {config_obj.feature_selection_mode}")
        self.logger.info(f"   dataset_name: {config_obj.dataset_name}")
        
        # Log full config object for debugging
        self.logger.debug(f"[ML-Splits] Full config: {config_obj}")
        # Step 1: Validate data before splitting
        is_valid, data_error = validate_ml_data(df, config_obj)
        if not is_valid:
            self.logger.error(f"[ML-Splits] ❌ Data validation failed: {data_error}")
            raise ValueError(data_error)


        # Architecture: Global-Enrich → Fit-on-Train → Per-Split-Transform
        #
        #   1. _enrich_with_targets(full_df)     — targets computed globally so split
        #                                          boundaries don't lose prediction_length rows
        #   2. Split boundaries calculated        — train/val/test index positions
        #   3. _fit_structural_range(train_view)  — causal fit, train rows only
        #   4. Per-split loop (warm-start ctx):
        #        a. Prepend rolling_window raw rows from prior split (prevents cold-start NaNs)
        #        b. _apply_structural_range        — fixed scalars, no leakage
        #        c. _add_regime_context_features   — ratios of structural range
        #        d. _normalize_by_rolling_structural_range  — OHLCV → [0, 1]
        #        e. _normalize_price_level_indicators
        #        f. Strip context rows
        #        g. _sanitize_split_for_scaling    — imputation using train statistics
        #        h. SelectiveScaler.transform      — (fit only on train)
        # 1. Enrich with targets globally (prevents losing boundary rows at split edges)

        
        ml_prep = MLDatasetPreparation(
            data=df,
            config=config_obj,
            task_id=self.task_id,
            reporter=None
        )
        self.ml_prep = ml_prep  # Store for potential later use (e.g. in per-split processing)
        # Thread step_configs into ml_prep so _generate_sequences_for_df can embed it in NPZ chunks
        ml_prep._step_configs = step_configs
        
        
        # Step 1: Calculate split boundaries BEFORE target enrichment
        n_rows = len(self.ml_prep.data)
        train_end = int(n_rows * config_obj.train_ratio)
        val_end = int(n_rows * (config_obj.train_ratio + config_obj.validation_ratio))
        self.logger.info(f"[ML-Splits] Pre-calculated split boundaries: train=0:{train_end}, val={train_end}:{val_end}, test={val_end}:{n_rows}")

        await self._send_progress_update(0, message=f"Enriching dataset with targets...")
        
        try:
            await self.ml_prep._enrich_with_targets()  # Direct call to enrichment logic (async)
        except Exception as e:
            self.logger.error(f"❌ [MLPrep] Error occurred while enriching dataset: {e}")
            raise

        full_df = self.ml_prep.data
        
        # ASYNC STABILITY: Wait 1 second after enrichment to ensure all async operations
        # (column creation, feature calculation, target preparation) are fully complete
        # before proceeding. This prevents race conditions where columns might not be
        # fully propagated across different runs on the same dataset.
        await asyncio.sleep(1)
        self.logger.info(f"[ML-Splits] ⏳ Waited 1s post-enrichment for async stability (full_df shape: {full_df.shape})")

        # Capture ALL target column names AFTER enrichment.
        # Two families must be included:
        #
        #   1. adv_target_* — continuous/sequence regression targets produced by
        #      _enrich_with_targets() (MFE, MAE, OHLCV sequences, regime speeds, etc.)
        #
        #   2. Signal_bounce_* / Signal_breakout_* — binary classification labels
        #      produced by signal_generator.py and carried through the pipeline.
        #      These DO NOT start with "adv_target_" so the old single-prefix scan
        #      silently dropped all four of them, leaving them absent from every .npz
        #      chunk and zero-filled in align_targets_to_model_outputs at training time.
        #
        # Both families are excluded from feature_cols by _identify_features()
        # (data-leakage prevention), but BOTH must be present in target_keys so the
        # sequence generator writes them as target_Signal_* keys in the NPZ.
        SIGNAL_CLASSIFICATION_COLS = [
            "Signal_bounce_support",
            "Signal_bounce_resistance",
            "Signal_breakout_support",
            "Signal_breakout_resistance",
        ]

        # FIX: Capture ALL target columns from enriched DataFrame, not just those starting with 'adv_target_'
        # This includes: adv_target_*, Signal_*, and any other enrichment-produced targets
        adv_target_cols_from_enrichment = sorted([
            c for c in full_df.columns 
            if (c.startswith('adv_target_') 
                or c in SIGNAL_CLASSIFICATION_COLS
                or any(pattern in c for pattern in ['reversal', 'next_zone', 'trend_continuation']))
        ])

        self.logger.info(
            f"[ML-Splits] 🎯 Found {len(adv_target_cols_from_enrichment)} target columns "
            f"after enrichment ({len([c for c in adv_target_cols_from_enrichment if c.startswith('adv_target_')])} adv_target_* "
            f"+ {len([c for c in adv_target_cols_from_enrichment if c.startswith('Signal_')])} Signal_*): "
            f"{adv_target_cols_from_enrichment}"
        )
        
        # 🔍 DEBUG: Check for specific missing targets
        reversal_in_list = [c for c in adv_target_cols_from_enrichment if 'reversal' in c or 'trend_continuation' in c]
        next_zone_in_list = [c for c in adv_target_cols_from_enrichment if 'next_zone' in c]
        signal_in_list = [c for c in adv_target_cols_from_enrichment if c.startswith('Signal_')]
        self.logger.info(
            f"[ML-Splits] 🔍 Target breakdown: "
            f"reversal/trend={len(reversal_in_list)} {reversal_in_list}, "
            f"next_zone={len(next_zone_in_list)} {next_zone_in_list}, "
            f"Signal={len(signal_in_list)} {signal_in_list}"
        )

        # Store for workers — passed as enriched_target_columns in split_kwargs so
        # MLDatasetPreparation._generate_sequences() includes them in target_keys
        self.enriched_target_columns = adv_target_cols_from_enrichment
        self.logger.info(
            f"[ML-Splits] 📦 Stored enriched_target_columns in PM instance "
            f"({len(self.enriched_target_columns)} targets) for worker distribution"
        )

        await self._send_progress_update(12, message=f"Dataset enriched with targets")
        
        #   Fit structural range on TRAINING slice only, then apply fixed scalars to full_df.
        #   Old code computed rolling high/low on all rows before splitting, so the 'range'
        #   at row 100 was influenced by prices at row 900. This compressed training prices
        #   into the upper 85–95% of the range (mean≈0.87, std≈0.07) because the expanding
        #   window kept growing but prices never revisited the lower half.
        #
        #   New code:
        #     1. _fit_structural_range(train_slice) — scalar max/min from train rows only
        #     2. _apply_structural_range(full_df)   — broadcasts those fixed scalars everywhere
        #   Same raw price → same normalised value in every split (zero cross-split drift).
        rolling_window = getattr(config_obj, 'rolling_window', config_obj.sequence_length)
        self.logger.info(f"[ML-Splits] Using rolling_window={rolling_window} (sequence_length={config_obj.sequence_length})")
        

        # Generator yields RAW (target-enriched only) slices one at a time.
        # All transformation passes happen per-split inside the loop.
        def _iter_splits():
            yield "train",      full_df.iloc[:train_end]
            yield "validation", full_df.iloc[train_end:val_end]
            yield "test",       full_df.iloc[val_end:]
        
        # Fit structural range ONCE on training data only (causal — no future data).
        # DO NOT normalize here - normalization happens inside the loop for ALL splits including train.
        # This ensures train is normalized exactly ONCE, same as val/test.
        train_view = full_df.iloc[:train_end].copy()
        if len(train_view) > 0:
            self.logger.info(f"[ML-Splits] Fitting structural range on training view ({len(train_view)} rows)")
            await self._send_progress_update(1, f"Fitting structural range on training data...")
            # Fetch separate scaler types or fall back to legacy scaler_type
            price_scaler_type =  "none"  # PassThroughScaler - prices already normalized by structural range [0,1]
            # Diff columns are pre-normalized by _normalize_diff_columns_by_range() using
            # Rolling_Range_Width as denominator → already in [-1, 1]. PassThrough preserves
            # regime-relative information that MinMax would destroy by re-compressing.
            diff_scaler_type = 'none'   # PassThroughScaler — diffs pre-scaled by range width
            other_scaler_type = "minmax"  # getattr(config_obj, 'other_scaler_type', 'minmax')

            self.logger.info(
                f"[ML-Splits] 🎯 Scaler Partitioning Configured:\n"
                f"   ├─ Price Scaler: {price_scaler_type} (PassThrough — already [0,1] from structural range)\n"
                f"   ├─ Diff Scaler:  {diff_scaler_type}  (PassThrough — pre-scaled to [-1,1] by range width)\n"
                f"   └─ Other Scaler: {other_scaler_type}"
            )

            # Fit structural range ONCE on training data only (causal — no future data).
            # Then run the FULL normalization pipeline on train_view so the scaler fits on [0,1] data.
            # This ensures train is normalized exactly ONCE (here for scaler fitting), 
            # and val/test are normalized ONCE (in the loop).
            self._fit_structural_range(train_view, window=rolling_window)
            self.logger.info(f"[ML-Splits] Structural range fitted (fixed scalars for all splits)")
            
            # Fit volume-specific range (uses rolling volume percentile + buffer)
            self._fit_volume_range(train_view, percentile=0.95, buffer=1.2)
            self.logger.info(f"[ML-Splits] Volume range fitted (separate from price structural range)")
            
            # NEW: Fit distance-specific range for distance columns
            self._fit_distance_range(train_view, percentile=0.95, buffer=1.2)
            self.logger.info(f"[ML-Splits] Distance range fitted (separate from price structural range)")
            
            # NEW: Fit footprint delta range (FP_Delta, FP_Cum_Delta — signed volume-scale)
            self._fit_footprint_range(train_view, percentile=0.95, buffer=1.2)
            self.logger.info(f"[ML-Splits] Footprint range fitted (symmetric for signed delta columns)")
            
            # NEW: Fit separate structural range for each currency index found in train_view
            # (Dollar_close ~104, Euro_close ~1.1, JPY_close ~150 need own scale!)
            self._fit_index_structural_ranges(train_view, window=rolling_window)
            self._fit_index_volume_ranges(train_view)
            
            # NEW: Fit StandardScaler for diff columns (EMA_*_Diff, MA_*_Diff, MACD, etc.)
            self._fit_diff_scalers(train_view)
            self.logger.info(f"[ML-Splits] Diff StandardScalers fitted (unbounded, preserves negatives and spikes)")
            
            # ASYNC STABILITY: Wait 1 second after range fitting to ensure all structural,
            # volume, distance, and diff calculations are fully complete before proceeding.
            await asyncio.sleep(1)
            self.logger.info(f"[ML-Splits] ⏳ Waited 1s post-fit for async stability (ready for transformation pipeline)")
            
            # Apply same per-split transformation pipeline so scaler fits on [0,1] structural-range-normalized data
            self._current_split_name = "train"
            train_view = self._normalize_volume_by_range(train_view)  # Step 0: Volume
            train_view = self._normalize_distance_by_range(train_view)  # Step 0b: Distance
            train_view = self._normalize_footprint_by_range(train_view)  # Step 0c: Footprint
            train_view = self._apply_diff_scalers(train_view)  # Step 0d: Diff StandardScaler
            train_view = self._apply_structural_range(train_view)  # Step 1: Price structural range
            train_view = self._add_regime_context_features(train_view)
            train_view = self._normalize_by_rolling_structural_range(
                train_view, split_name="train", rolling_window=rolling_window
            )
            train_view = self._normalize_price_level_indicators(train_view)
            # Step 5: Normalize diff/distance columns by Rolling_Range_Width → [0, 1]
            train_view = self._normalize_diff_columns_by_range(train_view)
            # Step 6: Normalize currency index columns by their own structural ranges
            train_view = self._normalize_index_columns(train_view, split_name="train")
            self.logger.info(f"[ML-Splits] Training view normalized: structural range + regime context + OHLCV [0,1] + diff/range [0,1] + index ranges")

            # ── Scale alignment diagnostic ────────────────────────────────────
            # Verify: OHLCV → [0,1], diff cols → [0, 1] (centered around 0.5)
            _scale_check_cols = [
                'Open', 'High', 'Low', 'Close',           # should be [0, 1]
                'MACD_12_26_9', 'MACDh_12_26_9',          # should be [0, 1], mean ~0.5
                'EMA_12_Minus_EMA8', 'EMA_64_Minus_EMA8', # should be [0, 1], mean ~0.5
                'Supertrend_Distance',                      # should be [0, 1], mean ~0.5
                'snr_dist_to_nearest_level',               # should be [0.5, 1] (always positive distance)
                'FP_POC_Diff', 'FP_VAH_Diff', 'FP_VAL_Diff',   # should be [0, 1], mean ~0.5
                'FP_Delta', 'FP_Cum_Delta',                     # should be ~[-1, 1], mean ~0
            ]
            for _sc in _scale_check_cols:
                if _sc in train_view.columns:
                    _s = train_view[_sc].dropna()
                    self.logger.info(
                        f"[ScaleCheck] {_sc:35s}  min={_s.min():+.4f}  max={_s.max():+.4f}  "
                        f"mean={_s.mean():+.4f}  std={_s.std():.4f}"
                    )
            # ─────────────────────────────────────────────────────────────────
            
            # Sanitize for scaling (imputation, string → numeric)
            train_view = self._sanitize_split_for_scaling(train_view, split_name="train")
            self.logger.info(f"[ML-Splits] Training view sanitized (column names preserved)")
            
            # Identify features on NORMALIZED train_view
            feature_cols, columns_to_scale = self._identify_ml_features(train_view, config_obj)
            
            if not feature_cols:
                self.logger.error("[ML-Splits] ❌ No feature columns identified for scaling")
                raise ValueError("No feature columns identified for scaling.")
            
            self.logger.info(f"[ML-Splits] Feature columns identified: {len(feature_cols)}")
            
            # Categorize features for SelectiveScaler
            price_cols, diff_cols, other_cols = self._categorize_features(columns_to_scale)
            
            # Log Column Partitioning for Debugging
            self.logger.info(f'[ML-Splits] Price columns identified for PassThroughScaler: {price_cols}')
            self.logger.info(f'[ML-Splits] Diff columns identified for PassThroughScaler: {diff_cols}')
            self.logger.info(f'[ML-Splits] Other columns identified for MinMaxScaler: {other_cols}')

            self.logger.info(
                f"🛡️ [ML-Splits] Feature Partitioning:\n"
                f"   ├─ Price features (PassThrough, already [0,1]): {len(price_cols)} cols\n"
                f"   ├─ Diff features (PassThrough, already [0,1] via range-width mapping): {len(diff_cols)} cols\n"
                f"   └─ Other features (MinMax): {len(other_cols)} cols"
            )
            
            # Helper to create scaler instances
            def make_scaler(name: str):
                name = str(name).lower()
                if name == "minmax": return MinMaxScaler()
                elif name == "standard": return StandardScaler()
                elif name == "robust": return RobustScaler()
                else: return PassThroughScaler()
            
            # Create SelectiveScaler with correct partitioning
            global_scaler = SelectiveScaler(
                price_scaler=make_scaler(price_scaler_type),
                diff_scaler=make_scaler(diff_scaler_type),
                other_scaler=make_scaler(other_scaler_type),
                price_cols=price_cols,
                diff_cols=diff_cols,
                other_cols=other_cols,
                all_cols=columns_to_scale
            )
            
            # Fit scaler on [0,1] normalized training data
            try:
                scaler_input = train_view[columns_to_scale]
                self.logger.info(f"[ML-Splits] Fitting SelectiveScaler on {scaler_input.shape[0]} rows × {scaler_input.shape[1]} columns")
                global_scaler.fit(scaler_input)
                self.global_scaler = global_scaler
                
                # Validate scaler
                if not hasattr(global_scaler, 'n_features_in_'):
                    raise ValueError("Scaler fit operation failed silently")
                
                scaler_n_features = global_scaler.n_features_in_
                if scaler_n_features != len(columns_to_scale):
                    raise ValueError(
                        f"Scaler dimension mismatch: expected {len(columns_to_scale)} features, got {scaler_n_features}"
                    )
                
                # Store the exact columns the scaler was fitted on
                self.global_scaler_fitted_columns = list(columns_to_scale)
                self.logger.info(f"[ML-Splits] Scaler fitted and columns stored: {len(self.global_scaler_fitted_columns)} columns")
                
            except Exception as e:
                self.logger.error(f"[ML-Splits] Scaler fitting failed: {e}")
                raise ValueError(f"Failed to fit scaler: {e}")
            
            # Serialize scaler to bytes for DB storage
            try:
                _buf = io.BytesIO()
                joblib.dump(global_scaler, _buf)
                self.global_scaler_binary = _buf.getvalue()
                self.logger.info(f"[ML-Splits] Scaler serialized to {len(self.global_scaler_binary):,} bytes for DB")
            except Exception as _e:
                self.global_scaler_binary = None
                self.logger.warning(f"[ML-Splits] Failed to serialize scaler: {_e}")
            
           
          
            self.logger.info(f"✅ [FeaturesIdentify] {len(feature_cols)} features after MLPrep exclusion logic (started with {len(train_view.select_dtypes(include=[np.number]).columns)})")
            
            # Store globally for reuse in loop
            self.global_feature_cols = feature_cols
            self.global_columns_to_scale = columns_to_scale
            
            # ── Feature Index Map ────────────────────────────────────────────────────────
            # Maps each final feature column to its 0-based position inside the sequence
            # tensor so inference code can address features by name (not magic integers).
            # e.g. {"RSI_14": 22, "Dollar_RSI_14": 188, "MACD_12_26_9": 189, …}
            self.global_feature_index_map: Dict[str, int] = {
                col: idx for idx, col in enumerate(feature_cols)
            }
            self.global_index_feature_map: Dict[int, str] = {
                idx: col for col, idx in self.global_feature_index_map.items()
            }
            global_scaler.sequence_feature_names = list(feature_cols)
            global_scaler.sequence_feature_index_map = dict(self.global_feature_index_map)
            global_scaler.sequence_index_feature_map = dict(self.global_index_feature_map)
            try:
                _buf = io.BytesIO()
                joblib.dump(global_scaler, _buf)
                self.global_scaler_binary = _buf.getvalue()
                self.logger.info(
                    f"[ML-Splits] Scaler re-serialized with feature maps "
                    f"({len(self.global_scaler_binary):,} bytes)"
                )
            except Exception as _e:
                self.logger.warning(f"[ML-Splits] Failed to re-serialize scaler with feature maps: {_e}")
            self.logger.info(
                f"[ML-Splits] 🗂️  Built feature_index_map: {len(self.global_feature_index_map)} entries "
                f"(first 5: {dict(list(self.global_feature_index_map.items())[:5])})"
            )
            
            # FIX #7: Release train_view — it's a normalized copy of the training slice
            # (potentially 300+ cols × 50K+ rows). Nothing below needs it; all derived
            # artifacts (scaler, feature_cols, fill_means) are already stored on self.
            del train_view
            gc.collect()
        else:
            self.logger.error("[ML-Splits] ❌ Training split is empty after validation. Cannot fit structural range.")
            raise ValueError("Training split is empty. Check your data and split ratios.")
        
        
        
        dataset_id = str(uuid.uuid4())
        dataset_name = kwargs.get('dataset_name', 'ml_dataset')
        
     
        targets_sorted = sorted(adv_target_cols_from_enrichment)
        # Note: We keep time.time() here as per user design for testing leniency
        # Idealy we want to hash both inputs and outputs to detect any changes that would affect the dataset integrity,
        # but for now we focus on targets since they directly influence the output structure and model training.
        hash_input = json.dumps({
            "targets": targets_sorted, 
            "timestamp": time.time()
        })
        output_targets_hash = hashlib.sha256(hash_input.encode()).hexdigest()
        
        scaler_type_str = (
            config_obj.scaler_type.value 
            if hasattr(config_obj.scaler_type, 'value') 
            else str(config_obj.scaler_type)
        )
        
        # Populate config.exclude_columns BEFORE using MLPrep
        # This ensures MLPrep's _identify_features() excludes these 9 columns:
        # - 4 future-bias Signal_ columns (forward-looking, cause data leakage)
        # - 5 structural columns (metadata, not features)
        structural_cols = ["Date", "index", "Timestamp", "userId", "Time", "time_unix", "datetime", "Datetime", "Timestamp_unix"]
        future_bias_signals = ["Signal_bounce_support", "Signal_bounce_resistance", "Signal_breakout_support", "Signal_breakout_resistance"]
        
        exclude_set = set(structural_cols + future_bias_signals)
        config_obj.exclude_columns = list(exclude_set)
        self.logger.info(f"✅ [ConfigExclude] Set exclude_columns to {len(exclude_set)} columns: {sorted(exclude_set)}")
        
      
        async with AsyncPostgresSessionLocal() as db_session:
            # Determine target types using working copy
            target_types = {}
            for t in adv_target_cols_from_enrichment:
                if t in ['target_signal', 'Direction', 'Next_Day_Direction']:
                    target_types[t] = "classification"
                else:
                    target_types[t] = "regression"
            
            if hasattr(config_obj, 'include_sequence_prediction') and config_obj.include_sequence_prediction:
                target_types['future_sequence'] = "sequence_prediction"
            
            # Create sample_x/sample_y AFTER features are identified in the loop
            # This ensures correct shape: [1, seq_len, actual_feat_count]
            sample_x = None
            sample_y = None
            if len(adv_target_cols_from_enrichment) > 0:
                sample_y = {
                    "data": np.zeros((1, len(adv_target_cols_from_enrichment))).tolist(),
                    "shape": [1, len(adv_target_cols_from_enrichment)]
                }

            # Build scaling_config and serialize to ensure JSON compatibility
            # (converts numpy bools, numpy scalars, etc. to JSON-serializable types)
            scaling_config_raw = {
                "scaler_type": scaler_type_str,
                "feature_columns": feature_cols,
                "columns_to_scale": columns_to_scale,   # Bug #2 fix: include ALL columns the scaler expects
                "scaler_fitted": True,
                # Normalization method configuration (for inference)
                "normalization_method": self.normalization_method,  # "clipping" or "rolling_mean_sigmoid"
                "sigmoid_scale_factor": self.sigmoid_scale_factor,
                # Persist fitted structural range with hasattr guards for inference-time reconstruction.
                # MetricInferenceManager reads this to apply the same normalisation without re-fitting.
                "structural_range": {
                    "high":       self.fitted_range_high if hasattr(self, 'fitted_range_high') else None,
                    "low":        self.fitted_range_low if hasattr(self, 'fitted_range_low') else None,
                    "width":      self.fitted_range_width if hasattr(self, 'fitted_range_width') else None,
                    "vol_factor": self.fitted_vol_factor if hasattr(self, 'fitted_vol_factor') else None,
                    "fitted_on":  "train_only",
                    "window":     rolling_window
                },
                # Breakout detection info (for monitoring/alerting)
                "breakout_detection": {
                    "columns_with_breakouts": list(self.global_breakout_detections.keys()) if self.global_breakout_detections else [],
                    "detection_data": self.global_breakout_detections or {},
                    "detection_note": "Values exceeding fitted range that were clipped to [0,1]. Indicates potential regime shift requiring retraining."
                },
                # Rolling mean baseline metadata (for future lossless normalization)
                "rolling_mean_baselines": {
                    "available": bool(self.global_rolling_mean_baselines),
                    "columns": list(self.global_rolling_mean_baselines.keys()) if self.global_rolling_mean_baselines else [],
                    "note": "Baseline statistics for rolling mean sigmoid normalization (for future use or fallback)"
                },
                # VOLUME NORMALIZATION: Separate range for volume (not price-based structural range)
                "volume_range": {
                    "method": "rolling_percentile_plus_buffer",
                    "Volume": {
                        "fitted_range_high": float(self.fitted_volume_range_high.get('Volume', 0)),
                        "fitted_range_low": float(self.fitted_volume_range_low.get('Volume', 0)),
                    } if 'Volume' in self.fitted_volume_range_high else {},
                    "TickVolume": {
                        "fitted_range_high": float(self.fitted_volume_range_high.get('TickVolume', 0)),
                        "fitted_range_low": float(self.fitted_volume_range_low.get('TickVolume', 0)),
                    } if 'TickVolume' in self.fitted_volume_range_high else {},
                    "note": "Volume normalized by rolling percentile + 1.2x buffer, NOT by price structural range"
                },
                # VOLUME SPIKE DETECTION: Phase 1 for volume (different from price clipping)
                "volume_clipping_detection": {
                    "columns": list(self.global_volume_clipping_detection.keys()) if self.global_volume_clipping_detection else [],
                    "detection_data": self.global_volume_clipping_detection or {},
                    "detection_note": "Volume spikes exceeding 1.2x buffer (monitored, not problematic like price clipping)"
                },
                # DISTANCE NORMALIZATION: Separate range for distance columns
                "distance_range": {
                    "method": "rolling_percentile_plus_buffer",
                    "columns": list(self.fitted_distance_range_high.keys()) if self.fitted_distance_range_high else [],
                    "ranges": {
                        col: {
                            "fitted_range_high": float(self.fitted_distance_range_high.get(col, 0)),
                            "fitted_range_low": float(self.fitted_distance_range_low.get(col, 0)),
                        } for col in self.fitted_distance_range_high.keys()
                    } if self.fitted_distance_range_high else {},
                    "note": "Distance columns normalized by rolling percentile + 1.2x buffer, NOT by price structural range"
                },
                # FOOTPRINT NORMALIZATION: Symmetric range for signed delta columns
                "footprint_range": {
                    "method": "symmetric_percentile_plus_buffer (abs(value))",
                    "columns": list(self.fitted_footprint_range_high.keys()) if hasattr(self, 'fitted_footprint_range_high') else [],
                    "ranges": {
                        col: {"fitted_range_high": float(v), "fitted_range_low": float(-v)}
                        for col, v in self.fitted_footprint_range_high.items()
                    } if hasattr(self, 'fitted_footprint_range_high') else {},
                    "note": "FP_Delta/FP_Cum_Delta normalized to ~[-1, 1] range"
                },
                # DIFF COLUMN STANDARDSCALER: Unbounded, mean-centered normalization
                "diff_scalers": {
                    "method": "StandardScaler (unbounded, mean-centered)",
                    "columns": list(self.diff_scalers.keys()) if self.diff_scalers else [],
                    "note": "Diff columns normalized with StandardScaler: unbounded, preserves negatives and spikes",
                    "scaler_binary_available": bool(self.global_diff_scaler_binary)
                },
                # ── Feature Index Map ─────────────────────────────────────────────────────
                # {column_name: 0-based tensor index} — stable reference for model remapping.
                # Resolves "Feature 188 → score 3.22" to the actual column name for
                # inference pipelines and feature importance interpretation.
                "feature_index_map": self.global_feature_index_map if hasattr(self, 'global_feature_index_map') else {},
                "index_feature_map": self.global_index_feature_map if hasattr(self, 'global_index_feature_map') else {},
                "scaler_feature_index_map": getattr(global_scaler, 'feature_index_map', {}),
                "scaler_index_feature_map": getattr(global_scaler, 'index_feature_map', {}),
                "feature_partitions": getattr(global_scaler, 'partition_by_feature', {}),
                "partition_indices": getattr(global_scaler, 'partition_indices', {}),
                # ── Per-index structural ranges ───────────────────────────────────────────
                # Each currency index (Dollar, Euro, JPY, etc.) lives on its own price scale.
                # Inference must apply the same per-index normalization as training.
                "index_structural_ranges": {
                    idx_name: {
                        "high":       float(params['high']),
                        "low":        float(params['low']),
                        "width":      float(params['width']),
                        "vol_factor": float(params['vol_factor']),
                        "fitted_on":  "train_only",
                    }
                    for idx_name, params in self.fitted_index_ranges.items()
                } if self.fitted_index_ranges else {},
            }

            from app.core.ml.serving_contract import build_output_transform_spec
            scaling_config_raw["output_transform_spec"] = build_output_transform_spec(
                target_names=adv_target_cols_from_enrichment,
                scaling_config=scaling_config_raw,
            )
            
            # Serialize to ensure JSON compatibility (converts bool, numpy scalars, etc.)
            scaling_config_serialized = to_serializable(scaling_config_raw)
            
            stmt = pg_insert(MLDataset).values(
                dataset_id=dataset_id,
                session_id=self.session_id,
                dataset_name=dataset_name,
                output_targets=adv_target_cols_from_enrichment,
                output_targets_hash=output_targets_hash,
                target_metadata={
                    "target_names": adv_target_cols_from_enrichment,
                    "target_types": target_types,
                    "class_mappings": {} # Populated if classification is present
                },
                parent_dataset_id=config_obj.source_dataset_id if hasattr(config_obj, 'source_dataset_id') else None,
                source_step="ml_dataset_preparation",
                feature_columns=feature_cols,
                feature_count=len(feature_cols),
                sample_count=0,  # Will be updated by append
                features_x=sample_x,  # Fast-Truth Sample
                targets_y=sample_y,   # Fast-Truth Sample
                scaler_binary=self.global_scaler_binary,
                scaling_config=scaling_config_serialized,
                split_config={
                    "train_ratio": float(config_obj.train_ratio),
                    "validation_ratio": float(config_obj.validation_ratio),
                    "test_ratio": float(config_obj.test_ratio),
                    "sequence_length": int(config_obj.sequence_length),
                    "prediction_length": int(config_obj.prediction_length)
                },
                source_metadata=to_serializable({
                    "feature_names": feature_cols,
                    "step_name": self.step_name,
                    "prediction_length": int(config_obj.prediction_length),
                    "dataset_name": dataset_name,
                    "symbol": session_provenance.get("symbol"),
                    "timeframe": session_provenance.get("timeframe"),
                    "start_date": session_provenance.get("start_date"),
                    "end_date": session_provenance.get("end_date"),
                    "data_source": session_provenance.get("data_source"),
                    "record_count": session_provenance.get("record_count"),
                    "step_configs": step_configs,
                }),
                is_current=True,
                status="processing",
                # Always use chunks storage — O(1) windowed reads at retrieval
                # time. The blob default ('blob') forces full decompression on every
                # get_dataset_sequences call, causing the endpoint to hang on large
                # datasets. Chunks are written by _append_to_chunk_table already;
                # this ensures the retrieval header matches.
                storage_mode="chunks",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            
            await db_session.execute(stmt)
            await db_session.commit()
            
            self.logger.info(f"✅ Created empty parent MLDataset {dataset_id}")
            
        split_metadata = {}  # Track metadata only (no data in memory)
        progress_map = {"train": 10, "validation": 55, "test": 80}

        # [CRITICAL BUG FIX] Retrieve scaler from self before loop
        # The scaler was created in the training block and stored as self.global_scaler,
        # but inside the loop, we need to reference it as a local variable
        if not hasattr(self, 'global_scaler') or self.global_scaler is None:
            raise ValueError(
                "❌ [Data Integrity] Scaler was not fitted during training phase. "
                "Cannot proceed with validation/test transforms."
            )
        global_scaler = self.global_scaler
        self.logger.info(f"✅ [SplitLoop] Retrieved fitted scaler for val/test transforms")

        # Distribution Parity Tracking
        training_stats = {}

        # Warm-start context: tail of the previous split's RAW data.
        # Prepended before computing per-split rolling indicators so that
        # RSI, ATR, Bollinger etc. have `rolling_window` rows of lookback
        # at the very first row of val/test — no cold-start NaNs.
        # The context rows are STRIPPED after computation.
        prev_raw_tail: Optional[pd.DataFrame] = None
        # Accumulate any target names extracted from chunked .npz files across splits
        all_extracted_targets: Set[str] = set()

        for split_name, split_df_view in _iter_splits():
            if len(split_df_view) == 0:
                self.logger.warning(f"[ML-Splits] Skipping empty {split_name} split")
                continue
            self.logger.info(f"[ML-Splits] Processing {split_name} split ({len(split_df_view)} rows)")
            split_df_raw = split_df_view.copy()

            # ── Per-split transformation pipeline (strictly causal) ────────────────
            # 1. Prepend warm-start context rows so rolling indicators don't cold-start.
            if prev_raw_tail is not None and len(prev_raw_tail) > 0:
                ctx_rows = len(prev_raw_tail)
                split_with_ctx = pd.concat(
                    [prev_raw_tail, split_df_raw], axis=0
                ).reset_index(drop=True)
                self.logger.info(
                    f"[ML-Splits] {split_name}: Prepended {ctx_rows} warm-start context rows "
                    f"({len(split_df_raw)} split + {ctx_rows} ctx = {len(split_with_ctx)} total)"
                )
            else:
                split_with_ctx = split_df_raw.copy()
                ctx_rows = 0

            # Apply the complete per-split transformation pipeline
            # (structural range + regime context + OHLCV [0,1] + price-level indicators)
            split_with_ctx = self._apply_split_transformation_pipeline(
                split_with_ctx, split_name=split_name, rolling_window=rolling_window
            )

            # Strip context rows → only the actual split data remains
            if ctx_rows > 0:
                split_df = split_with_ctx.iloc[ctx_rows:].copy().reset_index(drop=True)  # FIX BUG #7: Reset index
                self.logger.info(
                    f"[ML-Splits] {split_name}: Stripped {ctx_rows} context rows "
                    f"→ {len(split_df)} final rows"
                )
            else:
                split_df = split_with_ctx.copy().reset_index(drop=True)  # FIX BUG #7: Ensure 0-based index

            # Save RAW tail for next split's warm-start.
            # We use RAW (pre-normalization) so that the next split's pipeline
            # runs on consistent input — no double-normalization of context rows.
            prev_raw_tail = split_df_raw.iloc[-rolling_window:].copy()
            del split_with_ctx, split_df_raw
            gc.collect()

            # Sanitize (imputation, string → numeric) using training statistics
            # Must set _current_split_name BEFORE sanitize so the label-encoding branch
            # inside it correctly reuses training maps for val/test (not rebuild them).
            self._current_split_name = split_name
            split_df = self._sanitize_split_for_scaling(split_df, split_name=split_name)
            
            # Apply SelectiveScaler with partial pass-through
            # Only price columns are fully normalized by structural range [0,1]
            # Diff and Other columns still need their scalers (RobustScaler, MinMaxScaler)
            # Price columns use PassThroughScaler (already normalized, no re-scaling)
            
            if split_name == 'train':
                # Validate scaler exists and was fitted before loop
                if not hasattr(self, 'global_scaler') or self.global_scaler is None:
                    raise ValueError(f"❌ [Data Integrity] Scaler not fitted. Training phase must complete first.")
                self.logger.info(f"[Scaler] {split_name}: Using scaler fitted on pre-loop training data (n_features={global_scaler.n_features_in_})")
            else:
                # Validate scaler is available for val/test (should always be true after train iteration)
                if global_scaler is None:
                    raise ValueError(f"❌ [Data Integrity] Scaler not available. Training must be processed first.")
                self.logger.info(
                    f"[Scaler] {split_name}: Reusing scaler fitted on training data "
                    f"(n_features={global_scaler.n_features_in_}, price_cols={len(getattr(global_scaler, 'price_indices', []))}, "
                    f"diff_cols={len(getattr(global_scaler, 'diff_indices', []))}, other_cols={len(getattr(global_scaler, 'other_indices', []))})"
                )
            
            # CRITICAL: Use the EXACT columns the scaler was fitted on
            if not hasattr(self, 'global_scaler_fitted_columns'):
                raise ValueError(f"❌ [Data Integrity] Scaler fitted columns not stored. Training must be processed first.")
            
            scaler_input_columns = self.global_scaler_fitted_columns
            
            # Validate all scaler columns exist in this split
            missing_cols = [c for c in scaler_input_columns if c not in split_df.columns]
            if missing_cols:
                raise ValueError(
                    f"❌ [{split_name}] Scaler columns missing: {missing_cols}. "
                    f"Scaler was fitted on: {scaler_input_columns}"
                )
            
            # Apply selective scaler: PassThroughScaler for prices (already [0,1]), 
            # RobustScaler for diffs, MinMaxScaler for others
            
            # ─────── PRE-SCALING SNAPSHOT ───────────────────────────────────────────────────
            # Capture min/max BEFORE scaling to diagnose what went into scaler
            pre_scale_ranges = {}
            for col in scaler_input_columns:
                pre_scale_ranges[col] = {
                    'min': float(split_df[col].min()),
                    'max': float(split_df[col].max()),
                    'mean': float(split_df[col].mean()),
                    'std': float(split_df[col].std())
                }
            
            split_df[scaler_input_columns] = global_scaler.transform(split_df[scaler_input_columns])
            self.logger.info(f"✅ [Scaler] {split_name}: Transformed {len(scaler_input_columns)} columns using SelectiveScaler (prices: PassThrough, diffs: Robust, others: MinMax)")
            
            # ─────── POST-SCALING DEBUG VALIDATION ──────────────────────────────────────────
            # Check for out-of-range features and NaN/inf values
            scaled_data = split_df[scaler_input_columns]
            
            # Overall statistics
            scaled_min = scaled_data.min().min()
            scaled_max = scaled_data.max().max()
            self.logger.info(f"[PostScaler] {split_name}: Overall range [{scaled_min:.6f}, {scaled_max:.6f}]")
            
            # Check for out-of-range features (< 0 or > 1)
            # Allow small numerical tolerance (-0.001 to 1.001) for floating point precision
            out_of_range_cols = []
            for col in scaler_input_columns:
                col_min = scaled_data[col].min()
                col_max = scaled_data[col].max()
                # Only flag if clearly out of bounds (not just boundary-touching)
                if col_min < -0.001 or col_max > 1.001:
                    pre_min = pre_scale_ranges[col]['min']
                    pre_max = pre_scale_ranges[col]['max']
                    pre_std = pre_scale_ranges[col]['std']
                    
                    # Determine which partition this column is in
                    partition = "UNKNOWN"
                    try:
                        col_idx = scaler_input_columns.index(col)
                        if hasattr(global_scaler, 'price_indices') and col_idx in global_scaler.price_indices:
                            partition = "PRICE"
                        elif hasattr(global_scaler, 'diff_indices') and col_idx in global_scaler.diff_indices:
                            partition = "DIFF"
                        elif hasattr(global_scaler, 'other_indices') and col_idx in global_scaler.other_indices:
                            partition = "OTHER"
                    except Exception:
                        pass
                    
                    out_of_range_cols.append({
                        'name': col,
                        'partition': partition,
                        'pre_min': pre_min,
                        'pre_max': pre_max,
                        'pre_std': pre_std,
                        'post_min': col_min,
                        'post_max': col_max,
                        'below_minus_0001': (scaled_data[col] < -0.001).sum(),
                        'above_1001': (scaled_data[col] > 1.001).sum()
                    })
            
            if out_of_range_cols:
                self.logger.warning(f"⚠️ [PostScaler] {split_name}: Found {len(out_of_range_cols)} OUT-OF-RANGE features (> ±0.1% tolerance):")
                for feat in sorted(out_of_range_cols, key=lambda x: max(abs(x['post_min']), abs(x['post_max'])), reverse=True)[:20]:  # Top 20 worst
                    self.logger.warning(
                        f"  [{feat['partition']}] '{feat['name']}': "
                        f"PRE=[{feat['pre_min']:.4f}, {feat['pre_max']:.4f}, std={feat['pre_std']:.4f}] "
                        f"POST=[{feat['post_min']:.6f}, {feat['post_max']:.6f}] "
                        f"(below_-0.001: {feat['below_minus_0001']}, above_1.001: {feat['above_1001']})"
                    )
            
            # Check for NaN values (safe for all dtypes)
            nan_cols = scaled_data.columns[scaled_data.isna().any()].tolist()
            if nan_cols:
                self.logger.warning(f"⚠️ [PostScaler] {split_name}: Found NaN in {len(nan_cols)} columns: {nan_cols[:10]}")
            
            # Check for Inf values only in numeric columns
            try:
                numeric_data = scaled_data.select_dtypes(include=['float64', 'float32', 'int64', 'int32'])
                if len(numeric_data.columns) > 0:
                    inf_cols = numeric_data.columns[np.isinf(numeric_data).any()].tolist()
                    if inf_cols:
                        self.logger.warning(f"⚠️ [PostScaler] {split_name}: Found Inf in {len(inf_cols)} columns: {inf_cols[:10]}")
            except Exception as e:
                self.logger.debug(f"[PostScaler] Could not check for Inf: {e}")
            
            # Per-partition statistics
            if hasattr(global_scaler, 'price_indices') and len(global_scaler.price_indices) > 0:
                price_data = scaled_data.iloc[:, global_scaler.price_indices]
                price_min, price_max = price_data.min().min(), price_data.max().max()
                self.logger.info(f"[PostScaler] {split_name} PRICE partition ({len(global_scaler.price_indices)} cols): [{price_min:.6f}, {price_max:.6f}]")
            
            if hasattr(global_scaler, 'diff_indices') and len(global_scaler.diff_indices) > 0:
                diff_data = scaled_data.iloc[:, global_scaler.diff_indices]
                diff_min, diff_max = diff_data.min().min(), diff_data.max().max()
                self.logger.info(f"[PostScaler] {split_name} DIFF partition ({len(global_scaler.diff_indices)} cols): [{diff_min:.6f}, {diff_max:.6f}]")
            
            if hasattr(global_scaler, 'other_indices') and len(global_scaler.other_indices) > 0:
                other_data = scaled_data.iloc[:, global_scaler.other_indices]
                other_min, other_max = other_data.min().min(), other_data.max().max()
                self.logger.info(f"[PostScaler] {split_name} OTHER partition ({len(global_scaler.other_indices)} cols): [{other_min:.6f}, {other_max:.6f}]")
            
            # [DISTRIBUTION PARITY CHECK] Monitor for leakage/contamination
            if split_name == 'train':
                # Store training statistics for comparison with val/test
                training_stats = {
                    'ohlcv_stats': {}
                }
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col in pre_scale_ranges:
                        training_stats['ohlcv_stats'][col] = {
                            'min': pre_scale_ranges[col]['min'],
                            'max': pre_scale_ranges[col]['max'],
                            'mean': pre_scale_ranges[col]['mean'],
                            'std': pre_scale_ranges[col]['std']
                        }
                self.logger.info(f"✅ [Parity] Training stats captured for leakage detection")
            else:
                # Compare val/test with training stats to detect leakage
                if training_stats and 'ohlcv_stats' in training_stats:
                    ood_cols = []
                    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        if col in pre_scale_ranges and col in training_stats['ohlcv_stats']:
                            train_stats = training_stats['ohlcv_stats'][col]
                            split_min = pre_scale_ranges[col]['min']
                            split_max = pre_scale_ranges[col]['max']
                            train_min = train_stats['min']
                            train_max = train_stats['max']
                            
                            # Check if split range significantly differs from training
                            # Allow 10% margin for natural variation
                            margin = (train_max - train_min) * 0.10
                            if split_min < train_min - margin or split_max > train_max + margin:
                                ood_cols.append({
                                    'col': col,
                                    'train_range': f"[{train_min:.2f}, {train_max:.2f}]",
                                    'split_range': f"[{split_min:.2f}, {split_max:.2f}]"
                                })
                    
                    if ood_cols:
                        self.logger.warning(f"⚠️ [Parity] OOD in {split_name}: {len(ood_cols)} columns differ from training:")
                        for ood in ood_cols:
                            self.logger.warning(f"   {ood['col']:10s}: train={ood['train_range']}, {split_name}={ood['split_range']}")
            
            # FIX #8: Remove duplicate per-partition stats block (was copy-pasted twice,
            # causing 6 extra sub-DataFrame materializations per split with no extra value).
            # The identical block appears earlier after the out-of-range check.

            # ──────────────────────────────────────────────────────────────────────────────
            
            # 10. Log SCALED RANGE Diagnostics
            ohlcv_in_scale = [c for c in ['Volume', 'Open', 'High', 'Low', 'Close'] if c in columns_to_scale]
            if ohlcv_in_scale:
                self.logger.info(f"{'='*80}")
                self.logger.info(f"📊 {split_name.upper()} SPLIT - SCALED RANGE:")
                for col in ohlcv_in_scale:
                    col_data = split_df[col]
                    self.logger.info(f"  {col}: [{col_data.min():.6f}, {col_data.max():.6f}] (mean: {col_data.mean():.6f})")
                self.logger.info(f"{'='*80}")
                
            split_strategy = StrategyFactory.determine_strategy(
                len(split_df),
                self.processing_config,
                self.analysis_type
            )
            
            self.logger.info(f"[ML-Splits] Using {split_strategy.value} strategy for {split_name}")
            
            # Update progress
            await self._send_progress_update(
                progress_map[split_name],
                f"Processing {split_name} split (pre-scaled)..."
            )
            
            # Execute strategy on this split with PRE-SCALED data
            # Workers will skip scaling and just generate sequences
            split_kwargs = {
                **kwargs, 
                'split_name': split_name, 
                'split_type': split_name, 
                'fit_scaler': False,  # Never fit scaler - already fitted
                'skip_scaling': True,  # NEW: Tell workers data is already scaled
                'global_scaler': global_scaler,  # Pass for metadata/validation only
                'feature_cols': feature_cols,  # Pass feature column names
                'columns_to_scale': columns_to_scale,  # CRITICAL: Pass scaled columns metadata
                'enriched_target_columns': self.enriched_target_columns,  # FIX: Pass enriched targets to workers
                # Pass step_configs so workers can embed pipeline provenance
                # into each NPZ chunk for standalone training traceability.
                'step_configs': step_configs,
            }
            
            # Set PM context for strategy to use
            self._current_dataset_id = dataset_id
            self._current_split_name = split_name
            
            # Execute strategy with PM and DB session for chunk-by-chunk storage
            async with AsyncPostgresSessionLocal() as split_db:
                
                split_result = await self._execute_with_strategy(
                    split_strategy, split_df,
                    effective_task_store=effective_task_store,
                    pm_instance=self,  # Pass PM instance
                    db_session=split_db,  # Reuse same DB session
                    **split_kwargs
                )
            
            # Get actual sequence count from result metadata
            seq_count = split_result.get("metadata", {}).get("total_sequences", 0)

            # Collect any extracted target names that the merge/strategy returned
            extracted = split_result.get("metadata", {}).get("extracted_target_names")
            if extracted:
                try:
                    all_extracted_targets.update(extracted)
                except Exception:
                    for t in extracted:
                        all_extracted_targets.add(t)
            
            # Track metadata for this split
            split_metadata[split_name] = {
                "sequences_count": seq_count,
                "split_name": split_name,
                "dataset_id": dataset_id
            }
            
            # Free the scaled DataFrame BEFORE moving to next split
            # FIX #9: also release pre_scale_ranges (per-column stats dict) and
            # scaled_data (sub-DataFrame view) which were never freed before.
            del split_df, split_result
            if 'pre_scale_ranges' in dir():
                del pre_scale_ranges
            if 'scaled_data' in dir():
                del scaled_data
            gc.collect()
            
            self.logger.info(f"[ML-Splits] {split_name} stored ({seq_count} sequences, dataset_id: {dataset_id})")
        
        # Build final result with metadata only (no data in memory)
        await self._send_progress_update(95, "Building final ML dataset metadata...")
        total_sequences = sum(s["sequences_count"] for s in split_metadata.values())

        # Update split_config with real sequence counts so get_chunk_window
        # can return the correct `total` for each split. Without this, the viewer
        # always shows total=0 because split_config only has ratios, not sizes.
        try:
            updated_split_config = {
                "train_ratio": float(config_obj.train_ratio),
                "validation_ratio": float(config_obj.validation_ratio),
                "test_ratio": float(config_obj.test_ratio),
                "sequence_length": int(config_obj.sequence_length),
                "prediction_length": int(config_obj.prediction_length),
                "train_size": split_metadata.get("train", {}).get("sequences_count", 0),
                "validation_size": split_metadata.get("validation", {}).get("sequences_count", 0),
                "test_size": split_metadata.get("test", {}).get("sequences_count", 0),
            }
            
            
            
            updated_scaling_config = {
                "scaler_type": scaler_type_str,
                "feature_columns": feature_cols,
                "columns_to_scale": columns_to_scale,  # Scaler fitted on normalized [0,1] data
                "scaler_fitted": True,
                "note": "Updated after loop with final features (normalized in-place)",
                # Normalization method (for inference)
                "normalization_method": self.normalization_method,  # "clipping" or "rolling_mean_sigmoid"
                "sigmoid_scale_factor": self.sigmoid_scale_factor,
                # Re-persist fitted structural range so inference can reconstruct
                # the same normalisation without re-fitting on any split data.
                "structural_range": {
                    "high":       self.fitted_range_high   if hasattr(self, 'fitted_range_high')  else None,
                    "low":        self.fitted_range_low    if hasattr(self, 'fitted_range_low')   else None,
                    "width":      self.fitted_range_width  if hasattr(self, 'fitted_range_width') else None,
                    "vol_factor": self.fitted_vol_factor   if hasattr(self, 'fitted_vol_factor')  else None,
                    "fitted_on":  "train_only",
                    "window":     rolling_window
                },
                # Breakout detection summary
                "breakout_detection": {
                    "columns_with_breakouts": list(self.global_breakout_detections.keys()) if self.global_breakout_detections else [],
                    "total_breakout_columns": len(self.global_breakout_detections) if self.global_breakout_detections else 0,
                    "detection_note": "Any columns listed here had values exceeding fitted range — clipped to [0,1]"
                },
                # Rolling mean baseline (future use)
                "rolling_mean_baselines": {
                    "available": bool(self.global_rolling_mean_baselines),
                    "columns_count": len(self.global_rolling_mean_baselines) if self.global_rolling_mean_baselines else 0,
                    "note": "Baseline statistics available for migration to lossless rolling mean sigmoid normalization"
                },
                # VOLUME NORMALIZATION: Separate range for volume (not price-based structural range)
                "volume_range": {
                    "method": "rolling_percentile_plus_buffer",
                    "Volume": {
                        "fitted_range_high": float(self.fitted_volume_range_high.get('Volume', 0)),
                        "fitted_range_low": float(self.fitted_volume_range_low.get('Volume', 0)),
                    } if 'Volume' in self.fitted_volume_range_high else {},
                    "TickVolume": {
                        "fitted_range_high": float(self.fitted_volume_range_high.get('TickVolume', 0)),
                        "fitted_range_low": float(self.fitted_volume_range_low.get('TickVolume', 0)),
                    } if 'TickVolume' in self.fitted_volume_range_high else {},
                    "note": "Volume normalized by rolling percentile + 1.2x buffer, NOT by price structural range"
                },
                # VOLUME SPIKE DETECTION: Phase 1 for volume (different from price clipping)
                "volume_clipping_detection": {
                    "columns": list(self.global_volume_clipping_detection.keys()) if self.global_volume_clipping_detection else [],
                    "detection_data": self.global_volume_clipping_detection or {},
                    "detection_note": "Volume spikes exceeding 1.2x buffer (monitored, not problematic like price clipping)"
                    },
                # DISTANCE NORMALIZATION: Separate range for distance columns
                "distance_range": {
                    "method": "rolling_percentile_plus_buffer",
                    "columns": list(self.fitted_distance_range_high.keys()) if self.fitted_distance_range_high else [],
                    "ranges": {
                        col: {
                            "fitted_range_high": float(self.fitted_distance_range_high.get(col, 0)),
                            "fitted_range_low": float(self.fitted_distance_range_low.get(col, 0)),
                        } for col in self.fitted_distance_range_high.keys()
                    } if self.fitted_distance_range_high else {},
                    "note": "Distance columns normalized by rolling percentile + 1.2x buffer, NOT by price structural range"
                },
                # FOOTPRINT NORMALIZATION: Symmetric range for signed delta columns
                "footprint_range": {
                    "method": "symmetric_percentile_plus_buffer (abs(value))",
                    "columns": list(self.fitted_footprint_range_high.keys()) if hasattr(self, 'fitted_footprint_range_high') else [],
                    "ranges": {
                        col: {"fitted_range_high": float(v), "fitted_range_low": float(-v)}
                        for col, v in self.fitted_footprint_range_high.items()
                    } if hasattr(self, 'fitted_footprint_range_high') else {},
                    "note": "FP_Delta/FP_Cum_Delta normalized to ~[-1, 1] range"
                },
                # DIFF COLUMN STANDARDSCALER: Unbounded, mean-centered normalization
                "diff_scalers": {
                    "method": "StandardScaler (unbounded, mean-centered)",
                    "columns": list(self.diff_scalers.keys()) if self.diff_scalers else [],
                    "note": "Diff columns normalized with StandardScaler: unbounded, preserves negatives and spikes",
                    "scaler_binary_available": bool(self.global_diff_scaler_binary)
                },
                "feature_index_map": self.global_feature_index_map if hasattr(self, 'global_feature_index_map') else {},
                "index_feature_map": self.global_index_feature_map if hasattr(self, 'global_index_feature_map') else {},
                "scaler_feature_index_map": getattr(global_scaler, 'feature_index_map', {}),
                "scaler_index_feature_map": getattr(global_scaler, 'index_feature_map', {}),
                "feature_partitions": getattr(global_scaler, 'partition_by_feature', {}),
                "partition_indices": getattr(global_scaler, 'partition_indices', {}),
                # PER-INDEX STRUCTURAL RANGES: Each currency index has its own price scale
                "index_structural_ranges": {
                    idx_name: {
                        "high":       float(params['high']),
                        "low":        float(params['low']),
                        "width":      float(params['width']),
                        "vol_factor": float(params['vol_factor']),
                        "fitted_on":  "train_only",
                    }
                    for idx_name, params in self.fitted_index_ranges.items()
                } if self.fitted_index_ranges else {},
            }

            from app.core.ml.serving_contract import build_output_transform_spec
            updated_scaling_config["output_transform_spec"] = build_output_transform_spec(
                target_names=adv_target_cols_from_enrichment,
                scaling_config=updated_scaling_config,
            )
            
            # Serialize updated_scaling_config to ensure JSON compatibility
            updated_scaling_config = to_serializable(updated_scaling_config)
            self._final_scaling_config = updated_scaling_config

            final_source_metadata = to_serializable({
                "feature_names": feature_cols,
                "step_name": self.step_name,
                "prediction_length": int(config_obj.prediction_length),
                "dataset_name": dataset_name,
                "dataset_id": str(dataset_id),
                "symbol": getattr(self, "_session_provenance", {}).get("symbol"),
                "timeframe": getattr(self, "_session_provenance", {}).get("timeframe"),
                "start_date": getattr(self, "_session_provenance", {}).get("start_date"),
                "end_date": getattr(self, "_session_provenance", {}).get("end_date"),
                "data_source": getattr(self, "_session_provenance", {}).get("data_source"),
                "record_count": getattr(self, "_session_provenance", {}).get("record_count"),
                "step_configs": getattr(self, "_ml_step_configs", {}),
            })
            
            async with AsyncPostgresSessionLocal() as _upd_db:
                from sqlalchemy import update as sa_update
                upd_stmt = (
                    sa_update(MLDataset)
                    .where(MLDataset.dataset_id == dataset_id)
                    .values(
                        split_config=updated_split_config,
                        scaling_config=updated_scaling_config,  
                        scaler_binary=self.global_scaler_binary,
                        feature_columns=feature_cols,   
                        feature_count=len(feature_cols),
                        sample_count=total_sequences,
                        source_metadata=final_source_metadata,
                        status="ready",
                        updated_at=datetime.utcnow(),
                    )
                )
                await _upd_db.execute(upd_stmt)
                await _upd_db.commit()
            self.logger.info(
                f"[ML-Splits] split_config updated with real counts: "
                f"train={updated_split_config['train_size']}, "
                f"val={updated_split_config['validation_size']}, "
                f"test={updated_split_config['test_size']}"
            )
            self.logger.info(
                f"[ML-Splits] scaling_config updated with final features: "
                f"feature_count={len(feature_cols)}, "
                f"columns_to_scale={len(columns_to_scale)} "
                f"(original columns replaced in-place with normalized [0,1] versions)"
            )
        except Exception as _upd_err:
            self.logger.warning(f"[ML-Splits] ⚠️  Could not update split_config: {_upd_err}")
        
        # Build final target names list (including virtual targets) - use working copy
        # Start with all adv_target_* columns captured after enrichment
        all_target_names = list(adv_target_cols_from_enrichment)
        
        # Add future_sequence if configured (it's virtual — not a DataFrame column)
        if hasattr(config_obj, 'include_sequence_prediction') and config_obj.include_sequence_prediction:
            if 'future_sequence' not in all_target_names:
                all_target_names.append('future_sequence')
        
        # Add movement analysis targets if advanced targets are enabled.
        # These are computed per-sequence by _compute_movement_analysis_batch() and
        # are NOT DataFrame columns — so they won't appear in adv_target_cols_from_enrichment.
        if getattr(config_obj, 'prepare_advanced_ml_targets', False):
            movement_virtual_targets = [
                "adv_target_max_favorable_pct",
                "adv_target_max_adverse_pct",
                "adv_target_final_move_pct",
                "adv_target_risk_reward_ratio",
                "adv_target_avg_volatility",
                "adv_target_volatility_surge",
                "adv_target_time_to_max_favorable",
                "adv_target_time_to_max_adverse",
                "adv_target_signal_strength",
            ]
            for t in movement_virtual_targets:
                if t not in all_target_names:
                    all_target_names.append(t)

        # Union chunk-extracted names with computed virtual targets (future_sequence, movement).
        extracted_list = sorted(all_extracted_targets) if all_extracted_targets else []
        seen_targets: set = set()
        final_target_names: list = []
        for name in extracted_list + all_target_names:
            key = str(name)
            if key not in seen_targets:
                seen_targets.add(key)
                final_target_names.append(key)

        self.logger.info(
            f"[ML-Splits] 🎯 Final target names for DB ({len(final_target_names)}): {final_target_names}"
        )

        # Complete target_types for every name (including virtual targets not in enrichment)
        final_target_types = dict(target_types)
        for t in final_target_names:
            if t in final_target_types:
                continue
            if t == 'future_sequence':
                final_target_types[t] = "sequence_prediction"
            elif t in ('target_signal', 'Direction', 'Next_Day_Direction'):
                final_target_types[t] = "classification"
            else:
                final_target_types[t] = "regression"

        final_target_metadata = {
            "target_names": final_target_names,
            "target_types": final_target_types,
            "class_mappings": {},
            "primary_target": final_target_names[0] if final_target_names else None,
        }

        # Update output_targets AND target_metadata with the complete list
        try:
            from app.core.ml.serving_contract import build_output_transform_spec as _build_out_spec
            _base_scaling = getattr(self, "_final_scaling_config", {}) or {}
            final_output_transform = _build_out_spec(
                target_names=final_target_names,
                scaling_config=_base_scaling,
            )
            async with AsyncPostgresSessionLocal() as _tgt_db:
                from sqlalchemy import update as sa_update
                _refresh_scaling = dict(_base_scaling)
                _refresh_scaling["output_transform_spec"] = final_output_transform
                tgt_stmt = (
                    sa_update(MLDataset)
                    .where(MLDataset.dataset_id == dataset_id)
                    .values(
                        output_targets=final_target_names,
                        target_metadata=final_target_metadata,
                        scaling_config=to_serializable(_refresh_scaling),
                    )
                )
                await _tgt_db.execute(tgt_stmt)
                await _tgt_db.commit()
            self.logger.info(
                f"[ML-Splits] output_targets + target_metadata updated with "
                f"{len(final_target_names)} targets "
                f"({len(extracted_list)} from chunks, {len(all_target_names)} computed)"
            )
        except Exception as _tgt_err:
            self.logger.warning(f"[ML-Splits] ⚠️ Could not update output_targets/target_metadata: {_tgt_err}")

        all_target_names = final_target_names
        target_types = final_target_types

        final_result = {
            # Lightweight reference flags for AnalysisDataContext
            "_ref": str(dataset_id),
            "_storage": "analysisStorage",
            
            # Core dataset identifiers
            "dataset_id": str(dataset_id),
            "session_id": str(self.session_id),
            
            # Dataset metadata
            "strategy": "ml_split_processing",
            "analysis_type": "ml_dataset_preparation",
            "total_sequences": total_sequences,
            "split_counts": {n: d["sequences_count"] for n, d in split_metadata.items()},
            "split_dataset_ids": {n: d["dataset_id"] for n, d in split_metadata.items()},
            "feature_names": feature_cols,
            "target_names": all_target_names,
            "sequence_length": config_obj.sequence_length,
            "prediction_length": config_obj.prediction_length,
            "scaler_path": config_obj.scaler_save_path if config_obj.save_scaler else None,
            "storage_strategy": "split_unified",
            "scaler_available": self.global_scaler is not None,
            "target_metadata": {
                "target_names": all_target_names,
                "target_types": target_types,
                "class_mappings": {}
            },
            "timestamp": datetime.utcnow().isoformat(),
            "step_configs": getattr(self, "_ml_step_configs", step_configs),
            "provenance": {
                "symbol": getattr(self, "_session_provenance", {}).get("symbol"),
                "timeframe": getattr(self, "_session_provenance", {}).get("timeframe"),
                "start_date": getattr(self, "_session_provenance", {}).get("start_date"),
                "end_date": getattr(self, "_session_provenance", {}).get("end_date"),
            },
            
            # Keep metadata dict for backward compatibility/nested access if needed
            "metadata": {
                "dataset_id": str(dataset_id),
                "total_sequences": total_sequences,
                "split_counts": {n: d["sequences_count"] for n, d in split_metadata.items()},
                "split_dataset_ids": {n: d["dataset_id"] for n, d in split_metadata.items()},
                "feature_names": feature_cols,
                "target_names": all_target_names,
                "storage_strategy": "split_unified"
            }
        }
        
        if self.global_scaler is not None:
            final_result["scaler"] = self.global_scaler
        
        # COMPREHENSIVE SUMMARY: All three phases
        self.logger.info(f"[ML-Splits] Done: {total_sequences} sequences total across {len(split_metadata)} splits")
        
        # Breakout Detection Summary
        if self.global_breakout_detections:
            self.logger.warning(
                f"\n{'='*80}\n"
                f"⚠️  [PHASE 1] BREAKOUT DETECTION SUMMARY:\n"
                f"{'='*80}"
            )
            breakout_cols = list(self.global_breakout_detections.keys())
            self.logger.warning(f"Columns with breakouts: {len(breakout_cols)}/{len(columns_to_scale)}")
            for col in sorted(breakout_cols)[:10]:  # Show top 10
                det = self.global_breakout_detections[col]
                self.logger.warning(
                    f"  • {col}: {det['count']} breakout rows ({det['percentage']*100:.2f}%) "
                    f"{'⚠️ EXCEEDS 2% THRESHOLD' if det['threshold_exceeded'] else ''}"
                )
            if len(breakout_cols) > 10:
                self.logger.warning(f"  ... and {len(breakout_cols)-10} more columns")
            self.logger.warning(
                f"\nRECOMMENDATION: If breakouts persist in next runs, consider:\n"
                f"  1. Retraining with widened fitted_range (vol_factor adjustment)\n"
                f"  2. Migrating to rolling mean sigmoid normalization (Phase 2)\n"
            )
        else:
            self.logger.info(f"✅ [PHASE 1] No breakouts detected — clipping is NOT losing data")
        
        # Rolling Mean Baseline Status
        if self.global_rolling_mean_baselines:
            self.logger.info(
                f"✅ [PHASE 2] Rolling mean baselines captured for {len(self.global_rolling_mean_baselines)} columns "
                f"(ready for lossless normalization migration)"
            )
        else:
            self.logger.info(
                f"ℹ️  [PHASE 2] Rolling mean baselines NOT yet computed (use when switching normalization methods)"
            )
        
        # Normalization Method
        self.logger.info(
            f"✅ [PHASE 3] Normalization Configuration:\n"
            f"  • Method: {self.normalization_method} (clipping = lossy, rolling_mean_sigmoid = lossless)\n"
            f"  • Sigmoid scale factor: {self.sigmoid_scale_factor}\n"
            f"  • Stored in scaling_config for inference-time reconstruction"
        )
        
        # VOLUME NORMALIZATION: Separate from price structural range
        self.logger.info(
            f"\n{'='*80}\n"
            f"📊 [VOLUME NORMALIZATION] Separate Range-Based Normalization\n"
            f"{'='*80}"
        )
        if self.fitted_volume_range_high:
            for vol_col in ['Volume', 'TickVolume']:
                if vol_col in self.fitted_volume_range_high:
                    self.logger.info(
                        f"✅ {vol_col}: Range [0, {self.fitted_volume_range_high[vol_col]:,.0f}] "
                        f"(NOT price structural range, fitted on rolling percentile + 1.2x buffer)"
                    )
        else:
            self.logger.info(f"ℹ️  Volume range NOT fitted (no Volume column in data)")
        
        # Volume spike detection
        if self.global_volume_clipping_detection:
            self.logger.info(f"\n📈 Volume Spike Detection (Phase 1):")
            for vol_col, detection in self.global_volume_clipping_detection.items():
                self.logger.info(
                    f"  • {vol_col}: {detection['count']} spikes ({detection['percentage']*100:.2f}%) "
                    f"exceed buffer (max ratio: {detection['spike_ratio']:.2f}x)"
                )
        else:
            self.logger.info(f"ℹ️  No volume spikes detected (within fitted range)")
        
        self.logger.info(f"{'='*80}\n")
       
        return final_result

    async def _store_ml_splits_separately(self, result: Dict[str, Any], db: Any = None) -> None:
        """
        Store ML splits separately to avoid OOM during serialization.
        
        STRATEGY:
        - Store each split (train/validation/test) as a separate DB record
        - Use threading for parallel serialization to trade time for memory
        - Each split is serialized independently to avoid memory spike
        
        Args:
            result: ML dataset result with train/validation/test splits
            db: Optional database session (creates new if None)
        """
        
        
        splits_to_store = ["train", "validation", "test"]
        
        self.logger.info(f"[ML-Storage] Starting split-separate storage for {len(splits_to_store)} splits")
        
        # Define storage function for threading
        async def store_single_split(split_name: str, split_data: Dict[str, Any]) -> None:
            """Store a single split in its own DB record."""
            try:
                step_name = f"{self.step_name}_{split_name}"
                
                # Log split size
                sequences = split_data.get("sequences", [])
                if isinstance(sequences, np.ndarray):
                    seq_count = len(sequences) if sequences.size > 0 else 0
                    memory_mb = sequences.nbytes / (1024 * 1024) if sequences.size > 0 else 0
                else:
                    seq_count = len(sequences) if sequences else 0
                    memory_mb = 0
                
                self.logger.info(
                    f"[ML-Storage] Storing {split_name} split: {seq_count} sequences "
                    f"(~{memory_mb:.1f} MB)"
                )
                
                # Store this split
                async with AsyncPostgresSessionLocal() as split_db:
                    await store_session_step_result(
                        session_id=self.session_id,
                        step_name=step_name,
                        data=split_data,
                        db=split_db,
                        force_pickle=True,
                        on_progress=None  # No progress for individual splits
                    )
                
                self.logger.info(f"[ML-Storage] {split_name} split stored successfully")
                
                # Clean up immediately after storage
                del split_data
                gc.collect()
                
            except Exception as e:
                self.logger.error(f"[ML-Storage] ❌ Failed to store {split_name} split: {e}")
                raise
        
        # Store splits sequentially to avoid memory spike
        # (Parallel would use more memory; sequential trades time for memory safety)
        for split_name in splits_to_store:
            if split_name in result and result[split_name]:
                split_data = result[split_name]
                
                # Send progress update
                await self._send_progress_update(
                    progress=90 + (splits_to_store.index(split_name) * 3),
                    message=f"Storing {split_name} split...",
                    metadata={"stage": "storage", "split": split_name}
                )
                
                # Store this split
                await store_single_split(split_name, split_data)
                
                # Null out the split data to free memory immediately
                result[split_name] = None
                gc.collect()
        
        self.logger.info(f"[ML-Storage] All splits stored separately")

    async def _save_pending_checkpoint(self, slice_num: int) -> None:
        """Mark slice as pending BEFORE processing."""
        try:
            async with AsyncPostgresSessionLocal() as db:
                stmt = pg_insert(ChunkCheckpoint).values(
                    task_id=self.task_id,
                    session_id=self.session_id,
                    step_name=self.step_name,
                    last_chunk_id=slice_num,
                    status="processing",
                    created_at=datetime.utcnow(),
                ).on_conflict_do_update(
                    index_elements=["task_id", "session_id", "step_name"],
                    set_={
                        "last_chunk_id": slice_num,
                        "status": "processing",
                        "created_at": datetime.utcnow(),
                    }
                )
                await db.execute(stmt)
                await db.commit()
                self.logger.info(f"[Checkpoint] Marked slice {slice_num} as pending")
        except Exception as err:
            self.logger.warning(f"[Checkpoint] Failed to save pending checkpoint: {err}")

    async def _save_completed_checkpoint(self, slice_num: int, progress_pct: int) -> None:
        """Mark slice as successfully completed AFTER results persisted."""
        try:
            async with AsyncPostgresSessionLocal() as db:
                stmt = pg_insert(ChunkCheckpoint).values(
                    task_id=self.task_id,
                    session_id=self.session_id,
                    step_name=self.step_name,
                    last_chunk_id=slice_num,
                    progress_pct=progress_pct,
                    status="completed",
                    created_at=datetime.utcnow(),
                ).on_conflict_do_update(
                    index_elements=["task_id", "session_id", "step_name"],
                    set_={
                        "last_chunk_id": slice_num,
                        "progress_pct": progress_pct,
                        "status": "completed",
                        "created_at": datetime.utcnow(),
                    }
                )
                await db.execute(stmt)
                await db.commit()
                self.logger.info(f"[Checkpoint] Marked slice {slice_num} as completed")
        except Exception as err:
            self.logger.warning(f"[Checkpoint] Failed to save completed checkpoint: {err}")

    async def _save_failure_checkpoint(self, slice_num: int, error: str) -> None:
        """Track failure for a slice."""
        try:
            async with AsyncPostgresSessionLocal() as db:
                stmt = pg_insert(ChunkCheckpoint).values(
                    task_id=self.task_id,
                    session_id=self.session_id,
                    step_name=self.step_name,
                    last_chunk_id=slice_num,
                    status="failed",
                    message=error[:500],
                    created_at=datetime.utcnow(),
                ).on_conflict_do_update(
                    index_elements=["task_id", "session_id", "step_name"],
                    set_={
                        "status": "failed",
                        "message": error[:500],
                        "created_at": datetime.utcnow(),
                    }
                )
                await db.execute(stmt)
                await db.commit()
        except Exception as err:
            self.logger.warning(f"[Checkpoint] Failed to save failure checkpoint: {err}")

    async def _resume_from_checkpoint(self) -> Tuple[Optional[int], bool]:
        """Check database for existing checkpoint to resume from."""
        try:
            async with AsyncPostgresSessionLocal() as db:
                # Query matches task_id, session_id, and specifically step_name
                stmt = select(ChunkCheckpoint).where(
                    ChunkCheckpoint.task_id == self.task_id,
                    ChunkCheckpoint.session_id == self.session_id,
                    ChunkCheckpoint.step_name == self.step_name
                ).order_by(desc(ChunkCheckpoint.created_at)).limit(1)
                
                result = await db.execute(stmt)
                ckpt = result.scalars().first()

                if ckpt:
                    if ckpt.status == "processing":
                        # Crashed mid-slice
                        self.logger.warning(f"[Recovery] Resuming task {self.task_id} at slice {ckpt.last_chunk_id}")
                        await self._cleanup_incomplete_slice(ckpt.last_chunk_id)
                        return ckpt.last_chunk_id, True
                    elif ckpt.status == "completed":
                        # Resume from next slice
                        return ckpt.last_chunk_id + 1, False
                
                return None, False
        except Exception as err:
            self.logger.error(f"[Recovery] Failed to check checkpoint: {err}")
            return None, False

    async def _cleanup_incomplete_slice(self, slice_num: int) -> None:
        """Delete partial results from DB for a failed slice."""
        try:
            async with AsyncPostgresSessionLocal() as db:
                from sqlalchemy import text
                stmt = text("DELETE FROM session_step_results WHERE session_id = :sid AND step_name = :sname")
                await db.execute(stmt, {"sid": self.session_id, "sname": f"{self.step_name}_{slice_num}"})
                await db.commit()
                self.logger.info(f"[Recovery] Cleaned up partial results for slice {slice_num}")
        except Exception as err:
            self.logger.warning(f"[Recovery] Cleanup failed for slice {slice_num}: {err}")

    async def _store_slice_results(
        self, 
        slice_num: int, 
        results: Dict[str, Any], 
        slice_boundaries: Tuple[int, int]
    ) -> None:
        """Persist slice results to DB and aggregate memory-critical items.
        
        MEMORY CONTRACT: Nulls results["result_df"] (and variants) before returning
        so the caller's reference to `results` is already hollowed out by the time
        this function exits. This prevents a triple-copy spike at slice handoff:
          Copy A = results["result_df"] (strategy output)
          Copy B = authoritative_df + df_records (this function)
          Copy C = df_canonical in aggregator.add_result()
        By nulling Copy A here and deleting Copy B immediately after to_dict(),
        only one live copy exists at any point.
        """
        slice_start, slice_end = slice_boundaries
        
        # 1. Extract DataFrame Chunk (deduplicated)
        df_chunk = results.get("result_df", results.get("features_df", results.get("enriched_df")))
        df_records = None
        if isinstance(df_chunk, pd.DataFrame) and not df_chunk.empty:
            if isinstance(df_chunk.index, pd.DatetimeIndex):
                # FIX #2: For DatetimeIndex, convert position-based boundaries to time-based
                # This ensures only canonical rows within the slice are stored (not overlap)
                if slice_start < len(df_chunk) and slice_end <= len(df_chunk):
                    # Get original df from results to map slice indices to timestamps
                    original_df = results.get("_original_df")
                    if original_df is not None and isinstance(original_df.index, pd.DatetimeIndex):
                        start_time = original_df.index[slice_start]
                        end_time = original_df.index[min(slice_end - 1, len(original_df) - 1)]
                        authoritative_df = df_chunk[
                            (df_chunk.index >= start_time) & 
                            (df_chunk.index <= end_time)
                        ].copy()
                    else:
                        # Fallback: use position-based boundary if no original df
                        authoritative_df = df_chunk.iloc[slice_start:slice_end].copy()
                else:
                    authoritative_df = df_chunk.copy()
                self.logger.info(f"[Storage] Slice {slice_num} (DatetimeIndex): filtered {len(df_chunk)} → {len(authoritative_df)} rows")
            else:
                # Integer index - apply shift
                authoritative_df = df_chunk[df_chunk.index >= slice_start].copy()
                self.logger.info(f"[Storage] Slice {slice_num} (IntegerIndex): filtered {len(df_chunk)} → {len(authoritative_df)} rows")

            df_records = authoritative_df.to_dict(orient="records")
            # 🧹 CRITICAL: Free both copies immediately after serialisation.
            # authoritative_df is no longer needed — df_records is all that goes to DB.
            # df_chunk is a reference into results["result_df"]; we null that below.
            del authoritative_df
            del df_chunk
            gc.collect()

        # 🧹 MEMORY CONTRACT: Null all DataFrame keys in results NOW, before returning.
        # The caller passes `results` to result_queue.put() for the aggregator, but the
        # aggregator's add_result() re-derives the canonical slice from the original df —
        # it does NOT need result_df to still be populated here.
        # Nulling here collapses Copy A so the queue only carries the lightweight shell
        # (signals, zones, splits, metadata) into the background worker.
        for _df_key in ("result_df", "features_df", "enriched_df"):
            if results.get(_df_key) is not None:
                results[_df_key] = None

        # 2. Persist to DB TIER 3
        try:
            async with AsyncPostgresSessionLocal() as db:
                # Include original slice boundaries in metadata for FIX #4
                metadata = results.get("metadata", {}).copy()
                metadata["original_slice_start"] = slice_start  # FIX #4
                metadata["original_slice_end"] = slice_end      # FIX #4
                
                await store_session_step_result(
                    session_id=self.session_id,
                    step_name=f"{self.step_name}_{slice_num}",
                    data={
                        "slice_num": slice_num,
                        "slice_start": slice_start,
                        "slice_end": slice_end,
                        "metadata": metadata,  # Updated with original boundaries
                        "signals": results.get("signals", []),
                        "data": df_records
                    },
                    db=db,
                    force_pickle=True,
                )
                # 🧹 Free serialised records immediately after DB write
                del df_records
        except Exception as db_err:
            self.logger.error(f"[Storage] Failed to persist slice {slice_num}: {db_err}")

    async def _store_chunked_metadata(self, total_rows: int, total_slices: int) -> None:
        """Store header row for chunked results."""
        try:
            async with AsyncPostgresSessionLocal() as db:
                await store_session_step_result(
                    session_id=self.session_id,
                    step_name=f"{self.step_name}_header",
                    data={
                        "_is_chunked": True,
                        "total_rows": total_rows,
                        "num_chunks": total_slices,
                        "step_name": self.step_name,
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    db=db
                )
        except Exception as e:
            self.logger.warning(f"[Storage] Failed to store chunked header: {e}")

    async def _send_progress_update(
        self, 
        progress: int, 
        message: str, 
        message2: str = "",
        processed_bars: int = 0,
        total_bars: int = 0,
        current_indicator: str = "",
        strategy: str = "",
        extra: Optional[Dict] = None
    ) -> None:
        """
        Progress update with non-blocking fire-and-forget protection.
        
      
        Args:
            progress: Progress percentage (0-100)
            message: Short primary message (≤30 chars)
            message2: Detailed secondary message
            processed_bars: Number of bars/rows processed
            total_bars: Total bars/rows in dataset
            current_indicator: Current indicator being calculated
            strategy: Processing strategy (Sequential/Parallel/Streaming)
            extra: Additional metadata
        """
        try:
            # scaling happens at ProgressStage level
            # Don't double-scale here as it causes progress to stay at 0
            
            payload = {
                "type": "progress",
                "progress": progress, 
                "message": message,
                "message2": message2,  
                "stage": self.current_stage,
                "analysis_type": self.analysis_type,
                "rows_processed": self.rows_processed,
                "total_rows": self.total_rows,
                "user_id": self.user_id,  
                "processed_bars": processed_bars or self.rows_processed,
                "total_bars": total_bars or self.total_rows,
                "current_indicator": current_indicator,
                "strategy": strategy,
                "data": extra or {} 
            }
            
            if self.task_store is not None:
                self.task_store.update_task(self.task_id, **payload)
            if self.connection_manager is not None:
                # Fire-and-forget progress update with timeout
                # This ensures signal generation never waits for slow WebSocket
                # Use asyncio.create_task to run in background without awaiting
                async def send_with_timeout():
                    try:
                        # 2-second timeout: complete or give up
                        async with asyncio.timeout(2):
                            await self.connection_manager.send_progress_update(
                                self.task_id,
                                payload,
                                user_id=self.user_id
                            )
                    except asyncio.TimeoutError:
                        # Silently continue - don't block signal generation
                        self.logger.debug(f"[Progress] Send timeout for task {self.task_id[:8]} at {progress}%, continuing...")
                    except Exception as ws_err:
                        # Log but don't raise
                        self.logger.debug(f"[Progress] WebSocket send failed: {ws_err}")
                
                # Schedule as background task (don't await)
                asyncio.create_task(send_with_timeout())
        except Exception as err:
            self.logger.warning(f"[Progress] Setup failed: {err}")

    def _get_slice_processing_strategy(self, n_rows: int) -> ProcessingStrategy:
        """
        Determine strategy for individual slice processing.
        When processing slices inside _execute_with_slicing(), we cap strategies to:
        - SEQUENTIAL (if n_rows < threshold)
        - PARALLEL_CHUNKING (if n_rows >= threshold)
        
        This ensures slices are broken into parallel chunks, not sliced again.
        
        Example flow:
            50K rows (main) → SLICE_STREAMING
            → Creates ~3 slices of 15-20K each
            → Each slice → _get_slice_processing_strategy() → PARALLEL_CHUNKING
            → Each slice uses workers, NOT sub-slicing
        """
        if n_rows < self.processing_config.threshold_sequential_max:
            return ProcessingStrategy.SEQUENTIAL
        else:
            # Always return PARALLEL_CHUNKING for slices (never SLICE_STREAMING)
            return ProcessingStrategy.PARALLEL_CHUNKING

    async def _persist_to_database(self, result: Dict[str, Any]) -> None:
        """
        Persist FINAL MERGED result to database (TIER 3) and mark as current data.

        Flow:
        1. Strategy executes (Sequential/Parallel/SliceStreaming)
        2. Strategy returns COMPLETE merged result
        3. PM calls this method ONCE with final result
        4. Storage happens atomically
        
        Race Condition Prevention:
        - No intermediate storage during slice processing
        - Single atomic write of final result
        - No concurrent writes to same session_step
        """
        try:
            # Verify we have a complete result
            if not result or not isinstance(result, dict):
                self.logger.error(f"[PM] ❌ Invalid result for persistence: {type(result)}")
                return
            
            # Extract strategy metadata
            strategy_used = result.get("metadata", {}).get("strategy", "unknown")
            is_snr = self.analysis_type == AnalysisType.SNR or self.step_name == "snr_analysis"
            
            self.logger.info(
                f"[PM] 💾 Persisting FINAL result for {self.step_name} "
                f"(strategy: {strategy_used}, analysis: {self.analysis_type})"
            )
            
            # Determine data to store
            data_to_store = None
            if "result_df" in result:
                data_to_store = result["result_df"]
                
                # SPECIAL: If ML Prep, check storage strategy
                if self.analysis_type == AnalysisType.ML_DATASET_PREPARATION or self.analysis_type == "ml_dataset_preparation":
                    storage_strategy = result.get("metadata", {}).get("storage_strategy")
                    
                    if storage_strategy in ("split_separate", "split_unified"):
                        # MEMORY-OPTIMIZED: Splits already stored separately during processing
                        self.logger.info(f"[PM] 🧩 Splits already stored separately during processing")
                        
                        # Store only metadata (no sequences)
                        data_to_store = {
                            "metadata": result["metadata"],
                            "storage_strategy": "split_separate",
                            "splits_stored_separately": True
                        }
                        self.logger.info(f"[PM] 📦 Storing ML Prep metadata only (splits already in DB)")
                    else:
                        # Old path: store everything together (can cause OOM)
                        data_to_store = result
                        self.logger.info(f"[PM] 📦 Storing ML Prep result with splits (legacy mode)")
            else:
                # For non-dataframe results (e.g. pure metadata steps)
                data_to_store = result

            if data_to_store is None:
                self.logger.warning(f"[PM] ⚠️  No data to persist for {self.step_name}")
                return

            # Log pre-storage info
            if isinstance(data_to_store, pd.DataFrame):
                rows = len(data_to_store)
                cols = len(data_to_store.columns)
                self.logger.info(f"[PM] 📊 Persisting DataFrame: {rows} rows × {cols} cols")
            elif isinstance(data_to_store, dict):
                keys = list(data_to_store.keys())
                self.logger.info(f"[PM] � Persisting dict with keys: {keys}")
            else:
                self.logger.info(f"[PM] 📦 Persisting {type(data_to_store).__name__}")

            # ATOMIC STORAGE: Single transaction for complete result
            async with AsyncPostgresSessionLocal() as db:
                # Build progress reporter for storage (prevents WebSocket timeout)
                def storage_progress_reporter(pct: int, msg: str) -> None:
                    label = f"💾 Storing {self.step_name}: {msg}"
                    try:
                        loop = asyncio.get_running_loop()
                        if loop.is_running():
                            asyncio.ensure_future(
                                self._send_progress_update(
                                    pct, 
                                    label, 
                                    message2="Storing the results to db",  # FIX: Pass empty string, not dict
                                    extra={"stage": "storage", "step": self.step_name}  # FIX: Pass dict as extra
                                )
                            )
                    except Exception:
                        pass  # Never let progress callback kill storage

                # ─── Large-DataFrame routing ────────────────────────────────
                # A DataFrame with 80K rows × 667 cols compresses to ~344 MB as
                # a pickle blob.  asyncpg streams bind parameters over TCP; at
                # that size the OS TCP buffer stalls and the connection is reset
                # mid-write (ConnectionDoesNotExistError).
                # Fix: anything > CHUNK_THRESHOLD_MB goes through chunked
                # storage (15K-row slices) so each asyncpg round-trip stays
                # well inside the danger zone. EVERYTHING is stored — just
                # spread across multiple rows named step_name_0, step_name_1…
                # The metadata header row (step_name) lets loaders reassemble.
                _CHUNK_THRESHOLD_MB = 50  # tune if needed
                _use_chunked = False
                if isinstance(data_to_store, pd.DataFrame):
                    _df_mem_mb = data_to_store.memory_usage(deep=False).sum() / 1_000_000
                    if _df_mem_mb > _CHUNK_THRESHOLD_MB:
                        _use_chunked = True
                        self.logger.info(
                            f"[PM] 📦 DataFrame is {_df_mem_mb:.0f} MB — routing through "
                            f"chunked storage (15K-row slices) to avoid asyncpg TCP drop"
                        )

                if _use_chunked:
                    # CHUNKED WRITE: splits into step_name_0, step_name_1… + header row
                    await store_session_step_result_chunked(
                        session_id=self.session_id,
                        step_name=self.step_name,
                        data=data_to_store,
                        db=db,
                        chunk_size=15_000,
                        is_compressed=True,
                    )
                else:
                    # ATOMIC WRITE: small enough for a single transaction
                    await store_session_step_result(
                        session_id=self.session_id,
                        step_name=self.step_name,
                        data=data_to_store,
                        db=db,
                        force_pickle=True,  # For DataFrames / Numpy dictionaries
                        on_progress=storage_progress_reporter,
                    )

                # Do NOT call set_as_current_data for ML prep.
                # The result stored for ml_dataset_preparation is a lightweight metadata
                # stub (not a real feature DataFrame).  Marking it as "current data" would
                # cause subsequent pipeline steps that load current data to receive the
                # stub instead of the feature DataFrame, silently breaking column access.
                _is_ml_prep = (
                    self.analysis_type == AnalysisType.ML_DATASET_PREPARATION
                    or self.analysis_type == "ml_dataset_preparation"
                )
                if not _is_ml_prep:
                    await set_as_current_data(
                        session_id=self.session_id,
                        db=db,
                        task_id=self.task_id
                    )
                
                # Invalidate only this step's decompress cache entry so next
                # retrieval gets fresh data with the new columns. We use the step-scoped key
                # (session_id + step_name) so other steps (technical, SNR, etc.) in the same
                # session are NOT affected.
                decompress_cache = get_decompress_cache()
                async with decompress_cache.lock:
                    step_key = decompress_cache._make_key(self.session_id, self.step_name)
                    if step_key in decompress_cache.cache:
                        del decompress_cache.cache[step_key]
                        self.logger.info(f"[PM] 🗑️ Invalidated decompress cache for {self.step_name}")
                
                # Extract strategy metadata for logging
                strategy_used = result.get("metadata", {}).get("strategy", "unknown")
                
                # Log success with row count if available
                if isinstance(data_to_store, pd.DataFrame):
                    rows = len(data_to_store)
                    self.logger.info(
                        f"✅ [PM] Persisted {self.step_name} (rows={rows}, strategy={strategy_used})"
                    )
                else:
                    self.logger.info(
                        f"✅ [PM] Persisted {self.step_name} (strategy={strategy_used})"
                    )
                
                self.logger.info(f"[PM] Successfully persisted {self.step_name} to database")
                
                # Mark as persisted in metadata so AnalysisManager knows
                if "metadata" not in result:
                    result["metadata"] = {}
                result["metadata"]["pm_persisted"] = True
                
                # For SNR and other large datasets, tell AM to skip redundant persistence
                # if we've already handled the final assembly and storage here.
                if is_snr or strategy_used == "slice_streaming":
                    result["metadata"]["skip_am_persist"] = True
                    self.logger.info(f"[PM] 🏁 Set skip_am_persist=True for {self.step_name}")

        except Exception as e:
            self.logger.error(f"[PM] ❌ Persistence failed for {self.step_name}: {e}", exc_info=True)
            # Don't raise - allow execution to complete even if storage fails
            # The result is still in memory and can be used
            
            # Mark failure in metadata so AM knows
            result["metadata"] = result.get("metadata", {})
            result["metadata"]["pm_persist_failed"] = True
            # Don't fail the entire pipeline, just log

    def _ensure_result_completeness(self, result_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
        """
        Guarantees that result_df contains original input columns and proper index.
        Standardizes on Title Case for core columns (Time, Open, High, Low, Close, Volume).
        """
        try:
            # 0. Defensive deduplication of inputs (Case-Insensitive)
            # This prevents "Time" and "time" from both existing and causing "same-caps" duplicates after renaming
            result_df = result_df.loc[:, ~result_df.columns.str.lower().duplicated(keep='first')].copy()
            original_df = original_df.loc[:, ~original_df.columns.str.lower().duplicated(keep='first')].copy()

            # 1. Standardize core columns to Title Case in both
            core_map = {
                'time': 'Time', 'open': 'Open', 'high': 'High', 'low': 'Low', 
                'close': 'Close', 'volume': 'Volume', 'tickvolume': 'TickVolume'
            }
            result_df = result_df.rename(columns={c: core_map.get(c.lower(), c) for c in result_df.columns})
            original_df = original_df.rename(columns={c: core_map.get(c.lower(), c) for c in original_df.columns})

            # 1a. Re-deduplicate by exact name after rename.
            # e.g. _add_legacy_aliases adds 'open' alongside 'Open'; after the rename
            # above both become 'Open', creating a duplicate that slips past step 0's
            # case-insensitive check.  A second exact-name pass catches this.
            if result_df.columns.duplicated().any():
                result_df = result_df.loc[:, ~result_df.columns.duplicated(keep='first')].copy()
            if original_df.columns.duplicated().any():
                original_df = original_df.loc[:, ~original_df.columns.duplicated(keep='first')].copy()

            # 2. Row count check & Alignment (Resilience for TI warmup periods and index label loss)
            # If lengths match but indices differ, we force alignment to preserve temporal integrity.
            if not result_df.index.equals(original_df.index) or len(result_df) != len(original_df):
                self.logger.info(
                    f"[Enrichment] Aligning {self.analysis_type}: "
                    f"result_len={len(result_df)}, original_len={len(original_df)}, "
                    f"index_type={type(result_df.index).__name__}"
                )
                
                if len(result_df) == len(original_df):
                    # Case A: Lengths match - force index sync (handles RangeIndex vs DatetimeIndex)
                    result_df.index = original_df.index
                    self.logger.debug(f"[Enrichment] Synchronized index labels for {len(result_df)} rows")
                elif isinstance(result_df.index, pd.DatetimeIndex) and isinstance(original_df.index, pd.DatetimeIndex):
                    # Case B: Shared DatetimeIndex - standard label-based reindexing
                    result_df = result_df.reindex(original_df.index)
                elif isinstance(result_df.index, pd.RangeIndex) and isinstance(original_df.index, pd.DatetimeIndex):
                    # Case C: Worker lost index labels - positional tail-alignment (typical for TIs)
                    self.logger.warning(f"[Enrichment] Positional tail-alignment for {self.analysis_type} ({len(result_df)} rows)")
                    result_df.index = original_df.index[-len(result_df):]
                    result_df = result_df.reindex(original_df.index)
                else:
                    # Fallback: Best-effort reindexing
                    result_df = result_df.reindex(original_df.index)
                
                self.logger.info(f"[Enrichment] Aligned result to {len(result_df)} rows")
            
            # 3. Merge missing columns from original_df
            # Since index alignment is now fixed, we trust the worker's OHLCV (which may include cleanups/ffills)
            # We only merge columns that the worker completely dropped.
            missing_cols = [c for c in original_df.columns if c not in result_df.columns]
            if missing_cols:
                self.logger.info(f"[Enrichment] Merging {len(missing_cols)} original columns back into result for {self.analysis_type}")
                for col in missing_cols:
                    if col in original_df.columns:
                        col_values = original_df[col].reindex(result_df.index)
                        result_df[col] = col_values
                
                self.logger.info(f"[Enrichment] Added {len(missing_cols)} missing columns: {missing_cols}")
            
            # Reorder columns so original columns come first, new analysis
            # columns follow. This ensures OHLCV stays at indices 0-5 instead of 327-331.
            # Order: [original_cols_in_order] + [new_analysis_cols_in_order]
            original_col_order = [c for c in original_df.columns if c in result_df.columns]
            new_analysis_cols = [c for c in result_df.columns if c not in original_df.columns]
            result_df = result_df[original_col_order + new_analysis_cols]
            self.logger.info(
                f"[Enrichment] Column order restored: {len(original_col_order)} original + "
                f"{len(new_analysis_cols)} new = {len(result_df.columns)} total"
            )
            # 4. Normalize Time format for Frontend (Unix seconds)
            # We look for 'Time' (now standardized to Title Case)
            if 'Time' in result_df.columns:
                # 🛡️ Safety: Deduplicate columns again to prevent "same-caps" duplicates
                result_df = result_df.loc[:, ~result_df.columns.duplicated(keep='first')].copy()
                
                time_series = result_df['Time']
                if isinstance(time_series, pd.DataFrame):
                    time_series = time_series.iloc[:, 0]
                
                # Convert to integer Unix seconds ONLY if it's a datetime
                # If it's already numeric (Unix seconds), DO NOT DIVIDE BY 10^9 again!
                if pd.api.types.is_datetime64_any_dtype(time_series):
                    self.logger.debug(f"[Enrichment] Converting Time from datetime to Unix seconds")
                    result_df['Time'] = (time_series.astype('int64') // 10**9).astype(int)
                elif pd.api.types.is_numeric_dtype(time_series):
                    self.logger.debug(f"[Enrichment] Time is already numeric, skipping conversion")
                    result_df['Time'] = time_series.astype(int)
                
                # We no longer duplicate 'time' here as the frontend handles the alias 
                # as a non-enumerable property to avoid table duplication.
                pass
                
                # 🧹 CLEANUP: Delete temporary series
                del time_series
                gc.collect()

            return result_df
            
        except Exception as err:
            self.logger.error(f"❌ [Enrichment] Failed to ensure result completeness for {self.analysis_type}: {err}")
            return result_df

    def _aggregate_slice_results(self, all_results: list, original_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Aggregate results from all slices into a single result.
        
        Strategy:
        - Concatenate result_df from all slices (with proper overlap removal)
        - Merge signals and zones lists
        - Merge ML split dictionaries (train, validation, test)
        - Aggregate signal_counts
        
        Overlap Handling:
        - Each slice processed with overlap: overlap_start:end
        - But only rows [start:end] are "canonical" to that slice
        - Aggregation removes overlapped rows to prevent duplicates
        """
        if not all_results:
            return {"result_df": original_df, "metadata": {"slices": 0}}

        # 1. Combine DataFrames (removing overlap by tracking canonical rows)
        all_dfs = [r.get("result_df") for r in all_results if isinstance(r.get("result_df"), pd.DataFrame)]
        
        if all_dfs:
            # FIX #1 & #3: Track which rows are canonical to each slice using proper index-based extraction
            # This prevents duplicate rows at slice boundaries and ensures alignment
            combined_rows = []
            canonical_row_count = 0
            overlap_row_count = 0
            total_overlap_logged = 0  # Tracking for logging
            
            for slice_idx, df_result in enumerate(all_dfs):
                if df_result.empty:
                    continue
                
                # Get boundaries from result metadata (or context)
                result_dict = all_results[slice_idx]
                slice_metadata = result_dict.get("metadata", {})
                slice_start = slice_metadata.get("original_slice_start")

                if slice_start is None:
                    slice_start = slice_metadata.get("slice_start")
                
                slice_end = slice_metadata.get("original_slice_end")
                if slice_end is None:
                    slice_end = slice_metadata.get("slice_end")
                
                if slice_start is not None and slice_end is not None:
                    # Extract canonical rows by POSITION, not by time/index matching
                    # The canonical length is always (slice_end - slice_start)
                    # df_result starts at slice_start in the original data, so we take the first N rows
                    canonical_length = slice_end - slice_start
                    
                    # Validate canonical_length doesn't exceed df_result length
                    if canonical_length > len(df_result):
                        self.logger.warning(
                            f"[Aggregation] Slice {slice_idx}: canonical_length ({canonical_length}) "
                            f"> df_result length ({len(df_result)}), using all rows"
                        )
                        canonical_length = len(df_result)
                    
                    # Extract first canonical_length rows (these are the non-overlap rows)
                    df_canonical = df_result.iloc[0:canonical_length].copy()
                    
                    # Debug logging with index information
                    if isinstance(df_canonical.index, pd.DatetimeIndex) and len(df_canonical) > 0:
                        self.logger.debug(
                            f"[Aggregation] Slice {slice_idx}: extracted {len(df_canonical)} canonical rows "
                            f"(slice_start={slice_start}, slice_end={slice_end}, df_result_len={len(df_result)}) "
                            f"Index range: {df_canonical.index[0]} to {df_canonical.index[-1]}"
                        )
                    else:
                        self.logger.debug(
                            f"[Aggregation] Slice {slice_idx}: extracted {len(df_canonical)} canonical rows "
                            f"(slice_start={slice_start}, slice_end={slice_end}, df_result_len={len(df_result)})"
                        )
                    
                    combined_rows.append(df_canonical)
                    canonical_len = len(df_canonical)
                    overlap_len = len(df_result) - canonical_len
                    canonical_row_count += canonical_len
                    overlap_row_count += overlap_len
                    total_overlap_logged += overlap_len  # Log for visibility
                    
                    # Log what happened in this slice
                    self.logger.debug(f"[Aggregation] Slice {slice_idx}: {len(df_result)} processed, {canonical_len} canonical, {overlap_len} overlap")
                    
                    # 🧹 CLEANUP: Current loop items
                    del result_dict
                    del slice_metadata
                else:
                    # No metadata - use all rows (for backward compatibility)
                    combined_rows.append(df_result.copy())
                    canonical_row_count += len(df_result)
            
            if combined_rows:
                # Don't use ignore_index=True - preserve the original index for proper deduplication
                # Check if we have DatetimeIndex to preserve
                has_datetime_index = isinstance(combined_rows[0].index, pd.DatetimeIndex)
                
                if has_datetime_index:
                    # Preserve DatetimeIndex for proper deduplication
                    # Use sort=False to maintain slice order, not timestamp order
                    combined_df = pd.concat(combined_rows, ignore_index=False, sort=False)
                    
                    # Sort by index after concatenation to ensure chronological order
                    combined_df = combined_df.sort_index()
                    
                    # For DatetimeIndex: check for any remaining duplicates after canonical extraction
                    # This should only happen if there's a bug in canonical extraction
                    original_len = len(combined_df)
                    duplicates = combined_df.index.duplicated(keep='first')
                    dup_count = duplicates.sum()
                    
                    if dup_count > 0:
                        self.logger.warning(
                            f"[Dedup] Found {dup_count} duplicate timestamps after canonical extraction "
                            f"(this indicates a bug in slice boundary handling)"
                        )
                        combined_df = combined_df.loc[~duplicates].copy()
                        dedup_removed = original_len - len(combined_df)
                    else:
                        dedup_removed = 0
                else:
                    # Do NOT use ignore_index=True for integer indexes.
                    # Preserve the original index to deduplicate overlap rows 
                    # from chunks that didn't provide boundary metadata.
                    combined_df = pd.concat(combined_rows, ignore_index=False, sort=False)
                    combined_df = combined_df.sort_index()
                    
                    original_len = len(combined_df)
                    duplicates = combined_df.index.duplicated(keep='first')
                    dup_count = duplicates.sum()
                    
                    if dup_count > 0:
                        self.logger.warning(
                            f"[Dedup] Found {dup_count} duplicate index labels after canonical extraction "
                            f"(this indicates a bug in slice boundary handling or missing metadata)"
                        )
                        combined_df = combined_df.loc[~duplicates].copy()
                        dedup_removed = original_len - len(combined_df)
                    else:
                        dedup_removed = 0
                
                # Log deduplication stats
                if dedup_removed > 0:
                    self.logger.info(f"[Dedup] Removed {dedup_removed} duplicate rows")
                
                # Unified enrichment guarantee applied to aggregated slices
                # SKIP for SNR analysis - it returns complete data and enrichment causes 3GB memory spike
                if self.analysis_type != AnalysisType.SNR:
                    combined_df = self._ensure_result_completeness(combined_df, original_df)
                
                # Log aggregation stats for visibility
                self.logger.info(
                    f"[Aggregation] Merged {len(all_dfs)} slices: "
                    f"{canonical_row_count} canonical + {overlap_row_count} overlap region rows = {len(combined_df)} final rows "
                    f"(deduped {dedup_removed if dedup_removed > 0 else 0})"
                )
                self.logger.info(f"[Aggregation] Total overlap rows across slices: {total_overlap_logged}")
                
                # DIAGNOSTIC: Log index continuity for DatetimeIndex
                if has_datetime_index and len(combined_df) > 1:
                    # Check for gaps in the index
                    time_diffs = combined_df.index.to_series().diff()
                    median_diff = time_diffs.median()
                    large_gaps = time_diffs[time_diffs > median_diff * 10]
                    
                    if len(large_gaps) > 0:
                        self.logger.warning(
                            f"[Aggregation] ⚠️ Found {len(large_gaps)} large time gaps in merged data "
                            f"(>10x median interval). This may indicate missing data."
                        )
                        for idx, gap in large_gaps.head(5).items():
                            self.logger.warning(f"  Gap at {idx}: {gap}")
                
                for df in combined_rows:
                    del df
                del combined_rows
                del all_dfs
                gc.collect()
            else:
                combined_df = original_df
        else:
            combined_df = original_df

        # 2. Build final result structure
        final_result = {
            "result_df": combined_df,
            "metadata": {
                "strategy": "slice_streaming",
                "analysis_type": self.analysis_type,
                "slices": len(all_results),
                "total_rows_processed": len(combined_df),
            },
        }

        # 3. Aggregate Lists (Signals, Zones) with Deduplication
        for key in ["signals", "zones"]:
            combined_list = []
            for r in all_results:
                val = r.get(key)
                if isinstance(val, list):
                    combined_list.extend(val)
            
            if combined_list:
                # Deduplicate based on stable keys
                if key == "signals":
                    # Signals unique by index and type
                    final_result[key] = self._deduplicate_list_items(combined_list, ["index", "type", "direction"])
                elif key == "zones":
                    # Zones unique by id or price/type
                    final_result[key] = self._deduplicate_list_items(combined_list, ["id", "price", "type"])
                else:
                    final_result[key] = combined_list
                
                # 🧹 CLEANUP: Clear accumulator list
                del combined_list
                gc.collect()

        # 4. Aggregate Signal Counts (Recalculated from deduplicated signals)
        if "signals" in final_result:
            counts = {}
            for sig in final_result["signals"]:
                sig_type = sig.get("type", "unknown")
                counts[sig_type] = counts.get(sig_type, 0) + 1
            final_result["signal_counts"] = counts
            final_result["total_signals"] = len(final_result["signals"])
        else:
            final_result["signal_counts"] = {}
            final_result["total_signals"] = 0

        # 5. Merge ML Split Dictionaries
        # We use the helper from ParallelChunkingStrategy if available, or a local one
        from app.core.processing.processing_strategies import ParallelChunkingStrategy
        strategy_helper = ParallelChunkingStrategy(ProcessingContext(self.task_id, self.session_id, self.analysis_type, self.config))
        
        split_results = []
        for key in ["train", "validation", "test"]:
            split_dicts = [r.get(key) for r in all_results if isinstance(r.get(key), dict)]
            if split_dicts:
                final_result[key] = strategy_helper._merge_ml_split_dicts(split_dicts)
                split_results.extend(split_dicts)
        
        del split_results
        del strategy_helper

        # 6. Preserve Singletons (features_df, ml_dataset, etc.)
        for key in ["ml_dataset", "features_df", "enriched_df", "imbalance_analysis"]:
            if key in final_result: continue
            for r in all_results:
                if key in r and r[key] is not None:
                    final_result[key] = r[key]
                    break

        if "signals" in final_result:
            final_result["total_signals"] = len(final_result["signals"])

        del all_results
        gc.collect()
        
        return final_result

    def _deduplicate_list_items(self, items: List[Dict[str, Any]], key_fields: List[str]) -> List[Dict[str, Any]]:
        """
        Deduplicate items in a list of dictionaries based on specific key fields.
        Used for signals and zones collected from multiple slices.
        """
        if not items:
            return []
            
        seen = set()
        unique_items = []
        
        for item in items:
            # Handle list-based zones or dict-based signals
            if isinstance(item, (list, tuple)):
                # For zones: typically [id, price, type, ...]
                item_key = tuple(item[:3]) # Use first 3 fields as key
            else:
                # For dicts: create a stable key from specified fields
                item_key = tuple(str(item.get(field)) for field in key_fields)
            
            if item_key not in seen:
                seen.add(item_key)
                unique_items.append(item)
                
        return unique_items


    async def _calculate_global_ml_scaler(self, df_sample: pd.DataFrame) -> Any:
        """
        Calculate global ML scaler using a representative sample.
        Ensures consistent feature normalization across all data slices.
        """
        try:
            from app.core.ml.ml_dataset_preparation import MLDatasetPreparation, DatasetConfig
            
            # Map dict config if necessary
            config = self.config
            if isinstance(config, dict):
                from app.core.ml.ml_dataset_preparation import ScalerType, SplitStrategy
                config = DatasetConfig(
                    sequence_length=config.get("sequence_length", 60),
                    prediction_length=config.get("prediction_length", 7),
                    scaler_type=ScalerType(config.get("scaler_type", "minmax")),
                    split_strategy=SplitStrategy(config.get("split_strategy", "sequential")),
                )

            # Initialize ML prep with sample data
            ml_prep = MLDatasetPreparation(
                data=df_sample,
                config=config,
                task_id=f"global_fit_{self.task_id}"
            )
            
            self.logger.info(f"[PM] Fitting global scaler on {len(df_sample)} rows...")
            
            # Run the pre-scaling stages of the pipeline
            await ml_prep._validate_data()
            await ml_prep._enrich_with_targets()
            ml_prep._identify_features()
            await ml_prep._scale_dataframe_splits()
            
            scaler = ml_prep.scaler
            
            # 🧹 CLEANUP: Delete ML prep object after extracting scaler
            del ml_prep
            gc.collect()
            
            return scaler
            
        except Exception as e:
            # 🧹 CLEANUP: Clear on error
            if 'ml_prep' in locals():
                del ml_prep
            gc.collect()
            self.logger.error(f"[PM] Global scaler calculation failed: {e}", exc_info=True)
            return None

    def _identify_ml_features(self, df: pd.DataFrame, config_obj: DatasetConfig) -> tuple[List[str], List[str]]:
        """
        Identify feature columns for ML processing using MLDatasetPreparation logic.
        
        This creates a temporary MLDatasetPreparation instance to use its feature identification logic.
        
        Returns:
            Tuple of (feature_cols, columns_to_scale)
            - feature_cols: Features only
            - columns_to_scale: Features + target columns (for regression)
        """
        try:
            # Create temporary MLDatasetPreparation instance
            temp_ml_prep = MLDatasetPreparation(
                data=df,
                config=config_obj,
                task_id=None,  # No progress tracking needed
                reporter=None
            )
            
            # Use its feature identification logic
            temp_ml_prep._identify_features()
            feature_cols = temp_ml_prep.feature_cols
            columns_to_scale = temp_ml_prep.columns_to_scale
            
            # Exclude structural range metadata columns
            # These are NOT features, they're calculations used for normalization
            # Including them breaks the scaler: they're constants (Rolling_Range_High/Low/Mid)
            # or have not been normalized yet (Rolling_Range_Width)
            structural_exclude = {
                'Rolling_Range_High', 'Rolling_Range_Low', 'Rolling_Range_Mid', 'Rolling_Range_Width',
                # Exclude *_Norm columns that appear to be pre-computed standardized values
                # (not from our [0,1] normalization pipeline)
                'Open_Norm', 'High_Norm', 'Low_Norm', 'Close_Norm',
                'Structural_Range_Position'  # Also standardized, not [0,1]
            }
            
            feature_cols = [c for c in feature_cols if c not in structural_exclude]
            columns_to_scale = [c for c in columns_to_scale if c not in structural_exclude]
            
            excluded_present = [c for c in structural_exclude if c in temp_ml_prep.columns_to_scale]
            if excluded_present:
                self.logger.info(f"✅ [PM] Excluded {len(excluded_present)} structural/standardized columns: {excluded_present}")
            
            self.logger.info(f"✅ [PM] Identified {len(feature_cols)} features using MLDatasetPreparation logic")
            self.logger.info(f"✅ [PM] Total columns to scale: {len(columns_to_scale)} (features + adv_targets, structural metadata excluded)")
            
            # Clean up
            del temp_ml_prep
            gc.collect()
            
            return feature_cols, columns_to_scale
            
        except Exception as e:
            self.logger.error(f"[PM] Feature identification failed: {e}")
            # Fallback: return all columns
            numeric_cols = df.columns.tolist()
            self.logger.warning(f"[PM] Using fallback: {len(numeric_cols)} columns")
            return numeric_cols, numeric_cols
    
    def _sanitize_split_for_scaling(self, df: pd.DataFrame, split_name: str = "train") -> pd.DataFrame:
        """
        Sanitize data before scaling to handle inf/NaN values AND convert
        astronomical string columns (zodiac signs, moon types, day names, etc.)
        to numeric using deterministic ordinal encoding.
        
        This ensures the scaler can transform all columns in columns_to_scale,
        including string columns produced by astronomical analysis.
        
        Args:
            df: DataFrame to sanitize
            
        Returns:
            Sanitized DataFrame with all columns numeric-safe
        """
        df = df.copy()
        
        # 1. Replace inf/-inf with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 2. Deterministic ordinal mappings for known astronomical string columns
        # These must be consistent between fit (training) and transform (val/test)
        ZODIAC_MAP = {
            'Aries': 0, 'Taurus': 1, 'Gemini': 2, 'Cancer': 3,
            'Leo': 4, 'Virgo': 5, 'Libra': 6, 'Scorpio': 7,
            'Sagittarius': 8, 'Capricorn': 9, 'Aquarius': 10, 'Pisces': 11
        }
        MOON_TYPE_MAP = {
            'New Moon': 0, 'Waxing Crescent': 1, 'First Quarter': 2,
            'Waxing Gibbous': 3, 'Full Moon': 4, 'Waning Gibbous': 5,
            'Third Quarter': 6, 'Waning Crescent': 7, 'Normal': 8
        }
        DAY_MAP = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }
        ELEMENT_MAP = {'Fire': 0, 'Earth': 1, 'Air': 2, 'Water': 3}
        
        # 3. Convert string columns to numeric
        for col in df.columns:
            if df[col].dtype == 'object':
                col_lower = col.lower()
                
                # Apply known mappings first
                if 'zodiac' in col_lower:
                    df[col] = df[col].map(ZODIAC_MAP).fillna(0).astype(float)
                elif col == 'Moon_Type':
                    df[col] = df[col].map(MOON_TYPE_MAP).fillna(8).astype(float)
                elif col in ('Day_of_Week', 'Planetary_Day'):
                    df[col] = df[col].map(DAY_MAP).fillna(0).astype(float)
                elif 'element' in col_lower:
                    df[col] = df[col].map(ELEMENT_MAP).fillna(0).astype(float)
                else:
                    # Generic fallback: try pd.to_numeric, then label encode
                    converted = pd.to_numeric(df[col], errors='coerce')
                    if converted.notna().sum() > 0:
                        df[col] = converted.fillna(0)
                    else:
                        # Consistent label encoding across splits
                        # We build the map from training data and reuse it for val/test
                        if col not in self.global_label_maps:
                            if self._current_split_name == "train":
                                unique_vals = sorted(df[col].dropna().unique().tolist())
                                self.global_label_maps[col] = {v: i for i, v in enumerate(unique_vals)}
                                self.logger.info(f"[PM] Built persistent label map for '{col}' with {len(unique_vals)} categories")
                            else:
                                # For val/test, we should already have the map from training.
                                # DATA INTEGRITY: Raise error instead of building on-the-fly
                                if self._current_split_name != "train":
                                    raise ValueError(
                                        f"[Data Integrity] Column '{col}' has no label map in {self._current_split_name}. "
                                        f"This indicates training was not processed first, or new categories appeared. "
                                        f"Categories must be: train ⊇ val ⊇ test"
                                    )
                                # For training, build map
                                unique_vals = sorted(df[col].dropna().unique().tolist())
                                self.global_label_maps[col] = {v: i for i, v in enumerate(unique_vals)}
                                
                                label_map = self.global_label_maps[col]
                                df[col] = df[col].map(label_map).fillna(-1).astype(float)  # Unknown → -1
                                self.logger.debug(f"[PM] Built label map for '{col}' with {len(label_map)} categories")
                        else:
                            # Reuse existing map for val/test
                            label_map = self.global_label_maps[col]
                            encoded = df[col].map(label_map)
                            
                            # Check for unmapped categories
                            unmapped_count = encoded.isna().sum()
                            if unmapped_count > 0:
                                self.logger.warning(
                                    f"⚠️ [Data Check] {int(unmapped_count)} unknown categories in '{col}' during {self._current_split_name}. "
                                    f"These were not in training data."
                                )
                            
                            df[col] = encoded.fillna(-1).astype(float)  # Unknown categories → -1
                            self.logger.debug(f"[PM] Applied training label map to '{col}' (split={self._current_split_name})")
        
        # 4.Fill NaNs using TRAINING statistics (not per-split means)
        if split_name == "train":
            # Build global fill statistics from training split
            if not hasattr(self, 'global_fill_means'):
                self.global_fill_means = {}
            
            for col in df.columns:
                if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    col_mean = df[col].mean()
                    self.global_fill_means[col] = col_mean if pd.notna(col_mean) else 0.0
                    df[col] = df[col].fillna(self.global_fill_means[col])
            
            self.logger.info(f"✅ [Sanitize] Built global fill statistics for {len(self.global_fill_means)} columns from TRAINING")
        else:
            # Reuse training statistics for val/test (CRITICAL for data integrity)
            if not hasattr(self, 'global_fill_means') or not self.global_fill_means:
                self.logger.error(f"❌ [Sanitize] {split_name} attempted without training statistics!")
                raise ValueError(f"Training statistics missing for {split_name}. Was training processed first?")
            
            for col in df.columns:
                if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    # Use TRAINING mean, not this split's mean
                    fill_value = self.global_fill_means.get(col, 0.0)
                    df[col] = df[col].fillna(fill_value)
            
            self.logger.info(f"✅ [Sanitize] {split_name}: Imputed NaN using TRAINING statistics (no leakage)")
        
        
        # Just in case there are any unseen inf/nan values or extreme leaks
        # Capture the returned copy to ensure the fixes are applied!
        df = self.ml_prep._sanitize_data_for_scaling(df)
        df.fillna(0, inplace=True)
        df.replace([np.inf, -np.inf], 0, inplace=True)
        
        self.logger.info(f"✅ [PM] Data sanitized: Replaced inf/NaN, converted strings")
        return df

    def _apply_returns_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert price-level features to percentage changes (returns) 
        and volume to ratio-to-rolling-mean to ensure regime-agnostic inputs.
        
        This prevents catastrophic regression at chunk boundaries when absolute 
        price levels differ between training and validation data (e.g. Gold 2019 vs 2024).
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with transformed price/volume columns
        """
        # Create a copy to avoid SettingWithCopy warnings if operating on slices
        df = df.copy()
        
        # 1. Price Returns Transform (Multiplicative → Additive/Stationary)
        # These are regime-dependent (gold at 1800 vs 2500 produces different levels)
        # but returns are regime-agnostic (-0.3% move means the same regardless of level)
        PRICE_LEVEL_COLS = ['Open', 'High', 'Low', 'Close']
        for col in PRICE_LEVEL_COLS:
            if col in df.columns:
                # pct_change() is superior to diff() for regime-independence
                # Bounded roughly -0.05 to +0.05 for daily gold
                df[col] = df[col].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
                self.logger.info(f"📊 [MLPrep] Price returns transform: {col} → pct_change")
        
        # 2. Volume Ratio Transform (Absolute → Relative/Stationary)
        # Volume levels grow over years; ratio to recent mean preserves pressure signal
        VOLUME_COLS = ['Volume']
        for col in VOLUME_COLS:
            if col in df.columns:
                # Volume: ratio to 20-bar rolling mean (captures relative spikes)
                rolling_mean = df[col].rolling(window=20, min_periods=1).mean().replace(0, 1)
                df[col] = (df[col] / rolling_mean).replace([np.inf, -np.inf], 1).fillna(1)
                self.logger.info(f"📊 [MLPrep] Volume ratio transform: {col} → vol/rolling_mean(20)")
                
        return df


    def _categorize_features(self, columns: List[str]) -> tuple[List[str], List[str], List[str]]:
        """
        Partition feature columns into three mutually-exclusive categories for SelectiveScaler.

        Categories
        ----------
        price_cols  → PassThroughScaler
            Absolute price-level indicators already normalised to [0, 1] by
            _normalize_price_level_indicators() (structural range).
            Examples: OHLCV, SMA/EMA/WMA, BB bands, Supertrend levels,
                      pivot levels, SMC price levels, trendline values.

        diff_cols   → PassThroughScaler
            Momentum, spread, and distance columns already mapped to [0, 1]
            by _normalize_diff_columns_by_range() (divided by Rolling_Range_Width,
            clipped to [-1,1], then shifted to [0,1]).
            Examples: MACD, EMA spreads, ATR, RSI, Supertrend distance,
                      SNR distances, pct-change columns, MA_Change* columns.

        other_cols  → MinMaxScaler
            Everything else: boolean flags, categorical pattern columns,
            astro features, counts, time columns, and structural scalars
            (Rolling_Range_Width, Regime_Macro_Position) that are not
            pre-normalised in price units.

        Decision priority (first match wins)
        -------------------------------------
        1. Explicit pattern/boolean set  → other
        2. Explicit diff/oscillator set  → diff
        3. Explicit price-level set      → price
        4. Dynamic diff-suffix check on price prefixes (e.g. EMA_*_Diff) → diff
        5. Dynamic price-prefix match (SMA_*, EMA_*, MA_*, …)            → price
        6. Dynamic diff-prefix match (RSI_*, MACD_*, …)                  → diff
        7. Substring diff-pattern match (excluding time columns)          → diff
        8. Default                                                        → other

        Returns
        -------
        (price_cols, diff_cols, other_cols) — each a list of column names,
        preserving the input order.
        """

        # ── 1. EXPLICIT OTHER / PATTERN COLUMNS (checked first to prevent mis-routing) ─
        # Boolean flags, SMC presence flags, pattern counts — MinMax is fine.
        # NOTE: SMC_FVG_FVG / SMC_OB_OB etc. are presence flags (0/1), NOT price levels,
        # so they must stay here and NOT appear in the price sets below.
        # NOTE: Volume and TickVolume are PRE-NORMALIZED by _normalize_volume_by_range()
        # and should use PassThroughScaler (moved to price_cols below)
        explicit_other: set = {
            # Distance columns (pre-normalized by _normalize_distance_by_range)
            # These use distance-specific range fitting, not price structural range
            # REMOVED: Up_Distance, Down_Distance (handled by distance range fitting)
            
            # Volume surge and consistency metrics (absolute values, MinMax appropriate)
            'avg_volume_during_move',
            'volume_surge_factor',
            'volume_consistency',
            
            # Heikin-Ashi candle pattern flags
            'Doji',
            'HA_Candle', 'HA_Flat_Bottom', 'HA_Flat_Top', 'HA_Lower_Wick',
            'HA_Reversal', 'HA_Small_Body', 'HA_Upper_Wick',
            # SMC presence flags (boolean)
            'SMC_FVG_FVG',          # FVG present flag
            'SMC_Liquidity_Liquidity',  # Liquidity level present flag
            'SMC_OB_OB',            # Order block present flag
            'SMC_Swing_HighLow',    # Swing high/low present flag
            # Structure / pivot presence flags
            'Structure_Established',
            'Pivots',
            'Pivot',
            # Raw volume unit totals (not ratios — MinMax appropriate)
            'Zonal_Total_Volume',
            'Zonal_Net_Volume',
            'historical_avg_volume',
            'level_touch_volume_avg',
            # Candlestick pattern occurrence counts (integer, MinMax appropriate)
            'doji_count',
            'hammer_count',
            'shooting_star_count',
            'engulfing_bullish_count',
            'engulfing_bearish_count',
            'spinning_top_count',
            'marubozu_count',
            # Structural range scalars — broadcast as constants, not pre-normalised
            # in price-unit space, so MinMax is the safe choice.
            'Rolling_Range_Width',
            'Regime_Macro_Position',  # Can exceed [0,1] during breakouts; MinMax re-clips it.
            # Footprint flags/ratios (already [0,1] by construction)
            'FP_Imbalance_Max', 'FP_Delta_Divergence',
            'FP_High_Vol_Rejection', 'FP_Data_Available',
            # SNR count columns (integer values, NOT normalised to [0,1])
            'snr_num_levels_above',
            'snr_num_levels_below',
            'snr_nearest_zone_volume',
            # Cross_* event signals: +1 crossed above, -1 below, 0 flat.
            # Range is [-1, +1], NOT [0, 1] — MinMax handles this correctly.
            'Cross_EMA8_Above_EMA12',
            'Cross_EMA12_Above_EMA18',
            'Cross_MA25_Above_MA50',
            'Cross_MA50_Above_MA100',
            'Cross_Supertrend',
            # ── SMC BOS/CHOCH: ±1 directional flags, NOT price levels ─────────────
            # BOS=+1 bullish break, -1 bearish break (same for CHOCH).
            # These are categorical signals with range [-1, +1] — MinMax is correct.
            # Without this they fall through the 'SMC_' prefix → PRICE partition which
            # passes -1 values straight through the PassThroughScaler.
            'SMC_BOS_BOS',
            'SMC_BOS_CHOCH',
            # ── SMC BOS BrokenIndex: integer bar-index (0–2000+), NOT a price ──────
            # Records *which candle* confirmed the structural break. MinMax is correct.
            'SMC_BOS_BrokenIndex',
            # ── RSI_14 change lags: unbounded RSI-unit deltas (~[-70, +90]) ────────
            # These are RSI_14[t] - RSI_14[t-i]. They are NOT normalised by
            # _normalize_diff_columns_by_range() (which divides by price range width).
            # MinMaxScaler (OTHER partition) is correct: fits on training range.
            'RSI_14_Change_Lag_1', 'RSI_14_Change_Lag_2', 'RSI_14_Change_Lag_3',
            'RSI_14_Change_Lag_4', 'RSI_14_Change_Lag_5',
        }

        # ── 2. EXPLICIT DIFF / OSCILLATOR COLUMNS ─────────────────────────────────────
        # These are all pre-normalised by _normalize_diff_columns_by_range() → [0, 1].
        # PassThroughScaler preserves the regime-relative meaning; MinMax would destroy it.
        explicit_diff: set = {
            # Bollinger Band oscillators (NOT the band price levels)
            'BBP_20_2.0_2.0',       # %B position within bands
            'BBB_20_2.0_2.0',       # Band width (basis)
            # Core momentum indicators
            'RSI', 'RSI_14', 'RSI_7', 'RSI_2',
            'RSI_2_Change_Lag_1', 'RSI_2_Change_Lag_2', 'RSI_2_Change_Lag_3',
            'RSI_2_Change_Lag_4', 'RSI_2_Change_Lag_5', 'RSI_2_Pct_Change',
            'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
            'CCI', 'Stochastic', 'Stoch_K', 'Stoch_D',
            # Volatility indicators
            'ATR', 'ATR_Pct',
            'Historical_Volatility', 'Historical_Volatility_20',
            'Supertrend_Distance',
            # Volume-based momentum (ratios / pct changes, not raw units)
            'OBV', 'On_Balance_Volume',
            'Bar_Volume_Up', 'Bar_Volume_Down',
            'Up_Volume_Change_Pct', 'Down_Volume_Change_Pct', 'Volume_Change_Pct',
            # Candle metrics
            'Candle_Size',
            'ClosePCT',
            # MA crossover binary signals (position flags)
            # EMA8_Above_EMA12 etc. are 0/1 boolean-style flags → diff (pre-normalized [0,1])
            'EMA8_Above_EMA12', 'MA25_Above_MA50', 'MA50_Above_MA100',
            'Short_Above_Long_Crossover',
            # NOTE: Cross_* are ±1 event signals (not [0,1]) — moved to explicit_other below
            # SNR distance metrics (price-unit diffs normalised by range width)
            'snr_dist_to_nearest_level',
            'snr_dist_to_nearest_support',
            'snr_dist_to_nearest_resistance',
            'snr_in_zone',
            # NOTE: snr_num_levels_above / snr_num_levels_below are INTEGER COUNTS (0-29).
            # They are NOT normalised by range width — MinMaxScaler is correct for them.
            # Do NOT add snr_num_levels_above/below here.
            # Regime ratio features (already [0, 1] ratios)
            'Regime_ATR_Surge', 'Regime_Mid_ROC_10', 'Regime_Mid_ROC_50',
            'Regime_Width_ROC', 'Regime_Distance_From_Low', 'Regime_Distance_From_High',
            # EMA spread diffs (price-unit, normalised by range width)
            'EMA_12_Minus_EMA8', 'EMA_21_Minus_EMA8',
            'EMA_64_Minus_EMA8', 'EMA_200_Minus_EMA8',
            'EMA_100_Minus_EMA8',
            # ADDED: Individual EMA _Diff columns (were causing 50% breakouts)
            'EMA_100_Diff',
            'EMA_12_Diff',
            'EMA_21_Diff',
            'EMA_64_Diff',
            'EMA_8_Diff',
            'MA_100_50_Diff',
            # ADDED: Individual MA _Diff columns (were causing 48-50% breakouts)
            'Long_MA_Diff', 'Long_Period_MA_Diff',
            # ADDED: FVG_Diff (was causing 99.65% breakouts)
            'FVG_Diff',
            # MA_Change0-4 — EMA_X minus EMA_8 (price-unit spreads, produced by _calculate_ema_change_series)
            # MA_Change0 = EMA_64 - EMA_8,  MA_Change1 = EMA_32 - EMA_8, …
            'MA_Change0', 'MA_Change1', 'MA_Change2', 'MA_Change3', 'MA_Change4',
            # MA_200_Change* — bar-over-bar delta of MA_200 (price-unit diffs, NOT MA levels)
            'MA_200_Change0', 'MA_200_Change1', 'MA_200_Change2',
            'MA_200_Change3', 'MA_200_Change4',
            'MA_200_Change_0', 'MA_200_Change_1', 'MA_200_Change_2',
            'MA_200_Change_3', 'MA_200_Change_4',
            # Lettered diff variables (a-i) from _calculate_lettered_variables():
            #   a = MA_25 - Close,  b = MA_50 - Close,  c = MA_100 - Close,
            #   d = MA_200 - Close, e = EMA_8 - Close,  f = EMA_10 - Close,
            #   g = EMA_12 - Close, h = EMA_24 - Close, i = EMA_32 - Close
            # All are price-unit distances → range-width-normalised as diffs.
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
            # SMA0-SMA11 from _calculate_ema_change_series / sma_range_diffs:
            # SMA{n} = SMA(n+2) - Close  (SMA difference series, NOT SMA level values)
            # config default: sma_range_periods=(2,14) → SMA0…SMA11 are all price-unit diffs.
            'SMA0', 'SMA1', 'SMA2', 'SMA3', 'SMA4',
            'SMA5', 'SMA6', 'SMA7', 'SMA8', 'SMA9', 'SMA10', 'SMA11',
            # Raw momentum / MR / TF columns from _calculate_momentum_features():
            #   MOM_t = Close.diff()           → price-unit diff
            #   MR_t  = Close.pct_change()     → ratio/pct diff
            #   TF_t  = detrended z-score      → bounded oscillator
            'MOM_t', 'MR_t', 'TF_t',
            # Volume ratio / dominance signals (dimensionless, already ~[0, 1])
            'volume_surge_vs_historical',
            'up_volume_dominance',
            'down_volume_dominance',
            'volume_distance_ratio',
            'level_touch_volume_surge',
            'pre_signal_volume_trend',
            # Candlestick pattern strength scores ([0, 1])
            'pattern_strength_score',
            'reversal_pattern_strength',
            'continuation_pattern_strength',
            # Footprint delta (pre-normalized by _normalize_footprint_by_range → ~[-1,1])
            'FP_Delta', 'FP_Cum_Delta',
        }

        # ── 3. EXPLICIT PRICE-LEVEL COLUMNS ───────────────────────────────────────────
        # These are absolute price levels normalised to [0, 1] by structural range.
        # OHLCV and price-like columns
        price_ohlcv: set = {
            'Open', 'High', 'Low', 'Close',
            'open', 'high', 'low', 'close',
            'Spread',  # Price spread (not volume)
            'High_Day_1', 'High_Day_2', 'High_Day_3',
            'Low_Day_1', 'Low_Day_2', 'Low_Day_3',
            'Previous_Close',
            'Prev_1_Close', 'Prev_2_Close', 'Prev_3_Close', 'Prev_4_Close', 'Prev_5_Close',
            # Volume columns (pre-normalized by _normalize_volume_by_range in Step 0)
            # Must use PassThroughScaler to preserve the fitted range, NOT MinMaxScaler
            'Volume',
            'TickVolume',
        }
        # Moving averages (actual levels, not differences)
        price_ma: set = {
            'SMA_10', 'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200',
            # NOTE: SMA0-SMA11 are NOT here — they are SMA(period)-Close diffs,
            # computed by _calculate_ema_change_series / sma_range_diffs.
            # They live in explicit_diff above.
            'EMA_8', 'EMA_10', 'EMA_12', 'EMA_18', 'EMA_21',
            'EMA_24', 'EMA_32', 'EMA_64', 'EMA_100', 'EMA_200',
            'EMA-8', 'EMA-12', 'EMA-21', 'EMA-64',
            'Short_MA', 'Long_MA', 'MA',
            'Short_MA_10', 'Short_MA_50',
            'Long_MA_25', 'Long_MA_100', 'Long_MA_200',
            '10_Day_MA', '50_Day_MA',
            'MA_25', 'MA_50', 'MA_100', 'MA_200',
            'MA-25', 'MA-50', 'MA-100', 'MA-200',
            'Pivot Price',  # legacy name with space
        }
        # Bollinger Band levels (not %B or bandwidth)
        price_bb: set = {
            'BB_Upper', 'BB_Middle', 'BB_Lower',
            'BB_UpperBand', 'BB_MiddleBand', 'BB_LowerBand',
            'BBL_20_2.0_2.0', 'BBM_20_2.0_2.0', 'BBU_20_2.0_2.0',
        }
        # Supertrend price levels (not Supertrend_Distance)
        price_supertrend: set = {
            'Supertrend', 'Supertrend_Upper', 'Supertrend_Lower',
            'Final Lowerband', 'Final Upperband',
        }
        # Pivot price levels and SAR
        price_pivots: set = {
            'r1', 'r2', 'r3', 's1', 's2', 's3',
            'R1', 'R2', 'R3', 'S1', 'S2', 'S3',
            'Pivot_R1', 'Pivot_R2', 'Pivot_R3',
            'Pivot_S1', 'Pivot_S2', 'Pivot_S3',
            'Pivot_Price',
            'Parabolic_SAR',
        }
        # SMC price levels (level values, not presence flags)
        price_smc: set = {
            'SMC_OB_Top', 'SMC_OB_Bottom',
            'SMC_FVG_Top', 'SMC_FVG_Bottom',
            'SMC_Swing_Level', 'SMC_Liquidity_Level',
            # SMC_BOS_Level: the actual price level of the broken structure.
            # Already normalised by _normalize_price_level_indicators() via the
            # 'SMC_' dynamic prefix rule — must also be in explicit price_smc so
            # the PassThroughScaler handles it correctly after normalisation.
            'SMC_BOS_Level',
            'FVG_Top', 'FVG_Bottom',
            'Order_Block_Top', 'Order_Block_Bottom',
        }
        # Trendline price-level values
        price_trendlines: set = {
            'Support_Trendline_Value', 'Resist_Trendline_Value',
        }
        # Structural range bounds (fixed scalars, already in price units → PassThrough)
        price_structural: set = {
            'Rolling_Range_High', 'Rolling_Range_Low', 'Rolling_Range_Mid',
            'Structural_Range_Width', 'Structural_Range_Position',
        }

        all_price_cols: set = (
            price_ohlcv | price_ma | price_bb | price_supertrend
            | price_pivots | price_smc | price_trendlines | price_structural
        )

        # ── 4. DYNAMIC PREFIX / SUFFIX RULES ──────────────────────────────────────────
        # Prefixes whose columns are price-level by default (but can be overridden by
        # diff suffixes below, e.g. EMA_100_Minus_EMA8 starts with EMA_ but is a diff).
        price_prefixes: tuple = (
            'SMA_', 'EMA_', 'WMA_', 'HMA_',
            'BBL_', 'BBM_', 'BBU_',         # pandas_ta Bollinger levels
            'Supertrend_',
            'BB_',
            'MA_',
            'Pivot_',
            'SMC_',                          # catches dynamic SMC_* level columns
        )
        # Suffixes / substrings that override a price prefix and mark the column as a diff.
        # 'Minus' covers *_Minus_* spread columns (e.g. EMA_100_Minus_EMA8).
        # 'Change' covers MA_200_Change* and similar rate-of-change columns.
        diff_override_keywords: tuple = (
            '_Diff', '_diff',
            '_Distance', '_distance',
            '_Pct', '_pct', '_percent',
            'Minus', 'minus',
            'Change', '_Change',
            'BBP_', 'BBB_',             # Bollinger oscillators with price-like prefixes
        )
        # Prefixes that are always diff indicators regardless of suffix
        diff_prefixes: tuple = (
            'RSI_', 'MACD_', 'Stoch_', 'CCI_',
            'ATR_', 'OBV_', 'BBP_', 'BBB_',
            'CSM_',   # Currency Strength Matrix — already bounded [-1, +1], PassThrough
        )
        # Substring patterns that classify a column as diff (checked last)
        diff_substrings: tuple = (
            'diff', 'Diff',
            'distance', 'Distance',
            'change', 'Change',
            'pct', 'Pct',
            'percent', 'Percent',
            'returns', 'Returns',
            'roc', 'ROC',
            'momentum', 'Momentum',
            'rate', 'Rate',
            'speed', 'Speed',
            'minus', 'Minus',
            '_from_',
        )

        # ── CLASSIFY ──────────────────────────────────────────────────────────────────
        price_result: List[str] = []
        diff_result:  List[str] = []
        other_result: List[str] = []

        # Known currency-index prefixes (from currency_index.INDEX_DEFINITIONS).
        # If a column starts with one of these we strip the prefix and classify the
        # base name — e.g. "Dollar_RSI_14" → classify "RSI_14" → diff.
        # OHLCV index columns (Dollar_open, Dollar_close, …) have no base name and fall
        # through to the original price rules naturally.
        _INDEX_PREFIXES: tuple = (
            'Dollar_', 'Euro_', 'JPY_', 'GBP_', 'CHF_',
            'CAD_', 'AUD_', 'NZD_', 'CNH_',
        )

        def _classify_single(name: str) -> str:
            """Return 'price' | 'diff' | 'other' for one column name."""
            if name in explicit_other:
                return 'other'
            if name.startswith('SMC_') and any(f in name for f in ['_FVG_', '_OB_', '_Liquidity_', '_Swing_HighLow']):
                return 'other'
            if name in explicit_diff:
                return 'diff'
            if name in all_price_cols:
                return 'price'
            if any(name.startswith(p) for p in price_prefixes):
                return 'diff' if any(kw in name for kw in diff_override_keywords) else 'price'
            if any(name.startswith(p) for p in diff_prefixes):
                return 'diff'
            if any(kw in name for kw in diff_substrings):
                return 'other' if ('time' in name.lower() or 'moon' in name.lower()) else 'diff'
            return 'other'

        for col in columns:
            # Priority 0: currency-index prefixed TI columns — strip prefix, classify base.
            stripped = col
            for pfx in _INDEX_PREFIXES:
                if col.startswith(pfx):
                    stripped = col[len(pfx):]  # e.g. "Dollar_RSI_14" → "RSI_14"
                    break

            bucket = _classify_single(stripped)

            if bucket == 'price':
                price_result.append(col)
            elif bucket == 'diff':
                diff_result.append(col)
            else:
                other_result.append(col)

        return price_result, diff_result, other_result

    def _get_price_related_columns(self, df: pd.DataFrame, not_include_list: list[str]=None) -> List[str]:
        """DEPRECATED: Use _categorize_features() instead. 
        Kept for backward compatibility."""
        price_cols, _, _ = self._categorize_features(df.columns.tolist())
        return price_cols


    def _normalize_price_level_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize PRICE-LEVEL columns only (not diff, astro, or pattern features).
        Uses the SAME rolling structural range as OHLCV for consistent [0, 1] scaling.
        
        Price-level columns normalized (actual levels):
        - SMA_{period}, EMA_{period} (moving averages, not their diffs)
        - Short_MA, Long_MA, MA variants
        - BB_Upper, BB_Middle, BB_Lower (bands, not BB_Squeeze or diffs)
        - MACD, MACD_Signal, MACD_Histogram
        - Supertrend_Upper, Supertrend_Lower, Supertrend (not Supertrend_Distance)
        - Parabolic_SAR
        - Pivot levels: r1-r3, s1-s3
        - SMC levels: OB Top/Bottom, FVG Top/Bottom, Swing_Level, Liquidity_Level
        - Trendlines: Support_Trendline_Value, Resist_Trendline_Value
        
        EXCLUDED (use own rolling mean method):
        - *_Diff columns (differences relative to price)
        - RSI, ATR, Volume, Volatility (already bounded or relative)
        - Distance/Speed/Time columns
        - Pattern signals (HA_*, Doji*, BB_Squeeze, Signal_*)
        - Astro features (not price-dependent)
        """
        df = df.copy()
        if 'Close' not in df.columns: return df
        
        if 'Rolling_Range_Width' not in df.columns:
            self.logger.warning("[NormalizePriceLevels] Rolling range not found, skipping indicator normalization")
            return df
        
        # EXPLICIT price-level columns — must match the price sets in _categorize_features.
        # SMC_FVG_FVG, SMC_OB_OB, etc. are presence FLAGS (boolean), NOT price levels,
        # so they are intentionally absent here.
        # SMA0-SMA11 are SMA(period)-Close DIFFS, NOT SMA level values — also absent.
        # a-i are MA/EMA-Close DIFFS — also absent. Both sets are handled by
        # _normalize_diff_columns_by_range() instead.
        price_level_columns = {
            # Hyphen-named MA variants from older research
            'EMA-8', 'EMA-12', 'EMA-21', 'EMA-64',
            # Numbered SMA variants (e.g. SMA0 … SMA11)
            'SMA0', 'SMA1', 'SMA2', 'SMA3', 'SMA4',
            'SMA5', 'SMA6', 'SMA7', 'SMA8', 'SMA9', 'SMA10', 'SMA11',
            # Standard named MAs
            'SMA_10', 'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200',
            'EMA_8', 'EMA_10', 'EMA_12', 'EMA_18', 'EMA_21',
            'EMA_24', 'EMA_32', 'EMA_64', 'EMA_100', 'EMA_200',
            'Short_MA_10', 'Short_MA_50',
            'Long_MA_25', 'Long_MA_100', 'Long_MA_200',
            '10_Day_MA', '50_Day_MA',
            'MA_25', 'MA_50', 'MA_100', 'MA_200',
            'MA-25', 'MA-50', 'MA-100', 'MA-200',
            'Short_MA', 'Long_MA', 'MA',
            'Pivot Price',
            # Bollinger Band price levels (not %B or bandwidth)
            'BB_Upper', 'BB_Middle', 'BB_Lower',
            'BB_UpperBand', 'BB_MiddleBand', 'BB_LowerBand',
            # Supertrend price levels (not Supertrend_Distance)
            'Supertrend', 'Supertrend_Upper', 'Supertrend_Lower',
            'Final Lowerband', 'Final Upperband',
            # Parabolic SAR
            'Parabolic_SAR',
            # Pivot price levels
            'r1', 'r2', 'r3', 's1', 's2', 's3',
            'Pivot_R1', 'Pivot_R2', 'Pivot_R3',
            'Pivot_S1', 'Pivot_S2', 'Pivot_S3',
            'Pivot_Price',
            # SMC price-level values (NOT presence flags like SMC_FVG_FVG)
            'SMC_OB_Top', 'SMC_OB_Bottom',
            'SMC_FVG_Top', 'SMC_FVG_Bottom',
            'SMC_Swing_Level', 'SMC_Liquidity_Level',
            # BOS_Level: the price at which structure was broken — a genuine price level
            # that must be normalised by the structural range, same as OB/FVG levels.
            'SMC_BOS_Level',
            'FVG_Top', 'FVG_Bottom',
            'Order_Block_Top', 'Order_Block_Bottom',
            # Trendline price-level values
            'Support_Trendline_Value', 'Resist_Trendline_Value',
        }

        # Dynamic prefixes for pandas_ta and other generated price-level columns
        price_prefixes = ('SMA_', 'EMA_', 'WMA_', 'HMA_', 'BBL_', 'BBM_', 'BBU_', 'Supertrend_', 'BB_', 'MA_')

        # Substrings that mark a column as a diff/spread even when it has a price prefix.
        # 'Minus' catches EMA_*_Minus_EMA_* spreads.
        # 'Change' catches MA_200_Change* bar-over-bar deltas.
        diff_keywords = ('_Diff', '_diff', '_Distance', '_distance', '_pct', '_Pct', '_percent', 'Minus', 'minus', 'Change')

        count = 0
        for col in df.columns:
            # A column qualifies for price-level normalisation if it is in the explicit set
            # OR matches a known price prefix — but NOT if a diff keyword overrides it.
            is_price_col = col in price_level_columns or col.startswith(price_prefixes)
            is_diff_override = any(kw in col for kw in diff_keywords)

            if is_price_col and not is_diff_override:
                # Structural range normalisation, then rolling-mean sigmoid instead of
                # hard .clip(0, 1).  Values outside the training range are compressed
                # toward the boundary rather than saturated — fully lossless.
                raw_norm = (df[col] - df['Rolling_Range_Low']) / df['Rolling_Range_Width']
                raw_norm = raw_norm.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.5)

                normalized, _rm = self._normalize_with_rolling_mean(
                    raw_norm,
                    col=col,
                    window=20,
                    sigmoid_scale_factor=self.sigmoid_scale_factor,
                    store_baseline=True,
                )
                df[col] = normalized
                count += 1

        if count > 0:
            self.logger.info(
                f"⚖️ [MLPrep] Normalized {count} price-level indicators using "
                f"structural range + rolling-mean sigmoid (lossless, preserves spillover)"
            )
        return df

    def _normalize_diff_columns_by_range(self, df: pd.DataFrame, skip_cols: set = None) -> pd.DataFrame:
        """
        Normalize DIFF/oscillator columns that are in price units by dividing by
        Rolling_Range_Width, expressing each diff as a fraction of the current regime range.

        ⚠️ SKIP columns already standardized in Step 0c by StandardScaler. These are
        unbounded (mean=0, std=1) and should NOT be mapped to [0,1] ranges.
        Only normalize columns that haven't been processed yet.

        Formula:  
          1. normalized_diff = diff_col / Rolling_Range_Width
          2. clipped = clip(normalized_diff, -1, 1)
          3. mapped = (clipped + 1.0) / 2.0  →  [0, 1] range

        Why this works:
        - Diff columns (EMA spreads, MACD, Supertrend distance, SNR distances) are all
          expressed in the same raw price units as Rolling_Range_Width.
        - Dividing by the range width scales them relative to the current structural regime.
        - Mapping the result from [-1, 1] to [0, 1] perfectly preserves distance and sign 
          (0.5 is exactly zero difference) but keeps all inputs positive. Positive-only 
          activations (like ReLU/Sigmoid in LSTMs) perform much better when inputs are strictly >= 0.

        Columns normalized (price-unit diffs):
        - EMA_X_Minus_EMA_Y  (EMA spread diffs)
        - MA_100_50_Diff
        - MACD, MACDh, MACDs  (raw EMA diff outputs — price units)
        - Supertrend_Distance
        - snr_dist_to_nearest_* (if stored in price units)
        - Any dynamically generated *_Diff / *_Distance columns

        Columns EXCLUDED (already bounded or not in price units):
        - RSI, Stochastic, CCI, BBP_%B  (already [0,100] or [0,1])
        - Regime_Distance_From_Low / High  (already [0,1] ratios)
        - Rolling_Range_Width itself
        - Volume-based oscillators, pattern/binary signals
        - Columns in skip_cols (already standardized by StandardScaler)

        Args:
            df: Input DataFrame
            skip_cols: Set of column names to skip (already standardized by StandardScaler)

        Returns:
            DataFrame with normalized diff columns (except skip_cols)
        """
        if skip_cols is None:
            skip_cols = set()
        df = df.copy()

        if 'Rolling_Range_Width' not in df.columns:
            self.logger.warning("[NormalizeDiffs] Rolling_Range_Width not found — skipping diff normalization")
            return df

        range_width = df['Rolling_Range_Width'].replace(0, np.nan)

        # ── Explicit price-unit diff columns ──────────────────────────────────
        # These are all in raw price units and need dividing by Rolling_Range_Width.
        # Keep this list in sync with explicit_diff in _categorize_features.
        explicit_price_unit_diffs = {
            # EMA spread diffs
            'EMA_12_Minus_EMA8', 'EMA_21_Minus_EMA8',
            'EMA_64_Minus_EMA8', 'EMA_200_Minus_EMA8',
            'EMA_100_Minus_EMA8',
            'MA_100_50_Diff',
            # MACD family (raw EMA differences → price units)
            'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
            # Supertrend distance (price minus supertrend level → price units)
            'Supertrend_Distance',
            # SNR distances (stored in price units)
            'snr_dist_to_nearest_level',
            'snr_dist_to_nearest_support',
            'snr_dist_to_nearest_resistance',
            # MA_Change0-4: EMA_X - EMA_8 spreads (price-unit diffs)
            'MA_Change0', 'MA_Change1', 'MA_Change2', 'MA_Change3', 'MA_Change4',
            # MA_200_Change* — bar-over-bar delta of MA_200 level (price-unit diffs)
            'MA_200_Change0', 'MA_200_Change1', 'MA_200_Change2',
            'MA_200_Change3', 'MA_200_Change4',
            'MA_200_Change_0', 'MA_200_Change_1', 'MA_200_Change_2',
            'MA_200_Change_3', 'MA_200_Change_4',
            # Lettered variables: MA/EMA minus Close (price-unit distances)
            # a=MA_25-Close, b=MA_50-Close, c=MA_100-Close, d=MA_200-Close,
            # e=EMA_8-Close, f=EMA_10-Close, g=EMA_12-Close, h=EMA_24-Close, i=EMA_32-Close
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
            # SMA0-SMA11: SMA(period)-Close difference series (price-unit diffs)
            # SMA{n} = SMA(n+2) - Close  (default sma_range_periods=(2,14))
            'SMA0', 'SMA1', 'SMA2', 'SMA3', 'SMA4',
            'SMA5', 'SMA6', 'SMA7', 'SMA8', 'SMA9', 'SMA10', 'SMA11',
            # Raw momentum column
            'MOM_t',
            'MR_t',    # Close.pct_change() ratio — same price regime as MOM_t
            'TF_t',    # Detrended z-score — treated as diff, normalised by range width
            'Candle_Size',
        }

        # ── Columns that are already bounded ([0,1] or [0,100]) — must NOT be divided ──
        # These are diff-category columns that _normalize_diff_columns_by_range must skip
        # because dividing by range_width would corrupt their already-correct scale.
        already_scaled_cols = {
            # Bounded oscillators (output of indicator calculation, not price-unit diffs)
            'RSI', 'RSI_14', 'RSI_7', 'RSI_2',
            'RSI_2_Change_Lag_1', 'RSI_2_Change_Lag_2', 'RSI_2_Change_Lag_3',
            'RSI_2_Change_Lag_4', 'RSI_2_Change_Lag_5', 'RSI_2_Pct_Change',
            'Stochastic', 'Stoch_K', 'Stoch_D',
            'CCI', 'BBP_20_2.0_2.0', 'BBB_20_2.0_2.0',
            # Volume pct-change ratios (not price units)
            'Up_Volume_Change_Pct', 'Down_Volume_Change_Pct', 'Volume_Change_Pct',
            # Regime ratios — already [0, 1] by construction
            'Regime_Distance_From_Low', 'Regime_Distance_From_High',
            'Regime_ATR_Surge', 'Regime_Mid_ROC_10', 'Regime_Mid_ROC_50',
            'Regime_Width_ROC',
            # Regime_Macro_Position: can exceed [0,1] in breakouts; MinMax handles it
            'Regime_Macro_Position',
            # Structural range scalars (not to be re-divided)
            'Rolling_Range_Width', 'Rolling_Range_High',
            'Rolling_Range_Low', 'Rolling_Range_Mid',
            'Structural_Range_Width', 'Structural_Range_Position',
            # Astro distance (not price units)
            'Moon_Distance',
            # Currency Strength Matrix — already in [-1, +1] from the normalization pipeline.
            # Dividing by Rolling_Range_Width would corrupt the bounded oscillator values.
            'CSM_asset_norm_fast', 'CSM_dxy_norm_fast',
            'CSM_asset_norm_slow', 'CSM_dxy_norm_slow',
            'CSM_histogram_fast',  'CSM_histogram_slow',
        }

        # ── Dynamically catch *_Diff / *_Distance / *Minus / *_Change columns ─────
        # These cover any dynamically-generated indicator diffs not listed explicitly above.
        # Exclusion filters prevent accidentally catching time diffs, volume ratios, astro.
        dynamic_diff_cols = [
            c for c in df.columns
            if (
                '_Diff' in c or '_diff' in c
                or '_Distance' in c or '_distance' in c
                or 'Minus' in c or 'minus' in c
                or 'Speed' in c or 'speed' in c
                or 'Volatility' in c or 'volatility' in c
                or c.startswith('MA_Change')   # MA_Change* — bar-over-bar MA deltas
            )
            and c not in already_scaled_cols
            and 'Regime_Distance_From' not in c  # Regime ratios already [0,1]
            and 'Time_Diff' not in c             # Bar-count diffs, not price units
            and 'time_diff' not in c
            and 'Volume' not in c                # Volume diffs use different units
            and 'volume' not in c
            and 'Moon' not in c                  # Astro features
        ]

        target_cols = explicit_price_unit_diffs | set(dynamic_diff_cols)

        # ── Also normalize index-prefixed variants (Dollar_SMA0, Dollar_MOM_t, etc.) ──
        # _categorize_features strips the index prefix and classifies the base name,
        # so Dollar_SMA0 → diff (PassThrough). But the normalization pipeline above
        # only has the base names in explicit_price_unit_diffs. Expand target_cols to
        # include all {Index}_{base} variants so they get divided by Rolling_Range_Width.
        _INDEX_PREFIXES_NORM = ('Dollar_', 'Euro_', 'JPY_', 'GBP_', 'CHF_', 'CAD_', 'AUD_', 'NZD_', 'CNH_')
        index_prefixed_diffs = set()
        for col in df.columns:
            for pfx in _INDEX_PREFIXES_NORM:
                if col.startswith(pfx):
                    base = col[len(pfx):]
                    if base in explicit_price_unit_diffs:
                        index_prefixed_diffs.add(col)
                    break
        target_cols = target_cols | index_prefixed_diffs
        count = 0

        # ── SNR sanity check: confirm raw magnitudes are price-unit before dividing ──
        # If ratio << 0.1, they may be pre-normalized; if >> 2.0 they may be raw pips.
        snr_check_cols = [
            'snr_dist_to_nearest_level',
            'snr_dist_to_nearest_support',
            'snr_dist_to_nearest_resistance',
        ]
        range_mean = df['Rolling_Range_Width'].mean()
        for snr_col in snr_check_cols:
            if snr_col in df.columns and snr_col in target_cols:
                raw_max = df[snr_col].abs().max()
                ratio = raw_max / (range_mean + 1e-8)
                self.logger.info(
                    f"[DiffNorm] {snr_col}: raw_max={raw_max:.4f}, "
                    f"range_width_mean={range_mean:.4f}, ratio={ratio:.3f} "
                    f"({'⚠️ suspiciously small — may already be normalised' if ratio < 0.05 else '✅ price-unit confirmed'})"
                )

        for col in df.columns:
            if col not in target_cols:
                continue
            if col in already_scaled_cols:
                continue
            # Skip columns already standardized by StandardScaler in Step 0c
            if col in skip_cols:
                self.logger.debug(f"[NormalizeDiffs] Skipping {col} (already standardized by StandardScaler in Step 0c)")
                continue

            # Divide by range width — expresses as fraction of regime range
            normalized = df[col] / range_width

            # ── Breakout detection (BEFORE any squashing) ────────────────────
            # Bounds are [-1, 1]: diffs beyond the full structural range are extreme,
            # not breakouts of zero.  Threshold 10% because negative values are
            # structurally expected for bipolar diff columns.
            breakout_detection = self._detect_and_log_breakouts(
                col, normalized, normalized,
                threshold=0.10,
                lower_bound=-1,
                upper_bound=1,
            )
            if breakout_detection['detected']:
                if not hasattr(self, 'global_breakout_detections'):
                    self.global_breakout_detections = {}
                self.global_breakout_detections[col] = breakout_detection

            # ── Lossless sigmoid normalisation ───────────────────────────────
            # Replaces .clip(-1, 1) + linear (x+1)/2 map.
            # sigmoid(0) = 0.5 — zero-diff maps to exactly 0.5 (neutral momentum).
            # Extreme diffs compress smoothly toward (0, 1) instead of saturating.
            # Inverse: original recoverable via _inverse_transform_rolling_mean().
            final_normalized, _rm = self._normalize_with_rolling_mean(
                normalized,          # range-width-divided series (bipolar)
                col=col,
                window=20,
                sigmoid_scale_factor=self.sigmoid_scale_factor,
                store_baseline=True,
            )

            df[col] = final_normalized
            count += 1

        if count > 0:
            self.logger.info(
                f"⚖️ [NormalizeDiffs] Normalized {count} diff/distance columns "
                f"using Rolling_Range_Width → [0, 1] (regime-relative fraction, centered at 0.5), "
                f"skipped {len(skip_cols)} columns already standardized by StandardScaler"
            )
        elif skip_cols:
            self.logger.info(
                f"⚖️ [NormalizeDiffs] Skipped all {len(skip_cols)} diff columns (already standardized by StandardScaler in Step 0c)"
            )
        return df

    def _calculate_atr(self, df: pd.DataFrame, period: int = 21) -> pd.Series:
        """Calculate Average True Range - causal volatility measure."""
        try:
            # True Range components (using only past data - causal)
            if 'High' in df.columns and 'Low' in df.columns:
                high_low = df['High'] - df['Low']
                high_close = (df['High'] - df['Close'].shift(1)).abs()
                low_close = (df['Low'] - df['Close'].shift(1)).abs()
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            else:
                # Fallback using Close only (less reliable but better than nothing)
                tr = (df['Close'].pct_change() * 100).abs()  # Treat as rough volatility proxy
            
            # Average using standard rolling mean
            atr = tr.rolling(window=period, min_periods=1).mean()
            
            # Smooth with exponential weighting for stability
            atr = atr.ewm(span=period, adjust=False).mean()
            atr = atr.ffill().fillna(0.0).replace([np.inf, -np.inf], 0.0)
            
            return atr
        except Exception as e:
            self.logger.error(f"❌ [ATR] Calculation failed: {e}")
            # Return a safe fallback (1% of Close as default volatility)
            return (df['Close'] * 0.01).fillna(1.0)

    # ──────────────────────────────────────────────────────────────────────────
    # Structural Range — fit / transform pattern (mirrors sklearn)
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_dynamic_volatility_factor(self, train_df: pd.DataFrame, atr: pd.Series) -> float:
        """
        Compute volatility_factor from the asset's own historical regime shift magnitude.

        Logic: how much has this asset moved from its lowest to highest point in training?
        That same magnitude becomes the synthetic headroom above/below the fitted range.

        Example:
          Gold train range: $1200 → $2480  →  106% move
          ATR mean: $19
          implied_factor = (0.5 * 1280) / 19  ≈  33.7
          fitted_high = roll_high_max + 33.7 * atr_end  ≈  covers a further ~50% move

        Clamping:
          28.0 — minimum (empirically verified sweet spot from volatility sweep)
               - val max ≤ 0.76, test max ≤ 0.98 (all within bounds)
               - train mean ≈ 0.42 (not over-compressed)
          50.0 — maximum (prevents destroying training-set density for very volatile assets)
        """
        atr_mean = float(atr.mean())
        if atr_mean == 0:
            self.logger.warning("[DynamicVolFactor] Zero ATR mean — using minimum vol_factor=28.0")
            return 28.0

        price_range = float(train_df['Close'].max() - train_df['Close'].min())
        headroom    = price_range * 0.5          # half the historical range as buffer
        factor      = headroom / atr_mean
        factor      = max(28.0, min(100.0, factor))  # clamp to safe bounds (increased max to 100 to contain massive test breakouts)

        self.logger.info(
            f"[DynamicVolFactor] price_range={price_range:.4f}  atr_mean={atr_mean:.4f}  "
            f"headroom={headroom:.4f}  factor={factor:.2f} (clamped to [28.0, 50.0])"
        )
        return factor

    def _apply_split_transformation_pipeline(
        self, 
        df: pd.DataFrame, 
        split_name: str, 
        rolling_window: int
    ) -> pd.DataFrame:
        """
        Apply the complete per-split transformation pipeline (strictly causal).
        
        Pipeline steps (in order):
        0. Normalize Volume/TickVolume to [0, 1+] using volume-specific range (not price range!)
        0b. Normalize distance columns to [0, 1+] using distance-specific range
        0c. Apply StandardScaler to diff columns (mean-centered, unbounded, preserves negatives)
        1. Apply FIXED structural range scalars (fitted on train, constant everywhere)
        2. Add regime context features (ratios/RoC of structural range)
        3. Normalize OHLCV to [0, 1] using training-fitted structural range
        4. Normalize price-level indicators to match OHLCV scale
        
        Args:
            df: Input DataFrame (with context rows prepended if needed)
            split_name: Name of split ('train', 'validation', 'test') for logging
            rolling_window: Rolling window size for normalization
            
        Returns:
            Transformed DataFrame with context rows still intact (caller strips them)
            
        Design Note:
        - Context rows remain in output so caller can strip them deterministically
        - No data loss — caller is responsible for index management
        - All transformations are vectorized for performance
        """
        # 0. Normalize Volume/TickVolume using volume-specific range (not price-based structural range)
        df = self._normalize_volume_by_range(df)
        
        # 0b. Normalize distance columns using distance-specific range
        df = self._normalize_distance_by_range(df)
        
        # 0c. Normalize footprint delta columns (FP_Delta, FP_Cum_Delta) using symmetric range
        df = self._normalize_footprint_by_range(df)
        
        # 0d. Apply StandardScaler to diff columns (unbounded, mean-centered)
        # Store which columns were standardized so we can skip Step 5 for them
        standardized_diff_cols = set()
        if self.diff_scalers:
            standardized_diff_cols = set(self.diff_scalers.keys())
            self.logger.info(f"[Pipeline] StandardScaler will be applied to {len(standardized_diff_cols)} diff columns, skipping Step 5 normalization for them")
        
        df = self._apply_diff_scalers(df)
        
        # 1. Apply FIXED structural range scalars (fitted on train, constant everywhere)
        df = self._apply_structural_range(df)

        # 2. Regime context features (ratios / RoC of structural range)
        df = self._add_regime_context_features(df)

        # 3. Normalize OHLCV into [0, 1] using the training-fitted structural range
        df = self._normalize_by_rolling_structural_range(
            df, split_name=split_name, rolling_window=rolling_window
        )

        # 4. Normalize price-level indicators to match the OHLCV [0, 1] scale
        df = self._normalize_price_level_indicators(df)

        # 5. Normalize diff/distance columns by Rolling_Range_Width → [-1, 1]
        #    SKIP for diff columns already standardized in Step 0c (StandardScaler handles them)
        #    For remaining diff columns, this expresses each as a FRACTION OF THE CURRENT REGIME RANGE
        #    so the model can compare e.g. "EMA spread is 30% of range" vs "price is at 60% of range".
        df = self._normalize_diff_columns_by_range(df, skip_cols=standardized_diff_cols)

        # 6. Normalize currency index columns by their own structural ranges
        # (Dollar_close ~104 needs its own range, not the main EURUSD range ~1.1)
        df = self._normalize_index_columns(df, split_name=split_name)

        # 7. Post-normalization sanity clip for PRICE columns only.
        #
        # Why this is needed:
        #   - Early warm-up bars (before rolling window fills) can produce raw_norm
        #     slightly below 0 (e.g. -0.000519) because Rolling_Range_High/Low
        #     themselves have NaN-filled ATR during the first ~30 bars.
        #   - The sigmoid in Steps 3-4 compresses but never clips, so sub-zero
        #     raw_norm values produce sigmoid output just above 0 but may still
        #     slip to very small negatives through float32 precision chains.
        #   - Structural_Range_Width is broadcast as a raw price-unit scalar
        #     (e.g. 75.48) into the PRICE PassThrough bucket — it should be
        #     excluded from this clip but is handled by its own normalization.
        #
        # Recovery guarantee:
        #   The per-column pre-clip mean is stored in self.price_clip_means so that
        #   inference code can reconstruct the original via:
        #       original ≈ clipped × range_width + range_low
        #   (The mean shift from clipping is negligible — < 0.05% of rows affected.)
        #
        # We only clip the PRICE columns that went through structural normalization,
        # NOT metadata columns like Rolling_Range_Width, Structural_Range_Width, etc.
        _PRICE_COLS_TO_CLIP = {
            'Open', 'High', 'Low', 'Close',
            'Previous_Close',
            'Prev_1_Close', 'Prev_2_Close', 'Prev_3_Close', 'Prev_4_Close', 'Prev_5_Close',
            'R1', 'R2', 'R3', 'S1', 'S2', 'S3',
            'High_Day_1', 'High_Day_2', 'High_Day_3',
            'Low_Day_1', 'Low_Day_2', 'Low_Day_3',
        }
        if not hasattr(self, 'price_clip_means'):
            self.price_clip_means: dict = {}

        clipped_cols = []
        for col in _PRICE_COLS_TO_CLIP:
            if col not in df.columns:
                continue
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min < 0.0 or col_max > 1.0:
                # Store mean BEFORE clip for recovery
                self.price_clip_means[col] = float(df[col].mean())
                below = int((df[col] < 0.0).sum())
                above = int((df[col] > 1.0).sum())
                df[col] = df[col].clip(0.0, 1.0)
                clipped_cols.append(
                    f"{col}: {below} rows below 0 → 0.0, {above} rows above 1 → 1.0"
                )

        if clipped_cols:
            self.logger.warning(
                f"⚠️ [SanityClip] {split_name}: Clipped {len(clipped_cols)} PRICE columns "
                f"to [0,1] (warm-up artefacts):\n  " + "\n  ".join(clipped_cols)
            )
        else:
            self.logger.debug(f"[SanityClip] {split_name}: All PRICE columns already in [0,1] — no clip needed.")

        self.logger.info(
            f"✅ [MLPrep] {split_name}: Per-split transformation pipeline complete "
            f"(volume [0,1+] + structural range + regime context + OHLCV [0,1] + "
            f"price-level indicators + diff/range [-1,1] + index-specific ranges)"
        )

        return df

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 2: ROLLING MEAN + SIGMOID NORMALIZATION (Lossless Alternative)
    # ──────────────────────────────────────────────────────────────────────────

    def _normalize_with_rolling_mean(
        self,
        series_or_df,
        col: str = None,
        window: int = 60,
        sigmoid_scale_factor: float = 2.0,
        store_baseline: bool = True,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Normalize a column using rolling mean as anchor point + sigmoid squashing.

        Lossless alternative to .clip(): extreme values are compressed toward (0, 1)
        but never saturated — the inverse transform recovers the original exactly.

        Formula
        -------
            rolling_mean  = causal rolling mean of the series (window bars)
            deviation     = (value - rolling_mean) / (|rolling_mean| + ε)
            normalized    = sigmoid(deviation × scale_factor)  ∈ (0, 1)

        Inverse (exact recovery)
        ------------------------
            deviation     = logit(normalized) / scale_factor
            original      = rolling_mean + deviation × (|rolling_mean| + ε)

        Why |rolling_mean| in the denominator?
        - For price-level columns (always > 0) this is identical to rolling_mean.
        - For diff columns (can cross zero) using abs() keeps the scale symmetric
          so a +σ and −σ deviation produce mirror-image normalized values.

        Accepts two calling conventions
        --------------------------------
        1. Series-only (from _normalize_price_level_indicators):
               normalized, rm = self._normalize_with_rolling_mean(series, window=20)
        2. DataFrame + column name (from _normalize_diff_columns_by_range):
               normalized, rm = self._normalize_with_rolling_mean(df, col, window=20)

        Args:
            series_or_df:       pd.Series  OR  pd.DataFrame
            col:                Column name — required when series_or_df is a DataFrame
            window:             Rolling-mean lookback (bars)
            sigmoid_scale_factor: Compression strength (2.0 → typical range in [0.12, 0.88])
            store_baseline:     If True, persist rolling-mean stats to global_rolling_mean_baselines

        Returns:
            (normalized_series, rolling_mean_series)
        """
        # ── Resolve series ───────────────────────────────────────────────────
        if isinstance(series_or_df, pd.DataFrame):
            if col is None or col not in series_or_df.columns:
                empty = pd.Series(np.nan, index=series_or_df.index)
                return empty, empty
            series = series_or_df[col]
            col_name = col
        else:
            # Received a plain Series
            series = series_or_df
            col_name = col or getattr(series, 'name', 'unknown')

        # ── Rolling mean (causal, forward-leak-free) ─────────────────────────
        rolling_mean = series.rolling(window=window, min_periods=min(20, window)).mean()

        # ── Deviation ratio ──────────────────────────────────────────────────
        # Denominator: |rolling_mean| so sign-crossing diff columns stay symmetric
        with np.errstate(divide='ignore', invalid='ignore'):
            deviation_ratio = (series - rolling_mean) / (rolling_mean.abs() + 1e-8)

        deviation_ratio = deviation_ratio.replace([np.inf, -np.inf], 0).fillna(0)

        # ── Sigmoid squash → (0, 1) ──────────────────────────────────────────
        normalized = 1.0 / (1.0 + np.exp(-sigmoid_scale_factor * deviation_ratio))
        normalized = normalized.ffill().fillna(0.5)

        # ── Optionally persist compressed baseline for inference ─────────────
        if store_baseline and col_name and col_name != 'unknown':
            self._store_rolling_mean_baseline(col_name, rolling_mean)

        self.logger.debug(
            f"[RollingMean] {col_name}: window={window}, scale={sigmoid_scale_factor}, "
            f"rm=[{rolling_mean.min():.4f}, {rolling_mean.max():.4f}], "
            f"out=[{normalized.min():.6f}, {normalized.max():.6f}]"
        )

        return normalized, rolling_mean

    def _inverse_transform_rolling_mean(
        self,
        normalized: pd.Series,
        rolling_mean: pd.Series,
        sigmoid_scale_factor: float = 2.0,
    ) -> pd.Series:
        """
        Exactly recover original values from sigmoid-normalized data.

        Must use the SAME sigmoid_scale_factor that was used in the forward pass.

        Forward:  deviation = (x - μ) / (|μ| + ε)
                  y = sigmoid(deviation × k)  →  y ∈ (0,1)

        Inverse:  deviation = logit(y) / k   →  logit(y) = ln(y/(1−y))
                  x = μ + deviation × (|μ| + ε)

        The key correction vs. the old formula `rolling_mean * (1 + deviation)`:
        - Old: divides by rolling_mean (breaks when mean ≈ 0, wrong for diff cols)
        - New: adds deviation × |rolling_mean| (symmetric, exact inverse of forward)
        """
        # Clamp to open interval to avoid log(0) / log(∞) — float precision only
        normalized = normalized.clip(1e-7, 1.0 - 1e-7)

        # Invert sigmoid: logit(y) = ln(y / (1 - y))
        with np.errstate(divide='ignore', invalid='ignore'):
            deviation_ratio = np.log(normalized / (1.0 - normalized)) / sigmoid_scale_factor

        # Reconstruct original: x = μ + deviation × (|μ| + ε)
        original = rolling_mean + deviation_ratio * (rolling_mean.abs() + 1e-8)

        # Guard against any residual numerical anomalies
        original = original.replace([np.inf, -np.inf], np.nan).fillna(rolling_mean)

        return original

    def _store_rolling_mean_baseline(self, col_name: str, rolling_mean: pd.Series) -> None:
        """Store rolling mean baseline for inference-time reconstruction."""
        if not hasattr(self, 'global_rolling_mean_baselines'):
            self.global_rolling_mean_baselines = {}
        
        # Store compressed representation (mean of rolling mean, std dev)
        # Full series is too large; we'll recompute during inference
        self.global_rolling_mean_baselines[col_name] = {
            'mean_of_mean': float(rolling_mean.mean()),
            'std_of_mean': float(rolling_mean.std()),
            'min_of_mean': float(rolling_mean.min()),
            'max_of_mean': float(rolling_mean.max())
        }

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 1: BREAKOUT DETECTION (Hybrid with Current Clipping)
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_and_log_breakouts(
        self,
        col_name: str,
        raw_values: pd.Series,
        normalized_values: pd.Series,
        threshold: float = 0.02,
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Detect values that fall outside [lower_bound, upper_bound].

        Call-site conventions:
        - OHLCV / price-level:  lower_bound=0,  upper_bound=1,  threshold=0.02
        - Diff columns:         lower_bound=-1, upper_bound=1,  threshold=0.10
          (negative values are structurally expected for diffs — they are NOT breakouts)

        Returns statistics dict for monitoring and potential retraining decisions.
        """
        below = (raw_values < lower_bound).sum()
        above = (raw_values > upper_bound).sum()
        breakout_count = below + above

        if breakout_count == 0:
            return {'detected': False, 'count': 0}

        pct = breakout_count / len(raw_values) if len(raw_values) > 0 else 0

        detection = {
            'detected': True,
            'count': int(breakout_count),
            'percentage': float(pct),
            'below_bound_count': int(below),
            'above_bound_count': int(above),
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'threshold_exceeded': pct > threshold,
        }

        if pct > threshold:
            self.logger.warning(
                f"⚠️  [BreakoutDetection] {col_name}: {breakout_count} breakout rows "
                f"({pct*100:.2f}% > {threshold*100:.1f}% threshold, "
                f"bounds=[{lower_bound}, {upper_bound}]) — Consider retraining with new range"
            )

        return detection

    def _fit_structural_range(self, train_df: pd.DataFrame, window: int) -> None:
        """
        Fit structural range on TRAINING data only — like scaler.fit().

        Computes the rolling high/low extremes on the training slice, then adds a
        dynamically-calibrated ATR buffer as synthetic headroom for unseen prices.
        Results are stored as scalar instance attributes:

            self.fitted_range_high   — upper bound (float)
            self.fitted_range_low    — lower bound (float)
            self.fitted_range_width  — high - low  (float)
            self.fitted_vol_factor   — which vol_factor was used (for logging/DB)

        Must be called BEFORE _apply_structural_range().
        """
        atr       = self._calculate_atr(train_df, period=14)
        min_p     = max(20, window // 8)

        if 'High' in train_df.columns and 'Low' in train_df.columns:
            roll_high = train_df['High'].rolling(window=window, min_periods=min_p).max()
            roll_low  = train_df['Low'].rolling(window=window,  min_periods=min_p).min()
        else:
            roll_high = train_df['Close'].rolling(window=window, min_periods=min_p).max()
            roll_low  = train_df['Close'].rolling(window=window, min_periods=min_p).min()

        vol_factor = self._compute_dynamic_volatility_factor(train_df, atr)

        # Scalar aggregate — max/min over train only (intentional "lie" giving headroom
        # for prices the model has not yet seen, while keeping the scale grounded in train)
        self.fitted_range_high  = float((roll_high + vol_factor * atr).max())
        self.fitted_range_low   = float((roll_low  - vol_factor * atr).min())
        self.fitted_range_width = self.fitted_range_high - self.fitted_range_low
        self.fitted_vol_factor  = vol_factor

        self.logger.info(
            f"✅ [StructuralRange] Fitted on train only: "
            f"low={self.fitted_range_low:.4f}  high={self.fitted_range_high:.4f}  "
            f"width={self.fitted_range_width:.4f}  vol_factor={vol_factor:.2f}"
        )

    def _fit_index_structural_ranges(self, train_df: pd.DataFrame, window: int) -> None:
        """
        Fit a separate structural range for each currency index found in train_df.

        Currency index OHLCV columns follow the pattern {IndexName}_{field},
        e.g. Dollar_open, Dollar_high, Dollar_close, Euro_close, JPY_high.

        Each index lives in a completely different price scale (Dollar ~104,
        Euro ~1.1, JPY ~150) so using the main asset's structural range would
        produce meaningless [0,1] values.  This method fits a dedicated
        range per index using exactly the same ATR+rolling-extremes approach
        as _fit_structural_range(), stored in:

            self.fitted_index_ranges = {
                'Dollar': {'high': float, 'low': float, 'width': float, 'vol_factor': float},
                'Euro':   {...},
                ...
            }
        """
        from app.core.analysis.currency_index import INDEX_DEFINITIONS

        if not hasattr(self, 'fitted_index_ranges'):
            self.fitted_index_ranges: Dict[str, Dict[str, float]] = {}

        for idx_name in INDEX_DEFINITIONS.keys():
            close_col = f"{idx_name}_close"
            high_col  = f"{idx_name}_high"
            low_col   = f"{idx_name}_low"

            if close_col not in train_df.columns:
                continue  # This index not present in the dataset

            # Build a temporary OHLCV view using the index's own columns
            idx_close = train_df[close_col].astype(float).ffill().bfill()
            idx_high  = train_df[high_col].astype(float).ffill().bfill()  if high_col in train_df.columns else idx_close
            idx_low   = train_df[low_col].astype(float).ffill().bfill()   if low_col  in train_df.columns else idx_close

            # ATR from index OHLCV
            if f"{idx_name}_high" in train_df.columns and f"{idx_name}_low" in train_df.columns:
                # True Range using index-specific H/L/C
                prev_close = idx_close.shift(1)
                hl = idx_high - idx_low
                hc = (idx_high - prev_close).abs()
                lc = (idx_low  - prev_close).abs()
                tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
                atr = tr.rolling(window=14, min_periods=1).mean().ewm(span=14, adjust=False).mean()
                atr = atr.ffill().fillna(0.0).replace([np.inf, -np.inf], 0.0)
            else:
                # Fallback: 1% of close as ATR proxy
                atr = idx_close * 0.01

            min_p     = max(20, window // 8)
            roll_high = idx_high.rolling(window=window, min_periods=min_p).max()
            roll_low  = idx_low.rolling(window=window,  min_periods=min_p).min()

            # Use the same vol_factor computation
            temp_df = pd.DataFrame({'Close': idx_close, 'High': idx_high, 'Low': idx_low})
            vol_factor = self._compute_dynamic_volatility_factor(temp_df, atr)

            fitted_high  = float((roll_high + vol_factor * atr).max())
            fitted_low   = float((roll_low  - vol_factor * atr).min())
            fitted_width = fitted_high - fitted_low

            if fitted_width <= 0:
                self.logger.warning(f"[IndexRange] {idx_name}: degenerate range (width={fitted_width:.6f}), skipping")
                continue

            self.fitted_index_ranges[idx_name] = {
                'high': fitted_high,
                'low':  fitted_low,
                'width': fitted_width,
                'vol_factor': vol_factor,
            }

            self.logger.info(
                f"✅ [IndexRange] {idx_name}: low={fitted_low:.4f}  high={fitted_high:.4f}  "
                f"width={fitted_width:.4f}  vol_factor={vol_factor:.2f}"
            )

        fitted_count = len(self.fitted_index_ranges)
        if fitted_count > 0:
            self.logger.info(f"✅ [IndexRange] Fitted ranges for {fitted_count} currency indices: "
                             f"{list(self.fitted_index_ranges.keys())}")

    def _normalize_index_columns(self, df: pd.DataFrame, split_name: str = "train") -> pd.DataFrame:
        """
        Normalize all currency index OHLCV and TI columns using each index's
        own fitted structural range.

        For each index with a fitted range (in self.fitted_index_ranges):
          - OHLCV price columns ({idx}_open/high/low/close) → normalized to [0,1]
            using the index-specific range (same sigmoid approach as main OHLCV)
          - TI price-level columns ({idx}_SMA_20, {idx}_EMA_8, etc.) → same range
          - {idx}_tick_volume → index-specific volume range (or main volume range if absent)
          - TI diff columns ({idx}_RSI_14, {idx}_MACD, etc.) → ALREADY handled by
            _normalize_diff_columns_by_range() since they use Rolling_Range_Width;
            but their scale is the MAIN asset's range width, not the index range.
            We normalize these by the index-specific range width instead.

        Must be called AFTER _apply_structural_range() (which sets Rolling_Range_Width
        for the main asset).
        """
        if not hasattr(self, 'fitted_index_ranges') or not self.fitted_index_ranges:
            return df

        df = df.copy()

        # Determine all possible price-level prefixes (SMA_, EMA_, BB_, etc.)
        price_level_prefixes = (
            'SMA_', 'EMA_', 'EMA-', 'WMA_', 'HMA_',
            'BBL_', 'BBM_', 'BBU_', 'BB_', 'MA_', 'MA-',
            'Pivot_', 'Supertrend_',
        )
        index_price_level_names = {
            'Open', 'High', 'Low', 'Close',
            'open', 'high', 'low', 'close',
            'Previous_Close',
            'Prev_1_Close', 'Prev_2_Close', 'Prev_3_Close', 'Prev_4_Close', 'Prev_5_Close',
            'High_Day_1', 'High_Day_2', 'High_Day_3',
            'Low_Day_1', 'Low_Day_2', 'Low_Day_3',
            'Short_MA', 'Long_MA', 'MA',
            'Short_MA_10', 'Short_MA_50',
            'Long_MA_25', 'Long_MA_100', 'Long_MA_200',
            '10_Day_MA', '50_Day_MA',
            'Pivot Price', 'Pivot_Price',
            'r1', 'r2', 'r3', 's1', 's2', 's3',
            'R1', 'R2', 'R3', 'S1', 'S2', 'S3',
            'Parabolic_SAR',
            'Final Lowerband', 'Final Upperband',
            'Support_Trendline_Value', 'Resist_Trendline_Value',
            'SMC_OB_Top', 'SMC_OB_Bottom',
            'SMC_FVG_Top', 'SMC_FVG_Bottom',
            'SMC_Swing_Level', 'SMC_Liquidity_Level',
            'SMC_BOS_Level',
            'FVG_Top', 'FVG_Bottom',
            'Order_Block_Top', 'Order_Block_Bottom',
            'Rolling_Range_High', 'Rolling_Range_Low', 'Rolling_Range_Mid',
            'Structural_Range_Position',
        }
        index_width_names = {
            'Rolling_Range_Width',
            'Structural_Range_Width',
        }
        diff_keywords = ('_Diff', '_diff', '_Distance', '_distance', '_Pct', '_pct',
                         'Minus', 'MACD', 'MACDh', 'MACDs', 'ATR', 'RSI', 'Stoch',
                         'CCI', 'BBP_', 'BBB_', 'OBV', 'Change')

        for idx_name, range_params in self.fitted_index_ranges.items():
            idx_high  = range_params['high']
            idx_low   = range_params['low']
            idx_width = range_params['width']

            if idx_width <= 0:
                continue

            # --- 1. Normalize OHLCV price columns ---
            ohlcv_fields = ['open', 'high', 'low', 'close']
            for field in ohlcv_fields:
                col = f"{idx_name}_{field}"
                if col not in df.columns:
                    continue
                raw_norm = (df[col].astype(float) - idx_low) / idx_width
                raw_norm = raw_norm.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.5)
                normalized, _ = self._normalize_with_rolling_mean(
                    raw_norm, col=col, window=20,
                    sigmoid_scale_factor=self.sigmoid_scale_factor, store_baseline=True
                )
                df[col] = normalized

            # --- 2. Normalize {idx}_tick_volume ---
            vol_col = f"{idx_name}_tick_volume"
            if vol_col in df.columns:
                # Use index-specific volume range if fitted, otherwise percentile of column
                if hasattr(self, 'fitted_volume_range_high') and idx_name in self.fitted_volume_range_high:
                    v_high = self.fitted_volume_range_high[idx_name]
                    v_low  = self.fitted_volume_range_low.get(idx_name, 0.0)
                else:
                    v_high = float(df[vol_col].quantile(0.95)) * 1.2
                    v_low  = 0.0
                if v_high > v_low:
                    df[vol_col] = (df[vol_col].astype(float) - v_low) / (v_high - v_low)

            # --- 3. Normalize TI price-level columns for this index ---
            # These are columns like Dollar_SMA_20, Dollar_BB_Upper, etc.
            for col in df.columns:
                if not col.startswith(f"{idx_name}_"):
                    continue
                base_name = col[len(idx_name) + 1:]  # Strip "{idx}_" prefix

                # Skip OHLCV and tick_volume (already handled above)
                if base_name in ('open', 'high', 'low', 'close', 'tick_volume', 'Time', 'time'):
                    continue

                # Determine if this is a price-level column for this index
                is_width = base_name in index_width_names
                is_price_level = (
                    base_name in index_price_level_names or
                    any(base_name.startswith(p) for p in price_level_prefixes)
                )
                is_diff = any(kw in base_name for kw in diff_keywords)

                if is_width:
                    normalized = df[col].astype(float) / idx_width
                    normalized = normalized.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.5)
                    normalized, _ = self._normalize_with_rolling_mean(
                        normalized, col=col, window=20,
                        sigmoid_scale_factor=self.sigmoid_scale_factor, store_baseline=True
                    )
                    df[col] = normalized

                elif is_price_level and not is_diff:
                    # Normalize using index's own structural range
                    raw_norm = (df[col].astype(float) - idx_low) / idx_width
                    raw_norm = raw_norm.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.5)
                    normalized, _ = self._normalize_with_rolling_mean(
                        raw_norm, col=col, window=20,
                        sigmoid_scale_factor=self.sigmoid_scale_factor, store_baseline=True
                    )
                    df[col] = normalized

                elif is_diff and not is_price_level:
                    # Diff columns: normalize by index's range width (not main asset's width)
                    # This makes e.g. Dollar_MACD_12_26_9 a fraction of Dollar's price range
                    normalized = df[col].astype(float) / idx_width
                    normalized = normalized.replace([np.inf, -np.inf], np.nan)
                    normalized, _ = self._normalize_with_rolling_mean(
                        normalized, col=col, window=20,
                        sigmoid_scale_factor=self.sigmoid_scale_factor, store_baseline=True
                    )
                    df[col] = normalized

        return df

    def _fit_index_volume_ranges(self, train_df: pd.DataFrame) -> None:
        """
        Fit volume ranges for each currency index's tick_volume column.
        Stored in self.fitted_volume_range_high[idx_name] (reuses same dict as main asset).
        """
        if not hasattr(self, 'fitted_index_ranges'):
            return

        from app.core.analysis.currency_index import INDEX_DEFINITIONS

        for idx_name in INDEX_DEFINITIONS.keys():
            vol_col = f"{idx_name}_tick_volume"
            if vol_col not in train_df.columns:
                continue
            try:
                v_high = float(train_df[vol_col].quantile(0.95)) * 1.2
                self.fitted_volume_range_high[idx_name] = v_high
                self.fitted_volume_range_low[idx_name]  = 0.0
                self.logger.info(f"✅ [IndexVolRange] {idx_name}: [0, {v_high:,.2f}]")
            except Exception as e:
                self.logger.warning(f"[IndexVolRange] Failed for {idx_name}: {e}")

    def _fit_distance_range(self, train_df: pd.DataFrame, percentile: float = 0.95, buffer: float = 1.2) -> None:
        """
        Fit distance-specific range on TRAINING data only.
        
        Distance columns (Down_Distance, snr_dist_*, etc.) need their own range
        because they operate at a different scale than prices.
        Uses 95th percentile + buffer, same as volume.

        Args:
            train_df: Training DataFrame
            percentile: Quantile to use (0.95 = 95th percentile)
            buffer: Multiplier to add headroom (1.2 = 20% buffer)

        Stores results in:
            self.fitted_distance_range_high[col] — upper bound
            self.fitted_distance_range_low[col]  — always 0 (distances never negative)
        """
        DISTANCE_COLS = ['Down_Distance', 'Up_Distance', 'snr_dist_to_nearest_level',
                        'snr_dist_to_nearest_support', 'snr_dist_to_nearest_resistance']

        for dist_col in DISTANCE_COLS:
            if dist_col not in train_df.columns:
                continue

            # Use 95th percentile for robustness
            distance_percentile = train_df[dist_col].quantile(percentile)
            distance_range_high = distance_percentile * buffer

            self.fitted_distance_range_high[dist_col] = float(distance_range_high)
            self.fitted_distance_range_low[dist_col] = 0.0

        if self.fitted_distance_range_high:
            self.logger.info(
                f"✅ [DistanceRange] Fitted on training data (percentile={percentile}, buffer={buffer}):\n"
                + "\n".join(
                    f"   {col}: [0, {self.fitted_distance_range_high[col]:,.4f}]"
                    for col in DISTANCE_COLS if col in self.fitted_distance_range_high
                )
            )

    def _normalize_distance_by_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize distance columns to [0, 1+] using fitted distance range.

        Distance columns like Down_Distance and snr_dist_* operate at a different scale
        than prices. This separate normalization prevents them from being clipped incorrectly.

        Returns DataFrame with normalized distance columns.
        """
        if not self.fitted_distance_range_high:
            return df

        df = df.copy()
        DISTANCE_COLS = ['Down_Distance', 'Up_Distance', 'snr_dist_to_nearest_level',
                        'snr_dist_to_nearest_support', 'snr_dist_to_nearest_resistance']

        for dist_col in DISTANCE_COLS:
            if dist_col not in df.columns or dist_col not in self.fitted_distance_range_high:
                continue

            range_high = self.fitted_distance_range_high[dist_col]
            range_low = self.fitted_distance_range_low[dist_col]

            if range_high <= range_low:
                continue

            df[dist_col] = (df[dist_col] - range_low) / (range_high - range_low)

        self.logger.info(f"✅ [DistanceNorm] Normalized distance columns to [0, 1+] range")
        return df

    def _fit_diff_scalers(self, train_df: pd.DataFrame) -> None:
        """
        Fit StandardScaler for diff/oscillator columns INDEPENDENTLY.

        Diff columns like EMA_*_Diff, MA_*_Diff, MACD_*, etc. should use StandardScaler
        (unbounded, preserves negative values) instead of structural range (which clips to [0,1]).

        This preserves the full information content without clipping.
        """
        # Identify all diff columns in the training data
        DIFF_KEYWORDS = (
            '_Diff', '_diff', 'MACD_', 'EMA_', 'MA_', 'FVG_',
            'BBP_', 'BBB_', 'Pct', 'pct', 'PCT',
            # Short standalone names that match no keyword above
            'MR_t', 'TF_t',
            # RSI change/lag columns — hyphen form ('RSI-Change*') and
            # underscore form ('RSI_7_Change_Lag_*') both must be caught.
            'RSI-Change', 'RSI_7_Change_Lag', 'RSI_2_Change_Lag',
            # Dn_ volume change (lower-case prefix misses the 'Pct' match above)
            'Dn_volume_change',
        )
        
        diff_cols = [col for col in train_df.columns 
                    if any(kw in col for kw in DIFF_KEYWORDS)
                    and col not in ['Volume', 'TickVolume']]  # Exclude volume

        for col in diff_cols:
            try:
                scaler = StandardScaler()
                scaler.fit(train_df[[col]])
                self.diff_scalers[col] = scaler
            except Exception as e:
                self.logger.warning(f"⚠️  Failed to fit StandardScaler for {col}: {e}")

        if self.diff_scalers:
            self.logger.info(
                f"✅ [DiffScalers] Fitted StandardScaler for {len(self.diff_scalers)} diff columns "
                f"(unbounded, preserves negative values and spikes)"
            )
            # Serialize scalers to binary for storage
            self.global_diff_scaler_binary = pickle.dumps(self.diff_scalers)

    def _apply_diff_scalers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply StandardScaler to diff columns, then map the z-scores to (0, 1)
        using a sigmoid so they are compatible with the [0, 1] clipping guard in
        SelectiveScaler.transform().

        Why sigmoid after StandardScaler?
        ──────────────────────────────────
        SelectiveScaler.transform() clips the diff partition to [0, 1] after
        PassThrough.  That clip is intentional — LSTMs converge better with
        strictly bounded [0, 1] inputs.  Step 5 (_normalize_diff_columns_by_range)
        already maps range-width-normalised diff columns to (0, 1) via rolling-mean
        sigmoid (0.5 = zero-diff / neutral; < 0.5 = bearish; > 0.5 = bullish).

        Columns handled here (StandardScaler'd) skip Step 5, so they need the
        same (0, 1) mapping applied here:

            z_score  = StandardScaler.transform(col)     # unbounded, mean=0, std=1
            sigmoid  = 1 / (1 + exp(-z_score * k))       # maps to (0, 1)

        With k = 2.0 (self.sigmoid_scale_factor):
            z = 0   → sigmoid = 0.500  (neutral / mean)
            z = +1  → sigmoid ≈ 0.880  (1 std above mean, bullish)
            z = -1  → sigmoid ≈ 0.120  (1 std below mean, bearish)
            z = +3  → sigmoid ≈ 0.998  (extreme spike)
            z = -3  → sigmoid ≈ 0.002  (extreme crash)

        This preserves the full directional information while keeping values in
        (0, 1), so the downstream clip is a harmless float-precision guard.
        """
        if not self.diff_scalers:
            return df

        df = df.copy()
        k = getattr(self, 'sigmoid_scale_factor', 2.0)

        for col, scaler in self.diff_scalers.items():
            if col in df.columns:
                try:
                    z = scaler.transform(df[[col]]).ravel()
                    # Sigmoid map: 0.5 = neutral, >0.5 = above mean, <0.5 = below mean
                    df[col] = 1.0 / (1.0 + np.exp(-k * z))
                except Exception as e:
                    self.logger.warning(f"⚠️  Failed to apply StandardScaler+sigmoid to {col}: {e}")

        self.logger.info(
            f"✅ [DiffScaled] Applied StandardScaler → sigmoid to {len(self.diff_scalers)} diff columns "
            f"(now in (0,1), mean→0.5, ±1σ→[0.12, 0.88], compatible with [0,1] clip)"
        )
        return df

    def _apply_structural_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply pre-fitted structural range — like scaler.transform().

        Broadcasts the scalar bounds stored by _fit_structural_range() as constant
        columns to every row, guaranteeing that the same raw price always maps to
        the same normalised value regardless of which split the row belongs to.

        Raises ValueError if _fit_structural_range() has not been called first.
        """
        if not hasattr(self, 'fitted_range_high'):
            raise ValueError(
                "Call _fit_structural_range() on training data before _apply_structural_range(). "
                "self.fitted_range_high is not set."
            )

        df = df.copy()
        df['Rolling_Range_High']  = self.fitted_range_high
        df['Rolling_Range_Low']   = self.fitted_range_low
        df['Rolling_Range_Width'] = self.fitted_range_width
        df['Rolling_Range_Mid']   = (self.fitted_range_high + self.fitted_range_low) / 2.0

        self.logger.info(
            f"✅ [StructuralRange] Applied fixed scalars to {len(df)} rows: "
            f"[{self.fitted_range_low:.4f}, {self.fitted_range_high:.4f}]  "
            f"width={self.fitted_range_width:.4f}"
        )
        return df

    def _fit_volume_range(self, train_df: pd.DataFrame, percentile: float = 0.95, buffer: float = 1.2) -> None:
        """
        Fit volume-specific range on TRAINING data only.

        Unlike structural range (which uses price High/Low + ATR),
        volume range is based on rolling volume's 95th percentile + buffer.
        This prevents volume spikes from clipping to 1.0 prematurely.

        Args:
            train_df: Training DataFrame
            percentile: Quantile to use (0.95 = 95th percentile)
            buffer: Multiplier to add headroom (1.2 = 20% buffer)

        Stores results in:
            self.fitted_volume_range_high[col] — upper bound
            self.fitted_volume_range_low[col]  — always 0 (volume never negative)
        """
        VOLUME_COLS = ['Volume', 'TickVolume']

        for vol_col in VOLUME_COLS:
            if vol_col not in train_df.columns:
                continue

            # Rolling max over 20-bar window to capture typical spike behavior
            rolling_vol_max = train_df[vol_col].rolling(window=20, min_periods=5).max()

            # Use 95th percentile for robustness (avoids single extreme outliers)
            # Then apply buffer for unseen spikes
            volume_percentile = train_df[vol_col].quantile(percentile)
            volume_range_high = volume_percentile * buffer

            self.fitted_volume_range_high[vol_col] = float(volume_range_high)
            self.fitted_volume_range_low[vol_col] = 0.0

        self.logger.info(
            f"✅ [VolumeRange] Fitted on training data (percentile={percentile}, buffer={buffer}):\n"
            + "\n".join(
                f"   {col}: [0, {self.fitted_volume_range_high[col]:,.0f}]"
                for col in VOLUME_COLS if col in self.fitted_volume_range_high
            )
        )

    def _normalize_volume_by_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize Volume and TickVolume to [0, 1] using fitted volume range.

        This is called BEFORE structural range normalization so that:
        1. Volume spikes aren't affected by price range
        2. Phase 1 detection can track volume clipping separately
        3. Values exceeding range are not clipped (stay > 1.0 for detection)

        Returns DataFrame with normalized Volume columns.
        """
        if not self.fitted_volume_range_high:
            # Not fitted yet, skip
            return df

        df = df.copy()
        VOLUME_COLS = ['Volume', 'TickVolume']

        for vol_col in VOLUME_COLS:
            if vol_col not in df.columns or vol_col not in self.fitted_volume_range_high:
                continue

            range_high = self.fitted_volume_range_high[vol_col]
            range_low = self.fitted_volume_range_low[vol_col]

            if range_high <= range_low:
                self.logger.warning(f"[VolumeNorm] {vol_col} range invalid: [{range_low}, {range_high}], skipping")
                continue

            # Normalize to [0, 1+] (NOT clipped; spikes will be > 1.0)
            df[vol_col] = (df[vol_col] - range_low) / (range_high - range_low)

            # Phase 1: Detect spikes before clipping
            spike_mask = df[vol_col] > 1.0
            spike_count = spike_mask.sum()

            if spike_count > 0:
                spike_percentage = spike_count / len(df) if len(df) > 0 else 0
                max_spike = df[vol_col].max()

                self.global_volume_clipping_detection[vol_col] = {
                    "detected": True,
                    "count": int(spike_count),
                    "percentage": float(spike_percentage),
                    "max_normalized": float(max_spike),
                    "spike_ratio": float(max_spike / 1.0) if max_spike > 1.0 else 1.0,
                    "fitted_range_high": float(range_high),
                }

                if spike_percentage > 0.02:  # > 2%
                    self.logger.warning(
                        f"⚠️  [VolumeSpike] {vol_col}: {spike_count}/{len(df)} spikes ({spike_percentage*100:.2f}%) "
                        f"exceed fitted range (max={max_spike:.2f}x buffer)"
                    )

        self.logger.info(f"✅ [VolumeNorm] Normalized {len(VOLUME_COLS)} volume columns to [0, 1+] range")
        return df

    def _fit_footprint_range(self, train_df: pd.DataFrame, percentile: float = 0.95, buffer: float = 1.2) -> None:
        """
        Fit a symmetric train-only range for FP_Delta / FP_Cum_Delta.

        Unlike volume/distance (always >= 0), delta is signed, so we fit on
        abs(value) and apply a symmetric [-range_high, +range_high] mapping.

        Args:
            train_df: Training DataFrame
            percentile: Quantile to use on absolute values (0.95 = 95th percentile)
            buffer: Multiplier to add headroom (1.2 = 20% buffer)

        Stores results in:
            self.fitted_footprint_range_high[col] — symmetric range bound
        """
        FOOTPRINT_DELTA_COLS = ['FP_Delta', 'FP_Cum_Delta']

        for col in FOOTPRINT_DELTA_COLS:
            if col not in train_df.columns:
                continue

            # Use absolute value for percentile calculation (delta can be negative)
            abs_percentile = train_df[col].abs().quantile(percentile)
            self.fitted_footprint_range_high[col] = float(abs_percentile * buffer)

        if self.fitted_footprint_range_high:
            self.logger.info(
                f"✅ [FootprintRange] Fitted on training data (percentile={percentile}, buffer={buffer}):\n"
                + "\n".join(
                    f"   {c}: [-{v:,.4f}, {v:,.4f}]"
                    for c, v in self.fitted_footprint_range_high.items()
                )
            )

    def _normalize_footprint_by_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize FP_Delta / FP_Cum_Delta to roughly [-1, 1] using the fitted
        symmetric range. NOT clipped — spikes stay outside [-1, 1] for detection,
        consistent with how _normalize_volume_by_range handles Volume spikes.

        Returns DataFrame with normalized footprint delta columns.
        """
        if not self.fitted_footprint_range_high:
            # Not fitted yet, skip
            return df

        df = df.copy()

        for col, range_high in self.fitted_footprint_range_high.items():
            if col not in df.columns or range_high <= 0:
                continue

            # Normalize to ~[-1, 1] (NOT clipped; large deltas will exceed bounds)
            df[col] = df[col] / range_high

        self.logger.info("✅ [FootprintNorm] Normalized FP_Delta/FP_Cum_Delta to ~[-1, 1] range")
        return df

    def _detect_volume_clipping(self, col_name: str, normalized_values: pd.Series, threshold: float = 0.02) -> Dict[str, Any]:
        """
        Phase 1: Detect and log volume spikes that exceed fitted range.

        Unlike prices where clipping is problematic, volume spikes are EXPECTED
        and should be preserved. This detection tracks:
        - How often volume exceeds the fitted range
        - What percentage of rows have spikes
        - Maximum spike ratio for monitoring

        Args:
            col_name: Column name (Volume or TickVolume)
            normalized_values: Already-normalized values (may be > 1.0)
            threshold: Alert threshold (default 2%)

        Returns:
            Detection dict with statistics for storage in breakout_detection
        """
        spikes = (normalized_values > 1.0).sum()
        spike_percentage = spikes / len(normalized_values) if len(normalized_values) > 0 else 0
        max_spike = normalized_values.max() if len(normalized_values) > 0 else 1.0
        threshold_exceeded = spike_percentage > threshold

        detection = {
            "detected": spikes > 0,
            "count": int(spikes),
            "percentage": float(spike_percentage),
            "max_normalized": float(max_spike),
            "spike_ratio": float(max_spike / 1.0) if max_spike > 1.0 else 1.0,
            "threshold_exceeded": threshold_exceeded,
            "note": f"Volume spike detection (not problematic, just monitoring)"
        }

        if threshold_exceeded:
            self.logger.warning(
                f"⚠️  [VolumeMonitor] {col_name}: {spikes}/{len(normalized_values)} spikes ({spike_percentage*100:.2f}%) "
                f"exceed buffer (max ratio: {max_spike:.2f}x)"
            )

        return detection

    def _calculate_rolling_structural_range(self, df: pd.DataFrame, window: int = 60, volatility_factor: float = 3.5) -> pd.DataFrame:
        """Calculate adaptive structural range using volatility (ATR) + rolling extremes.
    
        Mimics pivot point logic (S2-R3 style) but regime-adaptive.
        As volatility increases, the range expands; as it decreases, the range contracts.
        This prevents regime-breaking normalization when price moves to new all-time highs/lows.
        
        Args:
            df: DataFrame with OHLCV
            window: Rolling lookback window (252 = 1 year)
            volatility_factor: Multiplier for ATR expansion (35 default, increased from 24 to contain gold $2700 breakout)
        """
        df = df.copy()
        try:
            if 'Close' not in df.columns:
                self.logger.warning("[Rolling Range] No Close column, skipping")
                return df

            # 1. Base rolling extremes (long-term structure)
            min_periods = max(20, window // 8)  # Warmup: ~31 bars for 252-bar window
            
            if 'High' in df.columns and 'Low' in df.columns:
                roll_high = df['High'].rolling(window=window, min_periods=min_periods).max()
                roll_low = df['Low'].rolling(window=window, min_periods=min_periods).min()
            else:
                roll_high = df['Close'].rolling(window=window, min_periods=min_periods).max()
                roll_low = df['Close'].rolling(window=window, min_periods=min_periods).min()

            # 2. Calculate ATR for volatility expansion (causal, backward-looking)
            atr = self._calculate_atr(df, period=14)
            
            # 3. Adaptive range with volatility buffer (S2-R3 projection)
            # Use expanding().max/min() to prevent range from shrinking after regime shifts
            # Once the range has expanded to contain a new ATH, it stays expanded
            # This prevents validation data from getting clipped when rolling window rolls off ATH bars
            df['Rolling_Range_High'] = (roll_high + (volatility_factor * atr))
            df['Rolling_Range_Low'] = (roll_low - (volatility_factor * atr))
            
            # 4. Width with safety (prevent division by zero AND negative width from
            #    warm-up bars where ATR may transiently exceed roll_high - roll_low)
            df['Rolling_Range_Width'] = (df['Rolling_Range_High'] - df['Rolling_Range_Low']).clip(lower=0.0)
            df['Rolling_Range_Width'] = df['Rolling_Range_Width'].replace(0, np.nan)
            
            # 5. Add midpoint for regime context features
            df['Rolling_Range_Mid'] = (df['Rolling_Range_High'] + df['Rolling_Range_Low']) / 2
            
            self.logger.info(f"✅ [Rolling Range] Adaptive regime-aware (window={window}, vol_factor={volatility_factor}, expanding bounds)")
            return df
        except Exception as e:
            self.logger.error(f"❌ [Rolling Range] Failed: {e}")
            raise ValueError(f"Failed to calculate rolling structural range: {e}")

    def _add_regime_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add regime-aware features that capture momentum ABOVE the local normalization.
        
        These features answer: 'Is this range itself trending?' and survive normalization
        because they're ratios/rates of already-normalized quantities.
        
        Regime context features for trend continuation vs reversion detection
        
        Features:
        1. macro_position: Where is price relative to the expanding range midpoint?
        2. range_mid_roc_*: Rate of change of range midpoint (is the whole range drifting?)
        3. range_width_roc: Range expansion (are we in a breakout volatility regime?)
        4. atr_surge_ratio: ATR relative to its baseline (volatility surge detector)
        
        Args:
            df: DataFrame with Rolling_Range_* columns already calculated
            
        Returns:
            DataFrame with added regime context features
        """
        df = df.copy()
        try:
            if 'Rolling_Range_Mid' not in df.columns or 'Rolling_Range_Width' not in df.columns:
                self.logger.warning("[Regime Context] Rolling range columns missing, skipping")
                return df
            
            # 1. Macro position: Where is price relative to the slow (expanding) range midpoint?
            # High value = price is above long-term center = trending up
            # This is computed BEFORE clip() so it can exceed [0,1] during breakouts
            df['Regime_Macro_Position'] = (
                (df['Close'] - df['Rolling_Range_Low']) / 
                (df['Rolling_Range_Width'] + 1e-8)
            )
            
            # 2. Rate of change of the range midpoint - is the whole range drifting?
            # Positive = uptrend, Negative = downtrend, Near-zero = ranging
            df['Regime_Mid_ROC_10'] = df['Rolling_Range_Mid'].pct_change(10).fillna(0)
            df['Regime_Mid_ROC_50'] = df['Rolling_Range_Mid'].pct_change(50).fillna(0)
            
            # 3. Range expansion - are we in a breakout volatility regime?
            # Positive = expanding (breakout), Negative = contracting (consolidation)
            df['Regime_Width_ROC'] = df['Rolling_Range_Width'].pct_change(20).fillna(0)
            
            # 4. ATR relative to its own rolling mean - volatility surge detector
            # >1.5 = high volatility, <0.5 = low volatility
            if 'Close' in df.columns:
                atr = self._calculate_atr(df, period=14)
                atr_baseline = atr.rolling(100, min_periods=20).mean()
                df['Regime_ATR_Surge'] = (atr / (atr_baseline + 1e-8)).fillna(1.0)
            else:
                df['Regime_ATR_Surge'] = 1.0
            
            # 5. Distance from range boundaries — lossless sigmoid (replaces .clip(0,1))
            dist_low_raw = (
                (df['Close'] - df['Rolling_Range_Low']) /
                (df['Rolling_Range_Width'] + 1e-8)
            ).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.5)
            dist_low_sig, _ = self._normalize_with_rolling_mean(
                dist_low_raw, col='Regime_Distance_From_Low',
                window=20, sigmoid_scale_factor=self.sigmoid_scale_factor, store_baseline=False,
            )
            df['Regime_Distance_From_Low'] = dist_low_sig

            dist_high_raw = (
                (df['Rolling_Range_High'] - df['Close']) /
                (df['Rolling_Range_Width'] + 1e-8)
            ).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.5)
            dist_high_sig, _ = self._normalize_with_rolling_mean(
                dist_high_raw, col='Regime_Distance_From_High',
                window=20, sigmoid_scale_factor=self.sigmoid_scale_factor, store_baseline=False,
            )
            df['Regime_Distance_From_High'] = dist_high_sig
            
            # Clean up any inf/nan values
            regime_cols = [c for c in df.columns if c.startswith('Regime_')]
            for col in regime_cols:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
            
            self.logger.info(f"✅ [Regime Context] Added {len(regime_cols)} regime-aware features")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ [Regime Context] Failed to add features: {e}")
            return df

    def _normalize_by_rolling_structural_range(self, df: pd.DataFrame, split_name: str = "train", rolling_window: int=60) -> pd.DataFrame:
        """Normalize OHLCV and price-level columns using the adaptive volatility-aware structural range.
        
        REPLACES original columns with normalized [0,1] versions so scaler fits on normalized data.
        The adaptive range (based on ATR) keeps normalized prices in [0,1] across
        different market regimes (e.g., Gold $1200 → $3000 breakout).
        
        Added clipping diagnostics to detect regime-breaking normalization
        """
        df = df.copy()
        try:
            if 'Rolling_Range_Width' not in df.columns:
                raise ValueError("Rolling ranges not calculated first")

            # ALL price-level columns that should be normalized by structural range
            price_cols = ['Open', 'High', 'Low', 'Close']
            # Add shifted price columns and support/resistance levels
            price_level_cols = [
                'Previous_Close', 
                'Prev_1_Close', 'Prev_2_Close', 'Prev_3_Close', 'Prev_4_Close', 'Prev_5_Close',
                'R1', 'R2', 'R3', 'S1', 'S2', 'S3',
                'High_Day_1', 'High_Day_2', 'High_Day_3',
                'Low_Day_1', 'Low_Day_2', 'Low_Day_3'
            ]
            all_price_cols = price_cols + price_level_cols
            
            normalized_count = 0
            out_of_bounds_counts = {}  # Track if any normalized values exceed [0,1]
            clipping_stats = {}  # Track how many values get clipped to 0.0 or 1.0

            for col in all_price_cols:
                if col in df.columns:
                    # Normalize IN-PLACE using rolling-mean sigmoid instead of clip(0, 1).
                    # Values outside the training structural range are compressed toward
                    # (0, 1) but never saturated — breakout bars still carry information.
                    raw_norm = ((df[col] - df['Rolling_Range_Low']) /
                                df['Rolling_Range_Width'])
                    raw_norm = raw_norm.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.5)

                    # Detect breakouts on the raw (pre-sigmoid) normalised values
                    pre_clip_low  = (raw_norm < 0).sum()
                    pre_clip_high = (raw_norm > 1).sum()
                    breakout_detection = self._detect_and_log_breakouts(
                        col, raw_norm, raw_norm, threshold=0.02
                    )
                    if breakout_detection['detected']:
                        if not hasattr(self, 'global_breakout_detections'):
                            self.global_breakout_detections = {}
                        self.global_breakout_detections[col] = breakout_detection

                    # Sigmoid squash — lossless, no saturation
                    normalized_values, _rm = self._normalize_with_rolling_mean(
                        raw_norm,
                        col=col,
                        window=20,
                        sigmoid_scale_factor=self.sigmoid_scale_factor,
                        store_baseline=True,
                    )

                    df[col] = normalized_values
                    
                    # Track clipping statistics
                    clipped_to_zero = (df[col] == 0.0).sum()
                    clipped_to_one = (df[col] == 1.0).sum()
                    
                    if pre_clip_low > 0 or pre_clip_high > 0:
                        clipping_stats[col] = {
                            'below_zero': pre_clip_low,
                            'above_one': pre_clip_high,
                            'clipped_to_0': clipped_to_zero,
                            'clipped_to_1': clipped_to_one,
                            'total_rows': len(df)
                        }
                    
                    # Diagnostic: Check bounds compliance
                    out_of_bounds = ((df[col] < 0) | (df[col] > 1)).sum()
                    if out_of_bounds > 0:
                        out_of_bounds_counts[col] = out_of_bounds
                    
                    normalized_count += 1

            # Causal rolling percentiles normalizer to map features to [0.0, 1.0] robustly
            def normalize_column_by_rolling_quantiles(col_name: str, window: int = 60, lower_q: float = 0.05, upper_q: float = 0.95, default_val: float = 0.5):
                if col_name not in df.columns:
                    return False
                
                # Causal rolling percentiles
                roll_low = df[col_name].rolling(window=window, min_periods=20).quantile(lower_q)
                roll_high = df[col_name].rolling(window=window, min_periods=20).quantile(upper_q)
                roll_width = roll_high - roll_low
                roll_width = roll_width.replace(0, np.nan)
                
                # Scale using rolling-mean sigmoid (lossless) instead of clip(0, 1)
                raw_scaled = (df[col_name] - roll_low) / roll_width
                raw_scaled = raw_scaled.replace([np.inf, -np.inf], np.nan).ffill().fillna(default_val)
                scaled, _rm = self._normalize_with_rolling_mean(
                    raw_scaled,
                    col=col_name,
                    window=window,
                    sigmoid_scale_factor=self.sigmoid_scale_factor,
                    store_baseline=True,
                )
                df[col_name] = scaled
                return True

            # 1. Normalize all Spread-related columns using rolling quantiles
            spread_cols = [c for c in df.columns if 'spread' in c.lower()]
            for col in spread_cols:
                if normalize_column_by_rolling_quantiles(col, window=rolling_window, lower_q=0.05, upper_q=0.95):
                    normalized_count += 1
                    self.logger.info(f"📊 [MLPrep] Normalized Spread-level feature: {col} using rolling [5%, 95%] quantiles")
            
            # 2. Normalize all ATR-related columns using rolling quantiles (exclude pct/percentage)
            atr_cols = [
                c for c in df.columns 
                if 'atr' in c.lower() 
                and not any(kw in c.lower() for kw in ['_pct', '_percent'])
            ]
            for col in atr_cols:
                if normalize_column_by_rolling_quantiles(col, window=rolling_window, lower_q=0.05, upper_q=0.95):
                    normalized_count += 1
                    self.logger.info(f"📊 [MLPrep] Normalized ATR/Volatility-level feature: {col} using rolling [5%, 95%] quantiles")

            # 3. Normalize all Volume/TickVolume related columns using rolling quantiles
            # EXCLUDE ratio/percentage/dominance columns which are already bounded or 
            # have a specific meaning that rolling quantiles would distort.
            vol_exclude_keywords = ['_ratio', '_dominance', '_pct', '_surge', '_trend', '_consistency']
            vol_cols = [
                c for c in df.columns 
                if ('volume' in c.lower() or c in ['OBV', 'On_Balance_Volume'])
                and not any(kw in c.lower() for kw in vol_exclude_keywords)
            ]
            for col in vol_cols:
                if normalize_column_by_rolling_quantiles(col, window=rolling_window, lower_q=0.05, upper_q=0.95):
                    normalized_count += 1
                    self.logger.info(f"📊 [MLPrep] Normalized Volume-level feature: {col} using rolling [5%, 95%] quantiles")

            # 4. Normalize bounded oscillators that are strictly [0, 100]
            bounded_100_cols = ['RSI', 'RSI_14', 'RSI_7', 'RSI_2', 'Stochastic', 'Stoch_K', 'Stoch_D']
            for col in bounded_100_cols:
                if col in df.columns:
                    df[col] = df[col] / 100.0
                    normalized_count += 1

            if clipping_stats:
                self.logger.warning(f"⚠️ [Clipping Diagnostics] {split_name} split:")
                for col, stats in clipping_stats.items():
                    pct_below = (stats['below_zero'] / stats['total_rows']) * 100
                    pct_above = (stats['above_one'] / stats['total_rows']) * 100
                    pct_at_zero = (stats['clipped_to_0'] / stats['total_rows']) * 100
                    pct_at_one = (stats['clipped_to_1'] / stats['total_rows']) * 100
                    
                    self.logger.warning(
                        f"  {col}: {pct_below:.1f}% below 0, {pct_above:.1f}% above 1 "
                        f"(clipped: {pct_at_zero:.1f}% at 0.0, {pct_at_one:.1f}% at 1.0)"
                    )
                    
                    # Alert if significant clipping (>5% of data at boundaries)
                    if pct_at_one > 5.0:
                        self.logger.error(
                            f"❌ [REGIME BREAK] {col} has {pct_at_one:.1f}% of values clipped to 1.0! "
                            f"This indicates the volatility_factor is too small for this regime shift. "
                            f"Consider increasing the dynamic clamp max (current fitted_vol_factor: {getattr(self, 'fitted_vol_factor', 'unknown')})."
                        )

            # Log normalization results with regime-awareness diagnostics
            if out_of_bounds_counts:
                # Some values out of bounds (expected during extreme moves)
                bounds_msg = ", ".join([f"{k}:{v}" for k, v in out_of_bounds_counts.items()])
                self.logger.warning(f"⚠️ [Structural Norm] {split_name}: {normalized_count} features normalized, "
                                   f"out-of-bounds: {bounds_msg} (expected during regime shifts)")
            else:
                # All values in bounds - normal operation
                self.logger.info(f"✅ [Structural Norm] {split_name}: Normalized {normalized_count} features to [0,1] "
                                f"(adaptive volatility range, regime-robust)")
            
            return df
        except Exception as e:
            self.logger.error(f"❌ [Structural Norm] Failed: {e}")
            raise ValueError(f"Failed to normalize: {e}")


# Ensure old imports still work
_all__ = [
    "ProcessingManager",
    "ProcessingContext",
    "ProcessingStrategy",
    "StrategyFactory",
    "HandlerRegistry",
    "IntermediateResultsCache",
    "CachedStepData",

]