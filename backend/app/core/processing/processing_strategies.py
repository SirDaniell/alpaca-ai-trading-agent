"""
Processing Strategies - Reusable Strategy Implementations

This module provides reusable processing strategies that can be applied uniformly
across all analysis types (Technical, SNR, Astronomical, ML Preparation, Model Training).

Key Design Principles:
1. Strategy Pattern: Each strategy (Sequential, Parallel, SliceStreaming) is a reusable class
2. Handler Registry: Analysis-specific logic is registered and retrieved dynamically
3. Uniform Interface: All strategies use the same execute() signature
4. Data Flow: Input → Strategy → Handler → Result → Storage
5. Progress Tracking: Built-in WebSocket and task_store integration

Architecture:
    ProcessingManager (Orchestrator)
        ↓
    StrategyFactory (Selects strategy based on data size)
        ↓
    ProcessingStrategy (Sequential/Parallel/SliceStreaming)
        ↓
    HandlerRegistry (Retrieves analysis-specific handler)
        ↓
    Handler Function (analyze_snr_impl, analyze_ml_prep_impl, etc.)
        ↓
    Result (Enriched data + metadata)
"""

# Standard library imports
import gc
import logging
import asyncio
import dataclasses
import multiprocessing
import math
import os
import random
import io
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable, Tuple, Set
from enum import Enum
from dataclasses import dataclass
import tempfile
import httpx as _httpx

# Third-party imports
import pandas as pd
import numpy as np
import joblib
import asyncio
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

from app.core.analysis.currency_index import (
    CurrencyIndexCalculator,
    INDEX_DEFINITIONS,
    OHLCV_FIELDS,
    prepare_index_data,
)
from app.services.mt5_service import MT5Service

# App imports
from app.core.processing.tasks import TaskStore, TaskCancelledException, BroadcastingTaskStoreProxy, QueueProgressStoreProxy
from app.core.config import ProcessingConfig
from app.core.analysis.technical_indicators import TechnicalIndicators
from app.core.analysis.trading.signal_generator import generate_signals_sequential_with_progress
from app.core.analysis.trading.signal_generator_optimized import smart_chunk_dataframe
from app.core.analysis.astronomy.astronomical_optimized import generate_astronomical_data_optimized
from app.core.processing.progress_reporter import ProgressReporter, ThrottlingStrategy
from app.core.ml.ml_dataset_preparation import MLDatasetPreparation
from app.core.services.multiprocessing_utils import RowChunker
from app.core.processing.progress_utils import calculate_cumulative_progress
# : Store each chunk immediately to database
from app.core.data.session_data_loader import append_sequences_to_ml_dataset

from app.core.processing.processing_types import (
    ProcessingContext,
    ProcessingStrategy,
    ProcessingStrategyBase,
)

# NOTE: processing_handlers is intentionally NOT imported at module level.
# It creates a cycle:  processing_strategies → processing_handlers → processing_strategies
# Handler functions are imported lazily inside get_worker_map() and
# currency_indices_analysis_worker() which are only called at runtime.

# Optional imports (with fallback handling)
try:
    import psutil
except ImportError:
    psutil = None

# Convert dict config to DatasetConfig object if needed

from app.core.ml.ml_dataset_preparation import DatasetConfig
logger = logging.getLogger(__name__)

# Worker names for personality and frontend animation
WORKER_NAMES = [
    "Atlas", "Titan", "Phoenix", "Orion", "Nova", "Quantum", 
    "Cipher", "Matrix", "Vector", "Nexus", "Pulse", "Spark",
    "Blaze", "Storm", "Thunder", "Lightning", "Comet", "Meteor",
    "Rocket", "Turbo", "Nitro", "Boost", "Flash", "Dash"
]

def get_worker_names(n_workers: int) -> List[str]:
    """Generate unique worker names for personality and frontend animation."""
    if n_workers <= len(WORKER_NAMES):
        return random.sample(WORKER_NAMES, n_workers)
    else:
        # For systems with many cores, add numbers to names
        names = WORKER_NAMES.copy()
        for i in range(len(WORKER_NAMES), n_workers):
            base_name = WORKER_NAMES[i % len(WORKER_NAMES)]
            names.append(f"{base_name}-{i // len(WORKER_NAMES) + 2}")
        return names[:n_workers]


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class StandardChunk:
    """Standard chunk for non-SNR analysis types."""
    chunk_id: int
    data: pd.DataFrame
    global_start_idx: int


# ============================================================================
# MULTIPROCESSING WORKER FUNCTIONS (Module level for pickling/pool support)
# ============================================================================

def worker_init():
    """
    Initialize worker process with CPU throttling.

    Strategy: OS-level niceness + numpy/OpenBLAS thread limits.
    This is more effective than just reducing worker count because:
    - nice(15) tells the scheduler to yield to the main process under load
      (WebSocket handler, FastAPI event loop stay responsive)
    - Limiting numpy threads prevents each worker from spawning its own
      thread pool (default = all cores), which is the real cause of 100% CPU
      when you have e.g. 4 workers each using 8 numpy threads = 32 threads
    """
    # 1. Lower OS scheduling priority so the main process (FastAPI/WebSocket)
    #    always gets CPU time first. nice 15 = background-class priority.
    try:
        os.nice(15)
    except Exception:
        pass

    # 2. Cap numpy/OpenBLAS/MKL internal thread pools to 2 per worker.
    #    Without this, each worker spawns cpu_count() threads for BLAS ops,
    #    turning 4 workers into 4*N_cores threads all fighting for CPU.
    for env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS"):
        os.environ[env_var] = "2"

    # Apply to numpy if already imported in this worker
    try:
        np.__config__.blas_opt_info  # noqa – just a probe, no-op if missing
    except Exception:
        pass

def technical_analysis_worker(
    chunk_df: pd.DataFrame,
    config: Any,
    task_id: str,
    progress_proxy: Any,
    chunk_id: int,
    global_offset: int,
    slice_context: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Worker: Technical indicator analysis on a data chunk."""
    try:
        ti = TechnicalIndicators(config)
        
        # Calculate indicators
        result_df = ti.calculate_all_indicators(
            chunk_df,
            task_id=task_id,
            progress_store=progress_proxy,
            mode="training",
            slice_context=slice_context
        )
        
        return {
            "chunk_id": chunk_id,
            "result_df": result_df,
            "metadata": {"analysis_type": "technical"}
        }
    except Exception as e:
        import traceback
        stack_trace = traceback.format_exc()
        logger.error(f"❌ Technical worker {chunk_id} failed: {e}\n{stack_trace}")
        return {"chunk_id": chunk_id, "error": f"{str(e)}", "traceback": stack_trace}


async def currency_strength_matrix_worker(
    df: pd.DataFrame,
    config,
    **kwargs,
) -> dict:
    """
    Worker for Currency Strength Matrix calculation.

    Configuration is supplied directly by the caller. This worker is
    stateless and safe to execute in parallel.
    """
    from app.core.analysis.currency_index import (
        calculate_currency_strength_matrix,
    )

    try:
        # Resolve column names (case-insensitive)
        close_col_cfg = getattr(config, "close_column", "close")
        dxy_col_cfg = getattr(config, "dxy_column", "Dollar_close")

        col_lower_map = {c.lower(): c for c in df.columns}

        actual_close = (
            col_lower_map.get(close_col_cfg.lower())
            or col_lower_map.get("close")
        )

        actual_dxy = (
            dxy_col_cfg
            if dxy_col_cfg in df.columns
            else col_lower_map.get(dxy_col_cfg.lower())
        )

        if actual_close is None:
            raise ValueError(
                f"[CSM] Asset close column '{close_col_cfg}' not found."
            )

        if actual_dxy is None:
            dxy_candidates = [
                c for c in df.columns
                if "dollar" in c.lower() or "dxy" in c.lower()
            ]
            raise ValueError(
                f"[CSM] DXY close column '{dxy_col_cfg}' not found. "
                f"Ensure Currency Indices ran first. "
                f"Candidates: {dxy_candidates or 'none'}"
            )

        fast_period = int(getattr(config, "fast_period", 20))
        slow_period = int(getattr(config, "slow_period", 100))
        zscore_clamp = float(getattr(config, "zscore_clamp", 3.0))

        logger.info(
            "[CSM Worker] "
            f"close={actual_close}, "
            f"dxy={actual_dxy}, "
            f"fast={fast_period}, "
            f"slow={slow_period}"
        )

        csm_df = calculate_currency_strength_matrix(
            asset_close=df[actual_close],
            dxy_close=df[actual_dxy],
            fast_period=fast_period,
            slow_period=slow_period,
            zscore_clamp=zscore_clamp,
        )

        # Remove any previous CSM columns
        stale_cols = [c for c in df.columns if c.startswith("CSM_")]
        if stale_cols:
            df = df.drop(columns=stale_cols)

        result_df = pd.concat([df, csm_df], axis=1)

        return {
            "result_df": result_df,
            "features_df": result_df,
            "metadata": {
                "analysis_type": "currency_strength_matrix",
                "strategy": "currency_strength_matrix",
                "rows_processed": len(result_df),
                "csm_columns": list(csm_df.columns),
                "fast_period": fast_period,
                "slow_period": slow_period,
            },
        }

    except Exception as exc:
        logger.error(
            f"❌ [CSM Worker] Failed: {exc}",
            exc_info=True,
        )
        raise



def snr_analysis_worker(
    chunk_df: pd.DataFrame,
    confirmation_period: int,
    lookback_period: int,
    n_clusters: int,
    zone_width: float,
    min_distance_pct: float,
    lookforward_period: int,
    animation_step: int,
    task_id: str,
    progress_proxy: Any,
    chunk_id: int,
    global_offset: int,
    slice_context: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Worker: SNR analysis implementation (matches core function signature).
    Returns dict format for consistent merging across all worker types.
    """
    try:
        signals, zones, df_with_snr, ml_dataset, g_start, g_end, signal_counts = generate_signals_sequential_with_progress(
            price_data=chunk_df,
            lookback_period=lookback_period,
            confirmation_period=confirmation_period,
            n_clusters=n_clusters,
            zone_width=zone_width,
            min_distance_pct=min_distance_pct,
            lookforward_period=lookforward_period,
            animation_step=animation_step,
            task_id=task_id,
            progress_store=progress_proxy,
            chunk_id=chunk_id,
            global_index_offset=global_offset,
            slice_context=slice_context,
        )
        # Return normalized dict format (not tuple)
        return {
            "chunk_id": chunk_id,
            "signals": signals,
            "zones": zones,
            "result_df": df_with_snr,
            "ml_dataset": ml_dataset,
            "signal_counts": signal_counts,
            "g_start": g_start,
            "g_end": g_end,
            "metadata": {"analysis_type": "snr", "global_offset": global_offset}
        }
    except Exception as e:
        logger.error(f"SNR worker {chunk_id} failed: {e}")
        # Return dict with error flag (consistent with other workers)
        return {"chunk_id": chunk_id, "error": str(e)}


def astronomical_analysis_worker(
    chunk_df: pd.DataFrame,
    config: Any,
    task_id: str,
    progress_proxy: Any,
    chunk_id: int,
    global_offset: int,
    slice_context: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Worker: Astronomical analysis on a data chunk."""
    try:
        reporter = ProgressReporter(task_id, progress_proxy, slice_context)
        
        # Log chunk range for trace-level debugging
        if not chunk_df.empty:
            time_col = next((c for c in chunk_df.columns if c.lower() in ['time', 'date']), None)
            if time_col:
                start_val = chunk_df[time_col].iloc[0]
                end_val = chunk_df[time_col].iloc[-1]
                logger.info(f"🚀 [Astro-Worker-{chunk_id}] Processing range: {start_val} to {end_val} ({len(chunk_df)} rows)")

        result_df = generate_astronomical_data_optimized(
            price_data=chunk_df,
            observer_lat=config.observer_lat,
            observer_lon=config.observer_lon,
            house_system=config.house_system,
            zodiac_type=config.zodiac_type,
            use_minor_aspects=config.use_minor_aspects,
            aspect_orbs=config.aspect_orbs,
            selected_features=config.selected_features,
            include_asteroids=getattr(config, "include_asteroids", False),
            include_fixed_stars=getattr(config, "include_fixed_stars", False),
            reporter=reporter,
            chunk_id=chunk_id,  # Pass chunk_id so progress reports include it
        )
        
        return {
            "chunk_id": chunk_id,
            "result_df": result_df,
            "metadata": {"analysis_type": "astronomical"}
        }
    except Exception as e:
        import traceback
        stack_trace = traceback.format_exc()
        logger.error(f"❌ Astronomical worker {chunk_id} failed: {e}\n{stack_trace}")
        return {"chunk_id": chunk_id, "error": f"{str(e)}", "traceback": stack_trace}


def ml_prep_worker(
    chunk_df: pd.DataFrame,
    config: Any,
    task_id: str,
    progress_proxy: Any,
    chunk_id: int,
    global_offset: int,
    slice_context: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Worker: ML preparation (sequence generation) on a data chunk.
    
  ARCHITECTURE:
    - Receives pre-split data (train, validation, or test)
    - Uses provided scaler (fitted on training) or fits  one if training split
    - Does NOT split data again (ProcessingManager already split)
    - Core function is agnostic to which split it's processing
    """
    try:
        
        
        if isinstance(config, dict):
            # All frontend fields are now supported in DatasetConfig
            config_obj = DatasetConfig(**config)
        else:
            config_obj = config
        
        # PRODUCTION: Use the unified picklable ProgressReporter
        # Workers automatically switch to Proxy mode when unpickled
        reporter = None
        if slice_context and "reporter" in slice_context:
            reporter = slice_context["reporter"]
            # Ensure worker uses its specific progress_proxy
            reporter.task_store = progress_proxy
        else:
            # Fallback for manual spawning
            reporter = ProgressReporter(
                task_id=task_id, 
                task_store=progress_proxy,
                user_id=slice_context.get("user_id", "unknown") if slice_context else "unknown",
                throttling_strategy=ThrottlingStrategy.HYBRID
            )
        
        # Extract scaler information from slice_context
        global_scaler = None
        fit_scaler = True  # Default: fit scaler (for training split)
        split_type = "train"  # Default split type
        skip_scaling = False  # : Check if data is already scaled
        
        if slice_context:
            global_scaler_bytes = slice_context.get("global_scaler")
            fit_scaler = slice_context.get("fit_scaler", True)
            split_type = slice_context.get("split_type", "train")
            skip_scaling = slice_context.get("skip_scaling", False)  # 
            
            # Deserialize scaler if provided
            global_scaler = None
            if global_scaler_bytes is not None:
                try:
                    buffer = io.BytesIO(global_scaler_bytes)
                    global_scaler = joblib.load(buffer)
                    logger.info(f"[ML Worker {chunk_id}] Successfully deserialized scaler for {split_type} split: {type(global_scaler)}")
                except Exception as e:
                    logger.error(f"[ML Worker {chunk_id}] Failed to deserialize scaler: {e}")
                    global_scaler = None
            else:
                logger.info(f"[ML Worker {chunk_id}] No scaler bytes provided for {split_type} split (fit_scaler={fit_scaler})")
        
        # OPTIMIZATION: Skip validation if data is already scaled in PM
        if skip_scaling:
            logger.info(f"[ML Worker {chunk_id}] Data already scaled in PM - skipping scaler validation")
        elif not fit_scaler and global_scaler is None:
            error_msg = f"No scaler available for {split_type} split. Training split must be processed first."
            logger.error(f"[ML Worker {chunk_id}] {error_msg}")
            return {"chunk_id": chunk_id, "error": error_msg}
        
        # Initialize ML Prep with the chunk (which is already a split)
        ml_prep = MLDatasetPreparation(
            data=chunk_df,
            config=config_obj,
            task_id=task_id,
            reporter=reporter,
            scaler=global_scaler,  # Use provided scaler or None for training
            dataset_name=config_obj.dataset_name,
            skip_scaling=skip_scaling,
            is_pre_split=True
        )

        # If the parent split already computed feature metadata, reuse it exactly.
        ctx_feature_cols = None
        ctx_columns_to_scale = None
        if slice_context:
            ml_prep_meta = slice_context.get("ml_prep_metadata")
            if isinstance(ml_prep_meta, dict):
                ctx_feature_cols = ml_prep_meta.get("feature_cols")
                ctx_columns_to_scale = ml_prep_meta.get("columns_to_scale")
            else:
                ctx_feature_cols = slice_context.get("feature_cols")
                ctx_columns_to_scale = slice_context.get("columns_to_scale")

            if ctx_feature_cols is not None:
                ml_prep.feature_cols = list(ctx_feature_cols)
                ml_prep.columns_to_scale = list(ctx_columns_to_scale) if ctx_columns_to_scale is not None else list(ctx_feature_cols)
                logger.info(
                    f"[ML Worker {chunk_id}] Reusing parent MLPrep metadata: "
                    f"{len(ml_prep.feature_cols)} features, {len(ml_prep.columns_to_scale)} columns_to_scale"
                )

        # PREPROCESSING STEPS: Run validation, enrichment, and feature identification
        # These are synchronous methods that prepare the data before sequence generation
        logger.info(f"[ML Worker {chunk_id}] Running preprocessing steps for {split_type} split...")
        
        # Run async preprocessing steps synchronously in worker
        async def run_preprocessing():
            # Stage 1: Validate data structure
            await ml_prep._validate_data()
            
            # Stage 2: Identify features for scaling only if not already set
            if not getattr(ml_prep, 'feature_cols', None):
                ml_prep._identify_features()
            else:
                logger.info(f"[ML Worker {chunk_id}] Skipping feature identification; using parent metadata")
            
            logger.info(f"[ML Worker {chunk_id}] Preprocessing complete: {len(ml_prep.feature_cols)} features identified")
        
        # Run preprocessing
        asyncio.run(run_preprocessing())
        
        # Generate sequences for this chunk (which is already a pre-split and pre-scaled)
        # OPTIMIZATION: Pass skip_scaling flag to avoid redundant scaling in worker
        # FIX: Pass enriched_target_columns from slice_context to ensure all targets are collected
        enriched_target_columns = slice_context.get('enriched_target_columns') if slice_context else None
        result = asyncio.run(ml_prep._generate_sequences_for_df(
            chunk_df, 
            split_name=split_type,
            start_pct=0, 
            end_pct=100,
            enriched_target_columns=enriched_target_columns  # FIX: Pass to sequence generator
        ))
        
        # MEMORY OPTIMIZATION: Write sequences to a compressed temp file.
        # This avoids serializing large numpy arrays over the IPC boundary into the main
        # process heap. The main thread callback reads and persists the file, then deletes it.
        tmp_dir = os.environ.get("ML_WORKER_TMP_DIR", "/tmp")
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".npz", prefix=f"ml_chunk_{chunk_id}_", dir=tmp_dir)
        os.close(tmp_fd)  # Close file descriptor; np.savez_compressed will reopen by path

        sequences = result["sequences"]
        labels    = result["labels"]
        seq_count = len(sequences) if sequences is not None else 0

        # Build a flat targets dict for np.savez (keys must be strings)
        targets = result.get("targets", {})
        save_kwargs = {"sequences": sequences, "labels": labels}
        if isinstance(targets, dict):
            for k, v in targets.items():
                save_kwargs[f"target_{k}"] = np.asarray(v)

        try:
            np.savez_compressed(tmp_path, **save_kwargs)
        except Exception as _write_err:
            # FIX #10: If write fails, remove the incomplete file so it doesn't
            # accumulate as an orphan.  Re-raise so the caller returns an error dict.
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise _write_err

        # Free the large arrays before returning
        del sequences, labels, targets, result, save_kwargs

        result_dict = {
            "chunk_id": chunk_id,
            "tmp_path": tmp_path,
            "seq_count": seq_count,
            "target_names": config_obj.target_columns,
            "feature_names": ml_prep.feature_cols,
            "metadata": {
                "analysis_type": "ml_preparation",
                "split_type": split_type,
                "seq_count": seq_count,
                "target_names": config_obj.target_columns,
                "feature_names": ml_prep.feature_cols,
            }
        }

        logger.info(f"ML Prep worker {chunk_id}: Spooled {seq_count} sequences to {tmp_path}")
        
        return result_dict
        
    except Exception as e:
        import traceback
        stack_trace = traceback.format_exc()
        logger.error(f"❌ ML Prep worker {chunk_id} failed: {e}\n{stack_trace}")
        return {"chunk_id": chunk_id, "error": f"{str(e)}", "traceback": stack_trace}

def sequential_chunk_worker(
    worker_chunks_data: List[Any],
    analysis_type: str,
    config: Any,
    task_id: str,
    progress_proxy: Any,
    worker_id: int,
    slice_context: Optional[Dict] = None,
    worker_name: str = "Worker"
) -> List[Dict[str, Any]]:
    """
    Process multiple chunks sequentially in a single worker.
    
    Args:
        worker_chunks_data: List of chunk data objects (SimpleChunk or SNR chunks)
        analysis_type: The type of analysis to perform
        config: Analysis configuration
        task_id: Task identifier
        progress_proxy: Progress reporting proxy
        worker_id: Worker identifier
        slice_context: Slice context
        worker_name: Friendly name for this worker (for logging and frontend)
    
    Returns:
        List of results from all chunks processed by this worker
    """
    worker_results = []
    total_chunks = len(worker_chunks_data)
    
    logger.info(f"🚀 [{worker_name}] Starting work on {total_chunks} chunks")
    
    # Map analysis type to worker function
    worker_map = {
        "snr": snr_analysis_worker,
        "technical": technical_analysis_worker,
        "astronomical": astronomical_analysis_worker,
        "ml_preparation": ml_prep_worker,
        "ml_dataset_preparation": ml_prep_worker,  # FIX: Add explicit mapping for ML_DATASET_PREPARATION
        "currency_indices": currency_indices_analysis_worker,  # FIX: Register currency indices worker
    }
    
    worker_func = worker_map.get(analysis_type)
    if not worker_func:
        logger.error(f"❌ [{worker_name}] Analysis type {analysis_type} not supported for sequential chunks")
        return [{"error": f"Unsupported analysis type: {analysis_type}", "worker_id": worker_id, "worker_name": worker_name}]

    for chunk_idx, chunk in enumerate(worker_chunks_data):
        try:
            chunk_task_id = f"{task_id}_{worker_name}_c{chunk_idx}"
            
            logger.debug(f"⚡ [{worker_name}] Processing chunk {chunk_idx + 1}/{total_chunks} (ID: {chunk.chunk_id})")
            
            # Prepare arguments based on worker type
            if analysis_type == "snr":
                # SNR has special parameter signature
                # In SNR, chunks are often tuples or objects with data and other params
                # Assuming chunk is the object returned by smart_chunk_dataframe
                args = (
                    chunk.data,
                    config.confirmation_period,
                    config.lookback_period,
                    config.n_clusters,
                    config.zone_width,
                    config.min_distance_pct,
                    config.lookforward_period,
                    config.animation_step,
                    chunk_task_id,
                    progress_proxy,
                    chunk.chunk_id,
                    chunk.global_start_idx,
                    slice_context,
                )
            else:
                # Standard parameter signature for technical, astronomical, ml_prep
                args = (
                    chunk.data, 
                    config, 
                    chunk_task_id, 
                    progress_proxy, 
                    chunk.chunk_id, 
                    chunk.global_start_idx, 
                    slice_context
                )
            
            # Execute the original worker function
            chunk_result = worker_func(*args)
            worker_results.append(chunk_result)
            
            # Report progress for this chunk
            if progress_proxy:
                try:
                    # FIX: Use update_task() method instead of .put()
                    progress_proxy.update_task(
                        task_id=chunk_task_id,
                        progress=100,
                        message=f"{worker_name}: chunk {chunk_idx + 1}/{total_chunks} complete",
                        metadata={
                            "worker_id": worker_id,
                            "worker_name": worker_name
                        }
                    )
                except AttributeError:
                   pass
                
        except Exception as e:
            logger.error(f"💥 [{worker_name}] Chunk {chunk_idx} failed: {e}")
            cid = getattr(chunk, 'chunk_id', chunk_idx)
            worker_results.append({"error": str(e), "chunk_id": cid, "worker_name": worker_name})
    
    logger.info(f"✅ [{worker_name}] Completed all {total_chunks} chunks successfully")
    return worker_results

def currency_indices_analysis_worker(
    chunk_df: pd.DataFrame,
    config: Any,
    task_id: str,
    progress_proxy: Any,
    chunk_id: int,
    global_offset: int,
    slice_context: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Multiprocessing worker for currency-index calculation on one chunk/slice.
 
    This function runs in a spawned subprocess, so it MUST be:
      - a plain synchronous function (no async def)
      - self-contained (all imports local)
      - free of references to variables defined in the parent process
 
    It mirrors the logic of ``analyze_currency_indices_impl`` exactly, but
    wraps every async call with ``asyncio.run()``.
 
    Args:
        chunk_df:     The row-slice of the full DataFrame to process.
        config:       CurrencyIndexConfig instance.
        task_id:      Task identifier for progress reporting.
        progress_proxy: Proxy object with ``update_task(task_id, progress, message)``.
        chunk_id:     Ordinal index of this chunk (for error reporting).
        global_offset: Row offset of this chunk in the full dataset.
        slice_context: Optional dict with slice metadata (slice_num, total_slices, …).
 
    Returns:
        Dict with keys:
            chunk_id, global_start_idx, result_df, metadata
        or on error:
            chunk_id, global_start_idx, error (traceback string)
    """
    try:
        
        # ------------------------------------------------------------------
        # helpers
        # ------------------------------------------------------------------
 
        _logger = logging.getLogger(__name__)
 
        def report(progress: int, message: str) -> None:
            if progress_proxy:
                try:
                    progress_proxy.update_task(
                        task_id=task_id,
                        progress=progress,
                        message=f"Chunk {chunk_id}: {message}",
                    )
                except Exception:
                    pass
 
        # ------------------------------------------------------------------
        # STEP 0  Strip stale index columns
        # ------------------------------------------------------------------
        INDEX_NAMES = set(INDEX_DEFINITIONS.keys())
        stale_cols = [
            c for c in chunk_df.columns
            if any(c.startswith(f"{idx}_") for idx in INDEX_NAMES)
        ]
        if stale_cols:
            _logger.info(
                "[Currency Indices Worker] Stripping %d stale index columns",
                len(stale_cols),
            )
            chunk_df = chunk_df.drop(columns=stale_cols)

        # ------------------------------------------------------------------
        # TIME GUARD: Pin the original Time column before any pair merges.
        # pair-merge dtype reconciliation (int64 ↔ datetime64) can silently
        # produce NaT in the Time column when pandas coerces mismatched types.
        # We restore it unconditionally after all merges so Time is never
        # mutated regardless of what pandas does internally.
        # ------------------------------------------------------------------
        _time_col = next((c for c in chunk_df.columns if c.lower() == "time"), None)
        _pinned_time = chunk_df[_time_col].copy() if _time_col else None
        _pinned_time_dtype = _pinned_time.dtype if _pinned_time is not None else None

 
        # ------------------------------------------------------------------
        # STEP 1  Collect required pairs from selected indices
        # ------------------------------------------------------------------
        required_pairs: set = set()
        for idx_name in getattr(config, "selected_indices", []):
            if idx_name not in INDEX_DEFINITIONS:
                _logger.warning("[Currency Indices Worker] Unknown index: %s, skipping", idx_name)
                continue
            required_pairs.update(INDEX_DEFINITIONS[idx_name]["pairs"].keys())
 
        required_pairs = sorted(required_pairs)
        report(10, f"Found {len(required_pairs)} required pairs")
        _logger.info("[Currency Indices Worker] Required pairs: %s", required_pairs)
 
        # ------------------------------------------------------------------
        # STEP 2  Detect missing pair columns
        # ------------------------------------------------------------------
        available_cols = set(chunk_df.columns)
        missing_pairs: set = set()
        for pair in required_pairs:
            for field in OHLCV_FIELDS:
                if f"{field}_{pair}" not in available_cols:
                    missing_pairs.add(pair)
                    break
 
        # ------------------------------------------------------------------
        # STEP 2a  Fetch missing pairs from MT5
        # ------------------------------------------------------------------
        if missing_pairs:
            timeframe = getattr(config, "timeframe", None)
            if not timeframe or timeframe == "H1":
                delta = 3600
                if len(chunk_df) > 1:
                    t0 = chunk_df["Time"].iloc[0]
                    t1 = chunk_df["Time"].iloc[1]
                    if isinstance(t0, (int, float, np.integer, np.floating)):
                        delta = t1 - t0
                    else:
                        delta = (
                            pd.to_datetime(t1) - pd.to_datetime(t0)
                        ).total_seconds()
 
                if delta <= 60:       timeframe = "M1"
                elif delta <= 300:    timeframe = "M5"
                elif delta <= 900:    timeframe = "M15"
                elif delta <= 1800:   timeframe = "M30"
                elif delta <= 3600:   timeframe = "H1"
                elif delta <= 14400:  timeframe = "H4"
                else:                 timeframe = "D1"
 
            fetch_count = len(chunk_df) + 100
            t_last = chunk_df["Time"].iloc[-1]
            if isinstance(t_last, (int, float, np.integer, np.floating)):
                date_from = datetime.fromtimestamp(t_last)
            else:
                date_from = pd.to_datetime(t_last).to_pydatetime()
 
            report(20, f"Fetching {len(missing_pairs)} missing pairs from MT5")
            _logger.info(
                "[Currency Indices Worker] Fetching %d missing pairs (timeframe=%s)",
                len(missing_pairs), timeframe,
            )
 
            async def _fetch_pair_with_retry(mt5, pair, max_retries=3, timeout=200):
                # BUG FIX: timeout raised from 30 → 200s.
                # The HTTP client already has a 180s read timeout; if asyncio.wait_for
                # fires at 30s it cancels a request that is still in-flight at the
                # bridge, producing spurious "Timeout fetching X (attempt N)" warnings
                # even when the bridge would have responded successfully at ~60-90s.
                #
                # NOT-FOUND SHORT-CIRCUIT: If the bridge reports "Symbol not found" /
                # "Terminal: Not found" we return empty immediately — no retries.
                # These are permanent broker-side absences; retrying wastes 3× the
                # bridge's queue budget and floods the logs.
                _NOT_FOUND_PHRASES = ("not found", "terminal: not found", "symbol not found", "-4,")
                for attempt in range(1, max_retries + 1):
                    try:
                        res = await asyncio.wait_for(
                            mt5.fetch_ohlc_data_v2(
                                symbol=pair,
                                timeframe=timeframe,
                                count=fetch_count,
                                date_from=date_from,
                            ),
                            timeout=timeout,
                        )
                        if isinstance(res, list) and res:
                            return pair, res
                        if isinstance(res, dict):
                            if res.get("data"):
                                return pair, res["data"]
                            if "error" in res:
                                err_msg = str(res["error"]).lower()
                                if any(phrase in err_msg for phrase in _NOT_FOUND_PHRASES):
                                    _logger.warning(
                                        "[Currency Indices Worker] ⛔ %s not available on this broker — skipping (no retry): %s",
                                        pair, res["error"],
                                    )
                                    return pair, []   # permanent — don't retry
                                _logger.warning(
                                    "[Currency Indices Worker] MT5 error for %s (attempt %d): %s",
                                    pair, attempt, res["error"],
                                )
                    except asyncio.TimeoutError:
                        _logger.warning(
                            "[Currency Indices Worker] Timeout fetching %s (attempt %d/%d)",
                            pair, attempt, max_retries,
                        )
                    except Exception as exc:
                        exc_str = str(exc).lower()
                        if any(phrase in exc_str for phrase in _NOT_FOUND_PHRASES):
                            _logger.warning(
                                "[Currency Indices Worker] ⛔ %s not available on this broker — skipping (no retry): %s",
                                pair, exc,
                            )
                            return pair, []   # permanent — don't retry
                        _logger.error(
                            "[Currency Indices Worker] Error fetching %s (attempt %d/%d): %s",
                            pair, attempt, max_retries, exc,
                        )
                    if attempt < max_retries:
                        await asyncio.sleep(2)
                _logger.warning(
                    "[Currency Indices Worker] ⚠️ Could not fetch %s after %d attempts — index will use remaining pairs",
                    pair, max_retries,
                )
                return pair, []
 
            async def _fetch_all():
                # BUG FIX: always create a fresh AsyncClient in this subprocess.
                #
                # MT5Service uses a class-level shared AsyncClient (_client).
                # That client was created in the parent process's event loop.
                # asyncio.run() in the worker spawns a brand-new event loop, so
                # the parent's client connection pool is either None or bound to a
                # dead loop — every request silently hangs until asyncio.wait_for
                # fires.  Injecting a fresh client scoped to this loop fixes it.
                
                fresh_client = _httpx.AsyncClient(
                    limits=_httpx.Limits(
                        max_connections=20,
                        max_keepalive_connections=10,
                    ),
                    timeout=_httpx.Timeout(180.0, connect=10.0),
                )
                try:
                    mt5 = MT5Service()
                    mt5.client = fresh_client  # override shared client
 
                    try:
                        status = await mt5.get_status()
                        if isinstance(status, dict) and not status.get("terminal_connected", False):
                            _logger.warning(
                                "[Currency Indices Worker] MT5 not connected: %s", status
                            )
                    except Exception as exc:
                        _logger.warning(
                            "[Currency Indices Worker] Could not check MT5 status: %s", exc
                        )
 
                    sem = asyncio.Semaphore(5)
 
                    async def _bounded(pair):
                        async with sem:
                            return await _fetch_pair_with_retry(mt5, pair)
 
                    return await asyncio.gather(
                        *[_bounded(p) for p in sorted(missing_pairs)],
                        return_exceptions=True,
                    )
                finally:
                    await fresh_client.aclose()
 

            raw_results = asyncio.run(_fetch_all())
 
            # Normalise gather results (some may be exceptions)
            results = []
            for pair, item in zip(sorted(missing_pairs), raw_results):
                if isinstance(item, Exception):
                    _logger.error("[Currency Indices Worker] Exception for %s: %s", pair, item)
                    results.append((pair, []))
                else:
                    results.append(item)
 
            # Merge fetched pair data into chunk_df
            for pair, data in results:
                if not data:
                    _logger.warning("[Currency Indices Worker] No data for %s", pair)
                    continue
                try:
                    pair_df = pd.DataFrame(data)
 
                    # Map numeric column indices → names (MT5 array format)
                    if len(pair_df.columns) and isinstance(pair_df.columns[0], int):
                        pair_df.rename(
                            columns={
                                0: "time", 1: "open", 2: "high", 3: "low",
                                4: "close", 5: "tick_volume", 6: "spread", 7: "real_volume",
                            },
                            inplace=True,
                        )
 
                    if "time" in pair_df.columns:
                        pair_df.rename(columns={"time": "Time"}, inplace=True)
                    if "Time" not in pair_df.columns:
                        _logger.warning("[Currency Indices Worker] No Time column for %s, skipping", pair)
                        continue
 
                    # Align Time dtypes
                    df_time_dtype  = chunk_df["Time"].dtype
                    pair_time_dtype = pair_df["Time"].dtype
                    if pd.api.types.is_datetime64_any_dtype(df_time_dtype):
                        if not pd.api.types.is_datetime64_any_dtype(pair_time_dtype):
                            if pd.api.types.is_numeric_dtype(pair_time_dtype):
                                pair_df["Time"] = pd.to_datetime(pair_df["Time"], unit="s")
                            else:
                                pair_df["Time"] = pd.to_datetime(pair_df["Time"])
                    elif pd.api.types.is_numeric_dtype(df_time_dtype):
                        if not pd.api.types.is_numeric_dtype(pair_time_dtype):
                            if pd.api.types.is_datetime64_any_dtype(pair_time_dtype):
                                pair_df["Time"] = pair_df["Time"].astype(int) // 10 ** 9
                            else:
                                pair_df["Time"] = pd.to_datetime(pair_df["Time"]).astype(int) // 10 ** 9
                    else:
                        chunk_df["Time"] = pd.to_datetime(chunk_df["Time"])
                        if not pd.api.types.is_datetime64_any_dtype(pair_time_dtype):
                            if pd.api.types.is_numeric_dtype(pair_time_dtype):
                                pair_df["Time"] = pd.to_datetime(pair_df["Time"], unit="s")
                            else:
                                pair_df["Time"] = pd.to_datetime(pair_df["Time"])
 
                    # Rename OHLCV → prefixed names
                    rename_map = {}
                    for field in ["open", "high", "low", "close", "tick_volume", "real_volume"]:
                        if field in pair_df.columns:
                            rename_map[field] = f"{field}_{pair}"
                        elif field.capitalize() in pair_df.columns:
                            rename_map[field.capitalize()] = f"{field}_{pair}"
 
                    if not rename_map:
                        _logger.warning(
                            "[Currency Indices Worker] No OHLCV columns for %s in %s",
                            pair, list(pair_df.columns),
                        )
                        continue
 
                    pair_df.rename(columns=rename_map, inplace=True)
                    pair_df = pair_df[["Time"] + list(rename_map.values())]
 
                    chunk_df = chunk_df.merge(pair_df, on="Time", how="left")
                    for col in rename_map.values():
                        if col in chunk_df.columns:
                            chunk_df[col] = chunk_df[col].ffill().bfill()
 
                    _logger.info("[Currency Indices Worker] Merged %s", pair)
 
                except Exception as exc:
                    _logger.error("[Currency Indices Worker] Error merging %s: %s", pair, exc, exc_info=True)
                    continue
 
        # ------------------------------------------------------------------
        # STEP 2b  Verify all required columns now present
        # ------------------------------------------------------------------
        missing_columns = [
            f"{field}_{pair}"
            for pair in required_pairs
            for field in OHLCV_FIELDS
            if f"{field}_{pair}" not in chunk_df.columns
        ]
        if missing_columns:
            preview = missing_columns[:5]
            raise ValueError(
                f"Missing {len(missing_columns)} required columns after fetch: {preview}…"
            )
        report(30, f"Verified all {len(required_pairs)} pair columns")
 
        # ------------------------------------------------------------------
        # STEP 2c  Fill NaN gaps + fix zero tick_volume
        # ------------------------------------------------------------------
        # Three-pass strategy required for chunk-boundary safety:
        #
        #   Pass 1 – ffill().bfill()
        #     Handles interior gaps and gaps caused by MT5 time-alignment
        #     mismatches.  Works for any row that has at least one valid
        #     neighbour in the chunk.
        #
        #   Pass 2 – per-column median fallback
        #     ffill/bfill cannot fill NaNs when they sit at BOTH edges of
        #     the chunk (e.g. a pair that only has data in the middle of
        #     this slice).  Replace any survivors with the column median so
        #     the weighted-product formula never sees NaN.  Median is more
        #     robust than mean for price columns.
        #
        #   Pass 3 – inf / -inf replacement
        #     The weighted-product formula  scalar * ∏(col ^ exp)  produces
        #     ±inf when a price is extreme or when 0 is raised to a negative
        #     exponent (which ffill/bfill won't catch).  Replace with NaN
        #     then re-apply the median fallback so downstream code is clean.
        # ------------------------------------------------------------------
        pair_cols = [
            f"{field}_{pair}"
            for pair in required_pairs
            for field in ["open", "high", "low", "close", "tick_volume"]
            if f"{field}_{pair}" in chunk_df.columns
        ]
 
        # Pass 1 — ffill + bfill
        nan_before = chunk_df[pair_cols].isna().sum().sum()
        if nan_before > 0:
            _logger.info(
                "[Currency Indices Worker] Filling %d NaN gaps in pair cols (ffill+bfill)",
                nan_before,
            )
            chunk_df[pair_cols] = chunk_df[pair_cols].ffill().bfill()
            nan_after = chunk_df[pair_cols].isna().sum().sum()
            if nan_after > 0:
                _logger.warning(
                    "[Currency Indices Worker] %d NaN gaps survive ffill+bfill "
                    "(chunk-edge orphans) — applying per-column median fallback",
                    nan_after,
                )
                # Pass 2 — median fallback for chunk-edge survivors
                for col in pair_cols:
                    orphan_mask = chunk_df[col].isna()
                    if orphan_mask.any():
                        median_val = chunk_df[col].median()
                        fill_val = median_val if not np.isnan(median_val) else 1.0
                        chunk_df.loc[orphan_mask, col] = fill_val
                        _logger.warning(
                            "[Currency Indices Worker] %s: filled %d edge NaNs with %.6f",
                            col, orphan_mask.sum(), fill_val,
                        )
 
        # Fix zero / negative tick_volume before it enters the power formula
        for col in (c for c in pair_cols if c.startswith("tick_volume_")):
            zero_mask = chunk_df[col] <= 0
            if zero_mask.any():
                _logger.warning(
                    "[Currency Indices Worker] %s: %d zero/negative values → 1.0",
                    col, zero_mask.sum(),
                )
                chunk_df.loc[zero_mask, col] = 1.0
 
        # Pass 3 — replace ±inf in pair columns (pre-empt calculator blowup)
        inf_mask = np.isinf(chunk_df[pair_cols].values)
        if inf_mask.any():
            inf_count = inf_mask.sum()
            _logger.warning(
                "[Currency Indices Worker] Replacing %d ±inf values in pair cols with column median",
                inf_count,
            )
            chunk_df[pair_cols] = chunk_df[pair_cols].replace(
                [np.inf, -np.inf], np.nan
            )
            for col in pair_cols:
                still_nan = chunk_df[col].isna()
                if still_nan.any():
                    median_val = chunk_df[col].median()
                    chunk_df.loc[still_nan, col] = median_val if not np.isnan(median_val) else 1.0
 
        # ------------------------------------------------------------------
        # STEP 3  Pre-process then calculate indices
        # ------------------------------------------------------------------
        report(40, "Pre-processing merged pair data…")
        chunk_df = prepare_index_data(chunk_df)
 
        class _ProxyReporter:
            """Thin shim so CurrencyIndexCalculator can call reporter.report()."""
            def report(self, progress, message="", **kwargs):
                report(int(progress), str(message))
 
            def check_cancellation(self):
                pass
 
            def report_loop(self, current, total, message="", base_progress=0.0,
                            progress_range=10.0, **kwargs):
                if total > 0:
                    p = int(base_progress + (current / total) * progress_range)
                    report(p, message or f"{current}/{total}")
 
        calc = CurrencyIndexCalculator(chunk_df, reporter=_ProxyReporter())
        indices_dict = calc.calculate_indices(
            indices=getattr(config, "selected_indices", [])
        )
 
        calculated_indices = list(indices_dict.keys())
        report(50, f"Calculated {len(calculated_indices)} indices")
        _logger.info("[Currency Indices Worker] Calculated: %s", calculated_indices)
 
        # ------------------------------------------------------------------
        # STEP 4  Flatten indices → DataFrame and merge
        # ------------------------------------------------------------------
        frames: Dict[str, pd.Series] = {}
        for idx_name, cols_dict in indices_dict.items():
            for field_name, series in cols_dict.items():
                frames[f"{idx_name}_{field_name}"] = series
 
        indices_df = pd.DataFrame(frames, index=chunk_df.index)
 
        # ------------------------------------------------------------------
        # STEP 4b  Sanitise indices_df before merging
        # ------------------------------------------------------------------
        # CurrencyIndexCalculator uses rolling / cumulative operations that
        # produce NaN for the first N rows of any chunk (warmup period).
        # ±inf can appear when a price exponent produces overflow.
        # Neither is caught by the pair-column fill above because they arise
        # *inside* the calculator, not in the raw pair data.
        #
        # Strategy (applied per index column):
        #   1. Replace ±inf → NaN
        #   2. ffill().bfill()  — handles interior and trailing warmup rows
        #   3. Median fallback  — handles leading warmup rows at chunk start
        # ------------------------------------------------------------------
        index_cols = list(indices_df.columns)
 
        # Step 1: inf → NaN
        if np.isinf(indices_df.values).any():
            inf_count = np.isinf(indices_df.values).sum()
            _logger.warning(
                "[Currency Indices Worker] Replacing %d ±inf values in indices_df", inf_count
            )
            indices_df = indices_df.replace([np.inf, -np.inf], np.nan)
 
        # Step 2: ffill + bfill
        idx_nan_before = indices_df.isna().sum().sum()
        if idx_nan_before > 0:
            _logger.info(
                "[Currency Indices Worker] indices_df: %d NaNs before fill (warmup rows)",
                idx_nan_before,
            )
            indices_df = indices_df.ffill().bfill()
            idx_nan_after = indices_df.isna().sum().sum()
 
            # Step 3: median fallback for columns that are entirely NaN
            # (e.g. an index whose every required pair was missing for this chunk)
            if idx_nan_after > 0:
                _logger.warning(
                    "[Currency Indices Worker] %d NaNs remain in indices_df after ffill+bfill "
                    "— applying per-column median fallback",
                    idx_nan_after,
                )
                for col in index_cols:
                    still_nan = indices_df[col].isna()
                    if still_nan.any():
                        median_val = indices_df[col].median()
                        fill_val = median_val if not np.isnan(median_val) else 0.0
                        indices_df.loc[still_nan, col] = fill_val
                        _logger.warning(
                            "[Currency Indices Worker] %s: filled %d NaNs with %.6f",
                            col, still_nan.sum(), fill_val,
                        )
 
        # Drop any stale index columns from chunk_df before concat
        existing_index_cols = [c for c in chunk_df.columns if c in set(indices_df.columns)]
        if existing_index_cols:
            chunk_df = chunk_df.drop(columns=existing_index_cols)
 
        enriched_df = pd.concat([chunk_df, indices_df], axis=1)
        report(92, f"Merged {len(indices_df.columns)} index columns")

        # ── Length invariant: concat must never change row count ─────────────
        if len(enriched_df) != len(chunk_df):
            _logger.error(
                "[Currency Indices Worker] ❌ Row count changed after concat: "
                "chunk=%d, enriched=%d. Truncating to original chunk length.",
                len(chunk_df), len(enriched_df),
            )
            enriched_df = enriched_df.iloc[:len(chunk_df)]
 
        # ------------------------------------------------------------------
        # STEP 4.5  Drop intermediate pair columns
        # ------------------------------------------------------------------
        pair_columns_to_drop = [
            f"{field}_{pair}"
            for pair in required_pairs
            for field in ["open", "high", "low", "close", "tick_volume", "real_volume", "spread"]
            if f"{field}_{pair}" in enriched_df.columns
        ]
        if pair_columns_to_drop:
            enriched_df = enriched_df.drop(columns=pair_columns_to_drop)
            _logger.info(
                "[Currency Indices Worker] Dropped %d intermediate pair columns. Final shape: %s",
                len(pair_columns_to_drop), enriched_df.shape,
            )
 
        report(100, "Currency indices complete")

        # ------------------------------------------------------------------
        # TIME RESTORE: Unconditionally write back the original Time column.
        # Any merge or concat earlier in this worker may have coerced its
        # dtype (int64 ↔ datetime64 ↔ object) producing NaT sentinels.
        # Restoring from _pinned_time guarantees the timestamps that enter
        # the next pipeline stage are byte-for-byte identical to what came in.
        # ------------------------------------------------------------------
        if _time_col is not None and _pinned_time is not None:
            if _time_col in enriched_df.columns:
                enriched_df[_time_col] = _pinned_time.values  # .values avoids index re-alignment
            else:
                enriched_df.insert(0, _time_col, _pinned_time.values)
            _logger.info(
                "[Currency Indices Worker] Time column restored (dtype=%s, rows=%d)",
                _pinned_time_dtype, len(_pinned_time),
            )

        return {
            "chunk_id": chunk_id,
            "global_start_idx": global_offset,
            "result_df": enriched_df,
            "metadata": {
                "analysis_type": "currency_indices",
                "rows_processed": len(enriched_df),
                "indices_calculated": calculated_indices,
                "columns_added": len(indices_df.columns),
                "required_pairs_verified": len(required_pairs),
                "calculate_ti_for_indices": getattr(config, "calculate_ti_for_indices", None),
                "ti_config": getattr(config, "ti_config", None),
            },
        }
 
    except Exception:
        import traceback
        _logger = logging.getLogger(__name__)
        _logger.error(
            "❌ [Currency Indices Worker] chunk_id=%d failed:\n%s",
            chunk_id, traceback.format_exc(),
        )
        return {
            "chunk_id": chunk_id,
            "global_start_idx": global_offset,
            "error": traceback.format_exc(),
        }
 

#
# ============================================================================
# HANDLER REGISTRY - EXTENSIBLE ANALYSIS FUNCTIONS
# ============================================================================

from typing import Dict, Tuple, Callable, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# STRATEGY FACTORY
# ============================================================================

class StrategyFactory:
    """
    Factory for creating the appropriate processing strategy based on data size.

    This class lives in processing_strategies (not processing_utils) because it
    needs to instantiate the concrete strategy classes defined in this file.
    processing_utils re-exports it for backward compatibility.

    Strategy selection rules:
        ≤ threshold_sequential_max rows          → SEQUENTIAL
        ≤ threshold_parallel_max rows            → PARALLEL_CHUNKING
        snr / ml_* and > 30 000 rows             → SLICE_STREAMING
        currency_indices / 
        technical / astro
        and > 500 000 rows                     → SLICE_STREAMING
        anything else above threshold_parallel_max → SLICE_STREAMING
    """

    @staticmethod
    def determine_strategy(
        n_rows: int,
        config: Optional[Any] = None,
        analysis_type: Optional[str] = None,
    ) -> "ProcessingStrategy":
        """
        Return the ProcessingStrategy enum value appropriate for *n_rows*.

        Args:
            n_rows:        Number of rows in the dataset.
            config:        ProcessingConfig instance (uses defaults when None).
            analysis_type: Analysis type string — affects slice thresholds.

        Returns:
            ProcessingStrategy enum value.
        """
        from app.core.config import ProcessingConfig as _ProcessingConfig  # local to avoid leaking

        if config is None:
            config = _ProcessingConfig()

        if n_rows <= config.threshold_sequential_max:
            return ProcessingStrategy.SEQUENTIAL
        elif n_rows <= config.threshold_parallel_max:
            return ProcessingStrategy.PARALLEL_CHUNKING
        else:
            if analysis_type in ("snr", "ml_preparation", "ml_dataset_preparation", "model_training"):
                # ≤ 30k → parallel chunking; > 30k → slice streaming
                return (
                    ProcessingStrategy.SLICE_STREAMING
                    if n_rows > 30_000
                    else ProcessingStrategy.PARALLEL_CHUNKING
                )
            elif analysis_type in ("currency_indices", "technical", "astronomical"):
                # ~70 % vectorised — only slice for extreme datasets
                return (
                    ProcessingStrategy.SLICE_STREAMING
                    if n_rows > 500_000
                    else ProcessingStrategy.PARALLEL_CHUNKING
                )
            else:
                return ProcessingStrategy.SLICE_STREAMING

    @staticmethod
    def create_strategy(
        strategy_type: "ProcessingStrategy",
        context: "ProcessingContext",
        logger: Optional[Any] = None,
    ) -> "ProcessingStrategyBase":
        """
        Instantiate a strategy for *strategy_type*.

        Args:
            strategy_type: Which strategy to create.
            context:       Processing context with all required parameters.
            logger:        Logger instance.

        Returns:
            Concrete ProcessingStrategyBase instance.

        Note:
            context.task_store is intentionally NOT wrapped in a
            BroadcastingTaskStoreProxy here.  Handlers must use ProgressReporter
            explicitly for properly throttled WebSocket updates.
        """
        if strategy_type == ProcessingStrategy.SEQUENTIAL:
            return SequentialStrategy(context, logger)
        elif strategy_type == ProcessingStrategy.PARALLEL_CHUNKING:
            return ParallelChunkingStrategy(context, logger)
        elif strategy_type == ProcessingStrategy.SLICE_STREAMING:
            return SliceStreamingStrategy(context, logger)
        else:
            raise ValueError(f"Unknown or unsupported strategy: {strategy_type}")

class HandlerRegistry:
    """
    Registry for analysis processing handlers.

    Stores only the worker/handler callable. Configuration is managed
    by the parent process and passed directly to the worker.
    """

    _registry: Dict[str, Callable] = {}

    @classmethod
    def register(cls, analysis_type: str, handler: Callable) -> None:
        """
        Register an analysis processing handler.

        Args:
            analysis_type: Analysis type identifier
            handler: Worker or handler callable
        """
        cls._registry[analysis_type] = handler
        logger.info(f"✅ Registered handler for '{analysis_type}'")

    @classmethod
    def get(cls, analysis_type: str) -> Callable:
        """Get the registered handler."""
        if analysis_type not in cls._registry:
            raise ValueError(f"No handler registered for '{analysis_type}'")
        return cls._registry[analysis_type]

    @classmethod
    def is_registered(cls, analysis_type: str) -> bool:
        """Check whether a handler is registered."""
        return analysis_type in cls._registry

    @classmethod
    def get_all(cls) -> Dict[str, Callable]:
        """Return a snapshot of all registered handlers."""
        return dict(cls._registry)
def get_worker_map() -> Dict[str, Callable]:
    """
    Ensures all known analysis handlers are registered and returns
    a mapping of analysis_type -> handler callable.

    Configuration is supplied by the parent process when invoking
    the worker, so only the worker function itself is tracked here.
    """

    _handlers = [
        ("snr", snr_analysis_worker),
        ("technical", technical_analysis_worker),
        ("currency_indices", currency_indices_analysis_worker),
        ("astronomical", astronomical_analysis_worker),
        ("ml_preparation", ml_prep_worker),
        ("ml_dataset_preparation", ml_prep_worker),
    ]

    for analysis_type, handler in _handlers:
        if not HandlerRegistry.is_registered(analysis_type):
            HandlerRegistry.register(analysis_type, handler)

    return {
        analysis_type: HandlerRegistry.get(analysis_type)
        for analysis_type, _ in _handlers
    }

# ============================================================================
# MODULE-LEVEL ML PREP SHARED HELPER
# Extracted from ParallelChunkingStrategy so both Sequential and Parallel paths
# can call the same DB-persist logic without inheritance coupling.
# ============================================================================

_ml_prep_logger = logging.getLogger(__name__)

async def _save_ml_prep_chunks(
    final_result: Dict[str, Any],
    chunk_results: List[Dict[str, Any]],
    pm_instance=None,                  # PM instance — provides _current_dataset_id
    db_session=None,                   # AsyncSession for DB writes
    split_name: Optional[str] = None,  # e.g. "train" / "validation" / "test"
) -> None:
    """Persist ML prep chunk results to the database.

    Shared by SequentialStrategy and ParallelChunkingStrategy so both paths
    write sequences identically.  Previously this logic lived only on
    ParallelChunkingStrategy, meaning small val/test splits routed to
    SequentialStrategy were never saved (sequences_count stayed 0).

    Handles both in-memory arrays and disk-spooled .npz files produced by
    ml_prep_worker when the chunk exceeds the spooling threshold.
    """
    _ml_prep_logger.info(f"[ML Prep Merge] Processing {len(chunk_results)} chunk results")
    for idx, result in enumerate(chunk_results):
        _ml_prep_logger.info(f"[ML Prep Merge] Chunk {idx} keys: {list(result.keys())}")
        if "sequences" in result:
            seq = result["sequences"]
            _ml_prep_logger.info(
                f"[ML Prep Merge] Chunk {idx} sequences: "
                f"type={type(seq).__name__}, len={len(seq) if hasattr(seq, '__len__') else 'N/A'}"
            )

    if pm_instance is None or db_session is None:
        _ml_prep_logger.warning("[ML Prep Merge] No PM/DB provided, passing chunks to PM")
        final_result["chunk_results"] = chunk_results
        final_result.setdefault("metadata", {})["storage_mode"] = "chunked"
        return

    dataset_id = getattr(pm_instance, '_current_dataset_id', None)
    # split_name is an explicit param — no longer relies on undefined 'kwargs'
    split_name = split_name or getattr(pm_instance, '_current_split_name', None)

    if dataset_id is None or split_name is None:
        _ml_prep_logger.error(
            f"[ML Prep Merge] Missing dataset_id ({dataset_id}) or "
            f"split_name ({split_name}), cannot store chunks"
        )
        final_result["chunk_results"] = chunk_results
        return

    total_sequences = 0
    all_extracted_targets: set = set()

    for idx, chunk in enumerate(chunk_results):
        sequences       = chunk.get("sequences")
        labels          = chunk.get("labels")
        targets         = chunk.get("targets", {})
        sequence_metadata = chunk.get("sequence_metadata", [])

        # Support disk-spooled .npz files produced when chunk exceeds memory threshold
        tmp_path = chunk.get("tmp_path")
        if tmp_path and os.path.exists(tmp_path):
            try:
                data      = np.load(tmp_path, allow_pickle=False)
                sequences = np.array(data["sequences"])   # force copy — avoid use-after-free
                labels    = np.array(data["labels"])
                targets   = {}
                for k in data.files:
                    if k.startswith("target_"):
                        target_name = k[7:]
                        targets[target_name] = np.array(data[k])
                        all_extracted_targets.add(target_name)
                data.close()
                os.remove(tmp_path)
                _ml_prep_logger.info(
                    f"[ML Prep Merge] Loaded spooled chunk {idx} from {tmp_path}, "
                    f"found {len(targets)} target keys"
                )
            except Exception as e:
                _ml_prep_logger.error(
                    f"[ML Prep Merge] Failed to load spooled chunk {idx} from {tmp_path}: {e}"
                )

        if sequences is None or len(sequences) == 0:
            _ml_prep_logger.warning(f"[ML Prep Merge] Chunk {idx} has no sequences, skipping")
            continue

        chunk_seq_count = len(sequences)
        _ml_prep_logger.info(
            f"[ML Prep Merge] Storing chunk {idx}: {chunk_seq_count} sequences "
            f"to dataset {dataset_id} ({split_name} split)"
        )

        success = await append_sequences_to_ml_dataset(
            dataset_id=dataset_id,
            sequences=sequences,
            labels=labels,
            targets=targets,
            split_name=split_name,
            db=db_session,
            sequence_metadata=sequence_metadata,
        )

        if not success:
            _ml_prep_logger.error(f"[ML Prep Merge] Failed to store chunk {idx}")
            continue

        total_sequences += chunk_seq_count
        del sequences, labels, targets
        gc.collect()
        _ml_prep_logger.info(f"[ML Prep Merge] ✅ Chunk {idx} stored, freed memory")

    meta = final_result.setdefault("metadata", {})
    meta["total_sequences"]      = total_sequences
    meta["storage_mode"]         = "chunked_db"
    meta["split_name"]           = split_name
    meta["chunk_count"]          = len(chunk_results)
    meta["dataset_id"]           = dataset_id
    meta["extracted_target_names"] = sorted(all_extracted_targets)

    _ml_prep_logger.info(
        f"[ML Prep Merge] ✅ Extracted {len(all_extracted_targets)} unique target names: "
        f"{sorted(all_extracted_targets)}"
    )

    final_result["sequences"] = np.array([])
    final_result["labels"]    = np.array([])
    final_result["targets"]   = {}

    _ml_prep_logger.info(
        f"[ML Prep] ✅ Stored {total_sequences} sequences in {len(chunk_results)} chunks to database"
    )

# ============================================================================
# SEQUENTIAL STRATEGY
# ============================================================================

class SequentialStrategy(ProcessingStrategyBase):
    """Process data sequentially without parallelization."""

    _ML_PREP_TYPES = {"ml_preparation", "ml_dataset_preparation"}
    _CHUNK_WORKER_TYPES = {"technical"}

    async def execute(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Sequential processing for <1K rows.

        Calls the registered handler for this analysis type.

        ML prep workers (ml_prep_worker) are synchronous functions with a fixed
        positional signature:
            (chunk_df, config, task_id, progress_proxy, chunk_id, global_offset,
             slice_context)
        They cannot accept arbitrary **kwargs. ML-specific kwargs such as
        fit_scaler, skip_scaling, split_type, etc. must be packed into the
        slice_context dict and passed as the last positional argument.

        All other registered handlers are assumed to be async and are called
        with the usual (df, context, **kwargs) signature.
        """
        self.logger.info(
            f"[Sequential] Processing {len(df)} rows for {self.context.analysis_type}"
        )

        # GET handler from REGISTRY
        if not HandlerRegistry.is_registered(self.context.analysis_type):
            raise ValueError(
                f"No handler registered for '{self.context.analysis_type}'. "
                f"Register it with HandlerRegistry.register()"
            )

        handler = HandlerRegistry.get(self.context.analysis_type)

        # Wrap handler call in try/catch with error context
        try:
            if self.context.analysis_type in self._ML_PREP_TYPES:
                result = await self._execute_ml_prep(df, handler, **kwargs)
            elif self.context.analysis_type in self._CHUNK_WORKER_TYPES:
                result = await self._execute_chunk_worker(df, handler, **kwargs)
            else:
                result = await handler(df, self.context, **kwargs)

            # Validate result structure
            if not isinstance(result, dict):
                raise ValueError(f"Handler returned {type(result).__name__}, expected dict")

            if "metadata" not in result:
                result["metadata"] = {}

        except asyncio.CancelledError:
            self.logger.warning(f"[Sequential] Task cancelled for {self.context.analysis_type}")
            raise
        except Exception as handler_err:
            self.logger.error(
                f"❌ [Sequential] Handler failed for {self.context.analysis_type}: {handler_err}",
                exc_info=True
            )
            raise RuntimeError(
                f"Handler execution failed for analysis type '{self.context.analysis_type}': {handler_err}"
            ) from handler_err

        # Normalize result for compatibility
        result["metadata"]["strategy"] = ProcessingStrategy.SEQUENTIAL.value
        result["metadata"]["analysis_type"] = self.context.analysis_type

        return result

    def _build_sequential_slice_context(
        self, df: pd.DataFrame, **kwargs
    ) -> Dict[str, Any]:
        """Build the single-chunk context used by sequential chunk workers."""
        slice_context: Dict[str, Any] = {
            "slice_num": getattr(self.context, "slice_num", 0),
            "total_slices": getattr(self.context, "total_slices", 1),
            "slice_start": getattr(self.context, "slice_start", 0),
            "slice_end": getattr(self.context, "slice_end", len(df)),
            "total_dataset_rows": getattr(self.context, "total_dataset_rows", len(df)),
            "global_offset": getattr(self.context, "global_offset", 0),
            "user_id": getattr(self.context, "user_id", "unknown"),
            "connection_manager": None,
            "global_scaler": self._serialize_scaler(
                getattr(self.context, "global_scaler", None)
            ),
            "fit_scaler": kwargs.get("fit_scaler", True),
            "split_type": kwargs.get("split_type", "train"),
            "split_name": kwargs.get("split_name", "train"),
            "skip_scaling": kwargs.get("skip_scaling", False),
            "feature_cols": kwargs.get("feature_cols"),
            "columns_to_scale": kwargs.get("columns_to_scale"),
            "ml_prep_metadata": kwargs.get("ml_prep_metadata"),
            "enriched_target_columns": kwargs.get("enriched_target_columns"),
        }

        raw_slice_ctx = {}
        if hasattr(self.context.config, "slice_context"):
            raw_slice_ctx = getattr(self.context.config, "slice_context", {})
        elif isinstance(self.context.config, dict):
            raw_slice_ctx = self.context.config.get("slice_context", {})
        if isinstance(raw_slice_ctx, dict):
            slice_context.update(raw_slice_ctx)
            slice_context["connection_manager"] = None

        return slice_context

    async def _execute_chunk_worker(
        self, df: pd.DataFrame, handler: Callable, **kwargs
    ) -> Dict[str, Any]:
        """Invoke synchronous chunk-style workers from the sequential strategy."""
        slice_context = self._build_sequential_slice_context(df, **kwargs)
        self.logger.info(
            f"[Sequential-Worker] Calling {getattr(handler, '__name__', 'worker')} "
            f"as one chunk (rows={len(df)}, global_offset={slice_context['global_offset']})"
        )

        loop = asyncio.get_running_loop()
        worker_result = await loop.run_in_executor(
            None,
            lambda: handler(
                df,
                self.context.config,
                self.context.task_id,
                getattr(self.context, "task_store", None),
                0,
                slice_context["global_offset"],
                slice_context,
            ),
        )

        if isinstance(worker_result, dict) and worker_result.get("error"):
            raise RuntimeError(worker_result["error"])
        return worker_result if isinstance(worker_result, dict) else {"metadata": {}}

    async def _execute_ml_prep(
        self, df: pd.DataFrame, handler: Callable, **kwargs
    ) -> Dict[str, Any]:
        """
        Invoke ml_prep_worker with its required positional signature.

        ml_prep_worker is a synchronous multiprocessing worker.  It does NOT
        accept arbitrary **kwargs; ML-specific options (fit_scaler, skip_scaling,
        split_type, global_scaler, feature_cols, …) must arrive via the
        slice_context dict.  This method:
          1. Builds a slice_context that mirrors what _get_slice_context() builds
             for the parallel path, so the worker sees identical inputs.
          2. Runs the sync worker in the default thread-pool executor so the
             async event loop is not blocked.
        """
        # Build slice_context from kwargs — mirrors ParallelChunkingStrategy._get_slice_context()
        slice_context = self._build_sequential_slice_context(df, **kwargs)

        self.logger.info(
            f"[Sequential-ML] Calling ml_prep_worker "
            f"(fit_scaler={slice_context['fit_scaler']}, "
            f"skip_scaling={slice_context['skip_scaling']}, "
            f"split_type={slice_context['split_type']}, "
            f"rows={len(df)})"
        )

        # ── 2. Run sync worker in thread-pool (non-blocking) ─────────────────
        # Signature: (chunk_df, config, task_id, progress_proxy, chunk_id, global_offset, slice_context)
        loop = asyncio.get_running_loop()
        worker_result = await loop.run_in_executor(
            None,
            lambda: handler(
                df,
                self.context.config,
                self.context.task_id,
                getattr(self.context, "task_store", None),
                0,                            # chunk_id — single chunk
                slice_context["global_offset"],
                slice_context,
            ),
        )

        # ── 3. Persist to DB via shared module function ───────────────────────
        # Mirrors what ml_prep_callback does in the parallel path.
        # Previously this step was missing: worker ran but result was never saved,
        # causing val/test splits (< 1K rows → sequential) to store 0 sequences.
        pm_instance = kwargs.get("pm_instance")
        db_session  = kwargs.get("db_session")

        if pm_instance is None or db_session is None:
            self.logger.warning(
                "[Sequential-ML] No pm_instance/db_session — "
                "skipping DB persist. Sequences will NOT be saved."
            )
            return worker_result if isinstance(worker_result, dict) else {"metadata": {}}

        final_result: Dict[str, Any] = {"metadata": {}}
        await _save_ml_prep_chunks(
            final_result,
            [worker_result],        # single result wrapped in list — same API as parallel
            pm_instance=pm_instance,
            db_session=db_session,
            split_name=kwargs.get("split_name"),
        )
        self.logger.info(
            f"[Sequential-ML] ✅ Persisted "
            f"{final_result.get('metadata', {}).get('total_sequences', 0)} sequences "
            f"for split '{kwargs.get('split_name')}'"
        )
        return final_result


# ============================================================================
# PARALLEL CHUNKING STRATEGY
# ============================================================================

class ParallelChunkingStrategy(ProcessingStrategyBase):
    """
    Process data using parallel chunking with adaptive worker allocation.
    
    Core Principle: Single execution path that handles any worker:chunk ratio.
    - 8 cores + 8 chunks = 8 parallel workers
    - 7 cores + 8 chunks = 7 workers, one does double work
    - 16 cores + 4 chunks = 4 workers (no over-allocation)
    
    This is the foundational parallel processing strategy.
    """
    
    def __init__(self, context: ProcessingContext, logger: Optional[logging.Logger] = None):
        """Initialize with context and setup worker allocation."""
        super().__init__(context, logger)
        self._worker_count = None
        self._chunk_count = None

    async def execute(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Single execution method that handles all parallel processing scenarios.
        
        Flow:
        1. Determine optimal worker count based on system resources
        2. Create chunks based on data size and analysis type
        3. Distribute chunks among workers (round-robin if workers < chunks)
        4. Execute and merge results
        
        Returns consistent result structure regardless of worker:chunk ratio.
        """
        self.logger.info(f"[Parallel] Starting execution on {len(df)} rows for {self.context.analysis_type}")
        
        # Step 1: Prepare DataFrame
        df = self._prepare_dataframe(df)
        
        # Step 2: Determine worker allocation
        self._worker_count = self._calculate_optimal_workers()
        
        # Step 3: Create chunks
        chunks = self._create_chunks(df)
        self._chunk_count = len(chunks)
        
        # FIX: Cap workers at chunk count - never spawn more workers than chunks
        if self._worker_count > self._chunk_count:
            original_workers = self._worker_count
            self._worker_count = self._chunk_count
            self.logger.info(
                f"[Workers] Capped workers from {original_workers} → {self._worker_count} "
                f"(no need for more workers than chunks)"
            )
        
        # Step 4: Log execution plan
        self._log_execution_plan()
        
        # Step 5: Execute processing (pass kwargs to workers)
        # Define a callback for immediate storage if this is an ML prep task
        chunk_callback = None
        if self.context.analysis_type in ["ml_preparation", "ml_dataset_preparation"]:
            # Capture kwargs values into locals NOW — stable closure bindings,
            # no dependency on 'kwargs' being in scope when callback fires.
            _cb_pm_instance = kwargs.get('pm_instance')
            _cb_db_session  = kwargs.get('db_session')
            _cb_split_name  = kwargs.get('split_name')

            async def ml_prep_callback(chunk_result):
                chunks_to_process = chunk_result if isinstance(chunk_result, list) else [chunk_result]
                temp_result = {"metadata": {}}
                # _save_ml_prep_chunks handles both inline arrays and disk-spooled .npz files
                await _save_ml_prep_chunks(
                    temp_result, chunks_to_process,
                    pm_instance=_cb_pm_instance,
                    db_session=_cb_db_session,
                    split_name=_cb_split_name,
                )
                # Propagate lightweight metadata back onto stub results
                for r in chunks_to_process:
                    r["metadata"] = r.get("metadata", {})
                    r["metadata"]["total_sequences"] = temp_result["metadata"].get("total_sequences", 0)
                    r["dataset_id"] = temp_result["metadata"].get("dataset_id")

            chunk_callback = ml_prep_callback

        results = await self._execute_parallel_processing(chunks, df, chunk_callback=chunk_callback, **kwargs)
        
        self.logger.info(f"[Parallel] Completed execution with {self._worker_count} workers and {self._chunk_count} chunks")
        
        # Pass the streaming flag to _merge_results so it knows to skip DF concatenation
        # but still perform metadata aggregation.
        merge_kwargs = {**kwargs, "is_streaming": (chunk_callback is not None)}
        
        # Step 6: Merge and return (pass kwargs for PM/DB session)
        return await self._merge_results(results, df, **merge_kwargs)

    
    def _calculate_optimal_workers(self) -> int:
        """
        Calculate optimal number of workers based on system resources.
        
        Strategy:
        - Respect system CPU count
        - Consider memory constraints
        - Account for slice context (nested execution)
        - Apply analysis-specific optimizations
        
        Returns:
            Optimal worker count (1 to system_max)
        """
        cpu_cores = multiprocessing.cpu_count() or 4
        
        # Base worker calculation with conservative approach
        if cpu_cores <= 2:
            base_workers = cpu_cores  # Use all cores on low-end systems
        elif cpu_cores <= 4:
            base_workers = cpu_cores  # Use all cores on quad-core
        elif cpu_cores <= 8:
            base_workers = max(4, int(cpu_cores * 0.75))  # 75% utilization
        elif cpu_cores <= 16:
            base_workers = max(6, int(cpu_cores * 0.625))  # 62.5% utilization
        else:
            base_workers = max(8, int(cpu_cores * 0.5))   # 50% for high-core systems
        
        # Apply slice context adjustment (nested execution)
        if self.context.total_slices > 1:
            # We're inside a slice - be more conservative
            slice_factor = 0.75 if cpu_cores >= 8 else 0.85
            base_workers = min(base_workers, max(2, int(cpu_cores * slice_factor)))
            self.logger.debug(
                f"[Workers] Slice context detected ({self.context.slice_num + 1}/{self.context.total_slices}): "
                f"reduced to {base_workers} workers"
            )
        
        # ML-SPECIFIC: Reduce worker count for ML prep (memory-intensive, CPU-greedy)
        if self.context.analysis_type in ["ml_preparation", "ml_dataset_preparation"]:
            # ML prep is memory-intensive and CPU-greedy (numpy operations)
            # Use 50% of cores to leave headroom for system responsiveness
            ml_workers = max(2, int(cpu_cores * 0.5))
            if base_workers > ml_workers:
                self.logger.info(
                    f"[Workers] ML Prep throttling: {base_workers} → {ml_workers} workers "
                    f"(50% utilization for memory-intensive operations)"
                )
                base_workers = ml_workers
        
        # Apply absolute limits
        max_workers = 16  # Reasonable upper bound
        optimal_workers = max(1, min(base_workers, max_workers))
        
        self.logger.info(
            f"[Workers] System: {cpu_cores} cores → Optimal: {optimal_workers} workers "
            f"({(optimal_workers/cpu_cores)*100:.0f}% utilization)"
        )
        
        return optimal_workers
    
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare DataFrame for parallel processing.
        
        - Reset index if needed
        - Remove duplicate columns
        - Apply analysis-specific preprocessing
        """
        df_prepared = df.copy()
        
        # Normalizing the index is highly dangerous as it destroys temporal alignment.
        # We must preserve the original DatetimeIndex or RangeIndex so ProcessingManager
        # can reindex correctly after warmup rows are dropped.
        # Removing all reset_index(drop=True) calls.        
        # Remove duplicate columns
        df_prepared = df_prepared.loc[:, ~df_prepared.columns.duplicated(keep="last")]
        
        # Analysis-specific preprocessing
        if self.context.analysis_type == "ml_preparation":
            df_prepared = self._preprocess_ml_data(df_prepared)
        
        return df_prepared
    
    def _preprocess_ml_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for ML preparation analysis.
        
        ML-specific preprocessing:
        1. Ensure numeric data types for features
        2. Handle missing values (forward fill for time series)
        3. Validate required columns exist
        4. Sort by time index if present
        5. Remove any duplicate timestamps
        
        Returns:
            Preprocessed DataFrame ready for ML sequence generation
        """
        df_processed = df.copy()
        
        # 1. Sort by time index if present
        if 'time' in df_processed.columns:
            df_processed = df_processed.sort_values('time')
            self.logger.debug("[ML-Preprocess] Sorted by time column")
        elif isinstance(df_processed.index, pd.DatetimeIndex):
            df_processed = df_processed.sort_index()
            self.logger.debug("[ML-Preprocess] Sorted by datetime index")
        
        # 2. Remove duplicate timestamps if time column exists
        if 'time' in df_processed.columns:
            before_count = len(df_processed)
            df_processed = df_processed.drop_duplicates(subset=['time'], keep='first')
            after_count = len(df_processed)
            if before_count != after_count:
                self.logger.warning(
                    f"[ML-Preprocess] Removed {before_count - after_count} duplicate timestamps"
                )
        
        # 3. Validate required OHLCV columns for financial data (case-insensitive)
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        # Create case-insensitive column mapping
        col_lower_map = {col.lower(): col for col in df_processed.columns}
        missing_cols = required_cols - set(col_lower_map.keys())
        
        if missing_cols:
            self.logger.warning(
                f"[ML-Preprocess] Missing OHLCV columns: {missing_cols}. "
                f"ML prep may fail if these are required features."
            )
        else:
            # FIX #11: Rename columns to Title Case (not lowercase) for consistency
            # This aligns with ProcessingManager's expectation of Time/Open/High/Low/Close/Volume.
            title_case_map = {
                'open': 'Open', 'high': 'High', 'low': 'Low', 
                'close': 'Close', 'volume': 'Volume'
            }
            rename_map = {}
            for req_col in required_cols:
                if req_col in col_lower_map:
                    original_name = col_lower_map[req_col]
                    target_name = title_case_map[req_col]
                    if original_name != target_name:
                        rename_map[original_name] = target_name
            
            if rename_map:
                df_processed = df_processed.rename(columns=rename_map)
                self.logger.debug(f"[ML-Preprocess] Normalized column names to Title Case: {rename_map}")
        
        # 4. Ensure numeric data types for all feature columns
        # Identify numeric columns (exclude time/date columns)
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = [col for col in df_processed.columns if col not in numeric_cols and col != 'time']
        
        if non_numeric_cols:
            self.logger.debug(f"[ML-Preprocess] Non-numeric columns found: {non_numeric_cols}")
            # Try to convert to numeric where possible
            for col in non_numeric_cols:
                if col == 'time':
                    continue
                try:
                    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                    self.logger.debug(f"[ML-Preprocess] Converted {col} to numeric")
                except Exception as e:
                    self.logger.warning(f"[ML-Preprocess] Could not convert {col} to numeric: {e}")
        
        # 5. Handle missing values (forward fill for time series continuity)
        null_counts_before = df_processed.isnull().sum()
        if null_counts_before.any():
            self.logger.debug(
                f"[ML-Preprocess] Found null values: {null_counts_before[null_counts_before > 0].to_dict()}"
            )
            
            # Forward fill (use previous value for time series)
            df_processed = df_processed.fillna(method='ffill')
            
            # Backward fill for any remaining nulls at the start
            df_processed = df_processed.fillna(method='bfill')
            
            # If still nulls, fill with 0 (last resort)
            remaining_nulls = df_processed.isnull().sum()
            if remaining_nulls.any():
                df_processed = df_processed.fillna(0)
                self.logger.warning(
                    f"[ML-Preprocess] Filled remaining nulls with 0: "
                    f"{remaining_nulls[remaining_nulls > 0].to_dict()}"
                )
        
        # 6. Validate no infinite values
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        inf_mask = np.isinf(df_processed[numeric_cols]).any()
        if inf_mask.any():
            inf_cols = inf_mask[inf_mask].index.tolist()
            self.logger.warning(f"[ML-Preprocess] Found infinite values in: {inf_cols}")
            # Replace inf with large finite values
            df_processed[numeric_cols] = df_processed[numeric_cols].replace([np.inf, -np.inf], np.nan)
            df_processed = df_processed.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # 7. Log preprocessing summary
        self.logger.info(
            f"[ML-Preprocess] Complete: {len(df_processed)} rows, {len(df_processed.columns)} columns, "
            f"all numeric features validated"
        )
        
        return df_processed
    
    def _create_chunks(self, df: pd.DataFrame) -> List[Any]:
        """
        Create chunks based on data size and analysis type.
        
        Strategy:
        - Calculate optimal chunk size based on memory constraints
        - Use analysis-specific chunking logic
        - Ensure chunks don't exceed memory limits
        """
        # Calculate target chunk count based on memory constraints
        target_chunk_count = self._calculate_target_chunk_count(df)
        
        if self.context.analysis_type == "snr":
            return self._create_snr_chunks(df, target_chunk_count)
        elif self.context.analysis_type in ["ml_preparation", "ml_dataset_preparation"]:
            return self._create_ml_chunks(df, target_chunk_count)
        else:
            return self._create_standard_chunks(df, target_chunk_count)

    def _create_ml_chunks(self, df: pd.DataFrame, target_count: int) -> List[Any]:
        """
        Create ML-specific chunks with overlap to prevent sequence loss at boundaries.
        Overlap size must be at least (sequence_length + prediction_length).
        """
        # Get overlap parameters from config
        sequence_length = getattr(self.context.config, 'sequence_length', 100)
        prediction_length = getattr(self.context.config, 'prediction_length', 1)
        
        # We need enough overlap to complete any sequence that starts in this chunk
        overlap = sequence_length + prediction_length + 5 # +5 for safety margin
        
        self.logger.info(f"[ML-Chunks] Creating {target_count} chunks with {overlap} row overlap")
        
        # Use RowChunker with overlap
        chunks_df_list = RowChunker.chunk_dataframe_by_rows(df, target_count, overlap=overlap)
        
        chunks = []
        current_idx = 0

        # Bug #5 fix: advance by the canonical (non-overlap) chunk size so that
        # global_start_idx reflects only the rows each chunk "owns".  Counting all
        # overlapped rows caused every subsequent chunk's start offset to be too
        # large, misaligning signal/sequence anchors in the final merged DataFrame.
        canonical_size = max(1, len(chunks_df_list[0]) - overlap) if chunks_df_list else 1

        for i, chunk_df in enumerate(chunks_df_list):
            chunks.append(StandardChunk(
                chunk_id=i,
                data=chunk_df,
                global_start_idx=current_idx
            ))
            # Advance by canonical rows only (exclude overlap region)
            canonical_size = max(1, len(chunk_df) - overlap)
            current_idx += canonical_size

        return chunks
    
    def _calculate_target_chunk_count(self, df: pd.DataFrame) -> int:
        """
        Calculate optimal number of chunks based on memory constraints.
        
        Strategy: Use available memory efficiently while maintaining stability.
        - Measure actual memory footprint per row
        - Target 200MB per chunk (aggressive but safe)
        - Allow larger chunks if memory permits
        """
        # Memory-based chunk calculation
        if self.context.analysis_type == "snr":
            # SNR analysis is memory-intensive
            # Actual measurement: ~15-20MB per 1K rows (not 20MB as previously estimated)
            estimated_mb_per_1k_rows = 15  # More accurate estimate
            target_mb_per_chunk = 200  # Aggressive: 200MB per chunk (was 100MB)
            
            # Calculate safe rows per chunk
            safe_rows_per_chunk = int((target_mb_per_chunk / estimated_mb_per_1k_rows) * 1000)
            
            # Remove artificial cap - let memory calculation determine size
            # Only enforce minimum for parallelization efficiency
            safe_rows_per_chunk = max(1024, safe_rows_per_chunk)  # Min 1024, no max cap
            
            # Log the calculation for monitoring
            self.logger.info(
                f"[Chunks] Memory-based calculation: {estimated_mb_per_1k_rows}MB/1K rows, "
                f"target {target_mb_per_chunk}MB/chunk → {safe_rows_per_chunk} rows/chunk"
            )
            
            target_chunks = math.ceil(len(df) / safe_rows_per_chunk)
            max_chunk_size = safe_rows_per_chunk  # FIX: Define for log statement below
        else:
            # Standard analysis - less memory intensive
            # ML dataset preparation needs larger chunks for efficiency
            if self.context.analysis_type in ["ml_preparation", "ml_dataset_preparation"]:
                # Use centralized config for ML chunk size
                max_chunk_size = getattr(self.context.config, 'ml_chunk_size', 10240)
            else:
                # Use centralized config for standard chunk size
                max_chunk_size = getattr(self.context.config, 'safe_chunk_size', 2048)
            
            target_chunks = max(self._worker_count, math.ceil(len(df) / max_chunk_size))
        
        self.logger.info(f"[Chunks] Target chunk count: {target_chunks} for {len(df)} rows (chunk_size={max_chunk_size})")
        return target_chunks
    
    def _create_snr_chunks(self, df: pd.DataFrame, target_count: int) -> List[Any]:
        """Create SNR-specific chunks with overlap."""
        chunks = smart_chunk_dataframe(
            df,
            target_count,
            self.context.config.lookback_period,
            self.context.config.lookback_period,
            self.context.config.confirmation_period,
            self.context.config.lookforward_period,
        )
        
        # Log chunk statistics
        chunk_sizes = [chunk.total_rows_to_process for chunk in chunks]
        self.logger.info(
            f"[SNR-Chunks] Created {len(chunks)} chunks: "
            f"avg={sum(chunk_sizes)/len(chunk_sizes):.0f} rows, "
            f"max={max(chunk_sizes)} rows, sizes={chunk_sizes}"
        )
        
        return chunks
    
    def _create_standard_chunks(self, df: pd.DataFrame, target_count: int) -> List[Any]:
        """Create standard chunks for non-SNR analysis."""
        chunks_df_list = RowChunker.chunk_dataframe_by_rows(df, target_count)
        
        chunks = []
        current_idx = 0
        for i, chunk_df in enumerate(chunks_df_list):
            chunks.append(StandardChunk(
                chunk_id=i,
                data=chunk_df,
                global_start_idx=current_idx
            ))
            current_idx += len(chunk_df)
        
        # Log chunk statistics
        chunk_sizes = [len(chunk.data) for chunk in chunks]
        self.logger.info(
            f"[Standard-Chunks] Created {len(chunks)} chunks: "
            f"avg={sum(chunk_sizes)/len(chunk_sizes):.0f} rows, "
            f"max={max(chunk_sizes)} rows, sizes={chunk_sizes}"
        )
        
        return chunks
    
    def _log_execution_plan(self):
        """Log the execution plan for transparency."""
        worker_names = get_worker_names(self._worker_count)
        
        if self._worker_count >= self._chunk_count:
            execution_mode = f"{self._chunk_count} workers (1:1 mapping)"
        else:
            chunks_per_worker = math.ceil(self._chunk_count / self._worker_count)
            execution_mode = f"{self._worker_count} workers handling {chunks_per_worker} chunks each"
        
        self.logger.info(
            f"[Execution Plan] {execution_mode} | "
            f"Workers: {', '.join(worker_names[:3])}{'...' if len(worker_names) > 3 else ''}"
        )
    
    async def _execute_parallel_processing(
        self, 
        chunks: List[Any], 
        df: pd.DataFrame, 
        chunk_callback: Optional[Callable] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Execute parallel processing with unified worker distribution.
        
        Enhanced with:
        - Comprehensive error handling
        - Memory management and cleanup
        - Detailed progress tracking
        - Resource monitoring
        """
        # Memory monitoring at start
        if psutil:
            process = psutil.Process()
            mem_start = process.memory_info().rss / 1024 / 1024
            self.logger.info(f"[Parallel] Starting execution. Memory: {mem_start:.1f} MB")
        else:
            process = None
            mem_start = None
        
        # Set up progress tracking
        manager_mp = multiprocessing.Manager()
        progress_queue = manager_mp.Queue()
        progress_proxy = QueueProgressStoreProxy(progress_queue)
        
        # Build slice context with ML-specific parameters
        slice_context = self._get_slice_context(df, **kwargs)
        
        # Start progress listener
        loop = asyncio.get_running_loop()
        listener_task = asyncio.create_task(
            self._progress_listener(progress_queue, slice_context, self._chunk_count)
        )
        
        results = []
        
        try:
            # Determine execution strategy
            if self._worker_count >= self._chunk_count:
                # Direct 1:1 mapping
                self.logger.info(f"[Parallel] Using direct mapping: {self._chunk_count} chunks → {self._worker_count} workers")
                results = await self._execute_direct_mapping(chunks, progress_proxy, slice_context, chunk_callback=chunk_callback)
            else:
                # Round-robin distribution
                chunks_per_worker = math.ceil(self._chunk_count / self._worker_count)
                self.logger.info(f"[Parallel] Using round-robin: {self._chunk_count} chunks → {self._worker_count} workers ({chunks_per_worker} chunks/worker)")
                results = await self._execute_round_robin(chunks, progress_proxy, slice_context, chunk_callback=chunk_callback)
            
            # Memory monitoring after execution
            if process:
                mem_after = process.memory_info().rss / 1024 / 1024
                self.logger.info(f"[Parallel] Execution complete. Memory: {mem_after:.1f} MB (+{mem_after - mem_start:.1f} MB)")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ [Parallel] Execution failed: {e}", exc_info=True)
            raise
            
        finally:
            # Comprehensive cleanup
            try:
                # Stop progress listener
                progress_queue.put({"action": "stop"})
                await listener_task
                
                # Cleanup multiprocessing resources
                if 'manager_mp' in locals():
                    manager_mp.shutdown()
                
                # NO pool.join() needed - ProcessPoolExecutor context manager handles all cleanup
                # The 'with' statement in _execute_direct_mapping() already waits for completion
                
                # Force garbage collection
                gc.collect()
                
                self.logger.debug("[Parallel] Cleanup complete")
                
            except Exception as cleanup_error:
                self.logger.warning(f"⚠️ [Parallel] Cleanup error (non-fatal): {cleanup_error}")
    
    async def _execute_direct_mapping(
        self, 
        chunks: List[Any], 
        progress_proxy, 
        slice_context, 
        chunk_callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """Execute with direct 1:1 worker-to-chunk mapping using ProcessPoolExecutor for better asyncio compatibility."""
        import concurrent.futures
        
        worker_func = self._get_worker_function()
        chunk_args = self._prepare_chunk_arguments(chunks, progress_proxy, slice_context)
        
        try:
            self.logger.info(f"[DirectMapping] Starting {self._worker_count} workers for {len(chunks)} chunks")
            
            loop = asyncio.get_running_loop()
            
            # FIX: Use ProcessPoolExecutor instead of multiprocessing.Pool for better asyncio integration
            # OOM FIX: Use 'spawn' context to prevent TensorFlow/Memory from being copied via fork
            import multiprocessing
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self._worker_count,
                initializer=worker_init,
                mp_context=multiprocessing.get_context('spawn')
                # Note: ProcessPoolExecutor doesn't support maxtasksperchild directly
            ) as executor:
                
                # Submit all tasks and wrap for asyncio
                async_futures = [asyncio.wrap_future(executor.submit(worker_func, *args)) for args in chunk_args]
      
                
                # FIX: Wait for futures IN ORDER to preserve temporal sorting
                # as_completed() yields results as they finish (non-deterministic).
                # Iterating over the original futures list ensures results are processed
                # in the order they were submitted, which is CRITICAL for streaming/DB storage.
                results = []
                for idx, fut in enumerate(async_futures):
                    try:
                        result = await fut
                        
                        # Immediate Callback for Streaming Storage
                        if chunk_callback:
                            await chunk_callback(result)
                            # Return lightweight metadata only if callback handled storage
                            results.append({
                                "chunk_id": result.get("chunk_id", idx),
                                "status": "stored_via_callback",
                                "metadata": result.get("metadata", {})
                            })
                            # Free memory immediately
                            del result
                            gc.collect()
                        else:
                            results.append(result)
                            
                    except Exception as e:
                        self.logger.error(f"❌ [DirectMapping] Worker failed: {e}")
                        results.append({"error": str(e), "chunk_id": -1})
            
            self.logger.info(f"[DirectMapping] Completed successfully: {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ [DirectMapping] Failed: {e}")
            raise
            
        finally:
            # FIX: Cleanup arguments after execution
            try:
                del chunk_args
                if 'futures' in locals():
                    del futures
                gc.collect()
                gc.collect()
            except:
                pass
    
    async def _execute_round_robin(
        self, 
        chunks: List[Any], 
        progress_proxy, 
        slice_context,
        chunk_callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """Execute with round-robin chunk distribution using ProcessPoolExecutor for better asyncio compatibility."""
        import concurrent.futures
        
        
        # Distribute chunks among workers
        worker_assignments = self._distribute_chunks_round_robin(chunks)
        worker_names = get_worker_names(self._worker_count)
        
        # Prepare worker arguments
        worker_args = []
        for worker_id, worker_chunks in enumerate(worker_assignments):
            worker_name = worker_names[worker_id] if worker_id < len(worker_names) else f"Worker-{worker_id}"
            args = (
                worker_chunks,
                self.context.analysis_type,
                self.context.config,
                self.context.task_id,
                progress_proxy,
                worker_id,
                slice_context,
                worker_name
            )
            worker_args.append(args)
        
        try:
            self.logger.info(f"[RoundRobin] Starting {self._worker_count} workers for {len(chunks)} chunks")
            
            # FIX: Use ProcessPoolExecutor instead of multiprocessing.Pool for better asyncio integration
            # OOM FIX: Use 'spawn' context to prevent TensorFlow/Memory from being copied via fork
            import multiprocessing
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self._worker_count,
                initializer=worker_init,
                mp_context=multiprocessing.get_context('spawn')
            ) as executor:
                
                # Submit all worker tasks and wrap for asyncio
                async_futures = [asyncio.wrap_future(executor.submit(sequential_chunk_worker, *args)) for args in worker_args]
                
                # FIX: Wait for futures IN ORDER to preserve temporal sorting
                flattened_results = []
                for idx, fut in enumerate(async_futures):
                    try:
                        worker_results = await fut
                        
                        if isinstance(worker_results, list):
                            # Immediate Callback for Streaming Storage
                            if chunk_callback:
                                await chunk_callback(worker_results)
                                for res in worker_results:
                                    flattened_results.append({
                                        "chunk_id": res.get("chunk_id", -1),
                                        "status": "stored_via_callback",
                                        "metadata": res.get("metadata", {})
                                    })
                                del worker_results
                                gc.collect()
                            else:
                                flattened_results.extend(worker_results)
                        else:
                            self.logger.warning(f"⚠️ Unexpected worker result type: {type(worker_results)}")
                            flattened_results.append(worker_results)
                            
                    except Exception as e:
                        self.logger.error(f"❌ [RoundRobin] Worker failed: {e}")
                        flattened_results.append({"error": str(e), "worker_id": -1})
            
            self.logger.info(f"[RoundRobin] Completed successfully: {len(flattened_results)} total results")
            return flattened_results
            
        except Exception as e:
            self.logger.error(f"❌ [RoundRobin] Failed: {e}")
            raise
            
        finally:
            # FIX: Cleanup arguments after execution
            try:
                del worker_args
                del worker_assignments
                if 'futures' in locals():
                    del futures
                if 'flattened_results' in locals():
                    del flattened_results
                gc.collect()
                gc.collect()
            except:
                pass
    
    def _get_worker_function(self):
        """Get the appropriate worker function for the analysis type."""
        worker_map = get_worker_map()
        
        worker_func = worker_map.get(self.context.analysis_type)
        if not worker_func:
            supported_types = list(worker_map.keys())
            raise ValueError(
                f"Analysis type '{self.context.analysis_type}' not supported for parallel chunking. "
                f"Supported types: {supported_types}. "
                f"Note: MODEL_BUILD and MODEL_TRAINING use single-threaded execution."
            )
        
        self.logger.info(f"[Workers] Selected worker function: {worker_func.__name__} for {self.context.analysis_type}")
        return worker_func
    
    def _prepare_chunk_arguments(self, chunks: List[Any], progress_proxy, slice_context) -> List[tuple]:
        """Prepare arguments for chunk processing."""
        chunk_args = []
        for chunk in chunks:
            chunk_task_id = f"{self.context.task_id}_chunk_{chunk.chunk_id}"
            
            if self.context.analysis_type == "snr":
                args = (
                    chunk.data,
                    self.context.config.confirmation_period,
                    self.context.config.lookback_period,
                    self.context.config.n_clusters,
                    self.context.config.zone_width,
                    self.context.config.min_distance_pct,
                    self.context.config.lookforward_period,
                    self.context.config.animation_step,
                    chunk_task_id,
                    progress_proxy,
                    chunk.chunk_id,
                    chunk.global_start_idx,
                    slice_context
                )
            else:
                args = (
                    chunk.data,
                    self.context.config,
                    chunk_task_id,
                    progress_proxy,
                    chunk.chunk_id,
                    chunk.global_start_idx,
                    slice_context
                )
            
            chunk_args.append(args)
        
        return chunk_args
    
    def _distribute_chunks_round_robin(self, chunks: List[Any]) -> List[List[Any]]:
        """Distribute chunks round-robin among workers."""
        worker_assignments = [[] for _ in range(self._worker_count)]
        for i, chunk in enumerate(chunks):
            worker_id = i % self._worker_count
            worker_assignments[worker_id].append(chunk)
        return worker_assignments
    
 
    async def _progress_listener(self, progress_queue, slice_context, total_chunks):
        """Listen for progress updates from workers."""
        loop = asyncio.get_running_loop()
        chunk_progress_map: Dict[int, float] = {}
        
        self.logger.info(f"[Progress] Listener started for {total_chunks} chunks")
        
        while True:
            try:
                message = await loop.run_in_executor(None, progress_queue.get)
                if not message:
                    continue
                
                action = message.get("action")
                if action == "stop":
                    break
                
                if action in ["update", "progress"]:
                    data = message.get("data", message)
                    raw_progress = data.get("progress")
                    chunk_id = data.get("chunk_id", 0)
                    
                    if raw_progress is not None:
                        chunk_progress_map[chunk_id] = float(raw_progress)
                        
                        # Use average progress across all known chunks for both task store AND WebSocket
                        # Previously WebSocket used (completed_chunks/total) * 100 which stayed at 0
                        # until a chunk finished. Now both use avg so intermediate progress is visible.
                        avg_progress = sum(chunk_progress_map.values()) / max(len(chunk_progress_map), 1)
                        
                        # Update task store
                        if self.context.task_store:
                            self.context.task_store.update_task(
                                self.context.task_id,
                                progress=avg_progress,
                                message=data.get("message", "Processing...")
                            )
                        
                        # Send WebSocket update using avg progress (not completed-count)
                        if self.context.connection_manager:
                            completed_chunks = len([p for p in chunk_progress_map.values() if p >= 100])
                            
                            await self.context.connection_manager.send_progress_update(
                                self.context.task_id,
                                {
                                    "type": "progress",
                                    "progress": round(min(avg_progress, 99.9), 2),  # avg, not completed-count
                                    "message": data.get("message", "Processing..."),
                                    "chunk_id": chunk_id,
                                    "completed_chunks": completed_chunks,
                                    "total_chunks": total_chunks
                                }
                            )
                            
            except Exception as e:
                self.logger.warning(f"Progress listener error: {e}")
                break
        
        self.logger.info(f"[Progress] Listener stopped. Final: {len(chunk_progress_map)}/{total_chunks} chunks")
    
    async def _merge_results(self, results: List[Dict[str, Any]], original_df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Merge parallel processing results into final output.

        Enhanced with:
        - Comprehensive error handling for all analysis types
        - Memory management and cleanup
        - Detailed logging for debugging
        - Result validation
        """
        analysis_type = self.context.analysis_type  # cache once — used 6+ times below
        is_ml_prep = analysis_type in {"ml_preparation", "ml_dataset_preparation"}  # set lookup, not list

        if not results:
            self.logger.warning("[Merge] No results to merge - returning original DataFrame")
            return {
                "result_df": original_df,
                "metadata": {
                    "strategy": ProcessingStrategy.PARALLEL_CHUNKING.value,
                    "analysis_type": analysis_type,
                    "workers": self._worker_count,
                    "chunks": self._chunk_count,
                    "rows_processed": len(original_df),
                    "storage_mode": "streaming_db" if kwargs.get("is_streaming") else "standard",
                    "warning": "No chunk results received",
                },
            }

        is_streaming = kwargs.get("is_streaming", False)
        self.logger.info(f"[Merge] Processing {len(results)} chunk results for {analysis_type} (streaming={is_streaming})")

        # Initialization
        chunk_dfs = []
        failed_chunks = []
        metadata_accumulator = {}
        n_successful = 0

        results.sort(key=lambda x: x.get("chunk_id") if isinstance(x.get("chunk_id"), int) else 999)

        for i, result in enumerate(results):
            # --- classify ---
            if not isinstance(result, dict):
                failed_chunks.append({
                    "chunk_index": i,
                    "chunk_id": "unknown",
                    "error": f"Invalid result type: {type(result).__name__}",
                })
                continue

            if result.get("error"):
                failed_chunks.append({
                    "chunk_index": i,
                    "chunk_id": result.get("chunk_id", "unknown"),
                    "error": result["error"],
                })
                continue

            n_successful += 1

            # --- extract DataFrame (SKIP if streaming) ---
            if not is_streaming:
                df_chunk = result.get("result_df")
                if isinstance(df_chunk, pd.DataFrame) and not df_chunk.empty:
                    validation_error = self._validate_chunk_dataframe(df_chunk, result.get("chunk_id", "unknown"))
                    if validation_error:
                        self.logger.warning(f"⚠️ Chunk validation warning: {validation_error}")
                    chunk_dfs.append((result.get("global_start_idx", 0), df_chunk))
                elif not is_ml_prep:
                    self.logger.warning(f"⚠️ Chunk {result.get('chunk_id', 'unknown')} has no valid result_df")

            # --- accumulate metadata ---
            self._accumulate_chunk_metadata(metadata_accumulator, result)

        # Report failures
        if failed_chunks:
            # generator expression avoids building a temporary list just to join it
            error_msg = (
                f"Parallel processing failed on {len(failed_chunks)}/{len(results)} chunks: "
                + "; ".join(
                    f"chunk_{f['chunk_id']}: {f['error']}" for f in failed_chunks
                )
            )
            self.logger.error(f"❌ {error_msg}")

            if n_successful == 0:
                raise RuntimeError(f"All chunks failed: {error_msg}")

            self.logger.warning(f"⚠️ Continuing with {n_successful} successful chunks")

        # Merge DataFrames
        merged_df = original_df
        if chunk_dfs and not is_streaming:
            try:
                # Sort in-place (no second list).  key accesses index 0 of each tuple.
                chunk_dfs.sort(key=lambda x: x[0])

                sorted_dfs = [df for _, df in chunk_dfs]
                del chunk_dfs  # free tuple list before concat — concat may spike RAM

                total_rows = sum(len(df) for df in sorted_dfs)
                self.logger.info(
                    f"[Merge] Concatenating {len(sorted_dfs)} DataFrames "
                    f"(total rows: {total_rows}, columns: {sorted_dfs[0].columns.tolist()[:5]}...)"
                )

                # FIX #6: free each element from sorted_dfs immediately after concat
                # so peak memory is (all_chunks + result) not (all_chunks + all_chunks + result).
                # pd.concat with copy=False avoids an extra copy during concatenation itself.
                merged_df = pd.concat(sorted_dfs, ignore_index=False, copy=False)
                # Clear the list in-place so individual DataFrames can be GC'd while
                # the final merged_df is still being processed below.
                for _i in range(len(sorted_dfs)):
                    sorted_dfs[_i] = None
                del sorted_dfs
                
                # DEFENSIVE SORT: Ensure chronological order within the merged chunks
                # This is a safety measure in case chunk_id alignment had gaps or overlaps.
                if analysis_type in ["snr", "ml_preparation", "ml_dataset_preparation", "currency_indices"]:
                    time_col = next((c for c in merged_df.columns if c.lower() == 'time'), None)
                    if time_col:
                        try:
                            # Use monotonic check to skip sorting if already correct (performance)
                            if not merged_df[time_col].is_monotonic_increasing:
                                self.logger.info(f"[Merge] Non-monotonic data detected, sorting by '{time_col}'")
                                if not pd.api.types.is_datetime64_any_dtype(merged_df[time_col]):
                                    if pd.api.types.is_numeric_dtype(merged_df[time_col]):
                                        merged_df[time_col] = pd.to_datetime(merged_df[time_col], unit='s', errors='coerce')
                                    else:
                                        merged_df[time_col] = pd.to_datetime(merged_df[time_col], errors='coerce')
                                merged_df = merged_df.sort_values(time_col)

                            # Drop rows where Time is NaT (produced by pair-merge misalignment in currency_indices chunks)
                            nat_mask = merged_df[time_col].isna()
                            if nat_mask.any():
                                n_dropped = nat_mask.sum()
                                self.logger.warning(
                                    f"[Merge] Dropping {n_dropped} rows with NaT timestamps "
                                    f"(pair-merge misalignment in {analysis_type} chunks)"
                                )
                                merged_df = merged_df[~nat_mask].reset_index(drop=True)
                        except Exception as sort_err:
                            self.logger.warning(f"[Merge] Defensive sort failed: {sort_err}")
                
                gc.collect()

                self.logger.info(
                    f"✅ [Merge] Merged to {len(merged_df)} rows, {len(merged_df.columns)} columns"
                )

            except Exception as merge_error:
                self.logger.error(f"❌ [Merge] DataFrame concatenation failed: {merge_error}")
                merged_df = original_df
                metadata_accumulator["merge_error"] = str(merge_error)

        else:
            if is_ml_prep:
                self.logger.info(
                    "[Merge] No DataFrames to merge (expected for ML prep — data is in sequences/labels/targets)"
                )
            else:
                self.logger.warning("[Merge] No valid DataFrames to merge - using original")
            merged_df = original_df

        # Build final result
        final_result = {
            "result_df": merged_df,
            "metadata": {
                "strategy": ProcessingStrategy.PARALLEL_CHUNKING.value,
                "analysis_type": analysis_type,
                "workers": self._worker_count,
                "chunks": self._chunk_count,
                "successful_chunks": n_successful,
                "failed_chunks": len(failed_chunks),
                "rows_processed": len(merged_df),
                "original_rows": len(original_df),
                "storage_mode": "streaming_db" if is_streaming else "standard",
                **metadata_accumulator,
            },
        }

        # Analysis-specific result processing.
        # STREAMING GUARD: For ML prep in streaming mode, chunks were already
        # persisted atomically by the callback during worker execution. Calling
        # _process_analysis_specific_results again with metadata stubs would be a
        # no-op but generates confusing "no sequences, skipping" warnings.
        # Only call for non-streaming or non-ML-prep analysis types.
        _is_ml_prep = analysis_type in ("ml_preparation", "ml_dataset_preparation")
        if not (is_streaming and _is_ml_prep):
            await self._process_analysis_specific_results(
                final_result,
                [r for r in results if isinstance(r, dict) and not r.get("error")],
                pm_instance=kwargs.get('pm_instance'),
                db_session=kwargs.get('db_session')
            )
        else:
            self.logger.info(
                f"[Merge] Skipping _process_analysis_specific_results for {analysis_type} "
                f"(streaming=True — data already stored by callback)"
            )

        del metadata_accumulator, failed_chunks
        gc.collect()

        self.logger.info(
            f"✅ [Merge] Complete for {analysis_type}: "
            f"{len(merged_df)} rows, {n_successful} successful, "
            f"{len(failed_chunks) if 'failed_chunks' in dir() else 0} failures"
            # simpler: track n_failed = len(failed_chunks) before del, log that
        )

        return final_result
    def _validate_chunk_dataframe(self, df: pd.DataFrame, chunk_id: str) -> Optional[str]:
        """
        Validate chunk DataFrame structure based on analysis type.
        
        Returns:
            None if valid, error message string if invalid
        """
        try:
            if df is None or len(df) == 0:
                return f"Chunk {chunk_id}: DataFrame is empty"
            
            # Common validation for all analysis types
            if df.isnull().all().any():
                null_cols = df.columns[df.isnull().all()].tolist()
                return f"Chunk {chunk_id}: Columns are entirely null: {null_cols}"
            
            # Analysis-specific validation
            if self.context.analysis_type in ["technical", "snr", "astronomical"]:
                # These require OHLCV columns (case-insensitive check)
                required_cols = {"open", "high", "low", "close", "volume"}
                col_lower_set = {col.lower() for col in df.columns}
                missing_cols = required_cols - col_lower_set
                if missing_cols:
                    return f"Chunk {chunk_id}: Missing OHLCV columns: {missing_cols}"
            
            elif self.context.analysis_type in ["ml_preparation", "ml_dataset_preparation"]:
                # ML prep should have feature columns
                if len(df.columns) < 5:  # Minimum expected columns
                    return f"Chunk {chunk_id}: Too few columns for ML prep: {len(df.columns)}"
            
            return None  # Valid
            
        except Exception as e:
            return f"Chunk {chunk_id}: Validation error: {e}"
    
    def _accumulate_chunk_metadata(self, accumulator: Dict[str, Any], chunk_result: Dict[str, Any]) -> None:
        """
        Accumulate analysis-specific metadata from chunk results.
        
        Handles different result structures for each analysis type.
        ✅ FIX: For ML prep, accumulate total_sequences from chunk metadata
        """
        try:
            # Special handling for ML prep metadata
            is_ml_prep = self.context.analysis_type in ("ml_preparation", "ml_dataset_preparation")
            chunk_metadata = chunk_result.get("metadata", {})
            
            if is_ml_prep and "total_sequences" in chunk_metadata:
                # Accumulate total_sequences from each chunk's metadata
                accumulator.setdefault("total_sequences", 0)
                accumulator["total_sequences"] += chunk_metadata.get("total_sequences", 0)
                self.logger.info(
                    f"[ML Prep Metadata] Accumulated {chunk_metadata['total_sequences']} sequences, "
                    f"total so far: {accumulator['total_sequences']}"
                )
            
            # Skip standard fields (but NOT metadata for ML prep)
            skip_fields = {"chunk_id", "global_start_idx", "error", "result_df"}
            if not is_ml_prep:
                skip_fields.add("metadata")
            
            for key, value in chunk_result.items():
                if key in skip_fields or value is None:
                    continue
                
                # Handle different data types
                if isinstance(value, list):
                    # Lists (signals, zones, etc.): extend
                    accumulator.setdefault(key, []).extend(value)
                    
                elif isinstance(value, dict):
                    # Skip metadata if we already handled it above (ML prep)
                    if key == "metadata" and is_ml_prep:
                        continue
                    
                    # Dicts (signal_counts, ml_dataset): merge
                    if key not in accumulator:
                        accumulator[key] = {}
                    
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float)):
                            # Numbers: sum
                            accumulator[key][sub_key] = accumulator[key].get(sub_key, 0) + sub_value
                        elif isinstance(sub_value, list):
                            # Nested lists: extend
                            accumulator[key].setdefault(sub_key, []).extend(sub_value)
                        else:
                            # Other: keep last
                            accumulator[key][sub_key] = sub_value
                            
                elif isinstance(value, (int, float)):
                    # Scalar numbers: collect for aggregation
                    accumulator.setdefault(key, []).append(value)
                    
                else:
                    # Other types: keep last
                    accumulator[key] = value
                    
        except Exception as e:
            self.logger.warning(f"⚠️ Metadata accumulation error: {e}")
    
    async def _process_analysis_specific_results(
        self, 
        final_result: Dict[str, Any], 
        chunk_results: List[Dict[str, Any]],
        pm_instance=None,  # PM instance for config
        db_session=None    # DB session for transactions
    ) -> None:
        """
        Process analysis-specific results and add to final result.
        
        Handles aggregation of analysis-specific data like signals, zones, ML datasets, etc.
        """
        try:
            analysis_type = self.context.analysis_type
            
            if analysis_type == "snr":
                self._process_snr_results(final_result, chunk_results)
            elif analysis_type == "technical":
                self._process_technical_results(final_result, chunk_results)
            elif analysis_type == "astronomical":
                self._process_astronomical_results(final_result, chunk_results)
            elif analysis_type in ["ml_preparation", "ml_dataset_preparation"]:
                await self._process_ml_prep_results(final_result, chunk_results, pm_instance, db_session)
                
        except Exception as e:
            self.logger.error(f"❌ Analysis-specific processing failed: {e}")
            final_result["metadata"]["processing_error"] = str(e)
    
    def _process_snr_results(self, final_result: Dict[str, Any], chunk_results: List[Dict[str, Any]]) -> None:
        """Process SNR-specific results: signals, zones, signal_counts."""
        all_signals = []
        all_zones = []
        total_signal_counts = {}
        
        for result in chunk_results:
            # Collect signals
            signals = result.get("signals", [])
            if signals:
                all_signals.extend(signals)
            
            # Collect zones
            zones = result.get("zones", [])
            if zones:
                all_zones.extend(zones)
            
            # Aggregate signal counts
            signal_counts = result.get("signal_counts", {})
            for signal_type, count in signal_counts.items():
                total_signal_counts[signal_type] = total_signal_counts.get(signal_type, 0) + count
        
        # Add to final result
        if all_signals:
            final_result["signals"] = all_signals
            final_result["total_signals"] = len(all_signals)
            
        if all_zones:
            final_result["zones"] = all_zones
            
        if total_signal_counts:
            final_result["signal_counts"] = total_signal_counts
            
        self.logger.info(f"[SNR] Aggregated {len(all_signals)} signals, {len(all_zones)} zones")
    
    def _process_technical_results(self, final_result: Dict[str, Any], chunk_results: List[Dict[str, Any]]) -> None:
        """Process Technical Analysis results: indicators, statistics."""
        # Technical analysis typically just enriches the DataFrame
        # Most results are in the result_df itself
        indicator_stats = {}
        
        for result in chunk_results:
            metadata = result.get("metadata", {})
            if "indicators_calculated" in metadata:
                for indicator, stats in metadata["indicators_calculated"].items():
                    if indicator not in indicator_stats:
                        indicator_stats[indicator] = stats
        
        if indicator_stats:
            final_result["metadata"]["indicators_calculated"] = indicator_stats
            
        self.logger.info(f"[Technical] Processed {len(indicator_stats)} indicator types")
    
    def _process_astronomical_results(self, final_result: Dict[str, Any], chunk_results: List[Dict[str, Any]]) -> None:
        """Process Astronomical Analysis results: planetary data, aspects."""
        # Astronomical analysis typically enriches the DataFrame with planetary positions
        aspect_counts = {}
        
        for result in chunk_results:
            metadata = result.get("metadata", {})
            if "aspects_calculated" in metadata:
                for aspect_type, count in metadata["aspects_calculated"].items():
                    aspect_counts[aspect_type] = aspect_counts.get(aspect_type, 0) + count
        
        if aspect_counts:
            final_result["metadata"]["aspects_calculated"] = aspect_counts
            
        self.logger.info(f"[Astronomical] Processed {sum(aspect_counts.values())} aspects")
    
    async def _process_ml_prep_results(
        self,
        final_result: Dict[str, Any],
        chunk_results: List[Dict[str, Any]],
        pm_instance=None,
        db_session=None,
        split_name: Optional[str] = None,
    ) -> None:
        """Thin shim — delegates to module-level _save_ml_prep_chunks.
        Kept for backward compat; Sequential path calls the module fn directly.
        """
        await _save_ml_prep_chunks(
            final_result, chunk_results,
            pm_instance=pm_instance,
            db_session=db_session,
            split_name=split_name,
        )

    def _distribute_chunks_among_workers(self, chunks: List[Any], n_workers: int) -> List[List[Any]]:
        """Distribute chunks round-robin among workers."""
        worker_chunk_assignments = [[] for _ in range(n_workers)]
        for i, chunk in enumerate(chunks):
            worker_id = i % n_workers
            worker_chunk_assignments[worker_id].append(chunk)
        return worker_chunk_assignments

    def _get_slice_context(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Build slice context for progress and workers."""
        raw_slice_context = {}
        if hasattr(self.context.config, "slice_context"):
            raw_slice_context = getattr(self.context.config, "slice_context", {})
        elif isinstance(self.context.config, dict):
            raw_slice_context = self.context.config.get("slice_context", {})
        
        # Ensure we have the total dataset rows for accurate progress calculation
        # When in slice mode, len(df) is the slice length, not total dataset length
        total_rows = self.context.total_dataset_rows
        if not total_rows or total_rows == 0:
            # Fallback: if no total_dataset_rows in context, use current df length
            # This happens in non-slice mode (direct parallel chunking)
            total_rows = len(df)
            self.logger.debug(f"[Context] Using current df length as total_dataset_rows: {total_rows}")
        else:
            self.logger.debug(f"[Context] Using context total_dataset_rows: {total_rows} (current df: {len(df)})")
            
        slice_context = {
            "slice_num": self.context.slice_num,
            "total_slices": self.context.total_slices,
            "slice_start": self.context.slice_start,
            "slice_end": self.context.slice_end,
            "total_dataset_rows": total_rows,
            "global_offset": self.context.global_offset,
            "connection_manager": None,
            "user_id": getattr(self.context, "user_id", "unknown"),
            # ML-specific context
            "global_scaler": self._serialize_scaler(getattr(self.context, "global_scaler", None)),
            "fit_scaler": kwargs.get("fit_scaler", True),
            "split_type": kwargs.get("split_type", "train"),
            "split_name": kwargs.get("split_name", "train"),
            "skip_scaling": kwargs.get("skip_scaling", False),  # : Pass skip_scaling flag
            "feature_cols": kwargs.get("feature_cols"),
            "columns_to_scale": kwargs.get("columns_to_scale"),
            "ml_prep_metadata": kwargs.get("ml_prep_metadata"),
            # FIX: Pass enriched target columns so workers can build the correct target_keys list
            "enriched_target_columns": kwargs.get("enriched_target_columns"),
        }
        
        if isinstance(raw_slice_context, dict):
            slice_context.update(raw_slice_context)
            slice_context["connection_manager"] = None
        return slice_context

    async def _progress_listener(self, progress_queue, slice_context, total_chunks):
        """Unified progress listener for parallel execution."""
        loop = asyncio.get_running_loop()
        
        # Log progress context for debugging
        total_dataset_rows = slice_context.get("total_dataset_rows", 0)
        slice_num = slice_context.get("slice_num", 0)
        total_slices = slice_context.get("total_slices", 1)
        slice_start = slice_context.get("slice_start", 0)
        slice_end = slice_context.get("slice_end", 0)
        
        self.logger.info(
            f"[Parallel] Progress listener started for task {self.context.task_id}: "
            f"{total_chunks} chunks, slice {slice_num + 1}/{total_slices}, "
            f"rows {slice_start}-{slice_end} of {total_dataset_rows} total"
        )
        
        chunk_progress_map: Dict[int, float] = {}
        
        while True:
            try:
                message = await loop.run_in_executor(None, progress_queue.get)
                if not message: continue
                
                action = message.get("action")
                if action == "stop": break
                
                if action in ["update", "progress"]:
                    data = message.get("data", message)
                    raw_progress = data.get("progress")
                    cid = data.get("chunk_id", 0)
                    
                    if raw_progress is not None:
                        chunk_progress_map[cid] = float(raw_progress)
                        avg_progress = sum(chunk_progress_map.values()) / max(len(chunk_progress_map), 1)
                        
                        # Scale progress based on slice context
                        scaled_prog = calculate_cumulative_progress(avg_progress, slice_context)
                        
                        # Update task store
                        if self.context.task_store:
                            self.context.task_store.update_task(self.context.task_id, progress=scaled_prog, message=data.get("message", "Processing..."))
                        
                        # FIX: Use avg_progress for broadcast, not completed-count formula
                        # Old: prog = (completed_chunks * 100) / total_chunks → stays 0 until a chunk finishes
                        # : avg_progress across all reporting chunks → shows real-time progress
                        completed_chunks = len([p for p in chunk_progress_map.values() if p >= 100])
                        prog = min(99.9, avg_progress)
                        
                        # Enhanced logging for progress debugging with row information
                        if completed_chunks % 5 == 0 or completed_chunks == total_chunks:  # Log every 5 chunks
                            slice_start = slice_context.get("slice_start", 0)
                            slice_end = slice_context.get("slice_end", 0)
                            rows_in_slice = slice_end - slice_start
                            estimated_rows_processed = slice_start + (rows_in_slice * avg_progress / 100)
                            
                            self.logger.debug(
                                f"[Progress] Chunk {cid}: {raw_progress:.1f}% | "
                                f"Completed: {completed_chunks}/{total_chunks} | "
                                f"Avg: {avg_progress:.1f}% | Scaled: {scaled_prog:.1f}% | "
                                f"Rows: ~{estimated_rows_processed:.0f}/{total_dataset_rows} | "
                                f"Broadcast: {prog:.1f}%"
                            )
                        
                        broadcast_payload = {
                            **data,
                            "type": "progress",
                            "task_id": self.context.task_id,
                            "progress": round(prog, 2),
                            "chunk_id": cid,
                            "total_chunks": total_chunks,
                            "completed_chunks": completed_chunks,
                            "slice_info": f"{slice_num + 1}/{total_slices}"
                        }
                        
                        if self.context.connection_manager:
                            await self.context.connection_manager.send_progress_update(
                                self.context.task_id,
                                broadcast_payload
                            )
            except Exception as e:
                self.logger.warning(f"Progress listener error: {e}")
                break
        
        self.logger.info(
            f"[Parallel] Progress listener stopped. Final: {len(chunk_progress_map)}/{total_chunks} chunks completed"
        )





# ============================================================================
# SLICE STREAMING STRATEGY
# ============================================================================

class SliceStreamingStrategy(ProcessingStrategyBase):
    """
    Process large datasets (>5K rows) using slice-based streaming.
    
    Architecture: Top-layer orchestrator that delegates to ParallelChunkingStrategy
    
    Flow:
    1. Split large DataFrame into slices (5K rows each)
    2. Process each slice using ParallelChunkingStrategy 
    3. Aggregate results from all slices
    4. Return merged final result
    
    This creates a nested processing hierarchy:
    SliceStreamingStrategy → ParallelChunkingStrategy → Workers
    """

    async def execute(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Execute slice-based processing by delegating to ParallelChunkingStrategy.
        
        Simple and clean: slice → delegate → aggregate.
        """
        self.logger.info(f"[SliceStreaming] Starting {self.context.analysis_type} on {len(df)} rows")
        
        # Memory monitoring at slice-streaming level
        if psutil:
            process = psutil.Process()
            mem_start = process.memory_info().rss / 1024 / 1024
            self.logger.info(f"[SliceStreaming] Memory at start: {mem_start:.1f} MB")
        else:
            process = None
            mem_start = None

        # Step 1: Create slices
        slices = self._create_slices(df)
        total_slices = len(slices)
        
        self.logger.info(f"[SliceStreaming] Created {total_slices} slices of ~{self.context.processing_config.slice_size} rows each")
        
        # Step 2: Process each slice using ParallelChunkingStrategy
        all_slice_results = []
        
        for slice_idx, (start_idx, end_idx, overlap_start_idx) in enumerate(slices):
            # Check for cancellation
            if self.context.task_store and self.context.task_store.is_cancelled(self.context.task_id):
                raise TaskCancelledException(f"Task {self.context.task_id} cancelled at slice {slice_idx + 1}/{total_slices}")

            # Memory check before slice
            if process:
                mem_before_slice = process.memory_info().rss / 1024 / 1024
                self.logger.info(f"[SliceStreaming] Slice {slice_idx + 1}/{total_slices} starting. Memory: {mem_before_slice:.1f} MB")

            # Extract slice DataFrame
            slice_df = df.iloc[overlap_start_idx:end_idx].copy()
            
            # Create slice-specific context
            slice_context = dataclasses.replace(
                self.context,
                slice_num=slice_idx,
                total_slices=total_slices,
                slice_start=start_idx,
                slice_end=end_idx,
                total_dataset_rows=len(df),
                global_offset=overlap_start_idx,
                # Preserve global_scaler from original context
                global_scaler=getattr(self.context, 'global_scaler', None),
            )
            
            # Delegate to ParallelChunkingStrategy for this slice
            parallel_strategy = ParallelChunkingStrategy(slice_context, self.logger)
            slice_result = await parallel_strategy.execute(slice_df, **kwargs)
            self.logger.info(f"[Processing Complete ] Slice {slice_idx + 1}/{total_slices} processed with {len(slice_result.get('result_df', []))} rows in result")

            # Immediately extract the canonical trimmed DataFrame and null out
            # result_df inside slice_result so the full enriched copy is freed BEFORE
            # the next slice is processed.  Previously all result_df objects accumulated
            # in all_slice_results until _aggregate_slice_results was called, giving a
            # memory peak of (N_slices × slice_result_df_size) + concat output.
            # Now peak is: original_df + 1 slice_result_df + growing trimmed_dfs list.
            if isinstance(slice_result.get("result_df"), pd.DataFrame):
                _slice_df_result = slice_result["result_df"]
                # Trim to canonical boundary using the original_df index labels
                _start_idx, _end_idx, _ = self._create_slices(df)[slice_idx]
                _start_label = df.index[_start_idx] if _start_idx < len(df) else df.index[0]
                _end_idx_c = min(_end_idx - 1, len(df) - 1)
                _end_label = df.index[_end_idx_c] if _end_idx_c >= 0 else df.index[0]
                _trimmed = _slice_df_result.loc[
                    (_slice_df_result.index >= _start_label) &
                    (_slice_df_result.index <= _end_label)
                ].copy()
                # Null out the large DataFrame in-place so the original is GC-eligible
                slice_result["result_df"] = None
                del _slice_df_result
                # Stash the lightweight trimmed copy as a sentinel key
                slice_result["_trimmed_df"] = _trimmed
                del _trimmed

            # Store result (result_df is now None; only _trimmed_df + metadata remain)
            all_slice_results.append(slice_result)

            # Memory cleanup after slice
            del slice_df
            gc.collect()
            
            if process:
                mem_after_slice = process.memory_info().rss / 1024 / 1024
                self.logger.info(f"[SliceStreaming] Slice {slice_idx + 1}/{total_slices} complete. Memory: {mem_after_slice:.1f} MB")

            # Send progress update
            progress_pct = int(((slice_idx + 1) / total_slices) * 100)
            await self._send_progress(
                progress=min(progress_pct, 99),
                message=f"Processed slice {slice_idx + 1}/{total_slices}",
                stage="slice_processing",
            )

        # Step 3: Aggregate all slice results
        self.logger.info(f"[SliceStreaming] Aggregating {total_slices} slice results")
        
        if process:
            mem_before_agg = process.memory_info().rss / 1024 / 1024
            self.logger.info(f"[SliceStreaming] Starting aggregation. Memory: {mem_before_agg:.1f} MB")

        final_result = self._aggregate_slice_results(all_slice_results, df)

        # Final memory report
        if process and mem_start is not None:
            mem_end = process.memory_info().rss / 1024 / 1024
            self.logger.info(
                f"[SliceStreaming] Complete. Total memory delta: +{mem_end - mem_start:.1f} MB "
                f"(start: {mem_start:.1f} MB → end: {mem_end:.1f} MB)"
            )

        return final_result

    def _create_slices(self, df: pd.DataFrame) -> List[Tuple[int, int, int]]:
        """
        Create slice boundaries with required context overlap.
        """
        if not self.context.processing_config:
            self.context.processing_config = ProcessingConfig()
        
        cfg = self.context.processing_config
        total_rows = len(df)
        slice_size = cfg.slice_size
        
        # FIX: Calculate required overlap based on analysis type
        # If overlap is too small, we get gaps because analysis functions drop edge rows.
        required_overlap = self._get_required_overlap()
        
        # Ensure factor-based overlap is at least as large as required context
        min_overlap = max(int(slice_size * cfg.slice_overlap_factor), required_overlap)
        
        self.logger.info(f"[SliceCreation] total={total_rows}, size={slice_size}, required_context={required_overlap}, actual_overlap={min_overlap}")

        slices = []
        total_slices = max(1, -(-total_rows // slice_size))
        
        for i in range(total_slices):
            start_idx = i * slice_size
            end_idx = min(start_idx + slice_size, total_rows)
            overlap_start_idx = max(0, start_idx - min_overlap)
            slices.append((start_idx, end_idx, overlap_start_idx))
        
        return slices

    def _get_required_overlap(self) -> int:
        """Determine required context rows based on analysis type."""
        atype = self.context.analysis_type
        
        if atype == "snr":
            config = getattr(self.context, 'config', None)
            if config:
                lookback = getattr(config, 'lookback_period', 100)
                lookforward = getattr(config, 'lookforward_period', 100)
                return lookback + lookforward + 10
            return 500 # Safe default for SNR
            
        if atype in ["ml_preparation", "ml_dataset_preparation"]:
            config = getattr(self.context, 'config', None)
            if config:
                seq_len = getattr(config, 'sequence_length', 60)
                pred_len = getattr(config, 'prediction_length', 7)
                return seq_len + pred_len + 10
            return 200 # Safe default for ML
            
        return 200 # General safe default for indicators

    def _aggregate_slice_results(self, slice_results: List[Dict[str, Any]], original_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Aggregate results from all slices into final result.

        Memory contract (FIX #2 + #3):
        - execute() pre-trims each result_df into _trimmed_df and nulls result_df
          BEFORE appending to all_slice_results, so we never accumulate N full
          enriched DataFrames simultaneously.
        - Here we consume _trimmed_df from each slice_result one at a time and
          immediately null it out, keeping peak memory to:
              original_df  +  1 trimmed slice  +  running concat output
          instead of the old:
              original_df  +  N result_dfs  +  N trimmed copies  +  concat output
        """
        if not slice_results:
            self.logger.warning("[SliceAggregation] No slice results to aggregate")
            return {
                "result_df": original_df,
                "metadata": {
                    "strategy": "slice_streaming",
                    "analysis_type": self.context.analysis_type,
                    "slices": 0,
                    "warning": "No slice results"
                }
            }

        self.logger.info(f"[SliceAggregation] Merging {len(slice_results)} slice results")

        # Detect index type once from first non-empty slice
        has_datetime_index: Optional[bool] = None
        metadata_accumulator: Dict[str, Any] = {}

        # accumulate DataFrames piecemeal, null source immediately 
        # We build combined_df incrementally instead of collecting all trimmed
        # pieces into a list first, eliminating the list-of-all-trimmed-dfs peak.
        combined_df: Optional[pd.DataFrame] = None

        slices_config = self._create_slices(original_df)

        for i, slice_result in enumerate(slice_results):
            if not isinstance(slice_result, dict):
                continue

            # 1. Retrieve the pre-trimmed DataFrame written by execute() (FIX #1).
            #    Fall back to on-the-fly trimming for callers that bypass execute()
            #    (e.g. unit tests that call _aggregate_slice_results directly).
            trimmed_df = slice_result.pop("_trimmed_df", None)

            if trimmed_df is None:
                # Fallback path: trim result_df now (covers legacy callers / tests)
                raw_df = slice_result.get("result_df")
                if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
                    if i < len(slices_config):
                        start_idx, end_idx, _ = slices_config[i]
                        start_label = original_df.index[start_idx] if start_idx < len(original_df) else original_df.index[0]
                        end_idx_c   = min(end_idx - 1, len(original_df) - 1)
                        end_label   = original_df.index[end_idx_c] if end_idx_c >= 0 else original_df.index[0]
                        trimmed_df  = raw_df.loc[
                            (raw_df.index >= start_label) & (raw_df.index <= end_label)
                        ].copy()
                    else:
                        trimmed_df = raw_df.copy()
                # ── FIX #3: null the source reference immediately ──
                slice_result["result_df"] = None

            if isinstance(trimmed_df, pd.DataFrame) and not trimmed_df.empty:
                if has_datetime_index is None:
                    has_datetime_index = isinstance(trimmed_df.index, pd.DatetimeIndex)

                # Incremental concat: append one slice at a time so peak memory
                # is (combined_so_far + one_trimmed_slice) not (all_trimmed_slices).
                if combined_df is None:
                    combined_df = trimmed_df
                else:
                    combined_df = pd.concat([combined_df, trimmed_df], ignore_index=False, sort=False, copy=False)

                # ── FIX #3 (cont.): release trimmed slice immediately ──
                del trimmed_df
                gc.collect()

            # 2. Accumulate metadata / signals / zones (no DataFrame involved)
            slice_metadata = slice_result.get("metadata", {})
            self._accumulate_slice_metadata(metadata_accumulator, slice_result, slice_metadata)
            self._accumulate_analysis_results(metadata_accumulator, slice_result)

        # ── Post-loop sort & dedup (now done on the already-incremental combined_df) ──
        if combined_df is not None and not combined_df.empty:
            try:
                if has_datetime_index:
                    combined_df = combined_df.sort_index()
                    original_len = len(combined_df)
                    duplicates = combined_df.index.duplicated(keep='first')
                    dup_count = duplicates.sum()
                    if dup_count > 0:
                        self.logger.info(f"[SliceAggregation] Removing {dup_count} duplicate timestamps from overlaps")
                        combined_df = combined_df.loc[~duplicates].copy()
                        self.logger.info(f"[SliceAggregation] Removed {original_len - len(combined_df)} rows, {len(combined_df)} remain")
                    else:
                        self.logger.info(f"[SliceAggregation] No duplicate timestamps found")
                    self.logger.info(f"[SliceAggregation] ✅ Final result sorted chronologically: {len(combined_df)} rows")
                else:
                    cfg = self.context.processing_config
                    if cfg and getattr(cfg, 'slice_overlap_factor', 0) > 0:
                        combined_df = combined_df.drop_duplicates(keep="first")
                        self.logger.info(f"[SliceAggregation] Removed overlaps: {len(combined_df)} final rows")

                    time_col = next((c for c in combined_df.columns if c.lower() == 'time'), None)
                    if time_col:
                        try:
                            if not pd.api.types.is_datetime64_any_dtype(combined_df[time_col]):
                                if pd.api.types.is_numeric_dtype(combined_df[time_col]):
                                    combined_df[time_col] = pd.to_datetime(combined_df[time_col], unit='s', errors='coerce')
                                else:
                                    combined_df[time_col] = pd.to_datetime(combined_df[time_col], errors='coerce')
                            combined_df = combined_df.sort_values(time_col)
                            self.logger.info(f"[SliceAggregation] Sorted result by column '{time_col}'")
                        except Exception as sort_err:
                            self.logger.warning(f"[SliceAggregation] Failed to sort by {time_col}: {sort_err}")
            except Exception as e:
                self.logger.error(f"❌ [SliceAggregation] Post-loop sort/dedup failed: {e}")
                metadata_accumulator["aggregation_error"] = str(e)
        else:
            self.logger.warning("[SliceAggregation] No DataFrames to merge - using original")
            combined_df = original_df

        # Build final result
        final_result = {
            "result_df": combined_df,
            "metadata": {
                "strategy": "slice_streaming",
                "analysis_type": self.context.analysis_type,
                "slices": len(slice_results),
                "rows_processed": len(combined_df),
                "original_rows": len(original_df),
                **metadata_accumulator
            }
        }
        
        # Add analysis-specific aggregated results to top level
        self._finalize_analysis_results(final_result, metadata_accumulator)
        
        # FIX: Log BEFORE deleting slice_results
        self.logger.info(
            f"✅ [SliceAggregation] Complete: {len(combined_df)} rows from {len(slice_results)} slices"
        )
        
        # Final cleanup
        del slice_results
        del metadata_accumulator
        gc.collect()

        return final_result
    
    def _accumulate_slice_metadata(self, accumulator: Dict[str, Any], slice_result: Dict[str, Any], slice_metadata: Dict[str, Any]) -> None:
        """Accumulate metadata from slice results."""
        # Accumulate numeric metadata
        for key in ["workers", "chunks", "successful_chunks", "failed_chunks"]:
            if key in slice_metadata:
                accumulator.setdefault(f"total_{key}", 0)
                accumulator[f"total_{key}"] += slice_metadata[key]
        
        # Track slice-level info
        accumulator.setdefault("slice_info", []).append({
            "slice_num": slice_metadata.get("slice_num", len(accumulator.get("slice_info", []))),
            "rows": slice_metadata.get("rows_processed", 0),
            "workers": slice_metadata.get("workers", 0),
            "chunks": slice_metadata.get("chunks", 0)
        })
    
    def _accumulate_analysis_results(self, accumulator: Dict[str, Any], slice_result: Dict[str, Any]) -> None:
        """Accumulate analysis-specific results from slices."""
        analysis_type = self.context.analysis_type
        
        if analysis_type == "snr":
            # FIX: Only keep signals and zones that fall within the slice's "owned" range
            # to prevent duplicates from the overlap context.
            start_idx = slice_result.get("metadata", {}).get("slice_start", 0)
            end_idx = slice_result.get("metadata", {}).get("slice_end", 999999999)
            
            # 1. Filter and extend signals
            signals = slice_result.get("signals", [])
            if signals:
                unique_signals = [
                    s for s in signals 
                    if start_idx <= s.get("index", 0) < end_idx
                ]
                accumulator.setdefault("signals", []).extend(unique_signals)
            
            # 2. Filter and extend zones
            zones = slice_result.get("zones", [])
            if zones:
                # Zones often have a unique ID at index 0 (from detect_snr_levels)
                # and metadata at index 5. Only keep zones "owned" by this slice.
                unique_zones = []
                for z in zones:
                    has_index = len(z) > 5 and isinstance(z[5], dict) and "index" in z[5]
                    if not has_index or (start_idx <= z[5]["index"] < end_idx):
                        unique_zones.append(z)
                
                # Fallback: simple deduplication by zone ID
                existing_zone_ids = {z[0] for z in accumulator.get("zones", [])}
                for z in unique_zones:
                    if z[0] not in existing_zone_ids:
                        accumulator.setdefault("zones", []).append(z)
                        existing_zone_ids.add(z[0])
            
            if "signal_counts" in slice_result:
                if "signal_counts" not in accumulator:
                    accumulator["signal_counts"] = {}
                for signal_type, count in slice_result["signal_counts"].items():
                    # We should really re-calculate counts from unique_signals, but this is an approximation
                    accumulator["signal_counts"][signal_type] = accumulator["signal_counts"].get(signal_type, 0) + count
        
        elif analysis_type in ["ml_preparation", "ml_dataset_preparation"]:
            # ML Prep: sequences, labels, targets
            for key in ["sequences", "labels", "targets"]:
                if key in slice_result:
                    value = slice_result[key]
                    # Handle numpy arrays and lists differently
                    if isinstance(value, np.ndarray):
                        if value.size > 0:  # Check if array is non-empty
                            accumulator.setdefault(key, []).append(value)
                    elif value:  # For lists/dicts, use truthiness
                        accumulator.setdefault(key, []).extend(value if isinstance(value, list) else [value])
        
        # Handle ML splits (train/validation/test)
        for split_key in ["train", "validation", "test"]:
            if split_key in slice_result:
                accumulator.setdefault(f"ml_splits_{split_key}", []).append(slice_result[split_key])
    
    def _finalize_analysis_results(self, final_result: Dict[str, Any], accumulator: Dict[str, Any]) -> None:
        """Move analysis-specific results from metadata to top level."""
        analysis_type = self.context.analysis_type
        
        if analysis_type == "snr":
            # Move SNR results to top level
            for key in ["signals", "zones", "signal_counts"]:
                if key in accumulator:
                    final_result[key] = accumulator[key]
                    
                    # FIX: Ensure signals are sorted by index
                    if key == "signals" and isinstance(final_result[key], list):
                        try:
                            final_result[key].sort(key=lambda s: s.get("index", 0))
                            final_result["total_signals"] = len(final_result[key])
                            self.logger.info(f"[SliceAggregation] Sorted {len(final_result[key])} signals by index")
                        except Exception as sig_sort_err:
                            self.logger.warning(f"[SliceAggregation] Signal sorting failed: {sig_sort_err}")
        
        elif analysis_type in ["ml_preparation", "ml_dataset_preparation"]:
            # Move ML results to top level and CONCATENATE numpy arrays
            for key in ["sequences", "labels"]:
                if key in accumulator:
                    values = accumulator[key]
                    if values and isinstance(values[0], np.ndarray):
                        final_result[key] = np.concatenate(values, axis=0)
                    else:
                        final_result[key] = values
            
            # Targets is a dict of lists of arrays
            if "targets" in accumulator:
                final_result["targets"] = self._merge_ml_split_dicts(accumulator["targets"])
            
            # Handle ML splits
            for split_key in ["train", "validation", "test"]:
                split_list_key = f"ml_splits_{split_key}"
                if split_list_key in accumulator:
                    final_result[split_key] = self._merge_ml_split_dicts(accumulator[split_list_key])
