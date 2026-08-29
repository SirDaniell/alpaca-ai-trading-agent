"""
Shared session data loading utility.

Provides a single authoritative function for loading the most recently enriched
dataset for a given session ID. The priority chain ensures each analysis step
always builds on the richest available data stored in the DB.
"""

import gc
import logging
import hashlib
import pickle
import json
import zlib
import base64
import uuid
from sqlalchemy import select, and_, case, update, func, exc, literal_column
from typing import Union, List, Optional, Dict, Any, Callable

from app.database.models import SessionStepResult, DataSession, MLDataset, MLDatasetChunk, TrainedModelForAnalysis
from app.core.data.serializers import deserialize_data, serialize_data, to_serializable
from app.core.services.data_utils import restore_numeric_types, clean_dataframe
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy.dialects.postgresql import insert as pg_insert


from app.core.data.session_dataset_registry import CompressionHandler
from app.core.data.serializers import serialize_data, deserialize_data
import asyncio

logger = logging.getLogger(__name__)

# Global dictionary to track per-session concurrency locks for data loading
_SESSION_LOAD_LOCKS: Dict[str, asyncio.Lock] = {}

def _get_session_load_lock(session_id: str) -> asyncio.Lock:
    """Get or create an asyncio Lock for a specific session ID to prevent concurrent DB fetches."""
    if session_id not in _SESSION_LOAD_LOCKS:
        _SESSION_LOAD_LOCKS[session_id] = asyncio.Lock()
    return _SESSION_LOAD_LOCKS[session_id]



# ============================================================================
# CUSTOM EXCEPTIONS FOR DATA INTEGRITY
# ============================================================================

class SessionResultNotFound(Exception):
    """Result not found in database"""
    pass


class DataIntegrityError(Exception):
    """Data failed integrity checks (checksum mismatch, deserialization error)"""
    pass


class DataRetrievalError(Exception):
    """General error retrieving or validating data"""
    pass


class DataValidationError(Exception):
    """Raised when data structure validation fails (Issue #2)"""
    pass


class AtomicOperationError(Exception):
    """Raised when store+mark transaction fails (Issue #3)"""
    pass


# ============================================================================
# DATA INTEGRITY VERIFICATION
# ============================================================================

def verify_data_integrity(
    data_bytes: bytes,
    expected_hash: Optional[str] = None,
    step_name: str = "unknown"
) -> bool:
    """
    Verify data integrity using SHA-256 checksum.
    
    Args:
        data_bytes: Raw bytes after decompression
        expected_hash: Expected SHA-256 hash (if available)
        step_name: For logging context
    
    Returns:
        True if valid or no hash provided
        
    Raises:
        DataIntegrityError: If hash mismatch detected
    """
    if expected_hash is None:
        logger.debug(f"No hash provided for {step_name} - skipping verification")
        return True
    
    actual_hash = hashlib.sha256(data_bytes).hexdigest()
    
    if actual_hash != expected_hash:
        logger.error(
            f"❌ Hash mismatch for {step_name}: "
            f"expected {expected_hash}, got {actual_hash}"
        )
        raise DataIntegrityError(
            f"Data integrity check failed for {step_name}. "
            f"Data may be corrupted in database."
        )
    
    logger.debug(f"✅ Hash verified for {step_name}: {actual_hash[:8]}...")
    return True


# ============================================================================
# DATA VALIDATION (BUG FIX #5)
# ============================================================================

def validate_data_structure(
    data: Any,
    step_name: str = "unknown",
    min_rows: int = 1,
    expected_columns: Optional[List[str]] = None
) -> bool:
    """
    Validate data structure after deserialization from database.
    Prevents corrupted or malformed data from propagating downstream.
    
    Args:
        data: Data to validate (should be list of dicts or DataFrame)
        step_name: Step name for logging context
        min_rows: Minimum required rows
        expected_columns: If provided, check that all columns exist
        
    Returns:
        True if valid
        
    Raises:
        DataValidationError: If validation fails
    """
    # Check if data exists
    if data is None:
        raise DataValidationError(f"Data is None for {step_name}")
    
    # Convert to list of dicts if DataFrame
    if isinstance(data, pd.DataFrame):
        if data.empty and min_rows > 0:
            raise DataValidationError(
                f"{step_name}: DataFrame is empty (expected at least {min_rows} rows)"
            )
        data_rows = data.to_dict(orient='records')
    elif isinstance(data, list):
        data_rows = data
    elif isinstance(data, dict) and 'data' in data:
        data_rows = data['data']
    else:
        raise DataValidationError(
            f"{step_name}: Data is not a list, DataFrame, or dict with 'data' key. "
            f"Got type: {type(data)}"
        )
    
    # Check row count
    if len(data_rows) < min_rows:
        raise DataValidationError(
            f"{step_name}: Expected at least {min_rows} rows, got {len(data_rows)}"
        )
    
    # Check column structure (first row only)
    if data_rows and isinstance(data_rows[0], dict):
        columns = set(data_rows[0].keys())
        
        # Required base columns (minimal OHLCV)
        # Use case-insensitive check because system standardizes to uppercase 'Open', 'High', etc.
        required = {'open', 'high', 'low', 'close'}
        available_lower = {c.lower() for c in columns}
        missing = required - available_lower
        
        if missing:
            logger.warning(
                f"{step_name}: Missing required OHLC columns: {missing}. "
                f"Available: {columns}"
            )
        
        # Check specific expected columns if provided
        if expected_columns:
            missing_expected = set(expected_columns) - columns
            if missing_expected:
                raise DataValidationError(
                    f"{step_name}: Missing expected columns: {missing_expected}"
                )
    
    logger.info(f"✅ Data validation passed for {step_name}: {len(data_rows)} rows")
    return True


# Priority order: most-enriched step first.
# Each step in the pipeline adds columns on top of the previous step's output,
# so we always want the highest step that has been completed.
SESSION_STEP_PRIORITY = [
    'astronomical_analysis',  # Adds planetary / ephemeris columns
    'snr_analysis',            # Adds SNR signal columns
    'signal_generation',      # Alias / legacy step name for snr_analysis
    'technical_analysis',      # Adds indicator columns
    'footprint_ingestion',     # Pre-ingested tick features
    'currency_indices',        # Adds currency strength index columns (Dollar, Euro, JPY, etc.)
    'data_source',             # Raw OHLCV as stored by DataSource step
]

# SNR and Astronomical enrich independently and may run in either order.
# When loading the priority chain (step_name=None), merge all available peer
# outputs on Time so columns from both steps are always present.
PEER_MERGE_STEPS = frozenset({
    'astronomical_analysis',
    'snr_analysis',
    'signal_generation',
})

PEER_MERGE_STEP_ORDER = [
    'snr_analysis',
    'signal_generation',       # legacy alias — skipped when snr_analysis exists
    'astronomical_analysis',
]

ENRICHED_STEPS = frozenset({
    'astronomical_analysis',
    'snr_analysis',
    'technical_analysis',
    'currency_indices',
})
MIN_ENRICHED_COLS = 11


def _normalize_time_column_for_merge(df: pd.DataFrame, time_col: str = 'Time') -> pd.DataFrame:
    """Normalize time column dtype for peer merging."""
    if time_col not in df.columns:
        return df

    if pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df = df.copy()
        df[time_col] = (df[time_col].astype('int64') // 10**9).astype('int64')
        return df

    if pd.api.types.is_integer_dtype(df[time_col]):
        return df

    if pd.api.types.is_float_dtype(df[time_col]):
        df = df.copy()
        df[time_col] = df[time_col].astype('int64')
        return df

    try:
        parsed = pd.to_datetime(df[time_col], errors='coerce')
        if parsed.isna().all():
            logger.warning(
                f"Peer merge time normalization: '{time_col}' values could not be parsed to datetime"
            )
            return df

        df = df.copy()
        df[time_col] = (parsed.astype('int64') // 10**9).astype('int64')
    except Exception as ex:
        logger.warning(
            f"Peer merge time normalization failed for '{time_col}': {ex}"
        )
    return df


def _merge_peer_dataframes_on_time(
    frames: List[tuple],
) -> pd.DataFrame:
    """
    Merge peer step DataFrames on Time.

    frames: list of (step_name, DataFrame, stored_at) sorted by stored_at ascending
    so later runs win for overlapping columns.
    """
    if not frames:
        raise ValueError("frames must not be empty")
    if len(frames) == 1:
        return frames[0][1].copy()

    sorted_frames = sorted(frames, key=lambda x: x[2] or datetime.min)
    normalized_frames = [
        (name, _normalize_time_column_for_merge(df, 'Time'), stored_at)
        for name, df, stored_at in sorted_frames
    ]

    result = normalized_frames[0][1].copy()
    time_col = 'Time'

    if time_col not in result.columns:
        raise ValueError(f"Peer merge requires '{time_col}' column in base frame")

    for step_name, df, _stored_at in normalized_frames[1:]:
        if time_col not in df.columns:
            logger.warning(
                f"Peer step '{step_name}' missing '{time_col}' column — skipping merge"
            )
            continue

        # If a later peer already contains all existing columns and has at least
        # as many rows, it can be used directly without outer merging.
        current_cols = set(result.columns) - {time_col}
        incoming_cols = set(df.columns) - {time_col}
        if current_cols <= incoming_cols and len(df) >= len(result):
            logger.info(
                f"Peer step '{step_name}' contains all current columns and is newer; "
                f"using '{step_name}' result directly"
            )
            result = df.copy()
            continue

        same_shape = (
            len(df) == len(result)
            and df[time_col].equals(result[time_col])
        )
        if same_shape:
            for col in df.columns:
                if col != time_col:
                    result[col] = df[col].values
            continue

        overlap = [c for c in df.columns if c in result.columns and c != time_col]
        new_cols = [c for c in df.columns if c not in result.columns and c != time_col]
        merge_df = df[[time_col] + new_cols + overlap].copy()
        result = result.merge(merge_df, on=time_col, how='outer', suffixes=('', '__peer'))
        for col in overlap:
            peer_col = f"{col}__peer"
            if peer_col in result.columns:
                result[col] = result[peer_col].combine_first(result[col])
                result.drop(columns=[peer_col], inplace=True)
        result = result.sort_values(time_col).reset_index(drop=True)
        del merge_df

    from app.core.services.data_utils import normalize_dataframe_columns
    result = normalize_dataframe_columns(result)
    result = restore_numeric_types(result)
    return result


async def _load_and_merge_peer_steps(
    session_id: str,
    db,
    task_id: str,
    exclude_steps: set,
    _locked: bool = True,
    as_dataframe: bool = False,
) -> Optional[Union[List[Dict[str, Any]], pd.DataFrame]]:
    """
    Load SNR and Astronomical peer steps (respecting exclude_steps) and merge
    their columns on Time. Returns None when no peer step data exists.
    """
    loaded: List[tuple] = []
    snr_loaded = False
    # Track the most recent stored_at seen so far across loaded steps.
    # The pipeline runs: astro → snr (astro enriches, snr further enriches and stores the
    # merged result). If snr was stored AFTER astro, the snr DB result already contains all
    # astro columns. Loading astro separately and outer-merging would cause:
    #   - row-count mismatch (snr has more rows if rows were added in the snr step)
    #   - ~600 MB wasted memory reassembling chunks that are already in snr
    # Guard: after loading each peer step, compare its stored_at against the most recently
    # loaded step. If this step's stored_at is strictly OLDER, it was already incorporated
    # into the later step's result — skip it.
    latest_stored_at = None

    for step in PEER_MERGE_STEP_ORDER:
        if step in exclude_steps:
            continue
        if step == 'signal_generation' and snr_loaded:
            continue

        # Read stored_at BEFORE loading data so we can skip the expensive DB reassembly
        step_stored_at = None
        try:
            stmt = select(SessionStepResult.stored_at).where(
                and_(
                    SessionStepResult.session_id == session_id,
                    SessionStepResult.step_name == step,
                )
            )
            row = (await db.execute(stmt)).scalar_one_or_none()
            step_stored_at = row
        except Exception as ts_err:
            logger.debug(f"Task {task_id}: Could not read stored_at for '{step}': {ts_err}")

        # ── Pipeline-incorporation guard ──────────────────────────────────────
        # If a later step (already in `loaded`) was stored AFTER this step, the later
        # step's result already incorporates this step's columns. Skip to avoid:
        #   (a) double memory cost from re-assembling large chunked steps
        #   (b) row-count mismatches on outer-merge
        if latest_stored_at is not None and step_stored_at is not None:
            if step_stored_at < latest_stored_at:
                logger.info(
                    f"Task {task_id}: ⏭️  Skipping peer step '{step}' — its stored_at "
                    f"({step_stored_at}) is older than the most recently loaded step "
                    f"({latest_stored_at}), meaning its columns are already incorporated "
                    f"into the later pipeline result."
                )
                continue

        step_df = await _get_latest_session_data_impl(
            session_id=session_id,
            db=db,
            task_id=task_id,
            exclude_steps=set(),
            step_name=step,
            _locked=_locked,
            as_dataframe=True,  # ALWAYS get DataFrame for internal peer merging to prevent dict conversion OOM
        )
        if step_df is None or (isinstance(step_df, pd.DataFrame) and step_df.empty):
            continue

        if not isinstance(step_df, pd.DataFrame):
            step_df = pd.DataFrame(step_df)

        if step in ('snr_analysis', 'signal_generation'):
            snr_loaded = True

        # Update latest known stored_at
        if step_stored_at is not None:
            if latest_stored_at is None or step_stored_at > latest_stored_at:
                latest_stored_at = step_stored_at

        loaded.append((step, step_df, step_stored_at))

    if not loaded:
        return None

    if len(loaded) == 1:
        merged_df = loaded[0][1]
        source_desc = loaded[0][0]
    else:
        merged_df = _merge_peer_dataframes_on_time(loaded)
        source_desc = '+'.join(entry[0] for entry in loaded)

    # Clean up intermediate peer step lists
    del loaded
    gc.collect()

    if len(merged_df.columns) < MIN_ENRICHED_COLS:
        logger.warning(
            f"Task {task_id}: Peer-merge result has only {len(merged_df.columns)} columns "
            f"(expected >{MIN_ENRICHED_COLS}) — falling through to priority chain"
        )
        return None

    if as_dataframe:
        logger.info(
            f"Task {task_id}: ✅ Peer-merge from [{source_desc}] (as_dataframe=True): "
            f"{len(merged_df)} rows × {len(merged_df.columns)} cols"
        )
        return merged_df

    result_records = merged_df.to_dict(orient="records")
    try:
        validate_data_structure(
            result_records,
            step_name=f"peer_merge({source_desc})",
            min_rows=1,
        )
    except DataValidationError as val_err:
        logger.error(f"Peer-merge validation failed: {val_err}")
        raise

    logger.info(
        f"Task {task_id}: ✅ Peer-merge from [{source_desc}]: "
        f"{len(merged_df)} rows × {len(merged_df.columns)} cols"
    )
    return result_records


async def get_latest_session_data(
    session_id: str,
    db,
    task_id: str = "unknown",
    step_name: Optional[str] = None,
    as_dataframe: bool = False
):
    """
    Load data for a given session.
    
    If step_name is provided, load THAT SPECIFIC STEP ONLY.
    If step_name is None, walk SESSION_STEP_PRIORITY from most-enriched to least-enriched.
    Final fallback: DataSession.raw_data.

    Args:
        session_id: The session UUID to load data for.
        db: An async SQLAlchemy session (AsyncSession).
        task_id: Used only for log messages.
        step_name: If provided, load ONLY this step (e.g., 'ml_preparation', 'astronomical_analysis').
                   If None, load most-enriched step by priority.
        as_dataframe: If True, return pd.DataFrame directly (bypassing dict conversion to prevent OOM).

    Returns:
        A list of dicts (rows) or pd.DataFrame representing the dataset, or None if not found.
    """
    return await _get_latest_session_data_impl(
        session_id, db, task_id, exclude_steps=set(), step_name=step_name, as_dataframe=as_dataframe
    )


async def get_latest_session_data_excluding_step(
    session_id: str, 
    db, 
    exclude_step: str,
    task_id: str = "unknown",
    as_dataframe: bool = False
):
    """
    PHASE 16: Load most enriched data, but SKIP a specific step.
    
    Used by mutation steps (SNR, Technical, Astro) to prevent loading their own 
    previous output and causing data corruption on re-runs.
    
    Args:
        session_id: The session UUID
        db: Async SQLAlchemy session
        exclude_step: Step name to skip (e.g., 'snr_analysis', 'technical_analysis')
        task_id: For logging
        as_dataframe: If True, return pd.DataFrame directly (bypassing dict conversion to prevent OOM)
        
    Returns:
        Most enriched data EXCLUDING the specified step
    """
    return await _get_latest_session_data_impl(
        session_id, db, task_id, exclude_steps={exclude_step}, as_dataframe=as_dataframe
    )


async def _get_latest_session_data_impl(
    session_id: str, 
    db, 
    task_id: str,
    exclude_steps: set,
    step_name: Optional[str] = None,
    _locked: bool = False,
    as_dataframe: bool = False
):
    """
    Internal implementation of data loading with optional step filtering.
    
    Args:
        exclude_steps: Set of step names to skip (e.g., {'snr_analysis'})
        step_name: If provided, load ONLY this specific step (overrides priority chain)
        as_dataframe: If True, return pd.DataFrame directly
    """
    if not _locked:
        lock = _get_session_load_lock(session_id)
        async with lock:
            return await _get_latest_session_data_impl(
                session_id, db, task_id, exclude_steps, step_name, _locked=True, as_dataframe=as_dataframe
            )

    # If specific step_name is requested, look for THAT STEP ONLY
    if step_name:
        steps_to_check = [step_name]
        logger.info(f"Task {task_id}: Looking for specific step: '{step_name}'")
    else:
        # SNR ↔ Astronomical may run in any order — merge peer outputs first.
        peer_merged = await _load_and_merge_peer_steps(
            session_id, db, task_id, exclude_steps, _locked=_locked, as_dataframe=as_dataframe
        )
        if peer_merged is not None:
            return peer_merged

        # Otherwise, walk through the priority chain
        steps_to_check = SESSION_STEP_PRIORITY
        if peer_merged is not None:
            return peer_merged

        # Otherwise, walk through the priority chain
        steps_to_check = SESSION_STEP_PRIORITY
    
    for step in steps_to_check:
        # PHASE 16: Skip excluded steps to prevent loading own output
        if step in exclude_steps:
            logger.debug(f"Task {task_id}: Skipping step '{step}' (excluded by caller)")
            continue
        
        try:
            stmt = select(SessionStepResult).where(
                and_(
                    SessionStepResult.session_id == session_id,
                    SessionStepResult.step_name == step
                )
            )
            result = await db.execute(stmt)
            step_result = result.scalar_one_or_none()

            if step_result and (step_result.result_data or step_result.result_data_v2):
                logger.info(f"Task {task_id}: Found enriched data in step '{step}'")
            
                # PRIORITY 1 FIX: Try JSONB first (new format), fall back to pickle
                if step_result.is_using_jsonb and step_result.result_data_v2 is not None:
                    # New JSONB format (already dict from PostgreSQL)
                    logger.info(f"📦 Loading JSONB data for '{step}'")
                    raw_data = step_result.result_data_v2
                
                    # Verify hash if available
                    if step_result.result_hash:
                        try:
                            # Use json.dumps with sort_keys=True (MUST MATCH storage hash computation)
                            # Never use str() - always json.dumps for consistency
                            json_str = json.dumps(raw_data, sort_keys=True)
                            retrieved_hash = hashlib.sha256(json_str.encode()).hexdigest()
                            logger.debug(f"✅ RETRIEVAL HASH (JSONB): {retrieved_hash[:16]}...")
                            if retrieved_hash != step_result.result_hash:
                                logger.warning(f"⚠️ JSONB Hash mismatch for '{step}': expected {step_result.result_hash[:8]}..., got {retrieved_hash[:8]}... (Data may have been modified)")
                        except Exception as hash_err:
                            logger.warning(f"⚠️ Could not verify JSONB hash: {hash_err}")
                        
                elif step_result.result_data is not None:
                    # Old pickle format (fallback)
                    logger.info(f"💾 Loading Pickle data (legacy) for '{step}'")
                
                    # Verify hash if available (FIX #1.5)
                    if step_result.result_hash:
                        try:
                            binary_data = base64.b64decode(step_result.result_data)
                            retrieved_hash = hashlib.sha256(binary_data).hexdigest()
                            logger.debug(f"✅ RETRIEVAL HASH: {retrieved_hash[:16]}... for step '{step}'")
                            logger.debug(f"   Expected hash:   {step_result.result_hash[:16]}...")
                        
                            if retrieved_hash != step_result.result_hash:
                                logger.warning(
                                    f"⚠️ Hash mismatch for {session_id}/{step} (may be benign): "
                                    f"expected {step_result.result_hash[:8]}..., got {retrieved_hash[:8]}... "
                                    f"(Data size: {len(binary_data)} bytes)"
                                )
                            else:
                                logger.debug(f"✅ Hash verified for step '{step}'")
                        except Exception as hash_check_err:
                            logger.warning(f"⚠️ Could not verify hash for '{step}': {hash_check_err}")
                
                    # Deserialize pickle data
                    raw_data = deserialize_data(step_result.result_data, step_result.is_compressed)
                else:
                    # No data in either format
                    continue

                # Standard result format: {data: [...rows...], metadata: {...}}
                if isinstance(raw_data, dict):
                    logger.info(f"Task {task_id}: Retrieved data structure for step '{step}': {type(raw_data)}")

                    # ── CHUNKED POINTER: PM used chunked storage (large DataFrame) ──────────
                    # When a DataFrame exceeds ~50 MB the PM calls store_session_step_result_chunked(),
                    # which writes the data across `step_name_0`, `step_name_1`, … rows and stores
                    # a lightweight pointer dict `{_is_chunked: True, num_chunks: N, …}` as the
                    # main `step_name` row.  Without this block the loader returns the pointer dict
                    # to the caller, which then fails (no OHLCV columns, wrong type, etc.).
                    if raw_data.get('_is_chunked'):
                        num_chunks = raw_data.get('num_chunks', 0)
                        total_rows = raw_data.get('total_rows', '?')
                        logger.info(
                            f"Task {task_id}: Detected chunked storage for '{step}' "
                            f"({num_chunks} chunks, {total_rows} rows) — reassembling from DB…"
                        )
                        chunk_dfs: list = []
                        for chunk_idx in range(num_chunks):
                            chunk_step = f"{step}_{chunk_idx}"
                            try:
                                chunk_stmt = select(SessionStepResult).where(
                                    and_(
                                        SessionStepResult.session_id == session_id,
                                        SessionStepResult.step_name == chunk_step,
                                    )
                                )
                                chunk_result = await db.execute(chunk_stmt)
                                chunk_row = chunk_result.scalar_one_or_none()
                                if chunk_row is None:
                                    logger.warning(
                                        f"Task {task_id}: Chunk '{chunk_step}' missing — skipping"
                                    )
                                    continue
                                raw_res = chunk_row.result_data
                                is_comp = chunk_row.is_compressed
                                raw_v2 = chunk_row.result_data_v2
                                del chunk_row, chunk_result

                                if raw_res is not None:
                                    chunk_data = deserialize_data(
                                        raw_res, is_comp
                                    )
                                    del raw_res
                                elif raw_v2 is not None:
                                    chunk_data = raw_v2
                                    del raw_v2
                                else:
                                    logger.warning(f"Task {task_id}: Chunk '{chunk_step}' has no data")
                                    continue

                                # Each chunk may be a DataFrame or a list of dicts
                                if isinstance(chunk_data, pd.DataFrame):
                                    chunk_dfs.append(chunk_data)
                                elif isinstance(chunk_data, list) and chunk_data:
                                    chunk_dfs.append(pd.DataFrame(chunk_data))
                                elif isinstance(chunk_data, dict) and 'data' in chunk_data and chunk_data['data']:
                                    chunk_dfs.append(pd.DataFrame(chunk_data['data']))
                                else:
                                    logger.warning(
                                        f"Task {task_id}: Chunk '{chunk_step}' has unexpected type "
                                        f"{type(chunk_data).__name__} — skipping"
                                    )
                                del chunk_data
                            except Exception as chunk_err:
                                logger.warning(
                                    f"Task {task_id}: Failed to load chunk '{chunk_step}': {chunk_err}"
                                )

                        if not chunk_dfs:
                            logger.error(
                                f"Task {task_id}: No chunks reassembled for '{step}' — skipping step"
                            )
                            continue

                        reassembled = pd.concat(chunk_dfs, ignore_index=True)
                        del chunk_dfs
                        gc.collect()
                        logger.info(
                            f"Task {task_id}: ✅ Reassembled '{step}' from chunks: "
                            f"{len(reassembled)} rows × {len(reassembled.columns)} cols"
                        )
                        # Hand off to the DataFrame path below
                        data_rows = reassembled

                    elif 'data' in raw_data and raw_data['data']:
                        logger.info(f"Task {task_id}: Extracted 'data' field from '{step}' result")
                        data_rows = raw_data['data']
                    # signal_generation may store a secondary ml_dataset key
                    # Only use it as a last resort tabular source for that step
                    elif step == 'signal_generation' and 'ml_dataset' in raw_data and raw_data['ml_dataset']:
                        logger.info(f"Task {task_id}: Using 'ml_dataset' fallback from 'signal_generation'")
                        data_rows = raw_data['ml_dataset']
                    else:
                        data_rows = raw_data
                else:
                    data_rows = raw_data

                # CRITICAL: RESTORE NUMERIC TYPES & DOWNCAST FLOAT64
                # During serialization (JSON), NaN becomes None, which often causes
                # pandas to infer 'object' type upon re-loading. This drops features in ML prep.
                #
                # ⚠️ IMPORTANT: currency_indices and other steps store a raw DataFrame in pickle.
                # After deserialization data_rows IS a DataFrame — never use bool(DataFrame)
                # as that raises "ambiguous truth value". Always use isinstance() checks.
                if isinstance(data_rows, pd.DataFrame):
                    # DataFrame path — already deserialized, just normalize and return records
                    from app.core.services.data_utils import normalize_dataframe_columns
                    data_rows = normalize_dataframe_columns(data_rows)
                    data_rows = restore_numeric_types(data_rows)

                    # Downcast float64 to float32 to reduce memory footprint by 50% for 1000+ col DataFrames
                    float64_cols = data_rows.select_dtypes(include=['float64']).columns
                    if len(float64_cols) > 0:
                        data_rows[float64_cols] = data_rows[float64_cols].astype(np.float32)

                    # Enrichment sanity guard (same as list path below)
                    if step in ENRICHED_STEPS and len(data_rows.columns) < MIN_ENRICHED_COLS:
                        logger.warning(
                            f"Task {task_id}: Step '{step}' DataFrame has only "
                            f"{len(data_rows.columns)} columns "
                            f"(expected >{MIN_ENRICHED_COLS} for enriched data). "
                            f"Skipping — likely corrupted/partial write. Trying next step."
                        )
                        continue

                    if as_dataframe:
                        logger.info(
                            f"Task {task_id}: ✅ DataFrame step '{step}' (as_dataframe=True) "
                            f"({len(data_rows)} rows × {len(data_rows.columns)} cols)"
                        )
                        return data_rows

                    result_records = data_rows.to_dict(orient="records")
                    logger.info(
                        f"Task {task_id}: ✅ DataFrame step '{step}' "
                        f"({len(result_records)} rows × {len(data_rows.columns)} cols)"
                    )
                    return result_records

                if isinstance(data_rows, list) and data_rows:
                    df = pd.DataFrame(data_rows)
                    # Normalize all column name variations
                    from app.core.services.data_utils import normalize_dataframe_columns
                    df = normalize_dataframe_columns(df)
                    df = restore_numeric_types(df)
                    result_records = df.to_dict(orient="records")
                
                    # ── Enrichment sanity guard ──────────────────────────────────────
                    # Steps higher in SESSION_STEP_PRIORITY should contain more columns
                    # than raw OHLCV (8–10 columns).  If a step that is supposed to be
                    # enriched returns ≤10 columns it was likely corrupted by a stale
                    # or incorrect DB write (e.g. the astronomical double-write bug).
                    # Fall through to the next priority step rather than returning garbage.
                    if step in ENRICHED_STEPS and len(df.columns) < MIN_ENRICHED_COLS:
                        logger.warning(
                            f"Task {task_id}: Step '{step}' has only {len(df.columns)} columns "
                            f"(expected >{MIN_ENRICHED_COLS} for enriched data). "
                            f"Skipping — likely a corrupted/partial write. Trying next step."
                        )
                        continue
                    # ────────────────────────────────────────────────────────────────
                
                    # BUG FIX #5: Validate data structure before returning
                    try:
                        validate_data_structure(result_records, step_name=f"{step} (restored)", min_rows=1)
                    except DataValidationError as val_err:
                        logger.error(f"Data validation failed for {step}: {val_err}")
                        raise
                
                    logger.info(
                        f"Task {task_id}: ✅ Data validation passed for {step} (restored): "
                        f"{len(df)} rows"
                    )
                    return result_records

                # BUG FIX #5: Validate before returning
                try:
                    validate_data_structure(data_rows, step_name=step, min_rows=1)
                except DataValidationError as val_err:
                    logger.error(f"Data validation failed for {step}: {val_err}")
                    raise
            
                return data_rows

        except Exception as e:
            logger.warning(f"Task {task_id}: Could not load/deserialize step '{step}': {e}")
        continue

    # Final fallback: DataSession.raw_data (set at upload time)
    try:
        stmt = select(DataSession).where(DataSession.session_id == session_id)
        result = await db.execute(stmt)
        session = result.scalar_one_or_none()
        if session and session.raw_data:
            logger.info(f"Task {task_id}: Falling back to DataSession.raw_data")
            raw_data = deserialize_data(session.raw_data)
            if isinstance(raw_data, list):
                df = pd.DataFrame(raw_data)
                # Normalize all column name variations
                from app.core.services.data_utils import normalize_dataframe_columns
                df = normalize_dataframe_columns(df)
                df = restore_numeric_types(df)
                result_records = df.to_dict(orient="records")
                
                # BUG FIX #5: Validate fallback data
                try:
                    validate_data_structure(result_records, step_name="DataSession.raw_data (fallback)", min_rows=1)
                except DataValidationError as val_err:
                    logger.error(f"Data validation failed for fallback data: {val_err}")
                    raise
                
                return result_records
            return raw_data
    except Exception as e:
        logger.error(f"Task {task_id}: Final fallback to DataSession failed: {e}")

    return None


async def get_current_data(session_id: str, db, task_id: str = "unknown"):
    """
    PHASE 16: Fetch the CURRENT DATA POINTER for a session.
    
    Returns the data marked with is_current_data=TRUE, which represents
    the most recently enriched dataset (combining all mutations so far).
    
    Args:
        session_id: Session ID
        db: Async SQLAlchemy session
        task_id: For logging
        
    Returns:
        List of dicts (rows) representing current enriched data, or None if not found
    """
    try:
        stmt = select(SessionStepResult).where(
            and_(
                SessionStepResult.session_id == session_id,
                SessionStepResult.is_current_data == True
            )
        )
        result = await db.execute(stmt)
        step_result = result.scalar_one_or_none()
        
        if not step_result:
            logger.warning(f"Task {task_id}: No current_data pointer found for session {session_id}")
            return None
        
        logger.info(f"Task {task_id}: Loaded current_data from step '{step_result.step_name}'")
        
        # Deserialize and validate
        raw_data = deserialize_data(step_result.result_data, step_result.is_compressed)
        
        if isinstance(raw_data, dict):
            if 'data' in raw_data and raw_data['data']:
                data_rows = raw_data['data']
            else:
                data_rows = raw_data
        else:
            data_rows = raw_data
        
        # Restore numeric types
        if data_rows and isinstance(data_rows, list):
            df = pd.DataFrame(data_rows)
            df = restore_numeric_types(df)
            data_rows = df.to_dict(orient="records")
            del df # Free large dataframe after conversion
        
        # Cleanup and return
        del raw_data, result, step_result  # Free large objects
        gc.collect()
        return data_rows
        
    except Exception as e:
        logger.error(f"Task {task_id}: Failed to load current_data for session {session_id}: {e}")
        return None


async def set_as_current_data(session_id: str, db, task_id: str = "unknown"):
    """
    PHASE 16: Mark the most recently stored result as THE CURRENT DATA POINTER.
    ATOMIC: Uses single SQL UPDATE with CASE to prevent race conditions (FIX FOR ISSUE #1).
    
    When a mutation step (data_source, technical, snr, astro) finishes enrichment:
    1. It stores result via store_session_step_result (with is_current_data=FALSE)
    2. Then calls this to mark it as current_data=TRUE
    3. Any previous current_data is unmarked (set to FALSE)
    
    RACE-CONDITION-SAFE: Only one row per session can have is_current_data=TRUE
    even if called simultaneously by multiple steps. Uses atomic SQL UPDATE.
    
    Args:
        session_id: Session ID
        db: Async SQLAlchemy session
        task_id: For logging
    """
    try:
        # Find the latest stored result for this session
        latest_stmt = select(SessionStepResult.id).where(
            SessionStepResult.session_id == session_id
        ).order_by(SessionStepResult.stored_at.desc()).limit(1)
        
        result = await db.execute(latest_stmt)
        latest_id = result.scalar()
        
        if not latest_id:
            logger.warning(
                f"[Task {task_id}] No session results found for session {session_id}"
            )
            return
        
        # ATOMIC OPERATION: Single UPDATE using CASE
        # Sets is_current_data=TRUE only for latest_id, FALSE for everything else
        # This is atomic at database level - no race condition possible
        update_stmt = update(SessionStepResult).where(
            SessionStepResult.session_id == session_id
        ).values(
            is_current_data=case(
                (SessionStepResult.id == latest_id, True),
                else_=False
            )
        )
        
        await db.execute(update_stmt)
        await db.commit()
        
        logger.info(
            f"[Task {task_id}] ✅ ATOMIC MARKED CURRENT: session={session_id}, "
            f"result_id={latest_id}"
        )
        
    except Exception as e:
        await db.rollback()
        logger.error(
            f"[Task {task_id}] ❌ FAILED TO MARK CURRENT (ATOMIC): {str(e)}"
        )
        raise

    del update_stmt, latest_stmt, result  # Free memory
    gc.collect()

async def store_session_step_result(
    session_id: str,
    step_name: str,
    data: Union[pd.DataFrame, list, dict],
    db,
    is_compressed: bool = True,
    force_pickle: bool = False,
    pre_serialized_data: Optional[str] = None,
    pre_serialized_hash: Optional[str] = None,
    on_progress: Optional[Callable[[int, str], Any]] = None,
):

    """
    Centralized helper for storing analysis results to the database.
    Uses JSONB (preferred) with fallback to pickle if JSONB fails or is too large.
    Ensures data cleaning (NaN handling) and consistent serialization.
    
    FIX #1: Supports pre-serialized data to avoid CPU work inside DB lock.
    
    Args:
        session_id: Session ID
        step_name: Name of the analysis step (e.g. 'technical_analysis')
        data: Data to store (DataFrame, list, or dict)
        db: Async SQLAlchemy session
        is_compressed: Whether to compress data (only used for pickle fallback)
        force_pickle: If True, skip JSONB and use pickle directly (for large SNR results)
        pre_serialized_data: PRE-SERIALIZED data (optional, avoids CPU work inside lock)
        pre_serialized_hash: Hash of pre-serialized data (optional)
    
    TODO: Parallel Serialization for Large Datasets
    ─────────────────────────────────────────────────────────────────────────
    When data > 50K rows, implement chunk-based serialization:
    
    1. Break data into N chunks (optimal chunk size: ~10K rows)
    2. Serialize chunks in parallel using process pool (bypass GIL)
    3. Concatenate serialized chunks
    4. Store single blob to DB
    
    Pseudo-code:
        if len(data) > 50_000:
            chunks = split_into_chunks(data, chunk_size=10_000)
            serialized_chunks = await parallelize(
                serialize_chunk, chunks, max_workers=num_cpu_cores
            )
            final_data = concatenate_serialized(serialized_chunks)
        else:
            final_data = serialize_data(data)
    
    Expected gain: 2-3s → 0.8s (60% reduction) on 4-core machine
    ─────────────────────────────────────────────────────────────────────────
    """
    try:
        # ─────────────────────────────────────────────────────────
        # FIX #1: USE PRE-SERIALIZED DATA IF PROVIDED
        # ─────────────────────────────────────────────────────────
        if pre_serialized_data is not None:
            logger.info(
                f"✅ Using pre-serialized data for '{step_name}' "
                f"(size: {len(pre_serialized_data)/1000:.0f}KB)"
            )
            result_data = pre_serialized_data
            result_data_v2 = None
            is_using_jsonb = False
            result_hash = pre_serialized_hash
            
            # Skip to DB insert
            # Skip to DB insert
            stmt = pg_insert(SessionStepResult).values(
                session_id=session_id,
                step_name=step_name,
                result_data=result_data,
                result_data_v2=result_data_v2,
                is_using_jsonb=is_using_jsonb,
                result_hash=result_hash,
                is_compressed=is_compressed,
                stored_at=datetime.utcnow()
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=['session_id', 'step_name'],
                set_={
                    'result_data': literal_column('EXCLUDED.result_data'),
                    'result_data_v2': literal_column('EXCLUDED.result_data_v2'),
                    'is_using_jsonb': literal_column('EXCLUDED.is_using_jsonb'),
                    'result_hash': literal_column('EXCLUDED.result_hash'),
                    'is_compressed': literal_column('EXCLUDED.is_compressed'),
                    'stored_at': literal_column('EXCLUDED.stored_at'),
                }
            )
            await db.execute(stmt)
            await db.commit()
            
            logger.info(
                f"✅ Successfully stored pre-serialized result for '{step_name}' "
                f"(session: {session_id})"
            )
            return
        
        # ─────────────────────────────────────────────────────────
        # STANDARD PATH: SERIALIZE NOW (if not pre-serialized)
        # ─────────────────────────────────────────────────────────
        
        # 🔒 FIX #15: For all analysis steps, force pickle to ensure consistency
        # ✅ P4 FIX: Include 'data_source' to ensure uniform serialization format
        # Frontend expects ALL steps to be pickled for consistent I/O handling
        if force_pickle or step_name in ['data_source', 'snr_analysis', 'snr_analysis_ml_dataset', 'astronomical_analysis', 'technical_analysis', 'footprint_ingestion', 'currency_indices']:
            logger.info(f"📦 Using pickle format (force_pickle={force_pickle}) for step '{step_name}'")
            result_records = data  # Use as-is for pickle
            result_data_v2 = None
            result_data = None
            is_using_jsonb = False
            result_hash = None
            
            # Prep data for pickle
            if isinstance(data, pd.DataFrame):
                # Keep DataFrame as DataFrame object for efficient binary pickling
                result_records = clean_dataframe(data)
            elif isinstance(data, dict):
                result_records = data.copy()
                for key in ['data', 'ml_dataset', 'signals']:
                    if key in result_records and isinstance(result_records[key], list):
                        try:
                            temp_df = pd.DataFrame(result_records[key])
                            temp_df = clean_dataframe(temp_df)
                            result_records[key] = temp_df.to_dict(orient="records")
                        except Exception as clean_err:
                            logger.warning(f"Could not clean nested key '{key}': {clean_err}")
            
            # Use pickle format
            use_numpy_safe = step_name.startswith('ml_preparation') or step_name.startswith('snr_analysis') or isinstance(result_records, pd.DataFrame)

            serialized_data = serialize_data(result_records, is_compressed, numpy_safe=use_numpy_safe, on_progress=on_progress)
            result_data = serialized_data

            
            try:
                serialized_bytes = base64.b64decode(serialized_data)
                result_hash = hashlib.sha256(serialized_bytes).hexdigest()
                logger.info(f"✅ STORAGE: Using pickle for '{step_name}' ({len(serialized_data)/1000:.0f}KB compressed)")
            except Exception as hash_err:
                logger.warning(f"⚠️ Failed to compute hash for '{step_name}': {hash_err}")
        else:
            # Standard path: Try JSONB first
            # 1. Prepare data (Convert to list of records if DataFrame)
            if isinstance(data, pd.DataFrame):
                # Clean data to ensure no NaNs (numeric 0.0 instead of None)
                data = clean_dataframe(data)
                result_records = data.to_dict(orient="records")
            elif isinstance(data, dict):
                # If it's a dictionary, look for 'data' or 'ml_dataset' keys and clean them if they are lists
                result_records = data.copy()
            else:
                result_records = data
        
            if isinstance(result_records, dict):
                for key in ['data', 'ml_dataset', 'signals']:
                    if key in result_records and isinstance(result_records[key], list):
                        # Convert to DF, clean, then back to records
                        try:
                            temp_df = pd.DataFrame(result_records[key])
                            temp_df = clean_dataframe(temp_df)
                            result_records[key] = temp_df.to_dict(orient="records")
                        except Exception as clean_err:
                            logger.warning(f"Could not clean nested key '{key}': {clean_err}")

            # 2. Try JSONB first (preferred) - ONLY for non-pickle steps
            result_data_v2 = None
            result_data = None
            is_using_jsonb = False
            result_hash = None
            
            try:
                if on_progress:
                    on_progress(10, "Converting data to JSON-safe format...")
                
                # Convert to JSONB-safe format (single to_serializable call, no pickle)
                json_safe_data = to_serializable(result_records)

                
                # FIX #14: PRE-EMPTIVE SIZE CHECK (Postgres JSONB limit is ~256MB)
                # 200MB threshold to be safe and avoid performance degradation
                json_str_len = 0
                try:
                    # Simple length heuristic: JSON string length roughly matches byte size for ASCII
                    # We only do this for large-looking objects to avoid overhead
                    if hasattr(json_safe_data, '__len__') and len(json_safe_data) > 10000:
                        json_str_for_size = json.dumps(json_safe_data)
                        json_str_len = len(json_str_for_size)
                        del json_str_for_size # Free memory
                        
                        if json_str_len > 200_000_000: # 200MB threshold
                            logger.warning(
                                f"⚠️ Data for '{step_name}' is too large for JSONB ({json_str_len / 1_000_000:.1f} MB). "
                                f"Forcing Pickle fallback to avoid Postgres limits."
                            )
                            raise ValueError("Payload exceeds safe JSONB limit")
                except (TypeError, ValueError) as size_err:
                    if "Payload exceeds" in str(size_err): raise
                    logger.debug(f"Could not pre-calculate size for '{step_name}': {size_err}")

                result_data_v2 = json_safe_data
                is_using_jsonb = True
                
                logger.info(f"✅ JSONB serialization successful for step '{step_name}'")
                
                # FIX #1.5: Compute SHA-256 hash for integrity verification
                try:
                    # Always use json.dumps() with sort_keys=True for deterministic/consistent hash
                    json_str = json.dumps(json_safe_data, sort_keys=True)
                        
                    result_hash = hashlib.sha256(json_str.encode()).hexdigest()
                    logger.info(f"✅ STORAGE HASH: {result_hash[:16]}... for step '{step_name}' (JSONB)")
                    del json_str
                except Exception as hash_err:
                    logger.warning(f"⚠️ Failed to compute hash for '{step_name}': {hash_err}")
            except Exception as jsonb_err:
                # FALLBACK: Use pickle format if JSONB fails
                logger.warning(f"⚠️ JSONB serialization failed for step '{step_name}': {jsonb_err}")
                logger.warning(f"   Falling back to pickle format...")
                
                
                use_numpy_safe = step_name.startswith('ml_preparation')
                serialized_data = serialize_data(result_records, is_compressed, numpy_safe=use_numpy_safe, on_progress=on_progress)
                result_data = serialized_data
                is_using_jsonb = False

                
                try:
                    serialized_bytes = base64.b64decode(serialized_data)
                    result_hash = hashlib.sha256(serialized_bytes).hexdigest()
                    logger.info(f"✅ STORAGE HASH: {result_hash[:16]}... for step '{step_name}' (Pickle fallback)")
                except Exception as hash_err:
                    logger.warning(f"⚠️ Failed to compute hash for '{step_name}': {hash_err}")

        # 3. Upsert into database (for all paths - pickle or JSONB)
        stmt = None
        try:
            stmt = pg_insert(SessionStepResult).values(
                session_id=session_id,
                step_name=step_name,
                result_data=result_data,  # Pickle format (nullable if using JSONB)
                result_data_v2=result_data_v2,  # JSONB format
                is_using_jsonb=is_using_jsonb,  # Track which format
                result_hash=result_hash,
                is_compressed=is_compressed,
                stored_at=datetime.utcnow()
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=['session_id', 'step_name'],
                set_={
                    'result_data': literal_column('EXCLUDED.result_data'),
                    'result_data_v2': literal_column('EXCLUDED.result_data_v2'),
                    'is_using_jsonb': literal_column('EXCLUDED.is_using_jsonb'),
                    'result_hash': literal_column('EXCLUDED.result_hash'),
                    'is_compressed': literal_column('EXCLUDED.is_compressed'),
                    'stored_at': literal_column('EXCLUDED.stored_at'),
                }
            )
            if on_progress:
                on_progress(95, "Committing to database...")
                
            await db.execute(stmt)
            await db.commit()
            
            if on_progress:
                on_progress(100, "Persistence complete.")

        except Exception as sql_err:
            # FIX #14 (ENHANCED): Catch Postgres-level limit errors that bypass Python checks
            err_str = str(sql_err)
            if "ProgramLimitExceededError" in err_str or "maximum of 268435455 bytes" in err_str:
                logger.warning(
                    f"🚨 POSTGRES LIMIT EXCEEDED for '{step_name}' (SQL execution level). "
                    f"Retrying with compressed Pickle format..."
                )
                await db.rollback()
                
                use_numpy_safe = step_name.startswith('ml_preparation')
                serialized_data = serialize_data(result_records, is_compressed=True, numpy_safe=use_numpy_safe)
                
                try:
                    serialized_bytes = base64.b64decode(serialized_data)
                    new_hash = hashlib.sha256(serialized_bytes).hexdigest()
                except:
                    new_hash = result_hash # Fallback
                
                retry_stmt = pg_insert(SessionStepResult).values(
                    session_id=session_id,
                    step_name=step_name,
                    result_data=serialized_data,
                    result_data_v2=None,  # CLEAR JSONB
                    is_using_jsonb=False,
                    result_hash=new_hash,
                    is_compressed=True,
                    stored_at=datetime.utcnow()
                )
                retry_stmt = retry_stmt.on_conflict_do_update(
                    index_elements=['session_id', 'step_name'],
                    set_={
                        'result_data': literal_column('EXCLUDED.result_data'),
                        'result_data_v2': literal_column('EXCLUDED.result_data_v2'),
                        'is_using_jsonb': literal_column('EXCLUDED.is_using_jsonb'),
                        'result_hash': literal_column('EXCLUDED.result_hash'),
                        'is_compressed': literal_column('EXCLUDED.is_compressed'),
                        'stored_at': literal_column('EXCLUDED.stored_at'),
                    }
                )
                await db.execute(retry_stmt)
                await db.commit()
                is_using_jsonb = False # For logging
            elif "ConnectionDoesNotExistError" in err_str or "connection was closed in the middle" in err_str:
                # The asyncpg TCP connection was dropped mid-write.
                # This almost always means the payload (~result_data) was too large
                # for a single round-trip (seen with 300 MB+ pickle blobs for SNR DataFrames).
                # The session is now poisoned — we CANNOT retry on the same db handle.
                # Surface the error so _persist_to_database can mark pm_persist_failed=True
                # and the pipeline continues from in-memory (TIER 0a) rather than blocking.
                payload_mb = len(result_data) / 1_000_000 if result_data else 0
                logger.error(
                    f"❌ SQL Execution failed for '{step_name}': {sql_err}"
                )
                logger.warning(
                    f"⚠️  [ConnectionDrop] asyncpg TCP connection lost mid-write for '{step_name}'. "
                    f"Payload size: {payload_mb:.1f} MB. "
                    f"Session is poisoned — skipping retry. "
                    f"Result remains available in TIER 0a in-memory cache."
                )
                try:
                    await db.rollback()
                except Exception:
                    pass  # Session already dead, swallow rollback error
                raise
            else:
                # Other SQL error: rollback and raise
                logger.error(f"❌ SQL Execution failed for '{step_name}': {sql_err}")
                await db.rollback()
                raise
        
        if is_using_jsonb:
            logger.info(f"✅ Successfully stored step result for '{step_name}' (JSONB) (session: {session_id})")
        else:
            logger.info(f"✅ Successfully stored step result for '{step_name}' (Pickle) (session: {session_id})")
        
    except Exception as e:
        logger.error(f"❌ Failed to store session step result: {e}")
        await db.rollback()
        raise
    finally:
        # Free memory before end of function
        if 'stmt' in locals() and stmt is not None: del stmt
        if 'result_hash' in locals(): del result_hash
        gc.collect()
async def store_session_step_result_chunked(
    session_id: str,
    step_name: str,
    data: Union[pd.DataFrame, list],
    db,
    chunk_size: int = 10000,
    is_compressed: bool = True
):
    """
    Unified chunked storage for large analysis results.
    Splits data into multiple rows (step_name_0, step_name_1...) and stores
    a metadata header row (step_name) with total_rows and chunk_info.
    
    Args:
        session_id: Session ID
        step_name: Base step name (e.g. 'astronomical_analysis')
        data: DataFrame or list of rows to store
        db: Async SQLAlchemy session
        chunk_size: Number of rows per chunk (default 10,000)
        is_compressed: Whether to compress chunks
    """
    try:
        # Efficient slicing without converting entire DataFrame to dicts
        is_df = isinstance(data, pd.DataFrame)
        total_rows = len(data)
        num_chunks = (total_rows + chunk_size - 1) // chunk_size if total_rows > 0 else 0
        
        logger.info(f"💾 Chunking '{step_name}' into {num_chunks} chunks (total={total_rows}, size={chunk_size})")

        # 1. Store individual chunks
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, total_rows)
            if is_df:
                chunk_slice = data.iloc[start_idx:end_idx]
            else:
                chunk_slice = data[start_idx:end_idx]
            
            chunk_step_name = f"{step_name}_{i}"
            
            # Use base storage helper for the actual row write
            await store_session_step_result(
                session_id=session_id,
                step_name=chunk_step_name,
                data=chunk_slice,
                db=db,
                is_compressed=is_compressed,
                force_pickle=is_df
            )
            
            # Explicitly clear slice from memory after each chunk write
            del chunk_slice
            if i % 5 == 0: gc.collect()
            
        # 2. Store metadata header row (with _is_chunked=True)
        metadata = {
            "_is_chunked": True,
            "total_rows": total_rows,
            "chunk_size": chunk_size,
            "num_chunks": num_chunks,
            "step_name": step_name,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await store_session_step_result(
            session_id=session_id,
            step_name=step_name,
            data=metadata,
            db=db,
            is_compressed=False # Metadata is small
        )
        
        logger.info(f"✅ Finished chunked storage for '{step_name}' ({num_chunks} chunks stored)")
        
    except Exception as e:
        logger.error(f"❌ Failed chunked storage for '{step_name}': {e}")
        await db.rollback()
        raise
    finally:
        gc.collect()



# ============================================================================
# BATCH STREAMING: Write Incremental Batches to DB (Memory-Efficient)
# ============================================================================

async def store_batch_to_db(
    session_id: str,
    step_name: str,
    batch_data: Dict[str, Any],
    db,
    batch_number: int,
    total_batches: int,
    prev_batch_hashes: Optional[set] = None,
    task_store=None
) -> Dict[str, Any]:
    """
    Store a single batch of results with deduplication against previous batch's overlap window.
    
    CRITICAL: This function is the core of the streaming architecture. Instead of accumulating
    all 189K rows in memory and storing once, each 5K-row batch is:
    1. Deduplicated against overlapping rows from previous batch
    2. Serialized and stored to DB
    3. Cleared from memory
    
    Args:
        session_id: Session ID
        step_name: Analysis step ('snr_analysis', 'technical_analysis', 'astronomical_analysis')
        batch_data: Dict with batch results:
            {
                'signals': [...],           # SNR signals from this batch
                'indicators': [...],        # Technical indicators
                'astro_features': [...],    # Astronomical features
                'rows_processed': int,      # Number of data rows in batch
                'batch_range': (start, end) # Row indices in full dataset
            }
        db: Async SQLAlchemy session
        batch_number: Current batch number (1-indexed)
        total_batches: Total number of batches
        prev_batch_hashes: Set of hashes from previous batch's last 100 rows (for dedup)
        task_store: TaskStore for progress updates
    
    Returns:
        Dict with:
        {
            'rows_inserted': int,
            'rows_skipped': int (duplicates),
            'rows_failed': int,
            'batch_hashes': set,  # Pass to next batch for dedup
            'batch_number': int,
            'total_batches': int,
            'storage_format': 'jsonb' or 'pickle',
            'size_kb': float
        }
    """
    try:
        logger.info(f"🔄 [Batch {batch_number}/{total_batches}] Starting DB write...")
        
        # 1. Extract batch metadata
        rows_processed = batch_data.get('rows_processed', 0)
        batch_range = batch_data.get('batch_range', (0, rows_processed))
        
        # 2. Deduplicate this batch against previous batch's overlap window
        prev_batch_hashes = prev_batch_hashes or set()
        batch_hashes = set()
        rows_to_write = []
        rows_skipped: int = 0
        
        # Collect results from all analysis types in this batch
        for result_type in ['signals', 'indicators', 'astro_features']:
            if result_type not in batch_data or not batch_data[result_type]:
                continue
            
            for row in batch_data[result_type]:
                # Create hash for deduplication (use index + key fields)
                try:
                    sig_key = (
                        row.get('index', -1),
                        row.get('type', row.get('signal_type', 'unknown')),
                        row.get('price', 0.0)
                    )
                    sig_hash = hashlib.sha256(str(sig_key).encode()).hexdigest()
                    
                    # Skip if this row was already written in previous batch's overlap
                    if sig_hash in prev_batch_hashes:
                        rows_skipped += 1
                        logger.debug(f"⏭️ [Dedup] Skipping duplicate row at index {row.get('index')}")
                        continue
                    
                    batch_hashes.add(sig_hash)
                    rows_to_write.append(row)
                
                except Exception as row_err:
                    logger.warning(f"⚠️ [Dedup] Could not hash row {row}: {row_err}")
                    rows_to_write.append(row)
        
        rows_inserted = len(rows_to_write)
        rows_failed = 0
        
        if rows_inserted == 0:
            logger.warning(f"⚠️ [Batch {batch_number}] No rows to write (all duplicates or empty)")
            return {
                'rows_inserted': 0,
                'rows_skipped': rows_skipped,
                'rows_failed': 0,
                'batch_hashes': batch_hashes,
                'batch_number': batch_number,
                'total_batches': total_batches,
                'storage_format': 'none',
                'size_kb': 0
            }
        
        # 3. Serialize batch results (use pickle for large batches, JSONB for metadata)
        # For streaming writes, we prefer pickle to avoid size limits
        result_records = {
            'batch': {
                'number': batch_number,
                'total': total_batches,
                'range': batch_range,
                'rows_processed': rows_processed,
                'rows_inserted': rows_inserted,
                'rows_skipped': rows_skipped
            },
            'signals': batch_data.get('signals', []),
            'indicators': batch_data.get('indicators', []),
            'astro_features': batch_data.get('astro_features', [])
        }
        
        # Use pickle for batch storage (smaller, faster than JSONB for streaming)
        try:
            serialized_data = serialize_data(result_records, is_compressed=True, numpy_safe=False)
            size_kb = len(base64.b64decode(serialized_data)) / 1000
            storage_format = 'pickle'
            is_using_jsonb = False
            result_data = serialized_data
            result_data_v2 = None
            
            logger.info(f"✅ [Batch {batch_number}] Serialized to pickle ({size_kb:.1f}KB)")
        except Exception as pkl_err:
            logger.error(f"❌ [Batch {batch_number}] Pickle serialization failed: {pkl_err}")
            rows_failed = rows_inserted
            rows_inserted = 0
            result_data = None
            result_data_v2 = None
            storage_format = 'failed'
            size_kb = 0
        
        # 4. Store to database with UPSERT (preserves all batches, not replacing)
        if result_data:
            try:
                # Use unique task_id + batch_number as identifier for this batch
                # This way, each batch is stored separately (no overwrite)
                batch_step_name = f"{step_name}__batch_{batch_number}"
                
                stmt = pg_insert(SessionStepResult).values(
                    session_id=session_id,
                    step_name=batch_step_name,
                    result_data=result_data,
                    result_data_v2=result_data_v2,
                    is_using_jsonb=False,  # Always pickle for batches
                    result_hash=hashlib.sha256(result_data.encode()).hexdigest() if isinstance(result_data, str) else None,
                    is_compressed=True,
                    stored_at=datetime.utcnow()
                ).on_conflict_do_update(
                    index_elements=['session_id', 'step_name'],
                    set_={
                        'result_data': literal_column('EXCLUDED.result_data'),
                        'result_data_v2': literal_column('EXCLUDED.result_data_v2'),
                        'is_using_jsonb': literal_column('EXCLUDED.is_using_jsonb'),
                        'is_compressed': literal_column('EXCLUDED.is_compressed'),
                        'stored_at': literal_column('EXCLUDED.stored_at'),
                    }
                )
                
                await db.execute(stmt)
                await db.commit()
                
                logger.info(f"✅ [Batch {batch_number}] Stored {rows_inserted} rows to DB ({size_kb:.1f}KB)")
            
            except Exception as db_err:
                logger.error(f"❌ [Batch {batch_number}] DB write failed: {db_err}")
                await db.rollback()
                rows_failed = rows_inserted
                rows_inserted = 0
        
        # 5. Update task progress
        if task_store:
            progress_pct = int((batch_number / total_batches) * 100)
            task_store.update_task(
                task_id=f"batch_{batch_number}",  # Simplified for batch tracking
                progress=progress_pct,
                message=f"Written batch {batch_number}/{total_batches}",
                signals_found=rows_inserted
            )
        
        # 6. Return batch stats for caller
        result = {
            'rows_inserted': rows_inserted,
            'rows_skipped': rows_skipped,
            'rows_failed': rows_failed,
            'batch_hashes': batch_hashes,  # For next batch's dedup window
            'batch_number': batch_number,
            'total_batches': total_batches,
            'storage_format': storage_format,
            'size_kb': size_kb
        }
        
        logger.info(f"✅ [Batch {batch_number}] Complete: inserted={rows_inserted}, skipped={rows_skipped}, failed={rows_failed}")
        return result
    
    except Exception as e:
        logger.error(f"❌ store_batch_to_db failed for batch {batch_number}: {e}", exc_info=True)
        return {
            'rows_inserted': 0,
            'rows_skipped': 0,
            'rows_failed': 1,
            'batch_hashes': set(),
            'batch_number': batch_number,
            'total_batches': total_batches,
            'storage_format': 'failed',
            'size_kb': 0,
            'error': str(e)
        }


# ============================================================================
# PRIORITY 1: Metadata-Only Retrieval (No Full Deserialization)
# ============================================================================

async def get_step_metadata(
    session_id: str,
    step_name: str,
    db
) -> dict:
    """
    Get metadata about a step result WITHOUT deserializing full dataset.
    
    Perfect for status checks on multi-GB datasets. Returns size, format, 
    timestamp without loading data into memory.
    
    Performance: < 50ms (vs 1000ms+ for full retrieval)
    
    Args:
        session_id: Session ID
        step_name: Step name to check
        db: Async SQLAlchemy session
        
    Returns:
        Metadata dict with size, format, stored_at, etc.
    """
    try:
        stmt = select(SessionStepResult).where(
            and_(
                SessionStepResult.session_id == session_id,
                SessionStepResult.step_name == step_name
            )
        ).order_by(SessionStepResult.stored_at.desc()).limit(1)
        
        result = await db.execute(stmt)
        step_result = result.scalar_one_or_none()
        
        if not step_result:
            return {
                "session_id": session_id,
                "step_name": step_name,
                "found": False,
                "message": f"No data stored for step '{step_name}'"
            }
        
        # Compute size in bytes
        size_bytes = 0
        if step_result.result_data:
            size_bytes = len(step_result.result_data.encode()) if isinstance(step_result.result_data, str) else len(step_result.result_data)
        elif step_result.result_data_v2:
            size_bytes = len(str(step_result.result_data_v2).encode())
        
        metadata = {
            "session_id": session_id,
            "step_name": step_name,
            "found": True,
            "stored_at": step_result.stored_at.isoformat() if step_result.stored_at else None,
            "format": "JSONB" if step_result.is_using_jsonb else "Pickle",
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / 1_000_000, 2),
            "is_compressed": step_result.is_compressed,
            "has_hash": step_result.result_hash is not None
        }
        
        logger.info(f"📊 Metadata for '{step_name}': {metadata['size_mb']}MB ({metadata['format']})")
        return metadata
        
    except Exception as e:
        logger.error(f"❌ Failed to retrieve metadata for '{step_name}': {e}")
        return {
            "session_id": session_id,
            "step_name": step_name,
            "found": False,
            "error": str(e)
        }


async def store_ml_split_chunks(
    session_id: str,
    split_name: str,
    split_data: dict,
    db,
    chunk_size: int = 2000,
    is_compressed: bool = True,
) -> dict:
    """
    Store a single ML split (train/validation/test) into the DB as multiple chunk rows.

    Each chunk row is stored as:
        step_name = 'ml_preparation_{split_name}_{chunk_index}'

    This is the format expected by MLDatasetReader.loadWindow() on the frontend, which
    reads 'ml_chunk_{taskId}_{split}_{ci}' from IDB, and falls back to the windowed DB
    API that can now serve individual chunk rows without deserialising the entire blob.

    Args:
        session_id:   Session ID
        split_name:   'train', 'validation', or 'test'
        split_data:   Dict with keys: sequences, labels, targets, indices, sequence_metadata
        db:           Async SQLAlchemy session
        chunk_size:   Sequences per chunk row (default 500, keeps each row ≈ 50–200 MB)
        is_compressed: Whether to zlib-compress each chunk

    Returns:
        { 'chunk_count': int, 'seq_count': int }
    """
    import math
    sequences = split_data.get('sequences', [])
    seq_count = len(sequences) if hasattr(sequences, '__len__') else 0

    if seq_count == 0:
        logger.warning(f"store_ml_split_chunks: split '{split_name}' has 0 sequences, skipping")
        return {'chunk_count': 0, 'seq_count': 0}

    num_chunks = math.ceil(seq_count / chunk_size)
    logger.info(f"💾 Storing split '{split_name}': {seq_count} sequences → {num_chunks} chunks of {chunk_size}")

    labels          = split_data.get('labels', [])
    indices         = split_data.get('indices', [])
    sequence_meta   = split_data.get('sequence_metadata', [])
    targets_dict    = split_data.get('targets', {})

    for ci in range(num_chunks):
        start = ci * chunk_size
        end   = min(start + chunk_size, seq_count)

        chunk = {
            'sequences':         sequences[start:end] if hasattr(sequences, '__getitem__') else [],
            'labels':            labels[start:end]    if hasattr(labels, '__getitem__') else [],
            'indices':           indices[start:end]   if hasattr(indices, '__getitem__') else [],
            'sequence_metadata': sequence_meta[start:end] if hasattr(sequence_meta, '__getitem__') else [],
            'targets': {
                k: v[start:end] if hasattr(v, '__getitem__') else []
                for k, v in targets_dict.items()
            }
        }

        step_name_chunk = f'ml_preparation_{split_name}_{ci}'
        await store_session_step_result(
            session_id=session_id,
            step_name=step_name_chunk,
            data=chunk,
            db=db,
            is_compressed=is_compressed,
        )
        logger.info(f"  ✅ Chunk {ci}/{num_chunks-1}: seqs {start}–{end-1} → '{step_name_chunk}'")

    del sequences, labels, indices, sequence_meta, targets_dict  # Free memory
    gc.collect()
    return {'chunk_count': num_chunks, 'seq_count': seq_count}


async def retrieve_ml_split_full(
    session_id: str,
    split_name: str,
    db,
    task_id: str = "unknown"
) -> Optional[Dict[str, Any]]:
    """
    PHASE 5: Efficiently retrieves all chunks for a specific ML split from the database
    and reassembles them into a single dataset dictionary.

    Args:
        session_id:  Session ID
        split_name:  'train', 'validation', or 'test'
        db:          Async SQLAlchemy session
        task_id:     For logging

    Returns:
        Dict with keys: sequences, labels, targets, indices, sequence_metadata
        or None if no chunks found.
    """
    logger.info(f"Task {task_id}: Reassembling full ML split '{split_name}' for session {session_id}")

    try:
        # 1. Query all chunks for this split, ordered by index
        # step_name format: 'ml_preparation_{split_name}_{chunk_index}'
        search_pattern = f'ml_preparation_{split_name}_%'
        
        stmt = select(SessionStepResult).where(
            and_(
                SessionStepResult.session_id == session_id,
                SessionStepResult.step_name.like(search_pattern)
            )
        ).order_by(SessionStepResult.step_name) # Lexicographical order works for _0, _1, etc.
        # Note: If we have >9 chunks, _10 comes before _2. 
        # Better to sort by length then name or parse index.
        
        result = await db.execute(stmt)
        chunks = result.scalars().all()
        
        if not chunks:
            logger.warning(f"Task {task_id}: No chunks found for split '{split_name}'")
            return None

        # Correctly sort chunks by index (parse from step_name)
        def get_chunk_index(name):
            try:
                return int(name.split('_')[-1])
            except:
                return 0
        
        chunks.sort(key=lambda x: get_chunk_index(x.step_name))
        
        logger.info(f"Task {task_id}: Found {len(chunks)} chunks for split '{split_name}'")

        # 2. Reassemble
        all_sequences = []
        all_labels = []
        all_indices = []
        all_meta = []
        all_targets = {}

        for chunk_row in chunks:
            chunk_data = deserialize_data(chunk_row.result_data, chunk_row.is_compressed)
            
            if not isinstance(chunk_data, dict):
                logger.warning(f"Task {task_id}: Chunk {chunk_row.step_name} has invalid format")
                continue
                
            all_sequences.append(chunk_data.get('sequences', []))
            all_labels.append(chunk_data.get('labels', []))
            all_indices.append(chunk_data.get('indices', []))
            all_meta.append(chunk_data.get('sequence_metadata', []))
            
            # Merge targets dictionary
            chunk_targets = chunk_data.get('targets', {})
            if isinstance(chunk_targets, dict):
                for k, v in chunk_targets.items():
                    if k not in all_targets:
                        all_targets[k] = []
                    all_targets[k].append(v)

        # Flatten/concatenate lists
        # Sequences are typically numpy-safe lists of lists or arrays
        # np.concatenate is most efficient for sequences
        def safe_concat(lists):
            if not lists: return []
            try:
                # Filter out empty lists
                valid_lists = [l for l in lists if len(l) > 0]
                if not valid_lists: return []
                
                # Check if elements are numpy arrays or lists
                if isinstance(valid_lists[0], np.ndarray):
                    return np.concatenate(valid_lists, axis=0)
                elif isinstance(valid_lists[0], list):
                    # For performance, use list comprehension flattening if short, 
                    # else np.array concat then tolist()
                    res = []
                    for l in valid_lists:
                        res.extend(l)
                    return res
                return []
            except Exception as e:
                logger.error(f"Error flattening chunks: {e}")
                return []

        final_result = {
            'sequences':         safe_concat(all_sequences),
            'labels':            safe_concat(all_labels),
            'indices':           safe_concat(all_indices),
            'sequence_metadata': safe_concat(all_meta),
            'targets': {
                k: safe_concat(v) for k, v in all_targets.items()
            }
        }
        
        total_seqs = len(final_result['sequences'])
        logger.info(f"Task {task_id}: ✅ Reassembled {total_seqs} sequences for split '{split_name}'")
        
        del all_sequences, all_labels, all_indices, all_meta, all_targets, chunks, result  # Free memory
        gc.collect()
        return final_result

    except Exception as e:
        logger.error(f"Task {task_id}: Failed to reassemble ML split '{split_name}': {e}", exc_info=True)
        return None


# ============================================================================
# FIX #2: DATA STRUCTURE VALIDATION (ISSUE #2)
# ============================================================================

async def validate_dataframe_structure(
    df: pd.DataFrame,
    step_name: str,
    task_id: str = "unknown",
    required_columns: Optional[List[str]] = None
) -> bool:
    """
    Validate that loaded data structure is compatible with step expectations (FIX FOR ISSUE #2).
    
    Checks:
    - DataFrame not empty
    - Required columns present
    - Numeric columns are actually numeric
    - No critical NaN in required columns
    - Time column parseable if present
    
    Args:
        df: DataFrame to validate
        step_name: Step name (for logging)
        task_id: Task ID (for logging)
        required_columns: List of required column names
    
    Returns:
        True if valid
    
    Raises:
        DataValidationError: If validation fails
    """
    
    REQUIRED_BASE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume', 'Time']
    NUMERIC_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    try:
        # 1. Check not empty
        if df.empty:
            raise DataValidationError(
                f"[Task {task_id}] Step {step_name}: DataFrame is empty"
            )
        
        # 2. Check required columns
        cols_to_check = required_columns or REQUIRED_BASE_COLUMNS
        missing_cols = set(cols_to_check) - set(df.columns)
        if missing_cols:
            raise DataValidationError(
                f"[Task {task_id}] Step {step_name}: Missing columns: {missing_cols}. "
                f"Available: {list(df.columns)}"
            )
        
        # 3. Check numeric types
        for col in NUMERIC_COLUMNS:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    raise DataValidationError(
                        f"[Task {task_id}] Step {step_name}: Column '{col}' is not numeric. "
                        f"Type: {df[col].dtype}"
                    )
        
        # 4. Check NaN in critical columns (drop rows with NaN, warn if found)
        initial_rows = len(df)
        for col in cols_to_check:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    logger.warning(
                        f"[Task {task_id}] Step {step_name}: Column '{col}' has {nan_count} NaN values. "
                        f"Dropping rows with NaN in required columns."
                    )
                    df = df.dropna(subset=[col])
        
        final_rows = len(df)
        if initial_rows != final_rows:
            logger.info(
                f"[Task {task_id}] Step {step_name}: Dropped {initial_rows - final_rows} rows with NaN"
            )
        
        # 5. Check Time column parseable
        if 'Time' in df.columns:
            try:
                pd.to_datetime(df['Time'])
            except Exception as e:
                raise DataValidationError(
                    f"[Task {task_id}] Step {step_name}: Time column not parseable. Error: {str(e)}"
                )
        
        logger.info(
            f"[Task {task_id}] ✅ DATA VALIDATION PASSED: {step_name}. "
            f"Shape: {df.shape}, Columns: {len(df.columns)}"
        )
        
        return True
        
    except DataValidationError:
        raise
    except Exception as e:
        raise DataValidationError(
            f"[Task {task_id}] Unexpected error validating {step_name}: {str(e)}"
        )


# ============================================================================
# FIX #3: ATOMIC STORE + MARK OPERATION (ISSUE #3)
# ============================================================================

async def store_and_mark_current(
    session_id: str,
    step_name: str,
    data: Union[pd.DataFrame, list, dict],
    db,
    task_id: str = "unknown",
    is_compressed: bool = True
) -> bool:
    """
    ATOMIC operation: Store step result AND mark as current data pointer (FIX FOR ISSUE #3).
    
    If either operation fails, both are rolled back - no inconsistent state.
    This prevents the scenario where data is stored but not marked as current,
    causing subsequent steps to load stale data.
    
    Args:
        session_id: Session ID
        step_name: Step name
        data: Data to store (DataFrame, dict, or list)
        db: Async SQLAlchemy session
        task_id: Task ID (for logging)
        is_compressed: Whether to compress data
    
    Returns:
        True if successful
    
    Raises:
        AtomicOperationError: If store or mark fails (includes rollback)
    """
    
    try:
        logger.info(
            f"[Task {task_id}] Starting atomic store+mark for step={step_name}"
        )
        
        # OPERATION 1: Store result
        await store_session_step_result(
            session_id=session_id,
            step_name=step_name,
            data=data,
            db=db,
            is_compressed=is_compressed
        )
        logger.info(f"[Task {task_id}] ✅ STORED: step={step_name}")
        
        # OPERATION 2: Mark as current
        await set_as_current_data(session_id, db, task_id)
        logger.info(f"[Task {task_id}] ✅ MARKED: {step_name} is now current_data")
        
        # Explicit commit to ensure both operations are persisted
        await db.commit()
        logger.info(
            f"[Task {task_id}] ✅ ATOMIC OPERATION COMPLETE: "
            f"step={step_name}, session={session_id}"
        )
        
        return True
        
    except Exception as e:
        # If ANY operation fails, rollback BOTH
        await db.rollback()
        logger.error(
            f"[Task {task_id}] ❌ ATOMIC OPERATION FAILED - ROLLED BACK: {str(e)}"
        )
        
        # Re-raise with context
        raise AtomicOperationError(
            f"Failed to store and mark current for step {step_name}: {str(e)}"
        )


# ============================================================================
# FIX #2.1: CHUNK CHECKPOINT FUNCTIONS FOR RECOVERY
# ============================================================================

async def save_chunk_checkpoint(
    task_id: str,
    session_id: Optional[str],
    step_name: str,
    last_chunk_id: int,
    total_chunks: Optional[int],
    progress_pct: int,
    db
) -> None:
    """
    Save chunk processing checkpoint for recovery if process crashes.
    
    Call this after successfully processing and persisting each chunk.
    
    Args:
        task_id: Task ID
        session_id: Associated session_id (optional)
        step_name: Step being processed (technical, signals, etc.)
        last_chunk_id: ID of last successfully processed chunk
        total_chunks: Total chunks expected (if known)
        progress_pct: Progress 0-100
        db: Async DB session
    """
    try:
        from app.database.models import ChunkCheckpoint
        
        stmt = pg_insert(ChunkCheckpoint).values(
            task_id=task_id,
            session_id=session_id,
            step_name=step_name,
            last_successful_chunk_id=last_chunk_id,
            total_chunks_expected=total_chunks,
            chunks_processed=last_chunk_id + 1,  # 0-indexed to 1-indexed
            progress_percentage=progress_pct,
            last_checkpoint_time=datetime.utcnow(),
        ).on_conflict_do_update(
            index_elements=['task_id'],
            set_=dict(
                last_successful_chunk_id=last_chunk_id,
                chunks_processed=last_chunk_id + 1,
                progress_percentage=progress_pct,
                last_checkpoint_time=datetime.utcnow(),
                last_error=None,  # Clear error on success
            )
        )
        
        await db.execute(stmt)
        await db.commit()
        logger.debug(f"✅ Saved checkpoint: task={task_id}, chunk={last_chunk_id}/{total_chunks}, progress={progress_pct}%")
        
    except Exception as e:
        logger.error(f"❌ Failed to save chunk checkpoint for {task_id}: {e}")
        await db.rollback()
        # Don't raise - checkpoint failure shouldn't block processing


async def get_chunk_checkpoint(task_id: str, db) -> Optional[Dict[str, Any]]:
    """
    Retrieve last checkpoint for task to resume processing.
    
    Returns:
        Dict with checkpoint data or None if not found
        Example: {
            'task_id': 'xyz',
            'last_chunk_id': 42,
            'total_chunks': 100,
            'progress_pct': 43,
            'retry_count': 0
        }
    """
    try:
        from app.database.models import ChunkCheckpoint
        
        stmt = select(ChunkCheckpoint).where(
            ChunkCheckpoint.task_id == task_id
        )
        result = await db.execute(stmt)
        checkpoint = result.scalar_one_or_none()
        
        if checkpoint:
            logger.info(
                f"✅ Found checkpoint for {task_id}: "
                f"resume from chunk {checkpoint.last_successful_chunk_id + 1}, "
                f"progress={checkpoint.progress_percentage}%"
            )
            return {
                'task_id': checkpoint.task_id,
                'last_chunk_id': checkpoint.last_successful_chunk_id,
                'total_chunks': checkpoint.total_chunks_expected,
                'progress_pct': checkpoint.progress_percentage,
                'retry_count': checkpoint.retry_count,
            }
        else:
            logger.debug(f"No checkpoint found for {task_id} - starting from beginning")
            return None
            
    except Exception as e:
        logger.warning(f"⚠️ Failed to retrieve checkpoint for {task_id}: {e}")
        return None


async def record_chunk_error(
    task_id: str,
    error_message: str,
    db
) -> None:
    """
    Record error and increment retry count for failed chunk processing.
    """
    try:
        from app.database.models import ChunkCheckpoint
        
        # Get current checkpoint
        stmt = select(ChunkCheckpoint).where(ChunkCheckpoint.task_id == task_id)
        result = await db.execute(stmt)
        checkpoint = result.scalar_one_or_none()
        
        if checkpoint:
            # Update with error
            checkpoint.last_error = error_message[:500]  # Truncate to 500 chars
            checkpoint.retry_count += 1
            await db.execute(
                pg_insert(ChunkCheckpoint).values(
                    task_id=task_id,
                    last_error=error_message[:500],
                    retry_count=checkpoint.retry_count + 1,
                ).on_conflict_do_update(
                    index_elements=['task_id'],
                    set_=dict(
                        last_error=error_message[:500],
                        retry_count=checkpoint.retry_count + 1,
                    )
                )
            )
            await db.commit()
            logger.warning(
                f"⚠️ Recorded error for {task_id}: {error_message[:100]}... "
                f"(retry {checkpoint.retry_count + 1}/{checkpoint.max_retries})"
            )
    except Exception as e:
        logger.error(f"❌ Failed to record error for {task_id}: {e}")
        await db.rollback()


async def cleanup_old_checkpoints(older_than_hours: int = 24, db=None) -> int:
    """
    Clean up old completed checkpoints to save DB space.
    
    Returns: Number of checkpoints deleted
    """
    try:
        from app.database.models import ChunkCheckpoint
        from datetime import timedelta
        
        cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
        
        stmt = select(ChunkCheckpoint).where(
            ChunkCheckpoint.last_checkpoint_time < cutoff_time
        )
        result = await db.execute(stmt)
        old_checkpoints = result.scalars().all()
        
        delete_count = 0
        for checkpoint in old_checkpoints:
            await db.delete(checkpoint)
            delete_count += 1
        
        await db.commit()
        logger.info(f"✅ Cleaned up {delete_count} old checkpoints (older than {older_than_hours}h)")
        return delete_count
        
    except Exception as e:
        logger.error(f"❌ Failed to cleanup old checkpoints: {e}")
        await db.rollback()
        return 0


# ============================================================================
# PHASE 18: MLDataset Data Access Layer
# ============================================================================

async def create_ml_dataset(
    session_id: str,
    dataset_name: str,
    output_targets: List[str],
    features_x,           # np.ndarray or list
    targets_y,            # np.ndarray or list
    source_step: str,
    scaling_config: Dict[str, Any],
    split_config: Dict[str, Any],
    feature_columns: List[str],
    db,
    **metadata
) -> Optional[str]:
    """
    Create and store a new MLDataset record for a session.

    Uses JSONB storage (preferred) with a LargeBinary pickle fallback for
    datasets that are too large to fit in JSONB.

    Args:
        session_id:       Session UUID string
        dataset_name:     Human-readable name, e.g. 'snr_signals__type__direction'
        output_targets:   Target column names, e.g. ['signal_type']
        features_x:       Feature matrix (numpy array or list-of-lists)
        targets_y:        Target matrix (numpy array or list)
        source_step:      Source analysis step name, e.g. 'snr_analysis'
        scaling_config:   Scaler params {mean: [...], std: [...]}
        split_config:     Split info {train_size, val_size, test_size, ...}
        feature_columns:  Column names for features
        db:               Async SQLAlchemy session
        **metadata:       Extra fields: preprocessing_steps, null_percentage, etc.

    Returns:
        dataset_id (str UUID) on success, None on failure.
    """
    import numpy as np
    import pickle as _pickle
    from app.database.models import MLDataset
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    try:
        # 1. Compute output_targets_hash for uniqueness constraint
        targets_sorted = sorted(output_targets)
        output_targets_hash = hashlib.sha256(
            json.dumps(targets_sorted).encode()
        ).hexdigest()

        # 2. Coerce to numpy if needed
        try:
            import numpy as np
            if not isinstance(features_x, np.ndarray):
                features_x = np.array(features_x, dtype=float)
            if not isinstance(targets_y, np.ndarray):
                targets_y = np.array(targets_y)
        except Exception:
            pass  # Keep as-is; serialization below handles it

        # 3. Compute data_hash for integrity
        try:
            data_hash = hashlib.sha256(
                features_x.tobytes() + targets_y.tobytes()
            ).hexdigest()
        except Exception:
            data_hash = None

        # 4. Serialize features_x / targets_y (JSONB first, LargeBinary fallback)
        features_x_jsonb = None
        targets_y_jsonb = None
        features_x_pickle = None
        targets_y_pickle = None

        try:
            features_x_jsonb = {
                "data": features_x.tolist(),
                "shape": list(features_x.shape),
                "dtype": str(features_x.dtype),
            }
            targets_y_jsonb = {
                "data": targets_y.tolist(),
                "shape": list(targets_y.shape),
                "dtype": str(targets_y.dtype),
            }
            # Quick size sanity check
            fx_size = len(json.dumps(features_x_jsonb))
            if fx_size > 100_000_000:  # 100 MB threshold → prefer binary
                raise ValueError("Too large for JSONB")
        except Exception as jsonb_err:
            logger.warning(
                f"MLDataset: JSONB fallback for '{dataset_name}': {jsonb_err}"
            )
            features_x_jsonb = None
            targets_y_jsonb = None
            features_x_pickle = _pickle.dumps(features_x)
            targets_y_pickle = _pickle.dumps(targets_y)

        # 5. Upsert MLDataset record
        stmt = pg_insert(MLDataset).values(
            session_id=session_id,
            dataset_name=dataset_name,
            output_targets=output_targets,
            output_targets_hash=output_targets_hash,
            source_step=source_step,
            source_metadata=metadata.get("source_metadata"),
            features_x=features_x_jsonb,
            targets_y=targets_y_jsonb,
            features_x_pickle=features_x_pickle,
            targets_y_pickle=targets_y_pickle,
            feature_columns=feature_columns,
            feature_count=len(feature_columns),
            sample_count=int(len(features_x)),
            scaling_config=scaling_config,
            scaling_fitted=int(metadata.get("scaling_fitted", 1)),
            split_config=split_config,
            stratification_col=metadata.get("stratification_col"),
            null_percentage=metadata.get("null_percentage"),
            class_imbalance_ratio=metadata.get("class_imbalance_ratio"),
            preprocessing_steps=metadata.get("preprocessing_steps", []),
            data_hash=data_hash,
            is_current=True,
            status="ready",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        ).on_conflict_do_update(
            constraint="uq_ml_dataset_targets_hash",
            set_=dict(
                dataset_name=dataset_name,
                features_x=features_x_jsonb,
                targets_y=targets_y_jsonb,
                features_x_pickle=features_x_pickle,
                targets_y_pickle=targets_y_pickle,
                sample_count=int(len(features_x)),
                data_hash=data_hash,
                scaling_config=scaling_config,
                split_config=split_config,
                updated_at=datetime.utcnow(),
                status="ready",
            ),
        ).returning(MLDataset.dataset_id)

        result = await db.execute(stmt)
        await db.commit()
        dataset_id = str(result.scalar_one())
        logger.info(
            f"✅ MLDataset stored: {dataset_id} "
            f"({dataset_name}, {len(features_x)} samples)"
        )
        return dataset_id

    except Exception as e:
        await db.rollback()
        logger.error(f"❌ create_ml_dataset failed: {e}", exc_info=True)
        return None


async def append_sequences_to_ml_dataset(
    dataset_id: str,
    sequences: np.ndarray,
    labels: np.ndarray,
    targets: Dict[str, np.ndarray],
    split_name: str,  # 'train', 'validation', or 'test'
    db,
    use_chunk_table: bool = True,  # Feature flag: O(n) chunks vs O(n²) blob
    sequence_metadata: list = None,  # Per-sequence viewer metadata (OHLCV snapshots etc.)
) -> bool:
    """
    Append a chunk of sequences to an existing ML dataset split.

    Two storage modes available:
    1. BLOB (use_chunk_table=False): O(n²) but simpler, backward compatible
       - Decompress → Concat → Recompress (re-fetches all previous chunks)

    2. CHUNKS (use_chunk_table=True): O(n) modern approach
       - Just compress + INSERT (no re-fetching)
       - 4 chunks: 6.5x faster than blob
       - 100 chunks: 40x faster than blob

    Args:
        dataset_id: UUID of the existing dataset
        sequences: New sequences to append (np.ndarray)
        labels: New labels to append (np.ndarray)
        targets: Dict of target arrays to append {target_name: np.ndarray}
        split_name: Which split to append to ('train', 'validation', or 'test')
        db: Async SQLAlchemy session
        use_chunk_table: If True, use new chunk table (O(n)); else blob (O(n²))
        sequence_metadata: Optional list of per-sequence dicts (OHLCV snapshots,
            anchor_index, target_structure). Viewer-only; training ignores this.
    Returns:
        True on success, False on failure
    """
    try:
        # Validate split_name
        if split_name not in ['train', 'validation', 'test']:
            logger.error(f"❌ Invalid split_name: {split_name}")
            return False
        
        # 1. Load existing dataset record
        stmt = select(MLDataset).where(MLDataset.dataset_id == dataset_id)
        result = await db.execute(stmt)
        dataset = result.scalar_one_or_none()
        
        if not dataset:
            logger.error(f"❌ Dataset {dataset_id} not found for append operation")
            return False
        
        # ========================================================================
        # ROUTE: Choose storage mode
        # ========================================================================
        if use_chunk_table or dataset.storage_mode == 'chunks':
            logger.info(f"📦 [append CHUNKS] Using O(n) chunk table for {split_name} split")
            return await _append_to_chunk_table(
                dataset=dataset,
                sequences=sequences,
                labels=labels,
                targets=targets,
                split_name=split_name,
                db=db,
                sequence_metadata=sequence_metadata,
            )
        else:
            logger.info(f"📦 [append BLOB] Using O(n²) blob for {split_name} split (backward compat)")
            return await _append_to_blob(
                dataset=dataset,
                sequences=sequences,
                labels=labels,
                targets=targets,
                split_name=split_name,
                db=db
            )
        
    except Exception as e:
        logger.error(f"❌ append_sequences_to_ml_dataset failed: {e}", exc_info=True)
        return False


async def _append_to_chunk_table(
    dataset: "MLDataset",
    sequences: np.ndarray,
    labels: np.ndarray,
    targets: Dict[str, np.ndarray],
    split_name: str,
    db,
    sequence_metadata: list = None,
) -> bool:
    """
    ✅ NEW: O(n) append using chunk table.

    Just compress and INSERT — no fetching/re-processing previous chunks.
    Each chunk row stores its global_offset (first sequence index within the split)
    so the viewer can do O(1) windowed lookups without scanning all chunks.
    """
    try:
        from app.database.models import MLDatasetChunk

        # Convert to numpy if needed
        if not isinstance(sequences, np.ndarray):
            sequences = np.array(sequences, dtype=np.float32)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        if not isinstance(targets, dict):
            targets = {"target": np.array(targets) if not isinstance(targets, np.ndarray) else targets}

        # Compress each component
        sequence_data = CompressionHandler.compress(sequences)
        if isinstance(sequence_data, tuple):
            sequence_data = sequence_data[0]

        labels_data = pickle.dumps(labels) if labels is not None else None
        targets_data = pickle.dumps(targets) if targets else None

        # Compress sequence_metadata if provided (viewer-only, nullable)
        sequence_metadata_data = None
        if sequence_metadata:
            try:
                sequence_metadata_data = pickle.dumps(sequence_metadata)
            except Exception as _e:
                logger.warning(f"[append CHUNKS] Could not pickle sequence_metadata: {_e}")

        # Determine chunk_index and global_offset from existing chunks for this split.
        # Both are derived from the same query so we do it in one round-trip.
        stmt = select(
            func.coalesce(func.max(MLDatasetChunk.chunk_index), -1).label("max_index"),
            func.coalesce(func.sum(MLDatasetChunk.sequence_count), 0).label("total_seqs"),
        ).where(
            and_(
                MLDatasetChunk.dataset_id == dataset.dataset_id,
                MLDatasetChunk.split_name == split_name,
            )
        )
        result = await db.execute(stmt)
        row = result.one()
        new_chunk_index = row.max_index + 1
        global_offset = int(row.total_seqs)  # first seq index of this new chunk

        # Create chunk record
        chunk = MLDatasetChunk(
            dataset_id=dataset.dataset_id,
            session_id=dataset.session_id,
            split_name=split_name,
            chunk_index=new_chunk_index,
            global_offset=global_offset,
            sequence_data=sequence_data,
            labels_data=labels_data,
            targets_data=targets_data,
            sequence_metadata_data=sequence_metadata_data,
            sequence_count=len(sequences),
            compression_ratio=round(len(pickle.dumps(sequences)) / max(len(sequence_data), 1), 2),
            uncompressed_size_bytes=len(pickle.dumps(sequences)),
            compressed_size_bytes=len(sequence_data),
            is_verified=True,
        )

        db.add(chunk)
        
        # Update MLDataset metadata
        update_values = {
            'sample_count': dataset.sample_count + len(sequences),
            'updated_at': datetime.utcnow(),
            'storage_mode': 'chunks',  # Mark as using chunks
        }
        
        # Update split_config
        split_config = dataset.split_config or {}
        if isinstance(split_config, str):
            split_config = json.loads(split_config)
        
        size_key = f"{split_name}_size"
        split_config[size_key] = split_config.get(size_key, 0) + len(sequences)
        update_values['split_config'] = split_config
        
        # Populate target_metadata on first chunk
        if not dataset.target_metadata and targets:
            target_names = list(targets.keys())
            target_types = {}
            target_shapes = {}
            
            for target_name, target_array in targets.items():
                if isinstance(target_array, np.ndarray):
                    if len(target_array.shape) == 1:
                        target_shapes[target_name] = "scalar"
                        unique_vals = np.unique(target_array)
                        if len(unique_vals) <= 10 and np.all(unique_vals == unique_vals.astype(int)):
                            target_types[target_name] = "classification"
                        else:
                            target_types[target_name] = "regression"
                    else:
                        target_shapes[target_name] = f"array_{target_array.shape[1:]}"
                        target_types[target_name] = "sequence_prediction"
            
            update_values['target_metadata'] = {
                "target_names": target_names,
                "target_types": target_types,
                "target_shapes": target_shapes,
                "class_mappings": {},
                "primary_target": target_names[0] if target_names else None,
                "prediction_length": None
            }
        
        update_stmt = update(MLDataset).where(
            MLDataset.dataset_id == dataset.dataset_id
        ).values(**update_values)
        
        await db.execute(update_stmt)
        await db.commit()
        
        logger.info(
            f"✅ [append CHUNKS] Stored chunk {new_chunk_index} for {split_name} split "
            f"({len(sequences)} sequences, {len(sequence_data)} bytes compressed)"
        )
        
        return True
        
    except Exception as e:
        await db.rollback()
        logger.error(f"❌ _append_to_chunk_table failed: {e}", exc_info=True)
        return False


async def _append_to_blob(
    dataset: "MLDataset",
    sequences: np.ndarray,
    labels: np.ndarray,
    targets: Dict[str, np.ndarray],
    split_name: str,
    db
) -> bool:
    """
    ❌ OLD: O(n²) append using blob (backward compatible).
    
    Decompresses entire blob, concatenates, recompresses.
    Kept for backward compatibility but deprecated.
    """
    try:
        logger.info(f"📦 [append BLOB] Loading existing {split_name} data for dataset {dataset.dataset_id}")
        
        # Map split name to specific columns
        data_col = f"{split_name}_data_compressed"
        labels_col = f"{split_name}_labels"
        targets_col = f"{split_name}_targets"
        
        # Deserialize existing data
        existing_sequences = None
        existing_labels = None
        existing_targets = None
        
        data_compressed = getattr(dataset, data_col, None)
        if data_compressed:
            existing_sequences = CompressionHandler.decompress(data_compressed)
        
        labels_bytes = getattr(dataset, labels_col, None)
        if labels_bytes:
            existing_labels = pickle.loads(labels_bytes)
        
        targets_blob = getattr(dataset, targets_col, None)
        if targets_blob:
            existing_targets = pickle.loads(targets_blob)
        else:
            existing_metadata = dataset.source_metadata or {}
            if isinstance(existing_metadata, str):
                existing_metadata = json.loads(existing_metadata)
            
            targets_key = f'{split_name}_targets'
            if targets_key in existing_metadata:
                existing_targets = {
                    k: np.array(v) if isinstance(v, list) else v
                    for k, v in existing_metadata[targets_key].items()
                }
        
        # Convert new chunk to numpy
        if not isinstance(sequences, np.ndarray):
            sequences = np.array(sequences, dtype=np.float32)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        if not isinstance(targets, dict):
            targets = {"target": np.array(targets) if not isinstance(targets, np.ndarray) else targets}
        
        # Concatenate
        if existing_sequences is not None:
            combined_sequences = np.concatenate([existing_sequences, sequences], axis=0)
            logger.info(
                f"📦 [append BLOB] Concatenated sequences: {len(existing_sequences)} + {len(sequences)} "
                f"= {len(combined_sequences)}"
            )
        else:
            combined_sequences = sequences
            logger.info(f"📦 [append BLOB] First chunk: {len(sequences)} sequences")
        
        if existing_labels is not None:
            combined_labels = np.concatenate([existing_labels, labels], axis=0)
        else:
            combined_labels = labels
        
        # Merge targets
        combined_targets_dict = {}
        if existing_targets is not None and isinstance(existing_targets, dict):
            for target_name, new_target_array in targets.items():
                if target_name in existing_targets:
                    existing_array = existing_targets[target_name]
                    if not isinstance(existing_array, np.ndarray):
                        existing_array = np.array(existing_array)
                    if not isinstance(new_target_array, np.ndarray):
                        new_target_array = np.array(new_target_array)
                    
                    combined_targets_dict[target_name] = np.concatenate(
                        [existing_array, new_target_array],
                        axis=0
                    )
                else:
                    combined_targets_dict[target_name] = new_target_array
            
            for target_name, existing_array in existing_targets.items():
                if target_name not in combined_targets_dict:
                    combined_targets_dict[target_name] = existing_array
        else:
            combined_targets_dict = targets
        
        # Free intermediates
        del existing_sequences, existing_labels, existing_targets
        gc.collect()
        
        # Compress and store
        compressed_data = CompressionHandler.compress(combined_sequences)
        if isinstance(compressed_data, tuple):
            compressed_data = compressed_data[0]
        
        update_values = {
            data_col: compressed_data,
            labels_col: pickle.dumps(combined_labels),
            targets_col: pickle.dumps(combined_targets_dict),
            'sample_count': dataset.sample_count + len(sequences),
            'updated_at': datetime.utcnow(),
            'storage_mode': 'blob',  # Mark as using blob
        }
        
        # Update split config
        split_config = dataset.split_config or {}
        if isinstance(split_config, str):
            split_config = json.loads(split_config)
        
        size_key = f"{split_name}_size"
        split_config[size_key] = split_config.get(size_key, 0) + len(sequences)
        update_values['split_config'] = split_config
        
        # Populate target_metadata on first chunk
        if not dataset.target_metadata and combined_targets_dict:
            target_names = list(combined_targets_dict.keys())
            target_types = {}
            target_shapes = {}
            
            for target_name, target_array in combined_targets_dict.items():
                if isinstance(target_array, np.ndarray):
                    if len(target_array.shape) == 1:
                        target_shapes[target_name] = "scalar"
                        unique_vals = np.unique(target_array)
                        if len(unique_vals) <= 10 and np.all(unique_vals == unique_vals.astype(int)):
                            target_types[target_name] = "classification"
                        else:
                            target_types[target_name] = "regression"
                    else:
                        target_shapes[target_name] = f"array_{target_array.shape[1:]}"
                        target_types[target_name] = "sequence_prediction"
            
            update_values['target_metadata'] = {
                "target_names": target_names,
                "target_types": target_types,
                "target_shapes": target_shapes,
                "class_mappings": {},
                "primary_target": target_names[0] if target_names else None,
                "prediction_length": None
            }
        
        update_stmt = update(MLDataset).where(
            MLDataset.dataset_id == dataset.dataset_id
        ).values(**update_values)
        
        await db.execute(update_stmt)
        await db.commit()
        
        logger.info(
            f"✅ [append BLOB] Successfully appended chunk to {split_name} split of dataset {dataset.dataset_id} "
            f"(new total: {len(combined_sequences)} sequences)"
        )
        
        # Free memory
        del combined_sequences, combined_labels, combined_targets_dict
        gc.collect()
        
        return True
        
    except Exception as e:
        await db.rollback()
        logger.error(f"❌ _append_to_blob failed for {split_name}: {e}", exc_info=True)
        return False

        
        # 2. Map split name to specific columns (Restored Design)
        data_col = f"{split_name}_data_compressed"
        labels_col = f"{split_name}_labels"
        targets_col = f"{split_name}_targets"
        
        # 3. Deserialize existing data from split-specific columns
        existing_sequences = None
        existing_labels = None
        existing_targets = None
        
        # Sequences
        data_compressed = getattr(dataset, data_col, None)
        if data_compressed:
            existing_sequences = CompressionHandler.decompress(data_compressed)
        
        # Labels
        labels_bytes = getattr(dataset, labels_col, None)
        if labels_bytes:
            existing_labels = pickle.loads(labels_bytes)
        
        # Targets (from LargeBinary blob or source_metadata fallback)
        targets_blob = getattr(dataset, targets_col, None)
        if targets_blob:
            existing_targets = pickle.loads(targets_blob)
        else:
            # Fallback to source_metadata for backward compatibility
            existing_metadata = dataset.source_metadata or {}
            if isinstance(existing_metadata, str):
                import json
                existing_metadata = json.loads(existing_metadata)
            
            targets_key = f'{split_name}_targets'
            if targets_key in existing_metadata:
                existing_targets = {
                    k: np.array(v) if isinstance(v, list) else v
                    for k, v in existing_metadata[targets_key].items()
                }
        
        # 4. Convert new chunk to numpy if needed
        if not isinstance(sequences, np.ndarray):
            sequences = np.array(sequences, dtype=np.float32)
        
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        
        if not isinstance(targets, dict):
            # If targets is a single array, wrap it
            targets = {"target": np.array(targets) if not isinstance(targets, np.ndarray) else targets}
        
        # 5. Concatenate existing + new chunk (sequences)
        if existing_sequences is not None:
            combined_sequences = np.concatenate([existing_sequences, sequences], axis=0)
            logger.info(
                f"📦 [append] Concatenated sequences: {len(existing_sequences)} + {len(sequences)} "
                f"= {len(combined_sequences)}"
            )
        else:
            combined_sequences = sequences
            logger.info(f"📦 [append] First chunk: {len(sequences)} sequences")
        
        # 6. Concatenate labels
        if existing_labels is not None:
            combined_labels = np.concatenate([existing_labels, labels], axis=0)
        else:
            combined_labels = labels
        
        # 7. Concatenate targets (dict-based)
        combined_targets_dict = {}
        
        if existing_targets is not None and isinstance(existing_targets, dict):
            # Merge existing and new targets
            for target_name, new_target_array in targets.items():
                if target_name in existing_targets:
                    existing_array = existing_targets[target_name]
                    # Convert to numpy if needed
                    if not isinstance(existing_array, np.ndarray):
                        existing_array = np.array(existing_array)
                    if not isinstance(new_target_array, np.ndarray):
                        new_target_array = np.array(new_target_array)
                    
                    combined_targets_dict[target_name] = np.concatenate(
                        [existing_array, new_target_array],
                        axis=0
                    )
                else:
                    combined_targets_dict[target_name] = new_target_array
            
            # Add any existing targets not in new chunk
            for target_name, existing_array in existing_targets.items():
                if target_name not in combined_targets_dict:
                    combined_targets_dict[target_name] = existing_array
        else:
            # First chunk or legacy format
            combined_targets_dict = targets
        
        # 8. Free intermediate objects
        del existing_sequences, existing_labels, existing_targets
        gc.collect()
        
        # 9. Store combined data in split-specific columns
        compressed_data = CompressionHandler.compress(combined_sequences)
        if isinstance(compressed_data, tuple): compressed_data = compressed_data[0]
        
        update_values = {
            data_col: compressed_data,
            labels_col: pickle.dumps(combined_labels),
            targets_col: pickle.dumps(combined_targets_dict),
            'sample_count': dataset.sample_count + len(sequences),
            'updated_at': datetime.utcnow(),
        }
        
        # 10. Update split metadata
        existing_split_config = dataset.split_config or {}
        if isinstance(existing_split_config, str): existing_split_config = json.loads(existing_split_config)
        
        size_key = f"{split_name}_size"
        existing_split_config[size_key] = existing_split_config.get(size_key, 0) + len(sequences)
        update_values['split_config'] = existing_split_config

        # 11. Populate target_metadata (if first chunk and not yet set)
        if not dataset.target_metadata and combined_targets_dict:
            target_names = list(combined_targets_dict.keys())
            target_types = {}
            target_shapes = {}
            
            # ✅ FIX #1: Analyze each target to determine its type
            for target_name, target_array in combined_targets_dict.items():
                if isinstance(target_array, np.ndarray):
                    # Determine shape
                    if len(target_array.shape) == 1:
                        target_shapes[target_name] = "scalar"
                        
                        # Determine type: classification or regression
                        unique_vals = np.unique(target_array)
                        if len(unique_vals) <= 10 and np.all(unique_vals == unique_vals.astype(int)):
                            target_types[target_name] = "classification"
                            logger.info(f"[Target] {target_name}: classification ({len(unique_vals)} classes)")
                        else:
                            target_types[target_name] = "regression"
                            logger.info(f"[Target] {target_name}: regression (continuous values)")
                    else:
                        # Multi-dimensional: sequence prediction
                        target_shapes[target_name] = f"array_{target_array.shape[1:]}"
                        target_types[target_name] = "sequence_prediction"
                        logger.info(f"[Target] {target_name}: sequence_prediction (shape: {target_array.shape})")
                else:
                    target_types[target_name] = "unknown"
                    target_shapes[target_name] = "unknown"
            
            update_values['target_metadata'] = {
                "target_names": target_names,
                "target_types": target_types,  # ✅ NOW POPULATED!
                "target_shapes": target_shapes,  # ✅ NEW!
                "class_mappings": {
                    "signal_type": {
                        0: "bounce_support",
                        1: "bounce_resistance",
                        2: "breakout_support",
                        3: "breakout_resistance",
                        4: "no_signal"
                    }
                },
                "primary_target": target_names[0] if target_names else None,
                "prediction_length": None  # Derived from actual target shape at training time
            }
            
            logger.info(f"[Target Metadata] Populated for {len(target_names)} targets: {target_types}")
        
        update_stmt = update(MLDataset).where(
            MLDataset.dataset_id == dataset_id
        ).values(**update_values)
        
        await db.execute(update_stmt)
        await db.commit()
        
        logger.info(
            f"✅ [append] Successfully appended chunk to {split_name} split of dataset {dataset_id} "
            f"(new total: {len(combined_sequences)} sequences)"
        )
        
        # 11. Free memory
        del combined_sequences, combined_labels, combined_targets_dict
        gc.collect()
        
        return True
        
    except Exception as e:
        await db.rollback()
        logger.error(f"❌ append_sequences_to_ml_dataset failed for {split_name}: {e}", exc_info=True)
        return False


def _extract_prediction_length(source_meta: dict, split_config_raw) -> int | None:
    """
    Resolve prediction_length from the two places it may be stored.

    Priority:
    1. source_metadata.prediction_length  (set by analysis_manager during ML prep)
    2. split_config.prediction_length     (set by the dataset builder)
    3. None — caller must derive from actual target shape (backend auto-corrects)

    Never falls back to a hardcoded default so the backend's shape-validation
    logic in execute_model_build() can correct it from real data.
    """
    # 1. source_metadata
    val = source_meta.get("prediction_length")
    if val and int(val) > 0:
        return int(val)

    # 2. split_config
    if split_config_raw:
        sc = safe_json_loads(split_config_raw) if isinstance(split_config_raw, str) else (split_config_raw or {})
        val = sc.get("prediction_length")
        if val and int(val) > 0:
            return int(val)

    return None


async def get_ml_datasets_for_session(
    session_id: str,
    db,
) -> List[Dict[str, Any]]:
    """
    List all ready MLDatasets for a session (lightweight — no feature data).

    Returns:
        List of dataset metadata dicts suitable for dropdowns.
    
    TRACE: Logs dataset retrieval with step_configs and target information for frontend
    """
    from sqlalchemy import text
    import json

    try:
        logger.info(
            f"📚 [get_ml_datasets_for_session] STARTING QUERY\n"
            f"   • session_id: {session_id}"
        )

        # Query the ml_datasets table directly using the schema from phase21 migration
        query = text("""
            SELECT 
                dataset_id, session_id, dataset_name, source_step,
                output_targets, feature_count, sample_count, feature_columns,
                split_config, source_metadata, target_metadata,
                created_at, compression_ratio, compressed_size_mb
            FROM ml_datasets 
            WHERE session_id = :session_id
            ORDER BY created_at DESC
        """)
        
        result = await db.execute(query, {'session_id': session_id})
        rows = result.fetchall()

        logger.debug(
            f"🔄 [get_ml_datasets_for_session] DATABASE QUERY COMPLETE\n"
            f"   • Total rows returned: {len(rows)}"
        )

        def safe_json_loads(json_str):
            """Safely deserialize JSON strings, handling None and malformed JSON."""
            if not json_str:
                return None
            try:
                return json.loads(json_str)
            except (json.JSONDecodeError, TypeError):
                return json_str

        import hashlib

        def compute_feature_hash(feature_cols: list) -> str:
            """Stable SHA-256 of sorted feature column names — used for model compatibility checks."""
            if not feature_cols:
                return ""
            canonical = sorted(str(f) for f in feature_cols)
            return hashlib.sha256(",".join(canonical).encode()).hexdigest()[:16]

        datasets = []
        for idx, row in enumerate(rows):
            feature_cols = safe_json_loads(row.feature_columns) if hasattr(row, 'feature_columns') and row.feature_columns else [f"feature_{i}" for i in range(row.feature_count or 0)]
            source_meta = safe_json_loads(row.source_metadata) or {} if row.source_metadata else {}
            output_targets = safe_json_loads(row.output_targets) if row.output_targets else []
            target_metadata = safe_json_loads(row.target_metadata) if row.target_metadata else {}
            split_config = safe_json_loads(row.split_config) if row.split_config else {}
            step_configs = source_meta.get("step_configs", {})
            
            logger.debug(
                f"📋 [get_ml_datasets_for_session] PROCESSING ROW [{idx + 1}/{len(rows)}]\n"
                f"   • dataset_id: {row.dataset_id}\n"
                f"   • dataset_name: {row.dataset_name}\n"
                f"   • source_step: {row.source_step}\n"
                f"   • output_targets (base columns): {output_targets}\n"
                f"   • feature_count: {row.feature_count}\n"
                f"   • sample_count: {row.sample_count}\n"
                f"   • sequence_length: {source_meta.get('sequence_length', 60)}\n"
                f"   • step_configs keys: {list(step_configs.keys())}\n"
                f"   • prepare_advanced_ml_targets: {step_configs.get('prepare_advanced_ml_targets', False)}\n"
                f"   • split_config keys: {list(split_config.keys())}"
            )
            
            dataset_dict = {
                "dataset_id": str(row.dataset_id),
                "session_id": session_id,
                "dataset_name": row.dataset_name,
                # ✅ CRITICAL: Use actual feature_columns from database
                "feature_columns": feature_cols,
                "feature_hash": compute_feature_hash(feature_cols),
                "output_targets": output_targets,
                "target_metadata": target_metadata,
                "sample_count": row.sample_count or 0,
                "created_at": row.created_at if row.created_at else None,
                "feature_count": row.feature_count or 0,
                "source_step": row.source_step,
                "compression_ratio": float(row.compression_ratio) if row.compression_ratio else 0.0,
                "compressed_size_mb": float(row.compressed_size_mb) if row.compressed_size_mb else 0.0,
                # Extract from source_metadata (stored by analysis_manager)
                "sequence_length": source_meta.get("sequence_length", 60),
                # prediction_length: prefer source_metadata, fall back to split_config.
                # Never hardcode 7 — let the backend auto-correct if still wrong.
                "prediction_length": _extract_prediction_length(source_meta, row.split_config),
                # Step configs that produced this dataset (for inference provenance)
                "step_configs": step_configs,
                # Extract split sizes from split_config
                "split_config": split_config,
            }
            
            datasets.append(dataset_dict)
            
            logger.debug(
                f"✅ [get_ml_datasets_for_session] DATASET ADDED TO RESPONSE\n"
                f"   • dataset_id: {dataset_dict['dataset_id']}\n"
                f"   • dataset_name: {dataset_dict['dataset_name']}\n"
                f"   • output_targets in dict: {dataset_dict['output_targets']}\n"
                f"   • step_configs in dict: {dataset_dict['step_configs']}"
            )

        logger.info(
            f"✅ [get_ml_datasets_for_session] COMPLETE\n"
            f"   • Total datasets retrieved: {len(datasets)}\n"
            f"   • Dataset names: {[d['dataset_name'] for d in datasets]}"
        )

        # Debug logging for first dataset
        if datasets and len(datasets) > 0:
            first = datasets[0]
            logger.debug(
                f"📊 [get_ml_datasets_for_session] FIRST DATASET STRUCTURE\n"
                f"   • dataset_id: {first['dataset_id']}\n"
                f"   • dataset_name: {first['dataset_name']}\n"
                f"   • output_targets: {first['output_targets']}\n"
                f"   • step_configs: {first['step_configs']}\n"
                f"   • feature_count: {first['feature_count']}\n"
                f"   • sequence_length: {first['sequence_length']}\n"
                f"   • split_config keys: {list(first['split_config'].keys())}"
            )
        
        return datasets
    except Exception as e:
        logger.error(f"❌ [get_ml_datasets_for_session] FAILED: {e}", exc_info=True)
        return []


async def get_ml_dataset(
    dataset_id: str,
    db,
) -> Optional[tuple]:
    """
    Retrieve a complete MLDataset for model training.

    Returns:
        (features_x: np.ndarray, targets_y: np.ndarray, metadata: dict) or None.
    """
    import numpy as np
    import pickle as _pickle
    from app.database.models import MLDataset
    from sqlalchemy import select

    try:
        stmt = select(MLDataset).where(MLDataset.dataset_id == dataset_id)
        result = await db.execute(stmt)
        dataset = result.scalar_one_or_none()

        if not dataset:
            logger.warning(f"MLDataset not found: {dataset_id}")
            return None

        # Load features_x
        if dataset.features_x:
            features_x = np.array(dataset.features_x["data"])
        elif dataset.features_x_pickle:
            features_x = _pickle.loads(dataset.features_x_pickle)
        else:
            raise ValueError(f"MLDataset {dataset_id} has no feature data")

        # Load targets_y
        if dataset.targets_y:
            targets_y = np.array(dataset.targets_y["data"])
        elif dataset.targets_y_pickle:
            targets_y = _pickle.loads(dataset.targets_y_pickle)
        else:
            raise ValueError(f"MLDataset {dataset_id} has no target data")

        meta = {
            "dataset_id": str(dataset.dataset_id),
            "dataset_name": dataset.dataset_name,
            "output_targets": dataset.output_targets,
            "feature_columns": dataset.feature_columns,
            "scaling_config": dataset.scaling_config,
            "split_config": dataset.split_config,
            "source_step": dataset.source_step,
            "sample_count": dataset.sample_count,
            "feature_count": dataset.feature_count,
        }

        logger.info(
            f"✅ MLDataset loaded: {dataset_id} "
            f"({dataset.sample_count} samples, {dataset.feature_count} features)"
        )
        return features_x, targets_y, meta

    except Exception as e:
        logger.error(f"❌ get_ml_dataset failed for {dataset_id}: {e}", exc_info=True)
        return None


# ============================================================================
# PHASE 19: TRAINED MODEL MANAGEMENT (Multi-Model Architecture)
# ============================================================================

async def create_trained_model(
    dataset_id: Union[str, uuid.UUID],
    session_id: str,
    model_name: str,
    architecture_config: Dict[str, Any],
    training_config: Dict[str, Any],
    metrics: Dict[str, Any],
    val_accuracy: float,
    model_binary: Optional[bytes] = None,
    scaler_binary: Optional[bytes] = None,
    db=None,
    **kwargs
) -> uuid.UUID:
    """
    Create a new trained model record linked to a dataset.
    Increments version automatically if same name exists for dataset.
    """
    # Check for existing versions to increment
    stmt = (
        select(func.max(TrainedModelForAnalysis.version))
        .where(
            and_(
                TrainedModelForAnalysis.dataset_id == dataset_id,
                TrainedModelForAnalysis.model_name == model_name
            )
        )
    )
    result = await db.execute(stmt)
    max_version = result.scalar() or 0
    new_version = max_version + 1
    
    model = TrainedModelForAnalysis(
        dataset_id=dataset_id,
        session_id=session_id,
        model_name=model_name,
        version=new_version,
        architecture_config=architecture_config,
        training_config=training_config,
        metrics=metrics,
        val_accuracy=val_accuracy,
        model_binary=model_binary,
        scaler_binary=scaler_binary,
        is_best_model=kwargs.get("is_best_model", False),
        training_time_ms=kwargs.get("training_time_ms"),
    )
    
    db.add(model)
    await db.flush()
    await db.refresh(model)
    
    logger.info(
        f"Created TrainedModel {model.model_id} (v{new_version}) "
        f"for dataset {dataset_id}"
    )
    return model.model_id


async def get_trained_models_for_dataset(
    dataset_id: Union[str, uuid.UUID], 
    db
) -> List[Dict[str, Any]]:
    """
    List all trained models for a specific dataset (metadata only).
    Does NOT return large binary weights.
    """
    stmt = (
        select(
            TrainedModelForAnalysis.model_id,
            TrainedModelForAnalysis.dataset_id,
            TrainedModelForAnalysis.model_name,
            TrainedModelForAnalysis.version,
            TrainedModelForAnalysis.metrics,
            TrainedModelForAnalysis.val_accuracy,
            TrainedModelForAnalysis.is_best_model,
            TrainedModelForAnalysis.created_at
        )
        .where(TrainedModelForAnalysis.dataset_id == dataset_id)
        .order_by(TrainedModelForAnalysis.val_accuracy.desc())
    )
    result = await db.execute(stmt)
    
    models = []
    for row in result:
        models.append({
            "model_id": str(row.model_id),
            "dataset_id": str(row.dataset_id),
            "model_name": row.model_name,
            "version": row.version,
            "metrics": row.metrics,
            "val_accuracy": float(row.val_accuracy) if row.val_accuracy else None,
            "is_best_model": row.is_best_model,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        })
    return models


async def get_trained_model(
    model_id: Union[str, uuid.UUID], 
    db
) -> Optional[Dict[str, Any]]:
    """
    Retrieve full trained model data, including serialized weights/scalers.
    """
    stmt = select(TrainedModelForAnalysis).where(TrainedModelForAnalysis.model_id == model_id)
    result = await db.execute(stmt)
    model = result.scalar_one_or_none()
    
    if not model:
        return None
        
    return {
        "model_id": str(model.model_id),
        "dataset_id": str(model.dataset_id),
        "session_id": model.session_id,
        "model_name": model.model_name,
        "version": model.version,
        "architecture_config": model.architecture_config,
        "training_config": model.training_config,
        "metrics": model.metrics,
        "val_accuracy": float(model.val_accuracy) if model.val_accuracy else None,
        "model_binary": model.model_binary,
        "scaler_binary": model.scaler_binary,
        "is_best_model": model.is_best_model,
        "created_at": model.created_at.isoformat(),
        "updated_at": model.updated_at.isoformat(),
    }


async def mark_best_model(
    model_id: Union[str, uuid.UUID], 
    db
) -> bool:
    """
    Set 1 model as 'best' per dataset, clearing others.
    """
    # 1. Get the dataset_id for this model
    stmt = select(TrainedModelForAnalysis.dataset_id).where(TrainedModelForAnalysis.model_id == model_id)
    res = await db.execute(stmt)
    dataset_id = res.scalar()
    
    if not dataset_id:
        return False
        
    # 2. Reset all models for this dataset
    await db.execute(
        update(TrainedModelForAnalysis)
        .where(TrainedModelForAnalysis.dataset_id == dataset_id)
        .values(is_best_model=False)
    )
    
    # 3. Set this specific model as best
    await db.execute(
        update(TrainedModelForAnalysis)
        .where(TrainedModelForAnalysis.model_id == model_id)
        .values(is_best_model=True)
    )
    
    await db.flush()
    return True


async def get_ml_dataset_targets_shape(
    session_id: str,
    dataset_name: str,
    db,
    split_type: str = "train",
) -> Optional[tuple]:
    """
    ✅ NEW: Get the targets shape for a dataset split.
    
    Used to determine the correct output layer size for model building.
    For autoencoders, targets shape is (N, sequence_length, features).
    
    Args:
        session_id: Session ID
        dataset_name: Dataset name (e.g., "ml_raw_20260418_256")
        db: Database connection
        split_type: "train", "validation", or "test"
    
    Returns:
        Tuple of target shape (e.g., (3433, 60, 14)) or None if not found
    """
    import pickle
    from sqlalchemy import text
    
    try:
        logger.info(f"🔍 [get_ml_dataset_targets_shape] Looking for targets: session={session_id[:8]}, name={dataset_name}, split={split_type}")
        
        # Query ml_datasets table for targets
        query = text("""
            SELECT train_targets, validation_targets, test_targets
            FROM ml_datasets
            WHERE session_id = :session_id AND dataset_name = :dataset_name
        """)
        
        result = await db.execute(query, {
            'session_id': session_id,
            'dataset_name': dataset_name
        })
        row = result.fetchone()
        
        if not row:
            logger.warning(f"⚠️ Dataset not found: session={session_id[:8]}, dataset_name={dataset_name}")
            return None
        
        # Map split_type to the correct column
        split_map = {
            'train': row.train_targets,
            'validation': row.validation_targets,
            'test': row.test_targets,
        }
        
        if split_type not in split_map:
            logger.error(f"❌ Invalid split_type: {split_type}")
            return None
        
        targets_bytes = split_map[split_type]
        
        if not targets_bytes:
            logger.warning(f"⚠️ No targets for {split_type} split")
            return None
        
        # Deserialize targets dict
        try:
            targets_dict = pickle.loads(targets_bytes)
            if isinstance(targets_dict, dict):
                # Get first target's shape (usually only one target)
                first_target = next(iter(targets_dict.values()))
                targets_shape = first_target.shape
                logger.info(f"✅ Targets shape: {targets_shape}")
                return targets_shape
            else:
                logger.warning(f"⚠️ Targets is not a dict: {type(targets_dict)}")
                return None
        except Exception as e:
            logger.error(f"❌ Failed to deserialize targets: {e}")
            return None
        
    except Exception as e:
        logger.error(f"❌ Failed to get targets shape: {e}", exc_info=True)
        return None


async def get_ml_dataset_metadata(
    session_id: str,
    dataset_name: str,
    db
) -> Optional[Dict[str, Any]]:
    """
    Retrieve only metadata for an ML dataset (no heavy blobs).
    Useful for model building where only shapes are needed.
    """
    from sqlalchemy import text
    try:
        query = text("""
            SELECT 
                dataset_id, dataset_name, feature_count, sample_count,
                split_config, target_metadata, feature_columns, output_targets,
                features_x, targets_y
            FROM ml_datasets
            WHERE session_id = :session_id AND dataset_name = :dataset_name
        """)
        
        result = await db.execute(query, {
            'session_id': session_id,
            'dataset_name': dataset_name
        })
        row = result.fetchone()
        
        if not row:
            return None
            
        return {
            "dataset_id": str(row.dataset_id),
            "dataset_name": row.dataset_name,
            "feature_count": row.feature_count,
            "sample_count": row.sample_count,
            "split_config": row.split_config or {},
            "target_metadata": row.target_metadata or {},
            "feature_columns": row.feature_columns or [],
            "output_targets": row.output_targets or [],
            "sample_x": row.features_x,
            "sample_y": row.targets_y
        }
    except Exception as e:
        logger.error(f"❌ Error fetching ML dataset metadata: {e}")
        return None


async def _load_from_chunks(
    dataset_id: str,
    session_id: str,
    split_type: str,
    db
) -> Optional[Dict[str, Any]]:
    """
    Load ML dataset from chunk table (O(n) storage approach).
    
    Reconstructs the full split by loading and concatenating chunks.
    Compatible with lazy loading - just concatenates and returns arrays.
    """
    try:
        logger.info(f"📦 [_load_from_chunks] Loading {split_type} split from chunks for dataset {dataset_id}")
        
        # Query all chunks for this split, ordered by index
        stmt = select(MLDatasetChunk).where(
            and_(
                MLDatasetChunk.dataset_id == dataset_id,
                MLDatasetChunk.split_name == split_type
            )
        ).order_by(MLDatasetChunk.chunk_index)
        
        result = await db.execute(stmt)
        chunks = result.scalars().all()
        
        if not chunks:
            logger.warning(f"⚠️ No chunks found for {split_type} split")
            return None
        
        logger.info(f"✅ Found {len(chunks)} chunks for {split_type} split")
        
        # Decompress and concatenate all chunks
        all_sequences = []
        all_labels = []
        all_targets = {}
        
        for i, chunk in enumerate(chunks):
            try:
                # Decompress sequence data
                seq_data = CompressionHandler.decompress(chunk.sequence_data)
                all_sequences.append(seq_data)
                
                # Decompress labels if present
                if chunk.labels_data:
                    labels_data = pickle.loads(chunk.labels_data)
                    all_labels.append(labels_data)
                
                # Decompress targets if present
                if chunk.targets_data:
                    targets_data = pickle.loads(chunk.targets_data)
                    for target_name, target_array in targets_data.items():
                        if target_name not in all_targets:
                            all_targets[target_name] = []
                        all_targets[target_name].append(target_array)
                
                logger.info(f"  ✓ Chunk {i}: {len(seq_data)} sequences decompressed")
                
            except Exception as chunk_err:
                logger.error(f"❌ Failed to decompress chunk {i}: {chunk_err}")
                continue
        
        # Concatenate all chunks
        if all_sequences:
            combined_sequences = np.concatenate(all_sequences, axis=0)
            logger.info(f"✅ Concatenated {len(chunks)} chunks: {combined_sequences.shape[0]} total sequences")
        else:
            logger.warning(f"⚠️ No sequences found in chunks")
            return None
        
        # Concatenate labels
        combined_labels = None
        if all_labels:
            combined_labels = np.concatenate(all_labels, axis=0)
            logger.info(f"✅ Concatenated labels: {combined_labels.shape}")
        
        # Concatenate targets
        combined_targets = {}
        for target_name, target_arrays in all_targets.items():
            combined_targets[target_name] = np.concatenate(target_arrays, axis=0)
            logger.info(f"✅ Concatenated {target_name}: {combined_targets[target_name].shape}")
        
        # Return in same format as blob-based retrieval
        return {
            "sequences": combined_sequences,
            "labels": combined_labels,
            "targets": combined_targets,
        }

    except Exception as e:
        logger.error(f"❌ _load_from_chunks failed: {e}", exc_info=True)
        return None


async def _spool_from_chunks(
    dataset_id: str,
    dataset_name: str,
    split_type: str,
    dataset_cache_dir: str,
    db
) -> List[str]:
    """
    Spool chunks from DB directly to disk to prevent RAM exhaustion.
    Returns list of spooled .npz file paths.
    """
    import os
    import numpy as np
    import pickle
    from app.core.data.session_dataset_registry import CompressionHandler
    from app.database.models import MLDatasetChunk
    from sqlalchemy import select

    try:
        os.makedirs(dataset_cache_dir, exist_ok=True)
        
        # Query all chunks for this split, ordered by index
        stmt = select(MLDatasetChunk).where(
            and_(
                MLDatasetChunk.dataset_id == dataset_id,
                MLDatasetChunk.split_name == split_type
            )
        ).order_by(MLDatasetChunk.chunk_index)
        
        result = await db.execute(stmt)
        chunks = result.scalars().all()
        
        if not chunks:
            logger.warning(f"⚠️ No chunks found for spooling {split_type} split")
            return []
            
        logger.info(f"🔄 [Lazy Spooling] Spooling {len(chunks)} chunks for {dataset_name}/{split_type} to disk...")
        spooled_paths = []
        
        for i, chunk in enumerate(chunks):
            chunk_path = os.path.join(dataset_cache_dir, f"chunk_{i:04d}.npz")
            
            # Skip if already exists (O(1) resume)
            if os.path.exists(chunk_path):
                spooled_paths.append(chunk_path)
                continue
                
            # Decompress sequence data
            seq_data = CompressionHandler.decompress(chunk.sequence_data)
            
            # Prepare targets if present
            targets_dict = None
            if chunk.targets_data:
                targets_dict = pickle.loads(chunk.targets_data)
            
            # Save as NPZ using keys expected by LazySequenceGenerator
            if targets_dict:
                # Merge target dict into flat savez arguments
                save_args = {"sequences": seq_data}
                for tname, tarr in targets_dict.items():
                    save_args[f"target_{tname}"] = tarr
                # Also provide 'targets' for backward compat
                save_args["targets"] = targets_dict
                np.savez_compressed(chunk_path, **save_args)
            else:
                np.savez_compressed(chunk_path, sequences=seq_data)
                
            spooled_paths.append(chunk_path)
            logger.info(f"  ✓ Spooled chunk {i}: {len(seq_data)} sequences")
            
            # Aggressive GC to keep memory low during spooling
            del seq_data, targets_dict
            if i % 5 == 0: gc.collect()

        return spooled_paths

    except Exception as e:
        logger.error(f"❌ _spool_from_chunks failed: {e}", exc_info=True)
        return []


async def get_chunk_window(
    dataset_id: str,
    split_name: str,
    offset: int,
    limit: int,
    db,
) -> Optional[Dict[str, Any]]:
    """
    O(1) windowed read from the chunk table.

    Uses global_offset to find only the chunk(s) that overlap [offset, offset+limit).
    Decompresses only those chunks — never touches the rest of the dataset.

    Returns a dict with the same shape as get_dataset_sequences expects:
        sequences, labels, targets, sequence_metadata, total
    where total is the sum of sequence_count across ALL chunks for this split
    (read from the header row, not by scanning chunks).
    """
    from app.database.models import MLDatasetChunk, MLDataset

    try:
        end = offset + limit

        # 1. Get total sequence count for this split from the header row.
        #    split_config stores {train_size, validation_size, test_size}.
        header_stmt = select(MLDataset.split_config).where(
            MLDataset.dataset_id == dataset_id
        )
        header_result = await db.execute(header_stmt)
        split_config = header_result.scalar_one_or_none() or {}
        if isinstance(split_config, str):
            import json as _json
            split_config = _json.loads(split_config)

        size_key = f"{split_name}_size"
        total = int(split_config.get(size_key, 0))

        # ✅ FIX: For legacy datasets whose split_config was written before the
        # size-key update (only ratios stored, not actual counts), fall back to
        # summing sequence_count across all chunk rows for this split.
        # One aggregate query — no blob decompression needed.
        if total == 0:
            from sqlalchemy import func as _func
            count_stmt = select(
                _func.coalesce(_func.sum(MLDatasetChunk.sequence_count), 0)
            ).where(
                and_(
                    MLDatasetChunk.dataset_id == dataset_id,
                    MLDatasetChunk.split_name == split_name,
                )
            )
            count_result = await db.execute(count_stmt)
            total = int(count_result.scalar() or 0)
            if total > 0:
                logger.info(
                    f"[get_chunk_window] split_config missing '{size_key}'; "
                    f"derived total={total} by summing chunk rows"
                )

        # 2. Fetch only the chunks whose range overlaps [offset, end).
        #    A chunk overlaps when:
        #      global_offset < end  AND  global_offset + sequence_count > offset
        stmt = (
            select(MLDatasetChunk)
            .where(
                and_(
                    MLDatasetChunk.dataset_id == dataset_id,
                    MLDatasetChunk.split_name == split_name,
                    MLDatasetChunk.global_offset < end,
                    (MLDatasetChunk.global_offset + MLDatasetChunk.sequence_count) > offset,
                )
            )
            .order_by(MLDatasetChunk.chunk_index)
        )
        result = await db.execute(stmt)
        chunks = result.scalars().all()

        if not chunks:
            logger.warning(
                f"[get_chunk_window] No chunks overlap [{offset},{end}) "
                f"for dataset={dataset_id} split={split_name}"
            )
            return {"sequences": [], "labels": [], "targets": {}, "sequence_metadata": [], "total": total}

        logger.info(
            f"[get_chunk_window] {len(chunks)} chunk(s) cover [{offset},{end}) "
            f"for {split_name} (total={total})"
        )
        # Log chunk inventory so caller can see what we are working with
        for _ci, _ch in enumerate(chunks):
            logger.info(
                f"  📦 chunk[{_ci}] index={_ch.chunk_index} "
                f"global_offset={_ch.global_offset} "
                f"sequence_count={_ch.sequence_count} "
                f"compressed_bytes={_ch.compressed_size_bytes} "
                f"has_labels={_ch.labels_data is not None} "
                f"has_targets={_ch.targets_data is not None} "
                f"has_seq_meta={_ch.sequence_metadata_data is not None}"
            )

        # 3. Decompress and slice each chunk to the requested window.
        out_sequences: list = []
        out_labels: list = []
        out_targets: Dict[str, list] = {}
        out_predictions: Dict[str, list] = {}
        out_metadata: list = []

        for chunk in chunks:
            chunk_start = chunk.global_offset
            chunk_end = chunk_start + chunk.sequence_count

            # Local slice within this chunk
            local_start = max(0, offset - chunk_start)
            local_end = min(chunk.sequence_count, end - chunk_start)
            slice_len = local_end - local_start

            logger.info(
                f"  🔬 chunk[{chunk.chunk_index}] "
                f"global=[{chunk_start},{chunk_end}) "
                f"local_slice=[{local_start},{local_end}) "
                f"→ {slice_len} sequences"
            )

            # ── Decompress sequences ─────────────────────────────────────
            seq_np = CompressionHandler.decompress(chunk.sequence_data)
            logger.info(
                f"    sequences: raw_shape={seq_np.shape} dtype={seq_np.dtype} "
                f"min={float(seq_np.min()):.6f} max={float(seq_np.max()):.6f} "
                f"mean={float(seq_np.mean()):.6f} std={float(seq_np.std()):.6f}"
            )
            seq_slice = seq_np[local_start:local_end]
            out_sequences.extend(seq_slice.tolist())
            logger.info(
                f"    sequences sliced: shape={seq_slice.shape} "
                f"min={float(seq_slice.min()):.6f} max={float(seq_slice.max()):.6f}"
            )

            # ── Decompress labels ────────────────────────────────────────
            if chunk.labels_data:
                labels_np = pickle.loads(chunk.labels_data)
                lbl_slice = labels_np[local_start:local_end]
                if hasattr(lbl_slice, "tolist"):
                    lbl_list = lbl_slice.tolist()
                else:
                    lbl_list = list(lbl_slice)
                out_labels.extend(lbl_list)

                # Distribution summary
                try:
                    import numpy as _np
                    unique_lbls, counts = _np.unique(
                        labels_np if hasattr(labels_np, "__iter__") else [labels_np],
                        return_counts=True
                    )
                    dist_str = ", ".join(
                        f"{int(lv)}:{int(ct)}" for lv, ct in zip(unique_lbls, counts)
                    )
                    logger.info(
                        f"    labels: full_shape={labels_np.shape if hasattr(labels_np,'shape') else len(labels_np)} "
                        f"dtype={labels_np.dtype if hasattr(labels_np,'dtype') else type(labels_np).__name__} "
                        f"distribution=[{dist_str}] "
                        f"slice_len={len(lbl_list)} "
                        f"slice_sample={lbl_list[:5]}"
                    )
                except Exception as _le:
                    logger.info(f"    labels: slice_len={len(lbl_list)} (dist unavailable: {_le})")
            else:
                logger.info(f"    labels: None (chunk has no labels_data)")

            # ── Decompress targets ───────────────────────────────────────
            if chunk.targets_data:
                targets_dict = pickle.loads(chunk.targets_data)
                logger.info(f"    targets: keys={list(targets_dict.keys())}")
                for tname, tarr in targets_dict.items():
                    tarr_np = np.array(tarr) if not isinstance(tarr, np.ndarray) else tarr
                    slice_vals = tarr_np[local_start:local_end]
                    if tname not in out_targets:
                        out_targets[tname] = []
                    out_targets[tname].extend(
                        slice_vals.tolist() if hasattr(slice_vals, "tolist") else list(slice_vals)
                    )
                    try:
                        logger.info(
                            f"      target '{tname}': full_shape={tarr_np.shape} "
                            f"dtype={tarr_np.dtype} "
                            f"min={float(tarr_np.min()):.6f} max={float(tarr_np.max()):.6f} "
                            f"mean={float(tarr_np.mean()):.6f} "
                            f"zeros={int((tarr_np == 0).sum())} "
                            f"slice_len={len(slice_vals)} "
                            f"slice_sample={slice_vals[:3].tolist() if hasattr(slice_vals,'tolist') else list(slice_vals)[:3]}"
                        )
                    except Exception as _te:
                        logger.info(f"      target '{tname}': slice_len={len(slice_vals)} (stats unavailable: {_te})")
            else:
                logger.info(f"    targets: None (chunk has no targets_data)")

            # ── Decompress sequence_metadata (viewer-only, nullable) ─────
            if chunk.sequence_metadata_data:
                try:
                    meta_list = pickle.loads(chunk.sequence_metadata_data)
                    out_metadata.extend(meta_list[local_start:local_end])
                    logger.info(
                        f"    sequence_metadata: full_len={len(meta_list)} "
                        f"slice_len={len(meta_list[local_start:local_end])}"
                    )
                except Exception as _e:
                    logger.warning(f"[get_chunk_window] Could not unpickle sequence_metadata: {_e}")
            else:
                logger.info(f"    sequence_metadata: None (chunk has no sequence_metadata_data)")

            # ── Decompress predictions (if available) ────────────────────
            if hasattr(chunk, 'predictions_data') and chunk.predictions_data:
                try:
                    pred_dict = CompressionHandler.decompress(chunk.predictions_data)
                    chunk_predictions = pred_dict.get("predictions", [])
                    
                    # Convert to numpy for consistent slicing
                    pred_np = np.array(chunk_predictions)
                    slice_preds = pred_np[local_start:local_end]
                    
                    if "predictions" not in out_predictions:
                        out_predictions["predictions"] = []
                    
                    out_predictions["predictions"].extend(
                        slice_preds.tolist() if hasattr(slice_preds, "tolist") else list(slice_preds)
                    )
                    logger.info(f"    predictions: slice_len={len(slice_preds)}")
                except Exception as _pe:
                    logger.warning(f"[get_chunk_window] Could not decompress predictions: {_pe}")

        # ── Final window summary ─────────────────────────────────────────
        logger.info(
            f"[get_chunk_window] ✅ Window assembled: "
            f"sequences={len(out_sequences)} "
            f"labels={len(out_labels)} "
            f"targets={{{', '.join(f'{k}:{len(v)}' for k,v in out_targets.items())}}} "
            f"seq_meta={len(out_metadata)} "
            f"total={total}"
        )
        if out_sequences:
            import numpy as _np
            _s = _np.array(out_sequences[0])
            logger.info(
                f"  first_sequence: shape={_s.shape} dtype={_s.dtype} "
                f"min={float(_s.min()):.6f} max={float(_s.max()):.6f}"
            )

        return {
            "sequences": out_sequences,
            "labels": out_labels,
            "targets": out_targets,
            "predictions": out_predictions,
            "sequence_metadata": out_metadata,
            "total": total,
        }

    except Exception as e:
        logger.error(f"❌ get_chunk_window failed: {e}", exc_info=True)
        return None


async def get_ml_dataset_splits_by_name(
    session_id: str,
    dataset_name: str,
    db,
    split_type: str = "train",  # "train", "validation", or "test"
    prefer_lazy: bool = False,
) -> Optional[Union[pd.DataFrame, Dict[str, Any]]]:
    """
    ✅ CORRECT: Retrieve ML dataset splits from ml_datasets table (Phase 21 schema).
    
    Queries the ml_datasets table which stores compressed train/validation/test data.
    If prefer_lazy is True, it will spool data to disk and return file paths.
    
    Args:
        session_id: Session ID
        dataset_name: Dataset name (e.g., "ml_raw_20260418_256")
        db: Database connection
        split_type: "train", "validation", or "test"
        prefer_lazy: If True, returns a dict with file paths for LazySequenceGenerator
    
    Returns:
        DataFrame (standard) or Dict (lazy) with the requested split, or None if not found
    """
    import pickle
    import numpy as np
    import os
    from sqlalchemy import text
    from app.core.data.session_dataset_registry import CompressionHandler
    
    try:
        logger.info(f"🔍 [get_ml_dataset_splits_by_name] Looking for dataset: session={session_id[:8]}, name={dataset_name}, split={split_type} (prefer_lazy={prefer_lazy})")
        
        # 1. Check for existing lazy cache if prefer_lazy is True
        cache_root = os.path.join("Backend", "data", "ml_cache")
        dataset_cache_dir = os.path.join(cache_root, dataset_name, split_type)
        
        if prefer_lazy and os.path.exists(dataset_cache_dir):
            existing_chunks = sorted([os.path.join(dataset_cache_dir, f) for f in os.listdir(dataset_cache_dir) if f.endswith(".npz")])
            if existing_chunks:
                logger.info(f"🚀 [Lazy Cache] HIT: Found {len(existing_chunks)} chunks for {dataset_name}/{split_type}")
                return {
                    "data_type": "lazy_npz",
                    "file_paths": existing_chunks,
                    "dataset_name": dataset_name,
                    "split_type": split_type
                }
        
        # 2. Auto-detect storage mode: Check for chunks first
        # This enables seamless migration from blob → chunks without changing callers
        stmt = select(
            MLDataset.dataset_id, 
            MLDataset.storage_mode,
            MLDataset.scaler_binary,
            MLDataset.scaling_config
        ).where(
            and_(
                MLDataset.session_id == session_id,
                MLDataset.dataset_name == dataset_name
            )
        )
        result = await db.execute(stmt)
        dataset_info = result.first()
        
        if not dataset_info:
            logger.warning(f"⚠️ Dataset not found in ml_datasets: session={session_id[:8]}, dataset_name={dataset_name}")
            return None
        
        dataset_id, storage_mode, scaler_binary, scaling_config = dataset_info
        logger.info(f"✅ Found dataset {dataset_name}: storage_mode={storage_mode}")

        # Automatically export artefacts to the dataset cache root if prefer_lazy=True
        if prefer_lazy:
            dataset_root = os.path.join(cache_root, dataset_name)
            os.makedirs(dataset_root, exist_ok=True)
            
            # Save scaler.joblib
            scaler_path = os.path.join(dataset_root, "scaler.joblib")
            if scaler_binary and not os.path.exists(scaler_path):
                try:
                    with open(scaler_path, "wb") as f:
                        f.write(scaler_binary)
                    logger.info(f"✅ Extracted scaler binary to {scaler_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to save scaler binary: {e}")
            
            # Save feature_index_map.json
            if scaling_config:
                try:
                    import json
                    scfg = scaling_config if isinstance(scaling_config, dict) else json.loads(scaling_config)
                    feat_map = scfg.get("feature_index_map")
                    if feat_map:
                        map_path = os.path.join(dataset_root, "feature_index_map.json")
                        # Always overwrite — stale maps from previous runs must be replaced
                        # so the file reflects the current run's feature_index_map from scaling_config.
                        with open(map_path, "w") as f:
                            json.dump({
                                "feature_index_map": feat_map,
                                "feature_columns": list(feat_map.keys()),
                                "step_configs": scfg.get("step_configs", {}),
                            }, f, indent=2)
                        logger.info(f"✅ Extracted feature index map to {map_path} ({len(feat_map)} features)")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to save feature index map: {e}")
        
        # If using chunks, load from chunk table
        if storage_mode == 'chunks':
            if prefer_lazy:
                logger.info(f"📦 [Auto-Detect] Spooling {split_type} split from chunks to disk (Lazy Mode)")
                spooled_paths = await _spool_from_chunks(
                    dataset_id=dataset_id,
                    dataset_name=dataset_name,
                    split_type=split_type,
                    dataset_cache_dir=dataset_cache_dir,
                    db=db
                )
                if spooled_paths:
                    return {
                        "data_type": "lazy_npz",
                        "file_paths": sorted(spooled_paths),
                        "dataset_name": dataset_name,
                        "split_type": split_type
                    }
                logger.warning(f"⚠️ Spooling failed for {dataset_name}, falling back to RAM load")

            logger.info(f"📦 [Auto-Detect] Loading {split_type} split from chunk table into RAM")
            return await _load_from_chunks(
                dataset_id=dataset_id,
                session_id=session_id,
                split_type=split_type,
                db=db
            )
        
        # 3. Fall back to blob-based retrieval (backward compat)
        logger.info(f"📦 [Auto-Detect] Loading {split_type} split from blob (storage_mode={storage_mode})")
        
        # Query ml_datasets table using Phase 21 schema
        query = text("""
            SELECT 
                train_data_compressed, validation_data_compressed, test_data_compressed,
                train_labels, validation_labels, test_labels,
                train_targets, validation_targets, test_targets,
                feature_count, source_metadata
            FROM ml_datasets
            WHERE session_id = :session_id AND dataset_name = :dataset_name
        """)
        
        result = await db.execute(query, {
            'session_id': session_id,
            'dataset_name': dataset_name
        })
        row = result.fetchone()
        
        if not row:
            logger.warning(f"⚠️ Dataset not found in ml_datasets after auto-detect: session={session_id[:8]}")
            return None
        
        # Map split_type to the correct column
        split_map = {
            'train': (row.train_data_compressed, row.train_labels, row.train_targets),
            'validation': (row.validation_data_compressed, row.validation_labels, row.validation_targets),
            'test': (row.test_data_compressed, row.test_labels, row.test_targets),
        }
        
        data_compressed, labels_bytes, targets_bytes = split_map[split_type]
        
        if not data_compressed:
            logger.warning(f"⚠️ No compressed data for {split_type} split")
            return None
        
        # Decompress the data
        try:
            sequences_np = CompressionHandler.decompress(data_compressed)
            logger.info(f"✅ Decompressed {split_type} data from blob: shape={sequences_np.shape}")
        except Exception as decomp_err:
            logger.error(f"❌ Failed to decompress {split_type} data: {decomp_err}")
            return None
        
        # Load targets if present
        targets_np = None
        if targets_bytes:
            try:
                import zlib
                import base64
                
                # Try three formats in order (new → old → legacy)
                targets_dict = None
                
                # Format 1: NEW - Direct pickle (current fixed format)
                try:
                    targets_dict = pickle.loads(targets_bytes)
                    logger.debug(f"✅ [TARGETS] Loaded from NEW pickle format")
                except Exception as e1:
                    logger.debug(f"   Format 1 (new pickle) failed: {e1}")
                    
                    # Format 2: OLD - Compressed pickle (serialize_data format)
                    try:
                        decompressed = zlib.decompress(targets_bytes)
                        targets_dict = pickle.loads(decompressed)
                        logger.info(f"✅ [TARGETS] Loaded from OLD compressed format (backward compat)")
                    except Exception as e2:
                        logger.debug(f"   Format 2 (old compressed) failed: {e2}")
                        
                        # Format 3: LEGACY - Base64 encoded stored as string
                        try:
                            targets_str = targets_bytes.decode('utf-8', errors='ignore')
                            if targets_str.startswith(('eyJ', 'gA', 'gI')):
                                targets_b64 = base64.b64decode(targets_str)
                                decompressed = zlib.decompress(targets_b64)
                                targets_dict = pickle.loads(decompressed)
                                logger.info(f"✅ [TARGETS] Loaded from LEGACY base64 format (backward compat)")
                        except Exception as e3:
                            logger.debug(f"   Format 3 (legacy base64) failed: {e3}")
                            targets_dict = None
                
                if isinstance(targets_dict, dict):
                    target_arrays = []
                    for k, v in targets_dict.items():
                        if not isinstance(v, np.ndarray):
                            try:
                                v = np.array(v)
                            except Exception:
                                continue
                        
                        if isinstance(v, np.ndarray):
                            if len(v.shape) <= 2 and len(v) > 0:
                                if len(v.shape) == 2 and v.shape[1] > 1:
                                    target_arrays.append(v)
                                else:
                                    target_arrays.append(v.ravel())
                    
                    if target_arrays:
                        targets_np = np.column_stack(target_arrays)
                        logger.info(f"✅ Loaded targets shape: {targets_np.shape}")
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to load targets: {e}")
        
        # 4. Handle Spooling to Disk if prefer_lazy is True
        if prefer_lazy:
            try:
                os.makedirs(dataset_cache_dir, exist_ok=True)
                chunk_size = 5000  # Size of each disk chunk
                num_sequences = len(sequences_np)
                spooled_paths = []
                
                logger.info(f"🔄 [Lazy Spooling] Starting spooling for {num_sequences} sequences to {dataset_cache_dir}...")
                
                for i in range(0, num_sequences, chunk_size):
                    chunk_idx = i // chunk_size
                    chunk_end = min(i + chunk_size, num_sequences)
                    chunk_path = os.path.join(dataset_cache_dir, f"chunk_{chunk_idx:04d}.npz")
                    
                    # Prepare chunk data
                    chunk_x = sequences_np[i:chunk_end]
                    chunk_y = targets_np[i:chunk_end] if targets_np is not None else None
                    
                    # Save as NPZ
                    if chunk_y is not None:
                        np.savez_compressed(chunk_path, sequences=chunk_x, targets=chunk_y)
                    else:
                        np.savez_compressed(chunk_path, sequences=chunk_x)
                    
                    spooled_paths.append(chunk_path)
                
                logger.info(f"✅ [Lazy Spooling] Successfully spooled {len(spooled_paths)} chunks to disk")
                
                # Cleanup memory immediately
                del sequences_np
                if targets_np is not None: del targets_np
                gc.collect()
                
                return {
                    "data_type": "lazy_npz",
                    "file_paths": sorted(spooled_paths),
                    "dataset_name": dataset_name,
                    "split_type": split_type
                }
            except Exception as spool_err:
                logger.error(f"❌ [Lazy Spooling] Failed to spool data to disk: {spool_err}")
                # Fall through to standard RAM return if spooling fails
        
        # 4. Standard Return (RAM-based)
        return {"sequences": sequences_np, "targets": targets_np}
        
    except Exception as e:
        logger.error(f"❌ Failed to retrieve splits: {e}", exc_info=True)
        return None
