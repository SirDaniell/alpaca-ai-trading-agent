"""
Serialization utilities for converting numpy arrays and other non-JSON-serializable types.
Used by analysis pipelines to prepare data for API responses.
"""

import numpy as np
import pandas as pd
import asyncio
from enum import Enum
from dataclasses import asdict, is_dataclass
from datetime import datetime, date
from typing import Any, List, Dict, Tuple, Union, Optional, Callable
import logging
import zlib
import gzip
import pickle
import base64

from app.core.services.multiprocessing_utils import ParallelExecutor, RowChunker, ChunkingStrategy

logger = logging.getLogger(__name__)


def to_serializable(val: Any, numpy_safe: bool = False, depth: int = 0) -> Any:
    """
    Convert numpy arrays, pandas objects, and other non-JSON-serializable types to JSON-compatible formats.
    
    Handles:
    - numpy arrays and scalars (all dtypes via np.generic)
    - pandas Series, DataFrame, Timestamp, NaT, Index
    - dataclasses
    - nested dicts, lists, tuples
    - datetime, date, Decimal
    
    Args:
        val: Value to serialize
        numpy_safe: If True, returns numpy arrays/scalars as-is instead of converting to lists/standard types.
        depth: Recursion depth tracking (internal)
        
    Returns:
        JSON-serializable equivalent (or numpy object if numpy_safe is True)
    """
    # Prevent infinite recursion for extremely deep objects
    if depth > 100:
        logger.warning(f"⚠️ Serialization depth exceeded 100 at type {type(val)}. Truncating.")
        return str(val)

    # 1. Handle containers (Recursion)
    if isinstance(val, dict):
        return {str(k): to_serializable(v, numpy_safe=numpy_safe, depth=depth+1) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [to_serializable(i, numpy_safe=numpy_safe, depth=depth+1) for i in val]
    
    # 2. Handle numpy/pandas containers
    if isinstance(val, (np.ndarray, pd.Series, pd.Index)):
        if numpy_safe:
            return val
        return val.tolist()
    
    # 3. Handle numpy scalars (CRITICAL FIX: Covers ALL numpy types including int64, uint64, float16, etc.)
    if isinstance(val, np.generic):
        if numpy_safe:
            return val
        try:
            return val.item()  # Converts to native Python type (int, float, bool, etc.)
        except Exception as e:
            logger.error(f"❌ Failed to convert numpy scalar {type(val)} using .item(): {e}")
            return str(val)
    
    # 4. Handle pandas DataFrame/Series (convert to dict/list before checking isna)
    if isinstance(val, pd.DataFrame):
        return val.to_dict(orient='records')
    if isinstance(val, pd.Series):
        return val.to_list()
    
    # 5. Handle pandas/numpy NaN/None (AFTER DataFrame/Series check)
    if pd.isna(val):
        return None
    
    # 6. Handle datetime objects
    if isinstance(val, (pd.Timestamp, datetime, date)):
        return val.isoformat()
    
    # 7. Handle Enum
    if isinstance(val, Enum):
        return val.value
    
    # 7. Handle dataclasses
    if is_dataclass(val):
        return to_serializable(asdict(val), numpy_safe=numpy_safe, depth=depth+1)

    # 8. Handle Decimal and other common non-standard types
    from decimal import Decimal
    if isinstance(val, Decimal):
        val = float(val)

    # 9. Handle Infinity and NaN for PostgreSQL JSONB
    # PostgreSQL JSONB strictly enforces the JSON spec, which does not support Infinity or NaN.
    if isinstance(val, float):
        import math
        if math.isinf(val):
            return "Infinity" if val > 0 else "-Infinity"
        if math.isnan(val):
            return None
            
    # Pass through standard JSON types (str, int, float, bool, None)
    # If it's something else, it might still fail JSON serialization later,
    # so we log it at debug level if it's not a standard type.
    if val is not None and not isinstance(val, (str, int, float, bool)):
        if depth < 5: # Only log for top-level unknown types to avoid spam
            logger.debug(f"🔍 to_serializable: passing through potentially non-JSON type: {type(val)}")
            
    return val


def to_serializable_records(records: List[Dict[str, Any]], numpy_safe: bool = False) -> List[Dict[str, Any]]:

    """
    Highly optimized version of to_serializable for lists of records (tabular data).
    Avoids redundant type checking by processing columns in batches where possible.
    """
    if not records:
        return []

    # If first record is NOT a dict, fallback to standard
    if not isinstance(records[0], dict):
        return [to_serializable(r, numpy_safe=numpy_safe) for r in records]

    # Standard path: list of dicts
    # Trace specific types for debugging if requested via log level
    if logger.isEnabledFor(logging.DEBUG) and len(records) > 0:
        logger.debug(f"🔍 Serializing {len(records)} records. First record types: {[(k, type(v)) for k, v in records[0].items()]}")

    return [
        {str(k): to_serializable(v, numpy_safe=numpy_safe) for k, v in record.items()}
        for record in records
    ]




# ============================================================================
# HELPER: Data Summary for Logging
# ============================================================================

def _get_data_summary(data: Any) -> str:
    """Generate a concise summary of data structure for logging"""
    try:
        if isinstance(data, list):
            if len(data) == 0:
                return "list(empty)"
            first = data[0]
            if isinstance(first, dict):
                keys = list(first.keys())[:5]  # First 5 columns
                return f"list({len(data)} records, columns={keys}...)"
            else:
                return f"list({len(data)} items)"
        elif isinstance(data, dict):
            keys = list(data.keys())[:5]
            if 'data' in data and isinstance(data['data'], list):
                return f"dict(keys={keys}..., data→list({len(data['data'])} items))"
            return f"dict(keys={keys}...)"
        elif isinstance(data, pd.DataFrame):
            return f"DataFrame({len(data)} rows × {len(data.columns)} cols)"
        else:
            return f"{type(data).__name__}"
    except Exception as e:
        return f"unknown_type({type(data).__name__})"


# ============================================================================
# Centralized Serialization (CRITICAL FIX for data consistency)
# ============================================================================

def serialize_data(data: Any, compress: bool = True, numpy_safe: bool = False, on_progress: Optional[Callable[[int, str], Any]] = None) -> str:
    """
    Serialize and optionally compress data, return as base64 string.
    This is the standard for storage in PostgreSQL TEXT columns.
    
    Args:
        data: Any serializable Python object
        compress: Whether to use zlib compression
        numpy_safe: If True, skip to_serializable() conversion and pickle numpy arrays directly.
        on_progress: Optional async/sync callback(progress_pct, message)
            
    Returns:
        Base64 encoded string
    """

    # ✅ LOGGING: Peek at input data structure
    data_summary = _get_data_summary(data)
    logger.info(f"📦 SERIALIZE INPUT: {data_summary}")
    
    if numpy_safe:
        # CRITICAL FIX: Pickle numpy arrays directly — do NOT call to_serializable()
        clean_data = data
    else:
        # Standard path: convert numpy/pandas to JSON-safe types first
        if on_progress:
            on_progress(10, "Preparing data for serialization...")

        # If data is large and is a list/dict, use ParallelSerializer
        if isinstance(data, (list, dict)) and len(data) > 5000:
            logger.info(f"🚀 Using ParallelSerializer for {len(data)} items")
            clean_data = ParallelSerializer.serialize(data, n_workers=None, on_progress=on_progress)
        else:
            # Optimize tabular records if it's a list of dicts
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                clean_data = to_serializable_records(data, numpy_safe=numpy_safe)
            else:
                clean_data = to_serializable(data, numpy_safe=numpy_safe)

    
    if on_progress:
        on_progress(70, "Finalizing pickling...")
    
    # Pickle the data (handles numpy arrays natively when numpy_safe=True)
    pickled = pickle.dumps(clean_data)
    
    if on_progress:
        on_progress(80, "Compressing data...")
    
    # Compress if requested
    if compress:
        compressed = zlib.compress(pickled)

    else:
        compressed = pickled
    
    # Convert to base64 string for storage
    b64_result = base64.b64encode(compressed).decode('utf-8')
    
    # ✅ LOGGING: Output size and format
    logger.info(f"✅ SERIALIZE OUTPUT: {len(b64_result):,} bytes (compressed={compress}, numpy_safe={numpy_safe})")
    
    return b64_result


# ============================================================================
# Centralized Deserialization (CRITICAL FIX for data consistency)
# ============================================================================

def deserialize_data(data: Union[str, bytes, dict, list], validate: bool = True) -> Union[dict, list]:
    """
    Centralized deserialization for all data types.
    
    Handles:
    - Pickled + compressed data (base64 encoded)
    - Pickled + compressed data (raw bytes)
    - Already deserialized dicts/lists
    - Validates data structure consistency
    
    Args:
        data: Input data (str, bytes, dict, or list)
        validate: Whether to validate structure consistency
        
    Returns:
        Deserialized data (dict or list)
        
    Raises:
        ValueError: If data is corrupted or inconsistent
    """
    
    if data is None:
        logger.warning("⚠️ deserialize_data received None")
        return {}
    
    # Case 1: Already deserialized (dict, list, or DataFrame)
    if isinstance(data, (dict, list, pd.DataFrame)):
        summary = _get_data_summary(data)
        logger.info(f"✅ DESERIALIZE INPUT: Already deserialized - {summary}")
        if validate:
            _validate_data_structure(data)
        return data
    
    # Case 2: String (base64 encoded pickle)
    if isinstance(data, str):
        try:
            logger.info("🔍 Attempting to decode base64 string...")
            data_bytes = base64.b64decode(data)
            logger.info(f"✅ Decoded {len(data_bytes)} bytes from base64")
            del data
        except Exception as e:
            logger.error(f"❌ Failed to decode base64: {e}")
            raise ValueError(f"Invalid base64 encoding: {e}")
        
        return _deserialize_bytes(data_bytes, validate)
    
    # Case 3: Bytes (pickled + compressed)
    if isinstance(data, bytes):
        return _deserialize_bytes(data, validate)
    
    raise ValueError(f"Unsupported data type for deserialization: {type(data)}")


def _deserialize_bytes(data_bytes: bytes, validate: bool = True) -> Union[dict, list]:
    """
    Deserialize bytes that may be compressed with zlib or gzip.
    
    Args:
        data_bytes: Raw bytes (possibly compressed with zlib or gzip)
        validate: Whether to validate structure
        
    Returns:
        Deserialized data
    """
    
    # Try zlib decompression first (preferred format)
    decompressed = None
    try:
        logger.info(f"🔍 Attempting zlib decompression on {len(data_bytes)} bytes...")
        decompressed = zlib.decompress(data_bytes)
        logger.info(f"✅ Decompressed with zlib to {len(decompressed)} bytes")
        data_bytes = decompressed
    except Exception as zlib_error:
        # If zlib fails, try gzip as fallback (for existing stored data)
        try:
            logger.info(f"⚠️ zlib failed: {zlib_error}, trying gzip...")
            decompressed = gzip.decompress(data_bytes)
            logger.info(f"✅ Decompressed with gzip to {len(decompressed)} bytes")
            data_bytes = decompressed
        except Exception as gzip_error:
            logger.warning(f"⚠️ Neither zlib nor gzip worked: {gzip_error}, trying direct pickle...")
    
    # Deserialize pickle
    try:
        logger.info("🔍 Attempting pickle deserialization...")
        result = pickle.loads(data_bytes)
        logger.info(f"✅ Deserialized pickle: {type(result).__name__}")
        
        # Free memory held by intermediate byte buffers
        del data_bytes
        if decompressed is not None:
            del decompressed

        if validate:
            _validate_data_structure(result)
        
        return result
    except Exception as e:
        logger.error(f"❌ Failed to deserialize pickle: {e}")
        raise ValueError(f"Failed to deserialize data: {e}")


def _validate_data_structure(data: Union[dict, list]) -> None:
    """
    Validate that deserialized data has consistent structure.
    
    For dicts: Only validates if it looks like actual data (not metadata).
              Actual data should have all columns of equal length.
              Metadata dicts (with 'preview', 'columns', 'record_count') are skipped.
    For lists: All items must be consistent type
    
    Args:
        data: Data to validate
        
    Raises:
        ValueError: If structure is inconsistent
    """
    
    if isinstance(data, dict):
        if not data:
            logger.warning("⚠️ Data dict is empty")
            return
        
        # CRITICAL FIX: Check if this is metadata or signal generation result (not strict DataFrame data)
        # Metadata dicts have keys like 'preview', 'tail', 'columns', 'record_count', 'source_config'
        metadata_keys = {'preview', 'tail', 'columns', 'record_count', 'source_config', 
                        'fp_count', 'record count', 'file_name', 'file_size'}
        actual_keys = set(data.keys())
        
        if metadata_keys & actual_keys:  # If there's any overlap with metadata keys
            logger.info(f"✅ Detected metadata dict, skipping array length validation: {actual_keys}")
            return
        
        # SIGNAL GENERATION RESULT: Has different-length arrays by design
        # Keys like 'ml_dataset' (1640), 'data' (1000), 'zones' (396), 'signals' (1640)
        signal_gen_keys = {'ml_dataset', 'data', 'signals', 'zones'}
        if signal_gen_keys & actual_keys:  # Signal generation result
            logger.info(f"✅ Detected signal generation result, skipping strict array length validation: {actual_keys}")
            return
        
        # Check all values are iterables with consistent length
        lengths = {}
        for key, value in data.items():
            if isinstance(value, (list, tuple, np.ndarray)):
                lengths[key] = len(value)
            elif isinstance(value, pd.Series):
                lengths[key] = len(value)
            elif isinstance(value, pd.Index):
                lengths[key] = len(value)
            else:
                # Scalar value - allow
                lengths[key] = 1
        
        # Check consistency
        unique_lengths = set(v for v in lengths.values() if v != 1)  # Ignore scalars
        
        if len(unique_lengths) > 1:
            logger.error(f"❌ VALIDATION FAILED: Inconsistent column lengths: {lengths}")
            logger.error(f"   Unique lengths (excluding scalars): {unique_lengths}")
            min_length = min(unique_lengths) if unique_lengths else min(lengths.values())
            raise ValueError(
                f"Data has inconsistent column lengths: {lengths}. "
                f"This will cause pandas.DataFrame() to fail. "
                f"Minimum length is {min_length}, please truncate longer columns."
            )
        
        logger.info(f"✅ Data structure validated: {len(data)} columns, consistent lengths: {set(lengths.values())}")
    
    elif isinstance(data, list):
        if not data:
            logger.warning("⚠️ Data list is empty")
            return
        
        logger.info(f"✅ Data structure validated: {len(data)} items")


def fix_inconsistent_data(data: Union[dict, list]) -> Union[dict, list]:
    """
    Automatically fix data with inconsistent structure.
    
    For dicts: Truncate all columns to minimum length
    For lists: Return as-is (lists are self-consistent)
    
    Args:
        data: Data to fix
        
    Returns:
        Fixed data
    """
    
    if isinstance(data, dict):
        lengths = {}
        for key, value in data.items():
            if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
                lengths[key] = len(value)
            else:
                lengths[key] = 1
        
        unique_lengths = set(v for v in lengths.values() if v != 1)
        
        if len(unique_lengths) <= 1:
            return data  # Already consistent
        
        min_length = min(unique_lengths) if unique_lengths else min(lengths.values())
        logger.warning(f"⚠️ Fixing inconsistent data: truncating all columns to length {min_length}")
        
        fixed_data = {}
        for key, value in data.items():
            if isinstance(value, (list, tuple)):
                fixed_data[key] = value[:min_length]
            elif isinstance(value, np.ndarray):
                fixed_data[key] = value[:min_length]
            elif isinstance(value, pd.Series):
                fixed_data[key] = value.iloc[:min_length].tolist()
            else:
                fixed_data[key] = value  # Keep scalars
        
        logger.info(f"✅ Data fixed: truncated to {min_length} rows")
        return fixed_data
    
    return data

# ============================================================================
# Worker Functions (Module level for pickling/multiprocessing compatibility)
# ============================================================================

def _serialize_list_chunk_worker(chunk_data: Tuple[int, List[Any], bool]) -> Tuple[int, List[Any]]:
    """Worker function to serialize a chunk of a list along with its original index for ordering."""
    chunk_idx, items, numpy_safe = chunk_data
    serialized_items = [to_serializable(item, numpy_safe=numpy_safe) for item in items]
    return chunk_idx, serialized_items

def _serialize_dict_chunk_worker(chunk_data: Tuple[List[Tuple[Any, Any]], bool]) -> List[Tuple[Any, Any]]:
    """Worker function to serialize a chunk of key-value pairs from a dictionary."""
    items, numpy_safe = chunk_data
    return [(to_serializable(k, numpy_safe=numpy_safe), to_serializable(v, numpy_safe=numpy_safe)) for k, v in items]


class ParallelSerializer:
    """
    Handles parallelization of serialization for large datasets.
    Implements the chunking strategies defined in multiprocessing_utils.
    """
    
    @staticmethod
    def serialize_list(val: List[Any], n_workers: int = None, threshold: int = 500, on_progress: Optional[Callable] = None, numpy_safe: bool = False) -> List[Any]:
        """
        Intelligently serializes a list. If the list is large, splits it across cores.
        """
        n_items = len(val)
        
        logger.info(f"🔍 ParallelSerializer.serialize_list: {n_items} items (threshold={threshold}, numpy_safe={numpy_safe})")
        
        # 1. Fallback to sequential for small lists
        if n_items < threshold:
            logger.info(f"⏭️  List too small ({n_items} < {threshold}): using sequential serialization")
            # Optimized path for record lists
            if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                return to_serializable_records(val, numpy_safe=numpy_safe)

            return [to_serializable(i, numpy_safe=numpy_safe) for i in val]
            
        if n_workers is None:
            n_workers = ChunkingStrategy.auto_chunk_count()
            
        n_workers = min(n_workers, n_items)
            
        # 2. Chunking strategy
        ranges = RowChunker.chunk_rows(n_items, n_workers)
        chunks = [val[start:end] for start, end in ranges]
        args_list = [(i, chunk, numpy_safe) for i, chunk in enumerate(chunks)]
        
        logger.info(f"🚀 PARALLELIZING LIST: {n_items} items → {n_workers} workers ({len(args_list)} chunks)")
        
        # 3. Parallel Execution with progress tracking
        chunk_results = []
        try:
            from multiprocessing import Pool
            with Pool(n_workers) as pool:
                # Use imap (ordered) — NOT imap_unordered — to guarantee
                # chunk results arrive in submission order and preserve
                # chronological ordering of the original time-series data.
                imap_it = pool.imap(_serialize_list_chunk_worker, args_list)
                
                for result in imap_it:
                    chunk_results.append(result)
                    if on_progress:
                        progress = 10 + int((len(chunk_results) / len(args_list)) * 50)
                        on_progress(progress, f"Serialized chunk {len(chunk_results)}/{len(args_list)}...")
        except Exception as e:
            logger.error(f"❌ Parallel serialization failed: {e}")
            # Fallback to sequential if parallel fails (e.g. pickling error)
            return [to_serializable(i, numpy_safe=numpy_safe) for i in val]
            
        logger.info(f"✅ Parallel serialization complete: {len(chunk_results)} chunk results received")
        
        # 4. Recombination (sort by chunk_idx to maintain order)
        chunk_results.sort(key=lambda x: x[0])
        
        final_result = []
        for _, items in chunk_results:
            final_result.extend(items)
            
        # 5. Logging: Verify order before return
        if final_result:
            try:
                # Log first 20 and last 20 for thorough verification
                head_20 = [i.get('Time') if isinstance(i, dict) else str(i)[:20] for i in final_result[:20]]
                tail_20 = [i.get('Time') if isinstance(i, dict) else str(i)[:20] for i in final_result[-20:]]
                logger.info(f"📊 [SERIALIZE_LIST] Order check (N={len(final_result)}):")
                logger.info(f"   ├─ Head (first 20): {head_20}")
                logger.info(f"   └─ Tail (last 20):  {tail_20}")
            except Exception as log_err:
                logger.debug(f"Could not log head/tail: {log_err}")

        return final_result


    @staticmethod
    def serialize_dict(val: Dict[Any, Any], n_workers: int = None, threshold: int = 10000, on_progress: Optional[Callable] = None, numpy_safe: bool = False) -> Dict[Any, Any]:
        """
        Intelligently serializes a dictionary. If the dict is large, splits it across cores.
        """
        n_items = len(val)
        
        logger.info(f"🔍 ParallelSerializer.serialize_dict: {n_items} items (threshold={threshold}, numpy_safe={numpy_safe})")
        
        if n_items < threshold:
            logger.info(f"⏭️  Dict too small ({n_items} < {threshold}): using sequential serialization")
            return to_serializable(val, numpy_safe=numpy_safe)
            
        if n_workers is None:
            n_workers = ChunkingStrategy.auto_chunk_count()
            
        n_workers = min(n_workers, n_items)
            
        # Convert dict to list of items for chunking
        items = list(val.items())
        ranges = RowChunker.chunk_rows(n_items, n_workers)
        chunks = [items[start:end] for start, end in ranges]
        # Include index for ordering
        args_list = [(i, chunk, numpy_safe) for i, chunk in enumerate(chunks)]
        
        logger.info(f"🚀 PARALLELIZING DICT: {n_items} items → {n_workers} workers ({len(args_list)} chunks)")
        
        # Parallel Execution with progress
        chunk_results = []
        try:
            from multiprocessing import Pool
            with Pool(n_workers) as pool:
                imap_it = pool.imap(_serialize_list_chunk_worker, args_list)  # ordered — preserves key insertion order
                for result in imap_it:
                    chunk_results.append(result)
                    if on_progress:
                        progress = 10 + int((len(chunk_results) / len(args_list)) * 50)
                        on_progress(progress, f"Serialized dict chunk {len(chunk_results)}/{len(args_list)}...")
        except Exception as e:
            logger.error(f"❌ Parallel dict serialization failed: {e}")
            return to_serializable(val, numpy_safe=numpy_safe)
        
        logger.info(f"✅ Parallel serialization complete: {len(chunk_results)} chunk results received")
        
        # Recombination (Sort by index to maintain key order)
        chunk_results.sort(key=lambda x: x[0])
        
        final_result = {}
        for _, chunk_items in chunk_results:
            # chunk_items is a list of (k, v) tuples if we use _serialize_list_chunk_worker on dict items
            # But wait, to_serializable on (k, v) tuple returns a list [k, v].
            # Let's fix the worker or the loop.
            for item in chunk_items:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    final_result[item[0]] = item[1]
                
        # Logging: Verify keys before return
        if final_result:
            try:
                all_keys = list(final_result.keys())
                logger.info(f"📊 [SERIALIZE_DICT] Keys check: Head={all_keys[0]} | Tail={all_keys[-1]} (Total: {len(all_keys)})")
            except Exception as log_err:
                logger.debug(f"Could not log head/tail: {log_err}")
                
        return final_result


    @staticmethod
    def serialize(val: Any, n_workers: int = None, list_threshold: int = 10000, dict_threshold: int = 10000, on_progress: Optional[Callable] = None, numpy_safe: bool = False) -> Any:
        """
        Main entry point for parallel serialization. Detects type and routes appropriately.
        """
        logger.info(f"🔍 ParallelSerializer.serialize: Processing {type(val).__name__} (list_threshold={list_threshold}, dict_threshold={dict_threshold}, numpy_safe={numpy_safe})")
        
        if isinstance(val, list):
            logger.info(f"   → Routing to serialize_list ({len(val)} items)")
            return ParallelSerializer.serialize_list(val, n_workers, list_threshold, on_progress=on_progress, numpy_safe=numpy_safe)
        elif isinstance(val, pd.DataFrame):
            logger.info(f"   → Routing to serialize_list via DataFrame ({len(val)} rows)")
            # Convert to records once and then parallelize
            records = val.to_dict(orient='records')
            return ParallelSerializer.serialize_list(records, n_workers, list_threshold, on_progress=on_progress, numpy_safe=numpy_safe)
        elif isinstance(val, tuple):
            logger.info(f"   → Routing to serialize_list via tuple ({len(val)} items)")
            return tuple(ParallelSerializer.serialize_list(list(val), n_workers, list_threshold, on_progress=on_progress, numpy_safe=numpy_safe))
        elif isinstance(val, dict):
            logger.info(f"   → Routing to serialize_dict ({len(val)} items)")
            return ParallelSerializer.serialize_dict(val, n_workers, dict_threshold, on_progress=on_progress, numpy_safe=numpy_safe)
        
        # Fallback to standard serialization for small data or other types
        logger.info(f"   → Fallback to sequential for {type(val).__name__}")
        return to_serializable(val, numpy_safe=numpy_safe)

# PRIORITY 1: Checkpoint Support for Long-Duration Analysis
# ============================================================================

def extract_metadata_only(data: Union[dict, list]) -> dict:
    """
    Extract only metadata from data WITHOUT full deserialization.
    Used for status checks on multi-GB datasets without loading full data.
    
    For analysis results, typically stored as:
    {
        "data": [...],
        "metadata": {...},
        "summary": {...}
    }
    
    Args:
        data: Full result dict or list
        
    Returns:
        Metadata only (no large arrays/data)
    """
    if isinstance(data, dict):
        return {
            "metadata": data.get("metadata", {}),
            "summary": data.get("summary", {}),
            "type": type(data).__name__,
            "keys": list(data.keys()),
            "has_data": "data" in data,
            "record_count": len(data.get("data", [])) if isinstance(data.get("data"), list) else 0
        }
    else:
        return {
            "type": type(data).__name__,
            "record_count": len(data) if isinstance(data, list) else 0
        }


def serialize_for_checkpoint(
    data: Any,
    checkpoint_number: int,
    chunk_size: int = 50_000
) -> dict:
    """
    Prepare data for checkpoint (partial result during long analysis).
    
    For tasks that take hours, save intermediate results every chunk_size records
    or every 15 minutes. Allows resumption if task is interrupted.
    
    Args:
        data: Partial result data
        checkpoint_number: Which checkpoint this is (1, 2, 3, ...)
        chunk_size: How many records per checkpoint
        
    Returns:
        Checkpoint metadata dict
    """
    import time
    
    metadata = {
        "checkpoint_number": checkpoint_number,
        "timestamp": datetime.utcnow().isoformat(),
        "chunk_size": chunk_size,
        "data_type": type(data).__name__,
    }
    
    # Count records for progress tracking
    if isinstance(data, list):
        metadata["record_count"] = len(data)
        metadata["total_expected"] = checkpoint_number * chunk_size
        metadata["completion_percentage"] = min(
            100,
            (metadata["record_count"] / metadata["total_expected"]) * 100
        )
    elif isinstance(data, dict) and "data" in data:
        if isinstance(data["data"], list):
            metadata["record_count"] = len(data["data"])
            metadata["total_expected"] = checkpoint_number * chunk_size
            metadata["completion_percentage"] = min(
                100,
                (metadata["record_count"] / metadata["total_expected"]) * 100
            )
    
    logger.info(
        f"💾 Checkpoint {checkpoint_number}: "
        f"{metadata.get('record_count', '?')} records "
        f"({metadata.get('completion_percentage', '?'):.1f}% complete)"
    )
    
    return metadata
