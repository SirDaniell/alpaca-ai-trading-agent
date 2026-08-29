# Backend/app/core/data

> Data serialization, session management, and ML dataset registry for the trading analysis platform.

---

## Overview

The `data/` module is the **persistence layer** for the entire analysis pipeline. It handles three critical responsibilities:

1. **Serialization** — Converting Python/NumPy/Pandas objects to JSON-safe or pickle formats for PostgreSQL storage and API responses
2. **Session data loading** — Managing the enrichment pipeline where each analysis step (technical → SNR → astronomical) builds on the previous step's output
3. **ML dataset registry** — Three-tier caching system (memory → LRU cache → PostgreSQL) for compressed ML training datasets

This module sits between the analysis engines (`analysis/`, `ml/`) and the database layer. Every analysis result flows through these serializers before storage. Every dataset load for training or inference queries this registry.

**Key design decisions:**

- **Parallel serialization:** Large datasets (>5K rows) are split across CPU cores to bypass Python's GIL, achieving 60% speed improvement
- **Dual format storage:** JSONB preferred for <200MB results (queryable), pickle fallback for larger data (compressed)
- **Mutation safety:** Steps can exclude their own previous output when loading data to prevent corruption on re-runs
- **ZSTANDARD compression:** ML datasets compressed at 4:1 ratio (75% size reduction) for efficient storage

---

## Module structure

```
data/
├── serializers.py              # Core serialization engine (JSON/pickle/compression)
├── session_data_loader.py      # Session enrichment pipeline & data loading
├── session_dataset_registry.py # Three-tier ML dataset cache with compression
└── __init__.py                 # (empty)
```

---

## Files in this module

### serializers.py — Serialization Engine

> Converts Python objects (numpy arrays, pandas DataFrames, datetime, Enum) to JSON-safe or pickle formats with optional compression and parallel processing.

**Key classes:** `ParallelSerializer`  
**Key functions:** `to_serializable()`, `serialize_data()`, `deserialize_data()`, `validate_data_structure()`

[→ Full documentation](#serializerspy)

---

### session_data_loader.py — Session Data Pipeline

> Authoritative data loading system that walks the enrichment priority chain (astronomical → SNR → technical → raw) to load the most recent dataset for each analysis step.

**Key functions:** `get_latest_session_data()`, `store_session_step_result()`, `set_as_current_data()`, `store_batch_to_db()`

[→ Full documentation](#session_data_loaderpy)

---

### session_dataset_registry.py — ML Dataset Cache

> Three-tier caching architecture (TIER 0: pointer, TIER 1: LRU memory cache, TIER 2: PostgreSQL) with ZSTANDARD compression for ML training datasets.

**Key classes:** `SessionDatasetRegistry`, `CompressionHandler`, `DatasetMetadata`

[→ Full documentation](#session_dataset_registrypy)

---

# serializers.py

> Core serialization engine for converting Python/NumPy/Pandas objects to JSON-safe or pickle formats with parallel processing and compression.

[→ Source: `Backend/app/core/data/serializers.py`](../../../Backend/app/core/data/serializers.py)

---

## Overview

This file solves the **type conversion problem**: Python's native types (numpy arrays, pandas DataFrames, datetime objects, NaN values) are not JSON-serializable, but the API and database require JSON-compatible formats.

**What it does:**

- Recursively converts nested data structures (dicts, lists, DataFrames) to JSON-safe types
- Handles all numpy scalar types (int64, float32, uint8, etc.) via `np.generic` detection
- Replaces NaN/Infinity with None/"Infinity" strings (PostgreSQL JSONB compliance)
- Parallelizes serialization for large datasets (>5K items) across CPU cores
- Compresses data with zlib for PostgreSQL TEXT column storage
- Validates data structure consistency (equal column lengths) before DataFrame construction

**Where it sits:**

- **Called by:** `session_data_loader.store_session_step_result()`, `session_dataset_registry.register_dataset()`, all API endpoints returning analysis results
- **Calls:** `multiprocessing.Pool` for parallel execution, `pickle` and `zlib` for compression

**Design decisions:**

1. **Why parallel serialization?** Large datasets (100K+ rows) take 2-3 seconds to serialize sequentially. Splitting across 4 cores reduces this to 0.8 seconds (60% improvement).
2. **Why pickle + zlib instead of JSON?** Pickle preserves numpy array dtypes exactly. JSON loses precision on float64 and cannot represent NaN natively.
3. **Why validate structure?** Pandas DataFrame constructor fails silently if columns have mismatched lengths. Validation catches this before storage.

---

## Quick reference

| Symbol | Type | Purpose |
|--------|------|---------|
| `to_serializable()` | function | Recursively convert any Python object to JSON-safe format |
| `to_serializable_records()` | function | Optimized batch conversion for list-of-dicts (tabular data) |
| `serialize_data()` | function | Pickle + compress data, return base64 string for DB storage |
| `deserialize_data()` | function | Decompress + unpickle data from base64 string or bytes |
| `validate_data_structure()` | function | Check column length consistency (prevents pandas errors) |
| `fix_inconsistent_data()` | function | Auto-truncate columns to minimum length |
| `ParallelSerializer` | class | Parallel serialization for large datasets (>5K items) |
| `extract_metadata_only()` | function | Get metadata without full deserialization (fast status checks) |
| `serialize_for_checkpoint()` | function | Prepare checkpoint metadata for long-running tasks |

[→ Jump to source: Backend/app/core/data/serializers.py](../../../Backend/app/core/data/serializers.py)

---

## Functions

### to_serializable(val, numpy_safe=False, depth=0) → Any

[→ Source: `Backend/app/core/data/serializers.py` line 24](../../../Backend/app/core/data/serializers.py#L24)

Recursively converts Python objects to JSON-serializable equivalents. This is the **core conversion function** used throughout the system.

**What it does:**

Handles 9 categories of non-JSON types:
1. Containers (dict, list, tuple) — recurse into nested structures
2. NumPy/Pandas arrays (ndarray, Series, Index) — convert to lists
3. NumPy scalars (int64, float32, etc.) — extract native Python value via `.item()`
4. Pandas DataFrames — convert to list of dicts
5. NaN/None values — replace with `None`
6. Datetime objects — convert to ISO 8601 strings
7. Enums — extract `.value`
8. Dataclasses — convert to dict via `asdict()`
9. Infinity/NaN floats — replace with strings (PostgreSQL JSONB compliance)

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| val | Any | Yes | Value to serialize (any Python object) |
| numpy_safe | bool | No (default: False) | If True, return numpy arrays/scalars as-is instead of converting |
| depth | int | No (default: 0) | Recursion depth tracker (internal use) |

**Returns:** JSON-serializable equivalent (str, int, float, bool, None, list, dict) — or numpy object if `numpy_safe=True`

**Raises:**
- Logs warning if depth exceeds 100 (prevents infinite recursion)
- Returns `str(val)` as fallback for unknown types

**The conversion logic:**

```
if isinstance(val, dict):
    return {str(k): to_serializable(v) for k, v in val.items()}

if isinstance(val, np.generic):  # Covers ALL numpy scalar types
    return val.item()  # Converts to native Python type

if pd.isna(val):
    return None

if isinstance(val, float) and math.isinf(val):
    return "Infinity" or "-Infinity"
```

**Code example:**

```python
import numpy as np
import pandas as pd
from app.core.data.serializers import to_serializable

# NumPy scalar conversion
val = np.int64(42)
result = to_serializable(val)
print(result, type(result))  # → 42 <class 'int'>

# Nested structure with NaN
data = {
    'prices': np.array([1.0, 2.5, np.nan]),
    'timestamp': pd.Timestamp('2024-01-15 10:30:00'),
    'volume': np.uint64(1000)
}
result = to_serializable(data)
print(result)
# → {
#     'prices': [1.0, 2.5, None],
#     'timestamp': '2024-01-15T10:30:00',
#     'volume': 1000
# }
```

**Edge cases:**

- **Empty containers:** Returns empty list/dict as-is
- **Circular references:** Depth limit (100) prevents infinite loops, returns `str(val)` at limit
- **Unknown types:** Logs debug message and passes through (may fail JSON serialization later)
- **numpy_safe=True:** Used for ML preparation where numpy arrays must be preserved for training

---

### to_serializable_records(records, numpy_safe=False) → List[Dict[str, Any]]

[→ Source: `Backend/app/core/data/serializers.py` line 119](../../../Backend/app/core/data/serializers.py#L119)

Optimized version of `to_serializable()` for **tabular data** (list of dicts). Avoids redundant type checking by processing columns in batches.

**What it does:**

For lists of dicts (typical DataFrame output), this function is 20-30% faster than calling `to_serializable()` on each record individually because it:
- Checks if first element is a dict (early exit if not)
- Processes all records in a single list comprehension
- Logs type information for debugging (first record only)

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| records | List[Dict[str, Any]] | Yes | List of record dicts (tabular data) |
| numpy_safe | bool | No (default: False) | If True, preserve numpy arrays |

**Returns:** List[Dict[str, Any]] — JSON-safe list of dicts

**Code example:**

```python
import pandas as pd
from app.core.data.serializers import to_serializable_records

df = pd.DataFrame({
    'price': [1.0, 2.5, 3.7],
    'volume': [100, 200, 150],
    'timestamp': pd.date_range('2024-01-01', periods=3)
})

records = df.to_dict(orient='records')
result = to_serializable_records(records)

print(result[0])
# → {
#     'price': 1.0,
#     'volume': 100,
#     'timestamp': '2024-01-01T00:00:00'
# }
```

**Performance:**

- **Small datasets (<1K rows):** ~5% faster than `to_serializable()`
- **Large datasets (>10K rows):** ~25% faster due to reduced function call overhead

---

### serialize_data(data, compress=True, numpy_safe=False, on_progress=None) → str

[→ Source: `Backend/app/core/data/serializers.py` line 169](../../../Backend/app/core/data/serializers.py#L169)

Serialize and optionally compress data, return as **base64 string** for PostgreSQL TEXT column storage.

**What it does:**

1. Convert data to JSON-safe format via `to_serializable()` (unless `numpy_safe=True`)
2. For large datasets (>5K items), use `ParallelSerializer` for 60% speed boost
3. Pickle the cleaned data
4. Compress with zlib (optional)
5. Encode as base64 string

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| data | Any | Yes | Python object to serialize |
| compress | bool | No (default: True) | Whether to use zlib compression |
| numpy_safe | bool | No (default: False) | If True, skip `to_serializable()` and pickle numpy arrays directly |
| on_progress | Callable[[int, str], Any] | No | Optional progress callback (percent, message) |

**Returns:** str — Base64-encoded string (ready for PostgreSQL TEXT column)

**The process:**

```
Input data (dict/list/DataFrame)
    ↓
to_serializable() [unless numpy_safe=True]
    ↓
pickle.dumps()
    ↓
zlib.compress() [if compress=True]
    ↓
base64.b64encode()
    ↓
Output: base64 string
```

**Code example:**

```python
from app.core.data.serializers import serialize_data
import pandas as pd

df = pd.DataFrame({
    'price': [1.0, 2.5, 3.7],
    'volume': [100, 200, 150]
})

# Standard serialization (compressed)
result = serialize_data(df.to_dict(orient='records'), compress=True)
print(len(result))  # → ~200 bytes (compressed)

# Uncompressed
result_uncompressed = serialize_data(df.to_dict(orient='records'), compress=False)
print(len(result_uncompressed))  # → ~800 bytes (raw pickle)

# Compression ratio: 4:1 (75% reduction)
```

**Performance:**

- **Small data (<5K items):** Sequential processing, ~10ms
- **Large data (>50K items):** Parallel processing, ~800ms (vs 2-3s sequential)

**Edge cases:**

- **numpy_safe=True:** Used for ML datasets where numpy dtypes must be preserved exactly
- **Empty data:** Returns small base64 string (~50 bytes)
- **Progress callback:** Called at 10%, 70%, 80%, 100% for UI updates

---

### deserialize_data(data, validate=True) → Union[dict, list]

[→ Source: `Backend/app/core/data/serializers.py` line 246](../../../Backend/app/core/data/serializers.py#L246)

Centralized deserialization for all data types. Handles pickled + compressed data from database.

**What it does:**

1. Detect input format (str, bytes, dict, list)
2. Decode base64 if string
3. Decompress with zlib or gzip (tries both)
4. Unpickle data
5. Validate structure consistency (optional)

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| data | Union[str, bytes, dict, list] | Yes | Input data (base64 string, raw bytes, or already deserialized) |
| validate | bool | No (default: True) | Whether to validate structure consistency |

**Returns:** Union[dict, list] — Deserialized data

**Raises:**
- `ValueError` — If base64 decoding fails or pickle deserialization fails
- `DataValidationError` — If validation enabled and structure inconsistent

**The process:**

```
Input: base64 string or bytes
    ↓
base64.b64decode() [if string]
    ↓
zlib.decompress() or gzip.decompress()
    ↓
pickle.loads()
    ↓
validate_data_structure() [if validate=True]
    ↓
Output: dict or list
```

**Code example:**

```python
from app.core.data.serializers import serialize_data, deserialize_data

# Round-trip test
original = {'prices': [1.0, 2.5, 3.7], 'volume': [100, 200, 150]}
serialized = serialize_data(original, compress=True)
restored = deserialize_data(serialized, validate=True)

print(restored == original)  # → True
```

**Edge cases:**

- **Already deserialized:** If input is dict/list, returns as-is (no-op)
- **Compression format:** Tries zlib first, falls back to gzip for legacy data
- **Validation failure:** Raises `DataValidationError` with details about inconsistent columns

---

### validate_data_structure(data) → None

[→ Source: `Backend/app/core/data/serializers.py` line 349](../../../Backend/app/core/data/serializers.py#L349)

Validate that deserialized data has **consistent structure** (all columns same length). Prevents pandas DataFrame construction errors.

**What it does:**

For dicts: Checks that all array-like values have the same length (ignoring scalars). Skips validation for metadata dicts (with keys like 'preview', 'columns', 'record_count') and signal generation results (with intentionally different-length arrays).

For lists: Validates item count only (lists are self-consistent).

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| data | Union[dict, list] | Yes | Data to validate |

**Returns:** None (raises exception if invalid)

**Raises:**
- `ValueError` — If dict has inconsistent column lengths

**The validation logic:**

```python
# For each column in dict:
lengths = {}
for key, value in data.items():
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        lengths[key] = len(value)
    else:
        lengths[key] = 1  # Scalar

# Check all non-scalar lengths are equal
unique_lengths = set(v for v in lengths.values() if v != 1)
if len(unique_lengths) > 1:
    raise ValueError(f"Inconsistent column lengths: {lengths}")
```

**Code example:**

```python
from app.core.data.serializers import validate_data_structure

# Valid data (all columns length 3)
valid = {
    'price': [1.0, 2.5, 3.7],
    'volume': [100, 200, 150],
    'symbol': 'EURUSD'  # Scalar ignored
}
validate_data_structure(valid)  # ✓ No exception

# Invalid data (mismatched lengths)
invalid = {
    'price': [1.0, 2.5, 3.7],      # Length 3
    'volume': [100, 200, 150, 50]  # Length 4
}
try:
    validate_data_structure(invalid)
except ValueError as e:
    print(e)  # → "Inconsistent column lengths: {'price': 3, 'volume': 4}"
```

**Edge cases:**

- **Metadata dicts:** Skipped (keys like 'preview', 'columns', 'record_count')
- **Signal generation results:** Skipped (keys like 'ml_dataset', 'signals', 'zones' have different lengths by design)
- **Empty dict:** Logs warning but does not raise exception

---

## Classes

### ParallelSerializer

[→ Source: `Backend/app/core/data/serializers.py` line 543](../../../Backend/app/core/data/serializers.py#L543)

> Handles parallelization of serialization for large datasets using multiprocessing to bypass Python's GIL.

**Inherits from:** None (static methods only)

**Used by:** `serialize_data()` when dataset size exceeds threshold (5K items for lists, 10K for dicts)

**Why it exists:**

Python's Global Interpreter Lock (GIL) prevents true parallelism in threads. For CPU-bound serialization of 100K+ rows, sequential processing takes 2-3 seconds. By splitting work across processes (which bypass GIL), we achieve 60% speed improvement on 4-core machines.

**What it does NOT do:**

- Does not handle small datasets (<500 items) — overhead exceeds benefit
- Does not preserve order unless using `imap()` (ordered iterator)
- Does not handle pickling errors gracefully — falls back to sequential on failure

---

#### Methods

##### serialize_list(val, n_workers=None, threshold=500, on_progress=None, numpy_safe=False) → List[Any]

[→ Source: `Backend/app/core/data/serializers.py` line 551](../../../Backend/app/core/data/serializers.py#L551)

Intelligently serializes a list. If list is large (>threshold), splits across CPU cores.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| val | List[Any] | Yes | List to serialize |
| n_workers | int | No | Number of worker processes (default: auto-detect CPU count) |
| threshold | int | No (default: 500) | Minimum list size for parallel processing |
| on_progress | Callable | No | Progress callback (percent, message) |
| numpy_safe | bool | No (default: False) | Preserve numpy arrays |

**Returns:** List[Any] — Serialized list (JSON-safe)

**The algorithm:**

```
1. If len(val) < threshold:
     → Use sequential to_serializable_records() or to_serializable()
     
2. Else:
     → Split list into n_workers chunks
     → Submit chunks to multiprocessing.Pool.imap()
     → Collect results in order (imap preserves submission order)
     → Concatenate chunks
     → Return final list
```

**Performance formula:**

```
speedup = sequential_time / parallel_time
        ≈ n_workers × 0.7  (70% efficiency due to overhead)

For 100K rows on 4-core machine:
  Sequential: 2.5 seconds
  Parallel:   0.9 seconds
  Speedup:    2.78× (close to theoretical 2.8×)
```

**Code example:**

```python
from app.core.data.serializers import ParallelSerializer
import numpy as np

# Large dataset (50K rows)
data = [
    {'price': np.float64(i * 1.5), 'volume': np.int64(i * 100)}
    for i in range(50000)
]

# Parallel serialization
result = ParallelSerializer.serialize_list(
    data,
    n_workers=4,
    threshold=500
)

print(len(result))  # → 50000
print(type(result[0]['price']))  # → <class 'float'> (native Python)
```

**Edge cases:**

- **Small lists (<500):** Falls back to sequential (overhead > benefit)
- **Pickling errors:** If worker fails, falls back to sequential processing
- **Order preservation:** Uses `imap()` (ordered) not `imap_unordered()` to maintain chronological order for time-series data

---

##### serialize_dict(val, n_workers=None, threshold=10000, on_progress=None, numpy_safe=False) → Dict[Any, Any]

[→ Source: `Backend/app/core/data/serializers.py` line 632](../../../Backend/app/core/data/serializers.py#L632)

Intelligently serializes a dictionary. If dict is large (>threshold), splits key-value pairs across cores.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| val | Dict[Any, Any] | Yes | Dictionary to serialize |
| n_workers | int | No | Number of worker processes (default: auto-detect) |
| threshold | int | No (default: 10000) | Minimum dict size for parallel processing |
| on_progress | Callable | No | Progress callback |
| numpy_safe | bool | No (default: False) | Preserve numpy arrays |

**Returns:** Dict[Any, Any] — Serialized dict (JSON-safe)

**The algorithm:**

```
1. Convert dict to list of (key, value) tuples
2. Split tuples into n_workers chunks
3. Serialize each chunk in parallel
4. Recombine into single dict (preserving key order)
```

**Code example:**

```python
from app.core.data.serializers import ParallelSerializer
import numpy as np

# Large dict (20K keys)
data = {
    f'feature_{i}': np.array([1.0, 2.0, 3.0])
    for i in range(20000)
}

result = ParallelSerializer.serialize_dict(
    data,
    n_workers=4,
    threshold=10000
)

print(len(result))  # → 20000
print(type(result['feature_0']))  # → <class 'list'>
```

**Edge cases:**

- **Small dicts (<10K):** Falls back to sequential
- **Key order:** Preserved via ordered recombination (Python 3.7+ dict order guarantee)

---

##### serialize(val, n_workers=None, list_threshold=10000, dict_threshold=10000, on_progress=None, numpy_safe=False) → Any

[→ Source: `Backend/app/core/data/serializers.py` line 693](../../../Backend/app/core/data/serializers.py#L693)

Main entry point for parallel serialization. Detects type and routes to appropriate method.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| val | Any | Yes | Value to serialize |
| n_workers | int | No | Number of workers |
| list_threshold | int | No (default: 10000) | Threshold for list parallelization |
| dict_threshold | int | No (default: 10000) | Threshold for dict parallelization |
| on_progress | Callable | No | Progress callback |
| numpy_safe | bool | No (default: False) | Preserve numpy arrays |

**Returns:** Any — Serialized value

**Routing logic:**

```
if isinstance(val, list):
    → serialize_list()
elif isinstance(val, pd.DataFrame):
    → Convert to records, then serialize_list()
elif isinstance(val, tuple):
    → Convert to list, serialize, convert back to tuple
elif isinstance(val, dict):
    → serialize_dict()
else:
    → Fallback to sequential to_serializable()
```

---

## Utility Functions

### extract_metadata_only(data) → dict

[→ Source: `Backend/app/core/data/serializers.py` line 728](../../../Backend/app/core/data/serializers.py#L728)

Extract only metadata from data **without full deserialization**. Used for status checks on multi-GB datasets without loading full data into memory.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| data | Union[dict, list] | Yes | Full result dict or list |

**Returns:** dict — Metadata only (no large arrays/data)

**Code example:**

```python
from app.core.data.serializers import extract_metadata_only

# Large result (100K rows)
result = {
    'data': [...],  # 100K rows
    'metadata': {'symbol': 'EURUSD', 'timeframe': 'H1'},
    'summary': {'total_signals': 42}
}

# Fast metadata extraction (no deserialization of 'data')
meta = extract_metadata_only(result)
print(meta)
# → {
#     'metadata': {'symbol': 'EURUSD', 'timeframe': 'H1'},
#     'summary': {'total_signals': 42},
#     'type': 'dict',
#     'keys': ['data', 'metadata', 'summary'],
#     'has_data': True,
#     'record_count': 100000
# }
```

**Performance:** <50ms (vs 1000ms+ for full deserialization)

---

### serialize_for_checkpoint(data, checkpoint_number, chunk_size=50000) → dict

[→ Source: `Backend/app/core/data/serializers.py` line 759](../../../Backend/app/core/data/serializers.py#L759)

Prepare data for checkpoint (partial result during long analysis). For tasks that take hours, save intermediate results every `chunk_size` records or every 15 minutes.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| data | Any | Yes | Partial result data |
| checkpoint_number | int | Yes | Which checkpoint this is (1, 2, 3, ...) |
| chunk_size | int | No (default: 50000) | How many records per checkpoint |

**Returns:** dict — Checkpoint metadata with progress tracking

**Code example:**

```python
from app.core.data.serializers import serialize_for_checkpoint

# After processing 50K rows (checkpoint 1)
checkpoint_meta = serialize_for_checkpoint(
    data=partial_results,
    checkpoint_number=1,
    chunk_size=50000
)

print(checkpoint_meta)
# → {
#     'checkpoint_number': 1,
#     'timestamp': '2024-01-15T10:30:00',
#     'chunk_size': 50000,
#     'data_type': 'list',
#     'record_count': 50000,
#     'total_expected': 50000,
#     'completion_percentage': 100.0
# }
```

---

## Summary

The `serializers.py` module is the **type conversion backbone** of the system. Every analysis result, every ML dataset, every API response flows through these functions. The parallel serialization optimization is critical for handling 100K+ row datasets without blocking the event loop for seconds.

**Key takeaways:**

1. **Use `to_serializable()` for small data** (<5K items) — simple, fast, no overhead
2. **Use `ParallelSerializer.serialize()` for large data** (>10K items) — 60% faster via multiprocessing
3. **Always validate structure** before DataFrame construction — catches mismatched column lengths early
4. **Use `numpy_safe=True` for ML datasets** — preserves exact dtypes for training

**Next file:** `session_data_loader.py`



---

# session_data_loader.py

> Authoritative data loading system that manages the enrichment pipeline and ensures each analysis step builds on the most recent dataset.

[→ Source: `Backend/app/core/data/session_data_loader.py`](../../../Backend/app/core/data/session_data_loader.py)

---

## Overview

This file solves the **data lineage problem**: In a multi-step analysis pipeline (data_source → technical → SNR → astronomical), each step enriches the dataset by adding columns. The question is: which dataset should the next step load?

**The answer:** Always load the **most enriched** dataset available, walking a priority chain from highest to lowest enrichment level.

**What it does:**

- Walks `SESSION_STEP_PRIORITY` list (astronomical → snr → technical → data_source) to find the most recent enriched dataset
- Prevents mutation steps from loading their own previous output (corruption safety)
- Atomically marks the latest result as "current data pointer" (race-condition-safe)
- Stores results in dual format: JSONB (preferred, <200MB) or pickle (fallback, compressed)
- Supports chunked storage for large results (10K rows per chunk)
- Implements batch streaming for incremental writes (5K-row batches with deduplication)
- Verifies data integrity with SHA-256 hashes on store/retrieve

**Where it sits:**

- **Called by:** All analysis steps (`technical_analysis`, `snr_analysis`, `astronomical_analysis`), ML preparation, model training
- **Calls:** `serializers.serialize_data()`, `serializers.deserialize_data()`, PostgreSQL `SessionStepResult` table

**Design decisions:**

1. **Why priority chain?** Each step adds columns. Loading the highest-priority step ensures the next step has all available features.
2. **Why exclude own output?** If SNR loads its own previous output and re-runs, it adds duplicate columns (e.g., `snr_signal`, `snr_signal_1`, `snr_signal_2`). Excluding prevents this.
3. **Why atomic mark?** If store succeeds but mark fails, subsequent steps load stale data. Atomic operation ensures consistency.
4. **Why dual format?** JSONB is queryable and preferred, but PostgreSQL has a 256MB limit. Pickle handles larger datasets.

---

## Quick reference

| Symbol | Type | Purpose |
|--------|------|---------|
| `SESSION_STEP_PRIORITY` | constant | Priority order for loading enriched data |
| `get_latest_session_data()` | function | Load most enriched dataset for session |
| `get_latest_session_data_excluding_step()` | function | Load data excluding specific step (mutation safety) |
| `get_current_data()` | function | Load dataset marked as current pointer |
| `set_as_current_data()` | function | Atomically mark latest result as current |
| `store_session_step_result()` | function | Store analysis result (JSONB or pickle) |
| `store_session_step_result_chunked()` | function | Store large result in 10K-row chunks |
| `store_batch_to_db()` | function | Write incremental 5K-row batch with deduplication |
| `validate_data_structure()` | function | Validate DataFrame structure before return |
| `verify_data_integrity()` | function | SHA-256 hash verification |
| `store_and_mark_current()` | function | Atomic store + mark operation |
| `create_ml_dataset()` | function | Create ML dataset record with JSONB/pickle storage |
| `append_sequences_to_ml_dataset()` | function | Append sequences to ML dataset (O(n) chunks or O(n²) blob) |
| `get_ml_datasets_for_session()` | function | List all ML datasets for session (metadata only) |

[→ Jump to source: Backend/app/core/data/session_data_loader.py](../../../Backend/app/core/data/session_data_loader.py)

---

## Constants

### SESSION_STEP_PRIORITY

[→ Source: `Backend/app/core/data/session_data_loader.py` line 155](../../../Backend/app/core/data/session_data_loader.py#L155)

Priority order for loading enriched data. Each step in the pipeline adds columns on top of the previous step's output, so we always want the highest step that has been completed.

```python
SESSION_STEP_PRIORITY = [
    'astronomical_analysis',  # Adds planetary / ephemeris columns
    'snr_analysis',            # Adds SNR signal columns
    'technical_analysis',      # Adds indicator columns
    'data_source',             # Raw OHLCV as stored by DataSource step
]
```

**Why this order?**

- `astronomical_analysis` is the most enriched (has all columns from previous steps + astronomical features)
- `snr_analysis` has technical indicators + SNR signals
- `technical_analysis` has raw OHLCV + indicators
- `data_source` is the baseline (raw OHLCV only)

**Usage:**

When a step needs data, it walks this list from top to bottom and loads the first step that has stored results. This ensures it always gets the richest dataset available.

---

## Core Functions

### get_latest_session_data(session_id, db, task_id="unknown", step_name=None) → List[Dict]

[→ Source: `Backend/app/core/data/session_data_loader.py` line 164](../../../Backend/app/core/data/session_data_loader.py#L164)

Load data for a given session. If `step_name` is provided, load THAT SPECIFIC STEP ONLY. If `step_name` is None, walk `SESSION_STEP_PRIORITY` from most-enriched to least-enriched.

**What it does:**

1. If `step_name` specified: Query `SessionStepResult` for that exact step
2. Else: Walk priority chain (astronomical → snr → technical → data_source)
3. For each step, try to load from `result_data_v2` (JSONB) or `result_data` (pickle)
4. Verify SHA-256 hash if available
5. Deserialize and validate structure
6. Restore numeric types (NaN handling)
7. Return list of dicts (rows)

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| session_id | str | Yes | Session UUID to load data for |
| db | AsyncSession | Yes | Async SQLAlchemy session |
| task_id | str | No (default: "unknown") | For logging context |
| step_name | str | No | If provided, load ONLY this step (overrides priority chain) |

**Returns:** List[Dict[str, Any]] — List of record dicts, or None if not found

**The algorithm:**

```
1. Determine steps_to_check:
     If step_name provided: [step_name]
     Else: SESSION_STEP_PRIORITY

2. For each step in steps_to_check:
     a. Query SessionStepResult WHERE session_id AND step_name
     b. If found:
          - Try result_data_v2 (JSONB) first
          - Verify hash if available
          - Deserialize
          - Extract 'data' field if dict
          - Restore numeric types
          - Validate structure
          - Return rows
     c. If not found or error: continue to next step

3. Final fallback: DataSession.raw_data (uploaded data)
```

**Code example:**

```python
from app.core.data.session_data_loader import get_latest_session_data

# Load most enriched data (walks priority chain)
data = await get_latest_session_data(
    session_id="abc-123",
    db=db,
    task_id="technical_analysis_456"
)

print(len(data))  # → 1000 rows
print(data[0].keys())  # → ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', ...]

# Load specific step only
snr_data = await get_latest_session_data(
    session_id="abc-123",
    db=db,
    step_name="snr_analysis"
)
```

**Edge cases:**

- **No data found:** Returns None (caller must handle)
- **Hash mismatch:** Logs warning but continues (data may have been modified)
- **Validation failure:** Raises `DataValidationError` with details

---

### get_latest_session_data_excluding_step(session_id, db, exclude_step, task_id="unknown") → List[Dict]

[→ Source: `Backend/app/core/data/session_data_loader.py` line 181](../../../Backend/app/core/data/session_data_loader.py#L181)

**PHASE 16 FIX:** Load most enriched data, but SKIP a specific step. Used by mutation steps (SNR, Technical, Astro) to prevent loading their own previous output and causing data corruption on re-runs.

**What it does:**

Same as `get_latest_session_data()`, but skips the specified step when walking the priority chain.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| session_id | str | Yes | Session UUID |
| db | AsyncSession | Yes | Async SQLAlchemy session |
| exclude_step | str | Yes | Step name to skip (e.g., 'snr_analysis', 'technical_analysis') |
| task_id | str | No | For logging |

**Returns:** List[Dict[str, Any]] — Most enriched data EXCLUDING the specified step

**Why this exists:**

**Problem:** If SNR analysis loads its own previous output and re-runs, it adds duplicate columns:
- First run: Adds `snr_signal`, `snr_strength`
- Second run (loading own output): Adds `snr_signal_1`, `snr_strength_1`
- Third run: Adds `snr_signal_2`, `snr_strength_2`

**Solution:** SNR calls `get_latest_session_data_excluding_step(..., exclude_step='snr_analysis')`, which returns data from `technical_analysis` or `data_source` instead.

**Code example:**

```python
# SNR analysis re-run (prevent loading own output)
data = await get_latest_session_data_excluding_step(
    session_id="abc-123",
    db=db,
    exclude_step='snr_analysis',  # Skip SNR's own output
    task_id="snr_rerun_789"
)

# Result: Loads from 'technical_analysis' or 'data_source' instead
# No duplicate columns!
```

---

### set_as_current_data(session_id, db, task_id="unknown") → None

[→ Source: `Backend/app/core/data/session_data_loader.py` line 289](../../../Backend/app/core/data/session_data_loader.py#L289)

**PHASE 16 FIX:** Mark the most recently stored result as THE CURRENT DATA POINTER. **ATOMIC:** Uses single SQL UPDATE with CASE to prevent race conditions.

**What it does:**

1. Find the latest stored result for this session (ORDER BY stored_at DESC LIMIT 1)
2. Execute atomic UPDATE: Set `is_current_data=TRUE` for latest, `FALSE` for all others
3. This is atomic at database level — no race condition possible even if called simultaneously

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| session_id | str | Yes | Session UUID |
| db | AsyncSession | Yes | Async SQLAlchemy session |
| task_id | str | No | For logging |

**Returns:** None (raises exception on failure)

**The atomic SQL:**

```sql
UPDATE session_step_results
SET is_current_data = CASE
    WHEN id = :latest_id THEN TRUE
    ELSE FALSE
END
WHERE session_id = :session_id
```

**Why atomic?**

**Problem:** If two steps finish simultaneously and both call `set_as_current_data()`:
- Step A: Query latest → ID=42
- Step B: Query latest → ID=43
- Step A: UPDATE id=42 to TRUE
- Step B: UPDATE id=43 to TRUE
- **Result:** Two rows marked as current (inconsistent state)

**Solution:** Single UPDATE with CASE ensures only one row can have `is_current_data=TRUE` per session.

**Code example:**

```python
# After storing result
await store_session_step_result(
    session_id="abc-123",
    step_name="snr_analysis",
    data=enriched_data,
    db=db
)

# Mark as current (atomic)
await set_as_current_data(
    session_id="abc-123",
    db=db,
    task_id="snr_analysis_456"
)
```

---

### store_session_step_result(session_id, step_name, data, db, is_compressed=True, force_pickle=False, pre_serialized_data=None, pre_serialized_hash=None, on_progress=None) → None

[→ Source: `Backend/app/core/data/session_data_loader.py` line 337](../../../Backend/app/core/data/session_data_loader.py#L337)

Centralized helper for storing analysis results to the database. Uses **JSONB (preferred)** with fallback to **pickle** if JSONB fails or is too large.

**What it does:**

1. If `pre_serialized_data` provided: Skip serialization, use directly (CPU optimization)
2. Else: Serialize data
   - If `force_pickle=True` or step in ['data_source', 'snr_analysis', 'astronomical_analysis', 'technical_analysis']: Use pickle
   - Else: Try JSONB first (convert to JSON-safe format)
   - If JSONB >200MB or fails: Fall back to pickle
3. Compute SHA-256 hash for integrity verification
4. UPSERT to `SessionStepResult` table (ON CONFLICT DO UPDATE)
5. Commit transaction

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| session_id | str | Yes | Session UUID |
| step_name | str | Yes | Analysis step name (e.g., 'technical_analysis') |
| data | Union[pd.DataFrame, list, dict] | Yes | Data to store |
| db | AsyncSession | Yes | Async SQLAlchemy session |
| is_compressed | bool | No (default: True) | Whether to compress data (pickle only) |
| force_pickle | bool | No (default: False) | If True, skip JSONB and use pickle directly |
| pre_serialized_data | str | No | PRE-SERIALIZED data (avoids CPU work inside lock) |
| pre_serialized_hash | str | No | Hash of pre-serialized data |
| on_progress | Callable | No | Progress callback (percent, message) |

**Returns:** None (raises exception on failure)

**The decision tree:**

```
if pre_serialized_data:
    → Use directly (skip serialization)
elif force_pickle or step in ['data_source', 'snr_analysis', 'astronomical_analysis', 'technical_analysis']:
    → Use pickle format
else:
    → Try JSONB:
        if size > 200MB or JSONB fails:
            → Fall back to pickle
```

**Code example:**

```python
import pandas as pd
from app.core.data.session_data_loader import store_session_step_result

# Store technical analysis result
df = pd.DataFrame({
    'Time': [...],
    'Open': [...],
    'RSI': [...],
    'MACD': [...]
})

await store_session_step_result(
    session_id="abc-123",
    step_name="technical_analysis",
    data=df,
    db=db,
    is_compressed=True,
    force_pickle=True  # Force pickle for consistency
)
```

**Edge cases:**

- **JSONB size limit:** If data >200MB, automatically falls back to pickle
- **PostgreSQL limit:** If JSONB fails at SQL execution level (>256MB), retries with compressed pickle
- **Pre-serialized optimization:** If caller pre-serializes data outside DB lock, passes `pre_serialized_data` to avoid CPU work during transaction

---

### store_batch_to_db(session_id, step_name, batch_data, db, batch_number, total_batches, prev_batch_hashes=None, task_store=None) → Dict

[→ Source: `Backend/app/core/data/session_data_loader.py` line 1009](../../../Backend/app/core/data/session_data_loader.py#L1009)

Store a single batch of results with **deduplication** against previous batch's overlap window. This is the core of the **streaming architecture**.

**What it does:**

Instead of accumulating all 189K rows in memory and storing once, each 5K-row batch is:
1. Deduplicated against overlapping rows from previous batch (using hash of index + key fields)
2. Serialized and stored to DB as separate row (`step_name__batch_{N}`)
3. Cleared from memory

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| session_id | str | Yes | Session UUID |
| step_name | str | Yes | Analysis step ('snr_analysis', 'technical_analysis', 'astronomical_analysis') |
| batch_data | Dict[str, Any] | Yes | Batch results with keys: 'signals', 'indicators', 'astro_features', 'rows_processed', 'batch_range' |
| db | AsyncSession | Yes | Async SQLAlchemy session |
| batch_number | int | Yes | Current batch number (1-indexed) |
| total_batches | int | Yes | Total number of batches |
| prev_batch_hashes | set | No | Set of hashes from previous batch's last 100 rows (for dedup) |
| task_store | TaskStore | No | TaskStore for progress updates |

**Returns:** Dict with batch statistics:
```python
{
    'rows_inserted': int,
    'rows_skipped': int,  # Duplicates
    'rows_failed': int,
    'batch_hashes': set,  # Pass to next batch for dedup
    'batch_number': int,
    'total_batches': int,
    'storage_format': 'pickle' or 'jsonb',
    'size_kb': float
}
```

**The deduplication algorithm:**

```
For each row in batch:
    1. Compute hash = SHA256(index + type + price)
    2. If hash in prev_batch_hashes:
         → Skip (duplicate from overlap window)
    3. Else:
         → Add to rows_to_write
         → Add hash to batch_hashes

Store rows_to_write to DB
Return batch_hashes for next batch
```

**Code example:**

```python
from app.core.data.session_data_loader import store_batch_to_db

# Process batches incrementally
prev_hashes = set()

for batch_num in range(1, total_batches + 1):
    batch_data = process_batch(batch_num)  # Returns signals, indicators, etc.
    
    result = await store_batch_to_db(
        session_id="abc-123",
        step_name="snr_analysis",
        batch_data=batch_data,
        db=db,
        batch_number=batch_num,
        total_batches=total_batches,
        prev_batch_hashes=prev_hashes
    )
    
    # Pass hashes to next batch for dedup
    prev_hashes = result['batch_hashes']
    
    print(f"Batch {batch_num}: inserted={result['rows_inserted']}, skipped={result['rows_skipped']}")
```

**Performance:**

- **Memory:** Constant (only one 5K-row batch in memory at a time)
- **Deduplication:** O(1) hash lookup per row
- **Storage:** Each batch stored separately (no re-fetching previous batches)

---

### store_and_mark_current(session_id, step_name, data, db, task_id="unknown", is_compressed=True) → bool

[→ Source: `Backend/app/core/data/session_data_loader.py` line 1577](../../../Backend/app/core/data/session_data_loader.py#L1577)

**ATOMIC operation:** Store step result AND mark as current data pointer. If either operation fails, both are rolled back — no inconsistent state.

**What it does:**

1. Store result via `store_session_step_result()`
2. Mark as current via `set_as_current_data()`
3. Explicit commit to ensure both operations are persisted
4. If ANY operation fails, rollback BOTH

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| session_id | str | Yes | Session UUID |
| step_name | str | Yes | Step name |
| data | Union[pd.DataFrame, list, dict] | Yes | Data to store |
| db | AsyncSession | Yes | Async SQLAlchemy session |
| task_id | str | No | For logging |
| is_compressed | bool | No (default: True) | Whether to compress data |

**Returns:** bool — True if successful

**Raises:** `AtomicOperationError` — If store or mark fails (includes rollback)

**Why atomic?**

**Problem:** If store succeeds but mark fails:
- Data is stored in DB
- But `is_current_data` flag not set
- Next step loads stale data (previous step's output)
- Analysis pipeline breaks

**Solution:** Wrap both operations in single transaction. If either fails, rollback both.

**Code example:**

```python
from app.core.data.session_data_loader import store_and_mark_current

try:
    success = await store_and_mark_current(
        session_id="abc-123",
        step_name="snr_analysis",
        data=enriched_data,
        db=db,
        task_id="snr_456"
    )
    
    if success:
        print("✅ Data stored and marked as current")
except AtomicOperationError as e:
    print(f"❌ Atomic operation failed: {e}")
    # Both store and mark were rolled back
```

---

## ML Dataset Functions

### create_ml_dataset(session_id, dataset_name, output_targets, features_x, targets_y, source_step, scaling_config, split_config, feature_columns, db, **metadata) → Optional[str]

[→ Source: `Backend/app/core/data/session_data_loader.py` line 1806](../../../Backend/app/core/data/session_data_loader.py#L1806)

Create and store a new MLDataset record for a session. Uses **JSONB storage (preferred)** with a **LargeBinary pickle fallback** for datasets that are too large to fit in JSONB.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| session_id | str | Yes | Session UUID string |
| dataset_name | str | Yes | Human-readable name (e.g., 'snr_signals__type__direction') |
| output_targets | List[str] | Yes | Target column names (e.g., ['signal_type']) |
| features_x | np.ndarray or list | Yes | Feature matrix |
| targets_y | np.ndarray or list | Yes | Target matrix |
| source_step | str | Yes | Source analysis step name (e.g., 'snr_analysis') |
| scaling_config | Dict | Yes | Scaler params {mean: [...], std: [...]} |
| split_config | Dict | Yes | Split info {train_size, val_size, test_size, ...} |
| feature_columns | List[str] | Yes | Column names for features |
| db | AsyncSession | Yes | Async SQLAlchemy session |
| **metadata | Dict | No | Extra fields: preprocessing_steps, null_percentage, etc. |

**Returns:** str — dataset_id (UUID) on success, None on failure

**Code example:**

```python
import numpy as np
from app.core.data.session_data_loader import create_ml_dataset

features = np.random.rand(1000, 60, 40)  # (samples, sequence_length, features)
targets = np.random.randint(0, 5, size=(1000,))  # Classification labels

dataset_id = await create_ml_dataset(
    session_id="abc-123",
    dataset_name="snr_signals__type__direction",
    output_targets=["signal_type"],
    features_x=features,
    targets_y=targets,
    source_step="snr_analysis",
    scaling_config={"mean": [...], "std": [...]},
    split_config={"train_size": 700, "val_size": 150, "test_size": 150},
    feature_columns=[f"feature_{i}" for i in range(40)],
    db=db
)

print(f"Created dataset: {dataset_id}")
```

---

### append_sequences_to_ml_dataset(dataset_id, sequences, labels, targets, split_name, db, use_chunk_table=True, sequence_metadata=None) → bool

[→ Source: `Backend/app/core/data/session_data_loader.py` line 1918](../../../Backend/app/core/data/session_data_loader.py#L1918)

Append a chunk of sequences to an existing ML dataset split. Two storage modes available:

1. **BLOB (use_chunk_table=False):** O(n²) but simpler, backward compatible
   - Decompress → Concat → Recompress (re-fetches all previous chunks)

2. **CHUNKS (use_chunk_table=True):** O(n) modern approach
   - Just compress + INSERT (no re-fetching)
   - 4 chunks: 6.5× faster than blob
   - 100 chunks: 40× faster than blob

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| dataset_id | str | Yes | UUID of the existing dataset |
| sequences | np.ndarray | Yes | New sequences to append |
| labels | np.ndarray | Yes | New labels to append |
| targets | Dict[str, np.ndarray] | Yes | Dict of target arrays to append {target_name: np.ndarray} |
| split_name | str | Yes | Which split to append to ('train', 'validation', or 'test') |
| db | AsyncSession | Yes | Async SQLAlchemy session |
| use_chunk_table | bool | No (default: True) | If True, use new chunk table (O(n)); else blob (O(n²)) |
| sequence_metadata | list | No | Optional list of per-sequence dicts (OHLCV snapshots, anchor_index, target_structure) |

**Returns:** bool — True on success, False on failure

**Performance comparison:**

```
Blob (O(n²)):
  Chunk 1: Decompress 0 + Concat 1K + Compress 1K = 50ms
  Chunk 2: Decompress 1K + Concat 1K + Compress 2K = 150ms
  Chunk 3: Decompress 2K + Concat 1K + Compress 3K = 300ms
  Chunk 4: Decompress 3K + Concat 1K + Compress 4K = 500ms
  Total: 1000ms

Chunks (O(n)):
  Chunk 1: Compress 1K + INSERT = 50ms
  Chunk 2: Compress 1K + INSERT = 50ms
  Chunk 3: Compress 1K + INSERT = 50ms
  Chunk 4: Compress 1K + INSERT = 50ms
  Total: 200ms (5× faster)
```

**Code example:**

```python
import numpy as np
from app.core.data.session_data_loader import append_sequences_to_ml_dataset

# Append chunk to training split
sequences = np.random.rand(1000, 60, 40)
labels = np.random.randint(0, 5, size=(1000,))
targets = {"signal_type": labels}

success = await append_sequences_to_ml_dataset(
    dataset_id="dataset-uuid-123",
    sequences=sequences,
    labels=labels,
    targets=targets,
    split_name="train",
    db=db,
    use_chunk_table=True  # Use O(n) chunks
)

if success:
    print("✅ Chunk appended successfully")
```

---

## Summary

The `session_data_loader.py` module is the **data pipeline orchestrator** for the entire system. It ensures that:

1. Each analysis step builds on the most enriched dataset available
2. Steps don't corrupt data by loading their own previous output
3. The "current data pointer" is always consistent (atomic operations)
4. Large datasets are stored efficiently (chunked or batched)
5. Data integrity is verified (SHA-256 hashes)

**Key takeaways:**

1. **Always use `get_latest_session_data_excluding_step()`** in mutation steps to prevent loading own output
2. **Always use `store_and_mark_current()`** instead of separate store + mark calls (atomic safety)
3. **Use chunked storage** for large results (>100K rows) to avoid memory issues
4. **Use batch streaming** for incremental writes (prevents 189K-row memory accumulation)

**Next file:** `session_dataset_registry.py`


---

# session_dataset_registry.py

> Three-tier caching architecture (TIER 0: pointer, TIER 1: LRU memory cache, TIER 2: PostgreSQL) with ZSTANDARD compression for ML training datasets.

[→ Source: `Backend/app/core/data/session_dataset_registry.py`](../../../Backend/app/core/data/session_dataset_registry.py)

---

## Overview

This file solves the **ML dataset access problem**: Training datasets are large (40-200MB per split), and fetching from PostgreSQL on every training run is slow (150-300ms). But keeping all datasets in memory is wasteful (most are accessed once and never again).

**The solution:** Three-tier caching with automatic eviction and compression.

**What it does:**

- **TIER 0:** Single pointer (`_current_pointer_id`) to the active training dataset (managed by AnalysisManager)
- **TIER 1:** LRU cache of last 5 accessed datasets in memory (4-hour TTL, ~40MB each)
- **TIER 2:** PostgreSQL `ml_datasets` table with ZSTANDARD-compressed arrays (persistent storage)
- **Compression:** `CompressionHandler` achieves 4:1 compression ratio (75% size reduction) using zstd level 10
- **Smart tiering:** `get_dataset()` checks TIER 1 first (<1ms), fetches from TIER 2 on miss (150-300ms), then caches
- **Lineage tracking:** `get_dataset_lineage()` traces ancestry chain (SNR → ML → future steps)
- **Regression targets:** Stores multi-target regression arrays separately (`train_targets`, `validation_targets`, `test_targets`)

**Where it sits:**

- **Called by:** `AnalysisManager` (ML preparation), model training workers, inference pipeline
- **Calls:** `CompressionHandler` for zstd compression, PostgreSQL `ml_datasets` table

**Design decisions:**

1. **Why three tiers?** TIER 0 (pointer) is instant, TIER 1 (cache) is fast (<1ms), TIER 2 (DB) is persistent. Each tier optimizes for different access patterns.
2. **Why LRU eviction?** Most datasets are accessed once during training, then never again. LRU keeps recently-used datasets hot.
3. **Why 4-hour TTL?** Training sessions typically last 1-2 hours. 4 hours ensures datasets stay cached during active development.
4. **Why ZSTANDARD?** Achieves better compression ratio than gzip (4:1 vs 3:1) with similar speed. Level 10 balances compression vs CPU time.

---

## Quick reference

| Symbol | Type | Purpose |
|--------|------|---------|
| `SessionDatasetRegistry` | class | Three-tier cache manager for ML datasets |
| `CompressionHandler` | class | ZSTANDARD compression/decompression utilities |
| `DatasetMetadata` | dataclass | Immutable metadata for a dataset |
| `TIER1_MAX_DATASETS` | constant | Max datasets in LRU cache (5) |
| `TIER1_TTL_MINUTES` | constant | Time-to-live for cache entries (240 = 4 hours) |
| `COMPRESSION_LEVEL` | constant | ZSTANDARD compression level (10) |

[→ Jump to source: Backend/app/core/data/session_dataset_registry.py](../../../Backend/app/core/data/session_dataset_registry.py)

---

## Data Classes

### DatasetMetadata

[→ Source: `Backend/app/core/data/session_dataset_registry.py` line 44](../../../Backend/app/core/data/session_dataset_registry.py#L44)

Immutable metadata for a dataset. Used for listing datasets without loading heavy arrays.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| dataset_id | str | Unique dataset identifier (UUID) |
| session_id | str | Parent session UUID |
| dataset_name | str | Human-readable name |
| source_step | str | "snr_analysis" or "ml_preparation" |
| parent_dataset_id | str | For lineage tracking (SNR → ML) |
| output_targets | Dict[str, Any] | Target configuration |
| feature_selection | Dict[str, Any] | Feature selection metadata |
| metadata | Dict[str, Any] | Additional metadata |
| compression_type | str | "zstandard" |
| compression_ratio | float | Original size / compressed size |
| uncompressed_size_mb | int | Original size in MB |
| compressed_size_mb | int | Compressed size in MB |
| target_metadata | Dict[str, Any] | Target types, shapes, class mappings |
| created_at | str | ISO 8601 timestamp |

**Code example:**

```python
from app.core.data.session_dataset_registry import DatasetMetadata

meta = DatasetMetadata(
    dataset_id="abc-123",
    session_id="session-456",
    dataset_name="snr_signals__type__direction",
    source_step="snr_analysis",
    parent_dataset_id=None,
    output_targets={"signal_type": "classification"},
    feature_selection={"mode": "rich", "feature_count": 40},
    metadata={},
    compression_type="zstandard",
    compression_ratio=4.2,
    uncompressed_size_mb=160,
    compressed_size_mb=38,
    target_metadata={"target_names": ["signal_type"]},
    created_at="2024-01-15T10:30:00"
)

print(meta.to_dict())  # Convert to dict for JSON serialization
```

---

## Classes

### CompressionHandler

[→ Source: `Backend/app/core/data/session_dataset_registry.py` line 73](../../../Backend/app/core/data/session_dataset_registry.py#L73)

> Handle ZSTANDARD compression/decompression for datasets.

**Inherits from:** None (static methods only)

**Used by:** `SessionDatasetRegistry.register_dataset()`, `SessionDatasetRegistry.get_dataset()`

**Why it exists:**

NumPy arrays are large (40-200MB per split). Storing uncompressed wastes disk space and slows down network transfer. ZSTANDARD achieves 4:1 compression ratio (75% reduction) with minimal CPU overhead.

---

#### Methods

##### compress(data) → Tuple[bytes, float, int]

[→ Source: `Backend/app/core/data/session_dataset_registry.py` line 79](../../../Backend/app/core/data/session_dataset_registry.py#L79)

Compress data using ZSTANDARD level 10.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| data | Any | Yes | Object to compress (will be pickled first) |

**Returns:** Tuple of (compressed_bytes, compression_ratio, compressed_size_mb)
- `compression_ratio`: original_size / compressed_size (e.g., 4.0 = 75% reduction)
- `compressed_size_mb`: Size in MB

**The algorithm:**

```
1. Pickle data with HIGHEST_PROTOCOL
2. Compress with zstd level 10
3. Calculate ratio = original_size / compressed_size
4. Return (compressed_bytes, ratio, size_mb)
```

**Code example:**

```python
import numpy as np
from app.core.data.session_dataset_registry import CompressionHandler

# Compress training data
train_data = np.random.rand(3000, 60, 40)  # 57.6 MB uncompressed

compressed, ratio, size_mb = CompressionHandler.compress(train_data)

print(f"Original: 57.6 MB")
print(f"Compressed: {size_mb:.1f} MB")
print(f"Ratio: {ratio:.1f}× (saved {(1 - 1/ratio)*100:.0f}%)")

# Output:
# Original: 57.6 MB
# Compressed: 13.7 MB
# Ratio: 4.2× (saved 76%)
```

**Performance:**

- **Speed:** ~200 MB/s compression, ~800 MB/s decompression (on modern CPU)
- **Ratio:** 4:1 typical for float32 arrays (better than gzip's 3:1)

---

##### decompress(compressed) → Any

[→ Source: `Backend/app/core/data/session_dataset_registry.py` line 115](../../../Backend/app/core/data/session_dataset_registry.py#L115)

Decompress ZSTANDARD data.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| compressed | bytes | Yes | Compressed bytes |

**Returns:** Any — Decompressed object

**Code example:**

```python
from app.core.data.session_dataset_registry import CompressionHandler

# Round-trip test
original = np.random.rand(1000, 60, 40)
compressed, _, _ = CompressionHandler.compress(original)
restored = CompressionHandler.decompress(compressed)

print(np.allclose(original, restored))  # → True
```

---

### SessionDatasetRegistry

[→ Source: `Backend/app/core/data/session_dataset_registry.py` line 138](../../../Backend/app/core/data/session_dataset_registry.py#L138)

> Manages datasets in a session with three-tier caching.

**Inherits from:** None

**Used by:** `AnalysisManager` (ML preparation), model training workers

**Why it exists:**

Without caching, every training run fetches 40-200MB from PostgreSQL (150-300ms latency). With three-tier caching:
- TIER 1 hit: <1ms (memory lookup)
- TIER 1 miss: 150-300ms (DB fetch + decompress), then cached for future hits

**What it does NOT do:**

- Does not handle model training (only dataset storage/retrieval)
- Does not validate dataset structure (caller's responsibility)
- Does not automatically delete old datasets (user must explicitly delete)

---

#### Constructor

```python
SessionDatasetRegistry(session_id: str, db_connection: Any)
```

Initialize registry for a session.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| session_id | str | Yes | Unique session identifier |
| db_connection | Any | Yes | Database connection object (PostgreSQL async driver) |

**Memory allocated:**

- ~50-200 MB for TIER 1 cache (5 datasets × 40 MB avg)
- Minimal runtime overhead otherwise

**Code example:**

```python
from app.core.data.session_dataset_registry import SessionDatasetRegistry

# Initialize registry
registry = SessionDatasetRegistry(
    session_id="abc-123",
    db_connection=db
)

# Use as async context manager (auto-cleanup)
async with registry:
    dataset = await registry.get_dataset("dataset-uuid-456")
    # ... use dataset ...
# Registry cleaned up automatically
```

---

#### Methods

##### register_dataset(dataset_id, dataset_name, train_data, validation_data, test_data, train_labels, validation_labels, test_labels, source_step, output_targets=None, feature_selection=None, metadata=None, parent_dataset_id=None, train_targets=None, validation_targets=None, test_targets=None) → bool

[→ Source: `Backend/app/core/data/session_dataset_registry.py` line 289](../../../Backend/app/core/data/session_dataset_registry.py#L289)

Register a new dataset in the registry.

**What it does:**

1. Compress data with ZSTANDARD
2. Store compressed in TIER 2 (DB)
3. Load to TIER 1 cache
4. Update TIER 0 pointer
5. Track metadata (lineage, feature_selection, etc.)

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| dataset_id | str | Yes | Unique dataset ID |
| dataset_name | str | Yes | Human-readable name |
| train_data | np.ndarray | Yes | Training feature array (sequences or flat) |
| validation_data | np.ndarray | Yes | Validation feature array |
| test_data | np.ndarray | Yes | Test feature array |
| train_labels | np.ndarray | Yes | Training labels |
| validation_labels | np.ndarray | Yes | Validation labels |
| test_labels | np.ndarray | Yes | Test labels |
| source_step | str | Yes | "snr_analysis" or "ml_preparation" |
| output_targets | Dict[str, Any] | No | {"signal_type": "bounce_support", ...} |
| feature_selection | Dict[str, Any] | No | {"mode": "rich", "feature_count": 40, ...} |
| metadata | Dict[str, Any] | No | Additional metadata dict |
| parent_dataset_id | str | No | For lineage tracking (SNR → ML) |
| train_targets | Dict[str, np.ndarray] | No | Dict of training regression targets {target_name: np.array(N, prediction_length)} |
| validation_targets | Dict[str, np.ndarray] | No | Dict of validation regression targets |
| test_targets | Dict[str, np.ndarray] | No | Dict of test regression targets |

**Returns:** bool — True if successful, False if failed

**Memory Impact:**

- Compresses data (75% reduction typical)
- Stores in TIER 1 cache (~40MB)
- DB storage only (does not keep copy in app memory beyond TIER 1)

**Code example:**

```python
import numpy as np
from app.core.data.session_dataset_registry import SessionDatasetRegistry

registry = SessionDatasetRegistry(session_id="abc-123", db_connection=db)

# Prepare splits
train_data = np.random.rand(3000, 60, 40)
val_data = np.random.rand(500, 60, 40)
test_data = np.random.rand(500, 60, 40)

train_labels = np.random.randint(0, 5, size=(3000,))
val_labels = np.random.randint(0, 5, size=(500,))
test_labels = np.random.randint(0, 5, size=(500,))

# Register dataset
success = await registry.register_dataset(
    dataset_id="dataset-uuid-456",
    dataset_name="snr_signals__type__direction",
    train_data=train_data,
    validation_data=val_data,
    test_data=test_data,
    train_labels=train_labels,
    validation_labels=val_labels,
    test_labels=test_labels,
    source_step="snr_analysis",
    output_targets={"signal_type": "classification"},
    feature_selection={"mode": "rich", "feature_count": 40}
)

if success:
    print("✅ Dataset registered and cached")
```

---

##### get_dataset(dataset_id) → Optional[Dict[str, Any]]

[→ Source: `Backend/app/core/data/session_dataset_registry.py` line 234](../../../Backend/app/core/data/session_dataset_registry.py#L234)

Fetch dataset with smart tiering.

**Strategy:**

1. Check TIER 1 cache (< 1ms) → return immediately
2. Miss: Fetch from TIER 2 DB (100-300ms)
3. Decompress and load to TIER 1
4. Return decompressed data

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| dataset_id | str | Yes | Dataset to fetch |

**Returns:** Dict with keys: train_data, validation_data, test_data, train_labels, validation_labels, test_labels, metadata — or None if not found

**Performance:**

- **TIER 1 hit:** ~0.1ms return time
- **TIER 1 miss:** ~150-300ms (DB fetch + decompress)

**Code example:**

```python
from app.core.data.session_dataset_registry import SessionDatasetRegistry

registry = SessionDatasetRegistry(session_id="abc-123", db_connection=db)

# First access (TIER 1 miss)
dataset = await registry.get_dataset("dataset-uuid-456")
# → 200ms (DB fetch + decompress + cache)

# Second access (TIER 1 hit)
dataset = await registry.get_dataset("dataset-uuid-456")
# → 0.1ms (memory lookup)

print(dataset['train_data'].shape)  # → (3000, 60, 40)
print(dataset['metadata']['dataset_name'])  # → "snr_signals__type__direction"
```

---

##### list_datasets(source_step=None, parent_dataset_id=None) → List[DatasetMetadata]

[→ Source: `Backend/app/core/data/session_dataset_registry.py` line 1009](../../../Backend/app/core/data/session_dataset_registry.py#L1009)

List all datasets in session.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| source_step | str | No | Filter by "snr_analysis" or "ml_preparation" |
| parent_dataset_id | str | No | Filter by parent (for lineage) |

**Returns:** List[DatasetMetadata] — List of dataset metadata objects

**Code example:**

```python
from app.core.data.session_dataset_registry import SessionDatasetRegistry

registry = SessionDatasetRegistry(session_id="abc-123", db_connection=db)

# List all datasets
datasets = await registry.list_datasets()

for ds in datasets:
    print(f"{ds.dataset_name}: {ds.sample_count} samples, {ds.compression_ratio:.1f}× compression")

# Filter by source step
snr_datasets = await registry.list_datasets(source_step="snr_analysis")
ml_datasets = await registry.list_datasets(source_step="ml_preparation")
```

---

##### get_dataset_lineage(dataset_id) → List[DatasetMetadata]

[→ Source: `Backend/app/core/data/session_dataset_registry.py` line 1117](../../../Backend/app/core/data/session_dataset_registry.py#L1117)

Get lineage chain (SNR → ML → ...future...). Shows full ancestry of a dataset.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| dataset_id | str | Yes | Starting dataset |

**Returns:** List[DatasetMetadata] — List of ancestor datasets (newest to oldest)

**Code example:**

```python
from app.core.data.session_dataset_registry import SessionDatasetRegistry

registry = SessionDatasetRegistry(session_id="abc-123", db_connection=db)

# Get lineage
lineage = await registry.get_dataset_lineage("ml-dataset-uuid-789")

for i, ds in enumerate(lineage):
    print(f"Step {i}: {ds.dataset_name} ({ds.source_step})")

# Output:
# Step 0: ml_prepared_v2 (ml_preparation)
# Step 1: snr_signals__type__direction (snr_analysis)
```

---

##### delete_dataset(dataset_id) → bool

[→ Source: `Backend/app/core/data/session_dataset_registry.py` line 1181](../../../Backend/app/core/data/session_dataset_registry.py#L1181)

Delete a dataset (both TIER 1 and TIER 2).

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| dataset_id | str | Yes | Dataset to delete |

**Returns:** bool — True if successful

**Code example:**

```python
from app.core.data.session_dataset_registry import SessionDatasetRegistry

registry = SessionDatasetRegistry(session_id="abc-123", db_connection=db)

# Delete dataset
success = await registry.delete_dataset("dataset-uuid-456")

if success:
    print("✅ Dataset deleted from cache and DB")
```

---

##### get_registry_stats() → Dict[str, Any]

[→ Source: `Backend/app/core/data/session_dataset_registry.py` line 1217](../../../Backend/app/core/data/session_dataset_registry.py#L1217)

Get registry statistics for monitoring.

**Returns:** Dict with cache stats, DB stats, etc.

**Code example:**

```python
from app.core.data.session_dataset_registry import SessionDatasetRegistry

registry = SessionDatasetRegistry(session_id="abc-123", db_connection=db)

stats = await registry.get_registry_stats()

print(stats)
# → {
#     "tier0_pointer": "dataset-uuid-456",
#     "tier1_cache_size": 3,
#     "tier1_max": 5,
#     "tier1_ttl_minutes": 240,
#     "tier2_total_datasets": 12,
#     "tier2_stored_mb": 456.7,
#     "tier2_original_mb": 1920.3,
#     "tier2_avg_compression": 4.2
# }
```

---

## Summary

The `session_dataset_registry.py` module is the **ML dataset cache manager** for the system. It ensures that:

1. Datasets are compressed efficiently (4:1 ratio, 75% size reduction)
2. Recently-used datasets are cached in memory (TIER 1 LRU cache)
3. All datasets are persisted to PostgreSQL (TIER 2)
4. Dataset lineage is tracked (SNR → ML → future steps)
5. Memory usage is bounded (max 5 datasets in cache, 4-hour TTL)

**Key takeaways:**

1. **Always use `register_dataset()`** to store new datasets (handles compression + caching automatically)
2. **Use `get_dataset()` for training** (smart tiering ensures fast access)
3. **Use `list_datasets()` for UI dropdowns** (lightweight, no heavy arrays)
4. **Use `get_dataset_lineage()` for provenance tracking** (shows full ancestry chain)

---
