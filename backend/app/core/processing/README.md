# Backend/app/core/processing

> Data processing orchestration, task management, progress tracking, and resource estimation for the trading analysis platform.

---

## Overview

The `processing/` module is the **orchestration layer** that coordinates the entire analysis pipeline. It sits between the API layer and the analysis engines, managing execution strategies, progress tracking, checkpointing, and result storage.

**Key responsibilities:**

1. **Strategy auto-selection** — Determines optimal processing strategy (Sequential, Parallel, Streaming) based on dataset size
2. **Task orchestration** — Coordinates multi-step analysis pipelines (technical → SNR → astronomical → ML)
3. **Progress tracking** — Real-time progress updates via WebSocket with monotonic guarantees
4. **Resource estimation** — Predicts memory, CPU, and time requirements before execution
5. **Checkpoint & recovery** — Saves intermediate results for long-running tasks
6. **Result aggregation** — Combines slice results while preventing memory peaks

**Where it sits:**

```
API Endpoint (analysis.py)
    ↓
ProcessingManager (orchestration)
    ↓
Strategy (Sequential/Parallel/Streaming)
    ↓
Handler (TechnicalIndicators/SNR/Astronomical/ML)
    ↓
Result Storage (TIER 1 cache + Database)
```

**Design decisions:**

1. **Why strategy auto-selection?** Different dataset sizes need different approaches. Small datasets (<10K rows) use sequential processing. Large datasets (>100K rows) use streaming to prevent memory issues.
2. **Why three-tier caching?** TIER 0 (pointer) is instant, TIER 1 (memory cache) is fast (<1ms), TIER 2 (database) is persistent. Each tier optimizes for different access patterns.
3. **Why monotonic progress?** Frontend progress bars must never go backwards. Pipeline-wide progress stages ensure each component reports within its allocated range.

---

## Module structure

```
processing/
├── processing_manager.py           # Main orchestrator (unified v3)
├── processing_strategies.py        # Strategy implementations
├── processing_handlers.py          # Analysis handlers
├── tasks.py                        # Task store & cancellation
├── progress_reporter.py            # Progress tracking
├── progress_utils.py               # Progress utilities
├── resource_estimator.py           # Resource estimation
├── smart_router.py                 # Smart routing logic
├── analysis_thread_coordinator.py  # Thread coordination
├── processing_manager_refactored.py # Refactored version
├── processing_manager_unified.py   # Unified version
└── __init__.py                     # Module exports
```

---

## Files in this module

### processing_manager.py — Main Orchestrator

> Unified orchestrator for all analysis types with automatic strategy selection, progress tracking, and result storage.

**Key classes:** `ProcessingManager`, `ProgressStage`, `PartialResultAggregator`, `IntermediateResultsCache`  
**Key enums:** `AnalysisType`

[→ Full documentation](#processing_managerpy)

---

### processing_strategies.py — Strategy Implementations

> Processing strategy implementations (Sequential, Parallel Chunking, Slice Streaming) with automatic selection logic.

**Key classes:** `ProcessingStrategy`, `ProcessingContext`, `StrategyFactory`, `HandlerRegistry`

[→ Full documentation](#processing_strategiespy)

---

### processing_handlers.py — Analysis Handlers

> Handler implementations for each analysis type (technical, SNR, astronomical, ML preparation, model training).

**Key functions:** `analyze_technical_impl()`, `analyze_snr_impl()`, `analyze_astronomical_impl()`, `analyze_ml_prep_impl()`

[→ Full documentation](#processing_handlerspy)

---

# processing_manager.py

> Unified orchestrator for all analysis types with automatic strategy selection.

[→ Source: `Backend/app/core/processing/processing_manager.py`](../../../Backend/app/core/processing/processing_manager.py)

---

## Overview

This file solves the **orchestration problem**: Different analysis types (technical indicators, SNR signals, astronomical features, ML preparation) have different performance characteristics and memory requirements. A one-size-fits-all approach either wastes resources (sequential for large datasets) or crashes (parallel for small datasets).

**What it does:**

- Auto-selects processing strategy based on dataset size and analysis type
- Builds execution context with all necessary parameters
- Delegates execution to selected strategy
- Tracks progress with monotonic guarantees (never goes backwards)
- Caches intermediate results (TIER 1: 30-minute TTL)
- Persists final results to database (TIER 2)
- Handles checkpointing and recovery for long-running tasks
- Aggregates slice results incrementally to prevent memory peaks

**Where it sits:**

- **Called by:** API endpoints (`analysis.py`), background workers
- **Calls:** `StrategyFactory`, `HandlerRegistry`, `IntermediateResultsCache`, `store_session_step_result()`

**Design decisions:**

1. **Why unified manager?** Previous versions had separate managers for each analysis type, leading to code duplication. Unified manager uses handler registry pattern.
2. **Why auto-selection?** Manual strategy selection is error-prone. Auto-selection uses proven heuristics (10K threshold for parallel, 100K for streaming).
3. **Why TIER 1 cache?** Multi-step pipelines (technical → SNR → astronomical) pass data between steps. Caching avoids redundant database round-trips.

---

## Quick reference

| Symbol | Type | Purpose |
|--------|------|---------|
| `ProcessingManager` | class | Main orchestrator for all analysis types |
| `ProgressStage` | enum | Pipeline-wide progress stages (0-100%) |
| `AnalysisType` | enum | Supported analysis types |
| `PartialResultAggregator` | class | Incremental result aggregation |
| `IntermediateResultsCache` | class | TIER 1 cache (30-minute TTL) |
| `CachedStepData` | class | Wrapper for cached data with expiration |

[→ Jump to source: Backend/app/core/processing/processing_manager.py](../../../Backend/app/core/processing/processing_manager.py)

---

## Enums

### ProgressStage

[→ Source: `Backend/app/core/processing/processing_manager.py` line 96](../../../Backend/app/core/processing/processing_manager.py#L96)

Pipeline-wide progress stages with allocated percentage ranges. Ensures monotonic progress across the entire analysis pipeline.

**Stages:**

| Stage | Range | Duration | Description |
|-------|-------|----------|-------------|
| API_ROUTING | 0-2% | ~50ms | Request validation & routing |
| DATA_LOADING | 2-5% | ~200ms | TIER cache lookup & decompression |
| STRATEGY_SETUP | 5-8% | ~100ms | Strategy selection & context building |
| CORE_PROCESSING | 8-98% | ~minutes | Main analysis (TechnicalIndicators, SNR, etc.) |
| SERIALIZATION | 98-99% | ~1s | Pickle encoding & compression |
| STORAGE | 99-100% | ~500ms | Database write & commit |

**Why these ranges?**

- **CORE_PROCESSING gets 90%** because it's the longest-running stage
- **SERIALIZATION and STORAGE get 2%** because they're fast but visible
- **API_ROUTING gets 2%** to show immediate feedback to user

**Methods:**

#### scale_progress(stage, local_progress) → float

[→ Source: `Backend/app/core/processing/processing_manager.py` line 118](../../../Backend/app/core/processing/processing_manager.py#L118)

Scale local progress (0-100) to global pipeline progress.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| stage | ProgressStage | Yes | Current pipeline stage |
| local_progress | float | Yes | Progress within the stage (0-100) |

**Returns:** float — Global progress percentage (0-100)

**The formula:**

```
global_progress = stage_start + (local_progress / 100) × (stage_end - stage_start)

where:
  stage_start = start percentage of stage
  stage_end   = end percentage of stage
```

**Example:**

```python
from app.core.processing.processing_manager import ProgressStage

# Halfway through core processing
global_pct = ProgressStage.scale_progress(ProgressStage.CORE_PROCESSING, 50)
print(global_pct)  # → 53.0 (8% + 50% of 90% = 53%)

# 75% through serialization
global_pct = ProgressStage.scale_progress(ProgressStage.SERIALIZATION, 75)
print(global_pct)  # → 98.75 (98% + 75% of 1% = 98.75%)
```

---

### AnalysisType

[→ Source: `Backend/app/core/processing/processing_manager.py` line 143](../../../Backend/app/core/processing/processing_manager.py#L143)

Supported analysis types.

**Values:**

```python
class AnalysisType(str, Enum):
    TECHNICAL = "technical"
    SNR = "snr"
    ASTRONOMICAL = "astronomical"
    ML_DATASET_PREPARATION = "ml_dataset_preparation"
    MODEL_BUILD = "model_build"
    MODEL_TRAINING = "model_training"
```

---

## Classes

### ProcessingManager

[→ Source: `Backend/app/core/processing/processing_manager.py` line 405](../../../Backend/app/core/processing/processing_manager.py#L405)

> Unified orchestrator for all analysis types.

**Inherits from:** None

**Used by:** API endpoints, background workers, analysis pipelines

**Why it exists:**

Before unified manager, each analysis type had its own orchestrator (TechnicalManager, SNRManager, AstroManager). This led to:
- Code duplication (same strategy selection logic repeated 5 times)
- Inconsistent progress tracking (each manager reported differently)
- Maintenance burden (bug fixes needed in 5 places)

Unified manager solves this with handler registry pattern: all analysis types use same code path, handlers registered at module load.

**What it does NOT do:**

- Does not implement analysis logic (delegates to handlers)
- Does not manage database connections (receives db session from caller)
- Does not handle authentication (caller's responsibility)

---

#### Constructor

```python
ProcessingManager(
    session_id: str,
    task_id: str,
    analysis_type: str,
    config: Any,
    task_store: Optional[TaskStore] = None,
    connection_manager: Optional[Any] = None,
    processing_config: Optional[ProcessingConfig] = None,
    step_name: Optional[str] = None,
    user_id: Optional[str] = "anonymous"
)
```

Initialize ProcessingManager.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| session_id | str | Yes | Session UUID |
| task_id | str | Yes | Task UUID for progress tracking |
| analysis_type | str | Yes | Analysis type ("technical", "snr", "astronomical", etc.) |
| config | Any | Yes | Analysis-specific config (TechnicalConfig, SNRConfig, etc.) |
| task_store | TaskStore | No | Task store for progress updates |
| connection_manager | Any | No | WebSocket connection manager |
| processing_config | ProcessingConfig | No | Processing configuration (thresholds, etc.) |
| step_name | str | No | Step name for database storage (defaults to "{analysis_type}_analysis") |
| user_id | str | No (default: "anonymous") | User ID for progress tracking |

**Code example:**

```python
from app.core.processing.processing_manager import ProcessingManager
from app.core.config import SNRConfig

pm = ProcessingManager(
    session_id="abc-123",
    task_id="task-456",
    analysis_type="snr",
    config=SNRConfig(
        timeframes=["M5", "M15", "H1"],
        min_touches=3,
        lookback_bars=100
    ),
    task_store=task_store,
    connection_manager=websocket_manager,
    user_id="user-789"
)

result = await pm.execute(df)
```

---

#### Methods

##### execute(df, **kwargs) → Dict[str, Any]

[→ Source: `Backend/app/core/processing/processing_manager.py` line 459](../../../Backend/app/core/processing/processing_manager.py#L459)

Main entry point: Execute analysis with automatic strategy selection.

**What it does:**

1. Auto-select strategy based on dataset size and analysis type
2. Send initial progress update (0%)
3. Execute with selected strategy
4. Ensure result completeness (merge original OHLCV columns if needed)
5. Cache result to TIER 1 (30-minute TTL)
6. Persist to database (TIER 2)
7. Send completion progress (100%)
8. Cleanup memory (delete original DataFrame, trigger GC)

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| df | pd.DataFrame | Yes | Input DataFrame (OHLCV data) |
| **kwargs | Dict | No | Additional parameters (dataset_name, split_name, etc.) |

**Returns:** Dict[str, Any] — Result dict with enriched data and metadata

**Raises:**
- `TaskCancelledException` — If task is cancelled
- `ValueError` — If handler not registered

**The execution flow:**

```
1. Auto-select strategy:
     if n_rows < 10K: Sequential
     elif n_rows < 100K: Parallel Chunking
     else: Slice Streaming

2. Send progress (0%): "Starting analysis..."

3. Execute:
     if ML Dataset Preparation:
         → Split data first, then process each split
     else:
         → Execute with selected strategy

4. Ensure completeness:
     → Merge original OHLCV columns if worker only returned features

5. Cache to TIER 1:
     → Store result with 30-minute TTL

6. Persist to database (TIER 2):
     → Send progress (98%): "Serializing..."
     → Send progress (99%): "Storing..."

7. Send completion (100%): "Complete"

8. Cleanup:
     → Delete original DataFrame
     → Trigger garbage collection
```

**Code example:**

```python
import pandas as pd
from app.core.processing.processing_manager import ProcessingManager
from app.core.config import TechnicalConfig

# Create manager
pm = ProcessingManager(
    session_id="abc-123",
    task_id="task-456",
    analysis_type="technical",
    config=TechnicalConfig(
        indicators=["RSI", "MACD", "BB"],
        rsi_period=14,
        macd_fast=12,
        macd_slow=26
    ),
    task_store=task_store
)

# Load data
df = pd.DataFrame({
    'Time': pd.date_range('2024-01-01', periods=10000, freq='5min'),
    'Open': np.random.rand(10000) * 100,
    'High': np.random.rand(10000) * 100,
    'Low': np.random.rand(10000) * 100,
    'Close': np.random.rand(10000) * 100,
    'Volume': np.random.randint(1000, 10000, 10000)
})

# Execute
result = await pm.execute(df)

# Result structure
print(result.keys())
# → ['result_df', 'metadata', 'signals', 'zones']

print(result['result_df'].columns.tolist())
# → ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'BB_upper', 'BB_lower']

print(result['metadata'])
# → {
#     'strategy': 'Parallel',
#     'n_rows': 10000,
#     'n_chunks': 4,
#     'execution_time_ms': 1234,
#     'indicators_calculated': ['RSI', 'MACD', 'BB']
# }
```

**Edge cases:**

- **Empty DataFrame:** Returns immediately with empty result
- **Task cancelled:** Raises `TaskCancelledException`, cleans up memory
- **Handler not registered:** Raises `ValueError` with helpful message
- **Memory pressure:** Triggers aggressive cleanup (clears all caches, forces GC)

---

### IntermediateResultsCache

[→ Source: `Backend/app/core/processing/processing_manager.py` line 267](../../../Backend/app/core/processing/processing_manager.py#L267)

> TIER 1 Cache: Intermediate analysis results between steps.

**Inherits from:** None (class with static methods)

**Why it exists:**

Multi-step pipelines (technical → SNR → astronomical) pass data between steps. Without caching:
- Each step fetches from database (200ms latency per step)
- 3 steps = 600ms wasted on database round-trips
- Database load increases unnecessarily

With TIER 1 cache:
- First step stores result in memory
- Next step retrieves from cache (<1ms)
- 3 steps = 2ms total (300× faster)

**Features:**

- 30-minute TTL (auto-expire to prevent stale data)
- Per-task scope: `{(task_id, step_name): CachedStepData}`
- Thread-safe (Python dict is atomic for basic operations)
- Falls back to database if cache miss

---

#### Methods

##### store(task_id, step_name, data, ttl_seconds=1800) → None

[→ Source: `Backend/app/core/processing/processing_manager.py` line 289](../../../Backend/app/core/processing/processing_manager.py#L289)

Cache intermediate result for next step.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| task_id | str | Yes | Task UUID |
| step_name | str | Yes | Step name (e.g., "technical_analysis") |
| data | Any | Yes | Data to cache (typically DataFrame) |
| ttl_seconds | int | No (default: 1800) | Time-to-live in seconds (30 minutes) |

**Code example:**

```python
from app.core.processing.processing_manager import IntermediateResultsCache

# Store technical analysis result
IntermediateResultsCache.store(
    task_id="task-456",
    step_name="technical_analysis",
    data=enriched_df,
    ttl_seconds=1800
)
```

---

##### retrieve(task_id, step_name) → Optional[Any]

[→ Source: `Backend/app/core/processing/processing_manager.py` line 299](../../../Backend/app/core/processing/processing_manager.py#L299)

Retrieve cached result, return None if expired or missing.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| task_id | str | Yes | Task UUID |
| step_name | str | Yes | Step name |

**Returns:** Any — Cached data, or None if expired/missing

**Code example:**

```python
from app.core.processing.processing_manager import IntermediateResultsCache

# Try to retrieve from cache
cached_df = IntermediateResultsCache.retrieve(
    task_id="task-456",
    step_name="technical_analysis"
)

if cached_df is not None:
    print("Cache hit! Using cached data")
else:
    print("Cache miss, loading from database")
    cached_df = await load_from_database()
```

---

## Summary

The `processing_manager.py` module is the **orchestration backbone** of the system. It ensures that:

1. Analysis executes with optimal strategy (Sequential/Parallel/Streaming)
2. Progress updates are monotonic (never go backwards)
3. Intermediate results are cached (TIER 1: 30-minute TTL)
4. Final results are persisted (TIER 2: database)
5. Memory is managed efficiently (incremental aggregation, cleanup)

**Key takeaways:**

1. **Always use ProcessingManager** for analysis execution (don't call handlers directly)
2. **Trust auto-selection** (proven heuristics for strategy selection)
3. **Monitor progress stages** (use ProgressStage.scale_progress for custom components)
4. **Leverage TIER 1 cache** (store intermediate results for multi-step pipelines)

---

## Module Status

**Documented:** 1/13 files
- ✅ processing_manager.py

**Remaining:** 12 files (strategies, handlers, tasks, progress, resource estimation, etc.)

**Next file:** `processing_strategies.py` or `processing_handlers.py`
