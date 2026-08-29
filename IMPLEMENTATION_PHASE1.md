# Phase 1 Implementation: Meta-Learner Signal Classification

**Status**: Starting  
**Objective**: Enable the meta learner to consume data, enrich it, generate signals, and classify signal quality  
**Target**: End-to-end flow with synthetic data, async/logging throughout

---

## What Phase 1 Builds

### Input
- Raw or synthetic OHLCV candles
- Market context (symbol, timeframe, session metadata)

### Processing
1. **Data Enrichment**
   - Compute technical indicators (RSI, MACD, Bollinger Bands, etc.)
   - Detect support/resistance levels
   - Calculate volatility and momentum features

2. **Signal Production**
   - Multi-timeframe RSI analysis
   - Divergence detection
   - Exhaustion and structural patterns
   - Signal bundle assembly

3. **Signal Classification**
   - Meta learner evaluates signal quality
   - Assigns reward scores
   - Labels confidence and category
   - Classifies actionability

### Output
- Classified signal with:
  - Type (trend, reversal, exhaustion, etc.)
  - Confidence score
  - Reward estimation
  - Forward move probability
  - Enriched feature context

---

## Async & Logging Architecture

All Phase 1 workflows must:

1. **Use async/await**
   - `async def` for I/O-bound operations (DB, file, API)
   - FastAPI and APScheduler for orchestration
   - `asyncio.gather()` for parallel tasks

2. **Log everything**
   - Use `logging.getLogger(__name__)` in all modules
   - Structured logging for signals (JSON-serializable events)
   - Observation points:
     - Data loading (source, rows, compression)
     - Feature calculation (indicator values, timing)
     - Signal generation (triggers, patterns, bundles)
     - Classification (scores, decisions, model predictions)
     - Errors and warnings (with context)

3. **No synchronous blocking**
   - No `requests.get()` without timeout
   - No heavy computation without progress logging
   - No direct process spawning (use ProcessPoolExecutor with async wrapper)

---

## File Organization

New/Modified files for Phase 1:

```
backend/app/
├── core/
│   ├── ml/
│   │   ├── signal_pipeline.py          [NEW] End-to-end async orchestration
│   │   ├── meta_learner.py             [EXISTING] Enhance logging
│   │   └── signal_meta_learner.py      [EXISTING] Verify async compatibility
│   └── data/
│       ├── synthetic_data_generator.py [NEW] Generate test candles
│       └── session_data_loader.py      [EXISTING] Logging enhancements
│
├── agent/
│   └── loop.py                         [MODIFY] Full async, structured logging
│
├── api/
│   └── routes/
│       ├── signals.py                  [NEW] Signal classification endpoint
│       └── datasets.py                 [NEW] Dataset management endpoints
│
└── main.py                             [MODIFY] Async startup, lifecycle logging
```

---

## Core Components

### 1. Synthetic Data Generator
File: `backend/app/core/data/synthetic_data_generator.py`

**Purpose**: Generate realistic OHLCV candles for testing

**Features**:
- Configurable price ranges, volatility, trend
- Multiple symbols
- Different timeframes
- Repeatable (seed-based)

**Example**:
```python
from app.core.data.synthetic_data_generator import SyntheticDataGenerator

gen = SyntheticDataGenerator(seed=42)
candles = gen.generate_session(
    symbol="AAPL",
    num_candles=500,
    start_price=150.0,
    volatility=0.02,
    trend=0.001
)
# candles = [{"time": 1234567890, "open": 150.1, "high": 150.5, ...}, ...]
```

### 2. Signal Pipeline (Orchestrator)
File: `backend/app/core/ml/signal_pipeline.py`

**Purpose**: Async coordination of data → enrichment → signals → classification

**Interface**:
```python
from app.core.ml.signal_pipeline import SignalPipeline

pipeline = SignalPipeline()
result = await pipeline.process_session(
    candles=[...],
    symbol="AAPL",
    logger=logger
)
# result = {
#   "signals": [...],
#   "enriched_data": {...},
#   "classifications": {...},
#   "metadata": {...}
# }
```

### 3. API Endpoint: /signal/classify
File: `backend/app/api/routes/signals.py`

**Purpose**: HTTP interface for signal classification

**POST /signal/classify**
```json
{
  "candles": [...],
  "symbol": "AAPL",
  "timeframe": "1H"
}
```

Response:
```json
{
  "signals": [
    {
      "id": "sig_1234",
      "type": "divergence_bullish",
      "confidence": 0.78,
      "reward_score": 2.3,
      "entry_point": 150.5,
      "forward_move_prob": 0.62,
      "enriched_features": {...}
    }
  ],
  "processing_ms": 234
}
```

### 4. Enhanced Agent Loop
File: `backend/app/agent/loop.py` (modified)

**Changes**:
- Full async/await
- Structured logging at every step
- Use SignalPipeline for classification
- Store results in DB with enriched metadata
- Report progress to WebSocket (if available)

---

## Logging Strategy

### Log Levels & Categories

**INFO** (default):
- Pipeline start/completion
- Data loading summary
- Signal generation (# of signals, types)
- Classification results (high-level)

**DEBUG** (when enabled):
- Feature values for each candle
- Intermediate calculations (RSI, divergence triggers)
- Cache hits/misses
- Processing timing

**WARNING**:
- Data quality issues (missing candles, outliers)
- Classification confidence below threshold
- Feature calculation failures (recoverable)

**ERROR**:
- DB connection failures
- Feature calculation fatal errors
- Unexpected data format

### Structured Logging (JSON)

For key events, emit JSON logs for downstream analysis:

```python
logger.info(json.dumps({
    "event": "signal_generated",
    "signal_id": sig_id,
    "symbol": "AAPL",
    "signal_type": "divergence_bullish",
    "confidence": 0.78,
    "reward_score": 2.3,
    "timestamp": datetime.utcnow().isoformat(),
    "processing_ms": 234
}))
```

---

## Testing & Validation

### Unit Tests
- Synthetic data generator (distributions, parameters)
- Feature calculations (technical indicators vs. reference)
- Signal detection (known patterns produce expected signals)
- Classification logic (reward scoring, confidence)

### Integration Tests
- End-to-end pipeline (synthetic data → classified signals)
- API endpoint (HTTP request → signal response)
- Database persistence (signal outcome storage)
- Logging verification (all expected logs present)

### Run Tests
```bash
pytest backend/tests/ -v -k "phase1"
```

---

## Success Criteria

By end of Phase 1:

- ✅ Synthetic data generator works
- ✅ Signal pipeline orchestrates all steps async
- ✅ Meta learner classifies signals with scores
- ✅ API endpoint responds to signal classification requests
- ✅ Agent loop runs continuously with full logging
- ✅ All operations are async (no blocking I/O)
- ✅ Database stores classified signals with metadata
- ✅ Tests cover critical paths
- ✅ README updated to reflect architecture

---

## Next: Phase 2 (Q-Learner)

Once Phase 1 is validated:
- Q-learner learns which signals to act on
- Policy layer converts signals → trade decisions
- Backtesting framework validates P&L
- (Alpaca integration remains deferred until P&L is proven)

