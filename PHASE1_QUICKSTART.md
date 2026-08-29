# Phase 1: Quick Start Guide

This guide walks you through running the Phase 1 implementation: meta-learner signal classification with synthetic data.

---

## Prerequisites

- Python 3.9+
- Backend dependencies installed

```bash
cd backend
pip install -r requirements.txt
pip install pytest pytest-asyncio  # For testing
```

---

## Step 1: Generate Synthetic Data

Test the synthetic data generator standalone:

```bash
cd backend
python -c "
from app.core.data.synthetic_data_generator import SyntheticDataGenerator

gen = SyntheticDataGenerator(seed=42)
candles = gen.generate_session('AAPL', num_candles=100, start_price=150.0)
print(f'Generated {len(candles)} candles')
print(f'First candle: {candles[0]}')
print(f'Last candle: {candles[-1]}')
"
```

**Expected output**:
```
SyntheticDataGenerator initialized with seed=42
Generating 100 candles for AAPL (start_price=150.0, volatility=0.02, trend=0.001)
Generated 100 candles for AAPL. Price range: 149.XX - 150.XX
Generated 100 candles
First candle: Candle(time=..., open=150.0, high=150.XX, low=149.XX, close=150.XX, volume=...)
Last candle: Candle(time=..., open=..., high=..., low=..., close=..., volume=...)
```

---

## Step 2: Run Signal Pipeline (Async)

Test the async signal pipeline:

```bash
cd backend
python -c "
import asyncio
from app.core.data.synthetic_data_generator import SyntheticDataGenerator
from app.core.ml.signal_pipeline import SignalPipeline

async def main():
    # Generate data
    gen = SyntheticDataGenerator(seed=42)
    candles = gen.generate_session('AAPL', num_candles=100)
    candles_dict = gen.to_dict_list(candles)
    
    # Process through pipeline
    pipeline = SignalPipeline()
    result = await pipeline.process_session(
        candles=candles_dict,
        symbol='AAPL',
        timeframe='1H'
    )
    
    # Display results
    print(f'\nPipeline Results:')
    print(f'  Signals generated: {len(result.signals)}')
    print(f'  Processing time: {result.processing_ms:.1f} ms')
    print(f'  Metadata: {result.metadata}')
    
    if result.signals:
        sig = result.signals[0]
        print(f'\n  First signal:')
        print(f'    ID: {sig.signal_id}')
        print(f'    Type: {sig.signal_type}')
        print(f'    Confidence: {sig.confidence:.2f}')
        print(f'    Reward Score: {sig.reward_score:.2f}')

asyncio.run(main())
"
```

**Expected output**:
```
SyntheticDataGenerator initialized with seed=42
Generating 100 candles for AAPL (...)
Pipeline START: AAPL 1H with 100 candles
Step 1: Validation OK (100 candles)
Step 2: Enrichment OK (5 feature groups)
Step 3: Generated X candidate signals
Step 4: Classified X signals (avg confidence: 0.XX)
Pipeline Results:
  Signals generated: X
  Processing time: XX.X ms
  Metadata: {...}
```

---

## Step 3: Run Full Test Suite

```bash
cd backend
pytest tests/test_phase1_implementation.py -v -s
```

**Expected output**:
```
tests/test_phase1_implementation.py::TestSyntheticDataGenerator::test_generator_initialization PASSED
tests/test_phase1_implementation.py::TestSyntheticDataGenerator::test_generate_session_basic PASSED
...
tests/test_phase1_implementation.py::TestIntegration::test_full_pipeline_e2e PASSED
========================== 20 passed in 2.34s ==========================
```

---

## Step 4: Test Async Parallelism

Test running multiple symbol processing in parallel:

```bash
cd backend
python -c "
import asyncio
from app.core.data.synthetic_data_generator import SyntheticDataGenerator
from app.core.ml.signal_pipeline import SignalPipeline

async def main():
    gen = SyntheticDataGenerator(seed=42)
    pipeline = SignalPipeline()
    
    # Process multiple symbols in parallel
    symbols = ['AAPL', 'GOOGL', 'TSLA', 'MSFT']
    tasks = []
    
    for symbol in symbols:
        candles = gen.generate_session(symbol, num_candles=50)
        candles_dict = gen.to_dict_list(candles)
        task = pipeline.process_session(candles_dict, symbol, '1H')
        tasks.append(task)
    
    print(f'Processing {len(tasks)} symbols in parallel...')
    results = await asyncio.gather(*tasks)
    
    print(f'\nResults:')
    for result in results:
        print(f\"  {result.metadata['symbol']}: {len(result.signals)} signals in {result.processing_ms:.1f}ms\")

asyncio.run(main())
"
```

**Expected output**:
```
Generating ... candles for AAPL (...)
Generating ... candles for GOOGL (...)
Generating ... candles for TSLA (...)
Generating ... candles for MSFT (...)
Processing 4 symbols in parallel...
Pipeline START: AAPL 1H with 50 candles
Pipeline START: GOOGL 1H with 50 candles
...
Results:
  AAPL: X signals in XX.X ms
  GOOGL: X signals in XX.X ms
  TSLA: X signals in XX.X ms
  MSFT: X signals in XX.X ms
```

---

## Step 5: Inject Test Patterns

Test pattern injection for realistic signal scenarios:

```bash
cd backend
python -c "
import asyncio
from app.core.data.synthetic_data_generator import SyntheticDataGenerator
from app.core.ml.signal_pipeline import SignalPipeline

async def main():
    gen = SyntheticDataGenerator(seed=42)
    pipeline = SignalPipeline()
    
    # Generate base candles
    candles = gen.generate_session('AAPL', num_candles=100)
    
    # Inject bullish divergence pattern
    patterned = gen.add_pattern(candles, 'bullish_divergence', intensity=1.5)
    patterned_dict = gen.to_dict_list(patterned)
    
    # Process through pipeline
    result = await pipeline.process_session(patterned_dict, 'AAPL')
    
    print(f'Pipeline result with bullish divergence pattern:')
    print(f'  Total signals: {len(result.signals)}')
    print(f'  Processing: {result.processing_ms:.1f}ms')

asyncio.run(main())
"
```

---

## Step 6: View Logging in Detail

Run a test with full DEBUG logging:

```bash
cd backend
python -c "
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

import asyncio
from app.core.data.synthetic_data_generator import SyntheticDataGenerator
from app.core.ml.signal_pipeline import SignalPipeline

async def main():
    gen = SyntheticDataGenerator(seed=42)
    pipeline = SignalPipeline()
    
    candles = gen.generate_session('AAPL', num_candles=30)
    candles_dict = gen.to_dict_list(candles)
    
    result = await pipeline.process_session(candles_dict, 'AAPL')
    print(f'\nCompleted: {len(result.signals)} signals')

asyncio.run(main())
"
```

**Expected output**:
```
2026-08-29 12:34:56,789 - app.core.data.synthetic_data_generator - INFO - SyntheticDataGenerator initialized with seed=42
2026-08-29 12:34:56,790 - app.core.data.synthetic_data_generator - INFO - Generating 30 candles for AAPL ...
2026-08-29 12:34:56,792 - app.core.ml.signal_pipeline - INFO - Pipeline START: AAPL 1H with 30 candles
2026-08-29 12:34:56,792 - app.core.ml.signal_pipeline - DEBUG - Step 1: Validating 30 candles
2026-08-29 12:34:56,792 - app.core.ml.signal_pipeline - INFO - Step 1: Validation OK (30 candles)
2026-08-29 12:34:56,793 - app.core.ml.signal_pipeline - DEBUG - Step 2: Enriching data
2026-08-29 12:34:56,800 - app.core.ml.signal_pipeline - INFO - Step 2: Enrichment OK (5 feature groups)
2026-08-29 12:34:56,800 - app.core.ml.signal_pipeline - DEBUG - Step 3: Generating signals
2026-08-29 12:34:56,801 - app.core.ml.signal_pipeline - DEBUG - Step 4: Classifying X signals
2026-08-29 12:34:56,802 - app.core.ml.signal_pipeline - INFO - Pipeline complete: ...
Completed: X signals
```

---

## Step 7: Start Backend Server (Optional)

Once validated, start the FastAPI server:

```bash
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

Server runs at `http://localhost:8000`. API endpoints (with Alpaca deferred):
- `GET /health` — Health check
- `GET /status` — System status
- `POST /signal/bundle` — Signal generation (legacy)

---

## Testing Checklist

- [ ] **Synthetic Data**: Candles generate correctly, deterministic with seed
- [ ] **Pipeline Async**: Runs without blocking, can process in parallel
- [ ] **Logging**: INFO logs at each stage, DEBUG logs intermediate steps
- [ ] **Signals**: Generated and classified with confidence/reward scores
- [ ] **Full Suite**: All 20+ tests pass

---

## Next Steps

Once all steps pass:

1. **Connect Real Indicators** — Replace placeholder methods with actual implementations
   - `app.core.analysis.technical_indicators.calculate_rsi()`
   - `app.core.market.mtf_rsi.calculate_mtf_rsi()`
   - `app.core.ml.meta_learner` (real reward scoring)

2. **Database Integration** — Store signals in PostgreSQL
   - Modify pipeline to call `db.session.add(SignalOutcome(...))`
   - Track enriched_features, classifications, and outcomes

3. **API Routes** — Expose pipeline via `/signal/classify` endpoint
   - FastAPI route that calls pipeline asynchronously
   - Return classified signals with metadata

4. **Agent Loop** — Integrate into `app.agent.loop`
   - Use pipeline instead of direct signal generation
   - Full structured logging for audit trail

---

## Troubleshooting

**Import error: `ModuleNotFoundError: No module named 'app'`**

Run from the `backend/` directory:
```bash
cd backend
python -c "from app.core.data.synthetic_data_generator import ..."
```

**Test failures: `FAILED ... asyncio.TimeoutError`**

This is normal for the first run. Tests may need pytest-asyncio configuration:
```bash
pip install pytest-asyncio
pytest tests/test_phase1_implementation.py --asyncio-mode=auto -v
```

**Logging not visible**

Enable DEBUG level:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Success Criteria

Phase 1 is complete when:

- ✅ Synthetic data generator produces valid OHLCV candles
- ✅ Signal pipeline processes data asynchronously
- ✅ All 20+ tests pass
- ✅ Logging captures every stage (validation, enrichment, signals, classification)
- ✅ Multiple symbols can be processed in parallel
- ✅ Patterns (divergences, gaps) can be injected and detected

---

## Questions?

Refer to:
- [IMPLEMENTATION_PHASE1.md](../IMPLEMENTATION_PHASE1.md) — Detailed design
- [backend/README.md](README.md) — Architecture overview
- [app/core/ml/signal_pipeline.py](app/core/ml/signal_pipeline.py) — Pipeline implementation
- [app/core/data/synthetic_data_generator.py](app/core/data/synthetic_data_generator.py) — Data generation
