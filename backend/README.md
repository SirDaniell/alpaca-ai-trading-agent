# Signal Intelligence & Options RL Execution Backend

This backend is an autonomous options trading agent and reinforcement learning engine built for the **Alpaca AI Trading Agents Hackathon**.

It features a **Two-Tier Reinforcement Learning Pipeline**:
1. **Tier 1 — Signal Meta-Learner**: Higher-timeframe (HTF) market bias, multi-horizon label scoring, and 6 decoupled auxiliary regression heads.
2. **Phase 1b — Meta-Learner Inference Pass**: Sequential, causal pass generating 10,000 high-conviction transition memories.
3. **Tier 2 — Dual-Branch Ensemble Options Q-Executor**: Lower-timeframe (LTF) trade execution with 1:1 functional parity across PyTorch (`q_executor.py`) and Keras (`keras_q_executor.py`).
4. **Hard Action Masking & Dynamic SNR**: Strict zero-lookahead support/resistance tracking with buyer/seller volume delta fallback.

---

## Benchmark Evaluation Scripts

Run walk-forward out-of-sample options benchmarks across 40,000 historical bars:

```bash
# PyTorch 40k-bar Evaluation
PYTHONPATH=. python3 scripts/evaluate_option_expiries.py --symbols SPY --framework pytorch --limit 40000

# Keras 40k-bar Evaluation
PYTHONPATH=. python3 scripts/evaluate_option_expiries.py --symbols SPY --framework keras --limit 40000
```

## Execution model

The backend is designed around async-first execution and logging-centered observability.

- Services and workflows are structured for asynchronous operation
- Data loading and enrichment are separated from final signal evaluation
- Signal and learner events are logged as structured artifacts
- The system is meant to observe, classify, and learn from outcomes before moving into trade-policy logic

Logging is part of the product surface: the backend is built to produce explainable, traceable signal events and learning records rather than only a final trade decision.

## Deferred integrations

The following integrations are explicitly deferred and are not the current active path:

- Alpaca live or paper trading integration
- MCP orchestration and external agent workflows
- Live order execution and portfolio automation
- Real-time broker routing and execution logic

These are future integration points, not the current implementation focus. The active architecture is signal intelligence first, meta-learning second, and trade execution later.

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

For the current stage, live broker credentials are not the active implementation path. Local configuration is mainly for application settings, dataset assets, and validation workflows.

```bash
cp .env.example .env
```

If you later add Alpaca integration, edit `.env`:
```env
DATABASE_URL=postgresql://lablab_user:lablab_pass@localhost:5432/lablab_trading
# Alpaca credentials (deferred integration - optional for now)
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### 3. Start PostgreSQL

**Option A: Docker (Recommended)**

```bash
docker-compose up postgres
```

**Option B: Local PostgreSQL Installation**

Ensure PostgreSQL 12+ is running:
```bash
sudo service postgresql start
createuser lablab_user -P  # Enter password: lablab_pass
createdb lablab_trading -O lablab_user
```

### 4. Initialize Database

```bash
python -c "from app.db.connection import init_db; init_db()"
```

### 5. Start the backend

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The server will start at `http://localhost:8000`.

Check that it's running:
```bash
curl http://localhost:8000/health
```

### 6. Run data or signal workflows

```bash
python -m app.agent.loop
```

These commands are intended for validating the signal pipeline and learning loop, not for live brokerage execution.

## Testing and development

Typical development flows include:

- dataset generation or loading
- feature enrichment and indicator validation
- signal bundle construction
- meta-learner reward and classification checks
- logging and analysis of signal outcomes

Run the backend tests with:

```bash
pytest tests/ -v
```

## Project Structure

The backend is organized around the signal-intelligence stack:

```
backend/
├── app/
│   ├── core/
│   │   ├── analysis/
│   │   │   ├── technical_indicators.py       # Feature generation
│   │   │   ├── support_resistance.py         # SNR zones
│   │   │   └── trading/
│   │   │       ├── signal_generator.py       # Signal detection
│   │   ├── market/
│   │   │   ├── mtf_rsi.py                    # Multi-timeframe RSI
│   │   │   ├── divergence_scale.py           # Divergence normalization
│   │   │   └── signal_events.py              # Signal bundle
│   │   ├── ml/
│   │   │   ├── meta_learner.py               # Reward calculation
│   │   │   ├── signal_meta_learner.py        # Signal classification
│   │   │   ├── ti_meta_features.py           # Meta features
│   │   │   ├── signal_outcome_labeling.py    # Outcome resolution
│   │   │   └── ml_dataset_preparation.py     # Feature engineering
│   │   ├── data/
│   │   │   ├── session_data_loader.py        # Data loading
│   │   │   ├── session_dataset_registry.py   # Dataset caching & management
│   │   │   └── serializers.py                # Compression & serialization
│   │   └── processing/
│   │       ├── processing_manager.py         # Task orchestration
│   │       └── processing_strategies.py      # Parallel execution
│   ├── db/
│   │   ├── connection.py                     # SQLAlchemy & session mgmt
│   │   └── models.py                         # SignalOutcome, LearnerCheckpoint
│   ├── agent/
│   │   └── loop.py                           # Signal generation orchestration
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/                           # API endpoints (datasets, models)
│   ├── utils/
│   │   └── alpaca_client.py                  # Alpaca API wrapper (deferred)
│   └── main.py                               # FastAPI app entry point
├── tests/
│   └── test_signal_pipeline.py               # Unit & integration tests
├── .env.example                              # Environment template
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
└── README.md
```

## Key modules

- **app/core/analysis/** — Technical feature generation, indicator calculation, data cleaning
- **app/core/market/** — Market event logic, MTF RSI, divergence detection, signal bundling
- **app/core/ml/** — Meta learner, reward scoring, feature preparation, classification pipelines
- **app/core/data/** — Dataset loading, registry, serialization, session handling
- **app/core/processing/** — Task orchestration, parallel execution, progress tracking
- **app/agent/** — Agent loop, continuous signal generation, logging
- **app/db/** — Database models and connection management
- **app/main.py** — FastAPI entry point for HTTP API and lifecycle management

## Important roadmap note

The implementation path is deliberately layered:

1. Build the signal-intelligence and meta-learning stack
2. Validate it on synthetic and curated data
3. Log and classify signal outcomes
4. Add a future Q-learner trade policy layer
5. Only then consider live execution integration

Alpaca and MCP are not the current active path. They are deferred capabilities intended for later stages, once the signal processing and classification stack is validated.

### "Signal generation failed"

Check logs:

```bash
# If running via Docker
docker-compose logs backend | tail -50

# If running locally
python -m app.agent.loop  # Run manually to see full error
```

## Next Steps

1. **Alpaca Paper Trading**: Confirm you have a paper trading account at https://app.alpaca.markets/paper
2. **Market Data**: Alpaca API is free for paper trading; no additional data subscription needed
3. **Signal Tuning**: Adjust `MARKET_SYMBOLS` and `AGENT_LOOP_INTERVAL_SECONDS` in `.env`
4. **Options Integration**: See `docs/strategy.md` for options strategy design
5. **Live Trading**: Switch to live API by using production Alpaca credentials (requires approval)

## Support

- Alpaca API Docs: https://docs.alpaca.markets
- FastAPI Docs: http://localhost:8000/docs (when running)
- Signal Architecture: See `docs/agent_handoff_signal_intelligence.md` for detailed logic flow

---

**Status**: Backend signal inference operational. Ready for options strategy integration.
