# Agent Handoff: Reusable Signal Intelligence from Existing Trading Repo

## Objective

This repo is the actual competition project workspace. The purpose of this handoff is to tell the next agent what to copy, what to ignore, and what to investigate from the original trading codebase before building the Alpaca options trading agent.

The key principle is simple:

> Do not copy the product UI or dashboard shell.
> Copy the signal intelligence, the label design, and the meta-learning logic that actually drives market decisions.

---

## 1. What to borrow from the existing repo

The valuable assets are not the frontend chart app. The valuable assets are the underlying intelligence stack.

### 1.1 Core signal / feature stack
These are the highest-value files to examine and selectively copy:

- /media/daniel-joseph/Linux_Data/Dan_backup/Projects/JavaScript/fin-dash-buddy-sos/Backend/app/core/analysis/technical_indicators.py
- /media/daniel-joseph/Linux_Data/Dan_backup/Projects/JavaScript/fin-dash-buddy-sos/Backend/app/core/analysis/trading/signal_generator.py
- /media/daniel-joseph/Linux_Data/Dan_backup/Projects/JavaScript/fin-dash-buddy-sos/Backend/app/core/analysis/trading/signal_generator_optimized.py
- /media/daniel-joseph/Linux_Data/Dan_backup/Projects/JavaScript/fin-dash-buddy-sos/Backend/app/core/ml/ml_dataset_preparation.py
- /media/daniel-joseph/Linux_Data/Dan_backup/Projects/JavaScript/fin-dash-buddy-sos/Backend/app/core/ml/ti_meta_features.py
- /media/daniel-joseph/Linux_Data/Dan_backup/Projects/JavaScript/fin-dash-buddy-sos/Backend/app/core/ml/signal_meta_learner.py

These files define the real system behavior:

- technical indicator features,
- SNR level extraction,
- event labeling (bounce vs breakout),
- feature bundles for meta-learning,
- forward target enrichment,
- and the signal-quality model loop.

### 1.2 MTF RSI and divergence layer
This is the most important context signal for regime detection:

- /media/daniel-joseph/Linux_Data/Dan_backup/Projects/JavaScript/fin-dash-buddy-sos/Frontend/src/lib/technical/mtf-rsi.ts
- /media/daniel-joseph/Linux_Data/Dan_backup/Projects/JavaScript/fin-dash-buddy-sos/Frontend/src/components/StockChart.tsx

The logic here is useful for understanding how the app interprets price regimes across timeframes.

Do not copy the chart shell; copy the logic pattern:

- compute RSI per timeframe,
- smooth and compare signals,
- detect crossovers and divergence,
- use that as contextual confirmation for market regime.

### 1.3 PostgreSQL persistence and meta-learner state
These are important if the competition agent needs to persist signal outcomes or checkpoint state:

- /media/daniel-joseph/Linux_Data/Dan_backup/Projects/JavaScript/fin-dash-buddy-sos/Backend/app/database/models.py
- /media/daniel-joseph/Linux_Data/Dan_backup/Projects/JavaScript/fin-dash-buddy-sos/Backend/app/database/connection.py
- /media/daniel-joseph/Linux_Data/Dan_backup/Projects/JavaScript/fin-dash-buddy-sos/Backend/app/database/repositories/learner_checkpoint_repository.py
- /media/daniel-joseph/Linux_Data/Dan_backup/Projects/JavaScript/fin-dash-buddy-sos/Backend/app/database/repositories/signal_outcome_repository.py
- /media/daniel-joseph/Linux_Data/Dan_backup/Projects/JavaScript/fin-dash-buddy-sos/Backend/migrations/versions/phase29_signal_outcomes.py
- /media/daniel-joseph/Linux_Data/Dan_backup/Projects/JavaScript/fin-dash-buddy-sos/Backend/migrations/versions/phase30_learner_checkpoints.py

These files give the pattern for persisting signal outcomes and model checkpoint metadata.

---

## 2. What to ignore completely

Ignore these unless a specific missing feature is discovered:

- Full dashboard shell and app orchestration
- StockChart chart rendering and canvas-heavy UI code
- WebSocket-driven live chart state management
- “analysis session” product screens
- non-core product metadata and persistence for the full app

The objective is to build a cleaner competition project, not to recreate the old product UI.

---

## 3. Core concepts to preserve from the original repo

### 3.1 Feature generation is historical-only and safe for input use
The technical indicators logic is intentionally designed around historical features, not lookahead leakage.

This is the key design rule:

- generate features from past bars only,
- keep target generation separate,
- never use future bars in the feature set.

### 3.2 SNR zones are not just visualization; they are structural context
The signal generator turns SNR support/resistance levels into actionable event labels.

The labels are meaningful:

- bounce_support
- bounce_resistance
- breakout_support
- breakout_resistance

This is a powerful event abstraction for options strategies.

### 3.3 Multi-target labels are more useful than a single return target
The target enrichment logic in `ml_dataset_preparation.py` is richer than a single “next close” prediction.

It adds:

- forward return targets,
- next-bar oscillator targets,
- next-zone liquidity targets,
- SNR sequence targets,
- reversal / continuation labels,
- multi-output signals for regime learning.

This is more aligned with an options trading decision engine than a toy regression model.

### 3.4 Meta-learning should evaluate whether a signal was actually valuable
The meta-learner is not a UI layer; it is the scoring engine that answers:

- did this signal structure actually pay off?
- what signal features predicted the next favorable move?
- what should the model trust more in future sessions?

This is exactly the sort of logic a competition bot can use to adapt and improve over time.

---

## 4. What to copy into this repo

### Required for a minimal but serious implementation

Create a lean core package in the new repo with roughly these modules:

- `signal_pipeline/features/technical_indicators.py`
- `signal_pipeline/features/mtf_rsi.py`
- `signal_pipeline/snr/signal_generator.py`
- `signal_pipeline/targets/ml_dataset_preparation.py`
- `signal_pipeline/meta/ti_meta_features.py`
- `signal_pipeline/meta/signal_meta_learner.py`
- `storage/postgres/models.py` (or equivalent minimal SQLAlchemy models)
- `storage/postgres/repositories/learner_checkpoint_repository.py`
- `storage/postgres/repositories/signal_outcome_repository.py`

These modules should be trimmed to the minimal version required by the competition build.

### Optional if you need to keep a simple API layer

- `api/routes/strategy_signal.py`
- `api/routes/meta_learning.py`

Keep these intentionally small.

---

## 5. Concrete file mapping from old repo to new repo

Use this mapping as the working checklist:

Old repo file -> New repo usage

- technical_indicators.py -> copy the feature-generation logic to the new repo core pipeline
- signal_generator.py -> copy the SNR + event-labeling logic
- ml_dataset_preparation.py -> copy target enrichment logic
- ti_meta_features.py -> copy curated meta learning feature bundle
- signal_meta_learner.py -> copy the meta learner logic and decision scoring
- mtf-rsi.ts -> port the regime logic into Python or a lightweight TS service
- signal_outcome_repository.py -> copy persistence pattern for outcomes
- learner_checkpoint_repository.py -> copy checkpoint persistence pattern
- models.py -> copy DB schema for outcome/checkpoint tracking

---

## 6. Research plan for the agent

Before building, the agent should do these investigations in the original codebase:

1. Open the indicator generation file and identify the minimum required historical features.
2. Open the SNR signal generator and understand the exact event conditions for bounce vs breakout.
3. Open the ML target generation file and decide which target set is worth keeping for the competition.
4. Open the meta learner files and check how signal outcomes are recorded.
5. Decide whether the repo needs a lightweight Postgres store or can use a simpler SQLite prototype for development.
6. Ignore all front-end visualization code unless explicitly needed for debugging.

---

## 7. Best implementation direction

The correct path for this competition project is:

1. build a backend signal pipeline,
2. create clean feature and target generation,
3. add a lightweight meta-learning layer,
4. persist outcomes and checkpoints in a small DB,
5. connect the model to Alpaca paper trading,
6. keep the UI minimal or omit it entirely.

Do not spend time recreating the old dashboard or product shell. The old repo’s edge is in the signal logic, not the UI.

---

## 8. Quick decision rules for the agent

### Do this
- copy logic modules that create and label market signals,
- port MTF RSI logic and SNR detection,
- preserve the forward target design,
- keep the meta-learning and persistence layers focused.

### Do not do this
- copy the old chart app, visual overlays, or full frontend shell,
- bring over product-level app state management,
- copy UI session code unless it’s required for debugging.

---

## 9. Final instruction to the next agent

The next agent should treat the original repo as a research and source book, not a direct template.

It should extract the reusable intelligence, create a compact but production-meaningful version in this repo, and then build the Alpaca options strategy on top of that signal pipeline.

The end goal is not to recreate the old app. The end goal is to preserve the decision-making stack that made the original system useful and repackage it into a competition-grade trading agent.

---

## 10. Important repo-local note

This file is meant to live in the new repo as a direct handoff to the agent that will do the work in this project. The next agent should come back to the original repo only for source investigation when a missing detail is required.

The intended workflow is:

- copy the logic you need,
- keep the repo lean,
- do not copy the UI,
- only reach back into the original project for missing implementation details.

---

## 11. Implementation notes from actual build (supplementary)

The prescriptive handoff above is a guide, not a mandate. The actual implementation in this repo is already built with key clarifications:

### Directory structure (ACTUAL)
NOT as prescribed in section 4, but rather:

```
backend/app/
├── core/
│   ├── market/
│   │   ├── mtf_rsi.py                    # Multi-timeframe RSI
│   │   ├── divergence_scale.py           # Divergence normalization
│   │   └── signal_events.py              # Signal bundle generation
│   ├── analysis/
│   │   ├── technical_indicators.py       # Feature generation
│   │   ├── support_resistance.py         # SNR zone detection
│   │   └── trading/
│   │       ├── signal_generator.py       # Event labeling
│   │       └── signal_generator_optimized.py
│   └── ml/
│       ├── ml_dataset_preparation.py     # Target enrichment
│       ├── ti_meta_features.py           # Meta features
│       ├── signal_meta_learner.py        # Signal scoring
│       └── meta_learner.py               # Reward calculation
├── db/
│   └── models.py                         # SignalOutcome + LearnerCheckpoint (SQLAlchemy)
├── agent/
│   └── loop.py                           # Continuous signal generation loop
├── utils/
│   └── alpaca_client.py                  # Alpaca paper trading integration
├── api/
│   └── __init__.py
└── main.py                               # FastAPI app with inline endpoints
```

### Database layer (ACTUAL)
NOT explicit repository classes, but direct SQLAlchemy ORM models:
- `SignalOutcome` — persists signal outcomes with MFE/MAE/reward tracking
- `LearnerCheckpoint` — persists learner state for future sessions

This is intentionally lean; repositories can be added later if CRUD logic grows.

### API layer (ACTUAL)
NOT separate route files, but inline endpoints in `main.py`:
- `GET /health` — service health
- `GET /status` — backend inference status
- `POST /signal/bundle` — compute signal bundle from candles

More endpoints (logs, positions, P&L) are TODO comments.

### Agent execution (ACTUAL)
`app/agent/loop.py` implements continuous signal generation loop (not documented in handoff).
- Runs scheduled checks for new candles
- Generates MTF RSI, divergence, signal bundle
- Persists outcomes to DB
- Ready for Alpaca paper trading integration via `app/utils/alpaca_client.py`

### PostgreSQL setup (ACTUAL)
Database connection is configured via environment variables (see `.env` file).
No explicit Alembic migrations yet; use SQLAlchemy `Base.metadata.create_all()` for table creation or add Alembic later.

### How to run (ACTUAL)
1. Install dependencies: `pip install -r requirements.txt`
2. Set up `.env` with PostgreSQL connection string (e.g., `DATABASE_URL=postgresql://user:pass@localhost/lablab`)
3. Run backend: `python -m uvicorn app.main:app --reload` or use Docker: `docker-compose up`
4. Backend tests: `pytest tests/` (all passing as of last build)

### What was actually chosen vs what handoff prescribed

| Aspect | Handoff Prescribed | Actual Implementation | Reason |
|--------|-------------------|----------------------|--------|
| Repositories | Explicit `*_repository.py` classes | Direct SQLAlchemy models | Leaner; repos can be added if CRUD logic scales |
| API routes | Separate `api/routes/*.py` files | Inline in `main.py` | Simpler for MVP; keeps all endpoints visible |
| Directory structure | `signal_pipeline/features/...` | `app/core/market/`, `app/core/analysis/`, `app/core/ml/` | Better semantic organization for complexity |
| Agent loop | Not mentioned | `app/agent/loop.py` implemented | Needed for continuous signal generation |
| Alpaca integration | Not mentioned | `app/utils/alpaca_client.py` implemented | Required for options strategy execution |
| Migrations | Alembic or equivalent | Not yet set up | Can add later; using SQLAlchemy models for now |

### Key design principles actually preserved

✅ No lookahead leakage in feature generation (historical-only)  
✅ SNR event labels (bounce_support, bounce_resistance, breakout_support, breakout_resistance)  
✅ Multi-target enrichment for options decision engine  
✅ Meta-learning for signal quality evaluation  
✅ Signal outcome persistence for audit trail  
✅ Backend-only inference (frontend is display layer)  
✅ Lean, focused implementation (not a UI-heavy product)  

---

## 12. Next agent: How to extend this

If future work requires:

**Database scale-up**: Add explicit repository classes in `app/db/repositories/` following the pattern in the reference repo.

**More API endpoints**: Add them to `main.py` or split into `app/api/routes/` if file size exceeds ~300 lines.

**Scheduled signal checks**: Extend `app/agent/loop.py` with APScheduler config for live market hours.

**Live trading**: Integrate Alpaca via `app/utils/alpaca_client.py` and call it from within signal outcomes.

**Migrations**: Add Alembic if schema changes become frequent; initialize with `alembic init alembic && alembic revision --autogenerate`.

**Testing coverage**: Current coverage includes unit tests for all core signal modules; add integration tests for agent loop and Alpaca integration as needed.

The foundation is solid. This handoff is now both a design guide and a real implementation reference.
