# AXE Genesis Backend

The backend is a FastAPI service for market signal generation, synthetic
meta-learner training, model registry workflows, performance persistence, and
Alpaca paper-account integration. It is experimental research software; broker
connectivity does not make the models or execution path production-ready.

The canonical project guide is the [root README](../README.md). This file
focuses on backend setup and development.

## Setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set database, paper-account, scheduler, and market settings in `.env`. Never
commit that file or any credentials.

## Run the API

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Useful endpoints:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/status
```

OpenAPI documentation is available at `http://localhost:8000/docs` and
`http://localhost:8000/openapi.json`.

## Tests and workflows

```bash
PYTHONPATH=. pytest tests/ -v --tb=short

PYTHONPATH=. python scripts/evaluate_option_expiries.py \
  --symbols SPY --framework pytorch --limit 40000

python scripts/manage_meta_models.py list
python scripts/manage_meta_models.py list --symbol AAPL
```

Run the focused model compatibility test with the project-approved Python
environment when available:

```bash
pytest tests/test_model_weight_compatibility.py -q
```

## API ownership

Routes are currently defined in `app/main.py`. The main implementation areas
are:

```text
app/
├── agent/                  # Scheduled signal and execution cycle
├── axe_paka_v1/            # Prototype model/runtime integration
├── core/
│   ├── analysis/            # Technical and support/resistance features
│   ├── market/              # MTF indicators, DXY, zones, signal events
│   ├── ml/                  # Models, datasets, training, registries
│   └── options/             # Options and Q-executor workflows
├── db/                      # SQLAlchemy models and persistence
├── main.py                  # FastAPI application and route definitions
└── utils/                   # Alpaca client and shared utilities
```

`scripts/` contains training, evaluation, audit, export, and monitoring CLIs.
`tests/` contains unit, integration, model, and workflow checks. Generated
datasets, logs, checkpoints, and weights are ignored by Git.

## Current limitations

- Use paper accounts only.
- Options execution, risk controls, and scheduler behavior require review
  before broker-backed experimentation.
- MCP orchestration is planned/deferred.
- The parity verifier has a known gradient-detach failure; do not treat the
  parity workflow as complete solely because notebook structure matches.
