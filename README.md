# Alpaca AI Trading Agents Hackathon — Project

Monorepo for an autonomous AI trading agent built on Alpaca's Trading API and MCP/CLI tooling for options-focused execution in the paper-trading environment.

## Structure
- `backend/` — Python autonomous trading agent (cloud-hosted, does the actual trading loop)
- `frontend/` — Vite + React dashboard for monitoring, not required to be always-on
- `docs/` — strategy notes, architecture handoff, and competition context

## Current status
Active development. The backend signal intelligence pipeline features an Multi-Head Online Reinforcement Meta-Learner with:
- **Full Technical Indicators Engine**: 200+ causal numeric indicators (`ti_meta_features.py`).
- **Sequential Decision Matrix**: 48-bar lookback window input tensor for PyTorch Deep Q-Network.
- **FX Pair-Constructed DXY Index & MTF Crosses**: Basket-derived Dollar index (`EURUSD`, `USDJPY`, `GBPUSD`, `USDCAD`, `USDSEK`, `USDCHF`), z-scored slow/fast divergence scale, and 3-way MTF RSI crosses.
- **Weighted Multi-Timeframe RSI**: Continuous HTF resampling (`H1`, `H4`, `D1`) with Wilder state tracking.
- **Multi-Head Network Architecture**: Shared PyTorch backbone feeding 6 specialized heads (`q_head`, `strength_head`, `pips_head`, `risk_head`, `liquidity_head`, `reversal_head`).
- **Feature Scaling & Serialization**: In-house `FeatureScaler` that serializes scaler state directly inside versioned checkpoints, eliminating normalisation drift at inference.
- **Temporal Train/Val/Test Data Splitting**: Chronological data partitioning (70% Train, 15% Val, 15% Test) with target enrichment performed prior to splitting and scaler fitting constrained to training data only.
- **Model Registry & Hot-Swappable Inference**: `MetaLearnerModelRegistry` for storing, listing, evaluating, pretraining, and hot-swapping active models per symbol.
- **Synthetic Trainer & Persistence**: Configurable market session simulation, 24-bar forward move outcomes (MFE, MAE, RL rewards), multi-task loss optimization, and SQLite checkpoint/session persistence.

## Competition context
This project is being prepared for the Alpaca AI Trading Agents Hackathon, where the required submission is an autonomous options trading agent operating on a fresh $100,000 paper-trading account and interfacing with Alpaca via the MCP server or CLI.

## How to Run Tests & Model Registry CLI

```bash
cd backend

# Run full backend test suite (unit tests + model registry + synthetic trainer)
PYTHONPATH=. /media/daniel-joseph/Linux_Data/Dan_backup/Projects/JavaScript/fin-dash-buddy-sos/dev_env/bin/python3 -m pytest tests/test_model_registry.py tests/test_synthetic_meta_trainer.py tests/test_signal_pipeline.py -v --tb=short

# Run synthetic meta-learner bootstrap training
PYTHONPATH=. /media/daniel-joseph/Linux_Data/Dan_backup/Projects/JavaScript/fin-dash-buddy-sos/dev_env/bin/python3 scripts/train_meta_learner_synthetic.py --num-candles 300 --train-steps 20 --scope bootstrap-ti-48-v1

# Manage Meta-Learner Models (List, Evaluate, Hot-Swap Active Model, Pretrain)
PYTHONPATH=. /media/daniel-joseph/Linux_Data/Dan_backup/Projects/JavaScript/fin-dash-buddy-sos/dev_env/bin/python3 scripts/manage_meta_models.py list
PYTHONPATH=. /media/daniel-joseph/Linux_Data/Dan_backup/Projects/JavaScript/fin-dash-buddy-sos/dev_env/bin/python3 scripts/manage_meta_models.py evaluate --checkpoint-id <uuid>
PYTHONPATH=. /media/daniel-joseph/Linux_Data/Dan_backup/Projects/JavaScript/fin-dash-buddy-sos/dev_env/bin/python3 scripts/manage_meta_models.py set-active --symbol EURUSD --checkpoint-id <uuid>
PYTHONPATH=. /media/daniel-joseph/Linux_Data/Dan_backup/Projects/JavaScript/fin-dash-buddy-sos/dev_env/bin/python3 scripts/manage_meta_models.py pretrain --symbol EURUSD --scope eurusd-base-v1 --num-candles 500
```

## Notes
- Use a dedicated paper-trading account for final judging.
- Keep `.env` local-only and never commit it.
- The reusable signal logic in `docs/` is based on the FinDash signal architecture and model update workflow notes.


