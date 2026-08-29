# Handover: Meta-learner, TI features, DXY miniseries, 48-bar window

**Date:** 2026-08-29  
**Project:** `lablab.ai Alpaca AI`  
**Original source book:** `/media/daniel-joseph/Linux_Data/Dan_backup/Projects/JavaScript/fin-dash-buddy-sos`

This file is the pickup point if the chat hits a limit. Read it before changing ML training or signal construction.

---

## Goal (user, current phase)

1. Meta-learner **can learn**, with **configurable data size**, persist sessions/outcomes/checkpoints to DB.
2. First train should be **massive** (historical or huge synthetic); later runs **load the checkpoint** and **fine-tune via replay**.
3. Training must use **`calculate_ti_features`** and the **full 200+ TI columns**, not a 16-key toy vector.
4. Decision always uses the **last 48 bars**.
5. Flow: **fetch OHLCV → TI → MTF SNR zones → DXY/index crosses → decide**.
6. Crosses are **not** “RSI vs itself”. Original StockChart miniseries:
   - DXY is **constructed from FX pairs** (ICE/FINEX Dollar basket).
   - MiniSeries: asset vs DXY, slow (structural) and fast (momentum) z-scored scales.
   - Three MTF RSI crosses: **index × MA14**, **DXY × MA14**, **index RSI × DXY RSI**.
   - Plus **DXY vs symbol** alignment crosses (`detectDxySymbolCrossSignals`).

Synthetic data is OK until historical ingest exists.

---

## What already works

### Endpoints (`backend/app/main.py`)

| Method | Path | Notes |
|---|---|---|
| POST | `/meta-learner/train-synthetic` | body: `num_candles`, `train_steps`, `seed`, `symbol`, `scope`, `persist` |
| GET | `/meta-learner/runs` | training runs |
| GET | `/meta-learner/sessions/{id}` | stored synthetic candles |
| GET | `/meta-learner/checkpoints?scope=` | checkpoint metadata (weights in JSON as `checkpoint_b64`) |

No `predict` endpoint yet. Agent loop does **not** call the meta-learner.

### Train CLI

```bash
cd backend
PYTHONPATH=. .venv/bin/python3 scripts/train_meta_learner_synthetic.py --num-candles 20000 --train-steps 2000 --scope bootstrap-v1
```

Same `scope` on a later run **reloads** the latest checkpoint then appends replay (fine-tune path). Old checkpoints were 16-dim; after the TI/48-bar contract bump they **will not** load — bump scope name (e.g. `bootstrap-ti-48-v1`).

### Persistence

- Default DB: SQLite `backend/data/meta_learner.db` (`DATABASE_URL` override).
- Models: `SignalOutcome`, `LearnerCheckpoint`, `SyntheticMarketSession`, `MetaLearnerTrainingRun` in `backend/app/db/models.py`.
- `connection.py` uses SQLite by default so this phase does not need Postgres.

### Tests that passed (pre-TI-window wiring)

`backend/tests/test_synthetic_meta_trainer.py` — 3 tests, with CPU torch in `.venv`.  
A sigmoid overflow on huge pip-scaled rewards was clamped in `signal_meta_learner.py` / `meta_learner.py`. ATR for stocks must be converted to pips via `get_instrument_metadata`.

Torch: `backend/.venv` has `torch 2.13.0+cpu`. Install was `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

---

## Original logic to preserve (do not simplify again)

### DXY from pairs

Source of truth:

`/media/daniel-joseph/Linux_Data/Dan_backup/Projects/JavaScript/fin-dash-buddy-sos/Backend/app/core/analysis/currency_index.py`

Copy into:

`backend/app/core/analysis/currency_index.py`

ICE/FINEX Dollar: `scalar * Π pair^weight` on `EURUSD, USDJPY, GBPUSD, USDCAD, USDSEK, USDCHF`.  
Columns: `{field}_{pair}` e.g. `close_EURUSD`.

Frontend: `HydrationManager` + `OptimizedIndicesComputationService` compute indices; StockChart reads `computedIndices.Dollar.timestampSeriesPerTf[tf]`.

### Miniseries / crosses

- `fin-dash-buddy-sos/Frontend/src/components/StockChart.tsx` — `plotMtfRsi`, divergence overlay ~1974 and ~3045.
- `MiniSeriesChart.tsx` — time-join DXY, `buildUnifiedDivergenceScale` slow+fast, 4-quadrant regime (STRONG_ASSET / WEAK_ASSET / WEAK_DXY / STRONG_DXY).
- `lib/technical/mtf-rsi.ts` — `detectPairCrosses` × 3 sources; **weighted MTF RSI** across H1+H4+D1 (not single-TF RSI).
- `lib/divergence-chart-scale.ts` — `detectDxySymbolCrossSignals`.
- `lib/technical/mtf-series.ts` — HTF compose-by last point at-or-before (causal).

MTF SNR: `buildMtfSnrZones` in StockChart — current TF + higher TFs, confluence ~0.15%.

---

## Files in this repo (state at handover)

| File | State |
|---|---|
| `app/core/analysis/currency_index.py` | **COPIED** Exact copy from original fin-dash-buddy-sos |
| `app/core/market/dollar_index.py` | **WIRED** `dollar_index_closes`, `pair_closes_to_dxy_candles` |
| `app/core/market/divergence_scale.py` | **WIRED** `detect_dxy_symbol_cross_signals`, z-score scales |
| `app/core/market/mtf_rsi.py` | **UPDATED** Ported weighted MTF RSI (H1/H4/D1) + HTF state resampling & 3-way crosses |
| `app/core/ml/ti_meta_features.py` | **REWRITTEN** full 200+ TI contract + context keys + 48-bar window helpers |
| `app/core/ml/decision_context.py` | **WIRED** `build_decision_feature_matrix` — TI + DXY scale + SNR + crosses |
| `app/core/ml/signal_meta_learner.py` | **WIRED MULTI-HEAD & FEATURE SCALER** 6 heads, `FeatureScaler` serialized inside checkpoints |
| `app/core/ml/synthetic_meta_trainer.py` | **WIRED TEMPORAL SPLITS** Train (70%)/Val (15%)/Test (15%), fit scaler on train only, val eval_metrics |
| `app/core/ml/meta_learner_registry.py` | **CREATED REGISTRY** Pretrain, list, evaluate, set-active (hot-swap), score_live per symbol |
| `scripts/manage_meta_models.py` | **CREATED CLI** Command-line model management (`list`, `evaluate`, `set-active`, `pretrain`) |
| `tests/test_model_registry.py` | **CREATED TESTS** 14 tests for FeatureScaler, temporal splits, registry listing, evaluation, and hot-swapping |

---

## Progress Status & Completed Work

1. **Original `currency_index.py` Copied**: Overwritten from original source of truth.
2. **Signal Meta-Learner 48-Bar Multi-Head Window Wired**:
   - `SIGNAL_META_LOOKBACK_BARS = 48`
   - `input_dim = DECISION_WINDOW_DIM` (200+ TI keys + CONTEXT_FEATURE_KEYS * 48 bars)
   - `hidden_dim = 256` (auto-selected when `input_dim > 128`)
   - `feature_contract_version` = `signal-meta-ti-seq-48-v1`
   - **6 Specialized Heads**: `q_head`, `strength_head`, `pips_head`, `risk_head` (MFE/MAE), `liquidity_head` (Next Zone dist/type), `reversal_head` (Reversal prob)
3. **Feature Scaling & Checkpoint Serialization**:
   - `FeatureScaler` performs `StandardScaler` on the 48-bar feature window
   - Scaler state (`mean_`, `scale_`, `n_samples_seen_`) is serialized into format_version 2 checkpoints
   - Scaler is fitted on training split ONLY, preventing data leakage into val/test/inference
4. **Temporal Train/Val/Test Data Splitting**:
   - Direct port of `processing_manager.py` pre-splitting strategy (70% Train, 15% Val, 15% Test)
   - Target enrichment occurs on full dataset first to preserve forward-looking 24-bar window bounds
   - Validation split computes `win_rate`, `avg_reward`, `avg_mfe_pips`, `avg_mae_pips`, `avg_mfe_mae_ratio`, `avg_reversal_prob`
5. **Model Registry & Hot-Swapping Infrastructure**:
   - `MetaLearnerModelRegistry` service handles symbol-specific model versioning
   - Atomic hot-swapping: `set_active(symbol, checkpoint_id)` switches active network weights and scaler together
   - `score_live(symbol, features, direction)` routes live predictions to active model
6. **Testing & Verification**:
   - Run unit test suite:
     ```bash
     PYTHONPATH=. /media/daniel-joseph/Linux_Data/Dan_backup/Projects/JavaScript/fin-dash-buddy-sos/dev_env/bin/python3 -m pytest tests/test_model_registry.py tests/test_synthetic_meta_trainer.py -v --tb=short
     ```
     Result: **21 passed (100% success rate)**.

---

## Next Steps / Future Work

1. Implement real historical OHLCV ingest feed alongside synthetic generator.
2. Add `POST /meta-learner/predict` REST endpoint accepting the latest 48 bars.
3. Apply `FeatureScaler` and `ModelRegistry` patterns to the upcoming Q-Learner.
