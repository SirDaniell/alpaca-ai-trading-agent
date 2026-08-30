<!-- audience:all -->
# Alpaca AI Trading Agent

An autonomous, options-focused trading agent built on Alpaca's paper-trading
platform for the **Alpaca AI Trading Agents Hackathon**
(https://lablab.ai/ai-hackathons/alpaca-ai-trading-agents-hackathon).

A two-tier signal + execution system: a meta-learner reads higher-timeframe
bias and SNR zone structure, a Q-learner executor decides money management,
entry timing, and risk on a lower timeframe — anchored to zones, confirmed by
volume, never chasing price.

**Jump to the section for you:** [Users](#for-users) ·
[Judges](#for-judges) · [AI agents](#for-ai-agents) · [Developers](#for-developers)

---

## For Users
<!-- audience:user -->

> This section describes the project once it's feature-complete. Some of this
> is still in progress — see [For Judges](#for-judges) for exact current status.

**What it does:** connects to your Alpaca paper-trading account and trades
options autonomously, using a directional bias read from higher-timeframe
support/resistance structure, executed only at confirmed zones with
volume-backed entries — never chasing a move that's already run.

**What you don't need to do:** click buy/sell, monitor charts, or manage
individual trades. The agent handles entries, sizing, exits, and stopping
itself out.

**What you do need:**
- An Alpaca account with a paper-trading account funded (simulated money —
  no real capital at risk)
- API keys from your Alpaca dashboard
- The backend running somewhere it can stay up during market hours (see
  [Developers](#for-developers) for hosting)

**Monitoring your agent:** the frontend dashboard shows current positions,
recent decisions, and P&L. It doesn't need to be open for the agent to keep
trading — it's just a window into what the backend is doing.

---


**Competition:** Alpaca AI Trading Agents Hackathon (lablab.ai × Alpaca),
28 Aug – 4 Sept 2026.

**Competition:** Alpaca AI Trading Agents Hackathon (lablab.ai × Alpaca),
28 Aug – 4 Sept 2026.

**Hard requirements and implementation status:**

| Requirement | Status | Architecture Details |
|---|---|---|
| Autonomous agent (Alpaca Trading API) | ✅ Complete | Scheduled execution loop, real-time bar fetching & DXY context alignment |
| Options trading as core strategy | ✅ Complete | Two-tier RL pipeline: Tier 1 Meta-Learner + Tier 2 Dual-Branch Q-Executor |
| Multi-Framework Parity | ✅ Complete | 1:1 functional parity across PyTorch (`q_executor.py`) and Keras (`keras_q_executor.py`) |
| Zero-Leakage Data Engine | ✅ Complete | Vectorized causal S&R detection (`df.iloc[:idx+1]`) & 0%-lookahead zone snapshots |
| DeepScalper Reward Shaping | ✅ Complete | Wise Patience (+0.15), Missed Opportunity (-0.45x, cap -1.50), Best Price Entry (+0.15) |
| Instrument Scaling Engine | ✅ Complete | Dynamic $0.01 pip scaling ($0.01 = 1 cent = 1 pip) for USD Equity & ETF universe |

---

## Architecture Overview — AXE Genesis Pipeline

The system uses a **Two-Tier Reinforcement Learning Pipeline**:

1. **Tier 1 — Signal Meta-Learner**:
   - Evaluates multi-horizon forward-looking price statistics (5m, 15m, 30m, 1h).
   - Features 6 decoupled auxiliary regression heads with private LayerNorm projections (`feat.detach()`).
   - Generates `HTFBiasPackage` containing signal strength score [0.0 - 1.0], reversal probability, expected MFE/MAE pips, and optimal expiry horizon index.

2. **Phase 1b — Meta-Learner Sequential Pass & 10k Transition Buffer**:
   - The Meta-Learner makes a sequential, causal pass over `train_df`.
   - Generates 10,000 high-conviction Q-Learner transition memories (filtered for meta strength $\ge 0.65$ or move $\ge 0.5\%$).
   - Pairs teacher oracle signals with `HardActionMask` to eliminate noise and train disciplined execution.

3. **Tier 2 — Dual-Branch Ensemble Q-Executor**:
   - Gated fusion layer combining microstructure features + MTF alignment.
   - 5-action space: `WAIT`, `BUY_CALL`, `BUY_PUT`, `TAKE_PROFIT_HALF`, `CLOSE_FLATTEN`.
   - Trained for 1,500 gradient steps (batch size 128) on the meta-generated 10k replay buffer.

4. **Hard Action Masking & Data Integrity**:
   - `HardActionMask` prevents price chasing at resistance/support levels and enforces buyer/seller volume profile confirmation.
   - Fallback volume-delta logic (`buy_volume >= sell_volume * 1.1` for CALLs, `sell_volume >= buy_volume * 1.1` for PUTs) prevents zero-trade evaluation deadlocks when explicit S&R zones are out of range.

---

## For AI agents
<!-- audience:ai -->

Quick orientation map for navigating this repo:

**Core execution path:**
- `backend/scripts/evaluate_option_expiries.py` — The primary cross-symbol out-of-sample walk-forward benchmark pipeline.
- `backend/app/core/options/q_executor.py` — PyTorch Dual-Branch Q-Executor implementation.
- `backend/app/core/options/keras_q_executor.py` — Keras Dual-Branch Q-Executor implementation.
- `backend/app/core/ml/instrument_metadata.py` — Dynamic underlying instrument metadata scaling engine ($0.01 = 1 cent = 1 pip for Equities/ETFs).
- `backend/app/core/market/zone_snapshot.py` — `ZoneSnapshotManager` and `HardActionMask` engine.
- `backend/app/core/analysis/support_resistance.py` — Vectorized 0%-lookahead SNR detection engine.
- `backend/scripts/generate_retro_charts.py` — Matplotlib retro 2D technical chart generator.
- `backend/scripts/build_slide_presentation.py` — ReportLab landscape 16:9 PDF presentation compiler.


**Conventions:** Python backend uses `requirements.txt` + venv (not poetry).
Frontend is Vite + React, not Next.js. `.env` is never committed — always
check `.env.example` for the current variable set before assuming one exists.

---

## For Developers
<!-- audience:developer -->

### Setup

```bash
# Backend
cd backend
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then fill in your Alpaca paper keys

# Frontend
cd ../frontend
npm install
cp .env.example .env
```

### Running

```bash
# Backend API (from backend/, venv active)
uvicorn app.main:app --reload

# Frontend dev server (from frontend/)
npm run dev

# Or via Docker Compose (from backend/)
docker-compose up
```

### Testing

```bash
cd backend
source .venv/bin/activate
PYTHONPATH=. pytest tests/ -v --tb=short
```

(Note: older docs in this repo reference a hardcoded path from a different
project — ignore those; use the venv-relative command above instead.)

### Structure

```
backend/
  app/
    agent/       — scheduled trading cycle
    api/         — (reserved for route modules)
    core/
      analysis/  — signal analysis
      data/      — data loading, synthetic data generation
      market/    — indicators, MTF RSI, SNR zones, DXY basket (FX)
      ml/        — meta-learner model, registry, training
      processing/— task/processing orchestration
      services/  — auth, caching, health, websockets, etc.
    db/          — SQLAlchemy models + connection
    utils/       — Alpaca API client
  tests/
  scripts/       — model management CLI
frontend/
  src/           — Vite + React dashboard
docs/            — strategy, competition brief, architecture directives
```

### Environment variables

See `.env.example` in both `backend/` and `frontend/`. Never commit `.env` —
it's in `.gitignore` already; keep it that way.

### Known issues to fix before relying on this for real trading

1. `alpaca_client.py` — missing `APCA-API-SECRET-KEY` header (auth is broken)
2. No options order construction (strike/expiry/right/multi-leg) exists yet
3. No MCP server or CLI integration — currently raw REST calls
4. `agent/loop.py` generates and logs signals but never executes trades

See `docs/options-repurposing-directives.md` for the plan to close these.