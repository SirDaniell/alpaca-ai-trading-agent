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

**Hard requirements and current status:**

| Requirement | Status |
|---|---|
| Autonomous agent (Alpaca Trading API) | 🟡 In progress — signal generation works, execution wiring in progress |
| Options trading as core strategy | 🟡 In progress — see `docs/options-repurposing-directives.md` |
| Integration via Alpaca MCP server or CLI | 🟡 In progress — currently uses direct REST, being migrated |
| Fresh dedicated paper account, $100k balance | ⬜ Set up at submission time, not during development |
| One-page write-up (AI logic, risk gates, infra) | ⬜ Pending — will live at `docs/write-up.md` |

We're being upfront here rather than overstating completion — full transparent
gap tracking lives in `docs/` (see [For AI agents](#for-ai-agents) for exact
file map). The one-page submission write-up will be the authoritative,
concise summary; everything else in `docs/` is working detail behind it.

**Where to look for the actual strategy logic:** `docs/strategy.md` for the
signal architecture, `docs/options-repurposing-directives.md` for how it's
being adapted to options with zone-anchored, volume-confirmed entries.

---

## For AI agents
<!-- audience:ai -->

Quick orientation map for navigating this repo without reading everything.

**Core execution path:**
- `backend/app/agent/loop.py` — the scheduled cycle: fetch data → generate
  signals → (execution wiring in progress, does not yet place trades)
- `backend/app/utils/alpaca_client.py` — Alpaca API wrapper. **Known bug:**
  `APCA-API-SECRET-KEY` header is never set, only the key ID — authenticated
  calls currently fail. Fix before relying on this for live calls.
- `backend/app/core/ml/` — meta-learner model code (registry, training,
  inference)
- `backend/app/core/market/` — signal engine: indicators, MTF RSI, SNR zones,
  divergence scale. **FX-specific pieces here (DXY basket) should not be
  modified** — see `docs/options-repurposing-directives.md`, Directive 1.
- `backend/app/db/` — SQLAlchemy models + connection for signal/outcome
  persistence

**Docs to read before making changes:**
- `docs/competition-brief.md` — full competition rules and requirements
- `docs/options-repurposing-directives.md` — architecture plan for the
  options build: two-tier meta-learner/Q-learner split, zone-anchored entry
  design, no-chase rule, zone snapshot/repaint handling, volume delta
  integration, reward function design
- `docs/strategy.md` — original signal architecture notes

**Known gaps (do not assume these are done just because older docs claim
so):**
- `GAPS_FIXED.md` and `BACKEND_IMPLEMENTATION_COMPLETE.md` mark signal
  generation and Alpaca integration as complete — true only for *data
  fetching and logging*, not for trade execution. No options support, no
  MCP/CLI integration, and `place_order()` is never called anywhere in the
  codebase as of this writing.
- The Q-learner executor described in `docs/options-repurposing-directives.md`
  does not exist yet as code — it's a design doc, not an implementation.

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