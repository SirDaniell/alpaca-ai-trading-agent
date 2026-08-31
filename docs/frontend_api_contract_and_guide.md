# 📘 AXE Genesis — Frontend API Contract & Integration Guide for Lovable

> **FastAPI Documentation URL**: `http://localhost:8000/docs`  
> **OpenAPI Specification JSON**: `http://localhost:8000/openapi.json`  
> **Backend Base URL**: `http://localhost:8000`

---

## 🎯 1. Overview & Frontend Purpose

This guide defines the complete REST API contract and synthetic data generation schema for building the **AXE Genesis Autonomous Options Trading Agent** frontend interface. 

The frontend acts as a **Real-Time Agent Monitoring & Model Lifecycle Control Center**. It allows users to observe autonomous trading behavior (Autopilot mode), track multi-period PnL performance, evaluate multi-horizon signal strength, and control reinforcement learning model checkpoints (pretrain, deploy, retrain, delete).

---

## 🖥️ 2. Core User Features & UI Requirements

Users of the AXE Genesis Trading Agent require the following visual components and control widgets:

### A. Agent Operations Control (Autopilot Status)
- **Agent Status Header**: Displays system health (`OK`), loop state (`RUNNING` / `PAUSED` / `STOPPED`), Alpaca API connection status, and NYSE market hours status.
- **Loop Control Buttons**:
  - `Start Autopilot`: Activates periodic automated trading cycles.
  - `Pause Autopilot`: Temporarily halts order placement while maintaining position tracking.
  - `Trigger Cycle Now`: Immediately executes a manual evaluation and execution pass.

### B. Multi-Period PnL & Account Performance Hierarchy
- **Performance Summary Cards (Exact Priority Order)**:
  1. **Session PnL** ($ and %) — accumulated since current dashboard boot.
  2. **Day PnL** ($ and %) — today's realized + unrealized account performance.
  3. **Week PnL** ($ and %) — trailing 7-day performance.
  4. **Month PnL** ($ and %) — trailing 30-day performance.
- **Key Performance Indicators (KPIs)**: Total Trades, Win Rate %, Wins vs Losses count, Current Portfolio Equity, Cash Balance, Buying Power.

### C. Live Position & Execution Feed
- **Open Positions Table**: Underlying Symbol, OCC Option Contract Code, Qty, Avg Fill Price, Current Price, Unrealized PnL ($ and %), Expiry Countdown timer.
- **Trade History Stream**: Completed trades with entry/exit timestamps, entry/exit prices, realized PnL, holding duration (e.g. 60s fast-expiry or 30m horizon), and exit reason (`EXPIRY_REACHED`, `TAKE_PROFIT`, `STOP_LOSS`).

### D. Model Registry & Training Management (Off-Line / On-Demand)
- **Registered Model Checkpoints Table**:
  - Symbol, Scope Name, Creation Timestamp, Active Flag, Scaler Status, Usable Training Samples.
  - Actions: `Pretrain Model`, `Activate (Hot-Swap)`, `View Evaluation`, `Delete Checkpoint`.
- **Pre-Training Modal / Runner**:
  - Parameters: Symbol (`AAPL`, `MSFT`, `SPY`, etc.), `num_candles`, `warmup_bars`, `train_steps`, `batch_size`, `seed`.
  - Live Training Progress & Metrics: Epoch Loss Curve, Win Rate %, Sharpe Ratio, Horizon Alignment Rate (5m, 15m, 30m, 1h).

### E. Signal Intelligence & Multi-Horizon Analysis
- **Live Signal Score Card**:
  - Tier 1 Meta-Learner Conviction Score $[0.0 - 1.0]$.
  - Directional Bias (`BULLISH` / `BEARISH` / `NEUTRAL`).
  - Recommended Option Expiry (`5m`, `15m`, `30m`, `1h`).
  - Reversal Risk Probability % and Expected MFE/MAE Pips.
  - DXY Basket Divergence indicator (Bullish/Bearish macro alignment).

---

## 📡 3. Complete REST API Endpoint Contract

All responses return standard JSON. Errors use standard HTTP status codes (`400 Bad Request`, `404 Not Found`, `500 Internal Server Error`).

---

### Category A: System Status & Health

#### 1. System Health Check
- **`GET /health`**
- **Response**:
```json
{
  "status": "ok",
  "mode": "backend-only",
  "signal_inference": true,
  "meta_learner_synthetic_train": true,
  "timestamp": "2026-08-31T11:30:00Z"
}
```

#### 2. Agent Overview Status
- **`GET /status`**
- **Response**:
```json
{
  "agent": "active",
  "loop_state": "running",
  "frontend_display_only": true,
  "backend_inference": ["mtf_rsi", "divergence_scale", "signal_bundle", "meta_learner"],
  "alpaca_connected": true,
  "market_open": true,
  "portfolio_value": 102450.75,
  "cash": 95200.50,
  "buying_power": 380802.00
}
```

---

### Category B: Agent Loop Control

#### 3. Start Agent Loop
- **`POST /agent/start`**
- **Request Body** *(Optional)*: `{"interval_seconds": 300}`
- **Response**:
```json
{
  "status": "started",
  "loop_state": "running",
  "interval_seconds": 300,
  "timestamp": "2026-08-31T11:30:05Z"
}
```

#### 4. Stop Agent Loop
- **`POST /agent/stop`**
- **Response**:
```json
{
  "status": "stopped",
  "loop_state": "paused",
  "timestamp": "2026-08-31T11:30:10Z"
}
```

#### 5. Trigger Single Execution Cycle
- **`POST /agent/run-cycle`**
- **Request Body** *(Optional)*: `{"symbol": "AAPL", "dry_run": false}`
- **Response**:
```json
{
  "status": "success",
  "cycle_id": "cyc-98124",
  "executed_action": "BUY_CALL",
  "symbol": "AAPL",
  "expiry_selected": "15m",
  "meta_conviction": 0.84,
  "timestamp": "2026-08-31T11:30:15Z"
}
```

---

### Category C: Multi-Period PnL & Performance Metrics

#### 6. Structured Multi-Period Performance Summary
- **`GET /performance/summary`**
- **Response**:
```json
{
  "session_start_time": "2026-08-31T08:00:00Z",
  "session_start_equity": 100000.00,
  "current_equity": 102450.75,
  "session_pnl": 2450.75,
  "session_pnl_pct": 2.45,
  "day_pnl": 1820.50,
  "day_pnl_pct": 1.81,
  "week_pnl": 5340.20,
  "week_pnl_pct": 5.49,
  "month_pnl": 12450.00,
  "month_pnl_pct": 13.83,
  "total_trades": 24,
  "wins": 18,
  "losses": 6,
  "win_rate_pct": 75.00
}
```

#### 7. Open Positions
- **`GET /positions`**
- **Response**:
```json
{
  "portfolio_value": 102450.75,
  "cash": 95200.50,
  "buying_power": 380802.00,
  "positions": [
    {
      "symbol": "AAPL260904C00235000",
      "underlying": "AAPL",
      "contract_type": "CALL",
      "qty": 2,
      "avg_fill_price": 3.45,
      "current_price": 4.10,
      "unrealized_pl": 130.00,
      "unrealized_plpc": 18.84
    }
  ]
}
```

#### 8. Historical Trade Log
- **`GET /performance/trades?limit=50`**
- **Response**:
```json
{
  "count": 1,
  "trades": [
    {
      "trade_id": "trd-001",
      "symbol": "AAPL260904C00235000",
      "side": "buy",
      "qty": 2,
      "asset_type": "option",
      "start_time": "2026-08-31T10:15:00Z",
      "end_time": "2026-08-31T10:45:00Z",
      "entry_price": 3.45,
      "exit_price": 4.10,
      "realized_pnl": 130.00,
      "realized_pnl_pct": 18.84,
      "hold_duration_sec": 1800,
      "exit_reason": "EXPIRY_REACHED"
    }
  ]
}
```

---

### Category D: Model Registry & Off-Line Pretraining

#### 9. List Model Checkpoints
- **`GET /models?symbol=AAPL`**
- **Response**:
```json
{
  "count": 1,
  "models": [
    {
      "checkpoint_id": "chk-aapl-v1",
      "symbol": "AAPL",
      "scope": "prod-v1",
      "active": true,
      "train_steps": 50,
      "final_loss": 1.24e-3,
      "metrics": {
        "scaler_fitted": true,
        "scaler_n_samples": 420
      },
      "created_at": "2026-08-31T09:00:00Z"
    }
  ]
}
```

#### 10. Pretrain Model on Synthetic / Real Data
- **`POST /models/pretrain`**
- **Request Body**:
```json
{
  "symbol": "MSFT",
  "scope": "dev-scope-v1",
  "num_candles": 1500,
  "warmup_bars": 80,
  "train_steps": 20,
  "batch_size": 32,
  "seed": 42,
  "persist": true
}
```
- **Response**:
```json
{
  "checkpoint_id": "chk-msft-v2",
  "symbol": "MSFT",
  "scope": "dev-scope-v1",
  "experiences_recorded": 26,
  "train_steps": 20,
  "final_loss": 2.14e-4,
  "metrics": {
    "scaler_fitted": true,
    "scaler_n_samples": 476,
    "val_loss": 3.12e-4,
    "win_rate_pct": 68.5
  }
}
```

#### 11. Activate (Hot-Swap) Active Model
- **`POST /models/{checkpoint_id}/activate`**
- **Response**:
```json
{
  "status": "activated",
  "checkpoint_id": "chk-msft-v2",
  "symbol": "MSFT",
  "timestamp": "2026-08-31T11:32:00Z"
}
```

#### 12. Delete Model Checkpoint
- **`DELETE /models/{checkpoint_id}`**
- **Response**:
```json
{
  "status": "deleted",
  "checkpoint_id": "chk-msft-v2"
}
```

---

## 🧪 4. Synthetic Data Generation Schema for Lovable / Mocks

When testing frontend UI components without a live backend connection, Lovable can generate synthetic market candles and signal bundles identical to the backend structure:

### Synthetic Market Candle Schema
```json
{
  "timestamp": "2026-08-31T11:30:00Z",
  "open": 150.25,
  "high": 151.80,
  "low": 149.90,
  "close": 151.40,
  "volume": 124500
}
```

### Synthetic Signal Bundle Payload
```json
{
  "candles": [
    {"timestamp": "2026-08-31T11:25:00Z", "open": 150.0, "high": 150.5, "low": 149.8, "close": 150.2, "volume": 10000},
    {"timestamp": "2026-08-31T11:30:00Z", "open": 150.25, "high": 151.8, "low": 149.9, "close": 151.4, "volume": 12450}
  ],
  "timeframes": {
    "H1": [
      {"timestamp": "2026-08-31T11:00:00Z", "close": 151.4, "rsi": 62.4}
    ]
  }
}
```

---

## 🛠️ 5. Instructions for Lovable Agent Prompt

Copy and paste the following prompt directly into **Lovable**:

```text
Build a modern, high-contrast Dark Mode (Slate/Cyan palette) Autonomous AI Options Trading Dashboard for "AXE Genesis".

Key Pages / Tabs:
1. "Autopilot Control & Execution Feed":
   - Top status header: Health (OK), Autopilot State (RUNNING/PAUSED), Alpaca Connected (True), Portfolio Value.
   - Action buttons: Start Autopilot, Stop Autopilot, Trigger Cycle Now.
   - Open Positions table & Historical Executed Trades stream (displaying entry/exit price, realized PnL $, hold duration, exit reason).

2. "Performance Analytics":
   - Top metrics grid displaying Session PnL ($ and %), Day PnL ($ and %), Week PnL ($ and %), Month PnL ($ and %).
   - KPI cards: Total Trades, Win Rate %, Wins vs Losses count, Cash Balance, Buying Power.

3. "Model Registry & Trainer":
   - Table of trained model checkpoints with active model indicator, symbol filter, pretrain modal trigger, and Hot-Swap/Activate button.
   - Pre-training Form: Symbol input, num_candles, batch_size, train_steps, with live loss curve visualization.

Connect all components to REST API base URL `http://localhost:8000` referencing FastAPI OpenAPI schema at `http://localhost:8000/openapi.json`.
```
