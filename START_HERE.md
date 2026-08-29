# Quick Integration Checklist - Ready to Run

## ⏰ Time to Start: ~5 minutes

Follow these steps to get the backend running with Alpaca paper trading.

---

## Step 1: Get Alpaca API Credentials (2 min)

1. Go to https://app.alpaca.markets
2. Create a free account or sign in
3. Click on your profile → Settings → API Keys
4. Click "Create New Key"
5. Keep the default settings (Paper Trading / Simulated)
6. Copy the **API Key** and **Secret Key**
7. Keep this tab open for next step

---

## Step 2: Set Up Environment (1 min)

From the backend directory:

```bash
cd backend
cp .env.example .env
```

Edit `.env` with your Alpaca credentials:

```env
# Paste your Alpaca credentials here
ALPACA_API_KEY=your_api_key_from_step_1
ALPACA_SECRET_KEY=your_secret_key_from_step_1

# Rest can stay as defaults
DATABASE_URL=postgresql://lablab_user:lablab_pass@localhost:5432/lablab_trading
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

Save and close the file.

---

## Step 3: Start Backend (2 min)

### Option A: Quick Start (Recommended for First Run)

```bash
./setup.sh
python -m uvicorn app.main:app --reload
```

This will:
- Create virtual environment
- Install dependencies
- Initialize database
- Start server on http://localhost:8000

### Option B: Docker (No Local Setup Needed)

```bash
docker-compose up
```

Wait for output like:
```
postgres | database system is ready to accept connections
backend | Uvicorn running on http://0.0.0.0:8000
```

---

## Step 4: Verify Connection (30 sec)

In a new terminal:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "ok",
  "mode": "backend-only",
  "signal_inference": true,
  "timestamp": "2026-08-29T12:34:56.789012"
}
```

Check Alpaca connection:

```bash
curl http://localhost:8000/status
```

Expected (with Alpaca connected):
```json
{
  "agent": "active",
  "frontend_display_only": true,
  "backend_inference": [...],
  "alpaca_connected": true,
  "market_open": true,
  "portfolio_value": 100000.00
}
```

---

## Step 5: Monitor Signal Generation

The agent loop runs automatically every 5 minutes (or manually):

### Option A: Watch Logs (While Server Running)
Look at the server output, should see periodic messages like:
```
INFO:app.agent.loop:Processing AAPL...
INFO:app.agent.loop:✓ AAPL signal generated: {...}
```

### Option B: Check API Logs

```bash
curl http://localhost:8000/logs
```

Response:
```json
{
  "count": 5,
  "outcomes": [
    {
      "signal_id": "AAPL_2026-08-29T12:34:56.789012",
      "symbol": "AAPL",
      "signal_type": "MTF_RSI_DIVERGENCE",
      "entry_time": "2026-08-29T12:34:56.789012",
      "entry_price": 234.56,
      "status": "unresolved",
      "created_at": "2026-08-29T12:34:56.789012"
    },
    ...
  ]
}
```

### Option C: Trigger Manual Cycle

```bash
python -m app.agent.loop
```

Output:
```
INFO:app.agent.loop:Account connected. Portfolio value: $100000.00
INFO:app.agent.loop:Processing AAPL...
INFO:app.agent.loop:Processing TSLA...
INFO:app.agent.loop:Processing GOOGL...
INFO:app.agent.loop:Processing MSFT...
INFO:app.agent.loop:Cycle complete
```

---

## What's Happening Behind the Scenes?

1. **Agent Loop** runs every 5 minutes (configurable)
2. **Fetches latest market data** from Alpaca for AAPL, TSLA, GOOGL, MSFT
3. **Calculates signals**:
   - Multi-timeframe RSI (14-period Wilder smoothing)
   - Divergence detection (normalized)
   - Support/resistance zones (SNR bounce/breakout labels)
4. **Persists to database** for tracking and meta-learning
5. **Exposes via REST API** for frontend and monitoring

---

## Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| "Connection refused" | Start Docker: `docker-compose up postgres` |
| "ALPACA_API_KEY not found" | Check `.env` file exists and credentials are filled |
| "No signals generated" | Market may be closed. Try during 9:30 AM - 4:00 PM EST |
| "ModuleNotFoundError" | Run `pip install -r requirements.txt` |
| "Port 8000 already in use" | Kill existing process or use different port: `--port 8001` |

---

## Next: Build Your Trading Strategy

Once you see signals flowing, the next step is to add your options trading logic.

See `docs/strategy.md` for integration points.

---

## API Reference (Quick)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Service health check |
| `/status` | GET | System status + Alpaca connectivity |
| `/logs` | GET | Recent signal outcomes (latest 50) |
| `/positions` | GET | Current positions from Alpaca |
| `/signal/bundle` | POST | Generate signals from candles |

More details: `backend/README.md`

---

## Support

- **Stuck?** Check `README.md` troubleshooting section
- **API docs** available at `http://localhost:8000/docs` (when running)
- **Code docs** in each module
- **Questions?** See `docs/agent_handoff_signal_intelligence.md` for architecture

---

**Status**: Ready to start! Just add Alpaca credentials and run. ✅
