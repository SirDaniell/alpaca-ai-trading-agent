# Start Here

AXE Genesis is a research prototype for market signal intelligence,
meta-learning, and Alpaca paper-account experimentation. It is not production
trading software and should be used with paper credentials only.

## First steps

1. Read the [root README](README.md) for project status, architecture, setup,
   API routes, and the curated repository map.
2. Configure the backend from [backend/.env.example](backend/.env.example).
3. Start the backend locally from `backend/`:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. Check `http://localhost:8000/health` and open the API reference at
   `http://localhost:8000/docs`.
5. Start the dashboard from `frontend/` with `npm install` and `npm run dev`.

## Read next

- [Signal intelligence handoff](docs/agent_handoff_signal_intelligence.md)
- [Frontend/API contract](docs/frontend_api_contract_and_guide.md)
- [Model update workflow](docs/model_update_workflow.md)
- [Training notebooks](notebooks/training)

Do not commit API keys, datasets, logs, checkpoints, or model weights.
