# AXE Genesis Dashboard

This directory contains the Vite/TanStack React dashboard for AXE Genesis. It
provides monitoring and model-management views backed by the FastAPI service;
it is not a standalone trading application.

See the [root README](../README.md) for the complete architecture and setup
guide, and the [frontend/API contract](../docs/frontend_api_contract_and_guide.md)
for the integration details.

## Views

- `/` — autopilot dashboard and current system state
- `/performance` — performance and trade analytics
- `/models` — model registry and training controls

## Development

Requirements: Node.js and npm.

```bash
cd frontend
npm install
npm run dev
```

The dev server normally runs on the Vite port shown in the terminal. Configure
the backend URL when needed:

```bash
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

Available checks:

```bash
npm run lint
npm run build
npm run preview
```

The backend must be running for live data, controls, and model operations.
Do not commit `.env` files or API credentials.
