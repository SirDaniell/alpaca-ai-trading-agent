import os
import logging
from datetime import datetime
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv

from app.core.market.mtf_rsi import calculate_mtf_rsi
from app.core.market.signal_events import build_signal_bundle
from app.db.connection import init_db, get_db
from app.db.models import LearnerCheckpoint, MetaLearnerTrainingRun, SignalOutcome, SyntheticMarketSession
from app.agent.loop import run_cycle
from app.utils.alpaca_client import AlpacaClient

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Alpaca Trading Agent - Backend")

# Initialize database on startup
@app.on_event("startup")
def startup():
    init_db()
    logger.info("Database initialized")
    
    # Start agent loop scheduler if enabled
    if os.getenv("AGENT_LOOP_ENABLED", "True").lower() == "true":
        interval = int(os.getenv("AGENT_LOOP_INTERVAL_SECONDS", "300"))
        scheduler = BackgroundScheduler()
        scheduler.add_job(run_cycle, "interval", seconds=interval, id="agent_loop")
        scheduler.start()
        logger.info(f"Agent loop scheduler started (interval: {interval}s)")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": "backend-only",
        "signal_inference": True,
        "meta_learner_synthetic_train": True,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/meta-learner/train-synthetic")
def train_meta_learner_synthetic(payload: dict = None, db: Session = Depends(get_db)):
    """Train the signal meta-learner on a configurable synthetic OHLCV session and persist it."""
    from app.core.ml.synthetic_meta_trainer import (
        SyntheticTrainConfig,
        result_to_dict,
        train_from_synthetic,
    )

    payload = payload or {}
    config = SyntheticTrainConfig(
        symbol=str(payload.get("symbol", "AAPL")),
        num_candles=int(payload.get("num_candles", 400)),
        seed=int(payload.get("seed", 42)),
        start_price=float(payload.get("start_price", 150.0)),
        volatility=float(payload.get("volatility", 0.015)),
        trend=float(payload.get("trend", 0.0008)),
        warmup_bars=int(payload.get("warmup_bars", 50)),
        train_steps=int(payload.get("train_steps", 40)),
        batch_size=int(payload.get("batch_size", 32)),
        persist=bool(payload.get("persist", True)),
        scope=str(payload.get("scope", "synthetic-default")),
    )
    result = train_from_synthetic(config=config, db=db)
    return result_to_dict(result)


@app.get("/meta-learner/runs")
def list_meta_learner_runs(limit: int = 20, db: Session = Depends(get_db)):
    runs = (
        db.query(MetaLearnerTrainingRun)
        .order_by(MetaLearnerTrainingRun.created_at.desc())
        .limit(limit)
        .all()
    )
    return {
        "count": len(runs),
        "runs": [
            {
                "run_id": r.run_id,
                "session_id": r.session_id,
                "symbol": r.symbol,
                "num_candles": r.num_candles,
                "experiences_recorded": r.experiences_recorded,
                "train_steps": r.train_steps,
                "final_loss": r.final_loss,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in runs
        ],
    }


@app.get("/meta-learner/sessions/{session_id}")
def get_synthetic_session(session_id: str, db: Session = Depends(get_db)):
    session = db.query(SyntheticMarketSession).filter(
        SyntheticMarketSession.session_id == session_id
    ).first()
    if session is None:
        return {"error": "not_found"}
    candles = session.candles or []
    return {
        "session_id": session.session_id,
        "symbol": session.symbol,
        "num_candles": session.num_candles,
        "seed": session.seed,
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "candle_count": len(candles),
        "candles_preview": candles[:5],
    }


@app.get("/meta-learner/checkpoints")
def list_learner_checkpoints(scope: str = "synthetic-default", db: Session = Depends(get_db)):
    rows = (
        db.query(LearnerCheckpoint)
        .filter(LearnerCheckpoint.scope == scope)
        .order_by(LearnerCheckpoint.created_at.desc())
        .limit(10)
        .all()
    )
    return {
        "count": len(rows),
        "checkpoints": [
            {
                "checkpoint_id": c.checkpoint_id,
                "scope": c.scope,
                "feature_schema_hash": c.feature_schema_hash,
                "training_counters": c.training_counters,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
            for c in rows
        ],
    }


@app.get("/status")
def status():
    client = AlpacaClient()
    account = client.get_account()
    return {
        "agent": "active",
        "frontend_display_only": True,
        "backend_inference": ["mtf_rsi", "divergence_scale", "signal_bundle", "meta_learner"],
        "alpaca_connected": account is not None,
        "market_open": client.is_market_open(),
        "portfolio_value": client.get_portfolio_value(),
    }


@app.post("/signal/bundle")
def signal_bundle(payload: dict):
    candles = payload.get("candles", [])
    index_candles = payload.get("index_candles", [])
    tf_data = payload.get("timeframes", {})
    
    bundle = build_signal_bundle(candles, index_candles)
    if tf_data:
        bundle["mtf_rsi"] = calculate_mtf_rsi("H1", tf_data)
    
    return bundle


@app.get("/logs")
def get_logs(db: Session = Depends(get_db)):
    """Get recent signal decision logs (latest 50 outcomes)."""
    try:
        outcomes = db.query(SignalOutcome)\
            .order_by(SignalOutcome.created_at.desc())\
            .limit(50)\
            .all()
        
        return {
            "count": len(outcomes),
            "outcomes": [
                {
                    "signal_id": o.signal_id,
                    "symbol": o.symbol,
                    "signal_type": o.signal_type,
                    "entry_time": o.entry_time.isoformat(),
                    "entry_price": o.entry_price,
                    "status": o.status,
                    "reward": o.reward,
                    "created_at": o.created_at.isoformat(),
                }
                for o in outcomes
            ]
        }
    except Exception as e:
        logger.error(f"Failed to fetch logs: {e}")
        return {"error": str(e), "count": 0, "outcomes": []}


@app.get("/positions")
def get_positions():
    """Get current positions and P&L from Alpaca."""
    client = AlpacaClient()
    try:
        positions = client.get_positions()
        account = client.get_account()
        
        return {
            "portfolio_value": account.get("portfolio_value") if account else None,
            "cash": account.get("cash") if account else None,
            "buying_power": account.get("buying_power") if account else None,
            "positions": [
                {
                    "symbol": p.get("symbol"),
                    "qty": float(p.get("qty", 0)),
                    "avg_fill_price": float(p.get("avg_fill_price", 0)),
                    "current_price": float(p.get("current_price", 0)),
                    "unrealized_pl": float(p.get("unrealized_pl", 0)),
                    "unrealized_plpc": float(p.get("unrealized_plpc", 0)),
                }
                for p in positions
            ] if positions else [],
        }
    except Exception as e:
        logger.error(f"Failed to fetch positions: {e}")
        return {"error": str(e), "positions": []}
