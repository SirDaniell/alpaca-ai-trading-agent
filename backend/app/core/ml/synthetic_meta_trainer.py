"""Train the online signal meta-learner on a configurable synthetic session and persist it."""

from __future__ import annotations

import base64
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
from sqlalchemy.orm import Session

from app.core.data.synthetic_data_generator import SyntheticDataGenerator
from app.core.market.dollar_index import DOLLAR_PAIRS, DOLLAR_START_PRICES, pair_closes_to_dxy_candles
from app.core.ml.decision_context import (
    build_decision_feature_matrix,
    decision_vector_at,
    infer_direction_from_row,
    last_bar_snapshot,
)
from app.core.ml.instrument_metadata import get_instrument_metadata
from app.core.ml.signal_meta_learner import (
    FEATURE_SCHEMA_HASH,
    SIGNAL_META_FEATURE_COUNT,
    SIGNAL_META_HORIZON_BARS,
    OnlineSignalMetaLearner,
)
from app.core.ml.ti_meta_features import (
    DECISION_FEATURE_KEYS,
    SIGNAL_META_FEATURE_CONTRACT_VERSION,
    SIGNAL_META_LOOKBACK_BARS,
)
from app.db.models import (
    LearnerCheckpoint,
    MetaLearnerTrainingRun,
    SignalOutcome,
    SyntheticMarketSession,
)

logger = logging.getLogger(__name__)


@dataclass
class SyntheticTrainConfig:
    symbol: str = "AAPL"
    num_candles: int = 400
    seed: int = 42
    start_price: float = 150.0
    volatility: float = 0.015
    trend: float = 0.0008
    interval_seconds: int = 3600
    warmup_bars: int = 80
    horizon_bars: int = SIGNAL_META_HORIZON_BARS
    train_steps: int = 40
    batch_size: int = 32
    persist: bool = True
    scope: str = "synthetic-default"
    target_sync_every: int = 10
    # Temporal split ratios (chronological — oldest=train, most recent=test)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    # test_ratio = 1 - train_ratio - val_ratio (most recent data, just before today)
    arch_version: str = "v1.0"
    notes: str = "synthetic"


@dataclass
class SyntheticTrainResult:
    run_id: str
    session_id: str
    symbol: str
    num_candles: int
    experiences_recorded: int
    train_steps: int
    final_loss: Optional[float]
    metrics: Dict[str, Any]
    checkpoint_id: Optional[str]
    weights_changed: bool


def _atr_pips(candles: List[Dict[str, Any]], index: int, symbol: str, lookback: int = 14) -> float:
    start = max(0, index - lookback + 1)
    window = candles[start:index + 1]
    if not window:
        return 10.0
    ranges = [float(c["high"]) - float(c["low"]) for c in window]
    price_atr = max(float(np.mean(ranges)), 1e-8)
    return price_atr * get_instrument_metadata(symbol).pip_scale


def _load_or_create_learner(db: Optional[Session], scope: str, symbol: str = "AAPL") -> OnlineSignalMetaLearner:
    """Load the most recent active checkpoint for (scope, symbol), else create fresh."""
    learner = OnlineSignalMetaLearner()
    if db is None:
        return learner
    row = (
        db.query(LearnerCheckpoint)
        .filter(LearnerCheckpoint.scope == scope, LearnerCheckpoint.symbol == symbol)
        .order_by(LearnerCheckpoint.created_at.desc())
        .first()
    )
    if row is None:
        return learner
    payload = row.payload or {}
    b64 = payload.get("checkpoint_b64")
    if not b64:
        return learner
    try:
        learner.import_checkpoint(base64.b64decode(b64))
        logger.info("Restored meta-learner checkpoint %s (steps=%s)", row.checkpoint_id, learner.total_steps)
    except Exception as exc:
        logger.warning("Could not restore checkpoint %s: %s — training from scratch", row.checkpoint_id, exc)
    return learner


def _persist_session(db: Session, config: SyntheticTrainConfig, candles: List[Dict[str, Any]]) -> SyntheticMarketSession:
    session = SyntheticMarketSession(
        session_id=str(uuid4()),
        symbol=config.symbol,
        num_candles=len(candles),
        seed=config.seed,
        start_price=config.start_price,
        volatility=config.volatility,
        trend=config.trend,
        interval_seconds=config.interval_seconds,
        candles=candles,
        created_at=datetime.now(timezone.utc),
    )
    db.add(session)
    db.flush()
    return session


def _persist_outcome(
    db: Session,
    *,
    run_id: str,
    signal_id: str,
    config: SyntheticTrainConfig,
    direction: str,
    entry_time: datetime,
    entry_price: float,
    features: Dict[str, float],
    stats,
) -> None:
    db.add(
        SignalOutcome(
            outcome_id=str(uuid4()),
            signal_id=signal_id,
            symbol=config.symbol,
            source_timeframe="H1",
            direction=direction,
            signal_type="synthetic_rsi_trend",
            entry_time=entry_time,
            entry_price=entry_price,
            horizon_bars=config.horizon_bars,
            horizon_seconds=config.horizon_bars * config.interval_seconds,
            feature_contract_version=SIGNAL_META_FEATURE_CONTRACT_VERSION,
            feature_names=list(DECISION_FEATURE_KEYS),
            feature_values=[features.get(k, 0.0) for k in DECISION_FEATURE_KEYS],
            status="resolved",
            resolved_at=datetime.now(timezone.utc),
            mfe_pips=stats.mfe_pips,
            mae_pips=stats.mae_pips,
            net_pips=stats.net_pips_24h,
            reversal_bar=stats.reversal_bar,
            reward=stats.reward,
            signal_strength=stats.signal_strength,
        )
    )


def _persist_checkpoint(
    db: Session,
    learner: OnlineSignalMetaLearner,
    config: SyntheticTrainConfig,
    metrics: Dict[str, Any],
    eval_metrics: Optional[Dict[str, Any]] = None,
    set_active: bool = True,
) -> str:
    """Persist model checkpoint; optionally deactivate older ones for this symbol+scope."""
    checkpoint_id = str(uuid4())
    encoded = base64.b64encode(learner.export_checkpoint()).decode("ascii")

    # Deactivate previous active checkpoints for this (scope, symbol)
    if set_active:
        db.query(LearnerCheckpoint).filter(
            LearnerCheckpoint.scope == config.scope,
            LearnerCheckpoint.symbol == config.symbol,
            LearnerCheckpoint.is_active == True,  # noqa: E712
        ).update({"is_active": False})

    db.add(
        LearnerCheckpoint(
            checkpoint_id=checkpoint_id,
            scope=config.scope,
            symbol=config.symbol,
            arch_version=config.arch_version,
            checkpoint_version="v2.0",
            feature_schema_hash=FEATURE_SCHEMA_HASH,
            training_counters={
                "total_steps": learner.total_steps,
                "buffer_size": len(learner.replay_buffer),
                "scaler_fitted": learner.scaler.fitted,
            },
            eval_metrics=eval_metrics,
            is_active=set_active,
            notes=config.notes,
            payload={
                "kind": "online_signal_meta_learner_v2",
                "checkpoint_b64": encoded,
                "metrics": metrics,
            },
            created_at=datetime.now(timezone.utc),
        )
    )
    return checkpoint_id


def _collect_split_vectors(
    feature_matrix,
    candles: List[Dict[str, Any]],
    start: int,
    end: int,
    config: SyntheticTrainConfig,
) -> np.ndarray:
    """Assemble raw flat feature vectors for a chronological slice without applying the scaler."""
    min_warmup = max(config.warmup_bars, SIGNAL_META_LOOKBACK_BARS)
    vectors = []
    for index in range(max(start, min_warmup), end - config.horizon_bars):
        window = decision_vector_at(feature_matrix, index)
        if window is not None:
            raw = np.asarray(window, dtype=np.float32).reshape(-1)
            if len(raw) >= SIGNAL_META_FEATURE_COUNT:
                vectors.append(raw[:SIGNAL_META_FEATURE_COUNT])
    return np.stack(vectors, axis=0) if vectors else np.zeros((0, SIGNAL_META_FEATURE_COUNT), dtype=np.float32)


def train_from_synthetic(
    config: Optional[SyntheticTrainConfig] = None,
    db: Optional[Session] = None,
    learner: Optional[OnlineSignalMetaLearner] = None,
) -> SyntheticTrainResult:
    """
    Generate synthetic bars, split them chronologically (train/val/test),
    fit the feature scaler on train only, label with 24-bar outcomes,
    run gradient updates, evaluate on val, and persist to DB.

    Split layout (chronological — oldest to most recent):
        TRAIN (70%)  |  VAL (15%)  |  TEST (15%)
         fit scaler     eval only     held-out
         fit model

    The scaler is frozen after training and stored in the checkpoint,
    ensuring identical normalisation at inference time regardless of which
    model version is active.
    """
    config = config or SyntheticTrainConfig()
    min_warmup = max(config.warmup_bars, SIGNAL_META_LOOKBACK_BARS)
    if config.num_candles < min_warmup + config.horizon_bars + 2:
        raise ValueError(
            f"num_candles must be >= lookback/warmup + horizon + 2 "
            f"({min_warmup + config.horizon_bars + 2})"
        )

    # ── 1. Generate synthetic OHLCV + DXY basket ────────────────────────────
    generator = SyntheticDataGenerator(seed=config.seed)
    candles = generator.to_dict_list(
        generator.generate_session(
            symbol=config.symbol,
            num_candles=config.num_candles,
            start_price=config.start_price,
            volatility=config.volatility,
            trend=config.trend,
            interval_seconds=config.interval_seconds,
        )
    )
    pair_closes = {}
    for offset, pair in enumerate(DOLLAR_PAIRS):
        pair_bars = generator.generate_session(
            symbol=pair,
            num_candles=config.num_candles,
            start_price=DOLLAR_START_PRICES.get(pair, 1.0),
            volatility=config.volatility * 0.6,
            trend=config.trend * (0.4 if offset % 2 == 0 else -0.5),
            interval_seconds=config.interval_seconds,
        )
        pair_closes[pair] = [c.close for c in pair_bars]
    dxy_candles = pair_closes_to_dxy_candles(pair_closes, [c["time"] for c in candles])
    feature_matrix, _ti_frame = build_decision_feature_matrix(candles, dxy_candles)

    # ── 2. Chronological split boundaries ───────────────────────────────────
    n = len(candles)
    train_end = int(n * config.train_ratio)
    val_end = int(n * (config.train_ratio + config.val_ratio))
    # test_end = n  (most recent data, just before today)
    logger.info(
        "[Trainer] Split: train=0:%d  val=%d:%d  test=%d:%d  (total=%d candles)",
        train_end, train_end, val_end, val_end, n, n,
    )

    # ── 3. Load / create learner ─────────────────────────────────────────────
    learner = learner or _load_or_create_learner(
        db if config.persist else None, config.scope, config.symbol,
    )

    # ── 4. Fit scaler on TRAINING slice only (causal — no future data) ──────
    logger.info("[Trainer] Collecting training vectors for scaler fit (%d → %d)...", min_warmup, train_end)
    train_vectors = _collect_split_vectors(feature_matrix, candles, 0, train_end, config)
    if len(train_vectors) > 0:
        learner.fit_scaler(train_vectors)
        logger.info("[Trainer] Scaler fitted on %d training vectors.", len(train_vectors))
    else:
        logger.warning("[Trainer] No training vectors collected — scaler unfitted.")

    # ── 5. Bookkeeping ───────────────────────────────────────────────────────
    weight_before = float(next(learner.net.parameters()).detach().abs().sum().item())
    run_id = str(uuid4())
    session_id = ""
    if config.persist and db is not None:
        session = _persist_session(db, config, candles)
        session_id = session.session_id

    # ── 6. Record TRAINING experiences (scaler already fitted + applied) ─────
    recorded = 0
    for index in range(min_warmup, train_end - config.horizon_bars):
        window = decision_vector_at(feature_matrix, index)
        next_window = decision_vector_at(feature_matrix, index + 1)
        snapshot = last_bar_snapshot(feature_matrix, index)
        direction = infer_direction_from_row(snapshot)
        entry = candles[index]
        future = candles[index + 1:index + 1 + config.horizon_bars]
        signal_id = f"{run_id}_train_{index}"
        stats = learner.record_experience(
            feature_dict=window,
            signal_id=signal_id,
            symbol=config.symbol,
            direction=direction,
            entry_price=float(entry["close"]),
            future_highs=np.array([c["high"] for c in future], dtype=np.float64),
            future_lows=np.array([c["low"] for c in future], dtype=np.float64),
            future_closes=np.array([c["close"] for c in future], dtype=np.float64),
            atr_pips=_atr_pips(candles, index, config.symbol),
            next_feature_dict=next_window,
            next_zone_dist_atr=float(snapshot.get("snr_distance", 2.0)),
            next_zone_type=1.0 if float(snapshot.get("snr_is_support", 0.0)) > 0 else 2.0,
        )
        recorded += 1
        if config.persist and db is not None:
            entry_time = datetime.fromtimestamp(int(entry["time"]), tz=timezone.utc)
            _persist_outcome(
                db,
                run_id=run_id,
                signal_id=signal_id,
                config=config,
                direction=direction,
                entry_time=entry_time,
                entry_price=float(entry["close"]),
                features=snapshot,
                stats=stats,
            )

    # ── 7. Gradient updates on training buffer ───────────────────────────────
    last_metrics: Dict[str, Any] = {"loss": 0.0, "buffer_size": len(learner.replay_buffer)}
    for step in range(config.train_steps):
        last_metrics = learner.train_step(batch_size=config.batch_size)
        if config.target_sync_every and (step + 1) % config.target_sync_every == 0:
            learner.sync_target_network()
    learner.sync_target_network()

    # ── 8. Evaluate on VALIDATION slice (scaler applied, no gradient update) ─
    val_rewards, val_mfes, val_maes, val_revs = [], [], [], []
    for index in range(max(train_end, min_warmup), val_end - config.horizon_bars):
        window = decision_vector_at(feature_matrix, index)
        snapshot = last_bar_snapshot(feature_matrix, index)
        direction = infer_direction_from_row(snapshot)
        entry = candles[index]
        future = candles[index + 1:index + 1 + config.horizon_bars]
        stats = learner.reward_calculator.calculate(
            signal_id=f"{run_id}_val_{index}",
            symbol=config.symbol,
            direction=direction,
            entry_price=float(entry["close"]),
            future_highs=np.array([c["high"] for c in future], dtype=np.float64),
            future_lows=np.array([c["low"] for c in future], dtype=np.float64),
            future_closes=np.array([c["close"] for c in future], dtype=np.float64),
            atr_pips=_atr_pips(candles, index, config.symbol),
        )
        val_rewards.append(stats.reward)
        val_mfes.append(stats.mfe_pips)
        val_maes.append(max(stats.mae_pips, 1e-6))
        val_revs.append(stats.reversal_prob)

    eval_metrics: Optional[Dict[str, Any]] = None
    if val_rewards:
        avg_reward = float(np.mean(val_rewards))
        eval_metrics = {
            "split": "val",
            "n_samples": len(val_rewards),
            "avg_reward": round(avg_reward, 4),
            "win_rate": round(float(np.mean([r > 0 for r in val_rewards])), 4),
            "avg_mfe_pips": round(float(np.mean(val_mfes)), 2),
            "avg_mae_pips": round(float(np.mean(val_maes)), 2),
            "avg_mfe_mae_ratio": round(float(np.mean([m / a for m, a in zip(val_mfes, val_maes)])), 4),
            "avg_reversal_prob": round(float(np.mean(val_revs)), 4),
        }
        logger.info("[Trainer] Val metrics: %s", eval_metrics)

    # ── 9. Persist checkpoint + run ──────────────────────────────────────────
    weight_after = float(next(learner.net.parameters()).detach().abs().sum().item())
    weights_changed = abs(weight_after - weight_before) > 1e-8
    final_loss = last_metrics.get("loss")
    metrics = {
        **last_metrics,
        "weight_l1_before": weight_before,
        "weight_l1_after": weight_after,
        "num_candles": config.num_candles,
        "train_end": train_end,
        "val_end": val_end,
        "warmup_bars": config.warmup_bars,
        "horizon_bars": config.horizon_bars,
        "lookback_bars": SIGNAL_META_LOOKBACK_BARS,
        "scaler_fitted": learner.scaler.fitted,
        "scaler_n_samples": learner.scaler.n_samples_seen_,
        "eval_metrics": eval_metrics,
        "run_id": run_id,
    }


    checkpoint_id = None
    if config.persist and db is not None:
        checkpoint_id = _persist_checkpoint(db, learner, config, metrics, eval_metrics=eval_metrics)
        db.add(
            MetaLearnerTrainingRun(
                run_id=run_id,
                session_id=session_id or None,
                scope=config.scope,
                symbol=config.symbol,
                num_candles=config.num_candles,
                warmup_bars=config.warmup_bars,
                horizon_bars=config.horizon_bars,
                experiences_recorded=recorded,
                train_steps=config.train_steps,
                batch_size=config.batch_size,
                final_loss=final_loss,
                metrics=metrics,
                notes=config.notes,
            )
        )
        db.commit()
        logger.info(
            "Synthetic train complete run=%s candles=%s experiences=%s loss=%s checkpoint=%s",
            run_id, config.num_candles, recorded, final_loss, checkpoint_id,
        )

    return SyntheticTrainResult(
        run_id=run_id,
        session_id=session_id,
        symbol=config.symbol,
        num_candles=config.num_candles,
        experiences_recorded=recorded,
        train_steps=config.train_steps,
        final_loss=final_loss,
        metrics=metrics,
        checkpoint_id=checkpoint_id,
        weights_changed=weights_changed,
    )


def result_to_dict(result: SyntheticTrainResult) -> Dict[str, Any]:
    return asdict(result)

