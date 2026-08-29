from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Boolean, Column, Float, Integer, String, DateTime, JSON, Index, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class SignalOutcome(Base):
    __tablename__ = "signal_outcomes"

    outcome_id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    signal_id = Column(String(255), nullable=False, unique=True, index=True)
    symbol = Column(String(64), nullable=False, index=True)
    source_timeframe = Column(String(16), nullable=False, default="H1")
    display_timeframe = Column(String(16), nullable=True)
    source_window = Column(String(16), nullable=True)
    direction = Column(String(16), nullable=False)
    signal_type = Column(String(64), nullable=False)
    entry_time = Column(DateTime(timezone=True), nullable=False, index=True)
    entry_price = Column(Float, nullable=False)
    horizon_bars = Column(Integer, nullable=False, default=24)
    horizon_seconds = Column(Integer, nullable=False, default=86400)
    feature_contract_version = Column(String(64), nullable=False)
    feature_names = Column(JSON, nullable=False)
    feature_values = Column(JSON, nullable=False)
    status = Column(String(16), nullable=False, default="unresolved", index=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    mfe_pips = Column(Float, nullable=True)
    mae_pips = Column(Float, nullable=True)
    net_pips = Column(Float, nullable=True)
    reversal_bar = Column(Integer, nullable=True)
    reward = Column(Float, nullable=True)
    signal_strength = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_signal_outcomes_symbol_entry", "symbol", "entry_time"),
        Index("idx_signal_outcomes_status_entry", "status", "entry_time"),
    )


class LearnerCheckpoint(Base):
    __tablename__ = "learner_checkpoints"

    checkpoint_id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    scope = Column(String(128), nullable=False, default="default", index=True)
    symbol = Column(String(64), nullable=False, default="AAPL", index=True)
    arch_version = Column(String(16), nullable=False, default="v1.0")
    checkpoint_version = Column(String(64), nullable=False, default="v1.0")
    feature_schema_hash = Column(String(64), nullable=False)
    training_counters = Column(JSON, nullable=False)
    eval_metrics = Column(JSON, nullable=True)  # win_rate, avg_reward, mfe_mae_ratio, n_eval_samples
    is_active = Column(Boolean, nullable=False, default=False, index=True)
    notes = Column(Text, nullable=True)
    payload = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False, index=True)

    __table_args__ = (
        Index("idx_learner_checkpoints_scope_created", "scope", "created_at"),
        Index("idx_learner_checkpoints_symbol_active", "symbol", "is_active"),
        Index("idx_learner_checkpoints_scope_symbol", "scope", "symbol"),
    )


class SyntheticMarketSession(Base):
    """Generated OHLCV used to train or replay the meta-learner."""

    __tablename__ = "synthetic_market_sessions"

    session_id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    symbol = Column(String(64), nullable=False, index=True)
    num_candles = Column(Integer, nullable=False)
    seed = Column(Integer, nullable=False, default=42)
    start_price = Column(Float, nullable=False, default=100.0)
    volatility = Column(Float, nullable=False, default=0.02)
    trend = Column(Float, nullable=False, default=0.001)
    interval_seconds = Column(Integer, nullable=False, default=3600)
    candles = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False, index=True)


class MetaLearnerTrainingRun(Base):
    """One synthetic (or later live) training pass, including how much data was used."""

    __tablename__ = "meta_learner_training_runs"

    run_id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    session_id = Column(String(36), nullable=True, index=True)
    scope = Column(String(128), nullable=False, default="synthetic-default", index=True)
    symbol = Column(String(64), nullable=False)
    num_candles = Column(Integer, nullable=False)
    warmup_bars = Column(Integer, nullable=False, default=50)
    horizon_bars = Column(Integer, nullable=False, default=24)
    experiences_recorded = Column(Integer, nullable=False, default=0)
    train_steps = Column(Integer, nullable=False, default=0)
    batch_size = Column(Integer, nullable=False, default=32)
    final_loss = Column(Float, nullable=True)
    metrics = Column(JSON, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False, index=True)
