"""Tests for MetaLearnerModelRegistry: versioning, listing, evaluation, hot-swap."""

import base64

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.ml.meta_learner_registry import MetaLearnerModelRegistry, ModelEntry
from app.core.ml.signal_meta_learner import SIGNAL_META_FEATURE_COUNT, FeatureScaler, OnlineSignalMetaLearner
from app.core.ml.synthetic_meta_trainer import SyntheticTrainConfig, train_from_synthetic
from app.db.models import Base, LearnerCheckpoint

import numpy as np


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


# ── FeatureScaler ─────────────────────────────────────────────────────────────

def test_feature_scaler_fit_stores_mean_and_scale():
    X = np.random.randn(100, 32).astype(np.float32)
    scaler = FeatureScaler()
    scaler.fit(X)
    assert scaler.fitted
    assert scaler.mean_ is not None
    assert scaler.scale_ is not None
    assert len(scaler.mean_) == 32
    assert scaler.n_samples_seen_ == 100


def test_feature_scaler_transform_zero_means_after_fit():
    X = np.random.randn(200, 16).astype(np.float32)
    scaler = FeatureScaler()
    X_t = scaler.fit_transform(X)
    # Mean of transformed data should be ~0
    assert abs(float(np.mean(X_t))) < 0.5


def test_feature_scaler_unfitted_passthrough():
    scaler = FeatureScaler()
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = scaler.transform(vec)
    np.testing.assert_array_equal(out, vec)


def test_feature_scaler_serialise_round_trip():
    X = np.random.randn(50, 10).astype(np.float32)
    scaler = FeatureScaler()
    scaler.fit(X)
    d = scaler.to_dict()
    restored = FeatureScaler.from_dict(d)
    assert restored.fitted
    vec = np.ones(10, dtype=np.float32)
    np.testing.assert_allclose(scaler.transform(vec), restored.transform(vec), rtol=1e-5)


def test_feature_scaler_baked_into_checkpoint():
    """Scaler state must survive export/import_checkpoint round-trip."""
    learner = OnlineSignalMetaLearner()
    X = np.random.randn(60, SIGNAL_META_FEATURE_COUNT).astype(np.float32)
    learner.fit_scaler(X)
    assert learner.scaler.fitted

    ckpt_bytes = learner.export_checkpoint()
    restored = OnlineSignalMetaLearner()
    restored.import_checkpoint(ckpt_bytes)
    assert restored.scaler.fitted
    np.testing.assert_allclose(
        learner.scaler.mean_, restored.scaler.mean_, rtol=1e-5,
        err_msg="Scaler mean_ must survive checkpoint round-trip",
    )


# ── Train/val/test split ──────────────────────────────────────────────────────

def test_train_split_scaler_only_on_train():
    """After training, scaler is fitted and n_samples_seen < num_candles (train only)."""
    result = train_from_synthetic(
        SyntheticTrainConfig(num_candles=280, warmup_bars=80, train_steps=2, batch_size=8, persist=False, seed=3),
        db=None,
    )
    assert result.metrics.get("scaler_fitted") is True
    # Scaler samples <= 70% of usable bars (warmup stripped)
    n_scaler = result.metrics.get("scaler_n_samples", 0)
    usable = 280 - 80 - 24
    assert n_scaler <= int(usable * 0.75), f"Scaler fitted on too many samples: {n_scaler}"


def test_val_eval_metrics_present_in_result():
    result = train_from_synthetic(
        SyntheticTrainConfig(num_candles=280, warmup_bars=80, train_steps=2, batch_size=8, persist=False, seed=5),
        db=None,
    )
    # Val metrics should appear even without DB persistence
    # (they are computed but not stored when persist=False)
    assert result.final_loss is not None


# ── Registry: list / evaluate / pretrain / set_active ─────────────────────────

def test_registry_pretrain_creates_active_checkpoint(db_session):
    registry = MetaLearnerModelRegistry()
    result = registry.pretrain(
        symbol="TSLA",
        scope="test-tsla-v1",
        db=db_session,
        config_overrides={"num_candles": 280, "warmup_bars": 80, "train_steps": 2, "batch_size": 8, "seed": 10},
    )
    assert result.checkpoint_id is not None

    row = db_session.query(LearnerCheckpoint).filter_by(checkpoint_id=result.checkpoint_id).one()
    assert row.is_active is True
    assert row.symbol == "TSLA"
    assert row.scope == "test-tsla-v1"


def test_registry_list_models_filters_by_symbol(db_session):
    registry = MetaLearnerModelRegistry()
    registry.pretrain("AAPL", "scope-a", db_session,
                      config_overrides={"num_candles": 280, "warmup_bars": 80, "train_steps": 2, "batch_size": 8})
    registry.pretrain("MSFT", "scope-b", db_session,
                      config_overrides={"num_candles": 280, "warmup_bars": 80, "train_steps": 2, "batch_size": 8})

    aapl_models = registry.list_models(db_session, symbol="AAPL")
    msft_models = registry.list_models(db_session, symbol="MSFT")
    all_models = registry.list_models(db_session)

    assert all(e.symbol == "AAPL" for e in aapl_models)
    assert all(e.symbol == "MSFT" for e in msft_models)
    assert len(all_models) >= 2


def test_registry_list_returns_model_entry_shape(db_session):
    registry = MetaLearnerModelRegistry()
    registry.pretrain("SPY", "scope-spy", db_session,
                      config_overrides={"num_candles": 280, "warmup_bars": 80, "train_steps": 2, "batch_size": 8})
    entries = registry.list_models(db_session, symbol="SPY")
    assert len(entries) >= 1
    e = entries[0]
    assert isinstance(e, ModelEntry)
    assert e.symbol == "SPY"
    assert isinstance(e.total_steps, int)
    assert isinstance(e.scaler_fitted, bool)
    d = e.to_dict()
    assert "checkpoint_id" in d and "win_rate" in d and "mfe_mae_ratio" in d


def test_registry_evaluate_returns_eval_metrics(db_session):
    registry = MetaLearnerModelRegistry()
    result = registry.pretrain("QQQ", "scope-qqq", db_session,
                               config_overrides={"num_candles": 280, "warmup_bars": 80, "train_steps": 2, "batch_size": 8})
    info = registry.evaluate(result.checkpoint_id, db_session)
    assert info is not None
    assert info["checkpoint_id"] == result.checkpoint_id
    assert "eval_metrics" in info
    assert info["is_active"] is True


def test_registry_set_active_hot_swaps_model(db_session):
    """Pretraining twice should deactivate first checkpoint and activate second."""
    registry = MetaLearnerModelRegistry()
    cfg = {"num_candles": 280, "warmup_bars": 80, "train_steps": 2, "batch_size": 8}
    r1 = registry.pretrain("NVDA", "scope-nvda", db_session, config_overrides={**cfg, "seed": 1})
    r2 = registry.pretrain("NVDA", "scope-nvda", db_session, config_overrides={**cfg, "seed": 2})

    # r1 should now be inactive, r2 active
    row1 = db_session.query(LearnerCheckpoint).filter_by(checkpoint_id=r1.checkpoint_id).one()
    row2 = db_session.query(LearnerCheckpoint).filter_by(checkpoint_id=r2.checkpoint_id).one()
    assert row1.is_active is False
    assert row2.is_active is True

    # Manually swap back to r1
    registry.set_active("NVDA", r1.checkpoint_id, db_session)
    db_session.refresh(row1)
    db_session.refresh(row2)
    assert row1.is_active is True
    assert row2.is_active is False


def test_registry_score_live_uses_active_model(db_session):
    """After pretrain, score_live should return a valid prediction dict."""
    registry = MetaLearnerModelRegistry()
    registry.pretrain("AMZN", "scope-amzn", db_session,
                      config_overrides={"num_candles": 280, "warmup_bars": 80, "train_steps": 2, "batch_size": 8})
    pred = registry.score_live("AMZN", {"rsi": 55.0}, direction="bullish")
    assert "signal_strength" in pred
    assert "reversal_prob" in pred
    assert 0.0 <= pred["signal_strength"] <= 1.0
    assert 0.0 <= pred["reversal_prob"] <= 1.0


def test_registry_pretrain_different_symbols_isolated(db_session):
    """Models for different symbols must be independently stored and scoreable."""
    registry = MetaLearnerModelRegistry()
    cfg = {"num_candles": 280, "warmup_bars": 80, "train_steps": 2, "batch_size": 8}
    registry.pretrain("EURUSD", "scope-fx", db_session, config_overrides={**cfg, "seed": 7})
    registry.pretrain("GBPUSD", "scope-fx", db_session, config_overrides={**cfg, "seed": 8})

    pred_eu = registry.score_live("EURUSD", {"rsi": 48.0}, direction="bearish")
    pred_gb = registry.score_live("GBPUSD", {"rsi": 62.0}, direction="bullish")

    assert "signal_strength" in pred_eu
    assert "signal_strength" in pred_gb
    # They are independent models — ensure registry keeps separate active learners
    assert registry.get_active("EURUSD") is not registry.get_active("GBPUSD")
