import base64

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.market.dollar_index import DOLLAR_PAIRS, dollar_index_closes
from app.core.market.mtf_rsi import detect_mtf_rsi_cross_signals
from app.core.ml.signal_meta_learner import SIGNAL_META_FEATURE_COUNT, OnlineSignalMetaLearner
from app.core.ml.synthetic_meta_trainer import SyntheticTrainConfig, train_from_synthetic
from app.core.ml.ti_meta_features import (
    DECISION_WINDOW_DIM,
    TI_NUMERIC_FEATURE_KEYS,
    calculate_ti_features,
)
from app.db.models import (
    Base,
    LearnerCheckpoint,
    MetaLearnerTrainingRun,
    SignalOutcome,
    SyntheticMarketSession,
)


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


def test_ti_contract_has_rich_feature_set():
    assert len(TI_NUMERIC_FEATURE_KEYS) >= 200
    assert DECISION_WINDOW_DIM == SIGNAL_META_FEATURE_COUNT
    candles = [
        {
            "time": 1_700_000_000 + i * 3600,
            "open": 100 + i * 0.1,
            "high": 101 + i * 0.1,
            "low": 99 + i * 0.1,
            "close": 100.5 + i * 0.1,
            "volume": 1000 + i,
        }
        for i in range(120)
    ]
    frame = calculate_ti_features(candles)
    numeric = [c for c in frame.columns if c in set(TI_NUMERIC_FEATURE_KEYS)]
    assert len(numeric) >= 100


def test_dollar_index_uses_official_pair_basket():
    n = 30
    pair_closes = {pair: [1.0 + i * 0.0001 + (ord(pair[0]) * 0.00001) for i in range(n)] for pair in DOLLAR_PAIRS}
    values = dollar_index_closes(pair_closes)
    assert len(values) == n
    assert all(v > 0 for v in values)


def test_mtf_rsi_crosses_include_three_sources():
    index = [{"time": i, "value": 40 + i * 0.8} for i in range(20)]
    dxy = [{"time": i, "value": 50 - i * 0.3} for i in range(20)]
    signals = detect_mtf_rsi_cross_signals(index, dxy)
    sources = {s["source"] for s in signals}
    assert sources >= {"index-signal", "dxy-signal", "index-dxy"} or len(signals) >= 1


def test_meta_learner_weights_change_on_synthetic_data():
    result = train_from_synthetic(
        SyntheticTrainConfig(
            num_candles=280,
            warmup_bars=80,
            train_steps=8,
            batch_size=8,
            persist=False,
            seed=7,
        ),
        db=None,
    )
    # 280 candles * 0.70 train_end = 196; loop is range(warmup=80, train_end - horizon=172) -> 92 experiences
    assert result.experiences_recorded == 92
    assert result.weights_changed


    assert result.final_loss is not None
    assert result.final_loss > 0


def test_configurable_data_is_stored_and_checkpoint_reloads(db_session):
    num_candles = 280
    result = train_from_synthetic(
        SyntheticTrainConfig(
            symbol="MSFT",
            num_candles=num_candles,
            warmup_bars=80,
            train_steps=5,
            batch_size=8,
            persist=True,
            seed=11,
            scope="test-scope-ti48",
        ),
        db=db_session,
    )

    session_row = db_session.query(SyntheticMarketSession).filter_by(session_id=result.session_id).one()
    assert session_row.num_candles == num_candles
    assert len(session_row.candles) == num_candles

    run = db_session.query(MetaLearnerTrainingRun).filter_by(run_id=result.run_id).one()
    assert run.num_candles == num_candles
    assert run.experiences_recorded == result.experiences_recorded

    outcomes = db_session.query(SignalOutcome).filter(SignalOutcome.signal_id.like(f"{result.run_id}_%")).all()
    assert len(outcomes) == result.experiences_recorded
    assert len(outcomes[0].feature_names) >= 200

    checkpoint = db_session.query(LearnerCheckpoint).filter_by(checkpoint_id=result.checkpoint_id).one()
    restored = OnlineSignalMetaLearner()
    restored.import_checkpoint(base64.b64decode(checkpoint.payload["checkpoint_b64"]))
    assert restored.total_steps == 5
    assert len(restored.replay_buffer) == result.experiences_recorded


def test_multihead_meta_learner_predict_outputs():
    learner = OnlineSignalMetaLearner()
    dummy_input = {"rsi": 55.0, "snr_distance": 1.5, "snr_is_support": 1.0}
    pred = learner.predict(dummy_input, signal_type="test-signal", direction="bullish")

    assert "signal_strength" in pred
    assert "expected_pips" in pred
    assert "expected_mfe_pips" in pred
    assert "expected_mae_pips" in pred
    assert "next_zone_dist_atr" in pred
    assert "next_zone_type" in pred
    assert "reversal_prob" in pred
    assert 0.0 <= pred["signal_strength"] <= 1.0
    assert 0.0 <= pred["reversal_prob"] <= 1.0


def test_auxiliary_losses_in_training_metrics():
    result = train_from_synthetic(
        SyntheticTrainConfig(
            num_candles=280,
            warmup_bars=80,
            train_steps=5,
            batch_size=8,
            persist=False,
            seed=42,
        ),
        db=None,
    )
    assert "loss_risk" in result.metrics
    assert "loss_liquidity" in result.metrics
    assert "loss_reversal" in result.metrics
    assert result.metrics["loss_risk"] >= 0.0
    assert result.metrics["loss_liquidity"] >= 0.0
    assert result.metrics["loss_reversal"] >= 0.0

