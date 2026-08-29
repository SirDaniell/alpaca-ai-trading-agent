import math

import numpy as np
import pandas as pd
import pytest

from app.core.market.mtf_rsi import (
    MTF_RSI_TIMEFRAMES,
    calculate_mtf_rsi,
    calculate_wilder_rsi,
    detect_mtf_rsi_cross_signals,
)
from app.core.market.divergence_scale import build_unified_divergence_scale, rolling_pct_change, rolling_z_score
from app.core.market.signal_events import build_signal_bundle
from app.core.ml.meta_learner import ForwardMoveRewardCalculator
from app.db.models import LearnerCheckpoint, SignalOutcome


@pytest.fixture
def sample_candles():
    closes = [100, 101, 102, 101, 103, 104, 103, 105, 106, 107, 106, 108]
    return [
        {"time": i, "open": c - 1, "high": c + 1, "low": c - 2, "close": c, "volume": 1000}
        for i, c in enumerate(closes)
    ]


def test_calculate_wilder_rsi_has_valid_range(sample_candles):
    rsi = calculate_wilder_rsi(sample_candles, period=7)
    assert len(rsi) == len(sample_candles)
    assert all(point["value"] is None or 0 <= point["value"] <= 100 for point in rsi)


def test_calculate_mtf_rsi_returns_timeframes_and_weighted_rsi(sample_candles):
    tf_data = {tf: sample_candles for tf in MTF_RSI_TIMEFRAMES}
    result = calculate_mtf_rsi("H1", tf_data)
    assert result["timeframe"] == "H1"
    assert "weighted_rsi" in result
    assert len(result["weighted_rsi"]) == len(sample_candles)


def test_detect_mtf_rsi_cross_signals_identifies_crossovers(sample_candles):
    index_series = [{"time": i, "value": 50 + i * 0.8} for i in range(len(sample_candles))]
    dxy_series = [{"time": i, "value": 50 + i * 0.4} for i in range(len(sample_candles))]
    signals = detect_mtf_rsi_cross_signals(index_series, dxy_series)
    assert isinstance(signals, list)
    assert all("type" in s for s in signals)


def test_divergence_scale_returns_normalized_series(sample_candles):
    asset = [c["close"] for c in sample_candles]
    idx = [max(50.0, c["close"] - 10) for c in sample_candles]
    scale = build_unified_divergence_scale(asset, {"Dollar": idx}, ["Dollar"], rolling_window=5)
    assert scale["length"] > 0
    assert len(scale["asset_norm"]) == len(asset)
    assert len(scale["indices_norm"]["Dollar"]) == len(asset)
    assert all(math.isfinite(v) for v in scale["asset_norm"])


def test_signal_bundle_contains_expected_signal_types(sample_candles):
    bundle = build_signal_bundle(sample_candles, sample_candles)
    assert set(bundle.keys()) >= {"signal_map", "signal_count", "signals"}
    assert bundle["signal_count"] >= 2
    assert any(item["type"] in {"bounce_support", "breakout_resistance", "rsi_cross", "rsi_divergence"} for item in bundle["signals"])


def test_reward_calculator_returns_valid_reward_and_strength():
    calc = ForwardMoveRewardCalculator(lookforward_bars=5)
    stats = calc.calculate(
        signal_id="sig-1",
        symbol="AAPL",
        direction="bullish",
        entry_price=100.0,
        future_highs=np.array([101.0, 103.0, 104.0, 105.0, 106.0]),
        future_lows=np.array([99.5, 98.0, 100.0, 101.0, 102.0]),
        future_closes=np.array([101.0, 102.0, 104.0, 105.0, 106.0]),
        atr_pips=10.0,
    )
    assert -1.0 <= stats.reward <= 1.0
    assert 0.0 <= stats.signal_strength <= 1.0
    assert stats.mfe_pips >= 0


def test_signal_and_checkpoint_models_are_defined():
    assert SignalOutcome.__tablename__ == "signal_outcomes"
    assert LearnerCheckpoint.__tablename__ == "learner_checkpoints"


def test_signal_outcome_repository_shape():
    outcome = SignalOutcome(
        signal_id="sig-1",
        symbol="AAPL",
        source_timeframe="H1",
        direction="bullish",
        signal_type="rsi_cross",
        entry_time=pd.Timestamp.utcnow(),
        entry_price=100.0,
        horizon_bars=24,
        horizon_seconds=86400,
        feature_contract_version="signal-meta-features-v1",
        feature_names=["rsi", "trend_alignment"],
        feature_values=[55.0, 0.5],
        status="unresolved",
    )
    assert outcome.signal_id == "sig-1"
    assert outcome.feature_names[0] == "rsi"
