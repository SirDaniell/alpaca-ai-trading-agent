"""
test_keras_models.py — Unit test suite for Keras 1D Dilated Conv Meta-Learner & Q-Executor models.
"""

import pytest
import numpy as np
import tensorflow as tf

from app.core.ml.keras_signal_meta_learner import (
    KerasOnlineSignalMetaLearner,
    build_keras_meta_network,
)
from app.core.options.keras_q_executor import (
    KerasOptionsQExecutor,
    build_keras_executor_q_network,
    HTFBiasPackage,
    AccountContext,
    ExecutionContext,
)
from app.core.market.zone_snapshot import ZoneSnapshotManager


def test_keras_meta_network_forward_pass():
    """Verify forward pass output shapes and values of KerasSignalMetaNetwork."""
    batch_size = 8
    lookback_bars = 48
    num_features = 238

    dummy_input = np.random.randn(batch_size, lookback_bars, num_features).astype(np.float32)
    model = build_keras_meta_network(num_features=num_features, lookback_bars=lookback_bars, num_actions=5)

    outputs = model(dummy_input, training=False)

    assert "q_values" in outputs
    assert outputs["q_values"].shape == (batch_size, 5)

    assert "bull_class" in outputs
    assert outputs["bull_class"].shape == (batch_size, 3)

    assert "Signal_bounce_support" in outputs
    assert outputs["Signal_bounce_support"].shape == (batch_size, 1)

    assert "snr_touch_1" in outputs
    assert outputs["snr_touch_1"].shape == (batch_size, 3)

    assert "pips_pred" in outputs
    assert outputs["pips_pred"].shape == (batch_size, 1)


def test_keras_meta_learner_training_step():
    """Verify experience recording and training step convergence."""
    learner = KerasOnlineSignalMetaLearner(num_features=238, lookback_bars=48, replay_capacity=100)

    # Record experiences
    for i in range(10):
        feature_dict = {f"feat_{j}": float(np.random.randn()) for j in range(238 * 48)}
        fut_highs = np.array([101.0, 102.0, 103.0])
        fut_lows = np.array([99.0, 98.0, 97.0])
        fut_closes = np.array([100.5, 101.5, 102.5])

        learner.record_experience(
            feature_dict=feature_dict,
            signal_id=f"sig_{i}",
            symbol="GLD",
            direction="bullish" if i % 2 == 0 else "bearish",
            entry_price=100.0,
            future_highs=fut_highs,
            future_lows=fut_lows,
            future_closes=fut_closes,
        )

    assert len(learner.buffer_x) == 10

    metrics = learner.train_step(batch_size=4)
    assert metrics["loss"] > 0
    assert "loss_q" in metrics
    assert "loss_pips" in metrics


def test_keras_q_executor_training_step():
    """Verify KerasOptionsQExecutor state construction, action selection, and training."""
    executor = KerasOptionsQExecutor()
    bias = HTFBiasPackage(direction="bullish", strength=0.8)
    account = AccountContext(equity=100000.0)
    exec_ctx = ExecutionContext(current_price=100.0, atr=2.0)
    zm = ZoneSnapshotManager()

    state = executor.construct_state_vector(bias, account, exec_ctx, zm)
    assert len(state) == 28

    mask = np.array([1, 1, 0, 0, 0], dtype=np.int32)
    action = executor.select_action(state, mask, eval_mode=True)
    assert action in [0, 1]

    # Record transition
    next_state = state.copy()
    executor.record_transition(state, action, reward=1.5, next_state=next_state, done=False, next_mask=mask)

    for _ in range(5):
        executor.record_transition(state, action, reward=1.5, next_state=next_state, done=False, next_mask=mask)

    loss = executor.train_step(batch_size=4)
    assert loss >= 0.0
