"""
test_keras_models.py — Unit test suite for Keras 1:1 AXE Genesis models.
"""

import pytest
import numpy as np
import tensorflow as tf

from app.core.ml.keras_signal_meta_learner import (
    KerasSignalMetaLearner,
    build_context_signals_model,
)
from app.core.options.keras_trade_executor import (
    KerasTradeExecutor,
    build_classification_ensemble_model,
)
from app.core.options.q_executor import (
    HTFBiasPackage,
    AccountContext,
    ExecutionContext,
)
from app.core.market.zone_snapshot import ZoneSnapshotManager


def test_keras_meta_network_forward_pass():
    """Verify forward pass output shapes and values of build_context_signals_model."""
    batch_size = 8
    lookback_bars = 48
    num_features = 238

    dummy_input = np.random.randn(batch_size, lookback_bars, num_features).astype(np.float32)
    model = build_context_signals_model(
        input_shape=(lookback_bars, num_features),
        continuous_feature_indices=list(range(num_features)),
        structure_indices=list(range(min(20, num_features))),
    )

    outputs = model(dummy_input, training=False)

    assert "q_values" in outputs
    assert outputs["q_values"].shape == (batch_size, 5)

    assert "bull_class" in outputs
    assert outputs["bull_class"].shape == (batch_size, 3)

    assert "Signal_bounce_support" in outputs
    assert outputs["Signal_bounce_support"].shape == (batch_size, 1)

    assert "reversal_prob" in outputs
    assert outputs["reversal_prob"].shape == (batch_size, 1)


def test_keras_meta_learner_training_step():
    """Verify experience recording and training step convergence."""
    learner = KerasSignalMetaLearner(num_features=238, lookback_bars=48, replay_capacity=100)

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

    assert len(learner._buf_x) == 10

    metrics = learner.train_step(batch_size=4)
    assert metrics["loss"] >= 0
    assert "loss_q" in metrics
    assert "loss_strength" in metrics


def test_keras_classification_ensemble_forward_pass():
    batch_size = 4
    seq_len = 1
    num_features = 28
    
    dummy_input = np.random.randn(batch_size, seq_len, num_features).astype(np.float32)
    model = build_classification_ensemble_model(input_shape=(seq_len, num_features), output_dim=5)
    
    outputs = model(dummy_input, training=False)
    
    assert "class_output" in outputs
    assert outputs["class_output"].shape == (batch_size, 5)
    assert "class_aux1" in outputs
    assert outputs["class_aux1"].shape == (batch_size, 5)
    assert "class_aux2" in outputs
    assert outputs["class_aux2"].shape == (batch_size, 5)


def test_keras_trade_executor_training_step():
    """Verify KerasTradeExecutor state construction, action selection, and training."""
    executor = KerasTradeExecutor(seq_len=1, n_features=28)
    bias = HTFBiasPackage(direction="bullish", strength=0.8)
    account = AccountContext(equity=100000.0)
    exec_ctx = ExecutionContext(current_price=100.0, atr=2.0)
    zm = ZoneSnapshotManager()

    state = executor.build_state_vector(bias, account, exec_ctx, zm)
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
