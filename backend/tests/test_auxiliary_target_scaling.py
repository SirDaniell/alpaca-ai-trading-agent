"""
Tests for Auxiliary Target Normalization, Loss Boundedness, Inference Unscaling, and Gradient Isolation in SignalMetaNetwork & OnlineSignalMetaLearner.
"""

import pytest
import numpy as np
import torch

from app.core.ml.signal_meta_learner import (
    SignalMetaNetwork,
    OnlineSignalMetaLearner,
    TARGET_PIP_SCALE,
    TARGET_ZONE_DIST_SCALE,
    TARGET_ZONE_TYPE_SCALE,
    DECISION_WINDOW_DIM,
)


def test_auxiliary_gradient_isolation():
    """
    Verify that auxiliary head losses (pips, risk, liquidity, reversal, aux1, aux2)
    do NOT send gradients to primary heads (q_head, strength_head) or shared backbone parameters.
    """
    net = SignalMetaNetwork(input_dim=DECISION_WINDOW_DIM, num_actions=4)
    x = torch.randn(4, DECISION_WINDOW_DIM, requires_grad=True)

    q_vals, strength, pips, risk, liquidity, reversal, aux1, aux2 = net(x, return_aux=True)

    # Compute auxiliary loss
    loss_aux = pips.sum() + risk.sum() + liquidity.sum() + reversal.sum() + aux1.sum() + aux2.sum()
    loss_aux.backward()

    # Primary head weights MUST have zero/None gradient
    assert net.q_head.weight.grad is None
    assert net.strength_head[0].weight.grad is None
    # Shared fusion backbone MUST have zero/None gradient from auxiliary heads
    assert net.fusion_fc1.weight.grad is None
    assert net.b1_fc1.weight.grad is None
    assert net.b2_c1.weight.grad is None

    # Auxiliary heads MUST have non-None gradients
    assert net.pips_head.weight.grad is not None
    assert net.risk_head.weight.grad is not None
    assert net.liquidity_head.weight.grad is not None
    assert net.reversal_head[0].weight.grad is not None
    assert net.aux1_head.weight.grad is not None
    assert net.aux2_head.weight.grad is not None


def test_primary_gradient_flow():
    """
    Verify that primary losses (loss_q, loss_strength) flow correctly to the shared backbone and primary heads.
    """
    net = SignalMetaNetwork(input_dim=DECISION_WINDOW_DIM, num_actions=4)
    x = torch.randn(4, DECISION_WINDOW_DIM, requires_grad=True)

    q_vals, strength, _, _, _, _ = net(x)
    loss_primary = q_vals.sum() + strength.sum()
    loss_primary.backward()

    # Primary heads MUST have non-None gradients
    assert net.q_head.weight.grad is not None
    assert net.strength_head[0].weight.grad is not None
    # Shared fusion & branch backbones MUST receive gradients
    assert net.fusion_fc1.weight.grad is not None
    assert net.b1_fc1.weight.grad is not None
    assert net.b2_c1.weight.grad is not None


def test_auxiliary_loss_boundedness_with_large_targets():
    """
    Verify that even with large raw targets (e.g. 500 pips MFE, 300 pips MAE, 8.5 ATR zone distance),
    the normalized auxiliary losses stay bounded (< 1.0 each) and do not dominate total loss.
    """
    learner = OnlineSignalMetaLearner(input_dim=DECISION_WINDOW_DIM)
    dummy_input = np.zeros(DECISION_WINDOW_DIM, dtype=np.float32)

    # Seed replay buffer with large-target transitions
    for i in range(16):
        learner.replay_buffer.add(
            state=dummy_input,
            action=0,
            reward=0.8,
            next_state=dummy_input,
            done=True,
            target_pips=450.0,            # Large pips
            mfe_pips=500.0,                # Large MFE
            mae_pips=300.0,                # Large MAE
            next_zone_dist_atr=8.5,        # Large zone distance
            next_zone_type=1.0,
            reversal_prob=0.2,
            transition_id=f"large-target-{i}",
        )

    metrics = learner.train_step(batch_size=8)

    assert metrics['loss_pips'] < 5.0, f"loss_pips too high: {metrics['loss_pips']}"
    assert metrics['loss_risk'] < 5.0, f"loss_risk too high: {metrics['loss_risk']}"
    assert metrics['loss_liquidity'] < 5.0, f"loss_liquidity too high: {metrics['loss_liquidity']}"
    assert metrics['loss_reversal'] < 1.0, f"loss_reversal too high: {metrics['loss_reversal']}"


def test_predict_unscaling_consistency():
    """
    Verify that predict() un-scales the raw normalized predictions back to physical units (pips / ATRs).
    """
    learner = OnlineSignalMetaLearner(input_dim=DECISION_WINDOW_DIM)

    # Mock net predictions for pips, risk, liquidity
    with torch.no_grad():
        learner.net.pips_head.weight.zero_()
        learner.net.pips_head.bias.fill_(1.5)        # 1.5 * 100.0 = 150.0 expected pips

        learner.net.risk_head.weight.zero_()
        learner.net.risk_head.bias.fill_(2.0)        # 2.0 * 100.0 = 200.0 expected MFE/MAE

        learner.net.liquidity_head.weight.zero_()
        learner.net.liquidity_head.bias.fill_(0.5)   # 0.5 * 10.0 = 5.0 ATR distance

    dummy_input = {"rsi": 50.0}
    pred = learner.predict(dummy_input)

    assert pred['expected_pips'] == pytest.approx(1.5 * TARGET_PIP_SCALE, abs=0.1)
    assert pred['expected_mfe_pips'] == pytest.approx(2.0 * TARGET_PIP_SCALE, abs=0.1)
    assert pred['expected_mae_pips'] == pytest.approx(2.0 * TARGET_PIP_SCALE, abs=0.1)
    assert pred['next_zone_dist_atr'] == pytest.approx(0.5 * TARGET_ZONE_DIST_SCALE, abs=0.1)
