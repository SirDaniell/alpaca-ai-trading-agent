"""
keras_signal_meta_learner.py — Keras-based Meta-Learner & Dilated CNN Architecture.

Inspired by AXE Genesis / tt.py model design:
- 1D Dilated Conv Multiscale Tower (_multiscale_research_conv_tower)
- Dedicated structural heads for SNR Zone Bounce / Breakout (Support & Resistance)
- SNR Touch Sequence classification heads (snr_touch_1, snr_touch_2)
- Bull / Bear / Action Q-classification heads
- Bounded auxiliary targets with tf.stop_gradient gradient isolation
"""

import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, Any, Tuple, Optional, List

from app.core.ml.signal_meta_learner import (
    SIGNAL_META_FEATURE_COUNT,
    SIGNAL_META_LOOKBACK_BARS,
    TARGET_PIP_SCALE,
    TARGET_ZONE_DIST_SCALE,
    FeatureScaler,
)

logger = logging.getLogger(__name__)

# Ensure TF logging is muted unless warning
tf.get_logger().setLevel('ERROR')


# ── Dilated Conv1D Research Blocks ─────────────────────────────────────────────

def _research_conv_block(x: tf.Tensor, filters: List[int], kernel_size: int = 6, name_prefix: str = "res_conv") -> tf.Tensor:
    """
    Dilated 1D Convolutional block with Batch Normalization, LeakyReLU, and Dropout.
    Applies increasing dilation rates (1, 2, 4...) across consecutive layers to capture wide receptive fields.
    """
    for i, f in enumerate(filters):
        dilation_rate = 2 ** (i % 3)  # 1, 2, 4
        x = layers.Conv1D(
            filters=f,
            kernel_size=kernel_size,
            padding="same",
            dilation_rate=dilation_rate,
            name=f"{name_prefix}_conv{i}",
        )(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_bn{i}")(x)
        x = layers.LeakyReLU(name=f"{name_prefix}_lrelu{i}")(x)
        x = layers.Dropout(0.2, name=f"{name_prefix}_drop{i}")(x)
    return x


def _multiscale_research_conv_tower(
    x: tf.Tensor,
    seq_len: int,
    name_prefix: str = "multiscale_tower",
    filters: Tuple[int, ...] = (128, 128, 64, 64, 64, 64),
    kernel_size: int = 6,
    fusion_dim: int = 64,
) -> tf.Tensor:
    """
    Multi-Scale Temporal Feature Extractor from tt.py:
    Evaluates input sequences concurrently across 3 temporal scales:
      1. Full sequence (0 to seq_len)
      2. Mid sequence (recent 50% of sequence)
      3. Short sequence (recent 30% of sequence)
    Concatenates GlobalAveragePooling1D across all scales into a dense fusion representation.
    """
    half = seq_len // 2
    recent = max(3, int(seq_len * 0.3))

    # Scale 1: Full sequence
    full_conv = _research_conv_block(x, list(filters), kernel_size=kernel_size, name_prefix=f"{name_prefix}_full")
    full_emb = layers.GlobalAveragePooling1D(name=f"{name_prefix}_full_gap")(full_conv)

    # Scale 2: Mid sequence (recent 50%)
    half_sl = layers.Lambda(lambda t: t[:, half:, :], name=f"{name_prefix}_sl_half")(x)
    half_conv = _research_conv_block(half_sl, list(filters), kernel_size=kernel_size, name_prefix=f"{name_prefix}_half")
    half_emb = layers.GlobalAveragePooling1D(name=f"{name_prefix}_half_gap")(half_conv)

    # Scale 3: Short sequence (recent 30%)
    rec_sl = layers.Lambda(lambda t: t[:, -recent:, :], name=f"{name_prefix}_sl_rec")(x)
    rec_conv = _research_conv_block(rec_sl, list(filters), kernel_size=kernel_size, name_prefix=f"{name_prefix}_rec")
    rec_emb = layers.GlobalAveragePooling1D(name=f"{name_prefix}_rec_gap")(rec_conv)

    # Fuse multi-resolution embeddings
    fused = layers.Concatenate(name=f"{name_prefix}_concat")([full_emb, half_emb, rec_emb])
    fused = layers.Dense(fusion_dim, activation="relu", name=f"{name_prefix}_fd")(fused)
    fused = layers.BatchNormalization(name=f"{name_prefix}_fbn")(fused)
    fused = layers.Dropout(0.2, name=f"{name_prefix}_fdrop")(fused)
    return fused


# ── Structural SNR Zone Bounce / Breakout Heads ────────────────────────────────

def build_bounce_breakout_heads(x: tf.Tensor, seq_len: int) -> Dict[str, tf.Tensor]:
    """
    Dedicated binary classification heads for key SNR zone reactions:
    - Signal_bounce_support
    - Signal_bounce_resistance
    - Signal_breakout_support
    - Signal_breakout_resistance
    """
    outputs = {}
    for name in ["Signal_bounce_support", "Signal_bounce_resistance", "Signal_breakout_support", "Signal_breakout_resistance"]:
        cx = _research_conv_block(x, [128, 128, 64, 64], kernel_size=6, name_prefix=f"{name}_tower")
        cx = layers.GlobalAveragePooling1D(name=f"{name}_gap")(cx)
        fc = layers.Dense(32, activation="relu", name=f"{name}_fc")(cx)
        fc = layers.BatchNormalization(name=f"{name}_bn")(fc)
        fc = layers.Dropout(0.2, name=f"{name}_drop")(fc)
        outputs[name] = layers.Dense(1, activation="sigmoid", name=name)(fc)
    return outputs


def build_snr_sequence_heads(x: tf.Tensor, seq_len: int) -> Dict[str, tf.Tensor]:
    """
    Two-head ordered SNR zone sequence classifier from tt.py:
    Predicts ordered pair (snr_touch_1, snr_touch_2) over SNR zones:
      0 = Support Touch, 1 = Resistance Touch, 2 = None / No-Touch
    """
    outputs = {}
    for name in ["snr_touch_1", "snr_touch_2"]:
        fused = _multiscale_research_conv_tower(x, seq_len, name_prefix=f"{name}_tower")
        outputs[name] = layers.Dense(3, activation="softmax", name=name)(fused)
    return outputs


# ── Keras Meta Network Architecture ──────────────────────────────────────────

def build_keras_meta_network(
    num_features: int = 238,
    lookback_bars: int = SIGNAL_META_LOOKBACK_BARS,
    num_actions: int = 5,
) -> keras.Model:
    """
    Builds the Keras Multiscale Dilated CNN Meta-Learner Model.
    
    Inputs:
        sequence_input: (batch_size, lookback_bars, num_features) 3D tensor
    
    Outputs:
        - q_values: (batch_size, 5) Q-values for action policy
        - bull_class: (batch_size, 3) Softmax (0=BEAR, 1=WAIT, 2=BULL)
        - bounce_breakout heads: 4 binary classification probabilities
        - snr_touch_1, snr_touch_2: 3-class softmax touch sequence predictions
        - pips_pred: (batch_size, 1) Continuous expected pips
        - risk_pred: (batch_size, 1) Continuous risk ratio
        - reversal_prob: (batch_size, 1) Continuous reversal probability
    """
    inputs = keras.Input(shape=(lookback_bars, num_features), name="sequence_input")

    # 1. Main Multiscale Dilated CNN Backbone
    shared_features = _multiscale_research_conv_tower(
        inputs,
        seq_len=lookback_bars,
        name_prefix="meta_backbone",
        filters=(128, 128, 64, 64, 64, 64),
        kernel_size=6,
        fusion_dim=128,
    )

    outputs = {}

    # 2. Primary Policy & Classification Heads (Trained on Backbone)
    q_dense = layers.Dense(64, activation="relu", name="q_fc")(shared_features)
    outputs["q_values"] = layers.Dense(num_actions, activation=None, name="q_values")(q_dense)

    bull_class_dense = layers.Dense(64, activation="relu", name="bull_class_fc")(shared_features)
    outputs["bull_class"] = layers.Dense(3, activation="softmax", name="bull_class")(bull_class_dense)

    # 3. Structural SNR Zone Reaction Heads
    outputs.update(build_bounce_breakout_heads(inputs, lookback_bars))
    outputs.update(build_snr_sequence_heads(inputs, lookback_bars))

    # 4. Auxiliary Heads with Gradient Isolation (tf.stop_gradient)
    # Stop gradients so auxiliary task losses do NOT corrupt shared backbone features
    isolated_features = layers.Lambda(lambda f: tf.stop_gradient(f), name="gradient_isolation")(shared_features)

    # Pips Head
    pips_fc = layers.Dense(32, activation="relu", name="pips_fc")(isolated_features)
    outputs["pips_pred"] = layers.Dense(1, activation="relu", name="pips_pred")(pips_fc)

    # Risk Head
    risk_fc = layers.Dense(32, activation="relu", name="risk_fc")(isolated_features)
    outputs["risk_pred"] = layers.Dense(1, activation="sigmoid", name="risk_pred")(risk_fc)

    # Reversal Prob Head
    rev_fc = layers.Dense(32, activation="relu", name="rev_fc")(isolated_features)
    outputs["reversal_prob"] = layers.Dense(1, activation="sigmoid", name="reversal_prob")(rev_fc)

    model = keras.Model(inputs=inputs, outputs=outputs, name="KerasSignalMetaNetwork")
    return model


# ── Keras Meta Learner Wrapper ────────────────────────────────────────────────

class KerasOnlineSignalMetaLearner:
    """
    High-level API Wrapper maintaining parity with OnlineSignalMetaLearner.
    Uses Keras 1D Dilated Conv models for signal classification and Q-learning.
    """

    def __init__(
        self,
        num_features: int = 238,
        lookback_bars: int = SIGNAL_META_LOOKBACK_BARS,
        replay_capacity: int = 20000,
        learning_rate: float = 3e-4,
    ):
        self.num_features = num_features
        self.lookback_bars = lookback_bars
        self.replay_capacity = replay_capacity
        self.scaler = FeatureScaler()

        # Build Keras Model
        self.model = build_keras_meta_network(
            num_features=num_features,
            lookback_bars=lookback_bars,
            num_actions=5,
        )

        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        # Buffer arrays
        self.buffer_x: List[np.ndarray] = []
        self.buffer_q_targets: List[np.ndarray] = []
        self.buffer_pips: List[float] = []
        self.buffer_risk: List[float] = []
        self.buffer_direction: List[int] = []

    def fit_scaler(self, X: np.ndarray) -> None:
        """Fit scaler on feature matrices."""
        self.scaler.fit(X)

    def _prepare_input_tensor(self, feature_dict_or_array: Any) -> np.ndarray:
        """Helper to convert feature input to (1, lookback_bars, num_features)."""
        if isinstance(feature_dict_or_array, dict):
            # Extract features into array
            feats = np.array(list(feature_dict_or_array.values()), dtype=np.float32)
        else:
            feats = np.asarray(feature_dict_or_array, dtype=np.float32)

        if feats.ndim == 1:
            # Flat vector -> reshape to (lookback_bars, num_features)
            if len(feats) == self.lookback_bars * self.num_features:
                feats = feats.reshape(self.lookback_bars, self.num_features)
            else:
                # Pad or truncate to expected size
                target_len = self.lookback_bars * self.num_features
                if len(feats) < target_len:
                    feats = np.pad(feats, (0, target_len - len(feats)))
                else:
                    feats = feats[:target_len]
                feats = feats.reshape(self.lookback_bars, self.num_features)

        elif feats.ndim == 2 and feats.shape != (self.lookback_bars, self.num_features):
            feats = np.resize(feats, (self.lookback_bars, self.num_features))

        return np.expand_dims(feats, axis=0)

    def record_experience(
        self,
        feature_dict: Dict[str, float],
        signal_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        future_highs: np.ndarray,
        future_lows: np.ndarray,
        future_closes: np.ndarray,
    ) -> None:
        """Record trade experience into replay buffer."""
        x_tensor = self._prepare_input_tensor(feature_dict)[0]  # (lookback, num_features)

        dir_idx = 2 if direction == "bullish" else (0 if direction == "bearish" else 1)
        pips = float((np.max(future_highs) - entry_price) * TARGET_PIP_SCALE) if direction == "bullish" else float((entry_price - np.min(future_lows)) * TARGET_PIP_SCALE)
        risk = float(np.std(future_closes) / max(1e-4, entry_price))

        q_targets = np.zeros(5, dtype=np.float32)
        if direction == "bullish":
            q_targets[1] = 1.0  # BUY_CALL
        elif direction == "bearish":
            q_targets[2] = 1.0  # BUY_PUT
        else:
            q_targets[0] = 1.0  # WAIT

        self.buffer_x.append(x_tensor)
        self.buffer_q_targets.append(q_targets)
        self.buffer_pips.append(pips)
        self.buffer_risk.append(risk)
        self.buffer_direction.append(dir_idx)

        if len(self.buffer_x) > self.replay_capacity:
            self.buffer_x.pop(0)
            self.buffer_q_targets.pop(0)
            self.buffer_pips.pop(0)
            self.buffer_risk.pop(0)
            self.buffer_direction.pop(0)

    def train_step(self, batch_size: int = 64) -> Dict[str, float]:
        """Perform one Keras gradient step over sampled replay batch."""
        if len(self.buffer_x) < batch_size:
            return {"loss": 0.0, "buffer_size": len(self.buffer_x)}

        indices = np.random.choice(len(self.buffer_x), size=batch_size, replace=False)
        bx = np.array([self.buffer_x[i] for i in indices], dtype=np.float32)
        bq = np.array([self.buffer_q_targets[i] for i in indices], dtype=np.float32)
        bpips = np.array([self.buffer_pips[i] for i in indices], dtype=np.float32).reshape(-1, 1)
        brisk = np.array([self.buffer_risk[i] for i in indices], dtype=np.float32).reshape(-1, 1)
        bdir = np.array([self.buffer_direction[i] for i in indices], dtype=np.int32)

        with tf.GradientTape() as tape:
            preds = self.model(bx, training=True)

            loss_q = tf.reduce_mean(tf.keras.losses.mse(bq, preds["q_values"]))
            loss_bull = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(bdir, preds["bull_class"]))
            loss_pips = tf.reduce_mean(tf.keras.losses.huber(bpips, preds["pips_pred"]))
            loss_risk = tf.reduce_mean(tf.keras.losses.mse(brisk, preds["risk_pred"]))

            total_loss = loss_q + 0.5 * loss_bull + 0.2 * loss_pips + 0.2 * loss_risk

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return {
            "loss": float(total_loss.numpy()),
            "loss_q": float(loss_q.numpy()),
            "loss_strength": float(loss_bull.numpy()),
            "loss_pips": float(loss_pips.numpy()),
            "loss_risk": float(loss_risk.numpy()),
            "buffer_size": len(self.buffer_x),
        }

    def predict(self, feature_dict_or_array: Any) -> Dict[str, Any]:
        """Predict model outputs for live evaluation/inference."""
        bx = self._prepare_input_tensor(feature_dict_or_array)
        preds = self.model(bx, training=False)

        q_vals = preds["q_values"].numpy()[0]
        bull_probs = preds["bull_class"].numpy()[0]
        pips = float(preds["pips_pred"].numpy()[0, 0])
        risk = float(preds["risk_pred"].numpy()[0, 0])

        bounce_supp = float(preds.get("Signal_bounce_support", tf.zeros((1, 1))).numpy()[0, 0])
        bounce_res = float(preds.get("Signal_bounce_resistance", tf.zeros((1, 1))).numpy()[0, 0])

        return {
            "q_values": q_vals.tolist(),
            "signal_strength": float(bull_probs[2]),  # Bullish prob
            "pips_expected": pips,
            "risk_estimate": risk,
            "bounce_support_prob": bounce_supp,
            "bounce_resistance_prob": bounce_res,
        }
