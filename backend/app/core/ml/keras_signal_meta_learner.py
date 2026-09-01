"""
keras_signal_meta_learner.py
1:1 port of AXE Genesis (tt.py) architecture for the Meta-Learner pipeline.

Includes exact copies of:
  • FeatureGrouper, SelectContinuousFeatures, StopGradient, GridAttention  (tt.py lines 120-331)
  • _research_conv_block, _multiscale_research_conv_tower               (tt.py lines 459-484)
  • _branch_grid_attention                                               (tt.py lines 497-499)
  • build_bounce_breakout_head  — private 6-block tower per head         (tt.py lines 501-523)
  • build_snr_sequence_head    — private multiscale tower per head       (tt.py lines 525-543)
  • build_full_feature_encoder — private multiscale tower per head       (tt.py lines 586-598)
  • build_recurrent_regression_head — 3-branch GridAttention + StopGrad (tt.py lines 619-707)
  • build_context_signals_model — Sub-model 1/4                         (tt.py lines 1420-1456)
  
Plus KerasOnlineSignalMetaLearner wrapper that maintains API parity with
the existing OnlineSignalMetaLearner (PyTorch).
"""

import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Any, Dict, List, Optional

from app.core.ml.signal_meta_learner import (
    SIGNAL_META_LOOKBACK_BARS,
    TARGET_PIP_SCALE,
    FeatureScaler,
)

logger = logging.getLogger(__name__)
tf.get_logger().setLevel("ERROR")


# ── Custom layers (1:1 from tt.py) ───────────────────────────────────────────

class FeatureGrouper(keras.layers.Layer):
    """tt.py line 120 — TimeDistributed Dense projection to num_groups."""
    def __init__(self, num_groups=64, **kwargs):
        super().__init__(**kwargs)
        self.num_groups = num_groups
        self._grouper = None

    def build(self, input_shape):
        self._grouper = keras.layers.TimeDistributed(
            keras.layers.Dense(self.num_groups, use_bias=False)
        )
        self._grouper.build(input_shape)
        super().build(input_shape)

    def call(self, x):
        return self._grouper(x)

    def get_config(self):
        cfg = super().get_config()
        cfg["num_groups"] = self.num_groups
        return cfg


class SelectContinuousFeatures(keras.layers.Layer):
    """tt.py line 142 — gather a fixed index list along the feature axis."""
    def __init__(self, continuous_indices, **kwargs):
        super().__init__(**kwargs)
        self._idx_list = list(continuous_indices)
        self._idx_tensor = tf.constant(self._idx_list, dtype=tf.int32)

    def call(self, inputs):
        return tf.gather(inputs, self._idx_tensor, axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg["continuous_indices"] = self._idx_list
        return cfg


class StopGradient(keras.layers.Layer):
    """tt.py line 300."""
    def call(self, x):
        return tf.stop_gradient(x)


class GridAttention(keras.layers.Layer):
    """tt.py line 305 — joint time×feature attention."""
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.project    = layers.Dense(units, use_bias=False)
        self.time_score = layers.Dense(1,     use_bias=False)
        self.feat_score = layers.Dense(1,     use_bias=False)

    def build(self, input_shape):
        self.project.build(input_shape)
        self.time_score.build((input_shape[0], input_shape[1], self.units))
        self.feat_score.build((input_shape[0], self.units, input_shape[1]))
        super().build(input_shape)

    def call(self, x):
        h   = tf.nn.tanh(self.project(x))
        t_w = tf.nn.softmax(self.time_score(h), axis=1)
        h_T = tf.transpose(h, [0, 2, 1])
        f_w = tf.transpose(tf.nn.softmax(self.feat_score(h_T), axis=1), [0, 2, 1])
        return tf.reduce_sum(t_w * f_w * h, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_config(self):
        cfg = super().get_config()
        cfg["units"] = self.units
        return cfg


CUSTOM_OBJECTS = {
    "FeatureGrouper": FeatureGrouper,
    "SelectContinuousFeatures": SelectContinuousFeatures,
    "StopGradient": StopGradient,
    "GridAttention": GridAttention,
}


# ── Building-block helpers (1:1 from tt.py) ──────────────────────────────────

def _research_conv_block(x, filters, kernel_size, name_prefix):
    """tt.py line 459 — BN + LeakyReLU + Dropout after each Conv1D."""
    for i, f in enumerate(filters):
        x = layers.Conv1D(f, kernel_size, padding="same",
                          name=f"{name_prefix}_conv{i}")(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_bn{i}")(x)
        x = layers.LeakyReLU(name=f"{name_prefix}_lrelu{i}")(x)
        x = layers.Dropout(0.2, name=f"{name_prefix}_drop{i}")(x)
    return x


def _multiscale_research_conv_tower(tower_slice, seq_len, name_prefix,
                                    filters=(64, 64, 32, 32, 32, 32),
                                    kernel_size=6, fusion_dim=32):
    """tt.py line 467 — full / half / recent slice → GAP → concat → Dense."""
    half   = seq_len // 2
    recent = max(3, int(seq_len * 0.3))

    full_emb = layers.GlobalAveragePooling1D(name=f"{name_prefix}_full_gap")(
        _research_conv_block(tower_slice, filters, kernel_size,
                             f"{name_prefix}_full"))

    half_sl  = layers.Lambda(lambda t: t[:, half:, :],
                             name=f"{name_prefix}_sl_half")(tower_slice)
    half_emb = layers.GlobalAveragePooling1D(name=f"{name_prefix}_half_gap")(
        _research_conv_block(half_sl, filters, kernel_size,
                             f"{name_prefix}_half"))

    rec_sl   = layers.Lambda(lambda t: t[:, -recent:, :],
                             name=f"{name_prefix}_sl_rec")(tower_slice)
    rec_emb  = layers.GlobalAveragePooling1D(name=f"{name_prefix}_rec_gap")(
        _research_conv_block(rec_sl, filters, kernel_size,
                             f"{name_prefix}_rec"))

    fused = layers.Concatenate(name=f"{name_prefix}_concat")(
        [full_emb, half_emb, rec_emb])
    fused = layers.Dense(fusion_dim, activation="relu",
                         name=f"{name_prefix}_fd")(fused)
    fused = layers.BatchNormalization(name=f"{name_prefix}_fbn")(fused)
    fused = layers.Dropout(0.2, name=f"{name_prefix}_fdrop")(fused)
    return fused


def _branch_grid_attention(x_slice, name_prefix, num_groups=32, units=128):
    """tt.py line 497."""
    y = FeatureGrouper(num_groups=num_groups,
                       name=f"{name_prefix}_grouper")(x_slice)
    return GridAttention(units, name=f"{name_prefix}_attention")(y)


# ── Classification head helpers ───────────────────────────────────────────────

def _binary_head(emb, name, fc_dim=32):
    """tt.py line 601."""
    x = layers.Dense(fc_dim, activation="relu", name=f"{name}_fc1")(emb)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.Dropout(0.2, name=f"{name}_drop")(x)
    return layers.Dense(1, activation="sigmoid", name=name)(x)


def _multiclass_head(emb, name, num_classes, fc_dim=32):
    """tt.py line 607."""
    x = layers.Dense(fc_dim, activation="relu", name=f"{name}_fc1")(emb)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.Dropout(0.3, name=f"{name}_drop")(x)
    return layers.Dense(num_classes, activation="softmax", name=name)(x)


# ── Signal heads (1:1 from tt.py) ────────────────────────────────────────────

def build_bounce_breakout_head(inputs, structure_indices, seq_len):
    """
    tt.py line 501 — four binary heads, each with its OWN private 6-block
    Conv1D tower.  No shared trunk → no competing gradients.
    """
    outputs = {}
    for name in ["Signal_bounce_support", "Signal_bounce_resistance",
                 "Signal_breakout_support", "Signal_breakout_resistance"]:
        cx = _research_conv_block(inputs, [128, 128, 64, 64, 64, 64],
                                  kernel_size=6,
                                  name_prefix=f"{name}_tower")
        cx = layers.MaxPooling1D(2, name=f"{name}_tower_pool")(cx)
        cx = layers.Flatten(name=f"{name}_tower_flat")(cx)
        fused = layers.Dense(32, activation="relu",
                             name=f"{name}_tower_fc")(cx)
        fused = layers.BatchNormalization(name=f"{name}_tower_fc_bn")(fused)
        fused = layers.Dropout(0.2, name=f"{name}_tower_fc_drop")(fused)
        outputs[name] = _binary_head(fused, name, fc_dim=32)
    return outputs


def build_snr_sequence_head(inputs, seq_len):
    """
    tt.py line 525 — ordered (snr_touch_1, snr_touch_2) classifiers,
    each with its own private multiscale CNN tower.
    3 classes: R=0 / S=1 / None=2.
    """
    outputs = {}
    for name in ["snr_touch_1", "snr_touch_2"]:
        fused = _multiscale_research_conv_tower(
            inputs, seq_len, name_prefix=f"{name}_tower")
        outputs[name] = layers.Dense(3, activation="softmax", name=name)(fused)
    return outputs


def build_full_feature_encoder(inputs, seq_len, name,
                               filters=None, kernel_size=6, dense_dim=32):
    """tt.py line 586 — private per-head multiscale CNN over all features."""
    if filters is None:
        filters = [128, 128, 64, 64, 64, 64]
    return _multiscale_research_conv_tower(
        inputs, seq_len, name_prefix=f"{name}_tower",
        filters=tuple(filters), kernel_size=kernel_size,
        fusion_dim=dense_dim)


def build_recurrent_regression_head(inputs, continuous_feature_indices, name,
                                    output_dim=1, output_activation=None,
                                    seq_len=48, pass_longterm_context=False):
    """
    tt.py line 619 — 3-branch GridAttention regression head with
    StopGradient-isolated aux outputs fused via LSTM.
    """
    half   = seq_len // 2
    recent = max(3, int(seq_len * 0.3))
    nm     = name

    x_full = SelectContinuousFeatures(continuous_feature_indices,
                                      name=f"{nm}_select")(inputs)

    # Branch 1 — full sequence LSTM stack
    b1 = layers.LSTM(64, return_sequences=True, kernel_initializer="glorot_uniform",
                     recurrent_initializer="orthogonal",
                     name=f"{nm}_b1_lstm1")(x_full)
    b1 = layers.LayerNormalization(name=f"{nm}_b1_ln1")(b1)
    b1 = layers.LSTM(16, return_sequences=True, kernel_initializer="glorot_uniform",
                     recurrent_initializer="orthogonal",
                     name=f"{nm}_b1_lstm2")(b1)
    b1 = layers.LayerNormalization(name=f"{nm}_b1_ln2")(b1)
    b1_3d = b1
    b1    = layers.LSTM(16, return_sequences=False, kernel_initializer="glorot_uniform",
                        recurrent_initializer="orthogonal",
                        name=f"{nm}_b1_lstm3")(b1_3d)

    aux1 = layers.Dense(output_dim, activation=output_activation,
                        kernel_initializer="glorot_uniform",
                        name=f"{nm}_aux1")(b1)
    feat_x_time = _branch_grid_attention(b1_3d, f"{nm}_b1",
                                         num_groups=16, units=32)

    # Branch 2 — mid-term (recent 50 %)
    x_half  = layers.Lambda(lambda t: t[:, half:, :],
                             name=f"{nm}_b2_slice")(inputs)
    b2_proj = layers.TimeDistributed(layers.Dense(32),
                                     name=f"{nm}_b2_proj")(x_half)
    b2_proj = layers.LayerNormalization(name=f"{nm}_b2_ln")(b2_proj)
    b2      = _branch_grid_attention(b2_proj, f"{nm}_b2",
                                     num_groups=12, units=64)
    b2_in   = layers.Concatenate(name=f"{nm}_b2_b1_cat")([b2, b1]) \
              if pass_longterm_context else b2
    aux2    = layers.Dense(output_dim, activation=output_activation,
                           kernel_initializer="glorot_uniform",
                           name=f"{nm}_aux2")(b2_in)

    # Branch 3 — short-term (recent 30 %)
    x_rec   = layers.Lambda(lambda t: t[:, -recent:, :],
                             name=f"{nm}_b3_slice")(inputs)
    b3_proj = layers.TimeDistributed(layers.Dense(32),
                                     name=f"{nm}_b3_proj")(x_rec)
    b3_proj = layers.LayerNormalization(name=f"{nm}_b3_ln")(b3_proj)
    b3      = _branch_grid_attention(b3_proj, f"{nm}_b3",
                                     num_groups=18, units=64)
    b3_in   = layers.Concatenate(name=f"{nm}_b3_b1_cat")([b3, b1]) \
              if pass_longterm_context else b3
    aux3    = layers.Dense(output_dim, activation=output_activation,
                           kernel_initializer="glorot_uniform",
                           name=f"{nm}_aux3")(b3_in)

    # StopGradient isolation before fusion
    h1 = StopGradient(name=f"{nm}_sg1")(aux1)
    h2 = StopGradient(name=f"{nm}_sg2")(aux2)
    h3 = StopGradient(name=f"{nm}_sg3")(aux3)

    # LSTM fusion
    merged = layers.Concatenate(name=f"{nm}_merge")(
        [b1, b2, b3, feat_x_time, h1, h2, h3])
    merged = layers.LayerNormalization(name=f"{nm}_merge_ln")(merged)
    merged = layers.Reshape((1, merged.shape[-1]),
                             name=f"{nm}_fusion_reshape")(merged)
    merged = layers.LSTM(64, return_sequences=False,
                         kernel_initializer="glorot_uniform",
                         recurrent_initializer="orthogonal",
                         name=f"{nm}_fusion")(merged)
    final  = layers.Dense(output_dim, activation=output_activation,
                          name=name)(merged)

    return {name: final,
            f"{name}_aux1": aux1,
            f"{name}_aux2": aux2,
            f"{name}_aux3": aux3}


# ── Sub-model 1/4 (1:1 from tt.py build_context_signals_model) ───────────────

def build_context_signals_model(input_shape,
                                continuous_feature_indices=None,
                                structure_indices=None):
    """
    tt.py line 1420 — Sub-model 1/4.

    Outputs:
        Signal_bounce_support, Signal_bounce_resistance,
        Signal_breakout_support, Signal_breakout_resistance,
        reversal_prob (+ _aux1/2/3),
        trend_continuation_prob (+ _aux1/2/3),
        bull_conf, bear_conf, bull_class,
        reversal_held,
        q_values  ← trading-policy head (not in original, added for this pipeline)
    """
    n_feat = input_shape[-1]
    if continuous_feature_indices is None:
        continuous_feature_indices = list(range(n_feat))
    if structure_indices is None:
        structure_indices = list(range(min(20, n_feat)))

    inputs  = keras.Input(shape=input_shape, name="ctx_input")
    seq_len = input_shape[0]
    outputs: Dict[str, Any] = {}

    # Bounce / breakout heads (private towers)
    outputs.update(build_bounce_breakout_head(inputs, structure_indices,
                                              seq_len))

    # Recurrent regression heads with GridAttention
    for head_name in ["reversal_prob", "trend_continuation_prob"]:
        outputs.update(build_recurrent_regression_head(
            inputs, continuous_feature_indices, head_name,
            output_dim=1, output_activation=None, seq_len=seq_len))

    # Bull / bear confidence + direction class (private CNN encoders)
    bull_conf_emb  = build_full_feature_encoder(inputs, seq_len, "bull_conf_enc")
    bear_conf_emb  = build_full_feature_encoder(inputs, seq_len, "bear_conf_enc")
    bull_class_emb = build_full_feature_encoder(inputs, seq_len, "bull_class_enc")

    outputs["bull_conf"]  = _binary_head(bull_conf_emb,  "bull_conf",  fc_dim=32)
    outputs["bear_conf"]  = _binary_head(bear_conf_emb,  "bear_conf",  fc_dim=32)
    outputs["bull_class"] = _multiclass_head(bull_class_emb, "bull_class",
                                             num_classes=3, fc_dim=32)

    # reversal_held: gated by already-learned reversal_prob (tt.py line 1399)
    rev_prob = outputs["reversal_prob"]
    rev_ctx  = layers.GlobalAveragePooling1D(name="rev_held_ctx_gap")(inputs)
    rev_ctx  = layers.Dense(32, activation="relu",
                             name="rev_held_ctx_fc")(rev_ctx)
    rev_ctx  = layers.LayerNormalization(name="rev_held_ctx_ln")(rev_ctx)
    gate_in  = layers.Concatenate(name="rev_held_gate_in")([rev_prob, rev_ctx])
    gate     = layers.Dense(16, activation="relu",
                             name="rev_held_gate_fc1")(gate_in)
    gate     = layers.Dropout(0.2, name="rev_held_gate_drop")(gate)
    outputs["reversal_held"] = layers.Dense(1, activation="sigmoid",
                                            name="reversal_held")(gate)

    # Q-values head (action policy, added for this pipeline)
    q_emb = build_full_feature_encoder(inputs, seq_len, "q_values_enc")
    outputs["q_values"] = layers.Dense(5, activation=None,
                                       name="q_values")(q_emb)

    return keras.Model(inputs=inputs, outputs=outputs,
                       name="context_signals_model")


# ── High-level wrapper ────────────────────────────────────────────────────────

class KerasSignalMetaLearner:
    """
    Keras-based replacement for OnlineSignalMetaLearner (PyTorch).
    Uses build_context_signals_model — 1:1 AXE Genesis architecture.
    API is intentionally identical to the PyTorch version so the eval
    script can swap between frameworks with a single --framework flag.
    """

    def __init__(self,
                 num_features: int = 238,
                 lookback_bars: int = SIGNAL_META_LOOKBACK_BARS,
                 replay_capacity: int = 20_000,
                 learning_rate: float = 3e-4):
        self.num_features    = num_features
        self.lookback_bars   = lookback_bars
        self.replay_capacity = replay_capacity
        self.scaler          = FeatureScaler()

        self.model = build_context_signals_model(
            input_shape=(lookback_bars, num_features),
            continuous_feature_indices=list(range(num_features)),
            structure_indices=list(range(min(20, num_features))),
        )
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        # Replay buffers
        self._buf_x:    List[np.ndarray] = []
        self._buf_q:    List[np.ndarray] = []
        self._buf_dir:  List[int]        = []
        self._buf_pips: List[float]      = []

    # ── helpers ──────────────────────────────────────────────────────────────

    def _to_tensor(self, feature_dict_or_array) -> np.ndarray:
        if isinstance(feature_dict_or_array, dict):
            arr = np.array(list(feature_dict_or_array.values()), dtype=np.float32)
        else:
            arr = np.asarray(feature_dict_or_array, dtype=np.float32)

        arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)

        target = self.lookback_bars * self.num_features
        if arr.ndim == 1:
            if arr.size != target:
                arr = np.resize(arr, target)
            arr = arr.reshape(self.lookback_bars, self.num_features)
        elif arr.ndim == 2 and arr.shape != (self.lookback_bars, self.num_features):
            arr = np.resize(arr, (self.lookback_bars, self.num_features))

        return np.expand_dims(arr, 0)   # (1, T, F)

    # ── public API ───────────────────────────────────────────────────────────

    def record_experience(self, feature_dict, signal_id, symbol,
                          direction, entry_price,
                          future_highs, future_lows, future_closes):
        x = self._to_tensor(feature_dict)[0]

        dir_idx = 2 if direction == "bullish" else (0 if direction == "bearish" else 1)
        raw_diff = (np.max(future_highs) - entry_price) if direction == "bullish" \
                   else (entry_price - np.min(future_lows))
        pips_norm = float(np.clip(raw_diff / max(1.0, entry_price) * TARGET_PIP_SCALE, 0.0, 2.0))

        q_target = np.zeros(5, dtype=np.float32)
        q_target[1 if direction == "bullish" else (2 if direction == "bearish" else 0)] = 1.0

        self._buf_x.append(x)
        self._buf_q.append(q_target)
        self._buf_dir.append(dir_idx)
        self._buf_pips.append(pips_norm)

        if len(self._buf_x) > self.replay_capacity:
            self._buf_x.pop(0)
            self._buf_q.pop(0)
            self._buf_dir.pop(0)
            self._buf_pips.pop(0)

    def train_step(self, batch_size: int = 64) -> Dict[str, float]:
        if len(self._buf_x) < batch_size:
            return {"loss": 0.0, "buffer_size": len(self._buf_x)}

        idx   = np.random.choice(len(self._buf_x), size=batch_size, replace=False)
        bx    = np.array([self._buf_x[i]    for i in idx], dtype=np.float32)
        bq    = np.array([self._buf_q[i]    for i in idx], dtype=np.float32)
        bdir  = np.array([self._buf_dir[i]  for i in idx], dtype=np.int32)
        bpips = np.array([self._buf_pips[i] for i in idx], dtype=np.float32).reshape(-1, 1)

        # bull label: 1 if bullish (dir==2), else 0
        bull_label = (bdir == 2).astype(np.float32).reshape(-1, 1)
        bear_label = (bdir == 0).astype(np.float32).reshape(-1, 1)

        if not np.all(np.isfinite(bx)):
            return {"loss": 0.0, "loss_q": 0.0, "loss_strength": 0.0, "loss_pips": 0.0, "loss_risk": 0.0, "loss_liquidity": 0.0, "loss_reversal": 0.0, "loss_aux1": 0.0, "loss_aux2": 0.0, "buffer_size": len(self._buf_x)}

        with tf.GradientTape() as tape:
            preds = self.model(bx, training=True)

            # Q-values: MSE vs one-hot direction target
            loss_q = tf.reduce_mean(tf.keras.losses.mse(bq, preds["q_values"]))

            # Direction classification: CE on bull_class (3-class)
            loss_str = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    bdir, preds["bull_class"]))

            # Reversal prob: BCE vs bear_label (bearish = reversal risk)
            loss_rev = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(
                    bear_label, preds["reversal_prob"]))

            # Bull/Bear confidence: BCE
            loss_bull = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(
                    bull_label, preds["bull_conf"]))
            loss_bear = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(
                    bear_label, preds["bear_conf"]))

            # Aux regression heads — reversal + trend continuation (MSE vs pips proxy)
            # All aux outputs are (B,1); bpips is (-1,1) — shapes match.
            loss_aux = tf.add_n([
                tf.reduce_mean(tf.keras.losses.mse(bpips, preds["reversal_prob_aux1"])),
                tf.reduce_mean(tf.keras.losses.mse(bpips, preds["reversal_prob_aux2"])),
                tf.reduce_mean(tf.keras.losses.mse(bpips, preds["reversal_prob_aux3"])),
                tf.reduce_mean(tf.keras.losses.mse(bpips, preds["trend_continuation_prob_aux1"])),
                tf.reduce_mean(tf.keras.losses.mse(bpips, preds["trend_continuation_prob_aux2"])),
                tf.reduce_mean(tf.keras.losses.mse(bpips, preds["trend_continuation_prob_aux3"])),
            ]) / 6.0

            loss = (loss_q
                    + 0.5  * loss_str
                    + 0.3  * loss_rev
                    + 0.2  * loss_bull
                    + 0.2  * loss_bear
                    + 0.15 * loss_aux)

            if not tf.math.is_finite(loss):
                return {"loss": 0.0, "loss_q": 0.0, "loss_strength": 0.0, "loss_pips": 0.0, "loss_risk": 0.0, "loss_liquidity": 0.0, "loss_reversal": 0.0, "loss_aux1": 0.0, "loss_aux2": 0.0, "buffer_size": len(self._buf_x)}

        grads = tape.gradient(loss, self.model.trainable_variables)
        valid_grads_and_vars = [
            (g, v) for g, v in zip(grads, self.model.trainable_variables) if g is not None
        ]
        self.optimizer.apply_gradients(valid_grads_and_vars)

        return {
            "loss":           float(loss.numpy()),
            "loss_q":         float(loss_q.numpy()),
            "loss_strength":  float(loss_str.numpy()),
            "loss_pips":      float(loss_aux.numpy()),
            "loss_risk":      float(loss_aux.numpy()),
            "loss_liquidity": 0.0,
            "loss_reversal":  float(loss_rev.numpy()),
            "loss_aux1":      float(loss_bull.numpy()),
            "loss_aux2":      float(loss_bear.numpy()),
            "buffer_size":    len(self._buf_x),
        }

    def predict(self, feature_dict_or_array) -> Dict[str, Any]:
        bx    = self._to_tensor(feature_dict_or_array)
        preds = self.model(bx, training=False)

        bull_probs  = preds["bull_class"].numpy()[0]
        q_vals      = preds["q_values"].numpy()[0]
        bounce_supp = float(preds["Signal_bounce_support"].numpy()[0, 0])
        bounce_res  = float(preds["Signal_bounce_resistance"].numpy()[0, 0])

        return {
            "q_values":               q_vals.tolist(),
            "signal_strength":        float(bull_probs[2]),
            "pips_expected":          0.0,
            "risk_estimate":          0.0,
            "bounce_support_prob":    bounce_supp,
            "bounce_resistance_prob": bounce_res,
        }


# Alias kept for backward compat with existing imports
KerasOnlineSignalMetaLearner = KerasSignalMetaLearner
