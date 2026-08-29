"""
custom_keras_objects.py
========================
Single source of truth for every custom Keras class / loss function used by
AXE proprietary models (Genesis v1, Chimera v8_hybrid).

Custom layers required by baseline_v1_best.keras (audit via config.json):
  - CategoryFeatureSlice   [unknown module]  — feature column gatherer
  - SelectContinuousFeatures [baseline_encoder_v7] — continuous feature selector
  - FeatureGrouper           [baseline_encoder_v7] — grouped feature projection
  - GridAttention            [unknown module]  — time + feature axis attention
  - StopGradient             [unknown module]  — gradient blocker for aux heads
  - CandleReconciliation     [unknown module]  — pure-arithmetic candle enforcer

Loss functions serialized into compile config:
  - hw_point_loss_d0.85_h0.01_dw0.2  (sequence heads — Genesis huber_delta=0.01)
  - hw_point_loss_d0.85_h0.05_dw0.2  (sequence heads — Chimera huber_delta=0.05)
  - _loss                              (fallback inner-fn name for closures)

Rules:
  - Every class uses @keras.saving.register_keras_serializable(package="Custom")
    so Keras locates them by name when deserializing .keras archives.
  - Loss closures are NOT registered as classes; they are keyed by __name__ in
    the custom_objects dict passed to load_model().
  - Import this module BEFORE any tf.keras.models.load_model() call.

Usage:
    from app.core.ml.custom_keras_objects import register_all, get_custom_objects
    register_all()
    model = tf.keras.models.load_model(path,
                                        custom_objects=get_custom_objects(),
                                        safe_mode=False)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentinel: track whether classes have been registered already
# ---------------------------------------------------------------------------
_REGISTERED = False


# ---------------------------------------------------------------------------
# Custom Keras layer classes
# ---------------------------------------------------------------------------

def _register_custom_layers() -> Dict[str, Any]:
    """
    Register all custom layer classes once and return a dict of {name: class}.
    Calling multiple times is safe — Keras deduplicates internally.
    """
    import tensorflow as tf
    import keras

    registered: Dict[str, Any] = {}

    # ── CategoryFeatureSlice ─────────────────────────────────────────────────
    # Gathers a fixed subset of feature columns from the raw input tensor.
    # Used by both baseline_v1 and baseline_encoder_v8_hybrid tower routing.
    existing = keras.saving.get_registered_object("Custom>CategoryFeatureSlice")
    if existing is not None:
        registered["CategoryFeatureSlice"] = existing
    else:
        @keras.saving.register_keras_serializable(package="Custom")
        class CategoryFeatureSlice(tf.keras.layers.Layer):
            """Gathers a fixed set of feature columns from the raw input tensor."""

            def __init__(self, indices: List[int], **kwargs: Any) -> None:
                super().__init__(**kwargs)
                self._indices = list(indices)
                self._gather_idx = tf.constant(self._indices, dtype=tf.int32)

            def call(self, x: Any) -> Any:
                return tf.gather(x, self._gather_idx, axis=-1)

            def get_config(self) -> Dict[str, Any]:
                cfg = super().get_config()
                cfg.update({"indices": self._indices})
                return cfg

        registered["CategoryFeatureSlice"] = CategoryFeatureSlice
        logger.debug("[CustomKeras] Registered CategoryFeatureSlice")

    # ── SelectContinuousFeatures ─────────────────────────────────────────────
    # Selects only continuous features by pre-computed index list.
    # Defined in baseline_encoder_v7.py (module="baseline_encoder_v7").
    existing = keras.saving.get_registered_object("Custom>SelectContinuousFeatures")
    if existing is not None:
        registered["SelectContinuousFeatures"] = existing
    else:
        @keras.saving.register_keras_serializable(package="Custom")
        class SelectContinuousFeatures(tf.keras.layers.Layer):
            """Selects only continuous features by index."""

            def __init__(self, continuous_indices: List[int], **kwargs: Any) -> None:
                super().__init__(**kwargs)
                self._continuous_indices = list(continuous_indices)
                self._gather_idx = tf.constant(self._continuous_indices, dtype=tf.int32)

            def call(self, inputs: Any) -> Any:
                return tf.gather(inputs, self._gather_idx, axis=-1)

            def get_config(self) -> Dict[str, Any]:
                cfg = super().get_config()
                cfg.update({"continuous_indices": self._continuous_indices})
                return cfg

        registered["SelectContinuousFeatures"] = SelectContinuousFeatures
        logger.debug("[CustomKeras] Registered SelectContinuousFeatures")

    # ── FeatureGrouper ───────────────────────────────────────────────────────
    # Grouped feature projection via TimeDistributed Dense.
    # Defined in baseline_encoder_v7.py.
    existing = keras.saving.get_registered_object("Custom>FeatureGrouper")
    if existing is not None:
        registered["FeatureGrouper"] = existing
    else:
        @keras.saving.register_keras_serializable(package="Custom")
        class FeatureGrouper(tf.keras.layers.Layer):
            """Projects features into grouped representations."""

            def __init__(self, num_groups: int = 64, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                self.num_groups = num_groups
                self.grouper: Any = None

            def build(self, input_shape: Any) -> None:
                self.grouper = tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Dense(self.num_groups, use_bias=False)
                )
                self.grouper.build(input_shape)
                super().build(input_shape)

            def call(self, x: Any) -> Any:
                return self.grouper(x)

            def get_config(self) -> Dict[str, Any]:
                cfg = super().get_config()
                cfg.update({"num_groups": self.num_groups})
                return cfg

        registered["FeatureGrouper"] = FeatureGrouper
        logger.debug("[CustomKeras] Registered FeatureGrouper")

    # ── StopGradient ─────────────────────────────────────────────────────────
    # Passes values through at inference; blocks gradients during training.
    existing = keras.saving.get_registered_object("Custom>StopGradient")
    if existing is not None:
        registered["StopGradient"] = existing
    else:
        @keras.saving.register_keras_serializable(package="Custom")
        class StopGradient(tf.keras.layers.Layer):
            """Blocks gradients for aux heads; identity at inference."""

            def call(self, x: Any) -> Any:
                return tf.stop_gradient(x)

        registered["StopGradient"] = StopGradient
        logger.debug("[CustomKeras] Registered StopGradient")

    # ── GridAttention ────────────────────────────────────────────────────────
    # Attends over time and feature axes independently.
    existing = keras.saving.get_registered_object("Custom>GridAttention")
    if existing is not None:
        registered["GridAttention"] = existing
    else:
        @keras.saving.register_keras_serializable(package="Custom")
        class GridAttention(tf.keras.layers.Layer):
            """2D attention over time and feature axes. Input (B,T,F) → Output (B,units)."""

            def __init__(self, units: int, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                self.units = units
                self.project = tf.keras.layers.Dense(units, use_bias=False)
                self.time_score = tf.keras.layers.Dense(1, use_bias=False)
                self.feat_score = tf.keras.layers.Dense(1, use_bias=False)

            def call(self, x: Any) -> Any:
                h = tf.nn.tanh(self.project(x))                    # (B, T, units)
                t_w = tf.nn.softmax(self.time_score(h), axis=1)    # (B, T, 1)
                h_T = tf.transpose(h, [0, 2, 1])                   # (B, units, T)
                f_w = tf.nn.softmax(self.feat_score(h_T), axis=1)  # (B, units, 1)
                f_w = tf.transpose(f_w, [0, 2, 1])                 # (B, 1, units)
                return tf.reduce_sum(t_w * f_w * h, axis=1)        # (B, units)

            def get_config(self) -> Dict[str, Any]:
                cfg = super().get_config()
                cfg.update({"units": self.units})
                return cfg

        registered["GridAttention"] = GridAttention
        logger.debug("[CustomKeras] Registered GridAttention")

    # ── CandleReconciliation ─────────────────────────────────────────────────
    # Pure-arithmetic geometric constraint enforcement for OHLCV candle heads.
    # No trainable weights.
    existing = keras.saving.get_registered_object("Custom>CandleReconciliation")
    if existing is not None:
        registered["CandleReconciliation"] = existing
    else:
        @keras.saving.register_keras_serializable(package="Custom")
        class CandleReconciliation(tf.keras.layers.Layer):
            """Enforces high>=open,close and low<=open,close; softplus on volume."""

            def call(self, inputs: Any) -> Any:
                open_s, high_s, low_s, close_s, vol_s = inputs
                high_f = tf.maximum(tf.maximum(high_s, open_s), close_s)
                low_f = tf.minimum(tf.minimum(low_s, open_s), close_s)
                vol_f = tf.nn.softplus(vol_s)
                stacked = tf.stack([open_s, high_f, low_f, close_s, vol_f], axis=-1)
                return open_s, high_f, low_f, close_s, vol_f, stacked

        registered["CandleReconciliation"] = CandleReconciliation
        logger.debug("[CustomKeras] Registered CandleReconciliation")

    return registered


# ---------------------------------------------------------------------------
# Loss factory functions
# ---------------------------------------------------------------------------

def horizon_weighted_point_loss(
    decay: float = 0.85,
    huber_delta: float = 0.01,
    dir_weight: float = 0.20,
) -> Any:
    """Per-point sequence loss: horizon-weighted Huber + direction penalty."""
    import tensorflow as tf

    def _loss(y_true: Any, y_pred: Any) -> Any:
        y_true = tf.cast(y_true, y_pred.dtype)
        horizon = tf.shape(y_pred)[1]
        steps = tf.cast(tf.range(horizon), y_pred.dtype)
        weights = tf.pow(tf.cast(decay, y_pred.dtype), steps)
        weights = weights / tf.reduce_sum(weights)
        err = y_true - y_pred
        abs_err = tf.abs(err)
        huber = tf.where(
            abs_err < huber_delta,
            0.5 * tf.square(err),
            huber_delta * (abs_err - 0.5 * huber_delta),
        )
        hw_loss = tf.reduce_mean(tf.reduce_sum(weights * huber, axis=1))
        delta_true = y_true[:, 1:] - y_true[:, :-1]
        delta_pred = y_pred[:, 1:] - y_pred[:, :-1]
        dir_err = tf.nn.relu(-delta_true * delta_pred)
        dir_loss = tf.reduce_mean(tf.reduce_sum(weights[:-1] * dir_err, axis=1))
        return hw_loss + dir_weight * dir_loss

    _loss.__name__ = f"hw_point_loss_d{decay}_h{huber_delta}_dw{dir_weight}"
    return _loss


def horizon_mae_loss(decay: float = 0.85) -> Any:
    """Per-step MAE with exponential horizon weights (aux hint heads)."""
    import tensorflow as tf

    def _loss(y_true: Any, y_pred: Any) -> Any:
        y_true = tf.cast(y_true, y_pred.dtype)
        horizon = tf.shape(y_pred)[1]
        steps = tf.cast(tf.range(horizon), y_pred.dtype)
        weights = tf.pow(tf.cast(decay, y_pred.dtype), steps)
        weights = weights / tf.reduce_sum(weights)
        abs_err = tf.abs(y_true - y_pred)
        return tf.reduce_mean(tf.reduce_sum(weights * abs_err, axis=1))

    _loss.__name__ = f"horizon_mae_{decay}"
    return _loss


def weighted_binary_crossentropy(pos_weight: float = 1.0) -> Any:
    """Weighted BCE for imbalanced signal classification."""
    import tensorflow as tf
    import keras

    def _loss(y_true: Any, y_pred: Any) -> Any:
        eps = keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        y_true = tf.cast(y_true, y_pred.dtype)
        return -tf.reduce_mean(
            pos_weight * y_true * tf.math.log(y_pred)
            + (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        )

    _loss.__name__ = f"weighted_bce_pw{pos_weight}"
    return _loss


def huber_scalar_loss(delta: float = 1.0) -> Any:
    """Huber loss for scalar regression targets not bounded to [0,1]."""
    import tensorflow as tf

    def _loss(y_true: Any, y_pred: Any) -> Any:
        y_true = tf.cast(y_true, y_pred.dtype)
        err = y_true - y_pred
        abs_err = tf.abs(err)
        return tf.reduce_mean(
            tf.where(
                abs_err < delta,
                0.5 * tf.square(err),
                delta * (abs_err - 0.5 * delta),
            )
        )

    _loss.__name__ = f"huber_scalar_d{delta}"
    return _loss


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_all() -> None:
    """
    Register all custom Keras objects.
    Idempotent — safe to call multiple times.
    Must be called before any load_model() invocation on AXE models.
    """
    global _REGISTERED
    _register_custom_layers()
    _REGISTERED = True
    logger.debug("[CustomKeras] All custom layers registered")


def get_custom_objects() -> Dict[str, Any]:
    """
    Return a custom_objects dict for tf.keras.models.load_model().

    Covers all loss names serialized into AXE Genesis v1 compile config:
      - hw_point_loss_d0.85_h0.01_dw0.2   (sequence heads)
      - hw_point_loss_d0.85_h0.05_dw0.2   (Chimera variant)
      - _loss                               (fallback inner-fn name)
    Plus all custom layer classes.
    """
    layer_classes = _register_custom_layers()

    # Pre-instantiate loss callables at their exact serialized __name__ keys
    hw_genesis = horizon_weighted_point_loss(decay=0.85, huber_delta=0.01, dir_weight=0.20)
    hw_chimera = horizon_weighted_point_loss(decay=0.85, huber_delta=0.05, dir_weight=0.20)
    hmae = horizon_mae_loss(decay=0.85)
    wbce_1 = weighted_binary_crossentropy(pos_weight=1.0)
    wbce_4 = weighted_binary_crossentropy(pos_weight=4.0)
    huber_1 = huber_scalar_loss(delta=1.0)

    # Fallback for closures serialized with bare name "_loss"
    _loss_fallback = horizon_weighted_point_loss(decay=0.85, huber_delta=0.01, dir_weight=0.20)
    _loss_fallback.__name__ = "_loss"

    objs: Dict[str, Any] = {}
    objs.update(layer_classes)
    objs.update({
        hw_genesis.__name__: hw_genesis,
        hw_chimera.__name__: hw_chimera,
        hmae.__name__: hmae,
        wbce_1.__name__: wbce_1,
        wbce_4.__name__: wbce_4,
        huber_1.__name__: huber_1,
        "_loss": _loss_fallback,
    })
    return objs
