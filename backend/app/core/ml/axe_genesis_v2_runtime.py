"""
AXE Genesis V2 Unified Runtime
══════════════════════════════════════════════════════════════════════════════
Orchestrates the active AXE Genesis V2 inference stack into a single runtime.

Active sub-models:
  - 5 Series Models (Open, High, Low, Close, Volume)
  - 1 Volume CNN Complement Model (optional, when saved)
  - 1 Context Model (Signals, Zones, Trade Quality, Probes)
  - 1 Classification Ensemble Model (4-class signal type head: CNN + DilatedCNN + Dense)

Inference Pipeline:
  1. Series Models: Predict raw 12-step OHLCV sequence forecasts from 90-step input sequence.
  2. Volume Complement: Blend the standard volume model with the saved volume-CNN alternative when present.
  3. Reconciliation: Enforce strict candle geometry constraints (High >= Open/Close, Low <= Open/Close).
  4. Context Model: Compute signal probabilities, liquidity zone distributions, and trade quality indicators.
  5. Classification Ensemble: Predict signal type via 3-branch CNN/Dilated/Dense fused ensemble.
     Output: 4-class softmax — 0=bounce_support, 1=bounce_resistance, 2=breakout_support,
                                3=breakout_resistance
"""

import os
import json
import logging
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger(__name__)

# ── Paths & Constants ─────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[3]
_DEFAULT_CHECKPOINT_DIR = _PROJECT_ROOT / "Backend" / "Backend" / "axe-genesis"

from app.core.ml.inference_feature_pipeline import resolve_dataset_cache_dir
_DEFAULT_DATASET_DIR = resolve_dataset_cache_dir("ml_raw_20260809_410")

SEQ_LEN = 90
N_FEATURES = 663
FORECAST_STEPS = 12

# ── Classification Ensemble Constants ─────────────────────────────────────────
# 4-class signal head: must match the notebook Cell 10 / ENS_SIGNAL_KEYS ordering.
N_SIGNAL_CLASSES = 4
SIGNAL_CLASS_NAMES = [
    "bounce_support",      # 0
    "bounce_resistance",   # 1
    "breakout_support",    # 2
    "breakout_resistance", # 3
]


# ── Custom Keras Layers ───────────────────────────────────────────────────────
class StopGradient(layers.Layer):
    """Layer that stops gradient propagation during backprop."""
    def call(self, x):
        return tf.stop_gradient(x)


class GridAttention(layers.Layer):
    """2D Grid attention mechanism over time and feature dimensions."""
    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.project = layers.Dense(units, use_bias=False)
        self.time_score = layers.Dense(1, use_bias=False)
        self.feat_score = layers.Dense(1, use_bias=False)

    def call(self, x):
        h = tf.nn.tanh(self.project(x))
        t_w = tf.nn.softmax(self.time_score(h), axis=1)
        h_T = tf.transpose(h, [0, 2, 1])
        f_w = tf.transpose(tf.nn.softmax(self.feat_score(h_T), axis=1), [0, 2, 1])
        return tf.reduce_sum(t_w * f_w * h, axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg["units"] = self.units
        return cfg


class FeatureGrouper(layers.Layer):
    """Group continuous features into low-dimensional representations."""
    def __init__(self, num_groups: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.num_groups = num_groups
        self.grouper = None

    def build(self, input_shape):
        self.grouper = layers.TimeDistributed(layers.Dense(self.num_groups, use_bias=False))
        self.grouper.build(input_shape)
        super().build(input_shape)

    def call(self, x):
        return self.grouper(x)

    def get_config(self):
        cfg = super().get_config()
        cfg["num_groups"] = self.num_groups
        return cfg


class SelectContinuousFeatures(layers.Layer):
    """Slices specific feature indices out of input tensor."""
    def __init__(self, continuous_indices: List[int], **kwargs):
        super().__init__(**kwargs)
        self.continuous_indices = tf.constant(continuous_indices, dtype=tf.int32)

    def call(self, x):
        return tf.gather(x, self.continuous_indices, axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg["continuous_indices"] = self.continuous_indices.numpy().tolist()
        return cfg


class CategoryFeatureSlice(layers.Layer):
    """Gathers categorical/structural feature indices."""
    def __init__(self, indices: List[int], **kwargs):
        super().__init__(**kwargs)
        self._indices = list(indices)
        self._gather_idx = tf.constant(self._indices, dtype=tf.int32)

    def call(self, x):
        return tf.gather(x, self._gather_idx, axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg["indices"] = self._indices
        return cfg


# ── Feature Categorization Helpers ────────────────────────────────────────────
def build_category_indices(feature_names: List[str]) -> Dict[str, List[int]]:
    CATEGORY_ORDER = ["structure", "conviction", "momentum", "flow", "candle", "session", "astro", "other"]
    EXACT_TOKEN_OVERRIDES = {"structure": {"r1", "r2", "r3", "s1", "s2", "s3"}, "astro": {"mc", "ic"}}
    DEFAULT_CATEGORY_KEYWORDS = {
        "structure":  ["pivot", "trendline", "support_", "resist_", "smc_", "bb_upper", "bb_lower",
                       "bb_middle", "bb_mid", "bb_squeeze", "snr_", "low_day_", "high_day_",
                       "parabolic_sar", "psar", "supertrend", "fp_poc", "fp_vah", "fp_val"],
        "conviction": ["candle_bull_score", "candle_bear_score", "pinbar", "reversal_",
                       "meanrev_", "trend_strength", "ha_reversal"],
        "momentum":   ["rsi", "macd", "mom_t", "mr_t", "tf_t", "regime_speed", "price_velocity",
                       "atr", "volatility_regime", "volatility_expansion", "volatility_bull",
                       "volatility_bear", "historical_volatility", "csm_", "mom_delta"],
        "flow":       ["volume", "tickvolume", "tick_volume", "obv", "vsi_20", "vol_zscore",
                       "fp_delta", "fp_cum_delta", "fp_imbalance", "fp_data_available",
                       "fp_high_vol_rejection"],
        "candle":     ["open", "high", "low", "close", "spread", "body_ratio", "close_pos",
                       "wick_r", "_dd", "bar_dir", "is_up_bar", "is_down_bar", "ha_candle",
                       "ha_flat", "doji", "ma_", "ema_", "sma_", "short_ma", "long_ma"],
        "session":    ["session", "hour", "minute", "day_of_week", "planetary_day"],
        "astro":      ["moon_", "sun_position", "sun_house", "sun_zodiac", "sun_speed",
                       "solstice", "equinox", "_position_deg", "_house", "_zodiac",
                       "_retrograde", "_station", "node_", "lilith", "part_of_fortune",
                       "ascendant", "descendant", "imum_coeli", "earth_helio",
                       "synodic", "conjunction", "sextile", "square", "trine", "opposition",
                       "eclipse", "dominant_element", "_speed"],
    }
    buckets = {c: [] for c in CATEGORY_ORDER}
    for idx, name in enumerate(feature_names):
        lname = name.lower()
        tokens = set(lname.split("_"))
        placed = False
        for cat in CATEGORY_ORDER:
            if cat == "other":
                continue
            if (tokens & EXACT_TOKEN_OVERRIDES.get(cat, set())) or \
               any(kw in lname for kw in DEFAULT_CATEGORY_KEYWORDS.get(cat, [])):
                buckets[cat].append(idx)
                placed = True
                break
        if not placed:
            buckets["other"].append(idx)
    return buckets


SERIES_FEATURE_GROUPS = {
    "open":   ["candle", "conviction"],
    "high":   ["candle", "momentum", "structure"],
    "low":    ["candle", "momentum", "structure"],
    "close":  ["candle", "momentum", "flow", "conviction"],
    "volume": ["flow", "momentum"],
}


def build_per_series_feature_indices(cat_idx: Dict[str, List[int]]) -> Dict[str, List[int]]:
    def merge(*groups):
        seen = set()
        out = []
        for g in groups:
            for i in g:
                if i not in seen:
                    seen.add(i)
                    out.append(i)
        return sorted(out)
    return {
        s: merge(*[cat_idx.get(c, []) for c in cats])
        for s, cats in SERIES_FEATURE_GROUPS.items()
    }


def candle_reconcile_np(raw_open, raw_high, raw_low, raw_close, raw_vol):
    high = np.maximum(np.maximum(raw_high, raw_open), raw_close)
    low  = np.minimum(np.minimum(raw_low,  raw_open), raw_close)
    vol  = np.log1p(np.exp(raw_vol))
    return np.stack([raw_open, high, low, raw_close, vol], axis=-1).astype(np.float32)


# ── Model Architecture Builders ───────────────────────────────────────────────
def save_model_config_metadata(model: keras.Model, series_name: str,
                               input_shape: Tuple[int, int],
                               per_series_indices: Dict[str, List[int]],
                               save_dir: str):
    """
    Ensures complete information for model reconstruction by saving a JSON configuration file.
    """
    config_path = os.path.join(save_dir, f"v2_series_{series_name}_config.json")
    indices = per_series_indices.get(series_name, [])
    meta = {
        "series_name": series_name,
        "input_shape": list(input_shape),
        "continuous_feature_indices": indices,
        "n_continuous_features": len(indices),
        "n_total_features": input_shape[1],
        "params_count": model.count_params(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(config_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved model config metadata → %s", config_path)


def build_series_model(series_name: str, input_shape: Tuple[int, int],
                       per_series_indices: Dict[str, List[int]],
                       output_dim: int, seq_len: int) -> keras.Model:
    half   = seq_len // 2
    recent = int(seq_len * 0.3)
    nm     = series_name

    raw_input = keras.Input(shape=input_shape, name=f"{nm}_input")

    # ── Branch 1: LSTM Stack consumes continuous features (>400) for series
    s_idx = per_series_indices.get(series_name, [])
    if not s_idx:
        s_idx = list(range(input_shape[1]))
    x_cont = SelectContinuousFeatures(s_idx, name=f"{nm}_select")(raw_input)

    b1 = layers.LSTM(256, return_sequences=True, kernel_initializer="glorot_uniform",
                     recurrent_initializer="orthogonal", name=f"{nm}_b1_lstm1")(x_cont)
    b1 = layers.LayerNormalization(name=f"{nm}_b1_ln1")(b1)
    b1 = layers.LSTM(48, return_sequences=True, kernel_initializer="glorot_uniform",
                     recurrent_initializer="orthogonal", name=f"{nm}_b1_lstm2")(b1)
    b1 = layers.LayerNormalization(name=f"{nm}_b1_ln2")(b1)
    b1 = layers.LSTM(48, return_sequences=False, kernel_initializer="glorot_uniform",
                     recurrent_initializer="orthogonal", name=f"{nm}_b1_lstm3")(b1)

    aux1 = layers.Dense(output_dim, kernel_initializer="zeros", name=f"{nm}_aux1")(b1)

    # ── Branch 2 & 3: Grid Attention consumes ALL raw input features (>600)
    x_half = layers.Lambda(lambda t: t[:, half:, :], name=f"{nm}_b2_slice")(raw_input)
    b2_y   = FeatureGrouper(num_groups=32, name=f"{nm}_b2_grouper")(x_half)
    b2     = GridAttention(128, name=f"{nm}_b2_attention")(b2_y)
    aux2   = layers.Dense(output_dim, kernel_initializer="zeros", name=f"{nm}_aux2")(b2)

    x_rec  = layers.Lambda(lambda t: t[:, -recent:, :], name=f"{nm}_b3_slice")(raw_input)
    b3_y   = FeatureGrouper(num_groups=64, name=f"{nm}_b3_grouper")(x_rec)
    b3     = GridAttention(128, name=f"{nm}_b3_attention")(b3_y)
    aux3   = layers.Dense(output_dim, kernel_initializer="zeros", name=f"{nm}_aux3")(b3)

    h1 = StopGradient(name=f"{nm}_sg1")(aux1)
    h2 = StopGradient(name=f"{nm}_sg2")(aux2)
    h3 = StopGradient(name=f"{nm}_sg3")(aux3)

    merged = layers.Concatenate(name=f"{nm}_merge")([b1, b2, b3, h1, h2, h3])
    merged = layers.LayerNormalization(name=f"{nm}_merge_ln")(merged)
    merged = layers.Dense(128, activation="tanh", name=f"{nm}_fusion")(merged)
    raw_out = layers.Dense(output_dim, name=f"{nm}_raw")(merged)

    return keras.Model(
        inputs=raw_input,
        outputs={
            f"{nm}_sequence": layers.Activation("linear", name=f"{nm}_sequence")(raw_out),
            f"{nm}_aux1": aux1,
            f"{nm}_aux2": aux2,
            f"{nm}_aux3": aux3,
        },
        name=f"series_model_{nm}"
    )


def build_volume_series_model(series_name: str, input_shape: Tuple[int, int],
                               per_series_indices: Dict[str, List[int]],
                               output_dim: int, seq_len: int) -> keras.Model:
    """
    Dedicated Volume Series Model Builder.
    Uses 3-branch LSTM + Grid Attention + Dense(128, tanh) fusion.
    Branch 1 LSTM processes continuous features; Branches 2 & 3 process full raw_input tensor (>600).
    """
    half   = seq_len // 2
    recent = int(seq_len * 0.3)
    nm     = series_name

    raw_input = keras.Input(shape=input_shape, name=f"{nm}_input")

    # ── Branch 1: LSTM Stack processes continuous features for volume
    s_idx = per_series_indices.get(series_name, [])
    if not s_idx:
        s_idx = list(range(input_shape[1]))
    x_cont = SelectContinuousFeatures(s_idx, name=f"{nm}_select")(raw_input)

    b1 = layers.LSTM(256, return_sequences=True, kernel_initializer="glorot_uniform",
                     recurrent_initializer="orthogonal", name=f"{nm}_b1_lstm1")(x_cont)
    b1 = layers.LayerNormalization(name=f"{nm}_b1_ln1")(b1)
    b1 = layers.LSTM(48, return_sequences=True, kernel_initializer="glorot_uniform",
                     recurrent_initializer="orthogonal", name=f"{nm}_b1_lstm2")(b1)
    b1 = layers.LayerNormalization(name=f"{nm}_b1_ln2")(b1)
    b1 = layers.LSTM(48, return_sequences=False, kernel_initializer="glorot_uniform",
                     recurrent_initializer="orthogonal", name=f"{nm}_b1_lstm3")(b1)

    aux1 = layers.Dense(output_dim, kernel_initializer="zeros", name=f"{nm}_aux1")(b1)

    # ── Branch 2 & 3: Grid Attention consumes ALL raw input features (>600)
    x_half = layers.Lambda(lambda t: t[:, half:, :], name=f"{nm}_b2_slice")(raw_input)
    b2_y   = FeatureGrouper(num_groups=32, name=f"{nm}_b2_grouper")(x_half)
    b2     = GridAttention(128, name=f"{nm}_b2_attention")(b2_y)
    aux2   = layers.Dense(output_dim, kernel_initializer="zeros", name=f"{nm}_aux2")(b2)

    x_rec  = layers.Lambda(lambda t: t[:, -recent:, :], name=f"{nm}_b3_slice")(raw_input)
    b3_y   = FeatureGrouper(num_groups=64, name=f"{nm}_b3_grouper")(x_rec)
    b3     = GridAttention(128, name=f"{nm}_b3_attention")(b3_y)
    aux3   = layers.Dense(output_dim, kernel_initializer="zeros", name=f"{nm}_aux3")(b3)

    h1 = StopGradient(name=f"{nm}_sg1")(aux1)
    h2 = StopGradient(name=f"{nm}_sg2")(aux2)
    h3 = StopGradient(name=f"{nm}_sg3")(aux3)

    merged = layers.Concatenate(name=f"{nm}_merge")([b1, b2, b3, h1, h2, h3])
    merged = layers.LayerNormalization(name=f"{nm}_merge_ln")(merged)
    merged = layers.Dense(128, activation="tanh", name=f"{nm}_fusion")(merged)
    raw_out = layers.Dense(output_dim, name=f"{nm}_raw")(merged)

    return keras.Model(
        inputs=raw_input,
        outputs={
            f"{nm}_sequence": layers.Activation("linear", name=f"{nm}_sequence")(raw_out),
            f"{nm}_aux1": aux1,
            f"{nm}_aux2": aux2,
            f"{nm}_aux3": aux3,
        },
        name=f"series_model_{nm}"
    )



def build_volume_cnn_model(input_shape: Tuple[int, int],
                            output_dim: int,
                            per_series_indices: Optional[Dict[str, List[int]]] = None,
                            embedding_dim: int = 64,
                            dropout_rate: float = 0.2) -> keras.Model:
    """
    CNN ensemble for volume sequence forecasting.

    Mirrors notebook Cell 1 build_volume_cnn_model exactly so that
    v2_series_volume_best.weights.h5 loads correctly at inference time.

    Architecture (3-branch fusion):
      Branch 1: Standard Causal CNN (local spike detection)
      Branch 2: Dilated Causal CNN with residual blocks (multi-scale patterns)
      Branch 3: Dense current-bar branch (immediate regime context)
      Fusion: Concatenate → LayerNorm → Dense(128, tanh) → Dense(output_dim)

    Uses the FULL input tensor — no SelectContinuousFeatures pre-selection.
    per_series_indices["volume"] can be empty depending on routing context;
    CNN filters learn volume-relevant features spatially without pre-selection.
    """
    nm  = "volume"
    inp = keras.Input(shape=input_shape, name=f"{nm}_input")
    x_sel = inp  # Full input (batch, 90, 663) — CNN learns feature relevance

    # ── Branch 1: Standard Causal CNN ────────────────────────────────────────
    b1 = layers.Conv1D(128, kernel_size=6, padding="causal",
                       name=f"{nm}_b1_conv1")(x_sel)
    b1 = layers.BatchNormalization(name=f"{nm}_b1_bn1")(b1)
    b1 = layers.LeakyReLU(name=f"{nm}_b1_lrelu1")(b1)
    b1 = layers.Dropout(dropout_rate, name=f"{nm}_b1_drop1")(b1)
    b1 = layers.Conv1D(128, kernel_size=6, padding="causal",
                       name=f"{nm}_b1_conv2")(b1)
    b1 = layers.BatchNormalization(name=f"{nm}_b1_bn2")(b1)
    b1 = layers.LeakyReLU(name=f"{nm}_b1_lrelu2")(b1)
    b1 = layers.Dropout(dropout_rate, name=f"{nm}_b1_drop2")(b1)
    b1 = layers.MaxPooling1D(pool_size=2, name=f"{nm}_b1_pool")(b1)
    b1 = layers.Flatten(name=f"{nm}_b1_flatten")(b1)
    b1 = layers.Dense(embedding_dim, activation="relu", name=f"{nm}_b1_proj")(b1)
    aux1 = layers.Dense(output_dim, kernel_initializer="zeros", name=f"{nm}_aux1")(b1)

    # ── Branch 2: Dilated Causal CNN (multi-scale temporal patterns) ─────────
    b2 = layers.Conv1D(128, kernel_size=3, padding="causal",
                       name=f"{nm}_b2_stem")(x_sel)
    b2 = layers.BatchNormalization(name=f"{nm}_b2_stem_bn")(b2)
    for rate, tag in zip([4, 8, 16, 32, 64], ["d4", "d8", "d16", "d32", "d64"]):
        r_ = layers.Conv1D(128, kernel_size=3, padding="causal", dilation_rate=rate,
                           name=f"{nm}_b2_{tag}_conv")(b2)
        r_ = layers.BatchNormalization(name=f"{nm}_b2_{tag}_bn")(r_)
        r_ = layers.LeakyReLU(name=f"{nm}_b2_{tag}_act")(r_)
        r_ = layers.Dropout(dropout_rate, name=f"{nm}_b2_{tag}_drop")(r_)
        b2 = layers.Add(name=f"{nm}_b2_{tag}_add")([b2, r_])
    b2 = layers.Conv1D(128, kernel_size=1, name=f"{nm}_b2_head")(b2)
    b2 = layers.GlobalAveragePooling1D(name=f"{nm}_b2_gap")(b2)
    b2 = layers.Dense(embedding_dim, activation="relu", name=f"{nm}_b2_proj")(b2)
    aux2 = layers.Dense(output_dim, kernel_initializer="zeros", name=f"{nm}_aux2")(b2)

    # ── Branch 3: Dense current-bar branch (immediate regime context) ────────
    b3 = layers.Lambda(lambda t: t[:, -1, :], name=f"{nm}_b3_last")(x_sel)
    b3 = layers.Dense(256, activation="relu",    name=f"{nm}_b3_d1")(b3)
    b3 = layers.BatchNormalization(name=f"{nm}_b3_bn1")(b3)
    b3 = layers.Dropout(dropout_rate, name=f"{nm}_b3_drop1")(b3)
    b3 = layers.Dense(128, activation="relu",    name=f"{nm}_b3_d2")(b3)
    b3 = layers.BatchNormalization(name=f"{nm}_b3_bn2")(b3)
    b3 = layers.Dropout(dropout_rate, name=f"{nm}_b3_drop2")(b3)
    b3 = layers.Dense(embedding_dim, activation="relu", name=f"{nm}_b3_proj")(b3)
    aux3 = layers.Dense(output_dim, kernel_initializer="zeros", name=f"{nm}_aux3")(b3)

    # ── Fusion: StopGradient + Concatenate → LayerNorm → Dense(128,tanh) ─────
    h1 = StopGradient(name=f"{nm}_sg1")(aux1)
    h2 = StopGradient(name=f"{nm}_sg2")(aux2)
    h3 = StopGradient(name=f"{nm}_sg3")(aux3)

    merged = layers.Concatenate(name=f"{nm}_merge")([b1, b2, b3, h1, h2, h3])
    merged = layers.LayerNormalization(name=f"{nm}_merge_ln")(merged)
    merged = layers.Dense(128, activation="tanh", name=f"{nm}_fusion")(merged)
    raw_out = layers.Dense(output_dim, name=f"{nm}_raw")(merged)

    return keras.Model(
        inputs=inp,
        outputs={
            f"{nm}_sequence": layers.Activation("linear", name=f"{nm}_sequence")(raw_out),
            f"{nm}_aux1": aux1,
            f"{nm}_aux2": aux2,
            f"{nm}_aux3": aux3,
        },
        name=f"series_model_{nm}"
    )


# ── Classification Ensemble Branch Builders ───────────────────────────────────
# Mirror the notebook Cell 1 builders exactly so weight shapes match when loading
# v2_classification_ensemble_best.weights.h5 from checkpoints.

def build_cnn_feature_branch(inputs, embedding_dim: int = 32, filters: int = 128,
                              kernel_size: int = 6, dropout_rate: float = 0.2,
                              name: str = 'cnn_branch'):
    """Two-block causal Conv1D + MaxPool + Flatten + Dense embedding (matches notebook Cell 1)."""
    x = layers.Conv1D(filters, kernel_size=kernel_size, padding='causal',
                      name=f'{name}_conv1')(inputs)
    x = layers.BatchNormalization(name=f'{name}_bn1')(x)
    x = layers.LeakyReLU(name=f'{name}_lrelu1')(x)
    x = layers.Dropout(dropout_rate, name=f'{name}_drop1')(x)
    x = layers.Conv1D(filters, kernel_size=kernel_size, padding='causal',
                      name=f'{name}_conv2')(x)
    x = layers.BatchNormalization(name=f'{name}_bn2')(x)
    x = layers.LeakyReLU(name=f'{name}_lrelu2')(x)
    x = layers.Dropout(dropout_rate, name=f'{name}_drop2')(x)
    x = layers.MaxPooling1D(pool_size=2, name=f'{name}_pool')(x)
    x = layers.Flatten(name=f'{name}_flatten')(x)
    return layers.Dense(embedding_dim, activation='relu', name=f'{name}_proj')(x)


def build_dilated_cnn_feature_branch(inputs, embedding_dim: int = 32, num_filters: int = 128,
                                      kernel_size: int = 3,
                                      dilation_rates: tuple = (4, 8, 16, 32, 64),
                                      dropout_rate: float = 0.2, l2_lambda: float = 0.01,
                                      name: str = 'dilated_branch'):
    """Dilated causal CNN: stem → 5 residual dilated blocks → GAP → Dense (matches notebook Cell 1)."""
    from tensorflow.keras.regularizers import l2 as keras_l2

    # Stem
    x = layers.Conv1D(num_filters, kernel_size=kernel_size, padding='causal',
                      name=f'{name}_stem_conv')(inputs)
    x = layers.BatchNormalization(name=f'{name}_stem_bn')(x)
    x = layers.LeakyReLU(name=f'{name}_stem_lrelu')(x)
    x = layers.Dropout(dropout_rate, name=f'{name}_stem_drop')(x)
    residual = x

    # Dilated stack: Cell 1 chains each dilated block into the next, then adds
    # a single projected stem shortcut before the head convolution.
    for i, d in enumerate(dilation_rates):
        x = layers.Conv1D(num_filters, kernel_size=kernel_size, padding='causal',
                          dilation_rate=d,
                          kernel_regularizer=keras_l2(l2_lambda),
                          name=f'{name}_dil{i}_conv')(x)
        x = layers.BatchNormalization(name=f'{name}_dil{i}_bn')(x)
        x = layers.LeakyReLU(name=f'{name}_dil{i}_lrelu')(x)
        x = layers.Dropout(dropout_rate, name=f'{name}_dil{i}_drop')(x)

    # Residual shortcut projection (1x1 conv to align channels)
    proj = layers.Conv1D(num_filters, kernel_size=1, padding='causal',
                         name=f'{name}_shortcut_proj')(residual)
    x = layers.Add(name=f'{name}_add')([proj, x])

    # Head conv
    x = layers.Conv1D(256, kernel_size=1, padding='causal',
                      name=f'{name}_head_conv')(x)
    x = layers.BatchNormalization(name=f'{name}_head_bn')(x)
    x = layers.LeakyReLU(name=f'{name}_head_lrelu')(x)
    x = layers.Dropout(dropout_rate, name=f'{name}_head_drop')(x)

    x = layers.GlobalAveragePooling1D(name=f'{name}_gap')(x)
    return layers.Dense(embedding_dim, activation='relu', name=f'{name}_proj')(x)


def build_current_dense_branch(inputs, embedding_dim: int = 32, dropout_rate: float = 0.2,
                                name: str = 'current_branch'):
    """Flatten → Dense → BN → Dropout embedding (matches notebook Cell 1)."""
    x = layers.Flatten(name=f'{name}_flatten')(inputs)
    x = layers.Dense(embedding_dim, activation='relu', name=f'{name}_dense')(x)
    x = layers.BatchNormalization(name=f'{name}_bn')(x)
    return layers.Dropout(dropout_rate, name=f'{name}_drop')(x)


def build_classification_ensemble_model(input_shape: Tuple[int, int],
                                         output_dim: int = N_SIGNAL_CLASSES,
                                         embedding_dim: int = 32,
                                         dropout_rate: float = 0.2) -> keras.Model:
    """3-branch fused classification ensemble (matches notebook Cell 1 builder exactly).

    Branches: CNN + Dilated Causal CNN + Dense → Concatenate → Dense(128) → Dense(64) → Softmax.
    Output: (batch, output_dim) softmax — class_output head.
    Weight file: v2_classification_ensemble_best.weights.h5
    """
    inp = keras.Input(shape=input_shape, name='ensemble_input')

    cnn_out     = build_cnn_feature_branch(inp, embedding_dim=embedding_dim)
    dilated_out = build_dilated_cnn_feature_branch(inp, embedding_dim=embedding_dim)
    dense_out   = build_current_dense_branch(inp, embedding_dim=embedding_dim)

    x = layers.Concatenate(name='ensemble_concat')([cnn_out, dilated_out, dense_out])
    x = layers.Dense(128, activation='relu', name='ensemble_dense1')(x)
    x = layers.BatchNormalization(name='ensemble_bn1')(x)
    x = layers.Dropout(dropout_rate, name='ensemble_drop1')(x)
    x = layers.Dense(64, activation='relu', name='ensemble_dense2')(x)
    x = layers.BatchNormalization(name='ensemble_bn2')(x)
    x = layers.Dropout(dropout_rate, name='ensemble_drop2')(x)

    out = layers.Dense(output_dim, activation='softmax', name='class_output')(x)
    return keras.Model(inputs=inp, outputs=out, name='classification_ensemble')


def load_keras_v3_weights_by_shape(model: keras.Model, weight_file: Path) -> int:
    """
    Load a Keras 3 `.weights.h5` file by matching exact layer weight shapes.

    Keras 3 stores weight-only files under generic traversal names
    (`layers/conv1d_5/...`) rather than the explicit runtime layer names. Small
    graph traversal differences can swap adjacent compatible layers and make
    `model.load_weights()` fail even when the architecture is semantically the
    same. This fallback keeps normal Keras loading as the first choice, then
    assigns each weighted runtime layer from the first unused H5 group with the
    exact same variable shapes.
    """
    with h5py.File(weight_file, "r") as h5:
        layer_root = h5.get("layers")
        if layer_root is None:
            raise ValueError(f"No 'layers' group found in {weight_file}")

        h5_groups = []
        for group_name in layer_root.keys():
            var_group = layer_root[group_name].get("vars")
            if var_group is None:
                continue
            values = [
                np.asarray(var_group[key])
                for key in sorted(var_group.keys(), key=lambda item: int(item))
            ]
            if values:
                h5_groups.append((group_name, values, [tuple(v.shape) for v in values]))

    used = set()
    loaded_layers = 0
    for layer in model.layers:
        layer_weights = layer.weights
        if not layer_weights:
            continue

        target_shapes = [tuple(w.shape) for w in layer_weights]
        match_idx = None
        for idx, (_group_name, _values, value_shapes) in enumerate(h5_groups):
            if idx in used:
                continue
            if value_shapes == target_shapes:
                match_idx = idx
                break
            # Slicing fallback for class_output layer if H5 checkpoint has 5 classes instead of 4
            if layer.name == "class_output" and len(value_shapes) == 2 and len(target_shapes) == 2:
                if value_shapes[0][0] == target_shapes[0][0]:
                    match_idx = idx
                    break

        if match_idx is None:
            raise ValueError(
                f"No matching H5 weight group for layer '{layer.name}' "
                f"with shapes {target_shapes}"
            )

        group_name, values, _value_shapes = h5_groups[match_idx]
        if layer.name == "class_output" and values[0].shape != layer_weights[0].shape:
            # Slice weights to match target dimension (e.g. 5 -> 4 classes)
            out_dim = layer_weights[0].shape[-1]
            sliced_values = [values[0][:, :out_dim], values[1][:out_dim]]
            layer.set_weights(sliced_values)
        else:
            layer.set_weights(values)
        used.add(match_idx)
        loaded_layers += 1
        logger.debug(
            "[V2Runtime] Shape-loaded ensemble layer %s from H5 group %s",
            layer.name,
            group_name,
        )

    return loaded_layers


def build_refiner_model(forecast_steps: int) -> keras.Model:
    inp = keras.Input(shape=(forecast_steps, 5), name="refiner_input")
    x = layers.Flatten(name="refiner_flat")(inp)
    x = layers.Dense(128, activation="relu", name="refiner_d1")(x)
    x = layers.LayerNormalization(name="refiner_ln1")(x)
    x = layers.Dense(128, activation="relu", name="refiner_d2")(x)
    x = layers.LayerNormalization(name="refiner_ln2")(x)
    x = layers.Dense(forecast_steps * 5, name="refiner_d3")(x)
    x = layers.Reshape((forecast_steps, 5), name="refiner_reshape")(x)
    x = layers.Add(name="refiner_residual")([inp, x])

    open_out  = layers.Lambda(lambda t: t[:, :, 0], name="open_sequence")(x)
    high_out  = layers.Lambda(lambda t: t[:, :, 1], name="high_sequence")(x)
    low_out   = layers.Lambda(lambda t: t[:, :, 2], name="low_sequence")(x)
    close_out = layers.Lambda(lambda t: t[:, :, 3], name="close_sequence")(x)
    vol_out   = layers.Lambda(lambda t: t[:, :, 4], name="volume_sequence")(x)

    return keras.Model(
        inputs=inp,
        outputs={
            "open_sequence":   open_out,
            "high_sequence":   high_out,
            "low_sequence":    low_out,
            "close_sequence":  close_out,
            "volume_sequence": vol_out,
        },
        name="refiner_model"
    )


def build_context_model(input_shape: Tuple[int, int],
                        continuous_feature_indices: List[int],
                        structure_indices: List[int],
                        forecast_steps: int,
                        n_zone_candidates: int = 7) -> keras.Model:
    def _conv_block(x, filters, kernel_size, pfx):
        for i, f in enumerate(filters):
            x = layers.Conv1D(f, kernel_size, padding="same", name=f"{pfx}_conv{i}")(x)
            x = layers.BatchNormalization(name=f"{pfx}_bn{i}")(x)
            x = layers.LeakyReLU(name=f"{pfx}_lrelu{i}")(x)
            x = layers.Dropout(0.2, name=f"{pfx}_drop{i}")(x)
        return x

    def _ms_tower(tower_slice, seq_len, pfx, filters=(128, 128, 64, 64, 64, 64), kernel_size=6, fd=64):
        half   = seq_len // 2
        recent = int(seq_len * 0.3)
        fe = layers.GlobalAveragePooling1D(name=f"{pfx}_full_gap")(_conv_block(tower_slice, filters, kernel_size, f"{pfx}_full"))
        hs = layers.Lambda(lambda t: t[:, half:, :], name=f"{pfx}_sl_half")(tower_slice)
        he = layers.GlobalAveragePooling1D(name=f"{pfx}_half_gap")(_conv_block(hs, filters, kernel_size, f"{pfx}_half"))
        rs = layers.Lambda(lambda t: t[:, -recent:, :], name=f"{pfx}_sl_rec")(tower_slice)
        re = layers.GlobalAveragePooling1D(name=f"{pfx}_rec_gap")(_conv_block(rs, filters, kernel_size, f"{pfx}_rec"))
        f  = layers.Concatenate(name=f"{pfx}_concat")([fe, he, re])
        f  = layers.Dense(fd, activation="relu", name=f"{pfx}_fd")(f)
        f  = layers.BatchNormalization(name=f"{pfx}_fbn")(f)
        return layers.Dropout(0.2, name=f"{pfx}_fdrop")(f)

    def _binary_head(emb, name, fc_dim=32):
        x = layers.Dense(fc_dim, activation="relu", name=f"{name}_fc1")(emb)
        x = layers.BatchNormalization(name=f"{name}_bn")(x)
        x = layers.Dropout(0.3, name=f"{name}_drop")(x)
        return layers.Dense(1, activation="sigmoid", name=name)(x)

    def _multiclass_head(emb, name, n, fc_dim=32):
        x = layers.Dense(fc_dim, activation="relu", name=f"{name}_fc1")(emb)
        x = layers.BatchNormalization(name=f"{name}_bn")(x)
        x = layers.Dropout(0.3, name=f"{name}_drop")(x)
        return layers.Dense(n, activation="softmax", name=name)(x)

    def _branch1_lstm_stack(x, name_prefix):
        x = layers.LSTM(256, return_sequences=True, kernel_initializer="glorot_uniform",
                        recurrent_initializer="orthogonal", name=f"{name_prefix}_lstm1")(x)
        x = layers.LayerNormalization(name=f"{name_prefix}_ln1")(x)
        x = layers.LSTM(48, return_sequences=True, kernel_initializer="glorot_uniform",
                        recurrent_initializer="orthogonal", name=f"{name_prefix}_lstm2")(x)
        x = layers.LayerNormalization(name=f"{name_prefix}_ln2")(x)
        x = layers.LSTM(48, return_sequences=False, kernel_initializer="glorot_uniform",
                        recurrent_initializer="orthogonal", name=f"{name_prefix}_lstm3")(x)
        return x

    inputs  = keras.Input(shape=input_shape, name="ctx_input")
    seq_len = input_shape[0]
    outputs = {}

    bb_slice = CategoryFeatureSlice(structure_indices, name="bb_structure_slice")(inputs)
    bb_fused = _ms_tower(bb_slice, seq_len, "bb_tower")
    for sig in ["Signal_bounce_support", "Signal_bounce_resistance",
                "Signal_breakout_support", "Signal_breakout_resistance"]:
        outputs[sig] = _binary_head(bb_fused, sig, fc_dim=32)

    nz_slice = CategoryFeatureSlice(structure_indices, name="nz_structure_slice")(inputs)
    nz_fused = _ms_tower(nz_slice, seq_len, "nz_tower")
    outputs["next_zone_idx"]      = layers.Dense(n_zone_candidates, activation="softmax", name="next_zone_idx")(nz_fused)
    outputs["next_zone_bars"]     = layers.Dense(1, name="next_zone_bars")(nz_fused)
    outputs["next_zone_distance"] = layers.Dense(1, name="next_zone_distance")(nz_fused)
    outputs["next_zone_volume"]   = layers.Dense(1, name="next_zone_volume")(nz_fused)

    rp_x = SelectContinuousFeatures(continuous_feature_indices, name="rolling_probability_select")(inputs)
    rp_g = layers.GRU(64, return_sequences=False, name="rolling_probability_gru")(rp_x)
    rp_l = layers.LSTM(64, return_sequences=False, name="rolling_probability_lstm")(rp_x)
    rp_e = layers.Concatenate(name="rolling_probability_concat")([rp_g, rp_l])
    rp_e = layers.Dense(32, activation="relu", name="rolling_probability_dense")(rp_e)
    outputs["rolling_probability"] = layers.Dense(1, activation="sigmoid", name="rolling_probability")(rp_e)

    for head_name, dim in {"support_trendline": forecast_steps, "resist_trendline": forecast_steps,
                            "mfe": 1, "mae": 1, "reversal_prob": 1, "trend_continuation_prob": 1}.items():
        bsh_x = SelectContinuousFeatures(continuous_feature_indices, name=f"{head_name}_select")(inputs)
        bsh   = _branch1_lstm_stack(bsh_x, f"{head_name}_b1")
        outputs[head_name] = layers.Dense(dim, name=head_name)(bsh)

    cx = SelectContinuousFeatures(continuous_feature_indices, name="ctx_phase2_select")(inputs)
    cx = layers.Conv1D(16, 3, activation="relu", padding="same", name="ctx_phase2_conv1")(cx)
    cx = layers.MaxPooling1D(2, name="ctx_phase2_pool")(cx)
    cx = layers.Conv1D(32, 3, activation="relu", padding="same", name="ctx_phase2_conv2")(cx)
    cx = layers.GlobalAveragePooling1D(name="ctx_phase2_gap")(cx)
    outputs["reversal_held"] = _binary_head(cx, "reversal_held", fc_dim=32)
    outputs["bull_conf"]     = _binary_head(cx, "bull_conf",     fc_dim=32)
    outputs["bear_conf"]     = _binary_head(cx, "bear_conf",     fc_dim=32)
    outputs["bull_class"]    = _multiclass_head(cx, "bull_class", n=3, fc_dim=32)

    return keras.Model(inputs=inputs, outputs=outputs, name="context_model")


# ── AXE Genesis V2 Runtime Orchestrator Class ─────────────────────────────────
class AXEGenesisV2Runtime:
    """
    Unified runtime wrapper presenting AXE Genesis V2 as a single 7-model ensemble.

    Models loaded (7 total):
      - 5 series models (open/high/low/close/volume) — LSTM-based OHLCV forecasts
      - 1 context model — 19 heads for market signals, zones, extrema, reversals
      - 1 classification ensemble model (4-class signal type) — REQUIRED (not optional)

    Active optional complement:
      - Volume CNN variant — loaded when the saved checkpoint exists and blended into the volume forecast.

    Refiner model is not loaded; it is replaced by NumPy candle-geometry reconciliation.
    All required models must load successfully; missing weights or models = fatal error.
    """
    def __init__(self, checkpoint_dir: Optional[Path] = None, dataset_dir: Optional[Path] = None, variant_tag: str = "market"):
        self.checkpoint_dir = Path(checkpoint_dir or _DEFAULT_CHECKPOINT_DIR)
        self.dataset_dir    = Path(dataset_dir or _DEFAULT_DATASET_DIR)
        self.variant_tag    = variant_tag

        self._feature_names: List[str] = []
        self.series_models: Dict[str, keras.Model]  = {}
        self.volume_cnn_model: Optional[keras.Model] = None
        self.context_model: Optional[keras.Model]   = None
        self.ensemble_model: Optional[keras.Model]  = None  # REQUIRED (not optional)
        self.refiner_model: Optional[keras.Model]   = None  # legacy compatibility; not used in active runtime

        self._load_feature_map()
        self._build_and_load_all()

    def _load_feature_map(self):
        fpath = self.dataset_dir / "feature_index_map.json"
        if not fpath.exists():
            raise FileNotFoundError(f"Feature map not found at {fpath}")
        payload = json.loads(fpath.read_text())
        feature_map = payload.get("feature_index_map", payload)
        ordered = sorted(feature_map.items(), key=lambda kv: int(kv[1]))
        self._feature_names = [name for name, _ in ordered]
        if len(self._feature_names) != N_FEATURES:
            logger.warning(f"[V2Runtime] Expected {N_FEATURES} features, got {len(self._feature_names)}")

    def _build_and_load_all(self):
        logger.info(f"[V2Runtime] Rebuilding V2 architecture and loading weights from {self.checkpoint_dir}...")
        cat_idx = build_category_indices(self._feature_names)
        per_ser = build_per_series_feature_indices(cat_idx)
        cont_idx = sorted(
            idx for cat in {"candle", "momentum", "flow", "structure", "conviction"}
            for idx in cat_idx.get(cat, [])
        )
        struct_idx = cat_idx.get("structure", [])

        # 1. Series models — load from .keras (architecture+weights) to avoid
        #    builder drift. Falls back to rebuild+weight-only if .keras absent.
        CUSTOM_OBJS = {
            "StopGradient": StopGradient,
            "GridAttention": GridAttention,
            "FeatureGrouper": FeatureGrouper,
            "SelectContinuousFeatures": SelectContinuousFeatures,
            "CategoryFeatureSlice": CategoryFeatureSlice,
        }
        for sname in ["open", "high", "low", "close", "volume"]:
            vtag = f"_{self.variant_tag}" if self.variant_tag != "market" else ""
            keras_file  = self.checkpoint_dir / f"v2_series_{sname}{vtag}_best.keras"
            weight_file = self.checkpoint_dir / f"v2_series_{sname}{vtag}_best.weights.h5"
            if keras_file.exists():
                sm = tf.keras.models.load_model(
                    str(keras_file), custom_objects=CUSTOM_OBJS, compile=False, safe_mode=False
                )
                logger.info(f"  [V2Runtime] ✅ Series '{sname}' loaded from .keras ({sm.count_params():,} params)")
            elif weight_file.exists():
                # Fallback: rebuild using per-series config JSON indices if saved
                cfg_file = self.checkpoint_dir / f"v2_series_{sname}_config.json"
                if cfg_file.exists():
                    cfg = json.loads(cfg_file.read_text())
                    series_per = {sname: cfg["continuous_feature_indices"]}
                else:
                    series_per = per_ser
                if sname == "volume":
                    sm = build_volume_series_model(sname, (SEQ_LEN, N_FEATURES), series_per, FORECAST_STEPS, SEQ_LEN)
                else:
                    sm = build_series_model(sname, (SEQ_LEN, N_FEATURES), series_per, FORECAST_STEPS, SEQ_LEN)
                sm.load_weights(str(weight_file))
                logger.info(f"  [V2Runtime] ✅ Series '{sname}' loaded from weights ({sm.count_params():,} params)")
            else:
                raise FileNotFoundError(f"No artifact for series '{sname}' in {self.checkpoint_dir}")
            self.series_models[sname] = sm

        # 1b. Volume CNN complement model — use the saved alternative volume predictor
        vtag = f"_{self.variant_tag}" if self.variant_tag != "market" else ""
        vol_cnn_file = self.checkpoint_dir / f"v2_series_volume_cnn{vtag}_best.weights.h5"
        if vol_cnn_file.exists():
            vol_cfg_file = self.checkpoint_dir / "v2_series_volume_config.json"
            volume_indices = per_ser.get("volume", list(range(N_FEATURES)))
            if vol_cfg_file.exists():
                cfg = json.loads(vol_cfg_file.read_text())
                volume_indices = cfg.get("continuous_feature_indices", volume_indices)
            self.volume_cnn_model = build_volume_cnn_model(
                (SEQ_LEN, N_FEATURES),
                FORECAST_STEPS,
                {"volume": volume_indices},
                embedding_dim=64,
                dropout_rate=0.2,
            )
            self.volume_cnn_model.load_weights(str(vol_cnn_file))
            logger.info(
                "  [V2Runtime] ✅ Volume CNN complement loaded from %s "
                "(%s params)",
                vol_cnn_file,
                f"{self.volume_cnn_model.count_params():,}",
            )
        else:
            logger.info(
                "  [V2Runtime] ℹ️  No volume CNN complement weights found at %s; using base volume model only.",
                vol_cnn_file,
            )

        # 2. Context model
        ctx_file = self.checkpoint_dir / f"v2_context{vtag}_best.weights.h5"
        self.context_model = build_context_model((SEQ_LEN, N_FEATURES), cont_idx, struct_idx, FORECAST_STEPS)
        self.context_model.load_weights(str(ctx_file))
        logger.info(f"  [V2Runtime] ✅ Context model loaded ({self.context_model.count_params():,} params)")

        # 3. Classification ensemble model — REQUIRED (not optional, always loaded)
        ens_file = self.checkpoint_dir / f"v2_classification_ensemble{vtag}_best.weights.h5"
        if not ens_file.exists():
            raise FileNotFoundError(
                f"Classification ensemble weights REQUIRED but not found at {ens_file}\n"
                f"Ensemble is mandatory for signal type prediction (4-class: bounce_support, bounce_resistance, "
                f"breakout_support, breakout_resistance)\n"
                f"Train Cell 10 in the notebook to generate {ens_file.name}"
            )
        
        try:
            self.ensemble_model = build_classification_ensemble_model(
                input_shape=(SEQ_LEN, N_FEATURES),
                output_dim=N_SIGNAL_CLASSES,
            )
            try:
                self.ensemble_model.load_weights(str(ens_file))
            except Exception as load_exc:
                # Fallback: shape-matched loading for Keras v3 traversal mismatches
                loaded_layers = load_keras_v3_weights_by_shape(self.ensemble_model, ens_file)
                logger.info(
                    "  [V2Runtime] ℹ️  Classification ensemble loaded via "
                    "shape-matched Keras-v3 fallback: %s weighted layers",
                    loaded_layers,
                )
            logger.info(
                f"  [V2Runtime] ✅ Classification ensemble loaded (REQUIRED) "
                f"({self.ensemble_model.count_params():,} params) — "
                f"{N_SIGNAL_CLASSES} classes: {SIGNAL_CLASS_NAMES}"
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load classification ensemble from {ens_file}: {exc}\n"
                f"Ensemble is REQUIRED for this runtime configuration."
            )

    def _build_confidence_layer(self, raw_heads: Dict[str, Any]) -> Dict[str, Any]:
        """Extract confidence metrics from the model heads that actually drove the prediction."""
        def _safe_float(v: Any, default: float = 0.0) -> float:
            try:
                arr = np.asarray(v).squeeze()
                if arr.size == 0:
                    return float(default)
                return float(arr)
            except Exception:
                return float(default)

        def _softmax_confidence(v: Any) -> float:
            try:
                arr = np.asarray(v).reshape(-1)
                arr = np.clip(arr.astype(np.float32), 1e-12, None)
                arr = arr / np.sum(arr)
                return float(np.max(arr))
            except Exception:
                return 0.0

        def _margin_from_probs(v: Any) -> float:
            try:
                arr = np.asarray(v).reshape(-1)
                arr = np.clip(arr.astype(np.float32), 1e-12, None)
                arr = arr / np.sum(arr)
                if arr.size < 2:
                    return 0.0
                ranked = np.sort(arr)[::-1]
                return float(np.clip(ranked[0] - ranked[1], 0.0, 1.0))
            except Exception:
                return 0.0

        def _entropy_norm(v: Any) -> float:
            try:
                arr = np.asarray(v).reshape(-1)
                arr = np.clip(arr.astype(np.float32), 1e-12, None)
                arr = arr / np.sum(arr)
                if arr.size <= 1:
                    return 0.0
                entropy = -np.sum(arr * np.log(arr + 1e-12)) / np.log(arr.size)
                return float(np.clip(entropy, 0.0, 1.0))
            except Exception:
                return 0.0

        confidence: Dict[str, Any] = {}

        # Ensemble confidence from final 5-class signal head.
        ens_probs = raw_heads.get("class_output")
        if ens_probs is not None:
            ens_arr = np.asarray(ens_probs).reshape(-1)
            if ens_arr.size > 0:
                ens_arr = np.clip(ens_arr.astype(np.float32), 1e-12, None)
                ens_arr = ens_arr / np.sum(ens_arr)
                confidence["ensemble_confidence"] = float(np.max(ens_arr))
                confidence["ensemble_margin"] = float(np.clip(np.max(ens_arr) - np.partition(ens_arr, -2)[-2], 0.0, 1.0))
                confidence["ensemble_entropy"] = float(np.clip(-np.sum(ens_arr * np.log(ens_arr + 1e-12)) / np.log(ens_arr.size), 0.0, 1.0))
                confidence["prediction_confidence"] = float(np.max(ens_arr))

        if "signal_class_conf" in raw_heads:
            confidence["prediction_confidence"] = max(confidence.get("prediction_confidence", 0.0), _safe_float(raw_heads["signal_class_conf"], 0.0))

        # Context signal confidence from the binary signal head family.
        ctx_heads = [
            "Signal_bounce_support", "Signal_bounce_resistance",
            "Signal_breakout_support", "Signal_breakout_resistance",
        ]
        ctx_values = [ _safe_float(raw_heads[k], 0.0) for k in ctx_heads if k in raw_heads ]
        if ctx_values:
            top = max(ctx_values)
            sorted_vals = sorted(ctx_values, reverse=True)
            second = sorted_vals[1] if len(sorted_vals) > 1 else 0.0
            confidence["context_signal_confidence"] = float(np.clip(top, 0.0, 1.0))
            confidence["context_signal_margin"] = float(np.clip(top - second, 0.0, 1.0))

        if "next_zone_idx" in raw_heads:
            confidence["next_zone_top_confidence"] = _softmax_confidence(raw_heads["next_zone_idx"])
        if "bull_class" in raw_heads:
            confidence["bull_class_top_confidence"] = _softmax_confidence(raw_heads["bull_class"])

        # Optional nested aggregate for downstream consumers.
        confidence["confidence_layer"] = {
            "prediction_confidence": confidence.get("prediction_confidence", 0.0),
            "ensemble_confidence": confidence.get("ensemble_confidence", 0.0),
            "ensemble_margin": confidence.get("ensemble_margin", 0.0),
            "ensemble_entropy": confidence.get("ensemble_entropy", 0.0),
            "context_signal_confidence": confidence.get("context_signal_confidence", 0.0),
            "context_signal_margin": confidence.get("context_signal_margin", 0.0),
            "next_zone_top_confidence": confidence.get("next_zone_top_confidence", 0.0),
            "bull_class_top_confidence": confidence.get("bull_class_top_confidence", 0.0),
        }

        return confidence

    def predict(
        self,
        X: np.ndarray,
        feature_window: Optional[Any] = None,
        snr_features: Optional[Dict[str, Any]] = None,
        raw_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the model forward pass and return a well-defined dict structure.

        Returns:
          raw_only=True:  Dict of 25 canonical raw outputs (normalized, unscaled)
          raw_only=False: Contract-mapped outputs (UI/LLM-facing with derived fields)

        Output Structure (raw_only=True):
          {
            # OHLCV (5 heads) from series models + candle geometry reconciler
            'open_sequence': np.array shape (batch, 12),
            'high_sequence': np.array shape (batch, 12),
            'low_sequence': np.array shape (batch, 12),
            'close_sequence': np.array shape (batch, 12),
            'volume_sequence': np.array shape (batch, 12),

            # Context Model (19 heads)
            # Bounce/Breakout (4)
            'Signal_bounce_support': float [0, 1],
            'Signal_bounce_resistance': float [0, 1],
            'Signal_breakout_support': float [0, 1],
            'Signal_breakout_resistance': float [0, 1],
            # Next Zone (4)
            'next_zone_idx': np.array shape (7,) softmax,
            'next_zone_bars': float,
            'next_zone_distance': float,
            'next_zone_volume': float,
            # Probability (1)
            'rolling_probability': float [0, 1],
            # Trendlines (2)
            'support_trendline': np.array shape (batch, 12),
            'resist_trendline': np.array shape (batch, 12),
            # Extremum (2)
            'mfe': float,
            'mae': float,
            # Reversal (3)
            'reversal_prob': float [0, 1],
            'trend_continuation_prob': float [0, 1],
            'reversal_held': float [0, 1],
            # Classification (3)
            'bull_conf': float [0, 1],
            'bear_conf': float [0, 1],
            'bull_class': np.array shape (3,) softmax [p_bear, p_neutral, p_bull],

            # Ensemble (1)
            'class_output': np.array shape (5,) softmax [p_bounce_support, p_bounce_resistance,
                                                          p_breakout_support, p_breakout_resistance,
                                                          p_no_signal],
          }

        Total: 25 canonical outputs (all normalized)
        """
        if X.ndim == 2:
            X = np.expand_dims(X, axis=0)

        # Step 1: 5 Series models (+ optional volume-CNN complement)
        raw_preds = {}
        for sname, model in self.series_models.items():
            out = model.predict(X, verbose=0)
            if not isinstance(out, dict):
                out = dict(zip(model.output_names, out if isinstance(out, (list, tuple)) else [out]))
            raw_preds[sname] = out[f"{sname}_sequence"]

        if self.volume_cnn_model is not None:
            vol_cnn_out = self.volume_cnn_model.predict(X, verbose=0)
            if not isinstance(vol_cnn_out, dict):
                vol_cnn_out = dict(zip(self.volume_cnn_model.output_names,
                                       vol_cnn_out if isinstance(vol_cnn_out, (list, tuple)) else [vol_cnn_out]))
            raw_preds["volume_cnn"] = vol_cnn_out["volume_sequence"]
            raw_preds["volume"] = 0.5 * (raw_preds["volume"] + raw_preds["volume_cnn"])

        # Step 2: Candle reconciliation (NumPy only, no learnable refiner)
        stacked = candle_reconcile_np(
            raw_preds["open"], raw_preds["high"], raw_preds["low"],
            raw_preds["close"], raw_preds["volume"]
        )

        # Extract reconciled OHLCV as dict (identity outputs after reconciliation)
        reconciled_preds = {
            "open_sequence":   stacked[:, :, 0],
            "high_sequence":   stacked[:, :, 1],
            "low_sequence":    stacked[:, :, 2],
            "close_sequence":  stacked[:, :, 3],
            "volume_sequence": stacked[:, :, 4],
        }

        # Step 3: Context model
        ctx_preds = self.context_model.predict(X, verbose=0)
        if not isinstance(ctx_preds, dict):
            ctx_preds = dict(zip(self.context_model.output_names,
                                  ctx_preds if isinstance(ctx_preds, (list, tuple)) else [ctx_preds]))

        # Step 4: Classification ensemble (REQUIRED — always present)
        ens_preds: Dict[str, Any] = {}
        try:
            ens_raw = self.ensemble_model.predict(X, verbose=0)  # (batch, N_SIGNAL_CLASSES)
            probs = np.squeeze(np.asarray(ens_raw))
            if probs.ndim == 0:
                probs = np.array([probs])
            ens_preds["class_output"] = probs
            
            best_cls = int(np.argmax(probs))
            ens_preds["signal_class"]      = best_cls
            ens_preds["signal_class_name"] = SIGNAL_CLASS_NAMES[best_cls]
            ens_preds["signal_class_conf"] = float(probs[best_cls])
            logger.info(
                "[V2Runtime] Ensemble → class=%d (%s) conf=%.3f",
                best_cls, SIGNAL_CLASS_NAMES[best_cls], float(probs[best_cls])
            )
        except Exception as exc:
            raise RuntimeError(
                f"Classification ensemble prediction failed (REQUIRED): {exc}\n"
                f"This is a fatal error; ensemble must always produce output."
            )

        # Combine all outputs into single dict
        raw_heads: Dict[str, Any] = {}
        raw_heads.update(reconciled_preds)
        if self.volume_cnn_model is not None:
            raw_heads["volume_cnn_sequence"] = raw_preds["volume_cnn"]
        raw_heads.update(ctx_preds)
        raw_heads.update(ens_preds)

        # Confidence layer: expose the actual internal certainty values that drove the prediction.
        confidence_layer = self._build_confidence_layer(raw_heads)
        raw_heads.update(confidence_layer)

        if raw_only:
            return raw_heads

        # Contract mapping (separate derived view)
        mapped_heads = self._map_heads_to_contract(raw_heads, feature_window=feature_window, snr_features=snr_features)
        return mapped_heads

    def _map_heads_to_contract(self, raw_heads: Dict[str, Any], feature_window: Optional[Any] = None, snr_features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Map raw model output heads to the UI/LLM-facing contract.

        Produces a zone-level story covering:
          - direction_class / direction_label (3-class: Bear / Neutral / Bull)
          - zone_interaction type (bounce/breakout, from ensemble + context model)
          - reversal_prob / trend_continuation_prob
          - bull_prob, bull_strength, bear_strength (continuous conviction)
          - next_zone_eta_bars, next_zone_pct_away (zone timing)
          - risk_reward derived from MFE / MAE
          - zone_story dict: structured summary for UI cards and LLM prompts
        """
        mapped: Dict[str, Any] = {}
        for k, v in raw_heads.items():
            mapped[k] = np.squeeze(np.asarray(v)) if hasattr(v, "ndim") else v

        # ── Helper ─────────────────────────────────────────────────────────────
        def _s(key: str, default: float = 0.0) -> float:
            v = mapped.get(key, default)
            try:
                return float(list(np.asarray(v).flat)[0])
            except (TypeError, ValueError, IndexError):
                return default

        # ── 1. rolling_probability → bull_prob ───────────────────────────────
        if "rolling_probability" in mapped:
            mapped["bull_prob"] = mapped.pop("rolling_probability")

        # Also keep raw bull_prob as [0,1] scalar for frontend clampNumber compatibility
        if "bull_prob" in mapped:
            v = mapped["bull_prob"]
            mapped["bull_prob"] = float(list(np.asarray(v).flat)[0]) if hasattr(v, "__len__") else float(v)

        # ── 2. Signal_* → signal_* (lowercase for UI compatibility) ──────────
        for sig_key in ["Signal_bounce_support", "Signal_bounce_resistance",
                        "Signal_breakout_support", "Signal_breakout_resistance"]:
            if sig_key in mapped:
                mapped[sig_key.lower()] = mapped[sig_key]

        # ── 3. Continuous directional conviction ─────────────────────────────
        mapped["bull_strength"] = _s("bull_conf") if "bull_conf" in mapped else _s("bull_prob")
        if "bear_conf" in mapped:
            mapped["bear_strength"] = _s("bear_conf")

        # ── 4. Direction class (3-class: Bear / Neutral / Bull) ───────────────
        # Context model emits bull_class as 3-class softmax [Bear, Neutral, Bull]
        if "bull_class" in mapped:
            bc = np.asarray(mapped["bull_class"]).squeeze()
            if bc.ndim == 1 and len(bc) == 3:
                cls_idx = int(np.argmax(bc))
                cls_labels = ["Bear", "Neutral", "Bull"]
                mapped["direction_class"]  = cls_idx
                mapped["direction_label"]  = cls_labels[cls_idx]
                mapped["direction_conf"]   = round(float(bc[cls_idx]) * 100, 1)
                mapped["direction_probs"]  = bc.tolist()          # [p_bear, p_neutral, p_bull]
                mapped["direction_net"]    = round(float(bc[2] - bc[0]) * 100, 1)  # bull − bear
            else:
                # Legacy scalar: 0.0=Bear, 0.5=Neutral, 1.0=Bull
                raw_val = float(bc.flat[0]) if hasattr(bc, "flat") else float(bc)
                cls_idx = {0.0: 0, 0.5: 1, 1.0: 2}.get(round(raw_val * 2) / 2, 1)
                mapped["direction_class"] = cls_idx
                mapped["direction_label"] = ["Bear", "Neutral", "Bull"][cls_idx]
                mapped["direction_conf"]  = 0.0
                mapped["direction_net"]   = round((raw_val - 0.5) * 200, 1)

        # ── 5. Signal strength (max of 4 binary zone-signal heads) ────────────
        sig_vals = [
            _s(k) for k in ["signal_bounce_support", "signal_bounce_resistance",
                             "signal_breakout_support", "signal_breakout_resistance"]
            if k in mapped
        ]
        if sig_vals:
            mapped["signal_strength"] = float(max(sig_vals))

        # Dominant signal from context model binary heads
        _sig_scores = {
            "bounce_support":      _s("signal_bounce_support"),
            "bounce_resistance":   _s("signal_bounce_resistance"),
            "breakout_support":    _s("signal_breakout_support"),
            "breakout_resistance": _s("signal_breakout_resistance"),
        }
        best_ctx_sig = max(_sig_scores, key=_sig_scores.get)  # type: ignore[arg-type]
        mapped["ctx_signal_type"] = best_ctx_sig
        mapped["ctx_signal_conf"] = round(_sig_scores[best_ctx_sig] * 100, 1)

        # ── 6. Classification ensemble head ───────────────────────────────────
        if "signal_class" in mapped and "signal_class_conf" in mapped:
            cls_name = str(mapped.get("signal_class_name", ""))
            conf     = float(mapped["signal_class_conf"])
            if cls_name != "no_signal":
                mapped["signal_strength"] = max(mapped.get("signal_strength", 0.0), conf)
            if cls_name in ("bounce_support", "breakout_support"):
                mapped.setdefault("ensemble_bias", "BULL")
                mapped["ensemble_confidence"] = round(conf * 100, 1)
            elif cls_name in ("bounce_resistance", "breakout_resistance"):
                mapped.setdefault("ensemble_bias", "BEAR")
                mapped["ensemble_confidence"] = round(conf * 100, 1)
            else:
                mapped.setdefault("ensemble_bias", "NEUTRAL")
                mapped["ensemble_confidence"] = 0.0

        # BUG-8: Add binary bull_class alias (0=Bear, 1=Bull) for frontend compat
        # Frontend expects scalar 0 or 1 but V2 emits 3-class softmax via direction_class
        if "direction_class" in mapped:
            mapped["bull_class"] = int(mapped["direction_class"] == 2)  # 2 = Bull slot

        # BUG-7: Derive vol_surge from ensemble confidence (no model head for this)
        # Frontend uses vol_surge to widen the confidence band: higher = wider band.
        ens_conf = float(mapped.get("ensemble_confidence", 0.0)) / 100.0
        bull_p   = _s("bull_prob")
        # vol_surge high when ensemble is very confident OR bull_prob is extreme
        mapped["vol_surge"] = round(min(1.0, ens_conf * 0.7 + abs(bull_p - 0.5) * 0.6), 4)

        # ── 7. Reversal & continuation probabilities ──────────────────────────
        # Keep as [0,1] for frontend compatibility (StockChart reads > 0.45, > 0.65 etc.)
        if "reversal_prob" in mapped:
            mapped["reversal_prob"] = round(_s("reversal_prob"), 4)
        if "reversal_held" in mapped:
            mapped["reversal_held"] = round(_s("reversal_held"), 4)
        if "trend_continuation_prob" in mapped:
            mapped["trend_continuation_prob"] = round(_s("trend_continuation_prob"), 4)

        # ── 8. Zone timing / proximity ────────────────────────────────────────
        if "next_zone_bars" in mapped:
            eta = max(0, int(round(_s("next_zone_bars"))))
            mapped["next_zone_eta_bars"] = eta
            mapped["next_zone_bars"]     = eta   # frontend reads next_zone_bars
        if "next_zone_distance" in mapped:
            dist_val = max(0.0, float(_s("next_zone_distance")))
            mapped["next_zone_pct_away"]  = round(dist_val * 100, 3)
            mapped["next_zone_distance"]  = round(dist_val, 6)  # keep raw [0,1] ATR-norm

            # Reconstruct Target Zone Price (Method B: Continuous ATR Distance)
            atr_val = 15.0  # fallback ATR default
            if feature_window is not None and hasattr(feature_window, "columns"):
                for col in ["ATR_14", "atr", "ATR"]:
                    if col in feature_window.columns:
                        try:
                            s = pd.to_numeric(feature_window[col], errors="coerce").dropna()
                            if not s.empty:
                                atr_val = float(s.iloc[-1])
                                break
                        except Exception:
                            pass
            ref_c = float(feature_window["Close"].iloc[-1]) if feature_window is not None and "Close" in feature_window.columns else 0.0
            price_offset = dist_val * atr_val
            dir_label = mapped.get("direction_label", "Bull")
            if dir_label == "Bear":
                reconstructed_b = ref_c - price_offset
            else:
                reconstructed_b = ref_c + price_offset
            mapped["next_zone_price_continuous"] = round(reconstructed_b, 4)

        if "next_zone_idx" in mapped:
            # Reconstruct Target Zone Price (Method A: Discrete Zone Selection)
            idx_probs = np.squeeze(np.asarray(mapped["next_zone_idx"]))
            if idx_probs.ndim == 1 and len(idx_probs) > 0:
                target_slot = int(np.argmax(idx_probs))
                mapped["next_zone_target_slot"] = target_slot
                # Map slot index to active candidate zone if snr_features available
                if snr_features and isinstance(snr_features, dict) and "zones" in snr_features:
                    zones = snr_features.get("zones") or []
                    if 0 <= target_slot < len(zones):
                        mapped["next_zone_price_discrete"] = round(float(zones[target_slot].get("price", 0.0)), 4)

        if "next_zone_volume" in mapped:
            mapped["next_zone_volume_est"] = round(_s("next_zone_volume"), 4)
            mapped["next_zone_volume"]     = round(_s("next_zone_volume"), 4)

        # ── 9. Risk/reward ────────────────────────────────────────────────────
        if "mfe" in mapped and "mae" in mapped:
            mfe_v = _s("mfe")
            mae_v = _s("mae")
            mapped["risk_reward"] = round(float(mfe_v / (mae_v + 1e-6)), 3)

        # ── 10. zone_story — structured narrative for UI cards / LLM prompts ──
        zone_story: Dict[str, Any] = {}

        if "direction_label" in mapped:
            zone_story["direction"]          = mapped["direction_label"]
            zone_story["direction_net_pct"]  = mapped.get("direction_net", 0.0)
            zone_story["direction_conf_pct"] = mapped.get("direction_conf", 0.0)

        if "signal_class_name" in mapped and mapped.get("signal_class_name") != "no_signal":
            zone_story["zone_interaction"]     = mapped["signal_class_name"]
            zone_story["interaction_conf_pct"] = mapped.get("ensemble_confidence", 0.0)
        elif "ctx_signal_type" in mapped:
            zone_story["zone_interaction"]     = mapped["ctx_signal_type"]
            zone_story["interaction_conf_pct"] = mapped.get("ctx_signal_conf", 0.0)

        rev  = mapped.get("reversal_prob", None)
        cont = mapped.get("trend_continuation_prob", None)
        if rev  is not None: zone_story["reversal_prob_pct"]     = rev
        if cont is not None: zone_story["continuation_prob_pct"] = cont
        if rev  is not None and cont is not None:
            zone_story["likely_outcome"] = "REVERSAL" if rev > cont else "CONTINUATION"

        zone_story["bull_prob_pct"]    = round(_s("bull_prob") * 100, 1)
        zone_story["bull_strength_pct"]= round(mapped.get("bull_strength", 0.0) * 100, 1)
        zone_story["bear_strength_pct"]= round(mapped.get("bear_strength", 0.0) * 100, 1)

        if "next_zone_eta_bars" in mapped: zone_story["next_zone_eta_bars"] = mapped["next_zone_eta_bars"]
        if "next_zone_pct_away" in mapped: zone_story["next_zone_pct_away"] = mapped["next_zone_pct_away"]
        if "risk_reward" in mapped:        zone_story["risk_reward"] = mapped["risk_reward"]
        if "mfe" in mapped:                zone_story["mfe"] = round(_s("mfe"), 4)
        if "mae" in mapped:                zone_story["mae"] = round(_s("mae"), 4)

        # Consensus bias: ensemble + context model directional vote
        eb = mapped.get("ensemble_bias", "NEUTRAL")
        dl = mapped.get("direction_label", "Neutral")
        bull_votes = sum([eb == "BULL", dl == "Bull"])
        bear_votes = sum([eb == "BEAR", dl == "Bear"])
        zone_story["consensus_bias"] = (
            "BULL" if bull_votes > bear_votes else
            "BEAR" if bear_votes > bull_votes else "NEUTRAL"
        )

        mapped["zone_story"] = zone_story

        # ── 11. Serialize numpy → JSON-safe Python ────────────────────────────
        SEQUENCE_HEADS = {
            "close_sequence", "open_sequence", "high_sequence",
            "low_sequence", "volume_sequence",
            "support_trendline", "resist_trendline",
        }
        for k in list(mapped.keys()):
            v = mapped[k]
            if isinstance(v, np.ndarray):
                if v.ndim == 0:
                    mapped[k] = float(v)
                elif v.ndim == 1:
                    mapped[k] = v.tolist()
                else:
                    sq = v.squeeze()
                    mapped[k] = sq.tolist() if sq.ndim > 0 else float(sq)
            elif hasattr(v, "tolist") and not isinstance(v, (dict, str, list)):
                mapped[k] = v.tolist()

        return mapped
