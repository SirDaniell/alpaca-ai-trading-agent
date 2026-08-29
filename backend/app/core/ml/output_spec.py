"""
output_spec.py
══════════════════════════════════════════════════════════════════════════
Single source of truth for V8.3 multi-output model head definitions.

Imported by:
 - compile_model_v8_hybrid()  (baseline_encoder_v8_hybrid.py)
 - _trainer_fit / split_output_loss  (analysis_manager_training_mixin.py)
 - execute_model_training            (analysis_manager_ml_mixin.py)
 - LazySequenceGenerator            (ml_data_loader.py)

The key invariant: if the same frozenset and the same loss_spec are used
at compile time and at train time, core_loss / full_loss will always agree
with what gradient descent is actually optimising.

SYNC CHECKLIST — update ALL THREE sections when the model adds/removes heads:
  1. V8_3_CORE_OUTPUT_KEYS  — drives weight rollback / checkpoint selection
  2. V8_3_ALL_OUTPUT_KEYS   — must be a superset of compile_model's loss_dict keys
  3. V8_3_NPZ_TARGET_KEY_MAP — raw NPZ column (no "target_" prefix) → model output name
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np

# ──────────────────────────────────────────────────────────────────────────
# CORE OUTPUT KEYS
# These are the heads whose loss drives every weight-affecting decision:
#   - batch-level jury rollback
#   - epoch-level jury rollback
#   - checkpoint selection (which epoch's weights ship as the product)
#
# Everything NOT in this set still trains normally and is still logged —
# it just doesn't vote on "should we revert weights" or "is this epoch
# the best one."
# ──────────────────────────────────────────────────────────────────────────
V8_3_CORE_OUTPUT_KEYS: frozenset = frozenset({
    "main_output",
    "open_sequence",
    "high_sequence",
    "low_sequence",
    "volume_sequence",
    "bull_class",
    "Signal_bounce_support",
    "Signal_bounce_resistance",
    "Signal_breakout_support",
    "Signal_breakout_resistance",
})

# ──────────────────────────────────────────────────────────────────────────
# ALL OUTPUT KEYS (ordered to match compile_model_v8_hybrid loss_dict)
#
# Must include EVERY name that model.output_names produces.
# Missing a name here means LazySequenceGenerator won't activate multi_output
# mode for datasets that only produce that subset of targets.
# ──────────────────────────────────────────────────────────────────────────
V8_3_ALL_OUTPUT_KEYS: Tuple[str, ...] = (
    # TIER 0 — OHLCV
    "main_output", "open_sequence",
    "high_sequence", "low_sequence", "volume_sequence", "ohlcv_sequence",
    # TIER 1 — Signal classification
    "Signal_bounce_support", "Signal_bounce_resistance",
    "Signal_breakout_support", "Signal_breakout_resistance",
    "bull_class", "bull_conf", "bear_conf",
    # TIER 2 — Trade quality & probability
    "mfe", "mae", "risk_reward", "signal_strength",
    "time_to_max_favorable", "time_to_max_adverse",
    "bull_prob", "bull_strength", "bear_strength",
    # TIER 2b — Reversal / trend continuation (added in V8.3)
    "reversal_prob", "trend_continuation_prob", "reversal_held",
    # TIER 2c — Next zone prediction (liquidity zones)
    "next_zone_idx", "next_zone_bars", "next_zone_distance", "next_zone_volume",
    # TIER 3 — Encoder regularization
    "aux_momentum", "aux_volatility", "aux_drift",
    "vol_surge", "aux_output_1", "aux_output_2", "aux_output_3",
    # TIER 4 — Structural levels
    "support_trendline_next", "resist_trendline_next",
    # TIER 5 — Temporal probes (diagnostic)
    "probe_hour", "probe_session", "probe_day_of_week",
)

# ──────────────────────────────────────────────────────────────────────────
# NPZ KEY → MODEL OUTPUT NAME MAPPING
#
# Direction: raw NPZ column name (WITHOUT "target_" prefix) → model output name
#
# ml_dataset_preparation writes chunks as "target_{column_name}".
# LazySequenceGenerator strips the "target_" prefix before looking up here.
#
# Rules for updating:
#  - If ml_dataset_preparation renames a target column, update the LHS key.
#  - If build_model_v8_hybrid renames an output head, update the RHS value.
#  - If the horizon suffix changes (e.g. _8 → _12), update the LHS key.
# ──────────────────────────────────────────────────────────────────────────
V8_3_NPZ_TARGET_KEY_MAP: Dict[str, str] = {
    # TIER 0 — OHLCV sequences
    "future_sequence":          "main_output",        # primary close-price sequence
    "adv_target_Open_seq":      "open_sequence",
    "adv_target_High_seq":      "high_sequence",
    "adv_target_Low_seq":       "low_sequence",
    "adv_target_Volume_seq":    "volume_sequence",

    # TIER 1 — Signal classification (columns written by signal_generator.py)
    "Signal_bounce_support":    "Signal_bounce_support",
    "Signal_bounce_resistance": "Signal_bounce_resistance",
    "Signal_breakout_support":  "Signal_breakout_support",
    "Signal_breakout_resistance": "Signal_breakout_resistance",
    "adv_target_bull_class":    "bull_class",
    "adv_target_bull_conf":     "bull_conf",
    "adv_target_bear_conf":     "bear_conf",

    # TIER 2 — Trade quality & probability
    "adv_target_bull_prob":             "bull_prob",
    # ⚠️ Horizon suffix: ml_dataset_preparation uses _8 (8-bar forward window).
    # If you change the window in the pipeline, update both keys here.
    "adv_target_bull_strength_8":       "bull_strength",
    "adv_target_bear_strength_8":       "bear_strength",
    "adv_target_MFE":                   "mfe",
    "adv_target_MAE":                   "mae",
    "adv_target_max_favorable_pct":     "mfe",
    "adv_target_max_adverse_pct":       "mae",
    "adv_target_risk_reward_ratio":     "risk_reward",
    "adv_target_time_to_max_favorable": "time_to_max_favorable",
    "adv_target_time_to_max_adverse":   "time_to_max_adverse",
    "adv_target_signal_strength":       "signal_strength",

    # TIER 2b — Reversal / trend continuation
    "adv_target_reversal_prob":             "reversal_prob",
    "adv_target_trend_continuation_prob":   "trend_continuation_prob",
    "adv_target_reversal_held":             "reversal_held",

    # TIER 3 — Encoder regularization (regime indicators, next-bar)
    "adv_target_MOM_t_next":    "aux_momentum",
    "adv_target_ATR_next":      "aux_volatility",
    "adv_target_TF_t_next":     "aux_drift",
    "adv_target_volatility_surge": "vol_surge",
    # aux_output_1/2/3 are encoder regularizers that reuse the main close
    # sequence. The loader expands the single future_sequence target into
    # these output names after the normal one-to-one map pass.

    # TIER 4 — Structural levels (next-bar trendline & SNR distances)
    "adv_target_Support_Trendline_next":        "support_trendline_next",
    "adv_target_Resist_Trendline_next":         "resist_trendline_next",
    # TIER 5 — Next zone prediction (liquidity zones)
    "adv_target_next_zone_idx":         "next_zone_idx",
    "adv_target_next_zone_bars":        "next_zone_bars",
    "adv_target_next_zone_distance":    "next_zone_distance",
    "adv_target_next_zone_volume":      "next_zone_volume",

    # TIER 6 — Temporal probes. These remain sparse integer labels in the
    # loader, not one-hot arrays, because the models compile them with
    # sparse_categorical_crossentropy.
    "adv_target_hour_next":         "probe_hour",
    "adv_target_session_next":      "probe_session",
    "adv_target_day_of_week_next":  "probe_day_of_week",
}


def is_multi_output_model(model) -> bool:
    """Return True if the model's output is a dict (multi-output)."""
    try:
        names = getattr(model, "output_names", None)
        if names and len(names) > 1:
            return True
        return False
    except Exception:
        return False


def get_loss_spec_from_model(model) -> Optional[Tuple[Dict, Dict]]:
    """
    Extract (loss_dict, loss_weights) from a compiled Keras model.

    Tries three introspection paths in order:
      1. model.loss        — raw dict if user passed dict to compile()
      2. model.compiled_loss._losses  — Keras 3 CompileLoss internals
      3. None              — caller must use the shared-spec refactor path

    Returns (loss_dict, loss_weights) or (None, None) if not extractable.
    """
    loss_dict = None
    loss_weights = None

    # Path 1: model.loss is the raw dict passed to compile()
    raw_loss = getattr(model, "loss", None)
    if isinstance(raw_loss, dict):
        loss_dict = raw_loss

    # Path 2: Keras 3 CompileLoss object
    if loss_dict is None:
        compiled_loss = getattr(model, "compiled_loss", None)
        if compiled_loss is not None:
            losses_attr = getattr(compiled_loss, "_losses", None)
            if isinstance(losses_attr, dict):
                loss_dict = losses_attr

    # Loss weights
    raw_weights = getattr(model, "loss_weights", None)
    if isinstance(raw_weights, dict):
        loss_weights = raw_weights
    else:
        compiled_loss = getattr(model, "compiled_loss", None)
        if compiled_loss is not None:
            weights_attr = getattr(compiled_loss, "_loss_weights", None)
            if isinstance(weights_attr, dict):
                loss_weights = weights_attr

    return loss_dict, loss_weights


def split_output_loss(
    model,
    y_true: dict,
    y_pred: dict,
    loss_spec: Optional[Tuple[Dict, Dict]] = None,
    core_keys: frozenset = V8_3_CORE_OUTPUT_KEYS,
    import_tf=None,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Compute (core_loss, full_loss, per_head_losses) for a multi-output model.

    Uses the SAME per-output loss functions and weights the model was compiled
    with — pulled directly from the model via get_loss_spec_from_model(), so
    this can never disagree with what gradient descent is actually optimising.

    Args:
        model:       Compiled Keras model.
        y_true:      Dict of {output_name: target_array} — must match model outputs.
        y_pred:      Dict of {output_name: prediction_array} — model's forward pass.
        loss_spec:   Optional pre-fetched (loss_dict, loss_weights). If None,
                     extracted from model at call time (use pre-fetching in loops).
        core_keys:   Which output names count toward core_loss.
        import_tf:   Pass tf module to avoid re-importing inside @tf.function.

    Returns:
        core_loss:      float — weighted loss over core_keys only.
        full_loss:      float — weighted loss over ALL heads.
        per_head_losses: dict[str, float] — each head's individual weighted loss.

    Raises:
        RuntimeError if loss_spec cannot be extracted and no fallback is possible.
    """
    if import_tf is None:
        import tensorflow as tf
    else:
        tf = import_tf

    if loss_spec is None:
        loss_spec = get_loss_spec_from_model(model)

    loss_dict, loss_weights = loss_spec if loss_spec else (None, None)

    if loss_dict is None:
        raise RuntimeError(
            "split_output_loss: could not extract loss_dict from model. "
            "Ensure model is compiled with a dict loss, or pass loss_spec explicitly. "
            "See output_spec.py get_loss_spec_from_model() for supported paths."
        )

    if loss_weights is None:
        loss_weights = {k: 1.0 for k in loss_dict}

    per_head_losses: Dict[str, float] = {}
    core_loss_sum = 0.0
    full_loss_sum = 0.0

    for head_name, loss_fn in loss_dict.items():
        weight = loss_weights.get(head_name, 1.0)
        if weight == 0.0 or loss_fn is None:
            per_head_losses[head_name] = 0.0
            continue

        y_t = y_true.get(head_name)
        y_p = y_pred.get(head_name)

        if y_t is None or y_p is None:
            per_head_losses[head_name] = 0.0
            continue

        try:
            y_t_tf = tf.cast(tf.constant(y_t) if not hasattr(y_t, 'numpy') else y_t, tf.float32)
            y_p_tf = tf.cast(y_p, tf.float32)

            if callable(loss_fn):
                head_loss = float(tf.reduce_mean(loss_fn(y_t_tf, y_p_tf)))
            elif loss_fn == "mse":
                head_loss = float(tf.reduce_mean(tf.square(y_t_tf - y_p_tf)))
            elif loss_fn == "mae":
                head_loss = float(tf.reduce_mean(tf.abs(y_t_tf - y_p_tf)))
            elif loss_fn in ("binary_crossentropy",):
                bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
                head_loss = float(bce(y_t_tf, y_p_tf))
            elif loss_fn in ("categorical_crossentropy",):
                cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
                head_loss = float(cce(y_t_tf, y_p_tf))
            elif loss_fn in ("sparse_categorical_crossentropy",):
                scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
                head_loss = float(scce(y_t_tf, y_p_tf))
            else:
                per_head_losses[head_name] = 0.0
                continue

            weighted = head_loss * weight
            per_head_losses[head_name] = weighted
            full_loss_sum += weighted
            if head_name in core_keys:
                core_loss_sum += weighted

        except Exception:
            per_head_losses[head_name] = 0.0

    return core_loss_sum, full_loss_sum, per_head_losses
