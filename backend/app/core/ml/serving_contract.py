"""
Serving contract builders — single source of truth for training/inference parity.

A serving contract captures everything needed to:
  1. Reconstruct the exact feature tensor (input_contract)
  2. Run model.predict on scaled data
  3. Convert outputs back to real-world values for the UI (output_contract)

Used by ProcessingManager (write at ML prep), analysis_manager (snapshot at training),
FeatureServingPipeline (read at inference), and ModelTrainingStepPanel (export).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

BUNDLE_VERSION = "1.0"

# Heads that are already in display-ready units — no inverse transform.
_IDENTITY_HEADS = frozenset({
    "bull_class", "bull_prob", "bear_conf", "bull_conf",
    "bull_strength", "bear_strength",
    "class_output", "direction_probs",
    "Signal_bounce_support", "Signal_bounce_resistance",
    "Signal_breakout_support", "Signal_breakout_resistance",
    "reversal_prob", "trend_continuation_prob", "reversal_held",
    "signal_bounce_support", "signal_bounce_resistance",
    "signal_breakout_support", "signal_breakout_resistance",
    "signal_class", "signal_class_conf", "signal_strength", "direction_class",
    "direction_conf", "direction_net", "ctx_signal_conf",
    "ensemble_confidence", "vol_surge",
    "risk_reward",
    "next_zone_idx", "next_zone_bars", "next_zone_distance", "next_zone_volume",
    "next_zone_eta_bars", "next_zone_pct_away", "next_zone_volume_est",
    "probe_hour", "probe_session", "probe_day_of_week",
})

# OHLCV sequence heads normalized with structural range during training.
_STRUCTURAL_SEQUENCE_HEADS = frozenset({
    "main_output", "close_sequence", "open_sequence",
    "high_sequence", "low_sequence", "ohlcv_sequence",
})

# Regression heads stored as fractions of structural range or pip-scaled targets.
_RANGE_FRACTION_HEADS = frozenset({
    "support_trendline", "resist_trendline",
    "support_trendline_next", "resist_trendline_next",
    "snr_nearest_support_next", "snr_nearest_resistance_next",
    "snr_support_distance_next", "snr_resist_distance_next",
    "aux_momentum", "aux_volatility", "aux_drift",
    "aux_output_1", "aux_output_2", "aux_output_3",
})

# MFE/MAE targets are generated as return fractions relative to current Close.
_RETURN_FRACTION_HEADS = frozenset({
    "mfe", "mae",
})


def build_output_transform_spec(
    target_names: Optional[List[str]] = None,
    scaling_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Build per-output-head inverse transform rules for UI denormalization.

    Rules are inferred from head naming conventions aligned with output_spec.py
    and ProcessingManager structural-range normalization.
    """
    spec: Dict[str, Dict[str, Any]] = {}
    names = list(target_names or [])

    # Also include known model output names from output_spec if no targets passed
    if not names:
        try:
            from app.core.ml.output_spec import V8_3_ALL_OUTPUT_KEYS
            names = list(V8_3_ALL_OUTPUT_KEYS)
        except ImportError:
            names = []

    structural_ref = "structural_range"
    if scaling_config and scaling_config.get("structural_range"):
        structural_ref = "scaling_config.structural_range"

    for head in names:
        if head in _IDENTITY_HEADS or head.endswith("_class"):
            spec[head] = {"method": "identity"}
        elif head in _STRUCTURAL_SEQUENCE_HEADS or head.endswith("_sequence"):
            if head == "volume_sequence":
                spec[head] = {
                    "method": "volume_range_inverse",
                    "ref": "scaling_config.volume_range",
                    "note": "Volume sequence values normalized by fitted volume range",
                }
                continue
            has_sr = bool(scaling_config and scaling_config.get("structural_range"))
            norm_method = scaling_config.get("normalization_method") if scaling_config else None
            use_sigmoid = has_sr or norm_method in (None, "rolling_mean_sigmoid", "structural_range")
            spec[head] = {
                "method": (
                    "rolling_sigmoid_structural_inverse"
                    if use_sigmoid
                    else "structural_range_inverse"
                ),
                "ref": structural_ref,
                "note": "Sequence values normalized by fitted structural range, with rolling-sigmoid inverse",
            }
        elif head in _RETURN_FRACTION_HEADS:
            spec[head] = {
                "method": "return_fraction_to_points",
                "ref": "reference_close",
                "note": "Return fraction target → price points via reference close",
            }
        elif head in _RANGE_FRACTION_HEADS or head.startswith("adv_target_"):
            spec[head] = {
                "method": "range_fraction_inverse",
                "ref": structural_ref,
                "note": "Regression head as fraction of structural range width",
            }
        else:
            spec[head] = {
                "method": "identity",
                "note": "Unknown head — pass through; update spec after parity probe",
            }

    return spec


def build_session_provenance(
    session_row: Any,
    *,
    dataset_name: str,
    dataset_id: str,
    step_configs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build provenance block from a DataSession ORM row or dict."""
    if session_row is None:
        return {
            "dataset_name": dataset_name,
            "dataset_id": dataset_id,
            "step_configs": step_configs or {},
        }

    def _get(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    return {
        "session_id": str(_get(session_row, "session_id", "")),
        "symbol": _get(session_row, "symbol"),
        "timeframe": _get(session_row, "timeframe"),
        "start_date": _get(session_row, "start_date"),
        "end_date": _get(session_row, "end_date"),
        "data_source": _get(session_row, "data_source"),
        "record_count": _get(session_row, "record_count"),
        "dataset_name": dataset_name,
        "dataset_id": dataset_id,
        "step_configs": step_configs or {},
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }


def build_input_contract(
    *,
    scaling_config: Dict[str, Any],
    split_config: Dict[str, Any],
    feature_columns: List[str],
    step_configs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Input-side contract: everything needed to build the model tensor."""
    feature_index_map = scaling_config.get("feature_index_map", {})
    return {
        "sequence_length": int(
            split_config.get("sequence_length")
            or split_config.get("window_size")
            or 60
        ),
        "prediction_length": int(split_config.get("prediction_length", 1)),
        "feature_count": len(feature_columns),
        "feature_columns": list(feature_columns),
        "feature_index_map": feature_index_map,
        "columns_to_scale": list(scaling_config.get("columns_to_scale", [])),
        "scaling_config": scaling_config,
        "split_config": split_config,
        "step_configs": step_configs or {},
    }


def build_output_contract(
    *,
    target_names: Optional[List[str]] = None,
    scaling_config: Optional[Dict[str, Any]] = None,
    output_transform_spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Output-side contract: head names + inverse transform rules."""
    transform = output_transform_spec or build_output_transform_spec(
        target_names=target_names,
        scaling_config=scaling_config,
    )
    return {
        "target_names": list(target_names or []),
        "output_transform_spec": transform,
    }


def build_serving_contract(
    *,
    model_id: Optional[str] = None,
    provenance: Dict[str, Any],
    scaling_config: Dict[str, Any],
    split_config: Dict[str, Any],
    feature_columns: List[str],
    target_names: Optional[List[str]] = None,
    step_configs: Optional[Dict[str, Any]] = None,
    output_transform_spec: Optional[Dict[str, Any]] = None,
    custom_objects: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Assemble the full serving contract bundle."""
    out_spec = output_transform_spec or build_output_transform_spec(
        target_names=target_names,
        scaling_config=scaling_config,
    )
    # Ensure scaling_config carries output_transform_spec for downstream readers
    scaling_with_output = dict(scaling_config)
    scaling_with_output["output_transform_spec"] = out_spec

    return {
        "bundle_version": BUNDLE_VERSION,
        "model_id": model_id,
        "provenance": provenance,
        "input_contract": build_input_contract(
            scaling_config=scaling_with_output,
            split_config=split_config,
            feature_columns=feature_columns,
            step_configs=step_configs or provenance.get("step_configs"),
        ),
        "output_contract": build_output_contract(
            target_names=target_names,
            scaling_config=scaling_with_output,
            output_transform_spec=out_spec,
        ),
        "artifacts": {
            "scaler": "ml_datasets.scaler_binary",
            "model": "trained_models_analysis.model_binary",
            "custom_objects": custom_objects or [],
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
