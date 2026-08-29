"""
OutputDenormalizer — convert model head outputs to real-world values for the UI.

The model always consumes scaled/normalized inputs. Chart overlays, metric cards,
and forecast panels must receive prices, pips, and probabilities in human units.

Never apply input SelectiveScaler.inverse_transform to arbitrary output heads.
Use the persisted output_transform_spec from the serving contract instead.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class OutputDenormalizer:
    """Apply per-head inverse transforms defined in the serving contract."""

    def __init__(
        self,
        output_transform_spec: Dict[str, Dict[str, Any]],
        scaling_config: Optional[Dict[str, Any]] = None,
        *,
        reference_close: Optional[float] = None,
        pip_size: float = 0.0001,
    ):
        self.output_transform_spec = output_transform_spec or {}
        self.scaling_config = scaling_config or {}
        self.reference_close = reference_close
        self.pip_size = pip_size

    @classmethod
    def from_serving_contract(
        cls,
        contract: Dict[str, Any],
        *,
        reference_close: Optional[float] = None,
        pip_size: float = 0.0001,
    ) -> "OutputDenormalizer":
        output_contract = contract.get("output_contract", {})
        input_contract = contract.get("input_contract", {})
        scaling_config = input_contract.get("scaling_config", {})
        spec = (
            output_contract.get("output_transform_spec")
            or scaling_config.get("output_transform_spec")
            or {}
        )
        return cls(
            output_transform_spec=spec,
            scaling_config=scaling_config,
            reference_close=reference_close,
            pip_size=pip_size,
        )

    def _structural_range(self) -> Dict[str, Any]:
        return self.scaling_config.get("structural_range") or {}

    def _inverse_structural(self, normalized: Union[float, List[float]]) -> Union[float, List[float]]:
        """Map [0,1] normalized values back to price space using fitted structural range."""
        sr = self._structural_range()
        low = float(sr.get("low") or 0.0)
        width = float(sr.get("width") or 0.0)
        if width <= 0:
            logger.warning("[OutputDenormalizer] structural_range.width missing — returning normalized values")
            return normalized
        logger.info(
            "[OutputDenormalizer] structural_range inverse low=%s high=%s width=%s",
            low,
            sr.get("high"),
            width,
        )

        def _one(v: float) -> float:
            return low + float(v) * width

        if isinstance(normalized, list):
            return [_one(v) for v in normalized]
        return _one(float(normalized))

    def _inverse_rolling_sigmoid_structural(
        self,
        normalized: Union[float, List[float]],
        *,
        head_name: Optional[str] = None,
    ) -> Union[float, List[float]]:
        """
        Approximate inverse for OHLC targets trained after rolling-mean sigmoid normalization.

        Training used:
            raw_norm = (price - structural_low) / structural_width
            y = sigmoid(((raw_norm - rolling_mean) / abs(rolling_mean)) * k)

        Runtime reconstructs the live rolling-mean anchor from the raw OHLC
        feature_window and passes it through scaling_config["ohlc_rolling_mean_anchors"].
        If unavailable, fall back to the current close's structural position.
        """
        sr = self._structural_range()
        low = float(sr.get("low") or 0.0)
        width = float(sr.get("width") or 0.0)
        if width <= 0 or self.reference_close is None:
            return self._inverse_structural(normalized)

        scale = float(self.scaling_config.get("sigmoid_scale_factor") or 2.0)
        if scale <= 0:
            scale = 2.0

        anchors = self.scaling_config.get("ohlc_rolling_mean_anchors") or {}
        anchor = anchors.get(head_name) if head_name else None
        if anchor is None:
            anchor = (float(self.reference_close) - low) / width
            anchor_source = "reference_close"
        else:
            anchor_source = "live_ohlc_rolling_mean"
        if not np.isfinite(anchor):
            anchor = 0.5
        anchor = max(1e-6, float(anchor))

        logger.info(
            "[OutputDenormalizer] rolling-sigmoid structural inverse low=%s high=%s width=%s anchor=%s anchor_source=%s scale=%s",
            low,
            sr.get("high"),
            width,
            anchor,
            anchor_source,
            scale,
        )

        def _one(v: float) -> float:
            y = min(1.0 - 1e-7, max(1e-7, float(v)))
            deviation_ratio = float(np.log(y / (1.0 - y)) / scale)
            raw_norm = anchor + deviation_ratio * (abs(anchor) + 1e-8)
            return low + raw_norm * width

        if isinstance(normalized, list):
            return [_one(v) for v in normalized]
        return _one(float(normalized))

    def _inverse_range_fraction(self, value: Union[float, List[float]]) -> Union[float, List[float]]:
        """Map range-fraction regression outputs back to price/pip units."""
        sr = self._structural_range()
        width = float(sr.get("width") or 0.0)
        if width <= 0:
            return value

        def _one(v: float) -> float:
            # Fraction of range → absolute price delta
            return float(v) * width

        if isinstance(value, list):
            return [_one(v) for v in value]
        return _one(float(value))

    def _return_fraction_to_points(self, value: Union[float, List[float]]) -> Union[float, List[float]]:
        """Map return-fraction outputs like MFE/MAE back to price points."""
        ref = self.reference_close
        if ref is None:
            return value

        def _one(v: float) -> float:
            return float(v) * float(ref)

        if isinstance(value, list):
            return [_one(v) for v in value]
        return _one(float(value))

    def _inverse_volume_range(self, normalized: Union[float, List[float]]) -> Union[float, List[float]]:
        """Map normalized volume values back through the fitted TickVolume/Volume range."""
        volume_cfg = self.scaling_config.get("volume_range") or {}
        range_cfg = volume_cfg.get("TickVolume") or volume_cfg.get("Volume") or {}
        try:
            low = float(range_cfg.get("fitted_range_low", 0.0) or 0.0)
            high = float(range_cfg.get("fitted_range_high") or 0.0)
        except (TypeError, ValueError):
            return normalized
        width = high - low
        if width <= 0:
            return normalized
        logger.info(
            "[OutputDenormalizer] volume_range inverse low=%s high=%s width=%s",
            low,
            high,
            width,
        )

        def _one(v: float) -> float:
            return low + float(v) * width

        if isinstance(normalized, list):
            return [_one(v) for v in normalized]
        return _one(float(normalized))

    def denormalize_head(self, head_name: str, value: Any) -> Any:
        """Denormalize a single head value."""
        rule = self.output_transform_spec.get(head_name, {"method": "identity"})
        method = rule.get("method", "identity")

        if value is None:
            return None

        # Convert numpy arrays to Python lists/scalars
        if hasattr(value, "tolist"):
            value = value.tolist()

        if method == "identity":
            logger.info("[OutputDenormalizer] head='%s' method='identity' (passthrough)", head_name)
            if head_name in ("next_zone_bars", "next_zone_distance"):
                if isinstance(value, (int, float)):
                    return max(0, value) if head_name == "next_zone_bars" else max(0.0, float(value))
                elif isinstance(value, list) and value:
                    return [max(0, x) if head_name == "next_zone_bars" else max(0.0, float(x)) for x in value]
            return value
        if method == "structural_range_inverse":
            has_sr = bool(self.scaling_config and self.scaling_config.get("structural_range"))
            norm_m = self.scaling_config.get("normalization_method")
            if has_sr or norm_m in (None, "rolling_mean_sigmoid", "structural_range"):
                method = "rolling_sigmoid_structural_inverse"

        # Trendlines outputs from model are diff/offset fractions relative to Close
        if head_name in ("support_trendline", "resist_trendline"):
            method = "range_fraction_inverse"

        if method == "structural_range_inverse":
            logger.info(
                "[OutputDenormalizer] head='%s' method='structural_range_inverse' ref_close=%s",
                head_name,
                self.reference_close,
            )
            return self._inverse_structural(value)
        if method == "rolling_sigmoid_structural_inverse":
            logger.info(
                "[OutputDenormalizer] head='%s' method='rolling_sigmoid_structural_inverse' ref_close=%s",
                head_name,
                self.reference_close,
            )
            return self._inverse_rolling_sigmoid_structural(value, head_name=head_name)
        if method == "range_fraction_inverse":
            logger.info("[OutputDenormalizer] head='%s' method='range_fraction_inverse'", head_name)
            return self._inverse_range_fraction(value)
        if method == "return_fraction_to_points":
            logger.info("[OutputDenormalizer] head='%s' method='return_fraction_to_points' ref_close=%s", head_name, self.reference_close)
            return self._return_fraction_to_points(value)
        if method == "volume_range_inverse":
            logger.info("[OutputDenormalizer] head='%s' method='volume_range_inverse'", head_name)
            return self._inverse_volume_range(value)
        if method == "pip_from_close":
            # value in pips → price offset from reference close
            ref = self.reference_close
            if ref is None:
                logger.info("[OutputDenormalizer] head='%s' method='pip_from_close' missing reference_close; passthrough", head_name)
                return value
            pip = float(rule.get("pip_size", self.pip_size))
            logger.info("[OutputDenormalizer] head='%s' method='pip_from_close' ref_close=%s pip_size=%s", head_name, ref, pip)
            if isinstance(value, list):
                return [ref + float(v) * pip for v in value]
            return ref + float(value) * pip

        logger.warning("[OutputDenormalizer] Unknown method '%s' for head '%s' — identity passthrough", method, head_name)
        return value

    def denormalize_all(self, named_heads: Dict[str, Any]) -> Dict[str, Any]:
        """Denormalize all model output heads for UI consumption."""
        logger.info(
            "[OutputDenormalizer] Denormalizing %d heads using %d transform rules",
            len(named_heads),
            len(self.output_transform_spec),
        )
        denormalized = {
            head: self.denormalize_head(head, val)
            for head, val in named_heads.items()
        }
        return self._add_human_readable_aliases(denormalized)

    @staticmethod
    def _scalar(value: Any) -> Optional[float]:
        try:
            if isinstance(value, bool):
                return None
            if isinstance(value, (int, float)):
                return float(value)
            if hasattr(value, "tolist"):
                value = value.tolist()
            if isinstance(value, list) and len(value) == 1:
                return float(value[0])
        except (TypeError, ValueError):
            return None
        return None

    def _add_human_readable_aliases(self, heads: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add UI-friendly companions without removing model-native fields.

        Probabilities often need both forms:
          bull_prob=0.8069 for threshold logic, bull_prob_pct=80.69 for display.
        """
        out = dict(heads)

        probability_like = {
            "bull_prob", "bull_conf", "bear_conf", "bull_strength", "bear_strength",
            "reversal_prob", "trend_continuation_prob", "reversal_held",
            "signal_strength", "signal_class_conf",
            "signal_bounce_support", "signal_bounce_resistance",
            "signal_breakout_support", "signal_breakout_resistance",
            "Signal_bounce_support", "Signal_bounce_resistance",
            "Signal_breakout_support", "Signal_breakout_resistance",
            "vol_surge", "ensemble_confidence",
        }
        already_percent = {
            "direction_conf", "direction_net", "ctx_signal_conf",
        }

        for key in probability_like:
            if key not in out:
                continue
            scalar = self._scalar(out[key])
            if scalar is None:
                continue
            pct_value = float(scalar) * 100.0 if -1.0 <= float(scalar) <= 1.0 else float(scalar)
            out[f"{key}_pct"] = round(min(100.0, max(0.0, pct_value)), 2)

        for key in already_percent:
            if key not in out:
                continue
            scalar = self._scalar(out[key])
            if scalar is not None:
                out[f"{key}_pct"] = round(float(scalar), 2)

        if "next_zone_distance" in out:
            scalar = self._scalar(out["next_zone_distance"])
            if scalar is not None:
                out["next_zone_distance_points"] = round(float(scalar), 5)
                if self.reference_close:
                    out["next_zone_pct_away"] = round((float(scalar) / self.reference_close) * 100.0, 4)
                out["next_zone_direction"] = "above" if scalar >= 0 else "below"

        for key in ["next_zone_bars", "next_zone_eta_bars"]:
            scalar = self._scalar(out.get(key))
            if scalar is not None and scalar < 0:
                out[f"{key}_display"] = None
                out[f"{key}_status"] = "not_applicable"

        if "mfe" in out:
            scalar = self._scalar(out["mfe"])
            if scalar is not None:
                out["mfe_points"] = round(float(scalar), 5)
                if self.reference_close:
                    out["mfe_pct"] = round((float(scalar) / self.reference_close) * 100.0, 4)

        if "mae" in out:
            scalar = self._scalar(out["mae"])
            if scalar is not None:
                out["mae_points"] = round(float(scalar), 5)
                if self.reference_close:
                    out["mae_pct"] = round((float(scalar) / self.reference_close) * 100.0, 4)

        mfe_scalar = self._scalar(out.get("mfe"))
        mae_scalar = self._scalar(out.get("mae"))
        if mfe_scalar is not None and mae_scalar is not None and abs(mae_scalar) > 1e-9:
            if "risk_reward" in out and "risk_reward_model" not in out:
                out["risk_reward_model"] = out["risk_reward"]
            out["risk_reward"] = round(abs(float(mfe_scalar)) / abs(float(mae_scalar)), 3)

        story = out.get("zone_story")
        if isinstance(story, dict):
            story = dict(story)
            for key in ["reversal_prob_pct", "continuation_prob_pct"]:
                scalar = self._scalar(story.get(key))
                if scalar is not None and -1.0 <= scalar <= 1.0:
                    story[key] = round(min(100.0, max(0.0, scalar * 100.0)), 2)
            for raw_key, pct_key in [
                ("bull_prob", "bull_prob_pct"),
                ("bull_strength", "bull_strength_pct"),
                ("bear_strength", "bear_strength_pct"),
            ]:
                if pct_key not in story and raw_key in out:
                    scalar = self._scalar(out[raw_key])
                    if scalar is not None:
                        story[pct_key] = round(scalar * 100.0, 2)
            out["zone_story"] = story

        open_seq = out.get("open_sequence")
        high_seq = out.get("high_sequence")
        low_seq = out.get("low_sequence")
        close_seq = out.get("close_sequence")
        if all(isinstance(seq, list) for seq in [open_seq, high_seq, low_seq, close_seq]):
            n = min(len(open_seq), len(high_seq), len(low_seq), len(close_seq))
            volume_seq = out.get("volume_sequence") if isinstance(out.get("volume_sequence"), list) else None
            candles = []
            for i in range(n):
                candle = {
                    "step": i + 1,
                    "open": round(float(open_seq[i]), 5),
                    "high": round(float(high_seq[i]), 5),
                    "low": round(float(low_seq[i]), 5),
                    "close": round(float(close_seq[i]), 5),
                }
                if volume_seq is not None and i < len(volume_seq):
                    candle["volume"] = round(float(volume_seq[i]), 2)
                candles.append(candle)
            out["ohlcv_forecast"] = candles

        return out
