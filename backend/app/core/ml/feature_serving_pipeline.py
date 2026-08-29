"""
FeatureServingPipeline — training-identical feature reconstruction for inference.

Loads the serving contract from DB (primary) or disk bundle (proprietary fallback),
replays upstream enrichment from step_configs, applies PM normalization from
scaling_config, then SelectiveScaler.transform to produce the model input tensor.

Phase 3 skeleton: contract loading + strict validation. Full PM replay wiring
is Phase 4 (requires hydrating ProcessingManager fitted state from scaling_config).
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from app.core.ml.output_denormalizer import OutputDenormalizer
from app.core.ml.serving_contract import build_serving_contract

logger = logging.getLogger(__name__)


class ServingContractError(ValueError):
    """Raised when the serving contract is incomplete or inconsistent."""

    def __init__(self, message: str, *, missing_features: Optional[List[str]] = None):
        super().__init__(message)
        self.missing_features = missing_features or []


class FeatureServingPipeline:
    """
    Training-identical serving pipeline.

    Usage (after fresh pipeline run with serving contract stored):

        pipeline = FeatureServingPipeline.from_db(dataset_id, model_id)
        tensor, meta = pipeline.build_input_tensor(enriched_df)
        raw_heads = model.predict(tensor)
        ui_heads = pipeline.denormalize_outputs(raw_heads, reference_close=last_close)
    """

    def __init__(self, contract: Dict[str, Any], scaler: Any):
        self.contract = contract
        self.scaler = scaler
        self._input = contract.get("input_contract", {})
        self._output = contract.get("output_contract", {})
        self.scaling_config = self._input.get("scaling_config", {})
        self.sequence_length = int(self._input.get("sequence_length", 60))
        self.feature_columns = list(self._input.get("feature_columns", []))
        self.columns_to_scale = list(self._input.get("columns_to_scale", []))
        self.feature_index_map = self._input.get("feature_index_map", {})
        self.step_configs = self._input.get("step_configs", {})

    @staticmethod
    def _ordered_features(feature_index_map: Dict[str, Any]) -> List[str]:
        if not feature_index_map:
            return []
        ordered = [name for name, _ in sorted(feature_index_map.items(), key=lambda x: int(x[1]))]
        positions = [int(feature_index_map[name]) for name in ordered]
        if positions != list(range(len(positions))):
            raise ServingContractError("feature_index_map is not contiguous and zero-based.")
        return ordered

    @classmethod
    def from_contract_dict(cls, contract: Dict[str, Any], scaler_bytes: bytes) -> "FeatureServingPipeline":
        if not contract:
            raise ServingContractError("Empty serving contract.")
        if not scaler_bytes:
            raise ServingContractError("Scaler binary is required.")
        scaler = joblib.load(io.BytesIO(scaler_bytes))
        return cls(contract=contract, scaler=scaler)

    @classmethod
    def from_serving_contract_json(
        cls,
        contract_path: Path,
        scaler_path: Path,
    ) -> "FeatureServingPipeline":
        contract = json.loads(contract_path.read_text())
        scaler_bytes = scaler_path.read_bytes()
        return cls.from_contract_dict(contract, scaler_bytes)

    def validate_feature_window(self, df: pd.DataFrame) -> List[str]:
        """Return list of missing scaler columns (strict — no zero-fill)."""
        required = self.columns_to_scale or self.feature_columns
        return [c for c in required if c not in df.columns]

    def apply_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply persisted SelectiveScaler to the scaler column set."""
        scale_cols = self.columns_to_scale
        if not scale_cols:
            raise ServingContractError("Serving contract has no columns_to_scale.")

        missing = [c for c in scale_cols if c not in df.columns]
        if missing:
            raise ServingContractError(
                f"Live enrichment missing {len(missing)} scaler columns.",
                missing_features=missing,
            )

        scaled = self.scaler.transform(df[scale_cols])
        return pd.DataFrame(scaled, index=df.index, columns=scale_cols)

    def build_input_tensor(
        self,
        scaled_df: pd.DataFrame,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Select sequence features in contract order and reshape to (1, seq_len, n_features).

        Expects scaled_df to already have passed PM normalization + SelectiveScaler.
        """
        seq_features = self._ordered_features(self.feature_index_map)
        if not seq_features:
            seq_features = list(getattr(self.scaler, "sequence_feature_names", []) or self.feature_columns)

        missing = [c for c in seq_features if c not in scaled_df.columns]
        if missing:
            raise ServingContractError(
                f"Scaled frame missing {len(missing)} sequence features.",
                missing_features=missing,
            )

        if len(scaled_df) < self.sequence_length:
            raise ServingContractError(
                f"Need {self.sequence_length} rows, got {len(scaled_df)}."
            )

        window = scaled_df[seq_features].iloc[-self.sequence_length :].values.astype("float32")
        tensor = window.reshape(1, self.sequence_length, len(seq_features))
        meta = {
            "sequence_length": self.sequence_length,
            "feature_count": len(seq_features),
            "tensor_shape": list(tensor.shape),
        }
        return tensor, meta

    def get_output_denormalizer(
        self,
        *,
        reference_close: Optional[float] = None,
        pip_size: float = 0.0001,
    ) -> OutputDenormalizer:
        return OutputDenormalizer.from_serving_contract(
            self.contract,
            reference_close=reference_close,
            pip_size=pip_size,
        )

    def denormalize_outputs(
        self,
        named_heads: Dict[str, Any],
        *,
        reference_close: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Convert raw model heads to UI-ready real-world values."""
        denorm = self.get_output_denormalizer(reference_close=reference_close)
        return denorm.denormalize_all(named_heads)

    @staticmethod
    def map_raw_predictions(model: Any, raw_preds: Any) -> Dict[str, Any]:
        """Map Keras multi-head predict output to named head dict."""
        output_names = list(getattr(model, "output_names", []) or [])
        if isinstance(raw_preds, dict):
            return {k: _squeeze(v) for k, v in raw_preds.items()}

        if not isinstance(raw_preds, (list, tuple)):
            raw_preds = [raw_preds]

        named: Dict[str, Any] = {}
        for i, name in enumerate(output_names):
            if i >= len(raw_preds):
                break
            named[name] = _squeeze(raw_preds[i])
        return named


def _squeeze(val: Any) -> Any:
    arr = np.asarray(val)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1 and arr.shape[0] == 1:
        return float(arr[0])
    if arr.ndim >= 2 and arr.shape[0] == 1:
        squeezed = arr[0]
        if squeezed.ndim == 0:
            return float(squeezed)
        return squeezed.tolist()
    if hasattr(arr, "tolist"):
        return arr.tolist()
    return val
