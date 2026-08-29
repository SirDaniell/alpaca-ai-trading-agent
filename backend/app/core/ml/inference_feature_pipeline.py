"""
InferenceFeaturePipeline
========================
Reproduces the training-time feature-generation contract at inference time.

Fixes two root bugs in ProprietaryModelRuntime.predict():
  1. Early 90-row slice destroyed TI lookback history
  2. No step_config awareness — unknown which steps produced which features

Correct flow (matching analysis.tsx step order):
  1000 bars
    → run_currency_indices(supporting_ohlcv, step_config)  [MUST run first]
    → run_technical_analysis(step_config)
    → run_astronomical_features(step_config)          [if configured]
    → run_snr_features(step_config)                   [live, no look-ahead]
    → validate_and_fill(df) against feature_index_map [663 cols, zero-fill gaps]
    → scale(df)  ← SelectiveScaler over ALL rows (754 cols)
    → build_tensor(scaled_df) → (1, 90, 663) float32
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEBUG_CSV_EXPORT_ENABLED = os.getenv("ENABLE_INFERENCE_DEBUG_CSV", "0").lower() in {"1", "true", "yes", "on"}

import joblib
import numpy as np
import pandas as pd

from app.core.analysis.currency_index import (
    CurrencyIndexCalculator,
    INDEX_DEFINITIONS,
    prepare_index_data,
)
from app.core.analysis.technical_indicators import TechnicalIndicators, IndicatorConfig
from app.core.config import CurrencyIndexConfig
from app.core.analysis.support_resistance import (
    detect_snr_levels_sequential,
    create_clustered_zones_sequential,
    extract_snr_features,
)

logger = logging.getLogger(__name__)

# ── Sentinel dataset & paths ───────────────────────────────────────────────────
_BACKEND_ROOT = Path(__file__).resolve().parents[3]
_ACTIVE_DATASET = "ml_raw_20260808_895"


def resolve_dataset_cache_dir(dataset_name: str) -> Path:
    """Resolve dataset directory across potential workspace directory structures."""
    base = Path(__file__).resolve()
    for parent in base.parents:
        for candidate in [
            parent / "Backend" / "Backend" / "data" / "ml_cache" / dataset_name,
            parent / "Backend" / "data" / "ml_cache" / dataset_name,
            parent / "data" / "ml_cache" / dataset_name,
        ]:
            if candidate.exists() and candidate.is_dir():
                return candidate
    return base.parents[3] / "Backend" / "Backend" / "data" / "ml_cache" / dataset_name


class InferenceContractError(ValueError):
    """Raised when live data cannot reproduce the model's training feature contract."""
    def __init__(self, message: str, *, missing_features: Optional[List[str]] = None):
        super().__init__(message)
        self.missing_features = missing_features or []


# ── Default fallback step config (used when DB has no record) ─────────────────
_DEFAULT_STEP_CONFIGS: Dict[str, Any] = {
    "technical_analysis": {},
    "snr_analysis": {"lookback_period": 50},
}


class InferenceFeaturePipeline:
    """
    Per-dataset feature pipeline that reproduces training-time feature generation
    at live inference time.

    Construction:
        pipeline = InferenceFeaturePipeline(
            dataset_name="ml_raw_20260808_895",
            step_configs=configs,        # optional — skips DB lookup
            feature_map_path=Path(...),  # optional
            scaler_path=Path(...),       # optional
        )
    """

    # Class-level singleton cache: dataset_name → InferenceFeaturePipeline
    _pipeline_cache: Dict[str, "InferenceFeaturePipeline"] = {}

    def __init__(
        self,
        dataset_name: str = _ACTIVE_DATASET,
        step_configs: Optional[Dict[str, Any]] = None,
        scaling_config: Optional[Dict[str, Any]] = None,
        feature_map_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None,
        scaler: Optional[Any] = None,
    ) -> None:
        self._dataset_name = dataset_name

        # Resolve artifact paths (use overrides or derive from dataset_name)
        cache_dir = resolve_dataset_cache_dir(dataset_name)
        self._feature_map_path = feature_map_path or (cache_dir / "feature_index_map.json")
        self._scaler_path = scaler_path or (cache_dir / "scaler.joblib")

        # Load feature map → ordered feature list
        self._feature_names: List[str] = self._load_feature_names(self._feature_map_path)
        self._scaling_config: Dict[str, Any] = scaling_config or self._load_embedded_scaling_config(self._feature_map_path)

        # step_configs: supplied directly (offline/test) or will be loaded from DB async
        self._step_configs: Dict[str, Any] = step_configs if step_configs is not None else {}

        # Lazy-cached scaler. Prefer DB-loaded scaler_binary; filesystem path is fallback.
        self._scaler: Optional[Any] = scaler

        # Sequence length from step_configs or default
        ml_prep = self._step_configs.get("ml_dataset_preparation", {}) or {}
        self._seq_len: int = int(ml_prep.get("sequence_length", 90))

        self._ti_calc = TechnicalIndicators()

        logger.info(
            "[InferenceFeaturePipeline] Initialized for dataset=%s  features=%d  seq_len=%d",
            dataset_name, len(self._feature_names), self._seq_len,
        )

    # ── Class-level factory & cache ────────────────────────────────────────────

    @classmethod
    async def get_or_create(
        cls,
        dataset_name: str = _ACTIVE_DATASET,
        feature_map_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None,
    ) -> "InferenceFeaturePipeline":
        """Return cached pipeline or create+load step_configs from DB."""
        if dataset_name in cls._pipeline_cache:
            return cls._pipeline_cache[dataset_name]

        step_configs = await cls._load_step_configs_from_db(dataset_name)
        scaler = await cls._load_scaler_from_db(dataset_name)
        pipeline = cls(
            dataset_name=dataset_name,
            step_configs=step_configs,
            scaling_config=await cls._load_scaling_config_from_db(dataset_name),
            feature_map_path=feature_map_path,
            scaler_path=scaler_path,
            scaler=scaler,
        )
        cls._pipeline_cache[dataset_name] = pipeline
        return pipeline

    @staticmethod
    def _load_embedded_scaling_config(feature_map_path: Path) -> Dict[str, Any]:
        try:
            payload = json.loads(feature_map_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        scaling_config = payload.get("scaling_config")
        return scaling_config if isinstance(scaling_config, dict) else {}

    @classmethod
    async def _load_scaling_config_from_db(cls, dataset_name: str) -> Dict[str, Any]:
        try:
            from sqlalchemy import select, desc
            from app.api.routes.data.database import AsyncPostgresSessionLocal
            from app.database.models import MLDataset

            async with AsyncPostgresSessionLocal() as db:
                stmt = (
                    select(MLDataset.scaling_config)
                    .where(MLDataset.dataset_name == dataset_name)
                    .order_by(desc(MLDataset.created_at))
                    .limit(1)
                )
                result = await asyncio.wait_for(db.execute(stmt), timeout=8.0)
                scaling_config = result.scalar()
                if isinstance(scaling_config, str):
                    try:
                        scaling_config = json.loads(scaling_config)
                    except json.JSONDecodeError:
                        scaling_config = {}
                if isinstance(scaling_config, dict) and scaling_config:
                    logger.info(
                        "[InferenceFeaturePipeline] Loaded scaling_config from DB for dataset='%s' (keys=%s)",
                        dataset_name,
                        sorted(scaling_config.keys())[:12],
                    )
                    return scaling_config
        except Exception as err:
            logger.warning(
                "[InferenceFeaturePipeline] scaling_config DB lookup failed for dataset='%s': %s(%r)",
                dataset_name,
                type(err).__name__,
                err,
            )
        return {}

    @classmethod
    async def _load_scaler_from_db(cls, dataset_name: str) -> Optional[Any]:
        """Load the fitted SelectiveScaler from ml_datasets.scaler_binary."""
        try:
            from sqlalchemy import select, desc
            from app.api.routes.data.database import AsyncPostgresSessionLocal
            from app.database.models import MLDataset

            async with AsyncPostgresSessionLocal() as db:
                stmt = (
                    select(MLDataset.dataset_name, MLDataset.dataset_id, MLDataset.scaler_binary)
                    .where(MLDataset.dataset_name == dataset_name)
                    .where(MLDataset.scaler_binary.isnot(None))
                    .order_by(desc(MLDataset.created_at))
                    .limit(1)
                )
                result = await asyncio.wait_for(db.execute(stmt), timeout=8.0)
                row = result.first()
                if not row or not row.scaler_binary:
                    logger.warning(
                        "[InferenceFeaturePipeline] No DB scaler_binary found for dataset='%s'; filesystem scaler fallback will be used",
                        dataset_name,
                    )
                    return None

                scaler_blob = bytes(row.scaler_binary)
                scaler = joblib.load(io.BytesIO(scaler_blob))
                logger.info(
                    "[InferenceFeaturePipeline] Loaded scaler from DB ml_datasets.scaler_binary "
                    "for dataset='%s' (dataset_id=%s, bytes=%d, n_features=%s)",
                    row.dataset_name,
                    row.dataset_id,
                    len(scaler_blob),
                    getattr(scaler, "n_features_in_", None),
                )
                return scaler
        except Exception as err:
            logger.warning(
                "[InferenceFeaturePipeline] DB scaler lookup failed for dataset='%s': %s(%r); filesystem scaler fallback will be used",
                dataset_name,
                type(err).__name__,
                err,
            )
            return None

    @staticmethod
    def _extract_step_configs(record: Any) -> Optional[Dict[str, Any]]:
        if record is None or not getattr(record, "source_metadata", None):
            return None
        source_meta = record.source_metadata
        if isinstance(source_meta, str):
            try:
                source_meta = json.loads(source_meta)
            except json.JSONDecodeError:
                return None
        if isinstance(source_meta, dict):
            return source_meta.get("step_configs")
        return None

    @staticmethod
    def _coerce_currency_index_config(raw_cfg: Any) -> CurrencyIndexConfig:
        """Rebuild the canonical persisted config shape back to CurrencyIndexConfig.

        The analysis manager always works with a CurrencyIndexConfig object, but
        values stored in the database can come back as plain dicts or nested
        {'config': {...}} payloads. This normalizes that shape without changing the
        effective runtime behavior.
        """
        if raw_cfg is None:
            return CurrencyIndexConfig()
        if isinstance(raw_cfg, CurrencyIndexConfig):
            return raw_cfg

        payload: Dict[str, Any] = {}
        if isinstance(raw_cfg, dict):
            payload = dict(raw_cfg)
            if "config" in payload and isinstance(payload["config"], dict):
                payload = dict(payload["config"])
        elif hasattr(raw_cfg, "dict"):
            payload = dict(raw_cfg.dict())
        elif hasattr(raw_cfg, "__dict__"):
            payload = {k: v for k, v in raw_cfg.__dict__.items() if not k.startswith("_")}
        else:
            return CurrencyIndexConfig()

        selected = payload.get("selected_indices") or payload.get("selected_index") or ["Dollar", "Euro", "JPY"]
        if isinstance(selected, str):
            selected = [selected]
        selected = [str(idx) for idx in selected]

        verified_pairs = payload.get("verified_pairs") or {}
        if isinstance(verified_pairs, list):
            verified_pairs = {str(pair): True for pair in verified_pairs}
        elif isinstance(verified_pairs, set):
            verified_pairs = {str(pair): True for pair in verified_pairs}

        ti_routing = payload.get("calculate_ti_for_indices") or {}
        if isinstance(ti_routing, list):
            ti_routing = {str(idx): idx in selected for idx in selected if idx in ti_routing}
        elif isinstance(ti_routing, dict):
            ti_routing = {str(k): bool(v) for k, v in ti_routing.items()}
        else:
            ti_routing = {}

        ti_config = payload.get("ti_config") or {}
        if not isinstance(ti_config, dict):
            ti_config = {}

        return CurrencyIndexConfig(
            selected_indices=selected,
            verified_pairs={str(k): bool(v) for k, v in dict(verified_pairs).items()},
            fetch_missing_pairs=bool(payload.get("fetch_missing_pairs", True)),
            calculate_ti_for_indices=ti_routing,
            ti_config=ti_config,
            timeframe=str(payload.get("timeframe") or "H1"),
        )

    @classmethod
    async def _load_step_configs_from_db(cls, dataset_name: str) -> Dict[str, Any]:
        """
        Query MLDataset and SessionStepResult for dataset_name or latest session in DB.
        Resolves config steps across aliases (e.g. technical_config, technical_analysis_config).
        Falls back to _DEFAULT_STEP_CONFIGS if not found or on error.
        """
        try:
            from sqlalchemy import select, desc
            from app.api.routes.data.database import AsyncPostgresSessionLocal
            from app.database.models import MLDataset, SessionStepResult

            record = None
            step_configs: Dict[str, Any] = {}

            async with AsyncPostgresSessionLocal() as db:
                # 1. Try exact match on dataset_name
                stmt = select(MLDataset).where(MLDataset.dataset_name == dataset_name).order_by(desc(MLDataset.created_at))
                result = await db.execute(stmt)
                record = result.scalars().first()

                # 2. Fallback: get latest MLDataset session
                if record is None:
                    stmt_latest = select(MLDataset).order_by(desc(MLDataset.created_at))
                    res_latest = await db.execute(stmt_latest)
                    record = res_latest.scalars().first()

                if record and record.session_id:
                    # Query SessionStepResult for session_id to extract granular step configs
                    stmt_steps = (
                        select(SessionStepResult)
                        .where(SessionStepResult.session_id == record.session_id)
                        .order_by(desc(SessionStepResult.stored_at))
                    )
                    res_steps = await db.execute(stmt_steps)
                    step_rows = res_steps.scalars().all()

                    # Mapping table to standardize step config names
                    key_mapping = {
                        "technical_config": "technical_analysis",
                        "technical_analysis_config": "technical_analysis",
                        "snr_config": "snr_analysis",
                        "snr_analysis_config": "snr_analysis",
                        "ml_dataset_preparation_config": "ml_dataset_preparation",
                        "ml_prep_config": "ml_dataset_preparation",
                        "currency_indices_config": "currency_indices",
                        "currency_indices_ti_config": "currency_indices_ti",  # TI config for per-index enrichment
                        "footprint_ingestion_config": "footprint_ingestion",  # Footprint tick data config
                        "astronomical_config": "astronomical_analysis",
                        "astronomy_config": "astronomical_analysis",
                        "astronomical_analysis_config": "astronomical_analysis",
                    }

                    for step_row in step_rows:
                        raw_json = getattr(step_row, "result_data_v2", None)
                        if raw_json is None:
                            raw_json = getattr(step_row, "result_json", None)
                        if isinstance(raw_json, str):
                            try:
                                raw_json = json.loads(raw_json)
                            except json.JSONDecodeError:
                                raw_json = None
                        if isinstance(raw_json, dict):
                            # Unwrap nested 'config' or 'step_config' payload if present
                            unwrapped = raw_json.get("config", raw_json.get("step_config", raw_json))
                            if isinstance(unwrapped, dict) and unwrapped:
                                step_name = step_row.step_name
                                target_key = key_mapping.get(step_name, step_name)
                                if target_key not in step_configs:
                                    step_configs[target_key] = unwrapped

                # 3. Also pull step_configs embedded in MLDataset.source_metadata
                meta_configs = cls._extract_step_configs(record)
                if isinstance(meta_configs, dict):
                    for k, v in meta_configs.items():
                        if k not in step_configs and isinstance(v, dict) and v:
                            step_configs[k] = v

            if not step_configs:
                logger.warning(
                    "[InferenceFeaturePipeline] No non-empty step_configs found for '%s' or fallback — using defaults",
                    dataset_name,
                )
                return dict(_DEFAULT_STEP_CONFIGS)

            logger.info(
                "[InferenceFeaturePipeline] Loaded step_configs from DB for dataset='%s' (session_id=%s): keys=%s",
                getattr(record, "dataset_name", dataset_name), getattr(record, "session_id", None), sorted(step_configs.keys()),
            )
            return step_configs

        except Exception as err:
            logger.warning(
                "[InferenceFeaturePipeline] DB lookup failed (%s) — using default step configs", err
            )
            return dict(_DEFAULT_STEP_CONFIGS)

    # ── Feature-map loader ────────────────────────────────────────────────────

    @staticmethod
    def _load_feature_names(feature_map_path: Path) -> List[str]:
        try:
            payload = json.loads(feature_map_path.read_text())
            feature_map = payload.get("feature_index_map", payload)
            ordered = [name for name, _ in sorted(feature_map.items(), key=lambda kv: int(kv[1]))]
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as err:
            raise InferenceContractError(
                f"Cannot load feature map '{feature_map_path}': {err}"
            ) from err
        return ordered

    # ── Step 1: Currency Index Features ──────────────────────────────────────

    def run_currency_indices(
        self,
        df: pd.DataFrame,
        supporting_ohlcv: Dict[str, List[Dict[str, Any]]],
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        """Merge supporting OHLCV, build index OHLCV, run per-index TI."""
        cfg = self._coerce_currency_index_config(config)
        required_indices = [
            idx for idx in cfg.selected_indices
            if any(col.startswith(f"{idx}_") for col in self._feature_names)
        ]
        if not required_indices:
            logger.info("[InferenceFeaturePipeline] No currency indices required for dataset '%s' — skip", self._dataset_name)
            return df

        logger.info(
            "[InferenceFeaturePipeline] 🚀 Starting run_currency_indices: required=%s  input_shape=%s  supporting_pairs=%s",
            required_indices,
            df.shape,
            list(supporting_ohlcv.keys()) if supporting_ohlcv else [],
        )

        enriched = df.copy()

        # Merge supporting pair bars
        if supporting_ohlcv:
            enriched = self._merge_supporting_pairs(enriched, supporting_ohlcv)
            logger.info(
                "[InferenceFeaturePipeline] Merged %d supporting pair(s) — frame shape now: %s",
                len(supporting_ohlcv),
                enriched.shape,
            )

        # ── CRITICAL: reset integer index immediately after merge ────────────
        # _merge_supporting_pairs uses pd.DataFrame.merge(..., how='left').
        # When timestamps don't align 1:1 the left-join can produce a
        # non-contiguous or duplicate RangeIndex. Any subsequent pandas
        # arithmetic (df[a] - df[b], np.where, pd.DataFrame(dict)) on such
        # a DataFrame raises:
        #   ValueError: cannot reindex on an axis with duplicate labels
        # Resetting here gives a clean 0…N-1 index before ANY further ops.
        enriched = enriched.reset_index(drop=True)

        # Dedup timestamps — a duplicate Time after merge inflates row count
        # and corrupts per-column length checks later.
        if "Time" in enriched.columns and enriched.duplicated(subset=["Time"]).any():
            n_dup = int(enriched.duplicated(subset=["Time"]).sum())
            logger.warning(
                "[InferenceFeaturePipeline] %d duplicate timestamps after pair merge — keeping last",
                n_dup,
            )
            enriched = enriched.drop_duplicates(subset=["Time"], keep="last").reset_index(drop=True)

        # Build currency index OHLCV
        try:
            pair_enriched = prepare_index_data(enriched)
            index_frame = CurrencyIndexCalculator(pair_enriched).to_dataframe(required_indices)
            logger.info(
                "[InferenceFeaturePipeline] CurrencyIndexCalculator produced %d index OHLCV columns: %s",
                len(index_frame.columns),
                list(index_frame.columns),
            )
        except Exception as err:
            # Fail-fast: propagate as InferenceContractError so callers can surface the error
            logger.error("[InferenceFeaturePipeline] CurrencyIndexCalculator failed: %s", err, exc_info=True)
            raise InferenceContractError(
                f"CurrencyIndex calculation failed: {err}",
            ) from err

        if index_frame.empty:
            # Missing support-pair OHLCV is a valid degraded state for live inference.
            # The pipeline is allowed to continue and zero-fill the corresponding
            # index-prefixed features later in validate_and_fill().
            logger.warning(
                "[InferenceFeaturePipeline] CurrencyIndexCalculator produced empty frame — likely missing supporting pair OHLCV; continuing with zero-filled index columns"
            )
            return enriched.reset_index(drop=True)

        # Reset index_frame to match enriched's clean 0…N-1 RangeIndex
        index_frame = index_frame.reset_index(drop=True)
        if len(index_frame) != len(enriched):
            logger.warning(
                "[InferenceFeaturePipeline] Index frame length mismatch: index_frame=%d != enriched=%d — trimming",
                len(index_frame),
                len(enriched),
            )
            index_frame = index_frame.iloc[: len(enriched)].reset_index(drop=True)

        for col in index_frame.columns:
            if col in enriched.columns:
                # Column already exists — update in-place to avoid duplicates
                enriched[col] = index_frame[col].values
            else:
                # New column — add it (values array avoids any index re-alignment)
                enriched[col] = index_frame[col].values

        # Run TI per index — each index is a standalone mini OHLCV symbol.
        # Config routing: ci_config may contain per-index dicts, e.g.:
        #   {"Dollar": {"TI": True, ...}, "Euro": {"TI": False}}
        # Top-level "enable_ti" key is also supported as a global default.
        # If neither is present, TI runs for all required indices (backward compat).
        ci_ti_config = self._step_configs.get("currency_indices_ti", {}) or {}
        _ci_ti_global_default = bool(cfg.fetch_missing_pairs) if cfg else True

        for index_name in required_indices:
            ohlcv_map = {
                f"{index_name}_open": "Open",
                f"{index_name}_high": "High",
                f"{index_name}_low": "Low",
                f"{index_name}_close": "Close",
                f"{index_name}_tick_volume": "Volume",
            }
            if not all(c in enriched.columns for c in ohlcv_map):
                missing_ohlcv = [c for c in ohlcv_map if c not in enriched.columns]
                logger.warning(
                    "[InferenceFeaturePipeline] Index '%s' missing required OHLCV columns %s — skipping per-index TI",
                    index_name,
                    missing_ohlcv,
                )
                continue

            # Per-index TI flag: prefer the canonical config object, then fall back to
            # the original saved dict shape. This preserves the exact analysis-manager
            # semantics even when values were loaded from storage as plain dicts.
            index_ci_cfg = cfg.calculate_ti_for_indices.get(index_name, _ci_ti_global_default)
            run_ti_for_index = bool(index_ci_cfg)

            if not run_ti_for_index:
                logger.info(
                    "[InferenceFeaturePipeline] Per-index TI disabled for '%s' (config TI=False) — skipping",
                    index_name,
                )
                continue

            # Give ti_input a CLEAN 0…N-1 index — calculate_all_indicators
            # performs Series arithmetic that will crash if the index has
            # duplicates inherited from the parent enriched DataFrame.
            ti_input = (
                enriched[list(ohlcv_map.keys())]
                .rename(columns=ohlcv_map)
                .reset_index(drop=True)
            )

            # Use currency_indices_ti step config for per-index TI if available,
            # otherwise fall back to the shared _ti_calc instance.
            # This lets training control which indicators are computed per-index
            # independently of the main symbol TI config.
            _per_index_cfg = ci_ti_config.get(index_name, {}) if ci_ti_config else {}
            if _per_index_cfg:
                from app.core.analysis.technical_indicators import TechnicalIndicators as _TI, IndicatorConfig as _IC
                _valid_keys = set(_IC.__annotations__.keys())
                _filtered_cfg = {k: v for k, v in _per_index_cfg.items() if k in _valid_keys}
                _index_ti_calc = _TI(config=_IC(**_filtered_cfg))
            else:
                _index_ti_calc = self._ti_calc

            try:
                ti_out = _index_ti_calc.calculate_all_indicators(ti_input.copy(), mode="inference")
            except Exception as err:
                logger.warning("[InferenceFeaturePipeline] %s TI calculation failed: %s", index_name, err)
                continue

            ti_out = ti_out.reset_index(drop=True)

            base_cols = {"Open", "High", "Low", "Close", "Volume"}
            new_cols: Dict[str, pd.Series] = {}
            for col in ti_out.columns:
                if col in base_cols:
                    continue
                prefixed = f"{index_name}_{col}"
                if prefixed in enriched.columns:
                    continue
                s = ti_out[col]
                if len(s) != len(enriched):
                    continue
                # Use numpy array to bind to enriched's clean index — avoids
                # re-alignment issues when enriched.index != ti_out.index.
                new_cols[prefixed] = pd.Series(s.to_numpy(), index=enriched.index)

            if new_cols:
                enriched = pd.concat([enriched, pd.DataFrame(new_cols, index=enriched.index)], axis=1)
                logger.info(
                    "[InferenceFeaturePipeline] Added %d prefixed TI indicator columns for index '%s' (e.g., %s)",
                    len(new_cols),
                    index_name,
                    list(new_cols.keys())[:3],
                )

        # ── Dedup guard: remove any duplicate columns that may have been created ──
        if enriched.columns.duplicated().any():
            dup_cols = enriched.columns[enriched.columns.duplicated()].tolist()
            logger.warning(
                "[InferenceFeaturePipeline] Detected %d duplicate column(s), removing: %s",
                len(dup_cols),
                dup_cols[:10],
            )
            enriched = enriched.loc[:, ~enriched.columns.duplicated(keep='first')]

        # ── PREFIX VERIFICATION CHECK ───────────────────────────────────────────────
        # Verify that all required currency indices have properly prefixed columns
        # in the enriched DataFrame, and check for any un-prefixed column leaks.
        total_index_cols = 0
        for index_name in required_indices:
            prefixed_cols = [c for c in enriched.columns if c.startswith(f"{index_name}_")]
            total_index_cols += len(prefixed_cols)
            if not prefixed_cols:
                logger.warning(
                    "[InferenceFeaturePipeline] ⚠️ PREFIX VERIFICATION FAILED: Index '%s' has 0 prefixed columns in enriched frame!",
                    index_name,
                )
            else:
                logger.info(
                    "[InferenceFeaturePipeline] ✓ PREFIX VERIFICATION OK: Index '%s' has %d properly prefixed columns (e.g. %s)",
                    index_name,
                    len(prefixed_cols),
                    prefixed_cols[:3],
                )

        # Check for un-prefixed index leakage (bare column names without prefix)
        unprefixed_leaks = [c for c in enriched.columns if c in required_indices]
        if unprefixed_leaks:
            logger.warning(
                "[InferenceFeaturePipeline] ⚠️ PREFIX VERIFICATION WARNING: Found unprefixed index column(s): %s",
                unprefixed_leaks,
            )

        logger.info(
            "[InferenceFeaturePipeline] ✅ Currency indices step complete: indices=%s  total_index_cols=%d  total_frame_cols=%d",
            required_indices,
            total_index_cols,
            len(enriched.columns),
        )
        return enriched.reset_index(drop=True)

    # ── Step 2: Technical Analysis ────────────────────────────────────────────

    def run_technical_analysis(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Run TI on the full bar window using stored config."""
        df_clean = df.reset_index(drop=True)
        if df_clean.columns.duplicated().any():
            df_clean = df_clean.loc[:, ~df_clean.columns.duplicated(keep="first")].copy()

        valid_keys = set(IndicatorConfig.__annotations__.keys())
        filtered = {k: v for k, v in (config or {}).items() if k in valid_keys}
        ti = TechnicalIndicators(config=IndicatorConfig(**filtered))
        try:
            result = ti.calculate_all_indicators(df_clean.copy(), mode="inference")
            # Dedup guard: input df may be pre-enriched; TI can introduce duplicate columns
            if result.columns.duplicated().any():
                dup_names = result.columns[result.columns.duplicated(keep=False)].unique().tolist()
                logger.warning(
                    "[InferenceFeaturePipeline] run_technical_analysis: deduplicating %d col(s): %s",
                    len(dup_names), dup_names[:20],
                )
                result = result.loc[:, ~result.columns.duplicated(keep="first")].copy()
            logger.info(
                "[InferenceFeaturePipeline] TI complete: %d→%d columns",
                len(df.columns), len(result.columns),
            )
            return result.reset_index(drop=True)
        except Exception as err:
            logger.warning("[InferenceFeaturePipeline] TI failed (%s) — returning raw df", err)
            return df_clean

    # ── Step 3: Astronomical Features ─────────────────────────────────────────

    def run_astronomical_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Run astro features only if configured AND the feature map has astro columns."""
        if not config:
            return df
        has_astro = any(col.startswith("astro_") or "moon_" in col for col in self._feature_names)
        if not has_astro:
            return df
        try:
            from app.core.analysis.astronomy.astronomical import AstronomicalFeatureGenerator
            astro = AstronomicalFeatureGenerator(
                use_minor_aspects=config.get("use_minor_aspects", False),
                observer_lat=config.get("observer_lat", 0.0),
                observer_lon=config.get("observer_lon", 0.0),
            )
            target_features = list(astro.create_all_possible_features().keys())
            time_col = "Time" if "Time" in df.columns else "time" if "time" in df.columns else None
            if time_col is None:
                return df
            rows = []
            for _, row in df.iterrows():
                rows.append(astro.generate_features_for_date(row[time_col], target_features))
            astro_df = pd.DataFrame(rows, index=df.index)
            return pd.concat([df, astro_df], axis=1)
        except Exception as err:
            logger.warning("[InferenceFeaturePipeline] Astro features failed: %s", err)
            return df

    # ── Step 4: SNR Zone Features ─────────────────────────────────────────────

    def run_snr_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Compute SNR zone features for last bar only (no look-ahead)."""
        cfg = config or {}
        lookback = int(cfg.get("lookback_period", 50))
        min_dist = float(cfg.get("min_distance_pct", 0.02))
        n_clusters = int(cfg.get("n_clusters", 5))
        zone_width = float(cfg.get("zone_width", 0.01))

        current_index = len(df) - 1
        try:
            levels = detect_snr_levels_sequential(df, current_index, lookback, min_dist)
            slice_start = max(0, current_index - lookback)
            df_slice = df.iloc[slice_start: current_index + 1]
            zones = create_clustered_zones_sequential(levels, df_slice, n_clusters=n_clusters, zone_width=zone_width)
            curr_price = float(df["Close"].iloc[current_index])
            snr_feats = extract_snr_features(curr_price, levels, zones)
            for k, v in snr_feats.items():
                df.at[df.index[current_index], k] = v
            logger.info("[InferenceFeaturePipeline] SNR features added: %d keys", len(snr_feats))
        except Exception as err:
            logger.warning("[InferenceFeaturePipeline] SNR features failed: %s", err)
        return df

    # ── Step 5: Validate & zero-fill ──────────────────────────────────────────

    @staticmethod
    def _coerce_time_index(time_values: pd.Series) -> pd.DatetimeIndex:
        """Parse live Time values, including UNIX seconds/milliseconds."""
        if pd.api.types.is_numeric_dtype(time_values):
            numeric = pd.to_numeric(time_values, errors="coerce")
            median = float(numeric.dropna().median()) if numeric.notna().any() else 0.0
            unit = "ms" if median > 10_000_000_000 else "s"
            return pd.DatetimeIndex(pd.to_datetime(numeric, unit=unit, errors="coerce"))
        return pd.DatetimeIndex(pd.to_datetime(time_values, errors="coerce"))

    def _add_inference_aliases(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reconstruct direct training-time aliases from live/canonical columns.

        These are not new indicators; they are equivalent names the saved feature
        map may require because the training dataset preserved older column names.
        """
        aliases: Dict[str, Any] = {}

        direct_aliases = {
            "A": ["a"],
            "B": ["b"],
            "D": ["d"],
            "E": ["e"],
            "G": ["g"],
            "TickVolume": ["tick_volume", "Volume", "volume"],
            "Spread": ["spread"],
            "Day_of_week": ["day_of_week"],
            "Hour": ["hour"],
            "Minute": ["minute"],
            "Pivot": ["Pivots"],
            "Resist_trendline_val": ["Resist_Trendline_Value", "resist_trendline_val"],
            "Support_trendline_val": ["Support_Trendline_Value", "support_trendline_val"],
            "Ma10_diff": ["SMA_10_Diff", "ma10_diff"],
            "Ma20_diff": ["SMA_20_Diff", "ma20_diff"],
            "Ma50_diff": ["MA_50_Diff", "ma50_diff"],
            "Ma100_diff": ["MA_100_Diff", "ma100_diff"],
            "Price_longlong_period_diff": ["Price_Long_Long_Period_Diff", "price_longlong_period_diff"],
            "Price_longshort_period_diff": ["Price_Long_Short_Period_Diff", "price_longshort_period_diff"],
            "Price_shortlong_period_diff": ["Price_Short_Long_Period_Diff", "price_shortlong_period_diff"],
            "R1": ["r1", "Pivot_R1"],
            "R2": ["r2", "Pivot_R2"],
            "R3": ["r3", "Pivot_R3"],
            "S1": ["s1", "Pivot_S1"],
            "S2": ["s2", "Pivot_S2"],
            "S3": ["s3", "Pivot_S3"],
            "Session": ["session"],
            "Session_transition": ["session_transition"],
        }
        for target, sources in direct_aliases.items():
            if target in df.columns:
                continue
            for source in sources:
                if source in df.columns:
                    aliases[target] = df[source]
                    break

        if "Time" in df.columns:
            time_index = self._coerce_time_index(df["Time"])
            if "day_of_week" not in df.columns:
                aliases["day_of_week"] = time_index.dayofweek
            if "Day_of_week" not in df.columns:
                aliases["Day_of_week"] = time_index.dayofweek
            if "hour" not in df.columns:
                aliases["hour"] = time_index.hour
            if "Hour" not in df.columns:
                aliases["Hour"] = time_index.hour
            if "minute" not in df.columns:
                aliases["minute"] = time_index.minute
            if "Minute" not in df.columns:
                aliases["Minute"] = time_index.minute

        if "Spread" not in df.columns:
            aliases["Spread"] = 0.0

        if aliases:
            alias_df = pd.DataFrame(
                {
                    name: (
                        value.to_numpy() if isinstance(value, pd.Series)
                        else np.asarray(value) if hasattr(value, "__len__") and not isinstance(value, str)
                        else np.full(len(df), value)
                    )
                    for name, value in aliases.items()
                    if name not in df.columns
                },
                index=df.index,
            )
            if not alias_df.empty:
                df = pd.concat([df, alias_df], axis=1)
                logger.info(
                    "[InferenceFeaturePipeline] Added %d inference feature alias(es): %s",
                    len(alias_df.columns),
                    list(alias_df.columns)[:12],
                )

        return df

    def validate_and_fill(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Check all 663 feature columns against the enriched DataFrame.
        Zero-fill missing columns. Raise if >50% are absent.
        """
        df = self._add_inference_aliases(df)
        missing = [c for c in self._feature_names if c not in df.columns]
        present_count = len(self._feature_names) - len(missing)
        total = len(self._feature_names)

        # Separate structural/primary TI features from currency index features
        index_missing = [c for c in missing if any(c.startswith(f"{idx}_") for idx in INDEX_DEFINITIONS)]
        primary_missing = [c for c in missing if c not in index_missing]

        if missing:
            # Group by family for diagnostics
            groups: Dict[str, List[str]] = {
                "primary_TI": primary_missing, "Dollar_*": index_missing
            }
            summary = {k: len(v) for k, v in groups.items() if v}
            sample = missing[:10]
            logger.warning(
                "[InferenceFeaturePipeline] ⚠️ %d/%d features missing (Primary: %d, Currency Indices: %d) — sample: %s",
                len(missing), total, len(primary_missing), len(index_missing), sample,
            )
             
            # If >50% of PRIMARY technical indicators are missing, hard raise.
            # Currency indices (Dollar_*) are expected to be zero-filled if supporting pair OHLCV isn't loaded.
            total_primary = sum(1 for c in self._feature_names if not any(c.startswith(f"{idx}_") for idx in INDEX_DEFINITIONS))
            if total_primary > 0 and (len(primary_missing) / total_primary) > 0.5:
                raise InferenceContractError(
                    f"[InferenceFeaturePipeline] {len(primary_missing)}/{total_primary} primary features absent "
                    f"(>{50}% threshold). Pipeline likely failed completely.",
                    missing_features=missing,
                )

            # Zero-fill all missing
            for c in missing:
                df[c] = 0.0
        else:
            logger.info("[InferenceFeaturePipeline] ✅ %d/%d features present", total, total)

        return df

    @staticmethod
    def _sigmoid_normalize(values: pd.Series, *, window: int = 20, scale: float = 2.0, default: float = 0.5) -> pd.Series:
        """Match training's rolling-mean sigmoid squash for already range-normalized series."""
        series = pd.to_numeric(values, errors="coerce")
        rolling_mean = series.rolling(window=window, min_periods=min(20, window)).mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            deviation = (series - rolling_mean) / (rolling_mean.abs() + 1e-8)
        deviation = deviation.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out = 1.0 / (1.0 + np.exp(-float(scale) * deviation))
        return out.replace([np.inf, -np.inf], np.nan).ffill().fillna(default)

    @staticmethod
    def _linear_range_normalize(values: pd.Series, range_cfg: Dict[str, Any]) -> Optional[pd.Series]:
        high = range_cfg.get("fitted_range_high", range_cfg.get("high"))
        low = range_cfg.get("fitted_range_low", range_cfg.get("low", 0.0))
        try:
            high_f = float(high)
            low_f = float(low)
        except (TypeError, ValueError):
            return None
        width = high_f - low_f
        if width <= 0:
            return None
        return (pd.to_numeric(values, errors="coerce") - low_f) / width

    def _apply_training_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the train-fitted pre-scaler normalization persisted in scaling_config.

        execute_ml_with_splits() fits SelectiveScaler after this transform, with
        price/diff partitions as PassThrough. Live inference must therefore feed
        the scaler the same normalized domain, not raw OHLCV/indicator values.
        """
        cfg = self._scaling_config or {}
        sr = cfg.get("structural_range") or {}
        try:
            low = float(sr.get("low"))
            width = float(sr.get("width"))
        except (TypeError, ValueError):
            logger.warning(
                "[InferenceFeaturePipeline] scaling_config.structural_range unavailable; "
                "SelectiveScaler will receive raw price-domain columns."
            )
            return df
        if width <= 0:
            return df

        out = df.copy()
        sigmoid_scale = float(cfg.get("sigmoid_scale_factor") or 2.0)

        out["Rolling_Range_High"] = float(sr.get("high", low + width))
        out["Rolling_Range_Low"] = low
        out["Rolling_Range_Width"] = width
        out["Rolling_Range_Mid"] = low + width / 2.0

        close_raw = pd.to_numeric(out["Close"], errors="coerce") if "Close" in out.columns else None
        if close_raw is not None:
            out["Regime_Macro_Position"] = (
                (close_raw - out["Rolling_Range_Low"]) / (out["Rolling_Range_Width"] + 1e-8)
            ).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.5)
            out["Regime_Mid_ROC_10"] = out["Rolling_Range_Mid"].pct_change(10).fillna(0.0)
            out["Regime_Mid_ROC_50"] = out["Rolling_Range_Mid"].pct_change(50).fillna(0.0)
            out["Regime_Width_ROC"] = out["Rolling_Range_Width"].pct_change(20).fillna(0.0)
            try:
                high = pd.to_numeric(out.get("High", close_raw), errors="coerce")
                low_series = pd.to_numeric(out.get("Low", close_raw), errors="coerce")
                prev_close = close_raw.shift(1)
                true_range = pd.concat(
                    [(high - low_series), (high - prev_close).abs(), (low_series - prev_close).abs()],
                    axis=1,
                ).max(axis=1)
                atr = true_range.rolling(window=14, min_periods=1).mean().ewm(span=14, adjust=False).mean()
                atr_baseline = atr.rolling(100, min_periods=20).mean()
                out["Regime_ATR_Surge"] = (atr / (atr_baseline + 1e-8)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
            except Exception:
                out["Regime_ATR_Surge"] = 1.0
            out["Regime_Distance_From_Low"] = self._sigmoid_normalize(
                out["Regime_Macro_Position"], scale=sigmoid_scale
            )
            out["Regime_Distance_From_High"] = self._sigmoid_normalize(
                ((out["Rolling_Range_High"] - close_raw) / (out["Rolling_Range_Width"] + 1e-8))
                .replace([np.inf, -np.inf], np.nan).ffill().fillna(0.5),
                scale=sigmoid_scale,
            )

        volume_cfg = cfg.get("volume_range") or {}
        for col in ["Volume", "TickVolume"]:
            if col in out.columns:
                normalized = self._linear_range_normalize(out[col], volume_cfg.get(col) or {})
                if normalized is not None:
                    out[col] = normalized

        distance_ranges = ((cfg.get("distance_range") or {}).get("ranges") or {})
        for col, range_cfg in distance_ranges.items():
            if col in out.columns:
                normalized = self._linear_range_normalize(out[col], range_cfg)
                if normalized is not None:
                    out[col] = normalized

        price_level_cols = {
            "Open", "High", "Low", "Close",
            "Previous_Close",
            "Prev_1_Close", "Prev_2_Close", "Prev_3_Close", "Prev_4_Close", "Prev_5_Close",
            "R1", "R2", "R3", "S1", "S2", "S3",
            "High_Day_1", "High_Day_2", "High_Day_3",
            "Low_Day_1", "Low_Day_2", "Low_Day_3",
            "10_Day_MA", "50_Day_MA",
            "Short_MA", "Long_MA", "MA",
            "Short_MA_10", "Short_MA_50",
            "Long_MA_25", "Long_MA_100", "Long_MA_200",
            "MA-25", "MA-50", "MA-100", "MA-200",
            "EMA-8", "EMA-12", "EMA-21", "EMA-64",
            "Pivot Price", "Pivot_Price",
            "Supertrend", "Supertrend_Upper", "Supertrend_Lower",
            "Final Lowerband", "Final Upperband", "Parabolic_SAR",
            "Rolling_Range_High", "Rolling_Range_Low", "Rolling_Range_Mid",
            "Structural_Range_Position",
            "Support_Trendline_Value", "Resist_Trendline_Value",
            "SMC_OB_Top", "SMC_OB_Bottom", "SMC_FVG_Top", "SMC_FVG_Bottom",
            "SMC_Swing_Level", "SMC_Liquidity_Level", "SMC_BOS_Level",
            "FVG_Top", "FVG_Bottom", "Order_Block_Top", "Order_Block_Bottom",
        }
        price_prefixes = ("SMA_", "EMA_", "WMA_", "HMA_", "BBL_", "BBM_", "BBU_", "BB_", "MA_")
        diff_markers = ("_Diff", "_diff", "_Distance", "_distance", "_pct", "_Pct", "Minus", "minus", "Change")

        normalized_price = 0
        for col in list(out.columns):
            is_price = col in price_level_cols or col.startswith(price_prefixes)
            if not is_price or any(marker in col for marker in diff_markers):
                continue
            raw_norm = (pd.to_numeric(out[col], errors="coerce") - low) / width
            raw_norm = raw_norm.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.5)
            out[col] = self._sigmoid_normalize(raw_norm, scale=sigmoid_scale)
            normalized_price += 1

        explicit_diffs = {
            "EMA_12_Minus_EMA8", "EMA_21_Minus_EMA8", "EMA_64_Minus_EMA8", "EMA_200_Minus_EMA8",
            "EMA_100_Minus_EMA8", "MA_100_50_Diff", "MACD_12_26_9", "MACDh_12_26_9",
            "MACDs_12_26_9", "Supertrend_Distance", "snr_dist_to_nearest_level",
            "snr_dist_to_nearest_support", "snr_dist_to_nearest_resistance",
            "MA_Change0", "MA_Change1", "MA_Change2", "MA_Change3", "MA_Change4",
            "MA_200_Change0", "MA_200_Change1", "MA_200_Change2", "MA_200_Change3", "MA_200_Change4",
            "MA_200_Change_0", "MA_200_Change_1", "MA_200_Change_2", "MA_200_Change_3", "MA_200_Change_4",
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "A", "B", "C", "D", "E", "F", "G", "H", "I",
            "SMA0", "SMA1", "SMA2", "SMA3", "SMA4", "SMA5", "SMA6", "SMA7", "SMA8", "SMA9", "SMA10", "SMA11",
            "MOM_t", "MR_t", "TF_t", "Candle_Size",
        }
        bounded = {
            "RSI", "RSI_14", "RSI_7", "RSI_2", "Stochastic", "Stoch_K", "Stoch_D", "CCI",
            "Regime_Distance_From_Low", "Regime_Distance_From_High", "Regime_Macro_Position",
            "Rolling_Range_Width", "Rolling_Range_High", "Rolling_Range_Low", "Rolling_Range_Mid",
            "Structural_Range_Width", "Structural_Range_Position",
        }
        diff_cols = set()
        for col in out.columns:
            base = col.split("_", 1)[1] if any(col.startswith(f"{idx}_") for idx in INDEX_DEFINITIONS) else col
            dynamic = (
                "_Diff" in base or "_diff" in base or "_Distance" in base or "_distance" in base
                or "Minus" in base or "minus" in base or "Speed" in base or "speed" in base
                or "Volatility" in base or "volatility" in base or base.startswith("MA_Change")
            )
            if (base in explicit_diffs or dynamic) and base not in bounded and "Volume" not in base and "volume" not in base and "Time_Diff" not in base:
                diff_cols.add(col)

        for col in diff_cols:
            raw_norm = pd.to_numeric(out[col], errors="coerce") / width
            raw_norm = raw_norm.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
            out[col] = self._sigmoid_normalize(raw_norm, scale=sigmoid_scale)

        bounded_oscillators = {
            "RSI", "RSI_14", "RSI_7", "RSI_2", "Stochastic", "Stoch_K", "Stoch_D",
            "BBP_20_2.0_2.0", "BBB_20_2.0_2.0",
        }
        for col in list(out.columns):
            base = col.split("_", 1)[1] if any(col.startswith(f"{idx}_") for idx in INDEX_DEFINITIONS) else col
            if base not in bounded_oscillators:
                continue
            values = pd.to_numeric(out[col], errors="coerce")
            max_abs = float(values.abs().max()) if values.notna().any() else 0.0
            if max_abs > 1.5:
                out[col] = (values / 100.0).clip(0.0, 1.0).fillna(0.0)

        index_ranges = cfg.get("index_structural_ranges") or {}
        for idx_name, range_cfg in index_ranges.items():
            try:
                idx_low = float(range_cfg.get("low"))
                idx_width = float(range_cfg.get("width"))
            except (TypeError, ValueError):
                continue
            if idx_width <= 0:
                continue
            for col in list(out.columns):
                if not col.startswith(f"{idx_name}_"):
                    continue
                base = col[len(idx_name) + 1:]
                if base in {"open", "high", "low", "close"} or (
                    (base in price_level_cols or base.startswith(price_prefixes))
                    and not any(marker in base for marker in diff_markers)
                ):
                    raw_norm = (pd.to_numeric(out[col], errors="coerce") - idx_low) / idx_width
                    raw_norm = raw_norm.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.5)
                    out[col] = self._sigmoid_normalize(raw_norm, scale=sigmoid_scale)
                elif col in diff_cols:
                    raw_norm = pd.to_numeric(out[col], errors="coerce") / idx_width
                    raw_norm = raw_norm.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
                    out[col] = self._sigmoid_normalize(raw_norm, scale=sigmoid_scale)

        logger.info(
            "[InferenceFeaturePipeline] Applied training pre-scaler normalization "
            "(price=%d, diff=%d, index_ranges=%d)",
            normalized_price,
            len(diff_cols),
            len(index_ranges),
        )
        return out

    # ── Step 6: Scale ─────────────────────────────────────────────────────────

    def scale(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply SelectiveScaler over ALL rows (754 cols, zero-fill adv_target_*)."""
        if self._scaler is None:
            logger.info("[InferenceFeaturePipeline] Loading scaler from filesystem fallback: %s", self._scaler_path)
            self._scaler = joblib.load(self._scaler_path)

        scaler_cols: List[str] = list(
            getattr(self._scaler, "feature_names", None)
            or getattr(self._scaler, "all_cols", None)
            or []
        )
        if not scaler_cols:
            raise InferenceContractError("Scaler has no feature_names / all_cols attribute.")

        col_dict = {
            col: df[col].values if col in df.columns else np.zeros(len(df))
            for col in scaler_cols
        }
        scale_input = pd.DataFrame(col_dict, index=df.index)
        self._save_debug_snapshot(scale_input, "ml_pre_selective_scaler_input")

        try:
            scaled_arr = self._scaler.transform(scale_input)
        except Exception as err:
            raise InferenceContractError(f"SelectiveScaler.transform failed: {err}") from err

        return pd.DataFrame(
            scaled_arr if isinstance(scaled_arr, np.ndarray) else np.asarray(scaled_arr),
            index=df.index,
            columns=scaler_cols,
        )

    # ── Step 7: Build tensor ──────────────────────────────────────────────────

    def build_tensor(self, scaled_df: pd.DataFrame) -> np.ndarray:
        """
        Select 663 features in feature_map order, slice last seq_len rows,
        return (1, seq_len, 663) float32 with NaN/Inf → 0.
        """
        missing_seq = [c for c in self._feature_names if c not in scaled_df.columns]
        if missing_seq:
            raise InferenceContractError(
                f"Scaled frame missing {len(missing_seq)} sequence features.",
                missing_features=missing_seq,
            )

        window = scaled_df[self._feature_names].iloc[-self._seq_len:].values.astype("float32")
        if window.shape[0] < self._seq_len:
            raise InferenceContractError(
                f"Not enough rows after scaling: need {self._seq_len}, got {window.shape[0]}."
            )

        bad = ~np.isfinite(window)
        n_bad = int(bad.sum())
        if n_bad:
            logger.warning("[InferenceFeaturePipeline] Replacing %d NaN/Inf values with 0.0", n_bad)
            window[bad] = 0.0

        return window.reshape(1, self._seq_len, len(self._feature_names))

    def _save_debug_snapshot(self, df: pd.DataFrame, stage: str) -> Optional[str]:
        """Persist live inference feature frames used to debug the ML/scaler contract.

        CSV exports are disabled by default to avoid bloating disk space; re-enable with
        ENABLE_INFERENCE_DEBUG_CSV=1 in the environment.
        """
        if not DEBUG_CSV_EXPORT_ENABLED:
            return None

        if df is None or getattr(df, "empty", True):
            return None

        output_root = _BACKEND_ROOT / "Backend" / "artifacts" / "live_inference_debug"
        output_root.mkdir(parents=True, exist_ok=True)
        dataset = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in self._dataset_name)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = output_root / f"{dataset}_{timestamp}_{stage}.csv"

        try:
            df.to_csv(path, index=False)
            dollar_cols = sum(1 for col in df.columns if str(col).startswith("Dollar_"))
            logger.info(
                "[InferenceFeaturePipeline] Saved %s snapshot: %s (%d rows x %d cols, Dollar_*=%d)",
                stage,
                path,
                len(df),
                len(df.columns),
                dollar_cols,
            )
            return str(path)
        except Exception as exc:
            logger.warning("[InferenceFeaturePipeline] Failed to save %s snapshot: %s", stage, exc)
            return None

    async def _ensure_currency_index_data(
        self,
        df: pd.DataFrame,
        supporting_ohlcv: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        config: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Reuse the canonical prefetch logic before building the currency-index block."""
        if df is None or getattr(df, "empty", False):
            return df

        cfg = self._coerce_currency_index_config(config)
        required_indices = [
            idx for idx in cfg.selected_indices
            if any(col.startswith(f"{idx}_") for col in self._feature_names)
        ]
        if not required_indices:
            return df

        missing_pair_columns = []
        for idx_name in required_indices:
            for pair in INDEX_DEFINITIONS[idx_name]["pairs"]:
                for field in ["open", "high", "low", "close", "tick_volume"]:
                    col = f"{field}_{pair}"
                    if col not in df.columns:
                        missing_pair_columns.append(col)
                        break

        if not missing_pair_columns and not (supporting_ohlcv or {}):
            return df

        try:
            from app.core.analysis.analysis_manager import AnalysisManager

            analysis_manager = AnalysisManager()
            if hasattr(analysis_manager, "_prefetch_currency_pairs"):
                logger.info(
                    "[InferenceFeaturePipeline] Reusing AnalysisManager._prefetch_currency_pairs for %d missing pair columns",
                    len(missing_pair_columns),
                )
                return await analysis_manager._prefetch_currency_pairs(df, cfg)
        except Exception as err:
            logger.warning(
                "[InferenceFeaturePipeline] Shared currency-pair prefetch unavailable; falling back to in-memory merge: %s",
                err,
            )

        if supporting_ohlcv:
            return self._merge_supporting_pairs(df, supporting_ohlcv)
        return df

    async def build_feature_window(
        self,
        df_full: pd.DataFrame,
        supporting_ohlcv: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Run the full feature pipeline on df_full (all bars, no pre-slicing).

        Returns:
            (tensor, reference_close)
            tensor shape: (1, seq_len, 663) float32
            reference_close: Close of the last bar BEFORE slicing
        """
        import time
        t0 = time.perf_counter()
        logger.info(
            "[InferenceFeaturePipeline] 🎬 Starting build_feature_window for dataset='%s' (features_map=%d, seq_len=%d, input_bars=%d, initial_cols=%d)",
            self._dataset_name,
            len(self._feature_names),
            self._seq_len,
            len(df_full),
            len(df_full.columns),
        )

        configs = self._step_configs
        ta_config = configs.get("technical_analysis", {}) or {}
        ci_config_raw = configs.get("currency_indices", {}) or {}
        ci_config = self._coerce_currency_index_config(ci_config_raw)
        astro_config = configs.get("astronomical_analysis", {}) or {}
        snr_config = configs.get("snr_analysis", {}) or {}

        df_full = await self._ensure_currency_index_data(df_full, supporting_ohlcv or {}, ci_config)

        loop = asyncio.get_running_loop()

        # Steps 1-4 are CPU-bound — offload to thread pool
        def _cpu_pipeline(df: pd.DataFrame) -> pd.DataFrame:
            # IMPORTANT: Currency indices must run FIRST (before technical analysis)
            # This matches the step order in analysis.tsx and ensures supporting pair
            # data is available for any technical indicators that depend on it
            t_ci = time.perf_counter()
            df = self.run_currency_indices(df, supporting_ohlcv or {}, ci_config)
            logger.info("[InferenceFeaturePipeline] Step 1/4 (Currency Indices) done in %.3fs — shape: %s", time.perf_counter() - t_ci, df.shape)

            t_ta = time.perf_counter()
            df = self.run_technical_analysis(df, ta_config)
            logger.info("[InferenceFeaturePipeline] Step 2/4 (Technical Analysis) done in %.3fs — shape: %s", time.perf_counter() - t_ta, df.shape)

            if astro_config:
                t_astro = time.perf_counter()
                df = self.run_astronomical_features(df, astro_config)
                logger.info("[InferenceFeaturePipeline] Step 3/4 (Astronomical) done in %.3fs — shape: %s", time.perf_counter() - t_astro, df.shape)

            t_snr = time.perf_counter()
            df = self.run_snr_features(df, snr_config)
            logger.info("[InferenceFeaturePipeline] Step 4/4 (SNR) done in %.3fs — shape: %s", time.perf_counter() - t_snr, df.shape)

            return df

        enriched = await loop.run_in_executor(None, _cpu_pipeline, df_full.copy())

        # Capture reference close BEFORE slicing (REQ-6.3)
        reference_close = float(enriched["Close"].iloc[-1]) if "Close" in enriched.columns else None

        # Steps 5-7
        def _scale_pipeline(df: pd.DataFrame) -> np.ndarray:
            logger.info("[InferenceFeaturePipeline] Step 5/8 Applying train-fitted pre-scaler normalization...")
            df = self._apply_training_normalization(df)
            logger.info("[InferenceFeaturePipeline] Step 6/8 Validating and filling feature schema...")
            df = self.validate_and_fill(df)
            self._save_debug_snapshot(df, "ml_post_validate_pre_scaler")
            logger.info("[InferenceFeaturePipeline] Step 7/8 Applying SelectiveScaler...")
            scaled_df = self.scale(df)
            logger.info("[InferenceFeaturePipeline] Step 8/8 Building (1, %d, %d) model tensor...", self._seq_len, len(self._feature_names))
            return self.build_tensor(scaled_df)

        tensor = await loop.run_in_executor(None, _scale_pipeline, enriched)
        elapsed = time.perf_counter() - t0
        logger.info(
            "[InferenceFeaturePipeline] ✅ Feature window built successfully in %.3fs! Tensor shape: %s, reference_close: %s",
            elapsed,
            tensor.shape,
            reference_close,
        )
        return tensor, reference_close

    # ── Supporting pair merge (same logic as ProprietaryModelRuntime) ─────────

    @staticmethod
    def _normalise_time_column(df: pd.DataFrame) -> pd.DataFrame:
        if "Time" not in df.columns or pd.api.types.is_numeric_dtype(df["Time"]):
            return df
        out = df.copy()
        out["Time"] = pd.to_datetime(out["Time"], errors="coerce").astype("int64") // 10**9
        return out

    def _merge_supporting_pairs(
        self,
        df: pd.DataFrame,
        supporting_ohlcv: Dict[str, List[Dict[str, Any]]],
    ) -> pd.DataFrame:
        if "Time" not in df.columns or not supporting_ohlcv:
            return df
        enriched = self._normalise_time_column(df.copy())
        for pair, bars in supporting_ohlcv.items():
            pair_sym = pair.upper()
            records = []
            for b in bars:
                if isinstance(b, dict):
                    ts = b.get("timestamp") or b.get("time") or b.get("Time")
                    if ts is None:
                        continue
                    records.append({
                        "Time": ts,
                        f"open_{pair_sym}": b.get("open", b.get("Open")),
                        f"high_{pair_sym}": b.get("high", b.get("High")),
                        f"low_{pair_sym}": b.get("low", b.get("Low")),
                        f"close_{pair_sym}": b.get("close", b.get("Close")),
                        f"tick_volume_{pair_sym}": b.get("tick_volume", b.get("volume", b.get("Volume"))),
                    })
                elif isinstance(b, (list, tuple)) and len(b) >= 5:
                    records.append({
                        "Time": b[0],
                        f"open_{pair_sym}": b[1],
                        f"high_{pair_sym}": b[2],
                        f"low_{pair_sym}": b[3],
                        f"close_{pair_sym}": b[4],
                        f"tick_volume_{pair_sym}": b[5] if len(b) > 5 else 0.0,
                    })
            if not records:
                continue
            pair_df = pd.DataFrame(records)
            pair_df = self._normalise_time_column(pair_df)
            pair_df = pair_df.drop_duplicates(subset=["Time"], keep="last")
            val_cols = [c for c in pair_df.columns if c != "Time"]
            enriched = enriched.merge(pair_df[["Time"] + val_cols], on="Time", how="left")
            enriched = enriched.drop_duplicates(subset=["Time"], keep="last")
            enriched[val_cols] = enriched[val_cols].ffill().bfill()
        return enriched
