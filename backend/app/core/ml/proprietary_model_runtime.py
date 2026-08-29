"""Strict serving preflight for local AXE proprietary model artifacts."""

import io
import json
import logging
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import joblib
import numpy as np
import pandas as pd

from app.core.analysis.currency_index import (
    CurrencyIndexCalculator,
    INDEX_DEFINITIONS,
    OHLCV_FIELDS,
    prepare_index_data,
)
from app.core.analysis.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class ProprietaryModelContractError(ValueError):
    def __init__(self, message: str, missing_features: List[str] | None = None):
        super().__init__(message)
        self.missing_features = missing_features or []


class ProprietaryModelRuntime:
    # Canonical active dataset — shared across __init__ and all methods
    _ACTIVE_DATASET: str = "ml_raw_20260809_410"

    # Registry mapping model_id → metadata.
    # dataset_name is manually set here for proprietary (backend-trained) models.
    # On first inference, _resolve_session_from_dataset() queries DB:
    #   dataset_name → MLDataset.session_id → SessionStepResult.step_configs
    # This creates the full audit chain: model → dataset → session → step_configs → reproducibility.
    # Pipeline models (trained via workspace UI) have their dataset_id/session_id
    # already embedded in TrainedModelForAnalysis.step_configs (see models.py).
    _REGISTERED_MODELS: Dict[str, Dict[str, str]] = {
        "axe_genesis": {
            "dataset_name": "ml_raw_20260808_895",
            "checkpoint_dir": "",   # resolved at runtime from _contracts
        },
        "axe_chimera_v8_hybrid": {
            "dataset_name": "ml_raw_20260808_895",
            "checkpoint_dir": "",
        },
        "axe_genesis_v2": {
            "dataset_name": "ml_raw_20260809_410",
            "checkpoint_dir": "axe-genesis",
        },
    }

    # Per-model resolved session cache: model_id → session_id (str or None)
    _session_id_cache: Dict[str, Optional[str]] = {}
    _scaling_config_cache: Dict[str, Dict[str, Any]] = {}
    _DB_LOOKUP_TIMEOUT_SEC: float = 8.0

    def __init__(self) -> None:
        from app.core.ml.inference_feature_pipeline import resolve_dataset_cache_dir
        backend_root = Path(__file__).resolve().parents[3]
        data_root = backend_root / "Backend" / "data"
        cache_dir = resolve_dataset_cache_dir(self._ACTIVE_DATASET)

        self._contracts: Dict[str, Dict[str, Path]] = {
            "axe_genesis": {
                "artifact": data_root / "baseline_v1_best.keras",
                "feature_map": cache_dir / "feature_index_map.json",
                "scaler": cache_dir / "scaler.joblib",
            },
            "axe_chimera_v8_hybrid": {
                "artifact": data_root / "baseline_encoder_v8_hybrid_best.keras",
                "feature_map": cache_dir / "feature_index_map.json",
                "scaler": cache_dir / "scaler.joblib",
            },
            "axe_genesis_v2": {
                "artifact": data_root.parent / "axe-genesis" / "v2_context_best.weights.h5",
                "feature_map": cache_dir / "feature_index_map.json",
                "scaler": cache_dir / "scaler.joblib",
            },
        }
        self._ti_calc = TechnicalIndicators()

    @classmethod
    async def _resolve_session_from_dataset(cls, model_id: str) -> Optional[str]:
        """
        Resolve session_id for a model via DB lookup:
          1. Read _REGISTERED_MODELS[model_id].dataset_name
          2. Query MLDataset WHERE dataset_name = ? ORDER BY created_at DESC
          3. Return MLDataset.session_id (the session that produced this dataset)
          4. Falls back to latest MLDataset session if no exact name match
          5. Returns None on any DB error (caller degrades gracefully)

        Result is cached in _session_id_cache so subsequent calls are O(1).
        """
        if model_id in cls._session_id_cache:
            return cls._session_id_cache[model_id]

        registry_entry = cls._REGISTERED_MODELS.get(model_id)
        if not registry_entry:
            cls._session_id_cache[model_id] = None
            return None

        dataset_name = registry_entry.get("dataset_name") or cls._ACTIVE_DATASET

        try:
            from sqlalchemy import select, desc
            from app.api.routes.data.database import AsyncPostgresSessionLocal
            from app.database.models import MLDataset

            async def _query() -> Optional[str]:
                session_id: Optional[str] = None
                async with AsyncPostgresSessionLocal() as db:
                    stmt = (
                        select(MLDataset.session_id)
                        .where(MLDataset.dataset_name == dataset_name)
                        .order_by(desc(MLDataset.created_at))
                        .limit(1)
                    )
                    result = await db.execute(stmt)
                    row = result.scalar()
                    if row:
                        session_id = str(row)

                    if session_id is None:
                        stmt_latest = (
                            select(MLDataset.session_id)
                            .order_by(desc(MLDataset.created_at))
                            .limit(1)
                        )
                        result_latest = await db.execute(stmt_latest)
                        row_latest = result_latest.scalar()
                        if row_latest:
                            session_id = str(row_latest)
                            logger.warning(
                                "[AXE Runtime] No MLDataset found for dataset_name='%s', "
                                "fell back to latest session='%s'",
                                dataset_name,
                                session_id,
                            )
                return session_id

            session_id = await asyncio.wait_for(_query(), timeout=cls._DB_LOOKUP_TIMEOUT_SEC)

            cls._session_id_cache[model_id] = session_id
            logger.info(
                "[AXE Runtime] Resolved session_id='%s' for model='%s' (dataset='%s')",
                session_id,
                model_id,
                dataset_name,
            )
            return session_id

        except Exception as err:
            logger.warning(
                "[AXE Runtime] DB session resolution failed for model='%s': %s(%r) — continuing without session_id",
                model_id,
                type(err).__name__,
                err,
            )
            cls._session_id_cache[model_id] = None
            return None

    @staticmethod
    def _coerce_json_dict(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}

    @classmethod
    async def _load_dataset_scaling_config(cls, model_id: str) -> Dict[str, Any]:
        """Load the training scaling_config persisted with the model's MLDataset."""
        registry_entry = cls._REGISTERED_MODELS.get(model_id, {})
        dataset_name = registry_entry.get("dataset_name") or cls._ACTIVE_DATASET
        if dataset_name in cls._scaling_config_cache:
            return cls._scaling_config_cache[dataset_name]

        try:
            from sqlalchemy import select, desc
            from app.api.routes.data.database import AsyncPostgresSessionLocal
            from app.database.models import MLDataset

            async def _query() -> Dict[str, Any]:
                async with AsyncPostgresSessionLocal() as db:
                    stmt = (
                        select(MLDataset.scaling_config)
                        .where(MLDataset.dataset_name == dataset_name)
                        .order_by(desc(MLDataset.created_at))
                        .limit(1)
                    )
                    result = await db.execute(stmt)
                    return cls._coerce_json_dict(result.scalar())

            scaling_config = await asyncio.wait_for(_query(), timeout=cls._DB_LOOKUP_TIMEOUT_SEC)

            if scaling_config:
                logger.info(
                    "[AXE Runtime] Loaded scaling_config from DB for dataset='%s' "
                    "(keys=%s)",
                    dataset_name,
                    sorted(scaling_config.keys())[:12],
                )
            else:
                logger.warning(
                    "[AXE Runtime] No DB scaling_config found for dataset='%s'; "
                    "output denormalization will use feature-map fallback.",
                    dataset_name,
                )
            cls._scaling_config_cache[dataset_name] = scaling_config
            return scaling_config
        except Exception as err:
            logger.warning(
                "[AXE Runtime] DB scaling_config lookup failed for dataset='%s': %s(%r)",
                dataset_name,
                type(err).__name__,
                err,
            )
            cls._scaling_config_cache[dataset_name] = {}
            return {}

    @classmethod
    async def _ensure_dataset_artifacts_from_db(
        cls,
        model_id: str,
        *,
        feature_map_path: Path,
        scaler_path: Path,
    ) -> Dict[str, Any]:
        """
        Ensure local serving artifacts mirror the DB-backed MLDataset record.

        The DB remains the source of truth. Disk files are just the fast local
        cache used by joblib/Keras-serving code paths.
        """
        registry_entry = cls._REGISTERED_MODELS.get(model_id, {})
        dataset_name = registry_entry.get("dataset_name") or cls._ACTIVE_DATASET
        scaling_config = await cls._load_dataset_scaling_config(model_id)

        scaler_missing = not scaler_path.is_file()
        fmap_missing = not feature_map_path.is_file()
        if not scaler_missing and not fmap_missing:
            return scaling_config

        try:
            from sqlalchemy import select, desc
            from app.api.routes.data.database import AsyncPostgresSessionLocal
            from app.database.models import MLDataset

            async def _query():
                async with AsyncPostgresSessionLocal() as db:
                    stmt = (
                        select(MLDataset.scaler_binary, MLDataset.scaling_config)
                        .where(MLDataset.dataset_name == dataset_name)
                        .order_by(desc(MLDataset.created_at))
                        .limit(1)
                    )
                    result = await db.execute(stmt)
                    return result.first()

            row = await asyncio.wait_for(_query(), timeout=cls._DB_LOOKUP_TIMEOUT_SEC)

            if not row:
                return scaling_config

            scaler_binary, db_scaling_config = row
            db_scaling_config = cls._coerce_json_dict(db_scaling_config)
            if db_scaling_config:
                scaling_config = db_scaling_config
                cls._scaling_config_cache[dataset_name] = db_scaling_config

            scaler_path.parent.mkdir(parents=True, exist_ok=True)
            feature_map_path.parent.mkdir(parents=True, exist_ok=True)

            if scaler_missing and scaler_binary:
                scaler_path.write_bytes(scaler_binary)
                logger.info(
                    "[AXE Runtime] Extracted DB scaler_binary for dataset='%s' to %s",
                    dataset_name,
                    scaler_path,
                )

            if fmap_missing and scaling_config.get("feature_index_map"):
                feature_map = scaling_config["feature_index_map"]
                feature_map_path.write_text(json.dumps({
                    "feature_index_map": feature_map,
                    "feature_columns": list(feature_map.keys()),
                    "step_configs": scaling_config.get("step_configs", {}),
                }, indent=2))
                logger.info(
                    "[AXE Runtime] Extracted DB feature_index_map for dataset='%s' to %s (%d features)",
                    dataset_name,
                    feature_map_path,
                    len(feature_map),
                )

            return scaling_config
        except Exception as err:
            logger.warning(
                "[AXE Runtime] DB artifact extraction failed for dataset='%s': %s(%r)",
                dataset_name,
                type(err).__name__,
                err,
            )
            return scaling_config

    @staticmethod
    def _load_feature_names(feature_map_path: Path) -> List[str]:
        try:
            payload = json.loads(feature_map_path.read_text())
            feature_map = payload.get("feature_index_map", payload)
            ordered = [name for name, _ in sorted(feature_map.items(), key=lambda item: int(item[1]))]
            positions = [int(feature_map[name]) for name in ordered]
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ProprietaryModelContractError(
                f"Unable to load proprietary feature map '{feature_map_path.name}': {error}"
            ) from error
        if positions != list(range(len(positions))):
            raise ProprietaryModelContractError(
                f"Proprietary feature map '{feature_map_path.name}' is not contiguous and zero-based."
            )
        return ordered

    def _contract(self, model_id: str, variant_tag: str = "market") -> Dict[str, Path]:
        from app.core.ml.model_store import model_store
        from app.core.ml.inference_feature_pipeline import resolve_dataset_cache_dir
        
        meta = model_store.get_meta(model_id, variant_tag)
        if not meta:
            # Fallback to hardcoded contracts
            contract = self._contracts.get(model_id)
            if not contract:
                raise ProprietaryModelContractError(
                    f"Selected model '{model_id}' (variant '{variant_tag}') has no registered serving artifact contract."
                )
            return contract
            
        dataset_name = meta.get("dataset_name") or self._ACTIVE_DATASET
        cache_dir = resolve_dataset_cache_dir(dataset_name)
        backend_root = Path(__file__).resolve().parents[3]
        data_root = backend_root / "Backend" / "axe_genesis" / "weights"
        weights_path = data_root / meta.get("weights_path", "")
        
        return {
            "artifact": weights_path,
            "feature_map": cache_dir / "feature_index_map.json",
            "scaler": cache_dir / "scaler.joblib",
        }

    def required_indices(self, model_id: str, variant_tag: str = "market") -> List[str]:
        try:
            required_features = self._load_feature_names(self._contract(model_id, variant_tag)["feature_map"])
        except ProprietaryModelContractError:
            return []
        indices = [
            index_name
            for index_name in INDEX_DEFINITIONS.keys()
            if any(name.startswith(f"{index_name}_") for name in required_features)
        ]
        return indices

    def required_pairs(self, model_id: str, variant_tag: str = "market") -> List[str]:
        pairs: Set[str] = set()
        for index_name in self.required_indices(model_id, variant_tag):
            pairs.update(INDEX_DEFINITIONS[index_name]["pairs"].keys())
        return sorted(pairs)

    def is_model_ready(self, model_id: str, variant_tag: str = "market") -> bool:
        """Check if all serving artifacts for model_id exist on disk."""
        try:
            contract = self._contract(model_id, variant_tag)
            return all(path.is_file() for path in contract.values())
        except Exception:
            return False

    @staticmethod
    def _normalise_time_column(df: pd.DataFrame) -> pd.DataFrame:
        if "Time" not in df.columns:
            return df
        if pd.api.types.is_numeric_dtype(df["Time"]):
            return df
        normalised = df.copy()
        normalised["Time"] = pd.to_datetime(normalised["Time"], errors="coerce").astype("int64") // 10**9
        return normalised

    @staticmethod
    def _bars_to_pair_frame(pair: str, bars: List[Dict[str, Any]]) -> pd.DataFrame:
        records = []
        for bar in bars:
            timestamp = bar.get("timestamp") or bar.get("time") or bar.get("Time")
            if timestamp is None:
                continue
            records.append({
                "Time": timestamp,
                f"open_{pair}": bar.get("open"),
                f"high_{pair}": bar.get("high"),
                f"low_{pair}": bar.get("low"),
                f"close_{pair}": bar.get("close"),
                f"tick_volume_{pair}": bar.get("tick_volume", bar.get("volume")),
            })
        pair_df = pd.DataFrame(records)
        if pair_df.empty:
            return pair_df
        pair_df = ProprietaryModelRuntime._normalise_time_column(pair_df)
        return pair_df.drop_duplicates(subset=["Time"], keep="last")

    def _merge_supporting_pairs(
        self,
        feature_window: pd.DataFrame,
        supporting_ohlcv: Dict[str, List[Dict[str, Any]]],
    ) -> pd.DataFrame:
        if "Time" not in feature_window.columns or not supporting_ohlcv:
            return feature_window

        enriched = self._normalise_time_column(feature_window.copy())
        supplied_pairs = []
        for pair, bars in supporting_ohlcv.items():
            pair_symbol = pair.upper()
            pair_df = self._bars_to_pair_frame(pair_symbol, bars)
            if pair_df.empty:
                continue
            value_columns = [column for column in pair_df.columns if column != "Time"]
            enriched = enriched.merge(pair_df[["Time"] + value_columns], on="Time", how="left")
            enriched[value_columns] = enriched[value_columns].ffill().bfill()
            supplied_pairs.append(pair_symbol)

        if supplied_pairs:
            logger.info(
                "[AXE Runtime] Merged hydration supporting OHLCV pairs: %s",
                sorted(set(supplied_pairs)),
            )
        return enriched

    def _add_currency_index_features(self, feature_window: pd.DataFrame, model_id: str) -> pd.DataFrame:
        required_indices = self.required_indices(model_id)
        if not required_indices:
            return feature_window

        enriched = feature_window.copy()
        pair_enriched = prepare_index_data(enriched)
        index_frame = CurrencyIndexCalculator(pair_enriched).to_dataframe(required_indices)
        if index_frame.empty:
            return enriched

        index_frame.index = enriched.index
        for column in index_frame.columns:
            enriched[column] = index_frame[column]

        for index_name in required_indices:
            ohlcv_map = {
                f"{index_name}_open": "Open",
                f"{index_name}_high": "High",
                f"{index_name}_low": "Low",
                f"{index_name}_close": "Close",
                f"{index_name}_tick_volume": "Volume",
            }
            if not all(column in enriched.columns for column in ohlcv_map):
                continue
            ti_input = enriched[list(ohlcv_map.keys())].rename(columns=ohlcv_map)
            try:
                ti_output = self._ti_calc.calculate_all_indicators(ti_input, mode="inference")
            except Exception as error:
                logger.warning("[AXE Runtime] Could not compute %s index technical features: %s", index_name, error)
                continue
            base_cols = set(["Open", "High", "Low", "Close", "Volume"])
            prefixed_columns: Dict[str, pd.Series] = {}
            for column in ti_output.columns:
                if column in base_cols:
                    continue
                prefixed = f"{index_name}_{column}"
                if prefixed in enriched.columns:
                    continue
                series = ti_output[column]
                if len(series) != len(enriched):
                    logger.debug(
                        "[AXE Runtime] Skipping %s: length %d != feature window %d",
                        prefixed,
                        len(series),
                        len(enriched),
                    )
                    continue
                prefixed_columns[prefixed] = pd.Series(series.to_numpy(), index=enriched.index)
            if prefixed_columns:
                enriched = pd.concat([enriched, pd.DataFrame(prefixed_columns, index=enriched.index)], axis=1)

        logger.info(
            "[AXE Runtime] Currency index enrichment complete: indices=%s columns=%d",
            required_indices,
            len(enriched.columns),
        )
        return enriched

    # Class-level pipeline cache: dataset_name → InferenceFeaturePipeline
    _pipeline_cache: Dict[str, Any] = {}

    async def predict(
        self,
        model_id: str,
        symbol: str,
        timeframe: str,
        feature_window: pd.DataFrame,
        snr_features: Dict[str, float],
        supporting_ohlcv: Dict[str, List[Dict[str, Any]]] | None = None,
        variant_tag: str = "market",
    ) -> Dict[str, Any]:
        import asyncio
        del symbol, timeframe
        contract = self._contract(model_id, variant_tag)
        scaling_config = await self._ensure_dataset_artifacts_from_db(
            model_id,
            feature_map_path=contract["feature_map"],
            scaler_path=contract["scaler"],
        )
        missing_artifacts = [name for name, path in contract.items() if not path.is_file()]
        if missing_artifacts:
            raise ProprietaryModelContractError(
                f"Selected model '{model_id}' is missing required serving artifacts: {', '.join(missing_artifacts)}."
            )

        # ── Phase 3: Build feature tensor via AnalysisManager helper or InferenceFeaturePipeline ─────────
        # This replaces the old inline merge+TI+slice block.
        # The pipeline:
        #   1. Loads step_configs from DB (cached after first call)
        #   2. Runs TI → currency indices → astro → SNR on the FULL bar window
        #   3. Validates 663 features, zero-fills gaps, logs warnings
        #   4. Scales with SelectiveScaler over ALL rows
        #   5. Slices last 90 rows → (1, 90, 663) float32
        # The 90-row slice is LAST, not first — fixing the original lookback bug.
        from app.core.ml.inference_feature_pipeline import InferenceFeaturePipeline, InferenceContractError

        # Attempt to use AnalysisManager's helper to build the inference feature
        # window so the AnalysisManager can centralise any session/step-config
        # lookups and prefetch behaviour. Fall back to the pipeline directly
        # if AnalysisManager is not available or raises.
        prebuilt_tensor = None
        reference_close = None
        registry_entry = self._REGISTERED_MODELS.get(model_id, {})
        dataset_name = registry_entry.get("dataset_name") or self._ACTIVE_DATASET

        try:
            from app.core.analysis.analysis_manager import AnalysisManager

            am = AnalysisManager()
            try:
                prebuilt_tensor, reference_close = await am.build_inference_feature_window(
                    feature_window=feature_window,
                    supporting_ohlcv=supporting_ohlcv or {},
                    model_id=model_id,
                    dataset_name=dataset_name,
                )
                logger.info("[AXE Runtime] Built feature window via AnalysisManager helper: %s", getattr(prebuilt_tensor, 'shape', None))
            except Exception as am_err:
                logger.warning("[AXE Runtime] AnalysisManager helper failed, falling back to pipeline: %s", am_err)
                prebuilt_tensor = None
        except Exception:
            # AnalysisManager import may fail in constrained contexts; ignore and fallback
            prebuilt_tensor = None

        if prebuilt_tensor is None:
            # Resolve dataset_name from the model registry (proprietary models carry it explicitly).
            # _resolve_session_from_dataset() also populates _session_id_cache for audit logging.
            await self._resolve_session_from_dataset(model_id)  # warms _session_id_cache

            pipeline_key = f"{model_id}:{dataset_name}"
            if pipeline_key not in self._pipeline_cache:
                self._pipeline_cache[pipeline_key] = await InferenceFeaturePipeline.get_or_create(
                    dataset_name=dataset_name,
                    feature_map_path=contract["feature_map"],
                    scaler_path=contract["scaler"],
                )
            pipeline: InferenceFeaturePipeline = self._pipeline_cache[pipeline_key]

            try:
                logger.info(
                    "[AXE Runtime] Building feature window for model=%s rows=%s cols=%s supporting_pairs=%s",
                    model_id,
                    getattr(feature_window, 'shape', None),
                    list(feature_window.columns)[:10] if hasattr(feature_window, 'columns') else None,
                    list((supporting_ohlcv or {}).keys())[:10],
                )
                prebuilt_tensor, reference_close = await pipeline.build_feature_window(
                    feature_window, supporting_ohlcv or {}
                )
                logger.info("[AXE Runtime] Feature window built: tensor_shape=%s reference_close=%s", getattr(prebuilt_tensor, 'shape', None), reference_close)
            except Exception as pipe_err:
                # Surface pipeline errors with the expected contract error type
                if isinstance(pipe_err, InferenceContractError):
                    raise ProprietaryModelContractError(
                        str(pipe_err),
                        missing_features=getattr(pipe_err, "missing_features", []),
                    ) from pipe_err
                raise

        required_features = self._load_feature_names(contract["feature_map"])

        # ── Phase 4: Execute inference ─────────────────────────────────────────
        # _run_inference contains blocking operations (model load, predict).
        # asyncio.to_thread() keeps the event loop free.
        return await asyncio.to_thread(
            self._run_inference,
            model_id=model_id,
            variant_tag=variant_tag,
            contract=contract,
            feature_window=feature_window,
            required_features=required_features,
            prebuilt_tensor=prebuilt_tensor,
            reference_close=reference_close,
            scaling_config=scaling_config,
            snr_features=snr_features,
        )


    # ── Lazy caches: one scaler + one model per model_id (loaded on first call) ──
    # Avoids re-loading heavy .keras files on every inference call.
    _scaler_cache: Dict[str, Any] = {}
    _model_cache: Dict[str, Any] = {}

    def _load_scaler_cached(self, model_id: str, scaler_path: Path) -> Any:
        if model_id not in self._scaler_cache:
            logger.info("[AXE Runtime] Loading scaler for %s from %s", model_id, scaler_path.name)
            self._scaler_cache[model_id] = joblib.load(scaler_path)
        return self._scaler_cache[model_id]

    def _load_model_cached(self, model_id: str, artifact_path: Path) -> Any:
        if model_id not in self._model_cache:
            if not artifact_path.is_file():
                logger.error("[AXE Runtime] Model artifact not found for %s: %s", model_id, artifact_path)
                raise ProprietaryModelContractError(f"Model artifact missing: {artifact_path}")

            if model_id == "axe_genesis_v2":
                from app.core.ml.axe_genesis_v2_runtime import AXEGenesisV2Runtime
                logger.info("[AXE Runtime] Initializing AXEGenesisV2Runtime for %s", model_id)
                from app.core.ml.inference_feature_pipeline import resolve_dataset_cache_dir
                checkpoint_dir = artifact_path.parent
                registry_entry = self._REGISTERED_MODELS.get(model_id, {})
                dataset_name = registry_entry.get("dataset_name") or self._ACTIVE_DATASET
                dataset_dir = resolve_dataset_cache_dir(dataset_name)
                self._model_cache[model_id] = AXEGenesisV2Runtime(checkpoint_dir=checkpoint_dir, dataset_dir=dataset_dir)
            else:
                import tensorflow as tf  # lazy import — keeps startup fast
                from app.core.ml.custom_keras_objects import register_all, get_custom_objects

                register_all()

                logger.info("[AXE Runtime] Loading Keras model for %s from %s", model_id, artifact_path.name)
                self._model_cache[model_id] = tf.keras.models.load_model(
                    str(artifact_path),
                    custom_objects=get_custom_objects(),
                    safe_mode=False,
                )
            logger.info("[AXE Runtime] Model %s ready", model_id)
        return self._model_cache[model_id]

    def _run_inference(
        self,
        *,
        model_id: str,
        variant_tag: str,
        contract: Dict[str, Path],
        feature_window: pd.DataFrame,
        required_features: List[str],
        prebuilt_tensor: Optional[np.ndarray] = None,
        reference_close: Optional[float] = None,
        scaling_config: Optional[Dict[str, Any]] = None,
        snr_features: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Core inference execution:
          1. Load scaler (cached) — SelectiveScaler fitted during training.
          2. Build scaler input: select and order columns by scaler.feature_names
             (= all_cols = the exact column order the scaler was fitted on).
          3. SelectiveScaler.transform — applies price/diff/other sub-scalers.
          4. Extract sequence window in feature_index_map order → (1, seq_len, n_features).
          5. Load model (cached) and perform forward pass.
          6. Map raw output list to named head dict.
          7. Denormalize heads to real-world values via OutputDenormalizer.
        """
        if prebuilt_tensor is not None:
            # ── Fast path: tensor already built by InferenceFeaturePipeline ──────
            # Scaling and slicing already applied — skip scaler assembly entirely.
            X = prebuilt_tensor
            logger.info(
                "[AXE Runtime] Using prebuilt tensor %s for '%s' (skip re-scaling)",
                X.shape, model_id,
            )
        else:
            # ── Legacy path: inline scaling (backward compat for direct _run_inference calls) ──
            scaler = self._load_scaler_cached(model_id, contract["scaler"])
            scaler_cols: List[str] = list(getattr(scaler, "feature_names", None) or required_features)

            missing_scaler_cols = [c for c in scaler_cols if c not in feature_window.columns]
            if missing_scaler_cols:
                n_target = sum(1 for c in missing_scaler_cols if c.startswith("adv_target_") or c.startswith("target_"))
                n_other = len(missing_scaler_cols) - n_target
                logger.info(
                    "[AXE Runtime] Scaler has %d extra columns absent from live window "
                    "(%d output labels zero-filled, %d other zero-filled).",
                    len(missing_scaler_cols), n_target, n_other,
                )

            scale_input = pd.DataFrame(index=feature_window.index)
            for col in scaler_cols:
                scale_input[col] = feature_window[col].values if col in feature_window.columns else 0.0

            try:
                scaled_arr = scaler.transform(scale_input)
            except Exception as err:
                raise ProprietaryModelContractError(
                    f"SelectiveScaler.transform failed for '{model_id}': {err}"
                ) from err

            scaled_df = pd.DataFrame(
                scaled_arr if isinstance(scaled_arr, np.ndarray) else np.asarray(scaled_arr),
                index=feature_window.index,
                columns=scaler_cols,
            )

            sequence_length = 90
            missing_seq = [c for c in required_features if c not in scaled_df.columns]
            if missing_seq:
                raise ProprietaryModelContractError(
                    f"Scaled frame is missing {len(missing_seq)} sequence features after transform.",
                    missing_features=missing_seq,
                )

            window = scaled_df[required_features].iloc[-sequence_length:].values.astype("float32")
            if window.shape[0] < sequence_length:
                raise ProprietaryModelContractError(
                    f"Not enough rows for inference window: need {sequence_length}, got {window.shape[0]}."
                )
            X = window.reshape(1, sequence_length, len(required_features))

        # ── 5 & 6. Load model + predict (raw only) ───────────────────────────
        from app.core.ml.model_store import model_store
        model_or_v2 = model_store.get(model_id, variant_tag)
        if model_or_v2 is None:
            model_or_v2 = self._load_model_cached(model_id, contract["artifact"])
        if model_id == "axe_genesis_v2":
            raw_named_heads = model_or_v2.predict(
                X,
                feature_window=feature_window,
                snr_features=snr_features,
                raw_only=True,
            )
        else:
            raw_preds = model_or_v2.predict(X, verbose=0)
            if isinstance(raw_preds, dict):
                raw_named_heads = raw_preds
            else:
                raw_named_heads = {}
                for key, value in zip(getattr(model_or_v2, "output_names", []), raw_preds if isinstance(raw_preds, (list, tuple)) else [raw_preds]):
                    raw_named_heads[key] = value

        logger.info(
            "[AXE Runtime] Raw inference complete for %s — %d heads produced: %s",
            model_id,
            len(raw_named_heads),
            sorted(raw_named_heads.keys()),
        )

        return raw_named_heads

    @staticmethod
    @staticmethod
    def _with_live_ohlc_sigmoid_anchors(
        scaling_config: Dict[str, Any],
        feature_window: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Legacy transform hook retained only for audit/debug. It no longer mutates outputs."""
        return scaling_config

    def _inverse_scaled_output_targets(
        self,
        model_id: str,
        contract: Dict[str, Path],
        named_heads: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Legacy transform hook retained only for audit. It does not mutate raw outputs."""
        return named_heads


proprietary_model_runtime = ProprietaryModelRuntime()
