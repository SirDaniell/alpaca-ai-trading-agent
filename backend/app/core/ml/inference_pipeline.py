
from typing import Dict, Any, List, Optional
import asyncio
import concurrent.futures as _cf
import pandas as pd
import numpy as np
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import logging
from dataclasses import asdict
import json
import os

from app.database.repositories.analysis_repository import AnalysisRepository
from app.core.analysis.technical_indicators import TechnicalIndicators, IndicatorConfig
from app.core.analysis.astronomy.astronomical import AstronomicalFeatureGenerator
from app.core.analysis.support_resistance import (
    detect_snr_levels_sequential,
    create_clustered_zones_sequential,
    extract_snr_features
)
import joblib
import io
import base64

logger = logging.getLogger(__name__)

# Dedicated executors for CPU-bound / blocking operations in the inference pipeline.
# Keeps the async event loop free during pandas feature generation and model.predict().
_feature_executor = _cf.ThreadPoolExecutor(max_workers=4, thread_name_prefix="feature_worker")
_model_executor   = _cf.ThreadPoolExecutor(max_workers=4, thread_name_prefix="model_worker")


class InferenceContractError(ValueError):
    """Raised when live data cannot reproduce the selected model's training input."""

    def __init__(self, message: str, *, missing_features: Optional[List[str]] = None):
        super().__init__(message)
        self.missing_features = missing_features or []


class InferencePipeline:
    def __init__(self, db: AsyncSession):
        self.repo = AnalysisRepository(db)

    async def load_session_config(self, session_id: UUID) -> Dict[str, Any]:
        """
        Load component configurations from a saved analysis session.
        """
        session = await self.repo.get_session(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return {}
        
        configs = {}
        for step in session.steps:
            configs[step.step_key] = step.configuration or {}
            
            # Also load "selected_features" from 'feature_review' or 'ml_preparation'
            # Depending on where it's stored. Usually ml_preparation has the final list.
            
        return configs

    async def run_feature_generation(self, session_id: UUID, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Replicates the analysis pipeline feature generation for a given session on new input data.
        CPU-bound work is offloaded to a thread-pool executor so the async event loop
        remains responsive for WebSocket ticks and concurrent requests.
        """
        configs = await self.load_session_config(session_id)
        if not configs:
            raise ValueError(f"Could not load configurations for session {session_id}")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _feature_executor,
            self._run_feature_generation_sync,
            configs,
            input_data,
        )

    def _run_feature_generation_sync(self, configs: Dict[str, Any], input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Synchronous inner implementation of feature generation.
        Called exclusively from run_feature_generation via run_in_executor.
        """
        df = input_data.copy()

        # Ensure timestamp column availability (standardizing to 'time')
        if 'time' not in df.columns and 'Date' in df.columns:
            df['time'] = pd.to_datetime(df['Date'])
        elif 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])

        # ----------------------------------------------------------------
        # 1. Technical Analysis
        # ----------------------------------------------------------------
        if 'technical_analysis' in configs:
            logger.info("Applying Technical Analysis...")
            ta_config_dict = configs['technical_analysis']
            
            # Attempt to map dictionary to IndicatorConfig
            # We filter keys that are actually in IndicatorConfig fields to avoid kwargs error
            # This requires inspecting IndicatorConfig
            valid_keys = IndicatorConfig.__annotations__.keys()
            filtered_config = {k: v for k, v in ta_config_dict.items() if k in valid_keys}
            
            # Instantiate config
            indi_config = IndicatorConfig(**filtered_config)
            
            # Run Indicators
            ti = TechnicalIndicators(config=indi_config)
            df = ti.calculate_all_indicators(df)

        # ----------------------------------------------------------------
        # 2. Support & Resistance Analysis
        # ----------------------------------------------------------------
        if 'snr_analysis' in configs:
            logger.info("Applying SNR Analysis Features...")
            snr_config = configs['snr_analysis']
            lookback = snr_config.get('lookback_period', 200)
            min_dist = snr_config.get('min_distance_pct', 0.5)
            n_clusters = snr_config.get('n_clusters', 16)
            zone_width = snr_config.get('zone_width', 0.004)
            
            # For inference, we calculate based on the current window
            # Usually df would have enough history (lookback)
            if len(df) >= lookback:
                # We only need the features for the LAST row for live inference
                # but we'll calculate for all provided if it's a batch
                for i in range(len(df)):
                    # Detect levels up to current index
                    levels = detect_snr_levels_sequential(df, i, lookback, min_dist)
                    
                    # Create zones
                    # slice for zones calculation
                    if i >= lookback:
                        slice_start = i - lookback
                    else:
                        slice_start = 0
                    df_slice = df.iloc[slice_start : i + 1]
                    
                    zones = create_clustered_zones_sequential(
                        levels, df_slice, n_clusters=n_clusters, zone_width=zone_width
                    )
                    
                    # Extract features
                    curr_price = df['Close'].iloc[i]
                    snr_feats = extract_snr_features(curr_price, levels, zones)
                    
                    # Apply SNR features to DataFrame
                    for k, v in snr_feats.items():
                        df.at[df.index[i], k] = v
                        
                    logger.debug(f"Added SNR features at index {i}: {list(snr_feats.keys())}")

        # ----------------------------------------------------------------
        # 3. Astronomical Analysis
        # ----------------------------------------------------------------
        if 'astronomical_analysis' in configs:
            logger.info("Applying Astronomical Analysis...")
            astro_config = configs['astronomical_analysis']
            
            # Instantiate Generator
            astro = AstronomicalFeatureGenerator(
                use_minor_aspects=astro_config.get('use_minor_aspects', False),
                observer_lat=astro_config.get('observer_lat', 0.0),
                observer_lon=astro_config.get('observer_lon', 0.0),
                # Add other config mappings
            )
            
            # Determine features to generate (optimization)
            # If we know the needed columns from ML Prep, we only generate those.
            # Otherwise we generate all?
            # For now, generate comprehensive set or use default list
            target_features = astro.create_all_possible_features().keys()
            
            # Iterate rows
            astro_features_list = []
            for idx, row in df.iterrows():
                dt = row['time']
                feats = astro.generate_features_for_date(dt, list(target_features))
                astro_features_list.append(feats)
            
            astro_df = pd.DataFrame(astro_features_list)
            # Merge back
            df = pd.concat([df.reset_index(drop=True), astro_df.reset_index(drop=True)], axis=1)

        return df

    @staticmethod
    def _ordered_feature_names(feature_map: Dict[str, Any], source: str) -> List[str]:
        if not feature_map:
            return []
        try:
            ordered = [name for name, _ in sorted(feature_map.items(), key=lambda item: int(item[1]))]
            positions = [int(feature_map[name]) for name in ordered]
        except (TypeError, ValueError) as error:
            raise InferenceContractError(
                f"{source} contains an invalid feature-index map: {error}"
            ) from error
        if positions != list(range(len(positions))):
            raise InferenceContractError(
                f"{source} feature-index map is not a contiguous zero-based tensor schema."
            )
        return ordered

    async def load_feature_contract(self, session_id: UUID, scaler: Any) -> Dict[str, List[str]]:
        session = await self.repo.get_session(session_id)
        if not session:
            raise InferenceContractError("Inference session was not found.")
        dataset = await self.load_ml_dataset(session)
        if not dataset:
            raise InferenceContractError(
                "Session has no persisted MLDataset contract. Live inference requires the "
                "ProcessingManager feature map, scaling config, and fitted scaler."
            )

        scaling_config = getattr(dataset, "scaling_config", {}) or {}
        if isinstance(scaling_config, str):
            try:
                scaling_config = json.loads(scaling_config)
            except json.JSONDecodeError as error:
                raise InferenceContractError("Persisted ML scaling_config is not valid JSON.") from error

        pm_sequence_features = self._ordered_feature_names(
            scaling_config.get("feature_index_map", {}),
            "ProcessingManager feature_index_map",
        )
        pm_scaler_features = list(scaling_config.get("columns_to_scale", []) or [])
        dataset_features = list(getattr(dataset, "feature_columns", []) or [])
        scaler_sequence_features = list(getattr(scaler, "sequence_feature_names", []) or [])
        scaler_sequence_map = getattr(scaler, "sequence_feature_index_map", {}) or {}
        if scaler_sequence_map:
            mapped_scaler_sequence_features = self._ordered_feature_names(
                scaler_sequence_map,
                "SelectiveScaler sequence_feature_index_map",
            )
            if scaler_sequence_features and scaler_sequence_features != mapped_scaler_sequence_features:
                raise InferenceContractError(
                    "SelectiveScaler sequence names and sequence feature-index map disagree."
                )
            scaler_sequence_features = mapped_scaler_sequence_features

        scaler_features = list(getattr(scaler, "feature_names", []) or getattr(scaler, "all_cols", []) or [])
        if not pm_sequence_features or not pm_scaler_features or not scaler_sequence_features or not scaler_features:
            raise InferenceContractError(
                "ML-preparation artifacts are incomplete; feature map, scaling columns, and scaler schemas are all required."
            )
        if pm_sequence_features != scaler_sequence_features:
            raise InferenceContractError(
                "ProcessingManager feature_index_map does not match the persisted scaler sequence schema."
            )
        if dataset_features and dataset_features != pm_sequence_features:
            raise InferenceContractError(
                "Dataset feature_columns does not match the ProcessingManager feature_index_map."
            )
        if pm_scaler_features != scaler_features:
            raise InferenceContractError(
                "ProcessingManager columns_to_scale does not match the persisted scaler schema."
            )
        return {
            "sequence_features": pm_sequence_features,
            "scaler_features": pm_scaler_features,
        }

    async def load_scaler(self, session_id: UUID) -> Any:
        """
        Loads the fitted scaler for a given session.
        """
        session = await self.repo.get_session(session_id)
        if not session:
            return None
            
        dataset = await self.load_ml_dataset(session)
        if not dataset:
            return None
            
        try:
            scaler_blob = getattr(dataset, "scaler_binary", None) or getattr(dataset, "binary_data", None)
            if not scaler_blob:
                return None
            scaler_data = base64.b64decode(scaler_blob) if isinstance(scaler_blob, str) else scaler_blob
            scaler = joblib.load(io.BytesIO(scaler_data))
            return scaler
        except Exception as e:
            logger.error(f"Failed to load scaler: {e}")
            return None

    async def load_ml_dataset(self, session: Any) -> Any:
        ml_step = next((step for step in session.steps if step.step_key == "ml_preparation"), None)
        if not ml_step or not ml_step.result or not ml_step.result.dataset_id:
            return None
        from app.database.models import MLDataset
        result = await self.repo.db.execute(
            select(MLDataset).where(MLDataset.dataset_id == ml_step.result.dataset_id)
        )
        return result.scalar_one_or_none()

    async def load_model(self, model_id: UUID) -> Any:
        """
        Loads a trained model by ID.
        """
        # We need a repository or direct DB access for TrainedModel
        # For now, let's assume we can fetch it via SQLAlchemy
        from app.database.models import TrainedModel
        stmt = select(TrainedModel).where(TrainedModel.id == model_id)
        result = await self.repo.db.execute(stmt)
        model_rec = result.scalar_one_or_none()
        
        if not model_rec or not model_rec.model_path:
            logger.error(f"Model {model_id} not found or has no path")
            return None
        
        # Validate model file exists
        if not os.path.exists(model_rec.model_path):
            logger.error(f"Model file not found: {model_rec.model_path}")
            raise FileNotFoundError(f"Model file not found: {model_rec.model_path}")
            
        try:
            # Check architecture to determine loader
            if 'lstm' in model_rec.architecture_id.lower():
                import tensorflow as tf
                model = tf.keras.models.load_model(model_rec.model_path)
                return model
            else:
                # Fallback to joblib for sklearn models
                import joblib
                model = joblib.load(model_rec.model_path)
                return model
        except Exception as e:
            logger.error(f"Failed to load model from {model_rec.model_path}: {e}")
            return None

    async def predict(self, session_id: UUID, model_id: UUID, input_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs the full inference pipeline: Feature generation -> Scaling -> Prediction
        """
        # 1. Generate features
        enriched_df = await self.run_feature_generation(session_id, input_data)
        
        # 2. Load model
        model = await self.load_model(model_id)
        if not model:
            raise ValueError(f"Could not load model {model_id}")
            
        # 3. Prepare input for model
        # Most models expect a specific window/shape
        # We fetch the configuration from TrainedModel
        from app.database.models import TrainedModel
        stmt = select(TrainedModel).where(TrainedModel.id == model_id)
        result = await self.repo.db.execute(stmt)
        model_rec = result.scalar_one_or_none()
        
        sequence_length = model_rec.parameters.get('sequence_length', 60) if model_rec.parameters else 60
        
        scaler = await self.load_scaler(session_id)
        if scaler is None:
            raise InferenceContractError(
                "Selected model has no persisted fitted scaler. Live inference requires the "
                "training scaler and feature contract."
            )

        feature_contract = await self.load_feature_contract(session_id, scaler)
        sequence_feature_names = feature_contract["sequence_features"]
        scaler_feature_names = feature_contract["scaler_features"]

        missing_scaler_features = [name for name in scaler_feature_names if name not in enriched_df.columns]
        if missing_scaler_features:
            raise InferenceContractError(
                "Live feature enrichment is incomplete for the selected model. "
                "Refusing prediction instead of zero-filling missing trained features.",
                missing_features=missing_scaler_features,
            )

        non_numeric_features = [
            name for name in scaler_feature_names
            if not pd.api.types.is_numeric_dtype(enriched_df[name])
        ]
        if non_numeric_features:
            raise InferenceContractError(
                "Live feature enrichment produced non-numeric values for trained model inputs.",
                missing_features=non_numeric_features,
            )

        try:
            scaled_full = scaler.transform(enriched_df[scaler_feature_names])
        except Exception as error:
            raise InferenceContractError(
                f"Failed to apply the persisted training scaler: {error}"
            ) from error

        scaled_df = pd.DataFrame(
            scaled_full,
            index=enriched_df.index,
            columns=scaler_feature_names,
        )

        missing_sequence_features = [name for name in sequence_feature_names if name not in scaled_df.columns]
        if missing_sequence_features:
            raise InferenceContractError(
                "Persisted sequence feature map is incompatible with the persisted scaler schema.",
                missing_features=missing_sequence_features,
            )

        # Ensure we have enough data for the sequence
        if len(scaled_df) < sequence_length:
            raise InferenceContractError(
                f"Insufficient fully-enriched data for inference. Need {sequence_length} rows, got {len(scaled_df)}"
            )
            
        # Extract the last sequence
        last_sequence = scaled_df[sequence_feature_names].tail(sequence_length).values
        
        # Reshape for model (batch_size, sequence_length, n_features)
        X = last_sequence.reshape(1, sequence_length, len(sequence_feature_names))

        model_input_shape = getattr(model, "input_shape", None)
        if model_input_shape and len(model_input_shape) == 3:
            expected_sequence_length = model_input_shape[1]
            expected_feature_count = model_input_shape[2]
            if (
                (expected_sequence_length is not None and expected_sequence_length != sequence_length)
                or (expected_feature_count is not None and expected_feature_count != len(sequence_feature_names))
            ):
                raise InferenceContractError(
                    "Persisted model input shape does not match the selected session's scaler contract. "
                    f"Model expects ({expected_sequence_length}, {expected_feature_count}), "
                    f"contract provides ({sequence_length}, {len(sequence_feature_names)})."
                )
        
        # 4. Predict — offload blocking model.predict() to thread pool
        # TensorFlow/Keras predict() is CPU/GPU-bound and blocks for 50-500ms.
        loop = asyncio.get_running_loop()
        prediction = await loop.run_in_executor(_model_executor, model.predict, X)
        
        # Format result
        # Assuming classification for now (buy/sell/hold)
        if hasattr(prediction, 'tolist'):
            pred_list = prediction.tolist()[0]
        else:
            pred_list = prediction[0]
            
        return {
            "prediction": pred_list,
            "timestamp": enriched_df['time'].iloc[-1].isoformat() if 'time' in enriched_df.columns else None,
            "price": enriched_df['Close'].iloc[-1],
            "model_id": str(model_id),
            "session_id": str(session_id)
        }
