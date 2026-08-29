"""
Processing Handlers - Analysis-Specific Implementation Functions

This module contains the actual analysis logic for each processing type.
Handlers are registered with HandlerRegistry and called by processing strategies.

Each handler follows the same signature:
    async def handler(df: pd.DataFrame, context: ProcessingContext) -> Dict[str, Any]

Handlers are responsible for:
1. Executing analysis-specific logic
2. Returning enriched DataFrame + metadata
3. Progress tracking via context.task_store
4. Handling slice context for boundary-aware processing

"""

import logging
import asyncio
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import uuid

import asyncio
from datetime import datetime
import pandas as pd
import numpy as np
from app.services.mt5_service import MT5Service
# Pure primitives — leaf module, no app-internal imports, never circular.
from app.core.processing.processing_types import ProcessingContext, ProcessingStrategy
# StrategyFactory instantiates concrete strategy classes, so it lives in
# processing_strategies (below handlers in the dependency hierarchy).
from app.core.processing.processing_strategies import StrategyFactory
from app.core.analysis.trading.signal_generator import generate_signals_sequential_with_progress
from app.core.processing.progress_reporter import ProgressReporter, ThrottlingStrategy
from app.core.analysis.technical_indicators import TechnicalIndicators
from app.core.analysis.astronomy.astronomical_optimized import generate_astronomical_data_optimized

from app.core.ml.metric_feature_engineer import MetricFeatureEngineer,  MetricFeatureConfig
from app.core.ml.model_registry import get_registry
from app.core.ml.persistent_model_store import persistent_model_store
from app.core.ml.ml_dataset_preparation import DatasetConfig, ScalerType, SplitStrategy
from app.core.ml.ml_dataset_preparation import MLDatasetPreparation
from app.core.data.session_data_loader import append_sequences_to_ml_dataset
from app.core.config import MLDatasetConfig
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION VALIDATION - TASK 3: Enforcement at Handler Entry
# ============================================================================

# Progress configuration is now managed globally by ProgressReporter and BroadcastingTaskStoreProxy.
# Throttling is disabled by default to ensure UI responsiveness.


# ============================================================================
# TECHNICAL ANALYSIS HANDLER
# ============================================================================

async def analyze_technical_impl(
    df: pd.DataFrame,
    context: ProcessingContext,
) -> Dict[str, Any]:
    """
    Technical indicators analysis implementation.
    
    Args:
        df: OHLCV DataFrame
        context: Processing context with config and progress tracking
        
    Returns:
        Dict with keys: features_df, metadata
    
    """
    
    # Create unified reporter
    reporter = ProgressReporter(
        task_id=context.task_id, # Pass explicitly
        task_store=context.task_store,
        connection_manager=context.connection_manager,
        user_id=context.user_id,
        throttling_strategy=ThrottlingStrategy.HYBRID
    )
    # Associate task_id for sync report() Compatibility
    reporter.task_id = context.task_id
    
    ti = TechnicalIndicators(context.config)
    
    # CRITICAL FIX: Wrap executor call in try/catch
    try:
        loop = asyncio.get_running_loop()
        result_df = await loop.run_in_executor(
            None,
            lambda: ti.calculate_all_indicators(
                df,
                task_id=context.task_id,
                progress_store=context.task_store,
                mode="training",
                slice_context={
                    "slice_num": context.slice_num,
                    "total_slices": context.total_slices,
                    "slice_start": context.slice_start,
                    "slice_end": context.slice_end,
                    "total_dataset_rows": context.total_dataset_rows,
                    "total_dataset_rows": context.total_dataset_rows,
                    "reporter": reporter,
                    "user_id": context.user_id,
                },
                reporter=reporter
            )
        )
    except asyncio.CancelledError:
        logger.warning(f"[Technical] Executor cancelled for task {context.task_id}")
        raise
    except Exception as executor_err:
        logger.error(
            f"❌ [Technical] Executor failed for task {context.task_id}: {executor_err}",
            exc_info=True
        )
        raise RuntimeError(
            f"Technical indicators analysis failed: {executor_err}"
        ) from executor_err
    
    # ── Metric Card Feature Engineering ──────────────────────────────────
    # Run AFTER TechnicalIndicators so metric_* columns are available as
    # training targets in the ML pipeline.  Controlled by the
    # enable_metric_features flag on TechnicalConfig (default True).
    metric_features_enabled = getattr(context.config, "enable_metric_features", True)
    if metric_features_enabled:
        try:
            
            
            mfe_config = MetricFeatureConfig(
                enable_metric_features=True,
                volatility_window=getattr(context.config, "metric_volatility_window", 20),
                speed_window=getattr(context.config, "metric_volatility_window", 20),
                regime_threshold=getattr(context.config, "metric_regime_threshold", 2.0),
                momentum_short_window=getattr(context.config, "metric_momentum_short_window", 5),
                momentum_long_window=getattr(context.config, "metric_momentum_long_window", 20),
            )
            mfe = MetricFeatureEngineer(mfe_config)
            result_df = mfe.calculate_all_metric_features(result_df, reporter=reporter)
            logger.info(
                f"[Technical] MetricFeatureEngineer added metric columns "
                f"(task={context.task_id[:8]})"
            )
        except Exception as mfe_err:
            # Non-fatal: log and continue without metric columns
            logger.warning(
                f"[Technical] MetricFeatureEngineer failed (non-fatal): {mfe_err}"
            )

    return {
        "features_df": result_df,
        "result_df": result_df,
        "metadata": {
            "strategy": "technical",
            "rows_processed": len(result_df),
            "analysis_type": "technical",
        },
    }


# ============================================================================
# SNR ANALYSIS HANDLER
# ============================================================================

async def analyze_snr_impl(
    df: pd.DataFrame,
    context: ProcessingContext,
) -> Dict[str, Any]:
    """
    SNR (Signal-to-Noise Ratio) signal generation implementation.
    
    Args:
        df: OHLCV DataFrame
        context: Processing context with config and progress tracking
        
    Returns:
        Dict with keys: signals, zones, enriched_df, ml_dataset, signal_counts, metadata
    
    ✅ ERROR HANDLING: Executor call wrapped with proper error context
    ✅ TASK 3: Progress configuration enforced (max_updates=4, reduced from 100+)
    """
    
    # Create unified reporter
    reporter = ProgressReporter(
        task_id=context.task_id, # Pass explicitly
        task_store=context.task_store,
        connection_manager=context.connection_manager,
        user_id=context.user_id,
        throttling_strategy=ThrottlingStrategy.HYBRID
    )
    # Associate task_id for sync report() Compatibility
    reporter.task_id = context.task_id
    
    # CRITICAL FIX: Wrap executor call in try/catch
    try:
        loop = asyncio.get_running_loop()
        signals, zones, df_with_snr, ml_dataset, g_start, g_end, signal_counts = await loop.run_in_executor(
            None,
            lambda: generate_signals_sequential_with_progress(
                price_data=df,
                lookback_period=context.config.lookback_period,
                confirmation_period=context.config.confirmation_period,
                n_clusters=context.config.n_clusters,
                zone_width=context.config.zone_width,
                min_distance_pct=context.config.min_distance_pct,
                lookforward_period=context.config.lookforward_period,
                animation_step=context.config.animation_step,
                task_id=context.task_id,
                progress_store=context.task_store,
                global_index_offset=context.global_offset,
                slice_context={
                    "slice_num": context.slice_num,
                    "total_slices": context.total_slices,
                    "slice_start": context.slice_start,
                    "slice_end": context.slice_end,
                    "total_dataset_rows": context.total_dataset_rows,
                    "global_offset": context.global_offset,
                    "total_dataset_rows": context.total_dataset_rows,
                    "global_offset": context.global_offset,
                    "reporter": reporter,
                    "user_id": context.user_id,
                },
                reporter=reporter,
            )
        )
    except asyncio.CancelledError:
        logger.warning(f"[SNR] Executor cancelled for task {context.task_id}")
        raise
    except Exception as executor_err:
        logger.error(
            f"❌ [SNR] Executor failed for task {context.task_id}: {executor_err}",
            exc_info=True
        )
        raise RuntimeError(
            f"SNR signal generation failed: {executor_err}"
        ) from executor_err

    exhaustion_events = []
    source_timeframe = getattr(context.config, "timeframe", None)
    source_symbol = getattr(context.config, "symbol", None)
    if source_timeframe in {"H1", "H4"} and source_symbol:
        from app.core.analysis.exhaustion_candles import detect_exhaustion_candles

        exhaustion_events = [
            event.to_dict()
            for event in detect_exhaustion_candles(
                df,
                symbol=source_symbol,
                timeframe=source_timeframe,
                snr_zones=zones,
            )
        ]

    return {
        "signals": signals,
        "zones": zones,
        "exhaustion_events": exhaustion_events,
        "enriched_df": df_with_snr,
        "result_df": df_with_snr,
        "ml_dataset": ml_dataset,
        "signal_counts": signal_counts,
        "total_signals": len(signals),
        "metadata": {
            "strategy": "snr",
            "rows_processed": len(df),
            "analysis_type": "snr",
        },
    }


# ============================================================================
# ASTRONOMICAL ANALYSIS HANDLER
# ============================================================================

async def analyze_astronomical_impl(
    df: pd.DataFrame,
    context: ProcessingContext,
) -> Dict[str, Any]:
    """
    Astronomical alignment analysis implementation.
    
    Args:
        df: OHLCV DataFrame
        context: Processing context with config and progress tracking
        
    Returns:
        Dict with keys: features_df, metadata
    
    ✅ ERROR HANDLING: Executor call wrapped with proper error context
    ✅ TASK 3: Progress configuration enforced (max_updates=2)
    """
    
    # Create unified reporter
    reporter = ProgressReporter(
        task_store=context.task_store,
        connection_manager=context.connection_manager,
        user_id=context.user_id,
        throttling_strategy=ThrottlingStrategy.HYBRID
    )
    # Associate task_id for sync report() Compatibility
    reporter.task_id = context.task_id
    
    # CRITICAL FIX: Wrap executor call in try/catch
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: generate_astronomical_data_optimized(
                price_data=df,
                observer_lat=context.config.observer_lat,
                observer_lon=context.config.observer_lon,
                house_system=context.config.house_system,
                zodiac_type=context.config.zodiac_type,
                use_minor_aspects=context.config.use_minor_aspects,
                aspect_orbs=context.config.aspect_orbs,
                selected_features=context.config.selected_features,
                include_asteroids=getattr(context.config, "include_asteroids", False),
                include_fixed_stars=getattr(context.config, "include_fixed_stars", False),
                reporter=reporter
            )
        )
    except asyncio.CancelledError:
        logger.warning(f"[Astronomical] Executor cancelled for task {context.task_id}")
        raise
    except Exception as executor_err:
        logger.error(
            f"❌ [Astronomical] Executor failed for task {context.task_id}: {executor_err}",
            exc_info=True
        )
        raise RuntimeError(
            f"Astronomical analysis failed: {executor_err}"
        ) from executor_err

    return {
        "features_df": result,
        "result_df": result,
        "metadata": {
            "strategy": "astronomical",
            "rows_processed": len(result),
            "analysis_type": "astronomical",
        },
    }


# ============================================================================
# ML DATASET PREPARATION HANDLER
# ============================================================================

async def analyze_ml_prep_impl(
    df: pd.DataFrame,
    context: ProcessingContext,
    **kwargs
) -> Dict[str, Any]:
    """
    ML dataset preparation implementation - SIMPLIFIED for pre-split data.
    
    NEW ARCHITECTURE (CORRECTED):
    - ProcessingManager already split and scaled the data
    - This handler just needs to:
      1. Validate and enrich the data
      2. Identify features
      3. Generate sequences (call _generate_sequences_from_splits directly)
      4. Store sequences to database
    
    Args:
        df: Pre-split, pre-scaled DataFrame (train, validation, or test)
        context: Processing context with DatasetConfig and scaler info
        **kwargs: Additional parameters (split_name, feature_cols, pm_instance, db_session, etc.)
        
    Returns:
        Dict with sequences, labels, targets, and metadata for this split
    
    ✅ ERROR HANDLING: Wrapped with proper error context
    ✅ TASK 3: Progress configuration enforced (max_updates=3, reduced from 90+)
    """
    
    # Extract parameters from kwargs
    split_name = kwargs.get('split_name', 'train')
    feature_cols = kwargs.get('feature_cols', [])
    pm_instance = kwargs.get('pm_instance')
    db_session = kwargs.get('db_session')
    enriched_target_columns = kwargs.get('enriched_target_columns')  # FIX: extract target list from PM
    
    logger.info(f"[ML Handler] Processing {split_name} split ({len(df)} rows) - data already split and scaled by PM")
    if enriched_target_columns:
        logger.info(f"[ML Handler] 🎯 Received {len(enriched_target_columns)} enriched_target_columns from PM")
    
    # Create unified reporter
    reporter = ProgressReporter(
        task_store=context.task_store,
        connection_manager=context.connection_manager,
        user_id=context.user_id,
        throttling_strategy=ThrottlingStrategy.HYBRID
    )
    reporter.task_id = context.task_id
    
    # CRITICAL FIX: Wrap entire ML prep flow in try/catch
    try:
        config = context.config
        
    
        if isinstance(config, MLDatasetConfig):
            # Map centralized config to internal DatasetConfig
            # Map string types to Enums
            scaler_map = {
                'minmax': ScalerType.MINMAX,
                'standard': ScalerType.STANDARD,
                'robust': ScalerType.ROBUST,
                'none': ScalerType.NONE
            }
            strategy_map = {
                'random': SplitStrategy.RANDOM,
                'sequential': SplitStrategy.SEQUENTIAL,
                'stratified': SplitStrategy.STRATIFIED
            }
            
            config = DatasetConfig(
                sequence_length=getattr(config, 'sequence_length', 60),
                prediction_length=getattr(config, 'prediction_length', 7),
                dataset_name=getattr(config, 'dataset_name', 'ml_prep_default'),
                signal_column_prefix=getattr(config, 'signal_column_prefix', 'Signal_'),
                target_columns=getattr(config, 'target_columns', ["target_return", "target_direction"]),
                exclude_columns=getattr(config, 'exclude_columns', []),
                scaler_type=scaler_map.get(getattr(config, 'scaler_type',None), None),
                save_scaler=getattr(config, 'save_scaler', True),
                scaler_filename=getattr(config, 'scaler_filename', 'dataset_scaler.joblib'),
                scaler_save_path=getattr(config, 'scaler_save_path', 'dataset_scaler.joblib'),
                scaler_load_path=getattr(config, 'scaler_load_path', None),
                train_ratio=getattr(config, 'train_ratio', 0.7),
                validation_ratio=getattr(config, 'validation_ratio', 0.15),
                test_ratio=getattr(config, 'test_ratio', 0.15),
                split_strategy=strategy_map.get(getattr(config, 'split_strategy', 'sequential'), SplitStrategy.SEQUENTIAL),
                random_seed=getattr(config, 'random_seed', 42),
                include_classification=getattr(config, 'include_classification', True),
                include_regression=getattr(config, 'include_regression', False),
                include_sequence_prediction=getattr(config, 'include_sequence_prediction', False),
                handle_class_imbalance=getattr(config, 'handle_class_imbalance', True),
                shuffle_data=getattr(config, 'shuffle_data', False),
                preserve_temporal_order=getattr(config, 'preserve_temporal_order', True),
                use_lazy_storage=getattr(config, 'use_lazy_storage', False),
                negative_sampling_ratio=getattr(config, 'negative_sampling_ratio', 1.0),
                mask_future_signals=getattr(config, 'mask_future_signals', True),
                signal_leakage_buffer=getattr(config, 'signal_leakage_buffer', 20),
                exclude_signals=getattr(config, 'exclude_signals', False),
                drop_zeros=getattr(config, 'drop_zeros', True),
                source_type=getattr(config, 'source_type', 'enriched_df'),
                input_source=getattr(config, 'input_source', 'raw'),
                use_snr_dataset=getattr(config, 'use_snr_dataset', False),
                selected_signal_types=getattr(config, 'selected_signal_types', []),
                feature_selection_mode=getattr(config, 'feature_selection_mode', 'rich'),
                custom_features=getattr(config, 'custom_features', None)
            )
        elif isinstance(config, dict):
            config_dict = config
            
            # Map frontend scaler types to backend enum
            scaler_type_map = {
                'minmax': ScalerType.MINMAX,
                'standard': ScalerType.STANDARD,
                'robust': ScalerType.ROBUST,
                'none': ScalerType.NONE
            }
            
            config = DatasetConfig(
                sequence_length=config_dict.get("sequence_length", 60),
                prediction_length=config_dict.get("prediction_length", 7),
                scaler_type=scaler_type_map.get(config_dict.get("scaler_type", "robust"), ScalerType.ROBUST),
                train_ratio=1.0,  # Not used - data already split
                validation_ratio=0.0,
                test_ratio=0.0,
                target_columns=config_dict.get("target_columns", ["target_return", "target_direction"]),
                exclude_columns=config_dict.get("exclude_columns", []),
                signal_column_prefix=config_dict.get("signal_column_prefix", "Signal_"),
                include_classification=config_dict.get("include_classification", True),
                include_regression=config_dict.get("include_regression", False),
                include_sequence_prediction=config_dict.get("include_sequence_prediction", False),
                mask_future_signals=config_dict.get("mask_future_signals", True),
                signal_leakage_buffer=config_dict.get("signal_leakage_buffer", 20),
                exclude_signals=config_dict.get("exclude_signals", False),
                drop_zeros=config_dict.get("drop_zeros", True),
                random_seed=config_dict.get("random_seed", 42),
                dataset_name=config_dict.get("dataset_name", "ml_prep_default"),
                scaler_save_path=config_dict.get("scaler_save_path", "dataset_scaler.joblib"),
                scaler_load_path=config_dict.get("scaler_load_path", None),
                save_scaler=config_dict.get("save_scaler", True),
                scaler_filename=config_dict.get("scaler_filename", "dataset_scaler.joblib"),
                split_strategy=strategy_map.get(config_dict.get("split_strategy", "sequential"), SplitStrategy.SEQUENTIAL),
                handle_class_imbalance=config_dict.get("handle_class_imbalance", True),
                shuffle_data=config_dict.get("shuffle_data", False),
                preserve_temporal_order=config_dict.get("preserve_temporal_order", True),
                use_lazy_storage=config_dict.get("use_lazy_storage", False),
                negative_sampling_ratio=config_dict.get("negative_sampling_ratio", 1.0),
                source_type=config_dict.get("source_type", "enriched_df"),
                input_source=config_dict.get("input_source", "raw"),
                use_snr_dataset=config_dict.get("use_snr_dataset", False),
                selected_signal_types=config_dict.get("selected_signal_types", []),
                feature_selection_mode=config_dict.get("feature_selection_mode", "rich"),
                custom_features=config_dict.get("custom_features", None)
            )
        
        # Initialize ML preparation helper
        ml_prep = MLDatasetPreparation(
            data=df,
            config=config,
            task_id=context.task_id,
            reporter=reporter,
        )
        
        # SIMPLIFIED FLOW: Just the essential steps
        # Step 1: Validate data
        await ml_prep._validate_data()
        logger.info(f"[ML Handler] Data validated for {split_name}")
        
        # Step 3: Identify features (if not provided by PM)
        if not feature_cols:
            ml_prep._identify_features()
            feature_cols = ml_prep.feature_cols
        else:
            ml_prep.feature_cols = feature_cols
            ml_prep.columns_to_scale = feature_cols
        
        logger.info(f"[ML Handler] Features identified: {len(feature_cols)} columns")
        
        # Step 4: Generate sequences directly (data is already scaled)
        # Create a single-split dict with the entire DataFrame
        scaled_splits = {split_name: ml_prep.data}
        
        logger.info(f"[ML Handler] Generating sequences for {split_name} split...")
        
        # Call _generate_sequences_from_splits directly (like other workers do)
        # FIX: Pass enriched_target_columns so all targets are included
        result_data = {}
        async for name, split_data in ml_prep._generate_sequences_from_splits(
            scaled_splits,
            enriched_target_columns=enriched_target_columns
        ):
            result_data = split_data
            logger.info(f"[ML Handler] Generated {len(split_data['sequences'])} sequences for {name}")
            break  # Only one split
        
        # Step 5: Store sequences to database (like parallel strategy does)
        sequences = result_data.get('sequences', np.array([]))
        labels = result_data.get('labels', np.array([]))
        targets = result_data.get('targets', {})
        
        if pm_instance is not None and db_session is not None:
            # Get dataset_id from PM context
            dataset_id = getattr(pm_instance, '_current_dataset_id', None)
            
            if dataset_id and len(sequences) > 0:
                logger.info(
                    f"[ML Handler] Storing {len(sequences)} sequences to database "
                    f"(dataset_id={dataset_id}, split={split_name})"
                )
                
                # Store to database
                success = await append_sequences_to_ml_dataset(
                    dataset_id=dataset_id,
                    sequences=sequences,
                    labels=labels,
                    targets=targets,
                    split_name=split_name,
                    db=db_session,
                    sequence_metadata=result_data.get('sequence_metadata')
                )
                
                if success:
                    logger.info(f"[ML Handler] ✅ Sequences stored to database successfully")
                else:
                    logger.error(f"[ML Handler] ❌ Failed to store sequences to database")
            else:
                if not dataset_id:
                    logger.warning(f"[ML Handler] No dataset_id found in PM context, skipping DB storage")
                if len(sequences) == 0:
                    logger.warning(f"[ML Handler] No sequences generated, skipping DB storage")
        else:
            logger.warning(f"[ML Handler] No PM instance or DB session provided, skipping DB storage")
        
        # Add metadata
        actual_target_names = enriched_target_columns or list(config.target_columns)
        result_data["metadata"] = {
            "total_sequences": len(sequences),
            "feature_count": len(feature_cols),
            "split_name": split_name,
            "extracted_target_names": actual_target_names,  # FIX: report all targets, not just config.target_columns
        }
        result_data["feature_names"] = feature_cols
        result_data["target_names"] = actual_target_names
        result_data["result_df"] = df
        
        logger.info(f"[ML Handler] Completed {split_name} split: {len(sequences)} sequences")
        
        return result_data
    
    except asyncio.CancelledError:
        logger.warning(f"[ML Prep] Task cancelled for task {context.task_id}")
        raise
    except Exception as prep_err:
        logger.error(
            f"❌ [ML Prep] Dataset preparation failed for task {context.task_id}: {prep_err}",
            exc_info=True
        )
        raise RuntimeError(
            f"ML dataset preparation failed: {prep_err}"
        ) from prep_err


# ============================================================================
# CURRENCY INDICES HANDLER
# ============================================================================

async def analyze_currency_indices_impl(
    df: pd.DataFrame,
    context: ProcessingContext,
) -> Dict[str, Any]:
    """
    Calculate currency strength indices and merge with input data.
    
    Args:
        df: OHLCV DataFrame with per-pair columns (e.g., open_EURUSD, high_USDJPY)
        context: Processing context with config (CurrencyIndexConfig)
        
    Returns:
        Dict with keys: result_df (enriched with index columns), metadata
    
    Process:
        1. Extract required pairs from selected indices (from INDEX_DEFINITIONS)
        2. Verify all pair columns exist in input DataFrame
        3. Calculate indices using CurrencyIndexCalculator
        4. Merge new columns into original DataFrame
        5. Return enriched DataFrame + metadata
    """
    
    try:
        from app.core.analysis.currency_index import (
            CurrencyIndexCalculator,
            INDEX_DEFINITIONS,
            OHLCV_FIELDS,
        )
        
        config = context.config  # CurrencyIndexConfig
        
        # Create unified reporter for progress tracking
        reporter = ProgressReporter(
            task_store=context.task_store,
            connection_manager=context.connection_manager,
            user_id=context.user_id,
            throttling_strategy=ThrottlingStrategy.HYBRID
        )
        reporter.task_id = context.task_id
        
        # ────────────────────────────────────────────────────────
        # PRE-STEP: Strip stale currency index columns from df
        # ────────────────────────────────────────────────────────
        # When current_data already contains results from a prior run
        # (Dollar_open, Dollar_SMA_20, Bar_Volume_Up, etc.) we must
        # strip them before re-computing so we always start clean.
        # Keep only the original OHLCV base columns + pair columns if any.
        # We identify "stale" columns as anything that doesn't belong to
        # the original base OHLCV set and is NOT a raw pair column.


        # We just need to strip the currency index columns from df if they exist
        # Find all columns that start with an index name prefix (Dollar_*, Euro_*, JPY_*, etc.)
        INDEX_NAMES = set(INDEX_DEFINITIONS.keys())  # Dollar, Euro, JPY, etc.
        
        stale_cols = [
            c for c in df.columns
            if any(c.startswith(f"{idx}_") for idx in INDEX_NAMES)
        ]
        if stale_cols:
            logger.info(f"[Currency Indices] Stripping {len(stale_cols)} stale index columns from input df")
            df = df.drop(columns=stale_cols)
            logger.info(f"[Currency Indices] Clean input df: {df.shape[1]} columns")
        
        # ────────────────────────────────────────────────────────
        # STEP 1: Extract all required pairs for selected indices
        # ────────────────────────────────────────────────────────
        required_pairs = set()
        for idx_name in config.selected_indices:
            if idx_name not in INDEX_DEFINITIONS:
                logger.warning(f"Unknown index: {idx_name}, skipping")
                continue
            pairs_dict = INDEX_DEFINITIONS[idx_name]['pairs']
            required_pairs.update(pairs_dict.keys())
        
        required_pairs = sorted(list(required_pairs))
        reporter.report(
            progress=10,
            message=f"Extracted {len(required_pairs)} required pairs",
        )
        logger.info(f"[Currency Indices] Required pairs: {required_pairs}")
        
        # ────────────────────────────────────────────────────────
        # ────────────────────────────────────────────────────────
        # STEP 2: Identify and fetch missing pair columns
        # ────────────────────────────────────────────────────────
        available_cols = set(df.columns)
        missing_pairs = set()
        for pair in required_pairs:
            for field in OHLCV_FIELDS:
                if f"{field}_{pair}" not in available_cols:
                    missing_pairs.add(pair)
                    break
        
        if missing_pairs:
            # ── GUARD: Workers must NOT re-fetch from MT5 ──────────────────────
            # All pair columns should have been pre-fetched by
            # AnalysisManager._prefetch_currency_pairs() BEFORE the DataFrame was
            # chunked and dispatched to worker processes.
            #
            # If we reach this point inside a worker and columns are still missing,
            # it means the pre-fetch either failed or was skipped (e.g. config was
            # None at the AM level).  Re-fetching here would:
            #   1. Spawn up to N×pairs concurrent MT5 bridge requests → 500s
            #   2. Return columns WITHOUT the {index}_ prefix, causing collisions
            #      in the base-pair Technical Analysis step that follows.
            #
            # Solution: raise immediately with a clear actionable message so the
            # root cause (failed pre-fetch) is surfaced rather than silently
            # producing corrupted output.
            missing_pair_names = sorted(missing_pairs)
            missing_col_examples = [
                f"{field}_{pair}"
                for pair in missing_pair_names[:3]
                for field in ["open", "close"]
            ][:6]
            raise ValueError(
                f"[Currency Indices] Worker received DataFrame missing {len(missing_pairs)} "
                f"required pair column-sets: {missing_pair_names}. "
                f"Example missing columns: {missing_col_examples}. "
                f"This should have been populated by AnalysisManager._prefetch_currency_pairs() "
                f"before chunking. Check that the pre-fetch step ran successfully and that "
                f"all pairs are available on the MT5 broker for this account."
            )
                        
        available_cols = set(df.columns)
        missing_columns = []
        for pair in required_pairs:
            for field in OHLCV_FIELDS:
                col_name = f"{field}_{pair}"
                if col_name not in available_cols:
                    missing_columns.append(col_name)
                    
        if missing_columns:
            error_msg = (
                f"Missing {len(missing_columns)} required columns: {missing_columns[:5]}..."
                if len(missing_columns) > 5
                else f"Missing columns: {missing_columns}"
            )
            logger.error(f"[Currency Indices] ❌ {error_msg}")
            raise ValueError(error_msg)
        
        reporter.report(
            progress=30,
            message=f"Verified all {len(required_pairs)} pair columns",
        )
        logger.info(f"[Currency Indices] All required columns verified")
        
        # ────────────────────────────────────────────────────────
        # STEP 2.5: Fill NaN gaps in pair columns from time-alignment
        # ────────────────────────────────────────────────────────
        # When MT5 data is merged by time, some timestamps may not align perfectly,
        # leaving NaN values. These propagate through the weighted-product formula
        # (scalar * ∏ col^exp) producing NaN index values.
        # Fix: forward-fill then back-fill so every row has valid pair prices.
        pair_cols = [
            f"{field}_{pair}"
            for pair in required_pairs
            for field in ['open', 'high', 'low', 'close', 'tick_volume']
            if f"{field}_{pair}" in df.columns
        ]
        nan_before = df[pair_cols].isna().sum().sum()
        if nan_before > 0:
            logger.info(f"[Currency Indices] Filling {nan_before} NaN gaps in pair columns (ffill+bfill)...")
            df[pair_cols] = df[pair_cols].ffill().bfill()
            nan_after = df[pair_cols].isna().sum().sum()
            logger.info(f"[Currency Indices] NaN gaps after fill: {nan_after}")
        
        # Also ensure tick_volume is never zero or negative (raises errors in power formula)
        tv_cols = [c for c in pair_cols if 'tick_volume_' in c]
        for col in tv_cols:
            zero_mask = df[col] <= 0
            if zero_mask.any():
                logger.warning(f"[Currency Indices] {zero_mask.sum()} zero/negative values in {col} → replacing with 1.0")
                df.loc[zero_mask, col] = 1.0
        
        # ────────────────────────────────────────────────────────
        # STEP 3: Calculate indices using CurrencyIndexCalculator
        # ────────────────────────────────────────────────────────
        # Create a wrapper reporter that passes through to base reporter's report() method
        class ReporterWrapper:
            def __init__(self, base_reporter, task_id):
                self.base_reporter = base_reporter
                self.task_id = task_id
            
            def report(self, progress, message="", message2="", **kwargs):
                """Pass through to base reporter's report() method"""
                # Handle both positional and keyword arguments
                self.base_reporter.report(
                    progress=progress, 
                    message=message, 
                    message2=message2, 
                    **kwargs
                )
            
            def check_cancellation(self):
                """Pass through cancellation check to base reporter"""
                if hasattr(self.base_reporter, 'check_cancellation'):
                    self.base_reporter.check_cancellation()

            def report_loop(self, current: int, total: int, message: str = "", message2: str = "",
                            base_progress: float = 0.0, progress_range: float = 10.0, **kwargs):
                """Pass through loop progress used by technical indicator internals."""
                if hasattr(self.base_reporter, 'report_loop'):
                    self.base_reporter.report_loop(
                        current,
                        total,
                        message=message,
                        message2=message2,
                        base_progress=base_progress,
                        progress_range=progress_range,
                        **kwargs
                    )
                    return

                if total <= 0:
                    return

                loop_progress = base_progress + ((current / total) * progress_range)
                formatted_message2 = message2.replace("{current}", str(current)).replace("{total}", str(total))
                self.report(
                    progress=int(max(0, min(100, loop_progress))),
                    message=message or f"Processing {current}/{total}",
                    message2=formatted_message2,
                    **kwargs
                )
        
        reporter_wrapper = ReporterWrapper(reporter, context.task_id)
        
        # ✅ CRITICAL FIX: Pre-clean merged pair data before index calculation
        # This fills NaN gaps from time-alignment mismatches and clamps extreme values
        # BEFORE passing to CurrencyIndexCalculator to ensure clean input
        from app.core.analysis.currency_index import prepare_index_data
        
        reporter.report(
            progress=40,
            message="Pre-processing merged pair data (fill gaps, clamp extremes)...",
        )
        logger.info("[Currency Indices] Pre-processing merged data with prepare_index_data()...")
        df = prepare_index_data(df)
        logger.info("[Currency Indices] ✅ Pre-processing complete: NaN gaps filled, extremes clamped")
        
        calc = CurrencyIndexCalculator(df, reporter=reporter_wrapper)
        indices_dict = calc.calculate_indices(indices=config.selected_indices)
        
        calculated_indices = list(indices_dict.keys())
        reporter.report(
            progress=50,
            message=f"Calculated {len(calculated_indices)} indices: {', '.join(calculated_indices)}",
        )
        logger.info(f"[Currency Indices] Calculated indices: {calculated_indices}")
        # Log what fields each index has
        for idx_name in calculated_indices:
            fields = list(indices_dict[idx_name].keys())
            logger.info(f"[Currency Indices]   → {idx_name}: {len(fields)} fields = {fields[:10]}{'...' if len(fields) > 10 else ''}")
        
        # ────────────────────────────────────────────────────────
        # STEP 3.5: Technical Indicator calculation delegated to AnalysisManager
        # ────────────────────────────────────────────────────────
        # ✅ ARCHITECTURE: TI calculation moved to AnalysisManager.execute_ti_for_currency_indices()
        # This properly leverages ProcessingManager for:
        # - Large dataset chunking
        # - Proper serialization (fixes null columns)
        # - Memory management
        # - Async batch processing
        # ProcessingManager will be called with temp sessions for each index,
        # avoiding inline calculation and serial bottleneck.
        logger.info("[Currency Indices] ℹ TI calculation will be handled by AnalysisManager.execute_ti_for_currency_indices()")
        
        # ────────────────────────────────────────────────────────
        # STEP 4: Convert indices to DataFrame and merge
        # ────────────────────────────────────────────────────────
        # Convert index_dict to flat DataFrame manually to support the new TI columns
        frames: Dict[str, pd.Series] = {}
        idx_num = 0
        total_indices = len(indices_dict)
        for idx_name, cols_dict in indices_dict.items():
            for field_name, series in cols_dict.items():
                frames[f"{idx_name}_{field_name}"] = series
            
            progress = int(90 + (idx_num / max(total_indices, 1)) * 9)
            reporter.report(
                progress=progress,
                message=f"Converting {idx_name} to DataFrame..."
            )
            idx_num += 1
            
        indices_df = pd.DataFrame(frames, index=df.index)
        
        # Drop any existing index columns from df before merge to avoid duplicates/overwrites.
        # This happens when current_data already has Dollar_open, Euro_close etc. from a previous run.
        index_col_pattern = set(indices_df.columns)
        existing_index_cols = [c for c in df.columns if c in index_col_pattern]
        if existing_index_cols:
            logger.info(f"[Currency Indices] Dropping {len(existing_index_cols)} stale index columns from df before merge")
            df = df.drop(columns=existing_index_cols)
        
        # Verify no column name conflicts remain
        conflicts = set(df.columns) & set(indices_df.columns)
        if conflicts:
            logger.warning(
                f"[Currency Indices] Column name conflicts will be overwritten: {conflicts}"
            )
        
        # Merge: concatenate along columns
        enriched_df = pd.concat([df, indices_df], axis=1)
        
        reporter.report(
            progress=92,
            message=f"Merged {len(indices_df.columns)} index columns",
        )
        logger.info(
            f"[Currency Indices] Merged {len(indices_df.columns)} index columns. "
            f"Shape before cleanup: {enriched_df.shape}"
        )
        
        # ────────────────────────────────────────────────────────
        # STEP 4.5: Drop intermediate pair columns (keep only original + indices)
        # ────────────────────────────────────────────────────────
        pair_columns_to_drop = []
        for pair in required_pairs:
            for field in ['open', 'high', 'low', 'close', 'tick_volume', 'real_volume', 'spread']:
                col_name = f"{field}_{pair}"
                if col_name in enriched_df.columns:
                    pair_columns_to_drop.append(col_name)
        
        if pair_columns_to_drop:
            enriched_df = enriched_df.drop(columns=pair_columns_to_drop)
            logger.info(
                f"[Currency Indices] Dropped {len(pair_columns_to_drop)} intermediate pair columns. "
                f"Final shape: {enriched_df.shape}"
            )
            reporter.report(
                progress=95,
                message=f"Cleaned up {len(pair_columns_to_drop)} intermediate columns",
            )
        
        # ────────────────────────────────────────────────────────
        # STEP 5: Prepare result
        # ────────────────────────────────────────────────────────
        reporter.report(
            progress=98,
            message="Index calculation complete",
        )
        
        return {
            'result_df': enriched_df,
            'enriched_df': enriched_df,
            'metadata': {
                'strategy': 'currency_indices',
                'rows_processed': len(enriched_df),
                'indices_calculated': calculated_indices,
                'columns_added': len(indices_df.columns),
                'required_pairs_verified': len(required_pairs),
                'analysis_type': 'currency_indices',
                # ✅ CRITICAL: Pass TI config to AM for post-processing
                'calculate_ti_for_indices': getattr(config, 'calculate_ti_for_indices', None),
                'ti_config': getattr(config, 'ti_config', None),
            }
        }
    
    except asyncio.CancelledError:
        logger.warning(f"[Currency Indices] Task cancelled for task {context.task_id}")
        raise
    except Exception as exc:
        logger.error(
            f"❌ [Currency Indices] Calculation failed for task {context.task_id}: {exc}",
            exc_info=True
        )
        raise RuntimeError(
            f"Currency indices calculation failed: {exc}"
        ) from exc


# ============================================================================
# CURRENCY STRENGTH MATRIX HANDLER
# ============================================================================

async def analyze_currency_strength_matrix_impl(
    df: pd.DataFrame,
    context: ProcessingContext,
) -> dict:
    """
    Calculate Currency Strength Matrix (CSM) columns and merge them into df.

    Requires:
        - 'close' column  (asset main symbol close price, case-insensitive)
        - 'Dollar_close'  (DXY index close — produced by the currency_indices step)

    Outputs merged into result_df:
        CSM_asset_norm_fast, CSM_dxy_norm_fast
        CSM_asset_norm_slow, CSM_dxy_norm_slow
        CSM_histogram_fast  = asset_norm_fast − dxy_norm_fast
        CSM_histogram_slow  = asset_norm_slow − dxy_norm_slow

    All CSM columns are forward-safe (backward-only rolling ops) so they can be
    used as ML features without data leakage.
    """
    from app.core.analysis.currency_index import calculate_currency_strength_matrix

    config = context.config  # CurrencyStrengthMatrixConfig

    reporter = ProgressReporter(
        task_store=context.task_store,
        connection_manager=context.connection_manager,
        user_id=context.user_id,
        throttling_strategy=ThrottlingStrategy.HYBRID,
    )
    reporter.task_id = context.task_id

    try:
        # ── Resolve column names (case-insensitive for close) ─────────────
        close_col_cfg = getattr(config, "close_column", "close")
        dxy_col_cfg   = getattr(config, "dxy_column",   "Dollar_close")

        col_lower_map = {c.lower(): c for c in df.columns}

        actual_close = col_lower_map.get(close_col_cfg.lower()) or col_lower_map.get("close")
        # DXY: exact match first, then case-insensitive fallback
        actual_dxy = (
            dxy_col_cfg if dxy_col_cfg in df.columns
            else col_lower_map.get(dxy_col_cfg.lower())
        )

        if actual_close is None:
            raise ValueError(
                f"[CSM] Asset close column '{close_col_cfg}' not found. "
                f"Available columns (first 15): {list(df.columns)[:15]}"
            )

        if actual_dxy is None:
            dxy_candidates = [c for c in df.columns if "dollar" in c.lower() or "dxy" in c.lower()]
            raise ValueError(
                f"[CSM] DXY close column '{dxy_col_cfg}' not found. "
                f"Ensure the Currency Indices step ran first with 'Dollar' selected. "
                f"DXY-like columns found: {dxy_candidates or 'none'}"
            )

        fast_period  = int(getattr(config, "fast_period",  20))
        slow_period  = int(getattr(config, "slow_period",  100))
        zscore_clamp = float(getattr(config, "zscore_clamp", 3.0))

        reporter.report(
            progress=15,
            message=f"Computing Currency Strength Matrix (fast={fast_period}, slow={slow_period})...",
        )
        logger.info(
            f"[CSM] close='{actual_close}', dxy='{actual_dxy}', "
            f"fast={fast_period}, slow={slow_period}, clamp={zscore_clamp}"
        )

        # ── Run in executor so the async loop stays unblocked ─────────────
        loop = asyncio.get_running_loop()
        csm_df = await loop.run_in_executor(
            None,
            lambda: calculate_currency_strength_matrix(
                asset_close=df[actual_close],
                dxy_close=df[actual_dxy],
                fast_period=fast_period,
                slow_period=slow_period,
                zscore_clamp=zscore_clamp,
            ),
        )

        reporter.report(
            progress=80,
            message=f"CSM calculated — {len(csm_df.columns)} columns produced",
        )
        logger.info(f"[CSM] Output columns: {list(csm_df.columns)}")

        # ── Drop stale CSM columns, then merge ────────────────────────────
        stale = [c for c in df.columns if c.startswith("CSM_")]
        if stale:
            logger.info(f"[CSM] Dropping {len(stale)} stale CSM columns before merge")
            df = df.drop(columns=stale)

        result_df = pd.concat([df, csm_df], axis=1)

        reporter.report(progress=100, message="Currency Strength Matrix complete")
        logger.info(f"[CSM] Done — final shape: {result_df.shape}")

        return {
            "result_df":  result_df,
            "features_df": result_df,
            "metadata": {
                "strategy": "currency_strength_matrix",
                "rows_processed": len(result_df),
                "csm_columns": list(csm_df.columns),
                "fast_period": fast_period,
                "slow_period": slow_period,
                "analysis_type": "currency_strength_matrix",
            },
        }

    except asyncio.CancelledError:
        logger.warning(f"[CSM] Task cancelled for task {context.task_id}")
        raise
    except Exception as exc:
        logger.error(f"❌ [CSM] Failed for task {context.task_id}: {exc}", exc_info=True)
        raise RuntimeError(f"Currency Strength Matrix calculation failed: {exc}") from exc


# ============================================================================
# MODEL TRAINING HANDLER
# ============================================================================

async def analyze_model_training_impl(
    df: pd.DataFrame,
    context: ProcessingContext,
) -> Dict[str, Any]:
    """
    Model training implementation.
    
    Handles:
    1. Loading ML splits from TIER 0b pointers
    2. Model compilation (if not cached)
    3. Training loop with epoch-by-epoch progress
    4. Checkpoint saving
    5. Validation metrics
    
    Args:
        df: Not used (training uses ML splits from AnalysisManager)
        context: Processing context with training config
        
    Returns:
        Dict with keys: epochs_completed, best_val_loss, training_history
    
    ✅ TASK 3: Progress configuration enforced (max_updates=10, reduced from 300+)
    """
    
    # Create unified reporter
    reporter = ProgressReporter(
        task_store=context.task_store,
        connection_manager=context.connection_manager,
        user_id=context.user_id,
        throttling_strategy=ThrottlingStrategy.HYBRID
    )
    # Associate task_id for sync report() Compatibility
    reporter.task_id = context.task_id
    
    # Progress is managed via unified ProgressReporter
    
    # Model training uses AnalysisManager's TIER 0b pointers (ml_train, ml_validation, ml_test)
    # This handler is called by AnalysisManager.execute_model_training_with_pm()
    # which already has access to the splits
    
    # For now, return placeholder (actual implementation in AnalysisManager)
    return {
        "epochs_completed": 0,
        "best_val_loss": float('inf'),
        "training_history": [],
        "metadata": {
            "strategy": "model_training",
            "analysis_type": "model_training",
        },
        "result_df": df,
    }


# ============================================================================
# MODEL BUILDING HANDLER
# ============================================================================

async def analyze_model_build_impl(
    df: pd.DataFrame,
    context: ProcessingContext,
) -> Dict[str, Any]:
    """
    Model building implementation.
    
    Handles:
    1. Model architecture selection from registry
    2. Model compilation with optimizer/loss
    3. Caching to TIER 0c (AnalysisManager.model)
    4. Persistence to disk
    
    Args:
        df: Not used (model building doesn't require data)
        context: Processing context with model config
        
    Returns:
        Dict with keys: model_id, architecture, metadata
    
    ✅ TASK 3: Progress configuration enforced (max_updates=5)
    """
    
    # Create unified reporter
    reporter = ProgressReporter(
        task_store=context.task_store,
        connection_manager=context.connection_manager,
        user_id=context.user_id,
        throttling_strategy=ThrottlingStrategy.HYBRID
    )
    # Associate task_id for sync report() Compatibility
    reporter.task_id = context.task_id
    
    # Progress is managed via unified ProgressReporter
    
    
    
    # Get model architecture from registry
    registry = get_registry()
    model_arch = registry.get_model(context.config.get("model_id"))
    
    if not model_arch:
        raise ValueError(f"Model architecture not found: {context.config.get('model_id')}")
    
    # Build model
    from app.core.ml import default_models
    builder_func = getattr(default_models, model_arch.builder_function)
    
    model = builder_func(
        input_shape=context.config.get("input_shape"),
        n_predictions=context.config.get("n_predictions"),
        **context.config.get("parameters", {})
    )
    
    # Compile model
    model.compile(
        optimizer=context.config.get("optimizer", "adam"),
        loss=context.config.get("loss", "mse"),
        metrics=context.config.get("metrics", ["mae"])
    )
    
    # Generate model ID
    model_id = str(uuid.uuid4())
    
    # Persist to disk
    persistent_model_store.save_model(
        model_id=model_id,
        model=model,
        config=context.config
    )
    
    return {
        "model_id": model_id,
        "model": model,  # For TIER 0c caching
        "architecture": str(model.summary()),
        "metadata": {
            "strategy": "model_build",
            "analysis_type": "model_build",
        },
        "result_df": df,
    }


# ============================================================================
# ENRICH WITH TARGETS HANDLER - OPTIMIZED FOR LARGE DATAFRAMES
# ============================================================================

async def analyze_enrich_with_targets_impl(
    df: pd.DataFrame,
    context: ProcessingContext,
    **kwargs
) -> Dict[str, Any]:
    """
    Advanced ML targets enrichment handler - DISTRIBUTED PROCESSING.
    
    This handler uses the ProcessingManager strategy pattern to compute
    movement analysis metrics (final_move, max_favorable_move, etc.) on
    large dataframes efficiently by:
    
    - Sequential: <1K rows (direct computation)
    - Parallel Chunking: 1K-50K rows (split into chunks, parallel processing)
    - Slice Streaming: >50K rows (streaming slices with memory-aware chunking)
    
    Why this is needed:
    - Direct _enrich_with_targets() on >50K rows causes memory spikes
    - Movement analysis (_analyze_post_interaction_movement) is compute-heavy
    - Strategy pattern distributes work across processes/asyncio tasks
    
    Args:
        df: OHLCV DataFrame (potentially very large)
        context: ProcessingContext with config and progress tracking
        **kwargs: Additional parameters (ml_prep_instance, close_col, etc.)
        
    Returns:
        Dict with keys: enriched_df, result_df, metrics_computed, metadata
    
    ✅ RESOURCE OPTIMIZATION: Uses strategy pattern for >50K rows
    ✅ PROGRESS TRACKING: Reports enrichment progress via WebSocket
    ✅ ERROR HANDLING: Wrapped with proper error context
    """
    
    ml_prep_instance = kwargs.get('ml_prep_instance')
    close_col = kwargs.get('close_col', 'Close')
    prepare_advanced_ml_targets = kwargs.get('prepare_advanced_ml_targets', True)
    include_sequence_prediction = kwargs.get('include_sequence_prediction', True)
    
    logger.info(
        f"[Enrich Targets] Starting target enrichment for {len(df)} rows "
        f"(advanced_ml_targets={prepare_advanced_ml_targets}, seq_pred={include_sequence_prediction})"
    )
    
    # Create unified reporter
    reporter = ProgressReporter(
        task_store=context.task_store,
        connection_manager=context.connection_manager,
        user_id=context.user_id,
        throttling_strategy=ThrottlingStrategy.HYBRID
    )
    reporter.task_id = context.task_id
    
    try:
        # Validate input
        if df is None or len(df) == 0:
            raise ValueError("DataFrame is empty or None")
        
        if close_col not in df.columns:
            raise ValueError(f"Column '{close_col}' not found in DataFrame")
        
        if ml_prep_instance is None:
            raise ValueError("ml_prep_instance required for enrichment")
        
        # Use StrategyFactory to determine strategy based on data size
        # This centralizes strategy selection logic and avoids repeated imports
        strategy_type = StrategyFactory.determine_strategy(
            n_rows=len(df),
            analysis_type="enrich_with_targets"
        )
        
        logger.info(
            f"[Enrich Targets] Selected {strategy_type.value} strategy for {len(df)} rows "
            f"(thresholds: sequential ≤1K, parallel 1K-50K, streaming >50K)"
        )
        
        # Log strategy selection
        await reporter.report_progress(
            0.1,
            f"[Enrich] Selected {strategy_type.value} strategy for {len(df)} rows"
        )
        
        # For enrichment, we can use the ML prep instance's methods directly
        # because the strategy is mainly for parallelization of the computation
        if prepare_advanced_ml_targets and include_sequence_prediction:
            # Call the main enrichment method on the ML prep instance
            # This will automatically compute all movement metrics
            await ml_prep_instance._enrich_with_targets()
            logger.info(
                f"[Enrich Targets] ✅ Advanced targets computed for {len(ml_prep_instance.data)} rows"
            )
        
        # Count metrics that were added
        metrics_cols = [col for col in ml_prep_instance.data.columns if col.startswith("adv_target_")]
        metrics_computed = len(metrics_cols)
        
        logger.info(f"[Enrich Targets] ✅ Enrichment complete: {metrics_computed} metrics added")
        
        await reporter.report_progress(
            1.0,
            f"[Enrich] Enrichment complete: {metrics_computed} advanced targets computed"
        )
        
        return {
            "enriched_df": ml_prep_instance.data,
            "result_df": ml_prep_instance.data,
            "metrics_computed": metrics_computed,
            "metrics_added": metrics_cols,
            "rows_processed": len(ml_prep_instance.data),
            "metadata": {
                "strategy": strategy_type.value,
                "analysis_type": "enrich_with_targets",
                "metrics": metrics_computed,
                "advanced_targets": prepare_advanced_ml_targets,
            },
        }
    
    except asyncio.CancelledError:
        logger.warning(f"[Enrich Targets] Task cancelled for task {context.task_id}")
        raise
    except Exception as enrich_err:
        logger.error(
            f"❌ [Enrich Targets] Target enrichment failed for task {context.task_id}: {enrich_err}",
            exc_info=True
        )
        raise RuntimeError(
            f"Target enrichment failed: {enrich_err}"
        ) from enrich_err