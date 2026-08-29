"""
ML Dataset Preparation Validation & Error Handling

This module implements the 3-tier validation hierarchy:
- Layer 1 (AM): Config validation, data availability
- Layer 2 (PM): Data split validation, scaler validation
- Layer 3 (MDP): Data structure validation, feature validation

All validation functions follow the pattern:
    validate_X(...) -> Tuple[bool, Optional[str]]
    Returns: (is_valid, error_message)
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from app.core.ml.ml_dataset_preparation import DatasetConfig

logger = logging.getLogger(__name__)


# ============================================================================
# LAYER 1: ANALYSIS MANAGER VALIDATION (Config & Data Availability)
# ============================================================================

def validate_ml_config(config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate ML preparation config before processing (AM Level).
    
    Checks:
    - Required fields present
    - Value ranges valid
    - Split ratios sum to 1.0
    - Target columns valid
    
    Args:
        config: Config dictionary from frontend
        
    Returns:
        (is_valid, error_message)
    """
    # Required fields
    required_fields = ['sequence_length', 'prediction_length', 'target_columns']
    for field in required_fields:
        if field not in config:
            return False, f"Missing required field: {field}"
    
    # Value validation
    if config['sequence_length'] <= 0:
        return False, f"sequence_length must be positive, got {config['sequence_length']}"
    
    if config['prediction_length'] < 0:
        return False, f"prediction_length cannot be negative, got {config['prediction_length']}"
    
    # Split ratios
    train_ratio = config.get('train_ratio', 0.7)
    val_ratio = config.get('validation_ratio', 0.15)
    test_ratio = config.get('test_ratio', 0.15)
    total_ratio = train_ratio + val_ratio + test_ratio
    
    if not np.isclose(total_ratio, 1.0):
        return False, f"Split ratios must sum to 1.0, got {total_ratio}"
    
    # Target columns validation
    if not isinstance(config.get('target_columns'), list):
        return False, "target_columns must be a list"
    
    if len(config.get('target_columns', [])) == 0:
        return False, "target_columns cannot be empty"
    
    return True, None


def validate_ml_data(df: pd.DataFrame, config: DatasetConfig) -> Tuple[bool, Optional[str]]:
    """
    Validate data before ML processing (AM Level).
    
    Checks:
    - DataFrame not empty
    - Sufficient rows for sequencing
    - Training split will be large enough
    
    Args:
        df: Input DataFrame
        config: DatasetConfig object
        
    Returns:
        (is_valid, error_message)
    """
    # Empty check
    if df is None or len(df) == 0:
        return False, "DataFrame is empty"
    
    # Row count check
    min_required = config.sequence_length + config.prediction_length
    if len(df) < min_required:
        return False, (
            f"Insufficient data: need {min_required} rows "
            f"(sequence_length={config.sequence_length} + "
            f"prediction_length={config.prediction_length}), "
            f"got {len(df)}"
        )
    
    # Split size check
    train_end = int(len(df) * config.train_ratio)
    if train_end < config.sequence_length:
        return False, (
            f"Training split too small: {train_end} rows < "
            f"{config.sequence_length} sequence_length"
        )
    
    return True, None


# ============================================================================
# LAYER 2: PROCESSING MANAGER VALIDATION (Splits & Scaler)
# ============================================================================

def validate_ml_splits(
    splits: Dict[str, pd.DataFrame],
    config: DatasetConfig
) -> Tuple[bool, Optional[str]]:
    """
    Validate data splits before processing (PM Level).
    
    Checks:
    - All splits present
    - No empty splits
    - Training split large enough for sequences
    
    Args:
        splits: Dict with train/validation/test DataFrames
        config: DatasetConfig object
        
    Returns:
        (is_valid, error_message)
    """
    required_splits = ["train", "validation", "test"]
    
    # Check all splits present
    for split_name in required_splits:
        if split_name not in splits:
            return False, f"Missing split: {split_name}"
    
    # Check no empty splits
    for split_name in required_splits:
        if len(splits[split_name]) == 0:
            return False, f"{split_name.capitalize()} split is empty"
    
    # Check training split size
    train_df = splits["train"]
    min_train_rows = config.sequence_length + config.prediction_length
    if len(train_df) < min_train_rows:
        return False, (
            f"Training split too small: {len(train_df)} rows < "
            f"{min_train_rows} required for sequence generation"
        )
    
    return True, None


def validate_scaler(scaler: Any, feature_cols: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate fitted scaler before use (PM Level).
    
    Checks:
    - Scaler not None
    - Scaler has transform method
    - Feature columns not empty
    - Scaler was fitted (has n_features_in_)
    - Feature count matches
    
    Args:
        scaler: Fitted scaler object
        feature_cols: List of feature column names
        
    Returns:
        (is_valid, error_message)
    """
    if scaler is None:
        return False, "Scaler is None"
    
    if not hasattr(scaler, 'transform'):
        return False, f"Scaler missing transform method: {type(scaler)}"
    
    if len(feature_cols) == 0:
        return False, "No feature columns to scale"
    
    # Check if scaler was fitted
    if not hasattr(scaler, 'n_features_in_'):
        return False, "Scaler was not fitted"
    
    if scaler.n_features_in_ != len(feature_cols):
        return False, (
            f"Scaler feature count mismatch: "
            f"fitted on {scaler.n_features_in_}, got {len(feature_cols)}"
        )
    
    return True, None


def validate_ml_features(
    df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[bool, Optional[str]]:
    """
    Validate feature columns before scaling (PM Level).
    
    Checks:
    - Feature columns not empty
    - All feature columns exist in DataFrame
    - All feature columns are numeric
    
    Args:
        df: Input DataFrame
        feature_cols: List of feature column names
        
    Returns:
        (is_valid, error_message)
    """
    if len(feature_cols) == 0:
        return False, "No feature columns identified"
    
    # Check all columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        return False, f"Missing feature columns: {missing_cols}"
    
    # Check all columns are numeric
    non_numeric = []
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric.append(col)
    
    if non_numeric:
        return False, f"Non-numeric feature columns: {non_numeric}"
    
    return True, None


# ============================================================================
# LAYER 3: ML DATASET PREPARATION VALIDATION (Data Structure)
# ============================================================================

def validate_ml_data_structure(
    df: pd.DataFrame,
    config: DatasetConfig
) -> Tuple[bool, Optional[str]]:
    """
    Validate data structure for ML processing (MDP Level).
    
    Checks:
    - Required OHLCV columns present
    - Sufficient data points for sequencing
    - Signal columns or SNR features present
    
    Args:
        df: Input DataFrame
        config: DatasetConfig object
        
    Returns:
        (is_valid, error_message)
    """
    # Check for required OHLCV columns (case-insensitive)
    required_cols = ["open", "high", "low", "close", "volume"]
    available_cols_lower = [col.lower() for col in df.columns]
    
    missing_cols = [col for col in required_cols if col not in available_cols_lower]
    if missing_cols:
        return False, (
            f"Missing required columns: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Check for sufficient data points
    min_required = config.sequence_length + config.prediction_length
    if len(df) < min_required:
        return False, (
            f"Insufficient data points ({len(df)}) for sequence generation. "
            f"Need at least {min_required} rows "
            f"(sequence_length={config.sequence_length} + "
            f"prediction_length={config.prediction_length})."
        )
    
    # Check for signal columns or SNR features
    confirmed_signal_cols = [
        "Signal_bounce_support",
        "Signal_bounce_resistance",
        "Signal_breakout_support",
        "Signal_breakout_resistance"
    ]
    
    has_confirmed_signals = any(col in df.columns for col in confirmed_signal_cols)
    has_snr_features = any(col.startswith("snr_") for col in df.columns)
    
    if not has_confirmed_signals and not has_snr_features:
        return False, (
            f"No signal columns or SNR features found. Expected:\n"
            f"  - Confirmed signals: Signal_bounce_*, Signal_breakout_*\n"
            f"  - SNR features: snr_dist_to_nearest_*, snr_in_zone, snr_num_levels_*\n"
            f"Available: {list(df.columns)}"
        )
    
    return True, None


def validate_sequence_generation_result(
    result: Dict[str, Any],
    split_name: str
) -> Tuple[bool, Optional[str]]:
    """
    Validate sequence generation result (MDP Level).
    
    Checks:
    - Sequences array not empty
    - Labels array not empty
    - Sequences and labels have same length
    - Sequence shape correct
    
    Args:
        result: Result dict from sequence generation
        split_name: Name of the split (for error messages)
        
    Returns:
        (is_valid, error_message)
    """
    if "sequences" not in result:
        return False, f"{split_name}: Missing 'sequences' in result"
    
    if "labels" not in result:
        return False, f"{split_name}: Missing 'labels' in result"
    
    sequences = result["sequences"]
    labels = result["labels"]
    
    if len(sequences) == 0:
        return False, f"{split_name}: No sequences generated"
    
    if len(labels) == 0:
        return False, f"{split_name}: No labels generated"
    
    if len(sequences) != len(labels):
        return False, (
            f"{split_name}: Sequence/label count mismatch: "
            f"{len(sequences)} sequences vs {len(labels)} labels"
        )
    
    # Check sequence shape
    if len(sequences.shape) != 3:
        return False, (
            f"{split_name}: Invalid sequence shape: {sequences.shape}. "
            f"Expected (n_samples, sequence_length, n_features)"
        )
    
    return True, None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_validation_summary(
    config: Dict[str, Any],
    df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Get comprehensive validation summary for debugging.
    
    Args:
        config: Config dictionary
        df: Input DataFrame
        
    Returns:
        Dict with validation results for all checks
    """
    from app.core.ml.ml_dataset_preparation import DatasetConfig
    
    summary = {
        "config_valid": False,
        "data_valid": False,
        "data_structure_valid": False,
        "errors": []
    }
    
    # Config validation
    is_valid, error = validate_ml_config(config)
    summary["config_valid"] = is_valid
    if not is_valid:
        summary["errors"].append(f"Config: {error}")
    
    if is_valid:
        # Convert to DatasetConfig for further validation
        try:
            config_obj = DatasetConfig(**config)
            
            # Data validation
            is_valid, error = validate_ml_data(df, config_obj)
            summary["data_valid"] = is_valid
            if not is_valid:
                summary["errors"].append(f"Data: {error}")
            
            # Data structure validation
            is_valid, error = validate_ml_data_structure(df, config_obj)
            summary["data_structure_valid"] = is_valid
            if not is_valid:
                summary["errors"].append(f"Structure: {error}")
                
        except Exception as e:
            summary["errors"].append(f"Config conversion: {str(e)}")
    
    summary["all_valid"] = len(summary["errors"]) == 0
    
    return summary
