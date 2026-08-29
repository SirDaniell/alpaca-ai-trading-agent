"""
Shared data cleaning and preparation utilities.
Ensures data consistency and type integrity across the analysis pipeline.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)

def clean_dataframe(df: pd.DataFrame, fill_value: float = 0.0) -> pd.DataFrame:
    """
    Clean DataFrame by handling NaN, inf, and preserving numeric types.
    Prevents the 'Pointer' logic from losing features due to type coercion to 'object'.
    
    Args:
        df: DataFrame to clean
        fill_value: Value to use for filling NaNs in numeric columns
        
    Returns:
        Cleaned DataFrame with consistent types and no NaN/inf
    """
    if df.empty:
        return df

    # Work on a copy to avoid side effects
    df = df.copy()

    # 1. Replace inf and -inf with NaN first
    df = df.replace([np.inf, -np.inf], np.nan)

    # 2. Iterate through columns and handle by type
    # Deduplicate columns first to avoid 'DataFrame has no attribute dtype' errors
    df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()

    for col in df.columns:
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            # Forward fill then backward fill to preserve local trends if possible
            df[col] = df[col].ffill().bfill()
            # If still has NaN (all values were NaN or at the edges), fill with default
            df[col] = df[col].fillna(fill_value)
        else:
            # For non-numeric columns (like Zodiac, Day), we might want to keep them as is
            # or fill with 'Unknown'
            df[col] = df[col].fillna("Unknown")

    return df

def restore_numeric_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt to restore numeric types to columns that were coerced to 'object'
    during serialization (e.g. numeric columns containing None).
    
    Args:
        df: DataFrame with potential type coercion
        
    Returns:
        DataFrame with restored numeric types where possible
    """
    # Deduplicate columns first to avoid 'DataFrame has no attribute dtype' errors
    df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()

    for col in df.columns:
        series = df[col]
        # Only process if single column (Series). If duplicate exists (DataFrame), we skip as deduplication above handled it.
        if isinstance(series, pd.Series) and series.dtype == 'object':

            # Attempt to convert to numeric, ignoring errors for truly categorical columns
            converted = pd.to_numeric(df[col], errors='coerce')
            
            # If we managed to convert a significant portion to numeric (and it wasn't all NaN)
            if not converted.isna().all():
                # Check if it was mostly numeric before
                # (e.g. if more than 50% are now numeric or it was clearly meant to be numeric)
                # For our case, if it CAN be numeric, it probably should be
                df[col] = converted.fillna(0.0)
                logger.info(f"✅ Restored numeric type for column: {col}")
                
    return df


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize all column name variations to standard OHLCV + Time format.
    Handles all possible variations from different data sources (CSV, MT5, Database).
    
    This function MUST be called early in the pipeline to ensure downstream
    components always work with consistent column names.
    
    Normalization Mapping:
    - Timestamp columns: time, timestamp, date, datetime, Date → 'Time'
    - Open: o, open, OPEN → 'Open'
    - High: h, high, HIGH → 'High'
    - Low: l, low, LOW → 'Low'
    - Close: c, close, CLOSE → 'Close'
    - Volume: v, vol, volume, VOLUME → 'Volume'
    - TickVolume: tick_volume, tick_vol, tickvolume → 'TickVolume'
    
    Args:
        df: DataFrame with any column name variations
        
    Returns:
        DataFrame with normalized column names
        
    Example:
        df = pd.DataFrame({'time': [...], 'close': [...], 'open': [...]})
        df = normalize_dataframe_columns(df)
        # Now has columns: 'Time', 'Close', 'Open'
    """
    if df.empty:
        return df
    
    # Mapping of all variations to standard names
    # Handled in two phases: lowercase matching + rename
    COLUMN_MAPPING = {
        # Timestamp variations (HIGHEST PRIORITY - many sources use different names)
        'time': 'Time',
        'timestamp': 'Time',
        'date': 'Time',
        'datetime': 'Time',
        'ts': 'Time',
        'unix_timestamp': 'Time',
        'epoch': 'Time',
        
        # OHLC variations (case-insensitive)
        'open': 'Open',
        'o': 'Open',
        'high': 'High',
        'h': 'High',
        'low': 'Low',
        'l': 'Low',
        'close': 'Close',
        'c': 'Close',
        
        # Volume variations
        'volume': 'Volume',
        'vol': 'Volume',
        'v': 'Volume',
        'tick_volume': 'TickVolume',
        'tick_vol': 'TickVolume',
        # MT5 Integer variations (Raw rates from bridge)
        '0': 'Time',
        '1': 'Open',
        '2': 'High',
        '3': 'Low',
        '4': 'Close',
        '5': 'TickVolume',
        '6': 'Spread',
        '7': 'Volume',
    }
    
    # Create reverse mapping: actual_column (lowercase) → standard_name
    # Convert column names to strings first, as some sources (MT5) may use integer column names
    lowercase_cols = {str(col).lower(): col for col in df.columns}
    rename_dict = {}
    
    # Apply mappings
    for norm_key, standard_name in COLUMN_MAPPING.items():
        if norm_key in lowercase_cols:
            actual_col = lowercase_cols[norm_key]
            # Only rename if not already the standard name
            if actual_col != standard_name:
                rename_dict[actual_col] = standard_name
                logger.debug(f"Normalizing column: {actual_col} → {standard_name}")
    
    # Special case: If we have both TickVolume (5) and Volume (7), 
    # many strategies expect 'Volume' to be 'TickVolume' for Forex
    if 'TickVolume' in rename_dict.values() and 'Volume' in rename_dict.values():
        logger.debug("Found both TickVolume and Volume, ensuring downstream compatibility")

    
    # Apply all renames at once
    if rename_dict:
        df = df.rename(columns=rename_dict)
        logger.debug(f"✅ Normalized {len(rename_dict)} column names: {list(rename_dict.values())}")
    
    # CRITICAL: Deduplicate columns (Case-normalization or mapping might have created duplicates)
    # We use case-insensitive deduplication to catch variants like 'Time' and 'time'
    if df.columns.str.lower().duplicated().any():
        before = len(df.columns)
        df = df.loc[:, ~df.columns.str.lower().duplicated(keep='first')].copy()
        logger.info(f"🛡️  Deduplicated columns (case-insensitive) in normalize_dataframe_columns: {before} → {len(df.columns)}")

    return df



def normalize_row_dict(row_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a single row dictionary's keys to standard format.
    Useful for converting individual row output before building DataFrames.
    
    Args:
        row_dict: Single row as dictionary with any column name variations
        
    Returns:
        Dictionary with normalized column names
        
    Example:
        row = {'time': '2026-04-05T10:00:00Z', 'close': 100.5}
        row = normalize_row_dict(row)
        # Now has keys: 'Time', 'Close'
    """
    COLUMN_MAPPING = {
        'time': 'Time',
        'timestamp': 'Time',
        'date': 'Time',
        'datetime': 'Time',
        'ts': 'Time',
        'unix_timestamp': 'Time',
        'epoch': 'Time',
        'open': 'Open',
        'o': 'Open',
        'high': 'High',
        'h': 'High',
        'low': 'Low',
        'l': 'Low',
        'close': 'Close',
        'c': 'Close',
        'volume': 'Volume',
        'vol': 'Volume',
        'v': 'Volume',
        'tick_volume': 'TickVolume',
        'tick_vol': 'TickVolume',
        'tickvolume': 'TickVolume',
    }
    
    normalized = {}
    for key, value in row_dict.items():
        # Look up standard name, otherwise keep original
        standard_key = COLUMN_MAPPING.get(key.lower(), key)
        # Rename if it's in the mapping and different
        if standard_key != key and key.lower() in COLUMN_MAPPING:
            normalized[standard_key] = value
        else:
            # Keep as-is for non-standard columns
            normalized[key] = value
    
    return normalized
