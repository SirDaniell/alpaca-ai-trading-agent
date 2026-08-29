# =====================================================================
# Data cleaning Helpers
# =====================================================================


from typing import Any, Dict, List
import numpy as np
import pandas as pd


def clean_value(value: Any) -> Any:
    """
    Clean a single value to ensure JSON serializability

    Args:
        value: Value to clean

    Returns:
        Cleaned value (None if NaN/inf, otherwise original value)
    """
    if value is None:
        return None

    # Handle numpy types
    if isinstance(value, (np.integer, np.floating)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value) if isinstance(value, np.floating) else int(value)

    # Handle pandas types
    if pd.isna(value):
        return None

    # Handle Python float/int
    if isinstance(value, (float, int)):
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return None
        return value

    # Handle boolean
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    return value


def clean_dataframe_for_json(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert DataFrame to list of dicts with all values cleaned for JSON serialization

    Args:
        df: DataFrame to clean

    Returns:
        List of dictionaries with cleaned values
    """
    if df.empty:
        return []

    # Replace inf and -inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Convert to records
    records = df.to_dict(orient="records")

    # Clean each record
    cleaned_records = []
    for record in records:
        cleaned_record = {}
        for key, value in record.items():
            cleaned_value = clean_value(value)
            # Only include non-None values or keep None if it's explicitly set
            cleaned_record[key] = cleaned_value
        cleaned_records.append(cleaned_record)

    return cleaned_records
