"""
Progress reporting utilities for slice-based streaming.

Provides helper functions to synchronize progress reporting across mutation steps
(SNR, Technical, Astronomical) when processing data using the slice-streaming architecture.
"""

from typing import Dict, Optional, Any


def calculate_cumulative_progress(
    local_progress_0_100: float,
    slice_context: Optional[Dict[str, int]] = None,
    sub_progress_weight: float = 0.2
) -> float:
    """
    Calculate cumulative progress accounting for actual dataset size and slice position.
    
    Uses total_dataset_rows and slice boundaries to calculate accurate progress
    based on how much of the total dataset has been processed, not just slice count.
    
    Args:
        local_progress_0_100: Local progress within current function (0-100)
        slice_context: Context dict with:
            - slice_num: Current slice number (0-based)
            - total_slices: Total number of slices to process
            - slice_start: Starting row of current slice
            - slice_end: Ending row of current slice
            - total_dataset_rows: Total rows in the entire dataset
            If None, returns local_progress unchanged
        sub_progress_weight: Weight allocated for sub-progress within slice (default 0.2 = 20%)
    
    Returns:
        Overall progress 0-100 accounting for actual data processed
        
    Example:
        >>> slice_ctx = {
        ...     "slice_num": 2, "total_slices": 4,
        ...     "slice_start": 20000, "slice_end": 30000,
        ...     "total_dataset_rows": 40000
        ... }
        >>> calculate_cumulative_progress(50.0, slice_ctx)
        # Slice 2/4 processing rows 20000-30000 of 40000 total at 50% local progress:
        # Completed rows: 20000 + (10000 * 0.5) = 25000 of 40000 = 62.5%
    """
    # No slice context or single slice: return local progress unchanged
    if slice_context is None or slice_context.get("total_slices", 1) <= 1:
        return float(local_progress_0_100)
    
    # Extract values from slice_context
    slice_num = slice_context.get("slice_num", 0)
    total_slices = slice_context.get("total_slices", 1)
    slice_start = slice_context.get("slice_start", 0)
    slice_end = slice_context.get("slice_end", 0)
    total_dataset_rows = slice_context.get("total_dataset_rows", 0)
    
    # If we don't have row information, fall back to slice-based calculation
    if not total_dataset_rows or slice_start == slice_end:
        # Fallback: slice-based progress calculation
        slice_base = (slice_num / total_slices) * (1 - sub_progress_weight)
        local_fraction = (local_progress_0_100 / 100.0)
        sub_progress = local_fraction * (sub_progress_weight / total_slices)
        cumulative = slice_base + sub_progress
        result_pct = int(cumulative * 100)
    else:
        # Row-based progress calculation (more accurate)
        rows_completed_before_slice = slice_start
        rows_in_current_slice = slice_end - slice_start
        rows_completed_in_slice = rows_in_current_slice * (local_progress_0_100 / 100.0)
        
        total_rows_completed = rows_completed_before_slice + rows_completed_in_slice
        result_pct = int((total_rows_completed / total_dataset_rows) * 100)
    
    # Safety cap: during slice processing, never hit 100%
    # Reserve 100% for ProcessingManager's final completion message
    is_final_slice = (slice_num == total_slices - 1)
    if is_final_slice and local_progress_0_100 >= 99.0:
        return min(99, result_pct)
    elif result_pct >= 100:
        return 99
    
    return result_pct


def format_progress_message(
    stage: str,
    slice_context: Optional[Dict[str, int]] = None,
    local_message: str = "",
    **extra_fields
) -> str:
    """
    Format progress message with slice context for clarity.
    
    Args:
        stage: Processing stage (e.g., "SNR Analysis", "Technical Indicators", "Astronomy")
        slice_context: Slice context dict (optional)
        local_message: The actual progress message (e.g., "Computing RSI...")
        **extra_fields: Additional context (e.g., rows_processed, bodies_calculated)
    
    Returns:
        Formatted message string with slice context
    """
    parts = [stage]
    
    if slice_context and slice_context.get("total_slices", 1) > 1:
        slice_num = slice_context.get("slice_num", 0)
        total_slices = slice_context.get("total_slices", 1)
        parts.append(f"[Slice {slice_num + 1}/{total_slices}]")
    
    if local_message:
        parts.append(local_message)
    
    return " ".join(parts)


def get_slice_info(slice_context: Optional[Dict[str, int]]) -> Dict[str, Any]:
    """
    Extract readable slice information from context.
    
    Args:
        slice_context: Slice context dict
    
    Returns:
        Dictionary with readable slice info for logging/UI
    """
    if slice_context is None or slice_context.get("total_slices", 1) <= 1:
        return {"is_sliced": False}
    
    # Extract values from slice_context (now guaranteed not None)
    slice_num = slice_context.get("slice_num", 0)
    total_slices = slice_context.get("total_slices", 1)
    slice_start = slice_context.get("slice_start", 0)
    slice_end = slice_context.get("slice_end", 0)
    total_dataset_rows = slice_context.get("total_dataset_rows", 0)
    
    return {
        "is_sliced": True,
        "current_slice": slice_num + 1,  # Convert to 1-based
        "total_slices": total_slices,
        "row_range": f"{slice_start}-{slice_end}",
        "total_rows": total_dataset_rows,
    }
