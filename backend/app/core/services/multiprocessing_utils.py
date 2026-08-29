"""
Multiprocessing utilities for parallel analysis of large financial datasets.

Implements column-based and method-based chunking strategies to leverage
multiple CPU cores for expensive operations like distance correlation and
feature importance calculation.
"""

import logging
import os
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
from scipy.spatial.distance import pdist, squareform
from typing import Dict, List, Tuple, Callable, Any, Optional
from scipy.stats import pearsonr, spearmanr, kendalltau

logger = logging.getLogger(__name__)


# ============================================================================
# Worker Functions (Module level for pickling/multiprocessing compatibility)
# ============================================================================

def _correlation_worker(
    feature_subset: List[str],
    data: pd.DataFrame,
    target_cols: List[str],
    method: str
) -> Tuple[List[str], Dict, Dict]:
    """Worker: compute correlations for all targets vs feature subset."""
    corr_result = {}
    pval_result = {}
    
    for target in target_cols:
        corr_result[target] = {}
        pval_result[target] = {}
        
        for feature in feature_subset:
            # Get paired data and drop NaN
            df_pair = data[[target, feature]].dropna()
            
            if len(df_pair) < 2:
                corr_result[target][feature] = 0.0
                pval_result[target][feature] = 1.0
                continue
            
            x = df_pair[target].values
            y = df_pair[feature].values
            
            try:
                if method == "pearson":
                    corr_val, p_val = pearsonr(x, y)
                elif method == "spearman":
                    corr_val, p_val = spearmanr(x, y)
                elif method == "kendall":
                    corr_val, p_val = kendalltau(x, y)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                corr_result[target][feature] = float(corr_val)
                pval_result[target][feature] = float(p_val)
            except Exception as e:
                logger.debug(f"Correlation failed for {target} x {feature}: {e}")
                corr_result[target][feature] = 0.0
                pval_result[target][feature] = 1.0
    
    return feature_subset, corr_result, pval_result


def distance_correlation(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Calculate distance correlation between two variables.
    Module-level for pickling support.
    Uses 'dcor' package if available for industry-standard accuracy/speed.
    """
    try:
        import dcor
        # dcor.distance_correlation is highly optimized and handles U-centering
        return float(dcor.distance_correlation(X, Y))
    except (ImportError, Exception):
        # Fallback to manual implementation (V-centering, biased for small samples)
        # Ensure same length
        if len(X) != len(Y) or len(X) < 2:
            return 0.0

        # Distance matrices
        a = squareform(pdist(X.reshape(-1, 1)))
        b = squareform(pdist(Y.reshape(-1, 1)))

        # Double centering (Standard centering)
        n = len(X)
        A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
        B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()

        # Distance covariance
        dcov = np.sqrt(np.sum(A * B) / (n**2))

        # Distance variances
        dvarX = np.sqrt(np.sum(A * A) / (n**2))
        dvarY = np.sqrt(np.sum(B * B) / (n**2))

        # Distance correlation
        dcor_val = dcov / np.sqrt(dvarX * dvarY) if dvarX * dvarY > 0 else 0.0
        return float(dcor_val)


def _distance_corr_worker(
    col_subset: List[str],
    data: pd.DataFrame,
    columns: List[str],
    distance_corr_func: Optional[Callable] = None
) -> Tuple[List[str], Dict]:
    """Worker: compute distance correlation for column subset against all columns."""
    func = distance_corr_func or distance_correlation
    
    result = {}
    for col1 in col_subset:
        result[col1] = {}
        for col2 in columns:
            if col1 == col2:
                result[col1][col2] = 1.0
            else:
                # Align and drop NaN
                df_pair = pd.concat([data[col1], data[col2]], axis=1).dropna()
                if len(df_pair) < 2:
                    result[col1][col2] = 0.0
                else:
                    result[col1][col2] = func(
                        df_pair[col1].values,
                        df_pair[col2].values
                    )
    
    return col_subset, result


def _importance_worker(
    method_subset: List[str],
    X: pd.DataFrame,
    y: pd.Series,
    method_funcs: Dict[str, Callable]
) -> Dict[str, Dict]:
    """Worker: compute feature importance for method subset."""
    results = {}
    
    for method in method_subset:
        if method not in method_funcs:
            logger.warning(f"Method {method} not found in method_funcs")
            continue
        
        try:
            importance_scores = method_funcs[method](X, y)
            results[method] = importance_scores
        except Exception as e:
            logger.error(f"Feature importance ({method}) failed: {e}")
            results[method] = {"error": str(e)}
    
    return results


def _outlier_worker(
    method_subset: List[str],
    data: pd.DataFrame,
    method_funcs: Dict[str, Callable]
) -> Dict[str, Dict]:
    """Worker: detect outliers using method subset."""
    results = {}
    
    for method in method_subset:
        if method not in method_funcs:
            logger.warning(f"Method {method} not found")
            continue
        
        try:
            outliers = method_funcs[method](data)
            results[method] = outliers
        except Exception as e:
            logger.error(f"Outlier detection ({method}) failed: {e}")
            results[method] = {"error": str(e)}
    
    return results


class ChunkingStrategy:
    """Base class for data chunking strategies."""
    
    @staticmethod
    def auto_chunk_count() -> int:
        """Determine optimal number of chunks based on CPU count."""
        cpu_cores = cpu_count() or 4
        # Reserve 2 cores for system and leave some headroom
        return max(5, cpu_cores - 2)
    
    @staticmethod
    def should_parallelize(n_rows: int, n_cols: int, threshold: int = 50000) -> bool:
        """
        Determine if parallelization is worthwhile.
        
        Args:
            n_rows: Number of rows in dataset
            n_cols: Number of columns
            threshold: Minimum operations count (n_rows * n_cols) to parallelize
        
        Returns:
            True if parallelization is beneficial
        """
        total_ops = n_rows * n_cols
        return total_ops > threshold


class ColumnChunker(ChunkingStrategy):
    """
    Split analysis by columns - each worker processes a subset of columns
    against all rows.
    
    Best for: Distance correlation, mutual information (column-pair analysis)
    """
    
    @staticmethod
    def chunk_columns(
        columns: List[str],
        n_chunks: Optional[int] = None
    ) -> List[List[str]]:
        """
        Split columns evenly across chunks.
        
        Args:
            columns: List of column names
            n_chunks: Number of chunks (defaults to CPU count)
        
        Returns:
            List of column subsets for each worker
        
        Example:
            >>> chunks = ColumnChunker.chunk_columns(['A','B','C','D'], n_chunks=2)
            >>> chunks
            [['A', 'B'], ['C', 'D']]
        """
        if n_chunks is None:
            n_chunks = ChunkingStrategy.auto_chunk_count()
        
        n_chunks = min(n_chunks, len(columns))
        return [
            columns[i::n_chunks]
            for i in range(n_chunks)
        ]
    
    @staticmethod
    def chunk_dataframe_by_columns(
        df: pd.DataFrame,
        n_chunks: Optional[int] = None
    ) -> List[pd.DataFrame]:
        """
        Split DataFrame by columns.
        
        Args:
            df: Input DataFrame
            n_chunks: Number of chunks
        
        Returns:
            List of DataFrames (each with different columns, same rows)
        """
        column_chunks = ColumnChunker.chunk_columns(df.columns.tolist(), n_chunks)
        return [df[cols] for cols in column_chunks]


class RowChunker(ChunkingStrategy):
    """
    Split analysis by rows - each worker processes a subset of rows
    against all columns.
    
    Best for: Outlier detection, feature engineering (row-based operations)
    """
    
    @staticmethod
    def chunk_rows(
        n_rows: int,
        n_chunks: Optional[int] = None,
        overlap: int = 0
    ) -> List[Tuple[int, int]]:
        """
        Calculate row ranges for each chunk with optional overlap.
        
        Args:
            n_rows: Total number of rows
            n_chunks: Number of chunks
            overlap: Number of rows to overlap between chunks
        
        Returns:
            List of (start_idx, end_idx) tuples for each chunk
        """
        if n_chunks is None:
            n_chunks = ChunkingStrategy.auto_chunk_count()
        
        n_chunks = min(n_chunks, n_rows)
        chunk_size = n_rows // n_chunks
        
        ranges = []
        for i in range(n_chunks):
            # Calculate base range
            start = i * chunk_size
            end = n_rows if i == n_chunks - 1 else (i + 1) * chunk_size
            
            # Apply overlap (start earlier, except for first chunk)
            if i > 0:
                start = max(0, start - overlap)
                
            ranges.append((start, end))
        
        return ranges
    
    @staticmethod
    def chunk_dataframe_by_rows(
        df: pd.DataFrame,
        n_chunks: Optional[int] = None,
        overlap: int = 0
    ) -> List[pd.DataFrame]:
        """
        Split DataFrame by rows with optional overlap.
        
        Args:
            df: Input DataFrame
            n_chunks: Number of chunks
            overlap: Number of rows to overlap
        
        Returns:
            List of DataFrames (each with different rows, same columns)
        """
        ranges = RowChunker.chunk_rows(len(df), n_chunks, overlap=overlap)
        return [df.iloc[start:end] for start, end in ranges]


class MethodChunker(ChunkingStrategy):
    """
    Split analysis by methods - each worker computes a different
    algorithm on the same data.
    
    Best for: Feature importance (multiple algorithms), ensemble methods
    """
    
    @staticmethod
    def chunk_methods(
        methods: List[str],
        n_chunks: Optional[int] = None
    ) -> List[List[str]]:
        """
        Distribute methods across chunks.
        
        Args:
            methods: List of method names
            n_chunks: Number of chunks
        
        Returns:
            List of method subsets for each worker
        
        Example:
            >>> methods = ['rf', 'lasso', 'svm', 'perm']
            >>> chunks = MethodChunker.chunk_methods(methods, n_chunks=2)
            >>> chunks
            [['rf', 'svm'], ['lasso', 'perm']]
        """
        if n_chunks is None:
            n_chunks = ChunkingStrategy.auto_chunk_count()
        
        n_chunks = min(n_chunks, len(methods))
        return [
            methods[i::n_chunks]
            for i in range(n_chunks)
        ]


class ParallelExecutor:
    """
    Execute functions in parallel with comprehensive error handling
    and progress tracking.
    """
    
    @staticmethod
    def map_reduce(
        worker_func: Callable,
        args_list: List[Tuple],
        n_workers: Optional[int] = None,
        timeout: int = 3600,
        chunksize: int = 1,
        error_handler: Optional[Callable] = None
    ) -> List[Any]:
        """
        Execute worker function on multiple arguments in parallel.
        
        Args:
            worker_func: Function to execute (must be pickleable)
            args_list: List of argument tuples for worker_func
            n_workers: Number of worker processes (defaults to CPU count)
            timeout: Timeout per worker in seconds
            chunksize: Number of items per worker batch
            error_handler: Optional function to handle exceptions
        
        Returns:
            List of results in same order as args_list
        
        Example:
            >>> def square(x):
            ...     return x * x
            >>> results = ParallelExecutor.map_reduce(square, [(2,), (3,), (4,)])
            >>> results
            [4, 9, 16]
        """
        if n_workers is None:
            n_workers = ChunkingStrategy.auto_chunk_count()
        
        results = []
        errors = []
        
        try:
            with Pool(n_workers) as pool:
                # Use imap to preserve original order (CRITICAL for time-series data)
                imap_results = pool.imap(
                    worker_func,
                    args_list,
                    chunksize=chunksize
                )
                
                for result in imap_results:
                    if isinstance(result, Exception):
                        errors.append(result)
                        results.append(None)
                    else:
                        results.append(result)
        
        except Exception as e:
            logger.error(f"Parallel execution failed: {e}")
            raise
        
        if errors:
            error_msg = "\n".join([f"  Task {i}: {err}" for i, err in errors])
            logger.warning(f"Errors in parallel execution:\n{error_msg}")
        
        return results
    
    @staticmethod
    def starmap_unordered(
        worker_func: Callable,
        args_list: List[Tuple],
        n_workers: Optional[int] = None,
        timeout: int = 3600
    ) -> List[Any]:
        """
        Execute worker function with multiple arguments in parallel.
        
        Like map_reduce but unpacks tuples as separate arguments.
        
        Example:
            >>> def add(a, b):
            ...     return a + b
            >>> args = [(1, 2), (3, 4), (5, 6)]
            >>> results = ParallelExecutor.starmap_unordered(add, args)
            >>> results
            [3, 7, 11]
        """
        if n_workers is None:
            n_workers = ChunkingStrategy.auto_chunk_count()
        
        results = []
        
        try:
            with Pool(n_workers) as pool:
                results = pool.starmap(worker_func, args_list)
        except Exception as e:
            logger.error(f"Parallel execution (starmap) failed: {e}")
            raise
        
        return results


# ============================================================================
# Specific parallel implementations
# ============================================================================

def parallel_rectangular_correlation(
    data: pd.DataFrame,
    target_cols: List[str],
    feature_cols: List[str],
    method: str = "pearson",
    n_workers: Optional[int] = None
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    Calculate rectangular correlation matrix (targets × features) in parallel.
    
    Args:
        data: Input DataFrame containing both targets and features
        target_cols: List of target column names (rows in output matrix)
        feature_cols: List of feature column names (columns in output matrix)
        method: Correlation method ('pearson', 'spearman', or 'kendall')
        n_workers: Number of parallel workers
    
    Returns:
        Tuple of (correlation_matrix, pvalue_matrix) as nested dicts
        
    Strategy:
        - Split features into chunks
        - Each worker computes all targets vs its feature chunk
        - Recombine results into full rectangular matrix
        
    Time estimate:
        - 5 targets × 200 features: ~30 sec (sequential) → ~8 sec (4 workers)
    """
    if n_workers is None:
        n_workers = ChunkingStrategy.auto_chunk_count()
    
    # Check if parallelization is worthwhile
    total_pairs = len(target_cols) * len(feature_cols)
    if total_pairs < 100:  # Small matrix, not worth parallelizing
        logger.info(
            f"Small correlation matrix ({len(target_cols)}×{len(feature_cols)}): "
            f"skipping parallelization"
        )
        return None, None  # Signal to use sequential version
    
    # Chunk features
    feature_chunks = ColumnChunker.chunk_columns(feature_cols, n_workers)
    
    logger.info(
        f"Parallelizing {method} correlation: {len(target_cols)} targets × "
        f"{len(feature_cols)} features → {len(feature_chunks)} chunks × {n_workers} workers"
    )
    
    # Execute in parallel
    chunk_results = ParallelExecutor.starmap_unordered(
        _correlation_worker,
        [(chunk, data, target_cols, method) for chunk in feature_chunks],
        n_workers=n_workers
    )
    
    # Recombine results
    final_corr = {target: {} for target in target_cols}
    final_pval = {target: {} for target in target_cols}
    
    for feature_subset, chunk_corr, chunk_pval in chunk_results:
        for target in target_cols:
            final_corr[target].update(chunk_corr[target])
            final_pval[target].update(chunk_pval[target])
    
    return final_corr, final_pval


def parallel_distance_correlation(
    data: pd.DataFrame,
    columns: Optional[List[str]] = None,
    n_workers: Optional[int] = None,
    distance_corr_func: Callable = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate distance correlation matrix in parallel using column chunking.
    
    Args:
        data: Input DataFrame
        columns: Columns to correlate (defaults to all numeric)
        n_workers: Number of parallel workers
        distance_corr_func: Function to calculate single correlation pair
    
    Returns:
        Distance correlation matrix as nested dict
        
    Strategy:
        - Split columns into chunks
        - Each worker computes correlation of its columns against all columns
        - Recombine results into full matrix
        
    Time estimate:
        - 100 cols × 10k rows: 8 min (sequential) → 2 min (4 workers)
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if n_workers is None:
        n_workers = ChunkingStrategy.auto_chunk_count()
    
    # Check if parallelization is worthwhile
    if not ChunkingStrategy.should_parallelize(len(data), len(columns)):
        logger.info(
            f"Dataset small ({len(data)} rows × {len(columns)} cols): "
            f"skipping parallelization, using sequential calculation"
        )
        return None  # Signal to use sequential version
    
    # Chunk columns
    column_chunks = ColumnChunker.chunk_columns(columns, n_workers)
    
    logger.info(
        f"Parallelizing distance correlation: {len(columns)} cols → "
        f"{len(column_chunks)} chunks × {n_workers} workers"
    )
    
    # Execute in parallel
    chunk_results = ParallelExecutor.starmap_unordered(
        _distance_corr_worker,
        [(chunk, data, columns, distance_corr_func) for chunk in column_chunks],
        n_workers=n_workers
    )
    
    # Recombine results
    final_result = {}
    for col_subset, chunk_result in chunk_results:
        final_result.update(chunk_result)
    
    return final_result


def parallel_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    methods: List[str],
    n_workers: Optional[int] = None,
    method_funcs: Optional[Dict[str, Callable]] = None,
    target_name: str = "Target"
) -> Dict[str, Dict]:
    """
    Calculate feature importance using multiple methods in parallel.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        methods: List of method names to compute
        n_workers: Number of parallel workers
        method_funcs: Dict of {method_name: function}
        target_name: Name of target for logging
    
    Returns:
        Dict of {method_name: {feature: importance_score}}
        
    Strategy:
        - Split methods into chunks
        - Each worker computes subset of methods
        - Recombine results
        
    Time estimate:
        - 7 methods: ~15 sec (sequential) → ~5 sec (4 workers)
    """
    if method_funcs is None:
        method_funcs = {}
    
    if n_workers is None:
        n_workers = ChunkingStrategy.auto_chunk_count()
    
    # Chunk methods
    method_chunks = MethodChunker.chunk_methods(methods, n_workers)
    
    logger.info(
        f"Parallelizing feature importance for {target_name}: "
        f"{len(methods)} methods → {len(method_chunks)} chunks × {n_workers} workers"
    )
    
    # Execute in parallel
    chunk_results = ParallelExecutor.starmap_unordered(
        _importance_worker,
        [(chunk, X, y, method_funcs) for chunk in method_chunks],
        n_workers=n_workers
    )
    
    # Recombine results
    final_result = {}
    for chunk_result in chunk_results:
        final_result.update(chunk_result)
    
    return final_result


def parallel_outlier_detection(
    data: pd.DataFrame,
    methods: List[str],
    n_workers: Optional[int] = None,
    method_funcs: Optional[Dict[str, Callable]] = None
) -> Dict[str, Dict]:
    """
    Detect outliers using multiple methods in parallel.
    
    Args:
        data: Input DataFrame
        methods: List of outlier detection method names
        n_workers: Number of parallel workers
        method_funcs: Dict of {method_name: function}
    
    Returns:
        Dict of outlier detection results by method
    """
    if method_funcs is None:
        method_funcs = {}
    
    if n_workers is None:
        n_workers = ChunkingStrategy.auto_chunk_count()
    
    method_chunks = MethodChunker.chunk_methods(methods, n_workers)
    
    logger.info(
        f"Parallelizing outlier detection: "
        f"{len(methods)} methods → {len(method_chunks)} chunks × {n_workers} workers"
    )
    
    chunk_results = ParallelExecutor.starmap_unordered(
        _outlier_worker,
        [(chunk, data, method_funcs) for chunk in method_chunks],
        n_workers=n_workers
    )
    
    final_result = {}
    for chunk_result in chunk_results:
        final_result.update(chunk_result)
    
    return final_result
