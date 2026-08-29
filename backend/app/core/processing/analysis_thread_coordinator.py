"""
Analysis Step Multi-Threading Coordinator

Ensures all analysis steps (Technical, SNR, Astronomical, Features, ML Prep) 
run with proper parallelization to avoid single-threaded bottlenecks.

Key Features:
- Auto-detect CPU count for optimal worker count
- ThreadPoolExecutor for I/O-bound operations
- ProcessPoolExecutor for CPU-bound analysis
- Progress aggregation across workers
- Memory-efficient chunk processing
"""

import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, Any, Optional, Callable, List
from functools import wraps
import os

logger = logging.getLogger(__name__)


class AnalysisThreadCoordinator:
    """
    Manages multi-threaded/multi-process execution of analysis steps.
    
    CRITICAL: Prevents single-threaded bottlenecks for large datasets.
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize coordinator with thread pool.
        
        Args:
            max_workers: Max workers (default: CPU count - 1)
        """
        if max_workers is None:
            max_workers = max(1, os.cpu_count() - 1)
        
        self.max_workers = max_workers
        self.thread_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=max_workers)
        
        logger.info(
            f"🚀 AnalysisThreadCoordinator initialized with {max_workers} workers"
        )
    
    async def run_analysis_step_parallel(
        self,
        step_name: str,
        chunks: List[Any],
        process_func: Callable[[Any, Dict[str, Any]], Any],
        config: Dict[str, Any],
        use_processes: bool = False,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run analysis on chunks in parallel.
        
        CRITICAL: Chunks processed simultaneously, not sequentially.
        
        Args:
            step_name: Name of analysis step (for logging)
            chunks: List of data chunks to process
            process_func: Function to process each chunk
            config: Configuration parameters
            use_processes: Use ProcessPoolExecutor (CPU-bound) vs ThreadPoolExecutor (I/O-bound)
            progress_callback: Optional callback for progress updates
        
        Returns:
            Aggregated results from all chunks
        """
        logger.info(
            f"🔄 {step_name}: Processing {len(chunks)} chunks "
            f"with {self.max_workers} workers..."
        )
        
        executor = self.process_executor if use_processes else self.thread_executor
        
        results = {
            "chunk_results": [],
            "aggregated": {},
            "total_time": 0,
        }
        
        import time
        start_time = time.time()
        
        # Submit all chunk jobs
        futures = {}
        for chunk_idx, chunk in enumerate(chunks):
            future = executor.submit(process_func, chunk, config)
            futures[future] = chunk_idx
        
        # Process completions as they finish
        completed = 0
        for future in as_completed(futures):
            chunk_idx = futures[future]
            completed += 1
            
            try:
                result = future.result()
                results["chunk_results"].append({
                    "chunk_index": chunk_idx,
                    "result": result
                })
                
                # Progress callback
                if progress_callback:
                    progress = int((completed / len(chunks)) * 100)
                    progress_callback(
                        progress,
                        f"{step_name}: {completed}/{len(chunks)} chunks complete"
                    )
                
                logger.debug(
                    f"{step_name}: Chunk {chunk_idx} completed "
                    f"({completed}/{len(chunks)})"
                )
                
            except Exception as e:
                logger.error(
                    f"{step_name}: Chunk {chunk_idx} failed: {e}",
                    exc_info=True
                )
                raise
        
        results["total_time"] = time.time() - start_time
        
        logger.info(
            f"✅ {step_name}: All {len(chunks)} chunks processed "
            f"in {results['total_time']:.2f}s"
        )
        
        return results
    
    async def run_in_executor(
        self,
        func: Callable,
        *args,
        use_process: bool = False,
        **kwargs
    ) -> Any:
        """
        Run function in thread/process pool asynchronously.
        
        Args:
            func: Function to run
            *args: Positional arguments
            use_process: Use ProcessPoolExecutor instead of ThreadPoolExecutor
            **kwargs: Keyword arguments
        
        Returns:
            Function result
        """
        loop = asyncio.get_event_loop()
        executor = self.process_executor if use_process else self.thread_executor
        
        return await loop.run_in_executor(
            executor,
            lambda: func(*args, **kwargs)
        )
    
    def shutdown(self):
        """Clean up thread/process pools."""
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        logger.info("🛑 AnalysisThreadCoordinator shut down")


# Global coordinator instance
_coordinator: Optional[AnalysisThreadCoordinator] = None


def get_analysis_coordinator() -> AnalysisThreadCoordinator:
    """Get or create global coordinator."""
    global _coordinator
    if _coordinator is None:
        _coordinator = AnalysisThreadCoordinator()
    return _coordinator


def async_analysis_step(step_name: str, use_processes: bool = False):
    """
    Decorator for analysis step functions.
    
    Automatically handles parallelization and progress reporting.
    
    Usage:
        @async_analysis_step("technical_analysis", use_processes=False)
        async def calculate_indicators(df, config):
            # Function runs in thread pool
            return result
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(data, config: Dict[str, Any], progress_callback=None):
            coordinator = get_analysis_coordinator()
            
            # Run in executor
            result = await coordinator.run_in_executor(
                func,
                data,
                config,
                use_process=use_processes
            )
            
            # Progress callback if provided
            if progress_callback:
                progress_callback(100, f"{step_name} complete")
            
            return result
        
        return wrapper
    return decorator


# Multi-threading configuration for each analysis step
ANALYSIS_STEP_CONFIG = {
    "technical_analysis": {
        "use_processes": False,  # I/O bound (database access)
        "chunk_method": "constant_size",
        "chunk_size": 1000000,  # 1M rows per chunk
    },
    "snr_analysis": {
        "use_processes": True,  # CPU bound (spatial calculations)
        "chunk_method": "smart_overlap",  # Overlapping chunks with lookback
        "chunk_size": 1000000,
    },
    "astronomical_analysis": {
        "use_processes": False,  # I/O bound (lookups)
        "chunk_method": "constant_size",
        "chunk_size": 1000000,
    },
    "feature_analysis": {
        "use_processes": False,  # I/O bound (rolling windows)
        "chunk_method": "constant_size",
        "chunk_size": 1000000,
    },
    "ml_preparation": {
        "use_processes": False,  # I/O bound (sequence creation)
        "chunk_method": "constant_size",
        "chunk_size": 1000000,
    },
}


async def ensure_multi_threading_for_step(
    step_name: str,
    data: Any,
    config: Dict[str, Any],
    process_func: Callable,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Wrapper that ensures a step uses multi-threading.
    
    CRITICAL: Prevents single-threaded processing for large datasets.
    
    Args:
        step_name: Name of the analysis step
        data: Input data (dataframe or chunks)
        config: Analysis configuration
        process_func: Function to process data
        progress_callback: Optional progress callback
    
    Returns:
        Processing results
    """
    step_config = ANALYSIS_STEP_CONFIG.get(step_name, {})
    
    coordinator = get_analysis_coordinator()
    
    # If data is already chunked, use parallel processing
    if isinstance(data, list):
        chunks = data
    else:
        # Convert dataframe to chunks
        chunk_size = step_config.get("chunk_size", 1000000)
        chunks = [data.iloc[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    
    # Process chunks in parallel
    results = await coordinator.run_analysis_step_parallel(
        step_name=step_name,
        chunks=chunks,
        process_func=process_func,
        config=config,
        use_processes=step_config.get("use_processes", False),
        progress_callback=progress_callback,
    )
    
    return results
