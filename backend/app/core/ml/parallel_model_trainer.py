"""
NOTE: Experimental feature, currently under construction
Parallel model training utilities for training multiple models on different datasets
simultaneously using multiprocessing across CPU cores.

Follows the same chunking/worker/executor pattern as multiprocessing_utils.py
but adapted for model training instead of correlation analysis.

Architecture:
    - ModelChunker: Split models/datasets into chunks for parallel execution
    - _train_single_model_worker(): Module-level function for pickling support
    - TrainingExecutor: Execute trainers in parallel using Pool
    - train_models_parallel(): Public API for parallel training submission
"""

import logging
import asyncio
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TrainingJob:
    """Definition of a single model training job."""
    model_id: str
    dataset_id: str
    model_config: Dict[str, Any]  # architecture, layers, dropout, etc.
    hyperparams: Dict[str, Any]   # epochs, batch_size, learning_rate, etc.
    session_id: str
    
    def __hash__(self):
        return hash((self.model_id, self.dataset_id))


@dataclass
class TrainingResult:
    """Result of training a single model."""
    model_id: str
    dataset_id: str
    success: bool
    accuracy: Optional[float] = None
    loss: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    training_time_seconds: float = 0.0
    epochs_completed: int = 0
    final_epoch: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    trained_model_path: Optional[str] = None
    timestamp: datetime = None
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary for JSON serialization."""
        return {
            "model_id": self.model_id,
            "dataset_id": self.dataset_id,
            "success": self.success,
            "accuracy": float(self.accuracy) if self.accuracy is not None else None,
            "loss": float(self.loss) if self.loss is not None else None,
            "precision": float(self.precision) if self.precision is not None else None,
            "recall": float(self.recall) if self.recall is not None else None,
            "f1_score": float(self.f1_score) if self.f1_score is not None else None,
            "training_time_seconds": float(self.training_time_seconds),
            "epochs_completed": int(self.epochs_completed),
            "final_epoch": int(self.final_epoch),
            "error_message": self.error_message,
            "trained_model_path": self.trained_model_path,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


# ============================================================================
# Worker Functions (Module-level for pickling/multiprocessing compatibility)
# ============================================================================

def _train_single_model_worker(
    model_id: str,
    dataset_id: str,
    model_config: Dict[str, Any],
    hyperparams: Dict[str, Any],
    session_id: str,
    get_dataset_func: Callable,
    build_model_func: Callable,
    execute_training_func: Callable
) -> TrainingResult:
    """
    Worker function: Train a single model on a dataset.
    
    This function runs in a separate process. It:
    1. Fetches the dataset from DB/cache
    2. Builds the model with given configuration
    3. Trains the model with given hyperparameters
    4. Returns metrics (accuracy, loss, F1, etc.)
    
    Args:
        model_id: Unique model identifier
        dataset_id: Dataset to train on
        model_config: Model architecture config (layers, units, dropout, etc.)
        hyperparams: Training hyperparameters (epochs, batch_size, learning_rate, etc.)
        session_id: Session context
        get_dataset_func: Callable to fetch dataset by ID
        build_model_func: Callable to build TensorFlow model from config
        execute_training_func: Callable to run training loop
    
    Returns:
        TrainingResult with metrics and status
    
    Note:
        All callables must be pickleable (not lambda, not class methods)
    """
    start_time = datetime.now()
    result = TrainingResult(
        model_id=model_id,
        dataset_id=dataset_id,
        success=False,
        timestamp=start_time
    )
    
    try:
        logger.info(f"[Worker] Training {model_id} on {dataset_id} (session: {session_id})")
        
        # Step 1: Fetch dataset
        try:
            dataset = get_dataset_func(session_id, dataset_id)
            if dataset is None:
                raise ValueError(f"Dataset {dataset_id} not found")
            
            train_data = dataset.get("train_data")
            val_data = dataset.get("validation_data")
            test_data = dataset.get("test_data")
            train_labels = dataset.get("train_labels")
            val_labels = dataset.get("validation_labels")
            test_labels = dataset.get("test_labels")
            
            if train_data is None or train_labels is None:
                raise ValueError(f"Dataset {dataset_id} missing train data/labels")
            
            logger.debug(f"[Worker] Loaded dataset: {train_data.shape}")
        
        except Exception as e:
            result.error_message = f"Failed to load dataset: {str(e)}"
            logger.error(f"[Worker] {result.error_message}")
            return result
        
        # Step 2: Build model
        try:
            model = build_model_func(
                model_config=model_config,
                input_shape=train_data.shape[1:],  # (sequence_length, n_features)
                output_shape=train_labels.shape[1:] if len(train_labels.shape) > 1 else 1
            )
            logger.debug(f"[Worker] Model built successfully")
        
        except Exception as e:
            result.error_message = f"Failed to build model: {str(e)}"
            logger.error(f"[Worker] {result.error_message}")
            return result
        
        # Step 3: Train model
        try:
            training_result = execute_training_func(
                model=model,
                train_data=train_data,
                train_labels=train_labels,
                val_data=val_data,
                val_labels=val_labels,
                test_data=test_data,
                test_labels=test_labels,
                hyperparams=hyperparams
            )
            
            # Extract metrics from training result
            result.success = training_result.get("success", False)
            result.accuracy = training_result.get("accuracy")
            result.loss = training_result.get("loss")
            result.precision = training_result.get("precision")
            result.recall = training_result.get("recall")
            result.f1_score = training_result.get("f1_score")
            result.epochs_completed = training_result.get("epochs_completed", 0)
            result.final_epoch = training_result.get("final_epoch", 0)
            result.trained_model_path = training_result.get("model_path")
            
            logger.info(
                f"[Worker] {model_id} training complete: "
                f"accuracy={result.accuracy:.4f}, loss={result.loss:.4f}"
            )
        
        except Exception as e:
            result.error_message = f"Training failed: {str(e)}"
            logger.error(f"[Worker] {result.error_message}")
            return result
        
        finally:
            # Clean up large objects
            try:
                if 'model' in locals():
                    del model
                if 'dataset' in locals():
                    del dataset
            except:
                pass
    
    except Exception as e:
        result.error_message = f"Unexpected error during training: {str(e)}"
        logger.error(f"[Worker] {result.error_message}")
        return result
    
    finally:
        # Always set elapsed time
        elapsed = (datetime.now() - start_time).total_seconds()
        result.training_time_seconds = elapsed
        logger.info(f"[Worker] {model_id} finished in {elapsed:.2f} seconds")
    
    return result


# ============================================================================
# Chunking Strategies
# ============================================================================

class ModelChunker:
    """
    Split model training jobs into chunks for parallel execution.
    
    Similar to ColumnChunker/RowChunker but for model training tasks.
    Each chunk contains a subset of (model_id, dataset_id) pairs.
    """
    
    @staticmethod
    def auto_chunk_count() -> int:
        """Determine optimal number of chunks based on CPU count."""
        cpu_cores = cpu_count() or 4
        # Reserve 2 cores for system/I/O, leave some headroom
        return max(3, cpu_cores - 2)
    
    @staticmethod
    def chunk_training_jobs(
        jobs: List[TrainingJob],
        n_chunks: Optional[int] = None
    ) -> List[List[TrainingJob]]:
        """
        Split training jobs evenly across chunks.
        
        Args:
            jobs: List of TrainingJob objects
            n_chunks: Number of chunks (defaults to CPU count - 2)
        
        Returns:
            List of job subsets for each worker
        
        Example:
            >>> jobs = [TrainingJob(...), TrainingJob(...), TrainingJob(...), TrainingJob(...)]
            >>> chunks = ModelChunker.chunk_training_jobs(jobs, n_chunks=2)
            >>> len(chunks)
            2
            >>> len(chunks[0]) + len(chunks[1])
            4
        """
        if n_chunks is None:
            n_chunks = ModelChunker.auto_chunk_count()
        
        # Don't create more chunks than jobs
        n_chunks = min(n_chunks, len(jobs))
        
        if n_chunks == 1:
            return [jobs]
        
        # Distribute jobs round-robin across chunks
        chunks = [[] for _ in range(n_chunks)]
        for idx, job in enumerate(jobs):
            chunks[idx % n_chunks].append(job)
        
        return [chunk for chunk in chunks if chunk]  # Remove empty chunks


# ============================================================================
# Parallel Executor
# ============================================================================

class TrainingExecutor:
    """
    Execute model training jobs in parallel across multiple CPU cores.
    
    Follows the same pattern as ParallelExecutor from multiprocessing_utils.py
    but adapted for training tasks.
    """
    
    @staticmethod
    def execute_parallel(
        jobs: List[TrainingJob],
        get_dataset_func: Callable,
        build_model_func: Callable,
        execute_training_func: Callable,
        n_workers: Optional[int] = None,
        timeout: int = 3600
    ) -> Tuple[List[TrainingResult], Dict[str, Any]]:
        """
        Execute multiple training jobs in parallel.
        
        Args:
            jobs: List of TrainingJob objects to execute
            get_dataset_func: Function to fetch dataset by (session_id, dataset_id)
            build_model_func: Function to build model from config
            execute_training_func: Function to run training loop
            n_workers: Number of parallel workers (defaults to CPU count - 2)
            timeout: Timeout per worker in seconds
        
        Returns:
            Tuple of (training_results, aggregated_stats)
            - training_results: List of TrainingResult objects
            - aggregated_stats: Summary statistics {total_time, avg_accuracy, best_model, etc.}
        
        Example:
            >>> results, stats = TrainingExecutor.execute_parallel(
            ...     jobs=[job1, job2, job3],
            ...     get_dataset_func=get_dataset,
            ...     build_model_func=build_model,
            ...     execute_training_func=train_model
            ... )
            >>> stats['avg_accuracy']
            0.945
        """
        if n_workers is None:
            n_workers = ModelChunker.auto_chunk_count()
        
        n_workers = min(n_workers, len(jobs))
        
        logger.info(
            f"Starting parallel training: {len(jobs)} jobs → "
            f"{n_workers} workers"
        )
        
        # Prepare worker arguments
        worker_args = [
            (
                job.model_id,
                job.dataset_id,
                job.model_config,
                job.hyperparams,
                job.session_id,
                get_dataset_func,
                build_model_func,
                execute_training_func
            )
            for job in jobs
        ]
        
        results = []
        errors = []
        start_time = datetime.now()
        
        try:
            with Pool(n_workers) as pool:
                # Execute training jobs in parallel
                imap_results = pool.starmap(
                    _train_single_model_worker,
                    worker_args,
                    chunksize=1
                )
                
                for idx, result in enumerate(imap_results):
                    if isinstance(result, Exception):
                        error_msg = f"Job {idx} failed: {str(result)}"
                        logger.error(error_msg)
                        errors.append((idx, result))
                    else:
                        results.append(result)
                        status = "✓" if result.success else "✗"
                        logger.info(
                            f"{status} {result.model_id}: "
                            f"acc={result.accuracy:.4f if result.accuracy else None} "
                            f"time={result.training_time_seconds:.2f}s"
                        )
        
        except Exception as e:
            logger.error(f"Parallel execution failed: {e}")
            raise
        
        finally:
            elapsed = (datetime.now() - start_time).total_seconds()
        
        # Aggregate statistics
        stats = TrainingExecutor._aggregate_stats(results, elapsed, errors)
        
        logger.info(
            f"Parallel training complete: {len(results)} succeeded, "
            f"{len(errors)} failed, {elapsed:.2f}s total"
        )
        
        return results, stats
    
    @staticmethod
    def _aggregate_stats(
        results: List[TrainingResult],
        total_time: float,
        errors: List[Tuple[int, Exception]]
    ) -> Dict[str, Any]:
        """
        Aggregate statistics from all training results.
        
        Returns:
            Dict with summary statistics
        """
        successful = [r for r in results if r.success]
        
        if len(successful) == 0:
            return {
                "total_jobs": len(results),
                "successful": 0,
                "failed": len(results),
                "total_time": total_time,
                "errors": len(errors)
            }
        
        accuracies = [r.accuracy for r in successful if r.accuracy is not None]
        times = [r.training_time_seconds for r in successful]
        
        best_model = max(successful, key=lambda r: r.accuracy if r.accuracy else 0)
        
        return {
            "total_jobs": len(results),
            "successful": len(successful),
            "failed": len(results) - len(successful),
            "total_time_seconds": total_time,
            "avg_time_per_model": float(np.mean(times)) if times else 0,
            "avg_accuracy": float(np.mean(accuracies)) if accuracies else 0,
            "best_accuracy": float(best_model.accuracy) if best_model.accuracy else 0,
            "best_model_id": best_model.model_id,
            "worst_accuracy": float(np.min(accuracies)) if accuracies else 0,
            "errors": len(errors),
            "speedup_factor": f"~{len(results) / (total_time / max(times, default=1)):.1f}x" if times else "N/A"
        }


# ============================================================================
# Public API
# ============================================================================

async def train_models_parallel(
    session_id: str,
    training_jobs: List[Dict[str, Any]],
    get_dataset_func: Callable,
    build_model_func: Callable,
    execute_training_func: Callable,
    n_workers: Optional[int] = None,
    progress_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Public API for submitting parallel training jobs.
    
    Converts input dicts to TrainingJob objects, executes training in parallel,
    and returns aggregated results.
    
    Args:
        session_id: Session context
        training_jobs: List of dicts with keys:
            - model_id: str
            - dataset_id: str
            - model_config: dict (architecture config)
            - hyperparams: dict (training hyperparams)
        get_dataset_func: Function to fetch dataset
        build_model_func: Function to build model
        execute_training_func: Function to run training
        n_workers: Number of parallel workers
        progress_callback: Optional callback for progress updates (async)
    
    Returns:
        Dict with keys:
        - success: bool
        - results: List of training result dicts
        - stats: Aggregated statistics
        - job_ids: List of job IDs
    
    Example:
        >>> jobs = [
        ...     {"model_id": "lstm_001", "dataset_id": "ml_001", ...},
        ...     {"model_id": "gru_002", "dataset_id": "ml_002", ...},
        ...     {"model_id": "dense_003", "dataset_id": "ml_003", ...}
        ... ]
        >>> result = await train_models_parallel(
        ...     session_id="sess_abc",
        ...     training_jobs=jobs,
        ...     get_dataset_func=get_dataset,
        ...     build_model_func=build_model,
        ...     execute_training_func=train_model
        ... )
        >>> result['stats']['avg_accuracy']
        0.945
    """
    logger.info(f"Processing {len(training_jobs)} training jobs for session {session_id}")
    
    # Convert dicts to TrainingJob objects
    jobs = [
        TrainingJob(
            model_id=job["model_id"],
            dataset_id=job["dataset_id"],
            model_config=job.get("model_config", {}),
            hyperparams=job.get("hyperparams", {}),
            session_id=session_id
        )
        for job in training_jobs
    ]
    
    # Execute training in parallel (run in thread pool to not block event loop)
    loop = asyncio.get_event_loop()
    results, stats = await loop.run_in_executor(
        None,
        TrainingExecutor.execute_parallel,
        jobs,
        get_dataset_func,
        build_model_func,
        execute_training_func,
        n_workers,
        3600  # timeout
    )
    
    # Convert results to dicts for JSON serialization
    result_dicts = [r.to_dict() for r in results]
    
    # Call progress callback if provided
    if progress_callback:
        await progress_callback({
            "status": "complete",
            "completed": len(results),
            "total": len(jobs)
        })
    
    return {
        "success": True,
        "results": result_dicts,
        "stats": stats,
        "job_ids": [job.model_id for job in jobs]
    }
