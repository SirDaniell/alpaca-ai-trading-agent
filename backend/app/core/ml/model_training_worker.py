"""
Model Training Worker and Multiprocessing Components

Implements the SNR-style multiprocessing pattern for efficient model training.
Uses OOP principles with proper resource cleanup.

🔒 CONCURRENCY: Uses threading.RLock to protect shared resources from parallel processes
   (Keras/TensorFlow + multiprocessing can have resource contention issues)
"""

# 🔒 TensorFlow Configuration: Limit threading to reduce resource contention
# Must be set BEFORE TensorFlow imports
import os
os.environ['TF_CPP_THREAD_POOL_SIZE'] = '1'  # Limit inter-op threads
os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads
os.environ['MKL_NUM_THREADS'] = '1'  # Limit MKL threads
os.environ['NUMEXPR_NUM_THREADS'] = '1'  # Limit NumExpr threads
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Prevent GPU memory hogging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging noise

import logging
import asyncio
import gc
import numpy as np
import threading
from typing import Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from multiprocessing import Queue, Lock as ProcessLock
from tensorflow.keras.callbacks import Callback
import multiprocessing

logger = logging.getLogger(__name__)

# 🔒 GLOBAL LOCK: Protects critical sections across parallel processes
_global_training_lock = threading.RLock()


@dataclass
class TrainingMetrics:
    """Container for training metrics (memory efficient)."""
    epoch: int
    loss: float
    val_loss: Optional[float] = None
    mae: Optional[float] = None
    val_mae: Optional[float] = None
    mse: Optional[float] = None
    val_mse: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for progress updates."""
        return {
            k: v for k, v in vars(self).items() 
            if v is not None
        }
    
    def __del__(self):
        """Explicit cleanup."""
        pass


class WorkerProgressCallback(Callback):
    """
    Callback that sends progress updates to multiprocessing queue.
    
    Designed to run in worker process, writes to queue instead of
    using asyncio (which is not safe from worker processes).
    """
    
    def __init__(self, task_id: str, total_epochs: int, progress_queue: Queue):
        """
        Args:
            task_id: Unique identifier for this training task
            total_epochs: Total number of epochs to train
            progress_queue: multiprocessing.Queue for sending updates to main process
        """
        super().__init__()
        self.task_id = task_id
        self.total_epochs = total_epochs
        self.progress_queue = progress_queue
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Called after each epoch."""
        logs = logs or {}
        
        try:
            progress = int((epoch + 1) / self.total_epochs * 100)
            message = f"Epoch {epoch + 1}/{self.total_epochs}"
            
            # Extract metrics safely
            metrics_str = self._extract_metrics_string(logs)
            
            # Extract all loss metrics for frontend unpacking (like SNR pattern)
            loss_metrics = {
                "loss": float(logs.get("loss", 0)),
                "mae": float(logs.get("mae", logs.get("mean_absolute_error", 0))),
                "mse": float(logs.get("mse", logs.get("mean_squared_error", 0))),
                "val_loss": float(logs.get("val_loss", 0)),
                "val_mae": float(logs.get("val_mae", logs.get("val_mean_absolute_error", 0))),
                "val_mse": float(logs.get("val_mse", logs.get("val_mean_squared_error", 0)))
            }
            
            # 🔒 LOCK: Protect queue write from resource contention
            # Multiple processes can clash when writing to queue + stdout (tqdm)
            with _global_training_lock:
                # Send progress update via queue (thread/process safe)
                self.progress_queue.put({
                    "action": "update",
                    "task_id": self.task_id,
                    "data": {
                        "progress": progress,
                        "message": message,
                        "message2": metrics_str,
                        "current_epoch": epoch + 1,
                        "total_epochs": self.total_epochs,
                        # 🎯 Unpackable loss metrics for frontend display (like SNRMetrics)
                        "loss": loss_metrics["loss"],
                        "mae": loss_metrics["mae"],
                        "mse": loss_metrics["mse"],
                        "val_loss": loss_metrics["val_loss"],
                        "val_mae": loss_metrics["val_mae"],
                        "val_mse": loss_metrics["val_mse"],
                        "training_metrics": loss_metrics
                    }
                })
        except Exception as e:
            logger.error(f"[{self.task_id}] Error in WorkerProgressCallback: {e}")
    
    @staticmethod
    def _extract_metrics_string(logs: Dict) -> str:
        """Extract human-readable metrics string from logs."""
        metrics = []
        
        if "loss" in logs:
            metrics.append(f"Loss: {logs['loss']:.4f}")
        
        if "mae" in logs:
            metrics.append(f"MAE: {logs['mae']:.4f}")
        elif "mean_absolute_error" in logs:
            metrics.append(f"MAE: {logs['mean_absolute_error']:.4f}")
        
        if "val_loss" in logs:
            metrics.append(f"Val Loss: {logs['val_loss']:.4f}")
        
        if "val_mae" in logs:
            metrics.append(f"Val MAE: {logs['val_mae']:.4f}")
        elif "val_mean_absolute_error" in logs:
            metrics.append(f"Val MAE: {logs['val_mean_absolute_error']:.4f}")
        
        return " | ".join(metrics)


class TrainingWorker:
    """
    Worker that trains a model in a separate process.
    
    Uses OOP to encapsulate training logic and ensure proper cleanup.
    Communicates with main process via multiprocessing.Queue.
    """
    
    def __init__(self, task_id: str, progress_queue: Queue):
        """
        Args:
            task_id: Unique identifier for this training task
            progress_queue: multiprocessing.Queue for sending progress updates
        """
        self.task_id = task_id
        self.progress_queue = progress_queue
        self.logger = logging.getLogger(__name__)
    
    def train_model(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int,
        batch_size: int,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        validation_split: float = 0.2
    ) -> Dict[str, list]:
        """
        Train a model using the worker process.
        
        Args:
            model: TensorFlow model to train
            X_train: Training data (samples, timesteps, features)
            y_train: Training labels (targets)
            epochs: Number of epochs to train
            batch_size: Batch size for training
            validation_data: Optional (X_val, y_val) tuple
            validation_split: Fraction for validation if not explicit data
        
        Returns:
            Dict containing 'loss', 'val_loss', 'mae', 'val_mae', etc.
        """
        try:
            self.logger.info(f"[{self.task_id}] Worker starting training: "
                           f"X_train shape={X_train.shape}, epochs={epochs}, batch_size={batch_size}")
            
            # Create progress callback for this worker
            callback = WorkerProgressCallback(self.task_id, epochs, self.progress_queue)
            
            # 🔒 LOCK: Protect Keras/TensorFlow training from parallel process resource contention
            # Multiple processes training simultaneously can clash with CPU/GPU/IO operations
            with _global_training_lock:
                # Train model
                history = model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=validation_data,
                    validation_split=validation_split if validation_data is None else 0.0,
                    callbacks=[callback],
                    verbose=0  # Suppress default output
                )
            
            # Extract history (outside lock - post-processing)
            history_dict = {
                "loss": [float(x) for x in history.history.get("loss", [])],
                "mae": [float(x) for x in history.history.get("mae", history.history.get("mean_absolute_error", []))],
                "mse": [float(x) for x in history.history.get("mse", history.history.get("mean_squared_error", []))],
                "val_loss": [float(x) for x in history.history.get("val_loss", [])],
                "val_mae": [float(x) for x in history.history.get("val_mae", history.history.get("val_mean_absolute_error", []))],
                "val_mse": [float(x) for x in history.history.get("val_mse", history.history.get("val_mean_squared_error", []))],
            }
            
            self.logger.info(f"[{self.task_id}] Worker training completed successfully")
            return history_dict
            
        except Exception as e:
            self.logger.error(f"[{self.task_id}] Worker training failed: {e}", exc_info=True)
            # Send error via queue
            self.progress_queue.put({
                "action": "error",
                "task_id": self.task_id,
                "data": {
                    "error": str(e),
                    "status": "error"
                }
            })
            raise
        finally:
            # Cleanup
            del model
            gc.collect()
    
    def __del__(self):
        """Ensure queue is closed on cleanup."""
        try:
            if hasattr(self, 'progress_queue') and self.progress_queue:
                self.progress_queue.close()
        except:
            pass


class ProgressQueueDrainer:
    """
    Drains a multiprocessing.Queue and updates central task_store.
    
    Runs as an async task concurrent with worker processes.
    Updates task_store atomically to avoid race conditions.
    """
    
    def __init__(self, task_id: str, progress_queue: Queue, task_store: Any):
        """
        Args:
            task_id: Main task identifier
            progress_queue: multiprocessing.Queue to drain
            task_store: Central task store for updates
        """
        self.task_id = task_id
        self.progress_queue = progress_queue
        self.task_store = task_store
        self.logger = logging.getLogger(__name__)
        self.items_drained = 0
    
    async def drain_queue(self) -> int:
        """
        Continuously drain the queue and update task_store.
        
        Returns:
            Number of items drained
        """
        items_drained = 0
        
        while True:
            try:
                # Non-blocking get
                while not self.progress_queue.empty():
                    try:
                        item = self.progress_queue.get_nowait()
                        items_drained += 1
                        
                        if item.get("action") == "update":
                            tid = item.get("task_id")
                            data = item.get("data", {})
                            self.task_store.update_task(tid, **data)
                        
                        elif item.get("action") == "error":
                            tid = item.get("task_id")
                            data = item.get("data", {})
                            self.task_store.update_task(tid, **data)
                    
                    except Exception:
                        break
                
                # Check if training is done
                task = self.task_store.get_task(self.task_id)
                if task and task.status in ["completed", "error", "cancelled"]:
                    break
                
                # Sleep briefly to avoid busy-wait
                await asyncio.sleep(0.05)
                
            except Exception as e:
                self.logger.debug(f"Queue drain error: {e}")
                await asyncio.sleep(0.1)
        
        self.logger.info(f"[{self.task_id}] Queue drainer completed: {items_drained} items drained")
        return items_drained
    
    def __del__(self):
        """Cleanup."""
        self.items_drained = 0


class ProgressMonitor:
    """
    Monitors training progress and sends WebSocket updates.
    
    Runs as an async task concurrent with worker processes.
    Aggregates progress from task_store and sends updates to frontend.
    """
    
    def __init__(self, task_id: str, task_store: Any, send_update_func: Callable, user_id: str):
        """
        Args:
            task_id: Main task identifier
            task_store: Central task store to monitor
            send_update_func: Async function to send WebSocket updates
            user_id: User identifier for WebSocket routing
        """
        self.task_id = task_id
        self.task_store = task_store
        self.send_update_func = send_update_func
        self.user_id = user_id
        self.logger = logging.getLogger(__name__)
        self.updates_sent = 0
    
    async def monitor_progress(self) -> int:
        """
        Monitor and send progress updates.
        
        Returns:
            Number of updates sent
        """
        last_progress = -1
        
        while True:
            try:
                task = self.task_store.get_task(self.task_id)
                
                if not task or task.status in ["completed", "error", "cancelled"]:
                    break
                
                # Only send update if progress changed
                if task.progress != last_progress:
                    await self.send_update_func(
                        self.task_id,
                        self.task_store.to_dict(self.task_id),
                        self.user_id
                    )
                    self.updates_sent += 1
                    last_progress = task.progress
                
                # Check frequency: 300ms (vs SNR pattern)
                await asyncio.sleep(0.3)
                
            except asyncio.CancelledError:
                self.logger.debug(f"[{self.task_id}] Monitor cancelled")
                break
            except Exception as e:
                self.logger.debug(f"[{self.task_id}] Monitor error: {e}")
                await asyncio.sleep(0.5)
        
        self.logger.info(f"[{self.task_id}] Progress monitor completed: {self.updates_sent} updates sent")
        return self.updates_sent
    
    def __del__(self):
        """Cleanup."""
        self.updates_sent = 0


class TrainingOrchestrator:
    """
    Orchestrates the entire training process using multiprocessing.
    
    Manages:
    - Worker process pool
    - Progress queue
    - Async monitoring tasks
    - Resource cleanup
    
    Implements SNR-style parallelism for fast, responsive training.
    """
    
    def __init__(self, task_id: str, task_store: Any, send_update_func: Callable, user_id: str):
        """
        Args:
            task_id: Unique training task identifier
            task_store: Central task store for progress
            send_update_func: Async function to send WebSocket updates
            user_id: User identifier for routing
        """
        self.task_id = task_id
        self.task_store = task_store
        self.send_update_func = send_update_func
        self.user_id = user_id
        self.logger = logging.getLogger(__name__)
        
        self.progress_queue: Optional[Queue] = None
        self.manager: Optional[Any] = None
        self.drain_task: Optional[asyncio.Task] = None
        self.monitor_task: Optional[asyncio.Task] = None
        self.training_complete = asyncio.Event()
    
    async def train_traditional(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int,
        batch_size: int,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        validation_split: float = 0.2
    ) -> Dict[str, list]:
        """
        Train model using multiprocessing pool (SNR pattern).
        
        Args:
            model: TensorFlow model
            X_train: Training data
            y_train: Training labels
            epochs: Number of epochs
            batch_size: Batch size
            validation_data: Optional validation tuple
            validation_split: Validation split ratio
        
        Returns:
            Training history dict
        """
        try:
            self.logger.info(f"[{self.task_id}] Starting traditional training with multiprocessing")
            
            # Setup multiprocessing components
            self._setup_multiprocessing()
            
            # Create worker
            worker = TrainingWorker(self.task_id, self.progress_queue)
            
            # Start background tasks
            self.drain_task = asyncio.create_task(
                ProgressQueueDrainer(self.task_id, self.progress_queue, self.task_store).drain_queue()
            )
            self.monitor_task = asyncio.create_task(
                ProgressMonitor(self.task_id, self.task_store, self.send_update_func, self.user_id).monitor_progress()
            )
            
            # Run training in executor (non-blocking)
            loop = asyncio.get_event_loop()
            history = await loop.run_in_executor(
                None,
                worker.train_model,
                model, X_train, y_train, epochs, batch_size, 
                validation_data, validation_split
            )
            
            # Signal completion
            self.training_complete.set()
            
            # Wait for background tasks
            await asyncio.gather(self.drain_task, self.monitor_task)
            
            self.logger.info(f"[{self.task_id}] Training completed successfully")
            return history
            
        except Exception as e:
            self.logger.error(f"[{self.task_id}] Training failed: {e}", exc_info=True)
            self.training_complete.set()
            raise
        
        finally:
            self._cleanup_multiprocessing()
    
    def _setup_multiprocessing(self):
        """Initialize multiprocessing components."""
        try:
            self.manager = multiprocessing.Manager()
            self.progress_queue = self.manager.Queue()
            self.logger.info(f"[{self.task_id}] Multiprocessing components initialized")
        except Exception as e:
            self.logger.error(f"[{self.task_id}] Failed to setup multiprocessing: {e}")
            raise
    
    def _cleanup_multiprocessing(self):
        """Clean up multiprocessing resources."""
        try:
            # Cancel pending tasks
            if self.drain_task and not self.drain_task.done():
                self.drain_task.cancel()
            if self.monitor_task and not self.monitor_task.done():
                self.monitor_task.cancel()
            
            # Close progress queue
            if self.progress_queue:
                self.progress_queue.close()
                self.progress_queue = None
            
            # Shutdown manager
            if self.manager:
                self.manager.shutdown()
                self.manager = None
            
            # Force garbage collection
            gc.collect()
            
            self.logger.info(f"[{self.task_id}] Multiprocessing cleanup completed")
        except Exception as e:
            self.logger.error(f"[{self.task_id}] Error during cleanup: {e}")
    
    def __del__(self):
        """Ensure resources are cleaned up."""
        try:
            self._cleanup_multiprocessing()
        except:
            pass
