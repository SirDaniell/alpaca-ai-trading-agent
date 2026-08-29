import uuid
import asyncio
import logging
import gc
from enum import Enum
from dataclasses import dataclass, field
import time
from typing import Dict, Any, Optional, List


# --- Unified Task Store Implementation ---
class TaskType(str, Enum):
    TECHNICAL_ANALYSIS = "technical_analysis"
    SIGNAL_GENERATION = "signal_generation"
    ASTRONOMICAL_ANALYSIS = "astronomical_analysis"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    ML_DATASET_PREPARATION = "ml_dataset_preparation"
    ML_PREPARATION = "ml_preparation"  # Alias for consistency with frontend
    TIMEFRAME_ML_ANALYSIS = "timeframe_ml_analysis"
    MODEL_BUILD = "model_build"
    MODEL_TRAIN = "model_train"


@dataclass
class TaskMetadata:
    task_id: str
    task_type: TaskType
    status: str
    progress: float
    message: str
    user_id: str = "unknown"  # User who created the task - CRITICAL for multi-user isolation
    result: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    # Extra fields for progress tracking
    current_index: int = 0
    total: int = 0
    signals_found: int = 0
    message2: str = ""
    current_levels: List[Any] = field(default_factory=list)
    current_zones: List[Any] = field(default_factory=list)
    snr_feats: Dict[str, Any] = field(default_factory=dict)
    signal_counts: Dict[str, int] = field(default_factory=dict)
    processing_stage: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskCancelledException(Exception):
    """Exception raised when a task is cancelled and should stop processing"""
    pass


class TaskStore:
    """Centralized task store with type safety"""

    def __init__(self):
        self._store: Dict[str, TaskMetadata] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Thread safety for access from threadpool workers
        import threading
        self._lock = threading.RLock()
        
        # Start automatic cleanup
        self._start_cleanup_scheduler()

    @property
    def tasks(self) -> Dict[str, TaskMetadata]:
        """Backward compatibility for direct task access"""
        return self._store
    
    def _start_cleanup_scheduler(self):
        """Start automatic cleanup scheduler."""
        try:
            loop = asyncio.get_running_loop()
            self._cleanup_task = loop.create_task(self._periodic_cleanup())
        except RuntimeError:
            # No running loop (e.g. during initialization or tests)
            pass
    
    async def _periodic_cleanup(self):
        """Periodically clean up old tasks."""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await self._cleanup_old_tasks()
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Periodic cleanup error: {e}")
    
    async def _cleanup_old_tasks(self):
        """Clean up tasks older than thresholds."""
        current_time = time.time()
        tasks_to_clean = []
        
        with self._lock:
            for task_id, task in self._store.items():
                age_minutes = (current_time - task.created_at) / 60
                
                # Different thresholds based on status
                should_clean = False
                if task.status in ["completed", "error", "cancelled"]:
                    if age_minutes > 30:  # 30 minutes for completed tasks
                        should_clean = True
                elif task.status == "completed_cleaned":
                    if age_minutes > 120:  # 2 hours for cleaned tasks (metadata only)
                        should_clean = True
                elif age_minutes > 60:  # 1 hour for any other status (likely stuck)
                    should_clean = True
                
                if should_clean:
                    tasks_to_clean.append(task_id)
        
        # Clean up identified tasks (delete_task uses lock, so we call it outside or inside? 
        # RLock allows re-entry, so it's safe to call inside or outside, 
        # BUT iterating over keys while deleting is bad. We collected keys, so it's fine.)
        for task_id in tasks_to_clean:
            self.delete_task(task_id)
            import logging
            logging.getLogger(__name__).info(f"Auto-cleaned task {task_id}")
        
        if tasks_to_clean:
            gc.collect()  # Force garbage collection after cleanup

    def create_task(self, task_type: TaskType, user_id: str = "unknown", metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new task and return its ID"""
        task_id = f"{task_type.value[:4]}_{uuid.uuid4()}"
        with self._lock:
            self._store[task_id] = TaskMetadata(
                task_id=task_id,
                task_type=task_type,
                status="initializing",
                progress=0.0,
                message=f"Initializing {task_type.value}...",
                user_id=user_id or "unknown",  # Store user_id for isolation
                metadata=metadata or {}
            )
        return task_id

    def register_task(self, task: TaskMetadata):
        """Manually register a task (useful for chunk tasks with predefined IDs)"""
        with self._lock:
            self._store[task.task_id] = task

    def register_external_task(self, task_id: str, user_id: str = "unknown") -> None:
        """
        Register a task with a frontend-provided task_id.
        
        Used by the unified analysis_manager endpoint where the frontend
        generates the task_id. Without this, update_task() calls are silently
        dropped because the task_id was never registered via create_task().
        
        Args:
            task_id: Frontend-generated task UUID
            user_id: User who owns this task
        """
        with self._lock:
            if task_id not in self._store:
                self._store[task_id] = TaskMetadata(
                    task_id=task_id,
                    task_type=TaskType.TECHNICAL_ANALYSIS,  # Generic type for unified endpoint
                    status="initializing",
                    progress=0.0,
                    message="Initializing...",
                    user_id=user_id or "unknown",
                )


    def update_task(self, task_id: str, **kwargs):
        """Update task metadata"""
        with self._lock:
            if task_id in self._store:
                for key, value in kwargs.items():
                    if hasattr(self._store[task_id], key):
                        setattr(self._store[task_id], key, value)
                self._store[task_id].updated_at = time.time()

    # Backward-compatible helper used across older route codepaths.
    def update_task_progress(self, task_id: str, **kwargs):
        self.update_task(task_id, **kwargs)

    # Backward-compatible helper used across older route codepaths.
    def complete_task(self, task_id: str, result: Optional[Dict[str, Any]] = None):
        payload: Dict[str, Any] = {"status": "completed", "progress": 100}
        if result is not None:
            payload["result"] = result
        self.update_task(task_id, **payload)

    # Backward-compatible helper used across older route codepaths.
    def fail_task(self, task_id: str, error_message: str):
        self.update_task(task_id, status="error", message=error_message, progress=0)

    def get_task(self, task_id: str) -> Optional[TaskMetadata]:
        """Get task by ID"""
        with self._lock:
            # Return a copy or the object? 
            # Returning the object allows modification without lock, which defeats the purpose.
            # However, for simple reads, it might be okay. 
            # For strict safety, we should return a snapshot or ensure callers lock.
            # Given the usage, we'll return the object but rely on atomic updates via update_task.
            return self._store.get(task_id)

    def delete_task(self, task_id: str):
        """Delete task from store"""
        with self._lock:
            if task_id in self._store:
                self._store.pop(task_id, None)

    def to_dict(self, task_id: str, include_result: bool = True) -> Dict[str, Any]:
        """
        Convert task to dictionary for API response.
        
        ⚠️ CRITICAL: For large-result tasks, set include_result=False to prevent
        serialization overhead when polling status. Large results are persisted in
        DB and should be retrieved via dedicated endpoints.
        """
        task = self.get_task(task_id)
        if not task:
            return {}

        result_to_return = task.result if include_result else None
        
        return {
            "task_id": task.task_id,
            "task_type": task.task_type.value,
            "status": task.status,
            "progress": task.progress,
            "message": task.message,
            "message2": task.message2,
            "user_id": task.user_id,  # Include user_id in response
            "result": result_to_return,
            "created_at": task.created_at,
            "updated_at": task.updated_at,
            "current_index": task.current_index,
            "total": task.total,
            "signals_found": task.signals_found,
            "current_levels": task.current_levels,
            "current_zones": task.current_zones,
            "snr_feats": task.snr_feats,
            "signal_counts": task.signal_counts,
            "processing_stage": task.processing_stage,
            "metadata": task.metadata,
        }

    def is_cancelled(self, task_id: str) -> bool:
        """Check if task has been cancelled"""
        task = self.get_task(task_id)
        return bool(task and task.status == "cancelled")

    def check_cancellation(self, task_id: str):
        """Check if task is cancelled and raise exception if it is"""
        if self.is_cancelled(task_id):
            raise TaskCancelledException(f"Task {task_id} was cancelled by user")

    def get_active_tasks(self) -> Dict[str, Any]:
        """Get all active tasks (not completed or error)"""
        return {
            tid: self.to_dict(tid)
            for tid, t in self._store.items()
            if t.status not in ["completed", "error", "cancelled"]
        }



# Proxy classes for multi-process progress reporting
class QueueProgressStoreProxy:
    """
    A lightweight proxy that mimics TaskStore for worker processes.
    Instead of updating local memory, it sends updates to a multiprocessing Queue.
    """
    def __init__(self, progress_queue: Any):
        self._queue = progress_queue

    def update_task(self, task_id: str, **kwargs):
        """Pass update to the queue for the main process to handle"""
        try:
            # Standardize on 'update' action to match ProcessingManager listener
            self._queue.put({
                "action": "update",
                "task_id": task_id,
                "data": kwargs
            })
        except Exception as e:
            # We don't want to crash the worker if queue failed
            import logging
            logging.getLogger(__name__).warning(f"Failed to put progress in queue: {e}")

    def get_task(self, task_id: str):
        """
        Workers shouldn't rely on getting task state from other workers.
        We return a dummy object that supports min/max to avoid crashes.
        """
        @dataclass
        class DummyTask:
            progress: float = 0.0
            status: str = "processing"
        return DummyTask()

    def check_cancellation(self, task_id: str):
        """
        Checking cancellation via queue is too slow/complex for workers.
        We expect the main process to terminate the pool if cancelled.
        """
        pass

    def is_cancelled(self, task_id: str) -> bool:
        return False


class BroadcastingTaskStoreProxy:
    """
    DEPRECATED: This proxy previously intercepted TaskStore updates to broadcast Websocket messages.
    It has been disabled to fix a severe 'Double-Broadcasting' bug.
    
    All analysis steps should now explicitly instantiate a ProgressReporter(..., throttling_strategy=ThrottlingStrategy.HYBRID)
    which cleanly handles its own throttling and targeted Websocket broadcasts.
    """
    def __init__(self, target_store: Any, connection_manager: Any, loop: Optional[asyncio.AbstractEventLoop] = None):
        self._target = target_store
        
    def update_task(self, task_id: str, **kwargs):
        """Pass update to local storage only. Broadcasting acts strictly locally now."""
        self._target.update_task(task_id, **kwargs)
        
    def get_task(self, task_id: str):
        return self._target.get_task(task_id)

    def is_cancelled(self, task_id: str) -> bool:
        return self._target.is_cancelled(task_id)

    def check_cancellation(self, task_id: str):
        self._target.check_cancellation(task_id)


# Global TaskStore instance (Singleton)
task_store = TaskStore()

