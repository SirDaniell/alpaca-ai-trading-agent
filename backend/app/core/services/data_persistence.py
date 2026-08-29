"""
Data persistence layer: manage RAM ↔ Disk transitions
"""

import os
import json
import joblib
import pandas as pd
import logging
from typing import Dict, Optional, Tuple, Any
import time

logger = logging.getLogger(__name__)

class TaskDataPersistence:
    """
    Manages data lifecycle for a single task:
    - RAM cache for active data (<200MB)
    - Joblib disk storage for large data (>200MB)
    - Smart loading: reload from disk when needed
    """

    # Class-level store of all task instances
    _instances: Dict[str, 'TaskDataPersistence'] = {}

    @classmethod
    def get_instance(cls, task_id: str, max_ram_mb: int = 200) -> 'TaskDataPersistence':
        """Get or create persistence instance for task"""
        if task_id not in cls._instances:
            cls._instances[task_id] = cls(task_id, max_ram_mb)
        return cls._instances[task_id]

    @classmethod
    def cleanup_old_tasks(cls, older_than_seconds: int = 3600):
        """Remove old task data to save disk"""
        now = time.time()
        to_delete = []

        for task_id, instance in list(cls._instances.items()):
            age = now - instance.created_at
            if age > older_than_seconds:
                to_delete.append(task_id)
                instance.cleanup_disk()

        for task_id in to_delete:
            del cls._instances[task_id]

    def __init__(self, task_id: str, max_ram_mb: int = 200):
        self.task_id = task_id
        self.max_ram_bytes = max_ram_mb * 1_000_000
        self.created_at = time.time()

        # RAM cache: {key: (dataframe, timestamp)}
        self.ram_cache: Dict[str, Tuple[pd.DataFrame, float]] = {}

        # Disk index: {key: filepath}
        self.disk_index: Dict[str, str] = {}

        # Create task-specific disk directory
        self.disk_dir = f"/tmp/analysis/{self.task_id}"
        os.makedirs(self.disk_dir, exist_ok=True)

        logger.info(f"DataPersistence: Created for task {task_id}, max_ram={max_ram_mb}MB")

    def estimate_memory_usage(self) -> int:
        """Estimate current RAM usage in bytes"""
        total = 0
        for df, _ in self.ram_cache.values():
            if isinstance(df, pd.DataFrame):
                total += df.memory_usage(deep=True).sum()
        return total

    # ========================================================================
    # FIX #3.1: BACKPRESSURE CONTROL - Prevent OOM in large dataset processing
    # ========================================================================
    
    def check_backpressure(self, target_df_size: int, timeout_seconds: float = 60.0) -> bool:
        """
        Check if we should apply backpressure (wait before accepting new chunks).
        
        If memory usage is >80% of max, block further chunks until it drops to <50%.
        This prevents OOM when processing >1M rows with limited cores.
        
        Args:
            target_df_size: Size in bytes of chunk about to be processed
            timeout_seconds: How long to wait for memory to free up
            
        Returns:
            True if backpressure applied (had to wait)
            
        Raises:
            MemoryError: If timeout exceeded (processing stalled)
        """
        current_ram = self.estimate_memory_usage()
        backpressure_threshold = int(self.max_ram_bytes * 0.80)  # 80% full
        recovery_threshold = int(self.max_ram_bytes * 0.50)  # 50% target
        
        if current_ram > backpressure_threshold:
            logger.warning(
                f"\u26a0\ufe0f Backpressure: RAM {current_ram/1_000_000:.1f}MB / "
                f"{self.max_ram_bytes/1_000_000:.1f}MB ({current_ram/self.max_ram_bytes*100:.0f}%) - "
                f"waiting for processing to catch up..."
            )
            
            # Wait for memory to drop below recovery threshold
            waited = 0.0
            check_interval = 0.5  # seconds
            
            while waited < timeout_seconds:
                current_ram = self.estimate_memory_usage()
                if current_ram < recovery_threshold:
                    logger.info(
                        f"\u2705 Backpressure recovered: RAM now "
                        f"{current_ram/1_000_000:.1f}MB"
                    )
                    return True
                
                time.sleep(check_interval)
                waited += check_interval
            
            # Timeout - processing might be stuck
            raise MemoryError(
                f"Backpressure timeout: Memory didn't drop below "
                f"{recovery_threshold/1_000_000:.0f}MB within {timeout_seconds}s. "
                f"Processing may be stalled."
            )
        
        return False

    def save(
        self,
        key: str,
        df: pd.DataFrame,
        force_disk: bool = False
    ) -> Dict[str, str]:
        """
        Save dataframe (RAM first, spill to disk if needed)

        Returns: {"location": "ram" | "disk", "key": key}
        """
        if not isinstance(df, pd.DataFrame):
            # If not a dataframe, handle as generic data in RAM for now
            self.ram_cache[key] = (df, time.time())
            return {"location": "ram", "key": key}

        df_size = df.memory_usage(deep=True).sum()
        current_ram = self.estimate_memory_usage()

        # Decision: RAM or disk?
        if force_disk or (current_ram + df_size > self.max_ram_bytes):
            # Save to disk
            return self._save_to_disk(key, df)
        else:
            # Save to RAM
            self.ram_cache[key] = (df, time.time())
            logger.info(
                f"Persistence: Saved {key} to RAM ({df_size/1_000_000:.1f}MB, "
                f"total RAM: {(current_ram + df_size)/1_000_000:.1f}MB)"
            )
            return {"location": "ram", "key": key}

    def _save_to_disk(self, key: str, df: pd.DataFrame) -> Dict[str, str]:
        """Save to joblib file"""
        filepath = os.path.join(self.disk_dir, f"{key}.joblib")

        try:
            joblib.dump(df, filepath, compress=3)  # compress 0-9
            self.disk_index[key] = filepath
            
            # Remove from RAM cache if exists
            if key in self.ram_cache:
                del self.ram_cache[key]

            df_size = df.memory_usage(deep=True).sum()
            logger.info(
                f"Persistence: Saved {key} to disk ({df_size/1_000_000:.1f}MB) "
                f"at {filepath}"
            )

            return {"location": "disk", "key": key}
        except Exception as e:
            logger.error(f"Failed to save {key} to disk: {e}")
            raise

    def load(self, key: str) -> Any:
        """Load from RAM cache or disk"""
        # Try RAM first
        if key in self.ram_cache:
            df, timestamp = self.ram_cache[key]
            return df

        # Try disk
        if key in self.disk_index:
            filepath = self.disk_index[key]
            try:
                df = joblib.load(filepath)
                
                # Check if we should move it back to RAM
                df_size = df.memory_usage(deep=True).sum() if isinstance(df, pd.DataFrame) else 0
                if self.estimate_memory_usage() + df_size <= self.max_ram_bytes:
                    self.ram_cache[key] = (df, time.time())
                
                logger.info(f"Persistence: Loaded {key} from disk")
                return df
            except Exception as e:
                logger.error(f"Failed to load {key} from disk: {e}")
                raise

        # Not found
        raise KeyError(f"Cannot find {key} in RAM or disk")

    def get_data(self, key: str) -> Any:
        """Unified access: RAM cache first, then disk"""
        return self.load(key)

    def exists(self, key: str) -> bool:
        """Check if data exists"""
        return key in self.ram_cache or key in self.disk_index

    def delete(self, key: str) -> None:
        """Remove from memory (and optionally disk)"""
        if key in self.ram_cache:
            del self.ram_cache[key]

        if key in self.disk_index:
            filepath = self.disk_index[key]
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                del self.disk_index[key]
                logger.info(f"Persistence: Deleted {key}")
            except Exception as e:
                logger.error(f"Failed to delete {key}: {e}")

    def list_keys(self) -> list:
        """List all stored keys"""
        return list(set(self.ram_cache.keys()) | list(self.disk_index.keys()))

    def get_status(self) -> Dict:
        """Status snapshot"""
        return {
            "task_id": self.task_id,
            "ram_usage": {
                "estimate_mb": self.estimate_memory_usage() / 1_000_000,
                "max_mb": self.max_ram_bytes / 1_000_000
            },
            "stored_keys": {
                "ram": list(self.ram_cache.keys()),
                "disk": list(self.disk_index.keys())
            },
            "disk_path": self.disk_dir
        }

    def cleanup_disk(self) -> None:
        """Delete all disk files for this task"""
        import shutil
        try:
            if os.path.exists(self.disk_dir):
                shutil.rmtree(self.disk_dir)
                logger.info(f"Persistence: Cleaned up disk for task {self.task_id}")
        except Exception as e:
            logger.error(f"Failed to cleanup disk: {e}")
