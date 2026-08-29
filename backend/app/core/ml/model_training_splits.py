"""
Split-based incremental model training for large datasets.

Enables training on 1M+ sequences with constant memory footprint.
Handles splitting, checkpoint management, and progress reporting.
"""

import os
import json
import shutil
import logging
import numpy as np
import gc
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
import random
import asyncio
from pathlib import Path
import joblib
import time

logger = logging.getLogger(__name__)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def safe_concat_arrays(arr_list: List) -> np.ndarray:
    """
    FIX #5 HELPER: Safely concatenate a list of arrays, handling mixed types.
    If list contains numpy arrays, concatenate them. If mixed or all lists, flatten.
    """
    if not arr_list:
        return np.array([])
    
    # CRITICAL FIX: Robustly determine if we can concatenate directly
    # If ALL items are numpy arrays, we can use np.concatenate for speed
    if all(isinstance(item, np.ndarray) for item in arr_list):
        try:
            return np.concatenate(arr_list, axis=0)
        except ValueError as e:
            logger.warning(f"np.concatenate failed: {e}. Falling back to list-based concatenation.")
    
    # Mixed or all lists - flatten chunks into samples
    result = []
    for item in arr_list:
        if isinstance(item, np.ndarray):
            result.extend(item.tolist())
        elif isinstance(item, list):
            # Check if this item is a chunk (list of sequences) or a single sequence
            # In our append-chunk strategy, it should always be a chunk
            result.extend(item)
        else:
            result.append(item)
            
    if not result:
        return np.array([])
        
    # Final conversion to numpy – this may still fail if sequences are inhomogeneous
    try:
        return np.array(result)
    except ValueError as e:
        logger.error(f"Failed to create numpy array from concatenated result: {e}")
        # If it's a list of dicts or something non-numeric, return as-is
        if result and isinstance(result[0], (dict, list)):
            return result
        raise

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SplitMetadata:
    """Metadata for a single training split"""
    split_id: int
    start_idx: int
    end_idx: int
    num_samples: int
    feature_count: int
    sequence_length: int
    hash_value: str = ""  # For integrity verification
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass 
class SplitConfig:
    """Configuration for split training"""
    total_splits: int = 10
    persist_to_disk: bool = True
    disk_path: str = "/tmp/training_splits"
    checkpoint_on_split: bool = True  # Save weights after each split
    shuffle_splits_per_epoch: bool = True
    seed: int = 42
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class TrainingProgress:
    """Training progress snapshot"""
    epoch: int
    split_id: int
    total_epochs: int
    total_splits: int
    loss: float
    accuracy: Optional[float] = None
    progress_percent: float = 0.0
    elapsed_time: float = 0.0
    
    @property
    def message(self) -> str:
        return f"Epoch {self.epoch+1}/{self.total_epochs}, Split {self.split_id}/{self.total_splits-1}: loss={self.loss:.4f}"

@dataclass
class TrainingState:
    """Persistent state to allow training resumption"""
    epoch: int
    split_index: int  # Index in the split_order
    split_order: List[int]
    total_epochs: int
    batch_size: int
    history: Dict[str, List[float]]
    metrics_per_split: List[Dict]
    timestamp: float

    def to_dict(self) -> Dict:
        return asdict(self)

# ============================================================================
# SPLIT MANAGER
# ============================================================================

class TrainingSplitManager:
    """
    Manages data splitting and incremental loading for model training.
    
    Responsibilities:
    1. Divide dataset into N splits
    2. Persist splits to disk
    3. Load splits on-demand
    4. Order splits for each epoch (shuffle or sequential)
    5. Free memory after each split
    6. Verify data integrity
    """
    
    def __init__(self, 
                 task_id: str,
                 config: Optional[SplitConfig] = None):
        self.task_id = task_id
        self.config = config or SplitConfig()
        self.split_metadata: Dict[int, SplitMetadata] = {}
        self.current_split_id: Optional[int] = None
        self.current_split_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.target_column: Optional[str] = None  # 🔧 FIX #6: Store target column for extraction
        
        # Ensure disk path exists
        Path(self.config.disk_path).mkdir(parents=True, exist_ok=True)
        
        # Metadata files
        self.metadata_file = os.path.join(
            self.config.disk_path,
            f"{task_id}_metadata.json"
        )
        self.config_file = os.path.join(
            self.config.disk_path,
            f"{task_id}_config.json"
        )
        
        logger.info(f"Initialized TrainingSplitManager for task {task_id}")
        logger.info(f"  Splits: {self.config.total_splits}")
        logger.info(f"  Disk path: {self.config.disk_path}")

    # ========================================================================
    # CREATION AND SETUP
    # ========================================================================

    @staticmethod
    def create_from_data(
        task_id: str,
        sequences: np.ndarray,
        targets: np.ndarray,
        num_splits: int = 10,
        persist_to_disk: bool = True,
        disk_path: str = "/tmp/training_splits",
        seed: int = 42
    ) -> 'TrainingSplitManager':
        """
        Create a split manager and divide data into splits.
        
        Args:
            task_id: Unique identifier for this training task
            sequences: Training data (n_samples, ...)
            targets: Training labels (n_samples,)
            num_splits: Number of splits to create
            persist_to_disk: Save splits to disk for memory efficiency
            disk_path: Where to store split files
            seed: Random seed for reproducibility
            
        Returns:
            TrainingSplitManager with splits created
            
        Raises:
            ValueError: If data dimensions don't match
        """
        
        # Validate inputs
        if len(sequences) != len(targets):
            raise ValueError(
                f"Sequence count ({len(sequences)}) != "
                f"target count ({len(targets)})"
            )
        
        if len(sequences) < num_splits:
            raise ValueError(
                f"Cannot create {num_splits} splits from {len(sequences)} samples"
            )
        
        # Create config
        config = SplitConfig(
            total_splits=num_splits,
            persist_to_disk=persist_to_disk,
            disk_path=disk_path,
            seed=seed
        )
        
        # Create manager
        manager = TrainingSplitManager(task_id, config)
        
        # Calculate split dimensions
        total_samples = len(sequences)
        base_split_size = total_samples // num_splits
        
        logger.info(
            f"Creating {num_splits} splits from {total_samples} samples "
            f"({base_split_size} samples per split)"
        )
        
        # Create and persist splits
        for split_id in range(num_splits):
            
            # Calculate boundaries
            start_idx = split_id * base_split_size
            if split_id == num_splits - 1:
                # Last split gets remainder
                end_idx = total_samples
            else:
                end_idx = (split_id + 1) * base_split_size
            
            # Extract split data
            split_sequences = sequences[start_idx:end_idx]
            split_targets = targets[start_idx:end_idx]
            
            # Create metadata
            metadata = SplitMetadata(
                split_id=split_id,
                start_idx=start_idx,
                end_idx=end_idx,
                num_samples=end_idx - start_idx,
                feature_count=sequences.shape[-1],
                sequence_length=sequences.shape[1] if len(sequences.shape) >= 2 else 1
            )
            
            manager.split_metadata[split_id] = metadata
            
            # Persist to disk
            if persist_to_disk:
                split_path = manager._get_split_filepath(split_id)
                
                split_data = {
                    "sequences": split_sequences,
                    "targets": split_targets,
                    "metadata": asdict(metadata)
                }
                
                joblib.dump(split_data, split_path, compress=3)
                
                logger.info(
                    f"  Split {split_id}: {end_idx - start_idx} samples "
                    f"→ {split_path}"
                )
        
        # Save configuration
        manager._save_metadata()
        manager._save_config()
        
        logger.info(f"✓ Split creation complete: {total_samples} samples in {num_splits} splits")
        
        return manager

    @staticmethod
    async def create_from_db_chunks(
        session_id: str,
        split_name: str,
        task_id: str,
        db,
        num_splits: int = 10,
        disk_path: str = "/tmp/training_splits",
        seed: int = 42,
        target_column: str = None  # 🔧 FIX #6: Target column to extract
    ) -> 'TrainingSplitManager':
        """
        PHASE 5: Create a split manager by downloading chunked data from the DB 
        and persisting it to local disk as joblib files.
        
        This enables incremental training on large datasets without ever 
        loading the full split into memory.
        
        Args:
            session_id:  Session ID to load chunks from
            split_name:  'train', 'validation', or 'test'
            task_id:     Unique identifier for this training task
            db:          Async SQLAlchemy session
            num_splits:  Total splits to create on disk
            disk_path:   Where to store split files
            seed:        Random seed
            
        Returns:
            TrainingSplitManager ready for training
        """
        from app.database.models import SessionStepResult
        from sqlalchemy import select, and_
        from app.core.data.serializers import deserialize_data
        
        logger.info(f"🚀 Creating TrainingSplitManager from DB chunks: session={session_id}, split={split_name}")
        
        # 1. Query all chunks for this split
        search_pattern = f'ml_preparation_{split_name}_%'
        stmt = select(SessionStepResult).where(
            and_(
                SessionStepResult.session_id == session_id,
                SessionStepResult.step_name.like(search_pattern)
            )
        ).order_by(SessionStepResult.step_name)
        
        result = await db.execute(stmt)
        chunk_rows = result.scalars().all()
        
        if not chunk_rows:
            raise ValueError(f"No chunks found for session {session_id}, split {split_name}")

        # Sort chunks correctly
        def get_chunk_index(name):
            try: return int(name.split('_')[-1])
            except: return 0
        chunk_rows.sort(key=lambda x: get_chunk_index(x.step_name))
        
        logger.info(f"  Found {len(chunk_rows)} database chunks. Re-chunking into {num_splits} local files.")

        # 2. Initialize manager
        config = SplitConfig(
            total_splits=num_splits,
            persist_to_disk=True,
            disk_path=disk_path,
            seed=seed
        )
        manager = TrainingSplitManager(task_id, config)
        
        # 🔧 FIX #6: Store target column for extraction during load_split
        manager.target_column = target_column
        
        # 3. Process DB chunks and redistribute into local splits
        # We process DB chunks one-by-one to keep memory footprint low
        # But we need to know the total number of samples to calculate split boundaries
        total_samples = 0
        all_chunk_sizes = []
        
        # We need a first pass to get total samples or we can rely on the metadata row
        # To be safe, let's just use the metadata row if possible, or do a light pass
        from app.core.data.session_data_loader import get_latest_session_data
        metadata = await get_latest_session_data(session_id, db, step_name='ml_preparation')
        if metadata and 'split_counts' in metadata and split_name in metadata['split_counts']:
            total_samples = metadata['split_counts'][split_name]
            logger.debug(f"  Total samples from metadata: {total_samples}")
        else:
            # Fallback (slightly slower): peek into deserialized data of all chunks for size only
            # This is still better than reassembling everything in memory
            logger.info("  Metadata missing total samples, calculating from chunks...")
            for row in chunk_rows:
                # We only need the size, but deserialize_data usually returns the full object
                # Optimization: if we had a way to get just length... but we dont.
                chunk_data = deserialize_data(row.result_data, row.is_compressed)
                sz = len(chunk_data.get('sequences', []))
                all_chunk_sizes.append(sz)
                total_samples += sz
                del chunk_data # Clear memory
        
        if total_samples == 0:
            raise ValueError(f"Total samples for {split_name} is 0")
            
        base_split_size = total_samples // num_splits
        logger.info(f"  Total samples: {total_samples}, Split size: ~{base_split_size}")

        # 4. Stream DB chunks into local files
        # Since DB chunks (500) are likely smaller than requested local splits (e.g. 5000),
        # we accumulate DB chunks into a "wait list" until we reach base_split_size, then save.
        
        current_split_id = 0
        accumulated_sequences = []
        accumulated_targets = []
        accumulated_labels = []
        accumulated_indices = []
        accumulated_meta = []
        accumulated_targets_dict = {}
        samples_in_current_split = 0
        
        feature_count = 0
        sequence_length = 0

        for i, row in enumerate(chunk_rows):
            chunk_data = deserialize_data(row.result_data, row.is_compressed)
            seqs = chunk_data.get('sequences', [])
            chunk_sz = len(seqs)
            
            if i == 0 and chunk_sz > 0:
                s_arr = np.array(seqs)
                feature_count = s_arr.shape[-1]
                sequence_length = s_arr.shape[1] if len(s_arr.shape) >= 2 else 1

            # ✅ FIX #7: ALWAYS use append() for chunk accumulation
            # Mixing append (for arrays) and extend (for lists) causes inconsistent nested depth
            # which leads to "inhomogeneous shape" errors in safe_concat_arrays.
            accumulated_sequences.append(seqs)
            
            labels = chunk_data.get('labels', [])
            accumulated_labels.append(labels)
            
            indices = chunk_data.get('indices', [])
            accumulated_indices.append(indices)
            
            meta = chunk_data.get('sequence_metadata', [])
            accumulated_meta.append(meta)
            
            targets_dict = chunk_data.get('targets', {})
            for k, v in targets_dict.items():
                if k not in accumulated_targets_dict: 
                    accumulated_targets_dict[k] = []
                accumulated_targets_dict[k].append(v)
            
            samples_in_current_split += chunk_sz
            # ✅ FREE CHUNK MEMORY IMMEDIATELY after adding to accumulators
            del chunk_data, seqs, labels, indices, meta, targets_dict
            
            # ✅ Proactively clear reference in chunk_rows to free the DB object memory
            chunk_rows[i] = None 
            
            # If we reached split size OR it's the last chunk, save to disk
            is_last_chunk = (i == len(chunk_rows) - 1)
            if samples_in_current_split >= base_split_size or is_last_chunk:
                # Handle edge case: if we have multiple splits remaining but only one chunk left,
                # we might want to be more granular. But simpler is just to dump remainder for now.
                # Actually, if we're not at the last local split yet, we might want to keep filling.
                
                # Rule: Save if we have enough for a split AND we're not exceeding num_splits
                if current_split_id < num_splits - 1:
                    # FIX #5: Concatenate accumulated data first before slicing
                    concat_sequences = safe_concat_arrays(accumulated_sequences)
                    concat_labels = safe_concat_arrays(accumulated_labels)
                    concat_indices = safe_concat_arrays(accumulated_indices)
                    concat_meta = safe_concat_arrays(accumulated_meta)
                    
                    # FIX #2: CRITICAL - Slice the targets_dict properly before passing
                    sliced_targets = {
                        k: safe_concat_arrays(v)[:base_split_size] 
                        for k, v in accumulated_targets_dict.items()
                    }
                    
                    # Save a full split
                    manager._save_local_split(
                        current_split_id,
                        concat_sequences[:base_split_size],
                        concat_labels[:base_split_size],
                        sliced_targets,  # ✅ NOW SLICED CORRECTLY
                        concat_indices[:base_split_size],
                        concat_meta[:base_split_size],
                        feature_count,
                        sequence_length
                    )
                    
                    # Remove used samples from accumulator (using concatenated version for counting)
                    samples_used = base_split_size
                    
                    # ✅ BUG FIX #7: Always convert back to list (even if empty) to support .append() in next iteration
                    def to_list_safe(arr):
                        if isinstance(arr, np.ndarray):
                            return arr.tolist()
                        return list(arr)
                    
                    accumulated_sequences = to_list_safe(concat_sequences[base_split_size:])
                    accumulated_labels = to_list_safe(concat_labels[base_split_size:])
                    accumulated_indices = to_list_safe(concat_indices[base_split_size:])
                    accumulated_meta = to_list_safe(concat_meta[base_split_size:])
                    
                    for k in accumulated_targets_dict:
                        concat_k = safe_concat_arrays(accumulated_targets_dict[k])
                        accumulated_targets_dict[k] = to_list_safe(concat_k[base_split_size:])
                        # ✅ FREE TEMPORARY CONCATENATED TARGET ARRAY
                        del concat_k
                    
                    # ✅ FREE TEMPORARY CONCATENATED ARRAYS after slicing
                    del concat_sequences, concat_labels, concat_indices, concat_meta, sliced_targets
                    gc.collect()
                    
                    samples_in_current_split -= base_split_size
                    current_split_id += 1
                    
                    # If it's the last chunk, we might have remainder for the VERY last split
                    if is_last_chunk and samples_in_current_split > 0:
                         manager._save_local_split(
                            current_split_id,
                            accumulated_sequences,
                            accumulated_labels,
                            accumulated_targets_dict,
                            accumulated_indices,
                            accumulated_meta,
                            feature_count,
                            sequence_length
                        )
                         current_split_id += 1
                elif is_last_chunk:
                    # Last DB chunk AND last local split — dump everything remaining
                    concat_sequences = safe_concat_arrays(accumulated_sequences)
                    concat_labels = safe_concat_arrays(accumulated_labels)
                    concat_indices = safe_concat_arrays(accumulated_indices)
                    concat_meta = safe_concat_arrays(accumulated_meta)
                    concat_targets = {
                        k: safe_concat_arrays(v) for k, v in accumulated_targets_dict.items()
                    }
                    
                    manager._save_local_split(
                        current_split_id,
                        concat_sequences,
                        concat_labels,
                        concat_targets,
                        concat_indices,
                        concat_meta,
                        feature_count,
                        sequence_length
                    )
                    # ✅ FREE TEMPORARY CONCATENATED ARRAYS
                    del concat_sequences, concat_labels, concat_indices, concat_meta, concat_targets
                    samples_in_current_split = 0
                    current_split_id += 1
                    gc.collect()
        
        if samples_in_current_split > 0 and current_split_id >= manager.config.total_splits:
            logger.warning(
                f"  ⚠️ {samples_in_current_split} samples left over after filling {manager.config.total_splits} splits. "
                f"They will be discarded. Consider increasing num_splits."
            )

        # 5. Finalize manager
        manager.config.total_splits = current_split_id
        manager._save_metadata()
        manager._save_config()
        
        # ✅ FINAL MEMORY CLEANUP
        del accumulated_sequences, accumulated_labels, accumulated_indices, accumulated_meta, accumulated_targets_dict
        del chunk_rows, all_chunk_sizes
        gc.collect()
        
        logger.info(f"  ✅ Completed re-chunking. Created {current_split_id} local splits.")
        return manager

    def _save_local_split(self, split_id, seqs, labels, targets_dict, indices, meta, feat_count, seq_len, slice_targets=False, limit=0):
        """Internal helper to save a split file and update metadata"""
        
        # Slice targets if requested
        final_targets = {}
        if slice_targets and limit > 0:
            for k, v in targets_dict.items():
                final_targets[k] = v[:limit]
        else:
            final_targets = targets_dict

        metadata = SplitMetadata(
            split_id=split_id,
            start_idx=0, # Relative index within the set of splits
            end_idx=len(seqs),
            num_samples=len(seqs),
            feature_count=feat_count,
            sequence_length=seq_len
        )
        self.split_metadata[split_id] = metadata
        
        split_path = self._get_split_filepath(split_id)
        # Use joblib to save; IncrementalModelTrainer expects this
        # Note: IncrementalModelTrainer.train_incremental expects split_data to have "sequences" and "targets"
        # Since targets_dict can have multiple columns, we need to be careful.
        # But wait, TrainingSplitManager.create_from_data (original) uses "targets" as a single array.
        # Let's align with that by providing the first target or specific target if known?
        # Actually, model_builder.py extracts it during load_split, but load_split only returns (seqs, targets).
        # Wait, I need to check load_split in this file again.
        
        split_data = {
            "sequences": np.array(seqs),
            "targets": final_targets, # We store the dict; load_split will handle it
            "labels": np.array(labels),
            "indices": np.array(indices),
            "metadata": asdict(metadata)
        }
        joblib.dump(split_data, split_path, compress=3)
        logger.debug(f"    Saved local split {split_id}: {len(seqs)} samples")

    def _get_split_filepath(self, split_id: int) -> str:
        """Get filesystem path for a split"""
        return os.path.join(
            self.config.disk_path,
            f"{self.task_id}_split_{split_id:03d}.joblib"
        )

    def _save_metadata(self):
        """Persist split metadata to disk"""
        metadata_dict = {
            str(k): v.to_dict() for k, v in self.split_metadata.items()
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        logger.debug(f"Saved metadata: {self.metadata_file}")

    def _save_config(self):
        """Persist configuration to disk"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        logger.debug(f"Saved config: {self.config_file}")

    # ========================================================================
    # LOADING EXISTING SPLITS
    # ========================================================================

    @staticmethod
    def load_from_metadata(
        task_id: str,
        disk_path: str = "/tmp/training_splits"
    ) -> 'TrainingSplitManager':
        """
        Load a previously saved split manager from disk.
        
        Args:
            task_id: Identifier to restore
            disk_path: Where splits are stored
            
        Returns:
            Loaded TrainingSplitManager ready for training
            
        Raises:
            FileNotFoundError: If metadata file doesn't exist
        """
        
        config = SplitConfig(disk_path=disk_path)
        manager = TrainingSplitManager(task_id, config)
        
        # Load metadata
        metadata_file = os.path.join(disk_path, f"{task_id}_metadata.json")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            metadata_dict = json.load(f)
        
        manager.split_metadata = {
            int(k): SplitMetadata(**v) for k, v in metadata_dict.items()
        }
        
        # Load config
        config_file = os.path.join(disk_path, f"{task_id}_config.json")
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            manager.config = SplitConfig(**config_data)
        
        logger.info(
            f"Loaded split manager: {len(manager.split_metadata)} splits "
            f"({sum(m.num_samples for m in manager.split_metadata.values())} total samples)"
        )
        
        return manager

    # ========================================================================
    # SPLIT LOADING AND FREEING
    # ========================================================================

    def load_split(self, split_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a split into memory.
        
        Args:
            split_id: Which split (0 to total_splits-1)
            
        Returns:
            (sequences, targets) arrays where targets is extracted based on target_column if set
            
        Raises:
            ValueError: If split_id invalid or target_column not found
            FileNotFoundError: If split file missing
        """
        
        if split_id not in self.split_metadata:
            raise ValueError(f"Invalid split_id: {split_id}")
        
        metadata = self.split_metadata[split_id]
        split_path = self._get_split_filepath(split_id)
        
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Split file not found: {split_path}")
        
        # Load from disk
        split_data = joblib.load(split_path)
        sequences = split_data["sequences"]
        
        # 🔧 FIX #6: Support both single-array targets and dict-based targets with extraction
        targets = split_data.get("targets")
        if isinstance(targets, dict):
            # If target_column is set, extract the specific column; otherwise use 'labels' key
            if hasattr(self, 'target_column') and self.target_column:
                if self.target_column not in targets:
                    raise ValueError(
                        f"Target column '{self.target_column}' not found in targets dict. "
                        f"Available keys: {list(targets.keys())}"
                    )
                targets = targets[self.target_column]
            else:
                # 'labels' is a top-level key in the split file, not inside targets dict
                targets = split_data.get("labels")
                if targets is None:
                    raise ValueError(
                        f"No target_column set and no 'labels' array found in split file. "
                        f"Available targets dict keys: {list(split_data.get('targets', {}).keys())}"
                    )
        
        # Store reference for cleanup
        self.current_split_id = split_id
        self.current_split_data = (sequences, targets)
        
        logger.debug(
            f"Loaded split {split_id}: {len(sequences)} sequences "
            f"({sequences.nbytes / 1e6:.1f} MB){f' [target: {self.target_column}]' if hasattr(self, 'target_column') and self.target_column else ''}"
        )
        
        return sequences, targets

    def free_current_split(self):
        """Free current split from memory"""
        if self.current_split_id is not None:
            logger.debug(f"Freed split {self.current_split_id}")
        
        self.current_split_id = None
        self.current_split_data = None

    # ========================================================================
    # SPLIT ORDERING
    # ========================================================================

    def get_epoch_order(
        self,
        epoch: int,
        shuffle: bool = True,
        seed: Optional[int] = None
    ) -> List[int]:
        """
        Get the order to process splits for a given epoch.
        
        Args:
            epoch: Current epoch number (0-based)
            shuffle: Whether to shuffle split order
            seed: Random seed (default: config.seed + epoch)
            
        Returns:
            List of split_ids in processing order
            
        Note:
            For time series data:
            - Shuffling at split-level is valid because mini-batch SGD
              already shuffles at batch-level
            - Different shuffle per epoch improves generalization
            - Not the same as shuffling raw time series data
        """
        
        split_ids = list(range(self.config.total_splits))
        
        if not shuffle:
            return split_ids
        
        # Deterministic shuffle for reproducibility
        actual_seed = (seed if seed is not None else self.config.seed) + epoch
        
        rng = random.Random(actual_seed)
        rng.shuffle(split_ids)
        
        return split_ids

    def get_sequential_order(self) -> List[int]:
        """Get sequential split order (no shuffling)"""
        return list(range(self.config.total_splits))

    # ========================================================================
    # STATISTICS AND UTILITIES
    # ========================================================================

    def get_stats(self) -> Dict:
        """Get summary statistics"""
        total_samples = sum(m.num_samples for m in self.split_metadata.values())
        
        return {
            "task_id": self.task_id,
            "total_splits": self.config.total_splits,
            "total_samples": total_samples,
            "samples_per_split": total_samples // self.config.total_splits,
            "average_samples": int(total_samples / self.config.total_splits),
            "feature_count": self.split_metadata[0].feature_count if self.split_metadata else 0,
            "sequence_length": self.split_metadata[0].sequence_length if self.split_metadata else 0,
            "persist_to_disk": self.config.persist_to_disk,
            "memory_per_split_mb": (
                total_samples / self.config.total_splits * 
                self.split_metadata[0].feature_count * 
                self.split_metadata[0].sequence_length * 
                8 / 1e6
            )
        }

    def cleanup(self):
        """Remove all split files and metadata from disk"""
        if os.path.exists(self.config.disk_path):
            # Only remove files for this task
            for split_id in self.split_metadata.keys():
                split_path = self._get_split_filepath(split_id)
                if os.path.exists(split_path):
                    os.remove(split_path)
            
            # Remove metadata files
            if os.path.exists(self.metadata_file):
                os.remove(self.metadata_file)
            if os.path.exists(self.config_file):
                os.remove(self.config_file)
            
            logger.info(f"Cleaned up splits for task {self.task_id}")

# ============================================================================
# INCREMENTAL TRAINER
# ============================================================================

class IncrementalModelTrainer:
    """
    Trains a Keras/TensorFlow model using incremental split-based learning.
    
    Workflow:
    1. Load model (pre-compiled)
    2. Create split manager from training data
    3. For each epoch:
       - Get shuffled split order
       - For each split:
         - Load split into memory
         - Train one epoch on split (model.fit with epochs=1)
         - Free split memory
         - Report progress
    4. Save final weights
    
    Memory Usage:
    - Traditional (all at once): ~1GB for 1M sequences
    - Split-based (load 1 at a time): ~100MB constant
    """
    
    def __init__(self, 
                 model,
                 split_manager: TrainingSplitManager,
                 verbose: bool = True):
        self.model = model
        self.split_manager = split_manager
        self.verbose = verbose
        
        self.training_history = {
            "epochs": [],
            "loss": [],
            "mae": [],
            "mse": [],
            "val_loss": [],
            "val_mae": [],
            "val_mse": [],
        }
        
        self.metrics_per_split = []
        
        # Checkpoint directory
        self.checkpoint_dir = os.path.join(
            split_manager.config.disk_path, 
            f"{split_manager.task_id}_checkpoints"
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.weights_path = os.path.join(self.checkpoint_dir, "latest_weights.weights.h5")
        self.state_path = os.path.join(self.checkpoint_dir, "training_state.json")
        
        logger.info(
            f"Initialized IncrementalModelTrainer "
            f"for {split_manager.config.total_splits} splits"
        )

    async def train_incremental(
        self,
        epochs: int = 50,
        batch_size: int = 32,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        progress_callback: Optional[Callable] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Train model incrementally on splits.
        
        Args:
            epochs: Number of full passes through all splits
            batch_size: Mini-batch size (TensorFlow's batch_size parameter)
            validation_data: Optional (X_val, y_val) for validation
            progress_callback: Async function(progress_data) for UI updates
            verbose: Print progress to logger
            
        Returns:
            Training history dict with keys: epochs, loss, accuracy, etc.
        """
        
        stats = self.split_manager.get_stats()
        
        if verbose:
            logger.info(
                f"Starting incremental training\n"
                f"  Epochs: {epochs}\n"
                f"  Total splits: {stats['total_splits']}\n"
                f"  Samples per split: {stats['samples_per_split']}\n"
                f"  Mode: {('shuffled' if self.split_manager.config.shuffle_splits_per_epoch else 'sequential')}\n"
                f"  Memory per split: {stats['memory_per_split_mb']:.1f} MB"
            )
        
        total_iterations = epochs * stats['total_splits']
        iteration = 1  # Start from 1 for better percentage reporting
        
        # Initial checkpoint (save starting weights)
        self.save_checkpoint(-1, -1, [], epochs, batch_size)

        for epoch in range(epochs):
            
            # Get split order for this epoch
            split_order = self.split_manager.get_epoch_order(
                epoch,
                shuffle=self.split_manager.config.shuffle_splits_per_epoch
            )
            
            epoch_losses = []
            epoch_accuracies = []
            
            for split_index, split_id in enumerate(split_order):
                await self._train_single_split(
                    epoch=epoch,
                    split_id=split_id,
                    split_index=split_index,
                    split_order=split_order,
                    total_epochs=epochs,
                    batch_size=batch_size,
                    iteration=iteration,
                    total_iterations=total_iterations,
                    validation_data=validation_data,
                    progress_callback=progress_callback,
                    verbose=verbose
                )
                
                # Accumulate for epoch metrics
                last_metrics = self.metrics_per_split[-1]
                epoch_losses.append(last_metrics["loss"])
                epoch_accuracies.append(last_metrics.get("accuracy", last_metrics.get("mae", 0)))
                
                iteration += 1
            
            # Record epoch metrics
            self.training_history["epochs"].append(epoch)
            self.training_history["loss"].append(np.mean(epoch_losses))
            self.training_history["mae"].append(np.mean(epoch_accuracies))
            # Note: MSE could also be averaged here if needed
            
            # Final epoch checkpoint
            self.save_checkpoint(epoch, len(split_order) - 1, split_order, epochs, batch_size)
        
        if verbose:
            logger.info(f"✓ Training complete!")
            logger.info(
                f"  Final loss: {self.training_history['loss'][-1]:.4f}"
            )
        
        return self.training_history

    def get_history(self) -> Dict:
        """Get complete training history"""
        return self.training_history

    def get_metrics_summary(self) -> Dict:
        """Get summary metrics over all splits"""
        if not self.metrics_per_split:
            return {}
        
        all_losses = [m["loss"] for m in self.metrics_per_split]
        all_accs = [m.get("accuracy", m.get("mae", 0)) for m in self.metrics_per_split]
        
        return {
            "mean_loss": float(np.mean(all_losses)),
            "min_loss": float(np.min(all_losses)),
            "max_loss": float(np.max(all_losses)),
            "std_loss": float(np.std(all_losses)),
            "mean_accuracy": float(np.mean(all_accs)) if all_accs else None,
            "total_splits_trained": len(self.metrics_per_split),
        }

    def save_checkpoint(self, epoch: int, split_index: int, split_order: List[int], total_epochs: int, batch_size: int):
        """Save current weights and training state to disk"""
        try:
            # Save weights
            self.model.save_weights(self.weights_path)
            
            # Save state
            state = TrainingState(
                epoch=epoch,
                split_index=split_index,
                split_order=split_order,
                total_epochs=total_epochs,
                batch_size=batch_size,
                history=self.training_history,
                metrics_per_split=self.metrics_per_split,
                timestamp=time.time()
            )
            
            with open(self.state_path, 'w') as f:
                json.dump(state.to_dict(), f, indent=2)
            
            logger.info(f"Saved checkpoint: Epoch {epoch}, Split Index {split_index}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self) -> Optional[Dict]:
        """Load weights and state from disk if they exist"""
        if not os.path.exists(self.weights_path) or not os.path.exists(self.state_path):
            return None
        
        try:
            # Load weights
            self.model.load_weights(self.weights_path)
            
            # Load state
            with open(self.state_path, 'r') as f:
                state_data = json.load(f)
            
            # Restore history and metrics
            self.training_history = state_data["history"]
            self.metrics_per_split = state_data["metrics_per_split"]
            
            logger.info(f"Restored checkpoint from Epoch {state_data['epoch']}, Split Index {state_data['split_index']}")
            return state_data
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    async def resume_training(
        self,
        batch_size: Optional[int] = None,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        progress_callback: Optional[Callable] = None,
        verbose: bool = True
    ) -> Dict:
        """Resume training from the last saved checkpoint"""
        state_data = self.load_checkpoint()
        if not state_data:
            logger.warning("No checkpoint found to resume from.")
            return {}

        start_epoch = state_data["epoch"]
        start_split_index = state_data["split_index"] + 1
        split_order = state_data["split_order"]
        total_epochs = state_data["total_epochs"]
        batch_size = batch_size or state_data["batch_size"]

        iteration = (start_epoch * len(split_order)) + start_split_index
        total_iterations = total_epochs * len(split_order)

        logger.info(f"Resuming training from Epoch {start_epoch + 1}, Split Index {start_split_index}")

        # Continue the current epoch's remaining splits
        for split_index in range(start_split_index, len(split_order)):
            await self._train_single_split(
                epoch=start_epoch,
                split_id=split_order[split_index],
                split_index=split_index,
                split_order=split_order,
                total_epochs=total_epochs,
                batch_size=batch_size,
                iteration=iteration,
                total_iterations=total_iterations,
                validation_data=validation_data,
                progress_callback=progress_callback,
                verbose=verbose
            )
            iteration += 1

        # Continue with subsequent epochs
        for epoch in range(start_epoch + 1, total_epochs):
            split_order = self.split_manager.get_epoch_order(
                epoch,
                shuffle=self.split_manager.config.shuffle_splits_per_epoch
            )
            
            for split_index, split_id in enumerate(split_order):
                await self._train_single_split(
                    epoch=epoch,
                    split_id=split_id,
                    split_index=split_index,
                    split_order=split_order,
                    total_epochs=total_epochs,
                    batch_size=batch_size,
                    iteration=iteration,
                    total_iterations=total_iterations,
                    validation_data=validation_data,
                    progress_callback=progress_callback,
                    verbose=verbose
                )
                iteration += 1
            
            # Epoch-level checkpoint
            self.save_checkpoint(epoch, len(split_order) - 1, split_order, total_epochs, batch_size)

        return self.training_history

    async def _train_single_split(
        self,
        epoch: int,
        split_id: int,
        split_index: int,
        split_order: List[int],
        total_epochs: int,
        batch_size: int,
        iteration: int,
        total_iterations: int,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        progress_callback: Optional[Callable] = None,
        verbose: bool = True
    ):
        """Helper to train a single split and handle progress/checkpointing"""
        X_split, y_split = self.split_manager.load_split(split_id)
        
        try:
            history = self.model.fit(
                X_split, y_split,
                epochs=1,
                batch_size=batch_size,
                validation_data=validation_data,
                verbose=0
            )
            
            loss = float(history.history['loss'][0])
            mae = float(history.history.get('mae', history.history.get('mean_absolute_error', [0]))[0])
            mse = float(history.history.get('mse', history.history.get('mean_squared_error', [0]))[0])
            
            # Record metrics
            self.metrics_per_split.append({
                "epoch": epoch,
                "split": split_id,
                "loss": loss,
                "mae": mae,
                "mse": mse
            })
            
            # Progress reporting
            progress = (iteration / total_iterations) * 100
            if progress_callback:
                progress_data = TrainingProgress(
                    epoch=epoch,
                    split_id=split_id,
                    total_epochs=total_epochs,
                    total_splits=len(split_order),
                    loss=loss,
                    accuracy=mae, # Using mae for accuracy field
                    progress_percent=progress
                )
                await progress_callback(progress_data)
            
            if verbose:
                logger.info(
                    f"Epoch {epoch+1}/{total_epochs}, Split {split_id}/{len(split_order)-1}: "
                    f"loss={loss:.4f}, progress={progress:.1f}%"
                )
            
            # Per-split checkpoint if configured
            if self.split_manager.config.checkpoint_on_split:
                self.save_checkpoint(epoch, split_index, split_order, total_epochs, batch_size)

        finally:
            self.split_manager.free_current_split()
            del X_split, y_split
            gc.collect() # ✅ Ensure memory is reclaimed before next split
