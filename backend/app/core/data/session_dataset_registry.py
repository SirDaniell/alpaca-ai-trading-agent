"""
Session Dataset Registry with Three-Tier Caching Architecture.

Manages dataset lifecycle across TIER 0/1/2:
- TIER 0: Single pointer in AnalysisManager (current dataset for training)
- TIER 1: LRU cache in-memory (last 5 accessed datasets, 4-hour TTL)
- TIER 2: PostgreSQL database (all datasets, compressed with ZSTANDARD, persistent until deletion)

Memory Management:
- Explicitly clean up large objects after extraction
- TTL-based automatic eviction of TIER 1 entries
- LRU eviction when cache capacity exceeded
- Compress/decompress on-demand to reduce memory footprint

OOP Design:
- Encapsulation: Private cache (_tier1_cache, _tier1_timestamps, _ttl_cleanup_thread)
- Resource Management: __init__/__del__ for cleanup, context managers for DB connections
- Abstraction: Public interface hides DB implementation details
- Type Safety: Full type hints on all methods and parameters
"""

import logging
import zstandard as zstd
import pickle
import asyncio
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from collections import OrderedDict
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from sqlalchemy import text
import json
import hashlib
from enum import Enum

from app.core.data.serializers import serialize_data, deserialize_data

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes (Serializable)
# ============================================================================

@dataclass
class DatasetMetadata:
    """Immutable metadata for a dataset."""
    dataset_id: str  # Changed from 'id' to match DB schema
    session_id: str
    dataset_name: str
    source_step: str  # "snr_analysis" or "ml_preparation"
    parent_dataset_id: Optional[str] = None
    output_targets: Optional[Dict[str, Any]] = None
    feature_selection: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    compression_type: str = "zstandard"
    compression_ratio: float = 0.0
    uncompressed_size_mb: int = 0
    compressed_size_mb: int = 0
    target_metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


# ============================================================================
# Compression Utilities
# ============================================================================

class CompressionHandler:
    """Handle ZSTANDARD compression/decompression for datasets."""
    
    # Compression level: 10 = default balance of speed/ratio
    COMPRESSION_LEVEL = 10
    
    @staticmethod
    def compress(data: Any) -> Tuple[bytes, float, int]:
        """
        Compress data using ZSTANDARD.
        
        Args:
            data: Object to compress (will be pickled first)
        
        Returns:
            Tuple of (compressed_bytes, compression_ratio, compressed_size_mb)
            - compression_ratio: original_size / compressed_size (e.g., 4.0 = 75% reduction)
            - compressed_size_mb: Size in MB
        """
        try:
            # Serialize to bytes
            pickled = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            original_size = len(pickled)
            
            # Compress
            cctx = zstd.ZstdCompressor(level=CompressionHandler.COMPRESSION_LEVEL)
            compressed = cctx.compress(pickled)
            compressed_size = len(compressed)
            
            # Calculate metrics
            ratio = original_size / compressed_size if compressed_size > 0 else 0
            size_mb = compressed_size / (1024 * 1024)
            
            logger.debug(
                f"Compressed {original_size / (1024*1024):.2f}MB → {size_mb:.2f}MB "
                f"(ratio: {ratio:.2f}x)"
            )
            
            return compressed, ratio, size_mb
        
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            raise
    
    @staticmethod
    def decompress(compressed: bytes) -> Any:
        """
        Decompress ZSTANDARD data.
        
        Args:
            compressed: Compressed bytes
        
        Returns:
            Decompressed object
        """
        try:
            dctx = zstd.ZstdDecompressor()
            decompressed = dctx.decompress(compressed)
            data = pickle.loads(decompressed)
            return data
        
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            raise


# ============================================================================
# Session Dataset Registry
# ============================================================================

class SessionDatasetRegistry:
    """
    Manages datasets in a session with three-tier caching.
    
    Features:
    - TIER 1 LRU cache: Fast access to recent datasets (4-hour TTL)
    - TIER 2 DB storage: Persistent dataset storage (compressed)
    - Automatic compression/decompression
    - Memory-efficient resource cleanup
    - Dataset lineage tracking (SNR → ML)
    """
    
    # Configuration constants
    TIER1_MAX_DATASETS = 5
    TIER1_TTL_MINUTES = 240  # 4 hours
    TIER1_CLEANUP_INTERVAL_SECONDS = 60
    
    def __init__(self, session_id: str, db_connection: Any):
        """
        Initialize registry for a session.
        
        Args:
            session_id: Unique session identifier
            db_connection: Database connection object (PostgreSQL async driver)
        
        Memory allocated:
            - ~50-200 MB for TIER 1 cache (5 datasets × 40 MB avg)
            - Minimal runtime overhead otherwise
        """
        self.session_id: str = session_id
        self.db: Any = db_connection
        
        # TIER 0: Single pointer (managed by AnalysisManager)
        self._current_pointer_id: Optional[str] = None
        
        # TIER 1: LRU cache (most recent 5 datasets)
        self._tier1_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._tier1_timestamps: Dict[str, datetime] = {}
        
        # Background cleanup task for TTL expiration
        self._cleanup_task: Optional[asyncio.Task] = None
        
        logger.info(f"SessionDatasetRegistry initialized for session {session_id}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup resources."""
        await self.cleanup()
    
    async def cleanup(self) -> None:
        """
        Explicit cleanup of session resources.
        
        Memory Management:
        - Clears TIER 1 cache (frees ~50-200 MB)
        - Cancels cleanup task
        - Does NOT delete TIER 2 database records (user must explicitly delete)
        """
        try:
            # Clear TIER 1 cache
            self._tier1_cache.clear()
            self._tier1_timestamps.clear()
            
            # Cancel cleanup task
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            logger.info(f"Registry cleanup complete for session {self.session_id}")
        
        except Exception as e:
            logger.error(f"Error during registry cleanup: {e}")
    
    # ========================================================================
    # TIER 1 Cache Management
    # ========================================================================
    
    def _check_ttl(self) -> None:
        """
        Check and remove expired TIER 1 entries (TTL-based eviction).
        
        Private method called during dataset access.
        Removes entries older than TIER1_TTL_MINUTES.
        """
        now = datetime.now()
        expired_ids = [
            dataset_id for dataset_id, timestamp in self._tier1_timestamps.items()
            if (now - timestamp).total_seconds() > self.TIER1_TTL_MINUTES * 60
        ]
        
        for dataset_id in expired_ids:
            try:
                if dataset_id in self._tier1_cache:
                    # Explicit cleanup of large objects
                    data = self._tier1_cache.pop(dataset_id, {})
                    del data
                
                if dataset_id in self._tier1_timestamps:
                    del self._tier1_timestamps[dataset_id]
                
                logger.debug(f"Evicted expired TIER 1 entry: {dataset_id}")
            
            except Exception as e:
                logger.error(f"Error evicting {dataset_id}: {e}")
    
    def _evict_oldest(self) -> Optional[str]:
        """
        LRU eviction: Remove oldest (least recently used) dataset from TIER 1.
        
        Returns:
            ID of evicted dataset, or None if cache empty
        
        Memory Management:
        - Explicitly deletes data dictionary
        - Marks for garbage collection
        """
        if not self._tier1_cache:
            return None
        
        # Pop oldest (first) item from OrderedDict
        evicted_id, evicted_data = self._tier1_cache.popitem(last=False)
        
        try:
            # Explicit cleanup of large objects
            del evicted_data
        except:
            pass
        
        if evicted_id in self._tier1_timestamps:
            del self._tier1_timestamps[evicted_id]
        
        logger.debug(f"LRU eviction: {evicted_id} (cache size now: {len(self._tier1_cache)})")
        return evicted_id
    
    def _load_to_tier1(
        self,
        dataset_id: str,
        train_data: np.ndarray,
        validation_data: np.ndarray,
        test_data: np.ndarray,
        train_labels: np.ndarray,
        validation_labels: np.ndarray,
        test_labels: np.ndarray,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Load dataset into TIER 1 LRU cache.
        
        Evicts oldest entry if cache full (> 5 datasets).
        Updates access timestamp for TTL tracking.
        
        Args:
            dataset_id: Unique dataset identifier
            train_data, validation_data, test_data: Feature arrays
            train_labels, validation_labels, test_labels: Label arrays
            metadata: Dataset metadata dict
        
        Memory Impact:
        - Stores ~40MB per dataset average
        - Evicts LRU entry if cache exceeds TIER1_MAX_DATASETS
        """
        # Check TTL and evict expired entries
        self._check_ttl()
        
        # Evict oldest if at capacity
        if len(self._tier1_cache) >= self.TIER1_MAX_DATASETS:
            evicted = self._evict_oldest()
            logger.info(
                f"Cache full ({self.TIER1_MAX_DATASETS}): evicted {evicted}, "
                f"now {len(self._tier1_cache)}/{self.TIER1_MAX_DATASETS}"
            )
        
        # Store in cache with timestamp
        self._tier1_cache[dataset_id] = {
            "train_data": train_data,
            "validation_data": validation_data,
            "test_data": test_data,
            "train_labels": train_labels,
            "validation_labels": validation_labels,
            "test_labels": test_labels,
            "metadata": metadata
        }
        
        self._tier1_timestamps[dataset_id] = datetime.now()
        
        # Move to end (most recently used)
        self._tier1_cache.move_to_end(dataset_id)
        
        logger.info(
            f"Loaded {dataset_id} to TIER 1 cache "
            f"(size: {len(self._tier1_cache)}/{self.TIER1_MAX_DATASETS})"
        )
    
    # ========================================================================
    # Smart Tiering: Get Dataset
    # ========================================================================
    
    async def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch dataset with smart tiering.
        
        Strategy:
        1. Check TIER 1 cache (< 1ms) → return immediately
        2. Miss: Fetch from TIER 2 DB (100-300ms)
        3. Decompress and load to TIER 1
        4. Return decompressed data
        
        Args:
            dataset_id: Dataset to fetch
        
        Returns:
            Dict with keys: train_data, validation_data, test_data,
                           train_labels, validation_labels, test_labels, metadata
            Or None if not found
        
        Performance:
        - TIER 1 hit: ~0.1ms return time
        - TIER 1 miss: ~150-300ms (DB fetch + decompress)
        """
        # Check TTL first
        self._check_ttl()
        
        # ===== TIER 1 Hit =====
        if dataset_id in self._tier1_cache:
            # Update access time and move to end (most recently used)
            self._tier1_timestamps[dataset_id] = datetime.now()
            self._tier1_cache.move_to_end(dataset_id)
            
            logger.debug(f"TIER 1 cache hit for {dataset_id}")
            return self._tier1_cache[dataset_id]
        
        # ===== TIER 1 Miss → Fetch from TIER 2 =====
        logger.debug(f"TIER 1 cache miss for {dataset_id}, fetching from TIER 2 (DB)...")
        
        try:
            # Query database
            from sqlalchemy import text
            query = text("""
                SELECT 
                    dataset_id, session_id, dataset_name, source_step, parent_dataset_id,
                    output_targets, feature_selection, source_metadata,
                    train_data_compressed, validation_data_compressed, test_data_compressed,
                    train_labels, validation_labels, test_labels
                FROM ml_datasets
                WHERE dataset_id = :dataset_id AND session_id = :session_id
            """)
            
            result = await self.db.execute(query, {'dataset_id': dataset_id, 'session_id': self.session_id})
            row = result.fetchone()
            
            if row is None:
                logger.warning(f"Dataset {dataset_id} not found in DB for session {self.session_id}")
                return None
            
            # Decompress data
            train_data = CompressionHandler.decompress(row['train_data_compressed'])
            validation_data = CompressionHandler.decompress(row['validation_data_compressed'])
            test_data = CompressionHandler.decompress(row['test_data_compressed'])
            train_labels = row['train_labels']  # Already pickled
            validation_labels = row['validation_labels']
            test_labels = row['test_labels']
            
            # Parse JSON fields
            import json
            
            def safe_json_loads(json_str):
                """Safely deserialize JSON strings, handling None and malformed JSON."""
                if not json_str:
                    return None
                try:
                    return json.loads(json_str)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Failed to parse JSON: {e}, returning string value")
                    return json_str
            
            output_targets = safe_json_loads(row['output_targets'])
            feature_selection = safe_json_loads(row['feature_selection'])
            source_metadata = safe_json_loads(row['source_metadata'])
            
            # Prepare metadata
            metadata = {
                'dataset_id': row['dataset_id'],
                'dataset_name': row['dataset_name'],
                'source_step': row['source_step'],
                'parent_dataset_id': row['parent_dataset_id'],
                'output_targets': output_targets,
                'feature_selection': feature_selection,
                'metadata': source_metadata
            }
            
            # Load to TIER 1 cache
            self._load_to_tier1(
                dataset_id=dataset_id,
                train_data=train_data,
                validation_data=validation_data,
                test_data=test_data,
                train_labels=train_labels,
                validation_labels=validation_labels,
                test_labels=test_labels,
                metadata=metadata
            )
            
            logger.info(f"Fetched {dataset_id} from TIER 2 and loaded to TIER 1")
            return self._tier1_cache[dataset_id]
        
        except Exception as e:
            logger.error(f"Error fetching dataset {dataset_id}: {e}")
            return None
    
    # ========================================================================
    # Register Dataset
    # ========================================================================
    
    async def register_dataset(
        self,
        dataset_id: str,
        dataset_name: str,
        train_data: np.ndarray,
        validation_data: np.ndarray,
        test_data: np.ndarray,
        train_labels: np.ndarray,
        validation_labels: np.ndarray,
        test_labels: np.ndarray,
        source_step: str,
        output_targets: Optional[Dict[str, Any]] = None,
        feature_selection: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent_dataset_id: Optional[str] = None,
        # ✅ NEW: Add target parameters for regression target storage
        train_targets: Optional[Dict[str, np.ndarray]] = None,
        validation_targets: Optional[Dict[str, np.ndarray]] = None,
        test_targets: Optional[Dict[str, np.ndarray]] = None,
    ) -> bool:
        """
        Register a new dataset in the registry.
        
        Process:
        1. Compress data with ZSTANDARD
        2. Store compressed in TIER 2 (DB)
        3. Load to TIER 1 cache
        4. Update TIER 0 pointer
        5. Track metadata (lineage, feature_selection, etc.)
        
        Args:
            dataset_id: Unique dataset ID
            dataset_name: Human-readable name
            train_data: Training feature array (sequences or flat)
            validation_data: Validation feature array
            test_data: Test feature array
            train_labels: Training labels
            validation_labels: Validation labels
            test_labels: Test labels
            source_step: "snr_analysis" or "ml_preparation"
            output_targets: {"signal_type": "bounce_support", ...}
            feature_selection: {"mode": "rich", "feature_count": 40, ...}
            metadata: Additional metadata dict
            parent_dataset_id: For lineage tracking (SNR → ML)
            train_targets: Dict of training regression targets {target_name: np.array(N, prediction_length)}
            validation_targets: Dict of validation regression targets
            test_targets: Dict of test regression targets
        
        Returns:
            True if successful, False if failed
        
        Memory Impact:
        - Compresses data (75% reduction typical)
        - Stores in TIER 1 cache (~40MB)
        - DB storage only (does not keep copy in app memory beyond TIER 1)
        """
        try:
            # ========================================================================
            # STEP 0: Input Validation
            # ========================================================================
            logger.info(
                f"🔵 [REGISTER_START] dataset_id={dataset_id}, dataset_name={dataset_name}, "
                f"session_id={self.session_id}, source_step={source_step}"
            )
            
            # Validate required fields
            if not dataset_id or not isinstance(dataset_id, str):
                raise ValueError(f"❌ Invalid dataset_id: {dataset_id} (must be non-empty string)")
            if not dataset_name or not isinstance(dataset_name, str):
                raise ValueError(f"❌ Invalid dataset_name: {dataset_name} (must be non-empty string)")
            if not source_step or source_step not in ["snr_analysis", "ml_preparation"]:
                raise ValueError(f"❌ Invalid source_step: {source_step} (must be 'snr_analysis' or 'ml_preparation')")
            if not self.session_id:
                raise ValueError(f"❌ Invalid session_id: {self.session_id} (must be non-empty)")
            
            # Validate data arrays
            if train_data is None or len(train_data) == 0:
                raise ValueError(f"❌ Empty train_data")
            if validation_data is None or len(validation_data) == 0:
                raise ValueError(f"❌ Empty validation_data")
            if test_data is None or len(test_data) == 0:
                raise ValueError(f"❌ Empty test_data")
            
            # ========================================================================
            # STEP 0.5: EARLY HASH COMPUTATION + COLLISION DETECTION
            # ========================================================================
            # Compute hash early (before expensive compression) to detect duplicates fast
            
            def safe_json_dumps_early(obj):
                if obj is None:
                    return None
                def _serializer(o):
                    if isinstance(o, Enum):
                        return o.value
                    return str(o)
                try:
                    return json.dumps(obj, default=_serializer)
                except Exception:
                    return json.dumps(str(obj))
            
            output_targets_json_early = safe_json_dumps_early(output_targets)
            # Include dataset_id in hash so every run is unique — no false collisions
            output_targets_hash = hashlib.sha256(
                (f"{dataset_id}:{output_targets_json_early or ''}").encode('utf-8')
            ).hexdigest()[:64]
            
            # Check only name collision — hash is unique per run so no hash collisions possible
            collision_check = text("""
                SELECT COUNT(*) FROM ml_datasets
                WHERE session_id = :session_id AND dataset_name = :dataset_name
            """)
            
            result = await self.db.execute(collision_check, {
                'session_id': self.session_id,
                'dataset_name': dataset_name,
            })
            name_collision_count = result.scalar() or 0
            
            original_dataset_name = dataset_name
            hash_collision_exists = False  # Never collides — hash includes dataset_id
            name_collision_exists = name_collision_count > 0
            
            logger.debug(
                f"✅ [REGISTER_VALIDATE] All inputs valid - "
                f"train={len(train_data)}, val={len(validation_data)}, test={len(test_data)}"
            )
            
            # ========================================================================
            # STEP 1: Compress Datasets
            # ========================================================================
            logger.info(f"📦 [REGISTER_COMPRESS_START] Compressing {len(train_data)} train sequences...")
            try:
                train_compressed, ratio_train, size_train = CompressionHandler.compress(train_data)
                logger.debug(f"✅ [TRAIN_COMPRESS] ratio={ratio_train:.2f}x, size={size_train/1024/1024:.2f}MB")
            except Exception as e:
                raise RuntimeError(f"❌ Failed to compress train_data: {e}")
            
            try:
                val_compressed, ratio_val, size_val = CompressionHandler.compress(validation_data)
                logger.debug(f"✅ [VAL_COMPRESS] ratio={ratio_val:.2f}x, size={size_val/1024/1024:.2f}MB")
            except Exception as e:
                raise RuntimeError(f"❌ Failed to compress validation_data: {e}")
            
            try:
                test_compressed, ratio_test, size_test = CompressionHandler.compress(test_data)
                logger.debug(f"✅ [TEST_COMPRESS] ratio={ratio_test:.2f}x, size={size_test/1024/1024:.2f}MB")
            except Exception as e:
                raise RuntimeError(f"❌ Failed to compress test_data: {e}")
            
            # Compress labels (also reduces size)
            try:
                train_labels_compressed = pickle.dumps(train_labels, protocol=pickle.HIGHEST_PROTOCOL)
                val_labels_compressed = pickle.dumps(validation_labels, protocol=pickle.HIGHEST_PROTOCOL)
                test_labels_compressed = pickle.dumps(test_labels, protocol=pickle.HIGHEST_PROTOCOL)
                logger.debug(f"✅ [LABELS_SERIALIZE] {len(train_labels_compressed)/1024:.2f}KB train, {len(val_labels_compressed)/1024:.2f}KB val, {len(test_labels_compressed)/1024:.2f}KB test")
            except Exception as e:
                raise RuntimeError(f"❌ Failed to pickle labels: {e}")
            
            # ✅ Pickle targets directly (simple method, matches loader expectations)
            # IMPORTANT: Use pickle.dumps() NOT serialize_data() to avoid double-encoding issues
            # Loader expects: pickle.loads(targets_bytes) to work directly
            train_targets_compressed = None
            val_targets_compressed = None
            test_targets_compressed = None
            
            if train_targets:
                try:
                    # ✅ FIX: Use pickle.dumps directly, not serialize_data (which adds base64+compression)
                    train_targets_compressed = pickle.dumps(train_targets, protocol=pickle.HIGHEST_PROTOCOL)
                    target_sizes = {k: v.shape for k, v in train_targets.items()}
                    logger.info(f"✅ [TARGETS_SERIALIZE] Pickled train targets: {target_sizes}")
                    logger.debug(f"   Train targets size: {len(train_targets_compressed)/1024:.2f}KB")
                except Exception as e:
                    logger.error(f"❌ Failed to pickle train targets: {e}")
                    train_targets_compressed = None
            
            if validation_targets:
                try:
                    # ✅ FIX: Use pickle.dumps directly
                    val_targets_compressed = pickle.dumps(validation_targets, protocol=pickle.HIGHEST_PROTOCOL)
                    logger.info(f"✅ [TARGETS_SERIALIZE] Pickled validation targets: {len(validation_targets)} columns")
                    logger.debug(f"   Val targets size: {len(val_targets_compressed)/1024:.2f}KB")
                except Exception as e:
                    logger.error(f"❌ Failed to pickle validation targets: {e}")
                    val_targets_compressed = None
            
            if test_targets:
                try:
                    # ✅ FIX: Use pickle.dumps directly
                    test_targets_compressed = pickle.dumps(test_targets, protocol=pickle.HIGHEST_PROTOCOL)
                    logger.info(f"✅ [TARGETS_SERIALIZE] Pickled test targets: {len(test_targets)} columns")
                    logger.debug(f"   Test targets size: {len(test_targets_compressed)/1024:.2f}KB")
                except Exception as e:
                    logger.error(f"❌ Failed to pickle test targets: {e}")
                    test_targets_compressed = None
            
            # Calculate total sizes including targets
            target_sizes_total = sum([
                len(train_targets_compressed) if train_targets_compressed else 0,
                len(val_targets_compressed) if val_targets_compressed else 0,
                len(test_targets_compressed) if test_targets_compressed else 0
            ])
            
            total_original = (
                train_data.nbytes + validation_data.nbytes + test_data.nbytes +
                len(train_labels_compressed) + len(val_labels_compressed) + len(test_labels_compressed) +
                target_sizes_total
            )
            total_compressed = (
                len(train_compressed) + len(val_compressed) + len(test_compressed) +
                len(train_labels_compressed) + len(val_labels_compressed) + len(test_labels_compressed) +
                target_sizes_total
            )
            overall_ratio = total_original / total_compressed if total_compressed > 0 else 0
            
            logger.info(
                f"📊 [REGISTER_COMPRESS_DONE] {total_original / (1024*1024):.2f}MB → "
                f"{total_compressed / (1024*1024):.2f}MB (ratio: {overall_ratio:.2f}x)"
            )
            
            # Store in TIER 2 (DB)
            
            
            # Custom JSON encoder to handle non-serializable objects
            def safe_json_dumps(obj):
                """Safely serialize objects to JSON, handling enums and other non-serializable types."""
                if obj is None:
                    return None
                
                def json_serializer(o):
                    if isinstance(o, Enum):
                        return o.value
                    elif hasattr(o, '__dict__'):
                        # For custom objects, try to serialize their dict representation
                        return str(o)
                    elif hasattr(o, 'name'):
                        # For objects with a name attribute (like scalers)
                        return str(o)
                    else:
                        return str(o)
                
                try:
                    return json.dumps(obj, default=json_serializer)
                except Exception as e:
                    logger.warning(f"Failed to serialize object {type(obj)}: {e}, using string representation")
                    return json.dumps(str(obj))
            
            def safe_json_loads(json_str):
                """Safely deserialize JSON strings, handling None and already-parsed objects."""
                if not json_str:
                    return None
                if isinstance(json_str, (dict, list)):
                    return json_str
                try:
                    return json.loads(json_str)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Failed to parse JSON: {e}, returning as-is")
                    return json_str
            
            # Generate hash for output_targets — include dataset_id to ensure uniqueness per run
            output_targets_json = safe_json_dumps(output_targets)
            output_targets_hash = hashlib.sha256(
                (f"{dataset_id}:{output_targets_json or ''}").encode('utf-8')
            ).hexdigest()[:64]
            
            # ========================================================================
            # STEP 2: Prepare Database Parameters
            # ========================================================================
            logger.info(f"🗄️  [REGISTER_DB_PREPARE] Building INSERT query for ml_datasets...")
            
            # For 3D sequences (N, seq_len, features): feature_count = shape[2]
            # For 2D flat arrays (N, features): feature_count = shape[1]
            if len(train_data.shape) == 3:
                feature_count = train_data.shape[2]  # (N, seq_len, features)
            elif len(train_data.shape) == 2:
                feature_count = train_data.shape[1]
            else:
                feature_count = 1
            sample_count = train_data.shape[0]
            
            # ✅ CRITICAL FIX: Ensure feature_names matches actual data shape
            # Sometimes metadata has fewer feature names than actual features in data
            feature_names_from_metadata = metadata.get('feature_names', [])
            
            if len(feature_names_from_metadata) != feature_count:
                logger.warning(
                    f"⚠️ [FEATURE_COUNT_MISMATCH] Metadata has {len(feature_names_from_metadata)} feature_names "
                    f"but data has {feature_count} features. "
                    f"{'Padding' if len(feature_names_from_metadata) < feature_count else 'Truncating'} to match data."
                )
                
                if len(feature_names_from_metadata) < feature_count:
                    # Pad with generic names for missing features
                    feature_names_from_metadata = list(feature_names_from_metadata) + [
                        f"feature_{i}" for i in range(len(feature_names_from_metadata), feature_count)
                    ]
                else:
                    # Truncate if metadata has more names than features (shouldn't happen)
                    feature_names_from_metadata = feature_names_from_metadata[:feature_count]
            
            # Build parameter dict with extensive logging
            params = {
                'dataset_id': dataset_id,
                'session_id': self.session_id,
                'dataset_name': dataset_name,
                'source_step': source_step,
                'output_targets': output_targets_json,
                'output_targets_hash': output_targets_hash,
                # ✅ FIXED: Use corrected feature_names that matches actual data shape
                'feature_columns': safe_json_dumps(feature_names_from_metadata),
                'feature_count': feature_count,  # Always use actual count from data shape
                'sample_count': sample_count,
                'split_config': safe_json_dumps({
                    'train_size': len(train_data),
                    'validation_size': len(validation_data),
                    'test_size': len(test_data),
                    'split_method': 'sequential'
                }),
                'parent_dataset_id': parent_dataset_id,
                'feature_selection': safe_json_dumps(feature_selection),
                'metadata': safe_json_dumps(metadata),
                'train_data': train_compressed,
                'val_data': val_compressed,
                'test_data': test_compressed,
                'train_labels': train_labels_compressed,
                'val_labels': val_labels_compressed,
                'test_labels': test_labels_compressed,
                # ✅ NEW: Add target fields
                'train_targets': train_targets_compressed,
                'val_targets': val_targets_compressed,
                'test_targets': test_targets_compressed,
                'compression_type': "zstandard",
                'compression_ratio': overall_ratio,
                'uncompressed_size': total_original / (1024 * 1024),
                'compressed_size': total_compressed / (1024 * 1024),
                'target_metadata': safe_json_dumps({
                    "target_names": list(train_targets.keys()) if train_targets else [],
                    "target_types": {name: "regression" for name in (train_targets.keys() if train_targets else [])},
                    "class_mappings": {}
                })
            }
            
            logger.debug(f"📋 [REGISTER_DB_PARAMS] Keys: {list(params.keys())}")
            logger.debug(f"🔑 [REGISTER_DB_PARAMS_VALUES] dataset_id={params['dataset_id']}, session_id={params['session_id']}, dataset_name={params['dataset_name']}, source_step={params['source_step']}")
            logger.debug(f"📊 [REGISTER_DB_PARAMS_SIZE] feature_count={params['feature_count']}, sample_count={params['sample_count']}")
            
            # ========================================================================
            # STEP 2.5: HANDLE NAME COLLISION (auto-rename with version suffix)
            # ========================================================================
            if name_collision_exists:
                # Same name already exists in this session
                # Option 1: Auto-rename with version suffix (current behavior)
                # Option 2: Delete old and replace (if user wants to overwrite)
                
                # For ML preparation, we typically want to replace old datasets
                # Check if this is an ML preparation dataset
                if source_step == "ml_preparation":
                    # Delete the old dataset with the same name
                    logger.info(
                        f"🔄 [NAME_COLLISION] ML dataset '{original_dataset_name}' already exists. "
                        f"Deleting old version and replacing with new one..."
                    )
                    delete_query = text("""
                        DELETE FROM ml_datasets 
                        WHERE session_id = :session_id AND dataset_name = :dataset_name
                    """)
                    await self.db.execute(delete_query, {
                        'session_id': self.session_id,
                        'dataset_name': original_dataset_name
                    })
                    await self.db.commit()
                    logger.info(f"✅ [NAME_COLLISION] Deleted old dataset '{original_dataset_name}'")
                    # Keep the original name (no versioning needed)
                    dataset_name = original_dataset_name
                    params['dataset_name'] = dataset_name
                else:
                    # For other datasets (SNR, etc.), use versioning
                    version = name_collision_count + 1
                    dataset_name = f"{original_dataset_name}_v{version}"
                    params['dataset_name'] = dataset_name
                    logger.info(
                        f"🔄 [NAME_COLLISION] Dataset '{original_dataset_name}' already exists. "
                        f"Auto-renaming to '{dataset_name}'"
                    )
            
            query = text("""
                INSERT INTO ml_datasets (
                    dataset_id, session_id, dataset_name, source_step,
                    output_targets, output_targets_hash,
                    feature_columns, feature_count, sample_count,
                    split_config,
                    train_data_compressed, validation_data_compressed, test_data_compressed,
                    train_labels, validation_labels, test_labels,
                    train_targets, validation_targets, test_targets,
                    compression_type, compression_ratio, uncompressed_size_mb, compressed_size_mb,
                    parent_dataset_id, feature_selection, source_metadata, target_metadata
                ) VALUES (
                    :dataset_id, :session_id, :dataset_name, :source_step,
                    :output_targets, :output_targets_hash,
                    :feature_columns, :feature_count, :sample_count,
                    :split_config,
                    :train_data, :val_data, :test_data,
                    :train_labels, :val_labels, :test_labels,
                    :train_targets, :val_targets, :test_targets,
                    :compression_type, :compression_ratio, :uncompressed_size, :compressed_size,
                    :parent_dataset_id, :feature_selection, :metadata, :target_metadata
                )
            """)
            
            # ========================================================================
            # STEP 3: Execute Database INSERT
            # ========================================================================
            logger.info(f"🔄 [REGISTER_DB_EXECUTE] Executing INSERT for session={self.session_id}, dataset={dataset_name}...")
            try:
                result = await self.db.execute(query, params)
                logger.info(f"✅ [REGISTER_DB_SUCCESS] INSERT executed successfully")
                logger.debug(f"📍 [REGISTER_DB_RESULT] Result status: rowcount={result.rowcount if hasattr(result, 'rowcount') else 'N/A'}")
            except Exception as db_err:
                # Extract meaningful error message for frontend
                error_str = str(db_err)
                db_error_lower = error_str.lower()
                
                if "constraint" in db_error_lower:
                    if "unique" in db_error_lower:
                        user_msg = "Dataset with these parameters already exists. Please use a different name or configuration."
                    else:
                        user_msg = "Database constraint violated. Please check your input and try again."
                elif "database" in db_error_lower or "connection" in db_error_lower:
                    user_msg = "Database connection error. Please try again or contact support if problem persists."
                elif "violates" in db_error_lower:
                    user_msg = "Unable to save dataset: Conflict with existing data. Try with a different name."
                else:
                    first_line = error_str.split(chr(10))[0][:120]
                    user_msg = f"Failed to save dataset: {first_line}"
                
                logger.error(f"❌ [REGISTER_DB_FAILED] Database INSERT failed: {error_str[:200]}", exc_info=True)
                logger.error(f"📋 [REGISTER_DB_FAILED_PARAMS] dataset_id={params['dataset_id']}, session_id={params['session_id']}, dataset_name={params['dataset_name']}")
                raise RuntimeError(user_msg)
            
            
            # Load to TIER 1 cache
            self._load_to_tier1(
                dataset_id=dataset_id,
                train_data=train_data,
                validation_data=validation_data,
                test_data=test_data,
                train_labels=train_labels,
                validation_labels=validation_labels,
                test_labels=test_labels,
                metadata={
                    'dataset_id': dataset_id,
                    'dataset_name': dataset_name,
                    'source_step': source_step,
                    'parent_dataset_id': parent_dataset_id,
                    'output_targets': output_targets,
                    'feature_selection': feature_selection,
                    'metadata': metadata
                }
            )
            
            # Set TIER 0 pointer (current dataset)
            self._current_pointer_id = dataset_id
            
            logger.info(f"✓ Dataset {dataset_id} registered and cached")
            return True
        
        except Exception as e:
            logger.error(f"❌ Error registering dataset {dataset_id}: {e}", exc_info=True)
            logger.debug(f"📋 Dataset metadata: name={dataset_name}, source_step={source_step}, session={self.session_id}")
            return False
    
    # ========================================================================
    # TIER 0 Pointer Management
    # ========================================================================
    
    async def set_current_dataset_id(self, dataset_id: str) -> bool:
        """
        Set the current dataset pointer (TIER 0).
        
        This dataset will be used for model training.
        
        Args:
            dataset_id: Dataset to select for training
        
        Returns:
            True if successful
        """
        try:
            # Verify dataset exists
            from sqlalchemy import text
            query = text("SELECT dataset_id FROM ml_datasets WHERE dataset_id = :dataset_id AND session_id = :session_id")
            result = await self.db.execute(query, {'dataset_id': dataset_id, 'session_id': self.session_id})
            row = result.fetchone()
            
            if row is None:
                logger.error(f"Dataset {dataset_id} not found")
                return False
            
            self._current_pointer_id = dataset_id
            logger.info(f"Set TIER 0 pointer to {dataset_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error setting current dataset: {e}")
            return False
    
    async def get_current_dataset_id(self) -> Optional[str]:
        """Get current TIER 0 dataset pointer."""
        return self._current_pointer_id
    
    # ========================================================================
    # Dataset Querying
    # ========================================================================
    
    async def list_datasets(
        self,
        source_step: Optional[str] = None,
        parent_dataset_id: Optional[str] = None
    ) -> List[DatasetMetadata]:
        """
        List all datasets in session.
        
        Args:
            source_step: Filter by "snr_analysis" or "ml_preparation"
            parent_dataset_id: Filter by parent (for lineage)
        
        Returns:
            List of DatasetMetadata objects
        """
        try:
            from sqlalchemy import text
            
            query_str = "SELECT * FROM ml_datasets WHERE session_id = :session_id"
            params = {'session_id': self.session_id}
            
            if source_step:
                query_str += " AND source_step = :source_step"
                params['source_step'] = source_step
            
            if parent_dataset_id:
                query_str += " AND parent_dataset_id = :parent_dataset_id"
                params['parent_dataset_id'] = parent_dataset_id
            
            query_str += " ORDER BY created_at DESC"
            query = text(query_str)
            
            logger.info(f"🔵 [LIST_DATASETS_START] session_id={self.session_id}, source_filter={source_step}")
            logger.debug(f"📝 [LIST_DATASETS_QUERY] {query_str}")
            logger.debug(f"🔑 [LIST_DATASETS_PARAMS] {params}")
            
            try:
                result = await self.db.execute(query, params)
            except Exception as query_err:
                logger.error(f"❌ [LIST_DATASETS_QUERY_FAILED] SQL execution failed: {query_err}", exc_info=True)
                return []
            
            try:
                rows = result.fetchall()
            except Exception as fetch_err:
                logger.error(f"❌ [LIST_DATASETS_FETCH_FAILED] Failed to fetch results: {fetch_err}", exc_info=True)
                return []
            
            logger.info(f"✅ [LIST_DATASETS_FETCH] Retrieved {len(rows)} rows")
            
            if len(rows) == 0:
                logger.warning(
                    f"⚠️  [LIST_DATASETS_EMPTY] No datasets found - session={self.session_id}, "
                    f"source_step={source_step}. Check if datasets exist in DB."
                )
                return []
            
            logger.debug(f"📊 [LIST_DATASETS_PARSE] Parsing {len(rows)} rows...")

            import json
            
            def safe_json_loads(json_str):
                """
                Safely deserialize JSON strings, handling None and malformed JSON.
                ✅ FIXED: Also handles already-parsed dict/list objects (from SQLAlchemy deserialization)
                """
                if not json_str:
                    return None
                
                # ✅ NEW: Handle already-parsed objects (SQLAlchemy may deserialize automatically)
                if isinstance(json_str, (dict, list)):
                    return json_str
                
                # Original: Handle JSON strings
                try:
                    return json.loads(json_str)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"⚠️  [LIST_DATASETS_JSON_PARSE_FAIL] Failed to parse JSON: {e}")
                    return json_str
            
            result_list = []
            for i, row in enumerate(rows):
                try:
                    # ✅ CRITICAL FIX: Convert UUID to string and normalize output_targets
                    output_targets_parsed = safe_json_loads(row.output_targets)
                    
                    # If output_targets is a list, convert to dict {name: value}
                    if isinstance(output_targets_parsed, list):
                        output_targets_parsed = {target: None for target in output_targets_parsed}
                    
                    metadata = DatasetMetadata(
                        dataset_id=str(row.dataset_id),  # ✅ Convert UUID to string
                        session_id=row.session_id,
                        dataset_name=row.dataset_name,
                        source_step=row.source_step,
                        parent_dataset_id=row.parent_dataset_id,
                        output_targets=output_targets_parsed,  # ✅ Ensure dict format
                        feature_selection=safe_json_loads(row.feature_selection),
                        metadata=safe_json_loads(row.source_metadata),
                        compression_type=row.compression_type,
                        compression_ratio=row.compression_ratio,
                        uncompressed_size_mb=row.uncompressed_size_mb,
                        compressed_size_mb=row.compressed_size_mb,
                        target_metadata=safe_json_loads(row.target_metadata),  # ✅ NEW
                        created_at=row.created_at.isoformat() if row.created_at else None
                    )
                    result_list.append(metadata)
                    logger.debug(f"✅ [LIST_DATASETS_ROW_{i}] Parsed: {metadata.dataset_name}")
                except Exception as row_err:
                    logger.error(f"❌ [LIST_DATASETS_ROW_{i}_FAILED] {row_err}", exc_info=True)
                    continue
            
            logger.info(f"✅ [LIST_DATASETS_COMPLETE] Returned {len(result_list)}/{len(rows)} datasets")
            return result_list
        
        except Exception as e:
            logger.error(f"❌ [LIST_DATASETS_EXCEPTION] Error listing datasets: {e}", exc_info=True)
            logger.error(f"📋 [LIST_DATASETS_EXCEPTION_CONTEXT] session_id={self.session_id}, source_step={source_step}")
            return []
    
    async def get_dataset_lineage(self, dataset_id: str) -> List[DatasetMetadata]:
        """
        Get lineage chain (SNR → ML → ...future...).
        
        Shows full ancestry of a dataset.
        
        Args:
            dataset_id: Starting dataset
        
        Returns:
            List of ancestor datasets (newest to oldest)
        """
        try:
            from sqlalchemy import text
            
            lineage = []
            current_id = dataset_id
            
            import json
            
            def safe_json_loads(json_str):
                """
                Safely deserialize JSON strings, handling None and malformed JSON.
                ✅ FIXED: Also handles already-parsed dict/list objects (from SQLAlchemy deserialization)
                """
                if not json_str:
                    return None
                
                # ✅ NEW: Handle already-parsed objects (SQLAlchemy may deserialize automatically)
                if isinstance(json_str, (dict, list)):
                    return json_str
                
                # Original: Handle JSON strings
                try:
                    return json.loads(json_str)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Failed to parse JSON: {e}, returning string value")
                    return json_str
            
            while current_id:
                query = text("SELECT * FROM ml_datasets WHERE dataset_id = :dataset_id AND session_id = :session_id")
                result = await self.db.execute(query, {'dataset_id': current_id, 'session_id': self.session_id})
                row = result.fetchone()
                
                if row is None:
                    break
                
                # ✅ CRITICAL FIX: Convert UUID to string and normalize output_targets
                output_targets_parsed = safe_json_loads(row.output_targets)
                if isinstance(output_targets_parsed, list):
                    output_targets_parsed = {target: None for target in output_targets_parsed}
                
                lineage.append(
                    DatasetMetadata(
                        dataset_id=str(row.dataset_id),  # ✅ Convert UUID to string
                        session_id=row.session_id,
                        dataset_name=row.dataset_name,
                        source_step=row.source_step,
                        parent_dataset_id=row.parent_dataset_id,
                        output_targets=output_targets_parsed,  # ✅ Ensure dict format
                        feature_selection=safe_json_loads(row.feature_selection),
                        metadata=safe_json_loads(row.source_metadata),
                        compression_type=row.compression_type,
                        compression_ratio=row.compression_ratio,
                        uncompressed_size_mb=row.uncompressed_size_mb,
                        compressed_size_mb=row.compressed_size_mb,
                        target_metadata=safe_json_loads(row.target_metadata),  # ✅ NEW
                        created_at=row.created_at.isoformat() if row.created_at else None
                    )
                )
                
                current_id = row.parent_dataset_id
            
            return lineage
        
        except Exception as e:
            logger.error(f"Error getting dataset lineage: {e}")
            return []
    
    # ========================================================================
    # Deletion & Cleanup
    # ========================================================================
    
    async def delete_dataset(self, dataset_id: str) -> bool:
        """
        Delete a dataset (both TIER 1 and TIER 2).
        
        Args:
            dataset_id: Dataset to delete
        
        Returns:
            True if successful
        """
        try:
            # Remove from TIER 1 if present
            if dataset_id in self._tier1_cache:
                data = self._tier1_cache.pop(dataset_id, {})
                del data  # Explicit cleanup
            
            if dataset_id in self._tier1_timestamps:
                del self._tier1_timestamps[dataset_id]
            
            # Remove from TIER 2 (DB)
            from sqlalchemy import text
            query = text("DELETE FROM ml_datasets WHERE dataset_id = :dataset_id AND session_id = :session_id")
            await self.db.execute(query, {'dataset_id': dataset_id, 'session_id': self.session_id})
            
            # Clear pointer if current
            if self._current_pointer_id == dataset_id:
                self._current_pointer_id = None
            
            logger.info(f"Deleted dataset {dataset_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error deleting dataset: {e}")
            return False
    
    # ========================================================================
    # Statistics
    # ========================================================================
    
    async def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics for monitoring.
        
        Returns:
            Dict with cache stats, DB stats, etc.
        """
        try:
            from sqlalchemy import text
            query = text("""
                SELECT 
                    COUNT(*) as total_datasets,
                    SUM(compressed_size_mb) as total_stored_mb,
                    SUM(uncompressed_size_mb) as total_original_mb,
                    AVG(compression_ratio) as avg_compression_ratio
                FROM ml_datasets
                WHERE session_id = :session_id
            """)
            
            result = await self.db.execute(query, {'session_id': self.session_id})
            row = result.fetchone()
            
            return {
                "tier0_pointer": self._current_pointer_id,
                "tier1_cache_size": len(self._tier1_cache),
                "tier1_max": self.TIER1_MAX_DATASETS,
                "tier1_ttl_minutes": self.TIER1_TTL_MINUTES,
                "tier2_total_datasets": row['total_datasets'] or 0,
                "tier2_stored_mb": float(row['total_stored_mb'] or 0),
                "tier2_original_mb": float(row['total_original_mb'] or 0),
                "tier2_avg_compression": float(row['avg_compression_ratio'] or 0)
            }
        
        except Exception as e:
            logger.error(f"Error getting registry stats: {e}")
            return {}
