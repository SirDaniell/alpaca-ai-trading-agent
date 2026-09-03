import logging
import asyncio
import json
import gc
import random
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from tqdm import tqdm
from fastapi.concurrency import run_in_threadpool

from app.core.analysis.trading.signal_generator import _analyze_post_interaction_movement
from app.core.data.serializers import to_serializable, ParallelSerializer
from app.core.processing.progress_reporter import ProgressEvent, ProgressReporter, ThrottlingStrategy

# Alias for backward compatibility with training_dataset.py imports
_to_serializable = to_serializable

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class ScalerType(Enum):
    """Supported scaler types for data normalization."""

    MINMAX = "minmax"
    STANDARD = "standard"
    ROBUST = "robust"
    NONE = "none"


class PassThroughScaler:
    """A dummy scaler that passes features through unchanged."""
    def fit(self, X, y=None):
        import numpy as np
        self.n_features_in_ = X.shape[1] if hasattr(X, 'shape') else len(X[0])
        self.scale_ = np.ones(self.n_features_in_)
        self.data_min_ = np.zeros(self.n_features_in_)
        self.data_max_ = np.ones(self.n_features_in_)
        return self
    
    def transform(self, X):
        return X
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return X
    
    def inverse_transform(self, X):
        return X


class MultiPartitionScaler:
    """Dynamically scales price channels, difference metrics and other features with custom configured pipelines."""
    def __init__(self, price_scaler_type: str, diff_scaler_type: str, other_scaler_type: str, columns_to_scale: List[str]):
        self.price_scaler_type = price_scaler_type
        self.diff_scaler_type = diff_scaler_type
        self.other_scaler_type = other_scaler_type
        self.columns_to_scale = list(columns_to_scale)
        
        self.price_cols = []
        self.diff_cols = []
        self.other_cols = []
        
        self.price_scaler = None
        self.diff_scaler = None
        self.other_scaler = None

    def _get_scaler_instance(self, scaler_type: str):
        if scaler_type == "minmax":
            return MinMaxScaler()
        elif scaler_type == "standard":
            return StandardScaler()
        elif scaler_type == "robust":
            return RobustScaler()
        else:
            return PassThroughScaler()

    def partition_columns(self, X_df: pd.DataFrame):
        """Partition columns dynamically into Price, Diff, and Other features based on statistics and heuristics."""
        self.price_cols = []
        self.diff_cols = []
        self.other_cols = []
        
        price_keywords = [
            'close', 'open', 'high', 'low', 'price', 'vwap', 'sma', 'ema', 
            'wma', 'hma', 'bband', 'upper', 'lower', 'middle', 'channel', 
            'pivot', 'sup', 'res', 'support', 'resistance', 'level', 
            'highs', 'lows', 'target', 'mid', 'volume', 'tickvolume', 'spread', 'atr'
        ]
        
        diff_keywords = [
            'diff', 'osc', 'macd', 'rsi', 'momentum', 'change', 'velocity',
            'roc', 'speed', 'slope', 'delta', 'return', 'derivative', 'signal',
            'csm_',      # Currency Strength Matrix — bounded [-1, +1] oscillators
        ]
        
        for col in self.columns_to_scale:
            if col not in X_df.columns:
                continue
                
            col_lower = col.lower()
            
            # 1. Price Channels Heuristic
            is_price = any(kw in col_lower for kw in price_keywords)
            
            # 2. Diff/Oscillator Heuristic
            is_diff = any(kw in col_lower for kw in diff_keywords)
            
            # Fallback dynamic checks using dataframe statistics if available
            col_min = float(X_df[col].min()) if not X_df[col].empty else 0.0
            col_max = float(X_df[col].max()) if not X_df[col].empty else 0.0
            
            # If the column spans across zero with a sign relationship (+/-), classify as diff/oscillator
            if col_min < -1e-5 and col_max > 1e-5 and not is_price:
                is_diff = True
                
            if is_price and not is_diff:
                self.price_cols.append(col)
            elif is_diff:
                self.diff_cols.append(col)
            else:
                self.other_cols.append(col)
                
        logger.info(f"📊 [MultiPartitionScaler] Partitioned {len(self.columns_to_scale)} columns:")
        logger.info(f"   ├─ Price Channels ({len(self.price_cols)} cols): {self.price_cols[:5]}...")
        logger.info(f"   ├─ Diff/Oscillators ({len(self.diff_cols)} cols): {self.diff_cols[:5]}...")
        logger.info(f"   └─ Other Features ({len(self.other_cols)} cols): {self.other_cols[:5]}...")

    def fit(self, X, y=None):
        # Determine columns partitions using X as a DataFrame
        if isinstance(X, pd.DataFrame):
            self.partition_columns(X)
        else:
            # Fallback if numpy array is passed (flatten sequence scaling)
            # Reconstruct dummy DataFrame to reuse partitioning
            X_df = pd.DataFrame(X, columns=self.columns_to_scale[:X.shape[1]])
            self.partition_columns(X_df)
            
        # Instantiate scalers
        self.price_scaler = self._get_scaler_instance(self.price_scaler_type)
        self.diff_scaler = self._get_scaler_instance(self.diff_scaler_type)
        self.other_scaler = self._get_scaler_instance(self.other_scaler_type)
        
        # Fit partitioned scalers
        if isinstance(X, pd.DataFrame):
            if self.price_cols:
                self.price_scaler.fit(X[self.price_cols])
            if self.diff_cols:
                self.diff_scaler.fit(X[self.diff_cols])
            if self.other_cols:
                self.other_scaler.fit(X[self.other_cols])
        else:
            # NumPy path (flat sequence transform support)
            # Match columns by index
            price_indices = [self.columns_to_scale.index(c) for c in self.price_cols if c in self.columns_to_scale]
            diff_indices = [self.columns_to_scale.index(c) for c in self.diff_cols if c in self.columns_to_scale]
            other_indices = [self.columns_to_scale.index(c) for c in self.other_cols if c in self.columns_to_scale]
            
            if price_indices:
                self.price_scaler.fit(X[:, price_indices])
            if diff_indices:
                self.diff_scaler.fit(X[:, diff_indices])
            if other_indices:
                self.other_scaler.fit(X[:, other_indices])
                
        return self

    def transform(self, X):
        X_out = X.copy()
        
        if isinstance(X, pd.DataFrame):
            if self.price_cols:
                X_out[self.price_cols] = self.price_scaler.transform(X[self.price_cols])
            if self.diff_cols:
                X_out[self.diff_cols] = self.diff_scaler.transform(X[self.diff_cols])
            if self.other_cols:
                X_out[self.other_cols] = self.other_scaler.transform(X[self.other_cols])
        else:
            # NumPy path
            price_indices = [self.columns_to_scale.index(c) for c in self.price_cols if c in self.columns_to_scale]
            diff_indices = [self.columns_to_scale.index(c) for c in self.diff_cols if c in self.columns_to_scale]
            other_indices = [self.columns_to_scale.index(c) for c in self.other_cols if c in self.columns_to_scale]
            
            if price_indices:
                X_out[:, price_indices] = self.price_scaler.transform(X[:, price_indices])
            if diff_indices:
                X_out[:, diff_indices] = self.diff_scaler.transform(X[:, diff_indices])
            if other_indices:
                X_out[:, other_indices] = self.other_scaler.transform(X[:, other_indices])
                
        return X_out

    def inverse_transform(self, X):
        X_out = X.copy()
        
        if isinstance(X, pd.DataFrame):
            if self.price_cols:
                X_out[self.price_cols] = self.price_scaler.inverse_transform(X[self.price_cols])
            if self.diff_cols:
                X_out[self.diff_cols] = self.diff_scaler.inverse_transform(X[self.diff_cols])
            if self.other_cols:
                X_out[self.other_cols] = self.other_scaler.inverse_transform(X[self.other_cols])
        else:
            # NumPy path
            price_indices = [self.columns_to_scale.index(c) for c in self.price_cols if c in self.columns_to_scale]
            diff_indices = [self.columns_to_scale.index(c) for c in self.diff_cols if c in self.columns_to_scale]
            other_indices = [self.columns_to_scale.index(c) for c in self.other_cols if c in self.columns_to_scale]
            
            if price_indices:
                X_out[:, price_indices] = self.price_scaler.inverse_transform(X[:, price_indices])
            if diff_indices:
                X_out[:, diff_indices] = self.diff_scaler.inverse_transform(X[:, diff_indices])
            if other_indices:
                X_out[:, other_indices] = self.other_scaler.inverse_transform(X[:, other_indices])
                
        return X_out


class SplitStrategy(Enum):
    """Data splitting strategies."""

    RANDOM = "random"
    SEQUENTIAL = "sequential"
    STRATIFIED = "stratified"

# Stage definitions - Standardized professional pipeline (Split -> Scale -> Generate)
PREPARATION_STAGES = {
    'validating': {'start': 0, 'end': 10, 'label': 'Validating data structure...'},
    'enriching': {'start': 10, 'end': 20, 'label': 'Enriching with target returns...'},
    'scaling': {'start': 20, 'end': 35, 'label': 'Standardized feature scaling (dataframe-level)...'},
    'generating': {'start': 35, 'end': 85, 'label': 'Generating final sequences for each split...'},
    'analyzing': {'start': 85, 'end': 92, 'label': 'Analyzing class imbalance...'},
    'finalizing': {'start': 92, 'end': 100, 'label': 'Calculating metrics...'}
}


@dataclass
class DatasetConfig:
    """Configuration for dataset preparation."""

    # Sequence parameters
    sequence_length: int = 60
    prediction_length: int = 7
    
    # Normalization window (should match or be less than sequence_length)
    # This defines the rolling window for structural range calculation
    # Default: 4x sequence_length for statistical stability, but capped at sequence_length for consistency
    rolling_window: int = 60  # Match sequence_length by default (will be synced in __post_init__)

    # Dataset tracking/naming
    dataset_name: str = "ml_prep_default"  # User-friendly name for storing results
    
    # Signal detection
    signal_column_prefix: str = "Signal_"
    target_columns: List[str] = field(default_factory=list)

    # Columns to exclude from features
    exclude_columns: List[str] = field(default_factory=list)

    # Scaling configuration
    scaler_type: ScalerType = ScalerType.ROBUST
    scaler_save_path: Optional[str] = "dataset_scaler.joblib"
    scaler_load_path: Optional[str] = None
    
    # Dynamic Scaler Partitioning configurations
    price_scaler_type: str = "none"
    diff_scaler_type: str = "none"
    other_scaler_type: str = "minmax"  # Default to minmax for non-price/diff features, but can be overridden by frontend UI
    
    # NEW: Frontend UI fields for scaler management
    save_scaler: bool = True  # Whether to save the scaler after fitting
    scaler_filename: str = "dataset_scaler.joblib"  # Filename for saving scaler

    # Split configuration
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    split_strategy: SplitStrategy = SplitStrategy.SEQUENTIAL
    random_seed: int = 42

    # Multi-target configuration
    include_classification: bool = True
    include_regression: bool = False
    include_sequence_prediction: bool = False
    
    # Advanced options
    handle_class_imbalance: bool = False
    shuffle_data: bool = True
    preserve_temporal_order: bool = False
    use_lazy_storage: bool = False  # If True, sequences are written to disk in chunks
    
    # Negative sampling (None class)
    negative_sampling_ratio: float = 1.0  # 1.0 = equal number of negative vs positive samples

    # Data leakage protection
    mask_future_signals: bool = True
    signal_leakage_buffer: int = 20  # Number of bars at the end of a sequence to zero out for signal columns
    exclude_signals: bool = False   # Completely exclude signal columns from features
    
    source_type: str = "enriched_df"  # 'enriched_df' or 'snr_unprocessed'
    input_source: str = "raw"  # 'raw' or 'snr' (frontend UI field)
    use_snr_dataset: bool = False  # Whether using SNR dataset
    
    # SNR-specific configuration
    selected_signal_types: List[str] = field(default_factory=list)  # Selected SNR signal types
    feature_selection_mode: str = "rich"  # 'minimal', 'movement', 'volume_patterns', 'rich', 'custom'
    custom_features: Optional[Dict[str, bool]] = None  # Custom feature selection for SNR

    # Safety checks
    drop_zeros: bool = True  # If True, sequences with zero targets (likely corrupted) are dropped

    # Advanced ML Targets (sequence-prediction mode only)
    # When True, calculates future-looking targets: post-interaction movement metrics
    # AND next-candle momentum indicator values. These are NEVER included in the feature
    # set — they are pure look-ahead ground-truth targets.
    prepare_advanced_ml_targets: bool = False
    # Horizon used for vectorised movement analysis (number of future bars to inspect)
    advanced_target_lookforward: int = 20

    # ── Forward Reversal Probability Labels ─────────────────────────────────
    # Activated automatically when prepare_advanced_ml_targets=True.
    # Controls _compute_forward_reversal_labels() in ml_dataset_preparation.py.
    reversal_n_future: int = 8          # How many bars ahead to measure reversal/continuation
    reversal_decay: float = 0.85        # Exponential decay weight per bar (bar t+1 weighted most)
    reversal_hold_threshold: float = 0.65  # reversal_prob must be >= this for reversal_held=1

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Align other_scaler_type with the primary scaler_type for backward compatibility in tests
        if self.other_scaler_type == "standard" and self.scaler_type != ScalerType.ROBUST:
            if hasattr(self.scaler_type, 'value'):
                self.other_scaler_type = self.scaler_type.value
            else:
                self.other_scaler_type = str(self.scaler_type)

        # Validate split ratios
        total_ratio = self.train_ratio + self.validation_ratio + self.test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total_ratio}. "
                f"(train: {self.train_ratio}, val: {self.validation_ratio}, test: {self.test_ratio})"
            )

        if self.sequence_length <= 0:
            raise ValueError(
                f"sequence_length must be positive, got {self.sequence_length}"
            )

        if self.prediction_length < 0:
            raise ValueError(
                f"prediction_length cannot be negative, got {self.prediction_length}"
            )


@dataclass
class DatasetMetrics:
    """Metrics and statistics about the prepared dataset."""

    total_sequences: int = 0
    signal_distribution: Dict[str, int] = field(default_factory=dict)
    class_imbalance_ratio: float = 0.0

    train_size: int = 0
    validation_size: int = 0
    test_size: int = 0

    feature_count: int = 0
    sequence_shape: Tuple = ()

    target_shapes: Dict[str, Tuple] = field(default_factory=dict)
    target_statistics: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary format."""
        return {
            "total_sequences": self.total_sequences,
            "signal_distribution": self.signal_distribution,
            "class_imbalance_ratio": self.class_imbalance_ratio,
            "split_sizes": {
                "train": self.train_size,
                "validation": self.validation_size,
                "test": self.test_size,
            },
            "feature_count": self.feature_count,
            "sequence_shape": self.sequence_shape,
            "target_shapes": self.target_shapes,
            "target_statistics": self.target_statistics,
        }




# ProgressTracker class removed in favor of unified ProgressReporter infrastructure.

    def set_stage(self, stage_name: str, total_stages: int):
        """Set current processing stage."""
        self.current_stage = stage_name
        self.total_stages = total_stages

    async def complete(self, result: dict, **kwargs):
        """
        Send completion message with final result - UNIFIED approach.

        Args:
            result: Final result dictionary to send to client
            **kwargs: Additional data to include in completion message
        """
        if not self.task_id:
            logger.info(f"Completion: Task completed successfully")
            return

        try:
            logger.info(f"📥 ML Completion: Starting for task {self.task_id}")
            
            # Store in progress_store for later retrieval
            if self.progress_store:
                logger.info(f"💾 Storing result in task_store...")
                import time
                start_time = time.time()
                
                self.progress_store.update_task(
                    self.task_id,
                    status="completed",
                    progress=100,
                    message="Dataset preparation complete",
                    result=result
                )
                
                store_time = time.time() - start_time
                logger.info(f" Stored in task_store in {store_time:.2f}s")
            else:
                logger.warning(f"⚠️ No progress_store - result will NOT be persisted!")

            # Send completion message
            if self.reporter:
                await self.reporter.update(ProgressEvent(
                    task_id=self.task_id,
                    progress=100,
                    message="Dataset preparation complete",
                    stage="complete",
                    status="complete",
                    extra={
                        "result_info": {
                            "download_available": True,
                            "download_endpoint": f"/api/analysis/tasks/{self.task_id}/download",
                            "total_sequences": result.get("metrics", {}).get("total_sequences", 0) if isinstance(result.get("metrics"), dict) else 0,
                            "splits_available": ["train", "validation", "test"]
                        },
                        "result": result
                    }
                ))

            logger.info(f" Task {self.task_id}: Completion finished successfully")

        except Exception as e:
            logger.error(f"Error in complete(): {e}", exc_info=True)

    async def error(self, message: str, **kwargs):
        """
        Send error message to client - UNIFIED approach.

        Args:
            message: Error message
            **kwargs: Additional data to include in error message
        """
        if not self.task_id:
            logger.error(f"Error: {message}")
            return

        try:
            error_data = {
                "type": "error",
                "progress": 0,
                "message": message,
                "stage": self.current_stage,
                "user_id": self.user_id,  #  Include user_id for routing
                **kwargs,
            }

            # Update progress store (same as ProcessingManager)
            if self.progress_store:
                self.progress_store.update_task(self.task_id, **error_data)
                logger.debug(f"📥 ML Error: Updated task_store for {self.task_id}")

            # Send via connection_manager.send_progress_update()
            if self.connection_manager:
                await self.connection_manager.send_progress_update(
                    self.task_id,
                    error_data,
                    user_id=self.user_id  # Pass user_id for routing
                )
                logger.info(f"📤 ML Error: Sent to user {self.user_id} via unified connection_manager")
            else:
                logger.warning(f"⚠️ ML Error: No connection_manager available, skipping WebSocket error")

            logger.error(f" Task {self.task_id}: Error occurred: {message}")

        except Exception as e:
            logger.error(f"Error sending error message: {e}", exc_info=True)


class MLDatasetPreparation:
    """
    Comprehensive ML dataset preparation class for time series sequential data.
    
    CORRECTED ARCHITECTURAL FLOW (prevents data leakage):
    1. Validate data structure and required columns
    2. Enrich data with target return columns
    3. Generate sequences from raw (unscaled) time series data
    4. Split sequences into train/validation/test sets
    5. Fit scaler ONLY on training sequences
    6. Transform all splits using the fitted scaler
    7. Analyze class imbalance
    8. Calculate final metrics

    KEY DESIGN PRINCIPLE:
    - Scaler is NEVER fitted on validation or test data
    - Prevents information leakage from future data into scaler statistics
    - Follows standard ML best practices
    - All splits use the same scaler object fitted on training data only

    Responsibilities:
    - Data validation and preprocessing
    - Target feature enrichment (returns, directions)
    - Sequence generation from time series data
    - Signal-based classification labeling
    - Multi-output target preparation
    - Train/validation/test splitting with various strategies
    - Feature scaling with correct data leakage prevention
    - Class imbalance detection and reporting
    - Comprehensive progress tracking for frontend

    Example:
        ```python
        # Initialize with configuration
        config = DatasetConfig(
            sequence_length=60,
            prediction_length=7,
            train_ratio=0.7,
            validation_ratio=0.15,
            test_ratio=0.15
        )

        prep = MLDatasetPreparation(
            data=df,
            config=config,
            task_id="task_123",
            reporter=reporter
        )

        # Prepare dataset using corrected pipeline
        dataset = await prep.prepare_dataset()

        # Access properly scaled splits (no data leakage)
        X_train, y_train = dataset['train']['sequences'], dataset['train']['labels']
        X_val, y_val = dataset['validation']['sequences'], dataset['validation']['labels']
        X_test, y_test = dataset['test']['sequences'], dataset['test']['labels']
        # Note: Validation and test are scaled using scaler fitted on training only
        ```
    """

    # Signal type mapping
    SIGNAL_MAPPING = {
        "bounce_support": 0,
        "bounce_resistance": 1,
        "breakout_support": 2,
        "breakout_resistance": 3,
        "buy_signal": 0,  # Map generic buy to 0
        "sell_signal": 1, # Map generic sell to 1
        "buy": 0,
        "sell": 1,
    }

    # Confirmed signal columns that have FORWARD-LOOKING BIAS (must be excluded from features)
    # These columns are populated by signal_generator.py after analyzing future price movement
    # They represent CONFIRMED signals only after future data is known
    #  SNR features (snr_*) are pure lookback → NO LEAKAGE
    #  These Signal_* are future-confirmed → EXCLUDE from features
    CONFIRMED_SIGNAL_COLUMNS = [
        "Signal_bounce_support",
        "Signal_bounce_resistance",
        "Signal_breakout_support",
        "Signal_breakout_resistance"
    ]

    def __init__(
        self,
        data: pd.DataFrame,
        config: DatasetConfig,
        task_id: Optional[str] = None,
        reporter: Optional[ProgressReporter] = None,
        scaler: Optional[Any] = None,
        injected_dataset: Optional[List[Dict]] = None,
        dataset_name: Optional[str] = None,
        skip_scaling: bool = False,
        is_pre_split: bool = False,
    ):
        """
        Initialize dataset preparation.

        Args:
            data: Input DataFrame with price data and signals
            config: Dataset configuration
            task_id: Optional task ID for progress tracking
            reporter: Optional ProgressReporter instance
            scaler: Optional pre-fitted scaler
            injected_dataset: Pre-formed sequences (e.g. from SNR)
            dataset_name: Custom name for this preparation experiment
            skip_scaling: If True, data is already scaled (skip scaling step)
            is_pre_split: If True, data is already a single split (don't split again)
        """
        self.data = data
        self.config = config
        self.task_id = task_id
        self.reporter = reporter
        self.injected_dataset = injected_dataset
        self.dataset_name = dataset_name
        self.skip_scaling = skip_scaling
        self.is_pre_split = is_pre_split

        self.scaled_data: Optional[pd.DataFrame] = None
        self.scaler = scaler  # Use externally provided scaler if available
        self.metrics = DatasetMetrics()

        # DYNAMIC SIGNAL MAPPING: Respect selected_signal_types from frontend
        self.signal_mapping = self.SIGNAL_MAPPING.copy()
        if self.config.selected_signal_types:
            for idx, s_type in enumerate(self.config.selected_signal_types):
                self.signal_mapping[s_type] = idx
            logger.info(f"📊 [MLPrep] Dynamic signal mapping initialized")

        # Data containers
        self.sequences: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.target_data: Dict[str, np.ndarray] = {}

        # signal feature analysis
        from app.core.analysis.trading.signal_generator import _calculate_basic_movement_metrics
        self._calculate_basic_movement_metrics = _calculate_basic_movement_metrics

        logger.info(f"Initialized MLDatasetPreparation with {len(self.data)} rows (skip_scaling={skip_scaling}, is_pre_split={is_pre_split})")

    async def _prepare_dataset(self):
        """Async generator for dataset preparation - yields metadata, splits, and completion.
        
        NOTE: I haven't worked on this funcion for a while,  bugs may arise as other architecture around it has evolved. 
        Please review carefully and test with various configurations. Or use the _execute_ml_with_splits() method in processing_manager.py 
        which has the most up-to-date flow and error handling.


        Proper ML pipeline (standardized):
        1. Validate data structure
        2. Enrich with target variables
        3. Identify features and targets for scaling
        4. Split DATAFRAME into train/val/test slices
        5. Fit scaler ONLY on training dataframe slice
        6. Transform ALL dataframe slices with fitted scaler (consistent scaling for sequences/targets)
        7. Generate sequences individually for each scaled split
        8. Analyze class imbalance
        9. Calculate final metrics
        10. Yield metadata, each split, and completion (never accumulates all splits in memory)
        """
        try:
            if self.reporter:
                await self.reporter.report_async(
                    progress=0,
                    message="Starting standardized dataset preparation...",
                    message2=f"Preparing '{self.dataset_name or 'unnamed'}' | Source: {'SNR Dataset' if self.injected_dataset else 'Raw DataFrame'}"
                )

            # HYBRID PATH: If injected_dataset is provided, skip generation and go straight to split/scale
            if self.injected_dataset:
                logger.info(f"🚀 Using HYBRID PATH for {len(self.injected_dataset)} injected sequences")
                splits = await self._process_injected_dataset()
            else:
                # Stage 1: Validate data
                await self._validate_data()

                # Stage 2: Enrich with targets (before scaling/generation)
                await self._enrich_with_targets()

                # Stage 3: Professional Scaling & Generation flow
                # Identify columns that need scaling (features and targets)
                self._identify_features()
                
                # Skip scaling if data is already scaled (pre-split mode)
                if self.skip_scaling:
                    logger.info(f"[MLPrep] Skipping scaling - data is already scaled")
                    # Create a single-split dict with the entire DataFrame
                    scaled_splits = {
                        "train": self.data.copy(),
                        "validation": pd.DataFrame(),
                        "test": pd.DataFrame()
                    }
                else:
                    # Perform split-then-scale on the dataframe
                    scaled_splits = await self._scale_dataframe_splits()
            
            # FIX #5: CRITICAL - Delete scaled_splits immediately after use (no longer needed)
            # scaled_splits holds 3 scaled DataFrames (~400 MB), only needed during sequence generation
            # NOTE: Keep self.data around until after _calculate_metrics() finishes
            
            # Stage 4: Collect splits from async generator (one at a time, but all in dict for analysis)
            # CRITICAL: We must collect all splits before calling imbalance/metrics analysis
            # This is safe - we're just moving accumulation from serialization to analysis phase
            splits = {}
            async for split_name, split_data in self._generate_sequences_from_splits(scaled_splits):
                splits[split_name] = split_data
                logger.info(f"📥 Collected {split_name} split: {len(split_data['sequences'])} sequences")
            
            # FIX #5: CRITICAL - Delete scaled_splits immediately after use (no longer needed)
            # scaled_splits holds 3 scaled DataFrames (~400 MB), only needed during sequence generation
            logger.info(f"🧹 Clearing scaled_splits (no longer needed)...")
            del scaled_splits
            gc.collect()

            # Stage 5: Analyze class imbalance (uses complete splits dict)
            # Flatten labels for imbalance analysis
            all_labels = []
            if "train" in splits and splits["train"]["labels"].size > 0: all_labels.append(splits["train"]["labels"])
            if "validation" in splits and splits["validation"]["labels"].size > 0: all_labels.append(splits["validation"]["labels"])
            if "test" in splits and splits["test"]["labels"].size > 0: all_labels.append(splits["test"]["labels"])
            
            # FIX #6: Use local labels_array variable instead of storing on self
            # This prevents keeping the concatenated labels array in memory longer than needed
            labels_array = np.concatenate(all_labels) if all_labels else np.array([])
            self.labels = labels_array  # Store for metrics calculation only
            imbalance_analysis = await self._analyze_class_imbalance()
            logger.info(f" Class imbalance analysis complete")

            # Stage 6: Calculate metrics (requires all splits and labels)
            await self._calculate_metrics(splits)
            logger.info(f" Metrics calculated for all splits")
            
            # FIX #8: Delete original data NOW (after metrics calculation is done)
            # self.data was needed by _calculate_metrics, but no longer needed after this point
            logger.info(f"🧹 Clearing original data (metrics now complete)...")
            del self.data
            gc.collect()
            
            # FIX #6: Delete labels_array immediately after imbalance/metrics are done (no longer needed)
            logger.info(f"🧹 Clearing labels_array (no longer needed after analysis)...")
            del labels_array
            gc.collect()

            # ===== YIELD METADATA FIRST (before any heavy serialization) =====
            logger.info(f"📤 Yielding metadata to client...")
            yield {
                "type": "metadata",
                "task_id": self.task_id,
                "total_sequences": sum(len(splits[s]["sequences"]) for s in splits if splits[s].get("sequences") is not None and hasattr(splits[s]["sequences"], "size") and splits[s]["sequences"].size > 0),
                "feature_count": len(self.feature_cols) if hasattr(self, 'feature_cols') else 0,
                "splits": list(splits.keys()),
                "sequence_length": self.config.sequence_length,
                "prediction_length": self.config.prediction_length,
                "message": "Metadata ready, split downloads beginning..."
            }
            logger.info(f" Metadata yielded")

           
            split_counts = {}

            for split_name in ["train", "validation", "test"]:
                if split_name not in splits:
                    logger.warning(f"⚠️ Split '{split_name}' not found, skipping")
                    continue

                split_data = splits[split_name]
                sequences = split_data.get("sequences")
                seq_count = len(sequences) if sequences is not None else 0

                if seq_count == 0:
                    logger.warning(f"⚠️ Split '{split_name}' has no sequences, skipping")
                    continue

                split_counts[split_name] = seq_count
                logger.info(f"📤 Yielding {split_name} split ({seq_count} sequences) as raw numpy...")
                # Yield numpy arrays directly — no serialization overhead
                yield {
                    "type": "split",
                    "name": split_name,
                    "count": seq_count,
                    "data": split_data  # raw dict with numpy arrays
                }
                logger.info(f" {split_name} split yielded")

            # Delete splits dict immediately after all are yielded
            logger.info(f"🧹 Clearing splits dict after all yields (no longer needed)...")
            del splits
            gc.collect()

            # ===== FIX D: Metadata is just plain dicts/lists — no ParallelSerializer needed =====
            # These are tiny compared to sequence arrays. Use to_serializable() directly (fast, no pickle overhead here).
            logger.info(f"🔄 Building completion metadata (lightweight)...")
            feature_cols = self.feature_cols if hasattr(self, 'feature_cols') else []

            # ===== YIELD COMPLETION WITH PLAIN METADATA =====
            logger.info(f"🔄 Preparing completion message...")
            yield {
                "type": "completion",
                "task_id": self.task_id,
                "status": "completed",
                "message": "Dataset preparation complete",
                "metrics": self.metrics.to_dict(),
                "imbalance_analysis": imbalance_analysis,
                "config": self.config.__dict__ if hasattr(self.config, '__dict__') else {},
                "feature_names": feature_cols,
                "target_names": self.config.target_columns,
                "dataset_info": {
                    "feature_cols": feature_cols,
                    "target_cols": self.config.target_columns,
                    "sequence_length": self.config.sequence_length,
                    "prediction_length": self.config.prediction_length,
                    "feature_count": len(feature_cols),
                    "signal_types": list(self.metrics.signal_distribution.keys()),
                    # Named feature group index arrays — use to build dedicated sub-encoders.
                    # e.g. csm_indices = dataset_info['feature_groups'].get('csm', [])
                    #       csm_input = sequences[:, :, csm_indices]  (batch, seq_len, 6)
                    "feature_groups": getattr(self, '_feature_groups', {}),
                },
                "scaler_path": self.config.scaler_save_path,
                "split_counts": split_counts
            }
            logger.info(f" Completion yielded - dataset preparation finished")

        except Exception as e:
            error_msg = f"Dataset preparation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            # Send error message to client via WebSocket
            if self.reporter:
                await self.reporter.error(self.task_id, error_msg)
            # Yield error message instead of raising (since we're an async generator)
            yield {
                "type": "error",
                "task_id": self.task_id,
                "message": error_msg,
                "status": "failed"
            }
            logger.error(f" Dataset preparation failed - error yielded to client")

    async def _process_injected_dataset(self) -> Dict[str, Any]:
        """
        Process pre-formed sequences and targets (Hybrid Path).
        Useful for SNR datasets that already contain sequence lookbacks.
        """
        if self.reporter:
            await self.reporter.report_async(
                progress=25,
                message="Extracting injected sequences and targets...",
                message2=f"Loaded {len(self.injected_dataset)} pre-formed records."
            )

        n_samples = len(self.injected_dataset)
        train_idx = int(n_samples * self.config.train_ratio)
        val_idx = train_idx + int(n_samples * self.config.validation_ratio)

        # 1. Split indices
        train_data = self.injected_dataset[:train_idx]
        val_data = self.injected_dataset[train_idx:val_idx]
        test_data = self.injected_dataset[val_idx:]

        if self.reporter:
            await self.reporter.report_async(
                progress=35,
                message="Structuring splits for scaling...",
                message2=f"Splits: Train({len(train_data)}), Val({len(val_data)}), Test({len(test_data)})"
            )

        # 2. Extract into numpy arrays
        splits = {
            "train": self._extract_injected_data(train_data),
            "validation": self._extract_injected_data(val_data),
            "test": self._extract_injected_data(test_data)
        }

        # 3. Identify features from first sample
        if len(train_data) > 0 and len(train_data[0]["sequence"]) > 0:
            sample_seq_df = pd.DataFrame(train_data[0]["sequence"])
            self.feature_cols = list(sample_seq_df.columns)
            logger.info(f" Identified {len(self.feature_cols)} features from injected sequences")

        # 4. Professional Scaling (Fit on Train, Transform All)
        if self.reporter:
            await self.reporter.report_async(
                progress=50,
                message="Applying feature scaling to injected sequences...",
                message2=f"Strategy: Fit on {len(train_data)} Training samples."
            )
        
        splits = self._scale_injected_splits(splits)

        return splits

    def _extract_injected_data(self, data: List[Dict]) -> Dict[str, Any]:
        """Extract List[Dict] sequences/targets into standardized numpy dict."""
        if not data:
            return {"sequences": np.array([]), "labels": np.array([]), "targets": {}}

        sequences = []
        labels = []
        
        # Determine target mapping
        target_col = self.config.target_columns[0] if self.config.target_columns else "type"

        for item in data:
            # Sequence: (L, F)
            seq_df = pd.DataFrame(item["sequence"])
            sequences.append(seq_df.values)
            
            # Label: use SNR signal type or specific column
            label_val = 4  # Default: No signal
            if "targets" in item:
                # SNR specific: map signal type
                signal_type = item["targets"].get("type")
                label_val = self.SIGNAL_MAPPING.get(signal_type, 4)
            
            labels.append(label_val)

        return {
            "sequences": np.array(sequences),
            "labels": np.array(labels),
            "targets": {} # Not deeply used for metrics yet in this path
        }

    def _scale_injected_splits(self, splits: Dict[str, Any]) -> Dict[str, Any]:
        """Scale 3D numpy sequences correctly.
        
        Uses self.scaler if provided (externally fitted), otherwise fits a new one.
        """
        train_seq = splits["train"]["sequences"]
        if train_seq.size == 0:
            return splits

        # Flatten for scaling: (N, L, F) -> (N*L, F)
        n, l, f = train_seq.shape
        train_flat = train_seq.reshape(-1, f)

        # Use provided scaler or fit a new one
        if self.scaler is None:
            # Fit Scaler on training data
            logger.info(f"[Injected] Fitting new dynamic partitioned MultiPartitionScaler on training sequences")
            self.scaler = MultiPartitionScaler(
                price_scaler_type=getattr(self.config, 'price_scaler_type', 'none'),
                diff_scaler_type=getattr(self.config, 'diff_scaler_type', 'robust'),
                other_scaler_type=getattr(self.config, 'other_scaler_type', 'standard'),
                columns_to_scale=self.columns_to_scale if hasattr(self, 'columns_to_scale') and self.columns_to_scale else [f"feature_{i}" for i in range(f)]
            )
            self.scaler.fit(train_flat)
            logger.info(f"[Injected] MultiPartitionScaler fitted on {n} sequences with {f} features")
        else:
            logger.info(f"[Injected] Using externally provided scaler (type: {type(self.scaler).__name__})")

        # Transform all splits
        for name in ["train", "validation", "test"]:
            seq = splits[name]["sequences"]
            if seq.size > 0:
                sn, sl, sf = seq.shape
                seq_flat = seq.reshape(-1, sf)
                scaled_flat = self.scaler.transform(seq_flat)
                splits[name]["sequences"] = scaled_flat.reshape(sn, sl, sf)

        return splits

    async def _validate_data(self):
        """Validate input data structure and required columns."""
        stage_info = PREPARATION_STAGES['validating']
        if self.reporter:
            await self.reporter.report_async(
                progress=stage_info['start'],
                message=stage_info['label'],
                message2="Checking required columns: open, high, low, close, volume"
            )

        if self.data is None or len(self.data) == 0:
            raise ValueError("Input data is empty or None")

        # Check for required columns (case-insensitive)
        required_cols = ["open", "high", "low", "close", "volume"]
        available_cols_lower = [col.lower() for col in self.data.columns]

        missing_cols = [col for col in required_cols if col not in available_cols_lower]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Available columns: {list(self.data.columns)}"
            )

        # Check for sufficient data points relative to sequencing config
        min_required = self.config.sequence_length + self.config.prediction_length
        if len(self.data) < min_required:
            raise ValueError(
                f"Insufficient data points ({len(self.data)}) for sequence generation. "
                f"Need at least {min_required} rows (sequence_length={self.config.sequence_length} + "
                f"prediction_length={self.config.prediction_length})."
            )

        # Check for signal columns - EXPLICIT DETECTION
        # CONFIRMED Signal_* columns (future-looking, will be masked in sequences)
        confirmed_signal_cols = [col for col in self.data.columns if col in self.CONFIRMED_SIGNAL_COLUMNS]
        
        # SNR feature columns (pure lookback, safe to use as features)
        snr_feature_cols = [col for col in self.data.columns if col.startswith("snr_") and col != "snr_signals"]
        
        # Raw signal type columns (if present)
        raw_signal_cols = [col for col in self.data.columns if col in self.SIGNAL_MAPPING.keys()]
        
        # Only count CONFIRMED signal columns for validation (not SNR features)
        signal_cols_for_validation = confirmed_signal_cols + raw_signal_cols

        if not signal_cols_for_validation and not snr_feature_cols:
            raise ValueError(
                f"No signal columns or SNR features found. Expected:"
                f"\n  - Confirmed signals: Signal_bounce_*, Signal_breakout_*"
                f"\n  - SNR features: snr_dist_to_nearest_*, snr_in_zone, snr_num_levels_*"
                f"\n  - Raw signal types: bounce_support, breakout_support, etc."
                f"\nAvailable: {list(self.data.columns)}"
            )

        logger.info(
            f"Validation passed. Found {len(confirmed_signal_cols)} signal columns: {confirmed_signal_cols}"
        )
        if snr_feature_cols:
            logger.info(
                f"  + {len(snr_feature_cols)} SNR feature columns (pure lookback): {snr_feature_cols}"
            )
        if self.reporter:
            await self.reporter.report_async(
                progress=stage_info['end'],
                message=f"{stage_info['label']} ✓",
                message2=f"Found {len(confirmed_signal_cols)} signal columns + {len(snr_feature_cols)} SNR features"
            )

    async def _enrich_with_targets(self):
        """
        Enrich data with target return and direction columns if they are missing.
        Uses look-ahead calculations based on the 'close' price.

        These indicators are already calculated by technical_indicators.py

        When `prepare_advanced_ml_targets` is enabled (sequence-prediction mode),
        also calculates:
          - Post-interaction movement metrics (max-favorable-excursion, drawdown, etc.)
          - Next-candle momentum indicator values (MOM_t+1, MR_t+1, TF_t+1, RSI+1, …)
        All advanced targets are prefixed with 'adv_target_' so the feature
        identification pass can automatically exclude them from the input feature set.
        """
        stage_info = PREPARATION_STAGES['enriching']
        if self.reporter:
            await self.reporter.report_async(
                progress=stage_info['start'],
                message=stage_info['label'],
                message2="Calculating target returns for regression task..."
            )

        # Identify the base column for enrichment (e.g. 'Close')
        # Priority: 1. First target column from config, 2. Case-insensitive 'Close'
        primary_target = self.config.target_columns[0] if self.config.target_columns else 'Close'
        close_col = next((col for col in self.data.columns if col.lower() == primary_target.lower()), None)
        
        # Fallback to any 'close' if primary target not found or not in data
        if not close_col:
            close_col = next((col for col in self.data.columns if col.lower() == 'close'), None)
        
        if not close_col:
            logger.warning("No base column (Close/%s) found for target enrichment.", primary_target)
            return

        logger.info(f"Enriching dataset with targets using base column '{close_col}'")

        # Map target names to their look-ahead periods
        # We handle the default targets: Next_Day_Return, Next_3_Day_Return, Next_5_Day_Return
        target_mapping = {
            "Next_Day_Return": 1,
            "Next_3_Day_Return": 3,
            "Next_5_Day_Return": 5
        }

        # ALWAYS generate default targets (not from config, which may only have 'Close')
        # Config target_columns often contains just the base column ('Close'), not the target names
        # So we explicitly generate all standard targets for ML training
        generated_count = 0
        for target_col, period in target_mapping.items():
            if target_col in self.data.columns:
                # Skip if already exists but log it
                logger.debug(f"Target '{target_col}' already exists, skipping")
                continue
            
            try:
                # Calculate future return: (P_{t+n} - P_t) / P_t
                self.data[target_col] = (
                    self.data[close_col].shift(-period) - self.data[close_col]
                ) / self.data[close_col]
                logger.info(f" Generated target '{target_col}' (look-ahead={period} bars)")
                generated_count += 1
            except Exception as e:
                logger.error(f" Failed to generate '{target_col}': {e}")
        
        logger.info(f" Generated {generated_count} default targets: {list(target_mapping.keys())}")
        
        # ALSO process any additional targets from config (for custom targets)
        for target_col in self.config.target_columns:
            if target_col in self.data.columns:
                # Already exists or already generated
                continue

            if target_col in target_mapping:
                # Already handled above
                continue
            
            elif target_col == "Next_Day_Direction":
                # Direction is 1 if Next_Day_Return > 0, else 0
                if "Next_Day_Return" in self.data.columns:
                    self.data["Next_Day_Direction"] = (self.data["Next_Day_Return"] > 0).astype(int)
                else:
                    next_return = (
                        self.data[close_col].shift(-1) - self.data[close_col]
                    ) / self.data[close_col]
                    self.data["Next_Day_Direction"] = (next_return > 0).astype(int)
                logger.info("Calculated target 'Next_Day_Direction'")

        # Fill NaNs created by shift at the end of the dataset
        self.data.fillna(0.0, inplace=True)
        
        # Add momentum indicator targets (simple shifts)
        # These indicators are already calculated by technical_indicators.py
        # We just shift them to get "next bar" values for prediction
        if getattr(self.config, 'prepare_advanced_ml_targets', False):
            momentum_shifts = []
            
            # Core momentum features (MOM_t, MR_t, TF_t)
            for col in ['MOM_t', 'MR_t', 'TF_t']:
                if col in self.data.columns:
                    target_col = f'adv_target_{col}_next'
                    self.data[target_col] = self.data[col].shift(-1)
                    momentum_shifts.append(target_col)
            
            # RSI (all periods found)
            rsi_cols = [c for c in self.data.columns if c.startswith('RSI_') and 'Pct' not in c and 'Change' not in c]
            for rsi_col in rsi_cols:
                # Extract period: RSI_14 → 14
                suffix = rsi_col.replace('RSI_', '')
                target_col = f'adv_target_RSI_{suffix}_next'
                self.data[target_col] = self.data[rsi_col].shift(-1)
                momentum_shifts.append(target_col)
            
            # MACD histogram (all variants)
            macdh_cols = [c for c in self.data.columns if c.startswith('MACDh_')]
            for macdh_col in macdh_cols:
                suffix = macdh_col.replace('MACDh_', '')
                target_col = f'adv_target_MACDh_{suffix}_next'
                self.data[target_col] = self.data[macdh_col].shift(-1)
                momentum_shifts.append(target_col)
            
            # Bollinger %B (all variants)
            bb_cols = [c for c in self.data.columns if c.startswith('BBP_') or c.startswith('BBB_')]
            for bb_col in bb_cols:
                suffix = bb_col.replace('BBP_', '').replace('BBB_', '')
                target_col = f'adv_target_BB_{suffix}_next'
                self.data[target_col] = self.data[bb_col].shift(-1)
                momentum_shifts.append(target_col)
            
            # ATR (all variants)
            atr_cols = [c for c in self.data.columns if c.upper().startswith('ATR') and not c.endswith('_Pct')]
            for atr_col in atr_cols:
                suffix = atr_col.replace('ATR', '').replace('_', '')
                target_col = f'adv_target_ATR_{suffix}_next' if suffix else 'adv_target_ATR_next'
                self.data[target_col] = self.data[atr_col].shift(-1)
                momentum_shifts.append(target_col)
            
            if momentum_shifts:
                logger.info(f" [Enriching] Added {len(momentum_shifts)} momentum indicator targets via shift(-1)")
                logger.info(f"   Examples: {momentum_shifts[:5]}{'...' if len(momentum_shifts) > 5 else ''}")
            
            # Collect session & time features as targets for temporal ML learning.
            # If upstream analysis did not create these columns, derive them from Time
            # so every declared session/time target is materialized before scaling.
            session_time_shifts = []
            
            time_source = next(
                (c for c in ['Time', 'time', 'Datetime', 'datetime', 'Timestamp', 'timestamp', 'Date', 'date']
                 if c in self.data.columns),
                None
            )
            if time_source:
                try:
                    raw_time = self.data[time_source]
                    if pd.api.types.is_numeric_dtype(raw_time):
                        unit = 'ms' if raw_time.dropna().abs().median() > 1e12 else 's'
                        dt = pd.to_datetime(raw_time, unit=unit, errors='coerce', utc=True)
                    else:
                        dt = pd.to_datetime(raw_time, errors='coerce', utc=True)

                    hour = dt.dt.hour.fillna(0).astype(float)
                    session = pd.Series(
                        np.select(
                            [
                                (hour >= 0) & (hour < 8),
                                (hour >= 8) & (hour < 16),
                            ],
                            [0.0, 1.0],
                            default=2.0,
                        ),
                        index=self.data.index,
                    )
                    derived_time_cols = {
                        'session': session,
                        'session_transition': session.ne(session.shift(1)).astype(float).fillna(0.0),
                        'day_of_week': dt.dt.dayofweek.fillna(0).astype(float),
                        'hour': hour,
                        'minute': dt.dt.minute.fillna(0).astype(float),
                    }
                    for col, series in derived_time_cols.items():
                        if col not in self.data.columns:
                            self.data[col] = series
                except Exception as e:
                    logger.warning(f"⚠️ [Enriching] Could not derive session/time targets from {time_source}: {e}")

            session_cols = {
                'session': 'Session class (0=Asia, 1=US, 2=Europe)',
                'session_transition': 'Session boundary flag',
                'day_of_week': 'Day of week (0=Mon, 6=Sun)',
                'hour': 'Hour of day (0-23)',
                'minute': 'Minute of hour (0-59)',
            }
            
            for col, description in session_cols.items():
                if col in self.data.columns:
                    target_col = f'adv_target_{col}_next'
                    self.data[target_col] = self.data[col].shift(-1).fillna(0.0)
                    session_time_shifts.append(target_col)
            
            if session_time_shifts:
                logger.info(f" [Enriching] Added {len(session_time_shifts)} session/time targets via shift(-1)")
                logger.info(f"   Targets: {session_time_shifts}")
        
        # Also collect OHLCV sequence targets for multi-task learning
        if getattr(self.config, 'prepare_advanced_ml_targets', False):
            ohlcv_targets_added = self._collect_next_candle_ohlcv_targets()
            if ohlcv_targets_added:
                logger.info(f" [Enriching] {len(ohlcv_targets_added)} OHLCV sequence targets added for multi-task learning")

        # ── NEW ADVANCED TARGETS ────────────────────────────────────────────────────────────
        if getattr(self.config, 'prepare_advanced_ml_targets', False):

            # ── 1. Dual-Head BEAR/WAIT/BULL labels (primary classification targets)
            #       Ports create_dual_labels() from the Dual-Head v6/v7 writing file.
            #       Produces adv_target_bull_class {0,1,2}, adv_target_bull_prob [0,1],
            #       adv_target_bull_conf {0,1}, adv_target_bear_conf {0,1}.
            #       bull_conf and bear_conf are INDEPENDENT: WAIT bars → both=0.
            dual_labels_added = self._compute_dual_head_labels()
            if dual_labels_added:
                logger.info(
                    f" [Enriching] Dual-Head labels added: {dual_labels_added}"
                )

            # ── 2. Max Favorable/Adverse Excursion (MFE / MAE)
            mfe_mae_added = self._compute_mfe_mae()
            if mfe_mae_added:
                logger.info(
                    f" [Enriching] MFE/MAE targets added: {mfe_mae_added}"
                )

            # ── 3. Forward log-returns at fixed horizons
            log_ret_added = self._compute_forward_log_returns()
            if log_ret_added:
                logger.info(
                    f" [Enriching] Forward log-return targets added: {log_ret_added}"
                )

            # ── 4. Forward Bull & Bear strength (independent continuous signals)
            strength_added = self._compute_forward_bull_bear_strength()
            if strength_added:
                logger.info(
                    f" [Enriching] Forward bull/bear strength targets added: {strength_added}"
                )

            # ── 5. Forward Velocity targets
            #       Next-bar values of Price_Velocity_* (shift-1 targets) +
            #       forward window averages of directional velocity.
            velocity_added = self._compute_forward_velocity_targets()
            if velocity_added:
                logger.info(
                    f" [Enriching] Forward velocity targets added: {velocity_added}"
                )

            # ── 6. Forward Volatility Regime targets
            #       Next-bar Volatility_Regime/Expansion/Bull/Bear +
            #       forward window average of Volatility_Regime (expected volatility state).
            vol_regime_added = self._compute_forward_volatility_targets()
            if vol_regime_added:
                logger.info(
                    f" [Enriching] Forward volatility regime targets added: {vol_regime_added}"
                )

            # ── 7. Forward Regime Speed targets
            #       Next-bar Regime_Speed_* (shift-1 targets) +
            #       forward window average of Regime_Speed_Aligned (expected trend pace).
            regime_speed_added = self._compute_forward_regime_speed_targets()
            if regime_speed_added:
                logger.info(
                    f" [Enriching] Forward regime speed targets added: {regime_speed_added}"
                )

            # ── 8. Forward Currency Strength Matrix targets
            #       If CSM columns are present (produced by the currency_strength_matrix
            #       analysis step), shift histogram and norm columns forward by 1 bar
            #       so the model can learn to predict next-bar divergence regime.
            csm_added = self._compute_forward_csm_targets()
            if csm_added:
                logger.info(
                    f" [Enriching] Forward CSM targets added: {csm_added}"
                )
            
            # ── 9. Forward Structural Level targets (Trendlines & SNR)
            #       AUXILIARY ENCODER TRAINING TARGETS that teach the model market geometry:
            #       - Trendlines: Linear channel boundaries (constant slopes)
            #       - SNR levels: Support/resistance zones (stationary price levels)
            #       These are EASY to predict (deterministic) but force the encoder to
            #       learn "is price in an uptrend channel?", "approaching resistance?",
            #       "breakout imminent?" without dominating the main price prediction task.
            #       Use very low loss weights (0.05-0.10) in the model.
            structural_added = self._compute_forward_structural_targets()
            if structural_added:
                logger.info(
                    f" [Enriching] Forward Structural Level targets added: {len(structural_added)} columns"
                )
                logger.info(
                    f"   Trendlines: {[c for c in structural_added if 'Trendline' in c]}"
                )
                logger.info(
                    f"   SNR levels: {[c for c in structural_added if 'snr_' in c]}"
                )

            # ── Next-Zone Liquidity targets ────────────────────────────────────
            # "Price moves from liquidity to liquidity."
            # Identifies which clustered S/R zone price actually visited next
            # within n_future bars, not just a shift(-1) of the nearest zone.
            # The model learns: given all known zones at time t, which one did
            # price travel to, how many bars did it take, and how much volume
            # sat at that zone (institutional interest proxy)?
            #
            # Outputs:
            #   adv_target_next_zone_price    — price level of the reached zone
            #   adv_target_next_zone_volume   — total volume absorbed at that zone
            #   adv_target_next_zone_type     — 1=support, 2=resistance, 0=none reached
            #   adv_target_next_zone_bars     — bars taken to reach it
            #   adv_target_next_zone_distance — ATR-normalised distance from Close[t]
            next_zone_added = self._compute_next_zone_targets(
                n_future=getattr(self.config, 'next_zone_n_future', 20),
                zone_touch_pct=getattr(self.config, 'next_zone_touch_pct', 0.004),
            )
            if next_zone_added:
                logger.info(
                    f" [Enriching] Next-Zone Liquidity targets added: {next_zone_added}"
                )

            # ── SNR Zone Sequence targets ──────────────────────────────────────
            # Ordered two-touch prediction: which SNR zone does price reach first
            # (snr_touch_1), then after that first touch which zone is reached
            # second (snr_touch_2)?
            #
            # Label space per head: 0=resistance  1=support  2=none
            # Uses zone prices from bar t+1 (forward-look snapshot) — no leakage.
            # snr_touch_2 implicitly encodes bounce vs breakout: (R→S)=bounce_R,
            # (S→R)=bounce_S, (R→R)=breakout_R, (S→S)=breakout_S — directly
            # complementary to Signal_bounce_* / Signal_breakout_* heads.
            snr_seq_added = self._compute_snr_zone_sequence_targets(
                n_future=getattr(self.config, 'snr_seq_n_future', 30),
                zone_touch_pct=getattr(self.config, 'snr_seq_touch_pct', 0.003),
            )
            if snr_seq_added:
                logger.info(
                    f" [Enriching] SNR zone sequence targets added: {snr_seq_added}"
                )

            # ── 10. Forward Reversal Probability labels
            #       Ground-truth answer to: "did a reversal actually happen AND hold?"
            #
            #       Unlike Reversal_Score (a heuristic INPUT feature computed from current-bar
            #       signals), these are LOOK-AHEAD LABELS that record what price actually did
            #       over the next n bars from the perspective of the current trend direction.
            #
            #       At a high pivot (uptrend bar), a reversal = price goes DOWN next.
            #       At a low pivot (downtrend bar), a reversal = price goes UP next.
            #
            #       Outputs:
            #         adv_target_reversal_prob           [0,1]  decay-weighted fraction of next n
            #                                                    bars that moved against current trend
            #         adv_target_trend_continuation_prob [0,1]  1 - reversal_prob (mirror)
            #         adv_target_reversal_held           {0,1}  1 if reversal began AND the
            #                                                    momentum held for the full window
            #
            #       Example:
            #         Bar at a local high, current trend = UP (Close > EMA_21).
            #         Next 8 bars all go down → reversal_prob ≈ 1.0, reversal_held = 1
            #         Next 8 bars mixed (4 up, 4 down) → reversal_prob ≈ 0.5, reversal_held = 0
            #         Next 8 bars all go up → reversal_prob ≈ 0.0, reversal_held = 0
            reversal_labels_added = self._compute_forward_reversal_labels(
                n_future=getattr(self.config, 'reversal_n_future', 8),
                decay=getattr(self.config, 'reversal_decay', 0.85),
                hold_threshold=getattr(self.config, 'reversal_hold_threshold', 0.65),
            )
            if reversal_labels_added:
                logger.info(
                    f" [Enriching] Forward reversal labels added: {reversal_labels_added}"
                )
        
        # Log which target columns now exist
        all_targets = [c for c in self.data.columns if c.startswith("Next_") or c.startswith("adv_target_")]
        logger.info(f" [Enriching] Target columns now in data: {len(all_targets)} total")
        logger.info(f" [Enriching] These will be EXCLUDED from features (not used as inputs)")
        logger.info(f" [Enriching] These are used ONLY as training labels (y values)")

       
        if self.reporter:
            await self.reporter.report_async(
                progress=stage_info['end'],
                message=f"{stage_info['label']} ✓",
                message2=f"Added {len(all_targets)} target columns (basic + momentum)"
            )

    # =========================================================================
    # ADVANCED ML TARGETS
    # These methods compute forward-looking ground-truth columns that are used
    # ONLY as regression/multi-task targets, never as input features.
    # All columns are prefixed with 'adv_target_' so _identify_features()
    # automatically excludes them from the feature set.
    # =========================================================================

    def _collect_next_candle_ohlcv_targets(self) -> List[str]:
        """
        Collect next-bar OHLCV (candlestick) targets for multi-task learning.
        
        UPDATED ARCHITECTURE (for uniform target shapes):
        Instead of returning (pred_len,) sequences, we now create individual
        scalar targets for each future timestep. This makes all targets have
        the same shape, simplifying multi-output model design.
        
        For each bar i, creates pred_len separate scalar targets per OHLCV column:
            adv_target_Open_t1, adv_target_Open_t2, ..., adv_target_Open_t7
            adv_target_High_t1, adv_target_High_t2, ..., adv_target_High_t7
            adv_target_Low_t1, adv_target_Low_t2, ..., adv_target_Low_t7
            adv_target_Close_t1, adv_target_Close_t2, ..., adv_target_Close_t7
            adv_target_Volume_t1, adv_target_Volume_t2, ..., adv_target_Volume_t7
        
        Multi-task learning benefit:
            When the model predicts all 5 OHLCV columns simultaneously,
            it learns the constraint that H >= max(O,C), L <= min(O,C),
            and high volume indicates conviction. This regularizes the Close
            prediction and improves price structure understanding.
        
        Returns:
            List of new column names added to self.data.
            
        Example output (pred_len=7):
            ['adv_target_Open_t1', 'adv_target_Open_t2', ..., 'adv_target_Volume_t7']
            Total: 35 scalar targets (5 OHLCV × 7 timesteps)
        """
        added_cols: List[str] = []
        df = self.data
        pred_len = self.config.prediction_length

        # Resolve OHLCV column names (case-insensitive)
        col_map = {c.lower(): c for c in df.columns}
        open_col   = col_map.get("open",   None)
        high_col   = col_map.get("high",   None)
        low_col    = col_map.get("low",    None)
        close_col  = col_map.get("close",  None)
        vol_col    = col_map.get("volume", None)

        if not all([open_col, high_col, low_col, close_col, vol_col]):
            missing = [n for n, c in [("Open", open_col), ("High", high_col), 
                                      ("Low", low_col), ("Close", close_col), 
                                      ("Volume", vol_col)] if c is None]
            logger.warning(
                f"[AdvTargets] Cannot collect OHLCV targets — missing columns: {missing}. "
                f"Available: {list(df.columns)[:10]}..."
            )
            return added_cols

        # Create individual scalar columns for each timestep
        # This replaces the old (pred_len,) sequence approach with pred_len scalar targets
        ohlcv_mapping = [
            (open_col, "Open"),
            (high_col, "High"),
            (low_col, "Low"),
            (close_col, "Close"),
            (vol_col, "Volume")
        ]
        
        for source_col, name in ohlcv_mapping:
            for t in range(1, pred_len + 1):
                # Shift by -t to get t bars into the future
                target_col_name = f"adv_target_{name}_t{t}"
                df[target_col_name] = df[source_col].shift(-t).astype(np.float32)
                added_cols.append(target_col_name)

        logger.info(
            f" [AdvTargets] OHLCV candlestick targets collected: {len(added_cols)} scalar targets"
        )
        logger.info(
            f"   Created {len(ohlcv_mapping)} OHLCV columns × {pred_len} timesteps"
        )
        logger.info(
            f"   All targets are now scalar values (uniform shape across all target types)"
        )
        logger.info(
            f"   This enables cleaner multi-task learning with consistent head architectures"
        )
        return added_cols

    # =========================================================================
    # DUAL-HEAD ADVANCED TARGETS
    # All methods below compute FORWARD-LOOKING ground-truth columns.
    # They are prefixed with 'adv_target_' so _identify_features() automatically
    # excludes them from the feature set X.  Never used as inputs.
    # =========================================================================

    def _compute_dual_head_labels(
        self,
        lookahead: int = 1,
        n_future: int = 8,
        w_base: float = 0.35,
        w_regime: float = 0.65,
        bull_q: float = 0.65,
        bear_q: float = 0.35,
    ) -> List[str]:
        """
        Port of create_dual_labels() from the Dual-Head v6/v7 writing file.

        For each bar i the label blends:
          base_val   = 1 if log(Close[i+lookahead] / Close[i]) > 0 else 0
          regime_str = exp-decay-weighted mean of Candle_Bull_Score[i+1 .. i+1+n_future]
          raw_label  = w_base * base_val + w_regime * regime_str

        Raw labels are converted to percentile rank, then quantile-bucketed:
          prob >= bull_q  →  BULL  = 2
          prob <= bear_q  →  BEAR  = 0
          else            →  WAIT  = 1

        Outputs (all prefixed adv_target_ → auto-excluded from features):
          adv_target_bull_prob    float [0,1]   percentile rank (continuous regression)
          adv_target_bull_class   int   {0,1,2} BEAR/WAIT/BULL (primary classification)
          adv_target_bull_conf    float {0,1}   is BULL? (GRU bull head binary target)
          adv_target_bear_conf    float {0,1}   is BEAR? (GRU bear head binary target)

        KEY: bull_conf and bear_conf are INDEPENDENT signals.
          BULL bar  → bull_conf=1, bear_conf=0
          BEAR bar  → bull_conf=0, bear_conf=1
          WAIT bar  → bull_conf=0, bear_conf=0  ← model learns: output LOW for both
        This trains the two GRU sigmoid heads to fire independently, not as a softmax pair.
        """
        df = self.data
        n  = len(df)
        added_cols: List[str] = []

        # ── Resolve column names (case-insensitive) ────────────────────────────
        col_map    = {c.lower(): c for c in df.columns}
        close_col  = col_map.get("close", None)
        bull_score = "Candle_Bull_Score" if "Candle_Bull_Score" in df.columns else None

        if close_col is None:
            logger.warning("[AdvTargets] _compute_dual_head_labels: no Close column found, skipping")
            return added_cols

        if bull_score is None:
            logger.warning(
                "[AdvTargets] _compute_dual_head_labels: 'Candle_Bull_Score' not found. "
                "Ensure enable_candle_bull_score=True in IndicatorConfig. Falling back to "
                "pure log-return base label (w_regime=0)."
            )
            w_base   = 1.0
            w_regime = 0.0

        close_vals = df[close_col].values.astype(np.float64)
        scores     = (
            df[bull_score].values.astype(np.float64)
            if bull_score is not None
            else np.full(n, 0.5, dtype=np.float64)
        )

        # Exponential decay weights for regime window
        # w_k = 0.85^k, then normalised so they sum to 1
        decay      = 0.85
        wts_full   = np.array([decay ** k for k in range(n_future)], dtype=np.float64)
        wts_full  /= wts_full.sum()

        # Total bars consumed per label: lookahead + 1 (for base) + n_future (for regime)
        total_fwd  = lookahead + n_future
        n_labels   = n - total_fwd

        if n_labels <= 0:
            logger.warning(
                f"[AdvTargets] Dual-head labels need {total_fwd} future bars, "
                f"but data only has {n} rows. Skipping."
            )
            return added_cols

        # Pre-allocate output arrays (NaN for rows that cannot have a label)
        raw_labels  = np.full(n, np.nan, dtype=np.float64)

        for i in range(n_labels):
            # Base: did price go up after lookahead bars?
            next_close = close_vals[i + lookahead]
            curr_close = close_vals[i]
            if curr_close <= 0:
                continue
            base_val = 1.0 if np.log(next_close / curr_close) > 0 else 0.0

            # Regime: exponentially weighted Candle_Bull_Score over n_future bars
            regime_slice = scores[i + lookahead + 1 : i + lookahead + 1 + n_future]
            if len(regime_slice) == 0:
                regime_str = 0.5
            else:
                wts_slice  = wts_full[: len(regime_slice)]
                wts_slice  = wts_slice / wts_slice.sum()
                regime_str = float(np.dot(regime_slice, wts_slice))

            raw_labels[i] = w_base * base_val + w_regime * regime_str

        # Convert to percentile rank (only over valid rows)
        valid_mask = ~np.isnan(raw_labels)
        prob_arr   = np.full(n, np.nan, dtype=np.float64)
        if valid_mask.sum() > 0:
            # Rolling-window percentile rank (window=200 bars).
            #
            # WHY: scipy.stats.rankdata over the entire series guarantees a
            # balanced 35/30/35 BEAR/WAIT/BULL split *globally*, but says
            # nothing about the balance inside any contiguous temporal chunk.
            # Since markets trend for extended periods, a sequential val-slice
            # can land heavily skewed toward one class → the model learns to
            # always predict the majority class on val, producing exactly the
            # frozen identical-to-4-decimal-places accuracy observed in the
            # pretrained-continuation log (val_bear_conf = 0.6696 × 3 epochs).
            #
            # Rolling-window rank: each bar's prob = its percentile within the
            # trailing RANK_WINDOW bars only.  This keeps train/val/test class
            # distributions comparable across temporal splits.
            RANK_WINDOW = 200
            from scipy.stats import rankdata as _rankdata
            valid_indices = np.where(valid_mask)[0]
            for idx in valid_indices:
                win_start = max(0, idx - RANK_WINDOW + 1)
                win       = raw_labels[win_start : idx + 1]
                win_valid = win[~np.isnan(win)]
                if len(win_valid) < 2:
                    prob_arr[idx] = 0.5
                    continue
                r = _rankdata(win_valid, method='average')
                # Last element in win_valid is always raw_labels[idx]
                prob_arr[idx] = (r[-1] - 1) / max(len(r) - 1, 1)

        # Quantile classification
        class_arr     = np.where(valid_mask, 1, np.nan)          # default WAIT=1
        class_arr     = class_arr.astype(np.float64)
        class_arr[prob_arr >= bull_q] = 2.0                       # BULL
        class_arr[prob_arr <= bear_q] = 0.0                       # BEAR
        class_arr[~valid_mask]        = np.nan                    # no label

        # Store in dataframe
        df["adv_target_bull_prob"]  = prob_arr
        df["adv_target_bull_class"] = class_arr
        df["adv_target_bull_conf"]  = (class_arr == 2).astype(np.float64)  # is BULL?
        df["adv_target_bear_conf"]  = (class_arr == 0).astype(np.float64)  # is BEAR?

        # NaN-fill terminal rows (no label possible)
        for col in ["adv_target_bull_prob", "adv_target_bull_class",
                    "adv_target_bull_conf", "adv_target_bear_conf"]:
            df[col] = df[col].fillna(0.0)

        added_cols = [
            "adv_target_bull_prob", "adv_target_bull_class",
            "adv_target_bull_conf", "adv_target_bear_conf",
        ]

        # Log class distribution
        valid_classes = class_arr[valid_mask]
        n_bull = int((valid_classes == 2).sum())
        n_bear = int((valid_classes == 0).sum())
        n_wait = int((valid_classes == 1).sum())
        total  = n_bull + n_bear + n_wait
        logger.info(
            f" [AdvTargets] Dual-Head labels: "
            f"BEAR={n_bear} ({n_bear/total*100:.1f}%)  "
            f"WAIT={n_wait} ({n_wait/total*100:.1f}%)  "
            f"BULL={n_bull} ({n_bull/total*100:.1f}%)  "
            f"(bull_q={bull_q}, bear_q={bear_q})"
        )
        return added_cols

    def _compute_mfe_mae(self) -> List[str]:
        """
        Compute Max Favorable Excursion (MFE) and Max Adverse Excursion (MAE)
        over the next prediction_length bars.

        MFE_n = max(High[t+1 .. t+n] - Close[t]) / Close[t]   (max upside)
        MAE_n = min(Low[t+1  .. t+n] - Close[t]) / Close[t]   (max drawdown)

        Both are expressed as log-return equivalents (fractions, not percent).
        MFE >= 0, MAE <= 0  for all valid rows.

        Output columns:
          adv_target_MFE   [0, +inf)   maximum upside during window
          adv_target_MAE   (-inf, 0]   maximum adverse move during window
        """
        df      = self.data
        n       = len(df)
        pred_n  = self.config.prediction_length
        added_cols: List[str] = []

        col_map   = {c.lower(): c for c in df.columns}
        close_col = col_map.get("close", None)
        high_col  = col_map.get("high",  None)
        low_col   = col_map.get("low",   None)

        if not all([close_col, high_col, low_col]):
            missing = [k for k, v in {"close": close_col, "high": high_col, "low": low_col}.items() if v is None]
            logger.warning(f"[AdvTargets] _compute_mfe_mae: missing columns {missing}, skipping")
            return added_cols

        close_vals = df[close_col].values.astype(np.float64)
        high_vals  = df[high_col].values.astype(np.float64)
        low_vals   = df[low_col].values.astype(np.float64)

        mfe_arr = np.zeros(n, dtype=np.float64)
        mae_arr = np.zeros(n, dtype=np.float64)

        for i in range(n - pred_n):
            c = close_vals[i]
            if c <= 0:
                continue
            fwd_high = high_vals[i + 1 : i + 1 + pred_n]
            fwd_low  = low_vals[i + 1  : i + 1 + pred_n]
            # MFE: maximum gain (relative to current close)
            mfe_arr[i] = (np.max(fwd_high) - c) / c
            # MAE: maximum loss (negative, relative to current close)
            mae_arr[i] = (np.min(fwd_low)  - c) / c

        df["adv_target_MFE"] = mfe_arr
        df["adv_target_MAE"] = mae_arr

        added_cols = ["adv_target_MFE", "adv_target_MAE"]
        logger.info(
            f" [AdvTargets] MFE/MAE computed over {pred_n}-bar window. "
            f"MFE mean={mfe_arr.mean():.4f}, MAE mean={mae_arr.mean():.4f}"
        )
        return added_cols

    def _compute_forward_log_returns(self) -> List[str]:
        """
        Compute forward log-returns at fixed horizons: 1, 5, 10, 20 bars.

        adv_target_logret_N = log(Close[t+N] / Close[t])

        Terminal rows that cannot look N bars ahead are filled with 0.0.
        These are regression targets for multi-horizon return prediction.
        """
        df = self.data
        added_cols: List[str] = []

        col_map   = {c.lower(): c for c in df.columns}
        close_col = col_map.get("close", None)
        if close_col is None:
            logger.warning("[AdvTargets] _compute_forward_log_returns: no Close column, skipping")
            return added_cols

        close = df[close_col].replace(0, np.nan)

        for horizon in [1, 5, 10, 20]:
            col_name = f"adv_target_logret_{horizon}"
            fwd_close = close.shift(-horizon)
            log_ret   = np.log(fwd_close / close)
            df[col_name] = log_ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            added_cols.append(col_name)

        logger.info(
            f" [AdvTargets] Forward log-returns added at horizons: 1, 5, 10, 20 bars."
        )
        return added_cols

    def _compute_forward_bull_bear_strength(
        self,
        n_future: int = 8,
        decay: float = 0.85,
    ) -> List[str]:
        """
        Compute independent forward Bull and Bear regime strength signals.

        These are CONTINUOUS companions to the binary adv_target_bull/bear_conf.
        Both strengths are always provided — the strength of one cannot be directly
        substituted for the weakness of the other.

        adv_target_bull_strength_n  = exp-decay-weighted avg of Candle_Bull_Score[t+1..t+n]
        adv_target_bear_strength_n  = exp-decay-weighted avg of Candle_Bear_Score[t+1..t+n]
                                      (falls back to 1-Candle_Bull_Score if Bear score unavailable)

        Independence rationale:
          - For a WAIT scenario (scores ≈ 0.5 for all future bars):
              bull_strength ≈ 0.5, bear_strength ≈ 0.5  (both moderate — NOT forced mirrors)
          - For a strong BULL regime (scores ≈ 0.85):
              bull_strength ≈ 0.85, bear_strength ≈ 0.15
          - For a strong BEAR regime (scores ≈ 0.15):
              bull_strength ≈ 0.15, bear_strength ≈ 0.85
          - For a volatile doji-heavy regime (mixed bull/bear geometry):
              bull_strength ≈ 0.5, bear_strength ≈ 0.5  (BOTH moderate — correctly ambiguous)

        The model's two GRU sigmoid heads are each trained against ONE of these,
        letting them learn that high-uncertainty = both outputs moderate/low.
        """
        df       = self.data
        n        = len(df)
        added_cols: List[str] = []

        if "Candle_Bull_Score" not in df.columns:
            logger.warning(
                "[AdvTargets] _compute_forward_bull_bear_strength: "
                "'Candle_Bull_Score' not found. "
                "Enable enable_candle_bull_score=True in IndicatorConfig and re-run TI."
            )
            return added_cols

        scores     = df["Candle_Bull_Score"].values.astype(np.float64)
        # FIX: Use Candle_Bear_Score directly if available (independent score, NOT 1-bull).
        # Before: bear_sc = 1.0 - scores  (symmetric inversion — wrong for WAIT bars)
        # After:  Candle_Bear_Score is computed from bear-specific candle geometry,
        #         so a doji/WAIT bar independently scores ~0.5 on BOTH, not forced to mirror.
        if "Candle_Bear_Score" in df.columns:
            bear_sc = df["Candle_Bear_Score"].values.astype(np.float64)
            logger.info("[AdvTargets] Using Candle_Bear_Score (independent) for bear_strength target")
        else:
            bear_sc = 1.0 - scores   # legacy fallback: symmetric inversion
            logger.debug("[AdvTargets] Candle_Bear_Score not found, falling back to 1 - Candle_Bull_Score")

        # Exponential decay weights, normalised
        wts_full   = np.array([decay ** k for k in range(n_future)], dtype=np.float64)
        wts_full  /= wts_full.sum()

        col_name   = f"adv_target_bull_strength_{n_future}"
        col_bear   = f"adv_target_bear_strength_{n_future}"

        bull_str   = np.zeros(n, dtype=np.float64)
        bear_str   = np.zeros(n, dtype=np.float64)

        for i in range(n - n_future):
            bull_slice = scores[i + 1 : i + 1 + n_future]
            bear_slice = bear_sc[i + 1 : i + 1 + n_future]
            wts        = wts_full[: len(bull_slice)]
            wts        = wts / wts.sum()
            bull_str[i] = float(np.dot(bull_slice, wts))
            bear_str[i] = float(np.dot(bear_slice, wts))

        df[col_name] = np.clip(bull_str, 0.0, 1.0)
        df[col_bear] = np.clip(bear_str, 0.0, 1.0)

        added_cols = [col_name, col_bear]
        logger.info(
            f" [AdvTargets] Forward bull/bear strength over {n_future}-bar window: "
            f"bull_mean={bull_str.mean():.3f}, bear_mean={bear_str.mean():.3f}"
        )
        return added_cols

    # =========================================================================
    # NEW ADVANCED TARGET METHODS — Velocity, Volatility, Regime Speed
    # =========================================================================

    def _compute_forward_velocity_targets(self, n_future: int = 8, decay: float = 0.85) -> List[str]:
        """
        Compute forward-looking Price Velocity targets.

        Two target types are produced per velocity column:

        1. shift(-1) — next-bar value (sequence target for _seq suffix):
             adv_target_Price_Velocity_Bull_next  — what is the bull pip-rate next bar?
             adv_target_Price_Velocity_Bear_next  — what is the bear pip-rate next bar?
             adv_target_Price_Velocity_Net_next   — what is the net direction next bar?

        2. forward exp-decay-weighted average over n_future bars (regime outlook):
             adv_target_vel_bull_fwd_{n}   — expected bull velocity over next n bars
             adv_target_vel_bear_fwd_{n}   — expected bear velocity over next n bars
             adv_target_vel_net_fwd_{n}    — expected net velocity over next n bars

        Why two types?
          The shift(-1) targets give the model a single-step prediction anchor.
          The forward averages give it the regime outlook — "will velocity stay high?"
          These are independent and complementary for multi-task learning.

        Requires: Price_Velocity_Bull, Price_Velocity_Bear, Price_Velocity_Net
                  (computed in TI._calculate_price_velocity).
        """
        df         = self.data
        n          = len(df)
        added_cols: List[str] = []

        vel_cols = {
            "Price_Velocity_Bull": ("adv_target_Price_Velocity_Bull_next",
                                    f"adv_target_vel_bull_fwd_{n_future}"),
            "Price_Velocity_Bear": ("adv_target_Price_Velocity_Bear_next",
                                    f"adv_target_vel_bear_fwd_{n_future}"),
            "Price_Velocity_Net":  ("adv_target_Price_Velocity_Net_next",
                                    f"adv_target_vel_net_fwd_{n_future}"),
        }

        missing = [c for c in vel_cols if c not in df.columns]
        if missing:
            logger.warning(
                f"[AdvTargets] _compute_forward_velocity_targets: missing columns {missing}. "
                f"Ensure TI._calculate_price_velocity ran. Skipping."
            )
            return added_cols

        # Exponential decay weights for forward window
        wts_full = np.array([decay ** k for k in range(n_future)], dtype=np.float64)
        wts_full /= wts_full.sum()

        for src_col, (next_col, fwd_col) in vel_cols.items():
            vals = df[src_col].values.astype(np.float64)

            # 1. shift(-1) next-bar target
            df[next_col] = df[src_col].shift(-1).fillna(0.0)
            added_cols.append(next_col)

            # 2. Forward exp-decay average
            fwd_arr = np.zeros(n, dtype=np.float64)
            for i in range(n - n_future):
                window_vals = vals[i + 1 : i + 1 + n_future]
                wts         = wts_full[: len(window_vals)]
                wts         = wts / wts.sum()
                fwd_arr[i]  = float(np.dot(window_vals, wts))

            df[fwd_col] = np.clip(fwd_arr, -1.0, 3.0)   # Net in [-1,1], Bull/Bear in [0,3]
            added_cols.append(fwd_col)

        logger.info(
            f" [AdvTargets] Velocity targets: {len(added_cols)} columns "
            f"(3 next-bar + 3 forward-{n_future}-bar)"
        )
        return added_cols

    def _compute_forward_volatility_targets(self, n_future: int = 8, decay: float = 0.85) -> List[str]:
        """
        Compute forward-looking Volatility Regime targets.

        1. shift(-1) — next-bar value for all four volatility columns:
             adv_target_Volatility_Regime_next
             adv_target_Volatility_Expansion_next
             adv_target_Volatility_Bull_next
             adv_target_Volatility_Bear_next

        2. Forward exp-decay-weighted average for Volatility_Regime (the most useful):
             adv_target_vol_regime_fwd_{n}   — expected volatility level over next n bars
             adv_target_vol_expansion_fwd_{n} — expected expansion state over next n bars

        Why the forward average for Regime and Expansion?
          - Regime tells the model "will we still be in a high-vol state?"
          - Expansion tells the model "will volatility keep growing or start contracting?"
          - Bull/Bear split is already captured per-bar, shift(-1) is sufficient.

        Requires: Volatility_Regime, Volatility_Expansion, Volatility_Bull, Volatility_Bear
                  (computed in TI._calculate_volatility_regime).
        """
        df         = self.data
        n          = len(df)
        added_cols: List[str] = []

        # shift(-1) next-bar targets for all four columns
        shift_cols = [
            "Volatility_Regime",
            "Volatility_Expansion",
            "Volatility_Bull",
            "Volatility_Bear",
        ]

        for col in shift_cols:
            if col not in df.columns:
                logger.debug(f"[AdvTargets] _compute_forward_volatility_targets: '{col}' not found, skipping")
                continue
            target_col = f"adv_target_{col}_next"
            df[target_col] = df[col].shift(-1).fillna(0.5)   # fill with neutral 0.5
            added_cols.append(target_col)

        # Forward window averages for Regime and Expansion
        wts_full = np.array([decay ** k for k in range(n_future)], dtype=np.float64)
        wts_full /= wts_full.sum()

        fwd_targets = {
            "Volatility_Regime":    f"adv_target_vol_regime_fwd_{n_future}",
            "Volatility_Expansion": f"adv_target_vol_expansion_fwd_{n_future}",
        }

        for src_col, fwd_col in fwd_targets.items():
            if src_col not in df.columns:
                continue
            vals    = df[src_col].values.astype(np.float64)
            fwd_arr = np.zeros(n, dtype=np.float64)

            for i in range(n - n_future):
                window_vals = vals[i + 1 : i + 1 + n_future]
                wts         = wts_full[: len(window_vals)]
                wts         = wts / wts.sum()
                fwd_arr[i]  = float(np.dot(window_vals, wts))

            df[fwd_col] = np.clip(fwd_arr, 0.0, 1.0)
            added_cols.append(fwd_col)

        logger.info(
            f" [AdvTargets] Volatility regime targets: {len(added_cols)} columns "
            f"(4 next-bar + 2 forward-{n_future}-bar)"
        )
        return added_cols

    def _compute_forward_regime_speed_targets(self, n_future: int = 8, decay: float = 0.85) -> List[str]:
        """
        Compute forward-looking Regime Speed targets.

        1. shift(-1) — next-bar value for all four speed columns:
             adv_target_Regime_Speed_Bull_next
             adv_target_Regime_Speed_Bear_next
             adv_target_Regime_Speed_Aligned_next
             adv_target_Regime_Speed_Divergence_next

        2. Forward exp-decay-weighted average for Aligned and Divergence:
             adv_target_speed_aligned_fwd_{n}    — how fast will the trend advance?
             adv_target_speed_divergence_fwd_{n} — which direction will dominate?

        Why these two for forward averages?
          - Aligned is the primary "how fast is the trend going?" signal.
            A model seeing Aligned_fwd_8 = 0.8 knows the trend will stay fast.
          - Divergence tells the model whether the dominant direction is about to flip.
            Divergence_fwd_8 trending from +0.6 to 0 signals an impending slowdown.

        Requires: Regime_Speed_Bull, Regime_Speed_Bear, Regime_Speed_Aligned,
                  Regime_Speed_Divergence (computed in TI._calculate_regime_speed).
        """
        df         = self.data
        n          = len(df)
        added_cols: List[str] = []

        # shift(-1) next-bar targets for all four columns
        shift_cols = [
            "Regime_Speed_Bull",
            "Regime_Speed_Bear",
            "Regime_Speed_Aligned",
            "Regime_Speed_Divergence",
        ]

        for col in shift_cols:
            if col not in df.columns:
                logger.debug(f"[AdvTargets] _compute_forward_regime_speed_targets: '{col}' not found, skipping")
                continue
            target_col = f"adv_target_{col}_next"
            df[target_col] = df[col].shift(-1).fillna(0.0)
            added_cols.append(target_col)

        # Forward window averages for Aligned and Divergence
        wts_full = np.array([decay ** k for k in range(n_future)], dtype=np.float64)
        wts_full /= wts_full.sum()

        fwd_targets = {
            "Regime_Speed_Aligned":    f"adv_target_speed_aligned_fwd_{n_future}",
            "Regime_Speed_Divergence": f"adv_target_speed_divergence_fwd_{n_future}",
        }

        for src_col, fwd_col in fwd_targets.items():
            if src_col not in df.columns:
                continue
            vals    = df[src_col].values.astype(np.float64)
            fwd_arr = np.zeros(n, dtype=np.float64)
            clip_lo = -1.0 if "Divergence" in src_col else 0.0   # Divergence is signed

            for i in range(n - n_future):
                window_vals = vals[i + 1 : i + 1 + n_future]
                wts         = wts_full[: len(window_vals)]
                wts         = wts / wts.sum()
                fwd_arr[i]  = float(np.dot(window_vals, wts))

            df[fwd_col] = np.clip(fwd_arr, clip_lo, 1.0)
            added_cols.append(fwd_col)

        logger.info(
            f" [AdvTargets] Regime speed targets: {len(added_cols)} columns "
            f"(4 next-bar + 2 forward-{n_future}-bar)"
        )
        return added_cols

    def _compute_forward_csm_targets(self) -> List[str]:
        """
        Forward Currency Strength Matrix (CSM) targets.

        Shifts CSM columns by -1 (next-bar prediction) so the model can learn to
        predict the upcoming divergence regime between the asset and DXY.

        Source columns (produced by the currency_strength_matrix analysis step):
            CSM_histogram_fast  → adv_target_CSM_hist_fast_next
            CSM_histogram_slow  → adv_target_CSM_hist_slow_next
            CSM_asset_norm_fast → adv_target_CSM_asset_fast_next
            CSM_dxy_norm_fast   → adv_target_CSM_dxy_fast_next

        Returns [] when no CSM columns are present in self.data (i.e. when the
        currency_strength_matrix step was not included in the pipeline).
        """
        CSM_TARGET_MAP = {
            "CSM_histogram_fast":  "adv_target_CSM_hist_fast_next",
            "CSM_histogram_slow":  "adv_target_CSM_hist_slow_next",
            "CSM_asset_norm_fast": "adv_target_CSM_asset_fast_next",
            "CSM_dxy_norm_fast":   "adv_target_CSM_dxy_fast_next",
        }
        added: List[str] = []
        for src_col, tgt_col in CSM_TARGET_MAP.items():
            if src_col not in self.data.columns:
                continue
            self.data[tgt_col] = self.data[src_col].shift(-1)
            added.append(tgt_col)

        if added:
            logger.info(
                f" [AdvTargets] CSM targets: {len(added)} columns via shift(-1) — "
                f"model predicts next-bar fast/slow histogram regime"
            )
        return added

    def _compute_next_zone_targets(
        self,
        n_future: int = 20,
        zone_touch_pct: float = 0.004,
    ) -> List[str]:
        """
        Next-Zone Liquidity Target — "price moves from liquidity to liquidity".

        Theory
        ------
        At any bar `t`, price sits between multiple clustered S/R zones.
        The pivot levels (r1/r2/r3 above, s1/s2/s3 below) already exist as
        features in the model's input sequence — they ARE the candidate zones.
        This method asks: "given all six of those known candidates, which one
        did price visit first within the next n_future bars?"

        The answer is a single integer label 0–6:
          0 = r1 (nearest resistance)
          1 = r2
          2 = r3
          3 = s1 (nearest support)
          4 = s2
          5 = s3
          6 = none reached within window

        This gives the model a softmax head over exactly the zones it already
        sees in its features — making the task trivially well-posed:
          "Here are six candidate zones with their distances and volumes.
           Assign a probability to each one being visited next."

        Why this is better than regression targets:
        - The model can directly reason: "S1 is close + has high volume →
          high probability price visits S1 before R1"
        - Output is a proper probability distribution over known candidates —
          directly interpretable on the dashboard ("73% → S1")
        - Cross-entropy loss is cleaner than MSE for "which one" questions

        Additionally computes three regression auxiliaries (at low weight)
        to give the head extra gradient signal:
          adv_target_next_zone_bars     — bars to reach (0=none)
          adv_target_next_zone_distance — ATR-normalised distance of visited zone
          adv_target_next_zone_volume   — volume at the visited zone

        Parameters
        ----------
        n_future : int
            Look-ahead window in bars (default 20).
        zone_touch_pct : float
            Tolerance to consider a zone "touched" (default 0.004 = 0.4%).

        Returns
        -------
        List of added column names.
        """
        df = self.data

        # ── Column resolution ────────────────────────────────────────────────
        close_col = next((c for c in ("close", "Close", "close_5m", "Close_5m") if c in df.columns), None)
        high_col  = next((c for c in ("high", "High", "high_5m", "High_5m") if c in df.columns), None)
        low_col   = next((c for c in ("low", "Low", "low_5m", "Low_5m") if c in df.columns), None)
        atr_col   = next((c for c in ("ATR", "atr", "ATR_5m", "atr_5m") if c in df.columns), None)

        if not all([close_col, high_col, low_col]):
            logger.debug("[NextZone] OHLC columns missing — skipping next-zone targets")
            return []

        # Pivot level columns.  Accept both capitalised and lowercase variants.
        # r1/r2/r3 = resistance above price; s1/s2/s3 = support below price.
        # The probe confirmed these sit at indices 148–157 in the current schema.
        def _find_col(candidates):
            for c in candidates:
                if c in df.columns:
                    return c
                for suffix in ("_5m", "_5m_5m"):
                    if f"{c}{suffix}" in df.columns:
                        return f"{c}{suffix}"
            return None

        r1_col = _find_col(["r1", "R1", "r1_5m", "R1_5m", "Pivot_R1", "Pivot_R1_Diff", "Pivot_R1_Price", "r1_price"])
        r2_col = _find_col(["r2", "R2", "r2_5m", "R2_5m", "Pivot_R2", "Pivot_R2_Diff", "Pivot_R2_Price", "r2_price"])
        r3_col = _find_col(["r3", "R3", "r3_5m", "R3_5m", "Pivot_R3", "Pivot_R3_Diff", "Pivot_R3_Price", "r3_price"])
        s1_col = _find_col(["s1", "S1", "s1_5m", "S1_5m", "Pivot_S1", "Pivot_S1_Diff", "Pivot_S1_Price", "s1_price"])
        s2_col = _find_col(["s2", "S2", "s2_5m", "S2_5m", "Pivot_S2", "Pivot_S2_Diff", "Pivot_S2_Price", "s2_price"])
        s3_col = _find_col(["s3", "S3", "s3_5m", "S3_5m", "Pivot_S3", "Pivot_S3_Diff", "Pivot_S3_Price", "s3_price"])

        pivot_cols = [r1_col, r2_col, r3_col, s1_col, s2_col, s3_col]
        pivot_types = ["resistance", "resistance", "resistance",
                       "support",    "support",    "support"]
        pivot_names = ["r1", "r2", "r3", "s1", "s2", "s3"]

        # Fall back gracefully if some pivot columns are missing
        available = [(i, c, t, nm) for i, (c, t, nm) in
                     enumerate(zip(pivot_cols, pivot_types, pivot_names)) if c is not None]

        logger.info(f"🔍 [DIAGNOSTIC] _compute_next_zone_targets - Columns in df ({len(df.columns)} total): {list(df.columns[:25])}...")
        logger.info(f"🔍 [DIAGNOSTIC] _compute_next_zone_targets - Found pivot columns: {[c for c in pivot_cols if c is not None]}")

        if len(available) == 0:
            logger.warning("[NextZone] No pivot columns (r1/r2/r3/s1/s2/s3) found — "
                           "skipping next-zone targets")
            return []

        K = 6          # total candidate slots (r1/r2/r3/s1/s2/s3)
        NONE_IDX = K   # class label when no zone reached in window

        vol_col      = next((c for c in ("snr_nearest_zone_volume", "snr_nearest_zone_volume_5m") if c in df.columns), None)
        zonal_col    = next((c for c in ("Zonal_Total_Volume", "Zonal_Total_Volume_5m") if c in df.columns), None)

        n = len(df)

        # ── Pre-extract arrays ───────────────────────────────────────────────
        close_arr = df[close_col].values.astype(np.float32)
        high_arr  = df[high_col].values.astype(np.float32)
        low_arr   = df[low_col].values.astype(np.float32)
        zvol_arr  = df[vol_col].values.astype(np.float32) if vol_col in df.columns else np.zeros(n, np.float32)
        zonal_arr = df[zonal_col].values.astype(np.float32) if zonal_col else None

        # Build (n, len(available)) array of pivot prices
        pivot_price_mat = np.zeros((n, K), dtype=np.float32)
        for slot_idx, col, _, _ in available:
            vals = df[col].values.astype(np.float32)
            if "diff" in col.lower():
                # If col is a diff column (e.g. Pivot_R1_Diff = Close - R1), price = Close - diff_val
                pivot_price_mat[:, slot_idx] = close_arr - vals
            else:
                pivot_price_mat[:, slot_idx] = vals

        if atr_col:
            atr_arr = df[atr_col].values.astype(np.float32)
        else:
            close_diff = np.abs(np.diff(close_arr, prepend=close_arr[0]))
            atr_arr = np.convolve(close_diff, np.ones(14) / 14, mode="same").astype(np.float32)
        atr_arr = np.where(atr_arr > 0, atr_arr, 1e-6)

        # ── Output arrays ────────────────────────────────────────────────────
        next_zone_idx      = np.full(n, NONE_IDX, dtype=np.float32)   # 0–6
        next_zone_bars     = np.full(n, float(n_future + 1), np.float32)
        next_zone_distance = np.zeros(n, dtype=np.float32)
        next_zone_volume   = np.zeros(n, dtype=np.float32)

        logger.info(
            f"[NextZone] Computing next-zone distribution targets: "
            f"n_future={n_future}, touch_pct={zone_touch_pct:.3f}, "
            f"K={K} candidates ({len(available)} present), bars={n}"
        )

        # ── Main scan (one bar at a time, vectorised inner window) ───────────
        for t in range(n - 1):
            end = min(t + 1 + n_future, n)
            win_high = high_arr[t + 1 : end]
            win_low  = low_arr[t + 1  : end]
            cur_close = close_arr[t]

            best_k     = n_future + 1   # bars to first touched zone
            best_slot  = NONE_IDX
            best_vol   = 0.0
            best_price = 0.0

            for slot_idx, col, zone_type, _ in available:
                z_price = pivot_price_mat[t, slot_idx]
                if z_price <= 0:
                    continue

                if zone_type == "resistance":
                    # Resistance is ABOVE price — touched when High >= threshold
                    if z_price < cur_close:
                        continue   # stale pivot — resistance now below price, skip
                    threshold = z_price * (1.0 - zone_touch_pct)
                    touched   = np.where(win_high >= threshold)[0]
                else:
                    # Support is BELOW price — touched when Low <= threshold
                    if z_price > cur_close:
                        continue   # stale pivot — support now above price, skip
                    threshold = z_price * (1.0 + zone_touch_pct)
                    touched   = np.where(win_low <= threshold)[0]

                if len(touched) == 0:
                    continue

                k = int(touched[0])
                if k < best_k or (k == best_k and abs(z_price - cur_close) < abs(best_price - cur_close)):
                    best_k     = k
                    best_slot  = slot_idx
                    best_price = z_price
                    # Volume: prefer zonal signal-bar volume, fall back to nearest-zone volume
                    bar_idx = t + 1 + k
                    if zonal_arr is not None and bar_idx < n and zonal_arr[bar_idx] > 0:
                        best_vol = float(zonal_arr[bar_idx])
                    else:
                        best_vol = float(zvol_arr[t])

            if best_slot < NONE_IDX:
                next_zone_idx[t]      = float(best_slot)
                next_zone_bars[t]     = float(best_k)
                next_zone_distance[t] = abs(best_price - cur_close) / atr_arr[t]
                next_zone_volume[t]   = best_vol

        # ── Write to DataFrame ───────────────────────────────────────────────
        df["adv_target_next_zone_idx"]      = next_zone_idx       # 0–6 class label
        df["adv_target_next_zone_bars"]     = next_zone_bars
        df["adv_target_next_zone_distance"] = next_zone_distance
        df["adv_target_next_zone_volume"]   = next_zone_volume
        # 0=support, 1=resistance, -1=no zone reached. This is a stable
        # scalar companion to the categorical zone-index target.
        df["adv_target_next_zone_type"] = np.where(
            next_zone_idx >= NONE_IDX,
            -1.0,
            (next_zone_idx < 3).astype(np.float32),
        ).astype(np.float32)

        added = [
            "adv_target_next_zone_idx",
            "adv_target_next_zone_bars",
            "adv_target_next_zone_distance",
            "adv_target_next_zone_volume",
            "adv_target_next_zone_type",
        ]

        # ── Diagnostics ──────────────────────────────────────────────────────
        reached       = (next_zone_idx < NONE_IDX)
        r_count       = int(np.sum(next_zone_idx[reached] < 3))   # slots 0,1,2 = resistance
        s_count       = int(np.sum(next_zone_idx[reached] >= 3))  # slots 3,4,5 = support
        touched_count = int(reached.sum())
        avg_bars      = float(next_zone_bars[reached].mean()) if touched_count > 0 else float("nan")

        # Class distribution
        class_counts = {nm: int((next_zone_idx == i).sum()) for i, (_, _, _, nm) in enumerate(available)}
        class_counts["none"] = int((next_zone_idx == NONE_IDX).sum())

        logger.info(
            f"[NextZone] ✅ Added {len(added)} next-zone targets (softmax label + aux regression):\n"
            f"   Zones reached: {touched_count}/{n-1} ({100*touched_count/max(n-1,1):.1f}%)\n"
            f"   Resistance visited: {r_count}  |  Support visited: {s_count}  |  None: {class_counts['none']}\n"
            f"   Avg bars to reach: {avg_bars:.1f}\n"
            f"   Class distribution: {class_counts}"
        )
        return added

    def _compute_snr_zone_sequence_targets(
        self,
        n_future: int = 30,
        zone_touch_pct: float = 0.003,
    ) -> List[str]:
        """Ordered two-touch SNR zone sequence labels.

        Theory
        ------
        At bar t, price will move toward one of two SNR zones: the nearest
        resistance above or the nearest support below.  This method asks:

          1. Which zone does price reach FIRST within n_future bars?   → snr_touch_1
          2. After that first touch, which zone does price reach NEXT?  → snr_touch_2

        Label space (per head): 0 = resistance  |  1 = support  |  2 = none

        Leakage prevention
        ------------------
        Zone prices are read from bar t+1 (the next bar's SNR snapshot), NOT
        from bar t.  The current bar's SNR levels are visible in the model's
        input features — reading them at t would make the label trivially
        solvable from the input.  The t+1 snapshot breaks this correlation while
        remaining a valid, well-defined zone.

        Complementarity with bounce/breakout heads
        ------------------------------------------
        The pair (snr_touch_1, snr_touch_2) implicitly encodes the post-touch
        market reaction:

          (resistance → support):  price bounced from resistance  → bounce_resistance
          (support → resistance):  price bounced from support     → bounce_support
          (resistance → resistance): price broke through resistance → breakout_resistance
          (support → support):     price broke through support    → breakout_support
          (any → none):            weak move, zone held partially

        This gives the bounce/breakout heads a sequence-level supervision signal
        that their binary per-bar targets alone cannot provide.

        Parameters
        ----------
        n_future : int
            Total look-ahead window in bars (default 30).  Must be long enough
            for both touches to occur; 30 bars = 15h on M30.
        zone_touch_pct : float
            Fractional tolerance to consider a zone "touched" (default 0.003 =
            0.3%; tighter than next_zone_targets to favour high-confidence hits).

        Returns
        -------
        List of added column names, or [] if required columns are missing.
        """
        df = self.data
        n  = len(df)

        RES_CLASS  = 0   # resistance touched
        SUP_CLASS  = 1   # support touched
        NONE_CLASS = 2   # neither reached in window

        # ── Column resolution ─────────────────────────────────────────────────
        def _find(candidates):
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        close_col = _find(["close", "Close"])
        high_col  = _find(["high",  "High"])
        low_col   = _find(["low",   "Low"])
        res_col   = _find(["snr_nearest_resistance_level", "SNR_Resistance",
                            "snr_resistance_level", "Resistance_Level"])
        sup_col   = _find(["snr_nearest_support_level", "SNR_Support",
                            "snr_support_level", "Support_Level"])

        if not all([close_col, high_col, low_col, res_col, sup_col]):
            missing = [name for name, col in [
                ("close", close_col), ("high", high_col), ("low", low_col),
                ("snr_resistance", res_col), ("snr_support", sup_col)
            ] if col is None]
            logger.warning(
                f"[SNRSeq] Missing columns {missing} — skipping SNR sequence targets"
            )
            return []

        close_arr = df[close_col].values.astype(np.float32)
        high_arr  = df[high_col].values.astype(np.float32)
        low_arr   = df[low_col].values.astype(np.float32)
        res_arr   = df[res_col].values.astype(np.float32)
        sup_arr   = df[sup_col].values.astype(np.float32)

        touch_1     = np.full(n, NONE_CLASS, dtype=np.int8)
        touch_2     = np.full(n, NONE_CLASS, dtype=np.int8)
        touch_1_bar = np.full(n, n_future + 1, dtype=np.int16)

        def _first_hit(win_arr, zone_price, is_resistance):
            """Return bar offset (0-indexed within window) of first touch, or n+1."""
            if is_resistance:
                threshold = zone_price * (1.0 - zone_touch_pct)
                hits = np.where(win_arr >= threshold)[0]
            else:
                threshold = zone_price * (1.0 + zone_touch_pct)
                hits = np.where(win_arr <= threshold)[0]
            return int(hits[0]) if len(hits) > 0 else n_future + 1

        # ── Main scan ─────────────────────────────────────────────────────────
        for t in range(n - 2):
            # Read zone prices from t+1 (forward-look snapshot — no leakage)
            r_price = float(res_arr[t + 1])
            s_price = float(sup_arr[t + 1])
            cur_close = float(close_arr[t])

            # Stale-pivot guard: resistance must be above price, support below
            r_valid = (r_price > cur_close) and (r_price > 0)
            s_valid = (s_price < cur_close) and (s_price > 0)

            end = min(t + 1 + n_future, n)
            win_high = high_arr[t + 1 : end]
            win_low  = low_arr[t + 1  : end]

            # ── First touch ──────────────────────────────────────────────────
            r_k = _first_hit(win_high, r_price, True)  if r_valid else n_future + 1
            s_k = _first_hit(win_low,  s_price, False) if s_valid else n_future + 1

            if r_k <= s_k and r_k <= n_future:
                touch_1[t]     = RES_CLASS
                touch_1_bar[t] = r_k
            elif s_k < r_k and s_k <= n_future:
                touch_1[t]     = SUP_CLASS
                touch_1_bar[t] = s_k
            else:
                continue   # no first touch → second touch undefined

            # ── Second touch (scan restarts AFTER first touch bar) ───────────
            ft_abs = t + 1 + int(touch_1_bar[t])   # absolute bar index of 1st touch
            remaining = n_future - int(touch_1_bar[t]) - 1
            if remaining <= 0 or ft_abs + 1 >= n:
                continue

            # Read fresh SNR snapshot at the first-touch bar
            snap = min(ft_abs + 1, n - 1)
            r2_price  = float(res_arr[snap])
            s2_price  = float(sup_arr[snap])
            ft_close  = float(close_arr[ft_abs])

            r2_valid = (r2_price > ft_close) and (r2_price > 0)
            s2_valid = (s2_price < ft_close) and (s2_price > 0)

            w2_end   = min(ft_abs + 1 + remaining, n)
            win2_high = high_arr[ft_abs + 1 : w2_end]
            win2_low  = low_arr[ft_abs + 1  : w2_end]

            r2_k = _first_hit(win2_high, r2_price, True)  if r2_valid else remaining + 1
            s2_k = _first_hit(win2_low,  s2_price, False) if s2_valid else remaining + 1

            if r2_k <= s2_k and r2_k <= remaining:
                touch_2[t] = RES_CLASS
            elif s2_k < r2_k and s2_k <= remaining:
                touch_2[t] = SUP_CLASS
            # else: NONE_CLASS (default)

        # ── Write to DataFrame ────────────────────────────────────────────────
        df["adv_target_snr_touch_1"] = touch_1.astype(np.float32)
        df["adv_target_snr_touch_2"] = touch_2.astype(np.float32)

        added = ["adv_target_snr_touch_1", "adv_target_snr_touch_2"]

        # ── Diagnostics ───────────────────────────────────────────────────────
        t1_r = int((touch_1 == RES_CLASS).sum())
        t1_s = int((touch_1 == SUP_CLASS).sum())
        t1_n = int((touch_1 == NONE_CLASS).sum())
        t2_r = int((touch_2 == RES_CLASS).sum())
        t2_s = int((touch_2 == SUP_CLASS).sum())
        t2_n = int((touch_2 == NONE_CLASS).sum())

        # Sequence pair diagnostics: implied bounce/breakout counts
        bb_r = int(((touch_1 == RES_CLASS) & (touch_2 == SUP_CLASS)).sum())  # bounce_resistance
        bb_s = int(((touch_1 == SUP_CLASS) & (touch_2 == RES_CLASS)).sum())  # bounce_support
        bk_r = int(((touch_1 == RES_CLASS) & (touch_2 == RES_CLASS)).sum())  # breakout_resistance
        bk_s = int(((touch_1 == SUP_CLASS) & (touch_2 == SUP_CLASS)).sum())  # breakout_support

        logger.info(
            f"[SNRSeq] ✅ Added SNR ordered two-touch targets (n_future={n_future}, "
            f"touch_pct={zone_touch_pct:.3f}):\n"
            f"   1st touch → R: {t1_r}  S: {t1_s}  None: {t1_n}\n"
            f"   2nd touch → R: {t2_r}  S: {t2_s}  None: {t2_n}\n"
            f"   Implied: bounce_R={bb_r}  bounce_S={bb_s}  "
            f"breakout_R={bk_r}  breakout_S={bk_s}"
        )
        return added

    def _compute_forward_structural_targets(self) -> List[str]:
        """
        Forward Structural Level targets (Trendlines & SNR levels).
        
        These are AUXILIARY ENCODER TRAINING TARGETS that teach the model to recognize
        market structure (channels, support/resistance zones) without dominating the
        main price prediction task.
        
        Why these targets are brilliant:
        1. **Deterministic patterns**: Trendlines have constant slopes (linear regression),
           SNR levels are stationary (sticky zones) → easy for aux heads to learn
        2. **Forces structural awareness**: Encoder must learn "is price in uptrend channel?",
           "approaching resistance?", "breakout imminent?" to predict these
        3. **Low loss weight, high information**: Use very low weights (0.05-0.10) to avoid
           dominating main price head, but still inject critical geometry into hidden state
        4. **Regularization effect**: Predicting structure prevents overfitting to noise
        
        Source columns:
        - Trendlines (from technical_indicators.py):
            Support_Trendline_Value  → adv_target_Support_Trendline_next
            Resist_Trendline_Value   → adv_target_Resist_Trendline_next
        
        - SNR levels (from signal_generator.py extract_snr_features):
            snr_nearest_support_level      → adv_target_snr_nearest_support_next
            snr_nearest_resistance_level   → adv_target_snr_nearest_resistance_next
            snr_support_distance           → adv_target_snr_support_distance_next
            snr_resist_distance            → adv_target_snr_resist_distance_next
        
        Returns:
            List of added target column names
        """
        STRUCTURAL_TARGET_MAP = {
            # Trendline targets (linear channels)
            "Support_Trendline_Value": "adv_target_Support_Trendline_next",
            "Resist_Trendline_Value": "adv_target_Resist_Trendline_next",
            
            # SNR level targets (key zones)
            "snr_nearest_support_level": "adv_target_snr_nearest_support_next",
            "snr_nearest_resistance_level": "adv_target_snr_nearest_resistance_next",
            
            # SNR distance targets (ATR-normalized proximity)
            "snr_support_distance": "adv_target_snr_support_distance_next",
            "snr_resist_distance": "adv_target_snr_resist_distance_next",
        }
        
        added: List[str] = []
        trendline_count = 0
        snr_count = 0
        zone_volume_count = 0

        for src_col, tgt_col in STRUCTURAL_TARGET_MAP.items():
            if src_col not in self.data.columns:
                continue
            
            # Shift by -1 to get next-bar values
            self.data[tgt_col] = self.data[src_col].shift(-1)
            added.append(tgt_col)
            
            # Track category for logging
            if "Trendline" in src_col:
                trendline_count += 1
            elif src_col in ("snr_nearest_zone_volume", "Zonal_Total_Volume", "Zonal_Net_Volume"):
                zone_volume_count += 1
            elif "snr_" in src_col:
                snr_count += 1
        
        if added:
            logger.info(
                f" [AdvTargets] Structural level targets: {len(added)} columns via shift(-1) — "
                f"Trendlines: {trendline_count}, SNR levels: {snr_count}, "
                f"Zone volume: {zone_volume_count}"
            )
            logger.info(
                f"   → Model learns market geometry: channels, support/resistance zones, "
                f"breakout vs bounce patterns, institutional volume at key zones"
            )
        else:
            logger.debug(
                " [AdvTargets] No structural level columns found. Ensure:"
                "\n   - technical_indicators.py ran with enable_additional_features=True (for trendlines)"
                "\n   - signal_generator.py ran (for SNR features)"
            )
        
        return added

    def _compute_forward_reversal_labels(
        self,
        n_future: int = 8,
        decay: float = 0.85,
        hold_threshold: float = 0.65,
    ) -> List[str]:
        """
        Compute forward-looking reversal probability labels.

        THE QUESTION BEING ANSWERED
        ────────────────────────────
        At bar t, the current trend is either UP or DOWN (determined by whether
        Close[t] is above or below EMA_21, falling back to the sign of Close.diff()).

        A "reversal" at bar t means: the next n bars moved AGAINST that trend.
        A "continuation" means: the next n bars moved WITH that trend.

        This is the ground truth the model needs to learn to predict from the
        heuristic Reversal_Score and its sub-components. Without this label,
        the model has no way to know if a high Reversal_Score actually preceded
        a real reversal.

        OUTPUT COLUMNS
        ──────────────
        adv_target_reversal_prob [0,1]
            Exponentially decay-weighted fraction of the next n_future bars that
            moved AGAINST the current trend direction.
            - 1.0 = all bars immediately reversed and stayed reversed
            - 0.5 = mixed (half and half)
            - 0.0 = trend continued for all bars

        adv_target_trend_continuation_prob [0,1]
            Mirror of reversal_prob: 1 - reversal_prob.
            Explicitly provided so both targets are available as separate heads.

        adv_target_reversal_held {0,1}
            1 if the reversal was decisive AND maintained:
            - The very first bar went against the trend (immediate reversal)
            - reversal_prob >= hold_threshold (reversal dominated the window)
            This distinguishes "one bar blip then continuation" from "real reversal".

        EXAMPLE
        ────────
        Bar at t=100. Close=1.0950, EMA_21=1.0900 → trend = UP (bullish).
        A reversal would be: Close[101..108] mostly going DOWN.

        Scenario A — clean reversal:
            bars 101-108: all close < previous close
            → reversal_prob ≈ 1.0, reversal_held = 1, continuation_prob ≈ 0.0

        Scenario B — partial reversal:
            bars 101-104 go down, 105-108 recover
            decay weights front-loaded so early bars count more
            → reversal_prob ≈ 0.55, reversal_held = 0, continuation_prob ≈ 0.45

        Scenario C — no reversal (trend continues):
            bars 101-108: all close > previous close
            → reversal_prob ≈ 0.0, reversal_held = 0, continuation_prob ≈ 1.0

        TREND DIRECTION
        ───────────────
        Priority order:
          1. EMA_21 crossover (Close > EMA_21 = UP, Close < EMA_21 = DOWN)
          2. Supertrend direction column (Cross_Supertrend, renamed from Signal_Supertrend in V8.3)
          3. Close.diff() sign (fallback: positive diff = UP)

        WHY DECAY-WEIGHTED?
        ────────────────────
        A reversal that begins immediately (bar t+1) is much more meaningful than
        one that begins 6 bars later (maybe it's just mean reversion, not a trend
        change). Decay weights give bar t+1 the most importance, shrinking
        exponentially for each subsequent bar.

        RELATIONSHIP TO Reversal_Score
        ───────────────────────────────
        Reversal_Score (from TI._calculate_reversal_score) is an INPUT FEATURE:
        it measures current-bar signals that look like reversal conditions.

        adv_target_reversal_prob is the TRAINING LABEL:
        it records what actually happened in the next n bars.

        The model trains: "given Reversal_Score = 0.8 at a pivot, how often does
        adv_target_reversal_held = 1?" This calibrates the heuristic.

        Args:
            n_future: number of forward bars to look at (default 8)
            decay: exponential decay rate per bar (default 0.85, same as bull/bear strength)
            hold_threshold: minimum reversal_prob to consider reversal "held" (default 0.65)

        Returns:
            List of added column names, or [] if required columns are missing.
        """
        df         = self.data
        n          = len(df)
        added_cols: List[str] = []

        # ── Resolve Close column ─────────────────────────────────────────────
        col_map   = {c.lower(): c for c in df.columns}
        close_col = col_map.get("close", None)
        if close_col is None:
            logger.warning("[AdvTargets] _compute_forward_reversal_labels: no Close column, skipping")
            return added_cols

        close_vals = df[close_col].values.astype(np.float64)

        # ── Determine trend direction at each bar ────────────────────────────
        # UP = 1, DOWN = -1.  Result stored as float array.
        trend_dir = np.zeros(n, dtype=np.float64)

        if "EMA_21" in df.columns:
            ema_vals = df["EMA_21"].values.astype(np.float64)
            # UP when Close > EMA_21, DOWN when Close < EMA_21, 0 when equal (rare)
            trend_dir = np.where(close_vals > ema_vals, 1.0,
                        np.where(close_vals < ema_vals, -1.0, 0.0))
        elif "Cross_Supertrend" in df.columns:
            # Cross_Supertrend (renamed from Signal_Supertrend in V8.3) is +1=bull, -1=bear, 0=flat
            trend_dir = df["Cross_Supertrend"].values.astype(np.float64)
        elif "Signal_Supertrend" in df.columns:
            # Legacy name — kept for backward compatibility with older datasets
            trend_dir = df["Signal_Supertrend"].values.astype(np.float64)
        else:
            # Fallback: sign of one-bar momentum
            diff = np.diff(close_vals, prepend=close_vals[0])
            trend_dir = np.where(diff > 0, 1.0, np.where(diff < 0, -1.0, 0.0))

        # ── Exponential decay weights ────────────────────────────────────────
        wts_full  = np.array([decay ** k for k in range(n_future)], dtype=np.float64)
        wts_full /= wts_full.sum()

        # ── Per-bar directional outcomes for forward window ──────────────────
        # A "reversal bar" within the future window is one where Close moved
        # against the trend at bar t.  We use bar-to-bar close diff inside the
        # window: bar i goes against UP-trend if close[i] < close[i-1].
        # The first bar in the window is compared against close[t] (the anchor).

        reversal_prob_arr     = np.zeros(n, dtype=np.float64)
        reversal_held_arr     = np.zeros(n, dtype=np.float64)

        for i in range(n - n_future):
            td = trend_dir[i]
            if td == 0:
                # Flat trend — skip, leave at 0.0
                continue

            # Forward close sequence: bars t+1 through t+n_future
            fwd = close_vals[i + 1 : i + 1 + n_future]
            # Reference for first bar's direction is current close
            prev = np.concatenate([[close_vals[i]], fwd[:-1]])

            # Directional outcome at each future bar:
            #   bar_diff > 0 → UP bar;  bar_diff < 0 → DOWN bar
            bar_diffs = fwd - prev

            # Against-trend flag: 1 if this bar's direction opposes td
            #   td=+1 (UP trend) → reversal bar has bar_diff < 0
            #   td=-1 (DOWN trend) → reversal bar has bar_diff > 0
            against_trend = np.where(td > 0, bar_diffs < 0, bar_diffs > 0).astype(np.float64)

            wts = wts_full[: len(against_trend)]
            wts = wts / wts.sum()

            rev_prob = float(np.dot(against_trend, wts))
            reversal_prob_arr[i] = rev_prob

            # reversal_held = 1 if:
            #   a) the very first bar was already a reversal bar (immediate flip)
            #   b) the overall reversal_prob >= hold_threshold
            first_bar_reversed = bool(against_trend[0] > 0)
            reversal_held_arr[i] = 1.0 if (first_bar_reversed and rev_prob >= hold_threshold) else 0.0

        # ── Store in dataframe ───────────────────────────────────────────────
        df["adv_target_reversal_prob"]             = np.clip(reversal_prob_arr, 0.0, 1.0)
        df["adv_target_trend_continuation_prob"]   = np.clip(1.0 - reversal_prob_arr, 0.0, 1.0)
        df["adv_target_reversal_held"]             = reversal_held_arr

        added_cols = [
            "adv_target_reversal_prob",
            "adv_target_trend_continuation_prob",
            "adv_target_reversal_held",
        ]

        # Log distribution
        valid_n     = n - n_future
        rev_mean    = reversal_prob_arr[:valid_n].mean()
        held_count  = int(reversal_held_arr[:valid_n].sum())
        up_trend_n  = int((trend_dir[:valid_n] > 0).sum())
        dn_trend_n  = int((trend_dir[:valid_n] < 0).sum())

        logger.info(
            f" [AdvTargets] Reversal labels over {n_future}-bar window "
            f"(decay={decay}, hold_thresh={hold_threshold}):"
        )
        logger.info(
            f"   reversal_prob mean={rev_mean:.3f}   "
            f"reversal_held={held_count} ({held_count/max(valid_n,1)*100:.1f}%)"
        )
        logger.info(
            f"   Trend distribution: UP={up_trend_n} "
            f"({up_trend_n/max(valid_n,1)*100:.1f}%)  "
            f"DOWN={dn_trend_n} ({dn_trend_n/max(valid_n,1)*100:.1f}%)"
        )
        logger.info(
            f"   Trend direction source: "
            f"{'EMA_21' if 'EMA_21' in df.columns else 'Cross_Supertrend' if 'Cross_Supertrend' in df.columns else 'Signal_Supertrend (legacy)' if 'Signal_Supertrend' in df.columns else 'Close.diff()'}"
        )

        return added_cols

    def _convert_binary_columns_to_numeric(self) -> int:

        """
        Convert binary/categorical non-numeric columns to numeric values.
        Examples: yes/no → 1/0, up/down → 1/0, true/false → 1/0, etc.
        
        Returns:
            Number of columns converted
        """
        converted_count = 0
        non_numeric_cols = self.data.select_dtypes(exclude=[np.number]).columns
        
        # Binary mapping patterns (expand as needed)
        binary_mappings = {
            # Yes/No mappings
            ('yes', 'no'): {True: 1, False: 0},
            ('y', 'n'): {True: 1, False: 0},
            # Up/Down mappings
            ('up', 'down'): {True: 1, False: 0},
            ('u', 'd'): {True: 1, False: 0},
            # True/False mappings
            ('true', 'false'): {True: 1, False: 0},
            ('t', 'f'): {True: 1, False: 0},
            # Long/Short mappings (trading)
            ('long', 'short'): {True: 1, False: 0},
            ('l', 's'): {True: 1, False: 0},
            # Buy/Sell mappings (trading)
            ('buy', 'sell'): {True: 1, False: 0},
            # Bullish/Bearish mappings
            ('bullish', 'bearish'): {True: 1, False: 0},
            # On/Off mappings
            ('on', 'off'): {True: 1, False: 0},
            # Win/Loss mappings
            ('win', 'loss'): {True: 1, False: 0},
            ('winner', 'loser'): {True: 1, False: 0},
            # Positive/Negative mappings
            ('positive', 'negative'): {True: 1, False: 0},
            ('pos', 'neg'): {True: 1, False: 0},
        }
        
        for col in non_numeric_cols:
            unique_vals = self.data[col].dropna().unique()
            if len(unique_vals) == 0:
                logger.warning(f"Column {col} is empty, skipping")
                continue
            
            # Check if column has exactly 2 unique values
            if len(unique_vals) == 2:
                val_lower = [str(v).lower() for v in unique_vals]
                
                # Try to match against known binary patterns
                matched = False
                for (pattern1, pattern2), mapping in binary_mappings.items():
                    if (pattern1 in val_lower and pattern2 in val_lower) or \
                       (pattern2 in val_lower and pattern1 in val_lower):
                        # Found a match - convert this binary column
                        conversion_map = {
                            unique_vals[i]: (1 if val_lower[i] == pattern1 else 0)
                            for i in range(2)
                        }
                        self.data[col] = self.data[col].map(conversion_map).fillna(0).astype(float)
                        logger.info(f" Converted binary column '{col}': {unique_vals} → numeric")
                        converted_count += 1
                        matched = True
                        break
                
                if not matched:
                    # Try pd.to_numeric as fallback
                    try:
                        self.data[col] = pd.to_numeric(self.data[col], errors='coerce').fillna(0)
                        logger.info(f" Converted column '{col}' using pd.to_numeric: {unique_vals}")
                        converted_count += 1
                    except Exception as e:
                        logger.warning(f"⚠️ Could not convert column '{col}': {e}, leaving as is")
            
            elif len(unique_vals) <= 10:
                # Try ordinal encoding for low-cardinality categorical columns
                try:
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    self.data[col] = le.fit_transform(self.data[col].astype(str)).astype(float)
                    logger.info(f" Ordinal encoded low-cardinality column '{col}': {len(unique_vals)} unique values")
                    converted_count += 1
                except Exception as e:
                    logger.warning(f"⚠️ Could not ordinal encode column '{col}': {e}")
        
        return converted_count

    def _sanitize_data_for_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sanitize data before scaling to handle inf/NaN values.
        Uses col.describe() to audit ranges and avoids global mean leakage.
        """
        # Case-insensitive column identification for OHLCV
        cols_map = {c.lower(): c for c in df.columns}
        ohlcv_keys = ['open', 'high', 'low', 'close', 'volume', 'tickvolume']
        found_ohlcv = {k: cols_map[k] for k in ohlcv_keys if k in cols_map}

        # FORCE NUMERIC & DEEP FFILL
        for k, actual_col in found_ohlcv.items():
            df[actual_col] = pd.to_numeric(df[actual_col], errors='coerce')
            # 1. Close is the anchor for everything
            if k == 'close':
                df[actual_col] = df[actual_col].ffill()

        # SMART BAR RECONSTRUCTION (Case-Insensitive)
        if all(k in found_ohlcv for k in ['open', 'high', 'low', 'close']):
            c = found_ohlcv['close']
            o = found_ohlcv['open']
            h = found_ohlcv['high']
            l = found_ohlcv['low']

            # Open should be previous close
            df[o] = df[o].fillna(df[c].shift(1)).fillna(df[c])
            # High/Low derived from Open/Close
            df[h] = df[h].fillna(df[[o, c]].max(axis=1))
            df[l] = df[l].fillna(df[[o, c]].min(axis=1))
            # Final physical safety check
            df[h] = df[[h, o, c]].max(axis=1)
            df[l] = df[[l, o, c]].min(axis=1)
        
        # Fill Volume/TickVolume with 0
        for k in ['volume', 'tickvolume']:
            if k in found_ohlcv:
                df[found_ohlcv[k]] = df[found_ohlcv[k]].fillna(0)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 1. Replace inf/-inf with NaN immediately (prevents describe() from failing)
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # 2. Deep Clean: FFill all numeric columns first (standard for time-series)
        df[numeric_cols] = df[numeric_cols].ffill()
        
        for col in numeric_cols:
            # Audit the column using describe() to find gaps and scale issues
            stats = df[col].describe()
            
            # Skip if column is empty or non-numeric
            if stats.get('count', 0) == 0:
                continue

            # Fill NaNs using the median (50%) from describe instead of mean.
            if stats.get('count', 0) < len(df) and '50%' in stats:
                fill_val = stats['50%'] if pd.notna(stats['50%']) else 0.0
                df[col] = df[col].fillna(fill_val)
        
        df = df.fillna(0)
        logger.info(f" Data sanitized: Replaced inf/NaN, filled means, clipped extremes")
        return df

    def _identify_features(self):
        """Identify feature and target columns for consistent scaling.
        
        CRITICAL: Feature column order is preserved throughout processing.
        - This order MUST match the order of features in generated sequences
        - This order MUST match feature_names in the result
        This ensures chart labels correspond to correct data columns.
        """
        # Diagnostic: Log all available columns first
        all_cols = list(self.data.columns)
        logger.info(f"🔍 [MLPrep] Identifying features from {len(all_cols)} total columns")
        
        # Step 1: Convert binary/categorical non-numeric columns to numeric
        converted = self._convert_binary_columns_to_numeric()
        if converted > 0:
            logger.info(f" [MLPrep] Converted {converted} non-numeric columns to numeric")
        
        # 1. Structural/Metadata columns (must be excluded from features)
        structural_cols = ["Date", "index", "Timestamp", "userId", "Time", "time_unix", "datetime", "Datetime", "Timestamp_unix"]
        technical_metadata_cols = ["snr_signals", "signal_type", "Signal_Type", "time_index"]
        
        # 2.  DATA LEAKAGE PROTECTION: Exclude columns that use FUTURE data
        # These are generated by signal_generator.py after analyzing future price movement
        # Or they represent future returns (targets)
        leakage_cols = set(self.CONFIRMED_SIGNAL_COLUMNS)
        
        # Add return targets and directional future-looking columns
        future_looking_cols = [
            "Price_Change", "Direction", "Target_Close", "Close_next", 
            "target_signal", "Next_Day_Return", "Next_3_Day_Return", 
            "Next_5_Day_Return", "Next_Day_Direction", "Target_Direction",
            "Candle_Range"  
        ]
        leakage_cols.update([c for c in future_looking_cols if c in self.data.columns])
        
        # Only exclude CONFIRMED future-bias Signal_ columns + Next_ + Target_ prefixes
        # Lookback-only TI event columns (Cross_Supertrend, Cross_EMA8_Above_EMA12, etc.)
        # are safe features — they are named Cross_* not Signal_* precisely to make this clear.
        # Signal_bounce_support / Signal_breakout_* are excluded (forward-confirmed by signal_generator).
        # Already excluded CONFIRMED_SIGNAL_COLUMNS above - only exclude the blanket Next_* and Target_* prefixes
        next_cols_excluded = [c for c in self.data.columns if c.startswith("Next_")]
        target_cols_excluded = [c for c in self.data.columns if c.startswith("Target_")]
        adv_cols_excluded = [c for c in self.data.columns if c.startswith("adv_target_")]
        
        # Update leakage set from computed lists (avoid duplicate filtering)
        leakage_cols.update(next_cols_excluded)
        leakage_cols.update(target_cols_excluded)
        # dvanced ML target columns (adv_target_ prefix) — always exclude from features
        # These are forward-looking ground-truth targets, never inputs.
        leakage_cols.update(adv_cols_excluded)
        if next_cols_excluded or target_cols_excluded or adv_cols_excluded:
            logger.info(f"🎯 [MLPrep] TARGETS GENERATED & EXCLUDED from features:")
            if next_cols_excluded:
                logger.info(f"   • Next_* targets: {next_cols_excluded}")
            if target_cols_excluded:
                logger.info(f"   • Target_* targets: {target_cols_excluded}")
            if adv_cols_excluded:
                logger.info(f"   • adv_target_* targets: {adv_cols_excluded}")
            logger.info(f"   ➜ These {len(next_cols_excluded) + len(target_cols_excluded) + len(adv_cols_excluded)} columns will be used ONLY as training labels (y), NOT as inputs (X)")
        
        # Build the final exclusion set
        excluded_cols = set(self.config.exclude_columns or [])
        excluded_cols.update(leakage_cols)
        excluded_cols.update([c for c in technical_metadata_cols if c in self.data.columns])
        
        # Case-insensitive exclusion for structural columns
        all_df_cols_lower = {c.lower(): c for c in self.data.columns}
        for s_col in structural_cols:
            if s_col.lower() in all_df_cols_lower:
                excluded_cols.add(all_df_cols_lower[s_col.lower()])

        # Determine numeric columns explicitly
        
        numeric_df = self.data.select_dtypes(include=[np.number])
        numeric_cols = set(numeric_df.columns)
        
        # FEATURE SELECTION: Respect custom_features config if provided
        if self.config.custom_features:
            # Map of col_name -> bool
            # CRITICAL: Even if user selects a leakage column, we MUST override and exclude it
            self.feature_cols = [col for col in self.data.columns 
                                if col in self.config.custom_features 
                                and self.config.custom_features[col] is True
                                and col in numeric_cols
                                and col not in leakage_cols
                                and col not in excluded_cols]
            
            # Log if we overrode user selection for safety
            overridden = [c for c in self.config.custom_features 
                         if self.config.custom_features[c] is True 
                         and (c in leakage_cols or c in excluded_cols) 
                         and c in self.data.columns]
            if overridden:
                 logger.warning(f"🛡️ [MLPrep] Overrode user selection of {len(overridden)} forbidden/leakage columns: {overridden}")
                 
            logger.info(f"🎯 [MLPrep] Using {len(self.feature_cols)} custom features selected by user")
        else:
            # Automatic selection: all numeric minus excluded
            # (All columns are already numeric after _convert_binary_columns_to_numeric)
            self.feature_cols = [col for col in self.data.columns 
                                if col not in excluded_cols and col in numeric_cols]
            logger.info(f"🤖 [MLPrep] Auto-selected {len(self.feature_cols)} features (all converted to numeric)")
        
        # ===== DIAGNOSTIC BREAKDOWN WITH SNR/SIGNAL CATEGORIZATION =====
        non_numeric_cols = [c for c in all_cols if c not in numeric_cols]
        user_excluded = [c for c in all_cols if c in self.config.exclude_columns or []]
        future_excluded = [c for c in all_cols if c in future_looking_cols or c.startswith("Next_")]
        structural_excluded = [c for c in all_cols if c.lower() in [s.lower() for s in structural_cols]]
        metadata_excluded = [c for c in all_cols if c in technical_metadata_cols]
        
        # Analyze feature composition
        snr_features = [c for c in self.feature_cols if c.startswith("snr_")]
        csm_features = [c for c in self.feature_cols if c.startswith("CSM_")]
        confirmed_signals_excluded = [c for c in self.CONFIRMED_SIGNAL_COLUMNS if c in self.data.columns]
        # Legitimate Signal_ columns are those starting with Signal_ but NOT in CONFIRMED_SIGNAL_COLUMNS
        technical_signals_included = [c for c in self.feature_cols if c.startswith("Signal_") and c not in self.CONFIRMED_SIGNAL_COLUMNS]
        
        logger.info(f"🔍 [MLPrep] Column Breakdown ({len(all_cols)} total):")
        logger.info(f"   Numeric columns: {len(numeric_cols)}")
        logger.info(f"   Non-numeric columns: {len(non_numeric_cols)} (excluded by type)")
        if non_numeric_cols:
            logger.info(f"     Examples: {non_numeric_cols[:10]}{'...' if len(non_numeric_cols) > 10 else ''}")
        logger.info(f"  🚫 User exclude_columns: {len(user_excluded)}")
        logger.info(f"  🔮 Future-looking/Leakage excluded: {len(future_excluded)}")
        logger.info(f"  📅 Structural excluded: {len(structural_excluded)}")
        logger.info(f"  🏷️ Metadata excluded: {len(metadata_excluded)}")
        logger.info(f"   CONFIRMED Future-confirmed Signal_* EXCLUDED: {len(confirmed_signals_excluded)}")
        if confirmed_signals_excluded:
            logger.info(f"     Excluded (Leakage): {confirmed_signals_excluded}")
        
        logger.info(f'All Excluded columns: {excluded_cols}')
        logger.info(f"   SNR FEATURES INCLUDED: {len(snr_features)} (pure lookback)")
        if snr_features:
            logger.info(f"     Examples: {snr_features[:8]}{'...' if len(snr_features) > 8 else ''}")
        if csm_features:
            logger.info(f"   CSM FEATURES INCLUDED: {len(csm_features)} (Currency Strength Matrix, Partition B/diff)")
            logger.info(f"     Columns: {csm_features}")
        if technical_signals_included:
            logger.info(f"   Technical Signal_* INCLUDED: {len(technical_signals_included)} (lookback indicators)")
            logger.info(f"     Examples: {technical_signals_included[:5]}{'...' if len(technical_signals_included) > 5 else ''}")
        logger.info(f"  ➡️ FINAL FEATURES INCLUDED: {len(self.feature_cols)}")

        # Build named feature groups (index arrays into feature_cols) for the model builder.
        # Each group maps a semantic name → list of column-index integers so a dedicated
        # sub-encoder / attention head can be constructed without hardcoding column names.
        self._feature_groups: Dict[str, List[int]] = {}
        for group_name, prefix in [("snr", "snr_"), ("csm", "CSM_")]:
            group_indices = [
                i for i, c in enumerate(self.feature_cols) if c.startswith(prefix)
            ]
            if group_indices:
                self._feature_groups[group_name] = group_indices
        if self._feature_groups:
            logger.info(
                f"   📐 Feature groups registered: "
                + ", ".join(f"{k}({len(v)} cols)" for k, v in self._feature_groups.items())
            )
        
        # ===== CRITICAL VERIFICATION: Confirm advanced targets are NOT in sequences =====
        # Advanced targets should NEVER appear as features in sequence data
        # They are outputs only, not inputs
        adv_targets_in_features = [c for c in self.feature_cols if c.startswith('adv_target_')]
        if adv_targets_in_features:
            logger.error(f"❌ [CRITICAL] Advanced targets found in sequence features: {adv_targets_in_features}")
            logger.error(f"   These should be EXCLUDED from sequences (outputs only, not inputs)")
            raise ValueError(f"Advanced target leakage detected in feature_cols: {adv_targets_in_features}")
        else:
            logger.info(f"✅ [VERIFIED] No advanced targets in sequence features (adv_target_* properly excluded)")
        
        # ===== VERIFICATION: Confirm Next_* and Target_* targets are NOT in sequences =====
        next_targets_in_features = [c for c in self.feature_cols if c.startswith('Next_') or c.startswith('Target_')]
        if next_targets_in_features:
            logger.error(f"❌ [CRITICAL] Return targets found in sequence features: {next_targets_in_features}")
            raise ValueError(f"Target leakage detected in feature_cols: {next_targets_in_features}")
        else:
            logger.info(f"✅ [VERIFIED] No return targets in sequence features (Next_*/Target_* properly excluded)")
        
        # Log exclusion reasons for transparency
        if len(self.feature_cols) < len(all_cols):
            hidden = [c for c in all_cols if c not in self.feature_cols]
            logger.info(f"🚫 [MLPrep] Excluded {len(hidden)} non-feature columns")
       
        # Columns to scale: All features EXCEPT structural indices.
        # DYNAMIC LOG-SCALING: Catch any extreme magnitude columns (OBV, Money Flow, etc.)
        # We exclude Time-related columns as they need Standard/Robust scaling instead.
        time_terms = ['time', 'date', 'timestamp', 'unix', 'index']
        potential_log_cols = [
            c for c in self.data.columns 
            if pd.api.types.is_numeric_dtype(self.data[c])
            and not any(term in c.lower() for term in time_terms)
        ]
        
        
        # SYNC SCALING: Ensure ALL features are included in columns_to_scale
        # This prevents 1.0e18 unscaled timestamp leaks.
        self.columns_to_scale = [c for c in self.feature_cols]

        # Explicitly add target columns to scaling list (crucial for regression)
        if self.config.target_columns:
            for t_col in self.config.target_columns:
                if t_col in self.data.columns and t_col not in self.columns_to_scale:
                    if pd.api.types.is_numeric_dtype(self.data[t_col]):
                        self.columns_to_scale.append(t_col)
        
        # CRITICAL: Add advanced ML target columns to scaling (same scale as inputs)
        # These adv_target_* columns are regression targets that must be scaled [0,1]
        # so model learns to output [0,1], then inverse-transform at inference.
        # EXCLUDING CATEGORICAL TARGETS: Integer class labels (snr_touch_1/2, reversal_held, bull_class, etc.)
        # MUST remain unscaled integers to prevent corrupting sparse categorical loss functions.
        _CATEGORICAL_ADV_TARGETS = {
            'adv_target_snr_touch_1',
            'adv_target_snr_touch_2',
            'adv_target_reversal_held',
            'adv_target_bull_class',
            'adv_target_session_next',
            'adv_target_session_transition_next',
            'adv_target_day_of_week_next',
            'adv_target_hour_next',
            'adv_target_minute_next',
        }
        adv_targets_for_scaling = [
            c for c in self.data.columns 
            if c.startswith('adv_target_') and c not in _CATEGORICAL_ADV_TARGETS
        ]
        for adv_col in adv_targets_for_scaling:
            if adv_col not in self.columns_to_scale and pd.api.types.is_numeric_dtype(self.data[adv_col]):
                self.columns_to_scale.append(adv_col)
        
        if adv_targets_for_scaling:
            logger.info(f"✅ [MLPrep] Added {len(adv_targets_for_scaling)} advanced regression targets to scaling (excluding {len(_CATEGORICAL_ADV_TARGETS)} categorical targets): {adv_targets_for_scaling[:5]}{'...' if len(adv_targets_for_scaling) > 5 else ''}")

        # Audit check
        missing_scale = [c for c in self.feature_cols if c not in self.columns_to_scale]
        if missing_scale:
            logger.warning(f"⚠️ [MLPrep] Features missing from scaler: {missing_scale}")
        
        logger.info(f" [MLPrep] Identified {len(self.feature_cols)} features and {len(self.columns_to_scale)} columns to scale (including {len(adv_targets_for_scaling)} adv_targets)")

    async def _scale_dataframe_splits(self) -> Dict[str, pd.DataFrame]:
        """Split dataframe and scale based on training split (prevents leakage).
        
        NEW: If is_pre_split=True, treats entire DataFrame as a single split.
        This is used when ProcessingManager has already split and scaled the data.
        """
        stage_info = PREPARATION_STAGES['scaling']
        
        # Handle pre-split data (already split by ProcessingManager)
        if self.is_pre_split:
            logger.info(f"[MLPrep] Data is pre-split - treating entire DataFrame as single split")
            
            if self.reporter:
                await self.reporter.report_async(
                    progress=stage_info['start'],
                    message="Using pre-split data...",
                    message2=f"Processing {len(self.data)} rows as single split (already scaled)"
                )
            
            # Return the entire DataFrame as the only split
            # The split name doesn't matter since we're only processing one
            result = {
                "train": self.data.copy(),
                "validation": pd.DataFrame(),  # Empty
                "test": pd.DataFrame()  # Empty
            }
            
            if self.reporter:
                await self.reporter.report_async(
                    progress=stage_info['end'],
                    message="Pre-split data ready ✓",
                    message2=f"Single split with {len(self.data)} rows"
                )
            
            return result
        
        # ORIGINAL LOGIC: Split and scale when not pre-split
        n_rows = len(self.data)
        train_end = int(n_rows * self.config.train_ratio)
        val_end = int(n_rows * (self.config.train_ratio + self.config.validation_ratio))
        
        # Sequential splitting of the dataframe
        train_df = self.data.iloc[:train_end].copy()
        val_df = self.data.iloc[train_end:val_end].copy()
        test_df = self.data.iloc[val_end:].copy()
        
        if self.reporter:
            await self.reporter.report_async(
                progress=stage_info['start'],
                message="Scaling dataframe splits...",
                message2=f"Fitting scaler on training slice (rows 0 to {train_end})..."
            )
        
        # Sanitize data BEFORE scaler fitting
        # This prevents StandardScaler from failing on inf/NaN values
        logger.info(f"🧹 [MLPrep] Sanitizing splits before scaling...")
        train_df = self._sanitize_data_for_scaling(train_df)
        val_df = self._sanitize_data_for_scaling(val_df)
        test_df = self._sanitize_data_for_scaling(test_df)
        logger.info(f" All splits sanitized (inf/NaN replaced, extremes clipped)")
        
        # Fit scaler ONLY on training data for columns_to_scale
        # CRITICAL: Skip fit if scaler was provided externally (Global fit pattern)
        if self.scaler is None:
            logger.info(f"📊 [MLPrep] Fitting MultiPartitionScaler on columns_to_scale ({len(self.columns_to_scale)} columns)...")
            self.scaler = MultiPartitionScaler(
                price_scaler_type=getattr(self.config, 'price_scaler_type', 'none'),
                diff_scaler_type=getattr(self.config, 'diff_scaler_type', 'robust'),
                other_scaler_type=getattr(self.config, 'other_scaler_type', 'standard'),
                columns_to_scale=self.columns_to_scale
            )
            # Fit on training features/targets ONLY when not provided externally
            self.scaler.fit(train_df[self.columns_to_scale])
            logger.info(f" MultiPartitionScaler fitted on training slice (rows 0 to {train_end})")
            
            # Save scaler if requested by config
            if self.config.save_scaler and self.config.scaler_filename:
                try:
                    import joblib
                    import os
                    
                    # Create scalers directory if it doesn't exist
                    scaler_dir = os.path.dirname(self.config.scaler_filename) or "scalers"
                    os.makedirs(scaler_dir, exist_ok=True)
                    
                    # Save scaler
                    scaler_path = self.config.scaler_filename
                    joblib.dump(self.scaler, scaler_path)
                    logger.info(f"💾 Scaler saved to: {scaler_path}")
                    
                    # Update scaler_save_path for metadata
                    self.config.scaler_save_path = scaler_path
                except Exception as e:
                    logger.warning(f"⚠️ Failed to save scaler: {e}")
        else:
            logger.info(f" Using externally provided scaler (Global Fit mode)")
        
        # Transform all splits
        train_df[self.columns_to_scale] = self.scaler.transform(train_df[self.columns_to_scale])
        val_df[self.columns_to_scale] = self.scaler.transform(val_df[self.columns_to_scale])
        test_df[self.columns_to_scale] = self.scaler.transform(test_df[self.columns_to_scale])
        
        if self.reporter:
            await self.reporter.report_async(
                progress=stage_info['end'],
                message="Scaling complete ✓",
                message2=f"Dataframe splits scaled consistently (fit on train only)."
            )
        
        return {
            "train": train_df,
            "validation": val_df,
            "test": test_df
        }

    async def _generate_sequences_from_splits(self, scaled_splits: Dict[str, pd.DataFrame], enriched_target_columns: Optional[List[str]] = None):
        """Async generator that yields each split as it completes sequence generation.
        
        CRITICAL: This is an async generator that yields (split_name, split_data) tuples.
        
        Args:
            scaled_splits: Dict of {split_name: DataFrame}
            enriched_target_columns: Complete list of adv_target_* columns from enrichment phase.
                                     Passed through to _generate_sequences_for_df so workers use
                                     the correct target list instead of scanning a slice.
        """
        stage_info = PREPARATION_STAGES['generating']
        
        # Filter out empty splits
        non_empty_splits = [(name, df) for name, df in scaled_splits.items() if len(df) > 0]
        total_steps = len(non_empty_splits)
        
        for idx, (name, df) in enumerate(non_empty_splits):
            step_start = stage_info['start'] + (idx / total_steps) * (stage_info['end'] - stage_info['start'])
            step_end = stage_info['start'] + ((idx + 1) / total_steps) * (stage_info['end'] - stage_info['start'])
            
            logger.info(f"🔄 Generating sequences for {name} split ({len(df)} rows)...")
            
            # Always use the provided df (which is already correctly sliced/scaled)
            # for both features and metadata in pre-split/worker mode.
            split_data = await self._generate_sequences_for_df(
                df, 
                name, 
                step_start, 
                step_end,
                enriched_target_columns=enriched_target_columns  # FIX: forward to sequence gen
            )
            
            yield (name, split_data)

    def _compute_movement_analysis_batch(self, df: pd.DataFrame, positions: List[int]) -> Dict[int, Dict]:
        """
        ⚡ THREADED BATCH PROCESSOR: Compute movement analysis for multiple positions in parallel.
        
        This runs movement analysis calls in a thread pool instead of sequentially, reducing
        overhead when processing 100+ sequences. The ProcessingManager already uses parallel workers
        at a high level, so this fine-grained threading focuses on CPU-intensive analysis within each worker.
        
        Args:
            df: DataFrame with price data
            positions: List of signal position indices to analyze
            
        Returns:
            Dict[pos -> movement_analysis_result], with None for failed analyses
        """
        if not positions or not getattr(self.config, 'prepare_advanced_ml_targets', False):
            return {}
        
        results = {}
        
        # Use max 4 threads — higher concurrency risks GIL contention on data access
        max_workers = min(4, len(positions))
        
        def _compute_one_analysis(pos: int) -> Tuple[int, Optional[Dict]]:
            """Single worker task: compute movement analysis for one position."""
            try:
                signal_type = self._detect_signal_type_at_idx(df, df.index[pos])
                level_price = df['Close'].iloc[pos]
                
                movement_analysis = _analyze_post_interaction_movement(
                    df=df,
                    signal_index=pos,
                    level_price=level_price,
                    signal_type=signal_type,
                    lookforward_period=self.config.prediction_length,
                    lookback_period=self.config.sequence_length,
                    reporter=None,
                    base_progress=0
                )
                return (pos, movement_analysis)
            except Exception as e:
                logger.warning(f"⚠️ Movement analysis failed at pos {pos}: {e}")
                return (pos, None)
        
        # Submit all tasks to thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_compute_one_analysis, pos): pos for pos in positions}
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    pos, analysis = future.result()
                    results[pos] = analysis
                except Exception as e:
                    pos = futures[future]
                    logger.error(f" Thread pool error for pos {pos}: {e}")
                    results[pos] = None
        
        logger.info(f" [MLPrep] Batch movement analysis complete: {len(results)} positions, "
                   f"{sum(1 for a in results.values() if a is not None)} successful")
        
        return results

    async def _generate_sequences_for_df(self, df: pd.DataFrame, split_name: str, start_pct: float, end_pct: float, enriched_target_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
            Generate labeled training sequences from a single DataFrame slice for ML model training.

            This method is the core sequence factory of the pipeline. It scans a time-series
            DataFrame for trading signal occurrences, extracts fixed-length lookback windows
            centered around each signal event, pairs them with forward-looking target values,
            and returns a structured batch of (sequence, label, target) triples ready for
            model ingestion.

            Parameters
            ----------
            df : pd.DataFrame
                A time-series DataFrame slice containing OHLCV data, technical indicators,
                and signal columns. Rows represent time steps (bars), columns represent features.
            split_name : str
                Identifier for this data partition (e.g. 'train', 'val', 'test'). Used for
                logging, progress reporting, and disk cache directory naming.
            start_pct : float
                Starting progress percentage passed to the reporter (0.0 to 1.0). Used to
                compute incremental progress updates relative to the caller's progress range.
            end_pct : float
                Ending progress percentage passed to the reporter (0.0 to 1.0). Defines the
                upper bound of the progress range this method is responsible for reporting.

            Returns
            -------
            dict
                A dictionary with the following keys:

                In-memory mode (use_lazy_storage=False):
                    sequences : np.ndarray, shape (N, seq_len, num_features), dtype float32
                        Lookback windows. Each entry is a 2D array of shape (seq_len, num_features)
                        representing seq_len time steps of feature data ending just before
                        the signal position.
                    labels : np.ndarray, shape (N,), dtype int32
                        Class label for each sequence. Positive samples are assigned a signal
                        type code via signal_mapping (e.g. 0=BUY, 1=SELL). Negative samples
                        (no-signal rows) are assigned label 4 (NONE).
                    targets : dict[str, np.ndarray]
                        Regression or sequence prediction targets keyed by target column name.
                        Shape is (N,) for single-step targets or (N, pred_len) for multi-step
                        and future_sequence targets.
                    sequence_metadata : list[dict]
                        Per-sequence metadata dicts (timestamp, position, detected signal type,
                        active feature list, etc.) in the same order as sequences.
                    indices : list[int]
                        Sequential integer indices [0, 1, ..., N-1] for downstream dataset mapping.

                Lazy storage mode (use_lazy_storage=True):
                    data_type : str
                        Always 'lazy_npz'.
                    file_path : str
                        Absolute path to the compressed .npz file on disk containing sequences,
                        labels, and targets.
                    total_sequences : int
                        Number of valid sequences written to disk.
                    sequence_metadata : list[dict]
                        Same as in-memory mode.
                    indices : list[int]
                        Same as in-memory mode.

            Raises
            ------
            No exceptions are raised directly. Individual sequence errors are caught internally
            and logged as warnings; the affected position is skipped and processing continues.

            Notes
            -----
            Sequence Construction:
                For each candidate position `pos`, the lookback window is:
                    df[feature_cols].iloc[pos - seq_len + 1 : pos + 1]
                meaning the window contains seq_len bars of history ending exactly
                ON the signal bar. This ensures the model sees the trigger state.

            Boundary Safety:
                Only positions in the range [sequence_length, len(df) - prediction_length]
                are considered, ensuring every window has a full lookback history and every
                target has a full forward horizon.

            Positive vs. Negative Sampling:
                Positive positions  — rows where any signal column exceeds 0.5.
                Negative positions  — rows where no signal fires, randomly downsampled to
                                    int(n_positives * negative_sampling_ratio) samples
                                    using config.random_seed for reproducibility.

            Leakage Protection:
                If mask_future_signals=True, the last `signal_leakage_buffer` bars of all
                CONFIRMED_SIGNAL_COLUMNS within each lookback window are zeroed out before
                storage. SNR features are intentionally left intact as they are pure lookback
                indicators with no forward leakage.

            Feature Filtering:
                If exclude_signals=True, all CONFIRMED_SIGNAL_COLUMNS are removed from the
                feature set permanently (self.feature_cols is mutated in place) before
                any sequence is extracted.

            Lazy Storage:
                If use_lazy_storage=True, the full sequence arrays are compressed and written
                to:
                    app/data/temp_sequences/{task_id}/{split_name}/sequences.npz
                and only a lightweight pointer dict is returned, keeping RAM usage flat
                regardless of dataset size.

            Progress Reporting:
                If self.reporter is set, async progress updates are emitted every 100
                sequences, interpolated linearly between start_pct and 80% of end_pct.

            Example
            -------
            Assuming a 1000-row DataFrame with seq_len=50, pred_len=5, and 200 signal rows:

                # Safe positions: rows 50 to 995
                # Positive positions found: 200 (signal fired)
                # Negative positions sampled: 400 (negative_sampling_ratio=2.0)
                # Total sequences: up to 600

                result = await self._generate_sequences_for_df(df, 'train', 0.0, 0.8)
                result['sequences'].shape   # (600, 50, num_features)
                result['labels'].shape      # (600,)
                result['targets']['return_1d'].shape  # (600, 5)
            """
        # Determine active target keys based on config
        target_keys = []
        #  FIX: For classification tasks, we still need to generate targets (the actual values)
        # even though the primary output is labels. Targets are used for evaluation and analysis.
        if self.config.include_classification or self.config.include_regression:
            target_keys = list(self.config.target_columns)
        if self.config.include_sequence_prediction:
            target_keys.append('future_sequence')
        
        #  NEW: Automatically append Advanced ML Target keys for prediction/analysis tasks
        # FIX: Remove double-gate - collect advanced targets whenever enabled, not just for sequence prediction
        if getattr(self.config, 'prepare_advanced_ml_targets', False):
            
            # FIX: Use enriched_target_columns if provided (from PM), otherwise scan columns (fallback)
            if enriched_target_columns:
                # Use the pre-computed list from enrichment phase (passed from PM)
                # This is the most reliable method as it contains ALL targets from enrichment
                for target_col in enriched_target_columns:
                    if target_col not in target_keys:
                        target_keys.append(target_col)
                logger.info(
                    f"[ML Prep] 🎯 Using {len(enriched_target_columns)} enriched targets "
                    f"passed from PM (no column scanning needed): {enriched_target_columns[:10]}..."
                )
            else:
                # Fallback: scan columns and use hard-coded lists (current behavior)
                logger.warning(
                    f"[ML Prep] ⚠️ enriched_target_columns not provided, falling back to column scanning. "
                    f"This may miss some targets if DataFrame was sliced after enrichment."
                )
                
                # Momentum targets (already shifted in _enrich_with_targets)
                momentum_targets = []
                for col in self.data.columns:
                    if col.startswith('adv_target_') and col.endswith('_next'):
                        momentum_targets.append(col)
            
                # OHLCV sequence targets (already exist from _collect_next_candle_ohlcv_targets)
                ohlcv_targets = [
                    "adv_target_Open_seq",
                    "adv_target_High_seq",
                    "adv_target_Low_seq",
                    "adv_target_Close_seq",
                    "adv_target_Volume_seq"
                ]
                
                # Movement analysis targets (will be calculated per-sequence)
                # These map to keys returned by analyze_post_interaction_movement()
                movement_targets = [
                    "adv_target_max_favorable_pct",
                    "adv_target_max_adverse_pct",
                    "adv_target_final_move_pct",
                    "adv_target_risk_reward_ratio",
                    "adv_target_avg_volatility",
                    "adv_target_volatility_surge",
                    "adv_target_time_to_max_favorable",
                    "adv_target_time_to_max_adverse",
                    "adv_target_signal_strength",
                ]
                
                # FIX: Add Dual-Head classification targets (created in _compute_dual_head_labels)
                dual_head_targets = [
                    "adv_target_bull_prob",
                    "adv_target_bull_class",
                    "adv_target_bull_conf",
                    "adv_target_bear_conf",
                ]
                
                # FIX: Add MFE/MAE targets (created in _compute_mfe_mae)
                mfe_mae_targets = [
                    "adv_target_MFE",
                    "adv_target_MAE",
                ]
                
                # FIX: Add forward log return targets (created in _compute_forward_log_returns)
                logret_targets = [
                    "adv_target_logret_1",
                    "adv_target_logret_5",
                    "adv_target_logret_10",
                    "adv_target_logret_20",
                ]
                
                # FIX: Add bull/bear strength targets (created in _compute_forward_bull_bear_strength)
                strength_targets = [
                    "adv_target_bull_strength_8",
                    "adv_target_bear_strength_8",
                ]
                
                # FIX: Add session/time feature targets (created in _enrich_with_targets)
                session_time_targets = [
                    "adv_target_session_next",
                    "adv_target_session_transition_next",
                    "adv_target_day_of_week_next",
                    "adv_target_hour_next",
                    "adv_target_minute_next",
                ]
                
                # FIX: Add Price Velocity targets (next-bar shift + forward average)
                velocity_targets = [
                    "adv_target_Price_Velocity_Bull_next",
                    "adv_target_Price_Velocity_Bear_next",
                    "adv_target_Price_Velocity_Net_next",
                    "adv_target_vel_bull_fwd_8",
                    "adv_target_vel_bear_fwd_8",
                    "adv_target_vel_net_fwd_8",
                ]

                # FIX: Add Volatility Regime targets (next-bar shift + forward average)
                volatility_targets = [
                    "adv_target_Volatility_Regime_next",
                    "adv_target_Volatility_Expansion_next",
                    "adv_target_Volatility_Bull_next",
                    "adv_target_Volatility_Bear_next",
                    "adv_target_vol_regime_fwd_8",
                    "adv_target_vol_expansion_fwd_8",
                ]

                # FIX: Add Regime Speed targets (next-bar shift + forward average)
                regime_speed_targets = [
                    "adv_target_Regime_Speed_Bull_next",
                    "adv_target_Regime_Speed_Bear_next",
                    "adv_target_Regime_Speed_Aligned_next",
                    "adv_target_Regime_Speed_Divergence_next",
                    "adv_target_speed_aligned_fwd_8",
                    "adv_target_speed_divergence_fwd_8",
                ]

                # FIX: Add Reversal targets (created in _compute_forward_reversal_labels)
                reversal_targets = [
                    "adv_target_reversal_prob",
                    "adv_target_trend_continuation_prob",
                    "adv_target_reversal_held",
                ]

                # FIX: Add Next-Zone targets (created in _compute_next_zone_targets)
                next_zone_targets = [
                    "adv_target_next_zone_idx",
                    "adv_target_next_zone_bars",
                    "adv_target_next_zone_distance",
                    "adv_target_next_zone_volume",
                    # SNR ordered two-touch sequence (created in _compute_snr_zone_sequence_targets)
                    # 3-class label per head: 0=resistance, 1=support, 2=none
                    # snr_touch_2 encodes bounce vs breakout at the first-touch zone.
                    "adv_target_snr_touch_1",
                    "adv_target_snr_touch_2",
                ]

                # FIX: Add Structural Level targets (created in _compute_forward_structural_targets)
                structural_targets = [c for c in self.data.columns if c.startswith("adv_target_") and any(x in c for x in ["Trendline", "snr_"])]

                # FIX: Add Signal classification targets
                signal_targets = [c for c in self.data.columns if c.startswith("Signal_")]

                # Combine all advanced targets
                all_advanced_targets = (
                    momentum_targets + 
                    ohlcv_targets + 
                    movement_targets + 
                    dual_head_targets + 
                    mfe_mae_targets + 
                    logret_targets + 
                    strength_targets + 
                    session_time_targets +
                    velocity_targets +
                    volatility_targets +
                    regime_speed_targets +
                    reversal_targets +
                    next_zone_targets +
                    structural_targets +
                    signal_targets
                )
                
                # Add all advanced targets to target_keys
                for t in all_advanced_targets:
                    if t not in target_keys:
                        target_keys.append(t)
                
                logger.info(
                    f"[ML Prep] 🎯 Collected {len(all_advanced_targets)} advanced targets: "
                    f"momentum={len(momentum_targets)}, ohlcv={len(ohlcv_targets)}, "
                    f"movement={len(movement_targets)}, dual_head={len(dual_head_targets)}, "
                    f"mfe_mae={len(mfe_mae_targets)}, logret={len(logret_targets)}, "
                    f"strength={len(strength_targets)}, session_time={len(session_time_targets)}, "
                    f"velocity={len(velocity_targets)}, volatility={len(volatility_targets)}, "
                    f"regime_speed={len(regime_speed_targets)}"
                )

        # ── DIAGNOSTIC: Log the complete target_keys list ──────────────────────────────
        # Partition into buckets to make it easy to spot what's missing
        _seq_targets   = [k for k in target_keys if k.endswith('_seq') or k == 'future_sequence']
        _next_targets  = [k for k in target_keys if k.endswith('_next')]
        _scalar_targets = [k for k in target_keys if k not in _seq_targets and k not in _next_targets]
        logger.info(
            f"[ML Prep] 🎯 Final target_keys for {split_name} ({len(target_keys)} total):\n"
            f"   Sequence targets ({len(_seq_targets)}): {_seq_targets}\n"
            f"   Next-bar targets ({len(_next_targets)}): {_next_targets}\n"
            f"   Scalar targets   ({len(_scalar_targets)}): {_scalar_targets}"
        )
        # ──────────────────────────────────────────────────────────────────────────────
        
        sequence_metadata = []
        signal_prefixes = ["Signal_", "snr_", "signal_"]
        signal_suffixes = ["_signal", "_Signal"]
        
        # 1. Feature Filtering & Diagnostic Logging (Restored)
        snr_feature_cols = [c for c in df.columns if c.startswith("snr_")]
        
        # If exclude_signals config, drop CONFIRMED Signal_ columns from feature set
        feature_cols = list(self.feature_cols)
        if self.config.exclude_signals:
            n_before = len(feature_cols)
            feature_cols = [c for c in feature_cols if c not in self.CONFIRMED_SIGNAL_COLUMNS]
            n_removed = n_before - len(feature_cols)
            logger.info(f"🚫 [MLPrep] Excluding {n_removed} CONFIRMED Signal_* from features")
            self.feature_cols = feature_cols

       
        if self.config.mask_future_signals:
            # Log intent here; actual index computation happens post-mutation (see PRE-ALLOCATION)
            logger.info(
                f"🛡️ [MLPrep] Leakage protection REQUESTED: will zero last "
                f"{self.config.signal_leakage_buffer} bars of CONFIRMED Signal_* after feature setup"
            )
            logger.info(f"     SNR features ({len(snr_feature_cols)}) intact - pure lookback")
        
        # 2. Determine boundaries and identify Signal columns for discovery
        # CRITICAL: end_idx must account for the MAXIMUM lookahead required by any target
        target_mapping = {
            "Next_Day_Return": 1,
            "Next_3_Day_Return": 3,
            "Next_5_Day_Return": 5,
            "Next_Day_Direction": 1
        }
        max_lookahead = self.config.prediction_length
        for t_col in target_keys:
            if t_col in target_mapping:
                max_lookahead = max(max_lookahead, target_mapping[t_col])
        
        start_idx = self.config.sequence_length
        end_idx = len(df) - max_lookahead
        
        if end_idx <= start_idx:
            logger.warning(f"Slice '{split_name}' too small for sequence generation ({len(df)} rows)")
            return {
                "sequences": np.array([]), "labels": np.array([]),
                "targets": {k: np.array([]) for k in target_keys},
                "sequence_metadata": []
            }

        # Last-minute pass to catch NaNs created by split boundaries
        # This prevents a single NaN at T+14 from nuking the previous 14 valid samples.
        df = df.ffill().bfill()

        # Identify signal columns for positive sample discovery (triggers)
        # CRITICAL: We ONLY want to trigger positive samples on confirmed BUY/SELL/BOUNCE signals.
        # Technical indicators (like Signal_Supertrend) should be FEATURES, not triggers.
        # Triggers must be in CONFIRMED_SIGNAL_COLUMNS or have a mapping in SIGNAL_MAPPING.
        active_signal_cols = []
        for col in df.columns:
            if col in self.CONFIRMED_SIGNAL_COLUMNS:
                active_signal_cols.append(col)
                continue
                
            # Check if it's a known mapped signal type after cleaning prefixes/suffixes
            clean_name = col
            for p in ["Signal_", "snr_", "signal_"]:
                if clean_name.startswith(p):
                    clean_name = clean_name[len(p):]
                    break
            for s in ["_signal", "_Signal"]:
                if clean_name.endswith(s):
                    clean_name = clean_name[:-len(s)]
                    break
            
            if clean_name in self.SIGNAL_MAPPING and clean_name not in ["none", "no_signal"]:
                active_signal_cols.append(col)

        if active_signal_cols:
            logger.info(f"🎯 [MLPrep] Using {len(active_signal_cols)} columns as triggers for {split_name}: {active_signal_cols}")
            signal_mask = (df[active_signal_cols] > 0.5).any(axis=1).values
        else:
            # Fallback if no signals found
            logger.warning(f"⚠️ [MLPrep] No confirmed signal columns found for {split_name} triggers! Falling back to sequential sampling of all valid positions.")
            signal_mask = np.ones(len(df), dtype=bool)
        
        valid_pos_mask = np.zeros(len(df), dtype=bool)
        valid_pos_mask[start_idx:end_idx] = signal_mask[start_idx:end_idx]
        pos_positions = np.where(valid_pos_mask)[0].tolist()
        
        neg_range_mask = np.zeros(len(df), dtype=bool)
        neg_range_mask[start_idx:end_idx] = ~signal_mask[start_idx:end_idx]
        potential_neg_positions = np.where(neg_range_mask)[0].tolist()
        
        neg_positions = []
        if self.config.negative_sampling_ratio > 0 and pos_positions:
            target_neg = int(len(pos_positions) * self.config.negative_sampling_ratio)
            target_neg = min(target_neg, len(potential_neg_positions))

            # Bug F fix: use a local RNG instance instead of mutating global random state,
            # so concurrent split generation (train/val/test in parallel) stays reproducible.
            _rng = random.Random(self.config.random_seed)
            neg_positions = _rng.sample(potential_neg_positions, target_neg)

        all_positions = pos_positions + neg_positions
        
        # CRITICAL FIX: Filter positions that don't have enough future data for target sequences
        # Sequences at position `pos` need data from pos+1 to pos+1+pred_len for target computation.
        # Without this, sequence targets like adv_target_Open_seq will return None and skip the sample.
        pred_len = self.config.prediction_length
        seq_len = self.config.sequence_length
        valid_all_positions = [
            pos for pos in all_positions 
            if pos >= seq_len - 1 and pos + 1 + pred_len <= len(df)
        ]
        all_positions = valid_all_positions
        total_samples = len(all_positions)
        
        if total_samples == 0:
            return {
                "sequences": np.array([]), "labels": np.array([]),
                "targets": {k: np.array([]) for k in target_keys},
                "sequence_metadata": []
            }

        # PRE-ALLOCATION
        num_features = len(self.feature_cols)

        signal_feature_indices = [
            j for j, col in enumerate(self.feature_cols)
            if col in self.CONFIRMED_SIGNAL_COLUMNS
        ]
        if self.config.mask_future_signals and signal_feature_indices:
            logger.info(
                f"🛡️ [MLPrep] Leakage protection ACTIVE: zeroing last "
                f"{self.config.signal_leakage_buffer} bars of "
                f"{len(signal_feature_indices)} CONFIRMED Signal_* columns"
            )

        sequences_final = np.zeros((total_samples, seq_len, num_features), dtype=np.float32)
        labels_final = np.zeros(total_samples, dtype=np.int32)
        
        targets_final = {}
        target_masks_final = {}
        for k in target_keys:
            # Determine target shape based on target type
            # ─────────────────────────────────────────────────────────────────────
            # Sequence targets (_seq, _next suffixes, future_sequence): (n_sequences, pred_len)
            # Scalar targets (movement analysis, pre-shifted returns, primary targets): (n_sequences,)
            
            # Primary target columns (config.target_columns) are always scalars
            is_primary_target = k in self.config.target_columns
            
            # Sequence targets end with specific suffixes or are future_sequence
            is_sequence_target = (
                (k == 'future_sequence' 
                 or k.endswith('_seq')
                 or k.endswith('_next'))
                and not is_primary_target
            )
            
            if is_sequence_target and pred_len > 1:
                # Multi-step sequence target: shape (total_samples, pred_len)
                targets_final[k] = np.zeros((total_samples, pred_len), dtype=np.float32)
                target_masks_final[k] = np.zeros((total_samples, pred_len), dtype=np.float32)
            else:
                # Scalar target: shape (total_samples,)
                targets_final[k] = np.zeros(total_samples, dtype=np.float32)
                target_masks_final[k] = np.zeros(total_samples, dtype=np.float32)

        # FILLING LOOP
        actual_count = 0
        
        # ⚡ PERFORMANCE OPTIMIZATION: Pre-compute movement analysis for ALL positions in parallel.
        # Includes both positive signal positions AND randomly sampled negative positions.
        # This replaces the sequential per-sample computation that was 5-10x slower.
        movement_analysis_cache = {}
        if getattr(self.config, 'prepare_advanced_ml_targets', False):
            logger.info(f"🔄 [MLPrep] Pre-computing movement analysis for {len(all_positions)} positions ({len(pos_positions)} positive + {len(all_positions)-len(pos_positions)} negative) in parallel...")
            import time
            start_time = time.time()
            movement_analysis_cache = self._compute_movement_analysis_batch(df, all_positions)  # FIX: Include negative positions
            elapsed = time.time() - start_time
            logger.info(f" [MLPrep] Movement analysis pre-computed in {elapsed:.2f}s (5-10x faster than sequential)")
        
        for idx, pos in enumerate(all_positions):
            try:
                # Include the signal bar (pos) in the lookback window.
                # Previously used pos - seq_len : pos (gap at pos).
                # New logic uses pos - seq_len + 1 : pos + 1 (ends ON the signal bar).
                seq = df[self.feature_cols].iloc[pos - seq_len + 1 : pos + 1].values
                if seq.shape[0] != seq_len: continue

                # FEATURE VALIDATION: Skip samples with NaNs or Infs in input features.
                # Identifying specific columns helps trace the source of calculation errors.
                if not np.isfinite(seq).all():
                    mask = ~np.isfinite(seq)
                    bad_col_idx = np.unique(np.where(mask)[1])
                    bad_cols = [self.feature_cols[i] for i in bad_col_idx]
                    logger.warning(
                        f"⚠️ [MLPrep] skipping sample at pos={pos} ({split_name}) "
                        f"due to non-finite FEATURES in columns: {bad_cols}"
                    )
                    continue
                
                if self.config.mask_future_signals and signal_feature_indices:
                    buffer = min(self.config.signal_leakage_buffer, seq_len)
                    if buffer > 0:
                        seq[-buffer:, signal_feature_indices] = 0.0
                
                # PER-TARGET MASKING (replaces the old all-or-nothing shared gate).
                # OLD: any NaN in any target → break + continue → entire sample dropped.
                # NEW: NaN in a target → mask that target (0.0) but keep the sample alive
                #      for all other heads that DID have valid targets at this bar.
                # Only drop the sample if (a) future_sequence is invalid (shapes must match),
                # or (b) literally zero targets were valid (useless sample for every head).
                current_targets = {}
                current_masks = {}
                valid_target_count = 0
                essential_target_failed = False  # set True only if future_sequence fails
                
                # Calculate advanced movement analysis targets per-sequence
                movement_analysis = None
                movement_target_keys = {
                    "adv_target_max_favorable_pct",
                    "adv_target_max_adverse_pct",
                    "adv_target_final_move_pct",
                    "adv_target_risk_reward_ratio",
                    "adv_target_avg_volatility",
                    "adv_target_volatility_surge",
                    "adv_target_time_to_max_favorable",
                    "adv_target_time_to_max_adverse",
                    "adv_target_signal_strength",
                }
                
                if getattr(self.config, 'prepare_advanced_ml_targets', False):
                    # ⚡ LOOKUP: Use pre-computed movement analysis from cache (O(1) dict lookup).
                    # Previously: computed sequentially inside loop (5-10x slower).
                    # Now: all positive positions pre-computed in parallel before loop.
                    movement_analysis = movement_analysis_cache.get(pos, None)
                
                for t_col in target_keys:
                    t_val = None
                    is_valid = False
                    
                    if t_col in movement_target_keys:
                        if movement_analysis is not None:
                            metric_name = t_col.replace('adv_target_', '')
                            if metric_name in movement_analysis:
                                t_val = movement_analysis[metric_name]
                                if isinstance(t_val, (list, np.ndarray)):
                                    t_val = float(t_val[0]) if len(t_val) > 0 else np.nan
                                else:
                                    t_val = float(t_val)
                                if np.isfinite(t_val):
                                    is_valid = True
                    else:
                        t_val = self._get_target_value_from_df(df, t_col, pos)
                        if t_val is not None:
                            if t_col == 'future_sequence':
                                # future_sequence is the primary regression scaffold — must match
                                # pred_len exactly or the batch tensor shapes break at training time.
                                # Treated as an essential target: failure drops the whole sample.
                                if isinstance(t_val, np.ndarray) and len(t_val) == self.config.prediction_length and np.isfinite(t_val).all():
                                    is_valid = True
                                else:
                                    essential_target_failed = True
                            elif self.config.prediction_length > 1 and isinstance(t_val, np.ndarray):
                                # Bug B fix (multi-step): strict equality check on length.
                                if len(t_val) == self.config.prediction_length and np.isfinite(t_val).all():
                                    is_valid = True
                            else:
                                if np.isfinite(t_val).all():
                                    is_valid = True
                    
                    target_slot_shape = targets_final[t_col][actual_count].shape
                    if is_valid and t_val is not None:
                        current_targets[t_col] = t_val
                        if target_slot_shape != ():
                            current_masks[t_col] = np.ones(target_slot_shape, dtype=np.float32)
                        else:
                            current_masks[t_col] = 1.0
                        valid_target_count += 1
                    else:
                        if target_slot_shape != ():
                            current_targets[t_col] = np.zeros(target_slot_shape, dtype=np.float32)
                            current_masks[t_col] = np.zeros(target_slot_shape, dtype=np.float32)
                        else:
                            current_targets[t_col] = 0.0
                            current_masks[t_col] = 0.0

                # Drop only if future_sequence is broken (tensor shape constraint)
                # OR if truly no target was valid at this bar (no head would benefit).
                if essential_target_failed or valid_target_count == 0:
                    continue

                sequences_final[actual_count] = seq
                if idx < len(pos_positions):
                    labels_final[actual_count] = self.signal_mapping.get(self._detect_signal_type_at_idx(df, df.index[pos]), 4)
                else:
                    labels_final[actual_count] = 4
                
                for t_col in target_keys:
                    targets_final[t_col][actual_count] = current_targets[t_col]
                    target_masks_final[t_col][actual_count] = current_masks[t_col]
                
                metadata = self._create_sequence_metadata_from_df(df, pos, self.feature_cols)
                sequence_metadata.append(metadata)
                actual_count += 1
                
                if actual_count % 100 == 0:
                    progress_pct = start_pct + (actual_count / total_samples) * (end_pct - start_pct) * 0.8
                    if self.reporter:
                        await self.reporter.report_async(progress=progress_pct, message=f"Generated {actual_count}/{total_samples} {split_name} sequences...")

            except Exception as e:
                logger.warning(f"Error generating sequence at position {pos}: {e}")
                continue

        if actual_count < total_samples:
            sequences_final = sequences_final[:actual_count]
            labels_final = labels_final[:actual_count]
            for k in targets_final:
                targets_final[k] = targets_final[k][:actual_count]
                target_masks_final[k] = target_masks_final[k][:actual_count]

        # Pass 3: Disk Caching / Pointer Return
        if self.config.use_lazy_storage:
            import os
            cache_dir = f"app/data/temp_sequences/{self.task_id}/{split_name}"
            os.makedirs(cache_dir, exist_ok=True)
            
            chunk_path = f"{cache_dir}/sequences.npz"
            np.savez_compressed(
                chunk_path,
                sequences=sequences_final,
                labels=labels_final,
                step_configs=np.array([json.dumps(getattr(self, '_step_configs', {}))], dtype=object),
                **{f"target_{k}": v for k, v in targets_final.items()},
                **{f"target_mask_{k}": v for k, v in target_masks_final.items()}
            )
            
            return {
                "data_type": "lazy_npz",
                "file_path": chunk_path,
                "total_sequences": actual_count,
                "sequence_metadata": sequence_metadata,
                "indices": list(range(actual_count))
            }

        return {
            "sequences": sequences_final,
            "labels": labels_final,
            "targets": targets_final,
            "target_masks": target_masks_final,
            "sequence_metadata": sequence_metadata,
            "indices": list(range(actual_count))
        }

    def _detect_signal_type_at_idx(self, df: pd.DataFrame, idx_val: Any) -> str:
        """Detect which signal type occurred at a specific index value."""
        signal_prefixes = ["Signal_", "snr_", "signal_"]
        signal_suffixes = ["_signal", "_Signal"]
        
        row = df.loc[idx_val]
        for col in df.columns:
            if any(col.startswith(p) for p in signal_prefixes):
                if row[col] > 0.5:
                    s_type = col
                    for p in signal_prefixes:
                        if s_type.startswith(p): s_type = s_type[len(p):]; break
                    for s in signal_suffixes:
                        if s_type.endswith(s): s_type = s_type[:-len(s)]; break
                    
                    if s_type in self.SIGNAL_MAPPING:
                        return s_type
        return "none"


    

    def analyze_post_interaction_movement(
        self,
        df,
        signal_index,
        level_price,
        signal_type,
        lookforward_period=50,
        lookback_period=200,
        reporter=None, base_progress=0,
    ):
        """
        Comprehensive analysis of price movement after level interaction with enhanced volume analysis.
        Delegates to the field-proven standalone function in signal_generator.py.

        Args:
            df: Price dataframe
            signal_index: Index where signal occurred
            level_price: The key level price
            signal_type: Type of signal (bounce_support, breakout_resistance, etc.)
            lookforward_period: Number of candles to analyze ahead

        Returns:
            dict: Comprehensive movement analysis metrics
        """
        
        return _analyze_post_interaction_movement(
            df=df,
            signal_index=signal_index,
            level_price=level_price,
            signal_type=signal_type,
            lookforward_period=lookforward_period,
            lookback_period=lookback_period,
            reporter=reporter,
            base_progress=base_progress
        )
       

    def _get_target_value_from_df(self, df: pd.DataFrame, t_col: str, i: int) -> Union[float, np.ndarray]:
        """
        Return the target value for the sequence anchored at position i.

        DISPATCH TABLE (IN ORDER):
        ──────────────────────────────────────────────────────────────────
        t_col                              where it lives       return shape
        ──────────────────────────────────────────────────────────────────
        "future_sequence"                  virtual key          (pred_len,) array
        adv_target_*_next                  df[t_col].iloc[i]    scalar float
        adv_target_*_t{N}                  df[t_col].iloc[i]    scalar float (timestep targets)
        adv_target_* (scalar metrics)      df[t_col].iloc[i]    scalar float
        Next_* / *_Return / *_Direction    df[t_col].iloc[i]    scalar float
                                           (already shift(-N) from _enrich_with_targets)
        any other col, pred_len > 1        df[t_col].iloc[i+1:…] (pred_len,) array
        any other col, pred_len == 1       df[t_col].iloc[i+1]   scalar float
        ──────────────────────────────────────────────────────────────────

        KEY ARCHITECTURE UPDATE:
        • All adv_target_* columns are now scalar floats for uniform target shapes
        • OHLCV targets changed from sequences to individual timestep scalars (t1, t2, ..., t7)
        • This enables cleaner multi-output model design with consistent head architectures
        """
        pred_len = self.config.prediction_length
        adv_on = getattr(self.config, "prepare_advanced_ml_targets", False)
        drop_zeros = getattr(self.config, "drop_zeros", True)

        # ── 1. Virtual future_sequence ────────────────────────────────────────
        if t_col == "future_sequence":
            primary = self.config.target_columns[0] if self.config.target_columns else "Close"
            target_col = next(
                (c for c in [primary, primary.lower(), primary.upper()] if c in df.columns),
                None
            )
            if not target_col:
                target_col = next(
                    (c for c in ["Close", "close", "CLOSE"] if c in df.columns), None
                )
            if not target_col:
                return None

            fut = df[target_col].iloc[i + 1 : i + 1 + pred_len].values.astype(np.float32)
            if len(fut) != pred_len:
                return None
            if drop_zeros and np.any(fut == 0):
                return None
            return fut if np.isfinite(fut).all() else None

        # ── 2. adv_target_*_next and adv_target_*_t{N} (next-bar scalar targets) ──────
        #
        # UPDATED: All OHLCV targets are now individual scalars (t1, t2, ..., t7)
        # Pattern: adv_target_Close_t1, adv_target_Close_t2, etc.
        # All _next targets are also pre-shifted scalars, read from iloc[i].
        #
        if adv_on and t_col.startswith("adv_target_") and t_col in df.columns:
            # Check if it's a timestep target (ends with _t followed by digit)
            import re
            is_timestep = re.search(r'_t\d+$', t_col) is not None
            is_next = t_col.endswith("_next")
            
            if is_timestep or is_next:
                val = float(df[t_col].iloc[i])
                return val if np.isfinite(val) else None

        # ── 3. adv_target_final_move and other movement metrics (pre-computed) ───
        #
        # These metrics are pre-computed in _calculate_advanced_ml_targets() by calling
        # _analyze_post_interaction_movement() on the full dataset. Here we just read
        # the scalar value that was already calculated. This avoids repeated analysis calls.
        #
        if adv_on and t_col in [
            "adv_target_final_move",
            "adv_target_max_favorable_move", 
            "adv_target_max_adverse_move",
            "adv_target_final_move_pct",
            "adv_target_max_favorable_pct",
            "adv_target_max_adverse_pct",
            "adv_target_signal_strength",
            "adv_target_risk_reward",
        ] and t_col in df.columns:
            raw = df[t_col].iloc[i]
            val = float(raw)
            return val if np.isfinite(val) else None

        # ── 4. adv_target_* (scalar movement-metric columns) ────────────────
        #
        # _calculate_advanced_ml_targets() pre-computes all movement metrics
        # and stores them as plain float32 scalars. Read directly from iloc[i].
        # NO cache, NO fuzzy string matching — direct dispatch only.
        #

        
        if adv_on and t_col.startswith("adv_target_") and t_col in df.columns:
            raw = df[t_col].iloc[i]
            # Guard: should never be list post Fix-A, but handle gracefully
            if isinstance(raw, (list, np.ndarray)):
                arr = np.asarray(raw, dtype=np.float32)
                if arr.ndim == 1 and len(arr) == pred_len:
                    return arr if np.isfinite(arr).all() else None
                return None
            val = float(raw)
            return val if np.isfinite(val) else None

        # ── 4b. adv_target_*_seq (OHLCV channel sequence targets) ─────────────
        #
        # adv_target_Open_seq, adv_target_High_seq, etc. are VIRTUAL: they do
        # not exist as columns in df. Instead they are built on-the-fly as a
        # (pred_len,) array from the corresponding raw OHLCV column at
        # positions i+1 … i+pred_len, mirroring the future_sequence logic.
        #
        if adv_on and t_col.startswith("adv_target_") and t_col.endswith("_seq"):
            # e.g. "adv_target_Open_seq" → channel = "Open"
            channel = t_col[len("adv_target_"):-len("_seq")].capitalize()
            # Try canonical forms
            src_col = next(
                (c for c in [channel, channel.lower(), channel.upper()] if c in df.columns),
                None
            )
            if src_col is None:
                return None
            arr = df[src_col].iloc[i + 1 : i + 1 + pred_len].values.astype(np.float32)
            if len(arr) != pred_len:
                return None
            if drop_zeros and np.any(arr == 0):
                return None
            return arr if np.isfinite(arr).all() else None


            return None

        # ── 6. Pre-shifted return / direction columns (Fix B) ────────────────
        #
        # _enrich_with_targets writes these with shift(-N), so the future value
        # already lives at iloc[i]. Do NOT add +1 offset here.
        #
        is_pre_shifted = (
            t_col.startswith("Next_")
            or t_col.endswith("_Return")
            or t_col.endswith("_Direction")
        )
        if is_pre_shifted:
            val = float(df[t_col].iloc[i])
            if not np.isfinite(val):
                return None
            is_label = "Direction" in t_col or "Label" in t_col
            if drop_zeros and not is_label and val == 0.0:
                return None
            return val

        # ── 6b. Primary target columns (config.target_columns) as scalars ──────
        #
        # config.target_columns (e.g., 'Close') are prediction targets, not sequences.
        # Always treat them as scalars regardless of pred_len.
        #
        if t_col in self.config.target_columns:
            val = float(df[t_col].iloc[i])
            if not np.isfinite(val):
                return None
            if drop_zeros and val == 0.0:
                return None
            return val

        # ── 6c. Signal classification columns — current-bar binary labels ─────
        #
        # Signal_bounce_support / Signal_bounce_resistance /
        # Signal_breakout_support / Signal_breakout_resistance are CURRENT-BAR
        # labels: they record what price did AT bar i, not in the future.
        # Read iloc[i] directly.  NEVER apply drop_zeros — 0 means "no signal"
        # and is valid supervision data (the model must learn to output 0 too).
        #
        if t_col in self.CONFIRMED_SIGNAL_COLUMNS and t_col in df.columns:
            val = float(df[t_col].iloc[i])
            return val if np.isfinite(val) else None

        # ── 7. Standard multi-step raw target ────────────────────────────────
        if pred_len > 1:
            arr = df[t_col].iloc[i + 1 : i + 1 + pred_len].values.astype(np.float32)
            if len(arr) != pred_len:
                return None
            if drop_zeros and np.any(arr == 0):
                return None
            return arr if np.isfinite(arr).all() else None

        # ── 8. Standard single-step raw target ───────────────────────────────
        src_i = i + 1 if (i + 1 < len(df)) else i
        val = float(df[t_col].iloc[src_i])
        if not np.isfinite(val):
            return None
        is_label = "Direction" in t_col or "Label" in t_col or "Signal" in t_col
        if drop_zeros and not is_label and val == 0.0:
            return None
        return val

    def _create_sequence_metadata_from_df(self, df: pd.DataFrame, i: int, feature_cols: list) -> Dict[str, Any]:
        """
        Create metadata for a sequence from a specific dataframe slice to enable proper visualization.
        
        Args:
            df: Dataframe slice to extract from
            i: Current index (anchor point)
            feature_cols: List of feature column names
            
        Returns:
            Dictionary with OHLCV data, timestamps, and timeframe info
        """
        seq_len = self.config.sequence_length
        pred_len = self.config.prediction_length
        start_idx = i - seq_len + 1
        end_idx = i + 1
        
        ohlcv_cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
        
        # --- Integer chart indices (ALWAYS integer-based for sequence mode) ---
        sequence_chart_indices = list(range(seq_len))                              # [0..59]
        prediction_chart_indices = list(range(seq_len, seq_len + pred_len))        # [60..66]
        
        metadata = {
            'anchor_index': int(i),
            'sequence_length': int(seq_len),
            'prediction_length': int(pred_len),
            'prediction_start_index': seq_len,
            'prediction_indices': prediction_chart_indices,
            'sequence_indices': sequence_chart_indices,
        }

        # Real timestamps (DEPRECATED: Frontend uses integer indices for sequence mode. 
        # Keeping only timeframe_delta for potential future extrapolation)
        if 'time' in df.columns:
            ts = df['time'].iloc[start_idx:end_idx].values
            # Calculate timeframe delta (for prediction extrapolation)
            if len(ts) > 1:
                try:
                    deltas = np.diff(ts, dtype='timedelta64[s]').astype(float)
                    metadata['timeframe_delta'] = float(np.median(deltas))
                except (TypeError, ValueError):
                    metadata['timeframe_delta'] = 3600.0  # Default 1 hour
            else:
                metadata['timeframe_delta'] = 3600.0
        else:
            metadata['timeframe_delta'] = 1.0
        
        # Store OHLCV data if available (SCALED data from df)
        if ohlcv_cols:
            ohlcv_data = df[ohlcv_cols].iloc[start_idx:end_idx].values
            metadata['ohlcv'] = ohlcv_data.tolist()
            metadata['ohlcv_columns'] = ohlcv_cols
            
            # Store last close for prediction connection
            close_col = next((c for c in ['Close', 'close'] if c in df.columns), None)
            if close_col:
                metadata['last_close'] = float(df[close_col].iloc[i])

            # Future OHLCV for the prediction window (SCALED ground truth from same dataset)
            future_end = i + 1 + pred_len
            if future_end <= len(df):
                future_data = df[ohlcv_cols].iloc[i + 1 : future_end].values
                metadata['future_ohlcv'] = future_data.tolist()
            else:
                metadata['future_ohlcv'] = []
            
        # Add regression targets if present
        target_keys = list(self.config.target_columns)
        if self.config.include_sequence_prediction:
            target_keys.append('future_sequence')

        # Store metadata OHLCV visualization data only
        metadata['ohlcv_info'] = {}
            
        # Detect which target columns have OHLCV structure
        target_set_lower = {c.lower() for c in self.config.target_columns}
        has_full_ohlc_targets = all(c in target_set_lower for c in ['open', 'high', 'low', 'close'])
        
        target_ohlcv_structure = {}
        for target_col in target_keys:
            target_info = {}
            
            if target_col == 'future_sequence':
                target_info['is_ohlcv'] = has_full_ohlc_targets
                target_info['is_array'] = not has_full_ohlc_targets
                target_info['length'] = pred_len
                target_info['indices'] = prediction_chart_indices
            else:
                is_array = pred_len > 1 and not (target_col.endswith('_Return') or target_col.endswith('_Direction'))
                target_info['is_ohlcv'] = False 
                target_info['is_array'] = is_array
                target_info['length'] = pred_len if is_array else 1
                target_info['indices'] = prediction_chart_indices if is_array else [seq_len]

            target_ohlcv_structure[target_col] = target_info
        
        metadata['target_structure'] = target_ohlcv_structure
        return metadata


    async def _analyze_class_imbalance(self):
        """Analyze class distribution and generate recommendations."""
        stage_info = PREPARATION_STAGES['analyzing']
        
        if self.reporter:
            await self.reporter.report_async(
                progress=stage_info['start'],
                message=stage_info['label'],
                details="Checking class distribution..."
            )
        
        if self.labels is None or len(self.labels) == 0:
            logger.warning("No labels found for imbalance analysis.")
            return {}

        unique_labels, counts = np.unique(self.labels, return_counts=True)
        label_to_count = dict(zip(unique_labels, counts))
        
        # Map numeric labels back to signal names
        reverse_mapping = {v: k for k, v in self.SIGNAL_MAPPING.items()}
        signal_distribution = {
            reverse_mapping.get(label, f'class_{label}'): int(count)
            for label, count in label_to_count.items()
        }
        
        # Calculate imbalance ratio
        max_count = np.max(counts)
        min_count = np.min(counts[counts > 0]) if np.any(counts > 0) else 1
        imbalance_ratio = float(max_count / min_count)
        
        # Classify severity
        if imbalance_ratio < 1.5:
            severity = 'balanced'
        elif imbalance_ratio < 3.0:
            severity = 'moderate'
        elif imbalance_ratio < 5.0:
            severity = 'significant'
        else:
            severity = 'severe'
        
        # Generate recommendations
        recommendations = []
        if imbalance_ratio > 1.5:
            recommendations.append({
                'option': 'class_weights',
                'description': 'Adjust model class weights inversely to frequency',
                'strength': 'Lightweight, no data change',
                'tradeoff': 'May still bias toward majority'
            })
            recommendations.append({
                'option': 'negative_sampling_ratio',
                'description': 'Adjust negative sampling ratio in config',
                'strength': 'Direct control over "No Signal" samples',
                'tradeoff': 'May remove valuable negative samples'
            })
        
        if imbalance_ratio > 3.0:
            recommendations.append({
                'option': 'stratified_split',
                'description': 'Use stratified splitting to maintain ratios in each split',
                'strength': 'Preserves distribution across train/val/test',
                'tradeoff': 'May reduce temporal coherence'
            })
            recommendations.append({
                'option': 'focal_loss',
                'description': 'Use Focal Loss during model training',
                'strength': 'Mathematically designed for class imbalance',
                'tradeoff': 'Requires model architecture change'
            })
        
        imbalance_analysis = {
            'signal_distribution': signal_distribution,
            'class_imbalance_ratio': imbalance_ratio,
            'severity': severity,
            'recommendations': recommendations
        }
        
        self.metrics.class_imbalance_ratio = imbalance_ratio
        self.metrics.signal_distribution = signal_distribution
        
        if self.reporter:
            await self.reporter.report_async(
                progress=stage_info['end'],
                message=f"{stage_info['label']} ✓",
                details=f"Imbalance ratio: {imbalance_ratio:.1f}:1 ({severity})",
                imbalance_analysis=imbalance_analysis
            )
        
        return imbalance_analysis




    async def _calculate_metrics(self, splits: Dict[str, Dict[str, np.ndarray]]):
        """Calculate comprehensive dataset metrics."""
        stage_info = PREPARATION_STAGES['finalizing']
        if self.reporter:
            await self.reporter.report_async(
                progress=stage_info['start'],
                message=stage_info['label'],
                details="Computing distribution stats..."
            )

        # Update split sizes from actual splits
        self.metrics.train_size = len(splits["train"]["sequences"]) if "train" in splits else 0
        self.metrics.validation_size = len(splits["validation"]["sequences"]) if "validation" in splits else 0
        self.metrics.test_size = len(splits["test"]["sequences"]) if "test" in splits else 0

        # Calculate class imbalance (already done in _analyze_class_imbalance, but ensuring metrics are synced)
        if self.labels is not None and len(self.labels) > 0:
            unique_labels, counts = np.unique(self.labels, return_counts=True)
            if len(counts) > 1:
                max_count = np.max(counts)
                min_count = np.min(counts[counts > 0])
                self.metrics.class_imbalance_ratio = float(max_count / min_count)
            else:
                self.metrics.class_imbalance_ratio = 1.0

        # Calculate target statistics
        for target_col in self.config.target_columns:
            if target_col in self.target_data:
                target_values = self.target_data[target_col]

                if len(target_values) == 0:
                    logger.warning(f"Target column '{target_col}' has no values. Skipping metrics.")
                    continue

                self.metrics.target_shapes[target_col] = target_values.shape
                self.metrics.target_statistics[target_col] = {
                    "mean": float(np.mean(target_values)),
                    "std": float(np.std(target_values)),
                    "min": float(np.min(target_values)),
                    "max": float(np.max(target_values)),
                    "median": float(np.median(target_values)),
                }
            else:
                # CRITICAL FIX: Handle case where target column exists in data but not in target_data
                # This happens when include_regression=False but target columns are specified
                if target_col in self.data.columns:
                    target_values = self.data[target_col].dropna().values
                    if len(target_values) > 0:
                        logger.info(f"Using data column '{target_col}' for target statistics (not in target_data)")
                        self.metrics.target_shapes[target_col] = target_values.shape
                        self.metrics.target_statistics[target_col] = {
                            "mean": float(np.mean(target_values)),
                            "std": float(np.std(target_values)),
                            "min": float(np.min(target_values)),
                            "max": float(np.max(target_values)),
                            "median": float(np.median(target_values)),
                        }
                    else:
                        logger.warning(f"Target column '{target_col}' exists in data but has no valid values. Skipping metrics.")
                else:
                    logger.warning(f"Target column '{target_col}' not found in data or target_data. Skipping metrics.")

        total_sequences = self.metrics.train_size + self.metrics.validation_size + self.metrics.test_size
        self.metrics.total_sequences = total_sequences
        if self.reporter:
            await self.reporter.report_async(
                progress=stage_info['end'],
                message=f"{stage_info['label']} ✓",
                details=f"ML Prep Metrics finalized for {total_sequences} sequences."
            )
    
    def _extract_label_from_row(self, row: pd.Series) -> int:
        """Extract classification label from a DataFrame row."""
        # Check for confirmed signal columns first
        for signal_col in self.CONFIRMED_SIGNAL_COLUMNS:
            if signal_col in row.index and pd.notna(row[signal_col]) and row[signal_col] != 0:
                signal_type = signal_col.replace("Signal_", "")
                return self.SIGNAL_MAPPING.get(signal_type, 4)  # Default to "none"
        
        # Check for raw signal type columns
        for signal_type, label_value in self.SIGNAL_MAPPING.items():
            if signal_type in row.index and pd.notna(row[signal_type]) and row[signal_type] != 0:
                return label_value
        
        # Default: no signal
        return 4  # "none" class
