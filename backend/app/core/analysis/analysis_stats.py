import asyncio
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    shapiro,
    normaltest,
    anderson,
    jarque_bera,
    kstest,
    spearmanr,
    kendalltau,
    chi2_contingency,
    kruskal,
    levene,
    bartlett,
    f_oneway,
    ttest_ind,
    mannwhitneyu,
    pearsonr,
    linregress,
    boxcox,
)
from scipy.spatial.distance import pdist, squareform
from app.core.processing.tasks import TaskCancelledException
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    IsolationForest,
)
from sklearn.feature_selection import (
    mutual_info_regression,
    f_regression,
    SelectKBest,
    RFE,
    SelectFromModel,
)
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import mutual_info_score
from sklearn.cluster import DBSCAN
from sklearn.model_selection import cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
import json
import logging
import traceback


# Helper to safeguard slow-moving astronomical position features (e.g., Pluto) from being pruned
def is_astro_col(col_name: str) -> bool:
    astro_keywords = [
        "house", "aspect", "coordinate", "declination", "longitude", "latitude", "distance", "velocity",
        "phase", "retrograde", "planet", "pluto", "neptune", "uranus", "saturn", "jupiter", "mars",
        "venus", "mercury", "sun", "moon", "node", "part_of_fortune"
    ]
    name_lower = col_name.lower()
    return any(kw in name_lower for kw in astro_keywords)

from app.core.services.multiprocessing_utils import (
    parallel_rectangular_correlation,
    parallel_distance_correlation,
    distance_correlation as global_distance_correlation
)

warnings.filterwarnings("ignore")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Module-level worker functions for multiprocessing
# ============================================================================

def _nested_distance_corr_worker(
    feature_chunk: List[str],
    feature_to_idx: Dict[str, int],
    target_to_idx: Dict[str, int],
    target_cols: List[str],
    X_np: Union[np.ndarray, Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """
    Worker function for nested distance correlation computation.
    Computes distance correlation for a chunk of features against all targets.
    
    Args:
        feature_chunk: List of feature column names to process
        feature_to_idx: Mapping of feature names to column indices in X_np
        target_to_idx: Mapping of target names to column indices in X_np
        target_cols: List of target column names
        X_np: NumPy array OR dict with shared memory info: {"shm_name": str, "shape": tuple, "dtype": str}
    
    Returns:
        Dict mapping feature names to dicts of target names → distance correlation values
    """
    from multiprocessing import shared_memory
    
    # Attach to shared memory if provided
    shm = None
    if isinstance(X_np, dict) and "shm_name" in X_np:
        shm = shared_memory.SharedMemory(name=X_np["shm_name"])
        data = np.ndarray(X_np["shape"], dtype=X_np["dtype"], buffer=shm.buf)
    else:
        data = X_np
        
    try:
        chunk_result = {}
        
        for feature in feature_chunk:
            chunk_result[feature] = {}
            feat_idx = feature_to_idx[feature]
            feat_data = data[:, feat_idx]
            
            for target in target_cols:
                try:
                    tgt_idx = target_to_idx[target]
                    tgt_data = data[:, tgt_idx]
                
                    # Remove NaN values
                    valid_mask = ~(np.isnan(feat_data) | np.isnan(tgt_data))
                    if valid_mask.sum() < 2:
                        chunk_result[feature][target] = 0.0
                        continue
                    
                    feat_clean = feat_data[valid_mask]
                    tgt_clean = tgt_data[valid_mask]
                    
                    # Compute distance correlation
                    dcor = global_distance_correlation(feat_clean, tgt_clean)
                    chunk_result[feature][target] = float(dcor)
                except Exception as e:
                    logger.debug(f"Distance corr {feature}→{target}: {e}")
                    chunk_result[feature][target] = 0.0
        
        return chunk_result
    finally:
        if shm:
            shm.close()


class UltraComprehensiveFinancialAnalyzer:
    """
    ULTRA-COMPREHENSIVE financial data analyzer with exhaustive statistical analysis.

    Features include:
    - Comprehensive descriptive statistics
    - Multi-method correlation analysis (Pearson, Spearman, Kendall, Partial, Distance)
    - Advanced feature importance (8+ methods)
    - Multi-algorithm outlier detection (10+ methods)
    - Extensive normality testing (6+ tests)
    - Multicollinearity diagnostics (VIF, Condition Number, etc.)
    - Principal Component Analysis with advanced metrics
    - Mutual information and entropy analysis
    - Statistical hypothesis testing suite
    - Confidence intervals and effect sizes
    - Time series analysis (stationarity, autocorrelation)
    - Advanced distribution analysis
    - Heteroscedasticity testing
    - Data quality assessment
    - Feature interaction analysis
    - Non-linear relationship detection
    - Power transformations analysis
    - Robust statistical measures
    - Bayesian analysis components
    """

    def __init__(
        self,
        data: pd.DataFrame,
        selected_columns: Optional[List[str]] = None,
        target_column: Optional[Union[str, List[str]]] = None,
        datetime_column: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        task_id: str = None,
        progress_store: Any = None,
        progress_callback: callable = None,
    ):
        """
        Initialize the ultra-comprehensive analyzer.

        Args:
            data: DataFrame containing financial data
            selected_columns: Optional list of columns to run analysis on
            target_column: Name(s) of the target/dependent variable(s)
            datetime_column: Name of datetime column for time series analysis
            options: Dictionary of options for specific analysis modules
        """
        # Normalize target_column to a list
        if isinstance(target_column, str):
            self.target_columns = [target_column]
        elif isinstance(target_column, list):
            self.target_columns = target_column
        else:
            self.target_columns = []

        # Filter data to selected columns if provided
        if selected_columns:
            available_cols = [col for col in selected_columns if col in data.columns]
            # Ensure target and datetime columns are included if they exist
            for target in self.target_columns:
                if target not in available_cols and target in data.columns:
                    available_cols.append(target)
            if datetime_column and datetime_column not in available_cols and datetime_column in data.columns:
                available_cols.append(datetime_column)
            self.data = data[available_cols].copy()
        else:
            self.data = data.copy()

        # Backward compatibility for single target_column attribute
        self.target_column = self.target_columns[0] if self.target_columns else None
        self.datetime_column = datetime_column
        self.options = options or {}
        self.analysis_timestamp = datetime.utcnow().isoformat()
        self.task_id = task_id
        self.progress_store = progress_store
        self.progress_callback = progress_callback
        self._progress_sequence_id = 0  # ✅ FIX #5: Track message sequence for ordering
        
        # --- CRITICAL: Smart Data Cleaning ---
        # Ensure data is clean before ANY analysis to prevent crashes
        self.data = self._clean_data(self.data.copy())
        
        # Capture the event loop during initialization for thread-safe progress updates
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = None

        # Identify column types
        self.numeric_columns = self.data.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        self.categorical_columns = self.data.select_dtypes(
            exclude=[np.number]
        ).columns.tolist()

        if datetime_column and datetime_column in self.categorical_columns:
            self.categorical_columns.remove(datetime_column)

        # Prepare feature matrix and targets
        # CRITICAL FIX: Validate target columns are numeric
        valid_targets = [
            t for t in self.target_columns 
            if t in self.data.columns and self.data[t].dtype in [np.float64, np.int64, np.float32, np.int32]
        ]
        
        # Check for non-numeric targets that were specified
        if self.target_columns:
            non_numeric_targets = [
                t for t in self.target_columns 
                if t in self.data.columns and t not in valid_targets
            ]
            if non_numeric_targets:
                target_dtypes = {t: str(self.data[t].dtype) for t in non_numeric_targets}
                raise ValueError(
                    f"Target column(s) must be numeric for feature importance analysis. "
                    f"Non-numeric targets: {non_numeric_targets}. "
                    f"Data types: {target_dtypes}. "
                    f"Please select numeric target columns only."
                )
        
        self.has_target = len(valid_targets) > 0
        
        if self.has_target:
            # Features are numeric columns that are NOT targets and NOT datetime
            feature_candidates = [c for c in self.numeric_columns if c not in valid_targets]
            if datetime_column and datetime_column in feature_candidates:
                feature_candidates.remove(datetime_column)
                
            self.X = self.data[feature_candidates]
            self.targets_df = self.data[valid_targets]
            # y remains for backward compatibility (first target)
            self.y = self.targets_df[valid_targets[0]] if valid_targets else None
            self.feature_columns = self.X.columns.tolist()
            
            # --- PHASE 4: Pre-filter constant columns ---
            # Features with nearly zero variance cause NaNs in correlations and instability in models
            CONSTANT_THRESHOLD = 1e-8
            self.constant_feature_columns = [c for c in self.feature_columns if self.X[c].std() < CONSTANT_THRESHOLD and not is_astro_col(c)]
            if self.constant_feature_columns:
                logger.warning(f"Excluding constant feature columns from analysis: {self.constant_feature_columns}")
                self.feature_columns = [c for c in self.feature_columns if c not in self.constant_feature_columns]
                self.X = self.X[self.feature_columns]
        else:
            numeric_features = self.numeric_columns.copy()
            if datetime_column and datetime_column in numeric_features:
                numeric_features.remove(datetime_column)

            self.X = self.data[numeric_features]
            self.targets_df = pd.DataFrame()
            self.y = None
            self.feature_columns = self.X.columns.tolist()
            
            # Pre-filter constant columns here too
            CONSTANT_THRESHOLD = 1e-8
            self.constant_feature_columns = [c for c in self.feature_columns if self.X[c].std() < CONSTANT_THRESHOLD and not is_astro_col(c)]
            if self.constant_feature_columns:
                logger.warning(f"Excluding constant columns from analysis: {self.constant_feature_columns}")
                self.feature_columns = [c for c in self.feature_columns if c not in self.constant_feature_columns]
                self.X = self.X[self.feature_columns]

        if datetime_column and datetime_column in self.data.columns:
            self.datetime_data = self.data[datetime_column]
        else:
            self.datetime_data = None

        # Data scaling and transformations
        self.scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        self.power_transformer = PowerTransformer(method="yeo-johnson")
        
        # ===== PARALLELIZATION CONFIGURATION (PHASE 3) =====
        # Determine parallelization strategy based on dataset size and configuration
        self._should_parallelize_global = len(self.data) >= 1000
        logger.info(f"Global parallelization {'enabled' if self._should_parallelize_global else 'disabled'} "
                   f"(dataset: {len(self.data)} rows, {len(self.feature_columns)} features)")
        
        # Load user parallelization configuration with defaults
        self.parallelization_config = options.get('parallelization', {}) if options else {}
        
        # Define method-specific parallelization thresholds (PHASE 3 TASK 2)
        # Based on computational complexity (O(n), O(n²), etc.) and data dependency
        default_parallelization_profiles = {
            'basic_statistics': {'rows': 10000, 'features': 0},  # Vectorized, minimal benefit
            'distribution_analysis': {'rows': 5000, 'features': 0},  # Histograms fast
            'correlation_analysis': {'rows': 500, 'features': 100},  # O(n²) in features, parallelize for 100+ features
            'partial_correlations': {'rows': 500, 'features': 50},  # High complexity, parallelize for 50+ features
            'feature_importance': {'rows': 500, 'features': 0},  # Tree models expensive
            'outlier_analysis': {'rows': 5000, 'features': 0},  # Vectorized sklearn
            'normality_tests': {'rows': 5000, 'features': 0},  # Statistical tests
            'multicollinearity': {'rows': 1000, 'features': 0},  # VIF computation
            'pca_analysis': {'rows': 2000, 'features': 0},  # Eigendecomposition
            'mutual_information': {'rows': 1000, 'features': 0},  # MI calculation
            'statistical_tests': {'rows': 2000, 'features': 0},  # Hypothesis tests
            'confidence_intervals': {'rows': 1000, 'features': 0},  # CI bootstrap
            'effect_sizes': {'rows': 1000, 'features': 0},  # Effect size metrics
            'variance_analysis': {'rows': 1000, 'features': 0},  # ANOVA
            'skewness_kurtosis_analysis': {'rows': 5000, 'features': 0},  # Moments
            'feature_interactions': {'rows': 500, 'features': 0},  # Interaction detection
            'heteroscedasticity_analysis': {'rows': 2000, 'features': 0},  # Breusch-Pagan
            'data_quality_assessment': {'rows': 10000, 'features': 0},  # Data metrics
            'non_linear_relationships': {'rows': 1000, 'features': 0},  # Non-linear tests
            'power_transformations': {'rows': 1000, 'features': 0},  # Transform analysis
            'robust_statistics': {'rows': 2000, 'features': 0},  # Robust measures
        }
        
        # User can override defaults (PHASE 3 TASK 3)
        user_profiles = self.parallelization_config.get('method_thresholds', {})
        self._parallelization_profiles = {**default_parallelization_profiles, **user_profiles}
        
        # Global parallelization enable/disable flag
        self.parallelization_enabled = self.parallelization_config.get('enabled', True)
        self.parallelization_max_workers = self.parallelization_config.get('max_workers', None)
        
        if not self.parallelization_enabled:
            logger.info("Parallelization DISABLED (via user configuration)")

        try:
            if not self.X.empty:
                self.X_scaled = pd.DataFrame(
                    self.scaler.fit_transform(self.X),
                    columns=self.X.columns,
                    index=self.X.index,
                )
                self.X_robust_scaled = pd.DataFrame(
                    self.robust_scaler.fit_transform(self.X),
                    columns=self.X.columns,
                    index=self.X.index,
                )
                self.X_power_transformed = pd.DataFrame(
                    self.power_transformer.fit_transform(self.X),
                    columns=self.X.columns,
                    index=self.X.index,
                )
            else:
                self.X_scaled = pd.DataFrame()
                self.X_robust_scaled = pd.DataFrame()
                self.X_power_transformed = pd.DataFrame()
        except Exception as e:
            logger.error(f"Feature scaling failed: {e}")
            raise ValueError(
                f"Cannot scale features for analysis. "
                f"Ensure all features are numeric and contain valid values. "
                f"Error: {e}"
            ) from e
    
    def _should_parallelize_method(self, method_name: str) -> bool:
        """Determine if specific method should run in parallel (PHASE 3 TASK 2).
        
        Decision based on:
        1. Global parallelization enable/disable flag
        2. Dataset size (rows) vs method threshold
        3. Feature count vs method threshold
        
        Args:
            method_name: Name of the analysis method (e.g., 'correlation_analysis')
            
        Returns:
            True if method should parallelize, False for sequential execution
        """
        # Check global enable flag first
        if not self.parallelization_enabled:
            return False
        
        # Get method-specific profile (with fallback to default)
        profile = self._parallelization_profiles.get(
            method_name, 
            {'rows': 1000, 'features': 0}  # Conservative default
        )
        
        row_threshold = profile.get('rows', 1000)
        feature_threshold = profile.get('features', 0)
        
        # Check all criteria
        meets_row_requirement = len(self.data) > row_threshold
        meets_feature_requirement = (
            feature_threshold == 0 or 
            len(self.feature_columns) > feature_threshold
        )
        
        should_parallelize = meets_row_requirement and meets_feature_requirement
        
        # Debug logging
        if should_parallelize:
            logger.debug(f"Parallelizing {method_name}: "
                        f"{len(self.data)} rows (>{row_threshold}), "
                        f"{len(self.feature_columns)} features (>{feature_threshold})")
        
        return should_parallelize

    def _send_progress(self, progress: int, message: str, message2: str = None):
        # 1. Check for cancellation if store is available
        if self.task_id and self.progress_store and hasattr(self.progress_store, "check_cancellation"):
            self.progress_store.check_cancellation(self.task_id)

        if self.progress_callback and self.task_id:
            try:
                # Use the loop captured during initialization if possible, 
                # otherwise try to get the current one
                loop = self.loop
                if not loop:
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        logger.warning("No event loop available for progress update.")
                        return

                # ✅ FIX #5: Add sequence ID to ensure frontend processes messages in order
                self._progress_sequence_id += 1
                progress_payload = {
                    "progress": progress,
                    "message": message,
                    "status": "processing",
                    "sequence_id": self._progress_sequence_id,  # Ordering guarantee
                }
                if message2:
                    progress_payload["message2"] = message2
                
                # ✅ FIX #2: Use timeout to prevent blocking on queue overflow
                # run_coroutine_threadsafe returns a concurrent.futures.Future
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        self.progress_callback(
                            self.task_id,
                            progress_payload,
                        ),
                        loop,
                    )
                    # Wait max 1 second for the coroutine to be enqueued/processed
                    # This prevents indefinite blocking if event loop stalls
                    future.result(timeout=1.0)
                except TimeoutError:
                    logger.warning(
                        f"Progress update timeout for task {self.task_id} - event loop may be overloaded. "
                        f"Progress: {progress}%, Message: {message}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to enqueue progress update: {e}")
                    
            except Exception as e:
                logger.warning(f"Failed to send progress update: {e}")

    async def analyze_all(self) -> Dict[str, Any]:
        """Perform ALL available analyses with intelligent parallelization.
        
        PARALLELIZATION STRATEGY:
        - PHASE 3 TASK 1 (Global): Skip parallelization overhead for small datasets (< 1000 rows)
        - PHASE 3 TASK 2 (Method-Specific): Use intelligent per-method thresholds
        - PHASE 3 TASK 3 (User-Configurable): Allow runtime override via options
        
        Returns:
            Dict containing exhaustive analysis results in API-ready format
        """
        print("Starting ultra-comprehensive financial analysis...")
        logger.info(f"Parallelization config: enabled={self.parallelization_enabled}, "
                   f"global_threshold={'enabled' if self._should_parallelize_global else 'disabled'}")

        try:
            results = {}
            methods = [
                ("basic_statistics", self.calculate_comprehensive_statistics),
                ("distribution_analysis", self.analyze_distributions),
                ("correlation_analysis", self.analyze_correlations_comprehensive),
                ("partial_correlations", self.calculate_partial_correlations),
                ("feature_importance", self.calculate_feature_importance_comprehensive),
                ("outlier_analysis", self.detect_outliers_comprehensive),
                ("normality_tests", self.test_normality_comprehensive),
                ("multicollinearity", self.detect_multicollinearity_advanced),
                ("pca_analysis", self.perform_pca_comprehensive),
                ("mutual_information", self.calculate_mutual_information_advanced),
                ("statistical_tests", self.perform_statistical_tests_comprehensive),
                (
                    "confidence_intervals",
                    self.calculate_confidence_intervals_comprehensive,
                ),
                ("effect_sizes", self.calculate_effect_sizes_advanced),
                ("variance_analysis", self.analyze_variance_comprehensive),
                (
                    "skewness_kurtosis_analysis",
                    self.analyze_skewness_kurtosis_comprehensive,
                ),
                ("feature_interactions", self.analyze_feature_interactions_advanced),
                (
                    "heteroscedasticity_analysis",
                    self.test_heteroscedasticity_comprehensive,
                ),
                ("data_quality_assessment", self.assess_data_quality_comprehensive),
                ("non_linear_relationships", self.analyze_non_linear_relationships),
                ("power_transformations", self.analyze_power_transformations),
                ("robust_statistics", self.calculate_robust_statistics),
            ]
            total_steps = len(methods)

            base_progress = 20
            methods_progress_span = 75  # This part of analysis will go from 20% to 95%

            loop = asyncio.get_event_loop()
            for i, (key, method) in enumerate(methods):
                progress = base_progress + int(
                    ((i + 1) / total_steps) * methods_progress_span
                )
                message = (
                    f"Step {i+1}/{total_steps}: Analyzing {key.replace('_', ' ')}..."
                )
                self._send_progress(progress, message)
                try:
                    # ===== PHASE 3: INTELLIGENT PARALLELIZATION DECISION =====
                    # Determine if this method should parallelize
                    should_parallelize = self._should_parallelize_method(key)
                    
                    self._send_progress(progress, message, f"Executing: {key.replace('_', ' ')} "
                                                          f"({'parallel' if should_parallelize else 'sequential'})")
                    
                    # Execute with or without explicit thread pool
                    if should_parallelize:
                        # Use explicit thread pool for methods that benefit from parallelization
                        from concurrent.futures import ThreadPoolExecutor
                        executor = ThreadPoolExecutor(max_workers=self.parallelization_max_workers or 2)
                        try:
                            results[key] = await loop.run_in_executor(executor, method)
                        finally:
                            executor.shutdown(wait=True)
                    else:
                        # Sequential execution for small datasets or simple methods
                        results[key] = await loop.run_in_executor(None, method)
                        
                except Exception as e:
                    results[key] = {
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }

            await self._send_progress(96, "Analyzing target relationships...")
            if self.has_target:
                results.update(
                    {
                        "target_relationship_analysis": self.analyze_target_relationship_comprehensive(),
                        "predictive_power_assessment": self.assess_predictive_power(),
                    }
                )

            await self._send_progress(98, "Analyzing time series properties...")
            if self.datetime_data is not None:
                results.update(
                    {
                        "time_series_analysis": self.perform_time_series_analysis(),
                    }
                )

            await self._send_progress(99, "Generating summary...")
            results["executive_summary"] = self.generate_executive_summary(results)

            print("Analysis completed successfully!")
            return self._to_json_serializable(results)

        except Exception as e:
            print(f"Error during comprehensive analysis: {e}")
            traceback.print_exc()
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "metadata": {
                    "analysis_timestamp": self.analysis_timestamp,
                    "error": True,
                },
            }

    def get_visualization_data(self, analysis_type: str) -> Dict[str, Any]:
        """
        Return data formatted specifically for frontend visualization
        """
        if analysis_type == "descriptive":
            return self._format_descriptive_for_viz()
        elif analysis_type == "correlation":
            return self._format_correlation_for_viz()
        else:
            return {"error": f"Visualization format for {analysis_type} not available."}

    def _format_descriptive_for_viz(self) -> Dict[str, Any]:
        """Format descriptive stats for charts"""
        stats = self.calculate_comprehensive_statistics()
        return {
            "bar_chart_data": [
                {
                    "name": col,
                    "mean": s.get("mean"),
                    "median": s.get("median"),
                    "std": s.get("std"),
                    "missing_pct": s.get("missing_pct"),
                }
                for col, s in stats.items()
                if "error" not in s
            ],
            "summary_stats": {
                col: {"mean": s.get("mean"), "std": s.get("std"), "cv": s.get("cv")}
                for col, s in stats.items()
                if "error" not in s
            },
        }

    def _format_correlation_for_viz(self) -> Dict[str, Any]:
        """Format correlation results for frontend visualizations."""
        return self.analyze_correlations_comprehensive()

    def calculate_comprehensive_statistics(self) -> Dict[str, Any]:
        """Calculate exhaustive basic statistics for all features."""
        stats_dict = {}

        total_cols = len(self.numeric_columns)
        for i, col in enumerate(self.numeric_columns):
            try:
                progress = 20 + int((i/total_cols)*5)
                self._send_progress(progress, "Analyzing basic stats...", f"Processing: {col} ({i+1}/{total_cols})")
                data_col = self.data[col].dropna()
                if len(data_col) == 0:
                    continue

                # Descriptive statistics
                # PHASE 4: Cache redundant calculations for efficiency
                missing_count = int(self.data[col].isna().sum())
                
                stats_dict[col] = {
                    "count": int(len(data_col)),
                    "missing": missing_count,
                    "missing_pct": float(
                        missing_count / len(self.data) * 100
                    ),
                    "mean": float(data_col.mean()),
                    "median": float(data_col.median()),
                    "mode": (
                        float(data_col.mode()[0]) if len(data_col.mode()) > 0 else None
                    ),
                    "std": float(data_col.std()),
                    "variance": float(data_col.var()),
                    "min": float(data_col.min()),
                    "max": float(data_col.max()),
                    "range": float(data_col.max() - data_col.min()),
                    # PHASE 4: Cache quantile calculations (q1, q3 computed once)
                    "q1": float(q1_cached := data_col.quantile(0.25)),
                    "q3": float(q3_cached := data_col.quantile(0.75)),
                    "iqr": float(q3_cached - q1_cached),
                    "cv": float(
                        np.inf if data_col.mean() == 0 and data_col.std() > 0 
                        else (data_col.std() / data_col.mean() if data_col.mean() != 0 else 0)
                    ),
                    "sem": float(data_col.sem()),
                    "mad": float(np.median(np.abs(data_col - data_col.median()))),
                    "geometric_mean": (
                        float(stats.gmean(data_col[data_col > 0]))
                        if (data_col > 0).all()
                        else None
                    ),
                    "harmonic_mean": (
                        float(stats.hmean(data_col[data_col > 0]))
                        if (data_col > 0).all()
                        else None
                    ),
                    # Robust statistics
                    "trimmed_mean_5": float(stats.trim_mean(data_col, 0.05)),
                    "trimmed_mean_10": float(stats.trim_mean(data_col, 0.10)),
                    "winsorized_mean_5": float(self._winsorized_mean(data_col, 0.05)),
                    "winsorized_mean_10": float(self._winsorized_mean(data_col, 0.10)),
                    # Shape statistics
                    "skewness": float(stats.skew(data_col)),
                    "kurtosis": float(stats.kurtosis(data_col)),
                    "excess_kurtosis": float(stats.kurtosis(data_col, fisher=True)),
                    # Extreme values
                    "outlier_lower_bound": float(
                        data_col.quantile(0.25)
                        - 1.5 * (data_col.quantile(0.75) - data_col.quantile(0.25))
                    ),
                    "outlier_upper_bound": float(
                        data_col.quantile(0.75)
                        + 1.5 * (data_col.quantile(0.75) - data_col.quantile(0.25))
                    ),
                    # Information theory
                    "entropy": float(
                        stats.entropy((np.histogram(data_col, bins="auto")[0] + 1e-10) / 
                                    (np.histogram(data_col, bins="auto")[0] + 1e-10).sum())
                    ),  # Normalize histogram to probabilities
                }

                # Percentiles
                percentiles = {}
                for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                    percentiles[f"p{p}"] = float(data_col.quantile(p / 100))
                stats_dict[col]["percentiles"] = percentiles

            except Exception as e:
                stats_dict[col] = {"error": f"Could not calculate statistics: {str(e)}"}

        return stats_dict

    def analyze_correlations_comprehensive(self) -> Dict[str, Any]:
        """
        Calculate exhaustive correlation analysis with multiple methods and filtering modes.
        
        Modes:
        - "all": Full square matrix (all numeric columns vs all numeric columns)
        - "ohlcv_vs_features": Rectangular matrix (OHLCV columns vs other features)
        - "targets_vs_features": Rectangular matrix (target columns vs feature columns)
        
        Methods: pearson, spearman, kendall, distance
        """
        all_results = {}
        corr_options = self.options.get("correlation", {})
        methods = corr_options.get(
            "methods", ["pearson", "spearman", "kendall", "distance"]
        )
        threshold = corr_options.get("threshold", 0.7)
        
        # Support both 'mode' (legacy) and 'modes' (new)
        requested_modes = corr_options.get("modes", [])
        if not requested_modes:
            requested_modes = [corr_options.get("mode", "all")]

        logger.info(f"Starting comprehensive correlation analysis for modes: {requested_modes}")
        logger.info(f"Methods: {methods}, Threshold: {threshold}")

       
        # CONSTANT_THRESHOLD = 1e-10 was too strict and excluded features with small but meaningful variance
        CONSTANT_THRESHOLD = 1e-8
        
        # Safeguard slow-moving astronomical columns (like Pluto's zodiac house) from being pruned
        astro_keywords = [
            "house", "aspect", "coordinate", "declination", "longitude", "latitude", "distance", "velocity",
            "phase", "retrograde", "planet", "pluto", "neptune", "uranus", "saturn", "jupiter", "mars",
            "venus", "mercury", "sun", "moon", "node", "part_of_fortune"
        ]
        def is_astro_col(col_name):
            name_lower = col_name.lower()
            return any(kw in name_lower for kw in astro_keywords)
            
        constant_cols = [
            col for col in self.numeric_columns 
            if self.data[col].std() < CONSTANT_THRESHOLD and not is_astro_col(col)
        ]
        if constant_cols:
            logger.warning(f"Excluding constant columns (variance < {CONSTANT_THRESHOLD}): {constant_cols}")

        #
        if self.target_columns and len(self.target_columns) > 0:
           user_target_cols = [c for c in self.target_columns if c in self.data.columns and c not in constant_cols]
           logger.info(f"Using user-selected target columns for ohlcv_vs_features mode: {user_target_cols}")
        else:
            # Fallback to OHLCV if no targets specified
            ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
            user_target_cols = [c for c in ohlcv_cols if c in self.data.columns and c not in constant_cols]
            logger.info(f"No target columns specified, using OHLCV fallback: {user_target_cols}")

        # === OPTIMIZATION: Compute distance correlation ONCE per request ===
        # Since frontend can only display one distance matrix, compute it once and reuse
        distance_corr_cache = None
        if "distance" in methods:
            try:
                self._send_progress(
                    19, 
                    "Computing distance correlations...", 
                    "Computing distance matrix (computed once, reused for all modes)..."
                )
                distance_corr_cache = self._calculate_distance_correlation_matrix()
                
                if distance_corr_cache and isinstance(distance_corr_cache, dict) and len(distance_corr_cache) > 0:
                    actual_pairs = sum(len(v) if isinstance(v, dict) else 1 for v in distance_corr_cache.values())
                    logger.info(f"✅ Distance correlation computed once: {actual_pairs} pairs (will be reused for all {len(requested_modes)} mode(s))")
                    self._send_progress(
                        20, 
                        "Computing distance correlations...", 
                        f"✅ Distance matrix ready: {actual_pairs} pairs (reusing for all modes)"
                    )
                else:
                    logger.warning(f"⚠️ Distance correlation returned empty result")
                    distance_corr_cache = None
                    
            except TimeoutError as e:
                logger.error(f"❌ Distance correlation TIMEOUT: {e}")
                self._send_progress(
                    20, 
                    "Analyzing correlations...", 
                    f"❌ TIMEOUT: Distance correlation exceeded time limit. Skipping for all modes."
                )
                distance_corr_cache = None
            except Exception as e:
                logger.error(f"❌ Distance correlation ERROR: {e}", exc_info=True)
                self._send_progress(
                    20, 
                    "Analyzing correlations...", 
                    f"❌ ERROR in distance correlation: {str(e)[:80]}"
                )
                distance_corr_cache = None

        for m_idx, mode in enumerate(requested_modes):
            corr_analysis = {}
            mode_label = mode.replace("_", " ").title()
            
            self._send_progress(
                20 + (m_idx * 15),
                f"Analyzing correlations ({mode_label})...",
                f"Initializing {mode_label} correlation analysis with {len(methods)} methods"
            )

            # === Prepare data and column lists based on correlation mode ===
            if mode == "ohlcv_vs_features":
                # Rectangular: User-selected targets (rows) vs other features (columns)
                # ✅ FIX: Use user_target_cols instead of hardcoded available_ohlcv
                row_cols = user_target_cols
                col_cols = [c for c in self.numeric_columns if c not in user_target_cols and c not in constant_cols]
                data_for_correlation = self.data[row_cols + col_cols]
                logger.info(f"OHLCV mode: {len(row_cols)} target cols × {len(col_cols)} features")
            elif mode == "targets_vs_features":
                # Rectangular: Targets (rows) vs Features (columns)
                if not self.has_target:
                    logger.warning(f"Skipping targets_vs_features mode: no targets defined")
                    continue
                row_cols = self.targets_df.columns.tolist()
                col_cols = self.feature_columns
                data_for_correlation = pd.concat([self.targets_df, self.X], axis=1)
                logger.info(f"Targets mode: {len(row_cols)} targets × {len(col_cols)} features")
            else:  # all
                # Square: All numeric columns vs all numeric columns
                row_cols = [c for c in self.numeric_columns if c not in constant_cols]
                col_cols = row_cols
                data_for_correlation = self.data[row_cols]
                logger.info(f"All mode: {len(row_cols)} columns (square matrix)")

            # === Calculate correlation matrices for all modes ===
            for method in ["pearson", "spearman", "kendall"]:
                if method in methods:
                    self._send_progress(
                        22 + (m_idx * 15), 
                        f"Analyzing {mode_label} correlations...", 
                        f"Computing {method.title()} Matrix ({len(row_cols)}×{len(col_cols)} = {len(row_cols)*len(col_cols)} pairs)"
                    )
                    
                    # Call unified correlation calculator with proper mode parameters
                    corr_matrix, pvalues_matrix = self._calculate_correlation_matrix_subset(
                        data_for_correlation,
                        method,
                        row_cols,  # Y-axis labels
                        col_cols,  # X-axis labels
                        mode=mode,
                        progress_base=22 + (m_idx * 15),
                        progress_span=5
                    )
                    
                    corr_analysis[method] = {
                        "correlation_matrix": corr_matrix,
                        "p_values": pvalues_matrix,
                        "significant_pairs": self._find_significant_correlations(
                            corr_matrix, pvalues_matrix
                        ),
                        "strong_correlations": self._find_strong_correlations(
                            corr_matrix, threshold=threshold
                        ),
                        "mode": mode,
                        "row_labels": row_cols,  # NEW: Explicit axis labels
                        "col_labels": col_cols,  # NEW: Explicit axis labels
                        "is_rectangular": mode != "all"
                    }

            # === Add cached distance correlation to this mode (OPTIMIZATION: computed once, reused) ===
            if "distance" in methods and distance_corr_cache is not None:
                corr_analysis["distance"] = {
                    "correlation_matrix": distance_corr_cache,
                    "strong_correlations": self._find_strong_correlations(
                        distance_corr_cache, threshold=0.5
                    ),
                    "cached": True,  # Flag indicating this is the cached single computation
                }
                logger.info(f"✅ Distance correlation REUSED for mode '{mode}' (computed once, not recomputed)")

            # 2. Specific Target-Feature Correlations (Always run if targets exist)
            if self.has_target and mode != "ohlcv_vs_features":
                target_results = {}
                target_count = len(self.targets_df.columns)
                feature_count = len(self.feature_columns)
                
                logger.info(f"Computing target correlations: {target_count} targets × {feature_count} features")
                
                for target_idx, target_name in enumerate(self.targets_df.columns):
                    target_correlations = {}
                    y_target = self.targets_df[target_name]
                    
                    progress_pct = 33 + (m_idx * 15)  # Progress window for target correlations
                    self._send_progress(
                        progress_pct, 
                        "Analyzing correlations...", 
                        f"Target sensitivity ({target_idx + 1}/{target_count}): {target_name}"
                    )
                    
                    # PHASE 4 TASK 11: Pre-compute all paired data once instead of in loop
                    paired_data = self._precompute_paired_data_matrices(
                        self.feature_columns,
                        [target_name],
                        data_source="X"
                    )
                    
                    total_pairs = len(paired_data)
                    processed_pairs = 0
                    
                    for col_idx, col in enumerate(self.feature_columns):
                        pair_key = (col, target_name)
                        
                        # Use pre-computed pair instead of pd.concat in loop
                        if pair_key not in paired_data:
                            continue
                        
                        processed_pairs += 1
                        # Send granular progress: current pair being processed
                        pair_progress = progress_pct + int((processed_pairs / max(total_pairs, 1)) * 5)
                        self._send_progress(
                            pair_progress,
                            f"Analyzing {target_name} correlations...",
                            f"Pair {processed_pairs}/{total_pairs}: {col} vs {target_name}"
                        )
                        
                        df_pair = paired_data[pair_key]
                        x, y_series = df_pair[col], df_pair[target_name]
                        for method in ["pearson", "spearman", "kendall"]:
                            if method in methods:
                                try:
                                    if method == "pearson": 
                                        corr, pval = pearsonr(x, y_series)
                                    elif method == "spearman": 
                                        corr, pval = spearmanr(x, y_series)
                                    else: 
                                        corr, pval = kendalltau(x, y_series)

                                    if col not in target_correlations: 
                                        target_correlations[col] = {}
                                    target_correlations[col][method] = {
                                        "correlation": float(corr),
                                        "p_value": float(pval),
                                        "significant": bool(pval < 0.05),
                                    }
                                except Exception as e:
                                    logger.debug(f"Failed to compute {method} for {col} vs {target_name}: {e}")
                                    pass
                    
                    target_results[target_name] = target_correlations
                    logger.info(f"Completed {target_name} correlations: {len(target_correlations)} features analyzed")
                
                corr_analysis["target_correlations"] = target_results
                self._send_progress(34 + (m_idx * 15), "Analyzing correlations...", f"All {target_count} target correlations completed")
            
            # === STORE CONFIG FOR FRONTEND ===
            corr_analysis["config"] = {
                "mode": mode,
                "threshold": threshold,
                "methods": methods,
                "target_count": len(self.targets_df.columns) if self.has_target else 0
            }
            all_results[mode] = corr_analysis

        return all_results

    def calculate_feature_importance_comprehensive(self) -> Dict[str, Any]:
        """
        Calculate feature importance using exhaustive methods for all targets.
        """
        if not self.has_target:
            return {"error": "Target column not set for feature importance."}

        multi_target_results = {}
        fi_options = self.options.get("feature_importance", {})
        methods = fi_options.get(
            "methods",
            [
                "random_forest",
                "linear_regression",
                "lasso",
                "mutual_information",
                "f_statistic",
                "recursive_feature_elimination",
                "permutation_importance",
            ],
        )
        total_methods = len(methods)
        targets = self.targets_df.columns.tolist()
        total_tasks = total_methods * len(targets)
        task_count = 0

        self._send_progress(
            25, 
            f"Starting feature importance for {len(targets)} targets with {total_methods} methods.",
            f"Evaluating {total_methods} importance methods across {len(targets)} target(s)"
        )

        for target_name in targets:
            y_target = self.targets_df[target_name]
            importance_results = {}
            
            self._send_progress(
                25,
                f"Analyzing target: {target_name}",
                f"Preparing feature matrix ({len(self.X.columns)} features) for importance calculation"
            )
            
            for i, method in enumerate(methods):
                task_count += 1
                progress = 25 + int((task_count / total_tasks) * 65)
                
                # Granular progress: which method, which target, which feature count
                self._send_progress(
                    progress,
                    f"Target [{target_name}]: Calculating {method.replace('_', ' ').title()}...",
                    f"Method {i+1}/{total_methods}: {method.replace('_', ' ').title()} on {len(self.X.columns)} features"
                )

                try:
                    if method == "random_forest":
                        self._send_progress(
                            progress,
                            f"Target [{target_name}]: Training Random Forest...",
                            f"Fitting 100-tree ensemble with {len(self.X.columns)} features (n_jobs=-1)"
                        )
                        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                        rf.fit(self.X, y_target)
                        importance_results["random_forest"] = {
                            col: float(imp)
                            for col, imp in zip(self.X.columns, rf.feature_importances_)
                        }
                        self._send_progress(
                            progress + 1,
                            f"Target [{target_name}]: Random Forest complete",
                            f"Extracted importance scores for {len(self.X.columns)} features"
                        )

                    elif method == "linear_regression":
                        self._send_progress(
                            progress,
                            f"Target [{target_name}]: Computing Linear Regression Coefficients...",
                            f"Solving linear regression on scaled {len(self.X.columns)} features"
                        )
                        lr = LinearRegression()
                        lr.fit(self.X_scaled, y_target)
                        # Use absolute coefficients for importance ranking (scaled features)
                        importance_results["linear_regression"] = {
                            col: float(abs(coef)) for col, coef in zip(self.X.columns, lr.coef_)
                        }
                        importance_results["linear_regression_metadata"] = {
                            "note": "Absolute coefficients from scaled features",
                            "interpretation": "Larger magnitude = stronger linear relationship"
                        }

                    elif method == "lasso":
                        self._send_progress(
                            progress,
                            f"Target [{target_name}]: Running Lasso Regression...",
                            f"Training LassoCV with 5-fold cross-validation on {len(self.X_scaled.columns)} scaled features"
                        )
                        from sklearn.linear_model import LassoCV
                        lasso = LassoCV(random_state=42, cv=5)
                        lasso.fit(self.X_scaled, y_target)
                        importance_results["lasso"] = {
                            col: float(coef) for col, coef in zip(self.X.columns, lasso.coef_)
                        }
                        importance_results["lasso_alpha"] = float(lasso.alpha_)

                    elif method == "mutual_information":
                        self._send_progress(
                            progress,
                            f"Target [{target_name}]: Computing Mutual Information...",
                            f"Calculating entropy relationships with {len(self.X.columns)} features"
                        )
                        mi_scores = mutual_info_regression(self.X, y_target, random_state=42)
                        importance_results["mutual_information"] = {
                            col: float(score) for col, score in zip(self.X.columns, mi_scores)
                        }

                    elif method == "f_statistic":
                        self._send_progress(
                            progress,
                            f"Target [{target_name}]: Computing F-Statistic...",
                            f"Testing univariate feature significance with regression ({len(self.X.columns)} features)"
                        )
                        f_scores, f_pvalues = f_regression(self.X, y_target)
                        # Normalize F-scores to 0-1 range for cross-feature comparison
                        max_f = float(np.max(f_scores)) if np.max(f_scores) > 0 else 1.0
                        importance_results["f_statistic"] = {
                            col: {
                                "f_score": float(score),
                                "f_score_normalized": float(score / max_f),  # 0-1 range
                                "p_value": float(pval),
                                "significant": bool(pval < 0.05),
                            }
                            for col, score, pval in zip(self.X.columns, f_scores, f_pvalues)
                        }

                    elif method == "recursive_feature_elimination":
                        self._send_progress(
                            progress,
                            f"Target [{target_name}]: Running Recursive Feature Elimination...",
                            f"Eliminating features recursively to rank importance ({len(self.X_scaled.columns)} features)"
                        )
                        rfe = RFE(estimator=LinearRegression(), n_features_to_select=1)
                        rfe.fit(self.X_scaled, y_target)
                        
                        # Separate rankings from scores with metadata
                        max_rank = float(rfe.ranking_.max())
                        importance_results["recursive_feature_elimination"] = {
                            "rankings": {
                                col: int(rank) for col, rank in zip(self.X.columns, rfe.ranking_)
                            },
                            "importance_scores": {
                                col: float(1.0 / (rank / max_rank)) if rank > 0 else 0.0
                                for col, rank in zip(self.X.columns, rfe.ranking_)
                            },
                            "metadata": {
                                "method": "Recursive Feature Elimination",
                                "note": "ranking=1 is most important; higher values eliminated later",
                                "importance_scores": "inverse of normalized ranking for cross-method comparison"
                            }
                        }

                    elif method == "permutation_importance":
                        self._send_progress(
                            progress,
                            f"Target [{target_name}]: Computing Permutation Importance...",
                            f"Measuring feature impact via permutation on fitted forest"
                        )
                        # Use a fast RF for permutation importance if not already calculated
                        perf_rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
                        perf_rf.fit(self.X, y_target)
                        perm_importance = self._calculate_permutation_importance(
                            perf_rf, self.X, y_target
                        )
                        importance_results["permutation_importance"] = perm_importance
                except Exception as e:
                    logger.error(f"Error calculating {method} for target {target_name}: {e}")
                    importance_results[method] = {"error": str(e)}

            multi_target_results[target_name] = importance_results

        self._send_progress(
            95, 
            "Finalizing feature importance analysis.",
            f"Aggregating results from {total_methods} methods for {len(targets)} target(s)"
        )
        return multi_target_results

    def detect_outliers_comprehensive(self) -> Dict[str, Any]:
        """
        Detect outliers using exhaustive methods.
        """
        outlier_results = {"univariate": {}, "multivariate": {}}
        outlier_options = self.options.get("outliers", {})
        uni_methods = outlier_options.get(
            "univariate_methods", ["iqr", "z_score", "modified_z_score", "tukey"]
        )
        multi_methods = outlier_options.get(
            "multivariate_methods",
            ["isolation_forest", "elliptic_envelope", "dbscan", "mahalanobis"],
        )
        contamination = outlier_options.get("contamination", 0.1)

        # Univariate outlier detection
        total_uni_cols = len(self.numeric_columns)
        for i, col in enumerate(self.numeric_columns):
            progress = 70 + int((i/total_uni_cols)*5)
            self._send_progress(progress, "Detecting outliers...", f"Univariate: {col}")
            data_col = self.data[col].dropna()

            if len(data_col) == 0:
                continue
            
            outlier_results["univariate"][col] = {}

            if "iqr" in uni_methods:
                q1, q3 = data_col.quantile(0.25), data_col.quantile(0.75)
                iqr = q3 - q1
                lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                iqr_outliers = (
                    (data_col < lower_bound) | (data_col > upper_bound)
                ).sum()
                outlier_results["univariate"][col]["iqr_method"] = {
                    "outlier_count": int(iqr_outliers),
                    "outlier_percentage": float(iqr_outliers / len(data_col) * 100),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                }

            if "z_score" in uni_methods:
                z_scores = np.abs(stats.zscore(data_col))
                z_outliers = (z_scores > 3).sum()
                outlier_results["univariate"][col]["z_score_method"] = {
                    "outlier_count": int(z_outliers),
                    "outlier_percentage": float(z_outliers / len(data_col) * 100),
                }

            if "modified_z_score" in uni_methods:
                median = np.median(data_col)
                mad = np.median(np.abs(data_col - median))
                modified_z_scores = (
                    0.6745 * (data_col - median) / mad
                    if mad != 0
                    else np.zeros_like(data_col)
                )
                modified_z_outliers = (np.abs(modified_z_scores) > 3.5).sum()
                outlier_results["univariate"][col]["modified_z_score"] = {
                    "outlier_count": int(modified_z_outliers),
                    "outlier_percentage": float(
                        modified_z_outliers / len(data_col) * 100
                    ),
                }

            if "tukey" in uni_methods:
                q1, q3 = data_col.quantile(0.25), data_col.quantile(0.75)
                iqr = q3 - q1
                tukey_lower, tukey_upper = q1 - 3 * iqr, q3 + 3 * iqr
                tukey_outliers = (
                    (data_col < tukey_lower) | (data_col > tukey_upper)
                ).sum()
                outlier_results["univariate"][col]["tukey_fences"] = {
                    "outlier_count": int(tukey_outliers),
                    "outlier_percentage": float(tukey_outliers / len(data_col) * 100),
                    "lower_bound": float(tukey_lower),
                    "upper_bound": float(tukey_upper),
                }

        # Multivariate outlier detection
        try:
            if "isolation_forest" in multi_methods:
                self._send_progress(75, "Detecting outliers...", f"Training Isolation Forest ensemble on {len(self.X.columns)} scaled features")
                iso_forest = IsolationForest(
                    contamination=contamination, random_state=42
                )
                iso_labels = iso_forest.fit_predict(self.X_scaled)
                outlier_results["multivariate"]["isolation_forest"] = {
                    "outlier_count": int((iso_labels == -1).sum()),
                    "outlier_percentage": float(
                        (iso_labels == -1).sum() / len(iso_labels) * 100
                    ),
                    "outlier_indices": [int(i) for i in np.where(iso_labels == -1)[0]],
                }

            if "elliptic_envelope" in multi_methods:
                self._send_progress(76, "Detecting outliers...", f"Fitting Elliptic Envelope minimum covariance estimator")
                elliptic = EllipticEnvelope(
                    contamination=contamination, random_state=42
                )
                elliptic_labels = elliptic.fit_predict(self.X_scaled)
                outlier_results["multivariate"]["elliptic_envelope"] = {
                    "outlier_count": int((elliptic_labels == -1).sum()),
                    "outlier_percentage": float(
                        (elliptic_labels == -1).sum() / len(elliptic_labels) * 100
                    ),
                    "outlier_indices": [
                        int(i) for i in np.where(elliptic_labels == -1)[0]
                    ],
                }

            if "dbscan" in multi_methods:
                self._send_progress(77, "Detecting outliers...", f"Running DBSCAN density-based clustering algorithm")
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                dbscan_labels = dbscan.fit_predict(self.X_scaled)
                dbscan_outliers = (dbscan_labels == -1).sum()
                outlier_results["multivariate"]["dbscan"] = {
                    "outlier_count": int(dbscan_outliers),
                    "outlier_percentage": float(
                        dbscan_outliers / len(dbscan_labels) * 100
                    ),
                    "outlier_indices": [
                        int(i) for i in np.where(dbscan_labels == -1)[0]
                    ],
                }

            if "mahalanobis" in multi_methods:
                self._send_progress(78, "Detecting outliers...", f"Calculating Mahalanobis distance from centroid")
                try:
                    mahalanobis_dist = self._calculate_mahalanobis_distance(
                        self.X_scaled
                    )
                    mahalanobis_outliers = (
                        mahalanobis_dist > 3
                    ).sum()  # Using chi-square threshold
                    outlier_results["multivariate"]["mahalanobis_distance"] = {
                        "outlier_count": int(mahalanobis_outliers),
                        "outlier_percentage": float(
                            mahalanobis_outliers / len(mahalanobis_dist) * 100
                        ),
                        "outlier_indices": [
                            int(i) for i in np.where(mahalanobis_dist > 3)[0]
                        ],
                    }
                except Exception as e:
                    outlier_results["multivariate"]["mahalanobis_distance"] = {
                        "error": str(e)
                    }

        except Exception as e:
            outlier_results["multivariate"][
                "error"
            ] = f"Multivariate outlier detection failed: {str(e)}"

        return outlier_results

    def test_normality_comprehensive(self) -> Dict[str, Any]:
        """
        Perform exhaustive normality testing.
        """
        normality_results = {}

        total_norm_cols = len(self.numeric_columns)
        for i, col in enumerate(self.numeric_columns):
            progress = 75 + int((i/total_norm_cols)*5)
            self._send_progress(progress, "Testing normality...", f"Column: {col}")
            data_col = self.data[col].dropna()

            normality_results[col] = {}

            # Shapiro-Wilk test
            if len(data_col) <= 5000:
                shapiro_stat, shapiro_p = shapiro(data_col)
                normality_results[col]["shapiro_wilk"] = {
                    "statistic": float(shapiro_stat),
                    "p_value": float(shapiro_p),
                    "is_normal": bool(shapiro_p > 0.05),
                }

            # D'Agostino-Pearson test
            if len(data_col) >= 20:
                dagostino_stat, dagostino_p = normaltest(data_col)
                normality_results[col]["dagostino_pearson"] = {
                    "statistic": float(dagostino_stat),
                    "p_value": float(dagostino_p),
                    "is_normal": bool(dagostino_p > 0.05),
                }

            # Anderson-Darling test
            anderson_result = anderson(data_col)
            normality_results[col]["anderson_darling"] = {
                "statistic": float(anderson_result.statistic),
                "critical_values": {
                    f"{sig_level}": float(cv)
                    for sig_level, cv in zip(
                        anderson_result.significance_level,
                        anderson_result.critical_values,
                    )
                },
                "is_normal_5%": bool(
                    anderson_result.statistic < anderson_result.critical_values[2]
                ),
            }

            # Jarque-Bera test
            jb_stat, jb_p = jarque_bera(data_col)
            normality_results[col]["jarque_bera"] = {
                "statistic": float(jb_stat),
                "p_value": float(jb_p),
                "is_normal": bool(jb_p > 0.05),
            }

            # Kolmogorov-Smirnov test
            ks_stat, ks_p = kstest(
                data_col, "norm", args=(data_col.mean(), data_col.std())
            )
            normality_results[col]["kolmogorov_smirnov"] = {
                "statistic": float(ks_stat),
                "p_value": float(ks_p),
                "is_normal": bool(ks_p > 0.05),
            }

            # Lilliefors test (corrected KS test)
            try:
                from statsmodels.stats.diagnostic import lilliefors

                lillie_stat, lillie_p = lilliefors(data_col)
                normality_results[col]["lilliefors"] = {
                    "statistic": float(lillie_stat),
                    "p_value": float(lillie_p),
                    "is_normal": bool(lillie_p > 0.05),
                }
            except ImportError:
                pass

        return normality_results

    def detect_multicollinearity_advanced(self) -> Dict[str, Any]:
        """
        Advanced multicollinearity detection with VIF and condition number.
        """
        multicollinearity = {}

        # OPTIMIZATION: Variance Inflation Factor via correlation matrix inversion
        # Instead of fitting n regression models (O(n⁴)), compute VIF from correlation matrix inverse (O(n³) once)
        vif_data = {}
        try:
            self._send_progress(80, "Analyzing Collinearity...", "Computing VIF from correlation matrix...")
            
            # Compute correlation matrix once
            corr_matrix = self.X.corr()
            
            # Compute inverse of correlation matrix
            # VIF = diagonal of (X'X)^-1, which is proportional to correlation matrix inverse diagonal
            try:
                inv_corr = np.linalg.inv(corr_matrix.values)
                vif_values = np.diagonal(inv_corr)
                
                total_features = len(self.X.columns)
                for i, col in enumerate(self.X.columns):
                    # Add progress for each feature
                    if i % max(1, total_features // 10) == 0:
                        progress = 80 + int((i / total_features) * 5)
                        self._send_progress(
                            progress,
                            "Computing VIF scores...",
                            f"Analyzing feature {i+1}/{total_features}: {col}"
                        )
                    
                    vif = float(vif_values[i])
                    vif_data[col] = {
                        "vif": vif,
                        "interpretation": self._interpret_vif(vif),
                        "severity": (
                            "high" if vif > 10 else "moderate" if vif > 5 else "low"
                        ),
                    }
                
                logger.info(f"VIF calculation completed using matrix inversion method (O(n³) single computation)")
            except np.linalg.LinAlgError:
                # Singular matrix - features are perfectly multicollinear
                logger.warning("Correlation matrix is singular - perfect multicollinearity detected")
                for col in self.X.columns:
                    vif_data[col] = {
                        "vif": float('inf'),
                        "interpretation": "Perfect multicollinearity detected",
                        "severity": "high"
                    }
        except Exception as e:
            logger.error(f"VIF calculation failed: {e}")
            for col in self.X.columns:
                vif_data[col] = {"error": str(e)}

        multicollinearity["vif"] = vif_data

        # Condition Number
        try:
            corr_matrix = self.X.corr()
            eigenvalues = np.linalg.eigvals(corr_matrix)
            condition_number = np.sqrt(eigenvalues.max() / eigenvalues.min())
            multicollinearity["condition_number"] = {
                "value": float(condition_number),
                "interpretation": self._interpret_condition_number(condition_number),
                "severity": (
                    "high"
                    if condition_number > 30
                    else "moderate" if condition_number > 15 else "low"
                ),
            }
        except Exception as e:
            multicollinearity["condition_number"] = {"error": str(e)}

        # Correlation matrix analysis
        corr_matrix = self.X.corr()
        high_corr_pairs = []
        for i, col1 in enumerate(self.X.columns):
            for j, col2 in enumerate(self.X.columns[i + 1 :], i + 1):
                corr_val = corr_matrix.loc[col1, col2]
                if abs(corr_val) > 0.8:
                    high_corr_pairs.append(
                        {
                            "feature1": col1,
                            "feature2": col2,
                            "correlation": float(corr_val),
                            "severity": "very_high" if abs(corr_val) > 0.9 else "high",
                        }
                    )

        multicollinearity["high_correlation_pairs"] = high_corr_pairs

        # Eigenvalue analysis
        try:
            multicollinearity["eigenvalue_analysis"] = {
                "eigenvalues": [float(val) for val in eigenvalues],
                "small_eigenvalues": [float(val) for val in eigenvalues if val < 0.01],
                "condition_index": float(condition_number),
            }
        except Exception as e:
            multicollinearity["eigenvalue_analysis"] = {"error": str(e)}

        return multicollinearity

    def perform_pca_comprehensive(self) -> Dict[str, Any]:
        """
        Perform comprehensive Principal Component Analysis.
        OPTIMIZED: Limits components to min(100, n_features) to avoid unnecessary computation.
        """
        # OPTIMIZATION: Cap PCA components at 100 to reduce computation
        # Most use cases need only 10-50 components for 95%+ variance explanation
        max_useful_components = 100
        n_components = min(max_useful_components, len(self.X.columns), len(self.X))
        
        # Safeguard for extremely small datasets
        if n_components < 1 or len(self.X) < 2:
            return {
                "error": "Insufficient data for PCA (needs at least 2 rows/columns)",
                "metadata": {"total_rows": len(self.X), "is_sampled": False}
            }

        self._send_progress(
            85, 
            "Performing PCA...", 
            f"Computing {n_components} principal components from {len(self.X.columns)} features ({len(self.X)} samples)"
        )
        pca = PCA(n_components=n_components)
        pca_transformed = pca.fit_transform(self.X_scaled)
        
        # Add progress after transformation
        self._send_progress(
            87,
            "Performing PCA...",
            f"Transformation complete. Explained variance: {np.sum(pca.explained_variance_ratio_)*100:.1f}%"
        )

        # SAMPLE DATA FOR FRONTEND VISUALIZATION (prevent WebSocket crashes)
        # We process stats on ALL data, but only send 5000 points for the 3D plot
        viz_sample_size = 5000
        if len(pca_transformed) > viz_sample_size:
            # Create random indices
            rng = np.random.default_rng(42)
            indices = rng.choice(len(pca_transformed), size=viz_sample_size, replace=False)
            pca_viz_data = pca_transformed[indices]
            logger.info(f"Sampling PCA data for visualization: {len(pca_transformed)} -> {viz_sample_size}")
        else:
            pca_viz_data = pca_transformed
            indices = np.arange(len(pca_transformed))

        pca_results = {
            "explained_variance_ratio": [
                float(x) for x in pca.explained_variance_ratio_
            ],
            "cumulative_variance_ratio": [
                float(x) for x in np.cumsum(pca.explained_variance_ratio_)
            ],
            "singular_values": [float(x) for x in pca.singular_values_],
            "components": {
                f"PC{i+1}": {
                    col: float(val)
                    for col, val in zip(self.X.columns, pca.components_[i])
                }
                for i in range(len(pca.components_))
            },
            "n_components_95_variance": int(
                np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
            ),
            "n_components_99_variance": int(
                np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.99) + 1
            ),
            "total_variance_explained": float(np.sum(pca.explained_variance_ratio_)),
            "component_correlation": self._calculate_pca_feature_correlation(
                pca, self.X_scaled
            ),
            "transformed_data": {
                "x": [float(val) for val in pca_viz_data[:, 0].tolist()] if pca_viz_data.shape[1] > 0 else [],
                "y": [float(val) for val in pca_viz_data[:, 1].tolist()] if pca_viz_data.shape[1] > 1 else [],
                "z": [float(val) for val in pca_viz_data[:, 2].tolist()] if pca_viz_data.shape[1] > 2 else [],
                "indices": [int(i) for i in indices], # Send indices to match metadata
            },
            "metadata": {
               "total_rows": int(len(self.X)),
               "sampled_rows": int(len(pca_viz_data)),
               "is_sampled": bool(len(self.X) > viz_sample_size)
            }
        }

        # Scree plot data
        pca_results["scree_data"] = {
            "components": [
                f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))
            ],
            "variance_explained": [float(x) for x in pca.explained_variance_ratio_],
            "cumulative_variance": [
                float(x) for x in np.cumsum(pca.explained_variance_ratio_)
            ],
        }

        return pca_results

    # Additional comprehensive analysis methods would follow similar patterns...
    # [The remaining methods would be implemented with similar thoroughness]

    # Helper methods for various calculations
    def _calculate_correlation_matrix(self, method: str, progress_base: int = 30, progress_span: int = 5) -> Tuple[Dict, Dict]:
        """Calculate correlation matrix with p-values for given method."""
        corr_matrix = self.X.corr(method=method)
        pvalue_matrix = pd.DataFrame(
            np.ones((len(self.X.columns), len(self.X.columns))),
            columns=corr_matrix.columns,
            index=corr_matrix.index,
        )

        total_cells = len(self.X.columns) ** 2
        cell_count = 0
        for col1 in self.X.columns:
            for col2 in self.X.columns:
                cell_count += 1
                if cell_count % max(1, total_cells // 10) == 0:
                    progress = progress_base + int((cell_count / total_cells) * progress_span)
                    self._send_progress(progress, f"Computing {method}...", f"Pair: {col1} x {col2}")
                
                if col1 == col2:
                    pvalue_matrix.loc[col1, col2] = 0.0
                    continue

                df_pair = self.X[[col1, col2]].dropna()

                if len(df_pair) < 2:
                    pvalue_matrix.loc[col1, col2] = 1.0
                    continue

                x = df_pair[col1]
                y_series = df_pair[col2]

                if method == "pearson":
                    _, p_val = pearsonr(x, y_series)
                elif method == "spearman":
                    _, p_val = spearmanr(x, y_series)
                else:  # kendall
                    _, p_val = kendalltau(x, y_series)
                pvalue_matrix.loc[col1, col2] = p_val

        return corr_matrix.to_dict(), pvalue_matrix.to_dict()

    def _calculate_correlation_matrix_subset(
        self,
        data: pd.DataFrame,
        method: str,
        row_labels: List[str],
        col_labels: List[str],
        mode: str = "all",
        progress_base: int = 30,
        progress_span: int = 5
    ) -> Tuple[Dict, Dict]:
        """
        Calculate correlation matrix for specified row and column labels.
        
        Args:
            data: DataFrame containing all necessary columns
            method: Correlation method (pearson, spearman, kendall)
            row_labels: Column names for Y-axis (rows in result matrix)
            col_labels: Column names for X-axis (columns in result matrix)
            mode: Correlation mode for logging
            progress_base: Starting progress percentage
            progress_span: Progress range to use
            
        Returns:
            Tuple of (correlation_dict, pvalue_dict) where:
            - For rectangular: {row_label: {col_label: value}}
            - For square: {label: {label: value}}
        """
        try:
            # Validate inputs
            if not row_labels or not col_labels:
                logger.error(f"Empty labels: rows={len(row_labels)}, cols={len(col_labels)}")
                return {}, {}
            
            # Get numeric data only
            numeric_data = data.select_dtypes(include=[np.number])
            
            # Validate all labels exist
            missing_rows = [r for r in row_labels if r not in numeric_data.columns]
            missing_cols = [c for c in col_labels if c not in numeric_data.columns]
            
            if missing_rows:
                logger.warning(f"Missing row labels: {missing_rows}")
                row_labels = [r for r in row_labels if r in numeric_data.columns]
            if missing_cols:
                logger.warning(f"Missing col labels: {missing_cols}")
                col_labels = [c for c in col_labels if c in numeric_data.columns]
            
            if not row_labels or not col_labels:
                logger.error("No valid labels after filtering")
                return {}, {}
            
            # ✅ FIX: Use dict.fromkeys() instead of set() to preserve column ordering
            # set() loses column order which can cause index misalignment issues
            unique_cols_ordered = list(dict.fromkeys(row_labels + col_labels))
            working_data = numeric_data[unique_cols_ordered].copy()
            
            # Remove NaN values
            clean_data = working_data.dropna()
            
            if len(clean_data) < 2:
                logger.warning(f"Insufficient data after dropna: {len(clean_data)} rows")
                return {}, {}
            
            # === DEBUG LOGGING FOR CORRELATION INVESTIGATION ===
            logger.info(f"🔍 [CORRELATION DEBUG] Clean data shape: {clean_data.shape}")
            logger.info(f"🔍 [CORRELATION DEBUG] Row labels: {row_labels[:5]}... ({len(row_labels)} total)")
            logger.info(f"🔍 [CORRELATION DEBUG] Col labels: {col_labels[:5]}... ({len(col_labels)} total)")
            
            # Log sample variances
            sample_variances = clean_data.var()
            logger.info(f"🔍 [CORRELATION DEBUG] Sample variances (first 5):")
            for col in list(sample_variances.index)[:5]:
                logger.info(f"  {col}: {sample_variances[col]:.6f}")
            
            # Log first few rows
            logger.info(f"🔍 [CORRELATION DEBUG] First 3 rows of clean_data:")
            logger.info(f"\n{clean_data.head(3)}")
            
            # Compute and log a test correlation
            if len(row_labels) > 0 and len(col_labels) > 0:
                test_row = row_labels[0]
                test_col = col_labels[0]
                test_x = clean_data[test_row].values
                test_y = clean_data[test_col].values
                test_mask = ~(np.isnan(test_x) | np.isnan(test_y))
                if test_mask.sum() >= 3:
                    from scipy.stats import pearsonr as test_pearsonr
                    test_corr, test_pval = test_pearsonr(test_x[test_mask], test_y[test_mask])
                    logger.info(f"🔍 [CORRELATION DEBUG] Test correlation {test_row} vs {test_col}: {test_corr:.6f} (p={test_pval:.6f})")
                else:
                    logger.warning(f"🔍 [CORRELATION DEBUG] Insufficient data for test correlation: {test_mask.sum()} samples")
            
            logger.info(f"Computing {method} correlation: {len(row_labels)} rows × {len(col_labels)} cols, {len(clean_data)} samples")
            
            # Determine if rectangular or square
            is_rectangular = set(row_labels) != set(col_labels)
            
            if is_rectangular:
                # === RECTANGULAR MATRIX: row_labels (Y) vs col_labels (X) ===
                logger.info(f"Rectangular mode: {len(row_labels)} targets × {len(col_labels)} features")
                
                # Try parallel computation first
                try:
                    parallel_corr, parallel_pval = parallel_rectangular_correlation(
                        clean_data,
                        row_labels,
                        col_labels,
                        method=method,
                        n_workers=None
                    )
                    
                    if parallel_corr is not None and parallel_pval is not None:
                        logger.info(f"Parallel {method} correlation completed successfully")
                        return parallel_corr, parallel_pval
                except Exception as e:
                    logger.warning(f"Parallel correlation failed, using sequential: {e}")
                
                # Sequential fallback
                corr_matrix = {}
                pvalue_matrix = {}
                
                total_cells = len(row_labels) * len(col_labels)
                cell_count = 0
                
                for row_label in row_labels:
                    corr_matrix[row_label] = {}
                    pvalue_matrix[row_label] = {}
                    
                    for col_label in col_labels:
                        cell_count += 1
                        
                        if cell_count % max(1, total_cells // 20) == 0:
                            progress = progress_base + int((cell_count / total_cells) * progress_span)
                            percent = int((cell_count / total_cells) * 100)
                            self._send_progress(
                                progress,
                                f"Computing {method.title()} ({percent}%)...",
                                f"Pair {cell_count}/{total_cells}: [{row_label}] vs [{col_label}]"
                            )
                        
                        # Get paired data
                        x = clean_data[row_label].values
                        y = clean_data[col_label].values
                        
                        # Remove NaN pairs
                        mask = ~(np.isnan(x) | np.isnan(y))
                        x_clean = x[mask]
                        y_clean = y[mask]
                        
                        if len(x_clean) < 3:
                            corr_matrix[row_label][col_label] = 0.0
                            pvalue_matrix[row_label][col_label] = 1.0
                            continue
                        
                        # Check variance
                        if np.var(x_clean) < 1e-10 or np.var(y_clean) < 1e-10:
                            corr_matrix[row_label][col_label] = 0.0
                            pvalue_matrix[row_label][col_label] = 1.0
                            continue
                        
                        try:
                            if method == "pearson":
                                corr_val, p_val = pearsonr(x_clean, y_clean)
                            elif method == "spearman":
                                corr_val, p_val = spearmanr(x_clean, y_clean)
                            else:  # kendall
                                corr_val, p_val = kendalltau(x_clean, y_clean)
                            
                            corr_matrix[row_label][col_label] = float(corr_val) if not np.isnan(corr_val) else 0.0
                            pvalue_matrix[row_label][col_label] = float(p_val) if not np.isnan(p_val) else 1.0
                        except Exception as e:
                            logger.debug(f"Correlation failed for {row_label} vs {col_label}: {e}")
                            corr_matrix[row_label][col_label] = 0.0
                            pvalue_matrix[row_label][col_label] = 1.0
                
                logger.info(f"Rectangular {method} matrix computed: {total_cells} pairs")
                return corr_matrix, pvalue_matrix
            
            else:
                # === SQUARE MATRIX: All vs All ===
                logger.info(f"Square matrix: {len(row_labels)} columns")
                
                # Use pandas correlation for robustness
                try:
                    if method == "pearson":
                        pandas_corr = clean_data[row_labels].corr(method="pearson")
                    elif method == "spearman":
                        pandas_corr = clean_data[row_labels].corr(method="spearman")
                    else:  # kendall
                        pandas_corr = clean_data[row_labels].corr(method="kendall")
                    
                    logger.info(f"Pandas {method} correlation computed: {pandas_corr.shape}")
                except Exception as e:
                    logger.error(f"Pandas correlation failed: {e}")
                    return {}, {}
                
                # Calculate p-values
                pvalue_matrix = {}
                total_cells = len(row_labels) ** 2
                cell_count = 0
                
                for col1 in row_labels:
                    pvalue_matrix[col1] = {}
                    
                    for col2 in row_labels:
                        cell_count += 1
                        
                        if col1 == col2:
                            pvalue_matrix[col1][col2] = 0.0
                            continue
                        
                        if cell_count % max(1, total_cells // 10) == 0:
                            progress = progress_base + int((cell_count / total_cells) * progress_span)
                            self._send_progress(progress, f"Computing {method} p-values...", f"Pair: {col1} x {col2}")
                        
                        x = clean_data[col1].values
                        y = clean_data[col2].values
                        
                        mask = ~(np.isnan(x) | np.isnan(y))
                        x_clean = x[mask]
                        y_clean = y[mask]
                        
                        if len(x_clean) < 3:
                            pvalue_matrix[col1][col2] = 1.0
                            continue
                        
                        try:
                            if method == "pearson":
                                _, p_val = pearsonr(x_clean, y_clean)
                            elif method == "spearman":
                                _, p_val = spearmanr(x_clean, y_clean)
                            else:
                                _, p_val = kendalltau(x_clean, y_clean)
                            pvalue_matrix[col1][col2] = float(p_val) if not np.isnan(p_val) else 1.0
                        except:
                            pvalue_matrix[col1][col2] = 1.0
                
                return pandas_corr.to_dict(), pvalue_matrix
            
        except Exception as e:
            logger.error(f"Error calculating {method} correlation: {e}", exc_info=True)
            return {}, {}

    def _calculate_distance_correlation_matrix(self) -> Dict[str, Any]:
        """
        Calculate distance correlation matrix with smart configuration-aware computation.
        
        Configuration-aware: Respects correlation mode to determine which pairs to compute.
        - "all" mode: Computes all-vs-all numeric column pairs
        - "targets_vs_features" mode: Computes feature→target pairs only
        - "ohlcv_vs_features" mode: Computes OHLCV→other features pairs
        """
        # Get correlation config and current mode being processed
        corr_options = self.options.get("correlation", {})
        requested_modes = corr_options.get("modes", [corr_options.get("mode", "all")])
        
        # Determine which mode we're in for distance correlation
        # Only compute "all" if explicitly requested, otherwise use "targets_vs_features"
        should_compute_all = "all" in requested_modes
        
        if should_compute_all:
            # User explicitly wants all-vs-all correlations
            columns = self.X.columns.tolist()
            target_cols = columns
            feature_cols = columns
            pair_count = len(columns) * len(columns)
            logger.info(f"📊 Distance correlation MODE=ALL: Computing {len(columns)}×{len(columns)} = {pair_count} pairs")
            self._send_progress(30, "Computing Distance Matrix...", 
                              f"📊 PHASE 1: Preparing ALL numeric columns distance correlation ({pair_count} pairs total)...")
        else:
            # Only compute feature→target (skip feature→feature)
            target_cols = self.targets_df.columns.tolist() if self.has_target else []
            feature_cols = self.feature_columns
            
            if not target_cols:
                logger.warning("⚠️ Distance correlation: No targets available for feature→target mode, skipping")
                self._send_progress(30, "Computing Distance Matrix...", 
                                  "⚠️ No target columns found, skipping feature→target distance correlation")
                return {}
            
            pair_count = len(feature_cols) * len(target_cols)
            logger.info(f"📊 Distance correlation MODE=TARGETS_VS_FEATURES: Computing {len(feature_cols)} features × {len(target_cols)} targets = {pair_count} pairs")
            self._send_progress(30, "Computing Distance Matrix...", 
                              f"📊 PHASE 1: Preparing TARGETS_VS_FEATURES distance correlation ({len(feature_cols)}F × {len(target_cols)}T = {pair_count} pairs total)...")
        
        # Try optimized 2-level parallel approach
        try:
            self._send_progress(31, "Computing Distance Matrix...", 
                              f"📊 PHASE 2: Processing {pair_count} distance correlation pairs using parallelization (may take minutes for large datasets)...")
            distance_corr = self._compute_distance_correlation_parallel_nested(
                feature_cols, target_cols, should_compute_all
            )
            
            if distance_corr:
                self._send_progress(32, "Computing Distance Matrix...", 
                                  f"✅ Distance correlation completed via parallelization ({pair_count} pairs)")
                logger.info(f"✅ Distance correlation: Completed {pair_count} pairs using nested parallelization")
                
                # Transpose if needed to match {row: {col: value}} format
                # Row labels are target_cols, column labels are feature_cols
                
                # Let's do a proper transposition
                final_matrix = {}
                for r_label in target_cols:
                    final_matrix[r_label] = {}
                    for c_label in feature_cols:
                        # distance_corr is {feature: {target: value}}
                        if c_label in distance_corr and r_label in distance_corr[c_label]:
                            final_matrix[r_label][c_label] = distance_corr[c_label][r_label]
                
                return final_matrix
        except Exception as e:
            logger.warning(f"⚠️ Nested parallel distance correlation failed, falling back to sequential: {e}", exc_info=True)
            self._send_progress(31, "Computing Distance Matrix...", 
                              f"⚠️ Parallel approach failed, using sequential fallback for {pair_count} pairs...")

        # Fallback: Sequential with progress
        logger.info(f"⏳ Distance correlation: Using sequential fallback for {pair_count} pairs")
        distance_corr = self._compute_distance_correlation_sequential_smart(
            feature_cols, target_cols, should_compute_all
        )
        
        # Transpose to match {target: {feature: value}} format
        final_matrix = {}
        for r_label in target_cols:
            final_matrix[r_label] = {}
            for c_label in feature_cols:
                if c_label in distance_corr and r_label in distance_corr[c_label]:
                    final_matrix[r_label][c_label] = distance_corr[c_label][r_label]
        
        return final_matrix
    
    def _compute_distance_correlation_parallel_nested(
        self, feature_cols: List[str], target_cols: List[str], compute_all: bool
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """
        2-level nested parallelization for distance correlation:
        - Level 1: Chunk features across CPU cores
        - Level 2: Each core computes against targets without creating sub-pools
        
        This avoids contention because different cores process different feature subsets,
        and work is naturally staged (shorter operations complete first, freeing cores).
        """
        from app.core.services.multiprocessing_utils import ChunkingStrategy, ColumnChunker
        from multiprocessing import Pool
        
        n_features = len(feature_cols)
        n_targets = len(target_cols)
        total_pairs = n_features * n_targets
        
        # Decide on worker count based on dataset size
        # Distance correlation is very expensive (O(N^2) in distance computation)
        # Low threshold (200 pairs) because even a few hundred distance corrs benefit from parallelization
        should_use_parallel = total_pairs > 200
        
        if not should_use_parallel:
            logger.info(f"Distance correlation: Only {total_pairs} pairs, using sequential")
            return None  # Fall back to sequential
        
        try:
            n_workers = ChunkingStrategy.auto_chunk_count()
            
            # Level 1: Split features across workers
            feature_chunks = ColumnChunker.chunk_columns(feature_cols, n_workers)
            
            logger.info(f"Distance correlation: Nested parallelization - "
                       f"{len(feature_chunks)} feature chunks × {n_targets} targets = "
                       f"{total_pairs} total pairs")
            
            # PHASE 4: Use SharedMemory to prevent OOM
            # Create a shared memory block for the NumPy array
            from multiprocessing import shared_memory
            
            # Prepare data
            data_to_share = pd.concat([self.X, self.targets_df], axis=1).values.astype(np.float64)
            shm = shared_memory.SharedMemory(create=True, size=data_to_share.nbytes)
            try:
                # Map to NumPy array and copy data
                shm_array = np.ndarray(data_to_share.shape, dtype=data_to_share.dtype, buffer=shm.buf)
                shm_array[:] = data_to_share[:]
                
                shm_info = {
                    "shm_name": shm.name,
                    "shape": data_to_share.shape,
                    "dtype": str(data_to_share.dtype)
                }
                
                feature_to_idx = {name: i for i, name in enumerate(self.feature_columns)}
                target_to_idx = {name: i + len(self.feature_columns) for i, name in enumerate(self.targets_df.columns)}
                
                with Pool(n_workers) as pool:
                    results = pool.starmap(
                        _nested_distance_corr_worker,
                        [(chunk, feature_to_idx, target_to_idx, target_cols, shm_info) for chunk in feature_chunks]
                    )
                
                # Merge results
                final_distances = {}
                for chunk_res in results:
                    final_distances.update(chunk_res)
                
                return final_distances
            finally:
                # Always cleanup shared memory
                shm.close()
                shm.unlink()
        except ImportError:
            logger.warning("multiprocessing.shared_memory not available, using standard parallelization")
            # Minimal implementation without SHM if it failed to import (shouldn't happen on 3.12)
            # ... but for brevity, we'll just fall through to the general exception handler
            raise
        except Exception as e:
            # ✅ FIX: Prepare data from correct sources
            # Features come from self.X, targets come from self.targets_df (or self.data for OHLCV mode)
            # Build combined array with features first, then targets
            feature_data = self.X[feature_cols].values
            
            # Get target data from the appropriate source
            if all(col in self.targets_df.columns for col in target_cols):
                target_data = self.targets_df[target_cols].values
            else:
                # Fallback: get from self.data (for OHLCV mode where targets might be features)
                target_data = self.data[target_cols].values
            
            X_np = np.hstack([feature_data, target_data])
            
            feature_to_idx = {col: i for i, col in enumerate(feature_cols)}
            target_to_idx = {col: len(feature_cols) + i for i, col in enumerate(target_cols)}
            
            # Create worker arguments - pass numpy data directly
            worker_args = [
                (
                    chunk,                    # feature_chunk
                    feature_to_idx,           # mapping
                    target_to_idx,            # mapping
                    target_cols,              # which targets to compute
                    X_np                      # shared data
                )
                for chunk in feature_chunks
            ]
            
            # Level 2: Execute feature chunks in parallel with progress reporting
            distance_corr = {}
            completed_chunks = 0
            total_chunks = len(feature_chunks)
            total_pairs_processed = 0
            
            with Pool(n_workers) as pool:
                try:
                    # Use imap instead of starmap to get results as they complete
                    for chunk_idx, chunk_result in enumerate(pool.starmap(
                        _nested_distance_corr_worker,
                        worker_args,
                        chunksize=1
                    )):
                        if chunk_result:
                            distance_corr.update(chunk_result)
                            completed_chunks += 1
                            
                            # Calculate progress
                            chunk_progress = int((completed_chunks / total_chunks) * 100)
                            total_features_so_far = len(distance_corr)
                            
                            # Count pairs in this chunk
                            pairs_in_chunk = sum(len(targets) for targets in chunk_result.values())
                            total_pairs_processed += pairs_in_chunk
                            
                            # Get first and last features from this chunk for display
                            chunk_features = list(chunk_result.keys())
                            first_feature = chunk_features[0] if chunk_features else "unknown"
                            last_feature = chunk_features[-1] if len(chunk_features) > 1 else first_feature
                            
                            # Detailed progress message
                            if len(chunk_features) > 1:
                                feature_range = f"{first_feature}...{last_feature}"
                            else:
                                feature_range = first_feature
                            
                            self._send_progress(
                                31,
                                f"Computing Distance Matrix ({chunk_progress}%)...",
                                f"🔄 Chunk {completed_chunks}/{total_chunks}: {feature_range} × {n_targets} targets = {pairs_in_chunk} pairs | Total: {total_pairs_processed}/{total_pairs} pairs ({int(total_pairs_processed/total_pairs*100)}%)"
                            )
                            
                            logger.info(f"Distance corr: Chunk {completed_chunks}/{total_chunks} completed ({chunk_progress}%) - Features: {feature_range}, Pairs: {pairs_in_chunk}")
                            
                except Exception as e:
                    logger.error(f"Nested parallel distance correlation failed: {e}")
                    return None
                    return None
            
            return distance_corr if len(distance_corr) == n_features else None
        
        except Exception as e:
            logger.error(f"Failed to initialize parallel distance correlation: {e}")
            return None
    
    def _compute_distance_correlation_sequential_smart(
        self, feature_cols: List[str], target_cols: List[str], compute_all: bool
    ) -> Dict[str, Dict[str, float]]:
        """
        Sequential distance correlation respecting configuration.
        Only computes specified pairs (feature→target, not feature→feature unless "all").
        Includes granular progress updates.
        """
        distance_corr = {}
        
        if compute_all:
            # Compute all pairs including feature→feature
            all_cols = feature_cols
            total_pairs = len(all_cols) * len(all_cols)
            pairs_completed = 0
            
            logger.info(f"🔄 DISTANCE CORR SEQUENTIAL (ALL): Computing {len(all_cols)}² = {total_pairs} pairs")
            self._send_progress(31, "Computing Distance Matrix...", 
                              f"🔄 PHASE 2a: Starting ALL-PAIRS sequential computation (0/{total_pairs} pairs)...")
            
            for i, col1 in enumerate(all_cols):
                distance_corr[col1] = {}
                
                for j, col2 in enumerate(all_cols):
                    if col1 == col2:
                        distance_corr[col1][col2] = 1.0
                    else:
                        try:
                            # ✅ FIX: Get columns from correct source (self.X or self.data)
                            col1_data = self.X[col1] if col1 in self.X.columns else self.data[col1]
                            col2_data = self.X[col2] if col2 in self.X.columns else self.data[col2]
                            df_pair = pd.concat([col1_data, col2_data], axis=1).dropna()
                            
                            if len(df_pair) < 2:
                                distance_corr[col1][col2] = 0.0
                            else:
                                dcor = global_distance_correlation(df_pair.iloc[:, 0].values, df_pair.iloc[:, 1].values)
                                distance_corr[col1][col2] = float(dcor)
                        except Exception as e:
                            logger.debug(f"Distance corr {col1}↔{col2}: {e}")
                            distance_corr[col1][col2] = 0.0
                    
                    pairs_completed += 1
                    if pairs_completed % max(1, total_pairs // 20) == 0:  # Report every 5%
                        pct = int((pairs_completed / total_pairs) * 100)
                        progress = 31 + int((pairs_completed / total_pairs) * 1)
                        self._send_progress(progress, "Computing Distance Matrix...",
                                          f"🔄 PHASE 2a: ALL-PAIRS {pairs_completed}/{total_pairs} pairs ({pct}%) - {col1}↔{col2}")
                
                # Report after each feature row completes
                logger.info(f"Distance corr: {col1} completed ({i+1}/{len(all_cols)} features)")
        else:
            # Only compute feature→target (no feature→feature)
            total_pairs = len(feature_cols) * len(target_cols)
            pairs_completed = 0
            report_interval = max(1, total_pairs // 20)  # Report every 5% progress
            
            logger.info(f"🔄 DISTANCE CORR SEQUENTIAL (TARGETS_VS_FEATURES): Computing {len(feature_cols)} × {len(target_cols)} = {total_pairs} pairs")
            self._send_progress(31, "Computing Distance Matrix...", 
                              f"🔄 PHASE 2b: Starting TARGETS_VS_FEATURES sequential computation (0/{total_pairs} pairs)...")
            
            for feat_idx, feature in enumerate(feature_cols):
                distance_corr[feature] = {}
                
                for tgt_idx, target in enumerate(target_cols):
                    pairs_completed += 1
                    
                    # Send granular progress for every pair
                    progress = 31 + int((pairs_completed / total_pairs) * 1)
                    percent = int((pairs_completed / total_pairs) * 100)
                    
                    # Detailed message2: which pair, feature index, target index
                    if pairs_completed % report_interval == 0 or pairs_completed == total_pairs:
                        self._send_progress(
                            progress, 
                            f"Computing Distance Matrix ({percent}%)...",
                            f"🔄 PHASE 2b: Pair {pairs_completed}/{total_pairs} ({percent}%) - {feature}→{target} "
                            f"[Feature {feat_idx+1}/{len(feature_cols)}, Target {tgt_idx+1}/{len(target_cols)}]"
                        )
                    
                    try:
                        # ✅ FIX: Get target data from correct source
                        # Try targets_df first, fallback to self.data for OHLCV mode
                        if target in self.targets_df.columns:
                            target_series = self.targets_df[target]
                        else:
                            target_series = self.data[target]
                        
                        # Combine feature and target columns
                        data_pair = pd.concat([self.X[feature], target_series], axis=1).dropna()
                        
                        if len(data_pair) < 2:
                            distance_corr[feature][target] = 0.0
                        else:
                            feat_vals = data_pair.iloc[:, 0].values
                            tgt_vals = data_pair.iloc[:, 1].values
                            dcor = global_distance_correlation(feat_vals, tgt_vals)
                            distance_corr[feature][target] = float(dcor)
                    except Exception as e:
                        logger.error(f"Distance corr {feature}→{target}: {e}")
                        distance_corr[feature][target] = 0.0
                
                # Report after each feature completes
                logger.info(f"Distance corr: {feature} completed ({feat_idx+1}/{len(feature_cols)} features)")
        
        self._send_progress(32, "Computing Distance Matrix...",
                          f"✅ Distance correlation complete: {total_pairs} pairs processed successfully")
        logger.info(f"✅ Distance correlation FINISHED: {total_pairs} total pairs computed")
        return distance_corr

    def _distance_correlation_pair(self, X, Y):
        """
        Compute distance correlation between two arrays.
        Helper method for parallel processing.
        
        Args:
            X: First variable array
            Y: Second variable array
        
        Returns:
            Distance correlation value between -1 and 1
        """
        from scipy.spatial.distance import pdist, squareform
        
        X, Y = np.array(X), np.array(Y)

        # Ensure same length
        if len(X) != len(Y):
            return 0.0
        
        if len(X) < 2:
            return 0.0

        # Distance matrices
        a = squareform(pdist(X.reshape(-1, 1)))
        b = squareform(pdist(Y.reshape(-1, 1)))

        # Double centering
        n = len(X)
        A = (
            a
            - a.mean(axis=0, keepdims=True)
            - a.mean(axis=1, keepdims=True)
            + a.mean()
        )
        B = (
            b
            - b.mean(axis=0, keepdims=True)
            - b.mean(axis=1, keepdims=True)
            + b.mean()
        )

        # Distance covariance
        dcov = np.sqrt(np.sum(A * B) / (n**2))

        # Distance variances
        dvarX = np.sqrt(np.sum(A * A) / (n**2))
        dvarY = np.sqrt(np.sum(B * B) / (n**2))

        # Distance correlation
        dcor = dcov / np.sqrt(dvarX * dvarY) if dvarX * dvarY > 0 else 0.0
        return dcor

    def _calculate_permutation_importance(self, model, X, y, n_repeats: int = 10) -> Dict[str, float]:
        """Calculate permutation importance with proper data isolation."""
        baseline_score = model.score(X, y)
        importance_scores = {}

        for col in X.columns:
            scores = []

            for _ in range(n_repeats):
                # Create fresh deep copy for each repeat to prevent data corruption
                # (Soft copy could carry over state from previous permutations)
                X_permuted = X.copy(deep=True)
                X_permuted[col] = np.random.permutation(X_permuted[col].values)
                score = model.score(X_permuted, y)
                # Importance = decrease in model performance when feature is randomized
                scores.append(baseline_score - score)

            importance_scores[col] = float(np.mean(scores))

        return importance_scores

    def _calculate_mahalanobis_distance(self, X: pd.DataFrame) -> np.ndarray:
        """Calculate Mahalanobis distance for multivariate outlier detection (vectorized)."""
        cov_matrix = X.cov()
        inv_cov_matrix = np.linalg.pinv(cov_matrix)
        mean_vector = X.mean(axis=0)

        # Vectorized computation (50-100x faster than row iteration)
        diffs = X.values - mean_vector.values  # Shape: (n_samples, n_features)
        
        # Compute (diffs @ inv_cov @ diffs^T) for each row efficiently
        mahal_squared = (diffs @ inv_cov_matrix * diffs).sum(axis=1)
        distances = np.sqrt(mahal_squared)

        return distances

    def _calculate_pca_feature_correlation(self, pca, X_scaled) -> Dict[str, Any]:
        """Calculate correlation between original features and principal components."""
        component_corr = {}
        for i in range(len(pca.components_)):
            component_corr[f"PC{i+1}"] = {}
            for j, col in enumerate(X_scaled.columns):
                corr = np.corrcoef(X_scaled.iloc[:, j], pca.transform(X_scaled)[:, i])[
                    0, 1
                ]
                component_corr[f"PC{i+1}"][col] = float(corr)
        return component_corr

    def _winsorized_mean(self, data, proportiontocut):
        """Calculate winsorized mean."""
        data_sorted = np.sort(data)
        n = len(data)
        cut = int(proportiontocut * n)
        return np.mean(data_sorted[cut : n - cut])

    def _precompute_paired_data_matrices(
        self, 
        feature_cols: List[str], 
        target_cols: List[str],
        data_source: str = "X"
    ) -> Dict[Tuple[str, str], pd.DataFrame]:
        """Pre-compute all paired data matrices for correlation analysis (PHASE 4: TASK 11).
        
        Instead of computing pd.concat([col1, col2], axis=1).dropna() repeatedly in loops,
        this method pre-computes all pairs once and caches them.
        
        Improvement: 20-30% faster for correlation analysis with many pairs.
        
        Args:
            feature_cols: List of feature column names
            target_cols: List of target column names  
            data_source: 'X' for features, 'targets' for targets, 'data' for raw data
            
        Returns:
            Dict mapping (feature, target) tuples to paired DataFrames (NaN-dropped)
        """
        paired_data = {}
        
        # Select data source
        if data_source == "X":
            features_df = self.X
        elif data_source == "targets":
            features_df = self.targets_df
        else:
            features_df = self.data
        
        for feature in feature_cols:
            for target in target_cols:
                pair_key = (feature, target)
                
                try:
                    # Compute pair once (PHASE 4: Cache this)
                    df_pair = pd.concat([features_df[feature], self.targets_df[target]], axis=1).dropna()
                    
                    # Only store if we have sufficient data
                    if len(df_pair) > 1:
                        paired_data[pair_key] = df_pair
                except (KeyError, ValueError):
                    # Skip if columns don't exist or can't be paired
                    logger.debug(f"Could not create pair ({feature}, {target})")
                    continue
        
        logger.debug(f"Pre-computed {len(paired_data)} paired data matrices (PHASE 4: Task 11)")
        return paired_data

    def _find_significant_correlations(self, corr_matrix, pvalue_matrix, alpha=0.05):
        """
        Find statistically significant correlation pairs with Bonferroni correction.
        
        Applies Bonferroni correction to account for multiple comparisons and reduce
        false positive rate.
        """
        significant_pairs = []
        outer_keys = list(corr_matrix.keys())
        if not outer_keys:
            return []

        # Check if the matrix is square or rectangular
        sample_outer_key = outer_keys[0]
        inner_keys = list(corr_matrix[sample_outer_key].keys())
        
        is_square = set(outer_keys) == set(inner_keys)
        
        # Calculate number of comparisons for Bonferroni correction
        if is_square:
            n_features = len(outer_keys)
            n_comparisons = (n_features * (n_features - 1)) // 2  # Number of unique pairs
        else:
            n_comparisons = len(outer_keys) * len(inner_keys)  # All pairs in rectangular matrix
        
        # Apply Bonferroni correction
        corrected_alpha = alpha / n_comparisons if n_comparisons > 0 else alpha
        
        logger.info(f"Applying Bonferroni correction: {n_comparisons} comparisons (IsSquare={is_square}), "
                   f"corrected alpha = {corrected_alpha:.6f} (original alpha = {alpha})")

        if is_square:
            for i, f1 in enumerate(outer_keys):
                for j, f2 in enumerate(outer_keys[i + 1 :], i + 1):
                    if f2 in pvalue_matrix.get(f1, {}) and pvalue_matrix[f1][f2] < corrected_alpha:
                        significant_pairs.append(
                            {
                                "feature1": f1,
                                "feature2": f2,
                                "correlation": corr_matrix[f1][f2],
                                "p_value": pvalue_matrix[f1][f2],
                                "corrected_alpha": corrected_alpha,
                                "n_comparisons": n_comparisons,
                            }
                        )
        else:
            # Rectangular mode
            for f_outer in outer_keys:
                for f_inner in inner_keys:
                    if f_inner in pvalue_matrix.get(f_outer, {}) and pvalue_matrix[f_outer][f_inner] < corrected_alpha:
                        significant_pairs.append(
                            {
                                "feature1": f_outer,
                                "feature2": f_inner,
                                "correlation": corr_matrix[f_outer][f_inner],
                                "p_value": pvalue_matrix[f_outer][f_inner],
                                "corrected_alpha": corrected_alpha,
                                "n_comparisons": n_comparisons,
                            }
                        )

        return significant_pairs

    def _find_strong_correlations(self, corr_matrix, threshold=0.7):
        """Find strongly correlated feature pairs."""
        strong_pairs = []
        outer_keys = list(corr_matrix.keys())
        if not outer_keys:
            return []

        # Check if square or rectangular
        sample_outer_key = outer_keys[0]
        inner_keys = list(corr_matrix[sample_outer_key].keys())
        
        is_square = set(outer_keys) == set(inner_keys)

        if is_square:
            for i, f1 in enumerate(outer_keys):
                for j, f2 in enumerate(outer_keys[i + 1 :], i + 1):
                    # Use .get() to avoid KeyError
                    if f2 in corr_matrix.get(f1, {}):
                        corr = abs(corr_matrix[f1][f2])
                        if corr > threshold:
                            strong_pairs.append(
                                {
                                    "feature1": f1,
                                    "feature2": f2,
                                    "correlation": corr_matrix[f1][f2],
                                    "absolute_correlation": corr,
                                    "strength": "very_strong" if corr > 0.9 else "strong",
                                }
                            )
        else:
            # Rectangular mode
            for f_outer in outer_keys:
                for f_inner in inner_keys:
                    if f_inner in corr_matrix.get(f_outer, {}):
                        corr = abs(corr_matrix[f_outer][f_inner])
                        if corr > threshold:
                            strong_pairs.append(
                                {
                                    "feature1": f_outer,
                                    "feature2": f_inner,
                                    "correlation": corr_matrix[f_outer][f_inner],
                                    "absolute_correlation": corr,
                                    "strength": "very_strong" if corr > 0.9 else "strong",
                                }
                            )

        return strong_pairs

    def _interpret_vif(self, vif):
        """Interpret VIF value."""
        if vif > 10:
            return "Severe multicollinearity"
        elif vif > 5:
            return "Moderate multicollinearity"
        else:
            return "No significant multicollinearity"

    def _interpret_condition_number(self, condition_number):
        """Interpret condition number."""
        if condition_number > 30:
            return "Severe multicollinearity"
        elif condition_number > 15:
            return "Moderate multicollinearity"
        else:
            return "No significant multicollinearity"

    def _interpret_cohens_d(self, d):
        """Interpret Cohen's d effect size."""
        if abs(d) < 0.2:
            return "Very small effect"
        elif abs(d) < 0.5:
            return "Small effect"
        elif abs(d) < 0.8:
            return "Medium effect"
        else:
            return "Large effect"

    def generate_executive_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an executive summary of the analysis."""
        summary = {
            "key_findings": [],
            "recommendations": [],
            "data_quality_issues": [],
            "potential_risks": [],
        }

        # Add summary logic based on analysis results
        # This would analyze the comprehensive results and extract key insights

        return summary

    def analyze_distributions(self) -> Dict[str, Any]:
        """Comprehensive distribution analysis."""
        dist_analysis = {}

        for col in self.numeric_columns:
            data_col = self.data[col].dropna()
            if len(data_col) == 0:
                dist_analysis[col] = {"error": "No data"}
                continue

            # Calculate entropy with error handling for low-variance data
            try:
                # Try automatic binning first
                hist_counts, _ = np.histogram(data_col, bins="auto")
                # Normalize counts to probabilities (scipy.stats.entropy expects probabilities, not counts)
                hist_probs = (hist_counts + 1e-10) / np.sum(hist_counts + 1e-10)
                entropy_val = float(stats.entropy(hist_probs))
            except ValueError as e:
                # If automatic binning fails, use fixed bins or skip
                if "Too many bins" in str(e):
                    try:
                        # Use a smaller fixed number of bins
                        n_bins = min(10, max(2, len(data_col.unique())))
                        hist_counts, _ = np.histogram(data_col, bins=n_bins)
                        # Normalize counts to probabilities (scipy.stats.entropy expects probabilities, not counts)
                        hist_probs = (hist_counts + 1e-10) / np.sum(hist_counts + 1e-10)
                        entropy_val = float(stats.entropy(hist_probs))
                    except:
                        # If still fails, set to None
                        entropy_val = None
                else:
                    entropy_val = None
            
            dist_analysis[col] = {
                "skewness": float(stats.skew(data_col)),
                "kurtosis": float(stats.kurtosis(data_col, fisher=False)),
                "excess_kurtosis": float(stats.kurtosis(data_col, fisher=True)),
                "skew_interpretation": self._interpret_skewness(stats.skew(data_col)),
                "kurtosis_interpretation": self._interpret_kurtosis(
                    stats.kurtosis(data_col, fisher=False)
                ),
                "is_symmetric": bool(abs(stats.skew(data_col)) < 0.5),
                "percentiles": {
                    f"p{p}": float(data_col.quantile(p / 100))
                    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
                },
                "moment_3": float(stats.moment(data_col, moment=3)),
                "moment_4": float(stats.moment(data_col, moment=4)),
                "entropy": entropy_val,
                "is_multimodal": self._detect_multimodality(data_col),
            }

        return dist_analysis

    def calculate_partial_correlations(self) -> Dict[str, Any]:
        """Calculate partial correlations."""
        partial_corrs = {}

        for i, col1 in enumerate(self.feature_columns):
            for j, col2 in enumerate(self.feature_columns[i + 1 :], i + 1):
                control_vars = [
                    c
                    for c in self.feature_columns
                    if c not in [col1, col2] and c in self.numeric_columns
                ]
                if (
                    len(control_vars) > 0
                    and col1 in self.numeric_columns
                    and col2 in self.numeric_columns
                ):
                    partial_corr = self._partial_correlation(col1, col2, control_vars)
                    partial_corrs[f"{col1}_vs_{col2}"] = {
                        "partial_correlation": float(partial_corr[0]),
                        "p_value": float(partial_corr[1]),
                        "controlled_variables": control_vars,
                    }

        return partial_corrs

    def calculate_mutual_information_advanced(self) -> Dict[str, Any]:
        """Advanced mutual information analysis.
        OPTIMIZED: Only computes MI with targets (skip O(n²) pairwise feature combinations).
        For large feature sets (>50 features), skip pairwise to focus on target MI only.
        """
        mi_results = {"with_target": {} if self.y is not None else None}

        # OPTIMIZATION: Skip pairwise feature MI for high-dimensional data
        # Feature-feature MI is O(n²) expensive and rarely used for analysis
        skip_pairwise = len(self.numeric_columns) > 50
        
        if not skip_pairwise:
            # Pairwise for numeric (only if < 50 features)
            mi_results["pairwise"] = {}
            total_pairs = len(self.numeric_columns) * (len(self.numeric_columns) - 1)
            current_pair = 0
            for col1 in self.numeric_columns:
                mi_results["pairwise"][col1] = {}
                for col2 in self.numeric_columns:
                    if col1 != col2:
                        current_pair += 1
                        if current_pair % 5 == 0 or current_pair == total_pairs:
                            self._send_progress(90, "Analyzing Information...", f"Mutual Info: {col1} vs {col2} ({current_pair}/{total_pairs})")
                        
                        score = mutual_info_regression(
                            self.data[[col1]].fillna(0),
                            self.data[col2].fillna(0).values.ravel(),
                            random_state=42,
                        )[0]
                        mi_results["pairwise"][col1][col2] = float(score)
        else:
            logger.info(f"Skipping pairwise MI: {len(self.numeric_columns)} features exceed threshold (>50). Computing target MI only.")
            mi_results["pairwise"] = None

        # With target (Always compute - this is the valuable metric)
        if self.y is not None:
            self._send_progress(93, "Analyzing Information...", "Computing MI with Target...")
            scores = mutual_info_regression(self.X.fillna(0), self.y, random_state=42)
            mi_results["with_target"] = {
                col: float(s) for col, s in zip(self.X.columns, scores)
            }
        else:
            logger.info("No target available - skipping MI with target computation")

        return mi_results

    def perform_statistical_tests_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive statistical testing."""
        test_results = {"pairwise": {}, "group": {}}

        numeric_features = self.X.columns
        # Pairwise tests for numeric
        for i, col1 in enumerate(numeric_features):
            for col2 in numeric_features[i + 1 :]:
                d1, d2 = self.data[col1].dropna(), self.data[col2].dropna()
                if len(d1) < 2 or len(d2) < 2:
                    continue

                key = f"{col1}_vs_{col2}"
                test_results["pairwise"][key] = {}

                t_stat, t_p = ttest_ind(d1, d2)
                test_results["pairwise"][key]["t_test"] = {
                    "stat": float(t_stat),
                    "p": float(t_p),
                }

                mw_stat, mw_p = mannwhitneyu(d1, d2)
                test_results["pairwise"][key]["mann_whitney"] = {
                    "stat": float(mw_stat),
                    "p": float(mw_p),
                }

        if len(numeric_features) >= 2:
            groups = [self.data[col].dropna() for col in numeric_features]

            lev_stat, lev_p = levene(*groups)
            test_results["group"]["levene"] = {
                "stat": float(lev_stat),
                "p": float(lev_p),
            }

            bart_stat, bart_p = bartlett(*groups)
            test_results["group"]["bartlett"] = {
                "stat": float(bart_stat),
                "p": float(bart_p),
            }

            f_stat, f_p = f_oneway(*groups)
            test_results["group"]["anova"] = {"stat": float(f_stat), "p": float(f_p)}

            krus_stat, krus_p = kruskal(*groups)
            test_results["group"]["kruskal"] = {
                "stat": float(krus_stat),
                "p": float(krus_p),
            }

        return test_results

    def calculate_confidence_intervals_comprehensive(
        self, confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """Comprehensive confidence intervals."""
        ci_results = {}

        for col in self.numeric_columns:
            data_col = self.data[col].dropna()
            if len(data_col) < 2:
                continue

            mean_ci = stats.t.interval(
                confidence_level,
                len(data_col) - 1,
                loc=data_col.mean(),
                scale=data_col.sem(),
            )
            median_ci = self._bootstrap_ci(data_col, np.median, confidence_level)
            ci_results[col] = {
                "mean_ci": {"lower": float(mean_ci[0]), "upper": float(mean_ci[1])},
                "median_ci": {
                    "lower": float(median_ci[0]),
                    "upper": float(median_ci[1]),
                },
            }

        return ci_results

    def calculate_effect_sizes_advanced(self) -> Dict[str, Any]:
        """Advanced effect size calculations."""
        effect_sizes = {"pairwise": {}, "group": {}}

        numeric_features = self.X.columns
        for i, col1 in enumerate(numeric_features):
            for col2 in numeric_features[i + 1 :]:
                d1, d2 = self.data[col1].dropna(), self.data[col2].dropna()
                if len(d1) < 2 or len(d2) < 2:
                    continue
                mean_diff = d1.mean() - d2.mean()
                pooled_sd = np.sqrt((d1.var() + d2.var()) / 2)
                cohen_d = mean_diff / pooled_sd if pooled_sd != 0 else 0
                key = f"{col1}_vs_{col2}"
                effect_sizes["pairwise"][key] = {
                    "cohens_d": float(cohen_d),
                    "interpretation": self._interpret_cohens_d(abs(cohen_d)),
                }

        if self.y is not None:
            r_sq = {}
            for col in self.feature_columns:
                if col in self.numeric_columns:
                    # Align data
                    df_pair = pd.concat([self.data[col], self.y], axis=1).dropna()
                    if len(df_pair) < 2:
                        continue
                    corr, _ = pearsonr(df_pair[col], df_pair[self.target_column])
                    r_sq[col] = float(corr**2)
            effect_sizes["with_target"] = r_sq

        return effect_sizes

    def analyze_variance_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive variance analysis."""
        variance_analysis = {"per_feature": {}, "homogeneity": {}}

        for col in self.numeric_columns:
            data_col = self.data[col].dropna()
            if len(data_col) == 0:
                continue
            variance_analysis["per_feature"][col] = {
                "variance": float(data_col.var()),
                "std_dev": float(data_col.std()),
                "coeff_var": float(
                    np.inf if data_col.mean() == 0 and data_col.std() > 0 
                    else (data_col.std() / data_col.mean() if data_col.mean() != 0 else 0)
                ),
            }

        if len(self.X.columns) >= 2:
            groups = [self.data[col].dropna() for col in self.X.columns]
            lev_stat, lev_p = levene(*groups)
            variance_analysis["homogeneity"]["levene_test"] = {
                "statistic": lev_stat,
                "p_value": lev_p,
            }

        return variance_analysis

    def analyze_skewness_kurtosis_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive skewness and kurtosis analysis."""
        sk_analysis = {}
        for col in self.numeric_columns:
            data_col = self.data[col].dropna()
            if len(data_col) == 0:
                continue
            skew = stats.skew(data_col)
            kurt = stats.kurtosis(data_col, fisher=True)
            sk_analysis[col] = {
                "skewness": skew,
                "kurtosis": kurt,
                "skew_interpretation": self._interpret_skewness(skew),
                "kurtosis_interpretation": self._interpret_kurtosis(
                    kurt + 3
                ),  # convert to non-excess
            }
        return sk_analysis

    def analyze_feature_interactions_advanced(self) -> Dict[str, Any]:
        """Advanced feature interaction analysis."""
        interactions = {}

        if self.y is not None:
            from sklearn.preprocessing import PolynomialFeatures

            poly = PolynomialFeatures(
                degree=2, interaction_only=True, include_bias=False
            )
            X_poly = poly.fit_transform(self.X.fillna(0))
            poly_features = poly.get_feature_names_out(self.X.columns)

            model = LinearRegression()
            model.fit(X_poly, self.y)
            interactions["poly_coefficients"] = {
                feat: float(coef) for feat, coef in zip(poly_features, model.coef_)
            }

            interaction_indices = [
                i for i, feat in enumerate(poly_features) if " " in feat
            ]
            top_inter = sorted(
                [(poly_features[i], abs(model.coef_[i])) for i in interaction_indices],
                key=lambda x: x[1],
                reverse=True,
            )[:10]
            interactions["top_interactions"] = [
                {"interaction": f, "abs_coef": v} for f, v in top_inter
            ]

        return interactions

    def test_heteroscedasticity_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive heteroscedasticity testing."""
        het_results = {}

        if self.y is not None:
            for col in self.feature_columns:
                if col in self.numeric_columns:
                    X_subset = self.data[[col]].dropna()
                    y_subset = self.y.loc[X_subset.index]
                    if len(X_subset) < 2:
                        continue

                    model = LinearRegression().fit(X_subset, y_subset)
                    residuals = y_subset - model.predict(X_subset)

                    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X_subset)
                    white_stat, white_p, _, _ = het_white(residuals, X_subset)

                    het_results[f"{col}_vs_target"] = {
                        "breusch_pagan": {"stat": float(bp_stat), "p": float(bp_p)},
                        "white": {"stat": float(white_stat), "p": float(white_p)},
                    }

        return het_results

    def assess_data_quality_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive data quality assessment."""
        quality = {
            "duplicates": int(self.data.duplicated().sum()),
            "duplicate_rows": self.data[self.data.duplicated()].index.tolist(),
            "missing_values": self.data.isna().sum().to_dict(),
            "constant_columns": [
                col for col in self.data.columns if self.data[col].nunique() <= 1
            ],
            "high_missing_columns": [
                col
                for col, miss in self.data.isna().sum().items()
                if miss / len(self.data) > 0.5
            ],
            "data_types": {col: str(dtype) for col, dtype in self.data.dtypes.items()},
        }

        return quality

    def analyze_non_linear_relationships(self) -> Dict[str, Any]:
        """Non-linear relationship analysis."""
        # Reusing feature interaction as a proxy for non-linear relationships
        return self.analyze_feature_interactions_advanced()

    def analyze_power_transformations(self) -> Dict[str, Any]:
        """Power transformation analysis."""
        suggestions = {}
        for col in self.numeric_columns:
            skew = stats.skew(self.data[col].dropna())
            suggestions[col] = []
            if abs(skew) > 1:
                suggestions[col].append(
                    "Consider log or Box-Cox transformation for skewness"
                )
        return {"suggestions": suggestions}

    def calculate_robust_statistics(self) -> Dict[str, Any]:
        """Robust statistics calculation."""
        robust_stats = {}
        for col in self.numeric_columns:
            data_col = self.data[col].dropna()
            if len(data_col) == 0:
                continue
            robust_stats[col] = {
                "trimmed_mean_10": float(stats.trim_mean(data_col, 0.1)),
                "median_absolute_deviation": float(
                    np.median(np.abs(data_col - data_col.median()))
                ),
            }
        return robust_stats

    def analyze_target_relationship_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive target relationship analysis."""
        target_rel = {}
        if self.y is not None:
            for col in self.feature_columns:
                if col in self.numeric_columns:
                    df_pair = pd.concat([self.data[col], self.y], axis=1).dropna()
                    if len(df_pair) < 2:
                        continue
                    corr, p = pearsonr(df_pair[col], df_pair[self.target_column])
                    target_rel[col] = {"pearson_corr": float(corr), "p_value": float(p)}
        return target_rel

    def assess_predictive_power(self) -> Dict[str, Any]:
        """Predictive power assessment."""
        model_analysis = {}
        if self.y is not None:
            X = self.X.fillna(0)
            y = self.y
            model = LinearRegression()
            cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
            model_analysis["cv_scores_r2"] = [float(s) for s in cv_scores]
            model_analysis["mean_cv_score_r2"] = float(np.mean(cv_scores))
        return model_analysis

    def perform_time_series_analysis(self) -> Dict[str, Any]:
        """Time series analysis."""
        ts_analysis = {}
        if self.datetime_column and self.datetime_column in self.data.columns:
            for col in self.numeric_columns:
                series = self.data.set_index(self.datetime_column)[col]

                try:
                    adf_stat, adf_p, _, _, adf_cv, _ = adfuller(series.dropna())
                    kpss_stat, kpss_p, _, _ = kpss(series.dropna())
                    ts_analysis[col] = {
                        "adf_test": {
                            "stat": float(adf_stat),
                            "p": float(adf_p),
                            "stationary": adf_p < 0.05,
                        },
                        "kpss_test": {
                            "stat": float(kpss_stat),
                            "p": float(kpss_p),
                            "stationary": kpss_p > 0.05,
                        },
                    }
                except ValueError as e:
                    ts_analysis[col] = {"error": f"Statistical test failed: {str(e)}"}
                except Exception as e:
                    ts_analysis[col] = {"error": f"Analysis error: {str(e)}"}
        else:
            ts_analysis = {"note": "No datetime column specified."}

        return ts_analysis

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Smart data cleaning to handle Infinite and NaN values appropriately.
        - 'Diff' / 'Distance' columns: Inf -> High Value (User Requirement: 0 implies closeness, so Inf needs to be 'far')
        - Other columns: Inf -> NaN -> ffill/bfill/0
        """
        try:
            # Identify columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            # 1. Handle Infinite Values
            for col in numeric_cols:
                # Check if column has infinite values
                if np.isinf(df[col]).any():
                    is_diff_col = any(keyword in col for keyword in ['Diff', 'Distance', 'Divergence', 'Gap', 'Squeeze'])
                    
                    if is_diff_col:
                        # For diff columns, Infinite means "Very Far" or "Extreme Divergence"
                        # We replace Inf with a robust maximum value (e.g. 5x the max finite absolute value)
                        finite_vals = df[col][np.isfinite(df[col])]
                        if not finite_vals.empty:
                            robust_max = np.nanmax(np.abs(finite_vals)) * 5.0
                            if robust_max == 0: robust_max = 9999.0 # Fallback if all finite are 0
                        else:
                            robust_max = 9999.0 # Fallback if ALL are Inf
                            
                        # Replace +Inf with +Max, -Inf with -Max
                        df[col] = df[col].replace([np.inf], robust_max)
                        df[col] = df[col].replace([-np.inf], -robust_max)
                    else:
                        # For standard columns, Inf is likely an error -> treat as NaN
                        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

            # 2. Handle NaN Values
            # First, standard forward fill for time series continuity
            df = df.ffill()
            # Then backward fill for initial warmup periods
            df = df.bfill()
            # Finally fill any remaining (e.g. empty columns) with 0
            df = df.fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Data cleaning failed: {e}")
            # Fallback: Just simple fillna to ensure we don't crash, even if not 'smart'
            return df.replace([np.inf, -np.inf], np.nan).fillna(0)
            
    # Helper methods to be added
    def _to_json_serializable(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._to_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._to_json_serializable(i) for i in obj]
        if isinstance(obj, (np.ndarray, pd.Series)):
            return self._to_json_serializable(obj.tolist())
        if isinstance(obj, pd.DataFrame):
            return self._to_json_serializable(obj.to_dict(orient="records"))
        if isinstance(obj, (np.float64, float)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, (np.int64, np.int32, int)):
            return int(obj)
        if isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        if obj is None or isinstance(obj, str):
            return obj
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()

        return str(obj)

    def _detect_multimodality(self, data: pd.Series) -> bool:
        try:
            from scipy.signal import find_peaks

            kde = stats.gaussian_kde(data)
            x = np.linspace(data.min(), data.max(), 1000)
            y = kde(x)
            peaks, _ = find_peaks(y)
            return len(peaks) > 1
        except:
            return False

    def _partial_correlation(
        self, x: str, y: str, controls: List[str]
    ) -> Tuple[float, float]:
        try:
            df = self.data[[x, y] + controls].dropna()
            if len(df) < len(controls) + 3:
                return 0.0, 1.0
            model_y = LinearRegression().fit(df[controls], df[y])
            model_x = LinearRegression().fit(df[controls], df[x])
            res_y = df[y] - model_y.predict(df[controls])
            res_x = df[x] - model_x.predict(df[controls])
            
            # Check for constant residuals which cause pearsonr to return NaN
            if np.std(res_x) < 1e-10 or np.std(res_y) < 1e-10:
                return 0.0, 1.0
                
            corr, pval = pearsonr(res_x, res_y)
            return np.nan_to_num(corr, nan=0.0), np.nan_to_num(pval, nan=1.0)
        except Exception:
            return 0.0, 1.0

    def _bootstrap_ci(
        self, data: pd.Series, func, confidence_level, n_boot=1000
    ) -> Tuple[float, float]:
        if len(data) == 0:
            return (0.0, 0.0)
        boot_samples = np.random.choice(data, size=(n_boot, len(data)), replace=True)
        boot_stats = np.apply_along_axis(func, 1, boot_samples)
        return np.percentile(
            boot_stats,
            [(1 - confidence_level) / 2 * 100, (1 + (1 - confidence_level) / 2) * 100],
        )

    def _interpret_skewness(self, skew: float) -> str:
        abs_skew = abs(skew)
        if abs_skew < 0.5:
            return "Approximately symmetric"
        elif abs_skew < 1:
            return "Moderately skewed"
        else:
            return "Highly skewed"

    def _interpret_kurtosis(self, kurt: float) -> str:
        if kurt < 2.5:  # Looser bounds
            return "Platykurtic (light tails)"
        elif kurt > 3.5:
            return "Leptokurtic (heavy tails)"
        else:
            return "Mesokurtic (normal tails)"
