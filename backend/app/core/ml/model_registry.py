"""
Model Registry System
Central catalog of all available ML model architectures with metadata and parameters.
"""

from typing import Dict, List, Optional, Any, Callable
from pydantic import BaseModel, Field
from enum import Enum
from app.core.ml import default_models
import inspect
import numpy as np


class ParameterType(str, Enum):
    """Parameter types for model configuration."""

    INTEGER = "int"
    FLOAT = "float"
    BOOLEAN = "bool"
    SELECT = "select"
    LIST = "list"


class ModelParameter(BaseModel):
    """Definition of a model parameter."""

    name: str = Field(..., description="Parameter name")
    type: ParameterType = Field(..., description="Parameter type")
    default: Any = Field(..., description="Default value")
    min: Optional[float] = Field(None, description="Minimum value (for numeric types)")
    max: Optional[float] = Field(None, description="Maximum value (for numeric types)")
    options: Optional[List[str]] = Field(
        None, description="Available options (for select type)"
    )
    description: str = Field(..., description="Parameter description")
    required: bool = Field(True, description="Whether parameter is required")
    is_layer_configurable: bool = Field(
        False, description="Whether this parameter can be configured per-layer"
    )
    category: str = Field(
        "global", description="Parameter category: 'global', 'layer', or 'optimization'"
    )


class ModelCategory(str, Enum):
    """Model architecture categories."""

    LSTM = "lstm"
    CNN = "cnn"
    TCN = "tcn"
    TRANSFORMER = "transformer"
    HYBRID = "hybrid"
    CNN_3D = "3d_cnn"
    DNN = "dnn"
    PROPRIETARY = "proprietary"  # Private/validated architectures from Backend/Backend/data/


class InputRequirement(BaseModel):
    """Input data requirements for a model."""

    shape_type: str = Field(..., description="Shape type: '2D', '3D', '4D'")
    min_timesteps: Optional[int] = Field(None, description="Minimum timesteps required")
    min_features: Optional[int] = Field(None, description="Minimum features required")
    description: str = Field(..., description="Input requirements description")


class ModelArchitecture(BaseModel):
    """Complete model architecture definition."""

    id: str = Field(..., description="Unique model identifier")
    name: str = Field(..., description="Display name")
    category: ModelCategory = Field(..., description="Model category")
    description: str = Field(..., description="Model description")
    parameters: List[ModelParameter] = Field(..., description="Configurable parameters")
    builder_function: str = Field(..., description="Function name in default_models.py")
    input_requirements: InputRequirement = Field(
        ..., description="Input data requirements"
    )
    tags: List[str] = Field(default_factory=list, description="Search tags")
    complexity: str = Field(
        ..., description="Model complexity: 'low', 'medium', 'high'"
    )
    recommended_for: List[str] = Field(
        default_factory=list, description="Recommended use cases"
    )
    # ── I/O contract (for evolving multi-input / multi-output architectures) ──
    input_mode: str = Field(
        "single",
        description="'single' = one (batch, timesteps, features) input tensor; "
                    "'multi' = multiple named input branches (future)"
    )
    output_mode: str = Field(
        "single",
        description="'single' = one regression/classification head; "
                    "'multi' = multiple named output heads (future)"
    )
    supports_target_selection: bool = Field(
        True,
        description="Whether the frontend should offer the target-column picker. "
                    "Set False for autoencoder-style models that use x=y."
    )


class _SingletonMeta(type):
    """
    Metaclass that enforces singleton semantics at the __call__ level.
    When ModelRegistry() is invoked after the first construction, __init__
    is not called again — the already-initialised instance is returned
    directly.  This prevents duplicate _register_default_models() /
    discover_proprietary_models() calls for any caller that still uses
    ModelRegistry() directly instead of get_registry().

    The singleton instance is stored in the module-level ``_registry``
    variable so that resetting ``_registry = None`` (as done in tests via
    ``reg_module._registry = None``) also clears the cached instance here,
    giving tests proper isolation without needing extra teardown.
    """

    def __call__(cls, *args, **kwargs):
        # Delegate to the module-level _registry so the two caches stay in sync.
        # Importing at call-time avoids a circular-reference at class definition.
        import sys
        _mod = sys.modules[cls.__module__]
        if getattr(_mod, "_registry", None) is None:
            instance = super().__call__(*args, **kwargs)
            _mod._registry = instance
        return _mod._registry


class ModelRegistry(metaclass=_SingletonMeta):
    """
    Central registry of all available model architectures.
    Provides catalog, search, and retrieval functionality.

    Enforced singleton via _SingletonMeta: ModelRegistry() always returns the
    same instance; __init__ is called only once.  Prefer get_registry() defined
    below, but direct ModelRegistry() calls are safe and idempotent.
    """

    def __init__(self):
        self.models: Dict[str, ModelArchitecture] = {}
        self._register_default_models()

    def register_model(self, architecture: ModelArchitecture):
        """Register a new model architecture."""
        self.models[architecture.id] = architecture

    def get_model(self, model_id: str) -> Optional[ModelArchitecture]:
        """Get a specific model architecture by ID."""
        return self.models.get(model_id)

    def get_builder(self, model_id: str):
        """
        Compatibility adapter for callers expecting class-based builders.
        Returns a lightweight builder class with `build()`.
        """
        architecture = self.get_model(model_id)
        if architecture is None:
            return None
        build_fn = getattr(default_models, architecture.builder_function, None)
        if build_fn is None:
            return None

        class _FunctionBuilder:
            def __init__(self, config: Dict[str, Any]):
                self.config = config or {}

            def build(self):
                cfg = dict(self.config)
                input_shape = cfg.pop("input_shape", None)
                n_predictions = cfg.pop("n_predictions", 1)
                
                # ── Fields to strip for standard (non-proprietary) builders ──────────
                # Proprietary builders (category=PROPRIETARY) receive extra fields
                # such as target_cols, dataset_name, and ml_preparation_ref so
                # their adapters can perform required-targets validation and
                # locate the feature_index_map. Standard builders don't need them.
                _is_proprietary = (
                    architecture is not None
                    and architecture.category == ModelCategory.PROPRIETARY
                )

                cfg.pop("type", None)             # Model type — used for builder selection only
                cfg.pop("task_id", None)             # Task ID (metadata only)
                cfg.pop("prediction_length", None)   # Prediction length (sets n_predictions, not passed to builder)
                # Phase 19: identity fields — user-facing metadata, not model hyperparameters
                cfg.pop("model_name", None)
                cfg.pop("description", None)
                cfg.pop("tags", None)
                cfg.pop("is_public", None)
                cfg.pop("version", None)
                # Provenance fields — stripped for standard builders, KEPT for proprietary
                cfg.pop("feature_cols", None)        # Always strip (not useful to builders)
                cfg.pop("feature_hash", None)
                cfg.pop("step_configs", None)
                cfg.pop("model_id", None)
                cfg.pop("architecture_type", None)
                cfg.pop("loss", None)
                cfg.pop("metrics", None)
                cfg.pop("parameters", None)

                if not _is_proprietary:
                    # Standard builders: strip dataset routing fields
                    cfg.pop("ml_preparation_ref", None)
                    cfg.pop("dataset_id", None)
                    cfg.pop("dataset_name", None)
                    cfg.pop("target_cols", None)
                    cfg.pop("selected_targets", None)
                # Proprietary builders: ml_preparation_ref, dataset_id, dataset_name,
                # target_cols, selected_targets are intentionally PRESERVED so the
                # adapter can validate required targets and locate feature_index_map.
                
                if input_shape is None:
                    raise ValueError("input_shape is required for model build")
                
                # ✅ ULTIMATE FIX: Filter kwargs based on the actual function signature
                # This handles the unified ModelBuildConfig which contains fields for ALL models.
                # Only pass what THIS specific builder function (e.g. build_cnn_model) accepts.
                sig = inspect.signature(build_fn)
                valid_params = sig.parameters.keys()
                
                # Prepare final kwargs
                kwargs = {k: v for k, v in cfg.items() if k in valid_params}
                
                # Always pass core required params if they are in signature
                if "input_shape" in valid_params:
                    kwargs["input_shape"] = input_shape
                if "n_predictions" in valid_params:
                    kwargs["n_predictions"] = n_predictions
                
                return build_fn(**kwargs)

        return _FunctionBuilder

    def list_models(
        self,
        category: Optional[ModelCategory] = None,
        complexity: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[ModelArchitecture]:
        """
        List models with optional filtering.

        Args:
            category: Filter by category
            complexity: Filter by complexity level
            tags: Filter by tags (any match)
        """
        models = list(self.models.values())

        if category:
            models = [m for m in models if m.category == category]

        if complexity:
            models = [m for m in models if m.complexity == complexity]

        if tags:
            models = [m for m in models if any(tag in m.tags for tag in tags)]

        return models

    def search_models(self, query: str) -> List[ModelArchitecture]:
        """Search models by name, description, or tags."""
        query_lower = query.lower()
        results = []

        for model in self.models.values():
            if (
                query_lower in model.name.lower()
                or query_lower in model.description.lower()
                or any(query_lower in tag.lower() for tag in model.tags)
            ):
                results.append(model)

        return results

    def get_categories(self) -> List[Dict[str, Any]]:
        """Get all categories with model counts."""
        categories = {}
        for model in self.models.values():
            cat = model.category.value
            if cat not in categories:
                categories[cat] = {"category": cat, "count": 0, "models": []}
            categories[cat]["count"] += 1
            categories[cat]["models"].append(model.id)

        return list(categories.values())

    def _register_default_models(self):
        """Register all default model architectures with complete parameter definitions."""

        # ====================================================================
        # 1. LSTM ARCHITECTURE
        # ====================================================================

        self.register_model(
            ModelArchitecture(
                id="lstm",
                name="LSTM Network",
                category=ModelCategory.LSTM,
                description="Long Short-Term Memory network with optional bidirectional processing and attention mechanism",
                parameters=[
                    # Core architecture (GLOBAL)
                    ModelParameter(
                        name="lstm_units",
                        type=ParameterType.INTEGER,
                        default=64,
                        min=16,
                        max=512,
                        description="Number of LSTM units per layer",
                        category="layer",
                        is_layer_configurable=True,
                    ),
                    ModelParameter(
                        name="num_layers",
                        type=ParameterType.INTEGER,
                        default=2,
                        min=1,
                        max=5,
                        description="Number of LSTM layers",
                        category="global",
                        is_layer_configurable=False,
                    ),
                    ModelParameter(
                        name="bidirectional",
                        type=ParameterType.BOOLEAN,
                        default=False,
                        description="Use bidirectional LSTM",
                        category="global",
                        is_layer_configurable=False,
                    ),
                    # Attention mechanism (GLOBAL)
                    ModelParameter(
                        name="attention",
                        type=ParameterType.BOOLEAN,
                        default=False,
                        description="Enable multi-head attention",
                        category="global",
                        is_layer_configurable=False,
                    ),
                    ModelParameter(
                        name="attention_heads",
                        type=ParameterType.INTEGER,
                        default=4,
                        min=1,
                        max=8,
                        description="Number of attention heads",
                        category="global",
                        is_layer_configurable=False,
                    ),
                    ModelParameter(
                        name="attention_position",
                        type=ParameterType.SELECT,
                        default="after_lstm",
                        options=["after_lstm", "end"],
                        description="Where to apply attention",
                        category="global",
                        is_layer_configurable=False,
                    ),
                    # Regularization (MOSTLY LAYER-CONFIGURABLE)
                    ModelParameter(
                        name="dropout_rate",
                        type=ParameterType.FLOAT,
                        default=0.2,
                        min=0.0,
                        max=0.5,
                        description="Dropout rate for regularization",
                        category="layer",
                        is_layer_configurable=True,
                    ),
                    ModelParameter(
                        name="l2_lambda",
                        type=ParameterType.FLOAT,
                        default=0.01,
                        min=0.0,
                        max=0.1,
                        description="L2 regularization factor",
                        category="global",
                        is_layer_configurable=False,
                    ),
                    ModelParameter(
                        name="recurrent_dropout",
                        type=ParameterType.FLOAT,
                        default=0.0,
                        min=0.0,
                        max=0.3,
                        description="Recurrent dropout rate",
                        category="layer",
                        is_layer_configurable=True,
                    ),
                    # Optimization (GLOBAL)
                    ModelParameter(
                        name="learning_rate",
                        type=ParameterType.FLOAT,
                        default=0.001,
                        min=0.0001,
                        max=0.01,
                        description="Learning rate for optimizer",
                        category="optimization",
                        is_layer_configurable=False,
                    ),
                    ModelParameter(
                        name="optimizer",
                        type=ParameterType.SELECT,
                        default="adam",
                        options=["adam", "rmsprop", "adamw"],
                        description="Optimizer algorithm",
                        category="optimization",
                        is_layer_configurable=False,
                    ),
                ],
                builder_function="build_lstm_model",
                input_requirements=InputRequirement(
                    shape_type="3D",
                    min_timesteps=10,
                    min_features=1,
                    description="Requires 3D input (batch, timesteps, features)",
                ),
                tags=["lstm", "recurrent", "sequence", "attention", "bidirectional"],
                complexity="medium",
                recommended_for=[
                    "Trend forecasting",
                    "Multi-step prediction",
                    "Sequential patterns",
                    "Time series with memory",
                ],
            )
        )

        # ====================================================================
        # 2. CNN ARCHITECTURE
        # ====================================================================

        self.register_model(
            ModelArchitecture(
                id="cnn",
                name="1D CNN",
                category=ModelCategory.CNN,
                description="1D Convolutional Neural Network for local pattern extraction in sequences",
                parameters=[
                    # Architecture (MOSTLY LAYER-CONFIGURABLE)
                    ModelParameter(
                        name="num_conv_layers",
                        type=ParameterType.INTEGER,
                        default=3,
                        min=1,
                        max=6,
                        description="Number of convolutional layers",
                        category="global",
                        is_layer_configurable=False,
                    ),
                    ModelParameter(
                        name="filters",
                        type=ParameterType.INTEGER,
                        default=64,
                        min=16,
                        max=256,
                        description="Number of filters in first layer (doubles each layer)",
                        category="layer",
                        is_layer_configurable=True,
                    ),
                    ModelParameter(
                        name="kernel_size",
                        type=ParameterType.INTEGER,
                        default=3,
                        min=2,
                        max=7,
                        description="Kernel size for convolutions",
                        category="layer",
                        is_layer_configurable=True,
                    ),
                    ModelParameter(
                        name="use_pooling",
                        type=ParameterType.BOOLEAN,
                        default=True,
                        description="Use max pooling between layers",
                        category="global",
                        is_layer_configurable=False,
                    ),
                    ModelParameter(
                        name="pool_size",
                        type=ParameterType.INTEGER,
                        default=2,
                        min=2,
                        max=4,
                        description="Pooling window size",
                        category="global",
                        is_layer_configurable=False,
                    ),
                    # Regularization (LAYER-CONFIGURABLE)
                    ModelParameter(
                        name="dropout_rate",
                        type=ParameterType.FLOAT,
                        default=0.2,
                        min=0.0,
                        max=0.5,
                        description="Dropout rate",
                        category="layer",
                        is_layer_configurable=True,
                    ),
                    ModelParameter(
                        name="l2_lambda",
                        type=ParameterType.FLOAT,
                        default=0.01,
                        min=0.0,
                        max=0.1,
                        description="L2 regularization",
                        category="global",
                        is_layer_configurable=False,
                    ),
                    # Optimization (GLOBAL)
                    ModelParameter(
                        name="learning_rate",
                        type=ParameterType.FLOAT,
                        default=0.001,
                        min=0.0001,
                        max=0.01,
                        description="Learning rate",
                        category="optimization",
                        is_layer_configurable=False,
                    ),
                    ModelParameter(
                        name="optimizer",
                        type=ParameterType.SELECT,
                        default="adam",
                        options=["adam", "rmsprop"],
                        description="Optimizer",
                        category="optimization",
                        is_layer_configurable=False,
                    ),
                ],
                builder_function="build_cnn_model",
                input_requirements=InputRequirement(
                    shape_type="3D",
                    min_timesteps=10,
                    description="Requires 3D input (batch, timesteps, features)",
                ),
                tags=["cnn", "convolutional", "1d", "fast", "pattern"],
                complexity="low",
                recommended_for=[
                    "Short-term patterns",
                    "Feature extraction",
                    "Fast training",
                    "Lightweight inference",
                ],
            )
        )

        # ====================================================================
        # 3. TCN ARCHITECTURE
        # ====================================================================

        self.register_model(
            ModelArchitecture(
                id="tcn",
                name="Temporal Convolutional Network",
                category=ModelCategory.TCN,
                description="TCN with dilated causal convolutions for long-range temporal dependencies",
                parameters=[
                    # Architecture
                    ModelParameter(
                        name="num_filters",
                        type=ParameterType.INTEGER,
                        default=64,
                        min=16,
                        max=256,
                        description="Number of filters per layer",
                    ),
                    ModelParameter(
                        name="kernel_size",
                        type=ParameterType.INTEGER,
                        default=3,
                        min=2,
                        max=7,
                        description="Kernel size",
                    ),
                    ModelParameter(
                        name="num_blocks",
                        type=ParameterType.INTEGER,
                        default=3,
                        min=1,
                        max=5,
                        description="Number of TCN blocks",
                    ),
                    ModelParameter(
                        name="use_residual",
                        type=ParameterType.BOOLEAN,
                        default=True,
                        description="Use residual connections",
                    ),
                    # Regularization
                    ModelParameter(
                        name="dropout_rate",
                        type=ParameterType.FLOAT,
                        default=0.2,
                        min=0.0,
                        max=0.5,
                        description="Dropout rate",
                    ),
                    ModelParameter(
                        name="l2_lambda",
                        type=ParameterType.FLOAT,
                        default=0.01,
                        min=0.0,
                        max=0.1,
                        description="L2 regularization",
                    ),
                    # Optimization
                    ModelParameter(
                        name="learning_rate",
                        type=ParameterType.FLOAT,
                        default=0.001,
                        min=0.0001,
                        max=0.01,
                        description="Learning rate",
                    ),
                    ModelParameter(
                        name="optimizer",
                        type=ParameterType.SELECT,
                        default="adam",
                        options=["adam"],
                        description="Optimizer",
                    ),
                ],
                builder_function="build_tcn_model",
                input_requirements=InputRequirement(
                    shape_type="3D",
                    min_timesteps=20,
                    description="Requires 3D input with sufficient sequence length",
                ),
                tags=["tcn", "dilated", "causal", "long-range"],
                complexity="medium",
                recommended_for=[
                    "Long-range dependencies",
                    "Causal forecasting",
                    "Variable-length sequences",
                ],
            )
        )

        # ====================================================================
        # 4. TRANSFORMER ARCHITECTURE
        # ====================================================================


        self.register_model(
            ModelArchitecture(
                id="transformer",
                name="Transformer",
                category=ModelCategory.TRANSFORMER,
                description="Transformer model with multi-head attention and feed-forward networks",
                parameters=[
                    ModelParameter(
                        name="num_heads",
                        type=ParameterType.INTEGER,
                        default=4,
                        min=2,
                        max=16,
                        description="Number of attention heads",
                    ),
                    ModelParameter(
                        name="dff",
                        type=ParameterType.INTEGER,
                        default=128,
                        min=64,
                        max=512,
                        description="Feed-forward network dimension",
                    ),
                    ModelParameter(
                        name="num_transformer_blocks",
                        type=ParameterType.INTEGER,
                        default=2,
                        min=1,
                        max=6,
                        description="Number of transformer blocks",
                    ),
                    ModelParameter(
                        name="dropout_rate",
                        type=ParameterType.FLOAT,
                        default=0.2,
                        min=0.0,
                        max=0.5,
                        description="Dropout rate",
                    ),
                ],
                builder_function="build_transformer_model",
                input_requirements=InputRequirement(
                    shape_type="3D",
                    min_timesteps=10,
                    description="Requires 3D input (batch, timesteps, features)",
                ),
                tags=["transformer", "attention", "state-of-the-art", "advanced"],
                complexity="high",
                recommended_for=[
                    "Complex patterns",
                    "Long sequences",
                    "State-of-the-art performance",
                ],
            )
        )

        # ====================================================================
        # HYBRID MODELS
        # ====================================================================

        self.register_model(
            ModelArchitecture(
                id="cnn_lstm",
                name="CNN-LSTM Hybrid",
                category=ModelCategory.HYBRID,
                description="Combines CNN for feature extraction with LSTM for sequence modeling",
                parameters=[
                    ModelParameter(
                        name="conv_filters",
                        type=ParameterType.INTEGER,
                        default=64,
                        min=16,
                        max=256,
                        description="Number of CNN filters",
                    ),
                    ModelParameter(
                        name="lstm_units",
                        type=ParameterType.INTEGER,
                        default=100,
                        min=32,
                        max=256,
                        description="Number of LSTM units",
                    ),
                    ModelParameter(
                        name="num_conv_layers",
                        type=ParameterType.INTEGER,
                        default=2,
                        min=1,
                        max=4,
                        description="Number of CNN layers",
                    ),
                    ModelParameter(
                        name="num_lstm_layers",
                        type=ParameterType.INTEGER,
                        default=2,
                        min=1,
                        max=4,
                        description="Number of LSTM layers",
                    ),
                ],
                builder_function="build_cnn_lstm_hybrid",
                input_requirements=InputRequirement(
                    shape_type="3D",
                    min_timesteps=15,
                    description="Requires 3D input for CNN+LSTM processing",
                ),
                tags=["hybrid", "cnn", "lstm", "powerful"],
                complexity="medium",
                recommended_for=[
                    "Complex time series",
                    "Multi-scale patterns",
                    "Robust predictions",
                ],
            )
        )

        self.register_model(
            ModelArchitecture(
                id="tcn_lstm_attention",
                name="TCN-LSTM-Attention",
                category=ModelCategory.HYBRID,
                description="Advanced hybrid combining TCN, LSTM, and multi-head attention",
                parameters=[
                    ModelParameter(
                        name="num_filters",
                        type=ParameterType.INTEGER,
                        default=128,
                        min=64,
                        max=256,
                        description="Number of TCN filters",
                    ),
                    ModelParameter(
                        name="lstm_units",
                        type=ParameterType.INTEGER,
                        default=64,
                        min=32,
                        max=128,
                        description="Number of LSTM units",
                    ),
                    ModelParameter(
                        name="num_heads",
                        type=ParameterType.INTEGER,
                        default=8,
                        min=4,
                        max=16,
                        description="Number of attention heads",
                    ),
                    ModelParameter(
                        name="dilation_rate",
                        type=ParameterType.INTEGER,
                        default=2,
                        min=1,
                        max=16,
                        description="TCN dilation rate",
                    ),
                ],
                builder_function="build_tcn_lstm_attention_model",
                input_requirements=InputRequirement(
                    shape_type="3D",
                    min_timesteps=20,
                    description="Requires longer sequences for optimal performance",
                ),
                tags=[
                    "hybrid",
                    "tcn",
                    "lstm",
                    "attention",
                    "advanced",
                    "state-of-the-art",
                ],
                complexity="high",
                recommended_for=[
                    "Maximum performance",
                    "Complex patterns",
                    "Research applications",
                ],
            )
        )

        # ====================================================================
        # 3D CNN MODELS
        # ====================================================================

        self.register_model(
            ModelArchitecture(
                id="cnn_3d",
                name="3D CNN",
                category=ModelCategory.CNN_3D,
                description="3D Convolutional Neural Network for multi-dimensional data",
                parameters=[
                    ModelParameter(
                        name="filters",
                        type=ParameterType.INTEGER,
                        default=256,
                        min=64,
                        max=512,
                        description="Number of filters in first layer",
                    ),
                    ModelParameter(
                        name="num_conv_blocks",
                        type=ParameterType.INTEGER,
                        default=2,
                        min=1,
                        max=4,
                        description="Number of 3D conv blocks",
                    ),
                    ModelParameter(
                        name="dropout_rate",
                        type=ParameterType.FLOAT,
                        default=0.3,
                        min=0.0,
                        max=0.5,
                        description="Dropout rate",
                    ),
                ],
                builder_function="build_3d_cnn_model",
                input_requirements=InputRequirement(
                    shape_type="4D",
                    description="Requires 4D input (batch, depth, height, width, channels)",
                ),
                tags=["3d", "cnn", "multi-dimensional", "advanced"],
                complexity="high",
                recommended_for=["Multi-dimensional data", "Spatial-temporal patterns"],
            )
        )

        # ====================================================================
        # DNN MODELS
        # ====================================================================
        # MULTI-HEAD TCN
        # ====================================================================

        self.register_model(
            ModelArchitecture(
                id="multihead_tcn",
                name="Multi-Head TCN",
                category=ModelCategory.HYBRID,
                description="Multi-head dilated CNN with parallel branches for capturing multiple temporal scales",
                parameters=[
                    ModelParameter(
                        name="num_branches",
                        type=ParameterType.INTEGER,
                        default=3,
                        min=2,
                        max=5,
                        description="Number of parallel branches",
                    ),
                    ModelParameter(
                        name="num_filters",
                        type=ParameterType.INTEGER,
                        default=64,
                        min=16,
                        max=256,
                        description="Filters per branch",
                    ),
                    ModelParameter(
                        name="kernel_size",
                        type=ParameterType.INTEGER,
                        default=3,
                        min=2,
                        max=7,
                        description="Kernel size",
                    ),
                    ModelParameter(
                        name="attention_enabled",
                        type=ParameterType.BOOLEAN,
                        default=True,
                        description="Enable attention mechanism",
                    ),
                    ModelParameter(
                        name="attention_heads",
                        type=ParameterType.INTEGER,
                        default=4,
                        min=2,
                        max=8,
                        description="Number of attention heads",
                    ),
                    ModelParameter(
                        name="lstm_units",
                        type=ParameterType.INTEGER,
                        default=64,
                        min=32,
                        max=256,
                        description="LSTM units",
                    ),
                    ModelParameter(
                        name="dropout_rate",
                        type=ParameterType.FLOAT,
                        default=0.2,
                        min=0.0,
                        max=0.5,
                        description="Dropout rate",
                    ),
                    ModelParameter(
                        name="l2_lambda",
                        type=ParameterType.FLOAT,
                        default=0.01,
                        min=0.0,
                        max=0.1,
                        description="L2 regularization",
                    ),
                    ModelParameter(
                        name="learning_rate",
                        type=ParameterType.FLOAT,
                        default=0.001,
                        min=0.0001,
                        max=0.01,
                        description="Learning rate",
                    ),
                    ModelParameter(
                        name="optimizer",
                        type=ParameterType.SELECT,
                        default="adam",
                        options=["adam"],
                        description="Optimizer",
                    ),
                ],
                builder_function="build_multihead_tcn",
                input_requirements=InputRequirement(
                    shape_type="3D",
                    min_timesteps=20,
                    description="Requires 3D input with sufficient sequence length",
                ),
                tags=["multihead", "tcn", "parallel", "multi-scale", "advanced"],
                complexity="high",
                recommended_for=[
                    "Multi-scale features",
                    "Ensemble learning",
                    "Advanced time-series",
                    "Capturing multiple temporal scales",
                ],
            )
        )

        # ====================================================================
        # 3D CNN
        # ====================================================================

        self.register_model(
            ModelArchitecture(
                id="cnn_3d",
                name="3D CNN",
                category=ModelCategory.CNN_3D,
                description="3D Convolutional Neural Network for multi-dimensional spatial-temporal data",
                parameters=[
                    ModelParameter(
                        name="num_3d_layers",
                        type=ParameterType.INTEGER,
                        default=3,
                        min=1,
                        max=5,
                        description="Number of 3D conv layers",
                    ),
                    ModelParameter(
                        name="filters",
                        type=ParameterType.INTEGER,
                        default=32,
                        min=16,
                        max=128,
                        description="Filters (doubles each layer)",
                    ),
                    ModelParameter(
                        name="use_pooling",
                        type=ParameterType.BOOLEAN,
                        default=True,
                        description="Use 3D max pooling",
                    ),
                    ModelParameter(
                        name="hybrid_type",
                        type=ParameterType.SELECT,
                        default="none",
                        options=["none", "lstm", "tcn"],
                        description="Hybrid architecture type",
                    ),
                    ModelParameter(
                        name="lstm_units",
                        type=ParameterType.INTEGER,
                        default=64,
                        min=32,
                        max=256,
                        description="LSTM units (if hybrid_type=lstm)",
                    ),
                    ModelParameter(
                        name="dropout_rate",
                        type=ParameterType.FLOAT,
                        default=0.2,
                        min=0.0,
                        max=0.5,
                        description="Dropout rate",
                    ),
                    ModelParameter(
                        name="l2_lambda",
                        type=ParameterType.FLOAT,
                        default=0.01,
                        min=0.0,
                        max=0.1,
                        description="L2 regularization",
                    ),
                    ModelParameter(
                        name="learning_rate",
                        type=ParameterType.FLOAT,
                        default=0.001,
                        min=0.0001,
                        max=0.01,
                        description="Learning rate",
                    ),
                    ModelParameter(
                        name="optimizer",
                        type=ParameterType.SELECT,
                        default="adam",
                        options=["adam"],
                        description="Optimizer",
                    ),
                ],
                builder_function="build_3d_cnn_model",
                input_requirements=InputRequirement(
                    shape_type="3D",
                    min_timesteps=10,
                    description="Requires 3D input (auto-reshaped to 5D for 3D convolutions)",
                ),
                tags=["3d", "cnn", "spatial", "multi-dimensional"],
                complexity="medium",
                recommended_for=[
                    "Multi-dimensional data",
                    "Multi-market analysis",
                    "Spatial patterns",
                    "Multi-timeframe grids",
                ],
            )
        )

        # ====================================================================
        # DNN (OPTIONAL - for tabular data)
        # ====================================================================

        self.register_model(
            ModelArchitecture(
                id="dnn",
                name="Deep Neural Network",
                category=ModelCategory.DNN,
                description="Fully connected deep neural network for tabular/flattened data",
                parameters=[
                    ModelParameter(
                        name="units",
                        type=ParameterType.INTEGER,
                        default=64,
                        min=16,
                        max=256,
                        description="Number of units in first layer",
                    ),
                    ModelParameter(
                        name="num_dense_layers",
                        type=ParameterType.INTEGER,
                        default=4,
                        min=2,
                        max=8,
                        description="Number of dense layers",
                    ),
                    ModelParameter(
                        name="dropout_rate",
                        type=ParameterType.FLOAT,
                        default=0.2,
                        min=0.0,
                        max=0.5,
                        description="Dropout rate",
                    ),
                ],
                builder_function="build_dnn_model",
                input_requirements=InputRequirement(
                    shape_type="2D", description="Requires 2D input (batch, features)"
                ),
                tags=["dnn", "fully-connected", "tabular", "simple"],
                complexity="low",
                recommended_for=[
                    "Tabular data",
                    "Feature-based prediction",
                    "Simple patterns",
                ],
            )
        )
        
        # Finally, trigger dynamic discovery for proprietary models
        self._register_proprietary_models()


    def _register_proprietary_models(self):
        """
        Auto-discover and register verified proprietary models from Backend/Backend/data/.

        A model is registered only if:
          1. Its .py file defines MODEL_REGISTRY_METADATA with all required fields.
          2. Its declared .keras checkpoint exists on disk (1-epoch training gate).

        This method is called at the end of _register_default_models() so proprietary
        models are always included in the catalog on backend startup.
        """
        import logging
        logger = logging.getLogger(__name__)

        try:
            from app.core.ml.proprietary_model_reader import discover_proprietary_models
        except ImportError as e:
            logger.error(f"[Registry] Failed to import proprietary_model_reader: {e}")
            return

        try:
            discovered = discover_proprietary_models(require_verification=True)
        except Exception as e:
            logger.error(f"[Registry] Error during proprietary model discovery: {e}", exc_info=True)
            return

        for meta in discovered:
            model_id = meta.get("id", "unknown")
            try:
                # Build ModelParameter list from the metadata dict format
                raw_params = meta.get("parameters", [])
                parameters = []
                for p in raw_params:
                    try:
                        parameters.append(ModelParameter(
                            name=p["name"],
                            type=p.get("type", "float"),
                            default=p.get("default"),
                            min=p.get("min"),
                            max=p.get("max"),
                            options=p.get("options"),
                            description=p.get("description", ""),
                            required=p.get("required", True),
                            is_layer_configurable=p.get("is_layer_configurable", False),
                            category=p.get("category", "global"),
                        ))
                    except Exception as pe:
                        logger.warning(f"[Registry] Skipping malformed parameter in '{model_id}': {pe}")

                # Resolve category enum
                raw_cat = meta.get("category", "proprietary")
                try:
                    category = ModelCategory(raw_cat)
                except ValueError:
                    category = ModelCategory.PROPRIETARY

                # Build InputRequirement
                ir = meta.get("input_requirements", {})
                input_req = InputRequirement(
                    shape_type=ir.get("shape_type", "3D"),
                    min_timesteps=ir.get("min_timesteps"),
                    min_features=ir.get("min_features"),
                    description=ir.get("description", "3D input (batch, timesteps, features)"),
                )

                architecture = ModelArchitecture(
                    id=model_id,
                    name=meta["name"],
                    category=category,
                    description=meta.get("description", ""),
                    parameters=parameters,
                    builder_function=meta["builder_function"],
                    input_requirements=input_req,
                    tags=meta.get("tags", []),
                    complexity=meta.get("complexity", "high"),
                    recommended_for=meta.get("recommended_for", []),
                    input_mode=meta.get("input_mode", "single"),
                    output_mode=meta.get("output_mode", "multi"),
                    supports_target_selection=meta.get("supports_target_selection", False),
                )

                self.register_model(architecture)
                logger.info(
                    f"[Registry] ✅ Registered proprietary model: '{model_id}' "
                    f"('{meta['name']}') — checkpoint: {meta.get('_checkpoint_size_mb', '?')} MB"
                )

            except Exception as e:
                logger.error(
                    f"[Registry] Failed to register proprietary model '{model_id}': {e}",
                    exc_info=True
                )


# Global registry instance
_registry = None


def get_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
