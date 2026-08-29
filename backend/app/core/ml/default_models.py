"""
Default Model Architectures
Parameterized builders for 9 core neural network architectures.
Each builder consolidates multiple variants from MyModelss.py into a single configurable function.
"""

import keras
from keras import backend, models, layers, regularizers, Input, optimizers
from app.core.ml.PositionEncoding import PositionalEncoding


def _get_optimizer(optimizer_name, learning_rate):
    """Helper to get unified optimizer instance."""
    if optimizer_name == 'adam':
        return optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        return optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_name == 'adamw':
        return optimizers.AdamW(learning_rate=learning_rate)
    return optimizers.Adam(learning_rate=learning_rate)


def build_lstm_model(
    input_shape,
    n_predictions=4,
    # Architecture params
    lstm_units=64,
    num_layers=2,
    bidirectional=False,
    batch_normalization=False,
    add_dense_layers=False,
    # Attention params
    attention=False,
    attention_heads=4,
    attention_position='after_lstm',  # 'after_lstm' or 'end'
    # Regularization
    dropout_rate=0.2,
    l2_lambda=0.01,
    recurrent_dropout=0.0,

    dense_units=64,
    # Optimization
    learning_rate=0.001,
    optimizer='adam',
    # NEW: Per-layer configuration support
    layer_configs=None
):
    """
    Unified LSTM builder handling all variations:
    - Basic LSTM, Bidirectional LSTM, LSTM + Attention, LSTM + Transformer.
    
    Args:
        layer_configs: Optional list of dicts, each dict contains layer-specific params
                      [{layer_index: 0, lstm_units: 64, dropout_rate: 0.1}, ...]
                      If provided, overrides lstm_units, dropout_rate, recurrent_dropout per layer
    """
    backend.clear_session()
    
    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    # Build LSTM layers
    for i in range(num_layers):
        # Determine per-layer configuration
        if layer_configs and i < len(layer_configs):
            layer_config = layer_configs[i]
            layer_lstm_units = layer_config.get('lstm_units', lstm_units)
            layer_dropout = layer_config.get('dropout_rate', dropout_rate)
            layer_recurrent_dropout = layer_config.get('recurrent_dropout', recurrent_dropout)
        else:
            # Backward compatibility: use single value for all layers
            layer_lstm_units = lstm_units
            layer_dropout = dropout_rate
            layer_recurrent_dropout = recurrent_dropout
        
        return_sequences = (i < num_layers - 1) or attention
        
        lstm_layer = layers.LSTM(
            units=layer_lstm_units,  # Per-layer
            return_sequences=return_sequences,
            recurrent_dropout=layer_recurrent_dropout,  # Per-layer
            kernel_regularizer=regularizers.l2(l2_lambda)
        )
        
        if bidirectional:
            x = layers.Bidirectional(lstm_layer)(x)
        else:
            x = lstm_layer(x)
        
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        
        # Apply attention after LSTM layers if specified
        if attention and attention_position == 'after_lstm' and i == num_layers - 1:
            attention_output = layers.MultiHeadAttention(
                num_heads=attention_heads,
                key_dim=layer_lstm_units  # Use last layer's unit count
            )(x, x)
            x = layers.Add()([x, attention_output])
            x = layers.BatchNormalization()(x)
    
    # Flatten if still sequence
    if len(x.shape) > 2:
        x = layers.Flatten()(x)
    
    # Apply attention at end if specified
    if attention and attention_position == 'end':
        x = layers.Reshape((1, -1))(x)
        attention_output = layers.MultiHeadAttention(
            num_heads=attention_heads,
            key_dim=lstm_units  # Default units for this
        )(x, x)
        x = layers.Flatten()(attention_output)
    
    # Dense layers (use global dropout_rate)
    if add_dense_layers:
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(dense_units, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(x)
        x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = layers.Dense(n_predictions, activation='linear')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=_get_optimizer(optimizer, learning_rate), loss='mse', metrics=['mae', 'mse'])
    
    return model


def build_cnn_model(
    input_shape,
    n_predictions=4,
    # Architecture params
    num_conv_layers=3,
    filters=64,
    kernel_size=3,
    use_pooling=True,
    pool_size=2,
    # Regularization
    dropout_rate=0.2,
    l2_lambda=0.01,
    # Optimization
    learning_rate=0.001,
    optimizer='adam',
    # NEW: Per-layer configuration support
    layer_configs=None
):
    """
    Unified CNN builder for 1D convolutions.
    
    Args:
        layer_configs: Optional list of dicts, each with layer-specific params
                      [{layer_index: 0, filters: 64, kernel_size: 3, dropout_rate: 0.1}, ...]
    """
    backend.clear_session()
    
    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    # Build convolutional layers
    for i in range(num_conv_layers):
        # Determine per-layer configuration
        if layer_configs and i < len(layer_configs):
            layer_config = layer_configs[i]
            layer_filters = layer_config.get('filters', filters * (2 ** i) if i < 3 else filters * 8)
            layer_kernel_size = layer_config.get('kernel_size', kernel_size)
            layer_dropout = layer_config.get('dropout_rate', dropout_rate)
        else:
            # Backward compatibility: use single value for all layers
            layer_filters = filters * (2 ** i) if i < 3 else filters * 8
            layer_kernel_size = kernel_size
            layer_dropout = dropout_rate
        
        x = layers.Conv1D(
            filters=layer_filters,  # Per-layer
            kernel_size=layer_kernel_size,  # Per-layer
            padding='same',
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_lambda)
        )(x)
        x = layers.BatchNormalization()(x)
        
        if use_pooling and i < num_conv_layers - 1:
            x = layers.MaxPooling1D(pool_size=pool_size)(x)
        
        x = layers.Dropout(layer_dropout)(x)  # Per-layer
    
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers (use global dropout)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = layers.Dense(n_predictions, activation='linear')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=_get_optimizer(optimizer, learning_rate), loss='mse', metrics=['mae', 'mse'])
    
    return model


def build_tcn_model(
    input_shape,
    n_predictions=4,
    # Architecture params
    num_filters=64,
    kernel_size=3,
    dilation_rates=None,
    num_blocks=3,
    use_residual=True,
    # Regularization
    dropout_rate=0.2,
    l2_lambda=0.01,
    # Optimization
    learning_rate=0.001,
    optimizer='adam'
):
    """Temporal Convolutional Network with dilated causal convolutions."""
    backend.clear_session()
    
    if dilation_rates is None:
        dilation_rates = [1, 2, 4, 8, 16, 32]
    
    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    for block in range(num_blocks):
        for dilation_rate in dilation_rates[:min(len(dilation_rates), 6)]:
            residual = x
            x = layers.Conv1D(
                filters=num_filters,
                kernel_size=kernel_size,
                padding='causal',
                dilation_rate=dilation_rate,
                activation='relu',
                kernel_regularizer=regularizers.l2(l2_lambda)
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout_rate)(x)
            
            x = layers.Conv1D(
                filters=num_filters,
                kernel_size=kernel_size,
                padding='causal',
                dilation_rate=dilation_rate,
                activation='relu',
                kernel_regularizer=regularizers.l2(l2_lambda)
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout_rate)(x)
            
            if use_residual:
                if residual.shape[-1] != num_filters:
                    residual = layers.Conv1D(num_filters, 1, padding='same')(residual)
                x = layers.Add()([x, residual])
    
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = layers.Dense(n_predictions, activation='linear')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=_get_optimizer(optimizer, learning_rate), loss='mse', metrics=['mae', 'mse'])
    
    return model


def build_transformer_model(
    input_shape,
    n_predictions=4,
    # Architecture params
    num_heads=4,
    dff=128,
    num_transformer_blocks=2,
    use_positional_encoding=True,
    # Regularization
    dropout_rate=0.2,
    l2_lambda=0.01,
    # Optimization
    learning_rate=0.001,
    optimizer='adam'
):
    """Pure Transformer architecture with multi-head attention."""
    backend.clear_session()
    
    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    if use_positional_encoding:
        pos_encoding = PositionalEncoding()(x)
        x = layers.Add()([x, pos_encoding])
    
    for _ in range(num_transformer_blocks):
        # Attention
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[-1])(x, x)
        attention_output = layers.Dropout(dropout_rate)(attention_output)
        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # FFN
        ffn = layers.Dense(dff, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(x)
        ffn = layers.Dropout(dropout_rate)(ffn)
        ffn = layers.Dense(input_shape[-1], kernel_regularizer=regularizers.l2(l2_lambda))(ffn)
        ffn = layers.Dropout(dropout_rate)(ffn)
        x = layers.Add()([x, ffn])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = layers.Dense(n_predictions, activation='linear')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=_get_optimizer(optimizer, learning_rate), loss='mse', metrics=['mae', 'mse'])
    
    return model


def build_cnn_lstm_hybrid(
    input_shape,
    n_predictions=4,
    # CNN params
    num_cnn_layers=3,
    conv_filters=64,
    kernel_size=3,
    # LSTM params
    lstm_units=100,
    num_lstm_layers=2,
    use_bidirectional_lstm=False,
    # Regularization
    dropout_rate=0.2,
    l2_lambda=0.01,
    # Optimization
    learning_rate=0.001,
    optimizer='adam'
):
    """CNN-LSTM Hybrid for spatial-temporal feature learning."""
    backend.clear_session()
    
    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    # CNN
    for i in range(num_cnn_layers):
        x = layers.Conv1D(
            filters=conv_filters * (2 ** i) if i < 3 else conv_filters * 8,
            kernel_size=kernel_size,
            padding='same',
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_lambda)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(dropout_rate)(x)
    
    # LSTM
    for i in range(num_lstm_layers):
        return_sequences = i < num_lstm_layers - 1
        lstm_layer = layers.LSTM(units=lstm_units, return_sequences=return_sequences, kernel_regularizer=regularizers.l2(l2_lambda))
        x = layers.Bidirectional(lstm_layer)(x) if use_bidirectional_lstm else lstm_layer(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = layers.Dense(n_predictions, activation='linear')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=_get_optimizer(optimizer, learning_rate), loss='mse', metrics=['mae', 'mse'])
    
    return model


def build_multihead_tcn(
    input_shape,
    n_predictions=4,
    # Architecture params
    num_branches=3,
    dilation_rates=None,
    num_filters=64,
    kernel_size=3,
    # Attention params
    attention_enabled=True,
    attention_heads=4,
    # LSTM params
    lstm_units=64,
    # Regularization
    dropout_rate=0.2,
    l2_lambda=0.01,
    # Optimization
    learning_rate=0.001,
    optimizer='adam'
):
    """Multi-Head Dilated CNN with parallel branches."""
    backend.clear_session()
    
    if dilation_rates is None:
        dilation_rates = [[1, 2, 4], [1, 4, 8], [1, 8, 16]]
    
    inputs = layers.Input(shape=input_shape)
    
    branches = []
    for branch_idx in range(num_branches):
        x = inputs
        rates = dilation_rates[branch_idx] if branch_idx < len(dilation_rates) else [1, 2, 4]
        for dilation_rate in rates:
            x = layers.Conv1D(
                filters=num_filters,
                kernel_size=kernel_size,
                padding='causal',
                dilation_rate=dilation_rate,
                activation='relu',
                kernel_regularizer=regularizers.l2(l2_lambda)
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout_rate)(x)
        branches.append(x)
    
    x = layers.Concatenate()(branches) if len(branches) > 1 else branches[0]
    
    if attention_enabled:
        attention_output = layers.MultiHeadAttention(num_heads=attention_heads, key_dim=num_filters)(x, x)
        x = layers.Add()([x, attention_output])
        x = layers.BatchNormalization()(x)
    
    x = layers.LSTM(units=lstm_units, kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = layers.Dense(n_predictions, activation='linear')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=_get_optimizer(optimizer, learning_rate), loss='mse', metrics=['mae', 'mse'])
    
    return model


def build_3d_cnn_model(
    input_shape,
    n_predictions=4,
    # Architecture params
    num_3d_layers=3,
    filters=32,
    kernel_size=(3, 3, 3),
    use_pooling=True,
    hybrid_type=None,  # None, 'lstm', or 'tcn'
    # Hybrid params
    lstm_units=64,
    # Regularization
    dropout_rate=0.2,
    l2_lambda=0.01,
    # Optimization
    learning_rate=0.001,
    optimizer='adam'
):
    """3D CNN for multi-dimensional spatial-temporal data."""
    backend.clear_session()
    
    if len(input_shape) == 2:
        inputs = layers.Input(shape=input_shape)
        x = layers.Reshape((input_shape[0], input_shape[1], 1, 1, 1))(inputs)
    else:
        inputs = layers.Input(shape=input_shape)
        x = inputs
    
    for i in range(num_3d_layers):
        x = layers.Conv3D(
            filters=filters * (2 ** i),
            kernel_size=kernel_size,
            padding='same',
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_lambda)
        )(x)
        x = layers.BatchNormalization()(x)
        if use_pooling:
            x = layers.MaxPooling3D(pool_size=(2, 2, 1))(x)
        x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Flatten()(x)
    
    if hybrid_type == 'lstm':
        x = layers.Reshape((-1, lstm_units))(x)
        x = layers.LSTM(units=lstm_units, kernel_regularizer=regularizers.l2(l2_lambda))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
    elif hybrid_type == 'tcn':
        x = layers.Reshape((-1, filters * (2 ** (num_3d_layers - 1))))(x)
        x = layers.Conv1D(filters=64, kernel_size=3, padding='causal', activation='relu')(x)
        x = layers.GlobalAveragePooling1D()(x)
    
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = layers.Dense(n_predictions, activation='linear')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=_get_optimizer(optimizer, learning_rate), loss='mse', metrics=['mae', 'mse'])
    
    return model


def build_tcn_lstm_attention_model(
    input_shape,
    n_predictions=4,
    # Architecture params
    num_filters=128,
    kernel_size=3,
    dilation_rate=2,
    # LSTM params
    lstm_units=64,
    # Attention params
    num_heads=8,
    # Regularization
    dropout_rate=0.2,
    l2_lambda=0.01,
    # Optimization
    learning_rate=0.001,
    optimizer='adam'
):
    """Advanced hybrid combining TCN features, LSTM memory, and global attention."""
    backend.clear_session()
    
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv1D(
        filters=num_filters,
        kernel_size=kernel_size,
        padding='causal',
        dilation_rate=dilation_rate,
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_lambda)
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.LSTM(units=lstm_units, return_sequences=True, kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = layers.BatchNormalization()(x)
    
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=lstm_units)(x, x)
    x = layers.Add()([x, attention_output])
    x = layers.BatchNormalization()(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = layers.Dense(n_predictions, activation='linear')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=_get_optimizer(optimizer, learning_rate), loss='mse', metrics=['mae', 'mse'])
    
    return model


def build_dnn_model(
    input_shape,
    n_predictions=4,
    # Architecture params
    units=64,
    num_dense_layers=4,
    # Regularization
    dropout_rate=0.2,
    l2_lambda=0.01,
    # Optimization
    learning_rate=0.001,
    optimizer='adam'
):
    """Standard Deep Neural Network for tabular or pre-flattened data."""
    backend.clear_session()
    
    inputs = layers.Input(shape=input_shape)
    x = layers.Flatten()(inputs) if len(input_shape) > 1 else inputs
    
    curr_units = units
    for i in range(num_dense_layers):
        x = layers.Dense(curr_units, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        if i % 2 == 0 and curr_units > 16:
            curr_units //= 2
            
    # Output layer
    outputs = layers.Dense(n_predictions, activation='linear')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=_get_optimizer(optimizer, learning_rate), loss='mse', metrics=['mae', 'mse'])
    
    return model


# ══════════════════════════════════════════════════════════════════════════════
# PROPRIETARY MODEL ADAPTERS
# ══════════════════════════════════════════════════════════════════════════════
# Adapters bridge the standard registry builder interface:
#   (input_shape, n_predictions, **params)
# to proprietary models with specialised signatures.
#
# ADDING A NEW PROPRIETARY MODEL:
#   1. Add your model file to Backend/Backend/data/ with MODEL_REGISTRY_METADATA.
#   2. Add an adapter here following the build_axe_genesis_v1 pattern.
#   3. Set "builder_function": "your_adapter_name" in MODEL_REGISTRY_METADATA.
# ══════════════════════════════════════════════════════════════════════════════


def build_axe_genesis_v1(
    input_shape,
    n_predictions=12,
    # ── AXE Genesis hyperparameters (mirrored from MODEL_REGISTRY_METADATA) ──
    learning_rate: float = 3e-4,
    hw_decay: float = 0.85,
    huber_delta: float = 0.05,
    dir_weight: float = 0.20,
    breakout_pos_weight: float = 4.0,
    # ── Dataset routing — needed to load feature_index_map ────────────────────
    ml_preparation_ref: str = None,
    dataset_id: str = None,
    dataset_name: str = None,
    # ── Optional target list — used only for the required-targets check ────────
    target_cols=None,
    selected_targets=None,
    # Absorb any extra registry fields passed by _FunctionBuilder.build()
    **_ignored,
):
    """
    Adapter: bridges the standard registry builder interface to build_baseline_brain.

    This function is what the registry calls when the user selects "AXE Genesis Model"
    in the frontend. It:

      1. Validates that the dataset contains the required OHLCV target columns.
      2. Locates the feature_index_map.json for the chosen dataset.
      3. Computes continuous_feature_indices / structure_indices via
         build_category_indices (same logic as the standalone training script).
      4. Builds and compiles the full baseline_brain model.

    REQUIRED TARGET VALIDATION
    ──────────────────────────
    The AXE Genesis model requires OHLCV sequence targets to be present. If the
    chosen dataset is missing any of these, we raise a clear ValueError so the
    frontend can surface a helpful error message rather than silently producing a
    broken model.

    Args:
        input_shape:       (seq_len, n_features)  — from dataset metadata
        n_predictions:     number of forecast steps (ignored — read from input_shape chunk)
        learning_rate:     Adam LR
        hw_decay:          horizon weight decay for point loss
        huber_delta:       Huber δ for OHLCV heads
        dir_weight:        direction penalty weight
        breakout_pos_weight: positive-class weight for breakout BCE
        ml_preparation_ref: dataset reference (name/id) used to locate ML cache
        dataset_id:        dataset UUID
        dataset_name:      dataset slug name (used to find ML cache directory)
        target_cols:       full list of target column names in the selected dataset
        selected_targets:  user-selected targets (subset of target_cols)

    Returns:
        Compiled keras.Model ready for training.

    Raises:
        ValueError: if required OHLCV targets are missing from the dataset.
        FileNotFoundError: if feature_index_map.json cannot be located.
    """
    import os
    import json
    import sys
    import logging
    from pathlib import Path

    logger = logging.getLogger(__name__)

    # ── Step 1: Import proprietary module ────────────────────────────────────
    # Resolve path: this file is at Backend/app/core/ml/default_models.py
    _this_dir   = Path(__file__).resolve().parent          # .../app/core/ml/
    _backend    = _this_dir.parent.parent.parent           # .../Backend/
    _data_dir   = _backend / "Backend" / "data"
    _module_path = _data_dir / "baseline_v1.py"

    if not _module_path.exists():
        raise FileNotFoundError(
            f"AXE Genesis source not found at {_module_path}. "
            f"Ensure Backend/Backend/data/baseline_v1.py is present."
        )

    import importlib.util
    _spec = importlib.util.spec_from_file_location("_axe_genesis_v1", str(_module_path))
    _mod  = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

    build_baseline_brain    = _mod.build_baseline_brain
    compile_baseline_brain  = _mod.compile_baseline_brain
    build_category_indices  = _mod.build_category_indices
    REQUIRED_TARGETS        = _mod.MODEL_REGISTRY_METADATA.get("required_targets", [])

    # ── Step 2: Required-targets validation ──────────────────────────────────
    # Resolve the working target list from whichever source is available
    available_targets = set(target_cols or []) | set(selected_targets or [])

    if available_targets and REQUIRED_TARGETS:
        missing = [t for t in REQUIRED_TARGETS if t not in available_targets]
        if missing:
            raise ValueError(
                f"[AXE Genesis] Dataset is missing {len(missing)} required target(s):\n"
                f"  Missing : {missing}\n"
                f"  Required: {REQUIRED_TARGETS}\n"
                f"  Available: {sorted(available_targets)[:10]}{'...' if len(available_targets) > 10 else ''}\n\n"
                f"The AXE Genesis Model requires OHLCV sequence targets (future_sequence + "
                f"adv_target_Open/High/Low/Volume_seq) to be present in the dataset. "
                f"Select a dataset generated by the full ML pipeline (including adv_targets)."
            )
        logger.info(f"[AXE Genesis] ✅ Required targets validated ({len(REQUIRED_TARGETS)} present)")
    else:
        logger.warning(
            "[AXE Genesis] target_cols/selected_targets not provided — "
            "skipping required-targets check. Ensure dataset has OHLCV targets."
        )

    # ── Step 3: Locate feature_index_map.json ────────────────────────────────
    # Search strategy (in priority order):
    #   1. Backend/Backend/data/ml_cache/{dataset_name}/feature_index_map.json
    #   2. Backend/data/ml_cache/{dataset_name}/feature_index_map.json
    #   3. Backend/data/ml_cache/*/feature_index_map.json  (any dataset — last resort)
    _ds_name = dataset_name or ml_preparation_ref or dataset_id or ""
    _ds_name = _ds_name.replace("/", "_").strip()

    _search_roots = [
        _data_dir / "ml_cache",
        _backend / "data" / "ml_cache",
    ]

    feature_index_map_path = None
    for root in _search_roots:
        if _ds_name:
            candidate = root / _ds_name / "feature_index_map.json"
            if candidate.exists():
                feature_index_map_path = candidate
                break
        # Fallback: find any feature_index_map.json under this root
        candidates = sorted(root.glob("*/feature_index_map.json"))
        if candidates:
            feature_index_map_path = candidates[-1]  # most recent alphabetically
            logger.warning(
                f"[AXE Genesis] dataset_name='{_ds_name}' not matched — "
                f"using fallback feature_index_map: {feature_index_map_path}"
            )
            break

    if feature_index_map_path is None:
        raise FileNotFoundError(
            f"[AXE Genesis] Cannot locate feature_index_map.json for dataset '{_ds_name}'.\n"
            f"Searched: {[str(r) for r in _search_roots]}\n"
            f"Ensure the ML preparation step has been run for this dataset."
        )

    logger.info(f"[AXE Genesis] Loading feature_index_map from {feature_index_map_path}")
    with open(feature_index_map_path) as _f:
        _map_data = json.load(_f)
    feature_index_map = _map_data.get("feature_index_map", _map_data)
    feature_names = [k for k, _ in sorted(feature_index_map.items(), key=lambda kv: kv[1])]
    logger.info(f"[AXE Genesis] Feature map loaded: {len(feature_names)} features")

    # ── Step 4: Category routing ─────────────────────────────────────────────
    category_indices = build_category_indices(feature_names)
    structure_indices = category_indices.get("structure", [])

    # Continuous: candle + momentum + flow + structure + conviction
    CONTINUOUS_CATS = {"candle", "momentum", "flow", "structure", "conviction"}
    continuous_feature_indices = sorted(
        idx for cat in CONTINUOUS_CATS for idx in category_indices.get(cat, [])
    )
    logger.info(
        f"[AXE Genesis] Category routing complete — "
        f"structure: {len(structure_indices)}, continuous: {len(continuous_feature_indices)}"
    )

    # ── Step 5: Build ─────────────────────────────────────────────────────────
    seq_len    = input_shape[0]
    n_features = input_shape[1]
    # Use n_predictions as the forecast horizon; fall back to 12 (v1 default)
    forecast_steps = int(n_predictions) if n_predictions and n_predictions > 1 else 12

    logger.info(
        f"[AXE Genesis] Building model — input_shape={input_shape}, "
        f"forecast_steps={forecast_steps}"
    )

    model = build_baseline_brain(
        input_shape                = (seq_len, n_features),
        continuous_feature_indices = continuous_feature_indices,
        structure_indices          = structure_indices,
        forecast_steps             = forecast_steps,
        extra_series               = {
            "support_trendline":       forecast_steps,
            "resist_trendline":        forecast_steps,
            "mfe":                     1,
            "mae":                     1,
            "reversal_prob":           1,
            "trend_continuation_prob": 1,
            "reversal_held":           1,
        },
    )

    model = compile_baseline_brain(
        model,
        forecast_steps      = forecast_steps,
        learning_rate       = learning_rate,
        hw_decay            = hw_decay,
        huber_delta         = huber_delta,
        dir_weight          = dir_weight,
        breakout_pos_weight = breakout_pos_weight,
    )

    logger.info(f"[AXE Genesis] ✅ Model built and compiled — {model.count_params():,} parameters")
    return model


def build_axe_vortex_v8(
    input_shape,
    n_predictions=12,
    learning_rate: float = 3e-4,
    ml_preparation_ref: str = None,
    dataset_id: str = None,
    dataset_name: str = None,
    target_cols=None,
    selected_targets=None,
    **_ignored,
):
    """
    Adapter: bridges the standard registry builder interface to AXE Vortex (V8 Semantic Tower Encoder).
    """
    import json
    import logging
    from pathlib import Path
    import importlib.util

    logger = logging.getLogger(__name__)

    _this_dir   = Path(__file__).resolve().parent
    _backend    = _this_dir.parent.parent.parent
    _data_dir   = _backend / "Backend" / "data"
    _module_path = _data_dir / "baseline_encoderv8.py"

    if not _module_path.exists():
        raise FileNotFoundError(f"AXE Vortex source not found at {_module_path}.")

    _spec = importlib.util.spec_from_file_location("_axe_vortex_v8", str(_module_path))
    _mod  = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

    build_model_v8 = _mod.build_model_v8
    compile_model_v8 = _mod.compile_model_v8
    build_category_indices = _mod.build_category_indices

    _ds_name = (dataset_name or ml_preparation_ref or dataset_id or "").replace("/", "_").strip()
    _search_roots = [_data_dir / "ml_cache", _backend / "data" / "ml_cache"]
    feature_index_map_path = None
    for root in _search_roots:
        if _ds_name:
            candidate = root / _ds_name / "feature_index_map.json"
            if candidate.exists():
                feature_index_map_path = candidate
                break
        candidates = sorted(root.glob("*/feature_index_map.json"))
        if candidates:
            feature_index_map_path = candidates[-1]
            break

    if feature_index_map_path is None:
        raise FileNotFoundError(f"[AXE Vortex] Cannot locate feature_index_map.json for '{_ds_name}'.")

    with open(feature_index_map_path) as _f:
        _map_data = json.load(_f)
    feature_index_map = _map_data.get("feature_index_map", _map_data)
    feature_names = [k for k, _ in sorted(feature_index_map.items(), key=lambda kv: kv[1])]

    category_indices = build_category_indices(feature_names)
    session_indices = category_indices.get("session", [])
    close_col_idx = feature_index_map.get("Close", 3)

    forecast_steps = int(n_predictions) if n_predictions and n_predictions > 1 else 12

    model = build_model_v8(
        input_shape=input_shape,
        output_dim=forecast_steps,
        category_indices=category_indices,
        session_indices=session_indices,
        close_col_idx=close_col_idx,
    )
    model = compile_model_v8(model, learning_rate=learning_rate)

    logger.info(f"[AXE Vortex] ✅ Model built and compiled — {model.count_params():,} parameters")
    return model


def build_axe_chimera_v8_hybrid(
    input_shape,
    n_predictions=12,
    learning_rate: float = 3e-4,
    ml_preparation_ref: str = None,
    dataset_id: str = None,
    dataset_name: str = None,
    target_cols=None,
    selected_targets=None,
    **_ignored,
):
    """
    Adapter: bridges standard registry builder to AXE Chimera (V8 Hybrid CNN+Towers+LSTM).
    """
    import json
    import logging
    from pathlib import Path
    import importlib.util

    logger = logging.getLogger(__name__)

    _this_dir   = Path(__file__).resolve().parent
    _backend    = _this_dir.parent.parent.parent
    _data_dir   = _backend / "Backend" / "data"
    _module_path = _data_dir / "baseline_encoder_v8_hybrid.py"

    if not _module_path.exists():
        raise FileNotFoundError(f"AXE Chimera source not found at {_module_path}.")

    _spec = importlib.util.spec_from_file_location("_axe_chimera_v8_hybrid", str(_module_path))
    _mod  = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

    build_model_v8_hybrid = _mod.build_model_v8_hybrid
    compile_model_v8_hybrid = _mod.compile_model_v8_hybrid
    build_category_indices = _mod.build_category_indices

    _ds_name = (dataset_name or ml_preparation_ref or dataset_id or "").replace("/", "_").strip()
    _search_roots = [_data_dir / "ml_cache", _backend / "data" / "ml_cache"]
    feature_index_map_path = None
    for root in _search_roots:
        if _ds_name:
            candidate = root / _ds_name / "feature_index_map.json"
            if candidate.exists():
                feature_index_map_path = candidate
                break
        candidates = sorted(root.glob("*/feature_index_map.json"))
        if candidates:
            feature_index_map_path = candidates[-1]
            break

    if feature_index_map_path is None:
        raise FileNotFoundError(f"[AXE Chimera] Cannot locate feature_index_map.json for '{_ds_name}'.")

    with open(feature_index_map_path) as _f:
        _map_data = json.load(_f)
    feature_index_map = _map_data.get("feature_index_map", _map_data)
    feature_names = [k for k, _ in sorted(feature_index_map.items(), key=lambda kv: kv[1])]

    category_indices = build_category_indices(feature_names)
    CONTINUOUS_CATS = {"candle", "momentum", "flow", "structure", "conviction"}
    continuous_feature_indices = sorted(
        idx for cat in CONTINUOUS_CATS for idx in category_indices.get(cat, [])
    )
    close_col_idx = feature_index_map.get("Close", 3)

    forecast_steps = int(n_predictions) if n_predictions and n_predictions > 1 else 12

    model = build_model_v8_hybrid(
        input_shape=input_shape,
        output_dim=forecast_steps,
        continuous_feature_indices=continuous_feature_indices,
        category_indices=category_indices,
        close_col_idx=close_col_idx,
    )
    model = compile_model_v8_hybrid(model, learning_rate=learning_rate)

    logger.info(f"[AXE Chimera] ✅ Model built and compiled — {model.count_params():,} parameters")
    return model


def build_axe_genesis_v2(
    input_shape,
    n_predictions=12,
    learning_rate: float = 3e-4,
    **_ignored,
):
    """
    Adapter: bridges standard registry builder interface to AXE Genesis V2 Runtime.
    """
    from app.core.ml.axe_genesis_v2_runtime import AXEGenesisV2Runtime
    return AXEGenesisV2Runtime()


