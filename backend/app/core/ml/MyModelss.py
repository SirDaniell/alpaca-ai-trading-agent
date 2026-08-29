import os



import tqdm
import logging
import joblib
#from keras._tf_keras.keras.models import load_model
from sklearn.model_selection import GridSearchCV
#from tensorflow.keras import backend, models, layers
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.regularizers import l2
from keras.optimizers import Adam, AdamW

import keras
#from keras._tf_keras.keras import models, layers, optimizers, backend
from sklearn.model_selection import KFold
from app.core.ml.PositionEncoding import PositionalEncoding
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
from keras import backend , models, layers, regularizers, optimizers, Input
#from keras._tf_keras.keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import tensorflow as tf
tf.compat.v1.reset_default_graph()


import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='keras.backend.common.global_state')

from keras.optimizers import schedules

def build_lstm_transformer_model(input_shape, lstm_units=64, attention_heads=4, dropout_rate=0.2):
    backend.clear_session()
    
    inputs = layers.Input(shape=input_shape)
    
    # Bidirectional LSTM Layers with Batch Normalization
    x = layers.Bidirectional(layers.LSTM(units=lstm_units, return_sequences=True))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(units=lstm_units, return_sequences=True))(x)
    x = layers.BatchNormalization()(x)
    
    # Multi-Head Attention Layer
    attention = layers.MultiHeadAttention(num_heads=attention_heads, key_dim=lstm_units)(x, x)
    
    # Residual Connection
    x = layers.Add()([x, attention])
    
    # Another Bidirectional LSTM Layer with Batch Normalization
    x = layers.Bidirectional(layers.LSTM(units=lstm_units))(x)
    x = layers.BatchNormalization()(x)
    
    # Fully Connected Layers
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output Layer
    outputs = layers.Dense(2, activation='softmax')(x)  # 2 classes for categorical classification
    
    # Create Model
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

def build_enhanced_lstm_modelx(input_shape, lstm_units=64, attention_heads=4, dropout_rate=0.2, l2_reg=0.001):
    backend.clear_session()
    
    inputs = layers.Input(shape=input_shape)
    
    # Bidirectional LSTM Layers with Batch Normalization
    x = layers.LSTM(units=lstm_units, return_sequences=True, recurrent_dropout=0.1,
                                         kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LSTM(units=lstm_units, return_sequences=True, recurrent_dropout=0.1,
                                         kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    
    # Multi-Head Attention Layer
    attention = layers.MultiHeadAttention(num_heads=attention_heads, key_dim=lstm_units)(x, x)
    
    # Residual Connection
    x = layers.Add()([x, attention])
    
    # Another Bidirectional LSTM Layer with Batch Normalization
    x = layers.Bidirectional(layers.LSTM(units=lstm_units, kernel_regularizer=regularizers.l2(l2_reg)))(x)
    x = layers.BatchNormalization()(x)

    # Another Bidirectional LSTM Layer with Batch Normalization
    x = layers.Bidirectional(layers.LSTM(units=lstm_units, kernel_regularizer=regularizers.l2(l2_reg)))(x)
    x = layers.BatchNormalization()(x)
    
    # Fully Connected Layers
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output Layer
    outputs = layers.Dense(2, activation='softmax')(x)  # 2 classes for categorical classification
    
    # Learning Rate Schedule
    lr_schedule = schedules.ExponentialDecay(initial_learning_rate=0.001, 
                                             decay_steps=10000, decay_rate=0.9)
    
    # Create Model
    optimizer = optimizers.RMSprop(learning_rate=0.001, clipvalue=1.0)
    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model
def build_bidirectional_lstm_model(input_shape, units=64, dropout_rate=0.2, l2_lambda=0.01):
    backend.clear_session()
    model = models.Sequential()

    # Adjusting input shape to (time_steps, features)
    model.add(layers.Input(shape=input_shape))

    # Bidirectional LSTM layers (maintaining 3D shape with return_sequences=True)
    for _ in range(3):
        model.add(layers.LSTM(128, return_sequences=True, 
                                                   kernel_regularizer=regularizers.l2(l2_lambda)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

    # Second set of LSTM layers (also maintaining 3D shape)
    for _ in range(3):
        model.add(layers.LSTM(units, return_sequences=True, 
                                                   kernel_regularizer=regularizers.l2(l2_lambda)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

    for _ in range(2):
        model.add(layers.LSTM(32, return_sequences=True, 
                                                   kernel_regularizer=regularizers.l2(l2_lambda)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

    for _ in range(2):
        model.add(layers.LSTM(16, return_sequences=True, 
                                                   kernel_regularizer=regularizers.l2(l2_lambda)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dropout_rate))

    # Third set of smaller LSTM layers (ending with return_sequences=False)
    model.add(layers.LSTM(units, return_sequences=False, 
                                               kernel_regularizer=regularizers.l2(l2_lambda)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dropout_rate))

    # Dense layers before output
    for _ in range(2):
        model.add(layers.Dense(32, kernel_regularizer=regularizers.l2(l2_lambda)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dropout_rate))

    for _ in range(2):
        model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(l2_lambda)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dropout_rate))
    for _ in range(2):
        model.add(layers.Dense(32, kernel_regularizer=regularizers.l2(l2_lambda)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dropout_rate))

    for _ in range(2):
        model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(l2_lambda)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(32, kernel_regularizer=regularizers.l2(l2_lambda)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dropout_rate))

    # Output layer with softmax activation for classification
    model.add(layers.Dense(2, activation='softmax'))

    # Compile the model with RMSprop optimizer, categorical crossentropy, and accuracy metric
    optimizer = optimizers.RMSprop(learning_rate=0.001, clipvalue=1.0)
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    return model


def build_dnn_model(input_shape, units=64, dropout_rate=0.2, l2_lambda=0.01):
    backend.clear_session()
    model = models.Sequential()

    # Input layer
    model.add(layers.Input(shape=(input_shape,)))

    # Hidden layers with BatchNorm before activation and LeakyReLU or SELU
    for _ in range(3):
        model.add(layers.Dense(128, kernel_regularizer=l2(l2_lambda)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

    # Second set of layers
    for _ in range(2):
        model.add(layers.Dense(units, kernel_regularizer=l2(l2_lambda)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dropout_rate))

    # Third set of layers (smaller units)
    for _ in range(2):
        model.add(layers.Dense(32, kernel_regularizer=l2(l2_lambda)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dropout_rate))

    # Fourth set of layers (smallest units)
    for _ in range(2):
        model.add(layers.Dense(16, kernel_regularizer=l2(l2_lambda)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dropout_rate))

    # Output layer with Softmax activation for classification
    model.add(layers.Dense(2, activation='softmax'))

    # Compile the model with gradient clipping and a learning rate schedule
    optimizer = optimizers.RMSprop(learning_rate=0.001, clipvalue=1.0)
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

def build_cnn_model_(input_shape, filters=64, kernel_size=3, dropout_rate=0.2):
    model = models.Sequential()

    model.add(layers.Input(shape=input_shape))
    
    # 1D Convolutional layers
    model.add(layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(filters=filters * 2, kernel_size=kernel_size, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    
    # Flatten the sequence
    model.add(layers.Flatten())
    
    # Fully connected layers
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    
    # Output layer
    model.add(layers.Dense(2, activation='softmax'))  # Assuming binary classification
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])

    return model

def build_transformer_dnn_model(input_shape, num_heads=4, dff=128, units=64, dropout_rate=0.18, l2_lambda=0.01):
    backend.clear_session()
    
    inputs = layers.Input(shape=(input_shape,))
    
    # Reshape input for multi-head attention
    x = layers.Reshape((1, input_shape))(inputs)
    
    # Multi-Head Self-Attention
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_shape)(x, x)
    attn_output = layers.Dropout(dropout_rate)(attn_output)
    attn_output = layers.LayerNormalization(epsilon=1e-6)(attn_output)
    
    # Feed-forward network (FFN)
    x_ffn = layers.Dense(dff, activation='relu', kernel_regularizer=l2(l2_lambda))(attn_output)
    x_ffn = layers.Dense(input_shape)(x_ffn)
    x = layers.Add()([attn_output, x_ffn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Flatten and Dense Layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=l2(l2_lambda))(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(units, activation='relu', kernel_regularizer=l2(l2_lambda))(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(units, activation='relu', kernel_regularizer=l2(l2_lambda))(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=l2(l2_lambda))(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Output layer
    outputs = layers.Dense(2, activation='sigmoid')(x)  # Assuming binary classification
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

def build_cnn_transformer_model(input_shape, num_heads=5, dff=128, dropout_rate=0.2, l2_reg=0.001):
    inputs = layers.Input(shape=input_shape)
    
    # Positional Encoding
    pos_encoding = PositionalEncoding(input_shape[0], input_shape[1])(inputs)
    
    # First Transformer Block
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[-1])(pos_encoding, pos_encoding)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    x_ffn = layers.Dense(dff, activation='gelu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x_ffn = layers.BatchNormalization()(x_ffn)
    x_ffn = layers.Dense(input_shape[-1], kernel_regularizer=regularizers.l2(l2_reg))(x_ffn)
    x = layers.Add()([x, x_ffn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Repeat the Transformer block
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[-1])(x, x)
    attn_output = layers.Dropout(dropout_rate)(attn_output)
    attn_output = layers.LayerNormalization(epsilon=1e-6)(attn_output)
    x_ffn = layers.Dense(dff, activation='gelu', kernel_regularizer=regularizers.l2(l2_reg))(attn_output)
    x_ffn = layers.BatchNormalization()(x_ffn)
    x_ffn = layers.Dense(input_shape[-1], kernel_regularizer=regularizers.l2(l2_reg))(x_ffn)
    x = layers.Add()([attn_output, x_ffn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Flatten the sequence
    x = layers.Flatten()(x)
    
    # Fully connected layers with L2 regularization and Batch Normalization
    x = layers.Dense(128, activation='gelu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

  
    
    # Output layer
    outputs = layers.Dense(2, activation='softmax')(x)  # Assuming binary classification
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001, weight_decay=l2_reg), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model


def build_cnn_transformer_modelx(input_shape, num_heads=5, dff=128, dropout_rate=0.2, l2_reg=0.001):
    inputs = layers.Input(shape=input_shape)
    
    # Multi-Head Self-Attention layers
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[-1])(inputs, inputs)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Feed-forward network (FFN) with L2 regularization and Batch Normalization
    x_ffn = layers.Dense(dff, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x_ffn = layers.BatchNormalization()(x_ffn)
    x_ffn = layers.Dense(input_shape[-1], kernel_regularizer=regularizers.l2(l2_reg))(x_ffn)
    x = layers.Add()([x, x_ffn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Repeat the Transformer block
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[-1])(x, x)
    attn_output = layers.Dropout(dropout_rate)(attn_output)
    attn_output = layers.LayerNormalization(epsilon=1e-6)(attn_output)
    x_ffn = layers.Dense(dff, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(attn_output)
    x_ffn = layers.BatchNormalization()(x_ffn)
    x_ffn = layers.Dense(input_shape[-1], kernel_regularizer=regularizers.l2(l2_reg))(x_ffn)
    x = layers.Add()([attn_output, x_ffn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Flatten the sequence
    x = layers.Flatten()(x)
    
    # Fully connected layers with L2 regularization and Batch Normalization
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = layers.Dense(2, activation='softmax')(x)  # Assuming binary classification
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

def build_cnn_transformer_modelx(input_shape, num_heads=5, dff=128, dropout_rate=0.2):
    inputs = layers.Input(shape=input_shape)
    
    # Multi-Head Self-Attention layers
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[-1])(inputs, inputs)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Feed-forward network (FFN)
    x_ffn = layers.Dense(dff, activation='relu')(x)
    x_ffn = layers.Dense(input_shape[-1])(x_ffn)
    x = layers.Add()([x, x_ffn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Repeat the Transformer block for better depth
    #for _ in range(1):  # 5 heads implies repeating 4 times to have a total of 5 attention heads

    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[-1])(x, x)
    attn_output = layers.Dropout(dropout_rate)(attn_output)
    attn_output = layers.LayerNormalization(epsilon=1e-6)(attn_output)
    x_ffn = layers.Dense(dff, activation='relu')(attn_output)
    x_ffn = layers.Dense(input_shape[-1])(x_ffn)
    x = layers.Add()([attn_output, x_ffn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Flatten the sequence
    x = layers.Flatten()(x)
    
    # Fully connected layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = layers.Dense(2, activation='softmax')(x)  # Assuming binary classification
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model


def build_cnn_transformer_modelx(input_shape, num_filters=64, kernel_size=3, num_heads=4, transformer_units=128, dropout_rate=0.2):
    backend.clear_session()
    
    inputs = layers.Input(shape=input_shape)
    
    # Convolutional Layer Block
    x = layers.Conv1D(filters=num_filters, kernel_size=kernel_size, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters=num_filters, kernel_size=kernel_size, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # Transformer Block
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=num_filters)(x, x)
    attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output)
    
    # Feed Forward Network in Transformer Block
    ffn = layers.Dense(transformer_units, activation='relu')(attention_output)
    ffn = layers.Dense(num_filters)(ffn)
    
    # Residual Connection and Layer Normalization
    x = layers.Add()([attention_output, ffn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Global Pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Fully Connected Layers
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output Layer
    outputs = layers.Dense(2, activation='softmax')(x)  # 2 classes for categorical classification
    
    # Create Model
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model


def build_cnn_model(input_shape, filters=200, kernel_size=3, dropout_rate=0.2):
    model = models.Sequential()

    model.add(layers.Input(shape=input_shape))
    
    # 1D Convolutional layers

    #for _ in range(3):
    model.add(layers.Conv1D(filters=128, kernel_size=kernel_size, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dropout_rate))

    #for _ in range(2):
    model.add(layers.Conv1D(filters=128, kernel_size=kernel_size, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dropout_rate))
        
    #for _ in range(2):
    model.add(layers.Conv1D(filters=64, kernel_size=kernel_size, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dropout_rate))

    #for _ in range(2):
    model.add(layers.Conv1D(filters=64, kernel_size=kernel_size, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Conv1D(filters=64, kernel_size=kernel_size, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dropout_rate))


    model.add(layers.Conv1D(filters=64, kernel_size=kernel_size, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.MaxPooling1D(pool_size=2))

    # Flatten the sequence
    model.add(layers.Flatten())
    
    # Fully connected layers
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    
    # Output layer
    model.add(layers.Dense(2, activation='softmax'))  # Assuming binary classification
    optimizer = optimizers.RMSprop(learning_rate=0.001, clipvalue=1.0)
    model.compile(optimizer=optimizer, 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])

    return model
def build_cnn_modelX(input_shape, filters=64, kernel_size=3, dropout_rate=0.2):
    model = models.Sequential()

    model.add(layers.Input(shape=input_shape))
    
    # 1D Convolutional layers

   
    model.add(layers.Conv1D(filters=128, kernel_size=kernel_size, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dropout_rate))

 
    model.add(layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))

    
    
    # Flatten the sequence
    model.add(layers.Flatten())
    
    # Fully connected layers
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    
    # Output layer
    model.add(layers.Dense(2, activation='softmax'))  # Assuming binary classification
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])

    return model
def build_enhanced_cnn_model(input_shape, filters=64, kernel_size=3, dropout_rate=0.2, l2_lambda=0.01):
    backend.clear_session()
    model = models.Sequential()

    # Input layer
    model.add(layers.Input(shape=input_shape))  # input_shape should be (time_steps, features)
    
    # First Conv1D Block with BatchNorm, LeakyReLU, and Dropout
    for _ in range(2):
        model.add(layers.Conv1D(filters=filters, kernel_size=kernel_size, 
                                kernel_regularizer=regularizers.l2(l2_lambda), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dropout_rate))
    
    # Second Conv1D Block with increased filters
    for _ in range(2):
        model.add(layers.Conv1D(filters=filters * 2, kernel_size=kernel_size, 
                                kernel_regularizer=regularizers.l2(l2_lambda), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dropout_rate))

    # Flatten layer to convert 3D feature maps to 1D
    
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    
    # Fully connected layers with BatchNorm, LeakyReLU, and Dropout
    for _ in range(2):
        model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(l2_lambda)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dropout_rate))

    # Smaller Dense layer for finer feature extraction
    model.add(layers.Dense(64, kernel_regularizer=regularizers.l2(l2_lambda)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dropout_rate))

    # Output layer for binary classification with softmax activation
    model.add(layers.Dense(2, activation='softmax'))

    # Compile the model with RMSprop optimizer (using gradient clipping) and categorical crossentropy
    optimizer = optimizers.RMSprop(learning_rate=0.001, clipvalue=1.0)
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    return model

def build_enhanced_cnn_modelx(input_shape, filters=64, kernel_size=3, dropout_rate=0.2, l2_lambda=0.01):
    backend.clear_session()
    model = models.Sequential()

    # Input layer
    model.add(layers.Input(shape=input_shape))
    
    # First Conv1D Block with BatchNorm, LeakyReLU, and Dropout
    for _ in range(3):
        model.add(layers.Conv1D(filters=filters, kernel_size=kernel_size, 
                                kernel_regularizer=regularizers.l2(l2_lambda), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Dropout(dropout_rate))
    
    # Second Conv1D Block with increased filters
    for _ in range(2):
        model.add(layers.Conv1D(filters=filters * 2, kernel_size=kernel_size, 
                                kernel_regularizer=regularizers.l2(l2_lambda), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Dropout(dropout_rate))

    # Flatten layer to convert 3D feature maps to 1D
    model.add(layers.Flatten())
    
    # Fully connected layers with BatchNorm, LeakyReLU, and Dropout
    for _ in range(2):
        model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(l2_lambda)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dropout_rate))

    # Smaller Dense layer for finer feature extraction
    model.add(layers.Dense(64, kernel_regularizer=regularizers.l2(l2_lambda)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dropout_rate))

    # Output layer for binary classification with softmax activation
    model.add(layers.Dense(2, activation='softmax'))

    # Compile the model with RMSprop optimizer (using gradient clipping) and categorical crossentropy
    optimizer = optimizers.RMSprop(learning_rate=0.001, clipvalue=1.0)
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    return model



def build_deep_cnn_lstmx(input_shape, n_predictions=5, conv_filters=64, lstm_units=100, learning_rate=0.001):
    
    # Clear any previous session
    backend.clear_session()

    # Initialize a Sequential model
    model = models.Sequential()

    # 1. Input Layer
    model.add(layers.Input(shape=input_shape))
    

    # 12. First LSTM Layer
    model.add(layers.LSTM(lstm_units,return_sequences= True ))
  

    model.add(layers.LSTM(lstm_units, ))
    model.add(layers.LSTM(lstm_units, ))
    model.add(layers.LSTM(lstm_units, ))
    model.add(layers.LSTM(lstm_units, ))
    model.add(layers.LSTM(lstm_units, ))


    #

    # 17. Output Layer for Classification or Regression
    model.add(layers.Dense(2, activation='softmax'))  # For regression

    # Compile the model with Adam optimizer
    optimizer = optimizers.RMSprop(learning_rate=0.001, clipvalue=1.0)
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    return model


def build_deep_cnn_lstm_(input_shape, n_predictions=2, conv_filters=64, lstm_units=100, learning_rate=0.001):
    # Clear any previous session
    backend.clear_session()

    # Initialize a Sequential model
    model = models.Sequential()

    # 1. Input Layer
    model.add(layers.Input(shape=input_shape))
    
    # 2. First Convolutional Block with Batch Normalization and ReLU activation
    model.add(layers.Conv1D(filters=conv_filters, kernel_size=7, strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    # 3. Second Convolutional Block
    model.add(layers.Conv1D(filters=conv_filters * 2, kernel_size=5, strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    # 4. Optional Pooling Layer
    model.add(layers.MaxPooling1D(pool_size=2))

    # 5. Third Convolutional Block
    model.add(layers.Conv1D(filters=conv_filters * 4, kernel_size=3, strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # 6. Recurrent Layer (LSTM)
    model.add(layers.LSTM(units=lstm_units, return_sequences=False))

    # 7. Fully Connected Layer
    model.add(layers.Dense(units=128, activation='relu'))

    # 8. Output Layer
    model.add(layers.Dense(units=n_predictions, activation='linear'))

    # Compile the model
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='mse')

    return model


def build_deep_cnn_lstm(input_shape, n_predictions=2, conv_filters=128, lstm_units=128, l2_lambda = 0.01):
    # Clear any previous session
    backend.clear_session()

    # Initialize a Sequential model
    model = models.Sequential()

    # 1. Input Layer
    model.add(layers.Input(shape=input_shape))
    
    # 2. Deep Convolutional Blocks
    for i in range(4):
        model.add(layers.Conv1D(filters=conv_filters * (i + 1), kernel_size=3, strides=1, kernel_regularizer=regularizers.l2(l2_lambda), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.ReLU())
        

        model.add(layers.Conv1D(filters=conv_filters * (i + 1), kernel_size=3, strides=1, kernel_regularizer=regularizers.l2(l2_lambda), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.ReLU())
        
        model.add(layers.MaxPooling1D(pool_size=2, padding='same'))

    for i in range(2):
        model.add(layers.Conv1D(filters=conv_filters * (i + 1), kernel_size=3, strides=1, kernel_regularizer=regularizers.l2(l2_lambda), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.ReLU())
        

        model.add(layers.Conv1D(filters=conv_filters * (i + 1), kernel_size=3, strides=1, kernel_regularizer=regularizers.l2(l2_lambda), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.ReLU())
        
        model.add(layers.MaxPooling1D(pool_size=2, padding='same'))

    for i in range(2):
        model.add(layers.Conv1D(filters=conv_filters * (i + 1), kernel_size=3, strides=1, kernel_regularizer=regularizers.l2(l2_lambda), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.ReLU())
        

        model.add(layers.Conv1D(filters=conv_filters * (i + 1), kernel_size=3, strides=1, kernel_regularizer=regularizers.l2(l2_lambda), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.ReLU())
        
        model.add(layers.MaxPooling1D(pool_size=2, padding='same'))

    for i in range(2):
        model.add(layers.Conv1D(filters=conv_filters * (i + 1), kernel_size=3, strides=1, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.ReLU())

        model.add(layers.Conv1D(filters=conv_filters * (i + 1), kernel_size=3, strides=1, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        
        
        model.add(layers.MaxPooling1D(pool_size=2, padding='same'))

    # 3. Recurrent Layer (LSTM)
    model.add(layers.LSTM(units=lstm_units, return_sequences=True))
    model.add(layers.LSTM(units=lstm_units//2, return_sequences=False))

    # 4. Fully Connected Layer
    model.add(layers.Dense(units=lstm_units//2, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(units=lstm_units//2, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(units=lstm_units//2, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(units=lstm_units//2, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))

    # 5. Output Layer
    model.add(layers.Dense(units=n_predictions, activation='softmax'))

    optimizer = optimizers.RMSprop(learning_rate=0.001, clipvalue=1.0)
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    return model

def build_conv2d_lstm(input_shape, n_predictions=2, conv_filters=64, lstm_units=100, learning_rate=0.001):
    # Clear any previous session
    backend.clear_session()

    # Initialize a Sequential model
    model = models.Sequential()

    # 1. Input Layer (Reshaped for Conv2D)
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Reshape((input_shape[0], input_shape[1], 1)))  # Reshape to (time_steps, num_features, 1)

    # 2. First Convolutional Block with Conv2D
    model.add(layers.Conv2D(filters=conv_filters, kernel_size=(3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # 3. Second Convolutional Block
    model.add(layers.Conv2D(filters=conv_filters * 2, kernel_size=(3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    # 4. Optional Pooling Layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # 5. Flatten the output before passing to LSTM
    model.add(layers.Flatten())
    model.add(layers.RepeatVector(1))  # Reshape to (batch_size, 1, flattened_size)

    # 6. Recurrent Layer (LSTM)
    model.add(layers.LSTM(units=lstm_units, return_sequences=False))

    # 7. Fully Connected Layer
    model.add(layers.Dense(units=128, activation='relu'))

    # 8. Output Layer
    model.add(layers.Dense(units=n_predictions, activation='linear'))

    # Compile the model
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='mse')

    return model

def build_conv3d_lstm(input_shape, n_predictions=2, conv_filters=64, lstm_units=100, learning_rate=0.001):
    # Clear any previous session
    backend.clear_session()

    # Initialize a Sequential model
    model = models.Sequential()

    # 1. Input Layer (Reshaped for Conv3D)
    model.add(layers.Reshape((input_shape[0], input_shape[1], 1)))  # Reshape to (time_steps, num_features, 1)

    # Assuming input_shape = (depth, time_steps, num_features) and adding a channel dimension
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Reshape((input_shape[0], input_shape[1], input_shape[2], 1)))  # Reshape to (depth, time_steps, num_features, 1)

    # 2. First Convolutional Block with Conv3D
    model.add(layers.Conv3D(filters=conv_filters, kernel_size=(3, 3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # 3. Second Convolutional Block
    model.add(layers.Conv3D(filters=conv_filters * 2, kernel_size=(3, 3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    # 4. Optional Pooling Layer
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))

    # 5. Flatten the output before passing to LSTM
    model.add(layers.Flatten())
    model.add(layers.RepeatVector(1))  # Reshape to (batch_size, 1, flattened_size)

    # 6. Recurrent Layer (LSTM)
    model.add(layers.LSTM(units=lstm_units, return_sequences=False))

    # 7. Fully Connected Layer
    model.add(layers.Dense(units=128, activation='relu'))

    # 8. Output Layer
    model.add(layers.Dense(units=n_predictions, activation='linear'))

    # Compile the model
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='mse')

    return model

def dilated_causal_cnn_with_bn(input_shape, num_filters=128, kernel_size=3, dilation_rates=[1, 2, 4, 8]):
    model = models.Sequential()
    
    # Input layer
    model.add(layers.InputLayer(shape=input_shape))
    
    # Stacking multiple causal Conv1D layers with dilation, batch normalization, and dropout
    for dilation_rate in dilation_rates:
        model.add(layers.Conv1D(filters=num_filters,
                                kernel_size=kernel_size,
                                padding='causal',   
                                dilation_rate=dilation_rate,
                                activation=None))   
        model.add(layers.BatchNormalization())      
        model.add(layers.LeakyReLU())  
        model.add(layers.Dropout(0.2))  
    
    # Optional: Adding another layer without dilation and with doubled filters
    model.add(layers.Conv1D(filters=num_filters * 2,   
                            kernel_size=kernel_size,
                            padding='causal',
                            activation=None))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))
    
    # Flatten and Dense layers for final prediction
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))   # Intermediate dense layer
    model.add(layers.Dense(2, activation='softmax')) # Softmax for 2-class classification
    
    # Compile the model with RMSprop optimizer and categorical crossentropy
    optimizer = optimizers.RMSprop(learning_rate=0.001, clipvalue=1.0)
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    return model


def dilated_causal_cnn_with_bn(input_shape, num_filters=256, kernel_size=3, l2_lambda =0.01,
                                dropout_rate = 0.2, dilation_rates=[1, 2, 4, 8, 16]):
    model = models.Sequential()
    
    # Input layer
    model.add(layers.InputLayer(shape=input_shape))
    
    # Initial convolutional layer
    model.add(layers.Conv1D(filters=num_filters,
                            kernel_size=kernel_size,
                            padding='causal',   
                            activation=None))  
    
    
    # Stacking multiple causal Conv1D layers with dilation and batch normalization
    for dilation_rate in dilation_rates:
        model.add(layers.Conv1D(filters=num_filters,
                                kernel_size=kernel_size,
                                padding='causal',   
                                dilation_rate=dilation_rate,
                                activation=None))   
        model.add(layers.BatchNormalization())      
        model.add(layers.LeakyReLU())  
        model.add(layers.Dropout(0.1))  
    
    # Adding residual connections
    residual = model.layers[-1].output
    model.add(layers.Conv1D(filters=num_filters,
                            kernel_size=1,  # 1x1 convolution for residual connection
                            padding='causal',
                            activation=None))
    model.add(layers.BatchNormalization())
    model.add(layers.Add()([model.output, residual]))  # Add the residual connection

    # Adding another layer without dilation and with doubled filters
    model.add(layers.Conv1D(filters=num_filters * 2,   
                            kernel_size=kernel_size,
                            padding='causal',
                            activation=None))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.05))
    
    # Global Average Pooling instead of Flatten
    model.add(layers.GlobalAveragePooling1D())
    
    # Additional Dense layers for more complexity
    # Dense layers after LSTM layers (fully connected)
    for _ in range(3):
        model.add(layers.Dense(128, kernel_regularizer=l2(l2_lambda)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dropout_rate))

    # Second set of layers
    for _ in range(2):
        model.add(layers.Dense(64, kernel_regularizer=l2(l2_lambda)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dropout_rate))

    # Third set of layers (smaller units)
    for _ in range(2):
        model.add(layers.Dense(32, kernel_regularizer=l2(l2_lambda)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dropout_rate))

    # Fourth set of layers (smallest units)
    for _ in range(2):
        model.add(layers.Dense(16, kernel_regularizer=l2(l2_lambda)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dropout_rate))


    model.add(layers.Dense(2, activation='softmax'))  # Final output layer for classification
    
    # Compile the model with RMSprop optimizer and categorical crossentropy
    optimizer = optimizers.RMSprop(learning_rate=0.001, clipvalue=1.0)
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    return model


def dilated_causal_cnn_with_bn(input_shape, num_filters=128, kernel_size=3, l2_lambda = 0.01, dilation_rates=[1, 2, 4, 8, 16]):
    inputs = Input(shape=input_shape)
    
    x = layers.Conv1D(filters=num_filters, kernel_size=kernel_size, padding='causal', activation=None)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.2)(x)
    
    # Keep track of the initial input for residual connection
    residual = x
    
    # Apply dilated convolutions
    for dilation_rate in dilation_rates:
        x = layers.Conv1D(filters=num_filters, kernel_size=kernel_size, padding='causal', dilation_rate=dilation_rate, kernel_regularizer=regularizers.l2(l2_lambda), activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.2)(x)
    
    # Apply a residual connection
    x = layers.Conv1D(filters=num_filters, kernel_size=1, padding='causal', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])
    
    # Additional layers
    x = layers.Conv1D(filters=num_filters * 2,  kernel_size=kernel_size, padding='causal', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.2)(x)
    
    # Global Average Pooling instead of Flatten
    x = layers.GlobalAveragePooling1D()(x)
    
    # Additional Dense layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(2, activation='softmax')(x)  # Final output layer for classification
    
    # Create and compile the model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    optimizer = optimizers.RMSprop(learning_rate=0.001, clipvalue=1.0)
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

    
 
def multihead_dilated_causal_cnn_lstm(input_shape, num_filters=128, kernel_size=3, l2_lambda=0.01, dilation_rates=[1, 2, 4, 8, 16], lstm_units=64):
    inputs = Input(shape=input_shape)
    
    # First CNN branch with different kernel sizes and dilation rates (multi-head)
    branches = []
    for dilation_rate in dilation_rates:
        x = layers.Conv1D(filters=num_filters, kernel_size=kernel_size, padding='causal', 
                          dilation_rate=dilation_rate, activation=None)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.2)(x)
        branches.append(x)
    
    # Concatenate the outputs of the CNN branches (multi-head)
    x = layers.Concatenate()(branches) if len(branches) > 1 else branches[0]
    
    # Residual connection (if needed)
    residual = layers.Conv1D(filters=num_filters, kernel_size=1, padding='causal', activation=None)(inputs)
    residual = layers.BatchNormalization()(residual)
    x = layers.Add()([x, residual])
    
    # Further CNN processing if needed
    x = layers.Conv1D(filters=num_filters * 2, kernel_size=kernel_size, padding='causal', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.2)(x)

    # Adding LSTM layer to capture temporal dependencies
    x = layers.LSTM(lstm_units, return_sequences=False)(x)
    
    # Additional Dense layers after LSTM
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Final output layer for classification (softmax for multiclass)
    outputs = layers.Dense(2, activation='softmax')(x)
    
    # Create and compile the model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    optimizer = optimizers.RMSprop(learning_rate=0.001, clipvalue=1.0)
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model
def multihead_dilated_causal_cnn_lstm(input_shape, num_filters=128, kernel_size=3, l2_lambda=0.01, dilation_rates=[1, 2, 4, 8, 16], lstm_units=64):
    inputs = Input(shape=input_shape)
    
    # First CNN branch with different kernel sizes and dilation rates (multi-head)
    branches = []
    for dilation_rate in dilation_rates:
        x = layers.Conv1D(filters=num_filters, kernel_size=kernel_size, padding='causal', 
                          dilation_rate=dilation_rate, activation=None)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.2)(x)
        branches.append(x)
    
    # Concatenate the outputs of the CNN branches (multi-head)
    x = layers.Concatenate()(branches) if len(branches) > 1 else branches[0]
    
    # Residual connection (matching number of filters)
    residual = layers.Conv1D(filters=x.shape[-1], kernel_size=1, padding='causal', activation=None)(inputs)
    residual = layers.BatchNormalization()(residual)
    x = layers.Add()([x, residual])  # Now the shapes should match

    # Further CNN processing if needed
    x = layers.Conv1D(filters=num_filters * 2, kernel_size=kernel_size, padding='causal', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.2)(x)

    # Adding LSTM layer to capture temporal dependencies
    x = layers.LSTM(lstm_units, return_sequences=False)(x)
    
    # Additional Dense layers after LSTM
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Final output layer for classification (softmax for multiclass)
    outputs = layers.Dense(2, activation='softmax')(x)
    
    # Create and compile the model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    optimizer = optimizers.RMSprop(learning_rate=0.001, clipvalue=1.0)
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

def multihead_dilated_causal_cnn_lstm_attention(input_shape, num_filters=128, kernel_size=3, l2_lambda=0.01, dropout_rate = 0.2,
                                                dilation_rates=[1, 2, 4, 8, 16], lstm_units=64, num_heads=8):
    inputs = Input(shape=input_shape)
    
    # First CNN branch with different dilation rates (multi-head CNN)
    branches = []
    for dilation_rate in dilation_rates:
        x = layers.Conv1D(filters=num_filters, kernel_size=kernel_size, padding='same', 
                          dilation_rate=dilation_rate, activation=None)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.2)(x)
        branches.append(x)
    
    # Concatenate the outputs of the CNN branches (multi-head CNN)
    x = layers.Concatenate()(branches) if len(branches) > 1 else branches[0]
    
    # Residual connection (matching number of filters)
    residual = layers.Conv1D(filters=x.shape[-1], kernel_size=1, padding='same', activation=None)(inputs)
    residual = layers.BatchNormalization()(residual)
    x = layers.Add()([x, residual])  # Now the shapes should match

    # Multi-Head Self-Attention layer
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=num_filters)(x, x)
    x = layers.Add()([x, attention_output])  # Add the attention output to the CNN features
    
    # Further processing after attention
    x = layers.Conv1D(filters=num_filters * 2, kernel_size=kernel_size, padding='causal', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.2)(x)
    
    # LSTM layer to capture temporal dependencies
    x = layers.LSTM(lstm_units, return_sequences=False)(x)
    
    # Additional Dense layers after LSTM
    # 5. Dense Layers with Loops (preserve original structure)
    for _ in range(3):
        x = layers.Dense(128, kernel_regularizer=l2(l2_lambda))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)

    for _ in range(2):
        x = layers.Dense(64, kernel_regularizer=l2(l2_lambda))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(dropout_rate)(x)

    for _ in range(2):
        x = layers.Dense(32, kernel_regularizer=l2(l2_lambda))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(dropout_rate)(x)

    for _ in range(2):
        x = layers.Dense(16, kernel_regularizer=l2(l2_lambda))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(dropout_rate)(x)

    
    # Final output layer for classification (softmax for multiclass)
    outputs = layers.Dense(2, activation='softmax')(x)
    
    # Create and compile the model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    optimizer = optimizers.RMSprop(learning_rate=0.001, clipvalue=1.0)
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model



def build_conv1d_lstm_dnn_model(input_shape, num_filters=64, kernel_size=3, lstm_units=64, 
                                units=64, dropout_rate=0.2, l2_lambda=0.01):
    backend.clear_session()
    model = models.Sequential()

    # Input layer for sequential data (time_steps, features)
    model.add(layers.Input(shape=input_shape))  # Input should be 3D: (time_steps, features)

    # Conv1D layer to extract short-term temporal features
    model.add(layers.Conv1D(filters=num_filters, kernel_size=kernel_size, padding='same', activation=None))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv1D(filters=num_filters, kernel_size=kernel_size, padding='same', activation=None))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # LSTM layers to capture long-term temporal dependencies
    model.add(layers.LSTM(lstm_units, return_sequences=True))  # First LSTM layer
    model.add(layers.BatchNormalization())
    model.add(layers.LSTM(lstm_units, return_sequences=False))  # Second LSTM layer
    model.add(layers.BatchNormalization())

    model.add(layers.Dropout(0.1))

    # Dense layers after LSTM layers (fully connected)
    for _ in range(3):
        model.add(layers.Dense(128, kernel_regularizer=l2(l2_lambda)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

    # Second set of layers
    for _ in range(2):
        model.add(layers.Dense(units, kernel_regularizer=l2(l2_lambda)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dropout_rate))

    # Third set of layers (smaller units)
    for _ in range(2):
        model.add(layers.Dense(32, kernel_regularizer=l2(l2_lambda)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dropout_rate))

    # Fourth set of layers (smallest units)
    for _ in range(2):
        model.add(layers.Dense(16, kernel_regularizer=l2(l2_lambda)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dropout_rate))

    # Output layer with Softmax activation for classification
    model.add(layers.Dense(2, activation='softmax'))

    # Compile the model with gradient clipping and a learning rate schedule
    optimizer = optimizers.RMSprop(learning_rate=0.001, clipvalue=1.0)
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    return model


class SplitAndPadLayer(layers.Layer):
    def __init__(self, group_sizes, max_group_size, **kwargs):
        super(SplitAndPadLayer, self).__init__(**kwargs)
        self.group_sizes = group_sizes
        self.max_group_size = max_group_size

    def call(self, inputs):
        # Split the input based on computed group sizes
        splits = tf.split(inputs, num_or_size_splits=self.group_sizes, axis=-1)

        # Pad each split to match the max_group_size
        padded_splits = [
            tf.pad(split, paddings=[[0, 0], [0, 0], [0, self.max_group_size - tf.shape(split)[-1]]])
            for split in splits
        ]
        

        return padded_splits


def build_tcn_block(input_shape, filters, kernel_size,  l2_lambda=0.01, dropout_rate = 0.2,
                                                dilation_rates=[1, 2, 4, 8, 16],):
    

    inputs = layers.Input(shape=input_shape)
    
    x = inputs
    for dilation_rate in dilation_rates:
        x = layers.Conv1D(filters, kernel_size, padding='causal',  kernel_regularizer=l2(l2_lambda), dilation_rate=dilation_rate)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(dropout_rate)(x)
    
    # Check if the number of filters in inputs and x match
    if inputs.shape[-1] != filters:
        # Use Conv1D with kernel_size=1 to match the number of filters
        inputs = layers.Conv1D(filters, kernel_size=1, padding='same')(inputs)
    
    # Residual connection
    x = layers.Add()([x, inputs])
    x = layers.ReLU()(x)
    
    return models.Model(inputs=inputs, outputs=x)

def create_tcn_model(sequence_length, feature_count, num_groups=4, dropout_rate = 0.2, l2_lambda=0.01):
    # Calculate base group size
    group_size = feature_count // num_groups
    remainder = feature_count % num_groups

    # Determine sizes of each group
    group_sizes = [group_size + 1 if i < remainder else group_size for i in range(num_groups)]
    max_group_size = max(group_sizes)

    # Create TCN blocks, each processing a max-sized group
    tcn_blocks = []
    for _ in range(num_groups):
        tcn_block = build_tcn_block_((sequence_length, max_group_size), filters=34, kernel_size=2, dilation_rates=[1, 2, 4])
        tcn_blocks.append(tcn_block)

    # Main input for the entire sequence and feature set
    main_input = layers.Input(shape=(sequence_length, feature_count))
   

    # Apply your custom layer wherever you want to use the TensorFlow function
    # Use the custom layer to split and pad
    split_and_pad_layer = SplitAndPadLayer(group_sizes, max_group_size)
    padded_splits = split_and_pad_layer(main_input)

    

    # Pass each padded split through its corresponding TCN block
    tcn_outputs = [tcn_blocks[i](padded_splits[i]) for i in range(num_groups)]

    # Concatenate outputs from all TCN blocks
    merged_output = layers.Concatenate()(tcn_outputs)

    # Fully connected network
    x = layers.Flatten()(merged_output)
    
    # 5. Dense Layers with Loops (preserve original structure)
    for _ in range(3):
        x = layers.Dense(128, kernel_regularizer=l2(l2_lambda))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(dropout_rate)(x)

    for _ in range(2):
        x = layers.Dense(64, kernel_regularizer=l2(l2_lambda))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(dropout_rate)(x)

    for _ in range(2):
        x = layers.Dense(32, kernel_regularizer=l2(l2_lambda))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(dropout_rate)(x)

    for _ in range(2):
        x = layers.Dense(16, kernel_regularizer=l2(l2_lambda))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(dropout_rate)(x)


    

    # Output layer (adjust units for your task, e.g., regression or classification)
    output = layers.Dense(1, activation='sigmoid')(x)  # For binary classification

    # Build the full model
    model = models.Model(inputs=main_input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def build_tcn_block_(input_shape, filters, kernel_size, l2_lambda=0.01, dropout_rate=0.2,
                    dilation_rates=[1, 2, 4, 8, 16]):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    # TCN Block with dilation rates
    for dilation_rate in dilation_rates:
        x = layers.Conv1D(filters, kernel_size, padding='causal', 
                          kernel_regularizer=l2(l2_lambda), 
                          dilation_rate=dilation_rate)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(dropout_rate)(x)
    
    # Residual connection: Match the number of filters between `x` and `inputs`
    if inputs.shape[-1] != filters:
        # Use Conv1D with kernel_size=1 to match the number of filters if necessary
        residual = layers.Conv1D(filters, kernel_size=1, padding='same')(inputs)
    else:
        residual = inputs

    # Add the residual connection before moving into dense layers
    x = layers.Add()([x, residual])
    x = layers.ReLU()(x)
    
    # Now proceed with the dense layers
    x = layers.Dense(32, kernel_regularizer=l2(l2_lambda))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(32, kernel_regularizer=l2(l2_lambda))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(16, kernel_regularizer=l2(l2_lambda))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(16, kernel_regularizer=l2(l2_lambda))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(dropout_rate)(x)

    return models.Model(inputs=inputs, outputs=x)



def create_tcn_model_(sequence_length, feature_count, num_groups=4, dropout_rate = 0.2, l2_lambda=0.01):
    # Calculate base group size
    group_size = feature_count // num_groups
    remainder = feature_count % num_groups

    # Determine sizes of each group
    group_sizes = [group_size + 1 if i < remainder else group_size for i in range(num_groups)]
    max_group_size = max(group_sizes)

    # Create TCN blocks, each processing a max-sized group
    tcn_blocks = []
    for _ in range(num_groups):
        tcn_block = build_tcn_block_((sequence_length, max_group_size), filters=34, kernel_size=2, dilation_rates=[1, 2, 4])
        tcn_blocks.append(tcn_block)

    # Main input for the entire sequence and feature set
    main_input = layers.Input(shape=(sequence_length, feature_count))
   
    # Input layer
    main_input = layers.Input(shape=(sequence_length, feature_count))
    
    # Conv1D layer
    conv_output = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(main_input)

    # Custom layer for splitting and padding
    split_and_pad_layer = SplitAndPadLayer(group_sizes, max_group_size)
    padded_splits = split_and_pad_layer(conv_output)
        

    # Pass each padded split through its corresponding TCN block
    tcn_outputs = [tcn_blocks[i](padded_splits[i]) for i in range(num_groups)]

    # Concatenate outputs from all TCN blocks
    merged_output = layers.Concatenate()(tcn_outputs)

    # Fully connected network
    x = layers.Flatten()(merged_output)

    # Now proceed with the dense layers
    x = layers.Dense(32, kernel_regularizer=l2(l2_lambda))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(32, kernel_regularizer=l2(l2_lambda))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(16, kernel_regularizer=l2(l2_lambda))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(16, kernel_regularizer=l2(l2_lambda))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    
    # Output layer (adjust units for your task, e.g., regression or classification)
    output = layers.Dense(2, activation='softmax')(x)  # For binary classification

    # Build the full model
    model = models.Model(inputs=main_input, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

class SplitAndPadLayer(layers.Layer):
    def __init__(self, group_sizes, max_group_size, **kwargs):
        super(SplitAndPadLayer, self).__init__(**kwargs)
        self.group_sizes = group_sizes
        self.max_group_size = max_group_size

    def call(self, inputs):
        splits = []
        start = 0
        for size in self.group_sizes:
            if size > 0:  # Only process non-zero sized groups
                split = inputs[:, :, start:start + size]  # Extract the group
                if size < self.max_group_size:  # Only pad if necessary
                    pad_size = self.max_group_size - size
                    padded_split = tf.pad(split, paddings=[[0, 0], [0, 0], [0, pad_size]])  # Pad to max group size
                else:
                    padded_split = split  # No padding needed
                splits.append(padded_split)
                start += size
            else:
                # Handle zero-sized group (skip or log)
                print(f"Skipping group with size 0 at position {start}")

            #print(f"Split shape: {split.shape}, Padded shape: {padded_split.shape}")
        
        return splits

    def compute_output_shape(self, input_shape):
        # Each split will be padded to max_group_size
        return [tf.TensorShape([input_shape[0], input_shape[1], self.max_group_size]) for size in self.group_sizes if size > 0]

def create_custom_tcn_model(sequence_length, feature_count, num_groups=4, dropout_rate=0.2, l2_lambda=0.01):
    # Group size calculation
    group_size = feature_count // num_groups
    remainder = feature_count % num_groups

    # Ensure no group is empty
    group_sizes = [group_size + 1 if i < remainder else group_size for i in range(num_groups)]
    group_sizes = [size for size in group_sizes if size > 0]  # Remove any zero-sized groups

    max_group_size = max(group_sizes) if group_sizes else 1  # Set a default max size if no groups exist

    # Main input layer
    main_input = layers.Input(shape=(sequence_length, feature_count))

    # Main Conv1D layer to capture overall features
    main_conv_output = layers.Conv1D(filters=feature_count, kernel_size=3, padding='same', activation='relu')(main_input)
    main_conv_output = layers.BatchNormalization()(main_conv_output)
    main_conv_output = layers.LeakyReLU()(main_conv_output)
    main_conv_output = layers.Dropout(dropout_rate)(main_conv_output)

    # Custom layer for splitting and padding
    split_and_pad_layer = SplitAndPadLayer(group_sizes, max_group_size)
    padded_splits = split_and_pad_layer(main_conv_output)

    # TCN blocks for each group
    tcn_blocks = []
    for _ in range(len(group_sizes)):
        tcn_block = build_tcn_block((sequence_length, max_group_size), filters=max_group_size, kernel_size=2)
        tcn_blocks.append(tcn_block)

    # Process each group with its TCN block
    tcn_outputs = [tcn_blocks[i](padded_splits[i]) for i in range(len(padded_splits))]

    # Combine TCN outputs with the main Conv1D output
    combined_output = layers.Concatenate()([main_conv_output] + tcn_outputs)

    # Flatten the combined output
    x = layers.Flatten()(combined_output)

    # Dense layers for final prediction
    for units in [128, 128, 64, 64]:
        x = layers.Dense(units, kernel_regularizer=l2(l2_lambda))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(dropout_rate)(x)

    # Output layer for binary classification (adjust units and activation for other tasks)
    output = layers.Dense(1, activation='sigmoid')(x)

    # Build and compile the model
    model = models.Model(inputs=main_input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def build_tcn_block(input_shape, filters, kernel_size, l2_lambda=0.01, dropout_rate=0.2, dilation_rates=[1, 2, 4, 8, 16, 32]):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    # TCN block with dilation rates
    for dilation_rate in dilation_rates:
        x = layers.Conv1D(filters, kernel_size, padding='causal', kernel_regularizer=l2(l2_lambda), dilation_rate=dilation_rate)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(dropout_rate)(x)
    
    # Residual connection
    if inputs.shape[-1] != filters:
        residual = layers.Conv1D(filters, kernel_size=1, padding='same')(inputs)
    else:
        residual = inputs

    x = layers.Add()([x, residual])
    x = layers.ReLU()(x)

    # Additional Dense layers
    for units in [64, 64, 64, 64]:
        x = layers.Dense(units, kernel_regularizer=l2(l2_lambda))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(dropout_rate)(x)

    return models.Model(inputs=inputs, outputs=x)


def create_custom_lstm_model_(input_shape, num_features, n_groups = 4, lstm_units = 128, dense_units = 64, l2_lambda = 0.01, dropout_rate = 0.2):
    # Define the input
    input_layer = layers.Input(shape=input_shape)
    
    # Split the features into n groups
    group_size = num_features // n_groups
    lstm_outputs = []
    
    # Apply a different LSTM to each feature group
    lstm_output_1 = layers.LSTM(lstm_units, return_sequences=False)(input_layer)
    for i in range(n_groups):
        start = i * group_size
        end = start + group_size
        
        # Slice the input to get the respective group
        group_input = layers.Lambda(lambda x: x[:, :, start:end])(input_layer)
        
        # Define a separate LSTM for each group
        outputs = layers.LSTM(lstm_units, return_sequences=False)(group_input)
        # Additional Dense layers
        for units in [32, 32, 16, 16]:
            outputs = layers.Dense(units, kernel_regularizer=l2(l2_lambda))(outputs)
            outputs = layers.BatchNormalization()(outputs)
            outputs = layers.LeakyReLU()(outputs)
            outputs = layers.Dropout(dropout_rate)(outputs)
            lstm_outputs.append(outputs)
    
    # Concatenate the outputs of all LSTMs
    concatenated_output = layers.Concatenate()(lstm_outputs, lstm_output_1)
    
    # Dense layer for final prediction
    dense_output = layers.Dense(dense_units, activation='relu')(concatenated_output)
    output_layer = layers.Dense(2, activation='softmax')(dense_output)
    
    # Create the model
    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def create_custom_lstm_model__(input_shape, num_features, n_groups=4, lstm_units=128, dense_units=64):
    # Define the input
    input_layer = layers.Input(shape=input_shape)
    
    # Split the features into n groups
    group_size = num_features // n_groups
    lstm_outputs = []
    
    for i in range(n_groups):
        start = i * group_size
        end = start + group_size
        
        # Slice the input to get the respective group
        group_input = layers.Lambda(lambda x: x[:, :, start:end])(input_layer)
        
        # Define a separate LSTM for each group
        lstm_output = layers.LSTM(lstm_units, return_sequences=False)(group_input)
        lstm_outputs.append(lstm_output)
    
    # Concatenate the outputs of all LSTMs
    concatenated_output = layers.Concatenate()(lstm_outputs)
    
    # Dense layer for final prediction
    dense_output = layers.Dense(dense_units, activation='relu')(concatenated_output)
    output_layer = layers.Dense(2, activation='softmax')(dense_output)
    
    # Create the model
    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def create_custom_lstm_model(input_shape, num_features, n_groups = 4, lstm_units = 128, dense_units = 64, l2_lambda = 0.01, dropout_rate = 0.2):
    # Define the input
    input_layer = layers.Input(shape=input_shape)
    
    # Split the features into n groups
    group_size = num_features // n_groups
    lstm_outputs = []
    
    # Apply the first LSTM to the full input
    lstm_output_1 = layers.LSTM(lstm_units, return_sequences=False)(input_layer)
    
    # Apply a different LSTM to each feature group
    for i in range(n_groups):
        start = i * group_size
        end = start + group_size
        
        # Slice the input to get the respective group
        group_input = layers.Lambda(lambda x: x[:, :, start:end])(input_layer)
        
        # Define a separate LSTM for each group
        
        outputs = layers.LSTM(lstm_units, return_sequences=False)(group_input)
        # Additional Dense layers
        for units in [32, 32, 16, 16]:
            outputs = layers.Dense(units, kernel_regularizer=l2(l2_lambda))(outputs)
            outputs = layers.BatchNormalization()(outputs)
            outputs = layers.LeakyReLU()(outputs)
            outputs = layers.Dropout(dropout_rate)(outputs)
        lstm_outputs.append(outputs)
    
    # Concatenate the outputs of all LSTMs, including lstm_output_1
    concatenated_output = layers.Concatenate()([lstm_output_1] + lstm_outputs)
    
    # Dense layer for final prediction
    outputs = layers.Dense(dense_units, activation='relu')(concatenated_output)
    outputs = layers.BatchNormalization()(outputs)
    outputs = layers.LeakyReLU()(outputs)
    outputs = layers.Dropout(dropout_rate)(outputs)

    outputs = layers.Dense(dense_units//2, activation='relu')(outputs)
    outputs = layers.BatchNormalization()(outputs)
    outputs = layers.LeakyReLU()(outputs)
    outputs = layers.Dropout(dropout_rate)(outputs)


    output_layer = layers.Dense(2, activation='softmax')(outputs)
    
    # Create the model
    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def create_custom_lstm_model(input_shape, num_features, attention_heads=4, n_groups=20, lstm_units=128, dense_units=64, l2_lambda=0.01, dropout_rate=0.2):
    # Define the input
    input_layer = layers.Input(shape=input_shape)
    
    # Split the features into n groups
    group_size = num_features // n_groups
    lstm_outputs = []
    
    # Apply the first LSTM to the full input
  


    x = layers.Bidirectional(layers.LSTM(units=lstm_units, return_sequences=True))(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(dropout_rate)(x)


    x = layers.Bidirectional(layers.LSTM(units=lstm_units, return_sequences=False))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Multi-Head Attention Layer
    attention = layers.MultiHeadAttention(num_heads=attention_heads, key_dim=lstm_units)(x, x)
    
    # Residual Connection
    x = layers.Add()([x, attention])
    # Apply a different LSTM to each feature group


    for i in range(n_groups):
        start = i * group_size
        end = start + group_size
        
        # Slice the input to get the respective group
        group_input = layers.Lambda(lambda x: x[:, :, start:end])(input_layer)
        
        # Define a separate LSTM for each group
        outputs = layers.LSTM(lstm_units, return_sequences=False)(group_input)
        outputs = layers.BatchNormalization()(outputs)
        outputs = layers.LeakyReLU()(outputs)
        outputs = layers.Dropout(dropout_rate)(outputs)

        # Multi-Head Attention Layer
        attention = layers.MultiHeadAttention(num_heads=attention_heads, key_dim=lstm_units)(outputs, outputs)
        
        # Residual Connection
        outputs = layers.Add()([outputs, attention])
    
        # Additional Dense layers with Leaky ReLU, Batch Normalization, Dropout
        #for units in [32, 32, 16, 16]:
        outputs = layers.Dense(64, kernel_regularizer=l2(l2_lambda))(outputs)
        outputs = layers.BatchNormalization()(outputs)
        outputs = layers.LeakyReLU()(outputs)
        outputs = layers.Dropout(dropout_rate)(outputs)
        
        lstm_outputs.append(outputs)
    
    # Concatenate the outputs of all LSTMs, including lstm_output_1
    concatenated_output = layers.Concatenate()([x] + lstm_outputs)

    # Flatten the concatenated output to ensure it's 2D
    flattened_output = layers.Flatten()(concatenated_output)
    print("Shape after concatenation:", concatenated_output.shape)
    print("Shape after flattening:", flattened_output.shape)

    
    # Final dense layers for classification
    outputs = layers.Dense(dense_units, kernel_regularizer=l2(l2_lambda))(flattened_output)
    outputs = layers.BatchNormalization()(outputs)
    outputs = layers.LeakyReLU()(outputs)
    outputs = layers.Dropout(dropout_rate)(outputs)
    
    # Final dense layers for classification
    #for units in [32, 32, 16, 16]:
    outputs = layers.Dense(64, kernel_regularizer=l2(l2_lambda))(outputs)
    outputs = layers.BatchNormalization()(outputs)
    outputs = layers.LeakyReLU()(outputs)
    outputs = layers.Dropout(dropout_rate)(outputs)
    
    # Output layer (softmax for classification)
    print(outputs.shape)
    output_layer = layers.Dense(2, activation='softmax')(outputs)
    
    # Create and compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer)
    optimizer = optimizers.RMSprop(learning_rate=0.001, clipvalue=1.0)
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    return model




def create_model(input_shape, num_features, n_groups = 4, lstm_units = 128, dense_units = 128):
    """
    Creates a model with the specified architecture.

    Args:
        input_shape: The shape of the input data.
        num_features: The total number of features.
        n_groups: The number of groups to split the features into.
        lstm_units: The number of units in the LSTM layers.
        dense_units: The number of units in the dense layers.

    Returns:
        The compiled model.
    """

    # Input layer
    inputs = tf.keras.Input(shape=input_shape)

    # First LSTM layer
    x = layers.LSTM(lstm_units, return_sequences=True)(inputs)

    # Split features into n groups
    group_size = num_features // n_groups
    x_groups = [x[:, :, i * group_size: (i + 1) * group_size] for i in range(n_groups)]

    # Process each group with individual LSTM and Dense layers
    group_outputs = []
    for group in x_groups:
        group_output = layers.LSTM(lstm_units)(group)
        group_output = layers.Dense(dense_units)(group_output)
        group_outputs.append(group_output)

    # Concatenate group outputs with the first layer output

    concatenated_output = layers.Concatenate(axis=1)(group_outputs + [x[:, 0, :]])
    


    # Final LSTM and Dense layers
    output = layers.LSTM(lstm_units)(concatenated_output)
    output = layers.Dense(dense_units)(output)

    # Model compilation
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mse')  # Adjust loss and optimizer as needed

    return model




def create_custom_lstm_modelx(input_shape, num_features, attention_heads=4, n_groups=10, lstm_units=64, dense_units=64, l2_lambda=0.01, dropout_rate=0.2):
    # Define the input
    input_layer = layers.Input(shape=input_shape)
    
    # Split the features into n groups
    group_size = num_features // n_groups
    lstm_outputs = []
    
    # Apply the first LSTM to the full input
    # Bidirectional LSTM Layers with Batch Normalization
    x = layers.Bidirectional(layers.LSTM(units=lstm_units, return_sequences=True))(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Bidirectional(layers.LSTM(units=lstm_units, return_sequences=False))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Multi-Head Attention Layer
    attention = layers.MultiHeadAttention(num_heads=attention_heads, key_dim=lstm_units)(x, x)
    
    # Residual Connection
    x = layers.Add()([x, attention])
    
    # Apply a different LSTM to each feature group
    for i in range(n_groups):
        start = i * group_size
        end = start + group_size
        
        # Slice the input to get the respective group
        group_input = layers.Lambda(lambda x: x[:, :, start:end])(input_layer)
        
        # Define a separate LSTM for each group
        outputs = layers.Bidirectional(layers.LSTM(lstm_units//2, return_sequences=True))(group_input)
        outputs = layers.BatchNormalization()(outputs)
        outputs = layers.LeakyReLU()(outputs)
        outputs = layers.Dropout(dropout_rate)(outputs)

        outputs = layers.Bidirectional(layers.LSTM(lstm_units//4, return_sequences=False))(outputs)
        outputs = layers.BatchNormalization()(outputs)
        outputs = layers.LeakyReLU()(outputs)
        outputs = layers.Dropout(dropout_rate)(outputs)

        attention = layers.MultiHeadAttention(num_heads=attention_heads, key_dim=lstm_units)(outputs, outputs)
    
        # Residual Connection
        outputs = layers.Add()([outputs, attention])

        
        lstm_outputs.append(outputs)
    
    # Concatenate the outputs of all LSTMs, including lstm_output_1
    concatenated_output = layers.Concatenate()([x] + lstm_outputs)
    
    # Final dense layers for classification
    #for units in [32, 32, 16, 16]:
    # Flatten the concatenated output to ensure it's 2D
    flattened_output = layers.Flatten()(concatenated_output)
    
    # Final dense layers for classification
    outputs = layers.Dense(dense_units, kernel_regularizer=l2(l2_lambda))(flattened_output)
    outputs = layers.BatchNormalization()(outputs)
    outputs = layers.LeakyReLU()(outputs)
    outputs = layers.Dropout(dropout_rate)(outputs)

    outputs = layers.Dense(dense_units//4, kernel_regularizer=l2(l2_lambda))(concatenated_output)
    outputs = layers.BatchNormalization()(outputs)
    outputs = layers.LeakyReLU()(outputs)
    outputs = layers.Dropout(dropout_rate)(outputs)

    outputs = layers.Dense(dense_units//8, kernel_regularizer=l2(l2_lambda))(concatenated_output)
    outputs = layers.BatchNormalization()(outputs)
    outputs = layers.LeakyReLU()(outputs)
    outputs = layers.Dropout(dropout_rate)(outputs)
    
    # Output layer (softmax for classification)
    output_layer = layers.Dense(2, activation='softmax')(outputs)
    
    # Create and compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer)
    optimizer = optimizers.RMSprop(learning_rate=0.001, clipvalue=1.0)
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    return model


def create_custom_lstm_model(input_shape, num_features, attention_heads=4, n_groups=20, lstm_units=128, dense_units=64, l2_lambda=0.01, dropout_rate=0.2):
    input_layer = layers.Input(shape=input_shape)
    
    # Split the features into n groups
    group_size = num_features // n_groups
    lstm_outputs = []
    
    # Apply the first LSTM to the full input
    x = layers.Bidirectional(layers.LSTM(units=lstm_units, return_sequences=True))(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Bidirectional(layers.LSTM(units=lstm_units, return_sequences=False))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    # Reshape and apply Multi-Head Attention
    attention_input = layers.Reshape((1, lstm_units))(x)
    attention = layers.MultiHeadAttention(num_heads=attention_heads, key_dim=lstm_units)(attention_input, attention_input)
    attention = layers.Reshape((lstm_units,))(attention)
    x = layers.Add()([x, attention])

    for i in range(n_groups):
        start = i * group_size
        end = start + group_size

        # Slice the input to get the respective group
        group_input = layers.Lambda(lambda x: x[:, :, start:end])(input_layer)

        # Define a separate LSTM for each group
        outputs = layers.LSTM(lstm_units, return_sequences=False)(group_input)
        outputs = layers.BatchNormalization()(outputs)
        outputs = layers.LeakyReLU()(outputs)
        outputs = layers.Dropout(dropout_rate)(outputs)

        # Reshape and apply Multi-Head Attention
        outputs = layers.Reshape((1, lstm_units))(outputs)
        attention = layers.MultiHeadAttention(num_heads=attention_heads, key_dim=lstm_units)(outputs, outputs)
        attention = layers.Reshape((lstm_units,))(attention)
        outputs = layers.Add()([outputs, attention])

        outputs = layers.Dense(64, kernel_regularizer=l2(l2_lambda))(outputs)
        outputs = layers.BatchNormalization()(outputs)
        outputs = layers.LeakyReLU()(outputs)
        outputs = layers.Dropout(dropout_rate)(outputs)
        
        lstm_outputs.append(outputs)

    # Concatenate the outputs of all LSTMs, including lstm_output_1
    concatenated_output = layers.Concatenate()([x] + lstm_outputs)
    print("Shape after concatenation:", concatenated_output.shape)
    
    
    # Directly pass concatenated output to dense layers (no need to flatten)
    outputs = layers.Dense(dense_units, kernel_regularizer=l2(l2_lambda))(concatenated_output)
    outputs = layers.BatchNormalization()(outputs)
    outputs = layers.LeakyReLU()(outputs)
    outputs = layers.Dropout(dropout_rate)(outputs)

    outputs = layers.Dense(64, kernel_regularizer=l2(l2_lambda))(outputs)
    outputs = layers.BatchNormalization()(outputs)
    outputs = layers.LeakyReLU()(outputs)
    outputs = layers.Dropout(dropout_rate)(outputs)

    print("outputs Shape :", outputs.shape)
    # Output layer (softmax for classification)
    output_layer = layers.Dense(2, activation='softmax')(outputs)

    # Create and compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer)
    optimizer = optimizers.RMSprop(learning_rate=0.001, clipvalue=1.0)
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    return model
# Define a function to print the tensor values
def print_layer(x):
    tf.print(x)  # Use tf.print to print the actual values during execution
    return x

def create_custom_lstm_model(input_shape, num_features, attention_heads=4, n_groups=5, lstm_units=64, dense_units=64, l2_lambda=0.021, dropout_rate=0.2):
    input_layer = layers.Input(shape=input_shape)
    
    # Split the features into n groups
    group_size = num_features // n_groups
    lstm_outputs = []
    
    # Apply the first LSTM to the full input
    x = layers.LSTM(units=num_features*2, return_sequences=True)(input_layer)
    x = layers.BatchNormalization()(x)

    x = layers.LSTM(units=lstm_units, return_sequences=False)(x)
    x = layers.BatchNormalization()(x)


    # Multi-Head Attention Layer (no reshaping needed, assuming x has 3 dimensions)
    # attention = layers.MultiHeadAttention(num_heads=attention_heads, key_dim=lstm_units)(x, x)
    # x = layers.Add()([x, attention])

    x = layers.Dense(1, activation='sigmoid')(x)

    # Apply a different LSTM to each feature group
    for i in range(n_groups):
        start = i * group_size
        end = start + group_size

        # Slice the input to get the respective group

        # Extract the first feature (index 0) : 5 Top most features
        first_feature = layers.Lambda(lambda x: x[:, :, :16])(input_layer)

        # Slice the input to get the respective group
        group_input = layers.Lambda(lambda x: x[:, :, start:end])(input_layer)

        # Concatenate the first feature with the sliced group
        group_input = layers.Concatenate(axis=-1)([first_feature, group_input])

       


        outputs = layers.LSTM(lstm_units * 2, kernel_regularizer=l2(l2_lambda),  return_sequences=True)(group_input)
        outputs = layers.LayerNormalization()(outputs)

        outputs = layers.LSTM(lstm_units, kernel_regularizer=l2(l2_lambda),  return_sequences=False)(outputs)
        outputs = layers.LayerNormalization()(outputs)

      
        
        outputs = layers.Dense(1, activation='sigmoid')(outputs)

     
        lstm_outputs.append(outputs)    

    

    # Concatenate the outputs of all LSTMs, including lstm_output_1
    concatenated_output = layers.Concatenate()([x] + lstm_outputs)
   
    output = layers.Dense(16, kernel_regularizer=l2(l2_lambda))(concatenated_output)
    output = layers.Dense(4, kernel_regularizer=l2(l2_lambda))(output)
    output_layer = layers.Dense(2, activation='softmax')(output)

    # Create and compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer)
    optimizer = optimizers.RMSprop(learning_rate=0.01)
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model



def Build_lstm_Conv1d_model(input_shape, num_features, attention_heads=4, n_groups=5, 
                            lstm_units=64, dense_units=64, l2_lambda=0.021, dropout_rate=0.2,
                            dilation_rates=[1, 2, 4, 8, 16], kernel_size = 3):

    input_layer = layers.Input(shape=input_shape)
    
    # Split the features into n groups
    group_size = num_features // n_groups
    lstm_outputs = []
    
    # Apply the first LSTM to the full input
    x = layers.LSTM(units=num_features*2, return_sequences=True)(input_layer)
    x = layers.BatchNormalization()(x)

    x = layers.LSTM(units=lstm_units, return_sequences=False)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(1, activation='sigmoid')(x)

    # Apply a different LSTM to each feature group
    for i in range(n_groups):

        start = i * group_size
        end = start + group_size

        # Extract the first feature (index 0) : 5 Top most features
        first_feature = layers.Lambda(lambda x: x[:, :, :6])(input_layer)

        # Slice the input to get the respective group
        group_input = layers.Lambda(lambda x: x[:, :, start:end])(input_layer)

        # Concatenate the first feature with the sliced group
        group_input = layers.Concatenate(axis=-1)([first_feature, group_input])

        # TCN Block with dilation rates
        for dilation_rate in dilation_rates:
            outputs = layers.Conv1D(filters=num_features, kernel_size=kernel_size, padding='causal', 
                            kernel_regularizer=l2(l2_lambda), dilation_rate=dilation_rate)(group_input)
            
            outputs = layers.BatchNormalization()(outputs)
            outputs = layers.LeakyReLU()(outputs) 
            outputs = layers.Dropout(dropout_rate)(outputs) 

        outputs = layers.LSTM(lstm_units * 2, kernel_regularizer=l2(l2_lambda), return_sequences=True)(outputs)
        outputs = layers.LayerNormalization()(outputs)

        outputs = layers.LSTM(lstm_units, kernel_regularizer=l2(l2_lambda), return_sequences=False)(outputs)
        outputs = layers.LayerNormalization()(outputs)

        outputs = layers.Dense(1, activation='sigmoid')(outputs)
        lstm_outputs.append(outputs)

    # Concatenate the outputs of all LSTMs, including lstm_output_1
    concatenated_output = layers.Concatenate()([x] + lstm_outputs)

    output = layers.Dense(16, kernel_regularizer=l2(l2_lambda))(concatenated_output)
    output = layers.Dense(4, kernel_regularizer=l2(l2_lambda))(output)
    output_layer = layers.Dense(2, activation='softmax')(output)

    # Create and compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer)
    optimizer = optimizers.RMSprop(learning_rate=0.01)
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
 
    return model



def create_3d_cnn_model(input_shape):
    """
    Create a 3D Convolutional Neural Network model.

    Parameters:
    - input_shape: Tuple, the shape of the input data (time_steps, num_features, num_currencies, channels).
    - num_outputs: Integer, the number of outputs (e.g., next n close prices).

    Returns:
    - model: A compiled Keras Sequential model.
    """
    
    backend.clear_session()
    model = models.Sequential()

    # Input layer for sequential data (time_steps, features)
    model.add(layers.Input(shape=input_shape))
    
    # Add 3D Convolutional layers
    model.add(layers.Conv3D(filters=32, kernel_size=(2, 2, 2), activation='relu'))
    model.add(layers.Conv3D(filters=64, kernel_size=(2, 2, 2), activation='relu'))
    
    # Flatten the output to feed into Dense layers
    model.add(layers.Flatten())
    
    # Add Dense layers
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))  # Output layer for predicting next n close prices
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    
    return model


def create_3d_cnn_model(input_shape):
    """
    Create a 3D Convolutional Neural Network model.

    Parameters:
    - input_shape: Tuple, the shape of the input data (depth, height, width, channels).

    Returns:
    - model: A compiled Keras Sequential model.
    """
    
    backend.clear_session()
    model = models.Sequential()

    # Input layer for sequential data (depth, height, width, channels)
    model.add(layers.Input(shape=input_shape))
    
    # First Conv3D layer with 32 filters
    model.add(layers.Conv3D(filters=32, kernel_size=(2, 2, 2), activation='relu', padding='same'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))  # Add 3D max pooling
    
    # Second Conv3D layer with 64 filters
    model.add(layers.Conv3D(filters=64, kernel_size=(2, 2, 2), activation='relu', padding='same'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))  # Add another 3D max pooling
    
    # Flatten the output to feed into Dense layers
    model.add(layers.Flatten())
    
    # Dense layer for learning from extracted features
    model.add(layers.Dense(128, activation='relu'))
    
    # Output layer with softmax (assuming binary classification, adjust if needed)
    model.add(layers.Dense(2, activation='softmax'))  
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


    from tensorflow.keras import backend, layers, models

def create_3d_cnn_model(input_shape):
    """
    Create a 3D Convolutional Neural Network model.

    Parameters:
    - input_shape: Tuple, the shape of the input data (depth, height, width, channels).

    Returns:
    - model: A compiled Keras Sequential model.
    """
    
    backend.clear_session()
    model = models.Sequential()

    # Input layer for sequential data (depth, height, width, channels)
    model.add(layers.Input(shape=input_shape))
    
    # First Conv3D layer with 32 filters
    model.add(layers.Conv3D(filters=256, kernel_size=(2, 2, 2), activation='relu', padding='same'))
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2)))  # Pooling only spatial dimensions
    
    # Second Conv3D layer with 64 filters
    model.add(layers.Conv3D(filters=64, kernel_size=(2, 2, 2), activation='relu', padding='same'))
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2)))  # Pooling only spatial dimensions
    
    # Flatten the output to feed into Dense layers
    model.add(layers.Flatten())
    
    # Dense layer for learning from extracted features
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    # Output layer with softmax (assuming binary classification, adjust if needed)
    model.add(layers.Dense(2, activation='softmax'))  
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def create_3d_cnn_model(input_shape):
    backend.clear_session()
    model = models.Sequential()

    model.add(layers.Input(shape=input_shape))
    
    # First Conv3D layer with 256 filters
    model.add(layers.Conv3D(filters=256, kernel_size=(2, 2, 2), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2)))  # Pooling only spatial dimensions
    model.add(layers.Dropout(0.3))
    
    # Second Conv3D layer with 64 filters
    model.add(layers.Conv3D(filters=64, kernel_size=(2, 2, 2), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2)))  # Pooling only spatial dimensions
    model.add(layers.Dropout(0.3))
    
    # Flatten the output to feed into Dense layers
    model.add(layers.Flatten())
    
    # Dense layers with Dropout
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(16, activation='relu'))
    
    # Output layer (binary classification example)
    model.add(layers.Dense(2, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), 
                  loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def create_3d_cnn_model(input_shape, sequence_length):
    backend.clear_session()
    model = models.Sequential()

    model.add(layers.Input(shape=input_shape))
    
    # Enhanced Conv3D layers
    model.add(layers.Conv3D(filters=512, kernel_size=(sequence_length, 3, 3), activation='linear', padding='same'))
    model.add(layers.LeakyReLU(negative_slope=0.1))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv3D(filters=128, kernel_size=(sequence_length//5, 3, 3), activation='linear', padding='same'))
    model.add(layers.LeakyReLU(negative_slope=0.1))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(layers.Dropout(0.3))
    
    # Flatten the output to feed into Dense layers
    model.add(layers.Flatten())
    
    # Dense layers with L2 Regularization
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    
    # Output layer
    model.add(layers.Dense(2, activation='softmax'))
    
    # Compile the model with a different learning rate
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), 
                  loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def create_3d_cnn_model(input_shape):
    backend.clear_session()
    
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # First Conv3D block with 512 filters
    x = layers.Conv3D(filters=512, kernel_size=(3, 3, 3), padding='same')(inputs)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    # Second Conv3D block with 128 filters
    x = layers.Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    x = layers.Dropout(0.3)(x)

    
    # Flatten the 3D feature maps into a 1D vector
    x = layers.Flatten()(x)
    
    # Dense layers with L2 regularization and Dropout
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    
    # Output layer for binary classification
    outputs = layers.Dense(2, activation='softmax')(x)
    
    # Define the model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), 
                  loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model



def create_3d_cnn_model(input_shape, sequence_length = 4):
    backend.clear_session()
    
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # First Conv3D block
    x = layers.Conv3D(filters=512, kernel_size=(sequence_length, 3, 3), padding='same')(inputs)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)
    
    # Second Conv3D block
    x = layers.Conv3D(filters=128, kernel_size=(sequence_length, 3, 3), padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling3D(pool_size=(1, 3, 3))(x)
    x = layers.Dropout(0.2)(x)
    
    # Third Conv3D block
    x = layers.Conv3D(filters=128, kernel_size=(sequence_length, 3, 3), padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    # Flatten the 3D feature maps into a 1D vector
    x = layers.Flatten()(x)
    
    # Dense layers with L2 regularization and Dropout
    for _ in range(2):
        x = layers.Dense(128, kernel_regularizer=l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)

    for _ in range(2):
        x = layers.Dense(64, kernel_regularizer=l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)

    for _ in range(1):
        x = layers.Dense(32, kernel_regularizer=l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)

    for _ in range(1):
        x = layers.Dense(16, kernel_regularizer=l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.25)(x)
    
    # Output layer for binary classification
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    # Define the model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), 
                  loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


def build_deep_conv_modelx(frames_shape, features_shape):
    """
    Builds a deep convolutional model that combines 3D CNN for frame processing 
    and LSTM for feature sequences.

    Parameters:
    - frames_shape: tuple, shape of the frames input (depth, height, width, channels)
    - features_shape: tuple, shape of the features input (timesteps, num_features)

    Returns:
    - model: Compiled Keras model.
    """

    # Frames input (for 3D CNN)
    frames_input = Input(shape=frames_shape)  # e.g., (depth, height, width, channels)
    x = layers.Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same')(frames_input)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    x = layers.Dropout(0.2)(x)

   
    

    # Flatten CNN output for concatenation
    x = layers.Flatten()(x)  

    # Features input (for LSTM)
    features_input = Input(shape=features_shape)  # e.g., (timesteps, num_features)
    y = layers.LSTM(units=100, return_sequences=True)(features_input)
    y = layers.BatchNormalization()(y)

    y = layers.LSTM(units=50, return_sequences=False)(y)
    y = layers.BatchNormalization()(y)

    # Combine CNN and LSTM features
    combined = layers.Concatenate()([x, y])
    z = layers.Dense(64, activation='relu')(combined)
    output = layers.Dense(1, activation='sigmoid')(z)

    # Create and compile the model
    model = Model(inputs=[frames_input, features_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def create_tcn_conv(sequence_length, feature_count, num_groups=4, dropout_rate=0.2, l2_lambda=0.01):
    # Group size calculation
    group_size = feature_count // num_groups
    remainder = feature_count % num_groups

    # Ensure no group is empty
    group_sizes = [group_size + 1 if i < remainder else group_size for i in range(num_groups)]
    group_sizes = [size for size in group_sizes if size > 0]  # Remove any zero-sized groups

    max_group_size = max(group_sizes) if group_sizes else 1  # Set a default max size if no groups exist

    # Main input layer
    main_input = layers.Input(shape=(sequence_length, feature_count))

    # Main Conv1D layer to capture overall features
    main_conv_output = layers.Conv1D(filters=feature_count, kernel_size=3, padding='same', activation='relu')(main_input)
    main_conv_output = layers.BatchNormalization()(main_conv_output)
    main_conv_output = layers.LeakyReLU()(main_conv_output)
    main_conv_output = layers.Dropout(dropout_rate)(main_conv_output)

    # Custom layer for splitting and padding
    split_and_pad_layer = SplitAndPadLayer(group_sizes, max_group_size)
    padded_splits = split_and_pad_layer(main_conv_output)

    # TCN blocks for each group
    tcn_blocks = []
    for _ in range(len(group_sizes)):
        tcn_block = build_tcn_block((sequence_length, max_group_size), filters=max_group_size, kernel_size=2)
        tcn_blocks.append(tcn_block)

    # Process each group with its TCN block
    tcn_outputs = [tcn_blocks[i](padded_splits[i]) for i in range(len(padded_splits))]

    # Combine TCN outputs with the main Conv1D output
    combined_output = layers.Concatenate()([main_conv_output] + tcn_outputs)

    # Flatten the combined output
    x = layers.Flatten()(combined_output)

    # Dense layers for final prediction
    for units in [32, 32, 16, 16]:
        x = layers.Dense(units, kernel_regularizer=l2(l2_lambda))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(dropout_rate)(x)

    # Output layer for binary classification (adjust units and activation for other tasks)
    output = layers.Dense(2, activation='softmax')(x)

    # Build and compile the model
    model = models.Model(inputs=main_input, outputs=output)
    

    return model

def build_deep_conv_model(frames_shape, features_shape, sequence_length, feature_count):
    """
    Builds a deep convolutional model that combines 3D CNN for frame processing 
    and TCN (Temporal Convolutional Network) for sequential features.

    Parameters:
    - frames_shape: tuple, shape of the frames input (depth, height, width, channels)
    - features_shape: tuple, shape of the features input (timesteps, num_features)
    - sequence_length: int, the length of the input sequence for the TCN
    - feature_count: int, the number of features in the TCN model

    Returns:
    - model: Compiled Keras model.
    """

    # Frames input (for 3D CNN)
    frames_input = Input(shape=frames_shape)  # e.g., (depth, height, width, channels)
    x = layers.Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same')(frames_input)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    x = layers.Dropout(0.2)(x)

    # Second Layer
    x = layers.Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    x = layers.Dropout(0.2)(x)

    # Flatten CNN output for concatenation
    x = layers.Flatten()(x)

    # Features input (for TCN)
    features_input = Input(shape=features_shape)  # e.g., (timesteps, num_features)

    # Create the TCN model for the feature input
    tcn_model = create_tcn_conv(sequence_length, feature_count)
    tcn_output = tcn_model(features_input)

    # Combine CNN and TCN outputs
    combined = layers.Concatenate()([x, tcn_output])

    # Dense layers for final prediction
    z = layers.Dense(64, activation='relu')(combined)
    output = layers.Dense(1, activation='sigmoid')(z)

    # Create and compile the model
    model = models.Model(inputs=[frames_input, features_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def create_3d_cnn_model(input_shape, features_shape):
    # CNN input for the 3D frames
    cnn_input = Input(shape=input_shape)
    
    # Adjust kernel size for the depth dimension and add padding='same'
    x = layers.Conv3D(16, kernel_size=(1, 3, 3), activation='relu', padding='same')(cnn_input)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    
    # Additional Conv3D layer if needed
    x = layers.Conv3D(32, kernel_size=(1, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    
    # Global Average Pooling instead of Flatten
    x = layers.GlobalAveragePooling3D()(x)

    # LSTM input for the sequence features
    features_input = Input(shape=features_shape)  # e.g., (timesteps, num_features)
    y = layers.LSTM(units=100, return_sequences=True)(features_input)
    y = layers.BatchNormalization()(y)

    y = layers.LSTM(units=50, return_sequences=False)(y)
    y = layers.BatchNormalization()(y)

    # Combine CNN and LSTM outputs
    combined = layers.Concatenate()([x, y])

    # Fully connected layers
    z = layers.Dense(64, activation='relu')(combined)
    z = layers.Dense(32, activation='relu')(z)
    
    # Output layer
    outputs = layers.Dense(1, activation='sigmoid')(z)
    
    # Create the model with two inputs
    model = tf.keras.Model(inputs=[cnn_input, features_input], outputs=outputs)
    
    # Compile the model
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    return model
