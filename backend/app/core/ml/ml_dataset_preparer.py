import asyncio
import logging
from typing import List, Dict, Any
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.concurrency import run_in_threadpool
import pandas as pd
import numpy as np
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


logger = logging.getLogger(__name__)


class MLDatasetPreparer:
    """Utility class for preparing ML datasets with sequences, scaling, and splits"""
    
    def __init__(self, df: pd.DataFrame, config: Dict[str, Any]):
        self.df = df
        self.config = config
        self.scaler = None
        
    def prepare_dataset(self, task_id: str = None, progress_store=None):
        """
        Prepare ML dataset with sequences, scaling, and train/val/test splits
        
        Returns:
            Dict containing prepared datasets and metadata
        """
        try:
            # Extract configuration
            sequence_length = self.config.get('sequence_length', 60)
            target_columns = self.config.get('target_columns', [])
            exclude_columns = self.config.get('exclude_columns', [])
            test_size = self.config.get('test_size', 0.2)
            validation_size = self.config.get('validation_size', 0.1)
            scaler_type = self.config.get('scaler_type', 'standard')
            scaler_save_path = self.config.get('scaler_save_path')
            scaler_load_path = self.config.get('scaler_load_path')
            
            if progress_store and task_id:
                progress_store.update_task(
                    task_id,
                    status="processing",
                    progress=10,
                    message=f"Preparing features and targets (Data: {len(self.df)} rows)"
                )
            
            logger.info(
                f"ML Preparation Task {task_id if task_id else 'Direct'}: "
                f"Rows={len(self.df)}, SeqLen={sequence_length}, "
                f"Targets={target_columns}, Exclude={exclude_columns}"
            )
            
            # Enrich with targets if missing
            self._enrich_with_targets(target_columns)
            
            # Separate features and targets
            feature_cols = [col for col in self.df.columns if col not in exclude_columns]
            if target_columns:
                feature_cols = [col for col in feature_cols if col not in target_columns]
            
            # Skip 'time' column if it exists and isn't the index
            feature_cols = [col for col in feature_cols if col.lower() != 'time']
            
            # Universal numeric conversion for ML compatibility
            logger.info(f"Attempting numeric conversion for {len(feature_cols)} columns...")
            
            for col in feature_cols:
                # 1. Convert Booleans to int
                if self.df[col].dtype == bool or np.issubdtype(self.df[col].dtype, np.bool_):
                    self.df[col] = self.df[col].astype(int)
                
                # 2. Attempt pd.to_numeric for other non-numeric types
                elif not np.issubdtype(self.df[col].dtype, np.number):
                    try:
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    except Exception as e:
                        logger.warning(f"Failed to convert column '{col}' to numeric: {e}")
            
            # Robustly filter for numeric columns only and log offenders
            all_feature_cols = feature_cols
            feature_cols = self.df[all_feature_cols].select_dtypes(include=[np.number]).columns.tolist()
            
            non_numeric = [col for col in all_feature_cols if col not in feature_cols]
            if non_numeric:
                logger.warning(f"Excluding remaining non-numeric columns from features: {non_numeric}")
                for col in non_numeric:
                    type_info = type(self.df[col].iloc[0]) if not self.df[col].empty else "Unknown"
                    logger.error(f"Column '{col}' (type {type_info}) cannot be converted/used for scaling.")

            X = self.df[feature_cols].values
            
            # Double check for sequences in the resulting array
            # This is a last-resort check before passing to sklearn
            if X.dtype == object:
                logger.error("Feature matrix X has object dtype even after numeric filtering. Searching for sequences...")
                for i, col in enumerate(feature_cols):
                    if any(isinstance(val, (list, tuple, np.ndarray)) for val in self.df[col].dropna()):
                        logger.error(f"CRITICAL: Column '{col}' contains sequence data (lists/arrays).")
                raise ValueError("Feature matrix contains sequence data (lists/arrays) and cannot be scaled.")

            y = self.df[target_columns].values if target_columns else None
            
            if progress_store and task_id:
                progress_store.update_task(
                    task_id,
                    progress=30,
                    message=f"Scaling data with {scaler_type} scaler"
                )
            
            # Initialize scaler
            if scaler_load_path:
                import joblib
                self.scaler = joblib.load(scaler_load_path)
                X_scaled = self.scaler.transform(X)
            else:
                if scaler_type == 'standard':
                    self.scaler = StandardScaler()
                elif scaler_type == 'minmax':
                    self.scaler = MinMaxScaler()
                elif scaler_type == 'robust':
                    self.scaler = RobustScaler()
                else:
                    raise ValueError(f"Unknown scaler type: {scaler_type}")
                
                X_scaled = self.scaler.fit_transform(X)
                
                # Save scaler if path provided
                if scaler_save_path:
                    import joblib
                    joblib.dump(self.scaler, scaler_save_path)
            
            if progress_store and task_id:
                progress_store.update_task(
                    task_id,
                    progress=50,
                    message="Creating sequences"
                )
            
            # Create sequences
            X_sequences = []
            y_sequences = [] if y is not None else None
            
            if len(X_scaled) <= sequence_length:
                logger.warning(
                    f"Insufficient data for sequence creation: Rows={len(X_scaled)}, "
                    f"SeqLen={sequence_length}. Minimum rows required: {sequence_length + 1}"
                )
            
            for i in range(len(X_scaled) - sequence_length):
                X_sequences.append(X_scaled[i:i + sequence_length])
                if y is not None:
                    y_sequences.append(y[i + sequence_length])
            
            X_sequences = np.array(X_sequences)
            if y_sequences is not None:
                y_sequences = np.array(y_sequences)
            
            logger.info(f"Generated {len(X_sequences)} sequences from {len(X_scaled)} scaled rows.")
            
            if progress_store and task_id:
                progress_store.update_task(
                    task_id,
                    progress=70,
                    message="Splitting into train/val/test sets"
                )
            
            # Calculate split indices with floor and safety checks
            total_samples = len(X_sequences)
            
            if total_samples >= 3:
                # Normal split path: ensure at least 1 in each
                test_count = max(1, int(total_samples * test_size))
                validation_count = max(1, int(total_samples * validation_size))
                train_count = total_samples - test_count - validation_count
                
                # Safety fallback if ratios are too aggressive for small n
                if train_count <= 0:
                    train_count = 1
                    test_count = max(0, total_samples - train_count - validation_count)
                
                test_idx = total_samples - test_count
                val_idx = test_idx - validation_count
                
                X_train = X_sequences[:val_idx]
                X_val = X_sequences[val_idx:test_idx]
                X_test = X_sequences[test_idx:]
                
                if y_sequences is not None:
                    y_train = y_sequences[:val_idx]
                    y_val = y_sequences[val_idx:test_idx]
                    y_test = y_sequences[test_idx:]
                else:
                    y_train = y_val = y_test = None
            else:
                # Tiny dataset fallback: assign everything to train (or split minimally)
                logger.warning(f"Extremely small dataset ({total_samples} samples). Disabling standard splits.")
                X_train = X_sequences
                X_val = np.array([])
                X_test = np.array([])
                
                if y_sequences is not None:
                    y_train = y_sequences
                    y_val = np.array([])
                    y_test = np.array([])
                else:
                    y_train = y_val = y_test = None
            
            result = {
                'X_train_shape': X_train.shape,
                'X_val_shape': X_val.shape,
                'X_test_shape': X_test.shape,
                'y_train_shape': y_train.shape if y_train is not None else None,
                'y_val_shape': y_val.shape if y_val is not None else None,
                'y_test_shape': y_test.shape if y_test is not None else None,
                'feature_columns': feature_cols,
                'target_columns': target_columns,
                'sequence_length': sequence_length,
                'scaler_type': scaler_type,
                'total_sequences': total_samples,
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test)
            }
            
            if y_sequences is not None:
                y_train = y_sequences[:val_idx]
                y_val = y_sequences[val_idx:test_idx]
                y_test = y_sequences[test_idx:]
                
                result['y_train_shape'] = y_train.shape
                result['y_val_shape'] = y_val.shape
                result['y_test_shape'] = y_test.shape
            
            if progress_store and task_id:
                progress_store.update_task(
                    task_id,
                    progress=100,
                    message="Dataset preparation complete"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error preparing ML dataset: {str(e)}", exc_info=True)
            raise

    def _enrich_with_targets(self, target_columns: List[str]):
        """
        Enrich data with target return and direction columns if they are missing.
        Uses look-ahead calculations based on the 'close' price.
        """
        if not target_columns:
            return

        # Identify the close column (case-insensitive)
        close_col = next((col for col in self.df.columns if col.lower() == 'close'), None)
        
        if not close_col:
            logger.warning("No 'close' column found for target enrichment.")
            return

        # Map target names to their look-ahead periods
        target_mapping = {
            "Next_Day_Return": 1,
            "Next_3_Day_Return": 3,
            "Next_5_Day_Return": 5
        }

        for target_col in target_columns:
            if target_col in self.df.columns:
                continue

            if target_col in target_mapping:
                period = target_mapping[target_col]
                # Calculate future return
                self.df[target_col] = (
                    self.df[close_col].shift(-period) - self.df[close_col]
                ) / self.df[close_col]
            
            elif target_col == "Next_Day_Direction":
                # Direction: 1 if positive return, else 0
                if "Next_Day_Return" in self.df.columns:
                    self.df["Next_Day_Direction"] = (self.df["Next_Day_Return"] > 0).astype(int)
                else:
                    next_return = (
                        self.df[close_col].shift(-1) - self.df[close_col]
                    ) / self.df[close_col]
                    self.df["Next_Day_Direction"] = (next_return > 0).astype(int)

        # Fill NaNs created by shifts
        self.df = self.df.fillna(0.0)
