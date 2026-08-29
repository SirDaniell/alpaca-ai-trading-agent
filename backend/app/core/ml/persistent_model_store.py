import json
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class PersistentModelStore:
    """
    Handles persistence for model configuration metadata and trained model weights.
    
    - Stores model configs as JSON metadata
    - Stores trained model weights as Keras native format (.keras files)
    - Allows backend to recover models after server reloads
    """
    
    def __init__(self, storage_path: str = "data/model_configs.json", models_dir: str = "data/models"):
        # Use absolute path relative to the Backend directory (where this file's parent's parent's parent is)
        # Assuming path is app/core/ml/persistent_model_store.py
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.storage_path = os.path.abspath(os.path.join(base_dir, storage_path))
        self.models_dir = os.path.abspath(os.path.join(base_dir, models_dir))
        logger.info(f"📁 PersistentModelStore initialized: storage={self.storage_path}, models={self.models_dir}")
        self._ensure_storage_dir()
        
    def _ensure_storage_dir(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        if not os.path.exists(self.storage_path):
            with open(self.storage_path, 'w') as f:
                json.dump({}, f)

    def save_config(self, config_id: str, metadata: Dict[str, Any]):
        """Save model configuration metadata to disk."""
        try:
            configs = self._load_all()
            configs[config_id] = metadata
            with open(self.storage_path, 'w') as f:
                json.dump(configs, f, indent=2)
            logger.info(f"✅ Persisted model config {config_id} to {self.storage_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save model config {config_id}: {e}")

    def get_config(self, config_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve model configuration metadata from disk."""
        import time
        max_retries = 3
        retry_delay = 0.2  # seconds
        
        for attempt in range(max_retries):
            try:
                configs = self._load_all()
                if config_id in configs:
                    return configs[config_id]
                
                if attempt < max_retries - 1:
                    logger.warning(f"🔍 Model config {config_id} not found (attempt {attempt+1}/{max_retries}), retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"❌ Model config {config_id} not found in {self.storage_path} after {max_retries} attempts. Available keys: {list(configs.keys())[:5]}...")
            except Exception as e:
                logger.error(f"❌ Error retrieving model config {config_id}: {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(retry_delay)
        return None

    def save_model(self, model_id: str, model, config: Dict[str, Any] = None):
        """
        Save trained model weights to disk using Keras native format.
        
        Args:
            model_id: Unique model identifier
            model: Compiled Keras model instance
            config: Optional config dict to save with model metadata
        """
        try:
            # Use .keras extension for native Keras format (recommended by TensorFlow 2.16+)
            model_filename = f"{model_id}.keras"
            model_path = os.path.join(self.models_dir, model_filename)
            
            # Save model using Keras native format
            model.save(model_path)
            
            # Save config metadata
            if config:
                self.save_config(model_id, {
                    "model_id": model_id,
                    "path": model_path,
                    "filename": model_filename,
                    **config
                })
            
            logger.info(f"✅ Saved model {model_id} to {model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save model {model_id}: {e}")
            raise

    def load_model(self, model_id: str):
        """
        Load trained model from disk.
        
        Args:
            model_id: Unique model identifier
            
        Returns:
            Loaded Keras model instance or None if not found
        """
        try:
            import tensorflow as tf
            
            # Try .keras format first (new format)
            model_filename = f"{model_id}.keras"
            model_path = os.path.join(self.models_dir, model_filename)
            
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path)
                logger.info(f"✅ Loaded model {model_id} from {model_path} (.keras format)")
                return model
            
            # Fallback to directory format (old SavedModel format)
            model_dir_path = os.path.join(self.models_dir, model_id)
            if os.path.exists(model_dir_path):
                model = tf.keras.models.load_model(model_dir_path)
                logger.info(f"✅ Loaded model {model_id} from {model_dir_path} (SavedModel format)")
                return model
            
            logger.warning(f"Model not found: {model_id} (tried both .keras and SavedModel formats)")
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to load model {model_id}: {e}")
            return None

    def _load_all(self) -> Dict[str, Any]:
        """Load all configurations from disk."""
        try:
            if not os.path.exists(self.storage_path):
                return {}
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"❌ Error loading model configs: {e}")
            return {}

# Singleton instance
persistent_model_store = PersistentModelStore()
