"""
Proprietary Model Reader
══════════════════════════════════════════════════════════════════════════════
Scans the private `Backend/Backend/data/` directory for custom model
architectures that expose a MODEL_REGISTRY_METADATA manifest.

DEVELOPER WORKFLOW FOR ADDING A NEW PROPRIETARY MODEL
─────────────────────────────────────────────────────────
1. Create your model file in Backend/Backend/data/  (e.g. my_model_v2.py).
2. Include a top-level MODEL_REGISTRY_METADATA dict (see baseline_v1.py as template).
3. Train for at least 1 epoch from the __main__ entry point to produce a
   .keras checkpoint file in the same directory.
4. Restart the backend — the reader will auto-discover and register your model.

VERIFICATION GATE
─────────────────
A model is only registered if its declared checkpoint file (e.g.
`baseline_v1_best.keras`) exists on disk. This proves at least 1 epoch of
training completed successfully. Models without checkpoints are logged
as warnings but excluded from the catalog.
"""

import os
import importlib
import importlib.util
import logging
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Resolve the private data directory ────────────────────────────────────────
# Layout: Backend/app/core/ml/proprietary_model_reader.py
#       → Backend/Backend/data/
_THIS_DIR = Path(__file__).resolve().parent                         # .../app/core/ml/
_BACKEND_ROOT = _THIS_DIR.parent.parent.parent                     # .../Backend/
_PROPRIETARY_DATA_DIR = _BACKEND_ROOT / "Backend" / "data"

# Alternative: allow env override for CI / deployment
PROPRIETARY_DATA_DIR = Path(
    os.environ.get("PROPRIETARY_MODEL_DIR", str(_PROPRIETARY_DATA_DIR))
)


def _load_module_metadata(filepath: Path) -> Optional[Dict[str, Any]]:
    """
    Dynamically import a .py file and extract its MODEL_REGISTRY_METADATA dict.
    Returns None if the file doesn't define MODEL_REGISTRY_METADATA or fails to import.
    
    SAFETY: Only reads the metadata attribute — does NOT execute __main__ blocks
    or trigger training. The module's Keras/TF imports may be heavy, but they are
    already loaded in the backend process.
    """
    module_name = f"_proprietary_model_{filepath.stem}"
    
    try:
        import sys
        _parent_dir = str(filepath.parent)
        if _parent_dir not in sys.path:
            sys.path.insert(0, _parent_dir)
            _added_to_path = True
        else:
            _added_to_path = False

        try:
            spec = importlib.util.spec_from_file_location(module_name, str(filepath))
            if spec is None or spec.loader is None:
                logger.warning(f"[ProprietaryReader] Could not create import spec for {filepath.name}")
                return None
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            if _added_to_path:
                sys.path.remove(_parent_dir)
        
        metadata = getattr(module, "MODEL_REGISTRY_METADATA", None)
        if metadata is None:
            return None  # Not a registrable model file — silently skip
        
        if not isinstance(metadata, dict):
            logger.warning(f"[ProprietaryReader] {filepath.name}: MODEL_REGISTRY_METADATA is not a dict, skipping")
            return None
        
        # Validate required fields
        required_fields = {"id", "name", "category", "description", "builder_function"}
        missing = required_fields - set(metadata.keys())
        if missing:
            logger.warning(
                f"[ProprietaryReader] {filepath.name}: MODEL_REGISTRY_METADATA missing "
                f"required fields: {missing}, skipping"
            )
            return None
        
        # Attach the source module reference for dynamic builder resolution
        metadata["_source_module"] = module
        metadata["_source_path"] = str(filepath)
        
        return metadata
    
    except Exception as e:
        logger.error(f"[ProprietaryReader] Failed to import {filepath.name}: {e}", exc_info=True)
        return None


def _verify_checkpoint(metadata: Dict[str, Any], data_dir: Path) -> bool:
    """
    Verification gate: check that the model's declared .keras checkpoint exists.
    This proves at least 1 epoch of training completed successfully.
    """
    checkpoint_file = metadata.get("checkpoint_file")
    if not checkpoint_file:
        logger.warning(
            f"[ProprietaryReader] Model '{metadata['id']}' has no checkpoint_file declared — "
            f"cannot verify 1-epoch training gate. Skipping."
        )
        return False
    
    checkpoint_path = data_dir / checkpoint_file
    if not checkpoint_path.exists():
        logger.warning(
            f"[ProprietaryReader] Model '{metadata['id']}': checkpoint "
            f"'{checkpoint_file}' not found at {checkpoint_path}. "
            f"Train for at least 1 epoch to produce this file. Skipping."
        )
        return False
    
    # Log verification success with file size
    size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    logger.info(
        f"[ProprietaryReader] ✅ Model '{metadata['id']}' verified — "
        f"checkpoint '{checkpoint_file}' ({size_mb:.1f} MB)"
    )
    metadata["_checkpoint_path"] = str(checkpoint_path)
    metadata["_checkpoint_size_mb"] = round(size_mb, 1)
    return True


def discover_proprietary_models(
    data_dir: Optional[Path] = None,
    require_verification: bool = True,
) -> List[Dict[str, Any]]:
    """
    Scan the proprietary data directory for model files with MODEL_REGISTRY_METADATA.
    
    Args:
        data_dir: Override the default proprietary data directory.
        require_verification: If True (default), only return models with verified
            .keras checkpoints. If False, return all discovered models (useful for
            debugging/listing).
    
    Returns:
        List of metadata dicts for verified proprietary models. Each dict contains
        all fields from MODEL_REGISTRY_METADATA plus:
            _source_module:      the imported Python module
            _source_path:        absolute path to the .py file
            _checkpoint_path:    absolute path to the .keras checkpoint (if verified)
            _checkpoint_size_mb: checkpoint file size in MB (if verified)
    """
    scan_dir = data_dir or PROPRIETARY_DATA_DIR
    
    if not scan_dir.exists():
        logger.warning(f"[ProprietaryReader] Data directory does not exist: {scan_dir}")
        return []
    
    logger.info(f"[ProprietaryReader] Scanning {scan_dir} for proprietary models...")
    
    discovered = []
    skipped = []
    
    # Scan all .py files (exclude __pycache__, tests, utilities, training scripts, debug scripts)
    _skip_prefixes = (
        "test_", "check_", "debug_", "inspect_",
        "audit_", "probe_", "diag_", "describe_",
        "apply_", "add_", "baseline_fit", "baseline_model_fit",
        "build_", "identify_", "phase1_", "verify_",
        "kaggle_baseline", "axe_", "cell", "genesis_",
        "train_", "eval_", "hybrid_", "add_", "repair_",
    )
    # Exact-name exclusions (cannot use prefix because names are too short/generic)
    _skip_exact = {"tt.py", "n"}

    for py_file in sorted(scan_dir.glob("*.py")):
        # Skip utility/diagnostic/training scripts by prefix
        if py_file.name.startswith(_skip_prefixes):
            continue
        # Skip exact-name matches (scratch files, etc.)
        if py_file.name in _skip_exact:
            continue
        if py_file.name == "__init__.py":
            continue
        
        # 🛡️ DEFENSIVE: Fast content check before importing.
        # Standalone scripts (e.g. cell1_train_open.py, eval scripts) carry top-level
        # side-effects that execute when imported via exec_module. Only import files
        # that explicitly declare MODEL_REGISTRY_METADATA.
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            if "MODEL_REGISTRY_METADATA" not in content:
                continue
        except Exception:
            continue

        metadata = _load_module_metadata(py_file)
        if metadata is None:
            continue  # Not a registrable model
        
        # Verification gate
        if require_verification:
            if not _verify_checkpoint(metadata, scan_dir):
                skipped.append(metadata["id"])
                continue
        
        discovered.append(metadata)
    
    logger.info(
        f"[ProprietaryReader] Discovery complete: "
        f"{len(discovered)} verified, {len(skipped)} skipped (unverified: {skipped})"
    )
    
    return discovered


def get_proprietary_builder(metadata: Dict[str, Any]):
    """
    Extract the builder function from a discovered proprietary model's metadata.
    
    The metadata's `_source_module` contains the imported module. We look for
    the `builder_function` name on that module.
    
    Returns:
        The callable builder function, or None if not found.
    """
    module = metadata.get("_source_module")
    builder_name = metadata.get("builder_function")
    
    if module is None or builder_name is None:
        return None
    
    # For proprietary models, the builder_function in metadata points to the
    # ADAPTER name in default_models.py (e.g. "build_baseline_brain_v1"),
    # not the raw function in the source module. The adapter is registered
    # separately in default_models.py.
    return None  # Adapter pattern — builder lives in default_models.py
