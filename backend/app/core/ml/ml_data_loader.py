import numpy as np
import logging
import os
import gc
from typing import List, Dict, Any, Optional, Iterator, Tuple, Union

logger = logging.getLogger(__name__)

_SCALAR_OUTPUTS = {
    "Signal_bounce_support", "Signal_bounce_resistance",
    "Signal_breakout_support", "Signal_breakout_resistance",
    "bull_prob", "bull_conf", "bear_conf", "bull_strength", "bear_strength",
    "mfe", "mae", "risk_reward", "signal_strength",
    "reversal_prob", "trend_continuation_prob", "reversal_held",
    "next_zone_bars", "next_zone_distance", "next_zone_volume",
    "vol_surge",
}
_SPARSE_OUTPUTS = {"probe_hour", "probe_session", "probe_day_of_week", "next_zone_idx"}


def _infer_output_placeholder(name: str, batch_size: int, horizon: int) -> np.ndarray:
    if name == "ohlcv_sequence":
        return np.zeros((batch_size, horizon, 5), dtype=np.float32)
    if name == "bull_class":
        return np.zeros((batch_size, 3), dtype=np.float32)
    if name in _SPARSE_OUTPUTS:
        return np.zeros((batch_size,), dtype=np.int32)
    if name in _SCALAR_OUTPUTS:
        return np.zeros((batch_size, 1), dtype=np.float32)
    return np.zeros((batch_size, horizon), dtype=np.float32)

# Lazy import to avoid circular deps — only used at runtime
def _get_output_spec():
    try:
        from app.core.ml.output_spec import V8_3_ALL_OUTPUT_KEYS, V8_3_NPZ_TARGET_KEY_MAP
        return set(V8_3_ALL_OUTPUT_KEYS), V8_3_NPZ_TARGET_KEY_MAP
    except ImportError as _e:
        # Do NOT swallow silently: an ImportError here means output_spec.py is
        # missing, has a circular import, or wasn't deployed. The consequence is
        # that multi_output auto-detection always returns False, and the entire
        # multi-output training pipeline silently degrades to legacy single-target
        # mode with no obvious cause in the logs.
        logger.error(
            "CRITICAL: Failed to import output_spec — multi_output auto-detection "
            "will be disabled and LazySequenceGenerator will fall back to single-target "
            "mode for ALL datasets. Fix the import before training a V8.3 model. "
            f"Import error: {_e}"
        )
        return set(), {}


class LazySequenceGenerator:
    """
    Memory-efficient data generator for ML training.
    Loads sequences from disk-cached .npz files batch-by-batch.

    Multi-output mode
    ─────────────────
    When ``multi_output=True`` (or auto-detected because ``selected_targets``
    covers all V8_3 output keys), ``flow()`` yields ``batch_y`` as a
    **dict** keyed by model output name rather than a single flat array.
    This matches the shape that Keras expects when ``model.compile()`` was
    called with a dict loss (i.e. any V8.3 model).
    """

    def __init__(
        self,
        file_paths: List[str],
        batch_size: int = 32,
        shuffle: bool = True,
        autoencoder_mode: bool = False,
        target_column: str = None,
        selected_targets: Optional[List[str]] = None,
        micro_val_holdback: float = 0.0,  # ✅ Hold back % of train for batch-level audit
        multi_output: Optional[bool] = None,  # ✅ NEW: Force / auto-detect dict-y mode
        npz_map: Optional[Dict[str, str]] = None,  # ✅ Custom NPZ key → output name mapping
    ):
        """
        Initialize the generator.
        """
        self.file_paths = [p for p in file_paths if os.path.exists(p)]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.autoencoder_mode = autoencoder_mode
        self.target_column = target_column
        self.selected_targets = selected_targets or []
        self.micro_val_holdback = micro_val_holdback

        # ✅ Multi-output detection: if selected_targets covers all V8_3 outputs,
        # or multi_output=True is set explicitly, yield y as a dict of named arrays
        # instead of a single flat tensor.  This is the prerequisite for training
        # any model compiled with a dict loss (V8.3+).
        all_keys, default_npz_map = _get_output_spec()
        # Use custom map if provided, otherwise use default v8 spec
        self._npz_key_map = npz_map if npz_map is not None else default_npz_map
        
        if multi_output is not None:
            self.multi_output = multi_output
        else:
            # Auto-detect: if selected_targets has >1 item that are known model-output
            # names, switch to dict mode. BUGFIX: selected_targets is typically a
            # list of RAW npz column names (e.g. 'adv_target_Open_seq'), not model
            # output names (e.g. 'open_sequence') — comparing raw names directly
            # against all_keys (model output names) almost always finds zero
            # overlap even when this genuinely is a multi-output dataset, so map
            # each selected target through npz_map before checking.
            if len(self.selected_targets) > 1 and all_keys:
                mapped_selected = {self._npz_key_map.get(t, t) for t in self.selected_targets}
                self.multi_output = bool(mapped_selected & all_keys)
            else:
                self.multi_output = False
        self.all_keys = all_keys

        # Load feature index map if available (for extracting probe targets)
        self.feature_index_map = {}
        if self.file_paths:
            import json
            # File paths are e.g. .../ml_cache/{dataset}/{split}/chunk.npz
            parent_dir = os.path.dirname(os.path.dirname(self.file_paths[0]))
            map_path = os.path.join(parent_dir, "feature_index_map.json")
            if os.path.exists(map_path):
                try:
                    with open(map_path, "r") as f:
                        data = json.load(f)
                        self.feature_index_map = data.get("feature_index_map", {})
                    # logger.info(f"✅ [LazyLoader] Loaded feature_index_map from {map_path}")
                except Exception as e:
                    logger.warning(f"⚠️ [LazyLoader] Failed to load feature_index_map: {e}")

        # Micro-validation Jury Pool (Fixed per epoch, rotated each epoch)
        self.jury_x = None
        # In multi_output mode jury_y is a dict[str, ndarray]; otherwise ndarray
        self.jury_y = None
        self.jury_indices = {}  # Map of {path: set(indices_to_skip)}
        
        # Metadata
        self.total_sequences = 0
        self.chunk_metadata = []
        self.pass_count = 0
        self.total_batches_yielded = 0
        
        self._index_chunks()
        
 
        self.actual_batches_per_epoch = sum(max(1, int(np.ceil(chunk["count"] / self.batch_size))) for chunk in self.chunk_metadata)
        self.num_batches = self.actual_batches_per_epoch
        
        logger.info(
            f"🚀 [LazyLoader] Initialized with {len(self.file_paths)} chunks, {self.total_sequences} sequences.\n"
            f"   ├─ Ideal batches: {int(np.ceil(self.total_sequences / self.batch_size))}\n"
            f"   ├─ Actual batches (per epoch): {self.actual_batches_per_epoch}\n"
            f"   └─ Target Column: {self.target_column or 'Auto (first available)'}\n"
            f"   └─ Selected Targets: {self.selected_targets or 'Auto detect'}"
        )
        
    def _index_chunks(self):
        """Scan all chunks to determine total sequence count."""
        for path in self.file_paths:
            try:
                # Load metadata only
                data = np.load(path, mmap_mode='r', allow_pickle=True)
                
                # Support both 'sequences' (legacy) and 'x' (spooled)
                count = 0
                if 'sequences' in data:
                    count = len(data['sequences'])
                elif 'x' in data:
                    count = len(data['x'])
                else:
                    logger.warning(f"[LazyLoader] No 'sequences' or 'x' in {path}")
                    continue
                    
                self.chunk_metadata.append({
                    "path": path,
                    "count": count
                })
                self.total_sequences += count
                logger.debug(f"  ├─ Indexed chunk {os.path.basename(path)}: {count} sequences")
                data.close()
            except Exception as e:
                logger.error(f"❌ [LazyLoader] Failed to index {path}: {e}")

    def __len__(self) -> int:
        """Total number of batches available per epoch."""
        return self.actual_batches_per_epoch

    def flow(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generator function yielding (batch_x, batch_y) tuples.
        
        ✅ CRITICAL FIX: Yields exactly ONE epoch of batches, then stops.
        Create a new generator instance (or call flow() again) for the next epoch.
        
        🔴 CRITICAL FIX: NEVER SKIP PARTIAL BATCHES
        All samples must be included in training, no silent drops.
        """
        # Track statistics for diagnostics
        total_samples_yielded = 0
        total_batches_yielded = 0
        
        epoch_samples = 0
        epoch_batches = 0
        chunks = list(self.chunk_metadata)
        
        if self.shuffle:
            np.random.shuffle(chunks)
        
        # 📊 Log epoch start
        logger.info(
            f"🔄 [LazyLoader] flow() START: "
            f"shuffling={self.shuffle}, {len(chunks)} chunks, "
            f"total_sequences={self.total_sequences}"
        )
        
        # ✅ NO WHILE TRUE — iterate through chunks once, then return naturally
        for chunk_idx, chunk in enumerate(chunks):  
            try:
                data = np.load(chunk["path"], allow_pickle=True)
                # peek in data
                # logger.info(f"📊 [LazyLoader] Data keys: {data.keys()}")
                # logger.info(f"📊 [LazyLoader] Data: {data}")
                # Support both 'sequences' (legacy) and 'x' (spooled)
                x = data["sequences"] if "sequences" in data else data["x"]
                
                chunk_samples = len(x)
                chunk_batches = 0
                
                # 📊 Log chunk load
                logger.info(
                    f"📂 [LazyLoader] Chunk {chunk_idx+1}/{len(chunks)}: "
                    f"loaded {chunk_samples} sequences from {os.path.basename(chunk['path'])}"
                )
                
                # 🎯 Debug: Log available keys
                # logger.debug(f"   Available keys in chunk: {list(data.files)}")
                
                # Target resolution logic
                targets_dict = {}
                
                # 🔍 [DEBUG] Deep inspection of keys
                all_keys = list(data.files)
                # logger.info(f"📊 [LazyLoader] Available keys in chunk: {all_keys}")
                # for k in all_keys:
                #     try:
                #         val = data[k]
                #         shape = val.shape if hasattr(val, 'shape') else 'scalar/dict'
                #         logger.info(f"   ├─ Key: {k} | Shape: {shape} | Type: {type(val)}")
                #     except Exception as e:
                #         logger.warning(f"   ├─ Key: {k} | Error reading: {e}")

                # 1. Explicit target selection if provided
                if self.selected_targets:
                    for requested_target in self.selected_targets:
                        prefixed = f"target_{requested_target}"
                        if prefixed in data:
                            targets_dict[requested_target] = data[prefixed]
                            # logger.info(f"   🎯 Using explicit selected target: {prefixed}")
                        elif requested_target in data:
                            targets_dict[requested_target] = data[requested_target]
                            # logger.info(f"   🎯 Using explicit selected target: {requested_target}")
                        else:
                            logger.warning(f"⚠️ [LazyLoader] Selected target '{requested_target}' not found in chunk.")
                elif self.target_column:
                    # Check for target_ prefix first (Standard MLPrep)
                    prefixed = f"target_{self.target_column}"
                    if prefixed in data:
                        targets_dict["primary"] = data[prefixed]
                        # logger.info(f"   🎯 Using explicit prefixed target: {prefixed}")
                    elif self.target_column in data:
                        targets_dict["primary"] = data[self.target_column]
                        # logger.info(f"   🎯 Using explicit target: {self.target_column}")
                    else:
                        logger.warning(f"⚠️ [LazyLoader] Target '{self.target_column}' not found in chunk.")
                
                # 2. Auto-detection if no explicit target or not found.
                # IMPORTANT: exclude target_mask_* keys — those are per-target validity
                # masks written by _generate_sequences_for_df and must not be treated
                # as training targets (they'd map to nonsense raw names like 'mask_foo').
                if not targets_dict:
                    for k in data.files:
                        if k.startswith("target_") and not k.startswith("target_mask_") and k != "targets":
                            targets_dict[k[7:]] = data[k]
                            # logger.info(f"   🔍 Found prefixed target: {k} -> {k[7:]}")
                    
                    if not targets_dict and "y" in data:
                        targets_dict["primary"] = data["y"]
                        # logger.info(f"   🔍 Found fallback target 'y'")
                    
                    if not targets_dict and "future_sequence" in data:
                        targets_dict["future_sequence"] = data["future_sequence"]
                        # logger.info(f"   🔍 Found direct target 'future_sequence'")
                    
                    if not targets_dict and "targets" in data:
                        t_data = data["targets"]
                        if isinstance(t_data, np.ndarray) and t_data.ndim > 0:
                            targets_dict["primary"] = t_data
                            # logger.info(f"   🔍 Using 'targets' as primary array")
                        elif hasattr(t_data, 'item') and isinstance(t_data.item(), dict):
                            item_dict = t_data.item()
                            # logger.info(f"   🔍 'targets' is a dict-wrapper: {list(item_dict.keys())}")
                            for tk, tv in item_dict.items():
                                targets_dict[tk] = tv
                
                # 📊 Final check
                if targets_dict:
                    pass  # logger.info(f"✅ [LazyLoader] Final target keys resolved: {list(targets_dict.keys())}")
                else:
                    logger.warning(f"❌ [LazyLoader] NO TARGETS RESOLVED! Loss will be calculated against inputs.")

                labels = data["labels"] if "labels" in data.files else None

                # Load per-target validity masks (written by _generate_sequences_for_df).
                # Keys are "target_mask_{raw_target_name}" in the NPZ; strip "target_mask_" to
                # get the raw target name, then resolve to a model output name via _npz_key_map.
                # These become sample_weight entries so Keras zeros out that head's loss
                # for samples where the raw target was NaN at sequence-generation time.
                masks_dict = {}   # raw_target_name -> full mask array (all samples in chunk)
                for k in data.files:
                    if k.startswith("target_mask_"):
                        raw_key = k[len("target_mask_"):]
                        masks_dict[raw_key] = data[k]
                
                # Apply jury mask: skip indices held back for this epoch's jury
                forbidden = self.jury_indices.get(chunk["path"], set())
                indices = [i for i in range(len(x)) if i not in forbidden]
                indices = np.array(indices)
                
                if self.shuffle:
                    np.random.shuffle(indices)
            
                for batch_num, start_idx in enumerate(range(0, len(indices), self.batch_size)):
                    end_idx = min(start_idx + self.batch_size, len(indices))
                    batch_size_actual = end_idx - start_idx
                    
                    batch_indices = indices[start_idx:end_idx]
                    batch_x = x[batch_indices]
                    
                    if self.autoencoder_mode:
                        batch_y = batch_x
                    else:
                        if targets_dict:
                            # ─────────────────────────────────────────────────────
                            # MULTI-OUTPUT PATH: build a dict keyed by model output
                            # name so Keras model.compute_loss(x, y_dict, y_pred)
                            # can match targets to the correct output head.
                            # ─────────────────────────────────────────────────────
                            if self.multi_output:
                                batch_y = {}
                                skipped_unmapped = []
                                # BUGFIX: previously this looped over selected_targets
                                # (raw npz-derived names like 'adv_target_Open_seq')
                                # and did `if tk in targets_dict: batch_y[tk] = ...`
                                # FIRST — since targets_dict is keyed by those exact
                                # raw names (that's how it was populated above), this
                                # branch always won and inserted the RAW name into
                                # batch_y, never reaching the npz_key_map lookup. A
                                # second loop then ALSO added the correctly-mapped
                                # name (since it wasn't in batch_y yet under THAT
                                # key), so batch_y ended up with BOTH
                                # 'adv_target_Open_seq' AND 'open_sequence' for the
                                # same array. Neither the model nor Keras has an
                                # output called 'adv_target_Open_seq', so this raw
                                # leftover key broke Keras's y_true/y_pred structural
                                # match ("different structures") even though the
                                # correctly-mapped key was also present.
                                #
                                # Fix: map EVERY raw key through _npz_key_map first,
                                # in one pass. Only keep it under a name that either
                                # (a) has a known mapping, or (b) is itself already a
                                # real model output name (self.all_keys) — anything
                                # else is a raw target with no corresponding model
                                # head and must be dropped, not injected verbatim.
                                for raw_key, arr in targets_dict.items():
                                    model_name = self._npz_key_map.get(raw_key)
                                    if model_name is None:
                                        if self.all_keys and raw_key not in self.all_keys:
                                            skipped_unmapped.append(raw_key)
                                            continue
                                        model_name = raw_key  # no map needed/available; raw name IS a valid output name
                                    value = arr[batch_indices].astype(np.float32)
                                    if model_name == "next_zone_idx":
                                        label_source = value
                                        if np.nanmax(label_source) <= 1.0:
                                            label_source = np.round(label_source * 6.0)
                                        value = np.clip(label_source, 0, 6).astype(np.int32)
                                    elif model_name == "bull_class" and value.ndim == 1:
                                        labels = np.clip(value.astype(int), 0, 2)
                                        value = np.eye(3, dtype=np.float32)[labels]
                                    elif model_name == "risk_reward":
                                        value = np.clip(value / 50.0, 0.0, 10.0)
                                    elif model_name == "vol_surge" and np.nanmax(value) > 1.0:
                                        value = (value > 1.5).astype(np.float32)
                                    elif model_name in _SCALAR_OUTPUTS and value.ndim == 1:
                                        value = value[:, None]
                                    batch_y[model_name] = value
                                if skipped_unmapped:
                                    logger.debug(
                                        f"   ↷ [LazyLoader] {len(skipped_unmapped)} raw target(s) have no "
                                        f"corresponding model output and were skipped (not model outputs, "
                                        f"not in npz_key_map): {skipped_unmapped}"
                                    )
                                if not batch_y:
                                    # Last resort: autoencoder fallback
                                    batch_y = batch_x

                                # One raw target can supervise several model heads.
                                # Keep these expansions data-driven from existing
                                # chunk arrays instead of relying on synthetic
                                # virtual keys that never exist in the NPZ.
                                close_target = targets_dict.get("future_sequence")
                                if close_target is not None:
                                    close_batch = close_target[batch_indices].astype(np.float32)
                                    for aux_name in ("aux_output_1", "aux_output_2", "aux_output_3"):
                                        if aux_name in self.all_keys and aux_name not in batch_y:
                                            batch_y[aux_name] = close_batch
                                
                                # Probe heads (V8.3 temporal sanity check)
                                # Use NPZ target sequences, NOT feature input values.
                                # sparse_categorical_crossentropy expects (batch,) int32.
                                if {"probe_hour", "probe_session", "probe_day_of_week"} & self.all_keys:
                                    hour_npz = targets_dict.get("adv_target_hour_next")
                                    session_npz = targets_dict.get("adv_target_session_next")
                                    dow_npz = targets_dict.get("adv_target_day_of_week_next")
                                    if hour_npz is not None:
                                        h = hour_npz[batch_indices]
                                        batch_y["probe_hour"] = np.round(
                                            (h[:, 0] if h.ndim == 2 else h) * 23
                                        ).astype(np.int32)
                                    if session_npz is not None:
                                        s = session_npz[batch_indices]
                                        batch_y["probe_session"] = np.round(
                                            (s[:, 0] if s.ndim == 2 else s) * 3
                                        ).astype(np.int32)
                                    if dow_npz is not None:
                                        d = dow_npz[batch_indices]
                                        batch_y["probe_day_of_week"] = np.round(
                                            (d[:, 0] if d.ndim == 2 else d) * 6
                                        ).astype(np.int32)

                                if self.all_keys:
                                    horizon_source = targets_dict.get("future_sequence")
                                    if horizon_source is None:
                                        horizon_source = targets_dict.get("adv_target_Close_seq")
                                    if horizon_source is None:
                                        horizon_source = targets_dict.get("adv_target_Open_seq")
                                    horizon = (
                                        int(horizon_source.shape[1])
                                        if horizon_source is not None and getattr(horizon_source, "ndim", 1) >= 2
                                        else 1
                                    )
                                    for output_name in self.all_keys:
                                        if output_name not in batch_y:
                                            batch_y[output_name] = _infer_output_placeholder(
                                                output_name, batch_size_actual, horizon
                                            )
                            # ─────────────────────────────────────────────────────
                            # SINGLE-OUTPUT PATH (legacy): resolve to one flat array
                            # ─────────────────────────────────────────────────────
                            elif self.selected_targets:
                                if len(self.selected_targets) == 1 and self.selected_targets[0] in targets_dict:
                                    batch_y = targets_dict[self.selected_targets[0]][batch_indices]
                                else:
                                    target_arrays = []
                                    for tk in self.selected_targets:
                                        if tk in targets_dict:
                                            arr = targets_dict[tk][batch_indices]
                                            target_arrays.append(arr.reshape(len(arr), -1))
                                    if target_arrays:
                                        batch_y = np.hstack(target_arrays)
                                    elif "primary" in targets_dict and self.target_column:
                                        batch_y = targets_dict["primary"][batch_indices]
                                    elif "future_sequence" in targets_dict:
                                        batch_y = targets_dict["future_sequence"][batch_indices]
                                    elif "y" in targets_dict:
                                        batch_y = targets_dict["y"][batch_indices]
                                    elif len(targets_dict) == 1:
                                        batch_y = list(targets_dict.values())[0][batch_indices]
                                    else:
                                        logger.warning(f"⚠️ [LazyLoader] Multiple targets found {list(targets_dict.keys())} and no explicit target_column or selected_targets resolved. Stacking them horizontally.")
                                        target_arrays = []
                                        for k in sorted(targets_dict.keys()):
                                            v = targets_dict[k][batch_indices]
                                            target_arrays.append(v.reshape(len(v), -1))
                                        batch_y = np.hstack(target_arrays)
                            else:
                                # 1. If explicit target_column was found, use it (highest priority)
                                if "primary" in targets_dict and self.target_column:
                                    batch_y = targets_dict["primary"][batch_indices]
                                # 2. If future_sequence exists and no specific column requested, prioritize it
                                elif "future_sequence" in targets_dict:
                                    batch_y = targets_dict["future_sequence"][batch_indices]
                                # 3. Standard 'y' fallback
                                elif "y" in targets_dict:
                                    batch_y = targets_dict["y"][batch_indices]
                                # 4. If only one target array exists, use it directly
                                elif len(targets_dict) == 1:
                                    batch_y = list(targets_dict.values())[0][batch_indices]
                                else:
                                    # Multiple targets -> stack them only as a last resort
                                    logger.warning(f"⚠️ [LazyLoader] Multiple targets found {list(targets_dict.keys())} and no explicit target_column set. Stacking them horizontally.")
                                    target_arrays = []
                                    for k in sorted(targets_dict.keys()):
                                        v = targets_dict[k][batch_indices]
                                        target_arrays.append(v.reshape(len(v), -1))
                                    batch_y = np.hstack(target_arrays)
                        elif labels is not None:
                            batch_y = labels[batch_indices]
                        else:
                            batch_y = batch_x  # Final fallback
                    
                    # 📊 Log batch (every Nth batch to avoid log spam)
                    if batch_num % 10 == 0 or batch_size_actual < self.batch_size:
                        logger.debug(
                            f"   Batch {batch_num+1}: "
                            f"samples {start_idx+1:5d}-{end_idx:5d} "
                            f"(size={batch_size_actual}/{self.batch_size}) "
                            f"partial={'YES ⚠️' if batch_size_actual < self.batch_size else 'NO ✓'}"
                        )
                        # 📊 DIAGNOSTIC: Log shapes for first batch of every chunk
                        if batch_num == 0:
                            if isinstance(batch_y, dict):
                                y_shape_info = {k: v.shape for k, v in batch_y.items()}
                                logger.info(
                                    f"📊 [LazyLoader] DIAGNOSTIC (multi-output): chunk={os.path.basename(chunk['path'])} batch=1\n"
                                    f"   ├─ x.shape: {batch_x.shape}\n"
                                    f"   └─ y keys/shapes: {y_shape_info}"
                                )
                            else:
                                logger.info(
                                    f"📊 [LazyLoader] DIAGNOSTIC: chunk={os.path.basename(chunk['path'])} batch=1\n"
                                    f"   ├─ x.shape: {batch_x.shape}\n"
                                    f"   ├─ y.shape: {batch_y.shape}\n"
                                    f"   └─ x[0,0,0]: {batch_x[0,0,0]:.4f} | y[0]: {batch_y[0] if np.isscalar(batch_y[0]) or len(batch_y[0].shape)==0 else batch_y[0,0] if len(batch_y[0].shape)>0 else '?'}"
                                )
                    
                    total_samples_yielded += batch_size_actual
                    total_batches_yielded += 1
                    epoch_samples += batch_size_actual
                    epoch_batches += 1
                    chunk_batches += 1
                    
                    # Build sample_weight dict from masks_dict.
                    # Maps raw_target_name → model_output_name via _npz_key_map,
                    # then slices the batch indices. If no masks were written (old NPZ
                    # files pre-fix), sample_weight is omitted entirely so existing
                    # training runs continue to work unchanged.
                    batch_sample_weight = {}
                    if masks_dict:
                        for raw_key, mask_arr in masks_dict.items():
                            model_name = self._npz_key_map.get(raw_key, raw_key)
                            # Only include if the resolved name is an actual model output
                            if self.all_keys and model_name not in self.all_keys:
                                continue
                            batch_sample_weight[model_name] = mask_arr[batch_indices].astype(np.float32)

                    # ✅ Yield dict with x, y, optional sample_weight, and raw target arrays.
                    # The raw target arrays are kept for backward-compat with batch_builder.
                    batch_result = {
                        'x': batch_x,
                        'y': batch_y,
                    }
                    if batch_sample_weight:
                        batch_result['sample_weight'] = batch_sample_weight

                    # Add raw target_* arrays (non-mask) for batch_builder lookup.
                    # Explicitly exclude target_mask_* to avoid polluting raw_y with mask arrays.
                    for k in data.files:
                        if k.startswith("target_") and not k.startswith("target_mask_"):
                            # Strip 'target_' prefix: target_adv_target_Close_seq → adv_target_Close_seq
                            name = k[7:]  # len("target_") == 7
                            batch_result[name] = data[k][batch_indices].astype(np.float32)

                    yield batch_result
                
                # 📊 Log chunk completion
                logger.info(
                    f"✅ [LazyLoader] Chunk {chunk_idx+1} complete: "
                    f"{chunk_batches} batches, {epoch_samples} samples"
                )
                
                # Cleanup
                data.close()
                del x, data
                gc.collect()
                
            except Exception as e:
                logger.error(f"[LazyLoader] Error in chunk {chunk['path']}: {e}")
                continue
        
        # ✅ One pass complete — generator exhausts here naturally (no while True, no break condition)
        logger.info(
            f"✅ [LazyLoader] flow() complete: "
            f"{total_batches_yielded} batches, {total_samples_yielded} samples"
        )
        # Generator falls off the end naturally ✅

    def _prepare_epoch_jury(self):
        """
        Randomly selects a subset of samples across all chunks to serve as the
        validation jury for the current epoch.

        In multi_output mode, jury_y is a dict keyed by model output name.
        In single-output mode, jury_y is a single flat ndarray (legacy).
        """
        self.jury_indices = {}
        all_jury_x = []
        # multi_output: list of dicts; single-output: list of arrays
        all_jury_y: list = []

        total_to_hold = int(self.total_sequences * self.micro_val_holdback)
        if total_to_hold < 1:
            return

        # Distribute holdback across chunks proportional to their size
        for chunk in self.chunk_metadata:
            chunk_hold = int(chunk["count"] * self.micro_val_holdback)
            if chunk_hold < 1:
                continue

            # Pick random indices to hold back
            indices = np.random.choice(chunk["count"], size=chunk_hold, replace=False)
            self.jury_indices[chunk["path"]] = set(indices.tolist())

            # Load the actual data for the jury pool
            try:
                data = np.load(chunk["path"], allow_pickle=True)
                x_full = data["sequences"] if "sequences" in data else data["x"]

                if self.multi_output:
                    # ── Multi-output: build a dict of arrays ──────────────
                    jury_y_chunk: Dict[str, np.ndarray] = {}
                    # Collect all target_ keys and map them to model output names
                    for k in data.files:
                        if k.startswith("target_"):
                            raw_name = k[7:]  # strip "target_"
                            model_name = self._npz_key_map.get(raw_name)
                            if model_name is None:
                                if self.all_keys and raw_name not in self.all_keys:
                                    continue
                                model_name = raw_name
                            value = data[k][indices].astype(np.float32)
                            if model_name == "next_zone_idx":
                                label_source = value
                                if np.nanmax(label_source) <= 1.0:
                                    label_source = np.round(label_source * 6.0)
                                value = np.clip(label_source, 0, 6).astype(np.int32)
                            elif model_name == "bull_class" and value.ndim == 1:
                                labels = np.clip(value.astype(int), 0, 2)
                                value = np.eye(3, dtype=np.float32)[labels]
                            elif model_name == "risk_reward":
                                value = np.clip(value / 50.0, 0.0, 10.0)
                            elif model_name == "vol_surge" and np.nanmax(value) > 1.0:
                                value = (value > 1.5).astype(np.float32)
                            elif model_name in _SCALAR_OUTPUTS and value.ndim == 1:
                                value = value[:, None]
                            jury_y_chunk[model_name] = value
                    close_npz = data.get("target_future_sequence")
                    if close_npz is not None:
                        close_jury = close_npz[indices].astype(np.float32)
                        for aux_name in ("aux_output_1", "aux_output_2", "aux_output_3"):
                            if aux_name in self.all_keys and aux_name not in jury_y_chunk:
                                jury_y_chunk[aux_name] = close_jury
                    # If selected_targets given, restrict to those keys
                    if self.selected_targets:
                        filtered = {}
                        for tk in self.selected_targets:
                            if tk in jury_y_chunk:
                                filtered[tk] = jury_y_chunk[tk]
                            else:
                                mapped = self._npz_key_map.get(tk)
                                if mapped and mapped in jury_y_chunk:
                                    filtered[mapped] = jury_y_chunk[mapped]
                        if filtered:
                            jury_y_chunk = filtered
                            
                    # Probe heads for validation jury — use NPZ targets, int32 labels
                    if {"probe_hour", "probe_session", "probe_day_of_week"} & self.all_keys:
                        hour_npz = data.get("target_adv_target_hour_next")
                        session_npz = data.get("target_adv_target_session_next")
                        dow_npz = data.get("target_adv_target_day_of_week_next")
                        if hour_npz is not None:
                            h = hour_npz[indices]
                            jury_y_chunk["probe_hour"] = np.round(
                                (h[:, 0] if h.ndim == 2 else h) * 23
                            ).astype(np.int32)
                        if session_npz is not None:
                            s = session_npz[indices]
                            jury_y_chunk["probe_session"] = np.round(
                                (s[:, 0] if s.ndim == 2 else s) * 3
                            ).astype(np.int32)
                        if dow_npz is not None:
                            d = dow_npz[indices]
                            jury_y_chunk["probe_day_of_week"] = np.round(
                                (d[:, 0] if d.ndim == 2 else d) * 6
                            ).astype(np.int32)

                    if self.all_keys:
                        horizon_source = data.get("target_future_sequence")
                        if horizon_source is None:
                            horizon_source = data.get("target_adv_target_Close_seq")
                        if horizon_source is None:
                            horizon_source = data.get("target_adv_target_Open_seq")
                        horizon = (
                            int(horizon_source.shape[1])
                            if horizon_source is not None and getattr(horizon_source, "ndim", 1) >= 2
                            else 1
                        )
                        for output_name in self.all_keys:
                            if output_name not in jury_y_chunk:
                                jury_y_chunk[output_name] = _infer_output_placeholder(
                                    output_name, len(indices), horizon
                                )

                    if jury_y_chunk:
                        all_jury_x.append(x_full[indices])
                        all_jury_y.append(jury_y_chunk)
                else:
                    # ── Single-output: resolve to one flat array ──────────
                    y_full = None
                    if self.selected_targets:
                        selected_arrays = []
                        for requested_target in self.selected_targets:
                            prefixed = f"target_{requested_target}"
                            if prefixed in data:
                                selected_arrays.append(data[prefixed])
                            elif requested_target in data:
                                selected_arrays.append(data[requested_target])
                        if len(selected_arrays) == 1:
                            y_full = selected_arrays[0]
                        elif len(selected_arrays) > 1:
                            y_full = np.hstack([arr.reshape(len(arr), -1) for arr in selected_arrays])
                    elif self.target_column:
                        prefixed = f"target_{self.target_column}"
                        if prefixed in data:
                            y_full = data[prefixed]
                        elif self.target_column in data:
                            y_full = data[self.target_column]

                    if y_full is None:
                        # Fallback auto-detection
                        for k in ["future_sequence", "y", "targets"]:
                            if k in data:
                                y_full = data[k]
                                if k == "targets" and hasattr(y_full, 'item'):
                                    item = y_full.item()
                                    if isinstance(item, dict):
                                        y_full = next(iter(item.values()))
                                break

                    if y_full is not None:
                        all_jury_x.append(x_full[indices])
                        all_jury_y.append(y_full[indices])

                data.close()
            except Exception as e:
                logger.error(f"⚠️ [LazyLoader] Failed to load jury from {chunk['path']}: {e}")

        if all_jury_x:
            self.jury_x = np.concatenate(all_jury_x, axis=0)
            if self.multi_output and all_jury_y and isinstance(all_jury_y[0], dict):
                # Merge list of per-chunk dicts into one dict of concatenated arrays
                combined: Dict[str, list] = {}
                for chunk_dict in all_jury_y:
                    for k, v in chunk_dict.items():
                        combined.setdefault(k, []).append(v)
                self.jury_y = {k: np.concatenate(v, axis=0) for k, v in combined.items()}
            else:
                self.jury_y = np.concatenate(all_jury_y, axis=0)
            n = len(self.jury_x)
            y_info = list(self.jury_y.keys()) if isinstance(self.jury_y, dict) else self.jury_y.shape
            logger.info(f"⚖️ [LazyLoader] Jury pool prepared: {n} samples held back. y: {y_info}")
        else:
            self.jury_x = None
            self.jury_y = None
