from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import asyncio
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import zstandard as zstd
from sqlalchemy import select, and_
from app.core.processing.processing_manager import ProcessingManager, AnalysisType
from app.core.processing.progress_reporter import ProgressReporter, ThrottlingStrategy
from app.api.routes.data.database import AsyncPostgresSessionLocal
from app.core.data.session_data_loader import SESSION_STEP_PRIORITY
from app.database.models import MLDatasetChunk, ModelPredictions
# ✅ Multi-output support
from app.core.ml.output_spec import (
    V8_3_CORE_OUTPUT_KEYS,
    split_output_loss,
    get_loss_spec_from_model,
    is_multi_output_model,
)


logger = logging.getLogger(__name__)


class TrainingEngineMixin:
    """Mixin: _train_model_async, _trainer_fit (training loop engine). _trainer_fit_v1 was deleted — confirmed dead code."""

    async def _train_model_async(
        self,
        model: Any,
        train_data: Any,
        val_data: Any,
        batch_size: int,
        epoch: int,
        task_id: str,
        is_generator: bool = False,
        train_targets: Any = None,
        val_targets: Any = None,
        reporter: Any = None,
        total_epochs: int = 1,
        last_val_metrics: Dict[str, Any] = None,
        target_column: str = None,
        weights_before: Any = None,  # ⚖️ NEW: For epoch-level jury validation
        train_data_obj: Any = None,  # ⚖️ NEW: Original data object for jury access
        max_m: int = 5               # 🚀 NEW: Dynamic micro-epoch budget
    ) -> Dict[str, Any]:
        """
        Pure async training loop for single epoch.
        """
        try:
            # 🔍 [DIAGNOSTIC] Deep inspection of inputs
            if epoch == 0:
                self.logger.info(f"🧪 [EPOCH {epoch+1}] TARGET VERIFICATION:")
                self.logger.info(f"   ├─ train_targets: type={type(train_targets)}, val={train_targets}")
                self.logger.info(f"   ├─ val_targets: type={type(val_targets)}, val={val_targets}")
                self.logger.info(f"   ├─ target_column (arg): {target_column}")
                self.logger.info(f"   └─ is_generator: {is_generator}")
                
                if isinstance(train_data, np.ndarray):
                    self.logger.info(f"   ├─ train_data (numpy): shape={train_data.shape}")
                elif hasattr(train_data, "__len__"):
                    self.logger.info(f"   ├─ train_data (obj): type={type(train_data)}, len={len(train_data)}")
            
            epoch_loss = 0.0
            epoch_mae = 0.0
            epoch_mse = 0.0
            history = None  # ✅ Guard against NameError in summary logs
            
            # ─────────────────────────────────────────────────────────
            # TRAINING PHASE
            # ─────────────────────────────────────────────────────────
            if is_generator:
                self.logger.info(f"[Epoch {epoch+1}] Training via Generator...")
                # 🎯 [NEW] Prepare jury pool for batch-level micro-validation (reshuffled each epoch)
                if hasattr(train_data, 'micro_val_holdback') and train_data.micro_val_holdback > 0:
                    train_data._prepare_epoch_jury()
                    logger.info(
                        f"⚖️ [EPOCH {epoch+1}] Jury pool ready for batch audits:\n"
                        f"   ├─ Jury samples: {len(train_data.jury_x) if train_data.jury_x is not None else 'N/A'}\n"
                        f"   ├─ Training holdback: {train_data.micro_val_holdback * 100:.0f}%\n"
                        f"   └─ Validation threshold: 2.0% (micro) vs 5.0% (epoch)"
                    )
                gen_flow = train_data.flow()
                num_batches = len(train_data)
                
                # 🎯 [DIAGNOSTIC] Batch Consumption State
                if epoch == 0:
                    self.logger.info(f"🎯 [EPOCH {epoch+1}] BATCH CONSUMPTION DIAGNOSTICS:")
                    self.logger.info(f"   ├─ Expected batches: {num_batches}")
                    self.logger.info(f"   ├─ Batch size: {batch_size}")
                    self.logger.info(f"   ├─ Generator type: {type(train_data).__name__}")
                    self.logger.info(f"   ├─ Generator shuffle: {getattr(train_data, 'shuffle', 'Unknown')}")
                    self.logger.info(f"   └─ Generator total sequences: {getattr(train_data, 'total_sequences', 'Unknown')}")
            else:
                num_batches = len(train_data) // batch_size
            
            # ✅ Warn if num_batches is suspiciously low (Audit Point 1)
            if epoch == 0 and num_batches < 10:
                self.logger.warning(
                    f"⚠️ [_train_model_async] SUSPICIOUSLY LOW num_batches: {num_batches}\n"
                    f"   len(train_data)={len(train_data)}, batch_size={batch_size}\n"
                    f"   This suggests train_data has only {len(train_data)} sequences!"
                )
            
            # ✅ CRITICAL CHECK: Ensure num_batches > 0 before entering loop (Audit Point 2)
            if num_batches == 0:
                self.logger.error(
                    f"❌ [EPOCH {epoch+1}] SKIPPING: num_batches=0!\n"
                    f"   len(train_data)={len(train_data)}, batch_size={batch_size}\n"
                    f"   This batch will not train at all!"
                )
                return {
                    "loss": float('inf'), "val_loss": float('inf'),
                    "mae": 0.0, "val_mae": 0.0, "mse": 0.0, "val_mse": 0.0,
                }
            
            # Shuffle  

            # -- shared progress reporter helper --
            async def _report(b_idx, b_loss, e_loss, e_mae, e_mse, count=None):
                # Use count for averages if provided (accurate when batches are skipped)
                divisor = count if count is not None else (b_idx + 1)
                avg_loss = e_loss / divisor if divisor > 0 else 0.0
                avg_mae  = e_mae  / divisor if divisor > 0 else 0.0
                avg_mse  = e_mse  / divisor if divisor > 0 else 0.0
                
                gp = int((((epoch * num_batches) + b_idx) /
                           (total_epochs * num_batches)) * 100)
                msg = (f"Epoch {epoch+1}/{total_epochs}: batch {b_idx+1}/{num_batches}"
                       f" - Batch Loss: {b_loss:.4f} | Avg Loss: {avg_loss:.4f}")
                vl = (last_val_metrics or {}).get("val_loss", 0.0) or 0.0
                vm = (last_val_metrics or {}).get("val_mae",  0.0) or 0.0
                vs = (last_val_metrics or {}).get("val_mse",  0.0) or 0.0
                tm = {
                    "loss": avg_loss, 
                    "mae": avg_mae, 
                    "mse": avg_mse,
                    "val_loss": vl, 
                    "val_mae": vm, 
                    "val_mse": vs,
                    "current_epoch": epoch + 1, "total_epochs": total_epochs,
                }
                if reporter:
                    await reporter.report_async(
                        progress=gp, 
                        message=msg,
                        trainingMetrics=tm,
                        loss=float(b_loss), 
                        avg_loss=float(avg_loss),
                    )
                else:
                    self.task_store.update_task(
                        task_id=task_id, 
                        status="processing",
                        progress=gp, 
                        message=msg,
                        metadata={
                            "loss": float(b_loss), 
                            "avg_loss": float(avg_loss),
                            "trainingMetrics": tm},
                    )

            if is_generator:
                # ── generator path ──────────────────────────────
                metric_names = [m.name if hasattr(m, 'name') else str(m) for m in model.metrics]
                batches_ran = 0
                
                # 📊 DIAGNOSTIC: Log expected batch count and sizes
                total_expected = num_batches * batch_size
                total_sequences = train_data.total_sequences if hasattr(train_data, 'total_sequences') else '?'
                diff = (total_expected - total_sequences) if isinstance(total_sequences, int) else '?'
                
                logger.info(
                    f"🎯 [EPOCH {epoch+1}] BATCH CONSUMPTION DIAGNOSTICS:\n"
                    f"   ├─ Expected batches: {num_batches}\n"
                    f"   ├─ Batch size: {batch_size}\n"
                    f"   ├─ Expected samples: {num_batches} × {batch_size} = {num_batches * batch_size}\n"
                    f"   ├─ Generator type: {type(train_data).__name__}\n"
                    f"   ├─ Generator shuffle: {train_data.shuffle if hasattr(train_data, 'shuffle') else '?'}\n"
                    f"   └─ Generator total sequences: {train_data.total_sequences if hasattr(train_data, 'total_sequences') else '?'}"
                )
                
                epoch_samples_training = 0
                samples_by_batch = []
                
                for batch_idx in range(num_batches):
                    try:
                        batch_x, batch_y = next(gen_flow)
                    except StopIteration:
                        self.logger.warning(f"⚠️ [EPOCH {epoch+1}] Generator exhausted early at batch {batch_idx}/{num_batches}")
                        break
                    except Exception as gen_err:
                        self.logger.error(f"❌ [EPOCH {epoch+1}] Generator error at batch {batch_idx}: {gen_err}")
                        break

                    # 🎯 TARGET ALIGNMENT VERIFICATION (Audit Point 3)
                    if batch_idx == 0:
                        self.logger.info(
                            f"🧪 [EPOCH {epoch+1}] TARGET VERIFICATION:\n"
                            f"   ├─ x.shape: {batch_x.shape}\n"
                            f"   ├─ y.shape: {batch_y.shape}\n"
                            f"   ├─ target_column: {target_column or 'None (Auto)'}\n"
                            f"   ├─ y_sample (future_seq): {batch_y[0].tolist() if hasattr(batch_y[0], 'tolist') else batch_y[0]}\n"
                            f"   └─ Aligned: {'YES ✓' if len(batch_x) == len(batch_y) else 'NO ❌'}"
                        )
                        if len(batch_x) != len(batch_y):
                            self.logger.error(f"❌ [EPOCH {epoch+1}] CRITICAL: Sample count mismatch! x={len(batch_x)}, y={len(batch_y)}")
                            break

                    # � DIAGNOSTIC: Track batch size
                    batch_size_actual = len(batch_x)
                    epoch_samples_training += batch_size_actual
                    samples_by_batch.append(batch_size_actual)
                    
                    if batch_idx % 20 == 0 or batch_size_actual < batch_size:
                        logger.debug(
                            f"   Batch {batch_idx+1:4d}/{num_batches}: "
                            f"size={batch_size_actual:4d} samples "
                            f"partial={'YES ⚠️' if batch_size_actual < batch_size else 'NO ✓'} "
                            f"cumulative={epoch_samples_training:6d}"
                        )

                    # 🚀 [MICRO-EPOCHS] Perform multiple weight updates on the same batch
                    # 🎯 [NEW] With batch-level jury audit and rollback on regression
                    m_idx = 0
                    # max_m is now passed as a parameter for dynamic decay
                    last_bl = float('inf')
                    bl, bm, bs = 0.0, 0.0, 0.0
                    
                    # ⚖️ BATCH AUDIT: Snapshot weights before micro-epochs
                    weights_before_batch = model.get_weights()
                    jury_loss_before = float('inf')
                    has_jury = hasattr(train_data, 'jury_x') and train_data.jury_x is not None
                    
                    if has_jury:
                        # Test against held-back jury pool BEFORE micro-epochs
                        j_size = min(batch_size, len(train_data.jury_x))
                        j_idx = np.random.choice(len(train_data.jury_x), size=j_size, replace=False)
                        jury_res_before = await asyncio.wait_for(
                            asyncio.to_thread(model.test_on_batch, train_data.jury_x[j_idx], train_data.jury_y[j_idx]),
                            timeout=300.0,
                        )
                        jury_loss_before = float(jury_res_before[0]) if isinstance(jury_res_before, (list, tuple)) else float(jury_res_before)
                    
                    while m_idx < max_m:
                        await asyncio.sleep(0)
                        try:
                            history = await asyncio.wait_for(
                                asyncio.to_thread(model.train_on_batch, batch_x, batch_y),
                                timeout=600.0,
                            )

                            if isinstance(history, (list, tuple)):
                                bl = float(history[0]) if len(history) > 0 else 0.0
                                bm = float(history[1]) if len(history) > 1 else 0.0
                                bs = float(history[2]) if len(history) > 2 else 0.0
                            else:
                                results = dict(zip(["loss"] + metric_names, 
                                            history if isinstance(history, (list, tuple)) else [history]))
                                
                                bl = float(results.get("loss", 0.0))
                                bm = float(results.get("mae", 0.0))
                                bs = float(results.get("mse", 0.0))
                            
                          
                            # New: < 0.5% relative improvement (or absolute < 1e-6 for tiny losses)
                            if last_bl > 0:
                                pct_improvement = (last_bl - bl) / last_bl
                                if (pct_improvement < 0.005 or (last_bl - bl) < 1e-6) and m_idx > 2:
                                    break
                            
                            last_bl = bl
                            m_idx += 1
                        except asyncio.TimeoutError:
                            self.logger.error(f"❌ Batch {batch_idx} timed out in generator — skipping")
                            bl = 0.0
                            break
                        except Exception as train_err:
                            self.logger.error(f"❌ Training error at batch {batch_idx}: {train_err}")
                            bl = 0.0
                            break
                    
                    # ⚖️ BATCH AUDIT: Test against jury pool AFTER micro-epochs for regression
                    if has_jury and np.isfinite(bl) and bl > 0:
                        j_size = min(batch_size, len(train_data.jury_x))
                        j_idx = np.random.choice(len(train_data.jury_x), size=j_size, replace=False)
                        jury_res_after = await asyncio.wait_for(
                            asyncio.to_thread(model.test_on_batch, train_data.jury_x[j_idx], train_data.jury_y[j_idx]),
                            timeout=300.0,
                        )
                        jury_loss_after = float(jury_res_after[0]) if isinstance(jury_res_after, (list, tuple)) else float(jury_res_after)
                        
                        # Rollback if jury loss regressed more than 2% (micro-val threshold)
                        if jury_loss_before > 0 and jury_loss_after > jury_loss_before * 1.02:
                            model.set_weights(weights_before_batch)
                            if batch_idx % 10 == 0:
                                logger.warning(
                                    f"⚖️ [BATCH AUDIT] Generator path - Rollback at batch {batch_idx+1}: "
                                    f"Jury loss regressed ({jury_loss_before:.4f} → {jury_loss_after:.4f}, "
                                    f"+{((jury_loss_after/jury_loss_before - 1) * 100):.1f}%)"
                                )
                            # Reset batch loss to indicate rollback
                            bl = jury_loss_after
                            continue
                            
                    if np.isfinite(bl):
                        epoch_loss += bl; epoch_mae += bm; epoch_mse += bs
                        batches_ran += 1
                    
                    if batch_idx % 4 == 0 or batch_idx == num_batches - 1:
                        await _report(batch_idx, bl, epoch_loss, epoch_mae, epoch_mse, count=batches_ran)
                
                final_batches_ran = batches_ran
                
                # 📊 DIAGNOSTIC: Comprehensive epoch summary
                min_batch_size = min(samples_by_batch) if samples_by_batch else 0
                max_batch_size = max(samples_by_batch) if samples_by_batch else 0
                avg_batch_size = epoch_samples_training / batches_ran if batches_ran > 0 else 0
                partial_batches = sum(1 for s in samples_by_batch if s < batch_size)
                
                logger.info(
                    f"📊 [EPOCH {epoch+1}] BATCH CONSUMPTION SUMMARY:\n"
                    f"   ├─ Batches ran: {batches_ran}/{num_batches} "
                    f"({'ALL RECEIVED ✅' if batches_ran == num_batches else f'INCOMPLETE ⚠️ ({num_batches - batches_ran} missing)'})\n"
                    f"   ├─ Total samples trained: {epoch_samples_training}\n"
                    f"   ├─ Batch sizes:\n"
                    f"   │  ├─ Min: {min_batch_size}\n"
                    f"   │  ├─ Max: {max_batch_size}\n"
                    f"   │  └─ Avg: {avg_batch_size:.1f}\n"
                    f"   └─ Partial batches (< {batch_size}): {partial_batches}\n"
                    f"\n"
                    f"   🔍 DATA QUALITY CHECK:\n"
                    f"   ├─ Expected samples: {num_batches * batch_size}\n"
                    f"   ├─ Actual samples: {epoch_samples_training}\n"
                    f"   ├─ Difference: {epoch_samples_training - (num_batches * batch_size)}\n"
                    f"   ├─ Loss this epoch: {epoch_loss / batches_ran if batches_ran > 0 else 'N/A':.4f}\n"
                    f"   ├─ History: {history}\n"
                    f"   └─ Status: {'DATA INTEGRITY OK ✅' if epoch_samples_training > 0 else 'NO DATA TRAINED ❌'}"
                )
                
                self.logger.info(f"📊 [EPOCH {epoch+1}] Generator pass complete. Batches: {batches_ran}/{num_batches}")
            else:
                queue: asyncio.Queue = asyncio.Queue(maxsize=3)

                # ✅ SHUFFLE: Re-shuffle every epoch for better generalization (Ref Code implementation)
                indices = np.random.permutation(len(train_data))

                async def _producer():
                    exc_to_raise = None
                    try:
                        for b_idx in range(num_batches):
                            start_idx = b_idx * batch_size
                            end_idx = start_idx + batch_size
                            batch_indices = indices[start_idx:end_idx]
                            
                            prepared = await asyncio.to_thread(
                                self._prepare_batch,
                                train_data, train_targets,
                                batch_indices,
                            )
                            await queue.put((b_idx, prepared))
                    except Exception as exc:
                        self.logger.error(f"❌ Batch producer error: {exc}")
                        exc_to_raise = exc
                    finally:
                        await queue.put(None)  # sentinel
                        if exc_to_raise:
                            raise exc_to_raise

                async def _consumer():
                    nonlocal epoch_loss, epoch_mae, epoch_mse
                    batches_ran = 0
                    
                    # Pre-fetch metric names for robust extraction (Bug 5)
                    metric_names = [m.name if hasattr(m, 'name') else str(m) for m in model.metrics]
                    
                    while True:
                        item = await queue.get()
                        if item is None:
                            break
                        b_idx, (b_x, b_y) = item
                        
                        # 🚀 [MICRO-EPOCHS] Perform multiple weight updates on the same batch
                        m_idx = 0
                        # max_m is now passed as a parameter for dynamic decay
                        last_bl = float('inf')
                        bl, bm, bs = 0.0, 0.0, 0.0
                        
                        # ⚖️ AUDIT SNAPSHOT
                        weights_before_batch = model.get_weights()
                        jury_loss_before = float('inf')
                        has_jury = hasattr(train_data, 'jury_x') and train_data.jury_x is not None
                        
                        if has_jury:
                            j_size = min(batch_size, len(train_data.jury_x))
                            j_idx = np.random.choice(len(train_data.jury_x), size=j_size, replace=False)
                            jury_res = await asyncio.to_thread(model.test_on_batch, train_data.jury_x[j_idx], train_data.jury_y[j_idx])
                            jury_loss_before = float(jury_res[0])

                        while m_idx < max_m:
                            await asyncio.sleep(0)
                            try:
                                history = await asyncio.wait_for(
                                    asyncio.to_thread(model.train_on_batch, b_x, b_y),
                                    timeout=600.0,
                                )
                                
                                # ✅ ROBUST EXTRACTION (Bug 5)
                                results = dict(zip(["loss"] + metric_names, 
                                               history if isinstance(history, (list, tuple)) else [history]))
                                
                                bl = float(results.get("loss", 0.0))
                                bm = float(results.get("mae") if results.get("mae") is not None else results.get("mean_absolute_error", 0.0))
                                bs = float(results.get("mse") if results.get("mse") is not None else results.get("mean_squared_error", 0.0))
                                
                                # 🔴 BUG FIX: Use RELATIVE improvement, not absolute 0.001
                                # Previous bug: 0.001 absolute threshold kills learning when loss < 0.01
                                # (e.g., 0.001→0.0009 has 0.0001 improvement < 0.001, exits early)
                                # New: < 0.5% relative improvement (or absolute < 1e-6 for tiny losses)
                                if last_bl > 0:
                                    pct_improvement = (last_bl - bl) / last_bl
                                    if (pct_improvement < 0.005 or (last_bl - bl) < 1e-6) and m_idx > 2:
                                        break
                                    
                                last_bl = bl
                                m_idx += 1
                                
                            except asyncio.TimeoutError:
                                self.logger.error(f"❌ Batch {b_idx} timed out — skipping")
                                bl = 0.0  # Ensure we don't use previous batch metrics
                                break
                                
                        # ⚖️ BATCH AUDIT: Test against jury pool AFTER micro-epochs for regression
                        if has_jury and np.isfinite(bl) and bl > 0:
                            j_size = min(batch_size, len(train_data.jury_x))
                            j_idx = np.random.choice(len(train_data.jury_x), size=j_size, replace=False)
                            jury_res_after = await asyncio.to_thread(model.test_on_batch, train_data.jury_x[j_idx], train_data.jury_y[j_idx])
                            jury_loss_after = float(jury_res_after[0]) if isinstance(jury_res_after, (list, tuple)) else float(jury_res_after)
                            
                            # Rollback if jury loss regressed more than 2% (micro-val threshold)
                            if jury_loss_before > 0 and jury_loss_after > jury_loss_before * 1.02:
                                model.set_weights(weights_before_batch)
                                if b_idx % 10 == 0:
                                    logger.warning(
                                        f"⚖️ [BATCH AUDIT] Numpy path - Rollback at batch {b_idx+1}: "
                                        f"Jury loss regressed ({jury_loss_before:.4f} → {jury_loss_after:.4f}, "
                                        f"+{((jury_loss_after/jury_loss_before - 1) * 100):.1f}%)"
                                    )
                                # Reset batch loss to indicate rollback
                                bl = jury_loss_after
                                continue
                        
                        # ✅ Guard against NaN/Inf and only accumulate if batch actually ran (Bug 3, 8)
                        if np.isfinite(bl) and (bl != 0.0 or bm != 0.0 or bs != 0.0):
                            epoch_loss += bl; epoch_mae += bm; epoch_mse += bs
                            batches_ran += 1
                            
                        if b_idx % 4 == 0 or b_idx == num_batches - 1:
                            # Use batches_ran for more accurate avg reporting during epoch
                            await _report(b_idx, bl, epoch_loss, epoch_mae, epoch_mse, count=batches_ran)
                    
                    self.logger.info(f"📊 [EPOCH {epoch+1}] Consumer pipeline complete. Batches: {batches_ran}/{num_batches}")
                    # Store the actual number of contributing batches for final averaging
                    return batches_ran

                prod_task = asyncio.create_task(_producer())
                cons_task = asyncio.create_task(_consumer())
                try:
                    # Capture batches_ran from consumer
                    results = await asyncio.gather(prod_task, cons_task)
                    final_batches_ran = results[1] if len(results) > 1 else num_batches
                except Exception:
                    prod_task.cancel()
                    cons_task.cancel()
                    raise
            
            # Average loss over batches
            avg_count = final_batches_ran
            if avg_count > 0:
                epoch_loss /= avg_count
                epoch_mae /= avg_count
                epoch_mse /= avg_count
            else:
                self.logger.error(f"❌ [TRAINING FAILURE] No valid batches processed at epoch {epoch+1}")
                epoch_loss = float('inf')
                epoch_mae = float('inf')
                epoch_mse = float('inf')
            
            # Validation
            if is_generator:
                try:
                    val_flow = val_data.flow()
                    val_loss_list = []
                    val_mae_list = []
                    val_mse_list = []
                    val_samples_list = [] # Track samples per batch for weighted avg
                    
                    max_val_batches = len(val_data)
                    
                    metric_names = [m.name if hasattr(m, 'name') else str(m) for m in model.metrics]

                    for v_idx in range(max_val_batches):
                        try:
                            v_batch = next(val_flow)
                            # ✅ FIX: Handle dict returns from flow() (new multi-target structure)
                            if isinstance(v_batch, dict):
                                v_x = v_batch.get('x')
                                v_y = v_batch.get('y')
                            else:
                                v_x, v_y = v_batch
                            
                            # v_loss_raw = await asyncio.to_thread(model.evaluate, v_x, v_y, verbose=0)
                            v_loss_raw = await asyncio.wait_for(
                                asyncio.to_thread(model.test_on_batch, v_x, v_y),
                                timeout=600.0
                            )
                            
                            if isinstance(v_loss_raw, (list, tuple, np.ndarray)):
                                lv = float(v_loss_raw[0]) if len(v_loss_raw) > 0 else 0.0
                                if not np.isfinite(lv):
                                    self.logger.warning(
                                        f"⚠️ [EPOCH {epoch+1}] Validation batch {v_idx} returned non-finite loss: {lv}\n"
                                        f"   ├─ x_bounds: [{np.min(v_x):.4f}, {np.max(v_x):.4f}]\n"
                                        f"   ├─ y_sample: {v_y[0].tolist() if hasattr(v_y[0], 'tolist') else v_y[0]}\n"
                                        f"   └─ Skipping batch."
                                    )
                                    continue
                                val_loss_list.append(lv)
                                val_mae_list.append(float(v_loss_raw[1]) if len(v_loss_raw) > 1 else 0.0)
                                val_mse_list.append(float(v_loss_raw[2]) if len(v_loss_raw) > 2 else 0.0)
                                val_samples_list.append(len(v_x))
                            else:
                                v_results = dict(zip(["loss"] + metric_names, 
                                             v_loss_raw if isinstance(v_loss_raw, (list, tuple, np.ndarray)) else [v_loss_raw]))
                                lv = float(v_results.get("loss", 0.0))
                                
                                if not np.isfinite(lv):
                                    self.logger.warning(
                                        f"⚠️ [EPOCH {epoch+1}] Validation batch {v_idx} returned non-finite loss: {lv}\n"
                                        f"   ├─ x_bounds: [{np.min(v_x):.4f}, {np.max(v_x):.4f}]\n"
                                        f"   ├─ y_sample: {v_y[0].tolist() if hasattr(v_y[0], 'tolist') else v_y[0]}\n"
                                        f"   └─ Skipping batch."
                                    )
                                    continue

                                val_loss_list.append(lv)
                                val_mae_list.append(float(v_results.get("mae") if v_results.get("mae") is not None else v_results.get("mean_absolute_error", 0.0)))
                                val_mse_list.append(float(v_results.get("mse") if v_results.get("mse") is not None else v_results.get("mean_squared_error", 0.0)))
                                val_samples_list.append(len(v_x))

                        except StopIteration:
                            break
                        except Exception as eval_err:
                            self.logger.warning(f"⚠️ Batch validation error: {eval_err}")
                            continue
                    
                    # ✅ WEIGHTED AVERAGE (Bug Fixed): Use sample weights to avoid partial batch bias
                    if val_loss_list and val_samples_list:
                        total_v_samples = sum(val_samples_list)
                        val_loss = sum(l * s for l, s in zip(val_loss_list, val_samples_list)) / total_v_samples
                        val_mae = sum(m * s for m, s in zip(val_mae_list, val_samples_list)) / total_v_samples
                        val_mse = sum(s * s_size for s, s_size in zip(val_mse_list, val_samples_list)) / total_v_samples
                    else:
                        val_loss, val_mae, val_mse = float('inf'), 0.0, 0.0
                    
                    # Log if validation came back as 0 despite having data
                    if val_loss == 0.0 and len(val_data) > 0:
                        self.logger.warning(f"⚠️ Validation metrics are 0.0 despite {len(val_data)} validation batches")
                        
                except Exception as gen_err:
                    self.logger.error(f"❌ Validation generator error: {gen_err}")
                    val_loss, val_mae, val_mse = float('inf'), 0.0, 0.0
            else:
                # Handle both DataFrame and numpy array
                if isinstance(val_data, pd.DataFrame):
                    val_x = val_data.values
                else:
                    val_x = val_data
                
                # ✅ FIXED: Use actual targets for validation
                if val_targets is not None:
                    val_y = val_targets
                else:
                    val_y = val_x  # Fallback: Autoencoder mode
                
                # ✅ FULL VALIDATION: Calculate total batches to cover everything (Bug Fixed)
                val_num_batches = (len(val_x) + batch_size - 1) // batch_size
                if val_num_batches == 0 and len(val_x) > 0:
                    val_num_batches = 1
                    
                val_loss_sum = 0.0
                val_mae_sum = 0.0
                val_mse_sum = 0.0
                val_samples_total = 0
                
                # Pre-fetch metric names
                metric_names = [m.name if hasattr(m, 'name') else str(m) for m in model.metrics]
                val_batches_ran = 0
                self.logger.debug(f"ℹ️ Running full validation on {val_num_batches} batches ({len(val_x)} samples)")

                for v_idx in range(val_num_batches):
                    start_idx = v_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(val_x))
                    
                    v_batch_x = val_x[start_idx:end_idx]
                    
                    # Ensure targets are also sliced correctly and handled as arrays
                    if isinstance(val_y, (pd.DataFrame, pd.Series)):
                        v_batch_y = val_y.iloc[start_idx:end_idx].values
                    else:
                        v_batch_y = val_y[start_idx:end_idx]
                    
                    # Yield to event loop
                    await asyncio.sleep(0)
                    
                    v_loss_raw = await asyncio.wait_for(
                        asyncio.to_thread(model.test_on_batch, v_batch_x, v_batch_y),
                        timeout=600.0
                    )
                    
                    v_results = dict(zip(["loss"] + metric_names, 
                                     v_loss_raw if isinstance(v_loss_raw, (list, tuple)) else [v_loss_raw]))
                    
                    lv = float(v_results.get("loss", 0.0))
                    if not np.isfinite(lv):
                        self.logger.warning(
                            f"⚠️ [EPOCH {epoch+1}] Validation batch {v_idx} (array) returned non-finite loss: {lv}\n"
                            f"   ├─ x_bounds: [{np.min(v_batch_x):.4f}, {np.max(v_batch_x):.4f}]\n"
                            f"   ├─ y_sample: {v_batch_y[0].tolist() if hasattr(v_batch_y[0], 'tolist') else v_batch_y[0]}\n"
                            f"   └─ Skipping batch."
                        )
                        continue

                    b_len = len(v_batch_x)
                    val_loss_sum += lv * b_len
                    val_mae_sum += float(v_results.get("mae", 0.0)) * b_len
                    val_mse_sum += float(v_results.get("mse", 0.0)) * b_len
                    val_samples_total += b_len
                    val_batches_ran += 1
                        
                if val_samples_total > 0:
                    val_loss = val_loss_sum / val_samples_total
                    val_mae = val_mae_sum / val_samples_total
                    val_mse = val_mse_sum / val_samples_total
                else:
                    val_loss = float('inf')
                    val_mae = 0.0
                    val_mse = 0.0
            
            # ⚖️ EPOCH-LEVEL JURY VALIDATION: Compare pre-epoch vs post-epoch on held-back jury pool
            # This catches entire-epoch regressions that batch audits may have missed
            if weights_before is not None and train_data_obj is not None and hasattr(train_data_obj, 'jury_x'):
                try:
                    jury_x = train_data_obj.jury_x
                    jury_y = train_data_obj.jury_y
                    
                    if jury_x is not None and len(jury_x) > 0 and np.isfinite(val_loss):
                        # 🔧 BUG #3 FIX: Capture post-epoch weights BEFORE modifying model
                        # Previously: post_epoch_weights were never captured, leading to pre-vs-pre comparison
                        post_epoch_weights = model.get_weights()  # ← CAPTURE POST-EPOCH FIRST
                        
                        # Test PRE-epoch weights on jury
                        model.set_weights(weights_before)
                        jury_loss_before = float(await asyncio.to_thread(model.test_on_batch, jury_x, jury_y))
                        
                        # Restore POST-epoch weights and test
                        model.set_weights(post_epoch_weights)
                        jury_loss_after = float(await asyncio.to_thread(model.test_on_batch, jury_x, jury_y))
                        
                        # Check if entire epoch regressed on jury (3% threshold for epoch-level, stricter than batch)
                        if jury_loss_before > 0 and jury_loss_after > jury_loss_before * 1.03:
                            pct_change = ((jury_loss_after / jury_loss_before) - 1) * 100
                            self.logger.warning(
                                f"⚖️ [EPOCH {epoch+1}] JURY REJECTED: Epoch regressed on jury pool\n"
                                f"   ├─ Jury loss: {jury_loss_before:.6f} → {jury_loss_after:.6f} (+{pct_change:.1f}%)\n"
                                f"   ├─ Val loss: {val_loss:.6f}\n"
                                f"   └─ Restoring pre-epoch weights (epoch will be retried with lower LR)"
                            )
                            # Return modified val_loss to signal regression to main loop
                            # This will trigger rollback logic
                            val_loss = jury_loss_after  # Override with jury verdict
                except Exception as e:
                    self.logger.debug(f"⚖️ Epoch-level jury audit failed: {e}")
            
            return {
                "loss": float(epoch_loss),
                "val_loss": float(val_loss),
                "mae": float(epoch_mae),
                "val_mae": float(val_mae),
                "mse": float(epoch_mse),
                "val_mse": float(val_mse),
            }
            
        except Exception as e:
            self.logger.error(f"Epoch training error: {str(e)}")
            return {
                "loss": float('inf'),
                "val_loss": float('inf'),
                "mae": 0.0,
                "val_mae": 0.0,
                "mse": 0.0,
                "val_mse": 0.0,
            }
    



    
    async def execute_model_build_with_pm(
        self,
        session_id: str,
        task_id: str,
        pm: ProcessingManager,
        model_config: Dict[str, Any],
        user_id: str = "anonymous",
    ) -> Dict[str, Any]:
        """
        Execute ML model building with injected ProcessingManager.
        
        Follows route pattern: PM instantiated in route, passed to AnalysisManager.
        
        Args:
            session_id: Session ID
            task_id: Task ID for progress tracking
            pm: Injected ProcessingManager (optimization layer)
            model_config: Model configuration dict
            
        Returns:
            dict with model_id, status, architecture
        """
        import dataclasses
        if dataclasses.is_dataclass(model_config) and not isinstance(model_config, type):
            model_config = dataclasses.asdict(model_config)
            
        try:
            self.logger.info(f"[{task_id}] Building model with PM optimization")
            
            # Build model
            result = await self.execute_model_build(
                session_id=session_id,
                task_id=task_id,
                model_config=model_config,
                user_id=user_id
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Model build with PM error: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def execute_model_training_with_pm(
        self,
        session_id: str,
        task_id: str,
        pm: ProcessingManager,
        train_config: Dict[str, Any],
        user_id: str = "anonymous",
    ) -> Dict[str, Any]:
        """
        Execute ML model training with injected ProcessingManager.
        
        Follows route pattern: PM instantiated in route for data streaming optimization.
        PM auto-selects strategy: SEQUENTIAL/PARALLEL_CHUNKING/SLICE_STREAMING
        
        Args:
            session_id: Session ID
            task_id: Task ID for progress tracking
            pm: Injected ProcessingManager (handles data streaming)
            train_config: Training configuration dict (model_id, epochs, batch_size, ml_preparation_ref)
            
        Returns:
            dict with epochs_completed, best_val_loss, status
        """
        import dataclasses
        if dataclasses.is_dataclass(train_config) and not isinstance(train_config, type):
            train_config = dataclasses.asdict(train_config)
            
        try:
            self.logger.info(f"[{task_id}] Training model with PM streaming optimization")
            
            # ✅ NEW: Extract ml_preparation_ref from train_config
            ml_prep_ref = train_config.get("ml_preparation_ref")
            
            # Train model
            result = await self.execute_model_training(
                session_id=session_id,
                task_id=task_id,
                model_id=train_config.get("model_id", ""),
                epochs=train_config.get("epochs", 50),
                batch_size=train_config.get("batch_size", 32),
                ml_preparation_ref=ml_prep_ref,  # ✅ Pass complex ref (handled in execute_model_training)
                user_id=user_id,
                is_classification=train_config.get("is_classification", False),
                selected_targets=train_config.get("selected_targets", []),
                # ✅ FIX: Resolve target_column from selected_targets if not explicitly set
                target_column=train_config.get("target_column") or (
                    (train_config.get("selected_targets") or [None])[0]
                ),
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Model training with PM error: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    # ────────────────────────────────────────────────────────────────
    # UNIFIED ANALYSIS EXECUTION (Single Step)
    # ────────────────────────────────────────────────────────────────

    async def _load_data_4_tier(
        self,
        session_id: str,
        task_id: str,
        request_data: Optional[List[Dict[str, Any]]] = None,
        exclude_step: str = None,
        data_type: str = "analysis",  # NEW: Route to correct TIER 0
        ml_dataset_name: str = None,  # ✅ NEW: Validate dataset name for ML pointers
        prefer_lazy: bool = False,    # ✅ NEW: Enable disk spooling for large ML data
    ) -> Tuple[Union[pd.DataFrame, Dict[str, Any]], str]:
        """
        Internal 4-tier data loader with dual TIER 0 pointers (analysis & ML splits).
        
        ✅ TIER 0a: `self.current_data` pointer - Analysis results (zero-latency)
        ✅ TIER 0b: `self.ml_train/val/test` pointers - ML splits (zero-latency) ← NEW
        
        Data Types:
        - "analysis": Check TIER 0a (current_data) first - used by mutation steps
        - "ml_train": Check TIER 0b (ml_train) - used by model training
        - "ml_validation": Check TIER 0b (ml_validation) - used by model training
        - "ml_test": Check TIER 0b (ml_test) - used by model evaluation
        
        Tier Priority:
        1. TIER 0: In-memory pointers (0a or 0b based on data_type) - ZERO latency
        2. TIER 1: Request Data (inline)
        3. TIER 2: AnalysisManager Cache (memory, dict lookup)
        4. TIER 3: Database (PostgreSQL)
        """
        # ─────────────────────────────────────────────────────────
        # TIER 0a: Analysis in-memory pointer (ZERO LATENCY)
        # 🔴 CRITICAL: MUST be fully merged complete dataset, not slice
        # ─────────────────────────────────────────────────────────
        if data_type == "analysis":
            if (self.current_data is not None and 
                self.current_session_id == session_id):
                
                # ✅ VALIDATION: Verify pointer is complete (not empty, not malformed)
                if len(self.current_data) == 0:
                    logger.error(
                        f"🔴 TIER 0a VALIDATION FAILED: current_data is EMPTY\n"
                        f"  ├─ This indicates slice was stored instead of merged result\n"
                        f"  └─ Check ProcessingManager._aggregate_slice_results()\n"
                    )
                    # Fall through to next tier to recover from DB
                elif len(self.current_data.columns) <= 8:
                    # Guard: if current_data is just raw OHLCV (≤8 cols: Time, Open, High,
                    # Low, Close, Volume, Spread, RealVolume) it is stale/under-enriched.
                    # The TA step and all downstream steps expect the most-enriched available
                    # result (100+ cols from TA, or 300+ from currency_indices). Using raw
                    # OHLCV would silently re-run TA without any prior enrichment context.
                    # Fall through to TIER 3 so the DB's highest-priority step is loaded.
                    logger.warning(
                        f"⚠️  TIER 0a VALIDATION FAILED: current_data has only "
                        f"{len(self.current_data.columns)} columns (likely raw OHLCV / stale pointer).\n"
                        f"  ├─ Falling through to TIER 3 to load most-enriched DB result.\n"
                        f"  └─ exclude_step={exclude_step}"
                    )
                    # Fall through to next tier
                else:
                    logger.info(
                        f"⚡ TIER 0a HIT: Analysis pointer\n"
                        f"  ├─ Rows: {len(self.current_data)} (COMPLETE MERGED DATASET)\n"
                        f"  ├─ Columns: {len(self.current_data.columns)}\n"
                        f"  └─ Latency: ZERO ms from previous step"
                    )
                    return self.current_data.copy(), "TIER0a_ANALYSIS_POINTER"
            
            logger.debug(f"⏭️  TIER 0a MISS: Session mismatch or not set")
        
        # ─────────────────────────────────────────────────────────
        # TIER 0b: ML split pointers (ZERO LATENCY)
        # 🔴 CRITICAL: MUST be complete splits, not slices or chunks
        # ✅ NEW: Validate dataset name matches to prevent cross-dataset contamination
        # ─────────────────────────────────────────────────────────
        elif data_type == "ml_train":
            if (self.ml_train is not None and 
                self.ml_session_id == session_id and
                (ml_dataset_name is None or self.ml_dataset_name == ml_dataset_name)):  # ✅ NEW: Validate dataset name
                
                # ✅ VALIDATION: Verify split is complete (not empty)
                if len(self.ml_train) == 0:
                    logger.error(
                        f"🔴 TIER 0b (ml_train) VALIDATION FAILED: Empty split\n"
                        f"  ├─ This indicates slice was stored instead of complete split\n"
                        f"  └─ Check ProcessingManager._aggregate_slice_results() and set_ml_data_pointers()\n"
                    )
                    # Fall through to next tier to recover from DB
                else:
                    logger.info(
                        f"⚡ TIER 0b HIT: ML train split\n"
                        f"  ├─ Rows: {len(self.ml_train)} (COMPLETE SPLIT)\n"
                        f"  ├─ Columns: {len(self.ml_train.columns)}\n"
                        f"  ├─ Dataset: {self.ml_dataset_name}\n"
                        f"  └─ Latency: ZERO ms"
                    )
                    return self.ml_train.copy(), "TIER0b_ML_TRAIN"
            elif self.ml_session_id == session_id and ml_dataset_name and self.ml_dataset_name != ml_dataset_name:
                logger.warning(
                    f"⚠️ TIER 0b MISMATCH (ml_train): Requested dataset '{ml_dataset_name}' but pointers are for '{self.ml_dataset_name}'\n"
                    f"  └─ Falling back to database to fetch correct dataset"
                )
            else:
                logger.debug(f"⏭️  TIER 0b MISS (ml_train): Not available")
        
        elif data_type == "ml_validation":
            if (self.ml_validation is not None and 
                self.ml_session_id == session_id and
                (ml_dataset_name is None or self.ml_dataset_name == ml_dataset_name)):  # ✅ NEW: Validate dataset name
                
                 # Guard: lazy dict pointers are valid, skip len() check
                if isinstance(self.ml_validation, dict):
                    return self.ml_validation, "TIER0b_ML_VALIDATION"  # lazy pointer, pass through
                
                # ✅ VALIDATION: Verify split is complete (not empty)
                if len(self.ml_validation) == 0:
                    logger.error(
                        f"🔴 TIER 0b (ml_validation) VALIDATION FAILED: Empty split\n"
                        f"  ├─ This indicates slice was stored instead of complete split\n"
                        f"  └─ Check ProcessingManager._aggregate_slice_results() and set_ml_data_pointers()\n"
                    )
                    # Fall through to next tier
                else:
                    logger.info(
                        f"⚡ TIER 0b HIT: ML validation split\n"
                        f"  ├─ Rows: {len(self.ml_validation)} (COMPLETE SPLIT)\n"
                        f"  ├─ Columns: {len(self.ml_validation.columns)}\n"
                        f"  ├─ Dataset: {self.ml_dataset_name}\n"
                        f"  └─ Latency: ZERO ms"
                    )
                    return self.ml_validation.copy(), "TIER0b_ML_VALIDATION"
            elif self.ml_session_id == session_id and ml_dataset_name and self.ml_dataset_name != ml_dataset_name:
                logger.warning(
                    f"⚠️ TIER 0b MISMATCH (ml_validation): Requested dataset '{ml_dataset_name}' but pointers are for '{self.ml_dataset_name}'\n"
                    f"  └─ Falling back to database to fetch correct dataset"
                )
            else:
                logger.debug(f"⏭️  TIER 0b MISS (ml_validation): Not available")
        
        elif data_type == "ml_test":
            if (self.ml_test is not None and 
                self.ml_session_id == session_id and
                (ml_dataset_name is None or self.ml_dataset_name == ml_dataset_name)):  # ✅ NEW: Validate dataset name
                
                # ✅ VALIDATION: Verify split is complete (not empty)
                if len(self.ml_test) == 0:
                    logger.error(
                        f"🔴 TIER 0b (ml_test) VALIDATION FAILED: Empty split\n"
                        f"  ├─ This indicates slice was stored instead of complete split\n"
                        f"  └─ Check ProcessingManager._aggregate_slice_results() and set_ml_data_pointers()\n"
                    )
                    # Fall through to next tier
                else:
                    logger.info(
                        f"⚡ TIER 0b HIT: ML test split\n"
                        f"  ├─ Rows: {len(self.ml_test)} (COMPLETE SPLIT)\n"
                        f"  ├─ Columns: {len(self.ml_test.columns)}\n"
                        f"  ├─ Dataset: {self.ml_dataset_name}\n"
                        f"  └─ Latency: ZERO ms"
                    )
                    return self.ml_test.copy(), "TIER0b_ML_TEST"
            elif self.ml_session_id == session_id and ml_dataset_name and self.ml_dataset_name != ml_dataset_name:
                logger.warning(
                    f"⚠️ TIER 0b MISMATCH (ml_test): Requested dataset '{ml_dataset_name}' but pointers are for '{self.ml_dataset_name}'\n"
                    f"  └─ Falling back to database to fetch correct dataset"
                )
            else:
                logger.debug(f"⏭️  TIER 0b MISS (ml_test): Not available")
        
        # TIER 1: Request Data
        if request_data:
            logger.info(f"✅ TIER 1: Loading {len(request_data)} rows from request")
            return pd.DataFrame(request_data), "TIER1_REQUEST"
        
        # TIER 2: Cache (Bypass for ML data types to prevent TIER 2 pollution)
        if session_id and data_type not in ["ml_train", "ml_validation", "ml_test"]:
            cached_data = await self.get_cached_data(session_id, task_id)
            if cached_data is not None:
                logger.info(f"✅ TIER 2: Cache HIT for session {session_id[:8]}... ({len(cached_data)} rows)")
                return pd.DataFrame(cached_data), "TIER2_CACHE"
        
        # TIER 3: Database
        if session_id:
            # ✅ NEW: For ML data types, load from ML dataset registry
            if data_type in ["ml_train", "ml_validation", "ml_test"]:
                logger.info(f"🔍 TIER 3: Cache MISS, fetching ML {data_type} from DB for session {session_id[:8]}...")
                
                from app.core.data.session_data_loader import get_ml_dataset_splits_by_name
                async with AsyncPostgresSessionLocal() as db:
                    # Map data_type to split_type
                    split_type_map = {
                        'ml_train': 'train',
                        'ml_validation': 'validation',
                        'ml_test': 'test'
                    }
                    split_type = split_type_map.get(data_type, 'train')
                    
                    # Try to load using the specific dataset_name
                    if ml_dataset_name:
                        logger.info(f"🔍 TIER 3: Attempting to load {split_type} split from dataset '{ml_dataset_name}'...")
                        split_data = await get_ml_dataset_splits_by_name(
                            session_id=session_id,
                            dataset_name=ml_dataset_name,
                            db=db,
                            split_type=split_type,
                            prefer_lazy=prefer_lazy
                        )
                        
                        if split_data is not None:
                            # ✅ FIXED: Log actual row count from dict or array
                            if isinstance(split_data, dict) and "sequences" in split_data:
                                row_count = len(split_data["sequences"])
                            elif isinstance(split_data, dict) and split_data.get("data_type") == "lazy_npz":
                                row_count = "LAZY_DISK_POINTER"
                            else:
                                row_count = len(split_data)
                                
                            logger.info(f"✅ TIER 3: Successfully loaded {split_type} split ({row_count})")
                            return split_data, f"TIER3_DATABASE_ML_SPLITS_{data_type.upper()}"
                        else:
                            logger.warning(f"⚠️ TIER 3: Failed to load {split_type} split from dataset '{ml_dataset_name}'")
                    else:
                        logger.warning(f"⚠️ TIER 3: No ml_dataset_name provided, cannot load ML data")
            
            # Fallback: Load latest analysis data (for non-ML data types)
            else:
                logger.info(f"🔍 TIER 3: Cache MISS, fetching standard data from DB for session {session_id[:8]}...")
                from app.core.data.session_data_loader import (
                    get_latest_session_data_excluding_step,
                    SESSION_STEP_PRIORITY,
                )
                async with AsyncPostgresSessionLocal() as db:
                    db_data = await get_latest_session_data_excluding_step(
                        session_id=session_id,
                        db=db,
                        exclude_step=exclude_step,
                        task_id=task_id,
                        as_dataframe=True,
                    )
                    if db_data is None or (isinstance(db_data, pd.DataFrame) and db_data.empty):
                        # Session not found in DB - could be expired/cleaned up
                        logger.warning(
                            f"⚠️ TIER 3: No data found for session {session_id[:8]}... in database.\n"
                            f"  ├─ This session may have been cleaned up or never persisted.\n"
                            f"  ├─ exclude_step={exclude_step}\n"
                            f"  └─ Checked steps: {SESSION_STEP_PRIORITY}"
                        )
                        raise ValueError(
                            f"Session {session_id} not found in database. "
                            f"It may have expired or been cleaned up. "
                            f"Please create a new analysis session."
                        )
                    
                    # ── Identify which step was actually loaded ──────────────────
                    # SESSION_STEP_PRIORITY walks most-enriched → least-enriched.
                    # Knowing which step was loaded tells us whether currency_indices
                    # data is present in what we're returning to the next step.
                    # This is the key diagnostic for the pointer propagation bug.
                    db_df = db_data if isinstance(db_data, pd.DataFrame) else pd.DataFrame(db_data)
                    loaded_step = "unknown"
                    for _step_name in SESSION_STEP_PRIORITY:
                        if _step_name == exclude_step:
                            continue
                        # Identify by column count: each step adds columns on top of previous.
                        # currency_indices adds Dollar_* columns; technical_analysis adds ~330+;
                        # data_source has ≤8 (OHLCV only).
                        # We can't query DB step_name here (already outside the db context),
                        # so infer from column count as a heuristic.
                        break  # we just need the first non-excluded step name as candidate
                    # Better: infer from column count
                    ncols = len(db_df.columns)
                    if ncols > 100:
                        loaded_step = "technical_analysis (or more enriched)"
                    elif ncols > 13:
                        loaded_step = "currency_indices (Dollar_* columns present)"
                    elif ncols > 8:
                        loaded_step = "footprint_ingestion or partial enrichment"
                    else:
                        loaded_step = "data_source (RAW OHLCV — currency_indices data NOT included)"
                    
                    logger.info(
                        f"✅ TIER 3: Loaded {len(db_data)} rows × {ncols} cols from database\n"
                        f"  ├─ Inferred step: {loaded_step}\n"
                        f"  ├─ exclude_step: {exclude_step}\n"
                        f"  └─ session: {session_id[:8]}"
                    )
                    
                    # ── TIER 3 → TIER 2 backfill ────────────────────────────────
                    # Populate in-memory cache so subsequent calls within the TTL
                    # window hit cache instead of DB (avoids re-initialization loop).
                    try:
                        await self.cache_session_data(
                            session_id=session_id,
                            data=db_data,
                            source_step="tier3_db_load",
                            ttl_seconds=1800  # 30 min TTL
                        )
                        logger.info(f"📌 TIER 3→2 backfill: Cached {len(db_data)} rows for session {session_id[:8]}")
                    except Exception as _cache_err:
                        logger.warning(f"⚠️ TIER 3→2 backfill failed (non-fatal): {_cache_err}")
                    
                    # ── TIER 3 → TIER 0a backfill ───────────────────────────────
                    # CRITICAL: Update the in-memory TIER 0a pointer so the NEXT
                    # step in this pipeline run (e.g. TA after loading currency_indices,
                    # or SNR after loading TA) gets the correct enriched data from
                    # memory instead of falling through to TIER 3 again.
                    #
                    # Without this backfill:
                    #   currency_indices runs → TIER 0a set to 349-col result ✅
                    #   Server restart / session reload → TIER 0a = None
                    #   TA runs → TIER 0a miss → TIER 3 loads currency_indices ✅
                    #   TA stores 680-col result → TIER 0a set to 680-col result ✅
                    #   SNR runs → TIER 0a hit ✅
                    #
                    # With this backfill, TIER 0a is warm after the TIER 3 load so
                    # the pipeline behaves identically whether or not there was a restart.
                    #
                    # Guard: only backfill if the loaded data is enriched enough to
                    # be useful as a TIER 0a starting point for the next step.
                    # Raw OHLCV (≤8 cols) is not useful — the next step would just
                    # re-enrich from scratch, which is correct.
                    if ncols > 8:
                        try:
                            self.current_data = db_df
                            self.current_session_id = session_id
                            logger.info(
                                f"📌 TIER 3→0a backfill: Set current_data to "
                                f"{len(db_df)} rows × {ncols} cols for session {session_id[:8]}\n"
                                f"  └─ Inferred step: {loaded_step}"
                            )
                        except Exception as _t0a_err:
                            logger.warning(f"⚠️ TIER 3→0a backfill failed (non-fatal): {_t0a_err}")
                    
                    return db_df, "TIER3_DATABASE"
        
        raise ValueError("Insufficient parameters to load data (no data or session_id)")
    
    async def cascade_clear_data_pointers_for_ml(
        self, 
        ml_session_id: str
    ) -> Dict[str, Any]:
        """
                
        When ML training pointers are set, clear analysis pointers to save memory.
        
        Memory Pattern:
        - Before: current_data (50K rows, 50MB) + ml_train/val/test (50MB)
        - After: ml_train/val/test only (50MB)
        - Saves: 50MB
        
        Args:
            ml_session_id: Session ID owning ML splits
        
        Returns:
            Dict with freed_mb and total_ml_mb metrics
        """
        
        # Calculate memory before clearing
        before_bytes = 0
        if self.current_data is not None:
            before_bytes = self.current_data.memory_usage(deep=True).sum()
        
        logger.info(
            f"🔄 CASCADE CLEAR: Before state\n"
            f"  ├─ current_data: {'Yes' if self.current_data is not None else 'No'} "
            f"({before_bytes/1e6:.1f}MB)\n"
            f"  ├─ current_session_id: {self.current_session_id}\n"
            f"  ├─ ml_train: {'Yes' if self.ml_train is not None else 'No'}\n"
            f"  ├─ ml_validation: {'Yes' if self.ml_validation is not None else 'No'}\n"
            f"  └─ ml_session_id: {self.ml_session_id}"
        )
        
        # CASCADE CLEAR: Remove analysis pointers (no longer needed)
        self.current_data = None
        self.current_session_id = None
        
        # Set ML session ID to track ownership
        self.ml_session_id = ml_session_id
        
        # Calculate memory after
        after_bytes = 0
        if self.ml_train is not None:
            after_bytes += self.ml_train.memory_usage(deep=True).sum()
        if self.ml_validation is not None:
            after_bytes += self.ml_validation.memory_usage(deep=True).sum()
        if self.ml_test is not None:
            after_bytes += self.ml_test.memory_usage(deep=True).sum()
        
        logger.info(
            f"🔄 CASCADE CLEAR: After state\n"
            f"  ├─ current_data: None (freed {before_bytes/1e6:.1f}MB!)\n"
            f"  ├─ current_session_id: None\n"
            f"  ├─ ml_session_id: {self.ml_session_id}\n"
            f"  └─ Total ML pointers: {after_bytes/1e6:.1f}MB"
        )
        
        return {
            'freed_mb': before_bytes / 1e6,
            'total_ml_mb': after_bytes / 1e6,
            'efficiency_gain': (before_bytes / 1e6)
        }


    async def _trainer_fit(
        self,
        model,
        train_data,
        num_batches,
        batch_size,
        epoch,
        task_id,
        reporter,
        total_epochs,
        last_val_metrics=None,
        target_column=None,
        max_m=5,
        dyn_threshold=1.15,
        # ── ports from _train_model_async ──────────────────────────
        val_data=None,
        train_targets=None,
        val_targets=None,
        is_generator=False,
        weights_before=None,
        train_data_obj=None,
        continual_state=None,  # ✅ UPGRADE: Persistent state across calls
    ):
        """
        TODO: Move this humongous functtion to ml/ module (see ml/trainer.py)
        Keras .fit() Clone with Sample-Weighted Metrics & Jury SOS Safety.
        """
        

        if num_batches == 0:
            self.logger.error(f"❌ [EPOCH {epoch+1}] num_batches=0 — skipping epoch")
            return {
                "loss": float('inf'), "val_loss": float('inf'),
                "mae": 0.0, "val_mae": 0.0, "mse": 0.0, "val_mse": 0.0,
            }

        # ── 1. Reset Stateful Metrics ────────────────────────
        for metric in model.metrics:
            metric.reset_state()

        # ── 2. Progress Reporter - Keep Frontend on the loop ─
        async def _report(b_idx, b_loss, epoch_loss, epoch_mae, epoch_mse, epoch_samples,
                          jury_loss_before=None, jury_loss_after=None, fresh_val=None,
                          core_epoch_loss=None, per_head_val_loss=None):
            # Fallback to batch metrics if no successful samples yet
            avg_loss = epoch_loss / epoch_samples if epoch_samples > 0 else b_loss
            avg_mae = epoch_mae / epoch_samples if epoch_samples > 0 else 0.0
            avg_mse = epoch_mse / epoch_samples if epoch_samples > 0 else 0.0
            avg_core = core_epoch_loss / epoch_samples if (core_epoch_loss and epoch_samples > 0) else avg_loss

            total_steps = total_epochs * num_batches
            current_step = (epoch * num_batches) + b_idx + 1
            gp = min(100, max(0, int(round((current_step / total_steps) * 100))))

            msg = (f"Epoch {epoch+1}/{total_epochs}: "
                   f"batch {b_idx+1}/{num_batches} - Loss: {b_loss:.4f}")

            if epoch_samples > 0:
                msg += f" | Avg: {avg_loss:.4f}"
            else:
                msg += " | Jury Auditing..."

            # Show jury core/full if available
            jb_core = _jury_core(jury_loss_before)
            ja_core = _jury_core(jury_loss_after)
            if jb_core is not None:
                if ja_core is not None:
                    reg_pct = (ja_core / jb_core - 1.0) * 100
                    msg += f" | Jury(core): {jb_core:.4f}→{ja_core:.4f} ({'+' if reg_pct > 0 else ''}{reg_pct:.1f}%)"
                else:
                    msg += f" | Jury(core): {jb_core:.4f}"

            v_src = fresh_val if fresh_val else (last_val_metrics or {})
            tm = {
                "loss": avg_loss, "mae": avg_mae, "mse": avg_mse,
                # NEW: core loss breakdown
                "core_loss": avg_core,
                "val_loss": v_src.get("val_loss", 0.0),
                "val_mae": v_src.get("val_mae", 0.0),
                "val_mse": v_src.get("val_mse", 0.0),
                "core_val_loss": v_src.get("core_val_loss", 0.0),  # NEW
                "jury_core_before": float(jb_core) if jb_core is not None else None,
                "jury_core_after": float(ja_core) if ja_core is not None else None,
                "per_head_val_loss": per_head_val_loss,  # NEW — dict or None
                "current_epoch": epoch + 1, "total_epochs": total_epochs,
            }
            if reporter:
                await reporter.report_async(progress=gp, message=msg,
                                            trainingMetrics=tm,
                                            loss=float(b_loss), avg_loss=float(avg_loss))
            else:
                self.task_store.update_task(
                    task_id=task_id, status="processing", progress=gp, message=msg,
                    metadata={"loss": float(b_loss), "avg_loss": float(avg_loss), "trainingMetrics": tm}
                )

            if b_idx % 10 == 0 or b_idx == num_batches - 1:
                self.logger.info(f"⚡ [EPOCH {epoch+1}] Batch {b_idx+1}/{num_batches} | "
                                 f"Loss: {b_loss:.5f} | Avg: {avg_loss:.5f} | Core: {avg_core:.5f}")

        # ── 3. Compiled SOS Training Step (Graph Mode) ────────────────────
        # NOTE: compiled_sos_step accepts y as either a single tensor (single-output
        # models) or a dict of tensors (multi-output models like V8.3).  Keras's
        # model.compute_loss handles both cases transparently — it routes each
        # entry in a dict y to the matching named output head.
        # b_mae/b_mse are computed against main_output (or the single output).
        @tf.function
        def compiled_sos_step(x, y, m_limit):
            last_loss = tf.constant(1e9, dtype=tf.float32)
            for m in tf.range(m_limit):
                with tf.GradientTape() as tape:
                    y_pred = model(x, training=True)
                    loss = model.compute_loss(x, y, y_pred)

                # ✅ NaN Safety: Prevent poisoning weights if loss goes NaN
                if tf.math.is_nan(loss) or tf.math.is_inf(loss):
                    tf.print("⚠️ NaN/Inf loss detected! Skipping gradient update.")
                    break

                grads = tape.gradient(loss, model.trainable_variables)

                # ✅ GRADIENT FILTERING: Prevent crash on None gradients AND NaN gradients
                valid_grads_vars = []
                for g, v in zip(grads, model.trainable_variables):
                    if g is not None:
                        g_clipped = tf.clip_by_norm(g, 1.0)
                        if not tf.math.reduce_any(tf.math.is_nan(g_clipped)) and not tf.math.reduce_any(tf.math.is_inf(g_clipped)):
                            valid_grads_vars.append((g_clipped, v))

                if valid_grads_vars:
                    model.optimizer.apply_gradients(valid_grads_vars)

                # Inline early-stop: < 0.5% relative improvement after 2 micro-steps
                if m > 2 and (last_loss - loss) / (last_loss + 1e-8) < 0.005: break
                last_loss = loss

            # MAE/MSE against main_output (or full output tensor for single-output models)
            y_pred_final = model(x, training=False)
            if isinstance(y_pred_final, dict):
                _yp = y_pred_final.get("main_output", next(iter(y_pred_final.values())))
                _yt = y["main_output"] if isinstance(y, dict) else y
            else:
                _yp = y_pred_final
                _yt = y
            b_mae = tf.reduce_mean(tf.abs(tf.cast(_yt, tf.float32) - _yp))
            b_mse = tf.reduce_mean(tf.square(tf.cast(_yt, tf.float32) - _yp))
            return loss, b_mae, b_mse

        # ── get_jury_loss: returns FULL aggregate loss (for logging) ───────────
        # Kept as a single-scalar function because it is called inside tf.function
        # contexts.  Core/full split is done outside graph mode via split_output_loss.
        @tf.function
        def get_jury_loss(jx, jy):
            """Graph-compiled helper — full aggregate loss for jury audits."""
            pred = model(jx, training=False)
            return model.compute_loss(jx, jy, pred)

        # ── compiled_val_step: graph-mode validation ────────────────────────────
        # Returns (full_loss, mae, mse) over main_output for per-batch accumulation.
        # Per-head breakdown is computed outside the graph loop via split_output_loss.
        @tf.function
        def compiled_val_step(vx, vy):
            """Graph-mode validation for maximum throughput."""
            pred = model(vx, training=False)
            v_loss = model.compute_loss(vx, vy, pred)
            if isinstance(pred, dict):
                _yp = pred.get("main_output", next(iter(pred.values())))
                _yt = vy["main_output"] if isinstance(vy, dict) else vy
            else:
                _yp = pred
                _yt = vy
            _yt_f = tf.cast(_yt, tf.float32)
            v_mae = tf.reduce_mean(tf.abs(_yt_f - _yp))
            v_mse = tf.reduce_mean(tf.square(_yt_f - _yp))
            return v_loss, v_mae, v_mse

        # ── 4. Helpers (Jury, Replay & Metric Extraction) ────────────────
        # ✅ UPGRADE: Extract state from persistent container
        if continual_state is None:
            # Fallback for isolated testing — must match full continual_state structure
            # so good_examples_storage persists correctly across epochs
            continual_state = {
                "engine_state": {
                    "jury_x": None, "jury_y": None,
                    "ptr": 0, "count": 0,
                    "good_ptr": 0, "good_count": 0
                },
                "replay_storage": [None] * 2000,
                "good_examples_storage": [None] * 1000  # ✅ FIX: persist forgetting state
            }

        engine_state = continual_state["engine_state"]
        replay_storage = continual_state["replay_storage"]          # HARD examples buffer
        good_examples_storage = continual_state["good_examples_storage"]  # GOOD examples buffer
        MAX_REPLAY_SAMPLES = len(replay_storage)
        MAX_GOOD_SAMPLES = len(good_examples_storage)

        # ── Pre-fetch loss spec once (not inside each batch) ──────────────────
        _loss_spec = get_loss_spec_from_model(model)
        _is_multi_output = is_multi_output_model(model)

        if (train_data_obj is not None and
            hasattr(train_data_obj, '_prepare_epoch_jury') and
            getattr(train_data_obj, 'micro_val_holdback', 0) > 0):
            try:
                train_data_obj._prepare_epoch_jury()
                if train_data_obj.jury_x is not None:
                    # ✅ MEMORY OPTIMIZATION: Zero-copy tf.constant caching
                    # Multi-output: cache dict of TF tensors; single-output: single tensor
                    if isinstance(train_data_obj.jury_y, dict):
                        engine_state["jury_x"] = tf.constant(train_data_obj.jury_x, dtype=tf.float32)
                        engine_state["jury_y"] = {
                            k: tf.constant(v, dtype=tf.float32)
                            for k, v in train_data_obj.jury_y.items()
                        }
                    else:
                        engine_state["jury_x"] = tf.constant(train_data_obj.jury_x, dtype=tf.float32)
                        engine_state["jury_y"] = tf.constant(train_data_obj.jury_y, dtype=tf.float32)

                n_jury = len(train_data_obj.jury_x) if train_data_obj.jury_x is not None else 0
                self.logger.info(
                    f"⚖️ [EPOCH {epoch+1}] Jury pool cached:\n"
                    f"   ├─ Samples: {n_jury}\n"
                    f"   └─ Holdback: {train_data_obj.micro_val_holdback * 100:.1f}%"
                )
            except Exception as j_err:
                self.logger.warning(f"⚠️ Could not cache jury pool: {j_err}")

        has_jury = (train_data_obj is not None and
                    hasattr(train_data_obj, 'jury_x') and
                    train_data_obj.jury_x is not None)

        async def run_jury(subset=True):
            """
            Holdback jury: returns (core_loss, full_loss) dict.
            ─ core_loss: weighted loss over V8_3_CORE_OUTPUT_KEYS — drives rollback decisions.
            ─ full_loss: total weighted aggregate — for logging/display only.
            Returns None if jury is not available.
            """
            if not has_jury or train_data_obj.jury_x is None:
                return None

            try:
                # Slice from cached TF tensors if available
                if engine_state["jury_x"] is not None:
                    jx_full = engine_state["jury_x"]
                    jy_full = engine_state["jury_y"]  # dict or tensor

                    if subset and tf.shape(jx_full)[0] > batch_size:
                        idx = tf.random.shuffle(tf.range(tf.shape(jx_full)[0]))[:batch_size]
                        fjx = tf.gather(jx_full, idx)
                        if isinstance(jy_full, dict):
                            fjy = {k: tf.gather(v, idx) for k, v in jy_full.items()}
                        else:
                            fjy = tf.gather(jy_full, idx)
                    else:
                        fjx, fjy = jx_full, jy_full

                    full_loss = float(get_jury_loss(fjx, fjy))

                    # Compute core loss via split_output_loss (eager, outside @tf.function)
                    if _is_multi_output and isinstance(fjy, dict):
                        pred = model(fjx, training=False)
                        pred_np = {k: v.numpy() if hasattr(v, 'numpy') else v for k, v in pred.items()}
                        fjy_np = {k: v.numpy() if hasattr(v, 'numpy') else v for k, v in fjy.items()}
                        core_loss, _, _ = split_output_loss(
                            model, fjy_np, pred_np,
                            loss_spec=_loss_spec, core_keys=V8_3_CORE_OUTPUT_KEYS, import_tf=tf
                        )
                    else:
                        core_loss = full_loss  # single-output: core == full

                    return {"core": core_loss, "full": full_loss}

                # Fallback to numpy path
                jx = train_data_obj.jury_x
                jy = train_data_obj.jury_y
                if subset and len(jx) > batch_size:
                    idx_np = np.random.choice(len(jx), batch_size, replace=False)
                    fjx_np, fjy_sub = jx[idx_np], (
                        {k: v[idx_np] for k, v in jy.items()} if isinstance(jy, dict)
                        else jy[idx_np]
                    )
                else:
                    fjx_np, fjy_sub = jx, jy

                fjx_tf = tf.constant(fjx_np, dtype=tf.float32)
                fjy_tf = (
                    {k: tf.constant(v, dtype=tf.float32) for k, v in fjy_sub.items()}
                    if isinstance(fjy_sub, dict)
                    else tf.constant(fjy_sub, dtype=tf.float32)
                )
                full_loss = float(get_jury_loss(fjx_tf, fjy_tf))

                if _is_multi_output and isinstance(fjy_sub, dict):
                    pred = model(fjx_tf, training=False)
                    pred_np = {k: v.numpy() if hasattr(v, 'numpy') else v for k, v in pred.items()}
                    core_loss, _, _ = split_output_loss(
                        model, fjy_sub, pred_np,
                        loss_spec=_loss_spec, core_keys=V8_3_CORE_OUTPUT_KEYS, import_tf=tf
                    )
                else:
                    core_loss = full_loss

                return {"core": core_loss, "full": full_loss}

            except Exception:
                # Final fallback: use model.test_on_batch, return full only
                try:
                    jx, jy = train_data_obj.jury_x, train_data_obj.jury_y
                    res = await asyncio.to_thread(model.test_on_batch, jx, jy)
                    fl = float(res[0]) if isinstance(res, (list, tuple, np.ndarray)) else float(res)
                    return {"core": fl, "full": fl}
                except Exception:
                    return None

        async def run_memory_check(subset_size=None):
            """
            Forgetting detector: returns core_loss on previously-easy examples.
            Uses good_examples_storage (low-loss samples), NOT hard examples.
            """
            good_count = engine_state.get("good_count", 0)
            if good_count == 0:
                return None
            n = subset_size or min(batch_size, good_count)
            indices = np.random.choice(good_count, n, replace=False)
            # Cast float16 back to float32 for TF inference
            rx = np.array([good_examples_storage[i][0] for i in indices], dtype=np.float32)
            ry_raw = [good_examples_storage[i][1] for i in indices]

            try:
                rx_tf = tf.constant(rx)
                # Determine if stored y examples are dict-shaped or array
                if isinstance(ry_raw[0], dict):
                    ry_tf = {k: tf.constant(np.array([item[k] for item in ry_raw], dtype=np.float32))
                             for k in ry_raw[0].keys()}
                else:
                    ry_tf = tf.constant(np.array(ry_raw, dtype=np.float32))

                full_loss = float(get_jury_loss(rx_tf, ry_tf))

                if _is_multi_output and isinstance(ry_tf, dict):
                    pred = model(rx_tf, training=False)
                    pred_np = {k: v.numpy() if hasattr(v, 'numpy') else v for k, v in pred.items()}
                    ry_np = {k: v.numpy() if hasattr(v, 'numpy') else np.array(v) for k, v in ry_tf.items()}
                    core_loss, _, _ = split_output_loss(
                        model, ry_np, pred_np,
                        loss_spec=_loss_spec, core_keys=V8_3_CORE_OUTPUT_KEYS, import_tf=tf
                    )
                    return core_loss
                return full_loss
            except Exception:
                res = await asyncio.to_thread(model.test_on_batch, rx,
                                              {k: v.numpy() for k, v in ry_tf.items()} if isinstance(ry_tf, dict) else np.array(ry_raw, dtype=np.float32))
                fl = float(res[0]) if isinstance(res, (list, tuple, np.ndarray)) else float(res)
                return fl

        async def run_epoch_jury():
            """Full holdback jury for epoch-level audit. Returns (core_loss, full_loss) or None."""
            if not has_jury or train_data_obj.jury_x is None:
                return None
            try:
                if engine_state["jury_x"] is not None:
                    full_loss = float(get_jury_loss(engine_state["jury_x"], engine_state["jury_y"]))
                    if _is_multi_output and isinstance(engine_state["jury_y"], dict):
                        pred = model(engine_state["jury_x"], training=False)
                        pred_np = {k: v.numpy() if hasattr(v, 'numpy') else v for k, v in pred.items()}
                        jy_np = {k: v.numpy() if hasattr(v, 'numpy') else np.array(v) for k, v in engine_state["jury_y"].items()}
                        core_loss, _, _ = split_output_loss(
                            model, jy_np, pred_np,
                            loss_spec=_loss_spec, core_keys=V8_3_CORE_OUTPUT_KEYS, import_tf=tf
                        )
                    else:
                        core_loss = full_loss
                    return {"core": core_loss, "full": full_loss}

                fjx = tf.constant(train_data_obj.jury_x, dtype=tf.float32)
                jy = train_data_obj.jury_y
                fjy = (
                    {k: tf.constant(v, dtype=tf.float32) for k, v in jy.items()}
                    if isinstance(jy, dict)
                    else tf.constant(jy, dtype=tf.float32)
                )
                full_loss = float(get_jury_loss(fjx, fjy))

                if _is_multi_output and isinstance(fjy, dict):
                    pred = model(fjx, training=False)
                    pred_np = {k: v.numpy() if hasattr(v, 'numpy') else v for k, v in pred.items()}
                    jy_np = {k: v.numpy() if hasattr(v, 'numpy') else np.array(v) for k, v in fjy.items()}
                    core_loss, _, _ = split_output_loss(
                        model, jy_np, pred_np,
                        loss_spec=_loss_spec, core_keys=V8_3_CORE_OUTPUT_KEYS, import_tf=tf
                    )
                else:
                    core_loss = full_loss
                return {"core": core_loss, "full": full_loss}

            except Exception as e1:
                try:
                    jx, jy = train_data_obj.jury_x, train_data_obj.jury_y
                    res = await asyncio.to_thread(model.test_on_batch, jx, jy)
                    fl = float(res[0]) if isinstance(res, (list, tuple, np.ndarray)) else float(res)
                    return {"core": fl, "full": fl}
                except Exception as e2:
                    self.logger.debug(f"⚖️ Epoch jury test failed. err1: {e1} | err2: {e2}")
                    return None

        def _jury_core(jury_result) -> Optional[float]:
            """Extract core_loss from run_jury / run_epoch_jury result."""
            if jury_result is None:
                return None
            if isinstance(jury_result, dict):
                return jury_result.get("core")
            return float(jury_result)  # legacy scalar

        def _jury_full(jury_result) -> Optional[float]:
            """Extract full_loss from run_jury / run_epoch_jury result."""
            if jury_result is None:
                return None
            if isinstance(jury_result, dict):
                return jury_result.get("full")
            return float(jury_result)

        def _sample_hard_weighted(n: int, current_epoch: int, decay: float = 0.88):
            """
            Sample n hard examples from replay_storage with age-priority weighting.

            Each stored example carries its epoch tag: (bx, by, stored_epoch).
            Weight = (1.0 / decay) ** (current_epoch - stored_epoch)
            
            This creates a FIFO-like priority where OLDER examples get significantly 
            higher sampling weights because the model saw them longest ago and is 
            most prone to catastrophically forgetting them.
            """
            count = engine_state["count"]
            if count == 0:
                return None, None
            n = min(n, count)
            # Build per-slot recency weights
            weights = np.zeros(count, dtype=np.float64)
            for i in range(count):
                item = replay_storage[i]
                if item is not None:
                    stored_epoch = item[2] if len(item) > 2 else 0
                    # ✅ PRIORITIZE OLDER EXAMPLES: Inverse of decay
                    age = max(0, current_epoch - stored_epoch)
                    weights[i] = (1.0 / decay) ** age
            w_sum = weights.sum()
            if w_sum == 0:
                weights = np.ones(count) / count  # Fallback: uniform
            else:
                weights /= w_sum
            indices = np.random.choice(count, n, replace=False, p=weights)
            rx = np.array([replay_storage[i][0] for i in indices], dtype=np.float32)
            ry = np.array([replay_storage[i][1] for i in indices], dtype=np.float32)
            return rx, ry

        # ── 5. Setup tf.data.Dataset ─────────────────────────────────────
        # ✅ FIX: flow() now yields dicts {'x': ..., 'y': ..., 'adv_target_*': ...}
        # We extract 'x' and 'y' keys; 'y' is already the resolved target for this run.
        def generator_fn():
            if hasattr(train_data, 'flow'):
                for batch in train_data.flow():
                    # ✅ FIX: LazyLoader.flow() returns dict with x, y, and multiple target_* keys
                    # Extract x and y from the dict batch
                    if isinstance(batch, dict):
                        batch_x = batch.get('x')
                        batch_y = batch.get('y')
                        if batch_x is not None and batch_y is not None:
                            yield batch_x, batch_y
                        else:
                            self.logger.error(f"⚠️ [generator_fn] Dict batch missing x or y: keys={list(batch.keys())}")
                            continue
                    else:
                        # Legacy tuple format (x, y)
                        try:
                            bx, by = batch
                            yield bx, by
                        except (TypeError, ValueError) as e:
                            self.logger.error(f"⚠️ [generator_fn] Failed to unpack batch: {e}, batch type: {type(batch)}")
                            continue
            else:
                for item in train_data:
                    if isinstance(item, dict):
                        batch_x = item.get('x')
                        batch_y = item.get('y')
                        if batch_x is not None and batch_y is not None:
                            yield batch_x, batch_y
                    else:
                        try:
                            bx, by = item
                            yield bx, by
                        except (TypeError, ValueError):
                            continue

        # ✅ FIX: Derive output_signature from the actual generator output — NOT model.output_shape.
        # model.output_shape reflects the Dense layer dimension (e.g. (None, 12)) which does NOT
        # match the stored label shape. Signal_* targets are scalars: y.shape=(N,).
        # Sample the first batch to get the true y shape. Fall back to None (TF auto-infers).
        output_signature = None
        if hasattr(train_data, 'flow'):
            try:
                temp_flow = train_data.flow()
                first_batch = next(temp_flow)
                sx = first_batch['x'] if isinstance(first_batch, dict) else first_batch[0]
                sy = first_batch['y'] if isinstance(first_batch, dict) else first_batch[1]
                x_spec = tf.TensorSpec(shape=(None,) + tuple(sx.shape[1:]), dtype=tf.float32)
                if isinstance(sy, dict):
                    y_spec = {
                        k: tf.TensorSpec(shape=(None,) + tuple(v.shape[1:]), dtype=tf.float32)
                        for k, v in sy.items()
                    }
                else:
                    y_spec = tf.TensorSpec(shape=(None,) + tuple(sy.shape[1:]), dtype=tf.float32)
                output_signature = (x_spec, y_spec)
                self.logger.info(
                    f"[output_signature] x={x_spec.shape}, "
                    f"y={'dict' if isinstance(sy, dict) else str(y_spec.shape)} "
                    f"(sampled from generator — model.output_shape not used)"
                )
            except Exception as fallback_err:
                self.logger.warning(
                    f"[output_signature] Sampling failed: {fallback_err} — "
                    f"passing None, TF will infer from first element"
                )
                output_signature = None

        dataset = tf.data.Dataset.from_generator(
            generator_fn, 
            output_signature=output_signature
        ).prefetch(tf.data.AUTOTUNE)

        # ── 6. Main Training Loop ─────────────────────────────────────────
        epoch_loss = epoch_mae = epoch_mse = 0.0
        core_epoch_loss = 0.0  # NEW: accumulate core-only loss (sample-weighted)
        epoch_samples = batches_ran = 0
        samples_by_batch = []
        bl = 0.0
        # ✅ FIX #9: Pre-initialize jury vars so final _report() is never undefined
        jury_loss_before = None
        jury_loss_after = None
        
        # ✅ FIX #10: Pre-calculate tensor constant for micro-steps to avoid creation overhead in loop
        m_limit_tensor = tf.constant(max_m, dtype=tf.int32)
        
        # ✅ ROLLBACK COOLDOWN - Proportional to epoch size so it doesn't strangle
        # learning on large datasets. ~20% of the epoch, between 20 and 300 batches.
        rollback_cooldown = 0
        COOLDOWN_DURATION = max(20, min(300, num_batches // 5))  # ~20% of epoch
        rollback_count = 0  # Track total rollbacks for statistics
        
        # ✅ PROACTIVE HARD REPLAY - Fire every REPLAY_INTERVAL batches regardless
        # of forgetting detection. Ensures rare regimes (transitions, ranging markets,
        # early downtrends) are consistently re-exposed to the model.
        REPLAY_INTERVAL = max(50, num_batches // 7)  # ~every 14% of the epoch
        self.logger.info(
            f"⚙️ [EPOCH {epoch+1}] Batch config: "
            f"num_batches={num_batches}, COOLDOWN={COOLDOWN_DURATION}, "
            f"REPLAY_INTERVAL={REPLAY_INTERVAL}"
        )
        
        forget_count = 0
        for batch_idx, (batch_x, batch_y) in enumerate(dataset):
            # 1. Yield to event loop to keep heartbeats/reports alive
            await asyncio.sleep(0)

            # 2. Break if we've reached the expected number of batches for this epoch
            if num_batches > 0 and batch_idx >= num_batches:
                break
                
            # ✅ DATA INTEGRITY CHECK: Reject NaN batches completely before they touch the model
            if tf.math.reduce_any(tf.math.is_nan(batch_x)) or tf.math.reduce_any(tf.math.is_nan(batch_y)):
                self.logger.warning(f"⚠️ [EPOCH {epoch+1}] Batch {batch_idx+1} contains NaN values in x or y! Skipping batch.")
                continue

            # ✅ COOLDOWN CHECK - Skip jury if in cooldown period
            jury_loss_before = None
            jury_loss_after = None
            weights_before_batch = None
            
            if rollback_cooldown > 0:
                rollback_cooldown -= 1
                if batch_idx % 50 == 0:
                    self.logger.debug(f"⏸️ [BATCH {batch_idx+1}] Jury cooldown: {rollback_cooldown} batches remaining")
                # Skip jury check during cooldown
            elif has_jury:
                # Run jury FIRST, only snapshot weights if jury is active.
                # Avoids copying all model weights on every batch when jury returns None.
                jury_loss_before = await run_jury(subset=True)
                if jury_loss_before is not None:
                    weights_before_batch = model.get_weights()

            # 2. Execute SOS step in thread to ensure frontend remains responsive
            # I know this looks redundant (should be res=(compiled_sos_step, batch_x, batch_y, m_limit_tensor) )but it is not
            # It is a way to decouple the CPU bound task of training from the IO bound task of keeping the frontend alive
            # So not a Bug and if there is a better  way please leme know. 
            res = await asyncio.to_thread(compiled_sos_step, batch_x, batch_y, m_limit_tensor)
            bl, bm, bs = [float(x) for x in res]
            
            batch_samples = int(batch_x.shape[0]) if hasattr(batch_x, 'shape') else len(batch_x) 
            do_rollback = False

            memory_loss_before = None
            if engine_state["count"] >= MAX_REPLAY_SAMPLES // 5:
                memory_loss_before = await run_memory_check()
            
            # FREE EXPLORATION: Epochs 1 & 2 (0-indexed: 0 & 1) run without any jury
            # restrictions, just like Keras .fit() would do greedily. This gives the model
            # time to find the loss basin before we start tightening safety.
            in_free_exploration = epoch < 2  # True for epoch 1 and 2 only
            
            if not in_free_exploration and rollback_cooldown == 0 and has_jury and jury_loss_before is not None and np.isfinite(bl) and bl > 0:
                jury_result_after = await run_jury(subset=True)
                jury_loss_after = _jury_full(jury_result_after)
                jury_core_after = _jury_core(jury_result_after)
                jury_core_before = _jury_core(jury_loss_before)
                if jury_core_after is not None and jury_core_before is not None:
                    regression = (jury_core_after / jury_core_before) - 1.0
                    
                    # ✅ DYNAMIC JURY THRESHOLD - Adapts to training phase (starts at epoch 3)
                    epoch_progress = epoch / total_epochs
                    
                    # Phase-aware threshold (Funnel Strategy — active from epoch 3 onwards)
                    if epoch_progress < 0.2:
                        # EARLY REFINEMENT: Relaxed but no longer free
                        jury_threshold = 1.12  # 12% tolerance
                    elif epoch_progress < 0.5:
                        # REFINEMENT PHASE: Tighten gradually
                        jury_threshold = 1.06  # 6% tolerance
                    elif epoch_progress < 0.8:
                        # CONVERGENCE PHASE: Strict but fair
                        jury_threshold = 1.03  # 3% tolerance
                    else:
                        # FINAL PHASE: Maximum precision
                        jury_threshold = 1.02  # 2% tolerance
                    
                    if regression > (jury_threshold - 1.0):  # Convert threshold to regression percentage
                        model.set_weights(weights_before_batch)
                        rollback_count += 1
                        rollback_cooldown = COOLDOWN_DURATION  # ✅ Activate cooldown

                        # Only log if significant (reduce log spam)
                        if batch_idx % 10 == 0 or regression > 0.20:
                            self.logger.warning(
                                f"⚖️ [BATCH {batch_idx+1}] Core-loss Rollback + Cooldown: "
                                f"core={jury_core_before:.6f} → {jury_core_after:.6f} "
                                f"full={_jury_full(jury_loss_before):.6f} → {jury_loss_after:.6f} "
                                f"(+{regression*100:.1f}% > {(jury_threshold-1)*100:.0f}% threshold, phase={epoch_progress:.1%}). "
                                f"Skipping jury for next {COOLDOWN_DURATION} batches."
                            )
                        do_rollback = True
            elif in_free_exploration and batch_idx == 0:
                self.logger.info(f"🆓 [EPOCH {epoch+1}] FREE EXPLORATION mode — jury disabled for this epoch ")
            
            # Memory check to detect and fix catastrophic forgetting
            # I have observed that models end to trigger FORGETTING when you have outliers in your data
            # So if you see this log much go back to data processing and inspect your data properly
            if not do_rollback and memory_loss_before is not None:
                memory_loss_after = await run_memory_check()
                if memory_loss_after is not None:
                    forgetting = (memory_loss_after / memory_loss_before) - 1.0
                    if forgetting > 0.30: # 30% Forgetting Threshold (Loosened per experimental observation)
                        forget_count +=1
                        self.logger.info(f"🧠 [MEMORY] Forgetting detected (+{forgetting*100:.1f}%) — Injecting reactive replay.")
                        n_r = min(batch_size, engine_state["count"])
                        # Use decay-weighted sampling: recent hard examples preferred
                        rx_np, ry_np = _sample_hard_weighted(n_r, epoch)
                        if rx_np is not None:
                            await asyncio.to_thread(compiled_sos_step, tf.constant(rx_np, dtype=tf.float32),
                                                tf.constant(ry_np, dtype=tf.float32), tf.constant(2, dtype=tf.int32))

                        if (forget_count / max(1, batch_idx + 1)) > 0.1:
                            self.logger.warning(f'Forget Count: {forget_count}')
                            self.logger.warning(f'Your Model is prone to forgetting while in training! Consider Data Audit (noisy data) or Architecture (fewer layers/neurons can cause forgetting).')
            
            # PROACTIVE HARD REPLAY: Periodically re-expose model to rare regimes
            # Uses decay-weighted sampling — recent batches bias the replay,
            # but older hard examples (rare regimes) still get a proportional chance.
            if (not do_rollback and
                engine_state["count"] >= MAX_REPLAY_SAMPLES // 5 and
                batch_idx > 0 and batch_idx % REPLAY_INTERVAL == 0):
                n_r = min(batch_size, engine_state["count"])
                rx_np, ry_np = _sample_hard_weighted(n_r, epoch)
                if rx_np is not None:
                    # Gentle 1 micro-step — remind, don't overwrite
                    await asyncio.to_thread(
                        compiled_sos_step,
                        tf.constant(rx_np, dtype=tf.float32),
                        tf.constant(ry_np, dtype=tf.float32),
                        tf.constant(1, dtype=tf.int32)
                    )
                    if batch_idx % (REPLAY_INTERVAL * 3) == 0:
                        self.logger.info(
                            f"🔁 [HARD REPLAY] Proactive injection at batch {batch_idx+1}: "
                            f"{n_r} decay-weighted samples (buf={engine_state['count']}, epoch={epoch+1})"
                        )

            if not do_rollback and np.isfinite(bl):
                epoch_loss += bl * batch_samples
                epoch_mae += bm * batch_samples
                epoch_mse += bs * batch_samples
                epoch_samples += batch_samples
                batches_ran += 1
                samples_by_batch.append(batch_samples)

                # NEW: accumulate core loss for this batch
                if _is_multi_output and isinstance(batch_y, dict):
                    try:
                        _pred_batch = model(batch_x, training=False)
                        _pred_np = {k: v.numpy() if hasattr(v, 'numpy') else v for k, v in _pred_batch.items()}
                        _yt_np = {k: (v.numpy() if hasattr(v, 'numpy') else np.array(v)) for k, v in batch_y.items()}
                        _core_bl, _, _ = split_output_loss(
                            model, _yt_np, _pred_np,
                            loss_spec=_loss_spec, core_keys=V8_3_CORE_OUTPUT_KEYS, import_tf=tf
                        )
                        core_epoch_loss += _core_bl * batch_samples
                    except Exception:
                        core_epoch_loss += bl * batch_samples  # fallback to full loss
                else:
                    core_epoch_loss += bl * batch_samples  # single-output: core == full

                # ====================== DUAL BUFFER STORAGE ======================
                # Two separate signals, two separate buffers:
                #   replay_storage        → HARD EXAMPLES (model struggled → force re-learning)
                #   good_examples_storage → GOOD EXAMPLES (model did well → detect forgetting)
                current_avg_loss = epoch_loss / epoch_samples if epoch_samples > 0 else bl
                difficulty_ratio = bl / (current_avg_loss + 1e-8)

                # ── HARD EXAMPLES: Store batches the model finds difficult ─────────────
                add_to_hard = False
                if engine_state["count"] < MAX_REPLAY_SAMPLES // 5: add_to_hard = True  # Fast fill
                elif difficulty_ratio > 1.25: add_to_hard = True                         # Significant struggle (rare regime)
                elif difficulty_ratio > 1.10 and np.random.rand() < 0.4: add_to_hard = True
                elif np.random.rand() < 0.05: add_to_hard = True                         # Small variety sample

                if add_to_hard:
                    num_to_add = min(6, len(batch_x))
                    indices = np.random.choice(len(batch_x), num_to_add, replace=False)
                    for i in indices:
                        try:
                            bx_val = batch_x[i].numpy().astype(np.float16) if hasattr(batch_x[i], 'numpy') else np.array(batch_x[i], dtype=np.float16)
                            by_val = batch_y[i].numpy().astype(np.float16) if hasattr(batch_y[i], 'numpy') else np.array(batch_y[i], dtype=np.float16)
                            # ✅ EPOCH TAG: store (x, y, epoch) so decay weighting works cross-epoch
                            replay_storage[engine_state["ptr"]] = (bx_val, by_val, epoch)
                            engine_state["ptr"] = (engine_state["ptr"] + 1) % MAX_REPLAY_SAMPLES
                            if engine_state["count"] < MAX_REPLAY_SAMPLES:
                                engine_state["count"] += 1
                        except Exception: pass

                # ── GOOD EXAMPLES: Store batches the model handled well ───────────────
                # These are used ONLY by run_memory_check() to detect forgetting.
                # Forgetting = model was once good at these, but now fails on them.
                if difficulty_ratio < 0.80 and np.random.rand() < 0.20:  # Low loss, sample 20%
                    good_count = engine_state.get("good_count", 0)
                    good_ptr = engine_state.get("good_ptr", 0)
                    num_to_add = min(3, len(batch_x))
                    indices = np.random.choice(len(batch_x), num_to_add, replace=False)
                    for i in indices:
                        try:
                            bx_val = batch_x[i].numpy().astype(np.float16) if hasattr(batch_x[i], 'numpy') else np.array(batch_x[i], dtype=np.float16)
                            by_val = batch_y[i].numpy().astype(np.float16) if hasattr(batch_y[i], 'numpy') else np.array(batch_y[i], dtype=np.float16)
                            good_examples_storage[good_ptr] = (bx_val, by_val)
                            good_ptr = (good_ptr + 1) % MAX_GOOD_SAMPLES
                            if good_count < MAX_GOOD_SAMPLES:
                                good_count += 1
                        except Exception: pass
                    engine_state["good_ptr"] = good_ptr
                    engine_state["good_count"] = good_count

                if difficulty_ratio > 1.20 and batch_idx % 25 == 0:
                    self.logger.info(
                        f"🧠 [BUFFERS] hard={engine_state['count']}/{MAX_REPLAY_SAMPLES} "
                        f"| good={engine_state.get('good_count',0)}/{MAX_GOOD_SAMPLES} "
                        f"| diff={difficulty_ratio:.2f}x"
                    )

                if batch_idx % 5 == 0 or batch_idx == num_batches - 1:
                    await _report(
                        batch_idx, bl, epoch_loss, epoch_mae, epoch_mse, epoch_samples,
                        jury_loss_before=jury_loss_before,
                        jury_loss_after=jury_loss_after
                    )
            else:
                # Still report progress during rollback/error
                if batch_idx % 5 == 0:
                    await _report(
                        batch_idx, bl, epoch_loss, epoch_mae, epoch_mse, epoch_samples,
                        jury_loss_before=jury_loss_before,
                        jury_loss_after=jury_loss_after
                    )

        # Diagnostics
        rollback_rate = (rollback_count / num_batches * 100) if num_batches > 0 else 0.0
        self.logger.info(
            f"📊 [EPOCH {epoch+1}] SUMMARY:\n"
            f"   ├─ Batches: {batches_ran}/{num_batches}\n"
            f"   ├─ Samples: {epoch_samples}\n"
            f"   ├─ Rollbacks: {rollback_count} ({rollback_rate:.1f}%)\n"
            f"   ├─ Replay Buffer: {engine_state['count']}\n"
            f"   └─ Partial Batches: {sum(1 for s in samples_by_batch if s < batch_size)}"
        )

        # ── 7. Validation Phase (Hardened Sample-Weighted Loop) ──────────
        val_loss = float('inf')   # inf = meaningful "no data" signal for upstream callers
        val_mae = val_mse = 0.0   # 0.0 avoids inf propagation into history/reporter
        if val_data is not None:
            v_loss_sum = v_mae_sum = v_mse_sum = 0.0
            v_core_loss_sum = 0.0                    # Bug 1 fix: accumulate core loss across full val set
            v_per_head_loss_sums: Optional[Dict[str, float]] = None  # dict[head_name -> weighted_sum]
            v_samples_total = 0
            
            # ✅ FIX: Auto-detect if validation data is a generator (don't rely on is_generator flag)
            is_val_generator = hasattr(val_data, 'flow') and callable(getattr(val_data, 'flow'))
            
            self.logger.info(
                f"🔍 [VAL {epoch+1}] Validation mode: {'Generator' if is_val_generator else 'Array'}\n"
                f"   ├─ val_data type: {type(val_data).__name__}\n"
                f"   ├─ val_targets type: {type(val_targets).__name__ if val_targets is not None else 'None'}\n"
                f"   ├─ val_targets value: {val_targets if isinstance(val_targets, str) else '(array)'}\n"
                f"   └─ has flow(): {hasattr(val_data, 'flow')}"
            )
            
            if is_val_generator:
                # ✅ FIX: val_data.flow() yields dicts {'x':..., 'y':...} — extract keys
                val_flow = val_data.flow()
                for v_idx in range(len(val_data)):
                    try:
                        v_batch = next(val_flow)
                        v_x = v_batch['x'] if isinstance(v_batch, dict) else v_batch[0]
                        v_y = v_batch['y'] if isinstance(v_batch, dict) else v_batch[1]
                        
                        # 📊 Log first batch for diagnostics
                        if v_idx == 0:
                            self.logger.info(
                                f"📊 [VAL] First batch from generator:\n"
                                f"   ├─ v_x shape: {v_x.shape}\n"
                                f"   ├─ v_y shape: {v_y.shape}\n"
                                f"   ├─ v_x range: [{np.min(v_x):.4f}, {np.max(v_x):.4f}]\n"
                                f"   └─ v_y range: [{np.min(v_y):.4f}, {np.max(v_y):.4f}]"
                            )
                        
                        # ✅ UPGRADE: Graph-mode validation step with timeout guard
                        v_res = await asyncio.wait_for(
                            asyncio.to_thread(compiled_val_step, tf.constant(v_x, dtype=tf.float32), tf.constant(v_y, dtype=tf.float32)),
                            timeout=120.0
                        )
                        vl, vmae, vmse = [float(val) for val in v_res]
                        
                        # 📊 Log first batch metrics
                        if v_idx == 0:
                            self.logger.info(
                                f"📊 [VAL] First batch metrics:\n"
                                f"   ├─ Batch loss: {vl:.6f}\n"
                                f"   ├─ Batch MAE: {vmae:.6f}\n"
                                f"   └─ Batch MSE: {vmse:.6f}"
                            )
                        
                        if np.isfinite(vl):
                            sl = len(v_x)
                            v_loss_sum += vl * sl
                            v_samples_total += sl
                            v_mae_sum += vmae * sl
                            v_mse_sum += vmse * sl
                            # Bug 1 fix: accumulate core val loss in the loop (not after)
                            if _is_multi_output and isinstance(v_y, dict):
                                try:
                                    _v_pred = model(tf.constant(v_x, dtype=tf.float32), training=False)
                                    _v_pred_np = {k: val.numpy() if hasattr(val, 'numpy') else val for k, val in _v_pred.items()}
                                    _v_y_np = {k: (val.numpy() if hasattr(val, 'numpy') else np.array(val)) for k, val in v_y.items()}
                                    _batch_core, _, _batch_per_head = split_output_loss(
                                        model, _v_y_np, _v_pred_np,
                                        loss_spec=_loss_spec, core_keys=V8_3_CORE_OUTPUT_KEYS, import_tf=tf
                                    )
                                    v_core_loss_sum += _batch_core * sl
                                    if _batch_per_head is not None:
                                        if v_per_head_loss_sums is None:
                                            v_per_head_loss_sums = {k: 0.0 for k in _batch_per_head}
                                        for k, hv in _batch_per_head.items():
                                            v_per_head_loss_sums[k] = v_per_head_loss_sums.get(k, 0.0) + hv * sl
                                except Exception:
                                    v_core_loss_sum += vl * sl  # fallback to full loss
                            else:
                                v_core_loss_sum += vl * sl  # single-output: core == full
                        else:
                            self.logger.warning(
                                f"⚠️ [VAL] Batch {v_idx} non-finite loss: {vl}\n"
                                f"   ├─ x_bounds: [{np.min(v_x):.4f}, {np.max(v_x):.4f}]\n"
                                f"   └─ Skipping batch."
                            )
                    except StopIteration: break
                    except Exception as eval_err:
                        self.logger.warning(f"⚠️ Val batch {v_idx} error: {eval_err}")
                        continue
            else:
                v_x_all = val_data.values if hasattr(val_data, 'values') else val_data
                v_y_all = val_targets
                
                if v_y_all == "LAZY_ON_DISK":
                    self.logger.error("❌ [VAL] val_targets='LAZY_ON_DISK' reached array path — skipping validation.")
                    v_batches = 0
                elif v_y_all is None:
                    self.logger.error("❌ [VAL] val_targets is None but model is not marked as Autoencoder. Skipping validation.")
                    v_batches = 0
                else:
                    v_batches = max(1, (len(v_x_all) + batch_size - 1) // batch_size)

                for v_idx in range(v_batches):
                    try:
                        s, e = v_idx * batch_size, min((v_idx + 1) * batch_size, len(v_x_all))
                        # ✅ UPGRADE: Graph-mode validation step with timeout guard
                        v_res = await asyncio.wait_for(
                            asyncio.to_thread(compiled_val_step, tf.constant(v_x_all[s:e], dtype=tf.float32), tf.constant(v_y_all[s:e], dtype=tf.float32)),
                            timeout=120.0
                        )
                        vl, vmae, vmse = [float(val) for val in v_res]

                        if np.isfinite(vl):
                            sl = e - s
                            v_loss_sum += vl * sl
                            v_samples_total += sl
                            v_mae_sum += vmae * sl
                            v_mse_sum += vmse * sl
                            # Bug 1 fix: accumulate core val loss (array path)
                            _v_y_batch = v_y_all[s:e]
                            if _is_multi_output and isinstance(_v_y_batch, dict):
                                try:
                                    _v_pred = model(tf.constant(v_x_all[s:e], dtype=tf.float32), training=False)
                                    _v_pred_np = {k: val.numpy() if hasattr(val, 'numpy') else val for k, val in _v_pred.items()}
                                    _v_y_np = {k: (val.numpy() if hasattr(val, 'numpy') else np.array(val)) for k, val in _v_y_batch.items()}
                                    _batch_core, _, _batch_per_head = split_output_loss(
                                        model, _v_y_np, _v_pred_np,
                                        loss_spec=_loss_spec, core_keys=V8_3_CORE_OUTPUT_KEYS, import_tf=tf
                                    )
                                    v_core_loss_sum += _batch_core * sl
                                    if _batch_per_head is not None:
                                        if v_per_head_loss_sums is None:
                                            v_per_head_loss_sums = {k: 0.0 for k in _batch_per_head}
                                        for k, hv in _batch_per_head.items():
                                            v_per_head_loss_sums[k] = v_per_head_loss_sums.get(k, 0.0) + hv * sl
                                except Exception:
                                    v_core_loss_sum += vl * sl
                            else:
                                v_core_loss_sum += vl * sl  # single-output: core == full
                        else:
                            self.logger.warning(f"⚠️ [VAL] Static batch {v_idx} non-finite loss: {vl}")
                    except Exception as v_err:
                        self.logger.warning(f"⚠️ [VAL] Static batch {v_idx} failed: {v_err}")
                        continue

            if v_samples_total > 0:
                val_loss = v_loss_sum / v_samples_total
                val_mae  = v_mae_sum / v_samples_total
                val_mse  = v_mse_sum / v_samples_total
                # Bug 1 fix: derive core/per-head from full-loop accumulators
                core_val_loss = v_core_loss_sum / v_samples_total
                per_head_val_loss = (
                    {k: s / v_samples_total for k, s in v_per_head_loss_sums.items()}
                    if v_per_head_loss_sums is not None else None
                )
            else:
                core_val_loss = float('inf')
                per_head_val_loss = None
            
            # Finalize: Reset model metrics after validation phase
            model.reset_metrics()
            
            self.logger.info(
                f"🧪 [VAL {epoch+1}] RESULTS:\n"
                f"   ├─ Validation loss (full): {val_loss:.6f}\n"
                f"   ├─ Validation loss (core): {core_val_loss:.6f}\n"
                f"   ├─ Validation MAE: {val_mae:.6f}\n"
                f"   ├─ Validation MSE: {val_mse:.6f}\n"
                f"   └─ Samples processed: {v_samples_total}"
            )

        # ── 8. Epoch-Level Jury Audit (Final Safety Net) ─────────────────
        epoch_jury_rejected = False  # upstream regression signal, separate from val_loss
        # ✅ FIX: Respect free exploration — skip epoch jury for epochs 1 & 2.
        if epoch >= 2 and weights_before is not None and has_jury and np.isfinite(val_loss):
            try:
                jx, jy = train_data_obj.jury_x, train_data_obj.jury_y
                if jx is not None and len(jx) > 0:
                    post_epoch_weights = model.get_weights()      # 1. Capture training result

                    model.set_weights(weights_before)             # 2. Measure baseline
                    j_result_before = await run_epoch_jury()
                    j_before = _jury_core(j_result_before)        # Use CORE loss for epoch decision

                    model.set_weights(post_epoch_weights)         # 3. Measure improvement
                    j_result_after = await run_epoch_jury()
                    j_after = _jury_core(j_result_after)          # Use CORE loss for epoch decision

                    # ✅ DYNAMIC EPOCH JURY THRESHOLD (Funnel Strategy)
                    # Mirrors the batch-level logic for consistency. Early epochs allow 
                    # more movement; later epochs require strict stability.
                    epoch_progress = epoch / total_epochs
                    if epoch_progress < 0.2:
                        e_jury_threshold = 1.12  # 12% tolerance (Early Refinement)
                    elif epoch_progress < 0.5:
                        e_jury_threshold = 1.06  # 6% tolerance (Refinement)
                    elif epoch_progress < 0.8:
                        e_jury_threshold = 1.03  # 3% tolerance (Convergence)
                    else:
                        e_jury_threshold = 1.02  # 2% tolerance (Final Precision)
                    
                    # ✅ CATCH-UP OVERRIDE: If the trainer is in Catch-Up Mode (dyn_threshold=2.0),
                    # respect that at the epoch level too.
                    e_jury_threshold = max(e_jury_threshold, dyn_threshold)
                    
                    # 🚀 VALIDATION OVERRULE: If the main validation set shows clear improvement,
                    # we should be much more suspicious of the tiny Jury Pool signal. 
                    # If val_loss dropped by >5%, we loosen the jury threshold by 5x.
                    prev_val = (last_val_metrics or {}).get("val_loss")
                    if prev_val and prev_val > 0 and val_loss < prev_val * 0.95:
                        self.logger.info(f"🚀 [VAL OVERRULE] Clear improvement in main validation ({prev_val:.4f} → {val_loss:.4f}). Relaxing Jury safety.")
                        e_jury_threshold = e_jury_threshold * 5.0 # Massive relaxation for massive gains

                    if j_before > 0 and j_after > j_before * e_jury_threshold:
                        pct = (j_after / j_before - 1) * 100
                        self.logger.warning(
                            f"⚖️ [EPOCH {epoch+1}] JURY REJECTED: {j_before:.6f} → {j_after:.6f} (+{pct:.1f}% > {(e_jury_threshold-1)*100:.0f}% threshold)\n"
                            f"   └─ val_loss kept as computed ({val_loss:.6f}), regression flagged upstream."
                        )
                        epoch_jury_rejected = True  # ✅ flag only — do NOT overwrite val_loss
            except Exception as e: self.logger.debug(f"⚖️ Epoch jury failed: {e}")

        # ── 9. Final Report (Fresh Metrics) ─────────────────────────────
        # Bug 1 fix: core_val_loss and per_head_val_loss are now accumulated across the
        # full validation set (section 7 loop above) — NOT from a single bolt-on batch.
        # The variables core_val_loss / per_head_val_loss were set in section 7.
        # For single-output models or when val_data is None, fall back to full val_loss.
        if val_data is None:
            core_val_loss = float(val_loss)  # val_loss is already float('inf') sentinel
            per_head_val_loss = None
        # else: core_val_loss and per_head_val_loss are already set by the section 7 loop

        res = {
            "loss": float(epoch_loss / epoch_samples) if epoch_samples > 0 else float('inf'),
            "core_loss": float(core_epoch_loss / epoch_samples) if epoch_samples > 0 else float('inf'),
            "val_loss": float(val_loss),
            "core_val_loss": float(core_val_loss),   # full-val-set average — drives checkpoint selection
            "per_head_val_loss": per_head_val_loss,  # full-val-set per-head averages or None
            "mae": float(epoch_mae / epoch_samples) if epoch_samples > 0 else 0.0,
            "val_mae": float(val_mae),
            "mse": float(epoch_mse / epoch_samples) if epoch_samples > 0 else 0.0,
            "val_mse": float(val_mse),
            "jury_rejected": epoch_jury_rejected,  # driven by core-loss internally
        }

        # Send one final report for the epoch with high-precision validation results
        last_reported_loss = bl if np.isfinite(bl) else 0.0
        await _report(
            num_batches - 1,
            last_reported_loss,
            epoch_loss,
            epoch_mae,
            epoch_mse,
            epoch_samples,
            jury_loss_before=jury_loss_before,
            jury_loss_after=jury_loss_after,
            fresh_val=res,
            core_epoch_loss=core_epoch_loss,
            per_head_val_loss=per_head_val_loss,
        )

        return res



    # _trainer_fit_v1 deleted — confirmed dead code (grep: only referenced in
    # analysis_manager_old.py, never called from execute_model_training or any live route).
    # The pre-patch single-output version was left behind after the V8.3 multi-output patch.
    # Keeping it would risk accidental re-wiring; the current _trainer_fit supersedes it.

    async def _generate_post_training_predictions(
        self,
        model,
        train_data,
        num_batches,
        batch_size,
        epoch,
        task_id,
        reporter,
        total_epochs,
        last_val_metrics=None,
        target_column=None,
        max_m=5,
        dyn_threshold=1.15,
        # ── ports from _train_model_async ──────────────────────────
        val_data=None,
        train_targets=None,
        val_targets=None,
        is_generator=False,
        weights_before=None,
        train_data_obj=None,
):
        """
        Clean Keras .fit() Clone with Sample-Weighted Metrics + Jury SOS Safety.
        Optimized for speed + stability. Ready for benchmarking.
        """
       
        

        if num_batches == 0:
            self.logger.error(f"❌ [EPOCH {epoch+1}] num_batches=0 — skipping epoch")
            return {"loss": float('inf'), "val_loss": float('inf'),
                    "mae": 0.0, "val_mae": 0.0, "mse": 0.0, "val_mse": 0.0}

        # ── 1. Reset Stateful Metrics ────────────────────────
        for metric in model.metrics:
            metric.reset_state()

        # ── 2. Progress Reporter ─────────────────────────────
        async def _report(b_idx, b_loss, epoch_loss, epoch_mae, epoch_mse, 
                        epoch_samples, fresh_val=None, jury_info=None):
            avg_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0.0
            avg_mae = epoch_mae / epoch_samples if epoch_samples > 0 else 0.0
            avg_mse = epoch_mse / epoch_samples if epoch_samples > 0 else 0.0

            gp = min(100, max(0, int(round(((epoch * num_batches + b_idx + 1) / 
                                        (total_epochs * num_batches)) * 100))))

            msg = f"Epoch {epoch+1}/{total_epochs}: batch {b_idx+1}/{num_batches} - Loss: {b_loss:.4f} | Avg: {avg_loss:.4f}"
            if jury_info:
                msg += f" | Jury: {jury_info}"

            v_src = fresh_val or (last_val_metrics or {})
            tm = {
                "loss": avg_loss, "mae": avg_mae, "mse": avg_mse,
                "val_loss": v_src.get("val_loss", 0.0),
                "val_mae": v_src.get("val_mae", 0.0),
                "val_mse": v_src.get("val_mse", 0.0),
                "current_epoch": epoch + 1, "total_epochs": total_epochs,
            }

            if reporter:
                await reporter.report_async(progress=gp, message=msg,
                                        trainingMetrics=tm, loss=float(b_loss), avg_loss=float(avg_loss))
            else:
                self.task_store.update_task(task_id=task_id, status="processing", 
                                        progress=gp, message=msg, metadata={"loss": float(b_loss)})

            if b_idx % 10 == 0 or b_idx >= num_batches - 1:
                self.logger.info(f"⚡ [EPOCH {epoch+1}] Batch {b_idx+1}/{num_batches} | Loss: {b_loss:.5f} | Avg: {avg_loss:.5f}")

        # ── 3. Compiled SOS Training Step ─────────────────────
        @tf.function
        def compiled_sos_step(x, y, m_limit):
            last_loss = tf.constant(1e9, dtype=tf.float32)
            for m in tf.range(m_limit):
                with tf.GradientTape() as tape:
                    y_pred = model(x, training=True)
                    loss = model.compute_loss(x, y, y_pred)

                grads = tape.gradient(loss, model.trainable_variables)
                grads_vars = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
                if grads_vars:
                    model.optimizer.apply_gradients(grads_vars)

                if m > 2:
                    if (last_loss - loss) / (last_loss + 1e-8) < 0.005:
                        break
                last_loss = loss

            y_pred_final = model(x, training=False)
            model.compute_metrics(x, y, y_pred_final)

            b_mae = tf.reduce_mean(tf.abs(y - y_pred_final))
            b_mse = tf.reduce_mean(tf.square(y - y_pred_final))
            return loss, b_mae, b_mse

        # ── 4. Jury & Helpers ─────────────────────────────────
        has_jury = (train_data_obj is not None and 
                    hasattr(train_data_obj, 'jury_x') and train_data_obj.jury_x is not None)

        replay_buffer = deque(maxlen=2000)   # Your intentional feature

        async def run_jury(subset=True):
            if not has_jury or train_data_obj.jury_x is None:
                return None
            jx, jy = train_data_obj.jury_x, train_data_obj.jury_y

            if subset and len(jx) > batch_size:
                idx = np.random.choice(len(jx), batch_size, replace=False)
                final_jx, final_jy = jx[idx], jy[idx]
            else:
                final_jx, final_jy = jx, jy

            # === INTENTIONAL REPLAY BUFFER MIXIN ===
            if replay_buffer:
                replay_list = list(replay_buffer)
                r_size = min(len(replay_list), batch_size // 2)
                if r_size > 0:
                    r_idx = np.random.choice(len(replay_list), r_size, replace=False)
                    rx = np.array([replay_list[i][0] for i in r_idx])
                    ry = np.array([replay_list[i][1] for i in r_idx])
                    final_jx = np.concatenate([final_jx, rx], axis=0)
                    final_jy = np.concatenate([final_jy, ry], axis=0)

            res = await asyncio.to_thread(model.test_on_batch, final_jx, final_jy)
            model.reset_metrics()   # Prevent pollution
            return float(res[0]) if isinstance(res, (list, tuple, np.ndarray)) else float(res)

        def extract_val_metrics(v_res):
            if isinstance(v_res, (list, tuple, np.ndarray)):
                vals = [float(x) for x in v_res]
                res_dict = {"loss": vals[0]}
                metric_names = [m.name.lower() for m in model.metrics if hasattr(m, 'name')]
                for i, name in enumerate(metric_names):
                    if i + 1 < len(vals):
                        res_dict[name] = vals[i + 1]
                # Fallbacks
                if "mae" not in res_dict and len(vals) > 1:
                    res_dict["mae"] = vals[1]
                if "mse" not in res_dict and len(vals) > 2:
                    res_dict["mse"] = vals[2]
                return res_dict
            return {"loss": float(v_res)}

        # ── 5. Dataset ───────────────────────────────────────
        # ✅ FIX: flow() yields dicts {'x': ..., 'y': ...} — extract keys
        def generator_fn():
            if hasattr(train_data, 'flow'):
                for batch in train_data.flow():
                    if isinstance(batch, dict):
                        yield batch['x'], batch['y']
                    else:
                        bx, by = batch
                        yield bx, by
            else:
                for item in train_data:
                    if isinstance(item, dict):
                        yield item['x'], item['y']
                    else:
                        bx, by = item
                        yield bx, by

        try:
            in_shape = tuple(model.input_shape[1:]) if model.input_shape else (None,)
            out_shape = tuple(model.output_shape[1:]) if hasattr(model, 'output_shape') else (None,)
            output_signature = (
                tf.TensorSpec(shape=(None,) + in_shape, dtype=tf.float32),
                tf.TensorSpec(shape=(None,) + out_shape, dtype=tf.float32)
            )
        except Exception:
            output_signature = None

        dataset = tf.data.Dataset.from_generator(
            generator_fn, output_signature=output_signature
        ).prefetch(tf.data.AUTOTUNE)

        # ── 6. Training Loop ─────────────────────────────────
        epoch_loss = epoch_mae = epoch_mse = 0.0
        epoch_samples = batches_ran = 0
        samples_by_batch = []

        for batch_idx, (batch_x, batch_y) in enumerate(dataset):
            await asyncio.sleep(0)  # Keep event loop responsive

            if num_batches > 0 and batch_idx >= num_batches:
                break

            weights_before_batch = model.get_weights() if has_jury else None
            jury_loss_before = await run_jury(subset=True) if has_jury else None

            # Training step
            bl_tensor, bm_tensor, bs_tensor = await asyncio.to_thread(
                compiled_sos_step, batch_x, batch_y, tf.constant(max_m, dtype=tf.int32)
            )
            bl, bm, bs = float(bl_tensor), float(bm_tensor), float(bs_tensor)
            batch_samples = len(batch_x)

            do_rollback = False
            jury_loss_after = None

            if has_jury and jury_loss_before is not None and np.isfinite(bl) and bl > 0:
                jury_loss_after = await run_jury(subset=True)
                if jury_loss_after is not None and jury_loss_after > jury_loss_before * 1.02:
                    model.set_weights(weights_before_batch)
                    self.logger.warning(f"⚖️ [BATCH {batch_idx+1}] Rollback: {jury_loss_before:.4f} → {jury_loss_after:.4f}")
                    do_rollback = True

            if not do_rollback and np.isfinite(bl):
                epoch_loss += bl * batch_samples
                epoch_mae += bm * batch_samples
                epoch_mse += bs * batch_samples
                epoch_samples += batch_samples
                batches_ran += 1
                samples_by_batch.append(batch_samples)

                # Add to replay buffer (your feature)
                for i in range(min(len(batch_x), 8)):
                    try:
                        replay_buffer.append((batch_x[i].numpy(), batch_y[i].numpy()))
                    except:
                        pass

            if batch_idx % 5 == 0 or batch_idx >= num_batches - 1:
                jury_info = f"{jury_loss_before:.4f}→{jury_loss_after:.4f}" if jury_loss_after else None
                await _report(batch_idx, bl, epoch_loss, epoch_mae, epoch_mse, epoch_samples, jury_info=jury_info)

        # Diagnostics
        self.logger.info(f"📊 [EPOCH {epoch+1}] SUMMARY: Batches={batches_ran}/{num_batches} | "
                        f"Samples={epoch_samples} | Partial={sum(s < batch_size for s in samples_by_batch)}")

        # ── 7. Validation Phase ───────────────────────────────
        val_loss = val_mae = val_mse = float('inf')
        if val_data is not None:
            v_loss_sum = v_mae_sum = v_mse_sum = 0.0
            v_samples_total = 0

            # Generator path
            if is_generator and hasattr(val_data, 'flow'):
                val_flow = val_data.flow()
                for v_idx in range(len(val_data)):
                    try:
                        v_batch = next(val_flow)
                        # ✅ FIX: Handle dict returns from flow() (new multi-target structure)
                        if isinstance(v_batch, dict):
                            v_x = v_batch.get('x')
                            v_y = v_batch.get('y')
                        else:
                            v_x, v_y = v_batch
                        
                        v_res = await asyncio.wait_for(
                            asyncio.to_thread(model.test_on_batch, v_x, v_y), timeout=600.0
                        )
                        vm = extract_val_metrics(v_res)
                        vl = vm.get("loss", 0.0)
                        if np.isfinite(vl):
                            sl = len(v_x)
                            v_loss_sum += vl * sl
                            v_mae_sum += vm.get("mae", 0.0) * sl
                            v_mse_sum += vm.get("mse", 0.0) * sl
                            v_samples_total += sl
                    except StopIteration:
                        break
                    except Exception as e:
                        self.logger.warning(f"⚠️ Val batch {v_idx} error: {e}")
                        continue
            else:
                # Array path
                v_x_all = getattr(val_data, 'values', val_data)
                v_y_all = val_targets if val_targets is not None else None
                if v_y_all is None:
                    self.logger.error("❌ Validation targets missing for supervised model!")
                else:
                    v_batches = max(1, (len(v_x_all) + batch_size - 1) // batch_size)
                    for v_idx in range(v_batches):
                        s = v_idx * batch_size
                        e = min(s + batch_size, len(v_x_all))
                        v_res = await asyncio.wait_for(
                            asyncio.to_thread(model.test_on_batch, v_x_all[s:e], v_y_all[s:e]), 
                            timeout=600.0
                        )
                        vm = extract_val_metrics(v_res)
                        vl = vm.get("loss", 0.0)
                        if np.isfinite(vl):
                            sl = e - s
                            v_loss_sum += vl * sl
                            v_mae_sum += vm.get("mae", 0.0) * sl
                            v_mse_sum += vm.get("mse", 0.0) * sl
                            v_samples_total += sl

            if v_samples_total > 0:
                val_loss = v_loss_sum / v_samples_total
                val_mae = v_mae_sum / v_samples_total
                val_mse = v_mse_sum / v_samples_total

        # ── 8. Epoch-Level Jury ───────────────────────────────
        epoch_jury_rejected = False
        if weights_before is not None and has_jury and np.isfinite(val_loss):
            try:
                post_weights = model.get_weights()
                model.set_weights(weights_before)
                j_before = await run_jury(subset=False)
                model.set_weights(post_weights)
                j_after = await run_jury(subset=False)

                if j_before > 0 and j_after > j_before * 1.03:
                    pct = (j_after / j_before - 1) * 100
                    self.logger.warning(f"⚖️ [EPOCH {epoch+1}] JURY REJECTED: {j_before:.5f} → {j_after:.5f} (+{pct:.1f}%)")
                    epoch_jury_rejected = True
            except Exception as e:
                self.logger.debug(f"Epoch jury failed: {e}")

        # ── 9. Final Result ───────────────────────────────────
        result = {
            "loss": float(epoch_loss / epoch_samples) if epoch_samples > 0 else float('inf'),
            "val_loss": float(val_loss),
            "mae": float(epoch_mae / epoch_samples) if epoch_samples > 0 else 0.0,
            "val_mae": float(val_mae),
            "mse": float(epoch_mse / epoch_samples) if epoch_samples > 0 else 0.0,
            "val_mse": float(val_mse),
            "jury_rejected": epoch_jury_rejected,
        }

        # Final progress update
        await _report(num_batches - 1 if num_batches > 0 else 0, 
                    bl, epoch_loss, epoch_mae, epoch_mse, epoch_samples, fresh_val=result)

        
        async def _generate_post_training_predictions(
            self,
            model,
            model_id: str,
            dataset_id: str,
            train_data: Any,
            val_data: Any,
            test_data: Any,
            task_id: str,
            reporter: Optional[ProgressReporter] = None
        ):
            """
            Generate predictions for train/validation/test splits and persist to ModelPredictions table.
            
            🔴 FIX: Store predictions WITH model_id so multiple models can have predictions on same dataset.
            This enables performance comparison across multiple trained models.
            
            Previously stored in MLDatasetChunk.predictions_data (no model_id) → overwrote previous model predictions
            Now stored in ModelPredictions with (model_id, dataset_id, chunk_index) unique constraint → preserves all predictions
            """
        

         
          
            self.logger.info(f"🔮 Generating post-training predictions for model {model_id[:8]} on dataset {dataset_id[:8]}...")
            if reporter:
                reporter.update(message="Generating post-training predictions for performance visualization...")

            splits = [
                ("train", train_data),
                ("validation", val_data),
                ("test", test_data)
            ]

            try:
                async with AsyncPostgresSessionLocal() as db:
                    for split_name, split_data in splits:
                        if split_data is None:
                            continue
                        
                        self.logger.info(f"  ├─ Processing {split_name} split...")
                        
                        # 1. Generate ALL predictions for this split
                        predictions = []
                        
                        # Use a local import to avoid circular dependencies if any
                        from app.core.ml.ml_data_loader import LazySequenceGenerator
                        
                        if not isinstance(split_data, LazySequenceGenerator):
                            # Small dataset (RAM)
                            # Expecting split_data to be np.ndarray or list
                            x_data = np.array(split_data)
                            
                            # If it's a tuple (x, y), extract x
                            if isinstance(split_data, tuple) and len(split_data) == 2:
                                x_data = split_data[0]
                                
                            # Predict in batches to avoid GPU OOM
                            batch_size = 1024
                            for i in range(0, len(x_data), batch_size):
                                batch_x = x_data[i:i+batch_size]
                                batch_pred = model.predict(batch_x, verbose=0)
                                predictions.append(batch_pred)
                            
                            if predictions:
                                all_predictions = np.concatenate(predictions, axis=0)
                            else:
                                continue
                        else:
                            # Large dataset (Lazy Generator)
                            # Use the file paths to ensure we match the DB chunks order
                            file_paths = getattr(split_data, 'file_paths', [])
                            if not file_paths:
                                self.logger.warning(f"  └─ No file paths found for lazy {split_name} split")
                                continue
                                
                            # Predict chunk by chunk
                            for path in file_paths:
                                try:
                                    data = np.load(path, mmap_mode='r', allow_pickle=True)
                                    x = data['sequences'] if 'sequences' in data else data['x']
                                    
                                    chunk_preds = []
                                    batch_size = 1024
                                    for i in range(0, len(x), batch_size):
                                        batch_x = x[i:i+batch_size]
                                        chunk_preds.append(model.predict(batch_x, verbose=0))
                                    
                                    predictions.append(np.concatenate(chunk_preds, axis=0))
                                    data.close()
                                except Exception as chunk_err:
                                    self.logger.error(f"Error predicting on chunk {path}: {chunk_err}")
                                    continue
                            
                            if predictions:
                                all_predictions = np.concatenate(predictions, axis=0)
                            else:
                                continue

                        # 2. Map predictions to chunks and persist to ModelPredictions table
                        stmt = select(MLDatasetChunk).where(
                            and_(
                                MLDatasetChunk.dataset_id == dataset_id,
                                MLDatasetChunk.split_name == split_name
                            )
                        ).order_by(MLDatasetChunk.chunk_index.asc())
                        
                        result = await db.execute(stmt)
                        chunks = result.scalars().all()
                        
                        if not chunks:
                            self.logger.warning(f"  └─ No chunks found in DB for {split_name} split")
                            continue
                            
                        # Slice all_predictions and store each chunk's predictions with model_id
                        cursor = 0
                        cctx = zstd.ZstdCompressor(level=3)
                        
                        created_count = 0
                        for chunk in chunks:
                            chunk_size = chunk.sequence_count
                            if cursor + chunk_size > len(all_predictions):
                                self.logger.warning(f"  └─ Prediction alignment mismatch: cursor={cursor}, chunk_size={chunk_size}, total={len(all_predictions)}")
                                break
                                
                            chunk_pred_slice = all_predictions[cursor:cursor+chunk_size]
                            
                            # Store as dict for future multi-target support
                            pred_dict = {"predictions": chunk_pred_slice}
                            
                            # Serialize and compress
                            serialized = pickle.dumps(pred_dict)
                            compressed = cctx.compress(serialized)
                            
                            # 🔴 FIX: Create ModelPredictions entry with model_id (not MLDatasetChunk)
                            # This preserves predictions from multiple models per dataset
                            try:
                                # Delete any existing predictions for this model/chunk combination
                                delete_stmt = await db.execute(
                                    select(ModelPredictions).where(
                                        and_(
                                            ModelPredictions.model_id == model_id,
                                            ModelPredictions.dataset_id == dataset_id,
                                            ModelPredictions.split_name == split_name,
                                            ModelPredictions.chunk_index == chunk.chunk_index
                                        )
                                    )
                                )
                                existing = delete_stmt.scalars().first()
                                if existing:
                                    await db.delete(existing)
                                
                                # Create new ModelPredictions entry
                                model_pred = ModelPredictions(
                                    model_id=model_id,
                                    dataset_id=dataset_id,
                                    split_name=split_name,
                                    chunk_index=chunk.chunk_index,
                                    sequence_count=chunk_size,
                                    predictions_data=compressed,
                                    compression_ratio=len(serialized) / len(compressed) if len(compressed) > 0 else 1.0,
                                    uncompressed_size_bytes=len(serialized),
                                    compressed_size_bytes=len(compressed),
                                    is_verified=True
                                )
                                db.add(model_pred)
                                created_count += 1
                            except Exception as pred_err:
                                self.logger.error(f"Error storing prediction for chunk {chunk.chunk_index}: {pred_err}")
                                continue
                            
                            cursor += chunk_size
                        
                        await db.commit()
                        self.logger.info(f"  └─ Created ModelPredictions entries for {created_count} chunks in {split_name} split (model={model_id[:8]})")

                self.logger.info(f"✅ Post-training predictions complete for model {model_id[:8]} on dataset {dataset_id[:8]}")
            except Exception as e:
                self.logger.error(f"❌ Failed to generate post-training predictions: {e}", exc_info=True)
