from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import asyncio
import numpy as np
import pandas as pd
from app.core.processing.processing_manager import ProcessingManager, AnalysisType, IntermediateResultsCache
from app.core.config import TechnicalConfig, SNRConfig, AstronomicalConfig, MLDatasetConfig, ModelBuildConfig, ModelTrainingConfig
from app.core.processing.progress_reporter import ProgressReporter, ThrottlingStrategy
from app.api.routes.data.database import AsyncPostgresSessionLocal
from app.core.data.session_data_loader import set_as_current_data, store_session_step_result
from app.core.services.decompress_cache import get_cache as get_decompress_cache

logger = logging.getLogger(__name__)


def _merge_footprint_into_df(
    df: pd.DataFrame,
    session_id: str,
) -> pd.DataFrame:
    """
    Load cached footprint parquet for this session's symbol/timeframe and
    left-join the raw fp_* columns onto df using the DatetimeIndex.

    This must run BEFORE TechnicalIndicators.calculate_all_indicators() so
    that _calculate_footprint_features() finds fp_poc / fp_delta / etc.

    Returns df unchanged if:
    - no parquet cache exists for symbol/timeframe
    - any load/merge error occurs (non-fatal, logged as warning)
    """
    try:
        from pathlib import Path
        import sqlalchemy as _sa
        from sqlalchemy import create_engine as _create_engine
        from app.core.config import settings as _settings

        # ── Resolve symbol & timeframe from DataSession (sync read) ─────────
        sync_url = (
            str(_settings.DATABASE_URL)
            .replace("+asyncpg", "")
            .replace("postgresql+asyncpg", "postgresql")
        )
        _engine = _create_engine(sync_url, pool_pre_ping=True, pool_size=1, max_overflow=0)
        with _engine.connect() as conn:
            row = conn.execute(
                _sa.text(
                    "SELECT symbol, timeframe FROM data_sessions WHERE session_id = :sid LIMIT 1"
                ),
                {"sid": str(session_id)},
            ).fetchone()

        if row is None:
            logger.warning(f"[FootprintMerge] No DataSession found for session_id={session_id}")
            return df

        symbol, timeframe = row[0], row[1]

        # ── Load parquet ─────────────────────────────────────────────────────
        # Cache lives at:  Backend/app/data/footprint/{symbol}_{timeframe}.parquet
        cache_dir = Path(__file__).parent.parent.parent / "data" / "footprint"
        cache_path = cache_dir / f"{symbol}_{timeframe}.parquet"

        if not cache_path.exists():
            logger.warning(
                f"[FootprintMerge] No cached footprint at {cache_path}. "
                f"Run Footprint Ingestion for {symbol} {timeframe} first."
            )
            return df

        fp_df = pd.read_parquet(cache_path)

        # ── Ensure DatetimeIndex on both sides ──────────────────────────────
        if not isinstance(fp_df.index, pd.DatetimeIndex):
            if "time" in fp_df.columns:
                fp_df = fp_df.set_index(pd.to_datetime(fp_df["time"]))
            else:
                logger.warning("[FootprintMerge] Footprint parquet has no datetime index, skipping.")
                return df

        # Normalise timezone to naive UTC so indices align
        fp_df.index = pd.to_datetime(fp_df.index).tz_localize(None) if fp_df.index.tz is not None else pd.to_datetime(fp_df.index)

        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("[FootprintMerge] Input df has no DatetimeIndex, skipping merge.")
            return df

        df.index = pd.to_datetime(df.index).tz_localize(None) if df.index.tz is not None else pd.to_datetime(df.index)

        # Only join the raw fp_* columns TI needs
        fp_cols = [c for c in fp_df.columns if c.startswith("fp_")]
        if not fp_cols:
            logger.warning("[FootprintMerge] No fp_* columns in parquet, skipping merge.")
            return df

        merged = df.join(fp_df[fp_cols], how="left")

        n_covered = int(merged["fp_data_available"].notna().sum()) if "fp_data_available" in merged.columns else "?"
        logger.info(
            f"✅ [FootprintMerge] Joined {len(fp_cols)} fp_* columns for "
            f"{symbol} {timeframe} — {n_covered}/{len(merged)} bars have tick data"
        )
        return merged

    except Exception as exc:
        logger.warning(
            f"⚠️ [FootprintMerge] Non-fatal error merging footprint data: {exc}. "
            f"Continuing without footprint features.",
            exc_info=True,
        )
        return df


class AnalysisExecutionMixin:
    """Mixin: analysis execution methods (technical, SNR, astronomical, currency indices, TI)."""

    async def execute_technical_analysis(
        self,
        df: pd.DataFrame,
        pm: ProcessingManager,
        session_id: str,
        task_id: str,
        user_id: str = "anonymous",
    ) -> Dict[str, Any]:
        """
        Execute technical analysis using centralized ProcessingManager.
        ✅ INTEGRATED: Part of AnalysisManager, no external helpers needed
        ✅ UPDATES: self.current_data with enriched DataFrame for next step
        """
        # ── Footprint merge: inject fp_* raw columns before TI runs ─────────
        # Only load the parquet when enable_footprint=True on the config.
        # _merge_footprint_into_df() is non-fatal: if the cache doesn't exist
        # or the join fails it returns df unchanged and logs a warning.
        enable_fp = getattr(pm.config, "enable_footprint", False)
        if enable_fp:
            logger.info(f"[Technical] enable_footprint=True — merging fp_* columns into df before TI")
            df = await asyncio.get_event_loop().run_in_executor(
                None, _merge_footprint_into_df, df, session_id
            )

        result = await pm.execute(df, user_id=user_id)
        
        # Store result to DB (if not slice_streaming which does it internally)
        try:
            async with AsyncPostgresSessionLocal() as db:
                strategy_used = result.get("metadata", {}).get("strategy")
                # Only store if not slice_streaming, as slice_streaming already stored chunks
                if strategy_used != "slice_streaming":
                    # Standardized result mapping: prioritize result_df
                    df_to_store = result.get("result_df", result.get("features_df", df))
                    
                    # 🛡️ Guard: if result_df was stripped (e.g. already a preview dict/str), fall back to raw df
                    if not isinstance(df_to_store, (pd.DataFrame, list)):
                        logger.warning(
                            f"⚠️ [Technical persist] result_df is {type(df_to_store).__name__}, not DataFrame/list. "
                            f"Falling back to raw input df ({len(df)} rows) for storage."
                        )
                        df_to_store = df

                    # ⚡ MEMORY OPTIMIZATION: Pass DataFrame directly to avoid OOM from to_dict(orient='records')
                    # Converting 80,000 rows × 100+ TI columns to list[dict] instantiates ~10 million Python dict/float objects,
                    # causing severe heap memory spikes and OOM crashes during technical analysis step.
                    
                    # Clear cache before updating
                    self.clear_cache(session_id)
                    logger.info(f"Cache invalidated for session {session_id[:8]}...")
                    
                    await store_session_step_result(
                        session_id=session_id,
                        step_name="technical_analysis",
                        data=df_to_store,  # Pass DataFrame directly!
                        db=db,
                        force_pickle=True  # Technical results can be large
                    )
                    await set_as_current_data(session_id, db, task_id)
                    
                    # Persist config snapshot so _collect_step_configs can include it
                    # in the step_configs chain passed to the ML preparation step.
                    # pm.config is always the authoritative TechnicalConfig — read it directly.
                    try:
                        _ta_config_obj = pm.config
                        if _ta_config_obj is None:
                            _ta_config_snapshot = {}
                        elif hasattr(_ta_config_obj, "__dict__"):
                            _ta_config_snapshot = {
                                k: v for k, v in vars(_ta_config_obj).items()
                                if not k.startswith("_")
                            }
                        elif isinstance(_ta_config_obj, dict):
                            _ta_config_snapshot = dict(_ta_config_obj)
                        else:
                            _ta_config_snapshot = {}

                        await store_session_step_result(
                            session_id=session_id,
                            step_name="technical_analysis_config",
                            data={"config": _ta_config_snapshot},
                            db=db,
                        )
                        logger.info(f"[TA] Persisted technical_analysis config snapshot ({len(_ta_config_snapshot)} keys)")

                        # If footprint is enabled, also store a dedicated footprint_config row
                        # so InferenceFeaturePipeline can find the exact fp_* parameters used.
                        if getattr(_ta_config_obj, "enable_footprint", False):
                            _fp_config_snapshot = {
                                "enable_footprint": True,
                                "footprint_cum_delta_window": getattr(_ta_config_obj, "footprint_cum_delta_window", 20),
                                "footprint_rejection_lookback": getattr(_ta_config_obj, "footprint_rejection_lookback", 1),
                            }
                            await store_session_step_result(
                                session_id=session_id,
                                step_name="footprint_config",
                                data={"config": _fp_config_snapshot},
                                db=db,
                            )
                            logger.info(f"[TA] Persisted footprint_config snapshot: {_fp_config_snapshot}")
                    except Exception as _cfg_err:
                        logger.warning(f"[TA] Could not persist config snapshot: {_cfg_err}")

                    num_rows = len(df_to_store) if isinstance(df_to_store, (pd.DataFrame, list)) else 0
                    # Populate cache with fresh results
                    await self.cache_session_data(
                        session_id=session_id,
                        data=df_to_store,
                        source_step='technical_analysis',
                        ttl_seconds=1800
                    )
                    logger.info(f"Cached {num_rows} rows after technical analysis (memory optimized)")
                    
                    # ✅ UPDATE self.current_data with enriched DataFrame for next step
                    self.current_data = df_to_store if isinstance(df_to_store, pd.DataFrame) else pd.DataFrame(df_to_store)
                    self.current_session_id = session_id
                    logger.info(f"Updated self.current_data with {len(self.current_data)} enriched rows (TIER 0 for next step)")
                else:
                    # 🔴 BUG FIX: slice_streaming stores internally BUT we must load the FULLY MERGED result
                    # PM's _aggregate_slice_results() returns complete merged DataFrame in result["result_df"]
                    df_merged = result.get("result_df")
                    if not isinstance(df_merged, pd.DataFrame):
                        df_merged = IntermediateResultsCache.retrieve(task_id, "technical_analysis")
                    
                    if isinstance(df_merged, pd.DataFrame) and len(df_merged.columns) >= len(df.columns):
                        self.current_data = df_merged  # Reference, not copy!
                        self.current_session_id = session_id
                        
                        mem_mb = df_merged.memory_usage(deep=True).sum() / (1024 * 1024)
                        logger.info(
                            f"✅ [MEMORY OPTIMIZED] Loaded FULLY MERGED technical result (reference): "
                            f"{len(self.current_data)} rows, {mem_mb:.1f} MB "
                            f"(from {len(result.get('metadata', {}).get('slices', [])) if 'slices' in result.get('metadata', {}) else '?'} slices)"
                        )
                    else:
                        logger.warning(f"⚠️ Could not extract merged result_df from slice_streaming — clearing TIER 0a pointer to force DB load")
                        if self.current_session_id == session_id:
                            self.current_data = None
                    
                    await set_as_current_data(session_id, db, task_id)
        except Exception as db_err:
            logger.warning(f"Could not persist technical analysis result: {db_err}")
        
        return result
    
    async def execute_snr_analysis(
        self,
        df: pd.DataFrame,
        pm: ProcessingManager,
        session_id: str,
        task_id: str,
        user_id: str = "anonymous",
    ) -> Dict[str, Any]:
        """
        Execute SNR analysis using centralized ProcessingManager.
        
        🔄 UNIFIED DATA HANDLING:
        - PM executes processing and caches to TIER 1
        - PM MAY store to TIER 3 (depending on strategy)
        - AM checks coordination flags (pm_persisted, skip_am_persist)
        - AM handles FINAL storage ONLY (after assembly/validation)
        
        ✅ COORDINATION PROTOCOL:
        - If PM stored: result["metadata"]["pm_persisted"] = True
        - If AM should skip re-store: result["metadata"]["skip_am_persist"] = True
        - AM RESPECTS these flags to avoid double writes
        """
        # Memory monitoring at SNR analysis level
        try:
            import psutil
            process = psutil.Process()
            mem_start = process.memory_info().rss / 1024 / 1024
            df_size = df.memory_usage(deep=True).sum() / (1024 * 1024)
            logger.info(
                f"[SNR] Starting analysis. Memory: {mem_start:.1f} MB, "
                f"Input DF: {len(df)} rows, {df_size:.1f} MB"
            )
        except ImportError:
            process = None
            mem_start = None
        
        result = await pm.execute(df, user_id=user_id)
        
        if process:
            mem_after_pm = process.memory_info().rss / 1024 / 1024
            logger.info(
                f"[SNR] After ProcessingManager: {mem_after_pm:.1f} MB "
                f"(+{mem_after_pm - mem_start:.1f} MB)"
            )
        
        # ✅ UPDATE TIER 0a POINTER: Set current_data with SNR enriched result
        # This ensures ML Preparation and Astronomical Analysis get the latest SNR data
        df_merged = result.get("result_df")
        if not isinstance(df_merged, pd.DataFrame):
            df_merged = IntermediateResultsCache.retrieve(task_id, "snr_analysis")
        if not isinstance(df_merged, pd.DataFrame):
            df_merged = IntermediateResultsCache.retrieve(task_id, "signal_generation")
            
        if isinstance(df_merged, pd.DataFrame) and len(df_merged.columns) >= len(df.columns):
            self.current_data = df_merged  # Reference to merged result
            self.current_session_id = session_id
            
            mem_mb = df_merged.memory_usage(deep=True).sum() / (1024 * 1024)
            logger.info(
                f"⚡ TIER 0a HIT: Analysis pointer updated with SNR result\n"
                f"   ├─ Rows: {len(self.current_data)} (COMPLETE MERGED DATASET)\n"
                f"   ├─ Columns: {len(self.current_data.columns)}\n"
                f"   └─ Latency: ZERO ms from previous step"
            )
        else:
            logger.warning(f"⚠️ Could not extract merged result_df from SNR analysis — clearing TIER 0a to force DB load on next step")
            if self.current_session_id == session_id:
                self.current_data = None
        
        # ✅ CAPTURE: SNR Unprocessed Dataset for direct ML prep transition
        if "ml_dataset" in result:
            ml_dataset_raw = result["ml_dataset"]
            
            # ✅ NEW: Reconstruct sequences from lightweight format if needed
            if isinstance(ml_dataset_raw, list) and len(ml_dataset_raw) > 0:
                first_item = ml_dataset_raw[0]
                
                # Check if it's the new lightweight format
                if "sequence_start" in first_item and "sequence_end" in first_item:
                    logger.info(f"🔄 [AM] Reconstructing ML sequences for in-memory cache...")
                    
                    from app.core.analysis.trading.signal_generator import reconstruct_ml_sequences
                    
                    result_df = result.get("result_df")
                    if result_df is not None and isinstance(result_df, pd.DataFrame):
                        self.unprocessed_dataset = reconstruct_ml_sequences(ml_dataset_raw, result_df)
                        logger.info(f"🎯 [AM] Captured SNR Unprocessed Dataset with {len(self.unprocessed_dataset)} reconstructed records")
                    else:
                        logger.warning("[AM] ⚠️ Cannot reconstruct sequences: result_df missing")
                        self.unprocessed_dataset = ml_dataset_raw  # Store lightweight format as fallback
                else:
                    # Already in full format
                    self.unprocessed_dataset = ml_dataset_raw
                    logger.info(f"🎯 [AM] Captured SNR Unprocessed Dataset with {len(self.unprocessed_dataset)} records")
            else:
                self.unprocessed_dataset = ml_dataset_raw
                logger.info(f"🎯 [AM] Captured SNR Unprocessed Dataset")
            
            self.unprocessed_session_id = session_id
        
        # Store result to DB (with coordination checks)
        try:
            async with AsyncPostgresSessionLocal() as db:
                strategy_used = result.get("metadata", {}).get("strategy")
                
                # ✅ FIX: Ensure boolean values (not DataFrames or other ambiguous types)
                pm_persisted_raw = result.get("metadata", {}).get("pm_persisted", False)
                skip_am_persist_raw = result.get("metadata", {}).get("skip_am_persist", False)
                
                # Convert to explicit boolean (handles DataFrame, None, etc.)
                # ✅ CRITICAL: Use isinstance check FIRST to avoid ambiguous truth value
                if isinstance(pm_persisted_raw, pd.DataFrame):
                    pm_persisted = False
                elif pm_persisted_raw is None:
                    pm_persisted = False
                else:
                    pm_persisted = bool(pm_persisted_raw)
                
                if isinstance(skip_am_persist_raw, pd.DataFrame):
                    skip_am_persist = False
                elif skip_am_persist_raw is None:
                    skip_am_persist = False
                else:
                    skip_am_persist = bool(skip_am_persist_raw)
                
                pm_persist_failed = result.get("metadata", {}).get("pm_persist_failed", False)
                
                logger.info(
                    f"[AM] 🔍 SNR coordination: pm_persisted={pm_persisted}, "
                    f"skip_am_persist={skip_am_persist}, strategy={strategy_used}"
                )

                # 🔄 Build a sync keepalive reporter for storage progress
                # This ensures the frontend doesn't timeout during large ML dataset storage
                def storage_progress_reporter(pct: int, msg: str) -> None:
                    label = f"💾 Persisting SNR Data: {msg}"

                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.ensure_future(
                                pm._send_progress_update(
                                    pct, label, {"stage": "storage", "step": "snr_analysis"}
                                )
                            )
                    except Exception:
                        pass

                # Strategy handling for SNR Persistence
                if not skip_am_persist and not pm_persisted:
                    df_to_store = result.get("result_df", result.get("features_df", df))
                    if not isinstance(df_to_store, (pd.DataFrame, list)):
                        df_to_store = df
                    
                    self.clear_cache(session_id)
                    num_rows = len(df_to_store) if isinstance(df_to_store, (pd.DataFrame, list)) else 0
                    await store_session_step_result(
                        session_id=session_id,
                        step_name="snr_analysis",
                        data=df_to_store,
                        db=db,
                        force_pickle=True,
                        on_progress=storage_progress_reporter
                    )

                    logger.info(f"✅ [AM] COMMITTED snr_analysis (session={session_id[:8]}, rows={num_rows})")
                    
                    # Persist SNR config snapshot for step_configs chain.
                    try:
                        _snr_config_obj = pm.config
                        if _snr_config_obj is None:
                            _snr_config_snapshot = {}
                        elif hasattr(_snr_config_obj, "__dict__"):
                            _snr_config_snapshot = {
                                k: v for k, v in vars(_snr_config_obj).items()
                                if not k.startswith("_")
                            }
                        elif isinstance(_snr_config_obj, dict):
                            _snr_config_snapshot = dict(_snr_config_obj)
                        else:
                            _snr_config_snapshot = {}

                        await store_session_step_result(
                            session_id=session_id,
                            step_name="snr_analysis_config",
                            data={"config": _snr_config_snapshot},
                            db=db,
                        )
                        logger.info(f"[SNR] Persisted snr_analysis_config snapshot ({len(_snr_config_snapshot)} keys)")
                    except Exception as _cfg_err:
                        logger.warning(f"[SNR] Could not persist config snapshot: {_cfg_err}")
                
                # 🔴 ALWAYS store ml_dataset (this is unique to SNR, not duplicated by PM)
                ml_dataset = result.get("ml_dataset")
                if ml_dataset is not None:
                    ml_dataset = result["ml_dataset"]
                    
                    # ✅ NEW: Reconstruct sequences from lightweight format if needed
                    if isinstance(ml_dataset, list) and len(ml_dataset) > 0:
                        first_item = ml_dataset[0]
                        
                        # Check if it's the new lightweight format (has sequence_start/sequence_end)
                        if "sequence_start" in first_item and "sequence_end" in first_item:
                            logger.info(f"[AM] 🔄 Reconstructing {len(ml_dataset)} ML sequences from lightweight format...")
                            
                            # Import reconstruction function
                            from app.core.analysis.trading.signal_generator import reconstruct_ml_sequences
                            
                            # Get the result_df for reconstruction
                            result_df = result.get("result_df")
                            if result_df is None or not isinstance(result_df, pd.DataFrame):
                                logger.error("[AM] ❌ Cannot reconstruct ML sequences: result_df missing")
                                ml_records = []
                            else:
                                # Reconstruct full sequences
                                ml_records = reconstruct_ml_sequences(ml_dataset, result_df)
                                logger.info(f"[AM] ✅ Reconstructed {len(ml_records)} ML sequences")
                        else:
                            # Already in full format (legacy or direct format)
                            ml_records = ml_dataset
                    elif isinstance(ml_dataset, pd.DataFrame):
                        # Convert DataFrame to records
                        ml_records = ml_dataset.to_dict(orient='records')
                    else:
                        ml_records = ml_dataset if isinstance(ml_dataset, list) else [ml_dataset]
                    
                    logger.info(f"[AM] 📝 Storing SNR ml_dataset: {len(ml_records)} records")
                    await store_session_step_result(
                        session_id=session_id,
                        step_name="snr_analysis_ml_dataset",
                        data=ml_records,
                        db=db,
                        force_pickle=True,
                        on_progress=storage_progress_reporter
                    )

                    logger.info(f"✅ [AM] COMMITTED snr_analysis_ml_dataset (rows={len(ml_records)})")
                else:
                    logger.warning(f"⚠️ [AM] SNR ml_dataset missing or empty")
                
                # Mark as current data (AM responsibility for SNR)
                await set_as_current_data(session_id, db, task_id)
            
            # Populate cache with enriched results
            if "snr_analysis" in locals():
                try:
                    await self.cache_session_data(
                        session_id=session_id,
                        data=result.get("result_records", result.get("result_df")),
                        source_step='snr_analysis',
                        ttl_seconds=1800
                    )
                    logger.info(f"[AM] 📌 Cached SNR results (NaN preserved)")
                except:
                    pass  # Cache failure non-fatal
            
            # ✅ UPDATE self.current_data with enriched DataFrame for next step
            self.current_session_id = session_id
            
            if process and mem_start is not None:
                mem_end = process.memory_info().rss / 1024 / 1024
                current_data_size = self.current_data.memory_usage(deep=True).sum() / (1024 * 1024)
                logger.info(
                    f"[SNR] Completed. Memory: {mem_end:.1f} MB "
                    f"(+{mem_end - mem_start:.1f} MB total), "
                    f"current_data: {len(self.current_data)} rows, {current_data_size:.1f} MB"
                )
                    
        except Exception as db_err:
            logger.warning(f"[AM] ⚠️  Could not persist SNR analysis result: {db_err}")
        
        return result
    
    async def execute_astronomical_analysis(
        self,
        df: pd.DataFrame,
        pm: ProcessingManager,
        session_id: str,
        task_id: str,
        user_id: str = "anonymous",
    ) -> Dict[str, Any]:
        """
        Execute astronomical analysis using centralized ProcessingManager.
        ✅ INTEGRATED: Part of AnalysisManager, no external helpers needed
        ✅ UPDATES: self.current_data with enriched DataFrame for next step

        DB WRITE STRATEGY:
        - PM._persist_to_database() already writes the enriched result to
          'astronomical_analysis' as its STEP 5.  We must NOT write a second
          time — on_conflict_do_update would stomp the correct enriched data
          with whatever df_to_store resolves to here.
        - This function's responsibility is:
            a) update self.current_data / self.current_session_id in-process
            b) call set_as_current_data so the pointer row is current
            c) refresh the AM-level session cache
        - Exception: slice_streaming stores slices internally but does NOT call
          set_as_current_data; we handle that pointer update here too.
        """
        result = await pm.execute(df, user_id=user_id)

        try:
            async with AsyncPostgresSessionLocal() as db:
                strategy_used = result.get("metadata", {}).get("strategy")

                if strategy_used != "slice_streaming":
                    # ── Recover enriched df from result (PM may have cleared it post-persist) ──
                    df_enriched = result.get("result_df")
                    if not isinstance(df_enriched, pd.DataFrame):
                        # PM cleared result_df to free memory after persisting — recover
                        # from TIER 1 cache using the correct step name.
                        df_enriched = IntermediateResultsCache.retrieve(
                            task_id, "astronomical_analysis"
                        )
                    if not isinstance(df_enriched, pd.DataFrame):
                        # Cache expired or unavailable — PM persisted correctly so DB is
                        # the source of truth; just log and proceed without self.current_data update.
                        logger.warning(
                            f"⚠️ [Astronomical] Could not recover enriched df from result or "
                            f"TIER 1 cache for session {session_id[:8]}. "
                            f"DB write by PM is authoritative; skipping self.current_data update."
                        )
                        # Still advance the current_data pointer in DB
                        await set_as_current_data(session_id, db, task_id)
                        return result

                    # ── PM already wrote to DB; just mark as current and refresh AM cache ──
                    await set_as_current_data(session_id, db, task_id)

                    # ── Persist astronomical config snapshot ──────────────────────
                    # pm.config is the authoritative AstronomicalConfig for this run.
                    # Stored as 'astronomical_analysis_config' so InferenceFeaturePipeline
                    # can reproduce the exact ephemeris settings at inference time.
                    # Key fields: observer_lat/lon, house_system, zodiac_type,
                    # use_minor_aspects, aspect_orbs, selected_features,
                    # include_asteroids, include_fixed_stars, precision_mode.
                    try:
                        _astro_config_obj = pm.config
                        if _astro_config_obj is None:
                            _astro_config_snapshot = {}
                        elif hasattr(_astro_config_obj, "__dict__"):
                            _astro_config_snapshot = {
                                k: v for k, v in vars(_astro_config_obj).items()
                                if not k.startswith("_")
                            }
                        elif isinstance(_astro_config_obj, dict):
                            _astro_config_snapshot = dict(_astro_config_obj)
                        else:
                            _astro_config_snapshot = {}

                        if _astro_config_snapshot:
                            await store_session_step_result(
                                session_id=session_id,
                                step_name="astronomical_analysis_config",
                                data={"config": _astro_config_snapshot},
                                db=db,
                            )
                            logger.info(
                                "[Astronomical] Persisted astronomical_analysis_config snapshot "
                                "(%d keys): house_system=%s, zodiac_type=%s, "
                                "use_minor_aspects=%s, selected_features=%d",
                                len(_astro_config_snapshot),
                                _astro_config_snapshot.get("house_system", "?"),
                                _astro_config_snapshot.get("zodiac_type", "?"),
                                _astro_config_snapshot.get("use_minor_aspects", "?"),
                                len(_astro_config_snapshot.get("selected_features") or []),
                            )
                    except Exception as _cfg_err:
                        logger.warning(
                            "[Astronomical] Could not persist config snapshot: %s", _cfg_err
                        )

                    # Refresh AM-level session cache directly with DataFrame (memory optimized)
                    self.clear_cache(session_id)
                    await self.cache_session_data(
                        session_id=session_id,
                        data=df_enriched,
                        source_step='astronomical_analysis',
                        ttl_seconds=1800
                    )
                    logger.info(
                        f"✅ [Astronomical] Cached {len(df_enriched)} enriched rows "
                        f"({len(df_enriched.columns)} cols) for session {session_id[:8]}"
                    )

                    # Update in-process current_data pointer
                    self.current_data = df_enriched
                    self.current_session_id = session_id

                else:
                    # slice_streaming: PM stored slices internally; load the merged result
                    df_merged = result.get("result_df")
                    if isinstance(df_merged, pd.DataFrame):
                        self.current_data = df_merged
                        self.current_session_id = session_id
                        mem_mb = df_merged.memory_usage(deep=True).sum() / (1024 * 1024)
                        logger.info(
                            f"✅ [Astronomical slice_streaming] Merged result: "
                            f"{len(self.current_data)} rows, {mem_mb:.1f} MB"
                        )
                    else:
                        logger.warning(
                            f"⚠️ [Astronomical slice_streaming] Could not extract merged result_df"
                        )
                    await set_as_current_data(session_id, db, task_id)

                    # ── Persist astronomical config snapshot ──────────────────────
                    # pm.config is the authoritative AstronomicalConfig for this run.
                    # Stored as 'astronomical_analysis_config' so InferenceFeaturePipeline
                    # can reproduce the exact ephemeris settings at inference time.
                    # Key fields: observer_lat/lon, house_system, zodiac_type,
                    # use_minor_aspects, aspect_orbs, selected_features,
                    # include_asteroids, include_fixed_stars, precision_mode.
                    try:
                        _astro_config_obj = pm.config
                        if _astro_config_obj is None:
                            _astro_config_snapshot = {}
                        elif hasattr(_astro_config_obj, "__dict__"):
                            _astro_config_snapshot = {
                                k: v for k, v in vars(_astro_config_obj).items()
                                if not k.startswith("_")
                            }
                        elif isinstance(_astro_config_obj, dict):
                            _astro_config_snapshot = dict(_astro_config_obj)
                        else:
                            _astro_config_snapshot = {}

                        if _astro_config_snapshot:
                            await store_session_step_result(
                                session_id=session_id,
                                step_name="astronomical_analysis_config",
                                data={"config": _astro_config_snapshot},
                                db=db,
                            )
                            logger.info(
                                "[Astronomical] Persisted astronomical_analysis_config snapshot "
                                "(%d keys): house_system=%s, zodiac_type=%s, "
                                "use_minor_aspects=%s, selected_features=%d",
                                len(_astro_config_snapshot),
                                _astro_config_snapshot.get("house_system", "?"),
                                _astro_config_snapshot.get("zodiac_type", "?"),
                                _astro_config_snapshot.get("use_minor_aspects", "?"),
                                len(_astro_config_snapshot.get("selected_features") or []),
                            )
                    except Exception as _cfg_err:
                        logger.warning(
                            "[Astronomical] Could not persist config snapshot: %s", _cfg_err
                        )

        except Exception as db_err:
            logger.warning(f"Could not finalise astronomical analysis result: {db_err}")

        return result
    
    async def _prefetch_currency_pairs(
        self,
        df: pd.DataFrame,
        config: Any,
    ) -> pd.DataFrame:
        """
        Pre-fetch all missing currency pair OHLCV data from MT5 ONCE in the
        async context, before df is split into parallel chunks.

        Without this, each of the N worker subprocesses independently calls the
        MT5 bridge for the same pairs — up to N*pairs concurrent HTTP requests
        all queued behind the bridge's single-thread executor.  Under that load
        the MT5 COM/Wine layer returns None for some symbols, the bridge raises
        RuntimeError and responds HTTP 500.

        Fetching once here (bounded concurrency=5, with per-pair retry) makes at
        most 3*len(pairs) total requests.  Workers find every pair column already
        present and raise ValueError immediately if any column is still missing.

        Timeout scaling (user's comment: 80k rows needs much longer than 200s):
          base 60s + 1s per 1000 rows, clamped to [60, 600] seconds per pair.
          Example: 80,000 rows => 60+80 = 140s per pair.
        """
        from app.core.processing.processing_strategies import INDEX_DEFINITIONS, OHLCV_FIELDS

        # ── collect required pairs ──────────────────────────────────────────
        required_pairs: list = []
        for idx_name in getattr(config, "selected_indices", []):
            if idx_name in INDEX_DEFINITIONS:
                for pair in INDEX_DEFINITIONS[idx_name]["pairs"]:
                    if pair not in required_pairs:
                        required_pairs.append(pair)

        if not required_pairs:
            return df

        available_cols = set(df.columns)
        missing_pairs = [
            p for p in required_pairs
            if any(
                f"{field}_{p}" not in available_cols
                for field in ["open", "high", "low", "close", "tick_volume"]
            )
        ]

        if not missing_pairs:
            logger.info("[CurrencyIndices pre-fetch] All pair columns already present — skip")
            return df

        # Record original row count — used at the end to detect fan-out from
        # duplicate timestamps in fetched pair data.
        _original_row_count = len(df)

        logger.info(
            "[CurrencyIndices pre-fetch] Fetching %d missing pairs from MT5 (single pass, before chunking)",
            len(missing_pairs),
        )

        # ── detect timeframe from row delta ─────────────────────────────────
        timeframe = getattr(config, "timeframe", None)
        if not timeframe:
            try:
                t0 = df["Time"].iloc[0]
                t1 = df["Time"].iloc[1]
                delta = (t1 - t0) if isinstance(t0, (int, float)) else (
                    pd.to_datetime(t1) - pd.to_datetime(t0)
                ).total_seconds()
                if   delta <= 60:    timeframe = "M1"
                elif delta <= 300:   timeframe = "M5"
                elif delta <= 900:   timeframe = "M15"
                elif delta <= 1800:  timeframe = "M30"
                elif delta <= 3600:  timeframe = "H1"
                elif delta <= 14400: timeframe = "H4"
                else:                timeframe = "D1"
            except Exception:
                timeframe = "H1"

        fetch_count = len(df) + 100
        t_last = df["Time"].iloc[-1]
        if isinstance(t_last, (int, float, np.integer, np.floating)):
            date_from = pd.Timestamp(t_last, unit="s").to_pydatetime()
        else:
            date_from = pd.to_datetime(t_last).to_pydatetime()

        # ── Scale per-pair timeout with dataset size ────────────────────────────
        # MT5 bridge fetches are proportional to row count.  Large datasets
        # (e.g. 80k rows of H1) can take 60-120s per pair over the COM bridge.
        # Formula: base 60s + 1s per 1000 rows, clamped to [60, 600] seconds.
        # Example: 80,000 rows => min(600, 60 + 80) = 140s per pair.
        _per_pair_timeout = int(max(60, min(600, 60 + (len(df) / 1000))))
        logger.info(
            "[CurrencyIndices pre-fetch] Per-pair timeout: %ds "
            "(dataset=%d rows, timeframe=%s, fetch_count=%d)",
            _per_pair_timeout, len(df), timeframe, fetch_count,
        )

        # ── fetch with bounded concurrency + per-pair retry ──────────────────
        # Semaphore keeps at most 5 pairs in-flight to avoid overwhelming the
        # MT5 COM bridge.  Each pair gets up to 3 attempts with exponential
        # backoff (2s, 4s) so transient bridge timeouts are self-healed.
        sem = asyncio.Semaphore(5)

        async def _fetch_one(pair: str):
            """Fetch one pair with retries and scaled timeout."""
            from app.services.mt5_service import MT5Service as _MT5Service
            mt5 = self.data_fetcher.mt5_service or _MT5Service()
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                async with sem:
                    try:
                        logger.info(
                            "[CurrencyIndices pre-fetch] %s — attempt %d/%d "
                            "(timeout=%ds, count=%d)",
                            pair, attempt, max_retries, _per_pair_timeout, fetch_count,
                        )
                        res = await asyncio.wait_for(
                            mt5.fetch_ohlc_data_v2(
                                symbol=pair,
                                timeframe=timeframe,
                                count=fetch_count,
                                date_from=date_from,
                            ),
                            timeout=_per_pair_timeout,
                        )
                        if isinstance(res, list) and res:
                            logger.info(
                                "[CurrencyIndices pre-fetch] OK %s — %d bars",
                                pair, len(res),
                            )
                            return pair, res
                        if isinstance(res, dict) and res.get("data"):
                            logger.info(
                                "[CurrencyIndices pre-fetch] OK %s — %d bars (dict)",
                                pair, len(res["data"]),
                            )
                            return pair, res["data"]
                        logger.warning(
                            "[CurrencyIndices pre-fetch] Empty response for %s "
                            "(attempt %d/%d): type=%s",
                            pair, attempt, max_retries, type(res).__name__,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "[CurrencyIndices pre-fetch] Timeout for %s "
                            "(attempt %d/%d, limit=%ds) — dataset may be very large",
                            pair, attempt, max_retries, _per_pair_timeout,
                        )
                    except Exception as exc:
                        logger.error(
                            "[CurrencyIndices pre-fetch] Error fetching %s "
                            "(attempt %d/%d): %s",
                            pair, attempt, max_retries, exc,
                        )

                # Exponential backoff between retries: 2s then 4s
                if attempt < max_retries:
                    backoff = 2 ** attempt
                    logger.info(
                        "[CurrencyIndices pre-fetch] Retrying %s in %ds...",
                        pair, backoff,
                    )
                    await asyncio.sleep(backoff)

            logger.error(
                "[CurrencyIndices pre-fetch] FAILED %s after %d attempts. "
                "Workers will raise ValueError if this pair is required.",
                pair, max_retries,
            )
            return pair, []

        results = await asyncio.gather(*[_fetch_one(p) for p in missing_pairs])

        # ── merge into df ────────────────────────────────────────────────────
        df_time_dtype = df["Time"].dtype
        for pair, data in results:
            if not data:
                logger.warning("[CurrencyIndices pre-fetch] No data for %s — workers will handle", pair)
                continue
            try:
                pair_df = pd.DataFrame(data)

                # Integer column indices → names (MT5 raw array format)
                if len(pair_df.columns) and isinstance(pair_df.columns[0], int):
                    pair_df.rename(columns={
                        0: "time", 1: "open", 2: "high", 3: "low",
                        4: "close", 5: "tick_volume", 6: "spread", 7: "real_volume",
                    }, inplace=True)

                if "time" in pair_df.columns:
                    pair_df.rename(columns={"time": "Time"}, inplace=True)
                if "Time" not in pair_df.columns:
                    logger.warning("[CurrencyIndices pre-fetch] No Time column for %s — skip", pair)
                    continue

                # Align Time dtypes so merge works correctly
                pair_time_dtype = pair_df["Time"].dtype
                if pd.api.types.is_datetime64_any_dtype(df_time_dtype):
                    if not pd.api.types.is_datetime64_any_dtype(pair_time_dtype):
                        if pd.api.types.is_numeric_dtype(pair_time_dtype):
                            pair_df["Time"] = pd.to_datetime(pair_df["Time"], unit="s")
                        else:
                            pair_df["Time"] = pd.to_datetime(pair_df["Time"])
                elif pd.api.types.is_numeric_dtype(df_time_dtype):
                    if not pd.api.types.is_numeric_dtype(pair_time_dtype):
                        if pd.api.types.is_datetime64_any_dtype(pair_time_dtype):
                            pair_df["Time"] = pair_df["Time"].astype(np.int64) // 10 ** 9
                        else:
                            pair_df["Time"] = pd.to_datetime(pair_df["Time"]).astype(np.int64) // 10 ** 9
                else:
                    df["Time"] = pd.to_datetime(df["Time"])
                    if pd.api.types.is_numeric_dtype(pair_time_dtype):
                        pair_df["Time"] = pd.to_datetime(pair_df["Time"], unit="s")
                    else:
                        pair_df["Time"] = pd.to_datetime(pair_df["Time"])

                rename_map = {}
                for field in ["open", "high", "low", "close", "tick_volume", "real_volume"]:
                    if field in pair_df.columns:
                        rename_map[field] = f"{field}_{pair}"
                    elif field.capitalize() in pair_df.columns:
                        rename_map[field.capitalize()] = f"{field}_{pair}"

                if not rename_map:
                    logger.warning(
                        "[CurrencyIndices pre-fetch] No OHLCV cols for %s in %s",
                        pair, list(pair_df.columns),
                    )
                    continue

                pair_df.rename(columns=rename_map, inplace=True)
                pair_df = pair_df[["Time"] + list(rename_map.values())]
                df = df.merge(pair_df, on="Time", how="left")
                for col in rename_map.values():
                    if col in df.columns:
                        df[col] = df[col].ffill().bfill()

                logger.info("[CurrencyIndices pre-fetch] ✅ Merged %s (%d rows)", pair, len(pair_df))

            except Exception as exc:
                logger.error(
                    "[CurrencyIndices pre-fetch] Merge error for %s: %s", pair, exc, exc_info=True
                )

        successful_pairs = sum(
            1 for p in missing_pairs
            if all(f"{field}_{p}" in df.columns for field in ["open", "close"])
        )
        logger.info(
            "[CurrencyIndices pre-fetch] Complete — %d/%d pairs merged. "
            "df shape: %s (+%d new cols).",
            successful_pairs, len(missing_pairs),
            df.shape, len(df.columns) - len(available_cols),
        )

        # ── Length invariant check ───────────────────────────────────────────
        # A left-join must never drop rows from the base DataFrame.
        # If the row count changed, something went wrong in the merge logic
        # (e.g. duplicate timestamps in a pair's data causing a fan-out).
        # Guard: log a WARNING (not raise) so the step degrades gracefully rather
        # than failing completely when a pair has duplicate timestamps.
        if len(df) != _original_row_count:
            logger.error(
                "[CurrencyIndices pre-fetch] ❌ Row count changed during merge: "
                "started=%d, ended=%d (delta=%+d). "
                "A pair with duplicate timestamps caused a fan-out. "
                "Deduplicating on Time to restore original length.",
                _original_row_count, len(df), len(df) - _original_row_count,
            )
            # Restore: keep first occurrence of each timestamp (preserves original order)
            df = df[~df["Time"].duplicated(keep="first")].reset_index(drop=True)
            if len(df) != _original_row_count:
                logger.warning(
                    "[CurrencyIndices pre-fetch] After dedup: %d rows (expected %d). "
                    "Some base-symbol timestamps may not have pair coverage.",
                    len(df), _original_row_count,
                )

        return df

    async def execute_currency_indices_analysis(
        self,
        df: pd.DataFrame,
        pm: ProcessingManager,
        session_id: str,
        task_id: str,
        config: Any = None,
        user_id: str = "anonymous",
    ) -> Dict[str, Any]:
        """
        Execute currency indices analysis using centralized ProcessingManager.
        ✅ INTEGRATED: Part of AnalysisManager, no external helpers needed
        ✅ UPDATES: self.current_data with enriched DataFrame for next step
        """
        # STEP 0: Pre-fetch all missing pair columns into df BEFORE chunking.
        # Each parallel worker would otherwise independently call the MT5 bridge
        # for the same pairs → up to (chunks × pairs) concurrent requests → 500s.
        #
        # IMPORTANT: We pass the pair-enriched df to pm.execute() so each chunk
        # already has the pair columns and workers skip the bridge fetch entirely.
        # However, the PM's "_ensure_result_completeness" uses the df it receives
        # as the "original" for column enrichment — meaning it would re-add the
        # pair columns to the final result (which the workers explicitly drop).
        #
        # Solution: record the original base columns BEFORE pre-fetch so we can
        # strip pair columns from the result after pm.execute() returns.
        # The PM stores df at the point of execute() call; by recording which
        # columns are "pair columns" added by pre-fetch, we can remove them from
        # the final result after the PM's enrichment step re-adds them.
        if config is None:
            config = getattr(pm, "config", None)

        _base_columns: set | None = None  # columns in df BEFORE pre-fetch
        if config is not None:
            _base_columns = set(df.columns)
            try:
                df_enriched = await self._prefetch_currency_pairs(df, config)
            except Exception as prefetch_err:
                logger.warning(
                    "[CurrencyIndices] Pre-fetch failed (%s) — workers will attempt individual fetches",
                    prefetch_err,
                )
                df_enriched = df
                _base_columns = None  # Reset — no pair columns were added
        else:
            df_enriched = df

        # Record input length for post-execution length guard
        _input_row_count = len(df)

        # KEY: pass the PAIR-ENRICHED df to pm.execute() so workers receive
        # chunk_df WITH pair columns already present (they skip the MT5 fetch).
        # But PM stores this df as `original_df` for its enrichment step, which
        # would re-add pair cols after workers drop them.
        #
        # Solution: after pm.execute() returns (and PM has already persisted
        # currency_indices_analysis), strip pair cols from result before the
        # mixin stores `currency_indices` (the TI-post-processed final step).
        # The `currency_indices_analysis` DB record will have pair cols but the
        # authoritative `currency_indices` record (read by downstream steps) will not.
        result = await pm.execute(df_enriched, user_id=user_id)

        # ── Strip pair columns that PM's enrichment re-added ─────────────────
        # The worker drops pair columns (open_EURUSD, close_USDJPY, etc.) from
        # each chunk result. But PM._ensure_result_completeness() re-adds ALL
        # columns from its stored original_df (which is the pair-enriched df we
        # passed in). Strip them from the final result so only the true output
        # columns remain: original base OHLCV + computed index cols (+ TI if enabled).
        result_df = result.get("result_df")
        if result_df is None:
            result_df = result.get("features_df")
        if isinstance(result_df, pd.DataFrame) and _base_columns is not None:
            from app.core.processing.processing_strategies import INDEX_DEFINITIONS, OHLCV_FIELDS
            pair_col_prefixes = tuple(
                f"{field}_"
                for field in list(OHLCV_FIELDS) + ["real_volume", "spread"]
            )
            pair_cols_to_drop = [
                c for c in result_df.columns
                if c not in _base_columns
                and any(c.startswith(pfx) for pfx in pair_col_prefixes)
                # Keep index value columns (e.g. Dollar_open, Dollar_close)
                and not any(
                    c.startswith(f"{idx_name}_")
                    for idx_name in INDEX_DEFINITIONS
                )
            ]
            if pair_cols_to_drop:
                logger.info(
                    "[CurrencyIndices] Dropping %d re-added pair columns from result "
                    "(added back by PM enrichment, not part of output): %s…",
                    len(pair_cols_to_drop), pair_cols_to_drop[:4],
                )
                result_df = result_df.drop(columns=pair_cols_to_drop)
                result["result_df"] = result_df

        # ── Row-length guard ──────────────────────────────────────────────────
        if isinstance(result_df, pd.DataFrame) and len(result_df) > 0:
            result_len = len(result_df)
            if result_len > _input_row_count:
                logger.error(
                    "[CurrencyIndices] ❌ Result LONGER than input: "
                    "input=%d → result=%d (+%d). Duplicate timestamps — truncating.",
                    _input_row_count, result_len, result_len - _input_row_count,
                )
                result_df = result_df.iloc[:_input_row_count]
                result["result_df"] = result_df
            elif result_len < _input_row_count:
                pct = (_input_row_count - result_len) / _input_row_count * 100
                (logger.warning if pct < 5 else logger.error)(
                    "[CurrencyIndices] ⚠️ Result shorter than input: "
                    "input=%d → result=%d (dropped %d = %.1f%%). "
                    "Acceptable when pair history is shorter than base symbol.",
                    _input_row_count, result_len, _input_row_count - result_len, pct,
                )
        
        # STEP 2: Extract result DataFrame and enhance with TI if needed
        result_df = result.get('result_df')
        if result_df is None:
            result_df = result.get('features_df')
        
        # STEP 3: The config (CurrencyIndexConfig) is passed directly from the caller.
        # It contains calculate_ti_for_indices populated by the frontend.
        # No need to read from metadata — pm.config IS the correct config.
        if config is None:
            # Last-resort fallback: try to read from pm.config
            config = getattr(pm, 'config', None)
        
        if config is None:
            logger.warning("[Currency Indices] No config available for TI post-processing")
        
        if isinstance(result_df, pd.DataFrame):
            logger.info(f"[Currency Indices] Post-processing with TI: checking config...")

            # Guard: if result_df already has TI columns from a cached re-run,
            # skip TI to avoid double-application and the OutOfBoundsDatetime crash
            # caused by index value columns (~50) being mistaken for Unix timestamps.
            ti_enabled_dict = None
            if isinstance(config, dict):
                ti_enabled_dict = config.get("calculate_ti_for_indices")
            elif config is not None:
                ti_enabled_dict = getattr(config, "calculate_ti_for_indices", None)

            ti_requested_indices = [idx for idx, enabled in (ti_enabled_dict or {}).items() if enabled]
            # Check ONLY for prefixed TI columns (e.g. Dollar_RSI_14).
            # Do NOT check for bare "RSI_14" — that column comes from the base-pair
            # Technical Analysis step and must not suppress index TI computation.
            already_has_ti = any(
                f"{idx_name}_RSI_14" in result_df.columns
                for idx_name in (ti_requested_indices or ["Dollar"])
            ) if ti_requested_indices else False

            if already_has_ti:
                logger.info(
                    "[Currency Indices] TI columns already present in result_df (%d cols) — skipping re-application",
                    len(result_df.columns),
                )
            else:
                # Apply TI post-processing if enabled (leverages PM for each index)
                result_df = await self.execute_ti_for_currency_indices(
                    result_df, config, pm, session_id, task_id, user_id
                )
                # Update result with enhanced DataFrame
                result['result_df'] = result_df
            logger.info(f"[Currency Indices] TI post-processing complete: {len(result_df.columns)} total columns")
        
        # Store result to DB (if not slice_streaming which does it internally)
        try:
            async with AsyncPostgresSessionLocal() as db:
                strategy_used = result.get("metadata", {}).get("strategy")
                if strategy_used != "slice_streaming":
                    # CRITICAL: Use the TI-enriched result_df (set above in result['result_df'] = result_df)
                    # Do NOT fall back to TIER 1 cache — that cache has the pre-TI 23-column version.
                    df_to_store = result.get("result_df")
                    if df_to_store is None:
                        df_to_store = result.get("features_df")
                    
                    # Only fall back to raw df if result_df is completely missing
                    if not isinstance(df_to_store, (pd.DataFrame, list)):
                        logger.warning(
                            f"⚠️ [Currency Indices persist] result_df is {type(df_to_store).__name__}, "
                            f"falling back to raw input df ({len(df)} rows)"
                        )
                        df_to_store = df

                    # ⚡ MEMORY OPTIMIZATION: Pass DataFrame directly to avoid OOM from to_dict(orient='records')
                    # Converting 80,000 rows × 150 columns to list[dict] instantiates ~12 million Python dict/float objects
                    # expanding 96MB of DataFrame buffer to >3.5GB heap memory and causing OOM killer.
                    
                    # Clear AM session cache AND decompress cache before updating with TI-enriched result
                    self.clear_cache(session_id)
                    logger.info(f"Cache invalidated for session {session_id[:8]}...")
                    
                    # Invalidate decompress cache for currency_indices (may hold pre-TI version)
                    try:
                        from app.core.services.decompress_cache import get_cache as get_decompress_cache
                        decompress_cache = get_decompress_cache()
                        async with decompress_cache.lock:
                            for step_key_suffix in ['currency_indices', 'currency_indices_analysis']:
                                step_key = decompress_cache._make_key(session_id, step_key_suffix)
                                if step_key in decompress_cache.cache:
                                    del decompress_cache.cache[step_key]
                                    logger.info(f"[Currency Indices] Invalidated decompress cache for '{step_key_suffix}'")
                    except Exception as cache_err:
                        logger.debug(f"[Currency Indices] Could not invalidate decompress cache: {cache_err}")
                    
                    await store_session_step_result(
                        session_id=session_id,
                        step_name="currency_indices",
                        data=df_to_store,  # Pass DataFrame directly!
                        db=db,
                        force_pickle=True  # Results can be large
                    )
                    await set_as_current_data(session_id, db, task_id)
                    num_rows = len(df_to_store) if isinstance(df_to_store, (pd.DataFrame, list)) else 0
                    logger.info(f"✅ Stored currency_indices enriched_df with {num_rows} records (memory optimized)")
                    
                    # Persist currency_indices config snapshot for step_configs chain
                    try:
                        _ci_config_snapshot = {}
                        if config is not None:
                            if isinstance(config, dict):
                                _ci_config_snapshot = dict(config)
                            elif hasattr(config, "dict"):
                                _ci_config_snapshot = config.dict()
                            elif hasattr(config, "__dict__"):
                                _ci_config_snapshot = {k: v for k, v in config.__dict__.items() if not k.startswith("_")}
                        # ── Fallback removed: self.config is current_data (a DataFrame).
                        # bool(DataFrame) raises "ambiguous truth value" — never use it
                        # in a boolean context. The config parameter is always the
                        # authoritative CurrencyIndexConfig passed from the caller.
                        # If config is None here, store an empty snapshot (non-fatal).
                        await store_session_step_result(
                            session_id=session_id,
                            step_name="currency_indices_config",
                            data={"config": _ci_config_snapshot},
                            db=db,
                        )
                        logger.info(f"[CI] Persisted currency_indices config snapshot ({len(_ci_config_snapshot)} keys)")

                        # ── If TI was requested for indices, also store the ti_config
                        # used so InferenceFeaturePipeline can reproduce exact TA settings.
                        _ti_enabled_dict = _ci_config_snapshot.get("calculate_ti_for_indices") or {}
                        _ti_requested = any(v for v in _ti_enabled_dict.values())
                        if _ti_requested:
                            _ti_config_snapshot = _ci_config_snapshot.get("ti_config") or {}
                            if _ti_config_snapshot:
                                await store_session_step_result(
                                    session_id=session_id,
                                    step_name="currency_indices_ti_config",
                                    data={"config": _ti_config_snapshot, "enabled_indices": _ti_enabled_dict},
                                    db=db,
                                )
                                logger.info(
                                    f"[CI] Persisted currency_indices_ti_config snapshot "
                                    f"({len(_ti_config_snapshot)} keys) for indices: "
                                    f"{[k for k, v in _ti_enabled_dict.items() if v]}"
                                )
                    except Exception as _cfg_err:
                        logger.warning(f"[CI] Could not persist config snapshot: {_cfg_err}")
                    
                    # Populate cache with fresh TI-enriched results
                    await self.cache_session_data(
                        session_id=session_id,
                        data=df_to_store,
                        source_step='currency_indices',
                        ttl_seconds=1800
                    )
                    logger.info(f"Cached {num_rows} rows after currency indices analysis")
                    
                    # ✅ UPDATE self.current_data with enriched DataFrame (final step, but keeps it consistent)
                    self.current_data = df_to_store if isinstance(df_to_store, pd.DataFrame) else pd.DataFrame(df_to_store)
                    self.current_session_id = session_id
                    logger.info(f"Updated self.current_data with {len(self.current_data)} enriched rows (final step)")
                else:
                    # 🔴 BUG FIX: slice_streaming stores internally BUT we must load the FULLY MERGED result
                    # PM's _aggregate_slice_results() returns complete merged DataFrame in result["result_df"]
                    df_merged = result.get("result_df")
                    if not isinstance(df_merged, pd.DataFrame):
                        df_merged = IntermediateResultsCache.retrieve(task_id, "currency_indices")
                    
                    if isinstance(df_merged, pd.DataFrame) and len(df_merged.columns) >= len(df.columns):
                        self.current_data = df_merged  # Reference, not copy!
                        self.current_session_id = session_id
                        
                        mem_mb = df_merged.memory_usage(deep=True).sum() / (1024 * 1024)
                        logger.info(
                            f"✅ [MEMORY OPTIMIZED] Loaded FULLY MERGED currency indices result (reference): "
                            f"{len(self.current_data)} rows, {mem_mb:.1f} MB "
                            f"(from {len(result.get('metadata', {}).get('slices', [])) if 'slices' in result.get('metadata', {}) else '?'} slices)"
                        )
                    else:
                        logger.warning(f"⚠️ Could not extract merged result_df from slice_streaming — clearing TIER 0a pointer to force DB load")
                        if self.current_session_id == session_id:
                            self.current_data = None
                    
                    await set_as_current_data(session_id, db, task_id)
        except Exception as db_err:
            logger.warning(f"Could not persist currency indices analysis result: {db_err}")
        return result

    # ────────────────────────────────────────────────────────────────
    # TI CALCULATION FOR CURRENCY INDICES
    # ────────────────────────────────────────────────────────────────

    async def execute_ti_for_currency_indices(
        self,
        result_df: pd.DataFrame,
        config: Any,
        pm,                    # ProcessingManager
        session_id: str,
        task_id: str,
        user_id: str = "anonymous",
    ) -> pd.DataFrame:
        """
        Calculate Technical Indicators for currency indices using ProcessingManager.
 
        Replaces the original method in AnalysisManager verbatim except for the
        DatetimeIndex fix applied in STEP 2 (marked ── FIX ──).
        """
        if not config:
            logger.info("[Currency Indices TI] No config provided, skipping TI calculation")
            return result_df
 
        # Handle both dict and object configs
        ti_enabled_dict = None
        if isinstance(config, dict):
            ti_enabled_dict = config.get("calculate_ti_for_indices")
        else:
            ti_enabled_dict = getattr(config, "calculate_ti_for_indices", None)
 
        if not ti_enabled_dict:
            logger.info("[Currency Indices TI] No TI calculation requested")
            return result_df
 
        ti_enabled_indices = [idx for idx, enabled in ti_enabled_dict.items() if enabled]
        if not ti_enabled_indices:
            logger.info("[Currency Indices TI] No indices have TI enabled")
            return result_df
 
        logger.info(f"[Currency Indices TI] Processing TI for: {ti_enabled_indices}")

        # Snapshot columns present before any TI is added — used by the
        # post-loop sanity check to find only newly-added columns.
        _cols_before_ti: set = set(result_df.columns)

        ti_calculated_count = 0
        for idx_name in ti_enabled_indices:
            try:
                # ── STEP 1: Locate index OHLCV columns ─────────────────────────
                ohlcv_cols = {
                    "open":   f"{idx_name}_open",
                    "high":   f"{idx_name}_high",
                    "low":    f"{idx_name}_low",
                    "close":  f"{idx_name}_close",
                    "volume": f"{idx_name}_tick_volume",
                }
 
                available_volume_col = None
                if f"{idx_name}_tick_volume" in result_df.columns:
                    available_volume_col = f"{idx_name}_tick_volume"
                elif f"{idx_name}_real_volume" in result_df.columns:
                    available_volume_col = f"{idx_name}_real_volume"
 
                missing_cols = [
                    col
                    for col in [
                        ohlcv_cols["open"],
                        ohlcv_cols["high"],
                        ohlcv_cols["low"],
                        ohlcv_cols["close"],
                    ]
                    if col not in result_df.columns
                ]
 
                if missing_cols or not available_volume_col:
                    logger.warning(
                        "[Currency Indices TI] Skipping %s — missing cols: %s, volume: %s",
                        idx_name, missing_cols, available_volume_col,
                    )
                    continue
 
                # ── STEP 2: Build minimal OHLCV DataFrame for PM ───────────────
                #
                # ── FIX ──────────────────────────────────────────────────────────
                # The original code used `index=result_df.index` which is a plain
                # RangeIndex after the currency-indices parallel workers run.
                # calculate_all_indicators() sets a DatetimeIndex internally, so
                # the TI result has DatetimeIndex while ti_df (= original_df inside
                # _ensure_result_completeness) has RangeIndex.  When row counts also
                # differ (warmup rows dropped), the fallback reindex branch fires:
                #
                #   result_df.reindex(original_df.index)   # DatetimeIndex by RangeIndex
                #
                # → zero label matches → 100% NaN on every TI column.
                #
                # Fix: parse the Time column into a DatetimeIndex NOW so both sides
                # of _ensure_result_completeness carry DatetimeIndex and Case B
                # (label-based reindex) aligns them correctly.
                # ─────────────────────────────────────────────────────────────────
 
                # Resolve Time values (prefer explicit Time column over index)
                # Guard: if `result_df` already has TI columns (from a cached re-run),
                # the Time column may have been consumed into the DatetimeIndex.
                # Check for it explicitly before falling through to synthesis.
                if "Time" in result_df.columns:
                    raw_time = result_df["Time"]
                elif "time" in result_df.columns:
                    raw_time = result_df["time"]
                elif pd.api.types.is_datetime64_any_dtype(result_df.index):
                    raw_time = result_df.index.to_series()
                else:
                    raw_time = None

                # Sanity-check numeric raw_time: if values are too small to be Unix
                # timestamps (< year 2000 ≈ 946684800s), they are likely index values
                # (e.g. Dollar ~50, JPY ~100) that ended up in the Time column due to
                # column mis-alignment.  Fall back to the DatetimeIndex in that case.
                if raw_time is not None and pd.api.types.is_numeric_dtype(raw_time):
                    sample_val = float(raw_time.iloc[0]) if len(raw_time) > 0 else 0
                    if sample_val < 946684800:  # before year 2000 — almost certainly not a valid timestamp
                        logger.warning(
                            "[Currency Indices TI] %s: 'Time' column contains suspicious values "
                            "(sample=%.2f < 946684800). Falling back to DatetimeIndex.",
                            idx_name, sample_val,
                        )
                        if pd.api.types.is_datetime64_any_dtype(result_df.index):
                            raw_time = result_df.index.to_series()
                        else:
                            raw_time = None
 
                # Build the datetime index
                if raw_time is not None:
                    if pd.api.types.is_datetime64_any_dtype(raw_time):
                        dt_index = pd.DatetimeIndex(raw_time.values)
                    elif pd.api.types.is_numeric_dtype(raw_time):
                        dt_index = pd.to_datetime(raw_time.values, unit="s", utc=False)
                    else:
                        try:
                            dt_index = pd.to_datetime(raw_time.values)
                        except Exception:
                            dt_index = None
                else:
                    dt_index = None
 
                # Warn and fall back to RangeIndex if parsing failed
                if dt_index is None:
                    logger.warning(
                        "[Currency Indices TI] Could not build DatetimeIndex for %s — "
                        "falling back to RangeIndex (TI warmup rows may produce NaN)",
                        idx_name,
                    )
                    use_index = result_df.index
                else:
                    use_index = dt_index
 
                ti_df = pd.DataFrame(
                    {
                        "Open":   result_df[ohlcv_cols["open"]].values,
                        "High":   result_df[ohlcv_cols["high"]].values,
                        "Low":    result_df[ohlcv_cols["low"]].values,
                        "Close":  result_df[ohlcv_cols["close"]].values,
                        "Volume": result_df[available_volume_col].values,
                    },
                    index=use_index,   # ← DatetimeIndex, not RangeIndex
                )
 
                # Add Time column so technical handlers can detect timeframe.
                # We store Unix seconds so the handler can parse it if needed.
                if dt_index is not None:
                    ti_df["Time"] = dt_index.astype(np.int64) // 10 ** 9
                elif raw_time is not None and pd.api.types.is_numeric_dtype(raw_time):
                    ti_df["Time"] = raw_time.values
                else:
                    # Synthesise hourly timestamps as last resort
                    logger.warning(
                        "[Currency Indices TI] No Time column for %s, synthesising hourly timestamps",
                        idx_name,
                    )
                    start_ts = int(pd.Timestamp("2020-01-01").timestamp())
                    ti_df["Time"] = [start_ts + i * 3600 for i in range(len(ti_df))]
 
                logger.info(
                    "[Currency Indices TI] Processing %s: shape=%s, nulls=%d, index_type=%s",
                    idx_name, ti_df.shape, ti_df.isna().sum().sum(),
                    type(ti_df.index).__name__,
                )
 
                # ── STEP 3: Call PM with AnalysisType.TECHNICAL ────────────────
                logger.info("[Currency Indices TI] Calling PM for %s", idx_name)
 
                ti_config_dict = {}
                if isinstance(config, dict):
                    ti_config_dict = config.get("ti_config") or {}
                else:
                    ti_config_dict = getattr(config, "ti_config", None) or {}
 
                # Import here to avoid circular imports at module level
                from app.core.processing.processing_manager import ProcessingManager, AnalysisType, IntermediateResultsCache
                from app.core.config import TechnicalConfig
 
                try:
                    ti_technical_config = (
                        TechnicalConfig(**ti_config_dict) if ti_config_dict else TechnicalConfig()
                    )
                except (TypeError, ValueError) as cfg_err:
                    logger.warning(
                        "[Currency Indices TI] Could not build TechnicalConfig: %s, using defaults",
                        cfg_err,
                    )
                    ti_technical_config = TechnicalConfig()
 
                ti_pm = ProcessingManager(
                    session_id=session_id,
                    task_id=task_id,
                    analysis_type=AnalysisType.TECHNICAL,
                    config=ti_technical_config,
                    task_store=pm.task_store,
                    connection_manager=pm.connection_manager,
                    processing_config=pm.processing_config,
                    user_id=user_id,
                )
 
                pm_result = await ti_pm.execute(ti_df)
 
                # ── STEP 4: Extract result DataFrame ───────────────────────────
                enriched_df = pm_result.get("result_df")
                if enriched_df is None:
                    enriched_df = pm_result.get("features_df")
 
                if not isinstance(enriched_df, pd.DataFrame):
                    cached_df = IntermediateResultsCache.retrieve(
                        task_id, f"technical_analysis__{idx_name}"
                    )
                    if isinstance(cached_df, pd.DataFrame):
                        enriched_df = cached_df
                        logger.info(
                            "[Currency Indices TI] Recovered %s TI from TIER 1 cache", idx_name
                        )
                    else:
                        logger.error(
                            "[Currency Indices TI] PM returned invalid result for %s, skipping",
                            idx_name,
                        )
                        continue
 
                logger.info(
                    "[Currency Indices TI] PM result for %s: shape=%s",
                    idx_name, enriched_df.shape,
                )
 
                # ── STEP 5: Identify TI columns (exclude base OHLCV + Time) ────
                #
                # BASE_OHLCV_COLS — exact names that must NEVER be prefixed or
                # mutated.  These are the canonical OHLCV columns of the result_df
                # and touching them would corrupt the base-pair price data.
                BASE_OHLCV_COLS = {
                    "Open", "High", "Low", "Close", "Volume", "Time", "time",
                    "TickVolume", "Spread", "RealVolume", "real_volume",
                }
                # Also exclude by lowercase so aliases ('open', 'tick_volume', …)
                # added by _add_legacy_aliases inside the inner TI run are caught.
                BASE_COL_LOWER = {c.lower() for c in BASE_OHLCV_COLS} | {
                    "tick_volume", "real_volume",
                    # Lowercase aliases produced by _add_legacy_aliases — must never
                    # be prefixed or they will collide with base-pair columns.
                    "is_up_bar", "is_down_bar",
                    "price_velocity_bull", "price_velocity_bear", "price_velocity_net",
                }

                # ── Deduplicate enriched_df first ──────────────────────────────
                # _add_legacy_aliases may have added 'open' alongside 'Open', then
                # _ensure_result_completeness renames 'open'→'Open' creating two
                # identically-named columns.  Deduplicate by exact name here so
                # enriched_df[col] always returns a Series, never a DataFrame.
                if enriched_df.columns.duplicated().any():
                    enriched_df = enriched_df.loc[
                        :, ~enriched_df.columns.duplicated(keep="first")
                    ].copy()
                    logger.debug(
                        "[Currency Indices TI] Deduplicated enriched_df for %s → %d cols",
                        idx_name, len(enriched_df.columns),
                    )

                ti_cols = [
                    c for c in enriched_df.columns
                    if c not in BASE_OHLCV_COLS           # exact guard — never touch OHLCV
                    and c.lower() not in BASE_COL_LOWER   # alias guard — 'open', 'time', etc.
                ]
                logger.info(
                    "[Currency Indices TI] Found %d TI columns for %s", len(ti_cols), idx_name
                )

                # ── STEP 6: Merge TI columns into result_df with index prefix ──
                #
                # ALL columns that survived step 5 are prefixed with {idx_name}_.
                # This guarantees no bare indicator name (e.g. Price_Velocity_Bull)
                # leaks into result_df and collides with the base-pair TI step.
                #
                # enriched_df has a DatetimeIndex; result_df has a RangeIndex.
                # We align by position using .values so there are no index-label
                # mismatches.  _ensure_result_completeness already NaN-padded any
                # warmup rows to the full 6518-row length, so .values is always
                # length-safe here.
                valid_ti_count = 0
                for col in ti_cols:
                    try:
                        series = enriched_df[col]

                        # Guard: if a dupe somehow survived, take only the first column
                        if isinstance(series, pd.DataFrame):
                            logger.warning(
                                "[Currency Indices TI] Column '%s' returned a DataFrame "
                                "(duplicate names) — taking first occurrence only.",
                                col,
                            )
                            series = series.iloc[:, 0]

                        nan_count = int(series.isna().sum())

                        if nan_count < len(series):
                            numeric_series = pd.to_numeric(series, errors="coerce").astype("float64")
                            prefixed = f"{idx_name}_{col}"

                            # Use .values to bypass any index-label alignment
                            # (enriched_df is DatetimeIndex; result_df is RangeIndex)
                            result_df[prefixed] = numeric_series.values

                            valid_ti_count += 1
                            if valid_ti_count <= 5:
                                pct = (nan_count / len(series)) * 100
                                logger.info(
                                    "[Currency Indices TI]   ✓ %s: %d/%d NaN (%.1f%%)",
                                    prefixed, nan_count, len(series), pct,
                                )
                        else:
                            logger.warning(
                                "[Currency Indices TI]   ✗ %s_%s: 100%% NaN (skipped)",
                                idx_name, col,
                            )
                    except Exception as col_err:
                        logger.warning(
                            "[Currency Indices TI]   ⚠ Failed to add %s: %s", col, col_err
                        )
 
                logger.info(
                    "[Currency Indices TI] Added %d/%d valid TI columns for %s",
                    valid_ti_count, len(ti_cols), idx_name,
                )
                ti_calculated_count += 1
 
            except Exception as exc:
                logger.error(
                    "[Currency Indices TI] Failed for %s: %s", idx_name, exc, exc_info=True
                )
 
        logger.info(
            "[Currency Indices TI] Completed: %d/%d indices processed, %d total columns",
            ti_calculated_count, len(ti_enabled_indices), len(result_df.columns),
        )

        # ── Sanity check: verify every newly-added column carries an index prefix ──
        # Compare against the snapshot taken at entry to find columns added by this fn.
        # Legitimate OHLCV columns (BASE_OHLCV_COLS) never carry a prefix — exclude them.
        BASE_OHLCV_COLS = {
            "Open", "High", "Low", "Close", "Volume", "Time", "time",
            "TickVolume", "Spread", "RealVolume", "real_volume",
        }
        known_prefixes = tuple(f"{idx}_" for idx in ti_enabled_indices)
        new_cols_added = [c for c in result_df.columns if c not in _cols_before_ti]
        unprefixed_new = [
            c for c in new_cols_added
            if c not in BASE_OHLCV_COLS
            and not any(c.startswith(p) for p in known_prefixes)
        ]
        if unprefixed_new:
            logger.warning(
                "[Currency Indices TI] ⚠ %d newly-added column(s) are missing an index prefix — "
                "they may collide with base-pair TI columns in the next step: %s",
                len(unprefixed_new), unprefixed_new[:20],
            )
        else:
            logger.info(
                "[Currency Indices TI] ✅ All %d newly-added columns carry an index prefix.",
                len(new_cols_added),
            )

        # ── Final dedup guard: ensure result_df has no duplicate column labels ──
        # Duplicate labels cause "cannot reindex on an axis with duplicate labels"
        # in every subsequent arithmetic step (TA, regime scoring, etc.) because
        # pandas tries to align Series by label during binary ops.  This can happen
        # when the inner TI run produces lowercase aliases ('open', 'close') that
        # slip through BASE_OHLCV_COLS and get merged as prefixed duplicates of
        # already-existing columns, or if this step is re-run on cached data that
        # already carries the Dollar_* columns from a prior run.
        if result_df.columns.duplicated().any():
            dup_mask = result_df.columns.duplicated(keep=False)
            dup_names = result_df.columns[dup_mask].unique().tolist()
            logger.warning(
                "[Currency Indices TI] ⚠ result_df has %d duplicate column name(s) "
                "before storage — deduplicating (keep first): %s",
                len(dup_names), dup_names[:20],
            )
            result_df = result_df.loc[:, ~result_df.columns.duplicated(keep="first")].copy()
            logger.info(
                "[Currency Indices TI] result_df after dedup: %d columns",
                len(result_df.columns),
            )

        logger.info(
            "[Currency Indices] TI post-processing complete: %d total columns",
            len(result_df.columns),
        )

        return result_df