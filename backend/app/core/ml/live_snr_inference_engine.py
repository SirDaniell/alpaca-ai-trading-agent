"""
Live SNR Inference Engine
Replicates the exact S&R zone proximity trigger used during model training.
Fires AXE model inference ONLY when price enters the zone_width proximity (1%) of a detected cluster zone.
"""

import logging
import os
import asyncio
import concurrent.futures as _cf
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from pydantic import BaseModel

from app.core.analysis.support_resistance import (
    detect_snr_levels_sequential,
    create_clustered_zones_sequential,
    extract_snr_features,
)
from app.core.analysis.technical_indicators import TechnicalIndicators
from app.services.mt5_service import MT5Service

# Dedicated executor for CPU-bound TI pipeline work inside the SNR engine.
# Keeps the async event loop unblocked during 2-10s pandas/numpy calculations.
_snr_ti_executor = _cf.ThreadPoolExecutor(max_workers=4, thread_name_prefix="snr_ti_worker")

logger = logging.getLogger(__name__)

DEBUG_CSV_EXPORT_ENABLED = os.getenv("ENABLE_LIVE_INFERENCE_DEBUG_CSV", "0").lower() in {"1", "true", "yes", "on"}

def _normalize_bar(b: Any) -> Dict[str, Any]:
    """Convert a bar (dict, list, or tuple) into a standardized dict."""
    if isinstance(b, dict):
        ts = b.get("timestamp") or b.get("time") or b.get("Time")
        o = b.get("open") if b.get("open") is not None else b.get("Open", 0.0)
        h = b.get("high") if b.get("high") is not None else b.get("High", 0.0)
        l = b.get("low") if b.get("low") is not None else b.get("Low", 0.0)
        c = b.get("close") if b.get("close") is not None else b.get("Close", 0.0)
        v = b.get("volume") if b.get("volume") is not None else b.get("Volume", b.get("tick_volume", 0.0))
        res = {
            "Time": ts,
            "Open": float(o or 0.0),
            "High": float(h or 0.0),
            "Low": float(l or 0.0),
            "Close": float(c or 0.0),
            "Volume": float(v or 0.0),
        }
        if b.get("tick_volume") is not None or b.get("TickVolume") is not None:
            res["TickVolume"] = float(b.get("tick_volume") if b.get("tick_volume") is not None else b.get("TickVolume"))
        if b.get("spread") is not None or b.get("Spread") is not None:
            res["Spread"] = float(b.get("spread") if b.get("spread") is not None else b.get("Spread"))
        if b.get("real_volume") is not None or b.get("RealVolume") is not None:
            res["RealVolume"] = float(b.get("real_volume") if b.get("real_volume") is not None else b.get("RealVolume"))
        return res
    elif isinstance(b, (list, tuple)):
        # Standard MT5 array format: [time, open, high, low, close, tick_volume, spread, real_volume]
        ts = b[0] if len(b) > 0 else None
        o = b[1] if len(b) > 1 else 0.0
        h = b[2] if len(b) > 2 else 0.0
        l = b[3] if len(b) > 3 else 0.0
        c = b[4] if len(b) > 4 else 0.0
        v = b[5] if len(b) > 5 else 0.0
        res = {
            "Time": ts,
            "Open": float(o or 0.0),
            "High": float(h or 0.0),
            "Low": float(l or 0.0),
            "Close": float(c or 0.0),
            "Volume": float(v or 0.0),
            "TickVolume": float(v or 0.0),
        }
        if len(b) > 6:
            res["Spread"] = float(b[6] or 0.0)
        if len(b) > 7:
            res["RealVolume"] = float(b[7] or 0.0)
        return res
    else:
        return {"Time": None, "Open": 0.0, "High": 0.0, "Low": 0.0, "Close": 0.0, "Volume": 0.0}


class LiveInferenceResult(BaseModel):
    triggered: bool
    reason: str
    symbol: str
    timeframe: str
    model_id: str
    current_price: float
    nearest_zone_price: Optional[float] = None
    distance_pct: Optional[float] = None
    snr_features: Optional[Dict[str, float]] = None
    zones: List[Dict[str, Any]] = []
    predictions: Optional[Dict[str, Any]] = None
    prediction_error: Optional[str] = None
    missing_features: List[str] = []
    pre_ml_snapshot_path: Optional[str] = None


class LiveSNRInferenceEngine:
    """
    Live SNR Inference Engine
    Replicates the exact S&R zone proximity trigger used during model training.
    Fires AXE model inference ONLY when price enters the zone_width proximity (1%) of a detected cluster zone.
    
    Training Trigger Replication:
      - Looks back 50 bars to detect S&R levels and K-means clustered zones.
      - Calculates distance_pct = abs(current_price - zone_price) / zone_price.
      - If distance_pct <= zone_width (default 0.01 / 1%):
    """

    def __init__(
        self,
        zone_width: float = 0.01,
        lookback_period: int = 50,
        min_distance_pct: float = 0.02,
        n_clusters: int = 5,
        min_bars_required: int = 200,   # need enough history for meaningful S&R
        model_input_window: int = 90,   # must match training sequence length
        mt5_service: Optional[MT5Service] = None,  # fresh bar fetcher
    ):
        self.zone_width = zone_width
        self.lookback_period = lookback_period
        self.min_distance_pct = min_distance_pct
        self.n_clusters = n_clusters
        self.min_bars_required = min_bars_required
        self.model_input_window = model_input_window
        self.ti_calc = TechnicalIndicators()
        self.mt5_service = mt5_service
        self._locks: Dict[str, asyncio.Lock] = {}
        self._zone_cache: Dict[str, Dict[str, Any]] = {}

    def _get_lock(self, key: str) -> asyncio.Lock:
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    def _save_pre_ml_snapshot(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Optional[str]:
        """Persist the enriched SNR result just before ML prep/scaling for debugging.

        CSV exports are disabled by default to avoid bloating disk space; re-enable by
        setting ENABLE_LIVE_INFERENCE_DEBUG_CSV=1 in the environment.
        """
        if not DEBUG_CSV_EXPORT_ENABLED:
            return None

        if df is None or getattr(df, "empty", True):
            return None

        output_root = Path(__file__).resolve().parents[3] / "Backend" / "artifacts" / "live_inference_debug"
        output_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        file_name = f"{symbol}_{timeframe}_{timestamp}_pre_ml_snapshot.csv"
        path = output_root / file_name

        try:
            df.to_csv(path, index=False)
            logger.info("[LiveSNR] Saved pre-ML SNR snapshot: %s (%d rows x %d cols)", path, len(df), len(df.columns))
            return str(path)
        except Exception as exc:
            logger.warning("[LiveSNR] Failed to save pre-ML SNR snapshot: %s", exc)
            return None

    async def evaluate_and_predict(
        self,
        model_id: str,
        symbol: str,
        timeframe: str,
        ohlcv_bars: List[Dict[str, Any]],
        supporting_ohlcv: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        predict_fn: Optional[Any] = None,
        force: bool = False,
        mt5_service: Optional[MT5Service] = None,
        broker_symbol: Optional[str] = None,
    ) -> LiveInferenceResult:
        """
        Evaluate live OHLCV bars against training S&R zone trigger logic.
        Fetches fresh 1000 raw bars from MT5 if available, falls back to frontend bars.
        """
        lock_key = f"{symbol}:{timeframe}"
        async with self._get_lock(lock_key):
            # ── Attempt to fetch fresh 1000 bars from MT5 (INFERENCE_BAR_COUNT) ───
            # Training used fresh raw bars, not pre-enriched frontend snapshots.
            # This fixes the duplicate-column crash when frontend sends 200-bar enriched data.
            INFERENCE_BAR_COUNT = 1000
            fresh_bars = None
            mt5_source = mt5_service or self.mt5_service
            fetch_symbol = broker_symbol or symbol

            if mt5_source:
                try:
                    logger.info(f"[LiveSNR] Attempting to fetch {INFERENCE_BAR_COUNT} fresh bars from MT5 for {fetch_symbol}...")
                    ohlc_data = await mt5_source.fetch_ohlc_data_v2(
                        symbol=fetch_symbol,
                        timeframe=timeframe,
                        count=INFERENCE_BAR_COUNT,
                    )

                    # Unwrap if nested
                    if isinstance(ohlc_data, dict) and "data" in ohlc_data:
                        ohlc_data = ohlc_data["data"]

                    if isinstance(ohlc_data, list) and ohlc_data:
                        fresh_bars = ohlc_data
                        logger.info(f"[LiveSNR] ✓ Fetched {len(fresh_bars)} fresh bars from MT5 for {fetch_symbol}")
                    elif fetch_symbol != symbol:
                        # Fallback attempt using canonical catalog symbol if primary broker symbol returned error/empty
                        logger.info(f"[LiveSNR] Primary broker symbol '{fetch_symbol}' returned no bars, trying fallback symbol '{symbol}'...")
                        fallback_data = await mt5_source.fetch_ohlc_data_v2(
                            symbol=symbol,
                            timeframe=timeframe,
                            count=INFERENCE_BAR_COUNT,
                        )
                        if isinstance(fallback_data, dict) and "data" in fallback_data:
                            fallback_data = fallback_data["data"]
                        if isinstance(fallback_data, list) and fallback_data:
                            fresh_bars = fallback_data
                            logger.info(f"[LiveSNR] ✓ Fetched {len(fresh_bars)} fresh bars from MT5 using fallback symbol '{symbol}'")
                except Exception as e:
                    logger.warning(f"[LiveSNR] MT5 fetch failed for {fetch_symbol}, falling back to frontend bars: {e}")
            
            # Use fresh bars if available, otherwise fall back to frontend bars
            raw_bars = fresh_bars if fresh_bars else ohlcv_bars
            normalized_bars = [_normalize_bar(b) for b in raw_bars]
            
            if len(normalized_bars) < self.min_bars_required:
                return LiveInferenceResult(
                    triggered=False,
                    reason=(
                        f"Insufficient OHLCV bars ({len(normalized_bars)} < {self.min_bars_required}). "
                        f"Need {self.min_bars_required}+ bars for reliable S&R detection."
                    ),
                    symbol=symbol,
                    timeframe=timeframe,
                    model_id=model_id,
                    current_price=float(normalized_bars[-1]["Close"]) if normalized_bars else 0.0,
                )

            last_bar_ts = normalized_bars[-1]["Time"]
            last_bar_close = float(normalized_bars[-1]["Close"])
            last_bar_high = float(normalized_bars[-1]["High"])
            last_bar_low = float(normalized_bars[-1]["Low"])
            bar_count = len(normalized_bars)

            cached = self._zone_cache.get(lock_key)
            if (
                cached
                and cached.get("bar_count") == bar_count
                and cached.get("last_bar_ts") == last_bar_ts
                and cached.get("last_bar_close") == last_bar_close
                and cached.get("last_bar_high") == last_bar_high
                and cached.get("last_bar_low") == last_bar_low
            ):

                logger.info(
                    "[LiveSNR] ⚡ Cache hit for %s/%s (%d bars) — reusing computed zones & TI features",
                    symbol,
                    timeframe,
                    bar_count,
                )
                df_full = cached["df_full"]
                # Restore raw_df (pre-TI) from cache; fall back to rebuilding from
                # normalized_bars if this is an old cache entry without the raw_df key.
                _cached_raw_df = cached.get("raw_df")
                raw_df = _cached_raw_df if _cached_raw_df is not None else pd.DataFrame(normalized_bars)
                levels = cached["levels"]
                zones = cached["zones"]
                serialized_zones = cached["serialized_zones"]
                snr_feats = cached["snr_feats"]
                current_price = float(df_full["Close"].iloc[-1])
            else:
                # Build pandas DataFrame across ALL received bars for full S&R detection.
                # Training collected signals on the entire historical dataset before slicing
                # the model input window — live inference must replicate this.
                df_full = pd.DataFrame(normalized_bars)

                # ── Keep raw OHLCV copy BEFORE TI enrichment ────────────────────────
                # InferenceFeaturePipeline.build_feature_window expects raw OHLCV — it
                # runs its own full pipeline internally:
                #   Step 1: run_currency_indices (raw 7 cols → ~300+ prefixed cols)
                #   Step 2: run_technical_analysis (enriched frame → >600 cols total)
                # Passing the TI-enriched df_full here would cause a second TI run on
                # a ~346-column DatetimeIndex frame, crashing with:
                #   ValueError: cannot reindex on an axis with duplicate labels
                #   ValueError: Data must be 1-dimensional, got ndarray of shape (1000, 2)
                _raw_cols = ["Time", "Open", "High", "Low", "Close", "Volume"]
                raw_df = df_full[[c for c in _raw_cols if c in df_full.columns]].copy()

                # Calculate all TI features for SNR zone detection only.
                # df_full (TI-enriched) is used ONLY for S&R zone detection below.
                # raw_df (pre-TI) is passed to predict_fn (InferenceFeaturePipeline).
                try:
                    loop = asyncio.get_running_loop()
                    df_full = await loop.run_in_executor(
                        _snr_ti_executor,
                        lambda: self.ti_calc.calculate_all_indicators(df_full, mode="inference"),
                    )
                except Exception as e:
                    logger.warning(f"Technical indicator calculation warning during live inference: {e}")

                current_index = len(df_full) - 1
                current_price = float(df_full["Close"].iloc[current_index])

                # 1. Detect S&R levels across the FULL history up to the current bar.
                #    lookback_period here is the rolling window used inside the sequential
                #    detector — running it at the final index spans the entire dataset.
                levels = detect_snr_levels_sequential(
                    df_full,
                    current_index,
                    self.lookback_period,
                    self.min_distance_pct,
                )

                # 2. Cluster levels into zones using the recent lookback slice for
                #    price-range context (zone width is still relative to current price).
                price_slice = df_full.iloc[max(0, current_index - self.lookback_period): current_index + 1]
                zones = create_clustered_zones_sequential(
                    levels,
                    price_slice,
                    n_clusters=self.n_clusters,
                    zone_width=self.zone_width,
                )
                serialized_zones = [
                    {
                        "id": int(zone_id),
                        "price": float(zone_price),
                        "levels": [list(level) for level in zone_levels],
                        "volume": {
                            key: float(value) if isinstance(value, (np.floating, np.integer)) else value
                            for key, value in zone_volume.items()
                        },
                    }
                    for zone_id, zone_price, zone_levels, zone_volume in zones
                ]
                logger.info(
                    "[LiveSNR] %s/%s produced %d zones: %s",
                    symbol,
                    timeframe,
                    len(serialized_zones),
                    [
                        {
                            "id": zone["id"],
                            "price": round(zone["price"], 5),
                            "up": zone["volume"].get("up_volume", 0),
                            "down": zone["volume"].get("down_volume", 0),
                        }
                        for zone in serialized_zones
                    ],
                )

                # 3. Extract SNR features from full-history levels
                snr_feats = extract_snr_features(current_price, levels, zones)

                # Store in cache for subsequent calls on identical bar window.
                # raw_df (pre-TI) is stored so cache-hit calls also pass clean OHLCV
                # to predict_fn without re-building from normalized_bars.
                self._zone_cache[lock_key] = {
                    "bar_count": bar_count,
                    "last_bar_ts": last_bar_ts,
                    "last_bar_close": last_bar_close,
                    "last_bar_high": last_bar_high,
                    "last_bar_low": last_bar_low,
                    "df_full": df_full,
                    "raw_df": raw_df,
                    "levels": levels,
                    "zones": zones,
                    "serialized_zones": serialized_zones,
                    "snr_feats": snr_feats,
                }


            # 4. Check proximity to any S&R zone
            triggered_zone = None
            min_dist_pct = float("inf")

            for zone_data in zones:
                # zone_data format: (cluster_id, zone_price, cluster_levels, volume_data)
                zone_price = float(zone_data[1])
                dist_pct = abs(current_price - zone_price) / zone_price
                if dist_pct < min_dist_pct:
                    min_dist_pct = dist_pct
                    triggered_zone = zone_price

            is_triggered = force or (min_dist_pct <= self.zone_width)

            # NOTE: We always run inference for display purposes.
            # The `triggered` flag tells the frontend whether price is AT a zone
            # (which is the training condition), but predictions are always computed
            # so the chart overlay is always visible to the user.
            # The frontend can use triggered=False to show a "Not at zone" indicator.

            # 5. Run inference if triggered or forced.
            #    Pass the full enriched window to the selected runtime so any
            #    secondary feature families can be computed before the final
            #    model-input slice is selected.
            predictions = None
            prediction_error = None
            missing_features: List[str] = []
            pre_ml_snapshot_path = self._save_pre_ml_snapshot(df_full.copy(), symbol, timeframe)

            if predict_fn:
                try:
                    predictions = await predict_fn(
                        model_id,
                        symbol,
                        timeframe,
                        raw_df.copy(),  # raw OHLCV — InferenceFeaturePipeline runs TI internally
                        snr_feats,
                        supporting_ohlcv or {},
                    )
                except Exception as e:
                    if hasattr(e, "missing_features"):
                        logger.warning("[LiveSNR] Selected-model contract unmet: %s", e)
                    else:
                        logger.error(f"Inference execution error: {e}", exc_info=True)
                    prediction_error = str(e)
                    missing_features = list(getattr(e, "missing_features", []) or [])

            reason_str = (
                "Forced manual inference" if force
                else f"S&R Zone proximity triggered ({min_dist_pct*100:.2f}% <= {self.zone_width*100:.1f}%)"
                if is_triggered
                else f"Price {current_price:.2f} is {min_dist_pct*100:.2f}% from nearest zone — inference run for display"
            )

            return LiveInferenceResult(
                triggered=is_triggered,
                reason=reason_str,
                symbol=symbol,
                timeframe=timeframe,
                model_id=model_id,
                current_price=current_price,
                nearest_zone_price=triggered_zone,
                distance_pct=min_dist_pct if min_dist_pct != float("inf") else None,
                snr_features=snr_feats,
                zones=serialized_zones,
                predictions=predictions,
                prediction_error=prediction_error,
                missing_features=missing_features,
                pre_ml_snapshot_path=pre_ml_snapshot_path,
            )


# Global singleton engine instance
live_snr_engine = LiveSNRInferenceEngine(
    zone_width=0.01,          # 1% proximity (matches training defaults)
    lookback_period=50,       # rolling window for S&R level detection per bar
    min_distance_pct=0.02,
    n_clusters=5,
    min_bars_required=200,    # minimum history to compute meaningful S&R zones
    model_input_window=90,    # matches training [90, 655] input_shape exactly
)
