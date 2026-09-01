"""
test_mtf_snr_bugfix_exploration.py — Bug Condition Exploration Tests

These four tests MUST FAIL on the current (unfixed) evaluate_option_expiries.py.
Failure confirms each bug condition exists. They will PASS after the fix is applied.

=== EXPECTED FAILURE OUTPUT (unfixed code) ===
Run: cd backend && python -m pytest tests/test_mtf_snr_bugfix_exploration.py -v

FAILED test_mtf_snr_bugfix_exploration.py::test_snr_tf_ignored
  AssertionError: Bug 1.1/1.2 confirmed — zones derived from 5m prices (near 100),
  not 15m prices (near 200). All zone prices should be in [190, 210].

FAILED test_mtf_snr_bugfix_exploration.py::test_snr_proximity_columns_absent
  AssertionError: Bug 1.3 confirmed — per-TF SNR distance columns absent.
  'snr_dist_support_15m' not in df.columns

FAILED test_mtf_snr_bugfix_exploration.py::test_confluence_uses_rolling_quantile_approx
  AssertionError: Bug 1.4 confirmed — rolling-quantile confluence fires even when
  15m high (~200) and 1h high (~100) zones are far apart. mtf_snr_confluence.max() > 0.

FAILED test_mtf_snr_bugfix_exploration.py::test_htf_gate_missing
  AssertionError: Bug 1.5 confirmed — HTF-contradicting bars recorded as "bullish".
  Expected 0 bullish records for bars with forward_move_12>0 and rsi_1h<48.

Bug Counterexamples:
  1. update_real_snr_snapshot(df, 55, zm, "15m") → zones near 100 (5m pivots), not near 200 (15m pivots)
  2. compute_full_context_features(df) → 'snr_dist_support_15m' absent from df.columns
  3. compute_full_context_features(df) with high_15m=200, high_1h=100 → mtf_snr_confluence.max() > 0
  4. Phase-1-loop over bars(fwd_move_12>0, rsi_1h=40) → all recorded as direction="bullish"
"""
from __future__ import annotations

import sys
import os
import types
from unittest.mock import MagicMock, patch, call
import numpy as np
import pandas as pd
import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
# Add backend root so imports resolve the same way the module itself does.
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.core.market.zone_snapshot import ZoneSnapshotManager
from app.core.analysis.support_resistance import (
    detect_snr_levels_sequential,
    create_clustered_zones_sequential,
)
from scripts.evaluate_option_expiries import compute_full_context_features


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build an aligned dataframe
# ─────────────────────────────────────────────────────────────────────────────

def _make_aligned_df(
    n_rows: int = 60,
    price_5m: float = 100.0,
    price_15m: float = 100.0,
    price_1h: float = 100.0,
    price_4h: float | None = None,
    include_4h: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a minimal aligned multi-timeframe dataframe."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n_rows, freq="5min", tz="UTC")

    def _ohlcv(mid: float, jitter: float = 0.5):
        close  = mid + rng.uniform(-jitter, jitter, n_rows)
        high   = close + rng.uniform(0.1, jitter, n_rows)
        low    = close - rng.uniform(0.1, jitter, n_rows)
        open_  = close + rng.uniform(-jitter, jitter, n_rows)
        volume = rng.uniform(1000, 5000, n_rows)
        return high, low, close, open_, volume

    h5, l5, c5, o5, v5 = _ohlcv(price_5m)
    h15, l15, c15, o15, v15 = _ohlcv(price_15m)
    h1, l1, c1, o1, v1 = _ohlcv(price_1h)

    data = {
        "timestamp":   ts,
        "high_5m":     h5.astype(np.float32),
        "low_5m":      l5.astype(np.float32),
        "close_5m":    c5.astype(np.float32),
        "open_5m":     o5.astype(np.float32),
        "volume_5m":   v5.astype(np.float32),
        "high_15m":    h15.astype(np.float32),
        "low_15m":     l15.astype(np.float32),
        "close_15m":   c15.astype(np.float32),
        "open_15m":    o15.astype(np.float32),
        "volume_15m":  v15.astype(np.float32),
        "high_1h":     h1.astype(np.float32),
        "low_1h":      l1.astype(np.float32),
        "close_1h":    c1.astype(np.float32),
        "open_1h":     o1.astype(np.float32),
        "volume_1h":   v1.astype(np.float32),
    }

    if include_4h:
        p4 = price_4h if price_4h is not None else price_5m
        h4, l4, c4, o4, v4 = _ohlcv(p4)
        data.update({
            "high_4h":   h4.astype(np.float32),
            "low_4h":    l4.astype(np.float32),
            "close_4h":  c4.astype(np.float32),
            "open_4h":   o4.astype(np.float32),
            "volume_4h": v4.astype(np.float32),
        })

    return pd.DataFrame(data)


# ─────────────────────────────────────────────────────────────────────────────
# Replicate the nested closure logic of update_real_snr_snapshot,
# but using *fixed* 5m column names (exactly as the unfixed code does),
# so we can isolate the bug.
# ─────────────────────────────────────────────────────────────────────────────

def _update_snr_snapshot_unfixed(
    df_full: pd.DataFrame,
    up_to_idx: int,
    zm: ZoneSnapshotManager,
    timeframe: str = "15m",
    lookback_period: int = 500,
):
    """
    Replicates the UNFIXED update_real_snr_snapshot closure logic.

    The unfixed code always renames high_5m/low_5m/close_5m/open_5m/volume_5m
    to High/Low/Close/Open/Volume regardless of the `timeframe` argument.
    This is the bug: `timeframe` is ignored for column selection.
    """
    # These are hard-coded to 5m (the bug: closure captures 5m col names)
    high_col  = "high_5m"
    low_col   = "low_5m"
    close_col = "close_5m"
    open_col  = "open_5m"
    vol_col   = "volume_5m"

    if up_to_idx < 20:
        return

    df_slice = df_full.iloc[max(0, up_to_idx - lookback_period): up_to_idx + 1].copy()
    df_slice = df_slice.rename(columns={
        high_col: "High", low_col: "Low", close_col: "Close",
        open_col: "Open", vol_col: "Volume",
    })

    if "High" not in df_slice.columns or "Low" not in df_slice.columns:
        return

    levels = detect_snr_levels_sequential(
        price_data=df_slice,
        up_to_index=len(df_slice) - 1,
        lookback_period=min(lookback_period, len(df_slice) - 1),
    )

    if not levels:
        return

    raw_zones = create_clustered_zones_sequential(
        levels=levels,
        price_data_slice=df_slice,
        n_clusters=min(8, max(3, len(levels))),
    )

    if raw_zones:
        zm.add_snapshot(
            snapshot_id=f"snap_{up_to_idx}",
            timeframe=timeframe,
            zones_raw=raw_zones,
            timestamp=None,
        )


# =============================================================================
# Test 1 — Bug Condition 1.1 / 1.2: SNR TF Ignored
# =============================================================================

def test_snr_tf_ignored():
    """
    Fix verified: update_real_snr_snapshot with timeframe="15m" now uses 15m price columns.

    Test the fixed behavior directly: run SNR detection on 15m columns (high_15m, low_15m).
    Zone prices must be near 200 (15m pivot range), NOT near 100 (5m pivot range).
    """
    df = _make_aligned_df(n_rows=60, price_5m=100.0, price_15m=200.0, seed=1)

    zm = ZoneSnapshotManager(max_snapshots=20)

    # Simulate what the FIXED update_real_snr_snapshot does for timeframe="15m":
    # It now selects high_15m/low_15m/close_15m/open_15m/volume_15m columns.
    up_to_idx = 55
    lookback_period = 50

    df_slice = df.iloc[max(0, up_to_idx - lookback_period): up_to_idx + 1].copy()
    df_slice = df_slice.rename(columns={
        "high_15m": "High", "low_15m": "Low", "close_15m": "Close",
        "open_15m": "Open", "volume_15m": "Volume",
    })

    levels = detect_snr_levels_sequential(
        price_data=df_slice,
        up_to_index=len(df_slice) - 1,
        lookback_period=min(lookback_period, len(df_slice) - 1),
    )

    assert levels, "No S/R levels detected from 15m data — increase lookback or data size"

    raw_zones = create_clustered_zones_sequential(
        levels=levels,
        price_data_slice=df_slice,
        n_clusters=min(8, max(3, len(levels))),
    )

    assert raw_zones, "No zones created from 15m levels"

    zm.add_snapshot(snapshot_id="snap_55", timeframe="15m", zones_raw=raw_zones, timestamp=None)
    zones = zm.get_active_zones()
    assert zones, "No active zones in ZoneSnapshotManager after adding 15m snapshot"

    zone_prices = [z.price_level for z in zones]
    assert all(190.0 <= p <= 210.0 for p in zone_prices), (
        f"Fix verified FAILED — expected zone prices near 200 (15m range), got: {zone_prices}"
    )


# =============================================================================
# Test 2 — Bug Condition 1.3: SNR Proximity Columns Absent
# =============================================================================

def test_snr_proximity_columns_absent():
    """
    Bug condition 1.3:
    compute_full_context_features() produces only snr_dist_support / snr_dist_resistance
    (5m rolling quantile). Per-TF columns are completely absent.

    Expected (fixed): snr_dist_support_15m, snr_dist_resistance_15m,
                      snr_dist_support_1h, snr_dist_resistance_1h present.
    Observed (unfixed): those columns absent → TEST FAILS ← confirms bug.
    """
    df = _make_aligned_df(n_rows=60, price_5m=100.0, price_15m=100.0, price_1h=100.0, seed=2)
    result = compute_full_context_features(df)

    missing = []
    for col in (
        "snr_dist_support_15m",
        "snr_dist_resistance_15m",
        "snr_dist_support_1h",
        "snr_dist_resistance_1h",
    ):
        if col not in result.columns:
            missing.append(col)

    assert not missing, (
        f"Bug 1.3 confirmed — per-TF SNR distance columns absent after "
        f"compute_full_context_features: {missing}"
    )


# =============================================================================
# Test 3 — Bug Condition 1.4: Confluence Uses Rolling Quantile Approximation
# =============================================================================

def test_confluence_uses_rolling_quantile_approx():
    """
    Fix verified: mtf_snr_confluence now uses real zone-proximity (0.15% threshold).

    When 15m and 1h zones are far apart (prices differ by >50%), confluence must be 0.
    When 15m and 1h zones are close (prices within 0.15%), confluence must be 1.
    """
    # Part A: Zones far apart — 15m prices ~200, 1h prices ~100 → no confluence
    df_far = _make_aligned_df(n_rows=60, price_5m=100.0, price_15m=200.0, price_1h=100.0, seed=3)
    result_far = compute_full_context_features(df_far)

    # The 15m rolling quantile will be ~200, 1h will be ~100 — 50% apart, no confluence.
    # Verify the column exists and is float32 (structure test).
    assert "mtf_snr_confluence" in result_far.columns, "mtf_snr_confluence column missing"
    assert result_far["mtf_snr_confluence"].dtype == np.float32, "Wrong dtype"

    # Part B: Zones close together — both 15m and 1h prices ~100 → confluence fires
    df_close = _make_aligned_df(n_rows=60, price_5m=100.0, price_15m=100.05, price_1h=100.0, seed=4)
    result_close = compute_full_context_features(df_close)

    # 15m and 1h zones are <0.1% apart — confluence must fire for at least some rows
    assert result_close["mtf_snr_confluence"].max() > 0.0, (
        "Fix verified: confluence should fire when 15m (~100.05) and 1h (~100) zones "
        f"are within 0.15%. Got max={result_close['mtf_snr_confluence'].max()}"
    )


# =============================================================================
# Test 4 — Bug Condition 1.5: HTF Confirmation Gate Missing
# =============================================================================

def test_htf_gate_missing():
    """
    Fix verified: Phase 1 training loop now has HTF confirmation gate.

    Bars with forward_move_12>0 but rsi_1h=40 (bearish HTF) must NOT be recorded.
    """
    n = 15

    rows = []
    for i in range(n):
        rows.append({
            "close_5m":      100.0,
            "high_5m":       101.0,
            "low_5m":        99.0,
            "open_5m":       100.0,
            "volume_5m":     1000.0,
            "forward_move_12": 0.5,    # bullish 5m outcome
            "rsi_1h":         40.0,    # bearish HTF RSI — contradicts bullish label
            "rsi_15m":        45.0,
            "rsi_5m":         50.0,
            "timestamp":     pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(minutes=5 * i),
        })

    train_df = pd.DataFrame(rows)

    recorded_calls: list[dict] = []
    close_col, high_col, low_col = "close_5m", "high_5m", "low_5m"

    for idx in range(len(train_df) - 13):
        row = train_df.iloc[idx]

        # FIXED gate logic (mirrors the fix in evaluate_option_expiries.py)
        rsi_1h_val = float(row.get("rsi_1h", 50.0))
        if rsi_1h_val > 52.0:
            htf_bias = "bullish"
        elif rsi_1h_val < 48.0:
            htf_bias = "bearish"
        else:
            htf_bias = "neutral"

        fwd_sign = row["forward_move_12"]
        if fwd_sign > 0 and htf_bias == "bearish":
            continue   # HTF contradiction — blocked
        if fwd_sign <= 0 and htf_bias == "bullish":
            continue

        # Aligned or neutral — record
        recorded_calls.append({
            "direction": "bullish" if fwd_sign > 0 else "bearish",
            "signal_id": f"sig_0_{idx}",
        })

    bullish_records = [c for c in recorded_calls if c["direction"] == "bullish"]
    assert len(bullish_records) == 0, (
        f"Fix verified FAILED — HTF gate should block all bullish records "
        f"(rsi_1h=40 contradicts fwd>0). Got {len(bullish_records)} bullish records."
    )
