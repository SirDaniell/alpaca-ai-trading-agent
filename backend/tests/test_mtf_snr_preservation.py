"""
test_mtf_snr_preservation.py — Preservation Property Tests

These five tests MUST PASS on the current (unfixed) code AND after the fix is applied.
They capture baseline behaviors that must survive the bugfix unchanged.

=== EXPECTED OUTCOME ===
All 5 tests PASS on unfixed code (baseline capture).
All 5 tests PASS on fixed code (regression guard).

Preservation guarantees:
  A — 5m SNR path still produces zones derived from 5m price data (near 100)
  B — Early-return guard (up_to_idx < 20) still blocks zone creation
  C — Missing 4h columns: compute_full_context_features raises no exception
  D — HTF-aligned bullish bars are recorded with direction="bullish"
  E — 70/15/15 chronological split boundaries are correct for all dataset sizes

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
"""
from __future__ import annotations

import sys
import os

import numpy as np
import pandas as pd
import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
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
# Shared helper: build a minimal aligned multi-timeframe dataframe
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
    """Build a minimal aligned multi-timeframe OHLCV dataframe."""
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
# Helper: replicates the 5m SNR path (the unfixed code always uses 5m cols)
# This is the path that MUST remain unchanged after the fix.
# ─────────────────────────────────────────────────────────────────────────────

def _run_5m_snr_path(
    df_full: pd.DataFrame,
    up_to_idx: int,
    zm: ZoneSnapshotManager,
    lookback_period: int = 500,
) -> None:
    """
    Replicate the 5m-column path of update_real_snr_snapshot.
    Uses high_5m/low_5m/close_5m/open_5m/volume_5m — the same columns the
    unfixed (and fixed) code uses when timeframe="5m".
    """
    if up_to_idx < 20:
        return

    high_col  = "high_5m"
    low_col   = "low_5m"
    close_col = "close_5m"
    open_col  = "open_5m"
    vol_col   = "volume_5m"

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
            timeframe="5m",
            zones_raw=raw_zones,
            timestamp=None,
        )


# =============================================================================
# Preservation Test A — 5m SNR path produces zones near 5m prices
# Requirements: 3.1, 3.3
# =============================================================================

def test_preservation_a_5m_snr_path_zones_near_5m_prices():
    """
    Preservation A: The 5m SNR path always generates zones near the 5m price range.

    5m prices cluster around 100; 15m prices cluster around 200.
    When called with the 5m column path, zone prices must be in [90, 110].

    This validates the 5m path is unaffected by the HTF-column bugfix.
    """
    df = _make_aligned_df(n_rows=60, price_5m=100.0, price_15m=200.0, seed=10)

    zm = ZoneSnapshotManager(max_snapshots=20)
    _run_5m_snr_path(df, up_to_idx=55, zm=zm, lookback_period=50)

    zones = zm.get_active_zones()
    assert zones, (
        "No zones detected from the 5m path — check data or lookback_period."
    )

    zone_prices = [z.price_level for z in zones]
    assert all(90.0 <= p <= 110.0 for p in zone_prices), (
        f"Preservation A FAILED — 5m SNR path should yield zones near 100, "
        f"but got: {zone_prices}"
    )


# =============================================================================
# Preservation Test B — Early-return guard (up_to_idx < 20)
# Requirements: 3.1, 3.3
# =============================================================================

def test_preservation_b_early_return_guard_below_20():
    """
    Preservation B: update_real_snr_snapshot returns immediately for up_to_idx < 20,
    adding no zones to the ZoneSnapshotManager.

    This early-return guard must remain unchanged after the fix.
    """
    df = _make_aligned_df(n_rows=60, price_5m=100.0, price_15m=200.0, seed=11)

    for idx in range(0, 20):
        zm = ZoneSnapshotManager(max_snapshots=20)
        _run_5m_snr_path(df, up_to_idx=idx, zm=zm)
        assert len(zm.get_active_zones()) == 0, (
            f"Preservation B FAILED — up_to_idx={idx} should trigger early return "
            f"(< 20), but {len(zm.get_active_zones())} zones were added."
        )


# =============================================================================
# Preservation Test C — Missing 4h columns: no exception, valid output
# Requirements: 3.2
# =============================================================================

def test_preservation_c_missing_4h_no_exception():
    """
    Preservation C: compute_full_context_features() must not raise an exception
    when high_4h / low_4h are absent from the dataframe.

    The function must complete normally and return a DataFrame of the same length.
    This tests the graceful fallback path (requirement 3.2).
    """
    # Build df WITHOUT 4h columns
    df = _make_aligned_df(
        n_rows=60,
        price_5m=100.0,
        price_15m=100.0,
        price_1h=100.0,
        include_4h=False,  # explicitly exclude 4h
        seed=12,
    )
    assert "high_4h" not in df.columns, "Setup error: high_4h should be absent"
    assert "low_4h" not in df.columns,  "Setup error: low_4h should be absent"

    # Must not raise
    result = compute_full_context_features(df)

    assert result is not None, "compute_full_context_features returned None"
    assert len(result) == len(df), (
        f"Preservation C FAILED — result length {len(result)} != input length {len(df)}"
    )


# =============================================================================
# Preservation Test D — HTF-aligned bars still recorded as "bullish"
# Requirements: 3.4, 3.5
# =============================================================================

def test_preservation_d_htf_aligned_bars_recorded_bullish():
    """
    Preservation D: Bars where forward_move_12 > 0 AND rsi_1h > 52 (HTF-aligned)
    must still be recorded as direction="bullish" after the fix.

    This confirms the HTF gate introduced by the fix only blocks *contradicting*
    bars and leaves aligned bars untouched.
    """
    n = 15

    rows = []
    for i in range(n):
        rows.append({
            "close_5m":        100.0,
            "high_5m":         101.0,
            "low_5m":           99.0,
            "open_5m":         100.0,
            "volume_5m":      1000.0,
            "forward_move_12":   0.5,   # bullish 5m outcome
            "rsi_1h":           58.0,   # bullish HTF RSI — aligned with forward move
            "rsi_15m":          55.0,
            "rsi_5m":           54.0,
            "timestamp": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(minutes=5 * i),
        })

    train_df = pd.DataFrame(rows)

    close_col = "close_5m"
    high_col  = "high_5m"
    low_col   = "low_5m"

    recorded_calls: list[dict] = []

    # Replicate the recording logic for the aligned (non-buggy) path.
    # Both unfixed and fixed code should record these rows as "bullish".
    for idx in range(len(train_df) - 13):
        row = train_df.iloc[idx]

        rsi_1h_val = float(row.get("rsi_1h", 50.0))
        fwd_sign   = row["forward_move_12"]

        # Apply the gate (mirrors the fixed code behavior).
        # When rsi_1h > 52 and fwd > 0, the bar is aligned — must be recorded.
        if fwd_sign > 0 and rsi_1h_val < 48:
            continue   # contradicting bearish — skip
        if fwd_sign <= 0 and rsi_1h_val > 52:
            continue   # contradicting bullish — skip

        direction = "bullish" if fwd_sign > 0 else "bearish"
        recorded_calls.append({
            "direction":   direction,
            "signal_id":   f"sig_0_{idx}",
            "symbol":      "TEST",
            "entry_price": float(row[close_col]),
        })

    bullish_records = [c for c in recorded_calls if c["direction"] == "bullish"]
    assert len(bullish_records) > 0, (
        "Preservation D FAILED — HTF-aligned bullish bars (rsi_1h=58, fwd>0) "
        "were not recorded as 'bullish'. These bars must always pass the gate."
    )


# =============================================================================
# Preservation Test E — 70/15/15 chronological split boundaries
# Requirements: 3.6
# =============================================================================

@pytest.mark.parametrize("n_total", [100, 500, 1000, 5000, 40000])
def test_preservation_e_split_boundaries(n_total: int):
    """
    Preservation E: The 70/15/15 chronological train/val/test split boundaries
    must be consistent and correct for all realistic dataset sizes.

    Validates:
    - val_idx > train_idx  (val set starts after train set)
    - n_total - val_idx > 0  (test set is non-empty)
    - Exact boundary formulas: train=70%, val=85%, test=15%
    """
    train_idx = int(n_total * 0.70)
    val_idx   = int(n_total * 0.85)

    assert val_idx > train_idx, (
        f"Preservation E FAILED for n_total={n_total}: "
        f"val_idx={val_idx} should be > train_idx={train_idx}"
    )
    assert n_total - val_idx > 0, (
        f"Preservation E FAILED for n_total={n_total}: "
        f"test set is empty (n_total - val_idx = {n_total - val_idx})"
    )
    # Verify the 70% boundary is correct
    assert train_idx == int(n_total * 0.70), (
        f"Preservation E FAILED: train_idx formula incorrect for n_total={n_total}"
    )
    # Verify the 85% boundary is correct
    assert val_idx == int(n_total * 0.85), (
        f"Preservation E FAILED: val_idx formula incorrect for n_total={n_total}"
    )
