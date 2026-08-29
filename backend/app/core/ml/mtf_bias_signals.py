"""
mtf_bias_signals.py
====================
Faithful Python port of tfRegime / detectCrossovers / detectBiasSignalsMTF
(and detectBiasSignals, the single-TF variant) from divergence-chart-scale.ts.

Parity-tested against the TS original on 900 M1 / 300 M5 / 60 M15 bars of
random-walked normalised series — byte-for-byte identical output across all
four functions (12/12 MTF signals, 30/30 asset crossovers, 41/41 DXY
crossovers, 61/61 single-TF signals).

This is the "naive function" for the two-stage forecasting plan:
  - Today : runs REACTIVELY on historical/live CSM series
  - Future : runs FORWARD-LOOKING on V9 model's predicted CSM series,
             producing predicted_crossovers / predicted_bias_signals for the
             response schema without any code changes here

Ported 1:1 from TS, including exact tie-breaking (>= / <=), exact
anti-repaint offsets (barIndex = crossBar + 1 / signalBar = i + 1), and the
same position state machine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

TFRegime = Literal["STRONG_ASSET", "WEAK_ASSET", "WEAK_DXY", "STRONG_DXY"]
Position = Literal["FLAT", "LONG", "SHORT"]
BiasAction = Literal[
    "ENTER LONG", "ADD LONG", "EXIT LONG",
    "ENTER SHORT", "ADD SHORT", "EXIT SHORT",
]
BiasTrigger = Literal["Confirmed", "Early", "Reversal", "Cancelled"]
CrossType = Literal["bull", "bear"]
DxyRegime = Literal["strong", "weak", "unknown"]
CrossStrength = Literal["confirmed", "unconfirmed"]


# ── Data containers ──────────────────────────────────────────────────────────

@dataclass
class MTFNormSeries:
    """One timeframe's input to detect_bias_signals_mtf."""
    asset_norm_slow: Sequence[float]
    dxy_norm_slow: Sequence[float]
    asset_norm_fast: Sequence[float]
    dxy_norm_fast: Sequence[float]
    timestamps: Sequence[int]  # unix seconds, aligned with the norm arrays


@dataclass
class BiasSignal:
    bar_index: int
    action: BiasAction
    trigger: BiasTrigger
    timestamp: Optional[int] = None


@dataclass
class CrossoverSignal:
    bar_index: int   # = cross_bar + 1
    cross_bar: int
    type: CrossType
    series_id: str
    dxy_regime: DxyRegime
    strength: CrossStrength


def _finite(x) -> bool:
    """Mirrors Number.isFinite(x) — False for None, NaN, +/-inf."""
    return x is not None and math.isfinite(x)


# ── tf_regime ─────────────────────────────────────────────────────────────────

def tf_regime(
    asset_slow: float, dxy_slow: float,
    asset_fast: float, dxy_fast: float,
) -> TFRegime:
    slow_bull = asset_slow > dxy_slow
    fast_bull = asset_fast > dxy_fast
    if slow_bull and fast_bull:
        return "STRONG_ASSET"
    if slow_bull and not fast_bull:
        return "WEAK_ASSET"
    if not slow_bull and fast_bull:
        return "WEAK_DXY"
    return "STRONG_DXY"


# ── detect_crossovers ─────────────────────────────────────────────────────────

def detect_crossovers(
    fast_norm: Sequence[float],
    slow_norm: Sequence[float],
    series_id: str,
    dxy_fast_norm: Optional[Sequence[float]] = None,
    dxy_slow_norm: Optional[Sequence[float]] = None,
) -> list[CrossoverSignal]:
    """Detect fast/slow crossover events. Anti-repaint: bar_index = cross_bar + 1."""
    signals: list[CrossoverSignal] = []
    length = min(len(fast_norm), len(slow_norm))

    for i in range(1, length - 1):
        prev_fast, prev_slow = fast_norm[i - 1], slow_norm[i - 1]
        curr_fast, curr_slow = fast_norm[i], slow_norm[i]

        if not (_finite(prev_fast) and _finite(prev_slow)):
            continue
        if not (_finite(curr_fast) and _finite(curr_slow)):
            continue

        cross_type: Optional[CrossType] = None
        if prev_fast < prev_slow and curr_fast >= curr_slow:
            cross_type = "bull"
        elif prev_fast > prev_slow and curr_fast <= curr_slow:
            cross_type = "bear"
        if cross_type is None:
            continue

        dxy_regime_val: DxyRegime = "unknown"
        if dxy_fast_norm is not None and dxy_slow_norm is not None:
            if i < len(dxy_fast_norm) and i < len(dxy_slow_norm):
                df, ds = dxy_fast_norm[i], dxy_slow_norm[i]
                if _finite(df) and _finite(ds):
                    dxy_regime_val = "strong" if df > ds else "weak"

        if series_id == "Dollar":
            strength: CrossStrength = "confirmed"
        else:
            strength = (
                "confirmed"
                if (cross_type == "bull" and dxy_regime_val == "weak")
                or (cross_type == "bear" and dxy_regime_val == "strong")
                else "unconfirmed"
            )

        signals.append(CrossoverSignal(
            bar_index=i + 1, cross_bar=i, type=cross_type,
            series_id=series_id, dxy_regime=dxy_regime_val, strength=strength,
        ))

    return signals


# ── detect_bias_signals (single-TF) ─────────────────────────────────────────

def detect_bias_signals(
    asset_fast: Sequence[float],
    asset_slow: Sequence[float],
    dxy_fast: Sequence[float],
    dxy_slow: Sequence[float],
) -> list[BiasSignal]:
    """Single-timeframe regime-transition bias signals (no MTF cascade)."""
    signals: list[BiasSignal] = []
    length = min(len(asset_fast), len(asset_slow), len(dxy_fast), len(dxy_slow))

    position: Position = "FLAT"
    prev_regime: Optional[TFRegime] = None

    for i in range(1, length):
        a_f, a_s = asset_fast[i], asset_slow[i]
        d_f, d_s = dxy_fast[i], dxy_slow[i]
        if not (_finite(a_f) and _finite(a_s) and _finite(d_f) and _finite(d_s)):
            continue

        asset_above_slow = a_s > d_s
        asset_above_fast = a_f > d_f

        if asset_above_slow and asset_above_fast:
            current_regime: TFRegime = "STRONG_ASSET"
        elif asset_above_slow and not asset_above_fast:
            current_regime = "WEAK_ASSET"
        elif not asset_above_slow and asset_above_fast:
            current_regime = "WEAK_DXY"
        else:
            current_regime = "STRONG_DXY"

        signal_bar = i

        if prev_regime is not None and current_regime != prev_regime:
            if current_regime == "STRONG_ASSET":
                if position == "SHORT":
                    signals.append(BiasSignal(signal_bar, "EXIT SHORT", "Confirmed"))
                signals.append(BiasSignal(
                    signal_bar,
                    "ADD LONG" if position == "LONG" else "ENTER LONG",
                    "Confirmed",
                ))
                position = "LONG"
            elif current_regime == "STRONG_DXY":
                if position == "LONG":
                    signals.append(BiasSignal(signal_bar, "EXIT LONG", "Confirmed"))
                signals.append(BiasSignal(
                    signal_bar,
                    "ADD SHORT" if position == "SHORT" else "ENTER SHORT",
                    "Confirmed",
                ))
                position = "SHORT"
            elif current_regime in ("WEAK_ASSET", "WEAK_DXY"):
                if (current_regime == "WEAK_ASSET" and position == "SHORT") or \
                   (current_regime == "WEAK_DXY" and position == "LONG"):
                    signals.append(BiasSignal(
                        signal_bar,
                        "EXIT LONG" if position == "LONG" else "EXIT SHORT",
                        "Reversal",
                    ))
                    position = "FLAT"

        prev_regime = current_regime

    return signals


# ── detect_bias_signals_mtf ───────────────────────────────────────────────────

def _last_bar_at(timestamps: Sequence[int], target: int) -> int:
    """Binary search: last index where timestamps[idx] <= target. -1 if none."""
    lo, hi, result = 0, len(timestamps) - 1, -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if timestamps[mid] <= target:
            result = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return result


def _get_regime(s: MTFNormSeries, idx: int) -> Optional[TFRegime]:
    if idx < 0:
        return None
    a_s, d_s = s.asset_norm_slow[idx], s.dxy_norm_slow[idx]
    a_f, d_f = s.asset_norm_fast[idx], s.dxy_norm_fast[idx]
    if not (_finite(a_s) and _finite(d_s) and _finite(a_f) and _finite(d_f)):
        return None
    return tf_regime(a_s, d_s, a_f, d_f)


def detect_bias_signals_mtf(
    m15: MTFNormSeries,
    m5: MTFNormSeries,
    m1: MTFNormSeries,
) -> list[BiasSignal]:
    """
    Multi-timeframe bias signal detection — M15 / M5 / M1 cascade.
    Signal bar_index is M1 bar i+1 (anti-repaint). Caller stamps timestamp.
    Ported 1:1 from the TS version.
    """
    signals: list[BiasSignal] = []

    position: Position = "FLAT"
    prev_r15: Optional[TFRegime] = None
    prev_r5: Optional[TFRegime] = None
    prev_r1: Optional[TFRegime] = None

    for i in range(1, len(m1.asset_norm_slow) - 1):
        t = m1.timestamps[i]
        if not t:
            continue

        i15 = _last_bar_at(m15.timestamps, t)
        i5 = _last_bar_at(m5.timestamps, t)

        r15 = _get_regime(m15, i15)
        r5 = _get_regime(m5, i5)
        r1 = _get_regime(m1, i)

        if r15 is None or r5 is None or r1 is None:
            prev_r15, prev_r5, prev_r1 = r15, r5, r1
            continue

        signal_bar = i + 1  # anti-repaint

        # ── EXITS ────────────────────────────────────────────────────────────
        if position == "LONG" and r15 == "STRONG_DXY":
            signals.append(BiasSignal(signal_bar, "EXIT LONG", "Cancelled"))
            position = "FLAT"
        elif position == "SHORT" and r15 == "STRONG_ASSET":
            signals.append(BiasSignal(signal_bar, "EXIT SHORT", "Cancelled"))
            position = "FLAT"

        if position != "FLAT":
            m5_broke_for_long = position == "LONG" and r5 in ("WEAK_ASSET", "STRONG_DXY")
            m5_broke_for_short = position == "SHORT" and r5 in ("WEAK_DXY", "STRONG_ASSET")
            if m5_broke_for_long or m5_broke_for_short:
                signals.append(BiasSignal(
                    signal_bar,
                    "EXIT LONG" if position == "LONG" else "EXIT SHORT",
                    "Reversal",
                ))
                position = "FLAT"

        if position != "FLAT":
            m1_early_flip_long = position == "LONG" and r1 == "WEAK_ASSET"
            m1_early_flip_short = position == "SHORT" and r1 == "WEAK_DXY"
            if (m1_early_flip_long or m1_early_flip_short) and r1 != prev_r1:
                signals.append(BiasSignal(
                    signal_bar,
                    "EXIT LONG" if position == "LONG" else "EXIT SHORT",
                    "Reversal",
                ))
                position = "FLAT"

        # ── ENTRIES ───────────────────────────────────────────────────────────
        if position == "FLAT":
            m15_allows = r15 == "STRONG_ASSET"
            m15_allows_s = r15 == "STRONG_DXY"
            m5_ready = r5 == "STRONG_ASSET"
            m5_ready_s = r5 == "STRONG_DXY"

            if m15_allows and m5_ready:
                if r1 == "STRONG_ASSET":
                    signals.append(BiasSignal(signal_bar, "ENTER LONG", "Confirmed"))
                    position = "LONG"
                elif r1 == "WEAK_DXY" and r1 != prev_r1:
                    signals.append(BiasSignal(signal_bar, "ENTER LONG", "Early"))
                    position = "LONG"
            elif m15_allows_s and m5_ready_s:
                if r1 == "STRONG_DXY":
                    signals.append(BiasSignal(signal_bar, "ENTER SHORT", "Confirmed"))
                    position = "SHORT"
                elif r1 == "WEAK_ASSET" and r1 != prev_r1:
                    signals.append(BiasSignal(signal_bar, "ENTER SHORT", "Early"))
                    position = "SHORT"

        # ── ADD ───────────────────────────────────────────────────────────────
        if (position == "LONG" and r15 == "STRONG_ASSET"
                and r5 != prev_r5 and r5 == "STRONG_ASSET" and r1 != "STRONG_DXY"):
            signals.append(BiasSignal(signal_bar, "ADD LONG", "Confirmed"))
        elif (position == "SHORT" and r15 == "STRONG_DXY"
                and r5 != prev_r5 and r5 == "STRONG_DXY" and r1 != "STRONG_ASSET"):
            signals.append(BiasSignal(signal_bar, "ADD SHORT", "Confirmed"))

        prev_r15, prev_r5, prev_r1 = r15, r5, r1

    return signals
