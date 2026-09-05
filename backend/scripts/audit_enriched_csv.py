#!/usr/bin/env python3
"""Audit AXE enriched CSVs against the notebook dataset contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "ATR_5m",
    "EMA_8_5m",
    "EMA_12_5m",
    "Bar_Volume_Up_5m",
    "Bar_Volume_Down_5m",
    "rsi_5m",
    "RSI_14_5m",
    "MACD_5m",
    "BB_Middle_5m",
    "Regime_Speed_Bull_5m",
    "target_dir_5m",
    "target_dir_15m",
    "target_dir_30m",
    "target_dir_1h",
    "forward_move_1",
    "forward_move_3",
    "forward_move_6",
    "forward_move_12",
    "mtf_snr_confluence",
    "snr_dist_support_5m",
    "snr_dist_resistance_5m",
    "dxy_snr_dist_support_5m",
    "dxy_snr_dist_resistance_5m",
]

META_TARGET_COLUMNS = [
    "adv_target_next_zone_idx",
    "adv_target_next_zone_bars",
    "adv_target_next_zone_distance",
    "adv_target_next_zone_type",
    "adv_target_next_zone_volume",
    "Volatility_Bull_next",
    "Volatility_Bear_next",
    "Volatility_Regime_next",
    "Volatility_Expansion_next",
    "Regime_Speed_Bull_next",
    "Regime_Speed_Bear_next",
    "Price_Velocity_Bull_next",
    "Price_Velocity_Bear_next",
    "Price_Velocity_Net_next",
    "vel_bull_fwd_8",
    "vel_bear_fwd_8",
    "vel_net_fwd_8",
]


def _indicator_columns(columns: Iterable[str]) -> list[str]:
    keys = ("EMA_", "ATR", "Bar_Volume", "MACD", "BB_", "Supertrend", "OBV", "SMA_")
    return [
        column
        for column in columns
        if column.endswith("_5m") and any(key in column for key in keys)
    ]


def _rate(series: pd.Series) -> float:
    return float(series.notna().mean()) if len(series) else 0.0


def audit_frame(frame: pd.DataFrame, name: str = "train", allow_sparse: Iterable[str] = ()) -> dict:
    allow_sparse = set(allow_sparse)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    indicators = _indicator_columns(frame.columns)
    sparse = [
        column
        for column in indicators
        if _rate(frame[column]) < 0.95 and column not in allow_sparse
    ]
    non_finite = []
    for column in REQUIRED_COLUMNS:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy()
            if not np.isfinite(values).all():
                non_finite.append(column)

    result = {
        "name": name,
        "n_rows": int(len(frame)),
        "n_cols": int(len(frame.columns)),
        "missing_required": missing,
        "required_nonnull": {column: _rate(frame[column]) for column in REQUIRED_COLUMNS if column in frame},
        "ti_nonnull_min": {
            column: _rate(frame[column]) for column in indicators
        },
        "empty_5m_indicator_count": len(sparse),
        "sparse_5m_indicators": sparse,
        "non_finite_required": non_finite,
        "meta_targets_present": [column for column in META_TARGET_COLUMNS if column in frame.columns],
        "meta_targets_missing": [column for column in META_TARGET_COLUMNS if column not in frame.columns],
    }

    if "rsi_5m" in frame.columns:
        rsi = pd.to_numeric(frame["rsi_5m"], errors="coerce")
        result["rsi_5m_mean"] = float(rsi.mean())
        result["rsi_5m_std"] = float(rsi.std())
        result["rsi_range"] = [float(rsi.min()), float(rsi.max())]
    else:
        result["rsi_5m_mean"] = None
        result["rsi_5m_std"] = None
        result["rsi_range"] = None

    direction_bad = []
    for column in ("target_dir_5m", "target_dir_15m", "target_dir_30m", "target_dir_1h"):
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").dropna().unique()
            if not set(values).issubset({0, 1}):
                direction_bad.append(column)
    result["non_binary_direction_targets"] = direction_bad

    # ── NEW: Variance (std) gates — non-null ≠ useful ────────────────────────
    TI_VARIANCE_COLS = ["ATR_5m", "EMA_8_5m", "EMA_12_5m", "Bar_Volume_Up_5m", "MACD_5m", "OBV_5m"]
    ti_zero_cols = [
        col for col in TI_VARIANCE_COLS
        if col in frame.columns and pd.to_numeric(frame[col], errors="coerce").std() < 1e-6
    ]
    result["ti_constant_zero_cols"] = ti_zero_cols

    # Zone targets must have more than 1 unique class
    zone_degenerate = []
    for col in ("adv_target_next_zone_idx", "adv_target_next_zone_type"):
        if col in frame.columns:
            n_unique = pd.to_numeric(frame[col], errors="coerce").nunique()
            if n_unique <= 1:
                zone_degenerate.append(f"{col} (nunique={n_unique})")
    result["zone_degenerate_targets"] = zone_degenerate

    # Vol/regime/velocity targets must have std > 0
    vel_zero = [
        col for col in ("Volatility_Bull_next", "Volatility_Bear_next", "Price_Velocity_Bull_next",
                        "Price_Velocity_Bear_next", "Regime_Speed_Bull_next", "vel_bull_fwd_8")
        if col in frame.columns and pd.to_numeric(frame[col], errors="coerce").std() < 1e-6
    ]
    result["constant_regime_velocity_targets"] = vel_zero

    # Summary of TI key stats for quick inspection
    result["ti_key_stats"] = {
        col: {"std": float(pd.to_numeric(frame[col], errors="coerce").std()),
              "zeros_pct": float((pd.to_numeric(frame[col], errors="coerce") == 0).mean() * 100)}
        for col in TI_VARIANCE_COLS if col in frame.columns
    }

    failures = []
    if missing:
        failures.append(f"missing required columns: {missing}")
    if sparse:
        failures.append(f"sparse 5m indicators (<95% non-null): {sparse}")
    if non_finite:
        failures.append(f"non-finite required columns: {non_finite}")
    if direction_bad:
        failures.append(f"non-binary direction targets: {direction_bad}")
    if result["rsi_5m_mean"] is None or not 20.0 < result["rsi_5m_mean"] < 80.0:
        failures.append("rsi_5m mean is outside (20, 80)")
    if result["rsi_5m_std"] is None or result["rsi_5m_std"] >= 40.0:
        failures.append("rsi_5m standard deviation is >= 40")
    if ti_zero_cols:
        failures.append(f"TI columns are constant-zero (std≈0): {ti_zero_cols}")
    if zone_degenerate:
        failures.append(f"Zone targets are degenerate: {zone_degenerate}")
    if vel_zero:
        failures.append(f"Regime/velocity targets are constant-zero: {vel_zero}")
    result["failures"] = failures
    result["passed"] = not failures
    return result



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", type=Path, help="Training CSV to audit")
    parser.add_argument("--val", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--manifest", type=Path, help="Optional output JSON path")
    parser.add_argument("--allow-sparse", action="append", default=[])
    args = parser.parse_args()

    reports = {}
    for name, path in (("train", args.train), ("val", args.val), ("test", args.test)):
        if path is None:
            continue
        if not path.exists():
            print(f"ERROR: CSV does not exist: {path}", file=sys.stderr)
            return 2
        reports[name] = audit_frame(pd.read_csv(path), name=name, allow_sparse=args.allow_sparse)
        report = reports[name]
        print(
            f"{name}: rows={report['n_rows']} cols={report['n_cols']} "
            f"rsi_mean={report['rsi_5m_mean']} rsi_std={report['rsi_5m_std']} "
            f"sparse_5m={report['empty_5m_indicator_count']}"
        )
        for failure in report["failures"]:
            print(f"  FAIL: {failure}", file=sys.stderr)

    payload = reports if len(reports) > 1 else next(iter(reports.values()))
    if args.manifest:
        args.manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if all(report["passed"] for report in reports.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
