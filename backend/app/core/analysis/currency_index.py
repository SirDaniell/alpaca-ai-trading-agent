"""
Currency Index Calculator
=========================
Computes currency strength indices — each as a full OHLCV dict — from
per-pair OHLCV data.

Supported indices
-----------------
  Dollar  — USDX  (ICE/FINEX official formula, scalar 50.143)
  Euro    — Wider ECB-style basket (non-standard, no scalar; user-supplied)
  ECX     — Euro Currency Index (ICE/FINEX, scalar 34.388)
  JPY     — Yen proxy index  (5-pair BIS-weighted, base-100)
  CNY     — Yuan proxy index (5-pair simplified CFETS basket, base-100)
  RUB     — Ruble proxy index (3-pair only: USD/EUR/CNY; post-2022 caveat)

  NOTE: JPY / CNY / RUB are *proxy* indices, not official published series.
  Weights are derived from BIS/CFETS trade-share data and renormalised to
  the available liquid pairs. Treat as directional signals, not benchmarks.

  NOTE (RUB): MOEX pairs have been illiquid since 2022 sanctions. The index
  is included for completeness but should be interpreted with caution.

Pair column naming convention (case-sensitive)
-----------------------------------------------
    {field}_{pair}
    e.g.  open_EURUSD  high_USDJPY  tick_volume_CNYJPY

Sign convention for exponents
------------------------------
    Pairs are expressed so that a *rising* column value means the index
    currency is *strengthening*.

    Dollar / ECX / Euro : pairs follow standard market quoting convention.
    JPY index : all pairs are XXXJPY (JPY in denominator).
                Positive exponent → index rises when JPY gains vs XXX.
    CNY index : all pairs are XXXCNY (CNY in denominator). Same logic.
    RUB index : all pairs are XXXRUB (RUB in denominator). Same logic.

@ 2026
"""

import logging
import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal, Union, Optional, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Index definitions
# ---------------------------------------------------------------------------
# Structure:
#   { name: { "scalar": float, "pairs": {pair: exponent}, "note": str } }
#
# Proxy index scalars are set to 100.0 (base-100 convention).
# Verified: weights in each proxy index sum to 1.0.
# ---------------------------------------------------------------------------

INDEX_DEFINITIONS: dict[str, dict] = {

    # ------------------------------------------------------------------
    # Dollar — Official ICE/FINEX USDX
    # ------------------------------------------------------------------
    "Dollar": {
        "scalar": 50.14348112,
        "pairs": {
            "EURUSD": -0.576,   # EUR/USD: EUR up → USD down
            "USDJPY":  0.136,
            "GBPUSD": -0.119,
            "USDCAD":  0.091,
            "USDSEK":  0.042,
            "USDCHF":  0.036,
        },
        "note": (
            "Official ICE/FINEX formula. "
            "Sum of |weights| = 0.576+0.136+0.119+0.091+0.042+0.036 = 1.0"
        ),
    },

    # ------------------------------------------------------------------
    # Euro — Wider basket (user-supplied, non-standard)
    # ------------------------------------------------------------------
    "Euro": {
        "scalar": 1.0,
        "pairs": {
            "EURGBP":  0.3060,
            "EURJPY":  0.1800,
            "EURCHF":  0.0785,
            "EURUSD": -0.3025,  # USD-denominated cross → negative
            "EURAUD": -0.0535,
            "EURCAD": -0.0260,
            "EURNZD": -0.0235,
        },
        "note": (
            "Non-standard wider basket supplied by user; no scalar (=1.0). "
            "Sum of |weights| ≈ 0.97. Verify weights with your data vendor."
        ),
    },

    # ------------------------------------------------------------------
    # ECX — Official Euro Currency Index (ICE/FINEX)
    # ------------------------------------------------------------------
    "ECX": {
        "scalar": 34.38805726,
        "pairs": {
            "EURUSD":  0.3155,
            "EURGBP":  0.3056,
            "EURJPY":  0.1891,
            "EURCHF":  0.1113,
            "EURSEK":  0.0785,
        },
        "note": "Official ICE/FINEX ECX formula. Weights sum = 1.0",
    },

    # ------------------------------------------------------------------
    # JPY — Japanese Yen proxy index
    #
    # Pairs expressed as XXXJPY (JPY in denominator).
    # A rising pair value means JPY is gaining against XXX → positive exp.
    #
    # Weights: BIS effective exchange rate trade shares renormalised to
    # 5 liquid pairs.
    #   USDJPY 0.3768 | EURJPY 0.2356 | GBPJPY 0.0826
    #   AUDJPY 0.1652 | CADJPY 0.1398     Sum = 1.0
    # ------------------------------------------------------------------
    "JPY": {
        "scalar": 100.0,
        "pairs": {
            "USDJPY":  0.3768,
            "EURJPY":  0.2356,
            "GBPJPY":  0.0826,
            "AUDJPY":  0.1652,
            "CADJPY":  0.1398,
        },
        "note": (
            "Proxy index — not an official published series. "
            "Weights from BIS trade-share data, renormalised to 5 liquid pairs. "
            "Scalar = 100 (base-100). Rising index = JPY strengthening."
        ),
    },

    # ------------------------------------------------------------------
    # CNY — Chinese Yuan proxy index (simplified CFETS)
    #
    # Pairs expressed as XXXCNY (CNY in denominator).
    # Weights: top-5 CFETS pairs by trade share, renormalised to sum=1.0.
    #   Raw CFETS:  USD 0.2159 | EUR 0.1624 | JPY 0.1153
    #               AUD 0.0601 | GBP 0.0316   (total raw = 0.5853)
    #   Renormalised: USD 0.3689 | EUR 0.2775 | JPY 0.1970
    #                 AUD 0.1027 | GBP 0.0539   Sum = 1.0
    # Full CFETS basket has 24 pairs; remaining ~37% weight (SGD, HKD,
    # CAD, CHF, etc.) redistributed proportionally across these 5.
    # ------------------------------------------------------------------
    "CNY": {
        "scalar": 100.0,
        "pairs": {
            "USDCNH":  0.3689,
            "EURCNH":  0.2775,
            "JPYCNH":  0.1970,
            "AUDCNH":  0.1027,
            "GBPCNH":  0.0539,
        },
        "note": (
            "Proxy index — simplified CFETS basket (top 5 of 24 pairs). "
            "Remaining ~37% CFETS weight (SGD, HKD, CAD, CHF…) redistributed. "
            "Scalar = 100 (base-100). Rising index = CNY strengthening. "
            "Uses CNH (Offshore Yuan) symbols to support standard broker feeds. "
            "For full accuracy use the official CFETS RMB Index."
        ),
    },

    # ------------------------------------------------------------------
    # RUB — Russian Ruble proxy index (3-pair only)
    #
    # ⚠️  MOEX pairs have been highly illiquid since 2022 sanctions.
    #
    # Pairs expressed as XXXRUB (RUB in denominator).
    # Weights: pre-sanction Russian trade share, renormalised to 3 pairs.
    #   USDRUB 0.52 | EURRUB 0.32 | CNYRUB 0.16    Sum = 1.0
    # ------------------------------------------------------------------
    "RUB": {
        "scalar": 100.0,
        "pairs": {
            "USDRUB":  0.52,
            "EURRUB":  0.32,
            "CNHRUB":  0.16,
        },
        "note": (
            "⚠️  PROXY ONLY — 3 pairs. "
            "MOEX-listed pairs illiquid since 2022 sanctions; "
            "treat as directional signal only, not a benchmark. "
            "Weights from pre-2022 Russian trade share. "
            "Scalar = 100 (base-100). Rising index = RUB strengthening."
        ),
    },
}

OHLCV_FIELDS = ("open", "high", "low", "close", "tick_volume")

IndexName = Literal["Dollar", "Euro", "ECX", "JPY", "CNY", "RUB"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _col(field: str, pair: str) -> str:
    """Return the expected DataFrame column name for a given field and pair."""
    return f"{field}_{pair}"


def _weighted_product(
    data: pd.DataFrame,
    field: str,
    pairs: dict[str, float],
    scalar: float,
) -> pd.Series:
    """
    Vectorised weighted geometric product: scalar * ∏ col^exp

    Optimisation (loop unroll):
      All pair columns are extracted into a 2-D NumPy matrix in one
      pandas call, cleaned with vectorised ops, exponentiated via
      NumPy broadcasting, then reduced with np.prod — eliminating the
      per-pair Python loop entirely.

      For P pairs and N rows the old code did P sequential Python
      iterations; this does one (N×P) matrix power + one reduction.
    """
    pair_list = list(pairs.keys())
    exponents = np.array(list(pairs.values()), dtype=np.float64)  # shape (P,)
    col_names = [_col(field, p) for p in pair_list]

    # Validate columns upfront (one pass)
    missing = [c for c in col_names if c not in data.columns]
    if missing:
        raise KeyError(f"Missing column(s) for index calculation: {missing}")

    # ── Extract full matrix in one call: shape (N, P) ─────────────────────
    mat: np.ndarray = data[col_names].to_numpy(dtype=np.float64)

    if field == "tick_volume":
        # Volumes: treat zero / NaN / negative as tiny positive
        mat = np.where(np.isfinite(mat) & (mat > 0), mat, 1e-8)
    else:
        # Price fields: bulk ffill/bfill then clamp
        if not np.all(np.isfinite(mat)):
            mat = (
                pd.DataFrame(mat, index=data.index)
                .ffill().bfill()
                .to_numpy(dtype=np.float64)
            )
        # Replace zeros/negatives with safe minimum
        mat = np.where(mat > 0, mat, 1e-8)
        mat = np.clip(mat, 1e-8, None)

    # ── Vectorised power: (N, P) ** (1, P)  →  (N, P) ───────────────────
    powered = np.power(mat, exponents[np.newaxis, :])   # broadcast

    # Guard against inf/nan from extreme exponentiation
    if not np.all(np.isfinite(powered)):
        powered = np.where(np.isfinite(powered), powered, np.nan)
        powered = (
            pd.DataFrame(powered, index=data.index)
            .ffill().bfill().fillna(1.0)
            .to_numpy(dtype=np.float64)
        )

    # ── Reduce across pairs: scalar * ∏_j powered[:, j] ──────────────────
    product = scalar * np.prod(powered, axis=1)   # shape (N,)

    # Final guard
    product = np.where(np.isfinite(product), product, scalar)
    return pd.Series(product, index=data.index, dtype=float)


# ---------------------------------------------------------------------------
# Data preparation helper
# ---------------------------------------------------------------------------

def prepare_index_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-process merged pair data for robust index calculation.

    Optimisation (vectorised / loop-unrolled):
      All column groups are operated on as a single DataFrame slice
      rather than column-by-column Python loops, which allows pandas
      to apply the operation in one C-level pass.

    Operations:
    - Forward/backward fills price and volume columns across time
    - Clamps negative/zero prices to safe positive values
    - Handles missing data gracefully

    Should be called before passing data to CurrencyIndexCalculator.
    """
    df = df.copy()

    # Identify OHLCV columns (one-pass comprehension)
    price_cols  = [c for c in df.columns if c.startswith(("open_", "high_", "low_", "close_"))]
    volume_cols = [c for c in df.columns if c.startswith("tick_volume_")]
    all_cols    = price_cols + volume_cols

    # ── Bulk ffill/bfill (single pandas call over entire sub-frame) ───────
    if all_cols:
        df[all_cols] = df[all_cols].ffill().bfill()

    # ── Bulk clip for price columns ───────────────────────────────────────
    if price_cols:
        df[price_cols] = df[price_cols].clip(lower=1e-8)

    # ── Bulk zero-replacement for volume columns ──────────────────────────
    if volume_cols:
        df[volume_cols] = df[volume_cols].replace(0, 1e-8)

    return df


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CurrencyIndexCalculator:
    """
    Calculate OHLCV index values for currency strength indices.

    Supported indices
    -----------------
    Dollar, Euro, ECX        — USD / EUR official and proxy
    JPY, CNY, RUB            — Yen, Yuan, Ruble proxy indices

    See INDEX_DEFINITIONS for full weight tables, sources, and caveats.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain columns in the form  {field}_{pair}
        e.g. open_EURUSD, high_USDJPY, tick_volume_AUDJPY

    Quick start
    -----------
    # For production with merged MT5 data, pre-clean first:
    df = prepare_index_data(merged_df)  # Fills gaps, clamps extremes
    calc = CurrencyIndexCalculator(df)

    # All indices (those missing pair columns are skipped with a log error)
    all_idx = calc.calculate_indices()

    # Single index
    jpy = calc.calculate_indices(indices="JPY")

    # Subset
    subset = calc.calculate_indices(indices=["Dollar", "ECX", "CNY"])

    # Flat DataFrame  ->  columns named {Index}_{field}
    df_out = calc.to_dataframe(["JPY", "CNY", "RUB"])

    # Print weight tables and caveats
    CurrencyIndexCalculator.describe()
    CurrencyIndexCalculator.describe("RUB")
    """

    def __init__(self, data: pd.DataFrame, reporter: Optional[Any] = None) -> None:
        self.data = data.copy()
        self.reporter = reporter

    # ------------------------------------------------------------------
    # Class-level utility
    # ------------------------------------------------------------------

    @classmethod
    def describe(cls, index: Union[str, None] = None) -> None:
        """
        Print a human-readable weight table and notes for one or all indices.

        Parameters
        ----------
        index : str | None
            Specific index name, or None to describe all.
        """
        targets = [index] if index else list(INDEX_DEFINITIONS.keys())
        for name in targets:
            defn = INDEX_DEFINITIONS[name]
            print(f"\n{'='*58}")
            print(f"  {name}   (scalar = {defn['scalar']})")
            print(f"{'='*58}")
            print(f"  {defn.get('note', '')}")
            print(f"\n  {'Pair':<12} {'Weight':>8}   Direction")
            print(f"  {'-'*44}")
            for pair, exp in defn["pairs"].items():
                arrow = "↑ strengthens index" if exp > 0 else "↑ weakens index"
                print(f"  {pair:<12} {exp:>8.4f}   {arrow}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_indices(
        self,
        indices: Union[IndexName, list[IndexName], None] = None,
    ) -> dict[str, dict[str, pd.Series]]:
        """
        Calculate OHLCV values for one or more currency indices.

        Optimisation (threading):
          Each index is independent.  A ThreadPoolExecutor computes all
          requested indices in parallel.  The reporter is called from the
          main thread after each future resolves (thread-safe progress).

        Parameters
        ----------
        indices : str | list[str] | None
            "Dollar", "Euro", "ECX", "JPY", "CNY", "RUB", or a list.
            Pass None (default) to attempt all six.

        Returns
        -------
        dict
            {
              "JPY": {
                "open":        pd.Series,
                "high":        pd.Series,
                "low":         pd.Series,
                "close":       pd.Series,
                "tick_volume": pd.Series,
              },
              ...
            }
        """
        requested     = self._resolve_indices(indices)
        total_indices = len(requested)
        results: dict[str, dict[str, pd.Series]] = {}

        if self.reporter:
            self.reporter.report(
                progress=50,
                message=f"Computing {total_indices} indices in parallel...",
            )

        # ── Dispatch all indices concurrently ─────────────────────────────
        # Increased from 6 to 16 to handle all indices concurrently (7 symbols + DXY + buffer)
        n_workers = min(total_indices, (os.cpu_count() or 4), 16)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            future_to_name = {
                pool.submit(self._compute_index, name): name
                for name in requested
            }
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    results[name] = future.result()
                    logger.info("✓ Computed %s index successfully.", name)
                except KeyError as e:
                    logger.error("✗ Missing data for %s index — skipping. %s", name, e)
                except Exception as e:
                    logger.error("✗ Unexpected error computing %s index: %s", name, e)

                # Progress update from main thread (thread-safe)
                if self.reporter:
                    progress = int(50 + (len(results) / total_indices) * 40)
                    self.reporter.report(
                        progress=progress,
                        message=f"Calculated {name} index ({len(results)}/{total_indices})",
                    )

        return results

    def to_dataframe(
        self,
        indices: Union[IndexName, list[IndexName], None] = None,
    ) -> pd.DataFrame:
        """
        Like calculate_indices() but returns a flat DataFrame.

        Column names follow the pattern  {IndexName}_{field}
        e.g. JPY_close, CNY_high, RUB_tick_volume.
        
        Emits progress updates during conversion.
        """
        index_dict = self.calculate_indices(indices)
        frames: dict[str, pd.Series] = {}
        
        total_indices = len(index_dict)
        idx_num = 0
        for idx_name, ohlcv in index_dict.items():
            for field_idx, field in enumerate(ohlcv.items()):
                field_name, series = field
                frames[f"{idx_name}_{field_name}"] = series
                
                # Emit fine-grained progress during DataFrame construction
                if self.reporter and field_idx == len(OHLCV_FIELDS) - 1:
                    progress = int(90 + (idx_num / max(total_indices, 1)) * 9)
                    self.reporter.report(
                        progress=progress,
                        message=f"Converting {idx_name} to DataFrame..."
                    )
            idx_num += 1
        
        # Final progress
        if self.reporter:
            self.reporter.report(
                progress=99,
                message=f"Flattening {len(frames)} columns..."
            )
        
        return pd.DataFrame(frames, index=self.data.index)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_indices(
        indices: Union[IndexName, list[IndexName], None],
    ) -> list[str]:
        """Normalise the `indices` argument to a validated list of names."""
        all_names = list(INDEX_DEFINITIONS.keys())

        if indices is None:
            return all_names

        if isinstance(indices, str):
            indices = [indices]

        unknown = [i for i in indices if i not in all_names]
        if unknown:
            raise ValueError(
                f"Unknown index name(s): {unknown}. "
                f"Valid options: {all_names}"
            )
        return list(indices)

    def _compute_index(self, name: str) -> dict[str, pd.Series]:
        """
        Compute OHLCV series for a single named index.

        Optimisation (threading):
          The 5 OHLCV fields are fully independent.  A ThreadPoolExecutor
          dispatches all 5 calls to _weighted_product concurrently.
          Because _weighted_product is dominated by NumPy C extensions
          (which release the GIL), true parallel speedup is achieved.
        """
        defn   = INDEX_DEFINITIONS[name]
        scalar: float            = defn["scalar"]
        pairs:  dict[str, float] = defn["pairs"]

        # Pre-check: verify all required columns exist (list-comp, no loop)
        missing_cols = [
            _col(field, pair)
            for pair in pairs
            for field in OHLCV_FIELDS
            if _col(field, pair) not in self.data.columns
        ]
        if missing_cols:
            logger.warning(
                "Missing %d columns for %s index: %s%s",
                len(missing_cols), name, missing_cols[:5],
                "..." if len(missing_cols) > 5 else "",
            )

        # ── Parallel field computation ─────────────────────────────────────
        n_workers = min(len(OHLCV_FIELDS), (os.cpu_count() or 2))

        def _compute_field(field: str):
            return field, _weighted_product(self.data, field, pairs, scalar)

        results: dict[str, pd.Series] = {}
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for field, series in pool.map(_compute_field, OHLCV_FIELDS):
                results[field] = series

        # Post-check: log any remaining NaNs
        for field, series in results.items():
            nan_count = int(series.isna().sum())
            if nan_count > 0:
                nan_pct = (nan_count / len(series)) * 100
                logger.warning(
                    "%s %s: %d NaNs (%.1f%%). "
                    "Consider calling prepare_index_data() before CurrencyIndexCalculator.",
                    name, field, nan_count, nan_pct,
                )

        return results


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
'''
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    # All pairs across all six indices
    all_pairs = [
        # Dollar
        "EURUSD", "USDJPY", "GBPUSD", "USDCAD", "USDSEK", "USDCHF",
        # Euro (extra)
        "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
        # ECX (extra)
        "EURSEK",
        # JPY
        "GBPJPY", "AUDJPY", "CADJPY",
        # CNY
        "USDCNH", "EURCNH", "JPYCNH", "AUDCNH", "GBPCNH",
        # RUB
        "USDRUB", "EURRUB", "CNHRUB",
    ]

    rows = 3
    dummy: dict[str, list] = {}
    for field in OHLCV_FIELDS:
        for pair in all_pairs:
            dummy[f"{field}_{pair}"] = [1.0] * rows

    df = pd.DataFrame(dummy)
    calc = CurrencyIndexCalculator(df)

    # --- describe ---
    print("\n\n=== INDEX DESCRIPTIONS ===")
    CurrencyIndexCalculator.describe()

    # --- all indices: when all pairs = 1.0, close should equal scalar ---
    print("\n\n=== CLOSE VALUES (all pairs = 1.0  →  should equal scalar) ===")
    all_idx = calc.calculate_indices()
    for idx_name, ohlcv in all_idx.items():
        scalar = INDEX_DEFINITIONS[idx_name]["scalar"]
        close_val = round(ohlcv["close"].iloc[0], 6)
        status = "✓" if abs(close_val - scalar) < 1e-6 else "✗"
        print(f"  {status}  {idx_name:<8}  close={close_val}  (expected {scalar})")

    # --- subset + flat DataFrame ---
    print("\n\n=== FLAT DATAFRAME (JPY / CNY / RUB) ===")
    print(calc.to_dataframe(["JPY", "CNY", "RUB"]).to_string())

    # --- single index ---
    print("\n\n=== SINGLE INDEX (Dollar) ===")
    dollar = calc.calculate_indices("Dollar")
    for field, series in dollar["Dollar"].items():
        print(f"  {field}: {series.tolist()}")

    # --- bad index name raises cleanly ---
    print("\n\n=== INVALID INDEX NAME ===")
    try:
        calc.calculate_indices("GBP")
    except ValueError as exc:
        print(f"  Caught expected ValueError: {exc}")'''

# ---------------------------------------------------------------------------
# Currency Strength Matrix
# ---------------------------------------------------------------------------

def _csm_rolling_pct_change(s: pd.Series, lookback: int) -> pd.Series:
    """
    Port of rollingPctChange() from divergence-chart-scale.ts.

    Each bar = ((close - close[i-lookback]) / close[i-lookback]) * 100.
    Bars within the first `lookback` positions return NaN (no prior reference).
    Only emits a value when the reference bar is a valid positive price.
    """
    ref = s.shift(lookback)
    pct = ((s - ref) / ref) * 100.0
    valid = (ref > 0) & np.isfinite(s) & np.isfinite(ref)
    return pct.where(valid, np.nan)


def _csm_rolling_zscore_clamp(pct: pd.Series, lookback: int, clamp: float = 3.0) -> pd.Series:
    """
    Port of rollingZScore() + clamp from divergence-chart-scale.ts.

    Backward-only rolling Z-score with an expanding warm-up window:
    starts emitting once MIN_SAMPLES = min(lookback, 5) valid bars are present,
    then grows up to `lookback` bars.  Uses ddof=1 to match the JS
    Math.sqrt(sqSum / (count - 1)) formula.

    Output is clamped to [-1, +1] via z / clamp (matching ZSCORE_CLAMP = 3).
    When std == 0 (flat segment), returns 0 instead of NaN/inf.
    Reuses the same lookback as rolling-% change so the fast window
    stays genuinely fast all the way through normalisation.
    """
    MIN_SAMPLES = min(lookback, 5)
    roll = pct.rolling(window=lookback, min_periods=MIN_SAMPLES)
    mean = roll.mean()
    std  = roll.std(ddof=1)
    z = (pct - mean) / std
    z = z.where(std > 0, 0.0)
    return (z / clamp).clip(-1.0, 1.0)


def calculate_currency_strength_matrix(
    asset_close: pd.Series,
    dxy_close: pd.Series,
    fast_period: int = 20,
    slow_period: int = 100,
    zscore_clamp: float = 3.0,
) -> pd.DataFrame:
    """
    Currency Strength Matrix (CSM)
    ==============================
    Faithful Python port of the frontend divergence-chart-scale.ts pipeline,
    producing the same normalised series and histograms that the Strength Matrix
    divergence chart displays - with identical non-repainting guarantees.

    Pipeline per period window (fast / slow):
      1. Rolling % change  : (close[i] - close[i-w]) / close[i-w] * 100
      2. Rolling Z-score   : backward-only, expanding from MIN_SAMPLES=5,
                             same w as step 1 (keeps fast genuinely fast)
      3. Clamp             : z / zscore_clamp, clipped to [-1, +1]
      4. Histogram         : asset_norm - dxy_norm

    Parameters
    ----------
    asset_close  : pd.Series  main symbol close prices  (column 'close')
    dxy_close    : pd.Series  Dollar index close prices  (column 'Dollar_close')
    fast_period  : int   fast lookback window   (default 20  = frontend DEFAULT_FAST_WINDOW)
    slow_period  : int   slow lookback window   (default 100 = frontend DEFAULT_SLOW_WINDOW)
    zscore_clamp : float  std-devs mapping to +-1 (default 3.0 = frontend ZSCORE_CLAMP)

    Returns
    -------
    pd.DataFrame with columns (same index as inputs):
        CSM_asset_norm_fast   -- asset normalised at fast window
        CSM_dxy_norm_fast     -- DXY normalised at fast window
        CSM_asset_norm_slow   -- asset normalised at slow window
        CSM_dxy_norm_slow     -- DXY normalised at slow window
        CSM_histogram_fast    = asset_norm_fast  - dxy_norm_fast
        CSM_histogram_slow    = asset_norm_slow  - dxy_norm_slow
    """
    idx = asset_close.index

    # Step 1: Rolling % change
    asset_pct_fast = _csm_rolling_pct_change(asset_close, fast_period)
    dxy_pct_fast   = _csm_rolling_pct_change(dxy_close,   fast_period)
    asset_pct_slow = _csm_rolling_pct_change(asset_close, slow_period)
    dxy_pct_slow   = _csm_rolling_pct_change(dxy_close,   slow_period)

    # Steps 2 & 3: Rolling Z-score + clamp
    asset_norm_fast = _csm_rolling_zscore_clamp(asset_pct_fast, fast_period, zscore_clamp)
    dxy_norm_fast   = _csm_rolling_zscore_clamp(dxy_pct_fast,   fast_period, zscore_clamp)
    asset_norm_slow = _csm_rolling_zscore_clamp(asset_pct_slow, slow_period, zscore_clamp)
    dxy_norm_slow   = _csm_rolling_zscore_clamp(dxy_pct_slow,   slow_period, zscore_clamp)

    # Step 4: Histograms (how much asset diverges from DXY)
    hist_fast = asset_norm_fast - dxy_norm_fast
    hist_slow = asset_norm_slow - dxy_norm_slow

    return pd.DataFrame(
        {
            "CSM_asset_norm_fast": asset_norm_fast.values,
            "CSM_dxy_norm_fast":   dxy_norm_fast.values,
            "CSM_asset_norm_slow": asset_norm_slow.values,
            "CSM_dxy_norm_slow":   dxy_norm_slow.values,
            "CSM_histogram_fast":  hist_fast.values,
            "CSM_histogram_slow":  hist_slow.values,
        },
        index=idx,
        dtype=np.float32,
    )
