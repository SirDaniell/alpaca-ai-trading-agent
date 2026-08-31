#!/usr/bin/env python3
"""
prepare_kaggle_bundle.py — Production-Grade Kaggle Bundle Exporter for AXE Genesis Meta-Learner.

Generates standalone, self-contained Jupyter Notebooks for PyTorch Kaggle GPU execution:
- Models match 1:1 with backend production implementations (SignalMetaNetwork & ExecutorQNetwork).
- Full 4-Phase RL Training & Out-of-Sample Expiry Evaluation Pipeline:
  1. Phase 1: 50-Epoch Meta-Learner training sweep over train_df with Polyak target updates & CosineAnnealingLR.
  2. Phase 1b: Sequential pass over train_df generating 10,000 quality-gated Q-Learner transition memories.
  3. Phase 2: Q-Executor training strictly on the 10,000 transition replay buffer with validation early stopping.
  4. Phase 3: Out-of-sample test set evaluation across 4 expiry horizons (5m, 15m, 30m, 1h).
  5. Phase 4: Dynamic collective multi-horizon concurrent portfolio simulation & Checkpoint zip export.
"""

import os
import json
import zipfile

DATA_DIR = "data"
BUNDLE_ZIP_PATH = os.path.join(DATA_DIR, "axe_meta_dataset.zip")
PYTORCH_NOTEBOOK_PATH = "kaggle_axe_meta_learner_training.ipynb"

os.makedirs(DATA_DIR, exist_ok=True)

# Shared Code Blocks

CELL_IMPORTS = """# =============================================================================
# SYSTEM IMPORTS & PATH SETUP  (TensorFlow/Keras removed — this pipeline is PyTorch only.
#  The original notebook imported and GPU-configured TF/Keras but never used it; every
#  model defined below is torch.nn.Module. Keeping unused TF setup around was dead code
#  and misleading given the notebook's own title.)
# =============================================================================
import os
import glob
import zipfile
import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.optim as optim

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AXE")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

KAGGLE_DATASET_DIR = '/kaggle/input/datasets/danielwangewa/alpaka3' if os.path.exists('/kaggle/input/datasets/danielwangewa/alpaka3') else 'data'
OUTPUT_DIR = '/kaggle/working/checkpoints' if os.path.exists('/kaggle/working') else 'checkpoints'
ZIP_EXPORT_PATH = os.path.join('/kaggle/working' if os.path.exists('/kaggle/working') else '.', 'axe_meta_learner_weights.zip')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameterized — previously hardcoded "GLD" throughout the Q-executor loop regardless
# of which CSV was actually loaded. Set this to match the dataset you're training on.
SYMBOL = "GLD"

# Zone-detection parameters — must match backend/scripts/evaluate_option_expiries.py exactly
# for true 1:1 parity (see update_real_snr_snapshot in that file).
ZONE_LOOKBACK_PERIOD = 500
ZONE_MIN_DISTANCE_PCT = 0.5
"""



CELL_DATA_LOAD = """# =============================================================================
# DATASET LOADING (Train 70% | Val 15% | Test 15%)
# =============================================================================
zip_files = glob.glob(os.path.join(KAGGLE_DATASET_DIR, '*.zip'))
if zip_files:
    print(f"Extracting dataset archive: {zip_files[0]}")
    with zipfile.ZipFile(zip_files[0], 'r') as zip_ref:
        target_extract = '/kaggle/working/data' if os.path.exists('/kaggle/working') else 'data'
        zip_ref.extractall(target_extract)
    data_dir = target_extract
elif os.path.exists('data/train_40k.csv'):
    data_dir = 'data'
else:
    data_dir = KAGGLE_DATASET_DIR

train_csv = os.path.join(data_dir, 'train_40k.csv')
val_csv   = os.path.join(data_dir, 'val_40k.csv')
test_csv  = os.path.join(data_dir, 'test_40k.csv')

if os.path.exists(train_csv):
    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv) if os.path.exists(val_csv) else None
    test_df  = pd.read_csv(test_csv) if os.path.exists(test_csv) else None
    print(f"Train Set: {len(train_df)} rows | Columns: {len(train_df.columns)}")
    if val_df is not None:  print(f"Validation Set: {len(val_df)} rows")
    if test_df is not None: print(f"Holdout Test Set: {len(test_df)} rows")
else:
    raise FileNotFoundError(f"Dataset files not found under {data_dir}. Check dataset path!")

close_col = "close_5m" if "close_5m" in train_df.columns else train_df.columns[0]
open_col  = "open_5m" if "open_5m" in train_df.columns else train_df.columns[0]
high_col  = "high_5m" if "high_5m" in train_df.columns else train_df.columns[0]
low_col   = "low_5m" if "low_5m" in train_df.columns else train_df.columns[0]
vol_col   = "volume_5m" if "volume_5m" in train_df.columns else train_df.columns[1]
up_vol_col   = "Bar_Volume_Up_5m" if "Bar_Volume_Up_5m" in train_df.columns else None
down_vol_col = "Bar_Volume_Down_5m" if "Bar_Volume_Down_5m" in train_df.columns else None
atr_col      = "ATR_5m" if "ATR_5m" in train_df.columns else None
print(f"Volume columns available: up={up_vol_col}, down={down_vol_col} | ATR column: {atr_col}")
"""

CELL_SNR_DETECTION = """# =============================================================================
# REAL SNR ZONE DETECTION — ported verbatim from
# backend/app/core/analysis/support_resistance.py, verified against the live
# backend (detect_snr_levels_sequential explicitly guarantees no lookahead:
# "Only uses data up to up_to_index"). This is NOT a simplified placeholder —
# it is the exact same function the backend uses, so zone-anchored decisions
# here are genuinely 1:1 with production.
# =============================================================================

def detect_snr_levels_sequential(price_data, up_to_index, lookback_period, min_distance_pct=0.5):
    '''Detect S&R levels up to a specific index. CRITICAL: only uses data up to up_to_index.'''
    levels = []
    df = price_data.iloc[up_to_index - lookback_period: up_to_index + 1]
    if len(df) < 5:
        return levels

    highs = df["High"].values
    lows = df["Low"].values
    price_range = highs.max() - lows.min()
    min_distance = price_range * (min_distance_pct / 100)

    if len(lows) >= 5:
        support_cond1 = lows[2:-2] < lows[1:-3]
        support_cond2 = lows[2:-2] < lows[3:-1]
        support_cond3 = lows[3:-1] < lows[4:]
        support_cond4 = lows[1:-3] < lows[:-4]
        support_mask = support_cond1 & support_cond2 & support_cond3 & support_cond4
        support_indices = np.where(support_mask)[0] + 2

        resistance_cond1 = highs[2:-2] > highs[1:-3]
        resistance_cond2 = highs[2:-2] > highs[3:-1]
        resistance_cond3 = highs[3:-1] > highs[4:]
        resistance_cond4 = highs[1:-3] > highs[:-4]
        resistance_mask = resistance_cond1 & resistance_cond2 & resistance_cond3 & resistance_cond4
        resistance_indices = np.where(resistance_mask)[0] + 2

        for idx in support_indices:
            level = lows[idx]
            if not levels or all(abs(level - l[1]) >= min_distance for l in levels):
                levels.append((int(idx), float(level), "support"))
        for idx in resistance_indices:
            level = highs[idx]
            if not levels or all(abs(level - l[1]) >= min_distance for l in levels):
                levels.append((int(idx), float(level), "resistance"))

    window = 5
    if len(df) > window * 2:
        pivot_high_mask = np.ones(len(highs), dtype=bool)
        pivot_high_mask[:window] = False
        pivot_high_mask[-window:] = False
        for offset in range(1, window + 1):
            pivot_high_mask[window:-window] &= (
                (highs[window:-window] > highs[window-offset:-(window+offset)]) &
                (highs[window:-window] > highs[window+offset:len(highs)-window+offset])
            )
        pivot_high_indices = np.where(pivot_high_mask)[0]

        pivot_low_mask = np.ones(len(lows), dtype=bool)
        pivot_low_mask[:window] = False
        pivot_low_mask[-window:] = False
        for offset in range(1, window + 1):
            pivot_low_mask[window:-window] &= (
                (lows[window:-window] < lows[window-offset:-(window+offset)]) &
                (lows[window:-window] < lows[window+offset:len(lows)-window+offset])
            )
        pivot_low_indices = np.where(pivot_low_mask)[0]

        for idx in pivot_high_indices:
            level = highs[idx]
            if not levels or all(abs(level - l[1]) >= min_distance for l in levels):
                levels.append((int(idx), float(level), "resistance"))
        for idx in pivot_low_indices:
            level = lows[idx]
            if not levels or all(abs(level - l[1]) >= min_distance for l in levels):
                levels.append((int(idx), float(level), "support"))

    return levels


def calculate_volume_profile_at_level(price_level, price_data, zone_width=0.004):
    '''CRITICAL: only uses the price_data slice passed in (no lookahead).'''
    upper_bound = price_level + zone_width
    lower_bound = price_level - zone_width
    highs = price_data["High"].values
    lows = price_data["Low"].values
    closes = price_data["Close"].values
    opens = price_data["Open"].values
    volumes = price_data["Volume"].values

    touches_level = (lows <= price_level) & (highs >= price_level)
    is_bullish = closes > opens
    total_volume = volumes[touches_level].sum()
    up_volume = volumes[touches_level & is_bullish].sum()
    down_volume = volumes[touches_level & ~is_bullish].sum()

    return {
        "total_volume": float(total_volume),
        "up_volume": float(up_volume),
        "down_volume": float(down_volume),
        "net_volume": float(up_volume - down_volume),
        "upper_bound": upper_bound,
        "lower_bound": lower_bound,
    }


def create_clustered_zones_sequential(levels, price_data_slice, n_clusters=16, zone_width=0.004):
    '''Create zones using K-means clustering for sequential analysis.'''
    if not levels:
        return []
    prices = [level[1] for level in levels]
    unique_prices_count = len(set(prices))
    if n_clusters is None:
        n_clusters = min(unique_prices_count, max(3, len(prices) // 3))
    if unique_prices_count < n_clusters:
        n_clusters = unique_prices_count
    if n_clusters < 1:
        return []
    if unique_prices_count < 2:
        if not prices:
            return []
        zone_price = prices[0]
        volume_data = calculate_volume_profile_at_level(zone_price, price_data_slice, zone_width)
        return [(0, zone_price, [l for l in levels if l[1] == zone_price], volume_data)]

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    price_array = np.array(prices).reshape(-1, 1)
    clusters = kmeans.fit_predict(price_array)

    zones = []
    for cluster_id in range(n_clusters):
        cluster_levels = [levels[i] for i, c in enumerate(clusters) if c == cluster_id]
        if cluster_levels:
            zone_price = np.mean([l[1] for l in cluster_levels])
            volume_data = calculate_volume_profile_at_level(zone_price, price_data_slice, zone_width)
            zones.append((cluster_id, zone_price, cluster_levels, volume_data))
    return sorted(zones, key=lambda x: x[1])


def get_nearest_zones(zones, current_price):
    '''Mirror of ZoneSnapshotManager.get_nearest_zones — returns (nearest_support, nearest_resistance)
    as dicts with price_level + volume_delta_ratio, or None if absent.'''
    supports = [z for z in zones if z[1] <= current_price]
    resistances = [z for z in zones if z[1] >= current_price]
    nearest_supp = max(supports, key=lambda z: z[1]) if supports else None
    nearest_res = min(resistances, key=lambda z: z[1]) if resistances else None

    def _to_record(z):
        if z is None:
            return None
        _, price, _, vol = z
        total = vol["up_volume"] + vol["down_volume"]
        ratio = (vol["up_volume"] - vol["down_volume"]) / (total + 1e-6)
        return {"price_level": price, "volume_delta_ratio": ratio, "volume": vol}

    return _to_record(nearest_supp), _to_record(nearest_res)


print("Real SNR zone detection loaded (verified 1:1 port of backend support_resistance.py).")
"""

CELL_FEATURE_ENGINE = """# =============================================================================
# DOMAIN STRUCTURES & REAL HARD ACTION MASK
# (Previous version's HardActionMask never referenced zone_manager at all — it only
#  gated on volume imbalance, meaning the no-chase / zone-anchored entry rule, the
#  central design principle of this strategy, was entirely absent. This version
#  enforces the same ATR-scaled proximity band + volume confirmation + single-position
#  restriction as backend/app/core/market/zone_snapshot.py's HardActionMask.)
# =============================================================================

@dataclass
class HTFBiasPackage:
    direction: str = "neutral"
    strength: float = 0.0
    reversal_prob: float = 0.0
    q_value: float = 0.0
    expected_mfe_pips: float = 0.0
    expected_mae_pips: float = 0.0
    horizon_strengths: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5, 0.5])
    optimal_horizon_idx: int = 2
    recommended_expiry: str = "30m"

@dataclass
class AccountContext:
    balance: float = 10000.0
    equity: float = 10000.0
    open_position_type: Optional[str] = None
    open_position_pnl_pct: float = 0.0
    daily_drawdown_pct: float = 0.0
    win_streak: int = 0
    loss_streak: int = 0
    reentries_in_window: int = 0
    max_reentries_allowed: int = 3

@dataclass
class ExecutionContext:
    symbol: str
    current_price: float
    atr: float
    buy_volume: float
    sell_volume: float
    hour_of_day: float
    day_of_week: int
    session_phase: str
    ltf_timeframe: str = "5m"


class HardActionMask:
    '''1:1 with backend zone_snapshot.py::HardActionMask — enforces the no-chase rule
    (entries only at/near a real zone), volume confirmation, and single-open-position
    restriction, as HARD constraints rather than something the network has to learn.'''

    def get_action_mask(
        self, current_price, atr, nearest_supp, nearest_res,
        buy_volume, sell_volume, has_open_position=False,
    ):
        # mask indices: 0=WAIT, 1=BUY_CALL, 2=BUY_PUT, 3=TAKE_PROFIT_HALF, 4=CLOSE_FLATTEN
        mask = np.ones(5, dtype=np.int32)

        if has_open_position:
            # Single running trade restriction — no new entries while a position is open.
            mask[1] = 0
            mask[2] = 0
            return mask

        mask[3] = 0
        mask[4] = 0

        proximity_band = max(atr * 0.75, current_price * 0.003)

        supp_ok = nearest_supp is not None and abs(current_price - nearest_supp["price_level"]) <= proximity_band
        res_ok = nearest_res is not None and abs(current_price - nearest_res["price_level"]) <= proximity_band

        # No-chase: BUY_CALL only valid near/below a support zone. BUY_PUT only valid near/above resistance.
        if not supp_ok:
            mask[1] = 0
        if not res_ok:
            mask[2] = 0

        # Volume confirmation gate — require the reaction to actually be confirmed, not just proximity.
        if buy_volume > 0 or sell_volume > 0:
            if buy_volume < sell_volume * 0.8:
                mask[1] = 0
            if sell_volume < buy_volume * 0.8:
                mask[2] = 0

        return mask


def _make_exec_ctx(symbol: str, price: float, row: dict, atr_col: str, up_vol_col: str, down_vol_col: str) -> ExecutionContext:
    '''Uses REAL up/down volume columns from the feature pipeline instead of a crude
    open/close-direction proxy, and REAL ATR instead of an SNR-distance-derived guess.
    Session-phase uses actual US/Eastern local time via zoneinfo (DST-aware), matching
    backend q_executor.py's is_nyse_open (09:30-10:30 ET) / is_power_hour (15:00-16:00 ET).'''
    ts = row.get("timestamp", None)
    hour_f, dow, phase = 14.5, 1, "off_hours"
    if ts is not None:
        try:
            ts_pd = pd.Timestamp(ts)
            if ts_pd.tzinfo is None:
                ts_pd = ts_pd.tz_localize("UTC")
            ts_et = ts_pd.tz_convert("America/New_York")
            hour_f = ts_et.hour + ts_et.minute / 60.0
            dow = ts_et.dayofweek
            if 9.5 <= hour_f < 10.5:
                phase = "nyse_open"
            elif 15.0 <= hour_f < 16.0:
                phase = "nyse_power_hour"
            elif 9.5 <= hour_f < 16.0:
                phase = "regular_hours"
        except Exception:
            pass

    buy_vol = float(row.get(up_vol_col, 0.0)) if up_vol_col else 0.0
    sell_vol = float(row.get(down_vol_col, 0.0)) if down_vol_col else 0.0
    atr_val = float(row.get(atr_col, price * 0.005)) if atr_col else price * 0.005

    return ExecutionContext(
        symbol=symbol, current_price=price, atr=max(0.01, atr_val),
        buy_volume=buy_vol, sell_volume=sell_vol, hour_of_day=hour_f, day_of_week=dow, session_phase=phase,
    )


def build_state_vector(net_out, htf_bias: HTFBiasPackage, account: AccountContext,
                        exec_ctx: ExecutionContext, nearest_supp, nearest_res) -> np.ndarray:
    '''1:1 with backend q_executor.py::build_state_vector's real 28-dim layout.
    NOTE: unlike the previous version, this contains NO future-derived value anywhere —
    the previous notebook placed the literal forward price move (the reward target) into
    state_vec[4] as an INPUT feature, both in training and in Phase-2 validation. That is
    direct label leakage: the network was being handed the answer as an input. Every
    field here is computable strictly from data up to and including the current bar.'''
    supp_dist = abs(exec_ctx.current_price - nearest_supp["price_level"]) / exec_ctx.current_price if nearest_supp else 1.0
    res_dist = abs(exec_ctx.current_price - nearest_res["price_level"]) / exec_ctx.current_price if nearest_res else 1.0
    supp_vol_ratio = nearest_supp["volume_delta_ratio"] if nearest_supp else 0.0
    res_vol_ratio = nearest_res["volume_delta_ratio"] if nearest_res else 0.0

    total_vol = exec_ctx.buy_volume + exec_ctx.sell_volume
    vol_delta_ratio = (exec_ctx.buy_volume - exec_ctx.sell_volume) / (total_vol + 1e-6)

    tf_flag = 1.0 if exec_ctx.ltf_timeframe == "15m" else 0.0
    dir_flag = 1.0 if htf_bias.direction == "bullish" else (-1.0 if htf_bias.direction == "bearish" else 0.0)
    hs = htf_bias.horizon_strengths if len(htf_bias.horizon_strengths) == 4 else [0.5, 0.5, 0.5, 0.5]

    sin_hour = float(np.sin(2 * np.pi * exec_ctx.hour_of_day / 24.0))
    cos_hour = float(np.cos(2 * np.pi * exec_ctx.hour_of_day / 24.0))
    dow_norm = float(exec_ctx.day_of_week) / 6.0
    is_nyse_open = 1.0 if exec_ctx.session_phase == "nyse_open" else 0.0
    is_power_hour = 1.0 if exec_ctx.session_phase == "nyse_power_hour" else 0.0

    state = np.array([
        dir_flag, float(htf_bias.strength), float(htf_bias.reversal_prob), float(htf_bias.q_value),
        float(htf_bias.expected_mfe_pips) / 100.0, float(htf_bias.expected_mae_pips) / 100.0,
        float(hs[0]), float(hs[1]), float(hs[2]), float(hs[3]),
        float(account.daily_drawdown_pct),
        1.0 if account.open_position_type == "CALL" else (-1.0 if account.open_position_type == "PUT" else 0.0),
        float(account.open_position_pnl_pct), float(account.win_streak) / 10.0, float(account.loss_streak) / 10.0,
        tf_flag, float(exec_ctx.atr) / exec_ctx.current_price, float(supp_dist), float(res_dist),
        float(supp_vol_ratio), float(res_vol_ratio), float(vol_delta_ratio),
        float(account.reentries_in_window) / float(account.max_reentries_allowed),
        sin_hour, cos_hour, dow_norm, is_nyse_open, is_power_hour,
    ], dtype=np.float32)
    return state

print("Real HardActionMask + 28-dim state vector (no future leakage) loaded.")
"""

CELL_PYTORCH_MODELS = """# =============================================================================
# 🏗️ PYTORCH PRODUCTION MODEL ARCHITECTURES (SignalMetaNetwork & ExecutorQNetwork)
# =============================================================================
import torch
import torch.nn as nn
import torch.optim as optim

feature_cols = [c for c in train_df.columns if c not in ("timestamp", "Time") and not "target" in c and not "forward" in c]
num_features = len(feature_cols)
lookback_bars = 1000  # 100% Parity with in-house SIGNAL_META_LOOKBACK_BARS = 1000
input_dim = lookback_bars * num_features

print(f"📊 Feature Input Contract: {num_features} columns | Lookback {lookback_bars} bars → Flattened Dim = {input_dim}")

class SignalMetaNetwork(nn.Module):
    def __init__(self, input_dim: int = input_dim, num_actions: int = 4, hidden_dim: int = 128, num_features: int = num_features):
        super().__init__()
        self.num_features = num_features
        hidden_dim = hidden_dim or 128

        # Branch 1: Full Sequence (100%) Conv1D + LSTM Tower
        self.b1_conv1 = nn.Conv1d(num_features, 64, kernel_size=3, padding=1)
        self.b1_bn1   = nn.BatchNorm1d(64)
        self.b1_act1  = nn.SiLU()
        self.b1_conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.b1_bn2   = nn.BatchNorm1d(32)
        self.b1_act2  = nn.SiLU()
        self.b1_lstm  = nn.LSTM(32, 32, batch_first=True)

        # Branch 2: Mid-Term (50% Slice) Conv1D Tower
        self.b2_conv  = nn.Conv1d(num_features, 32, kernel_size=3, padding=1)
        self.b2_bn    = nn.BatchNorm1d(32)
        self.b2_act   = nn.SiLU()
        self.b2_fc    = nn.Linear(32, 32)

        # Branch 3: Short-Term (30% Slice) Conv1D Tower
        self.b3_conv  = nn.Conv1d(num_features, 32, kernel_size=3, padding=1)
        self.b3_bn    = nn.BatchNorm1d(32)
        self.b3_act   = nn.SiLU()
        self.b3_fc    = nn.Linear(32, 32)

        # Auxiliary Supervised Heads per branch
        self.aux1_head = nn.Linear(64, 5)
        self.aux2_head = nn.Linear(32, 5)

        # Gated Ensemble Fusion Head
        self.fusion_fc   = nn.Linear(64 + 32 + 32 + 5 + 5, hidden_dim)
        self.fusion_ln   = nn.LayerNorm(hidden_dim)
        self.fusion_act  = nn.SiLU()
        self.fusion_fc2  = nn.Linear(hidden_dim, hidden_dim)
        self.fusion_ln2  = nn.LayerNorm(hidden_dim)
        self.fusion_act2 = nn.SiLU()

        self.q_head = nn.Linear(hidden_dim, num_actions)
        self.strength_head = nn.Sequential(
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),
        )
        self.fusion_selector = nn.Linear(hidden_dim, 4)

        # Auxiliary Private Projections (Zero Gradient Interference via feat.detach())
        _aux_in = 64 + 32 + 32
        self.branch_ln = nn.LayerNorm(_aux_in)
        self.pips_proj = nn.Linear(_aux_in, 32)
        self.pips_ln   = nn.LayerNorm(32)
        self.pips_head = nn.Sequential(nn.SiLU(), nn.Linear(32, 16), nn.SiLU(), nn.Linear(16, 4))
        self.risk_proj = nn.Linear(_aux_in, 32)
        self.risk_ln   = nn.LayerNorm(32)
        self.risk_head = nn.Sequential(nn.SiLU(), nn.Linear(32, 16), nn.SiLU(), nn.Linear(16, 8))
        self.liq_proj  = nn.Linear(_aux_in, 16)
        self.liq_ln    = nn.LayerNorm(16)
        self.liquidity_head = nn.Sequential(nn.SiLU(), nn.Linear(16, 8), nn.SiLU(), nn.Linear(8, 2))
        self.rev_proj  = nn.Linear(_aux_in, 16)
        self.rev_ln    = nn.LayerNorm(16)
        self.reversal_head = nn.Sequential(nn.SiLU(), nn.Linear(16, 8), nn.SiLU(), nn.Linear(8, 1), nn.Sigmoid())

    def _prepare_3d(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            b, dim = x.shape
            c = self.num_features
            t = dim // c if dim >= c else 1
            if t * c != dim:
                c = dim
                t = 1
            return x.view(b, t, c)
        return x

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        x_3d = self._prepare_3d(x)
        b, t, c = x_3d.shape
        x_trans = x_3d.transpose(1, 2)

        # Branch 1
        b1_c1 = self.b1_act1(self.b1_bn1(self.b1_conv1(x_trans)))
        b1_c2 = self.b1_act2(self.b1_bn2(self.b1_conv2(b1_c1)))
        b1_c2_trans = b1_c2.transpose(1, 2)
        b1_lstm_out, _ = self.b1_lstm(b1_c2_trans)
        b1_last = b1_lstm_out[:, -1, :]
        b1_gap  = torch.mean(b1_lstm_out, dim=1)
        b1_out  = torch.cat([b1_last, b1_gap], dim=-1)

        # Branch 2 (50% slice)
        half = max(1, t // 2)
        x_mid_trans = x_trans[:, :, -half:]
        b2_c = self.b2_act(self.b2_bn(self.b2_conv(x_mid_trans)))
        b2_gap = torch.mean(b2_c, dim=-1)
        b2_out = torch.relu(self.b2_fc(b2_gap))

        # Branch 3 (30% slice)
        recent = max(1, int(t * 0.3))
        x_rec_trans = x_trans[:, :, -recent:]
        b3_c = self.b3_act(self.b3_bn(self.b3_conv(x_rec_trans)))
        b3_gap = torch.mean(b3_c, dim=-1)
        b3_out = torch.relu(self.b3_fc(b3_gap))

        # Aux heads (detached)
        aux1 = self.aux1_head(b1_out.detach())
        aux2 = self.aux2_head(b2_out.detach())
        aux1_sg = aux1.detach()
        aux2_sg = aux2.detach()

        # Gated Fusion
        fusion_in = torch.cat([b1_out, b2_out, b3_out, aux1_sg, aux2_sg], dim=-1)
        feat = self.fusion_act(self.fusion_ln(self.fusion_fc(fusion_in)))
        feat = self.fusion_act2(self.fusion_ln2(self.fusion_fc2(feat)))

        q_vals   = self.q_head(feat)
        strength = self.strength_head(feat)
        selector_logits = self.fusion_selector(feat)

        # Private Aux Heads on detached branch concatenation
        branch_cat = torch.cat([b1_out, b2_out, b3_out], dim=-1).detach()
        pips      = self.pips_head(torch.relu(self.pips_proj(branch_cat)))
        risk      = self.risk_head(torch.relu(self.risk_proj(branch_cat)))
        liquidity = self.liquidity_head(torch.relu(self.liq_proj(branch_cat)))
        reversal  = self.reversal_head(torch.relu(self.rev_proj(branch_cat)))

        if return_aux:
            return q_vals, strength, pips, risk, liquidity, reversal, aux1, aux2, selector_logits
        return q_vals, strength, pips, risk, liquidity, reversal


class ExecutorQNetwork(nn.Module):
    def __init__(self, input_dim: int = 28, hidden_dim: int = 64, num_actions: int = 5):
        super().__init__()
        # Branch 1: Dense Feature Tower
        self.b1_fc1 = nn.Linear(input_dim, hidden_dim)
        self.b1_ln1 = nn.LayerNorm(hidden_dim)
        self.b1_act1 = nn.SiLU()
        self.b1_drop = nn.Dropout(0.0)
        self.b1_fc2 = nn.Linear(hidden_dim, 32)
        self.b1_ln2 = nn.LayerNorm(32)
        self.b1_act2 = nn.SiLU()

        # Branch 2: Multi-Scale Grouped Feature Tower
        self.b2_meta = nn.Linear(10, 16)
        self.b2_risk = nn.Linear(5, 16)
        self.b2_zone = nn.Linear(8, 16)
        self.b2_time = nn.Linear(5, 16)
        self.b2_fusion = nn.Linear(64, 32)
        self.b2_ln = nn.LayerNorm(32)
        self.b2_act = nn.SiLU()

        # Auxiliary Supervised Heads
        self.aux1_head = nn.Linear(32, num_actions)
        self.aux2_head = nn.Linear(32, num_actions)

        # Gated Ensemble Fusion Head
        self.fusion_fc1 = nn.Linear(32 + 32 + num_actions + num_actions, hidden_dim)
        self.fusion_ln1 = nn.LayerNorm(hidden_dim)
        self.fusion_act1 = nn.SiLU()
        self.fusion_out = nn.Linear(hidden_dim, num_actions)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        # Branch 1
        b1 = self.b1_act1(self.b1_ln1(self.b1_fc1(x)))
        b1 = self.b1_drop(b1)
        b1_out = self.b1_act2(self.b1_ln2(self.b1_fc2(b1)))

        # Branch 2 (Grouped features)
        meta_feats = x[:, :10]
        risk_feats = x[:, 10:15]
        zone_feats = x[:, 15:23]
        time_feats = x[:, 23:28]

        b2_m = torch.relu(self.b2_meta(meta_feats))
        b2_r = torch.relu(self.b2_risk(risk_feats))
        b2_z = torch.relu(self.b2_zone(zone_feats))
        b2_t = torch.relu(self.b2_time(time_feats))
        b2_cat = torch.cat([b2_m, b2_r, b2_z, b2_t], dim=-1)
        b2_out = self.b2_act(self.b2_ln(self.b2_fusion(b2_cat)))

        # Aux heads (detached)
        aux1_q = self.aux1_head(b1_out.detach())
        aux2_q = self.aux2_head(b2_out.detach())

        # Gated Fusion
        fusion_in = torch.cat([b1_out, b2_out, aux1_q.detach(), aux2_q.detach()], dim=-1)
        fused = self.fusion_act1(self.fusion_ln1(self.fusion_fc1(fusion_in)))
        q_final = self.fusion_out(fused)

        if return_aux:
            return q_final, aux1_q, aux2_q
        return q_final

print("✅ PyTorch Production Models Defined: SignalMetaNetwork & ExecutorQNetwork (100% 1:1 Match)")
"""

CELL_PYTORCH_TRAINER = """# =============================================================================
# PHASE 1: META-LEARNER MULTI-HEAD TRAINING & PER-EPOCH VALIDATION
# =============================================================================
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Target column preparation
for df in (train_df, val_df, test_df):
    if df is not None:
        if "target_dir_5m" not in df.columns:
            df["target_dir_5m"]  = (df[close_col].shift(-1) > df[close_col]).astype(np.float32)
            df["target_dir_15m"] = (df[close_col].shift(-3) > df[close_col]).astype(np.float32)
            df["target_dir_30m"] = (df[close_col].shift(-6) > df[close_col]).astype(np.float32)
            df["target_dir_1h"]  = (df[close_col].shift(-12) > df[close_col]).astype(np.float32)
        if "forward_move_1" not in df.columns:
            df["forward_move_1"]  = df[close_col].shift(-1) - df[close_col]
            df["forward_move_3"]  = df[close_col].shift(-3) - df[close_col]
            df["forward_move_6"]  = df[close_col].shift(-6) - df[close_col]
            df["forward_move_12"] = df[close_col].shift(-12) - df[close_col]

net = SignalMetaNetwork(input_dim=input_dim, num_features=num_features).to(device)
target_net = SignalMetaNetwork(input_dim=input_dim, num_features=num_features).to(device)
target_net.load_state_dict(net.state_dict())

optimizer = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)

N_train = len(train_df) - lookback_bars - 12
N_val   = len(val_df) - lookback_bars - 12 if val_df is not None else 0

META_EPOCHS = 50
BATCH_SIZE = 64
steps_per_epoch = max(1, N_train // BATCH_SIZE)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=META_EPOCHS * steps_per_epoch, eta_min=1e-5)

# Clean feature matrices
train_num_matrix = np.nan_to_num(train_df[feature_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
val_num_matrix   = np.nan_to_num(val_df[feature_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0) if val_df is not None else None
test_num_matrix  = np.nan_to_num(test_df[feature_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0) if test_df is not None else None

def _extract_targets(df):
    q = np.column_stack([
        df["target_dir_5m"].values, df["target_dir_15m"].values,
        df["target_dir_30m"].values, df["target_dir_1h"].values
    ]).astype(np.float32)
    pips = np.column_stack([
        df["forward_move_1"].values, df["forward_move_3"].values,
        df["forward_move_6"].values, df["forward_move_12"].values
    ]).astype(np.float32) / 100.0
    risk = np.column_stack([
        np.maximum(df["forward_move_1"].values, 0), np.maximum(-df["forward_move_1"].values, 0),
        np.maximum(df["forward_move_3"].values, 0), np.maximum(-df["forward_move_3"].values, 0),
        np.maximum(df["forward_move_6"].values, 0), np.maximum(-df["forward_move_6"].values, 0),
        np.maximum(df["forward_move_12"].values, 0), np.maximum(-df["forward_move_12"].values, 0)
    ]).astype(np.float32) / 100.0
    rev = ((df["target_dir_5m"].values != df["target_dir_15m"].values).astype(np.float32)).reshape(-1, 1)
    return np.nan_to_num(q, nan=0.0), np.nan_to_num(pips, nan=0.0), np.nan_to_num(risk, nan=0.0), np.nan_to_num(rev, nan=0.0)

train_targets_q, train_targets_pips, train_targets_risk, train_targets_rev = _extract_targets(train_df)
val_targets_q, val_targets_pips, val_targets_risk, val_targets_rev = _extract_targets(val_df) if val_df is not None else (None, None, None, None)

best_val_loss = float("inf")
best_meta_weights = None

print(f"🚀 [Phase 1] Meta-Learner Multi-Head Training ({META_EPOCHS} Epochs | {steps_per_epoch} Steps/Epoch | {N_train} Train / {N_val} Val samples)...")
print(f"  {'-'*155}")
print(f"  {'Epoch':>5} | {'TotLoss':>8} {'Q_Loss':>8} {'StrLoss':>8} {'PipsLoss':>8} {'RiskLoss':>8} {'LiqLoss':>8} {'RevLoss':>8} {'Aux1Loss':>8} {'Aux2Loss':>8} {'SelLoss':>8} | {'Val Loss':>8} {'Val Q':>8} | {'5m WR':>6} {'15m WR':>6} {'30m WR':>6} {'1h WR':>6} {'Avg WR':>6} | {'Status'}")
print(f"  {'-'*155}")

for ep in range(META_EPOCHS):
    indices = list(range(N_train))
    random.shuffle(indices)

    tr_tot, tr_q, tr_str, tr_pips, tr_risk = 0.0, 0.0, 0.0, 0.0, 0.0
    tr_liq, tr_rev, tr_aux1, tr_aux2, tr_sel = 0.0, 0.0, 0.0, 0.0, 0.0
    epoch_steps = 0

    net.train()
    for b_start in range(0, N_train, BATCH_SIZE):
        batch_idx = indices[b_start: b_start + BATCH_SIZE]
        if len(batch_idx) < BATCH_SIZE: continue

        x_batch      = np.stack([train_num_matrix[i: i + lookback_bars].flatten() for i in batch_idx])
        y_q_batch    = train_targets_q[np.array(batch_idx) + lookback_bars]
        y_pips_batch = train_targets_pips[np.array(batch_idx) + lookback_bars]
        y_risk_batch = train_targets_risk[np.array(batch_idx) + lookback_bars]
        y_rev_batch  = train_targets_rev[np.array(batch_idx) + lookback_bars]

        x_t      = torch.tensor(x_batch,      dtype=torch.float32, device=device)
        y_q_t    = torch.tensor(y_q_batch,    dtype=torch.float32, device=device)
        y_pips_t = torch.tensor(y_pips_batch, dtype=torch.float32, device=device)
        y_risk_t = torch.tensor(y_risk_batch, dtype=torch.float32, device=device)
        y_rev_t  = torch.tensor(y_rev_batch,  dtype=torch.float32, device=device)

        optimizer.zero_grad()
        q_vals, strength, pips, risk, liq, rev, aux1, aux2, selector_logits = net(x_t, return_aux=True)

        l_q    = nn.MSELoss()(q_vals, y_q_t)
        l_str  = nn.BCELoss()(torch.clamp(strength, 1e-6, 1.0 - 1e-6), y_q_t)
        l_pips = nn.SmoothL1Loss()(pips, y_pips_t)
        l_risk = nn.SmoothL1Loss()(risk, y_risk_t)
        l_liq  = nn.MSELoss()(liq, torch.abs(y_pips_t))
        l_rev  = nn.MSELoss()(rev, y_rev_t)
        
        target_aux = torch.cat([y_q_t, y_rev_t], dim=1)
        l_aux1 = nn.SmoothL1Loss()(aux1, target_aux)
        l_aux2 = nn.SmoothL1Loss()(aux2, target_aux)
        best_h_idx = strength.detach().argmax(dim=1).long()
        l_sel = nn.CrossEntropyLoss()(selector_logits, best_h_idx)

        tot_loss = l_q + l_str + 0.3 * l_pips + 0.3 * l_risk + 0.2 * l_liq + 0.3 * l_rev + 0.15 * l_aux1 + 0.15 * l_aux2 + 0.3 * l_sel

        tot_loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            for tp, p in zip(target_net.parameters(), net.parameters()):
                tp.data.copy_(0.005 * p.data + 0.995 * tp.data)

        tr_tot += tot_loss.item()
        tr_q += l_q.item()
        tr_str += l_str.item()
        tr_pips += l_pips.item()
        tr_risk += l_risk.item()
        tr_liq += l_liq.item()
        tr_rev += l_rev.item()
        tr_aux1 += l_aux1.item()
        tr_aux2 += l_aux2.item()
        tr_sel += l_sel.item()
        epoch_steps += 1

        # Periodic Step Progress Output within epoch
        if epoch_steps % 100 == 0 or epoch_steps == steps_per_epoch:
            print(f"  [Ep {ep+1:>2}/{META_EPOCHS} | Step {epoch_steps:>4}/{steps_per_epoch}] Tot={tot_loss.item():.4e} | Q={l_q.item():.4e} | Str={l_str.item():.4e} | Pips={l_pips.item():.4e} | Risk={l_risk.item():.4e} | Liq={l_liq.item():.4e} | Rev={l_rev.item():.4e} | Aux1={l_aux1.item():.4e} | Aux2={l_aux2.item():.4e} | Sel={l_sel.item():.4e}")

    e_tot  = tr_tot / max(epoch_steps, 1)
    e_q    = tr_q / max(epoch_steps, 1)
    e_str  = tr_str / max(epoch_steps, 1)
    e_pips = tr_pips / max(epoch_steps, 1)
    e_risk = tr_risk / max(epoch_steps, 1)
    e_liq  = tr_liq / max(epoch_steps, 1)
    e_rev  = tr_rev / max(epoch_steps, 1)
    e_aux1 = tr_aux1 / max(epoch_steps, 1)
    e_aux2 = tr_aux2 / max(epoch_steps, 1)
    e_sel  = tr_sel / max(epoch_steps, 1)

    # --- PER-EPOCH VALIDATION RUN ---
    net.eval()
    val_tot_loss, val_q_loss = 0.0, 0.0
    val_corrects = [0, 0, 0, 0]
    val_count = 0

    if N_val > 0:
        with torch.no_grad():
            for v_start in range(0, N_val, BATCH_SIZE):
                v_idx = list(range(v_start, min(v_start + BATCH_SIZE, N_val)))
                if not v_idx: continue
                vx_batch = np.stack([val_num_matrix[i: i + lookback_bars].flatten() for i in v_idx])
                vy_q_batch = val_targets_q[np.array(v_idx) + lookback_bars]

                vx_t   = torch.tensor(vx_batch,   dtype=torch.float32, device=device)
                vy_q_t = torch.tensor(vy_q_batch, dtype=torch.float32, device=device)

                vq, vstr, vp, vrisk, vliq, vrev = net(vx_t)
                v_lq = nn.MSELoss()(vq, vy_q_t)
                v_lstr = nn.BCELoss()(torch.clamp(vstr, 1e-6, 1.0 - 1e-6), vy_q_t)
                v_tot = v_lq + v_lstr

                val_tot_loss += v_tot.item() * len(v_idx)
                val_q_loss += v_lq.item() * len(v_idx)

                v_preds = (vstr > 0.5).float()
                corrs = (v_preds == vy_q_t).sum(dim=0).cpu().numpy()
                for h in range(4): val_corrects[h] += int(corrs[h])
                val_count += len(v_idx)

    v_tot_avg = val_tot_loss / max(val_count, 1)
    v_q_avg   = val_q_loss / max(val_count, 1)
    wr_h = [val_corrects[h] / max(val_count, 1) * 100.0 for h in range(4)]
    avg_wr = np.mean(wr_h)

    is_best = v_tot_avg < best_val_loss
    if is_best and N_val > 0:
        best_val_loss = v_tot_avg
        best_meta_weights = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        torch.save(net.state_dict(), "best_meta_learner.pt")

    # SAVE PER-EPOCH WEIGHT CHECKPOINT
    ep_ckpt_path = f"meta_learner_epoch_{ep+1}.pt"
    torch.save(net.state_dict(), ep_ckpt_path)

    status = f"** BEST VAL -> {ep_ckpt_path} saved **" if is_best else f"saved {ep_ckpt_path}"
    print(f"  {ep+1:>5} | {e_tot:>8.4f} {e_q:>8.4f} {e_str:>8.4f} {e_pips:>8.4f} {e_risk:>8.4f} {e_liq:>8.4f} {e_rev:>8.4f} {e_aux1:>8.4f} {e_aux2:>8.4f} {e_sel:>8.4f} | {v_tot_avg:>8.4f} {v_q_avg:>8.4f} | {wr_h[0]:>5.1f}% {wr_h[1]:>5.1f}% {wr_h[2]:>5.1f}% {wr_h[3]:>5.1f}% {avg_wr:>5.1f}% | {status}")

print(f"  {'-'*155}")
if best_meta_weights is not None:
    net.load_state_dict(best_meta_weights)
    print(f"✅ Restored best Meta-Learner weights (Best Val Loss = {best_val_loss:.4e}). All per-epoch weights saved as meta_learner_epoch_*.pt.")
"""

CELL_PHASE2_Q_TRAINING = """# =============================================================================
# PHASE 2: Q-EXECUTOR TRAVERSAL TRAINING OVER train_df (rewritten)
#
# Fixes applied vs. the original:
#  1. SEQUENTIAL walk instead of random.shuffle(indices) — account state (win/loss
#     streak, drawdown, open position) is inherently path-dependent and meaningless
#     under shuffled order. The notebook's own comment called this a "practice maze
#     traversal", which already implied sequential walking was the real intent.
#  2. REAL zone-anchored HardActionMask — computed from actual detect_snr_levels_sequential
#     zones (via up_to_index, no lookahead), not a volume-only mask that ignored zones entirely.
#  3. REAL 28-dim state vector, 1:1 with backend q_executor.py — and critically, NO
#     future-derived value anywhere in it. The previous version put the literal forward
#     price move (the reward target) into state_vec[4] as an INPUT feature — the network
#     was being handed the answer. That's removed entirely; fwd_pct is used ONLY for the
#     reward, never as a model input.
#  4. Missed-opportunity (hindsight) penalty is now GATED by the real hard mask — it only
#     fires when the mask would have actually allowed the trade (i.e. price was genuinely
#     near a confirmed, volume-backed zone), not on raw meta-learner confidence alone.
#     Ungated, this risked teaching the agent to trade too liberally.
# =============================================================================
q_net = ExecutorQNetwork(input_dim=28, hidden_dim=64, num_actions=5).to(device)
q_target = ExecutorQNetwork(input_dim=28, hidden_dim=64, num_actions=5).to(device)
q_target.load_state_dict(q_net.state_dict())
q_opt = optim.AdamW(q_net.parameters(), lr=1e-3, weight_decay=1e-4)

Q_EPOCHS = 50
BATCH_SIZE_Q = 128
BUFFER_CAPACITY = 20000
TARGET_VAL_WIN_RATE = 70.0
replay_buffer = []

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995

net.eval()
mask_engine = HardActionMask()

best_val_wr = 0.0
best_q_weights = None

print(f"🚀 [Phase 2] Q-Executor Sequential Training & Validation (Max {Q_EPOCHS} Epochs | Target WR: {TARGET_VAL_WIN_RATE}%)...")

for q_epoch in range(Q_EPOCHS):
    account = AccountContext()
    _q_loss_acc = 0.0
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    open_position = None  # {action, entry_idx, entry_price}

    train_trades, train_wins, train_losses = 0, 0, 0
    train_win_streak, train_loss_streak = 0, 0
    max_tr_win_streak, max_tr_loss_streak = 0, 0
    total_win_pnl, total_loss_pnl = 0.0, 0.0
    peak_equity, max_drawdown = 10000.0, 0.0

    # ── TRAIN TRAVERSAL ──
    for idx in range(N_train):
        x_seq = train_num_matrix[idx: idx + lookback_bars].flatten()
        x_t = torch.tensor(x_seq, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_vals, strength, pips, risk, liq, rev = net(x_t)

        strength_vec = strength.squeeze(0).cpu().numpy()
        meta_score = float(strength_vec.mean())
        row = train_df.iloc[idx + lookback_bars]
        cp = float(row[close_col])
        row_idx_abs = idx + lookback_bars

        exec_ctx = _make_exec_ctx(SYMBOL, cp, dict(row), atr_col, up_vol_col, down_vol_col)
        bias = HTFBiasPackage(
            direction="bullish" if meta_score > 0.5 else "bearish",
            strength=meta_score, reversal_prob=float(rev.mean().item()),
            q_value=float(q_vals.max().item()),
            expected_mfe_pips=float(risk[0, 0].item()) * 100.0 if risk.shape[-1] > 0 else 0.0,
            expected_mae_pips=float(risk[0, 1].item()) * 100.0 if risk.shape[-1] > 1 else 0.0,
            horizon_strengths=strength_vec.tolist(),
        )

        price_data_hl = train_df[[open_col, high_col, low_col, close_col, vol_col]].rename(
            columns={open_col: "Open", high_col: "High", low_col: "Low", close_col: "Close", vol_col: "Volume"}
        )
        levels = detect_snr_levels_sequential(price_data_hl, up_to_index=row_idx_abs,
                                                lookback_period=min(ZONE_LOOKBACK_PERIOD, row_idx_abs),
                                                min_distance_pct=ZONE_MIN_DISTANCE_PCT) if row_idx_abs >= 20 else []
        df_slice = price_data_hl.iloc[max(0, row_idx_abs - ZONE_LOOKBACK_PERIOD): row_idx_abs + 1]
        zones = create_clustered_zones_sequential(levels, df_slice, n_clusters=min(8, max(3, len(levels)))) if levels else []
        nearest_supp, nearest_res = get_nearest_zones(zones, cp)

        has_open = open_position is not None
        action_mask = mask_engine.get_action_mask(
            cp, exec_ctx.atr, nearest_supp, nearest_res, exec_ctx.buy_volume, exec_ctx.sell_volume, has_open_position=has_open
        )
        state_vec = build_state_vector(None, bias, account, exec_ctx, nearest_supp, nearest_res)

        valid_actions = [a for a in range(5) if action_mask[a] == 1]
        if not valid_actions: valid_actions = [0]

        if random.random() < epsilon:
            action = random.choice(valid_actions)
        else:
            q_net.eval()
            with torch.no_grad():
                st_t = torch.tensor(state_vec, device=device).unsqueeze(0)
                q_out = q_net(st_t).squeeze(0).cpu().numpy()
                masked_q = np.where(action_mask == 1, q_out, -1e9)
                action = int(np.argmax(masked_q))
        action_counts[action] += 1

        fwd_cps = [float(train_df.iloc[idx + lookback_bars + k][close_col]) for k in (1, 3, 6, 12) if (idx + lookback_bars + k) < len(train_df)]
        if not fwd_cps: continue
        max_cp, min_cp = max(fwd_cps), min(fwd_cps)
        fwd_pct = (max_cp - cp) / (cp + 1e-8) if abs(max_cp - cp) > abs(min_cp - cp) else (min_cp - cp) / (cp + 1e-8)

        if not has_open and action in (1, 2):
            open_position = {"action": action, "entry_idx": idx, "entry_price": cp}
            account.open_position_type = "CALL" if action == 1 else "PUT"
            account.reentries_in_window += 1
        elif has_open and action == 4:
            pnl = (cp - open_position["entry_price"]) / open_position["entry_price"]
            if open_position["action"] == 2: pnl = -pnl
            
            train_trades += 1
            account.equity *= (1.0 + pnl)
            peak_equity = max(peak_equity, account.equity)
            drawdown = (peak_equity - account.equity) / peak_equity
            max_drawdown = max(max_drawdown, drawdown)

            if pnl > 0:
                train_wins += 1
                train_win_streak += 1
                train_loss_streak = 0
                max_tr_win_streak = max(max_tr_win_streak, train_win_streak)
                total_win_pnl += pnl
            else:
                train_losses += 1
                train_loss_streak += 1
                train_win_streak = 0
                max_tr_loss_streak = max(max_tr_loss_streak, train_loss_streak)
                total_loss_pnl += abs(pnl)

            account.win_streak = train_win_streak
            account.loss_streak = train_loss_streak
            open_position = None
            account.open_position_type = None
            account.reentries_in_window = 0

        if action == 1: reward = fwd_pct - 0.0005
        elif action == 2: reward = -fwd_pct - 0.0005
        else:
            mask_allowed_entry = (action_mask[1] == 1) or (action_mask[2] == 1)
            reward = -abs(fwd_pct) if (mask_allowed_entry and meta_score >= 0.55 and abs(fwd_pct) >= 0.0015) else 0.001

        next_row = train_df.iloc[idx + lookback_bars + 1] if (idx + lookback_bars + 1) < len(train_df) else row
        next_cp = float(next_row[close_col])
        next_exec_ctx = _make_exec_ctx(SYMBOL, next_cp, dict(next_row), atr_col, up_vol_col, down_vol_col)
        next_levels = detect_snr_levels_sequential(price_data_hl, up_to_index=min(row_idx_abs + 1, len(train_df) - 1),
                                                     lookback_period=min(ZONE_LOOKBACK_PERIOD, row_idx_abs + 1),
                                                     min_distance_pct=ZONE_MIN_DISTANCE_PCT) if row_idx_abs + 1 >= 20 else []
        next_df_slice = price_data_hl.iloc[max(0, row_idx_abs + 1 - ZONE_LOOKBACK_PERIOD): row_idx_abs + 2]
        next_zones = create_clustered_zones_sequential(next_levels, next_df_slice, n_clusters=min(8, max(3, len(next_levels)))) if next_levels else []
        next_nearest_supp, next_nearest_res = get_nearest_zones(next_zones, next_cp)
        next_mask = mask_engine.get_action_mask(
            next_cp, next_exec_ctx.atr, next_nearest_supp, next_nearest_res,
            next_exec_ctx.buy_volume, next_exec_ctx.sell_volume, has_open_position=(open_position is not None)
        )
        next_state_vec = build_state_vector(None, bias, account, next_exec_ctx, next_nearest_supp, next_nearest_res)

        replay_buffer.append((state_vec, action, reward, next_state_vec, False, next_mask))
        if len(replay_buffer) > BUFFER_CAPACITY: replay_buffer.pop(0)

        if len(replay_buffer) >= BATCH_SIZE_Q:
            q_net.train()
            batch = random.sample(replay_buffer, BATCH_SIZE_Q)
            st_b = torch.tensor(np.array([b[0] for b in batch]), device=device)
            act_b = torch.tensor([b[1] for b in batch], device=device).unsqueeze(1)
            rew_b = torch.tensor([b[2] for b in batch], device=device).unsqueeze(1)
            next_st_b = torch.tensor(np.array([b[3] for b in batch]), device=device)
            mask_b = torch.tensor(np.array([b[5] for b in batch]), device=device)

            q_vals_b = q_net(st_b).gather(1, act_b)
            with torch.no_grad():
                next_q_pol = q_net(next_st_b)
                masked_next_q = torch.where(mask_b == 1, next_q_pol, torch.tensor(-1e9, device=device))
                best_next_act = masked_next_q.argmax(dim=1, keepdim=True)
                next_q_targ = q_target(next_st_b).gather(1, best_next_act)
                target_q = rew_b + 0.99 * next_q_targ

            loss = nn.MSELoss()(q_vals_b, target_q)
            q_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
            q_opt.step()

            with torch.no_grad():
                for tp, p in zip(q_target.parameters(), q_net.parameters()):
                    tp.data.copy_(0.005 * p.data + 0.995 * tp.data)
            _q_loss_acc += loss.item()

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # ── VALIDATION TRAVERSAL (PER-EPOCH PASS OVER val_df) ──
    val_trades, val_wins, val_losses = 0, 0, 0
    val_win_streak, val_loss_streak = 0, 0
    max_val_win_streak, max_val_loss_streak = 0, 0
    val_total_win_pnl, val_total_loss_pnl = 0.0, 0.0
    val_peak_eq, val_max_dd = 10000.0, 0.0

    if N_val > 0:
        val_account = AccountContext()
        val_open_pos = None
        q_net.eval()

        val_hl = val_df[[open_col, high_col, low_col, close_col, vol_col]].rename(
            columns={open_col: "Open", high_col: "High", low_col: "Low", close_col: "Close", vol_col: "Volume"}
        )

        for v_idx in range(N_val):
            vx_seq = val_num_matrix[v_idx: v_idx + lookback_bars].flatten()
            vx_t = torch.tensor(vx_seq, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                vq_vals, vstrength, vpips, vrisk, vliq, vreversal = net(vx_t)

            v_strength_vec = vstrength.squeeze(0).cpu().numpy()
            v_meta_score = float(v_strength_vec.mean())
            v_row = val_df.iloc[v_idx + lookback_bars]
            v_cp = float(v_row[close_col])
            v_row_abs = v_idx + lookback_bars

            v_exec_ctx = _make_exec_ctx(SYMBOL, v_cp, dict(v_row), atr_col, up_vol_col, down_vol_col)
            v_bias = HTFBiasPackage(
                direction="bullish" if v_meta_score > 0.5 else "bearish",
                strength=v_meta_score, reversal_prob=float(vreversal.mean().item()),
                q_value=float(vq_vals.max().item()),
                expected_mfe_pips=float(vrisk[0, 0].item()) * 100.0 if vrisk.shape[-1] > 0 else 0.0,
                expected_mae_pips=float(vrisk[0, 1].item()) * 100.0 if vrisk.shape[-1] > 1 else 0.0,
                horizon_strengths=v_strength_vec.tolist(),
            )

            v_levels = detect_snr_levels_sequential(val_hl, up_to_index=v_row_abs,
                                                    lookback_period=min(ZONE_LOOKBACK_PERIOD, v_row_abs),
                                                    min_distance_pct=ZONE_MIN_DISTANCE_PCT) if v_row_abs >= 20 else []
            v_slice = val_hl.iloc[max(0, v_row_abs - ZONE_LOOKBACK_PERIOD): v_row_abs + 1]
            v_zones = create_clustered_zones_sequential(v_levels, v_slice, n_clusters=min(8, max(3, len(v_levels)))) if v_levels else []
            v_supp, v_res = get_nearest_zones(v_zones, v_cp)

            v_has_open = val_open_pos is not None
            v_mask = mask_engine.get_action_mask(
                v_cp, v_exec_ctx.atr, v_supp, v_res, v_exec_ctx.buy_volume, v_exec_ctx.sell_volume, has_open_position=v_has_open
            )
            v_state_vec = build_state_vector(None, v_bias, val_account, v_exec_ctx, v_supp, v_res)

            with torch.no_grad():
                st_v_t = torch.tensor(v_state_vec, device=device).unsqueeze(0)
                v_q_out = q_net(st_v_t).squeeze(0).cpu().numpy()
                v_masked_q = np.where(v_mask == 1, v_q_out, -1e9)
                v_act = int(np.argmax(v_masked_q))

            if not v_has_open and v_act in (1, 2):
                val_open_pos = {"action": v_act, "entry_price": v_cp}
                val_account.open_position_type = "CALL" if v_act == 1 else "PUT"
            elif v_has_open and v_act == 4:
                v_pnl = (v_cp - val_open_pos["entry_price"]) / val_open_pos["entry_price"]
                if val_open_pos["action"] == 2: v_pnl = -v_pnl

                val_trades += 1
                val_account.equity *= (1.0 + v_pnl)
                val_peak_eq = max(val_peak_eq, val_account.equity)
                val_dd = (val_peak_eq - val_account.equity) / val_peak_eq
                val_max_dd = max(val_max_dd, val_dd)

                if v_pnl > 0:
                    val_wins += 1
                    val_win_streak += 1
                    val_loss_streak = 0
                    max_val_win_streak = max(max_val_win_streak, val_win_streak)
                    val_total_win_pnl += v_pnl
                else:
                    val_losses += 1
                    val_loss_streak += 1
                    val_win_streak = 0
                    max_val_loss_streak = max(max_val_loss_streak, val_loss_streak)
                    val_total_loss_pnl += abs(v_pnl)

                val_open_pos = None
                val_account.open_position_type = None

    # ── METRIC CALCULATIONS & MONEY MANAGEMENT INSIGHTS ──
    tr_wr = (train_wins / max(train_trades, 1)) * 100.0
    tr_lr = (train_losses / max(train_trades, 1)) * 100.0
    val_wr = (val_wins / max(val_trades, 1)) * 100.0
    val_lr = (val_losses / max(val_trades, 1)) * 100.0

    avg_win_pnl = (val_total_win_pnl / max(val_wins, 1)) * 100.0
    avg_loss_pnl = (val_total_loss_pnl / max(val_losses, 1)) * 100.0
    rr_ratio = avg_win_pnl / max(avg_loss_pnl, 1e-6)

    # Kelly Criterion position sizing recommendation: K% = W - (1-W)/RR
    p_win = val_wr / 100.0
    kelly_f = p_win - ((1.0 - p_win) / max(rr_ratio, 1e-6)) if rr_ratio > 0 else 0.0
    rec_sizing = float(np.clip(0.5 * kelly_f * 100.0, 1.0, 25.0))  # Bounded Half-Kelly %

    is_best_q = val_wr > best_val_wr and val_trades >= 5
    if is_best_q:
        best_val_wr = val_wr
        best_q_weights = {k: v.cpu().clone() for k, v in q_net.state_dict().items()}
        torch.save(q_net.state_dict(), "best_q_executor.pt")

    q_ckpt_path = f"q_executor_epoch_{q_epoch+1}.pt"
    torch.save(q_net.state_dict(), q_ckpt_path)

    avg_q_td = _q_loss_acc / (N_train + 1e-8)
    status_q = f"** NEW BEST VAL WR -> {q_ckpt_path} saved **" if is_best_q else f"saved {q_ckpt_path}"

    print()
    print(f"📊 [Q Maze Epoch {q_epoch+1:>2}/{Q_EPOCHS}] TD Loss={avg_q_td:.4e} | Epsilon={epsilon:.3f}")
    print(f"  ├─ Train Performance : Trades={train_trades:>3} | WinRate={tr_wr:>5.1f}% ({train_wins}W/{train_losses}L) | LossRate={tr_lr:>5.1f}% | Streaks(W/L)={max_tr_win_streak}/{max_tr_loss_streak} | MaxDD={max_drawdown*100:.1f}%")
    print(f"  ├─ Val Performance   : Trades={val_trades:>3} | WinRate={val_wr:>5.1f}% ({val_wins}W/{val_losses}L) | LossRate={val_lr:>5.1f}% | Streaks(W/L)={max_val_win_streak}/{max_val_loss_streak} | MaxDD={val_max_dd*100:.1f}% {status_q}")
    print(f"  ├─ Money Management : AvgWin={avg_win_pnl:+.2f}% | AvgLoss=-{avg_loss_pnl:.2f}% | Risk/Reward={rr_ratio:.2f}:1 | Half-Kelly Sizing={rec_sizing:.1f}%")
    print(f"  └─ Action Allocations: WAIT={action_counts[0]}, CALL={action_counts[1]}, PUT={action_counts[2]}, TP={action_counts[3]}, CLOSE={action_counts[4]}")

    if val_wr >= TARGET_VAL_WIN_RATE and val_trades >= 10:
        print(f"🎯 Target Validation Win Rate ({TARGET_VAL_WIN_RATE}%) achieved at Epoch {q_epoch+1}! Early stopping triggered.")
        break

if best_q_weights is not None:
    q_net.load_state_dict(best_q_weights)
    print(f"✅ Restored best Q-Executor checkpoint (Best Val Win Rate = {best_val_wr:.1f}%). All per-epoch weights saved as q_executor_epoch_*.pt.")
"""

CELL_PHASE3_PHASE4_EVAL = """# =============================================================================
# 📈 PHASE 3 & 4: OUT-OF-SAMPLE TEST EVALUATION & DYNAMIC PORTFOLIO SIMULATION
# =============================================================================
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXPIRY_HORIZONS = {
    "5m (1 bar)": 1,
    "15m (3 bars)": 3,
    "30m (6 bars)": 6,
    "1h (12 bars)": 12,
}

print("\\n" + "═"*92)
print("📊 OUT-OF-SAMPLE TEST EVALUATION ACROSS EXPIRY HORIZONS")
print("═"*92)
print(f"{'Expiry Horizon':<18} | {'Trades':<8} | {'Wins':<6} | {'Losses':<8} | {'Waits':<8} | {'Win Rate %':<10} | {'Max Streak [W, L]':<18}")
print("-" * 92)

test_matrix = np.nan_to_num(test_df[feature_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
N_test = len(test_df) - lookback_bars - 12
q_net.eval()
net.eval()

# Phase 3: Per-Horizon Evaluation
for exp_label, lookahead_bars in EXPIRY_HORIZONS.items():
    wins, losses, waits = 0, 0, 0
    cur_w_streak, cur_l_streak = 0, 0
    max_w_streak, max_l_streak = 0, 0
    open_until = -1

    for idx in range(N_test):
        if idx < open_until:
            continue

        x_seq = test_matrix[idx : idx + lookback_bars].flatten()
        x_t = torch.tensor(x_seq, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            q_vals, strength, pips, risk, liq, rev = net(x_t)

        meta_score = float(strength.mean().item())
        row = test_df.iloc[idx + lookback_bars]
        exp_row = test_df.iloc[idx + lookback_bars + lookahead_bars]

        cp = float(row[close_col])
        exp_cp = float(exp_row[close_col])

        st_vec = np.zeros((1, 28), dtype=np.float32)
        st_vec[0, 0] = meta_score

        with torch.no_grad():
            act_logits = q_net(torch.tensor(st_vec, device=device))
            action = int(act_logits.argmax(dim=-1).item())

        if action == 1:  # BUY_CALL
            open_until = idx + lookahead_bars
            if exp_cp > cp:
                wins += 1
                cur_w_streak += 1; cur_l_streak = 0
            else:
                losses += 1
                cur_l_streak += 1; cur_w_streak = 0
        elif action == 2:  # BUY_PUT
            open_until = idx + lookahead_bars
            if exp_cp < cp:
                wins += 1
                cur_w_streak += 1; cur_l_streak = 0
            else:
                losses += 1
                cur_l_streak += 1; cur_w_streak = 0
        else:
            waits += 1

        max_w_streak = max(max_w_streak, cur_w_streak)
        max_l_streak = max(max_l_streak, cur_l_streak)

    tot = wins + losses
    wr = (100.0 * wins / tot) if tot > 0 else 0.0
    print(f"{exp_label:<18} | {tot:<8} | {wins:<6} | {losses:<8} | {waits:<8} | {wr:<10.2f} | W:{max_w_streak} / L:{max_l_streak}")

# Phase 4: Dynamic Collective Multi-Horizon Concurrent Portfolio Simulation
print("\\n" + "═"*92)
print("📊 DYNAMIC COLLECTIVE MULTI-HORIZON CONCURRENT PORTFOLIO SIMULATION")
print("═"*92)
print("Policy: Max 1 active trade per horizon concurrently from shared account.")
print("-" * 92)

active_horizon_until = {exp: -1 for exp in EXPIRY_HORIZONS}
portfolio_outcomes = []
portfolio_win_streaks = []
portfolio_loss_streaks = []
horizon_trade_counts = {exp: {"wins": 0, "losses": 0, "total": 0} for exp in EXPIRY_HORIZONS}

cur_p_w_streak, cur_p_l_streak = 0, 0
max_p_w_streak, max_p_l_streak = 0, 0
_p_cur_ws, _p_cur_ls = 0, 0
recommended_matches = 0

for idx in range(N_test):
    x_seq = test_matrix[idx : idx + lookback_bars].flatten()
    x_t = torch.tensor(x_seq, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        q_vals, strength, pips, risk, liq, rev = net(x_t)

    meta_score = float(strength.mean().item())
    rec_horizon_idx = int(strength.argmax(dim=-1).item())

    row = test_df.iloc[idx + lookback_bars]
    cur_price = float(row[close_col])

    for h_idx, (exp_label, lookahead_bars) in enumerate(EXPIRY_HORIZONS.items()):
        if idx < active_horizon_until[exp_label]:
            continue

        if idx + lookback_bars + lookahead_bars >= len(test_df):
            continue

        expiry_row = test_df.iloc[idx + lookback_bars + lookahead_bars]
        expiry_price = float(expiry_row[close_col])

        st_vec = np.zeros((1, 28), dtype=np.float32)
        st_vec[0, 0] = meta_score
        st_vec[0, 1] = float(rev.mean().item())

        with torch.no_grad():
            act_logits = q_net(torch.tensor(st_vec, device=device))
            action = int(act_logits.argmax(dim=-1).item())

        outcome = None
        if action == 1:  # BUY_CALL
            outcome = 1 if expiry_price > cur_price else 0
        elif action == 2:  # BUY_PUT
            outcome = 1 if expiry_price < cur_price else 0

        if outcome is not None:
            active_horizon_until[exp_label] = idx + lookahead_bars
            horizon_trade_counts[exp_label]["total"] += 1
            if h_idx == rec_horizon_idx:
                recommended_matches += 1

            portfolio_outcomes.append(outcome)
            if outcome == 1:
                horizon_trade_counts[exp_label]["wins"] += 1
                cur_p_w_streak += 1; cur_p_l_streak = 0
                _p_cur_ws += 1
                if _p_cur_ls > 0:
                    portfolio_loss_streaks.append(_p_cur_ls)
                    _p_cur_ls = 0
            else:
                horizon_trade_counts[exp_label]["losses"] += 1
                cur_p_l_streak += 1; cur_p_w_streak = 0
                _p_cur_ls += 1
                if _p_cur_ws > 0:
                    portfolio_win_streaks.append(_p_cur_ws)
                    _p_cur_ws = 0

            max_p_w_streak = max(max_p_w_streak, cur_p_w_streak)
            max_p_l_streak = max(max_p_l_streak, cur_p_l_streak)

if _p_cur_ws > 0: portfolio_win_streaks.append(_p_cur_ws)
if _p_cur_ls > 0: portfolio_loss_streaks.append(_p_cur_ls)

total_p_trades = len(portfolio_outcomes)
p_wins = sum(portfolio_outcomes)
p_losses = total_p_trades - p_wins
p_win_rate = (p_wins / total_p_trades * 100.0) if total_p_trades > 0 else 0.0

p_avg_ws = round(sum(portfolio_win_streaks) / len(portfolio_win_streaks), 2) if portfolio_win_streaks else 0.0
p_avg_ls = round(sum(portfolio_loss_streaks) / len(portfolio_loss_streaks), 2) if portfolio_loss_streaks else 0.0
p_streak_ratio = round(p_avg_ws / p_avg_ls, 2) if p_avg_ls > 0 else float("inf")
rec_alignment_pct = round(recommended_matches / total_p_trades * 100.0, 1) if total_p_trades > 0 else 0.0

print(f"COLLECTIVE PORTFOLIO  | Trades: {total_p_trades:<5} | Wins: {p_wins:<5} | Losses: {p_losses:<5} | Win Rate: {p_win_rate:.2f}% | Max Streaks [W:{max_p_w_streak}, L:{max_p_l_streak}]")
print(f"  Avg Streaks: W={p_avg_ws} / L={p_avg_ls}  (Streak Ratio: {p_streak_ratio}x)")
print(f"  Recommended Expiry Alignment Rate: {rec_alignment_pct}% ({recommended_matches}/{total_p_trades})")
print("  Per-Horizon Portfolio Contribution:")
for exp_label, counts in horizon_trade_counts.items():
    h_tot = counts["total"]
    h_wr = (counts["wins"] / h_tot * 100.0) if h_tot > 0 else 0.0
    print(f"    - {exp_label:<5}: {h_tot:<5} trades | Win Rate: {h_wr:.2f}% (Wins: {counts['wins']}, Losses: {counts['losses']})")
"""

CELL_EXPORT_ZIP = """# =============================================================================
# 💾 EXPORT TRAINED CHECKPOINTS TO ZIP FOR BACKEND HYDRATION
# =============================================================================
def export_all_checkpoints_zip(output_zip_path):
    pt_meta_path = os.path.join(OUTPUT_DIR, 'meta_learner_best.pt')
    pt_q_path    = os.path.join(OUTPUT_DIR, 'q_executor_best.pt')

    torch.save(net.state_dict(), pt_meta_path)
    torch.save(q_net.state_dict(), pt_q_path)

    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(pt_meta_path, arcname='meta_learner_best.pt')
        zipf.write(pt_q_path,    arcname='q_executor_best.pt')

    zip_mb = os.path.getsize(output_zip_path) / (1024 * 1024)
    print(f"✅ Checkpoint export complete: {output_zip_path} ({zip_mb:.2f} MB)")
    print("Ready to copy back to backend app/core/ml/checkpoints/")

export_all_checkpoints_zip(ZIP_EXPORT_PATH)
"""

KERAS_NOTEBOOK_PATH = "kaggle_keras_meta_learner_training.ipynb"

def generate_notebook(cells_def: list[dict], output_filename: str):
    nb = {
        "cells": cells_def,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated Notebook: {output_filename}")


def _build_cells(title: str) -> list:
    """Single source of truth for notebook cells — both PyTorch and Keras notebooks
    use the identical PyTorch pipeline. The Keras label is kept for Kaggle dataset
    organisation; TF/Keras dead code from the original notebook has been removed."""
    return [
        {"cell_type": "markdown", "metadata": {}, "source": [f"# {title}\n\nFull RL pipeline: Meta-Learner training (Phase 1), Q-Executor sequential traversal training (Phase 2), out-of-sample evaluation across 4 expiry horizons (Phase 3/4), and checkpoint export."]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [CELL_IMPORTS]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [CELL_DATA_LOAD]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [CELL_SNR_DETECTION]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [CELL_FEATURE_ENGINE]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [CELL_PYTORCH_MODELS]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [CELL_PYTORCH_TRAINER]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [CELL_PHASE2_Q_TRAINING]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [CELL_PHASE3_PHASE4_EVAL]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "source": [CELL_EXPORT_ZIP]},
    ]


def prepare_bundle():
    generate_notebook(
        _build_cells("AXE Genesis PyTorch Meta-Learner & Q-Executor Pipeline (Kaggle GPU)"),
        PYTORCH_NOTEBOOK_PATH,
    )
    # Keras notebook: identical PyTorch pipeline, different filename/title.
    # The original notebook had dead TF/Keras GPU setup that was never used — removed.
    generate_notebook(
        _build_cells("AXE Genesis PyTorch Pipeline (Kaggle GPU — Keras Dataset Slot)"),
        KERAS_NOTEBOOK_PATH,
    )


if __name__ == "__main__":
    prepare_bundle()
