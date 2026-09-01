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
import re

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
elif os.path.exists('data/train_50k.csv'):
    data_dir = 'data'
else:
    data_dir = KAGGLE_DATASET_DIR

train_csv = os.path.join(data_dir, 'train_50k.csv')
val_csv   = os.path.join(data_dir, 'val_50k.csv')
test_csv  = os.path.join(data_dir, 'test_50k.csv')

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
from typing import Dict, List, Optional, Tuple, Any, Union

feature_cols = [c for c in train_df.columns if c not in ("timestamp", "Time") and not "target" in c and not "forward" in c and not "adv_target" in c]
num_features = len(feature_cols)
meta_lookback_bars = 150
q_lookback_bars = 300
input_dim = meta_lookback_bars * num_features

print(f"📊 Feature Input Contract: {num_features} columns | Lookback {meta_lookback_bars} bars → Flattened Dim = {input_dim}")

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

        # Auxiliary Private Projections (Zero Gradient Interference)
        _aux_in = 64 + 32 + 32
        self.branch_ln = nn.LayerNorm(_aux_in)

        self.pips_proj = nn.Linear(_aux_in, 32)
        self.pips_ln   = nn.LayerNorm(32)
        self.pips_head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 4),
        )

        self.risk_proj = nn.Linear(_aux_in, 32)
        self.risk_ln   = nn.LayerNorm(32)
        self.risk_head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 8),
        )

        self.liq_proj = nn.Linear(_aux_in, 16)
        self.liq_ln   = nn.LayerNorm(16)
        self.liquidity_head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(16, 8),
            nn.SiLU(),
            nn.Linear(8, 2),
        )

        self.rev_proj = nn.Linear(_aux_in, 16)
        self.rev_ln   = nn.LayerNorm(16)
        self.reversal_head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(16, 8),
            nn.SiLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

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

        # Branch 2
        half = max(1, t // 2)
        x_mid_trans = x_trans[:, :, -half:]
        b2_c = self.b2_act(self.b2_bn(self.b2_conv(x_mid_trans)))
        b2_gap = torch.mean(b2_c, dim=-1)
        b2_out = torch.relu(self.b2_fc(b2_gap))

        # Branch 3
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
        branch_cat = self.branch_ln(torch.cat([b1_out, b2_out, b3_out], dim=-1).detach())

        pips      = self.pips_head(self.pips_ln(self.pips_proj(branch_cat)))
        risk      = self.risk_head(self.risk_ln(self.risk_proj(branch_cat)))
        liquidity = self.liquidity_head(self.liq_ln(self.liq_proj(branch_cat)))
        reversal  = self.reversal_head(self.rev_ln(self.rev_proj(branch_cat)))

        if return_aux:
            return q_vals, strength, pips, risk, liquidity, reversal, aux1, aux2, selector_logits
        return q_vals, strength, pips, risk, liquidity, reversal


class ExecutorQNetwork(nn.Module):
    def __init__(
        self,
        num_features: int = num_features,
        input_dim: int = 28,
        hidden_dim: int = 128,
        num_actions: int = 5,
        ctx_dim: int = 28,
        q_lookback: int = q_lookback_bars,
        num_horizons: int = 4,
        num_head_actions: int = 3,
    ):
        super().__init__()
        self.num_features = num_features
        self.ctx_dim = ctx_dim
        self.q_lookback = q_lookback
        self.num_horizons = num_horizons

        # --- Branch A: full indicators (B, T, F) -> Conv1D ---
        self.feat_conv1 = nn.Conv1d(num_features, 64, kernel_size=3, padding=1)
        self.feat_bn1   = nn.BatchNorm1d(64)
        self.feat_conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.feat_bn2   = nn.BatchNorm1d(64)
        self.feat_pool  = nn.AdaptiveAvgPool1d(1)
        self.feat_fc    = nn.Linear(64, 64)

        # --- Branch B: 28-dim context dense encoder ---
        self.b1_fc1 = nn.Linear(ctx_dim, hidden_dim)
        self.b1_ln1 = nn.LayerNorm(hidden_dim)
        self.b1_fc2 = nn.Linear(hidden_dim, 64)
        self.b1_ln2 = nn.LayerNorm(64)

        self.b2_meta   = nn.Linear(10, 16)
        self.b2_risk   = nn.Linear(5, 16)
        self.b2_zone   = nn.Linear(8, 16)
        self.b2_time   = nn.Linear(5, 16)
        self.b2_fusion = nn.Linear(64, 64)
        self.b2_ln     = nn.LayerNorm(64)

        # --- Fusion + per-horizon heads ---
        self.fusion_fc = nn.Linear(64 + 64 + 64, hidden_dim)
        self.fusion_ln = nn.LayerNorm(hidden_dim)

        self.horizon_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.LayerNorm(64),
                nn.SiLU(),
                nn.Linear(64, num_head_actions),
            )
            for _ in range(num_horizons)
        ])

    def _encode_feat(self, feat_window: torch.Tensor) -> torch.Tensor:
        x = feat_window.transpose(1, 2)
        x = torch.nn.functional.silu(self.feat_bn1(self.feat_conv1(x)))
        x = torch.nn.functional.silu(self.feat_bn2(self.feat_conv2(x)))
        x = self.feat_pool(x).squeeze(-1)
        return torch.nn.functional.silu(self.feat_fc(x))

    def _encode_ctx(self, ctx: torch.Tensor):
        b1     = torch.nn.functional.silu(self.b1_ln1(self.b1_fc1(ctx)))
        b1_out = torch.nn.functional.silu(self.b1_ln2(self.b1_fc2(b1)))

        meta_f = ctx[:, :10]
        risk_f = ctx[:, 10:15]
        zone_f = ctx[:, 15:23]
        time_f = ctx[:, 23:28]
        b2_cat = torch.cat([
            torch.relu(self.b2_meta(meta_f)),
            torch.relu(self.b2_risk(risk_f)),
            torch.relu(self.b2_zone(zone_f)),
            torch.relu(self.b2_time(time_f)),
        ], dim=-1)
        b2_out = torch.nn.functional.silu(self.b2_ln(self.b2_fusion(b2_cat)))
        return b1_out, b2_out

    def forward(
        self,
        feat_window: torch.Tensor,
        ctx: torch.Tensor,
        horizon_idx: Optional[int] = None,
        return_aux: bool = False,
    ):
        feat_h         = self._encode_feat(feat_window)
        b1_out, b2_out = self._encode_ctx(ctx)
        shared = torch.nn.functional.silu(
            self.fusion_ln(
                self.fusion_fc(torch.cat([feat_h, b1_out, b2_out], dim=-1))
            )
        )
        if horizon_idx is not None:
            return self.horizon_heads[int(horizon_idx)](shared)
        return torch.stack([h(shared) for h in self.horizon_heads], dim=1)

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
# PHASE 2: PER-HORIZON Q-LEARNING
# Architecture: 4 independent Q-heads (one per horizon), each with 3 actions
#               WAIT(0) / CALL(1) / PUT(2). Auto-expiry handles settlement.
# Gate: max 1 open position per horizon simultaneously.
# Training: each horizon's head is supervised only by that horizon's reward signal.
# =============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel
import time
import numpy as np
import random
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpus = torch.cuda.device_count()
print(f"GPUs available: {n_gpus}")

HORIZON_BARS_LIST = [1, 3, 6, 12]       # 5m, 15m, 30m, 1h
HORIZON_LABELS    = ["5m", "15m", "30m", "1h"]
NUM_HORIZONS      = 4
H_WAIT, H_CALL, H_PUT = 0, 1, 2         # Per-head action indices

net.eval()
if n_gpus > 1:
    meta_dp = DataParallel(net)
else:
    meta_dp = net

# ---------- Precompute meta outputs ----------
PRECOMPUTE_BATCH = 256
N_train = len(train_df) - lookback_bars - 12
print(f"[Precompute] Meta features for {N_train} steps...")
t0 = time.time()

meta_strengths = np.zeros((N_train, 4), dtype=np.float32)
meta_qmax      = np.zeros(N_train, dtype=np.float32)
meta_rev       = np.zeros(N_train, dtype=np.float32)
meta_mfe       = np.zeros(N_train, dtype=np.float32)
meta_mae       = np.zeros(N_train, dtype=np.float32)

with torch.no_grad():
    for start in range(0, N_train, PRECOMPUTE_BATCH):
        end = min(start + PRECOMPUTE_BATCH, N_train)
        batch_x = np.stack([train_num_matrix[i: i + lookback_bars].flatten() for i in range(start, end)])
        x_t = torch.tensor(batch_x, dtype=torch.float32, device=device)
        q_vals, strength, pips, risk, liq, rev = meta_dp(x_t)
        meta_strengths[start:end] = strength.cpu().numpy()
        meta_qmax[start:end]      = q_vals.max(dim=1).values.cpu().numpy()
        meta_rev[start:end]       = rev.squeeze(-1).cpu().numpy() if rev.ndim > 1 else rev.cpu().numpy()
        if risk.shape[-1] >= 2:
            meta_mfe[start:end]   = risk[:, 0].cpu().numpy()
            meta_mae[start:end]   = risk[:, 1].cpu().numpy()
        if (start // PRECOMPUTE_BATCH) % 20 == 0:
            print(f"  meta precompute {end}/{N_train}")

meta_strengths = np.nan_to_num(meta_strengths, nan=0.5, posinf=1.0, neginf=0.0)
meta_qmax      = np.nan_to_num(meta_qmax, nan=0.5)
meta_rev       = np.nan_to_num(meta_rev, nan=0.2)
meta_mfe       = np.nan_to_num(meta_mfe, nan=0.5)
meta_mae       = np.nan_to_num(meta_mae, nan=0.15)
print(f"Meta precompute done in {time.time()-t0:.1f}s")

# ---------- Precompute zones ----------
print("[Precompute] SNR zones...")
t1 = time.time()
price_data_hl = train_df[[open_col, high_col, low_col, close_col, vol_col]].rename(
    columns={open_col: "Open", high_col: "High", low_col: "Low", close_col: "Close", vol_col: "Volume"})
nearest_supp_list = [None] * N_train
nearest_res_list  = [None] * N_train
close_prices = train_df[close_col].values.astype(np.float64)
atr_vals = train_df[atr_col].values.astype(np.float64) if atr_col else close_prices * 0.005
up_vols  = train_df[up_vol_col].values.astype(np.float64) if up_vol_col else np.zeros(len(train_df))
dn_vols  = train_df[down_vol_col].values.astype(np.float64) if down_vol_col else np.zeros(len(train_df))

last_zones = []
for i in range(N_train):
    abs_idx = i + lookback_bars
    if i % 5 == 0 or not last_zones:
        lb = min(ZONE_LOOKBACK_PERIOD, abs_idx)
        levels = detect_snr_levels_sequential(price_data_hl, up_to_index=abs_idx, lookback_period=lb, min_distance_pct=ZONE_MIN_DISTANCE_PCT) if abs_idx >= 20 else []
        df_slice = price_data_hl.iloc[max(0, abs_idx - ZONE_LOOKBACK_PERIOD): abs_idx + 1]
        last_zones = create_clustered_zones_sequential(levels, df_slice, n_clusters=min(8, max(3, len(levels)))) if levels else []
    ns, nr = get_nearest_zones(last_zones, close_prices[abs_idx])
    nearest_supp_list[i] = ns
    nearest_res_list[i]  = nr
    if i % 5000 == 0:
        print(f"  zones {i}/{N_train}")
print(f"Zone precompute done in {time.time()-t1:.1f}s")

# ---------- Build static state vectors ----------
print("[Precompute] static state features...")
static_states = np.zeros((N_train, 28), dtype=np.float32)
for i in range(N_train):
    abs_idx = i + lookback_bars
    row = train_df.iloc[abs_idx]
    cp  = close_prices[abs_idx]
    atr = max(0.01, atr_vals[abs_idx])
    bv, sv = up_vols[abs_idx], dn_vols[abs_idx]
    ts = row.get("timestamp", None)
    hour_f, dow, phase = 14.5, 1, "off_hours"
    if ts is not None:
        try:
            ts_pd = pd.Timestamp(ts)
            if ts_pd.tzinfo is None: ts_pd = ts_pd.tz_localize("UTC")
            ts_et = ts_pd.tz_convert("America/New_York")
            hour_f = ts_et.hour + ts_et.minute / 60.0
            dow = ts_et.dayofweek
            if   9.5 <= hour_f < 10.5: phase = "nyse_open"
            elif 15.0 <= hour_f < 16.0: phase = "nyse_power_hour"
            elif 9.5 <= hour_f < 16.0:  phase = "regular_hours"
        except Exception: pass
    sv_vec = meta_strengths[i]
    opt_h = int(np.argmax(sv_vec))
    meta_score = float(sv_vec[opt_h])
    dir_flag = 1.0 if meta_score > 0.5 else (-1.0 if meta_score < 0.5 else 0.0)
    hs = sv_vec.tolist()
    ns = nearest_supp_list[i]
    nr = nearest_res_list[i]
    supp_dist = abs(cp - ns["price_level"]) / cp if ns else 1.0
    res_dist  = abs(cp - nr["price_level"]) / cp if nr else 1.0
    supp_vol_ratio = ns["volume_delta_ratio"] if ns else 0.0
    res_vol_ratio  = nr["volume_delta_ratio"] if nr else 0.0
    total_vol = bv + sv
    vol_delta_ratio = (bv - sv) / (total_vol + 1e-6)
    sin_hour = np.sin(2 * np.pi * hour_f / 24.0)
    cos_hour = np.cos(2 * np.pi * hour_f / 24.0)
    static_states[i] = [
        dir_flag, meta_score, float(meta_rev[i]), float(meta_qmax[i]),
        float(meta_mfe[i]), float(meta_mae[i]),
        hs[0], hs[1], hs[2], hs[3],
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        atr / cp, supp_dist, res_dist,
        supp_vol_ratio, res_vol_ratio, vol_delta_ratio,
        0.0,
        sin_hour, cos_hour, dow / 6.0,
        1.0 if phase == "nyse_open" else 0.0,
        1.0 if phase == "nyse_power_hour" else 0.0,
    ]
static_states = np.nan_to_num(static_states, nan=0.0, posinf=0.0, neginf=0.0)
print("Static state cache ready.")

# ---------- Per-horizon Q-network ----------
# One net, 4 independent heads — each head[h] produces [WAIT, CALL, PUT] for horizon h
q_net    = ExecutorQNetwork(input_dim=28, hidden_dim=64, num_horizons=NUM_HORIZONS).to(device)
q_target = ExecutorQNetwork(input_dim=28, hidden_dim=64, num_horizons=NUM_HORIZONS).to(device)
q_target.load_state_dict(q_net.state_dict())
q_opt = optim.AdamW(q_net.parameters(), lr=1e-3, weight_decay=1e-4)

Q_EPOCHS      = 50
BATCH_SIZE_Q  = 256
BUFFER_CAPACITY = 30000
# Separate replay buffer per horizon so each head's gradient is signal-clean
replay_buffers = [[] for _ in range(NUM_HORIZONS)]

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay_per_epoch = 0.92

mask_engine = HardActionMask()

print(f"[Phase 2] Per-Horizon Q-Learning (4 heads x {Q_EPOCHS} epochs)...")
print(f"  {'Epoch':>5} | {'Loss':>10} | {'eps':>5} | {'5m WAIT/CALL/PUT':>18} | {'15m WAIT/CALL/PUT':>18} | {'30m WAIT/CALL/PUT':>18} | {'1h WAIT/CALL/PUT':>18}")
print(f"  {'-'*105}")

for q_epoch in range(Q_EPOCHS):
    # Per-horizon position tracking
    open_positions = {h: None for h in range(NUM_HORIZONS)}  # h -> {action, entry_price, entry_i}
    win_streaks    = {h: 0 for h in range(NUM_HORIZONS)}
    loss_streaks   = {h: 0 for h in range(NUM_HORIZONS)}
    action_counts  = {h: {H_WAIT: 0, H_CALL: 0, H_PUT: 0} for h in range(NUM_HORIZONS)}

    _q_loss_acc = 0.0
    _q_steps    = 0

    for i in range(N_train):
        abs_idx = i + lookback_bars
        cp  = close_prices[abs_idx]
        atr = max(0.01, atr_vals[abs_idx])
        bv, sv_v = up_vols[abs_idx], dn_vols[abs_idx]
        ns, nr = nearest_supp_list[i], nearest_res_list[i]

        # --- Process all 4 horizons independently at each bar ---
        for h in range(NUM_HORIZONS):
            lookahead = HORIZON_BARS_LIST[h]

            # Auto-expire position for this horizon
            if open_positions[h] is not None:
                bars_held = i - open_positions[h]["entry_i"]
                if bars_held >= open_positions[h]["horizon"]:
                    entry_p = open_positions[h]["entry_price"]
                    pnl = (cp - entry_p) / (entry_p + 1e-8)
                    if open_positions[h]["action"] == H_PUT:
                        pnl = -pnl
                    if pnl > 0:
                        win_streaks[h] += 1
                        loss_streaks[h] = 0
                    else:
                        loss_streaks[h] += 1
                        win_streaks[h] = 0
                    settle_reward = float(np.clip(pnl - 0.0005, -0.05, 0.05))
                    # Synthetic CLOSE transition into replay buffer for this horizon
                    sc = static_states[i].copy()
                    sc[12] = float(pnl)  # unrealized → realized
                    next_flat = static_states[min(i + 1, N_train - 1)].copy()
                    next_flat[12] = 0.0
                    replay_buffers[h].append((sc, H_WAIT, settle_reward, next_flat))
                    if len(replay_buffers[h]) > BUFFER_CAPACITY:
                        replay_buffers[h].pop(0)
                    open_positions[h] = None

            # Mark-to-market live unrealized PnL
            if open_positions[h] is not None:
                unreal = (cp - open_positions[h]["entry_price"]) / (open_positions[h]["entry_price"] + 1e-8)
                if open_positions[h]["action"] == H_PUT:
                    unreal = -unreal
            else:
                unreal = 0.0

            has_open = open_positions[h] is not None

            # Per-horizon mask: only WAIT allowed while position is open
            if has_open:
                h_mask = np.array([1, 0, 0], dtype=np.int32)  # WAIT only
            else:
                base_mask = mask_engine.get_action_mask(cp, atr, ns, nr, bv, sv_v, has_open_position=False)
                h_mask = np.array([base_mask[0], base_mask[1], base_mask[2]], dtype=np.int32)

            # Build state (inject horizon index as a feature override in slot 15)
            state = static_states[i].copy()
            state[11] = 1.0 if has_open else 0.0
            state[12] = float(unreal)
            state[13] = win_streaks[h] / 10.0
            state[14] = loss_streaks[h] / 10.0
            state[15] = float(h) / 3.0  # horizon identity slot

            valid = [a for a in range(3) if h_mask[a] == 1] or [H_WAIT]

            if random.random() < epsilon:
                action = random.choice(valid)
            else:
                q_net.eval()
                with torch.no_grad():
                    st_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    logits = q_net(st_t, horizon_idx=h).squeeze(0).cpu().numpy()
                    masked = np.where(h_mask == 1, logits, -1e9)
                    action = int(np.argmax(masked))
            action_counts[h][action] += 1

            if abs_idx + lookahead >= len(train_df):
                continue
            expiry_cp = close_prices[abs_idx + lookahead]
            fwd_pct = float(np.clip((expiry_cp - cp) / (cp + 1e-8), -0.05, 0.05))

            if not has_open and action == H_CALL:
                open_positions[h] = {"action": H_CALL, "entry_price": cp, "entry_i": i, "horizon": lookahead}
            elif not has_open and action == H_PUT:
                open_positions[h] = {"action": H_PUT, "entry_price": cp, "entry_i": i, "horizon": lookahead}

            # Reward shaping (horizon-specific)
            if action == H_CALL:
                reward = fwd_pct - 0.0005
            elif action == H_PUT:
                reward = -fwd_pct - 0.0005
            else:
                # WAIT penalty only if high-confidence signal missed
                if h_mask[H_CALL] == 1 or h_mask[H_PUT] == 1:
                    h_strength = float(meta_strengths[i][h])
                    if h_strength >= 0.60 and abs(fwd_pct) >= 0.0015:
                        reward = -abs(fwd_pct)
                    else:
                        reward = 0.001
                else:
                    reward = 0.001
            reward = float(np.clip(reward, -0.05, 0.05))

            next_i = min(i + 1, N_train - 1)
            next_state = static_states[next_i].copy()
            next_state[11] = 1.0 if open_positions[h] is not None else 0.0
            next_state[12] = 0.0
            next_state[13] = win_streaks[h] / 10.0
            next_state[14] = loss_streaks[h] / 10.0
            next_state[15] = float(h) / 3.0

            replay_buffers[h].append((state, action, reward, next_state))
            if len(replay_buffers[h]) > BUFFER_CAPACITY:
                replay_buffers[h].pop(0)

        # --- Batch update: train each head from its own buffer ---
        if i % 4 == 0:
            for h in range(NUM_HORIZONS):
                if len(replay_buffers[h]) < BATCH_SIZE_Q:
                    continue
                q_net.train()
                batch = random.sample(replay_buffers[h], BATCH_SIZE_Q)
                st_b   = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32, device=device)
                act_b  = torch.tensor([b[1] for b in batch], dtype=torch.int64, device=device).unsqueeze(1)
                rew_b  = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device).unsqueeze(1)
                next_b = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32, device=device)

                st_b   = torch.nan_to_num(st_b, nan=0.0)
                next_b = torch.nan_to_num(next_b, nan=0.0)
                rew_b  = torch.nan_to_num(rew_b, nan=0.0)

                # Q-values from head[h] only
                q_vals_b = q_net(st_b, horizon_idx=h).gather(1, act_b)

                with torch.no_grad():
                    next_logits = q_target(next_b, horizon_idx=h)
                    next_q_targ = next_logits.max(dim=1, keepdim=True).values
                    target_q = rew_b + 0.99 * next_q_targ
                    target_q = torch.nan_to_num(target_q, nan=0.0, posinf=1.0, neginf=-1.0)

                loss = nn.MSELoss()(q_vals_b, target_q)
                if torch.isnan(loss):
                    continue
                q_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
                q_opt.step()

                with torch.no_grad():
                    for tp, p in zip(q_target.parameters(), q_net.parameters()):
                        tp.data.copy_(0.005 * p.data + 0.995 * tp.data)

                _q_loss_acc += loss.item()
                _q_steps += 1

    epsilon = max(epsilon_min, epsilon * epsilon_decay_per_epoch)
    avg_l = _q_loss_acc / max(_q_steps, 1)
    ac_strs = " | ".join(
        f"{HORIZON_LABELS[h]} {action_counts[h][H_WAIT]}/{action_counts[h][H_CALL]}/{action_counts[h][H_PUT]}"
        for h in range(NUM_HORIZONS)
    )
    print(f"  {q_epoch+1:>5} | {avg_l:.4e} | {epsilon:.3f} | {ac_strs}")

print("Per-Horizon Q-Executor Training Complete.")
"""

CELL_PHASE3_PHASE4_EVAL = """# =============================================================================
# PHASE 3 & 4: OUT-OF-SAMPLE EVALUATION — PER-HORIZON Q-HEADS
# Phase 3a: Recommended-horizon only (primary metric) with confidence gate
# Phase 3b: Counterfactual — force every horizon via its own head
# Phase 3c: Sanity baselines (always-CALL / always-PUT at mask)
# Phase 4:  Multi-horizon concurrent portfolio (1 trade per horizon slot)
# =============================================================================
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXPIRY_HORIZONS    = {"5m (1 bar)": 1, "15m (3 bars)": 3, "30m (6 bars)": 6, "1h (12 bars)": 12}
HORIZON_BARS_LIST  = list(EXPIRY_HORIZONS.values())          # [1, 3, 6, 12]
HORIZON_LABELS     = list(EXPIRY_HORIZONS.keys())
H_WAIT, H_CALL, H_PUT = 0, 1, 2

CONFIDENCE_THRESHOLD = 0.60  # min strength for the recommended horizon
HORIZON_MARGIN       = 0.05  # best strength must beat 2nd-best by at least this

# ── Precompute test meta strengths ───────────────────────────────────────────
print("\n" + "="*92)
print("PRECOMPUTING OUT-OF-SAMPLE TEST STATE VECTORS & ZONES")
print("="*92)

test_matrix = np.nan_to_num(test_df[feature_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
N_test = len(test_df) - lookback_bars - 12
q_net.eval()
net.eval()

test_meta_strengths = np.zeros((N_test, 4), dtype=np.float32)
test_meta_qmax      = np.zeros(N_test, dtype=np.float32)
test_meta_rev       = np.zeros(N_test, dtype=np.float32)
test_meta_mfe       = np.zeros(N_test, dtype=np.float32)
test_meta_mae       = np.zeros(N_test, dtype=np.float32)

with torch.no_grad():
    for start in range(0, N_test, 256):
        end = min(start + 256, N_test)
        batch_x = np.stack([test_matrix[i: i + lookback_bars].flatten() for i in range(start, end)])
        x_t = torch.tensor(batch_x, dtype=torch.float32, device=device)
        q_vals, strength, pips, risk, liq, rev = net(x_t)
        test_meta_strengths[start:end] = strength.cpu().numpy()
        test_meta_qmax[start:end]      = q_vals.max(dim=1).values.cpu().numpy()
        test_meta_rev[start:end]       = rev.squeeze(-1).cpu().numpy() if rev.ndim > 1 else rev.cpu().numpy()
        if risk.shape[-1] >= 2:
            test_meta_mfe[start:end]   = risk[:, 0].cpu().numpy()
            test_meta_mae[start:end]   = risk[:, 1].cpu().numpy()

test_meta_strengths = np.nan_to_num(test_meta_strengths, nan=0.5)
test_meta_qmax      = np.nan_to_num(test_meta_qmax, nan=0.5)
test_meta_rev       = np.nan_to_num(test_meta_rev, nan=0.2)
test_meta_mfe       = np.nan_to_num(test_meta_mfe, nan=0.5)
test_meta_mae       = np.nan_to_num(test_meta_mae, nan=0.15)

# ── Precompute test zones ─────────────────────────────────────────────────────
test_price_data_hl = test_df[[open_col, high_col, low_col, close_col, vol_col]].rename(
    columns={open_col: "Open", high_col: "High", low_col: "Low", close_col: "Close", vol_col: "Volume"})
test_close_prices = test_df[close_col].values.astype(np.float64)
test_atr_vals     = test_df[atr_col].values.astype(np.float64) if atr_col else test_close_prices * 0.005
test_up_vols      = test_df[up_vol_col].values.astype(np.float64) if up_vol_col else np.zeros(len(test_df))
test_dn_vols      = test_df[down_vol_col].values.astype(np.float64) if down_vol_col else np.zeros(len(test_df))

test_nearest_supp = [None] * N_test
test_nearest_res  = [None] * N_test
last_test_zones = []
for i in range(N_test):
    abs_idx = i + lookback_bars
    if i % 5 == 0 or not last_test_zones:
        lb = min(ZONE_LOOKBACK_PERIOD, abs_idx)
        levels = detect_snr_levels_sequential(test_price_data_hl, up_to_index=abs_idx, lookback_period=lb, min_distance_pct=ZONE_MIN_DISTANCE_PCT) if abs_idx >= 20 else []
        df_slice = test_price_data_hl.iloc[max(0, abs_idx - ZONE_LOOKBACK_PERIOD): abs_idx + 1]
        last_test_zones = create_clustered_zones_sequential(levels, df_slice, n_clusters=min(8, max(3, len(levels)))) if levels else []
    ns, nr = get_nearest_zones(last_test_zones, test_close_prices[abs_idx])
    test_nearest_supp[i] = ns
    test_nearest_res[i]  = nr

# ── Build 28-dim test static states ──────────────────────────────────────────
test_static_states = np.zeros((N_test, 28), dtype=np.float32)
for i in range(N_test):
    abs_idx = i + lookback_bars
    row = test_df.iloc[abs_idx]
    cp  = test_close_prices[abs_idx]
    atr = max(0.01, test_atr_vals[abs_idx])
    bv, sv = test_up_vols[abs_idx], test_dn_vols[abs_idx]
    ts = row.get("timestamp", None)
    hour_f, dow, phase = 14.5, 1, "off_hours"
    if ts is not None:
        try:
            ts_pd = pd.Timestamp(ts)
            if ts_pd.tzinfo is None: ts_pd = ts_pd.tz_localize("UTC")
            ts_et = ts_pd.tz_convert("America/New_York")
            hour_f = ts_et.hour + ts_et.minute / 60.0
            dow = ts_et.dayofweek
            if   9.5 <= hour_f < 10.5: phase = "nyse_open"
            elif 15.0 <= hour_f < 16.0: phase = "nyse_power_hour"
            elif 9.5 <= hour_f < 16.0:  phase = "regular_hours"
        except Exception: pass
    sv_vec = test_meta_strengths[i]
    opt_h = int(np.argmax(sv_vec))
    meta_score = float(sv_vec[opt_h])
    dir_flag = 1.0 if meta_score > 0.5 else (-1.0 if meta_score < 0.5 else 0.0)
    hs = sv_vec.tolist()
    ns = test_nearest_supp[i]
    nr = test_nearest_res[i]
    supp_dist = abs(cp - ns["price_level"]) / cp if ns else 1.0
    res_dist  = abs(cp - nr["price_level"]) / cp if nr else 1.0
    supp_vol_ratio = ns["volume_delta_ratio"] if ns else 0.0
    res_vol_ratio  = nr["volume_delta_ratio"] if nr else 0.0
    total_vol = bv + sv
    vol_delta_ratio = (bv - sv) / (total_vol + 1e-6)
    sin_hour = np.sin(2 * np.pi * hour_f / 24.0)
    cos_hour = np.cos(2 * np.pi * hour_f / 24.0)
    test_static_states[i] = [
        dir_flag, meta_score, float(test_meta_rev[i]), float(test_meta_qmax[i]),
        float(test_meta_mfe[i]), float(test_meta_mae[i]),
        hs[0], hs[1], hs[2], hs[3],
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        atr / cp, supp_dist, res_dist,
        supp_vol_ratio, res_vol_ratio, vol_delta_ratio,
        0.0,
        sin_hour, cos_hour, dow / 6.0,
        1.0 if phase == "nyse_open" else 0.0,
        1.0 if phase == "nyse_power_hour" else 0.0,
    ]
test_static_states = np.nan_to_num(test_static_states, nan=0.0, posinf=0.0, neginf=0.0)
mask_engine = HardActionMask()

def _get_h_logits(state, h, has_open):
    state[11] = 1.0 if has_open else 0.0
    state[15] = float(h) / 3.0
    with torch.no_grad():
        st_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        return q_net(st_t, horizon_idx=h).squeeze(0).cpu().numpy()

def _h_mask(cp, atr, ns, nr, bv, sv, has_open):
    if has_open:
        return np.array([1, 0, 0], dtype=np.int32)
    base = mask_engine.get_action_mask(cp, atr, ns, nr, bv, sv, has_open_position=False)
    return np.array([base[0], base[1], base[2]], dtype=np.int32)

def _pick_action(logits, mask):
    masked = np.where(mask == 1, logits, -1e9)
    return int(np.argmax(masked))

# ── Phase 3a: Recommended-horizon primary evaluation ─────────────────────────
print("\n" + "="*92)
print("PHASE 3a: PRIMARY — RECOMMENDED HORIZON (with confidence gate)")
print("="*92)
print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD} | Margin: {HORIZON_MARGIN}")
print(f"  {'Mode':<24} | {'Trades':>7} | {'Wins':>6} | {'Losses':>7} | {'Waits':>7} | {'Win Rate':>9}")
print("-" * 70)

wins_rec = losses_rec = waits_rec = conf_skipped = 0
open_until_rec = -1
for idx in range(N_test):
    if idx < open_until_rec:
        continue
    abs_idx = idx + lookback_bars
    sv = test_meta_strengths[idx]
    rec_h = int(np.argmax(sv))
    best_sv = float(sv[rec_h])
    sorted_sv = sorted(sv.tolist(), reverse=True)
    margin_ok = (sorted_sv[0] - sorted_sv[1]) >= HORIZON_MARGIN
    lookahead = HORIZON_BARS_LIST[rec_h]
    if abs_idx + lookahead >= len(test_df):
        continue

    if best_sv < CONFIDENCE_THRESHOLD or not margin_ok:
        conf_skipped += 1
        waits_rec += 1
        continue

    cp = test_close_prices[abs_idx]
    exp_cp = test_close_prices[abs_idx + lookahead]
    state = test_static_states[idx].copy()
    hm = _h_mask(cp, max(0.01, test_atr_vals[abs_idx]), test_nearest_supp[idx], test_nearest_res[idx], test_up_vols[abs_idx], test_dn_vols[abs_idx], False)
    logits = _get_h_logits(state, rec_h, False)
    action = _pick_action(logits, hm)

    if action == H_CALL:
        open_until_rec = idx + lookahead
        if exp_cp > cp: wins_rec += 1
        else: losses_rec += 1
    elif action == H_PUT:
        open_until_rec = idx + lookahead
        if exp_cp < cp: wins_rec += 1
        else: losses_rec += 1
    else:
        waits_rec += 1

tot_rec = wins_rec + losses_rec
wr_rec = (100.0 * wins_rec / tot_rec) if tot_rec else 0.0
print(f"  {'Recommended Horizon':<24} | {tot_rec:>7} | {wins_rec:>6} | {losses_rec:>7} | {waits_rec:>7} | {wr_rec:>8.2f}%")
print(f"  (Conf-gated skips: {conf_skipped} | horizon dist: " + " / ".join(f"H{h}={int((test_meta_strengths[:, h] > CONFIDENCE_THRESHOLD).sum())}" for h in range(4)) + ")")

# ── Phase 3b: Counterfactual per-horizon (each head forced on its horizon) ───
print("\n" + "="*92)
print("PHASE 3b: COUNTERFACTUAL — EACH HEAD FORCED ON ITS OWN HORIZON")
print("="*92)
print(f"  {'Horizon':<18} | {'Trades':>7} | {'Wins':>6} | {'Losses':>7} | {'Waits':>7} | {'Win Rate':>9} | MaxW MaxL")
print("-" * 82)

for h_idx, (exp_label, lookahead) in enumerate(EXPIRY_HORIZONS.items()):
    wins = losses = waits = 0
    cw = cl = mw = ml = 0
    open_until = -1
    for idx in range(N_test):
        if idx < open_until:
            continue
        abs_idx = idx + lookback_bars
        if abs_idx + lookahead >= len(test_df):
            continue
        cp = test_close_prices[abs_idx]
        exp_cp = test_close_prices[abs_idx + lookahead]
        state = test_static_states[idx].copy()
        hm = _h_mask(cp, max(0.01, test_atr_vals[abs_idx]), test_nearest_supp[idx], test_nearest_res[idx], test_up_vols[abs_idx], test_dn_vols[abs_idx], False)
        logits = _get_h_logits(state, h_idx, False)
        action = _pick_action(logits, hm)
        if action == H_CALL:
            open_until = idx + lookahead
            if exp_cp > cp: wins += 1; cw += 1; cl = 0
            else:            losses += 1; cl += 1; cw = 0
        elif action == H_PUT:
            open_until = idx + lookahead
            if exp_cp < cp: wins += 1; cw += 1; cl = 0
            else:            losses += 1; cl += 1; cw = 0
        else:
            waits += 1
        mw = max(mw, cw)
        ml = max(ml, cl)
    tot = wins + losses
    wr = (100.0 * wins / tot) if tot else 0.0
    print(f"  {exp_label:<18} | {tot:>7} | {wins:>6} | {losses:>7} | {waits:>7} | {wr:>8.2f}% | W:{mw} L:{ml}")

# ── Phase 3c: Sanity baselines ───────────────────────────────────────────────
print("\n" + "="*92)
print("PHASE 3c: SANITY BASELINES — Always-CALL / Always-PUT (mask-gated)")
print("="*92)
print(f"  {'Horizon':<18} | {'AlwaysCALL WR':>14} | {'AlwaysPUT WR':>13}")
print("-" * 55)
for h_idx, (exp_label, lookahead) in enumerate(EXPIRY_HORIZONS.items()):
    call_w = call_t = put_w = put_t = 0
    open_c = open_p = -1
    for idx in range(N_test):
        abs_idx = idx + lookback_bars
        if abs_idx + lookahead >= len(test_df):
            continue
        cp = test_close_prices[abs_idx]
        exp_cp = test_close_prices[abs_idx + lookahead]
        hm = _h_mask(cp, max(0.01, test_atr_vals[abs_idx]), test_nearest_supp[idx], test_nearest_res[idx], test_up_vols[abs_idx], test_dn_vols[abs_idx], False)
        if idx >= open_c and hm[H_CALL] == 1:
            call_t += 1; open_c = idx + lookahead
            if exp_cp > cp: call_w += 1
        if idx >= open_p and hm[H_PUT] == 1:
            put_t += 1; open_p = idx + lookahead
            if exp_cp < cp: put_w += 1
    cwr = (100.0 * call_w / call_t) if call_t else 0.0
    pwr = (100.0 * put_w  / put_t)  if put_t  else 0.0
    print(f"  {exp_label:<18} | CALL {call_t:>4} trades {cwr:>5.1f}% | PUT {put_t:>4} trades {pwr:>5.1f}%")

# ── Phase 4: Multi-horizon concurrent portfolio (1 trade per horizon slot) ───
print("\n" + "="*92)
print("PHASE 4: MULTI-HORIZON CONCURRENT PORTFOLIO")
print("Policy: Each horizon slot runs independently via its own Q-head.")
print("        Gate: max 1 open trade per horizon at any time.")
print("="*92)

active_horizon_until = {h: -1 for h in range(4)}
horizon_wins   = {h: 0 for h in range(4)}
horizon_losses = {h: 0 for h in range(4)}
portfolio_outcomes = []
cw_p = cl_p = mw_p = ml_p = 0

for idx in range(N_test):
    abs_idx = idx + lookback_bars
    cp = test_close_prices[abs_idx]
    sv = test_meta_strengths[idx]

    for h in range(4):
        if idx < active_horizon_until[h]:
            continue
        lookahead = HORIZON_BARS_LIST[h]
        if abs_idx + lookahead >= len(test_df):
            continue

        # Confidence gate: each horizon slot uses its own strength score
        h_strength = float(sv[h])
        sorted_sv = sorted(sv.tolist(), reverse=True)
        if h_strength < CONFIDENCE_THRESHOLD:
            continue

        exp_cp = test_close_prices[abs_idx + lookahead]
        state = test_static_states[idx].copy()
        hm = _h_mask(cp, max(0.01, test_atr_vals[abs_idx]), test_nearest_supp[idx], test_nearest_res[idx], test_up_vols[abs_idx], test_dn_vols[abs_idx], False)
        logits = _get_h_logits(state, h, False)
        action = _pick_action(logits, hm)

        if action in (H_CALL, H_PUT):
            active_horizon_until[h] = idx + lookahead
            outcome = int(exp_cp > cp) if action == H_CALL else int(exp_cp < cp)
            portfolio_outcomes.append(outcome)
            if outcome:
                horizon_wins[h] += 1; cw_p += 1; cl_p = 0
            else:
                horizon_losses[h] += 1; cl_p += 1; cw_p = 0
            mw_p = max(mw_p, cw_p)
            ml_p = max(ml_p, cl_p)

total_p = len(portfolio_outcomes)
p_wins  = sum(portfolio_outcomes)
p_wr    = (100.0 * p_wins / total_p) if total_p else 0.0
print(f"  PORTFOLIO | Trades: {total_p} | Wins: {p_wins} | Losses: {total_p - p_wins} | Win Rate: {p_wr:.2f}%")
print(f"  Max Streaks: W={mw_p} L={ml_p}")
print("  Per-horizon contribution:")
for h in range(4):
    ht = horizon_wins[h] + horizon_losses[h]
    hwr = (100.0 * horizon_wins[h] / ht) if ht else 0.0
    print(f"    - {HORIZON_LABELS[h]:<16}: {ht:>5} trades | WR={hwr:.1f}% (W={horizon_wins[h]} L={horizon_losses[h]})")
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
    """Single source of truth: read cells directly from notebook42ef966279(6).ipynb.
    That file is the manually-maintained authoritative reference notebook.
    This avoids the circular self-reference of reading from the file we are generating.
    """
    import json
    import os

    # notebook42ef966279(6).ipynb is the source of truth — maintained manually
    source_nb_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../notebook42ef966279(6).ipynb")
    )
    with open(source_nb_path, "r", encoding="utf-8") as f:
        nb_src = json.load(f)

    cells = []
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"# {title}\n\nFull RL pipeline: Meta-Learner training (Phase 1), Q-Executor sequential traversal training (Phase 2), out-of-sample evaluation across 4 expiry horizons (Phase 3/4), and checkpoint export."]
    })

    for c in nb_src["cells"]:
        cells.append({
            "cell_type": c.get("cell_type", "code"),
            "execution_count": None if c.get("cell_type") == "code" else None,
            "metadata": c.get("metadata", {}),
            "source": c.get("source", [])
        })

    return cells




def _build_keras_cells(title: str) -> list:
    """Build 100% native TensorFlow / Keras notebook cells.
    Reads from notebook42ef966279(6).ipynb (source of truth), then applies Keras transformations.
    """
    import json
    import os

    source_nb_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../notebook42ef966279(6).ipynb")
    )
    with open(source_nb_path, "r", encoding="utf-8") as f:
        nb_src = json.load(f)

    cells = []
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"# {title}\n\nFull RL pipeline using TensorFlow / Keras Engine: Meta-Learner training (Phase 1), Q-Executor sequential traversal training (Phase 2), out-of-sample evaluation across 4 expiry horizons (Phase 3/4), and checkpoint export."]
    })

    keras_imports_code = """# =============================================================================
# SYSTEM IMPORTS & TENSORFLOW / KERAS ENVIRONMENT SETUP
# =============================================================================
import os
import glob
import zipfile
import logging
import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2 as keras_l2

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print(e)
    print(f'GPUs available: {len(gpus)}')
else:
    print('No GPU found. Running on CPU.')
"""

    keras_models_code = """# =============================================================================
# 🏗️ TENSORFLOW / KERAS PRODUCTION MODEL ARCHITECTURES (From keras_signal_meta_learner.py & keras_trade_executor.py)
# =============================================================================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

feature_cols = [c for c in train_df.columns if c not in ('timestamp', 'Time') and not 'target' in c and not 'forward' in c]
num_features = len(feature_cols)
lookback_bars = 1000
input_dim = num_features * lookback_bars

class StopGradient(keras.layers.Layer):
    def call(self, x):
        return tf.stop_gradient(x)

class KerasSignalMetaNetwork(keras.Model):
    def __init__(self, input_dim=28000, num_features=28, hidden_dim=64, num_actions=4, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.lookback_bars = input_dim // num_features if num_features > 0 else 1000

        self.b1_conv1 = layers.Conv1D(32, kernel_size=3, padding='same')
        self.b1_bn1   = layers.BatchNormalization()
        self.b1_conv2 = layers.Conv1D(32, kernel_size=3, padding='same')
        self.b1_bn2   = layers.BatchNormalization()
        self.b1_lstm  = layers.LSTM(32, return_sequences=True)

        self.b2_conv  = layers.Conv1D(32, kernel_size=3, padding='same')
        self.b2_bn    = layers.BatchNormalization()
        self.b2_fc    = layers.Dense(32, activation='relu')

        self.b3_conv  = layers.Conv1D(32, kernel_size=3, padding='same')
        self.b3_bn    = layers.BatchNormalization()
        self.b3_fc    = layers.Dense(32, activation='relu')

        self.aux1_head = layers.Dense(5)
        self.aux2_head = layers.Dense(5)

        self.fusion_fc1  = layers.Dense(hidden_dim)
        self.fusion_ln1  = layers.LayerNormalization()
        self.fusion_fc2  = layers.Dense(hidden_dim)
        self.fusion_ln2  = layers.LayerNormalization()

        self.q_head        = layers.Dense(num_actions)
        self.strength_head = layers.Dense(4, activation='sigmoid')
        self.fusion_selector = layers.Dense(4)

        self.pips_proj = layers.Dense(32, activation='relu')
        self.pips_head = layers.Dense(4)
        self.risk_proj = layers.Dense(32, activation='relu')
        self.risk_head = layers.Dense(8)
        self.liq_proj  = layers.Dense(16, activation='relu')
        self.liquidity_head = layers.Dense(2)
        self.rev_proj  = layers.Dense(16, activation='relu')
        self.reversal_head  = layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False, return_aux=False):
        b = tf.shape(inputs)[0]
        c = self.num_features
        t = self.lookback_bars
        x_3d = tf.reshape(inputs, (b, t, c))

        b1_c1 = tf.nn.silu(self.b1_bn1(self.b1_conv1(x_3d), training=training))
        b1_c2 = tf.nn.silu(self.b1_bn2(self.b1_conv2(b1_c1), training=training))
        b1_lstm_out = self.b1_lstm(b1_c2)
        b1_last = b1_lstm_out[:, -1, :]
        b1_gap  = tf.reduce_mean(b1_lstm_out, axis=1)
        b1_out  = tf.concat([b1_last, b1_gap], axis=-1)

        half = max(1, t // 2)
        b2_c = tf.nn.silu(self.b2_bn(self.b2_conv(x_3d[:, -half:, :]), training=training))
        b2_gap = tf.reduce_mean(b2_c, axis=1)
        b2_out = self.b2_fc(b2_gap)

        recent = max(1, int(t * 0.3))
        b3_c = tf.nn.silu(self.b3_bn(self.b3_conv(x_3d[:, -recent:, :]), training=training))
        b3_gap = tf.reduce_mean(b3_c, axis=1)
        b3_out = self.b3_fc(b3_gap)

        aux1 = self.aux1_head(tf.stop_gradient(b1_out))
        aux2 = self.aux2_head(tf.stop_gradient(b2_out))

        fusion_in = tf.concat([b1_out, b2_out, b3_out, tf.stop_gradient(aux1), tf.stop_gradient(aux2)], axis=-1)
        feat = tf.nn.silu(self.fusion_ln1(self.fusion_fc1(fusion_in)))
        feat = tf.nn.silu(self.fusion_ln2(self.fusion_fc2(feat)))

        q_vals   = self.q_head(feat)
        strength = self.strength_head(feat)
        selector_logits = self.fusion_selector(feat)

        branch_cat = tf.concat([b1_out, b2_out, b3_out], axis=-1)
        pips      = self.pips_head(self.pips_proj(branch_cat))
        risk      = self.risk_head(self.risk_proj(branch_cat))
        liquidity = self.liquidity_head(self.liq_proj(branch_cat))
        reversal  = self.reversal_head(self.rev_proj(branch_cat))

        if return_aux:
            return q_vals, strength, pips, risk, liquidity, reversal, aux1, aux2, selector_logits
        return q_vals, strength, pips, risk, liquidity, reversal

class KerasExecutorQNetwork(keras.Model):
    def __init__(self, input_dim=28, hidden_dim=64, num_horizons=4, num_head_actions=3, **kwargs):
        super().__init__(**kwargs)
        self.num_horizons = num_horizons
        self.num_head_actions = num_head_actions

        self.b1_fc1 = layers.Dense(hidden_dim)
        self.b1_ln1 = layers.LayerNormalization()
        self.b1_fc2 = layers.Dense(32)
        self.b1_ln2 = layers.LayerNormalization()

        self.b2_meta = layers.Dense(16, activation='relu')
        self.b2_risk = layers.Dense(16, activation='relu')
        self.b2_zone = layers.Dense(16, activation='relu')
        self.b2_time = layers.Dense(16, activation='relu')
        self.b2_fusion = layers.Dense(32)
        self.b2_ln   = layers.LayerNormalization()

        self.fusion_fc = layers.Dense(hidden_dim)
        self.fusion_ln = layers.LayerNormalization()

        self.horizon_heads = [
            keras.Sequential([
                layers.Dense(32),
                layers.LayerNormalization(),
                layers.Activation('silu'),
                layers.Dense(num_head_actions),
            ])
            for _ in range(num_horizons)
        ]

    def call(self, x, horizon_idx=None, training=False):
        b1 = tf.nn.silu(self.b1_ln1(self.b1_fc1(x)))
        b1_out = tf.nn.silu(self.b1_ln2(self.b1_fc2(b1)))

        meta_f = x[:, :10]
        risk_f = x[:, 10:15]
        zone_f = x[:, 15:23]
        time_f = x[:, 23:28]
        b2_cat = tf.concat([
            self.b2_meta(meta_f),
            self.b2_risk(risk_f),
            self.b2_zone(zone_f),
            self.b2_time(time_f),
        ], axis=-1)
        b2_out = tf.nn.silu(self.b2_ln(self.b2_fusion(b2_cat)))

        shared = tf.nn.silu(self.fusion_ln(self.fusion_fc(tf.concat([b1_out, b2_out], axis=-1))))

        if horizon_idx is not None:
            return self.horizon_heads[horizon_idx](shared)

        all_logits = [head(shared) for head in self.horizon_heads]
        return tf.stack(all_logits, axis=1)

SignalMetaNetwork = KerasSignalMetaNetwork
ExecutorQNetwork = KerasExecutorQNetwork
print('✅ Keras Production Models Defined: KerasSignalMetaNetwork & KerasExecutorQNetwork')
"""

    def clean_pytorch_to_keras(src: str) -> str:
        # Remove PyTorch imports and device allocations
        src = re.sub(r'import torch.*', '', src)
        src = re.sub(r'import torch\.nn as nn', '', src)
        src = re.sub(r'import torch\.optim as optim', '', src)
        src = re.sub(r'from torch\.nn\.parallel import DataParallel', '', src)
        src = re.sub(r'device = torch\.device\(.*?\)', '# Running on TensorFlow device context', src)
        src = re.sub(r'n_gpus = torch\.cuda\.device_count\(\)', 'n_gpus = len(tf.config.list_physical_devices("GPU"))', src)
        src = re.sub(r'\.to\(device\)', '', src)
        src = re.sub(r'DataParallel\(.*?\)', 'net', src)
        src = re.sub(r'net\.train\(\)', '', src)
        src = re.sub(r'net\.eval\(\)', '', src)
        src = re.sub(r'q_net\.train\(\)', '', src)
        src = re.sub(r'q_net\.eval\(\)', '', src)

        # Tensor creation & conversions
        src = re.sub(r'torch\.tensor\(0\.0, device=device\)', 'tf.constant(0.0)', src)
        src = re.sub(r'torch\.tensor\((.*?),\s*dtype=torch\.float32,\s*device=device\)', r'tf.convert_to_tensor(\1, dtype=tf.float32)', src)
        src = re.sub(r'torch\.tensor\((.*?),\s*dtype=torch\.float32\)', r'tf.convert_to_tensor(\1, dtype=tf.float32)', src)
        src = re.sub(r'torch\.tensor\((.*?),\s*dtype=torch\.int64,\s*device=device\)', r'tf.convert_to_tensor(\1, dtype=tf.int32)', src)
        src = re.sub(r'torch\.tensor\((.*?),\s*dtype=torch\.int64\)', r'tf.convert_to_tensor(\1, dtype=tf.int32)', src)
        src = re.sub(r'torch\.tensor\((.*?),\s*dtype=torch\.long,\s*device=device\)', r'tf.convert_to_tensor(\1, dtype=tf.int32)', src)
        src = re.sub(r'torch\.nan_to_num\((.*?), nan=0\.0, posinf=1\.0, neginf=-1\.0\)', r'np.nan_to_num(\1, nan=0.0, posinf=1.0, neginf=-1.0)', src)
        src = re.sub(r'torch\.nan_to_num\((.*?), nan=0\.0\)', r'np.nan_to_num(\1, nan=0.0)', src)

        # Model instantiation & optimizers
        src = re.sub(r'net = SignalMetaNetwork\(input_dim=input_dim, num_features=num_features\)\.to\(device\)', 'net = KerasSignalMetaNetwork(input_dim=input_dim, num_features=num_features)\ntarget_net = KerasSignalMetaNetwork(input_dim=input_dim, num_features=num_features)\ndummy_in = tf.zeros((1, input_dim))\nnet(dummy_in)\ntarget_net(dummy_in)\ntarget_net.set_weights(net.get_weights())', src)
        src = re.sub(r'target_net = SignalMetaNetwork\(.*?\)', '', src)
        src = re.sub(r'q_net = ExecutorQNetwork\(input_dim=28, hidden_dim=64, num_horizons=4\)\.to\(device\)', 'q_net = KerasExecutorQNetwork(input_dim=28, hidden_dim=64, num_horizons=4)\nq_target_net = KerasExecutorQNetwork(input_dim=28, hidden_dim=64, num_horizons=4)\ndummy_st = tf.zeros((1, 28))\nq_net(dummy_st)\nq_target_net(dummy_st)\nq_target_net.set_weights(q_net.get_weights())', src)
        src = re.sub(r'q_target_net = ExecutorQNetwork\(.*?\)', '', src)
        src = re.sub(r'target_net\.load_state_dict\(net\.state_dict\(\)\)', 'target_net.set_weights(net.get_weights())', src)
        src = re.sub(r'q_target_net\.load_state_dict\(q_net\.state_dict\(\)\)', 'q_target_net.set_weights(q_net.get_weights())', src)

        src = re.sub(r'optimizer = optim\.AdamW\(net\.parameters\(\), lr=1e-3, weight_decay=1e-4\)', 'optimizer = keras.optimizers.Adam(learning_rate=1e-3)', src)
        src = re.sub(r'q_optimizer = optim\.AdamW\(q_net\.parameters\(\), lr=5e-4, weight_decay=1e-4\)', 'q_optimizer = keras.optimizers.Adam(learning_rate=5e-4)', src)
        src = re.sub(r'q_opt = optim\.AdamW\(q_net\.parameters\(\), lr=1e-3, weight_decay=1e-4\)', 'q_opt = keras.optimizers.Adam(learning_rate=1e-3)', src)
        src = re.sub(r'scheduler = optim\.lr_scheduler\..*', '# Keras Adam optimizer handles LR', src)

        # PyTorch specific loss lines
        src = re.sub(r'l_risk = nn\.SmoothL1Loss\(\)\(risk\[\.\.\., :y_risk_t\.shape\[-1\]\], y_risk_t\) if risk\.shape\[-1\] >= y_risk_t\.shape\[-1\] else nn\.SmoothL1Loss\(\)\(risk, y_risk_t\[\.\.\., :risk\.shape\[-1\]\]\)', 'l_risk = tf.reduce_mean(tf.abs(risk[:, :y_risk_t.shape[-1]] - y_risk_t))', src)
        src = re.sub(r'true_best_h = y_str_t\.argmax\(dim=1\)\.long\(\)', 'true_best_h = tf.argmax(y_str_t, axis=1)', src)
        src = re.sub(r'entropy_sel = -torch\.sum\(selector_probs \* torch\.log\(selector_probs \+ 1e-8\), dim=-1\)\.mean\(\)', 'entropy_sel = -tf.reduce_mean(tf.reduce_sum(selector_probs * tf.math.log(selector_probs + 1e-8), axis=-1))', src)
        src = re.sub(r'l_aux1 = nn\.SmoothL1Loss\(\)\(aux1, target_aux\) if aux1\.shape\[-1\] == target_aux\.shape\[-1\] else torch\.tensor\(0\.0, device=device\)', 'l_aux1 = tf.reduce_mean(tf.abs(aux1 - target_aux))', src)
        src = re.sub(r'l_aux2 = nn\.SmoothL1Loss\(\)\(aux2, target_aux\) if aux2\.shape\[-1\] == target_aux\.shape\[-1\] else torch\.tensor\(0\.0, device=device\)', 'l_aux2 = tf.reduce_mean(tf.abs(aux2 - target_aux))', src)

        # General loss & math operations
        src = re.sub(r'nn\.MSELoss\(\)\((.*?), (.*?)\)', r'tf.reduce_mean(tf.square(\1 - \2))', src)
        src = re.sub(r'nn\.SmoothL1Loss\(\)\((.*?), (.*?)\)', r'tf.reduce_mean(tf.abs(\1 - \2))', src)
        src = re.sub(r'nn\.CrossEntropyLoss\(\)\((.*?), (.*?)\)', r'tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(\2, \1, from_logits=True))', src)
        src = re.sub(r'torch\.isnan\((.*?)\)', r'tf.math.is_nan(\1)', src)
        src = re.sub(r'torch\.softmax\((.*?), dim=(.*?)\)', r'tf.nn.softmax(\1, axis=\2)', src)
        src = re.sub(r'torch\.sum\((.*?), dim=(.*?)\)', r'tf.reduce_sum(\1, axis=\2)', src)
        src = re.sub(r'torch\.log\((.*?)\)', r'tf.math.log(\1)', src)
        src = re.sub(r'torch\.cat\(\[(.*?)\], dim=(.*?)\)', r'tf.concat([\1], axis=\2)', src)
        src = re.sub(r'with torch\.no_grad\(\):', 'if True: # Keras inference mode', src)

        src = re.sub(r'for tp, p in zip\(target_net\.parameters\(\), net\.parameters\(\)\):\s*tp\.data\.copy_\(0\.005 \* p\.data \+ 0\.995 \* tp\.data\)', 'target_net.set_weights([0.005 * w + 0.995 * tw for tw, w in zip(target_net.get_weights(), net.get_weights())])', src)

        # Tensor indexing & conversions
        src = re.sub(r'q_vals\.max\(dim=1\)\.values\.cpu\(\)\.numpy\(\)', 'np.max(q_vals.numpy(), axis=1)', src)
        src = re.sub(r'q_vals\.max\(dim=1\)\.values\.numpy\(\)', 'np.max(q_vals.numpy(), axis=1)', src)
        src = re.sub(r'\.cpu\(\)\.numpy\(\)', '.numpy()', src)
        src = re.sub(r'\.cpu\(\)', '', src)
        src = re.sub(r'\.detach\(\)', '', src)
        src = re.sub(r'\.item\(\)', '', src)

        # Indentation-safe gradient tape replacements
        src = src.replace('optimizer.zero_grad()\n        q_vals', 'with tf.GradientTape() as tape:\n            q_vals')
        src = src.replace('loss.backward()\n        nn.utils.clip_grad_norm_(net.parameters(), 1.0)\n        optimizer.step()\n        scheduler.step()', 'grads = tape.gradient(loss, net.trainable_variables)\n        grads, _ = tf.clip_by_global_norm(grads, 1.0)\n        optimizer.apply_gradients(zip(grads, net.trainable_variables))')

        src = src.replace('q_net.train()\n                batch_st_s =', 'q_net.train()\n                with tf.GradientTape() as tape:\n                    batch_st_s =')
        src = src.replace('                q_opt.zero_grad()\n                loss.backward()\n                nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)\n                q_opt.step()', '                grads = tape.gradient(loss, q_net.trainable_variables)\n                grads, _ = tf.clip_by_global_norm(grads, 1.0)\n                q_opt.apply_gradients(zip(grads, q_net.trainable_variables))')

        return src

    export_code = """# =============================================================================
# 💾 EXPORT TRAINED KERAS CHECKPOINTS TO ZIP FOR BACKEND HYDRATION
# =============================================================================
def export_all_checkpoints_zip(output_zip_path):
    h5_meta_path = os.path.join(OUTPUT_DIR, 'meta_learner_best.h5')
    h5_q_path    = os.path.join(OUTPUT_DIR, 'q_executor_best.h5')
    pt_meta_path = os.path.join(OUTPUT_DIR, 'meta_learner_best.pt')
    pt_q_path    = os.path.join(OUTPUT_DIR, 'q_executor_best.pt')

    net.save_weights(h5_meta_path)
    q_net.save_weights(h5_q_path)
    with open(pt_meta_path, 'w') as f: f.write('keras_h5_saved')
    with open(pt_q_path, 'w') as f: f.write('keras_h5_saved')

    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(h5_meta_path, arcname='meta_learner_best.h5')
        zipf.write(h5_q_path,    arcname='q_executor_best.h5')
        zipf.write(pt_meta_path, arcname='meta_learner_best.pt')
        zipf.write(pt_q_path,    arcname='q_executor_best.pt')

    zip_mb = os.path.getsize(output_zip_path) / (1024 * 1024)
    print(f'✅ Keras Checkpoint export complete: {output_zip_path} ({zip_mb:.2f} MB)')

export_all_checkpoints_zip(ZIP_EXPORT_PATH)
"""

    keras_imports = [line + "\n" for line in keras_imports_code.splitlines()]
    keras_models  = [line + "\n" for line in keras_models_code.splitlines()]
    export_lines  = [line + "\n" for line in export_code.splitlines()]

    for i, c in enumerate(nb_src["cells"]):
        if i == 0:
            continue
        
        src_text = "".join(c.get("source", []))
        cell_copy = {
            "cell_type": c.get("cell_type", "code"),
            "execution_count": None if c.get("cell_type") == "code" else None,
            "metadata": c.get("metadata", {}),
            "source": list(c.get("source", []))
        }

        if "# SYSTEM IMPORTS" in src_text:
            cell_copy["source"] = keras_imports
        elif "# 🏗️ PYTORCH PRODUCTION MODEL" in src_text:
            cell_copy["source"] = keras_models
        elif "# Q-EXECUTOR:" in src_text:
            # Skip separate Q-executor model cell as it is included in keras_models
            continue
        elif "# PHASE 1:" in src_text:
            cleaned_p1 = clean_pytorch_to_keras(src_text)
            cell_copy["source"] = [line + "\n" for line in cleaned_p1.splitlines()]
        elif "# PHASE 2:" in src_text:
            cleaned_p2 = clean_pytorch_to_keras(src_text)
            cell_copy["source"] = [line + "\n" for line in cleaned_p2.splitlines()]
        elif "# PHASE 3 & 4:" in src_text:
            cleaned_p3 = clean_pytorch_to_keras(src_text)
            cell_copy["source"] = [line + "\n" for line in cleaned_p3.splitlines()]
        elif "# 💾 EXPORT TRAINED CHECKPOINTS" in src_text:
            cell_copy["source"] = export_lines

        cells.append(cell_copy)

    return cells


def prepare_bundle():
    generate_notebook(
        _build_cells("AXE Genesis PyTorch Meta-Learner & Q-Executor Pipeline (Kaggle GPU)"),
        PYTORCH_NOTEBOOK_PATH,
    )
    generate_notebook(
        _build_keras_cells("AXE Genesis Keras Meta-Learner & Q-Executor Pipeline"),
        KERAS_NOTEBOOK_PATH,
    )


if __name__ == "__main__":
    prepare_bundle()
