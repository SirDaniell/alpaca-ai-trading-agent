"""
Online Reinforcement Meta-Learner for Signal Strength & Forward Movement Statistics.

This module provides a Contextual Q-Learning / Deep Reinforcement Learning meta-learner that
evaluates H1 and lower-timeframe signal quality based on feature state vectors (from IndexedDB /
DataFrame features) and 24-bar forward move outcomes (MFE, MAE, pip gain, reversal timing).

Architecture:
1. ForwardMoveRewardCalculator: 24-bar lookforward reward engine.
2. PrioritizedReplayBuffer: Experience replay queue with TD-error / reward priority sampling.
3. OnlineSignalMetaLearner: PyTorch Q-Network estimating signal strength [0.0 - 1.0] and expected pips.
"""

import math
import random
import logging
import hashlib
import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from app.core.ml.instrument_metadata import get_instrument_metadata
from app.core.ml.ti_meta_features import (
    DECISION_FEATURE_KEYS,
    DECISION_WINDOW_DIM,
    SIGNAL_META_FEATURE_CONTRACT_VERSION,
    SIGNAL_META_LOOKBACK_BARS,
)

logger = logging.getLogger(__name__)

SIGNAL_META_FEATURE_KEYS = tuple(DECISION_FEATURE_KEYS)
SIGNAL_META_FEATURE_COUNT = DECISION_WINDOW_DIM
SIGNAL_META_HORIZON_BARS = 24

# Target normalization constants for auxiliary heads
TARGET_PIP_SCALE = 100.0        # Scale pips, MFE, MAE by 100.0 so target range is ~[0.0 - 5.0]
TARGET_ZONE_DIST_SCALE = 10.0   # Scale ATR zone distance by 10.0 so target range is ~[0.0 - 1.0]
TARGET_ZONE_TYPE_SCALE = 2.0    # Scale zone type (0=none, 1=support, 2=resistance) by 2.0

# Aliases for service/contract interfaces
CANONICAL_FEATURE_NAMES = list(SIGNAL_META_FEATURE_KEYS)
FEATURE_SCHEMA_VERSION = SIGNAL_META_FEATURE_CONTRACT_VERSION
FEATURE_SCHEMA_HASH = hashlib.sha256(
    ",".join(SIGNAL_META_FEATURE_KEYS).encode() + f"|lookback={SIGNAL_META_LOOKBACK_BARS}".encode()
).hexdigest()[:16]
DEFAULT_HORIZON_BARS = SIGNAL_META_HORIZON_BARS
DEFAULT_HORIZON_SECONDS = SIGNAL_META_HORIZON_BARS * 3600


class FeatureScaler:
    """
    Per-training-run StandardScaler for the full decision feature vector.

    Fits once on training data, then frozen.
    Serialized into model checkpoints so inference always uses the correct scale.
    Switching symbols or scopes produces an independent scaler per checkpoint.

    Design notes:
    - Fits on the flat DECISION_WINDOW_DIM vector (48 * num_features)
    - mean_ and scale_ are stored in the checkpoint payload under "scaler"
    - At inference: scaler.transform(live_vec) → feed to net
    - No inverse transform needed: model outputs (pips, probs) are already interpretable
    """

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.n_samples_seen_: int = 0
        self.fitted: bool = False

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        """
        Fit on the training feature matrix.

        Args:
            X: shape (n_samples, input_dim) or (n_samples,) flat vectors
        Returns:
            self (for chaining)
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        self.mean_ = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        # Avoid division by zero: set scale to 1.0 where std is zero
        self.scale_ = np.where(std > 1e-8, std, 1.0).astype(np.float32)
        self.n_samples_seen_ = len(X)
        self.fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Apply standardization: (x - mean) / scale.
        Returns original array if scaler hasn't been fitted yet.
        """
        if not self.fitted:
            return np.asarray(x, dtype=np.float32)
        x = np.asarray(x, dtype=np.float32)
        return ((x - self.mean_) / self.scale_).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit on X and return transformed X."""
        self.fit(X)
        return self.transform(X)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fitted": self.fitted,
            "mean": self.mean_.tolist() if self.mean_ is not None else None,
            "scale": self.scale_.tolist() if self.scale_ is not None else None,
            "n_samples_seen": self.n_samples_seen_,
            "feature_contract_version": FEATURE_SCHEMA_VERSION,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeatureScaler":
        scaler = cls()
        if d.get("fitted") and d.get("mean") is not None:
            scaler.mean_ = np.array(d["mean"], dtype=np.float32)
            scaler.scale_ = np.array(d["scale"], dtype=np.float32)
            scaler.n_samples_seen_ = int(d.get("n_samples_seen", 0))
            scaler.fitted = True
        return scaler



HORIZON_BARS = (1, 3, 6, 12)
HORIZON_LABELS = ("5m", "15m", "30m", "1h")


@dataclass
class ForwardMoveStats:
    """Statistics capturing price trajectory across 4 multi-horizon lookforward windows (5m, 15m, 30m, 1h)."""
    signal_id: str
    symbol: str
    direction: str  # 'bullish' | 'bearish'
    entry_price: float
    lookforward_bars: int = 12
    mfe_pips: float = 0.0          # Max Favorable Excursion in pips (1h)
    mae_pips: float = 0.0          # Max Adverse Excursion in pips (1h)
    net_pips_24h: float = 0.0      # Net move in pips (1h)
    next_zone_dist_atr: float = 0.0 # ATR-normalized distance to next SNR zone
    next_zone_type: float = 0.0     # 0.0=none, 1.0=support, 2.0=resistance
    reversal_prob: float = 0.0      # Decay-weighted reversal probability [0.0 - 1.0]
    reversal_bar: int = -1         # Bar index where reversal occurred (-1 if no reversal)
    reward: float = 0.0            # Calculated RL reward score [-1.0, +1.0] (1h)
    signal_strength: float = 0.5   # Main normalized signal strength [0.0, 1.0]

    # Multi-Horizon Targets & Predictions (5m, 15m, 30m, 1h)
    horizon_strengths: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5, 0.5])
    horizon_rewards: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    horizon_mfes: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    horizon_maes: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    optimal_horizon_idx: int = 2
    recommended_expiry: str = "30m"


class ForwardMoveRewardCalculator:
    """
    Calculates multi-horizon forward move statistics and RL rewards over 4 expiry horizons:
    H1=1 bar (5m), H2=3 bars (15m), H3=6 bars (30m), H4=12 bars (1h).
    """

    def __init__(self, lookforward_bars: int = 12, pip_scale: Optional[float] = None, alpha_mae_penalty: float = 1.5):
        self.lookforward_bars = lookforward_bars
        self.pip_scale = pip_scale
        self.alpha_mae_penalty = alpha_mae_penalty

    def calculate(
        self,
        signal_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        future_highs: np.ndarray,
        future_lows: np.ndarray,
        future_closes: np.ndarray,
        atr_pips: Optional[float] = None,
        next_zone_dist_atr: Optional[float] = None,
        next_zone_type: Optional[float] = None,
        reversal_prob: Optional[float] = None,
    ) -> ForwardMoveStats:
        """
        Compute multi-horizon forward move stats for a signal over [1, 3, 6, 12] bars.
        """
        n_total = len(future_closes)
        if n_total == 0 or entry_price <= 0:
            return ForwardMoveStats(
                signal_id=signal_id,
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                lookforward_bars=self.lookforward_bars,
            )

        scale = self.pip_scale or get_instrument_metadata(symbol).pip_scale
        atr = atr_pips if (atr_pips is not None and atr_pips > 0) else 10.0
        is_bullish = direction.lower() == 'bullish'

        h_strengths, h_rewards, h_mfes, h_maes = [], [], [], []

        for h in HORIZON_BARS:
            n = min(n_total, h)
            highs = future_highs[:n]
            lows = future_lows[:n]
            closes = future_closes[:n]

            if is_bullish:
                fav_diffs = (highs - entry_price) * scale
                adv_diffs = (entry_price - lows) * scale
            else:
                fav_diffs = (entry_price - lows) * scale
                adv_diffs = (highs - entry_price) * scale

            mfe_h = float(np.max(fav_diffs)) if len(fav_diffs) > 0 else 0.0
            mae_h = float(np.max(adv_diffs)) if len(adv_diffs) > 0 else 0.0

            raw_reward = (mfe_h - self.alpha_mae_penalty * mae_h) / max(atr, 1.0)
            reward_h = float(math.tanh(raw_reward / 3.0))
            clipped = max(-60.0, min(60.0, raw_reward))
            strength_h = float(1.0 / (1.0 + math.exp(-clipped)))

            h_mfes.append(mfe_h)
            h_maes.append(mae_h)
            h_rewards.append(reward_h)
            h_strengths.append(strength_h)

        optimal_idx = int(np.argmax(h_strengths))
        recommended_expiry = HORIZON_LABELS[optimal_idx]

        # 1h overall calculations for legacy compatibility
        n_1h = min(n_total, 12)
        if is_bullish:
            net_24h = float((future_closes[n_1h - 1] - entry_price) * scale)
            adv_diffs_1h = (entry_price - future_lows[:n_1h]) * scale
        else:
            net_24h = float((entry_price - future_closes[n_1h - 1]) * scale)
            adv_diffs_1h = (future_highs[:n_1h] - entry_price) * scale

        reversal_threshold = max(0.5 * h_mfes[-1], 1.5 * atr)
        reversal_indices = np.where(adv_diffs_1h > reversal_threshold)[0]
        reversal_bar = int(reversal_indices[0]) if len(reversal_indices) > 0 else -1

        if reversal_prob is None:
            weights = np.exp(-0.1 * np.arange(n_1h))
            rev_events = (adv_diffs_1h > 1.0 * atr).astype(np.float32)
            calculated_rev_prob = float(np.sum(rev_events * weights) / (np.sum(weights) + 1e-6))
            calculated_rev_prob = max(0.0, min(1.0, calculated_rev_prob))
        else:
            calculated_rev_prob = max(0.0, min(1.0, float(reversal_prob)))

        zone_dist = float(next_zone_dist_atr) if next_zone_dist_atr is not None else 2.0
        zone_type = float(next_zone_type) if next_zone_type is not None else 0.0

        return ForwardMoveStats(
            signal_id=signal_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            lookforward_bars=12,
            mfe_pips=h_mfes[-1],
            mae_pips=h_maes[-1],
            net_pips_24h=net_24h,
            next_zone_dist_atr=zone_dist,
            next_zone_type=zone_type,
            reversal_prob=calculated_rev_prob,
            reversal_bar=reversal_bar,
            reward=h_rewards[-1],
            signal_strength=h_strengths[optimal_idx],
            horizon_strengths=h_strengths,
            horizon_rewards=h_rewards,
            horizon_mfes=h_mfes,
            horizon_maes=h_maes,
            optimal_horizon_idx=optimal_idx,
            recommended_expiry=recommended_expiry,
        )



class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Queue storing (State, Action, Reward, NextState, Done) tuples.
    Samples minibatches weighted by TD-error priorities.
    """

    def __init__(self, capacity: int = 5000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer: List[Dict[str, Any]] = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.transition_ids: set[str] = set()

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        target_pips: float = 0.0,
        mfe_pips: float = 0.0,
        mae_pips: float = 0.0,
        next_zone_dist_atr: float = 0.0,
        next_zone_type: float = 0.0,
        reversal_prob: float = 0.0,
        transition_id: Optional[str] = None,
        priority: Optional[float] = None,
    ):
        """Add experience transition to the ring buffer."""
        if transition_id is None:
            transition_id = hashlib.sha1(
                np.asarray(state, dtype=np.float32).tobytes()
                + str(action).encode()
                + str(reward).encode()
            ).hexdigest()
        if transition_id in self.transition_ids:
            return
        max_prio = self.priorities.max() if self.buffer else 1.0
        prio = priority if priority is not None else max_prio

        transition = {
            'state': np.array(state, dtype=np.float32),
            'action': int(action),
            'reward': float(reward),
            'next_state': np.array(next_state, dtype=np.float32),
            'done': bool(done),
            'target_pips': float(target_pips),
            'mfe_pips': float(mfe_pips),
            'mae_pips': float(mae_pips),
            'next_zone_dist_atr': float(next_zone_dist_atr),
            'next_zone_type': float(next_zone_type),
            'reversal_prob': float(reversal_prob),
            'transition_id': transition_id,
        }

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.transition_ids.discard(self.buffer[self.pos]['transition_id'])
            self.buffer[self.pos] = transition

        self.transition_ids.add(transition_id)
        self.priorities[self.pos] = prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int = 32) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        """Sample a minibatch weighted by priorities."""
        if len(self.buffer) == 0:
            raise ValueError("Cannot sample from empty replay buffer")

        current_size = len(self.buffer)
        prios = self.priorities[:current_size]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(current_size, size=min(batch_size, current_size), p=probs, replace=False)
        total = current_size
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states = np.array([self.buffer[idx]['state'] for idx in indices])
        actions = np.array([self.buffer[idx]['action'] for idx in indices])
        rewards = np.array([self.buffer[idx]['reward'] for idx in indices])
        next_states = np.array([self.buffer[idx]['next_state'] for idx in indices])
        dones = np.array([self.buffer[idx]['done'] for idx in indices])
        target_pips = np.array([self.buffer[idx]['target_pips'] for idx in indices])
        mfe_pips = np.array([self.buffer[idx].get('mfe_pips', 0.0) for idx in indices])
        mae_pips = np.array([self.buffer[idx].get('mae_pips', 0.0) for idx in indices])
        next_zone_dist_atr = np.array([self.buffer[idx].get('next_zone_dist_atr', 0.0) for idx in indices])
        next_zone_type = np.array([self.buffer[idx].get('next_zone_type', 0.0) for idx in indices])
        reversal_prob = np.array([self.buffer[idx].get('reversal_prob', 0.0) for idx in indices])

        batch = {
            'states': torch.tensor(states, dtype=torch.float32),
            'actions': torch.tensor(actions, dtype=torch.long),
            'rewards': torch.tensor(rewards, dtype=torch.float32),
            'next_states': torch.tensor(next_states, dtype=torch.float32),
            'dones': torch.tensor(dones, dtype=torch.float32),
            'target_pips': torch.tensor(target_pips, dtype=torch.float32),
            'mfe_pips': torch.tensor(mfe_pips, dtype=torch.float32),
            'mae_pips': torch.tensor(mae_pips, dtype=torch.float32),
            'next_zone_dist_atr': torch.tensor(next_zone_dist_atr, dtype=torch.float32),
            'next_zone_type': torch.tensor(next_zone_type, dtype=torch.float32),
            'reversal_prob': torch.tensor(reversal_prob, dtype=torch.float32),
        }

        return batch, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled indices after TD-error calculation."""
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = max(float(prio), 1e-5)

    def __len__(self) -> int:
        return len(self.buffer)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'capacity': self.capacity,
            'alpha': self.alpha,
            'beta': self.beta,
            'pos': self.pos,
            'transitions': [
                {
                    **transition,
                    'state': transition['state'].tolist(),
                    'next_state': transition['next_state'].tolist(),
                }
                for transition in self.buffer
            ],
            'priorities': self.priorities[:len(self.buffer)].tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> 'PrioritizedReplayBuffer':
        replay = cls(
            capacity=int(payload.get('capacity', 5000)),
            alpha=float(payload.get('alpha', 0.6)),
            beta=float(payload.get('beta', 0.4)),
        )
        for index, transition in enumerate(payload.get('transitions', [])):
            replay.add(
                state=np.asarray(transition['state'], dtype=np.float32),
                action=int(transition['action']),
                reward=float(transition['reward']),
                next_state=np.asarray(transition['next_state'], dtype=np.float32),
                done=bool(transition['done']),
                target_pips=float(transition.get('target_pips', 0.0)),
                mfe_pips=float(transition.get('mfe_pips', 0.0)),
                mae_pips=float(transition.get('mae_pips', 0.0)),
                next_zone_dist_atr=float(transition.get('next_zone_dist_atr', 0.0)),
                next_zone_type=float(transition.get('next_zone_type', 0.0)),
                reversal_prob=float(transition.get('reversal_prob', 0.0)),
                transition_id=str(transition.get('transition_id', f'restored-{index}')),
                priority=float(payload.get('priorities', [1.0] * (index + 1))[index]),
            )
        replay.pos = int(payload.get('pos', replay.pos)) % replay.capacity
        return replay


def _hidden_dim_for(input_dim: int) -> int:
    return 256 if input_dim > 128 else 64


class SignalMetaNetwork(nn.Module):
    """
    3D Temporal Conv1D + LSTM Ensemble Deep PyTorch Network for Signal Meta-Learning.
    Mirrors tt.py AXE Genesis architecture:
    - Preserves 3D Sequence Input shape (Batch, SeqLen, Features) without flattening.
    - Branch 1 (Full Sequence 100%): 2-Block Conv1D + LSTM(64) sequence encoder.
    - Branch 2 (Mid-Term 50% Slice): Private Conv1D temporal encoder over recent 50% sequence.
    - Branch 3 (Short-Term 30% Slice): Private Conv1D temporal encoder over recent 30% sequence.
    - Per-Branch Auxiliary Heads (aux1, aux2): Independent prediction heads on detached branch outputs.
    - StopGradient Isolation: aux predictions are detached before feeding into Gated Ensemble Fusion Head.
    - Gated Ensemble Fusion Head: Combines b1_out (128), b2_out (64), b3_out (64), aux1_sg (5), aux2_sg (5) -> (266) -> hidden_dim (256).
    """

    def __init__(self, input_dim: int = DECISION_WINDOW_DIM, num_actions: int = 4, hidden_dim: int = 256, num_features: int = 238):
        super().__init__()
        self.num_features = num_features
        hidden_dim = hidden_dim or _hidden_dim_for(input_dim)

        # Branch 1: Full Sequence (100%) Conv1D + LSTM Tower
        self.b1_conv1 = nn.Conv1d(num_features, 128, kernel_size=3, padding=1)
        self.b1_bn1   = nn.BatchNorm1d(128)
        self.b1_act1  = nn.SiLU()
        self.b1_conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.b1_bn2   = nn.BatchNorm1d(64)
        self.b1_act2  = nn.SiLU()
        self.b1_lstm  = nn.LSTM(64, 64, batch_first=True)

        # Branch 2: Mid-Term (50% Slice) Conv1D Tower
        self.b2_conv  = nn.Conv1d(num_features, 64, kernel_size=3, padding=1)
        self.b2_bn    = nn.BatchNorm1d(64)
        self.b2_act   = nn.SiLU()
        self.b2_fc    = nn.Linear(64, 64)

        # Branch 3: Short-Term (30% Slice) Conv1D Tower
        self.b3_conv  = nn.Conv1d(num_features, 64, kernel_size=3, padding=1)
        self.b3_bn    = nn.BatchNorm1d(64)
        self.b3_act   = nn.SiLU()
        self.b3_fc    = nn.Linear(64, 64)

        # Auxiliary Supervised Heads per branch
        self.aux1_head = nn.Linear(128, 5)  # [strength_5m, strength_15m, strength_30m, strength_1h, reversal_aux]
        self.aux2_head = nn.Linear(64, 5)

        # Gated Ensemble Fusion Head
        # Inputs: b1_out (128) + b2_out (64) + b3_out (64) + aux1_sg (5) + aux2_sg (5) = 266
        self.fusion_fc   = nn.Linear(128 + 64 + 64 + 5 + 5, hidden_dim)
        self.fusion_ln   = nn.LayerNorm(hidden_dim)
        self.fusion_act  = nn.SiLU()

        self.q_head = nn.Linear(hidden_dim, num_actions)
        self.strength_head = nn.Sequential(
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),
        )

        # Auxiliary Private Projections (Zero Gradient Interference)
        _aux_in = 128 + 64 + 64  # b1(128) + b2(64) + b3(64) = 256
        self.branch_ln = nn.LayerNorm(_aux_in)

        self.pips_proj = nn.Linear(_aux_in, 64)
        self.pips_ln   = nn.LayerNorm(64)
        self.pips_head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 4),
        )

        self.risk_proj = nn.Linear(_aux_in, 64)
        self.risk_ln   = nn.LayerNorm(64)
        self.risk_head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 8),
        )

        self.liq_proj = nn.Linear(_aux_in, 32)
        self.liq_ln   = nn.LayerNorm(32)
        self.liquidity_head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 2),
        )

        self.rev_proj = nn.Linear(_aux_in, 32)
        self.rev_ln   = nn.LayerNorm(32)
        self.reversal_head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def _prepare_3d(self, x: torch.Tensor) -> torch.Tensor:
        """Convert flat (B, T*C) or 2D inputs into 3D (B, T, C) sequence shape."""
        if x.ndim == 2:
            b, dim = x.shape
            c = self.num_features
            t = dim // c if dim >= c else 1
            if t * c != dim:
                # If feature dimension doesn't divide cleanly, pad or truncate
                c = dim
                t = 1
            return x.view(b, t, c)
        return x

    def forward(
        self, x: torch.Tensor, return_aux: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[Any, ...]:
        x_3d = self._prepare_3d(x)  # (B, T, C)
        b, t, c = x_3d.shape

        # Transpose to (B, C, T) for PyTorch Conv1D
        x_trans = x_3d.transpose(1, 2)

        # ── Branch 1: Full Sequence (100%) Conv1D + LSTM ────────────────────
        b1_c1 = self.b1_act1(self.b1_bn1(self.b1_conv1(x_trans)))
        b1_c2 = self.b1_act2(self.b1_bn2(self.b1_conv2(b1_c1)))
        b1_c2_trans = b1_c2.transpose(1, 2)  # (B, T, 64)
        b1_lstm_out, _ = self.b1_lstm(b1_c2_trans)  # (B, T, 64)
        b1_last = b1_lstm_out[:, -1, :]  # (B, 64)
        b1_gap  = torch.mean(b1_lstm_out, dim=1)  # (B, 64)
        b1_out  = torch.cat([b1_last, b1_gap], dim=-1)  # (B, 128)

        # ── Branch 2: Mid-Term (50% Slice) Conv1D ───────────────────────────
        half = max(1, t // 2)
        x_mid_trans = x_trans[:, :, -half:]
        b2_c = self.b2_act(self.b2_bn(self.b2_conv(x_mid_trans)))
        b2_gap = torch.mean(b2_c, dim=-1)  # (B, 64)
        b2_out = torch.relu(self.b2_fc(b2_gap))  # (B, 64)

        # ── Branch 3: Short-Term (30% Slice) Conv1D ──────────────────────────
        recent = max(1, int(t * 0.3))
        x_rec_trans = x_trans[:, :, -recent:]
        b3_c = self.b3_act(self.b3_bn(self.b3_conv(x_rec_trans)))
        b3_gap = torch.mean(b3_c, dim=-1)  # (B, 64)
        b3_out = torch.relu(self.b3_fc(b3_gap))  # (B, 64)

        # ── Auxiliary Supervision & StopGradient Detaching ───────────────────
        aux1 = self.aux1_head(b1_out.detach())
        aux2 = self.aux2_head(b2_out.detach())

        aux1_sg = aux1.detach()
        aux2_sg = aux2.detach()

        # ── Gated Ensemble Fusion Layer ──────────────────────────────────────
        fusion_in = torch.cat([b1_out, b2_out, b3_out, aux1_sg, aux2_sg], dim=-1)
        feat = self.fusion_act(self.fusion_ln(self.fusion_fc(fusion_in)))

        q_vals   = self.q_head(feat)
        strength = self.strength_head(feat)

        # ── Private-projection auxiliary heads (Zero Gradient Interference) ─
        branch_cat = self.branch_ln(torch.cat([b1_out, b2_out, b3_out], dim=-1).detach())  # (B, 256)

        pips      = self.pips_head(self.pips_ln(self.pips_proj(branch_cat)))
        risk      = self.risk_head(self.risk_ln(self.risk_proj(branch_cat)))
        liquidity = self.liquidity_head(self.liq_ln(self.liq_proj(branch_cat)))
        reversal  = self.reversal_head(self.rev_ln(self.rev_proj(branch_cat)))

        if return_aux:
            return q_vals, strength, pips, risk, liquidity, reversal, aux1, aux2
        return q_vals, strength, pips, risk, liquidity, reversal



class OnlineSignalMetaLearner:
    """
    Online Reinforcement Meta-Learner for Signal Strength.

    Features:
    - Contextual Q-Learning online parameter updates with Multi-Head Auxiliary Losses.
    - Auxiliary Heads: Risk (MFE/MAE), Liquidity (Next Zone), and Reversal Probability.
    - Experience Replay Queue for stable non-repainting online learning.
    - Signal Strength scoring [0.0 - 1.0] and 24-bar forward move estimation.
    - Decision input is the last 48 bars of the full TI + DXY/SNR context matrix.
    """

    def __init__(
        self,
        input_dim: int = DECISION_WINDOW_DIM,
        num_actions: int = 4,
        lr: float = 1e-3,
        gamma: float = 0.95,
        replay_capacity: int = 5000,
        lookback_bars: int = SIGNAL_META_LOOKBACK_BARS,
    ):
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.total_steps = 0
        self.lookback_bars = lookback_bars
        hidden_dim = _hidden_dim_for(input_dim)

        self.net = SignalMetaNetwork(input_dim=input_dim, num_actions=num_actions, hidden_dim=hidden_dim)
        self.target_net = SignalMetaNetwork(input_dim=input_dim, num_actions=num_actions, hidden_dim=hidden_dim)
        self.target_net.load_state_dict(self.net.state_dict())

        self.optimizer = optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        self.replay_buffer = PrioritizedReplayBuffer(capacity=replay_capacity)
        self.reward_calculator = ForwardMoveRewardCalculator(
            lookforward_bars=SIGNAL_META_HORIZON_BARS,
        )
        self.feature_keys = list(SIGNAL_META_FEATURE_KEYS)
        self.feature_contract_version = SIGNAL_META_FEATURE_CONTRACT_VERSION
        # Scaler fitted on training data only; frozen thereafter
        self.scaler = FeatureScaler()

    def fit_scaler(self, X: np.ndarray) -> None:
        """
        Fit the feature scaler on the TRAINING split only.
        Call this once after assembling all training feature vectors,
        before passing them to record_experience / train_step.
        The scaler is frozen after this call and shared across val/test/live.
        """
        self.scaler.fit(X)
        logger.info(
            "[MetaLearner] FeatureScaler fitted on %d training samples (%s features).",
            self.scaler.n_samples_seen_,
            X.shape[1] if np.asarray(X).ndim == 2 else "flat",
        )

    def extract_features(self, feature_input: Union[Dict[str, Any], List[float], np.ndarray]) -> np.ndarray:
        """Convert a 48-bar window, flat vector, or last-bar dict into the network input."""
        if isinstance(feature_input, np.ndarray):
            arr = np.asarray(feature_input, dtype=np.float32)
            if arr.ndim == 2:
                vec = arr.reshape(-1)
            else:
                vec = np.where(np.isfinite(arr.reshape(-1)), arr.reshape(-1), 0.0)
        elif isinstance(feature_input, (list, tuple)):
            raw = [float(x) if x is not None and np.isfinite(float(x)) else 0.0 for x in feature_input]
            vec = np.array(raw, dtype=np.float32)
        elif isinstance(feature_input, dict):
            per_bar = []
            for key in self.feature_keys:
                val = float(feature_input.get(key, 0.0))
                if math.isnan(val) or math.isinf(val):
                    val = 0.0
                per_bar.append(val)
            # Last-bar dict → pad left so the live bar is the last of 48
            window = np.zeros((self.lookback_bars, len(self.feature_keys)), dtype=np.float32)
            window[-1] = np.array(per_bar, dtype=np.float32)
            vec = window.reshape(-1)
        else:
            vec = np.zeros(self.input_dim, dtype=np.float32)

        if len(vec) < self.input_dim:
            vec = np.concatenate([np.zeros(self.input_dim - len(vec), dtype=np.float32), vec])
        vec = vec[:self.input_dim]
        # Apply frozen scaler (no-op if not yet fitted)
        return self.scaler.transform(vec)

    def predict(
        self,
        feature_dict: Dict[str, Any],
        signal_type: str = 'rsi-ma-cross',
        direction: str = 'bullish',
    ) -> Dict[str, Any]:
        """
        Evaluate a signal event and return its signal strength score and forward statistics.
        """
        x_vec = self.extract_features(feature_dict)
        x_tensor = torch.tensor(x_vec, dtype=torch.float32).unsqueeze(0)

        self.net.eval()
        with torch.no_grad():
            q_vals, strength_tensor, pips_tensor, risk_tensor, liquidity_tensor, reversal_tensor = self.net(x_tensor)

        action_idx = 0 if direction.lower() == 'bullish' else 1
        q_val = float(q_vals[0, action_idx].item())

        h_strengths = [float(strength_tensor[0, i].item()) for i in range(4)]
        opt_idx = int(np.argmax(h_strengths))
        rec_expiry = HORIZON_LABELS[opt_idx]
        strength = float(h_strengths[opt_idx])

        expected_pips = float(pips_tensor[0, opt_idx].item()) * TARGET_PIP_SCALE
        pred_mfe = float(risk_tensor[0, opt_idx * 2].item()) * TARGET_PIP_SCALE
        pred_mae = float(risk_tensor[0, opt_idx * 2 + 1].item()) * TARGET_PIP_SCALE
        pred_zone_dist = float(liquidity_tensor[0, 0].item()) * TARGET_ZONE_DIST_SCALE
        pred_zone_type = float(liquidity_tensor[0, 1].item()) * TARGET_ZONE_TYPE_SCALE
        pred_reversal_prob = float(reversal_tensor[0, 0].item())

        return {
            'signal_strength': round(strength, 4),
            'horizon_strengths': [round(s, 4) for s in h_strengths],
            'optimal_horizon_idx': opt_idx,
            'recommended_expiry': rec_expiry,
            'expected_pips': round(expected_pips, 2),
            'expected_mfe_pips': round(pred_mfe, 2),
            'expected_mae_pips': round(pred_mae, 2),
            'next_zone_dist_atr': round(pred_zone_dist, 2),
            'next_zone_type': round(pred_zone_type, 1),
            'reversal_prob': round(pred_reversal_prob, 4),
            'confidence': round(float(1.0 / (1.0 + math.exp(-q_val))), 4),
            'q_value': round(q_val, 4),
        }

    def record_experience(
        self,
        feature_dict: Dict[str, Any],
        signal_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        future_highs: np.ndarray,
        future_lows: np.ndarray,
        future_closes: np.ndarray,
        atr_pips: Optional[float] = None,
        next_feature_dict: Optional[Dict[str, Any]] = None,
        next_zone_dist_atr: Optional[float] = None,
        next_zone_type: Optional[float] = None,
        reversal_prob: Optional[float] = None,
    ) -> ForwardMoveStats:
        """
        Record a multi-horizon transition into the prioritized experience replay queue.
        """
        state = self.extract_features(feature_dict)
        next_state = self.extract_features(next_feature_dict) if next_feature_dict is not None else state

        stats = self.reward_calculator.calculate(
            signal_id=signal_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            future_highs=future_highs,
            future_lows=future_lows,
            future_closes=future_closes,
            atr_pips=atr_pips,
            next_zone_dist_atr=next_zone_dist_atr,
            next_zone_type=next_zone_type,
            reversal_prob=reversal_prob,
        )

        action = 0 if direction.lower() == 'bullish' else 1
        priority = float(abs(stats.reward) + 0.1)

        self.replay_buffer.add(
            state=state,
            action=action,
            reward=stats.reward,
            next_state=next_state,
            done=True,
            target_pips=stats.net_pips_24h,
            mfe_pips=stats.mfe_pips,
            mae_pips=stats.mae_pips,
            next_zone_dist_atr=stats.next_zone_dist_atr,
            next_zone_type=stats.next_zone_type,
            reversal_prob=stats.reversal_prob,
            transition_id=signal_id,
            priority=priority,
        )

        return stats

    def train_step(self, batch_size: int = 32) -> Dict[str, float]:
        """Perform one online gradient update step using sampled experience from the replay queue."""
        if len(self.replay_buffer) < max(batch_size, 1):
            return {'loss': 0.0, 'buffer_size': len(self.replay_buffer)}

        self.net.train()
        batch, indices, weights = self.replay_buffer.sample(batch_size=batch_size)

        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        target_pips = batch['target_pips']
        mfe_pips = batch['mfe_pips']
        mae_pips = batch['mae_pips']
        next_zone_dist_atr = batch['next_zone_dist_atr']
        next_zone_type = batch['next_zone_type']
        reversal_prob = batch['reversal_prob']
        weights_t = torch.tensor(weights, dtype=torch.float32)

        q_vals, strength_pred, pips_pred, risk_pred, liquidity_pred, reversal_pred, aux1, aux2 = self.net(states, return_aux=True)
        with torch.no_grad():
            next_q_vals, _, _, _, _, _ = self.target_net(next_states)
            max_next_q, _ = torch.max(next_q_vals, dim=1)
            target_q = rewards + (1.0 - dones) * self.gamma * max_next_q

        state_action_q = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)
        td_error = target_q - state_action_q
        loss_q = torch.mean(weights_t * (td_error ** 2))

        # Reward-to-strength target
        target_strength = torch.clamp((rewards.unsqueeze(1) + 1.0) / 2.0, 0.0, 1.0)
        loss_strength = torch.mean((strength_pred[:, :1] - target_strength) ** 2)

        # Scale pips & risk targets by TARGET_PIP_SCALE (100.0) so raw loss is ~0.01 - 0.5
        scaled_target_pips = target_pips / TARGET_PIP_SCALE
        loss_pips = torch.mean(weights_t * nn.functional.smooth_l1_loss(
            pips_pred[:, :1].squeeze(1), scaled_target_pips, reduction='none',
        ))

        # Auxiliary Head Losses (MFE/MAE scaled by TARGET_PIP_SCALE)
        target_risk = torch.stack([mfe_pips / TARGET_PIP_SCALE, mae_pips / TARGET_PIP_SCALE], dim=1)
        loss_risk = torch.mean(weights_t * nn.functional.smooth_l1_loss(
            risk_pred[:, :2], target_risk, reduction='none'
        ).mean(dim=1))

        # Liquidity targets: distance normalized by 10.0, type by 2.0
        # Guard: skip liquidity loss when all targets are zero (missing data)
        target_liquidity = torch.stack([
            next_zone_dist_atr / TARGET_ZONE_DIST_SCALE,
            next_zone_type / TARGET_ZONE_TYPE_SCALE
        ], dim=1)
        liq_mask = (target_liquidity.abs().sum(dim=1) > 1e-6).float()
        if liq_mask.sum() > 0:
            loss_liquidity = torch.mean(
                liq_mask * nn.functional.smooth_l1_loss(
                    liquidity_pred, target_liquidity, reduction='none'
                ).mean(dim=1)
            )
        else:
            loss_liquidity = torch.zeros(1, device=states.device).squeeze()

        loss_reversal = torch.mean(weights_t * ((reversal_pred.squeeze(1) - reversal_prob) ** 2))

        # Branch-specific auxiliary supervision losses
        target_aux = torch.cat([target_strength.repeat(1, 4), reversal_prob.unsqueeze(1)], dim=1)
        loss_aux1 = torch.mean(weights_t * nn.functional.smooth_l1_loss(aux1, target_aux, reduction='none').mean(dim=1))
        loss_aux2 = torch.mean(weights_t * nn.functional.smooth_l1_loss(aux2, target_aux, reduction='none').mean(dim=1))

        # Balanced Total Loss: raised weights for pips/risk/reversal to match Keras capacity
        total_loss = (
            loss_q
            + loss_strength
            + 0.3 * loss_pips
            + 0.3 * loss_risk
            + 0.05 * loss_liquidity
            + 0.3 * loss_reversal
            + 0.15 * loss_aux1
            + 0.15 * loss_aux2
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.total_steps += 1

        # Update priorities in replay buffer based on TD errors
        new_prios = np.abs(td_error.detach().cpu().numpy()) + 1e-4
        self.replay_buffer.update_priorities(indices, new_prios)

        return {
            'loss': float(total_loss.item()),
            'loss_q': float(loss_q.item()),
            'loss_strength': float(loss_strength.item()),
            'loss_pips': float(loss_pips.item()),
            'loss_risk': float(loss_risk.item()),
            'loss_liquidity': float(loss_liquidity.item()),
            'loss_reversal': float(loss_reversal.item()),
            'loss_aux1': float(loss_aux1.item()),
            'loss_aux2': float(loss_aux2.item()),
            'buffer_size': len(self.replay_buffer),
        }


    def sync_target_network(self):
        """Soft/hard sync of target network weights."""
        self.target_net.load_state_dict(self.net.state_dict())

    def export_checkpoint(self) -> bytes:
        """Serialize all mutable learner state for durable checkpoint storage."""
        payload = {
            'format_version': 2,
            'input_dim': self.input_dim,
            'num_actions': self.num_actions,
            'gamma': self.gamma,
            'total_steps': self.total_steps,
            'feature_contract_version': self.feature_contract_version,
            'feature_keys': list(self.feature_keys),
            'horizon_bars': SIGNAL_META_HORIZON_BARS,
            'lookback_bars': self.lookback_bars,
            'network': self.net.state_dict(),
            'target_network': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'replay': self.replay_buffer.to_dict(),
            'scaler': self.scaler.to_dict(),
        }
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        return buffer.getvalue()

    def import_checkpoint(self, checkpoint: bytes) -> None:
        """Restore a checkpoint only when its architecture and contract match."""
        payload = torch.load(io.BytesIO(checkpoint), map_location='cpu', weights_only=False)
        fmt = payload.get('format_version', 1)
        if fmt not in (1, 2):
            raise ValueError('Unsupported signal meta-learner checkpoint format')
        if payload.get('input_dim') != self.input_dim or payload.get('num_actions') != self.num_actions:
            raise ValueError('Checkpoint architecture does not match learner')
        if payload.get('feature_contract_version') != self.feature_contract_version:
            raise ValueError('Checkpoint feature contract does not match learner')
        if payload.get('feature_keys') != list(self.feature_keys):
            raise ValueError('Checkpoint feature ordering does not match learner')
        if payload.get('horizon_bars') != SIGNAL_META_HORIZON_BARS:
            raise ValueError('Checkpoint horizon does not match learner')
        if int(payload.get('lookback_bars', SIGNAL_META_LOOKBACK_BARS)) != self.lookback_bars:
            raise ValueError('Checkpoint lookback does not match learner')

        self.net.load_state_dict(payload['network'])
        self.target_net.load_state_dict(payload['target_network'])
        self.optimizer.load_state_dict(payload['optimizer'])
        self.replay_buffer = PrioritizedReplayBuffer.from_dict(payload['replay'])
        self.total_steps = int(payload.get('total_steps', 0))
        # Restore scaler (format_version >= 2); older checkpoints start unfitted
        if 'scaler' in payload:
            self.scaler = FeatureScaler.from_dict(payload['scaler'])
        else:
            self.scaler = FeatureScaler()

