"""
models.py — PyTorch Production Model Architectures (SignalMetaNetwork & ExecutorQNetwork)
Copied verbatim from notebooks/training/axe_signal_shaped_rl_training.ipynb for Axe-paka-v1.
"""

from typing import Any, Callable, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_META_LOOKBACK_BARS = 150
DEFAULT_Q_LOOKBACK = 150
DEFAULT_STATE_DIM = 28
DEFAULT_NUM_FEATURES = 333


class SignalMetaNetwork(nn.Module):
    """
    Tier 1 Signal Strength Meta-Learner PyTorch Network.
    Multi-branch Conv1D + LSTM ensemble architecture with auxiliary supervised heads.
    """

    def __init__(
        self,
        input_dim: int = DEFAULT_META_LOOKBACK_BARS * DEFAULT_NUM_FEATURES,
        num_actions: int = 4,
        hidden_dim: int = 128,
        num_features: int = DEFAULT_NUM_FEATURES,
    ):
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

        # Auxiliary Private Projections (connected to backbone)
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

        # Aux heads
        aux1 = self.aux1_head(b1_out)
        aux2 = self.aux2_head(b2_out)
        aux1_sg = aux1.detach()
        aux2_sg = aux2.detach()

        # Gated Fusion
        fusion_in = torch.cat([b1_out, b2_out, b3_out, aux1_sg, aux2_sg], dim=-1)
        feat = self.fusion_act(self.fusion_ln(self.fusion_fc(fusion_in)))
        feat = self.fusion_act2(self.fusion_ln2(self.fusion_fc2(feat)))

        q_vals   = self.q_head(feat)
        strength = self.strength_head(feat)
        selector_logits = self.fusion_selector(feat)

        # Auxiliary Heads
        branch_cat = self.branch_ln(torch.cat([b1_out, b2_out, b3_out], dim=-1))
        pips      = self.pips_head(self.pips_ln(self.pips_proj(branch_cat)))
        risk      = self.risk_head(self.risk_ln(self.risk_proj(branch_cat)))
        liquidity = self.liquidity_head(self.liq_ln(self.liq_proj(branch_cat)))
        reversal  = self.reversal_head(self.rev_ln(self.rev_proj(branch_cat)))

        if return_aux:
            return q_vals, strength, pips, risk, liquidity, reversal, aux1, aux2, selector_logits
        return q_vals, strength, pips, risk, liquidity, reversal


class ExecutorQNetwork(nn.Module):
    """
    Tier 2 Dual-Specialist Q-Executor PyTorch Network.
    Shared Conv1D indicator trunk + 28-dim context dense encoder + 4 per-horizon specialist towers.
    """

    def __init__(
        self,
        num_features: int = DEFAULT_NUM_FEATURES,
        input_dim: int = DEFAULT_STATE_DIM,
        hidden_dim: int = 128,
        num_actions: int = 5,
        ctx_dim: int = DEFAULT_STATE_DIM,
        q_lookback: int = DEFAULT_Q_LOOKBACK,
        num_horizons: int = 4,
        num_head_actions: int = 3,
        tower_dim: int = 96,
    ):
        super().__init__()
        self.num_features = num_features
        self.ctx_dim = ctx_dim
        self.q_lookback = q_lookback
        self.num_horizons = num_horizons
        self.tower_dim = tower_dim

        in_channels = max(num_features, 1)
        self.feat_conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.feat_bn1   = nn.BatchNorm1d(64)
        self.feat_conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.feat_bn2   = nn.BatchNorm1d(64)
        self.feat_pool  = nn.AdaptiveAvgPool1d(1)
        self.feat_fc    = nn.Linear(64, 64)

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

        self.fusion_fc = nn.Linear(64 + 64 + 64, hidden_dim)
        self.fusion_ln = nn.LayerNorm(hidden_dim)

        def _tower():
            return nn.Sequential(
                nn.Linear(hidden_dim, tower_dim),
                nn.LayerNorm(tower_dim),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(tower_dim, tower_dim),
                nn.LayerNorm(tower_dim),
                nn.SiLU(),
            )

        self.buy_towers = nn.ModuleList([_tower() for _ in range(num_horizons)])
        self.sell_towers = nn.ModuleList([_tower() for _ in range(num_horizons)])

        self.buy_side_heads = nn.ModuleList([
            nn.Linear(tower_dim, 2) for _ in range(num_horizons)
        ])
        self.sell_side_heads = nn.ModuleList([
            nn.Linear(tower_dim, 2) for _ in range(num_horizons)
        ])

        self.decision_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(tower_dim + tower_dim + hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 64),
                nn.SiLU(),
                nn.Linear(64, num_head_actions),
            )
            for _ in range(num_horizons)
        ])

    def _encode_feat(self, feat_window: torch.Tensor) -> torch.Tensor:
        x = feat_window.transpose(1, 2)
        x = F.silu(self.feat_bn1(self.feat_conv1(x)))
        x = F.silu(self.feat_bn2(self.feat_conv2(x)))
        x = self.feat_pool(x).squeeze(-1)
        return F.silu(self.feat_fc(x))

    def _encode_ctx(self, ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h      = F.silu(self.b1_ln1(self.b1_fc1(ctx)))
        b1_out = F.silu(self.b1_ln2(self.b1_fc2(h)))

        meta = F.silu(self.b2_meta(ctx[:, 0:10]))
        risk = F.silu(self.b2_risk(ctx[:, 10:15]))
        zone = F.silu(self.b2_zone(ctx[:, 15:23]))
        time = F.silu(self.b2_time(ctx[:, 23:28]))
        b2_out = F.silu(self.b2_ln(self.b2_fusion(torch.cat([meta, risk, zone, time], dim=-1))))
        return b1_out, b2_out

    def forward(
        self,
        feat_window: torch.Tensor,
        ctx: torch.Tensor,
        horizon_idx: Optional[int] = None,
        return_aux: bool = False,
        return_sides: bool = False,
    ) -> Any:
        feat_h         = self._encode_feat(feat_window)
        b1_out, b2_out = self._encode_ctx(ctx)
        shared = F.silu(
            self.fusion_ln(
                self.fusion_fc(torch.cat([feat_h, b1_out, b2_out], dim=-1))
            )
        )

        def _one(h_idx_int):
            buy_e = self.buy_towers[h_idx_int](shared)
            sell_e = self.sell_towers[h_idx_int](shared)
            buy_side = self.buy_side_heads[h_idx_int](buy_e)
            sell_side = self.sell_side_heads[h_idx_int](sell_e)
            fused = self.decision_heads[h_idx_int](torch.cat([buy_e, sell_e, shared], dim=-1))
            return fused, buy_side, sell_side

        if horizon_idx is not None:
            h = int(horizon_idx)
            fused, buy_side, sell_side = _one(h)
            if return_sides:
                return fused, buy_side, sell_side
            return fused

        outs, buys, sells = [], [], []
        for h in range(self.num_horizons):
            f, b, s = _one(h)
            outs.append(f)
            buys.append(b)
            sells.append(s)
        stacked = torch.stack(outs, dim=1)
        if return_sides:
            return stacked, torch.stack(buys, dim=1), torch.stack(sells, dim=1)
        return stacked


# ── Feature-window helpers ──────────────────────────────────────────────────

def build_feat_window(num_matrix: np.ndarray, abs_idx: int, q_lookback: int = DEFAULT_Q_LOOKBACK) -> np.ndarray:
    """
    Build lookahead-free feature window for index `abs_idx`.
    Left-pads with zeros if start index < 0.
    """
    start = abs_idx - q_lookback + 1
    if start >= 0:
        return num_matrix[start: abs_idx + 1].astype(np.float32)
    window = np.zeros((q_lookback, num_matrix.shape[1]), dtype=np.float32)
    available = num_matrix[: abs_idx + 1]
    window[-len(available):] = available
    return window


def build_feat_window_batch(num_matrix: np.ndarray, abs_indices: Any, q_lookback: int = DEFAULT_Q_LOOKBACK) -> np.ndarray:
    """Vectorized version of build_feat_window for a batch of indices."""
    return np.stack([build_feat_window(num_matrix, int(i), q_lookback) for i in abs_indices])


def make_get_h_logits(q_net: ExecutorQNetwork, num_matrix: np.ndarray, device: Union[str, torch.device], q_lookback: int = DEFAULT_Q_LOOKBACK) -> Callable:
    """Factory helper to construct horizon logits generator."""
    def _get_h_logits(state: np.ndarray, abs_idx: int, h: int, has_open: bool = False) -> np.ndarray:
        st = state.copy()
        st[11] = 1.0 if has_open else 0.0
        st[15] = float(h) / 3.0
        fw = build_feat_window(num_matrix, abs_idx, q_lookback)
        with torch.no_grad():
            fw_t = torch.tensor(fw[None, ...], dtype=torch.float32, device=device)
            st_t = torch.tensor(st[None, ...], dtype=torch.float32, device=device)
            return q_net(fw_t, st_t, horizon_idx=h).squeeze(0).cpu().numpy()
    return _get_h_logits
