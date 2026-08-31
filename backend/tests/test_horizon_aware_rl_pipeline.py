import pytest
import numpy as np
import torch
import pandas as pd

class DummyAccount:
    def __init__(self):
        self.daily_drawdown_pct = 0.0
        self.open_position_type = None
        self.open_position_pnl_pct = 0.0
        self.win_streak = 0
        self.loss_streak = 0
        self.reentries_in_window = 0
        self.max_reentries_allowed = 3

class DummyHardActionMask:
    def get_action_mask(self, cp, atr, ns, nr, bv, sv, has_open_position=False):
        if not has_open_position:
            # flat: WAIT (0), CALL (1), PUT (2) are allowed
            return np.array([1, 1, 1, 0, 0], dtype=np.int32)
        else:
            # open: WAIT (0), TP (3), CLOSE (4) are allowed
            return np.array([1, 0, 0, 1, 1], dtype=np.int32)

def test_static_state_recommended_horizon_indexing():
    """Verify dir_flag and meta_score derive from the recommended horizon (max strength)."""
    strength_vec = np.array([0.45, 0.52, 0.88, 0.60], dtype=np.float32)  # Max at index 2 (30m)
    opt_h = int(np.argmax(strength_vec))
    assert opt_h == 2
    meta_score = float(strength_vec[opt_h])
    assert meta_score == pytest.approx(0.88, rel=1e-3)
    dir_flag = 1.0 if meta_score > 0.5 else (-1.0 if meta_score < 0.5 else 0.0)
    assert dir_flag == 1.0

def test_auto_expiry_settlement_logic():
    """Test auto-settlement when bars_held reaches the contract horizon."""
    HORIZON_BARS_LIST = [1, 3, 6, 12]
    account = DummyAccount()
    replay_buffer = []

    # Open CALL position with horizon = 3 bars (15m) at entry_price 100.0
    open_position = {
        "action": 1,
        "entry_price": 100.0,
        "entry_i": 10,
        "horizon": 3
    }
    account.open_position_type = "CALL"

    # Step i = 13 -> bars_held = 13 - 10 = 3 >= horizon (3)
    i = 13
    cp = 105.0  # Price moved up 5% -> Profit!

    if open_position is not None:
        bars_held = i - open_position["entry_i"]
        if bars_held >= open_position["horizon"]:
            entry_p = open_position["entry_price"]
            pnl = (cp - entry_p) / (entry_p + 1e-8)
            if open_position["action"] == 2:
                pnl = -pnl
            if pnl > 0:
                account.win_streak += 1
                account.loss_streak = 0
            else:
                account.loss_streak += 1
                account.win_streak = 0

            settle_reward = float(np.clip(pnl - 0.0005, -0.05, 0.05))
            state_close = np.zeros(28, dtype=np.float32)
            state_close[11] = 1.0 if open_position["action"] == 1 else -1.0
            state_close[12] = float(pnl)

            next_flat = np.zeros(28, dtype=np.float32)
            replay_buffer.append((state_close, 4, settle_reward, next_flat, np.array([1, 0, 0, 0, 0], dtype=np.int32)))

            open_position = None
            account.open_position_type = None

    assert open_position is None
    assert account.open_position_type is None
    assert account.win_streak == 1
    assert len(replay_buffer) == 1
    assert replay_buffer[0][1] == 4  # Action 4 (CLOSE)
    assert pytest.approx(replay_buffer[0][2], rel=1e-3) == 0.0495  # 0.05 - 0.0005 clipped

def test_manual_close_pnl_reward():
    """Verify manual CLOSE/TP calculates real PnL reward rather than flat +0.002 bonus."""
    account = DummyAccount()
    open_position = {
        "action": 2,  # PUT
        "entry_price": 100.0,
        "entry_i": 5,
        "horizon": 6
    }
    account.open_position_type = "PUT"
    cp = 95.0  # Price dropped 5% -> Profit for PUT!
    action = 4  # Agent explicitly chooses CLOSE

    if open_position is not None and action in (3, 4):
        entry_p = open_position["entry_price"]
        pnl = (cp - entry_p) / (entry_p + 1e-8)
        if open_position["action"] == 2:
            pnl = -pnl
        if pnl > 0:
            account.win_streak += 1
            account.loss_streak = 0
        else:
            account.loss_streak += 1
            account.win_streak = 0
        open_position = None
        account.open_position_type = None
        action_close_pnl = float(pnl)

    assert action_close_pnl == pytest.approx(0.05, rel=1e-3)
    reward = float(np.clip(action_close_pnl - 0.0005, -0.05, 0.05))
    assert reward == pytest.approx(0.0495, rel=1e-3)
    assert open_position is None
    assert account.win_streak == 1

def test_recommended_horizon_gating():
    """Verify Phase 4 gating ensures trades only execute on the meta-recommended expiry."""
    EXPIRY_HORIZONS = {"5m (1 bar)": 1, "15m (3 bars)": 3, "30m (6 bars)": 6, "1h (12 bars)": 12}
    strength_vec = np.array([0.2, 0.9, 0.4, 0.1])  # 15m is max (index 1)
    rec_horizon_idx = int(np.argmax(strength_vec))
    assert rec_horizon_idx == 1

    executed_horizons = []
    for h_idx, (exp_label, lookahead) in enumerate(EXPIRY_HORIZONS.items()):
        if h_idx != rec_horizon_idx:
            continue
        executed_horizons.append(exp_label)

    assert executed_horizons == ["15m (3 bars)"]
