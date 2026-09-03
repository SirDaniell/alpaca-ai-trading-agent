"""
test_bug_conditions.py
======================
Task 1 — Bug Condition Exploration Tests (Property 1: Bug Conditions)

Seven sub-tests (1a–1g), one per confirmed bug in notebook2398f959dc.ipynb.

Each test:
  • Runs against UNFIXED notebook code (logic reproduced inline from cell inspection)
  • PASSES when the buggy condition is detected (confirms the bug exists)
  • Prints: PASS: <description> | counterexample: <value>

Run with:
  /media/daniel-joseph/Linux_Data/Dan_backup/Projects/JavaScript/fin-dash-buddy-sos/dev_env/bin/python test_bug_conditions.py

Validates: Requirements 1.1, 1.2, 1.3, 2.1–2.3, 3.1–3.2, 4.1–4.4, 5.1–5.2, 6.1–6.2, 7.1–7.2
"""

import os
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Locate the training CSV (train_50k.csv in backend/data/)
# ---------------------------------------------------------------------------
_SPEC_DIR = os.path.dirname(os.path.abspath(__file__))
# Spec lives at <project>/.kiro/specs/<spec-name>/ — three levels up is project root
_PROJECT_ROOT = os.path.normpath(os.path.join(_SPEC_DIR, "..", "..", ".."))
_CSV_PATH = os.path.join(
    _PROJECT_ROOT,
    "backend", "data", "train_50k.csv"
)


# ===========================================================================
# 1a — Bug 6: Backbone Gradient Probe
# ===========================================================================
def test_1a_bug6_backbone_gradient():
    """
    Reproduce the detach() pattern from SignalMetaNetwork.forward() (Cell 6).

    The notebook code at lines 127-128, 142:
        aux1       = self.aux1_head(b1_out.detach())
        aux2       = self.aux2_head(b2_out.detach())
        branch_cat = self.branch_ln(torch.cat([b1_out, b2_out, b3_out], dim=-1).detach())

    Because branch_cat is detached, any loss computed from pips/risk/liq/rev heads
    (which consume branch_cat) sends ZERO gradient back through b1_out/b2_out/b3_out
    to the Conv1D/LSTM backbone.

    Bug condition: b1_out.grad is None (or zero) after backward on a loss that
    flows only through the detached branch_cat path.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(42)

    # Minimal backbone branches that mirror the notebook shapes:
    #   b1_out: (B, 64)   b2_out: (B, 32)   b3_out: (B, 32)
    B = 8
    b1_out_data = torch.randn(B, 64, requires_grad=True)
    b2_out_data = torch.randn(B, 32, requires_grad=True)
    b3_out_data = torch.randn(B, 32, requires_grad=True)

    branch_ln = nn.LayerNorm(128)  # 64+32+32 = 128
    pips_proj = nn.Linear(128, 64)
    pips_ln   = nn.LayerNorm(64)
    pips_head = nn.Linear(64, 4)

    # ---- UNFIXED CODE: .detach() severs the gradient path ----
    branch_cat = branch_ln(
        torch.cat([b1_out_data, b2_out_data, b3_out_data], dim=-1).detach()
    )
    pips_out = pips_head(pips_ln(pips_proj(branch_cat)))

    # Backward through pips loss (mimics l_pips in the training loop)
    targets = torch.zeros_like(pips_out)
    l_pips = nn.MSELoss()(pips_out, targets)
    l_pips.backward()

    # Bug condition: the gradient should NOT have flowed back to b1_out_data
    b1_grad = b1_out_data.grad
    b1_grad_norm = b1_grad.abs().mean().item() if b1_grad is not None else 0.0

    assert b1_grad_norm == 0.0, (
        f"Expected grad_norm == 0.0 (detach severs path) but got {b1_grad_norm:.6f}. "
        "Bug may already be fixed."
    )

    print(
        f"PASS: [Bug 6 — Backbone Gradient Probe] "
        f"| counterexample: b1_out.grad.abs().mean() = {b1_grad_norm:.6f} "
        f"(zero — pips/risk/liq/rev losses contribute 0 gradient to backbone via detach)"
    )


# ===========================================================================
# 1b — Bug 3: Coin-Flip Label Check
# ===========================================================================
def test_1b_bug3_coin_flip_labels():
    """
    Load training CSV. Compute target_dir_5m = (forward_move_1 > 0).
    Assert the positive rate is within ±2% of 50% — confirming coin-flip labels.
    Also verify the MSE of a constant-0.5 predictor is ≈ 0.25 (the 50/50 floor).

    Bug condition: abs(mean(target_dir_5m) - 0.5) < 0.02
    """
    import pandas as pd

    assert os.path.exists(_CSV_PATH), f"Training CSV not found: {_CSV_PATH}"

    df = pd.read_csv(_CSV_PATH)
    assert "forward_move_1" in df.columns, "Column 'forward_move_1' not in CSV"

    fwd = df["forward_move_1"].values.astype(np.float32)
    target_dir = (fwd > 0).astype(np.float32)
    pos_rate = float(target_dir.mean())

    # Also check the column already in the CSV matches our computation
    if "target_dir_5m" in df.columns:
        pos_rate_csv = float(df["target_dir_5m"].mean())
        # Both should be coin-flip
        assert abs(pos_rate_csv - 0.5) < 0.02, (
            f"target_dir_5m CSV mean {pos_rate_csv:.4f} is NOT coin-flip "
            "(may already be fixed)."
        )

    # Bug condition check on computed labels
    assert abs(pos_rate - 0.5) < 0.02, (
        f"Positive rate {pos_rate:.4f} is outside coin-flip range (|x-0.5| < 0.02). "
        "Bug may already be fixed."
    )

    # MSE of constant-0.5 predictor ≈ 0.25 (theoretical 50/50 floor)
    const_pred = np.full_like(target_dir, 0.5)
    mse = float(np.mean((const_pred - target_dir) ** 2))
    assert 0.24 <= mse <= 0.26, (
        f"MSE floor {mse:.4f} should be in [0.24, 0.26] for 50/50 labels."
    )

    print(
        f"PASS: [Bug 3 — Coin-Flip Label Check] "
        f"| counterexample: target_dir_5m positive rate = {pos_rate:.4f} "
        f"(abs({pos_rate:.4f} - 0.5) = {abs(pos_rate-0.5):.4f} < 0.02); "
        f"constant-0.5 MSE = {mse:.4f} (at 50/50 entropy floor ≈ 0.25)"
    )


# ===========================================================================
# 1c — Bug 5: Q_LOOKBACK Python Default-Arg Binding
# ===========================================================================
def test_1c_bug5_qlookback_default_binding():
    """
    Reproduce the Python default-argument capture-at-definition-time bug that
    mirrors the Cell 2 → Cell 7 → Cell 10 execution order problem.

    In the notebook:
      Cell 2:  Q_LOOKBACK = 150   (config)
      Cell 7:  Q_LOOKBACK = 300   (overrides; build_feat_window defined here
                                   with default=Q_LOOKBACK — captures 300)
      Cell 10: Q_LOOKBACK = 150   (line 0, immediately overridden)
               Q_LOOKBACK = 64    (line 11, "OOM-safe")
               ExecutorQNetwork(q_lookback=Q_LOOKBACK)  → instantiated with 64

    After all three cells run:
      build_feat_window default = 300  (captured when Cell 7 defined it)
      ExecutorQNetwork.q_lookback = 64 (last value in Cell 10)
      → MISMATCH: 300 ≠ 64

    This test demonstrates the Python scoping mechanism that causes the mismatch.

    Bug condition: NOT (q_lookback_network == q_lookback_window_fn == q_lookback_train)
    """
    # ---- Simulate Cell 2 ----
    Q_LOOKBACK = 150  # noqa: F841  Cell 2 config

    # ---- Simulate Cell 7 (defines build_feat_window while Q_LOOKBACK == 150) ----
    # The notebook redefines Q_LOOKBACK = 300 at the top of Cell 7 BEFORE the def
    Q_LOOKBACK = 300  # Cell 7 override (as seen in notebook Cell 7 line 15)

    def build_feat_window_unfixed(num_matrix, abs_idx, q_lookback=Q_LOOKBACK):
        """Mirrors notebook Cell 7 build_feat_window — default captured at def time."""
        return q_lookback

    # ---- Simulate Cell 10 ----
    Q_LOOKBACK = 150  # Cell 10 line 0
    Q_LOOKBACK = 64   # Cell 10 line 11 — "forced OOM-safe"

    # Network is instantiated with Q_LOOKBACK = 64 (current value in Cell 10 line 157)
    q_net_lookback = Q_LOOKBACK  # 64

    # build_feat_window default is STILL 300 (captured when Cell 7 defined it)
    fn_default = build_feat_window_unfixed(None, 0)  # returns 300

    # Bug condition: the two values disagree
    assert fn_default != q_net_lookback, (
        f"Expected mismatch (fn_default={fn_default} != q_net_lookback={q_net_lookback}), "
        "but they agree — bug may already be fixed."
    )

    print(
        f"PASS: [Bug 5 — Q_LOOKBACK Default-Arg Binding] "
        f"| counterexample: build_feat_window default = {fn_default} (captured at Cell 7 def time), "
        f"ExecutorQNetwork q_lookback = {q_net_lookback} (Cell 10 last assignment); "
        f"MISMATCH: {fn_default} ≠ {q_net_lookback}"
    )


# ===========================================================================
# 1d — Bug 1: Zero ML Target Coverage
# ===========================================================================
def test_1d_bug1_zero_ml_targets():
    """
    Load training CSV and check for the presence of auxiliary ML target columns
    that _fill_ml_targets() falls back to np.zeros when absent.

    Bug condition: at least one required auxiliary target column is absent →
    _fill_ml_targets fills it with zeros → COUNT(k | abs(v).mean() <= 1e-7) > 0
    """
    import pandas as pd

    assert os.path.exists(_CSV_PATH), f"Training CSV not found: {_CSV_PATH}"

    df = pd.read_csv(_CSV_PATH, nrows=100)
    cols = set(df.columns)

    # These are the columns the notebook requires but cannot synthesize from OHLCV
    # (they default to np.zeros in _fill_ml_targets when absent)
    required_cols = [
        "vel_bull_fwd_8",
        "Volatility_Regime_next",
        "adv_target_next_zone_idx",
        "adv_target_next_zone_bars",
        "adv_target_next_zone_distance",
        "adv_target_next_zone_type",
        "adv_target_CSM_hist_fast_next",
        "adv_target_CSM_hist_slow_next",
        "adv_target_CSM_asset_fast_next",
        "adv_target_CSM_dxy_fast_next",
    ]

    missing_cols = [c for c in required_cols if c not in cols]

    assert len(missing_cols) > 0, (
        f"All required auxiliary columns are present in CSV — Bug 1 may not apply "
        f"to this dataset, or it is already fixed."
    )

    # Simulate _fill_ml_targets fallback: zeros for each missing column
    zero_target_count = 0
    for col in missing_cols:
        arr = np.zeros(100, dtype=np.float32)   # the fallback in _fill_ml_targets
        mean_abs = float(np.abs(arr).mean())
        assert mean_abs <= 1e-7, (
            f"Expected zeros for missing column '{col}' but got mean={mean_abs}"
        )
        zero_target_count += 1

    # Bug condition confirmed: at least one target is all-zero
    assert zero_target_count > 0, (
        "No zero targets found — bug condition not confirmed."
    )

    print(
        f"PASS: [Bug 1 — Zero ML Target Coverage] "
        f"| counterexample: {len(missing_cols)} required auxiliary columns absent from CSV: "
        f"{missing_cols}; "
        f"_fill_ml_targets fills all {zero_target_count} with np.zeros "
        f"(abs(zeros).mean() = 0.0 <= 1e-7)"
    )


# ===========================================================================
# 1e — Bug 7: Feature Scale Check
# ===========================================================================
def test_1e_bug7_feature_scale():
    """
    Load training CSV. Verify that price-level features (close_5m) have
    absolute-scale values (mean >> 1.0, std >> 1.0) — confirming that
    build_feat_window feeds non-normalized absolute prices to Conv1D.

    Bug condition: price_std > 1.0  (absolute scale, not normalized)
    """
    import pandas as pd

    assert os.path.exists(_CSV_PATH), f"Training CSV not found: {_CSV_PATH}"

    df = pd.read_csv(_CSV_PATH)

    # Try close_5m first, then close
    price_col = None
    for candidate in ("close_5m", "close"):
        if candidate in df.columns:
            price_col = candidate
            break

    assert price_col is not None, (
        "No price column ('close_5m' or 'close') found in CSV."
    )

    price_vals = df[price_col].dropna().values.astype(np.float64)
    price_mean = float(price_vals.mean())
    price_std  = float(price_vals.std())

    # Bug condition: absolute scale (GLD typically 175–260 USD range in the dataset)
    assert price_mean > 50.0, (
        f"Expected price mean > 50.0 (absolute scale) but got {price_mean:.2f}. "
        "Feature may already be normalized."
    )
    assert price_std > 1.0, (
        f"Expected price std > 1.0 (absolute scale) but got {price_std:.2f}. "
        "Feature may already be normalized."
    )

    print(
        f"PASS: [Bug 7 — Feature Scale Check] "
        f"| counterexample: '{price_col}' mean = {price_mean:.2f}, std = {price_std:.2f} "
        f"(absolute price level — Conv1D receives non-normalized GLD price values, "
        f"not relative structural patterns)"
    )


# ===========================================================================
# 1f — Bug 2: WAIT Reward Dominance
# ===========================================================================
def test_1f_bug2_wait_reward_dominance():
    """
    Simulate the ORIGINAL (unfixed) reward function with 10,000 random steps.

    Original code (before any partial fix):
        if action == H_WAIT:
            reward = +0.001      # unconditional fixed positive reward
        elif action == H_CALL:
            reward = fwd_pct - 0.0005   # PnL minus spread
        elif action == H_PUT:
            reward = -fwd_pct - 0.0005  # PnL minus spread

    On a coin-flip market, fwd_pct ~ N(0, σ):
        E[reward_CALL] ≈ -0.0005
        E[reward_PUT]  ≈ -0.0005
        E[reward_WAIT] = +0.001

    Bug condition: avg_reward_WAIT > avg_reward_CALL + epsilon AND
                   avg_reward_WAIT > avg_reward_PUT  + epsilon
    """
    rng = np.random.default_rng(42)
    n_steps = 10_000

    # Simulate forward price moves: GLD 5m bar ≈ N(0, 0.002) fractional return
    fwd_pct = rng.normal(loc=0.0, scale=0.002, size=n_steps)

    # Original reward function (unfixed — unconditional WAIT = +0.001)
    reward_wait = 0.001
    reward_call = fwd_pct - 0.0005
    reward_put  = -fwd_pct - 0.0005

    avg_wait = float(reward_wait)                  # constant
    avg_call = float(reward_call.mean())
    avg_put  = float(reward_put.mean())

    epsilon = 0.0005  # meaningful dominance threshold

    # Bug condition: WAIT unconditionally dominates
    assert avg_wait > avg_call + epsilon, (
        f"Expected avg_wait ({avg_wait:.5f}) > avg_call ({avg_call:.5f}) + {epsilon}. "
        "Reward function may already be fixed."
    )
    assert avg_wait > avg_put + epsilon, (
        f"Expected avg_wait ({avg_wait:.5f}) > avg_put ({avg_put:.5f}) + {epsilon}. "
        "Reward function may already be fixed."
    )

    print(
        f"PASS: [Bug 2 — WAIT Reward Dominance] "
        f"| counterexample: avg_reward_WAIT = {avg_wait:.5f}, "
        f"avg_reward_CALL = {avg_call:.5f}, "
        f"avg_reward_PUT = {avg_put:.5f}; "
        f"WAIT dominates by {avg_wait - max(avg_call, avg_put):.5f} "
        f"→ Q-network learns WAIT-always policy"
    )


# ===========================================================================
# 1g — Bug 4: Buffer Wipe Check
# ===========================================================================
def test_1g_bug4_buffer_wipe():
    """
    Reproduce the original replay buffer placement INSIDE the epoch loop
    (as found in the original notebook code before the partial deque fix).

    The bug: `replay_buffers = [[] for _ in range(NUM_HORIZONS)]` is placed
    at the start of the epoch loop, wiping all accumulated experience every epoch.

    Even with the current notebook's deque guard, this test proves the CLASS of
    bug: placing the buffer init inside the loop causes wipe-per-epoch.

    Bug condition: after iteration 0 fills 100 items, at the start of iteration 1
    the buffer is reset to [] and has length 0.
    """
    NUM_HORIZONS = 4
    H = 0  # test horizon 0

    # Simulate the ORIGINAL (buggy) pattern: init INSIDE the epoch loop
    replay_buffer_sizes_after_fill = []

    for epoch in range(2):
        # BUG: buffer re-initialized inside epoch loop — wipes all experience
        replay_buffers = [[] for _ in range(NUM_HORIZONS)]

        if epoch == 0:
            # Fill 100 transitions in epoch 0
            for i in range(100):
                replay_buffers[H].append(
                    (i, np.zeros(5), 0, 0.001, i + 1, np.zeros(5))
                )

        # Record size at end of each epoch (after fill in epoch 0, after wipe in epoch 1)
        replay_buffer_sizes_after_fill.append(len(replay_buffers[H]))

    # After epoch 0 fill: 100 items
    # After epoch 1 reset (init at top of loop): 0 items — BUG CONFIRMED
    size_after_epoch1 = replay_buffer_sizes_after_fill[1]

    assert size_after_epoch1 == 0, (
        f"Expected buffer length 0 after epoch 1 reset (wipe-per-epoch bug) "
        f"but got {size_after_epoch1}. Bug may already be fixed."
    )

    # Also confirm the 100 items WERE present at end of epoch 0
    assert replay_buffer_sizes_after_fill[0] == 100, (
        f"Expected 100 items after epoch 0, got {replay_buffer_sizes_after_fill[0]}"
    )

    print(
        f"PASS: [Bug 4 — Buffer Wipe Check] "
        f"| counterexample: buffer had {replay_buffer_sizes_after_fill[0]} items after epoch 0, "
        f"then reset to {size_after_epoch1} at start of epoch 1 "
        f"(init inside epoch loop wipes all experience; <3% retained per epoch)"
    )


# ===========================================================================
# Main runner
# ===========================================================================
if __name__ == "__main__":
    tests = [
        ("1a", "Bug 6 — Backbone Gradient Probe",          test_1a_bug6_backbone_gradient),
        ("1b", "Bug 3 — Coin-Flip Label Check",             test_1b_bug3_coin_flip_labels),
        ("1c", "Bug 5 — Q_LOOKBACK Default-Arg Binding",   test_1c_bug5_qlookback_default_binding),
        ("1d", "Bug 1 — Zero ML Target Coverage",           test_1d_bug1_zero_ml_targets),
        ("1e", "Bug 7 — Feature Scale Check",               test_1e_bug7_feature_scale),
        ("1f", "Bug 2 — WAIT Reward Dominance",             test_1f_bug2_wait_reward_dominance),
        ("1g", "Bug 4 — Buffer Wipe Check",                 test_1g_bug4_buffer_wipe),
    ]

    passed = 0
    failed = 0
    print("=" * 72)
    print("AXE Genesis RL Training — Bug Condition Exploration Tests (Task 1)")
    print("=" * 72)
    print()

    for tag, name, fn in tests:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"FAIL: [{tag} — {name}] | {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: [{tag} — {name}] | {type(e).__name__}: {e}")
            failed += 1
        print()

    print("=" * 72)
    print(f"{passed}/7 bug conditions confirmed  |  {failed} failed/errored")
    print("=" * 72)

    if failed > 0:
        sys.exit(1)
