# Directives: Repurposing the Signal Stack for Options (Keeping FX Intact)

## Core principle

**Don't touch the FX pipeline. Extract, don't rewrite.**

The 200+ indicator engine, MTF RSI, SNR zone logic, and the meta-learner network
architecture are not inherently FX — they operate on OHLCV candles, which any
instrument produces. The only FX-specific piece is the **DXY basket construction**
(EURUSD/USDJPY/GBPUSD/etc.), which stays exactly where it is, untouched, for
whatever other project needs it.

**Directive 1 — Extract the shared signal engine into an instrument-agnostic core.**
- Move/confirm these live in a shared location, taking only `candles` (OHLCV) as
  input, with no FX-specific assumptions baked in:
  - `calculate_ti_features` (200+ indicators)
  - `calculate_mtf_rsi` / `calculate_wilder_rsi`
  - `build_unified_divergence_scale`
  - `build_signal_bundle`
  - SNR zone detection/confluence scoring
  - `SignalMetaNetwork` (the 6-head DQN architecture itself — the *architecture*
    is reusable even though the current trained weights are FX-tuned)
- The **DXY basket module stays as an optional plug-in feature**, not a
  dependency. FX pipeline keeps using it. Options pipeline ignores it (or later,
  optionally feeds it in as a macro-context feature — e.g., dollar strength vs.
  SPY — but that's a nice-to-have, not now).

**Directive 2 — Two independent pipeline configs, one shared engine.**
- `pipeline_fx.py` (existing, unchanged) → forex pairs, DXY basket, current
  trained weights, current targets in pips.
- `pipeline_options.py` (new) → equity/ETF underlyings (whatever Alpaca options
  chain covers — e.g. SPY, QQQ, AAPL, etc.), retrained/fine-tuned meta-learner
  weights, targets reframed appropriately (see Directive 4).

This means zero risk of breaking the FX system while building the options one —
they share code, not state or weights.

---

## Architecture: two-tier decision system

This matches what you described — separate the **"what's the bias/opportunity"**
question from the **"how do we actually trade it"** question.

### Tier 1 — Meta-Learner (repurposed, higher timeframe bias)
- Runs on **HTF** (suggest H1/H4, D1 for the macro read) against the equity/ETF
  underlying's own price action — not DXY, since Alpaca doesn't trade FX pairs.
- Outputs (already exist in the model, just reinterpreted):
  - `strength_head` → conviction score for a directional bias
  - `q_head` → bull/bear/wait/hedge valuation
  - `reversal_head` → reversal probability — this is where your SNR-zone edge
    lives, since HTF SNR zones are exactly where reversals cluster
  - `risk_head` → MFE/MAE-style expected move envelope (needs reinterpreting
    for options — see Directive 4)
- Job: produce a **filtered directional bias package** — not a trade, not a
  size, just "here's what the market is likely to do and how confident we are."
  This is the "naive signals, already accurate, need filtering" layer you
  mentioned — the filtering *is* the strength/reversal thresholding that
  already exists (≥0.65 conviction, reversal suppression above 0.65).

### Tier 2 — Q-Learner Trade Executor (new, lower timeframe execution)
- Runs on **LTF** (suggest 5–15min) — this is the new component, separate model
  from the meta-learner.
- **Input (state):**
  - The Tier 1 bias package (direction, strength, reversal probability, expected
    move envelope)
  - Current account state (equity, open positions, today's realized P&L,
    current drawdown)
  - Recent trade outcome history (win/loss streak — informs risk scaling, not
    revenge sizing)
  - LTF price action / short-term SNR proximity (fine-tune entry timing within
    the HTF bias window)
- **Action space (this is the "money management + when to take a trade" layer):**
  - Trade / no-trade decision (abstaining is a valid, rewarded action)
  - Position size (as % of account risk, not fixed contracts)
  - Options structure selection (e.g., long call/put vs. defined-risk spread —
    start with one or two structures, expand later)
  - Exit/adjustment trigger (take profit, stop, or roll)
  - **Re-entry while the HTF bias persists** — see "Bias-Persistence Re-Entry"
    below. This is the "milk 1–2 bars while the signal holds" behavior, and
    it's the correct place for that logic — not a general size-scaling rule.
- **Reward function (needs explicit design, don't leave implicit):**
  - Primary: realized P&L per trade, risk-adjusted (e.g., P&L / max drawdown
    exposed on that trade)
  - Penalty: overtrading (churn cost — options have wider spreads than equities,
    so cost-per-trade matters more)
  - Penalty: exceeding a hard daily loss cap or max concurrent exposure — this
    should be a **hard gate outside the learned policy**, not something the
    Q-learner can override. Never let a learned policy be the only thing
    standing between the account and a blowup.

---

## Bias-Persistence Re-Entry (not martingale — important distinction)

This is the actual mechanic you described: if H1 shows a confirmed upward
bias, the executor should be free to take **more than one LTF trade within
that same H1 window** — closing a position (win, loss, or scratch) doesn't
mean the opportunity is over if the underlying thesis is still intact.

**How this differs from martingale (and why that distinction matters for the
write-up's risk gates):**
- Each re-entry uses the **same fixed risk sizing** as the first entry in that
  window — never scaled up because a prior attempt in the window lost.
  Martingale scales size *up* after a loss to recover it; this doesn't.
- Re-entry is **gated by the bias still being valid**, not by a loss having
  occurred. A win doesn't block a re-entry, and a loss doesn't trigger one —
  the only trigger is "is the H1 thesis still confirmed."

**Re-entry gate — what "still holding" should mean concretely:**
- `strength_head` conviction score still above the entry threshold (≥0.65 or
  whatever's tuned)
- `reversal_head` probability still below the suppression threshold — this is
  the direct signal that the bias hasn't started flipping
- Price hasn't broken the nearest opposing HTF SNR zone (i.e., the structural
  level that would invalidate the thesis)
- Optionally: a cap on max re-entries per H1 window, so the executor can't loop
  indefinitely on a single thesis even if all signals stay green — this is a
  simple, explicit ceiling worth having regardless of what the model learns.

**Bias invalidation — Q-learner's responsibility:**
- The moment any of the above gate conditions fail, the executor should stop
  re-entering and close/flatten any open position from that window rather than
  waiting for a stop-loss to be hit passively.
- This needs the executor polling Tier 1 outputs continuously (each LTF
  candle), not just once at the start of the H1 window — the bias can break
  mid-window and the executor needs to see that in near-real-time.

---

## Why LTF execution makes sense here

You're right that options P&L isn't about "pip growth" the way FX is — it's
about catching the right window of a directional move (plus theta/IV effects
if you use spreads). Running the executor on a shorter timeframe gives it more
decision points to time entries within the HTF bias window the meta-learner
already flagged, rather than waiting for one big multi-bar move. Directive:
keep the **bias on HTF** (this is where your SNR-zone reversal edge is
genuinely strong) and let the **executor's frequency be LTF**, but don't let
"more executions" become the reward target itself — frequency is a side effect
of good timing, not a goal.

---

## Concrete next steps for the local agent

1. Refactor: move/confirm shared signal engine modules are instrument-agnostic
   (no FX-only assumptions). Leave `app/core/market/` FX-specific pieces (DXY
   basket) untouched.
2. Create `app/core/options/` — new module for:
   - Options chain retrieval (strikes, expiries, greeks) — currently missing
     entirely from `alpaca_client.py`
   - Options order construction (OCC symbol format, multi-leg orders,
     `asset_class="option"`)
3. Fix the existing auth bug in `alpaca_client.py` — `APCA-API-SECRET-KEY`
   header is never set, only the key ID is. Every authenticated call currently
   fails.
4. Wire Alpaca access through **MCP server or CLI** (competition hard
   requirement) instead of raw `requests` calls — this replaces
   `alpaca_client.py`'s direct REST approach.
5. Create `app/agent/q_executor.py` — new Q-learner, separate from the existing
   meta-learner, implementing the state/action/reward design above.
6. Update `agent/loop.py` so it actually **calls the executor and places
   trades** — right now it only logs signals and never executes anything.
7. Retrain/fine-tune the meta-learner on equity/ETF underlyings instead of FX
   pairs (or run inference with the existing architecture but new weights —
   don't reuse FX-trained weights directly on stocks, the distributions differ).
8. Define the hard risk gates (daily loss cap, max position size, max
   concurrent option positions) as **code-level constraints outside the
   learned policy**, not something either model can bypass. These become your
   one-page write-up's "risk gates" section.

---

## Zone-Anchored Entry Design (no-chase rule, zone repainting, volume delta)

This is the core execution discipline for the Q-learner: **entries only happen
at or very near a valid zone, never chased once price has moved away from it.**
The research above supports this directly — zone-proximity as an explicit
input measurably helps, and it should be enforced as a hard rule, not just
something the model is hoped to learn on its own.

### No-chase entry rule — enforce as a hard filter, not a learned preference

Don't leave "should I chase this?" as something the Q-learner has to figure
out through trial and error — bake it in as an **action mask** applied before
the model's chosen action is allowed to execute:

- Define a proximity band around each zone (e.g., zone edge ± some ATR-scaled
  buffer, tuned per instrument/timeframe).
- **Sell setups**: entry only valid at, very near, or above a resistance zone.
- **Buy setups**: entry only valid at, very near, or below a support zone.
- If price is outside the proximity band, the "enter" action is masked off
  entirely — the Q-learner can still choose "wait," but "enter" isn't even on
  the table. This mirrors the straddle-option paper's approach of feeding
  zone-proximity as a first-class signal, but goes a step further by making it
  a hard constraint rather than a soft one, so a bad training run can't
  accidentally learn to chase.
- This also directly protects against the exact failure mode described in the
  proplynq zone-trading piece found during research: stops/entries placed too
  tight against a zone edge get caught by the routine stop-hunt/liquidity-grab
  price action that happens right at obvious levels. The proximity band should
  have real width, not sit exactly on the line.

### Zone snapshot & repaint handling

Zones are computed from a lookback window (swing-point detection with
reversal/breakout thresholds), which means by construction they can shift as
new swing points confirm later in the session. Directive:

1. **Snapshot, don't overwrite.** Every time zones are recalculated (e.g., on
   each new HTF bar close), write a new **timestamped, versioned zone set**
   rather than replacing the previous one in place. Keep a rolling history
   (e.g., last N sessions).
2. **The Q-learner reads from the union of recent valid snapshots**, not just
   the latest one. A zone from this morning's snapshot that hasn't been
   invalidated is still legitimate structure even if the newest recalculation
   moved or removed it — this matches your own observation that repainted
   zones often keep working after the repaint.
3. **Confluence weighting**: if the same price area shows up across multiple
   snapshots (today's and a recalculated version), treat that as a
   stronger/higher-confidence zone rather than just picking one.
4. **Explicit invalidation rule** — a zone should drop out of the active set
   only on a *confirmed* break (a closed candle beyond it, not just a wick),
   so the system isn't carrying dead levels forever, but also isn't discarding
   a zone just because the recalculation redrew it differently.

### Volume delta integration (buyer vs. seller volume)

Feed both buy volume and sell volume — not just net delta, but both raw
series — into two places:

- **Tier 1 (meta-learner)**: as an additional feature alongside the existing
  200+ indicators, since order-flow imbalance has real, literature-supported
  predictive value for near-term direction (see research above).
- **Tier 2 (Q-learner), as an entry confirmation gate at the zone**: don't
  just allow entry because price is near a zone — require the volume delta to
  actually confirm it. E.g., for a long at support: require selling volume to
  be drying up or buying volume delta to flip positive *at the zone*, rather
  than entering on proximity alone. This operationalizes "wait for the
  reaction, not just the touch" — the zone gets you close, volume confirms the
  rejection is real rather than the zone about to fail.

### Reward design note (informed by the straddle-option paper's finding)

Avoid a reward function that fires on every small P&L fluctuation while a
position is open — the research above showed this specifically destabilizes
DQN training (two baseline models failed to learn anything useful because of
it). Prefer a **delayed reward** structure: no reward while holding (unless a
hard stop is breached), reward realized on close, with a defined stop-loss
threshold that itself gets a distinct reward signal when hit correctly (i.e.,
the model should be reinforced for *correctly stopping out*, not just
punished for the loss). This is the same fix that made their model outperform
naive-reward baselines by a wide margin.

### Missed-opportunity penalty (hindsight bonus) — reward the setups it should have taken

This adds the piece you asked for: penalize the Q-learner for choosing "wait"
on a setup that the meta-learner correctly flagged and that then moved in the
recommended direction. Done naively, this risks teaching the agent to force
trades just to dodge the penalty — undoing the zone-discipline built above —
so it needs guardrails:

1. **Only applies when every hard entry gate was already satisfied.** If the
   no-chase mask, zone proximity, and volume confirmation gate all passed and
   the agent still chose "wait," *that's* a real missed opportunity worth
   penalizing. If any gate failed (price too far from zone, volume didn't
   confirm), never penalize — that's correct discipline, not negligence.
2. **Never applies when a hard risk gate blocked the trade** (daily loss cap
   hit, max concurrent positions reached, max re-entries per window already
   used). Respecting a risk limit should never be treated as a mistake.
3. **Structure it as a hindsight/potential-based bonus, training-time only** —
   this follows the approach used in DeepScalper (a risk-aware RL trading
   framework): add a term equal to the forward price move over a fixed horizon
   *h* (matching the meta-learner's existing lookforward window), scaled by a
   small weight, applied only when the agent was flat and a valid setup played
   out. Crucially, this shaping term is used only to guide training gradients —
   the unshaped, real P&L reward is still what's used to evaluate actual
   performance, so the shaping can't quietly make a bad policy look good.
4. **Keep the penalty weight small relative to real trade outcomes.** A missed
   opportunity should sting less than an actual bad loss stings, and less than
   a real win rewards — otherwise the agent over-corrects toward compulsive
   entries on every borderline setup just to avoid the regret term.
5. **Threshold the counterfactual move.** Only apply the penalty if the
   forward move was clean/significant (past some minimum size), not on noisy,
   marginal price wiggles — otherwise the agent gets punished for sensible
   caution on genuinely low-conviction setups.



The straddle-option paper fused HTF/LTF context into **one network** using
attention (short-term sequence + longer-period context combined via a
channel-attention module) rather than two separate models talking to each
other. That's an alternative to the two-tier meta-learner → Q-learner split
already planned in this doc. Two separate models is more modular and easier
to debug/iterate on separately (recommended given the timeline), but worth
knowing a single fused architecture is a proven alternative if the two-model
handoff becomes awkward in practice.

---

## What stays exactly as-is (do not modify)

- FX pipeline, DXY basket construction, current trained FX meta-learner weights
- `docs/HANDOVER_META_LEARNER.md`, `docs/agent_handoff_signal_intelligence.md`
  (FX-specific handoff docs — keep for the other project)
