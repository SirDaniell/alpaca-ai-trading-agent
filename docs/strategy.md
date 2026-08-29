# Strategy Notes & Architecture

The autonomous options trading agent uses an **Online Reinforcement Meta-Learner** to quantify signal strength [0.0 - 1.0] and expected pip returns over a 24-bar lookforward window.

## Signal Intelligence Stack

1. **200+ Causal Technical Indicators**: Computed via `calculate_ti_features` across price action, candle microstructure, trend velocity, and volatility regimes.
2. **Basket-Constructed DXY Index**: Geometrically derived Dollar index (`EURUSD`, `USDJPY`, `GBPUSD`, `USDCAD`, `USDSEK`, `USDCHF`).
3. **Multi-Timeframe SNR & MTF RSI**: Support/Resistance zone distances, confluence scoring, and weighted MTF RSI (`H1`, `H4`, `D1`) with Wilder state resampling.
4. **48-Bar Sequential Decision Window**: Input tensor flattened from a `(48, F)` matrix into a `DECISION_WINDOW_DIM` vector for PyTorch Deep Q-Network evaluation.

## Multi-Head Model Architecture & Auxiliary Enriched Targets

The Meta-Learner PyTorch Network (`SignalMetaNetwork`) uses a shared backbone feeding 6 specialized heads:

| Head | Output | Target Origin (`ml_dataset_preparation.py`) | Function / Trading Policy |
|---|---|---|---|
| `q_head` | Q-values `[bull, bear, wait, hedge]` | RL TD-Target | Contextual Q-Learning action valuation |
| `strength_head` | Signal strength `[0.0 - 1.0]` | Sigmoid-mapped reward | High-conviction entry threshold ($\ge 0.65$) |
| `pips_head` | Net 24-bar pips | `adv_target_logret_24` | Expected net move magnitude |
| `risk_head` | MFE pips, MAE pips | `adv_target_MFE`, `adv_target_MAE` | Reward-to-risk ratio $(MFE/MAE)$ for dynamic SL/TP |
| `liquidity_head` | Next zone distance (ATR) & type | `adv_target_next_zone_distance`, `adv_target_next_zone_type` | Structural target price & estimated trade duration |
| `reversal_head` | Reversal probability `[0.0 - 1.0]` | `adv_target_reversal_prob` | Entry suppression when `reversal_prob > 0.65` |

## Option Strategy & Execution Policy (Planned Competition Loop)

- **Signal Scoring**: Live signals evaluate $Q$-values and `signal_strength` score ($\ge 0.65$ threshold for trade execution).
- **Risk Gate**: Trade execution requires predicted $MFE / \max(MAE, 1.0) \ge 2.0$.
- **Dynamic TP/SL**: Take Profit target set to `next_zone_distance` and Stop Loss anchored to predicted `MAE`.
- **Execution Interface**: Alpaca options trading interface via MCP tool / CLI API.
- **Durable Learning**: Fine-tunes model weights via Prioritized Experience Replay on confirmed 24-bar forward move outcomes (MFE, MAE, next zone, reversal probability, RL rewards).


