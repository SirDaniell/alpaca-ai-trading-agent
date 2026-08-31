"""
Autonomous agent loop for Options Trading.
Runs on a schedule (cron / APScheduler / CLI).

Architecture:
1. Pull market data for Equity/ETF underlyings (SPY, QQQ, AAPL, etc.).
2. Compute HTF Zone Snapshots & Tier 1 Meta-Learner directional bias.
3. Compute HardActionMask (no-chase zone proximity & volume delta confirmation).
4. Invoke Tier 2 Q-Learner Executor (q_executor) to determine action (BUY_CALL, BUY_PUT, EXIT, WAIT).
5. Enforce non-bypassable code-level Hard Risk Gates (daily loss cap, position exposure cap).
6. Format OCC option contract and execute order via Alpaca Trading API / MCP CLI.
7. Log decision, transition, and execution result.
"""

import os
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from app.utils.alpaca_client import AlpacaClient
from app.utils.alpaca_cli_wrapper import AlpacaCLIWrapper
from app.core.market.mtf_rsi import calculate_wilder_rsi
from app.core.market.divergence_scale import build_unified_divergence_scale
from app.core.market.signal_events import build_signal_bundle
from app.core.analysis.support_resistance import detect_snr_levels_sequential, create_clustered_zones_sequential
from app.core.market.zone_snapshot import ZoneSnapshotManager, HardActionMask
from app.core.ml.signal_meta_learner import OnlineSignalMetaLearner
from app.core.options.q_executor import OptionsQExecutor, HTFBiasPackage, AccountContext, ExecutionContext, ACTION_BUY_CALL, ACTION_BUY_PUT, ACTION_TAKE_PROFIT_HALF, ACTION_CLOSE_FLATTEN, ACTION_WAIT
from app.core.options.options_order import select_target_option_contract
from app.core.options.pipeline_options import OptionsPipelineConfig
from app.db.connection import SessionLocal
from app.db.models import SignalOutcome

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

pipeline_config = OptionsPipelineConfig()
SYMBOLS = pipeline_config.symbols
TIMEFRAME = os.getenv("MARKET_TIMEFRAME", "5min")
LOOKBACK = int(os.getenv("MARKET_LOOKBACK_BARS", "500"))

# Global singleton instances
zone_manager = ZoneSnapshotManager(max_snapshots=20)
q_executor = OptionsQExecutor()
meta_learner = OnlineSignalMetaLearner()


def fetch_market_data(client: AlpacaClient, symbol: str) -> list:
    """Fetch OHLCV bars from Alpaca."""
    bars = client.get_bars(symbol, timeframe=TIMEFRAME, limit=LOOKBACK)
    if not bars:
        logger.warning(f"No bars fetched for {symbol}")
        return []

    candles = []
    for bar in bars:
        candles.append({
            "time": int(datetime.fromisoformat(bar["t"].replace("Z", "+00:00")).timestamp()),
            "open": float(bar["o"]),
            "high": float(bar["h"]),
            "low": float(bar["l"]),
            "close": float(bar["c"]),
            "volume": int(bar["v"]),
        })

    return sorted(candles, key=lambda x: x["time"])


def get_current_account_context(client: AlpacaClient) -> AccountContext:
    """Fetch live account health from Alpaca for risk gating."""
    acc = client.get_account()
    if not acc:
        return AccountContext()

    equity = float(acc.get("equity", 100000.0))
    last_equity = float(acc.get("last_equity", equity))
    drawdown_pct = max(0.0, (last_equity - equity) / (last_equity + 1e-6))

    positions = client.get_positions()
    open_pos_type = None
    pos_pnl_pct = 0.0

    if positions:
        first_pos = positions[0]
        sym = first_pos.get("symbol", "")
        if "C00" in sym or "CALL" in sym:
            open_pos_type = "CALL"
        elif "P00" in sym or "PUT" in sym:
            open_pos_type = "PUT"
        pos_pnl_pct = float(first_pos.get("unrealized_plpc", 0.0))

    return AccountContext(
        equity=equity,
        daily_pnl=equity - last_equity,
        daily_drawdown_pct=drawdown_pct,
        open_position_type=open_pos_type,
        open_position_pnl_pct=pos_pnl_pct,
    )


def check_hard_risk_gates(account: AccountContext, positions_count: int) -> bool:
    """
    Non-bypassable risk bounds check:
    1. Daily drawdown cap (3%).
    2. Max concurrent open positions cap.
    """
    if account.daily_drawdown_pct >= pipeline_config.max_daily_drawdown_pct:
        logger.warning("[RiskGate] Max daily drawdown limit (%.2f%%) reached! Trading suspended.", pipeline_config.max_daily_drawdown_pct * 100)
        return False

    if positions_count >= pipeline_config.max_concurrent_option_positions:
        logger.warning("[RiskGate] Max concurrent position cap (%d) reached!", pipeline_config.max_concurrent_option_positions)
        return False

    return True


def run_cycle():
    """Execute one complete trading loop cycle: fetch data, run Tier 1/2 models, check risk gates, execute orders via Alpaca CLI."""
    if not SYMBOLS:
        logger.warning("No symbols configured")
        return

    client = AlpacaClient()
    cli_wrapper = AlpacaCLIWrapper()
    db = SessionLocal()


    # Check if market is open
    if not client.is_market_open():
        logger.info("Market is closed, skipping cycle")
        db.close()
        return

    account_ctx = get_current_account_context(client)
    logger.info(f"Account connected. Portfolio Equity: ${account_ctx.equity:.2f}, Daily Drawdown: {account_ctx.daily_drawdown_pct:.2%}")

    positions = client.get_positions()
    positions_count = len(positions)

    # Process each symbol
    for symbol in SYMBOLS:
        try:
            logger.info(f"Processing {symbol}...")
            candles = fetch_market_data(client, symbol)
            if not candles or len(candles) < 30:
                continue

            import pandas as pd
            df_candles = pd.DataFrame(candles)
            df_candles.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)

            # 1. Update HTF SNR Zone Snapshots
            levels = detect_snr_levels_sequential(df_candles, up_to_index=len(df_candles)-1, lookback_period=50)
            zones = create_clustered_zones_sequential(levels, df_candles)
            zone_manager.add_snapshot(f"{symbol}_{datetime.now(timezone.utc).strftime('%H%M')}", "5m", zones)

            # Update invalidations on current closed candle
            current_close = float(df_candles["Close"].iloc[-1])
            current_high = float(df_candles["High"].iloc[-1])
            current_low = float(df_candles["Low"].iloc[-1])
            zone_manager.update_invalidation(current_close, current_high, current_low)

            # 2. Derive Tier 1 HTF Bias Package using OnlineSignalMetaLearner (PyTorch Multi-Head)
            direction = "bullish" if current_close > float(df_candles["Close"].iloc[-10]) else "bearish"
            feat_dict = {
                "close": current_close,
                "high": current_high,
                "low": current_low,
                "volume": float(df_candles["Volume"].iloc[-1]),
                "change": float(current_close - df_candles["Close"].iloc[-2]),
            }
            meta_pred = meta_learner.predict(feat_dict, signal_type="zone-breakout", direction=direction)

            htf_bias = HTFBiasPackage(
                direction=direction,
                strength=meta_pred["signal_strength"],
                reversal_prob=meta_pred["reversal_prob"],
                q_value=meta_pred["confidence"],
                expected_mfe_pips=meta_pred["expected_mfe_pips"],
                expected_mae_pips=meta_pred["expected_mae_pips"],
            )

            # 3. Time & Session Context
            now_utc = datetime.now(timezone.utc)
            hour_float = now_utc.hour + now_utc.minute / 60.0
            session_phase = "nyse_open" if 13.5 <= hour_float <= 14.5 else ("nyse_power_hour" if 19.0 <= hour_float <= 20.0 else "nyse_midday")

            exec_ctx = ExecutionContext(
                symbol=symbol,
                ltf_timeframe=pipeline_config.active_ltf_timeframe,
                current_price=current_close,
                atr=float(df_candles["High"].iloc[-14:].max() - df_candles["Low"].iloc[-14:].min()) / 2.0,
                buy_volume=float(df_candles["Volume"].iloc[-3:].mean() * 0.6),
                sell_volume=float(df_candles["Volume"].iloc[-3:].mean() * 0.4),
                hour_of_day=hour_float,
                day_of_week=now_utc.weekday(),
                session_phase=session_phase,
            )

            # 4. Compute HardActionMask (no-chase zone proximity & volume delta confirmation)
            has_pos = account_ctx.open_position_type is not None
            action_mask = q_executor.action_mask_engine.get_action_mask(
                current_price=current_close,
                atr=exec_ctx.atr,
                zone_manager=zone_manager,
                buy_volume=exec_ctx.buy_volume,
                sell_volume=exec_ctx.sell_volume,
                has_open_position=has_pos,
            )

            # 5. Invoke Tier 2 Q-Learner Executor
            state_vec = q_executor.build_state_vector(htf_bias, account_ctx, exec_ctx, zone_manager)
            chosen_action = q_executor.select_action(state_vec, action_mask, eval_mode=True)

            logger.info(f"Symbol {symbol} -> Action Mask: {action_mask}, Chosen Action: {chosen_action}")

            # 6. Check Non-Bypassable Hard Risk Gates
            risk_ok = check_hard_risk_gates(account_ctx, positions_count)

            # 7. Execute Options Order via Alpaca CLI Wrapper if action is BUY_CALL or BUY_PUT
            if risk_ok and chosen_action in (ACTION_BUY_CALL, ACTION_BUY_PUT):
                opt_type = "call" if chosen_action == ACTION_BUY_CALL else "put"
                target_contract = select_target_option_contract(symbol, current_close, opt_type)
                occ_ticker = target_contract["occ_symbol"]

                logger.info(f"🚀 EXECUTING OPTION ORDER VIA CLI: {occ_ticker} (Type: {opt_type.upper()}, Strike: ${target_contract['strike_price']})")
                order_res = cli_wrapper.submit_order(symbol=occ_ticker, side="buy", qty=1, is_option=True)
                logger.info(f"CLI Order Response: {order_res}")

            elif chosen_action in (ACTION_TAKE_PROFIT_HALF, ACTION_CLOSE_FLATTEN) and positions:
                first_pos = positions[0]
                pos_sym = first_pos.get("symbol")
                pos_qty = int(first_pos.get("qty", 1))
                close_qty = max(1, pos_qty // 2) if chosen_action == ACTION_TAKE_PROFIT_HALF else pos_qty

                logger.info(f"🛑 CLOSING OPTION POSITION VIA CLI: {pos_sym} Qty: {close_qty}")
                order_res = cli_wrapper.submit_order(symbol=pos_sym, side="sell", qty=close_qty, is_option=True)
                logger.info(f"CLI Close Response: {order_res}")


            # 8. Log outcome to DB
            outcome = SignalOutcome(
                signal_id=f"{symbol}_{datetime.now(timezone.utc).isoformat()}",
                symbol=symbol,
                source_timeframe=pipeline_config.active_ltf_timeframe,
                direction=htf_bias.direction.upper(),
                signal_type="OPTIONS_Q_EXECUTOR",
                entry_time=datetime.now(timezone.utc),
                entry_price=current_close,
                feature_contract_version="v2.0",
                feature_names=["strength", "reversal_prob", "action_mask", "chosen_action"],
                feature_values={
                    "strength": htf_bias.strength,
                    "reversal_prob": htf_bias.reversal_prob,
                    "action_mask": action_mask.tolist(),
                    "chosen_action": chosen_action,
                },
            )
            db.add(outcome)
            db.commit()

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            db.rollback()

    db.close()
    logger.info("Cycle complete")


if __name__ == "__main__":
    run_cycle()
