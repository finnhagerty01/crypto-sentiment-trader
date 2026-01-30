import time
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.append(str(Path(__file__).parent / 'src'))

from src.utils.config import TradingConfig
from src.data.reddit_client import RedditClient
from src.data.market_client import MarketClient
from src.data.archive import append_to_archive
from src.features.sentiment_advanced import EnhancedSentimentAnalyzer
from src.models.trading_model import ImprovedTradingModel
from src.execution.live import BinanceExecutor

# Risk Management
from src.risk.position_sizing import PositionSizer, RiskBudget
from src.risk.stop_loss import StopLossManager
from src.risk.exit_manager import ExitManager, ExitReason
from src.risk.portfolio import PortfolioRiskManager

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("trade_bot.log"), logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger("MainOrchestrator")

# --- CONFIGURATION ---
DRY_RUN = True  # <--- SAFETY ON: NO REAL MONEY
LOOP_INTERVAL = 3600

# PAPER TRADING STATE
STARTING_CAPITAL = 1000.0  # Starting capital in USDT

# --- RISK MANAGEMENT CONFIGURATION ---
RISK_CONFIG = {
    # Position Sizing
    'max_position_pct': 0.15,       # Max 15% of account per position
    'max_total_exposure': 0.50,     # Max 50% of account exposed
    'min_confidence': 0.55,         # Minimum confidence for any trade

    # Stop-Loss / Take-Profit
    'default_stop_pct': 0.03,       # 3% stop-loss
    'default_take_profit_pct': 0.06, # 6% take-profit (2:1 reward/risk)
    'atr_multiplier': 2.0,          # ATR multiplier for volatility-based stops

    # Exit Management
    'min_hold_hours': 2.0,          # Minimum hold before signal exit
    'max_hold_hours': 48.0,         # Maximum hold time (forced exit)

    # Daily Risk Limits
    'max_daily_loss': 0.05,         # Halt trading after 5% daily loss
    'max_drawdown': 0.15,           # Halt trading after 15% drawdown

    # Cooldown
    'cooldown_hours': 4,            # Wait after selling before re-buying
}


def main():
    logger.info(f"Starting Sleek Bot... DRY_RUN={DRY_RUN}")

    config = TradingConfig.from_yaml("configs/data.yaml")
    reddit_client = RedditClient(subreddits=config.subreddits)
    market_client = MarketClient(config)
    sentiment_analyzer = EnhancedSentimentAnalyzer(symbols=config.symbols)
    model = ImprovedTradingModel()
    executor = BinanceExecutor()

    # --- INITIALIZE PORTFOLIO & RISK MANAGEMENT ---
    portfolio = PortfolioRiskManager(
        initial_capital=STARTING_CAPITAL,
        max_positions=5,
        max_exposure=RISK_CONFIG['max_total_exposure'],
        max_single_position=RISK_CONFIG['max_position_pct'],
        max_correlated_exposure=0.30,
        max_sector_exposure=0.30,
        fee_per_side=config.fee_per_side,
        slippage_per_side=config.slippage_per_side
    )

    position_sizer = PositionSizer(
        account_value=STARTING_CAPITAL,
        max_position_pct=RISK_CONFIG['max_position_pct'],
        max_total_exposure=RISK_CONFIG['max_total_exposure']
    )

    stop_manager = StopLossManager(
        default_stop_pct=RISK_CONFIG['default_stop_pct'],
        default_take_profit_pct=RISK_CONFIG['default_take_profit_pct'],
        atr_multiplier=RISK_CONFIG['atr_multiplier']
    )

    exit_manager = ExitManager(
        min_hold_hours=RISK_CONFIG['min_hold_hours'],
        max_hold_hours=RISK_CONFIG['max_hold_hours'],
        stop_loss_pct=RISK_CONFIG['default_stop_pct'],
        take_profit_pct=RISK_CONFIG['default_take_profit_pct']
    )

    risk_budget = RiskBudget(
        account_value=STARTING_CAPITAL,
        max_daily_loss=RISK_CONFIG['max_daily_loss'],
        max_drawdown=RISK_CONFIG['max_drawdown']
    )

    # Track last daily reset
    last_daily_reset = datetime.now(timezone.utc).date()

    logger.info(f"Risk Management initialized: {RISK_CONFIG}")

    # --- 1. LOAD & MERGE DATA FIRST ---
    logger.info("--- LOADING DATASETS ---")
    
    # A. Load Existing Master Record
    data_path = Path("data/master_reddit.csv")
    if data_path.exists():
        logger.info("Loading master_reddit.csv from disk...")
        master_reddit = pd.read_csv(data_path)
        master_reddit['created_utc'] = pd.to_datetime(master_reddit['created_utc'])
    else:
        logger.info("No master record found. Starting fresh.")
        master_reddit = pd.DataFrame()

    # B. Fetch Recent History (Fill gaps)
    logger.info("Fetching recent Reddit history...")
    hist_reddit = reddit_client.fetch_historical(days=30)

    # Archive historical data (append-only, never deleted)
    if not hist_reddit.empty:
        append_to_archive(hist_reddit)

    # C. Merge
    if not master_reddit.empty:
        # Combine old + new
        master_reddit = pd.concat([master_reddit, hist_reddit]).drop_duplicates(subset=['id'])
        # Ensure all existing data is in archive
        append_to_archive(master_reddit)
    else:
        master_reddit = hist_reddit

    # D. Save merged version immediately (rolling window for live trading)
    master_reddit.to_csv(data_path, index=False)
    logger.info(f"Total Database Size: {len(master_reddit)} posts")

    # --- 2. INITIAL TRAINING (Using FULL Dataset) ---
    logger.info("--- INITIAL TRAINING ---")
    
    # Fetch matching market data for the full period
    # We ask for a bit more buffer (45 days) to ensure we cover the accumulated CSV
    hist_market = market_client.fetch_ohlcv(lookback_days=45)

    if master_reddit.empty or hist_market.empty:
        logger.critical("Insufficient data to train. Exiting.")
        return

    hist_sentiment = sentiment_analyzer.analyze(master_reddit)
    
    # Train
    train_df = model.prepare_features(hist_market, hist_sentiment, is_inference=False)
    if not train_df.empty:
        model.train(train_df)
    else:
        logger.warning("Feature prep returned empty DF (check timestamps). Skipping initial train.")

    last_train_time = datetime.now()
    RETRAIN_HOURS = 24

    # --- 3. LIVE LOOP ---
    while True:
        try:
            logger.info("--- STARTING LIVE CYCLE ---")
            current_time = datetime.now(timezone.utc)

            # --- DAILY RESET CHECK ---
            today = current_time.date()
            if today > last_daily_reset:
                logger.info("New day detected - resetting daily risk limits")
                risk_budget.reset_daily()
                last_daily_reset = today

            # --- CHECK IF TRADING IS HALTED ---
            if not risk_budget.can_trade():
                status = risk_budget.get_status()
                logger.warning(f"TRADING HALTED: {status['halt_reasons']}")
                logger.info(f"Sleeping {LOOP_INTERVAL}s (halted)...")
                time.sleep(LOOP_INTERVAL)
                continue

            # A. Fetch Live Data
            live_reddit = reddit_client.fetch_live(limit=500)
            if not live_reddit.empty:
                # Archive BEFORE applying rolling window (preserves all historical data)
                append_to_archive(live_reddit)

                master_reddit = pd.concat([master_reddit, live_reddit]).drop_duplicates(subset=['id'])
                # Keep last 30 days rolling for live trading efficiency
                cutoff = datetime.now(timezone.utc) - timedelta(days=30)
                master_reddit = master_reddit[master_reddit['created_utc'] > cutoff]

                master_reddit.to_csv(data_path, index=False)
                logger.info(f"Database updated: {len(master_reddit)} posts.")

            # Lookback increased to 5 days to support 48h lags
            live_market = market_client.fetch_ohlcv(lookback_days=5)

            # B. Analyze & Predict
            live_sentiment = sentiment_analyzer.analyze(master_reddit)
            pred_df = model.prepare_features(live_market, live_sentiment, is_inference=True)

            if pred_df.empty:
                logger.warning("No features generated. Waiting...")
                time.sleep(60)
                continue

            latest_ts = pred_df['timestamp'].max()

            # Sanity Check: Ensure we are predicting for the current hour
            current_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0).replace(tzinfo=None)
            logger.info(f"Latest Data Timestamp: {latest_ts} | Current UTC Hour: {current_hour}")

            latest_features = pred_df[pred_df['timestamp'] == latest_ts]
            logger.info(f"Predicting on {len(latest_features)} symbols (Time: {latest_ts})")

            signals = model.predict(latest_features)

            # C. Build price map for all relevant symbols
            symbols_to_price = set(signals.keys()) | set(portfolio.state.positions.keys())
            price_map = {}
            for sym in symbols_to_price:
                try:
                    price_map[sym] = executor.get_price(sym)
                except Exception:
                    pass

            # --- UPDATE ACCOUNT VALUE FOR POSITION SIZER ---
            current_equity = portfolio.equity(price_map)
            position_sizer.update_account_value(current_equity)
            risk_budget.update(current_equity)

            # --- CHECK STOP-LOSSES FIRST (Risk override - exits immediately) ---
            stop_actions = stop_manager.check_stops(price_map, pd.Timestamp(current_time))
            for symbol, action in stop_actions.items():
                if action and portfolio.has_position(symbol):
                    pos = portfolio.get_position(symbol)
                    price = price_map.get(symbol)

                    if DRY_RUN:
                        trade = portfolio.close_position(symbol, mid_price=price, reason=action, exit_time=pd.Timestamp(current_time))
                        if trade:
                            logger.warning(f"[PAPER] {action.upper()} EXIT {symbol} @ ${price:.4f} P&L=${trade['pnl']:.2f}")
                            risk_budget.record_trade(trade['pnl'])
                            stop_manager.remove_stop(symbol)
                    else:
                        order = executor.execute_order(symbol, "SELL", quantity=None)
                        if order:
                            trade = portfolio.close_position(symbol, mid_price=price, reason=action)
                            if trade:
                                risk_budget.record_trade(trade['pnl'])
                            stop_manager.remove_stop(symbol)

            # D. Process signals with risk management
            for symbol, signal in signals.items():
                action = signal['action']
                confidence = signal['confidence']
                price = price_map.get(symbol)
                if price is None:
                    continue

                # Check current holding state
                is_holding = portfolio.has_position(symbol)

                if action == "BUY":
                    # GATE 1: State Awareness - Don't double buy
                    if is_holding:
                        continue

                    # GATE 2: Cooldown Check (using PortfolioRiskManager)
                    in_cooldown, hours_remaining = portfolio.is_in_cooldown(
                        symbol,
                        RISK_CONFIG['cooldown_hours'],
                        pd.Timestamp(current_time)
                    )
                    if in_cooldown:
                        logger.info(f"COOLDOWN BLOCKED BUY {symbol}: {hours_remaining:.1f}h remaining")
                        continue

                    # GATE 3: Portfolio-level risk checks
                    can_open, block_reason = portfolio.can_open_position(symbol, 100, price_map)  # Quick check
                    if not can_open:
                        logger.info(f"PORTFOLIO BLOCKED {symbol}: {block_reason}")
                        continue

                    # GATE 4: Get volatility for position sizing (use ATR % if available)
                    symbol_features = latest_features[latest_features['symbol'] == symbol]
                    if not symbol_features.empty and 'atr_14_pct' in symbol_features.columns:
                        volatility = symbol_features['atr_14_pct'].iloc[0]
                        atr_value = symbol_features['atr_14'].iloc[0] if 'atr_14' in symbol_features.columns else None
                    else:
                        volatility = 0.02  # Default 2% if not available
                        atr_value = None

                    # GATE 5: Calculate position size using risk management
                    current_positions = {
                        sym: pos.value
                        for sym, pos in portfolio.state.positions.items()
                    }

                    position_calc = position_sizer.calculate_position(
                        symbol=symbol,
                        price=price,
                        confidence=confidence,
                        volatility=volatility,
                        current_positions=current_positions
                    )

                    if position_calc['size_usd'] <= 0:
                        logger.info(f"POSITION SIZING BLOCKED {symbol}: {position_calc.get('reason', 'size=0')}")
                        continue

                    notional_usd = position_calc['size_usd']
                    quantity = position_calc['quantity']

                    if DRY_RUN:
                        ok = portfolio.open_position(symbol, notional_usd=notional_usd, mid_price=price, entry_time=pd.Timestamp(current_time))
                        if ok:
                            logger.info(
                                f"[PAPER] BUY {symbol} ${notional_usd:.2f} ({position_calc['pct_of_account']:.1%} of acct) "
                                f"@ ${price:.4f} conf={confidence:.2f} sizing={position_calc['limiting_factor']}"
                            )
                            # Create stop-loss for the position
                            if atr_value and atr_value > 0:
                                stop_manager.create_atr_stop(symbol, price, quantity, atr_value)
                            else:
                                stop_manager.create_fixed_stop(symbol, price, quantity)
                        else:
                            logger.info(f"[PAPER] BUY skipped (insufficient cash) {symbol}")
                    else:
                        qty_rounded = round(quantity, 5)
                        order = executor.execute_order(symbol, "BUY", qty_rounded)
                        if order:
                            if atr_value and atr_value > 0:
                                stop_manager.create_atr_stop(symbol, price, qty_rounded, atr_value)
                            else:
                                stop_manager.create_fixed_stop(symbol, price, qty_rounded)

                elif action == "SELL":
                    # GATE 1: Can't sell what you don't have
                    if not is_holding:
                        continue

                    # GATE 2: Use ExitManager for hold time + risk override logic
                    pos = portfolio.get_position(symbol)
                    entry_price = pos.entry_price
                    entry_ts = pos.entry_time

                    exit_decision = exit_manager.should_exit(
                        symbol=symbol,
                        entry_price=entry_price,
                        current_price=price,
                        entry_time=entry_ts,
                        signal_says_sell=True,
                        current_time=pd.Timestamp(current_time)
                    )

                    if not exit_decision.can_exit:
                        logger.info(f"EXIT BLOCKED {symbol}: {exit_decision.details}")
                        continue

                    exit_reason = exit_decision.reason.value

                    # Execute sell
                    if DRY_RUN:
                        trade = portfolio.close_position(symbol, mid_price=price, reason=exit_reason, exit_time=pd.Timestamp(current_time))
                        if trade:
                            logger.info(
                                f"[PAPER] SELL {symbol} @ ${price:.4f} reason={exit_reason} "
                                f"P&L=${trade['pnl']:.2f} ({trade['pnl_pct']:.1%})"
                            )
                            risk_budget.record_trade(trade['pnl'])
                            stop_manager.remove_stop(symbol)
                    else:
                        order = executor.execute_order(symbol, "SELL", quantity=None)
                        if order:
                            trade = portfolio.close_position(symbol, mid_price=price, reason=exit_reason)
                            if trade:
                                risk_budget.record_trade(trade['pnl'])
                            stop_manager.remove_stop(symbol)

            # E. Log portfolio status
            summary = portfolio.get_summary(price_map)
            risk_status = risk_budget.get_status()
            active_stops = stop_manager.get_all_stops()

            logger.info(
                f"[PORTFOLIO] Equity=${summary['total_value']:.2f} | Cash=${summary['cash']:.2f} | "
                f"RealizedPnL=${summary['total_realized_pnl']:.2f} | Fees=${summary['total_fees_paid']:.2f} | "
                f"DailyPnL=${risk_status['daily_pnl']:.2f} | Drawdown={risk_status['drawdown']:.1%} | "
                f"Positions={summary['n_positions']} | ActiveStops={len(active_stops)}"
            )


            # D. Retrain Cycle
            if (datetime.now() - last_train_time).total_seconds() / 3600 >= RETRAIN_HOURS:
                logger.info("Retraining Model...")
                # Fetch full history again for retraining
                full_market = market_client.fetch_ohlcv(lookback_days=45)
                # Analyze ALL accumulated Reddit data
                full_sentiment = sentiment_analyzer.analyze(master_reddit)
                
                new_train = model.prepare_features(full_market, full_sentiment, is_inference=False)
                if not new_train.empty:
                    model.train(new_train)
                    last_train_time = datetime.now()
                else:
                    logger.warning("Retrain failed: empty features.")

            logger.info(f"Sleeping {LOOP_INTERVAL}s...")
            time.sleep(LOOP_INTERVAL)

        except KeyboardInterrupt:
            logger.info("Bot Stopped.")
            break
        except Exception as e:
            logger.error(f"Loop Error: {e}", exc_info=True)
            time.sleep(60)

if __name__ == "__main__":
    main()