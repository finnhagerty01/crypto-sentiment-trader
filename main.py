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
from src.data.archive import append_to_archive, load_archive
from src.features.sentiment_advanced import EnhancedSentimentAnalyzer
from src.models.trading_model import ImprovedTradingModel
from src.execution.live import BinanceExecutor

# Risk Management
from src.risk.position_sizing import PositionSizer, RiskBudget
from src.risk.stop_loss import StopLossManager
from src.risk.exit_manager import ExitManager
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
STARTING_CAPITAL = 1000.0 

# --- RISK CONFIGURATION ---
RISK_CONFIG = {
    'max_position_pct': 0.15,
    'max_total_exposure': 0.50,
    'min_confidence': 0.55,
    'default_stop_pct': 0.03,
    'default_take_profit_pct': 0.06,
    'atr_multiplier': 2.0,
    'min_hold_hours': 2.0,
    'max_hold_hours': 48.0,
    'max_daily_loss': 0.05,
    'max_drawdown': 0.15,
    'cooldown_hours': 4,
}


def main():
    logger.info(f"Starting Sleek Bot... DRY_RUN={DRY_RUN}")

    config = TradingConfig.from_yaml("configs/data.yaml")
    reddit_client = RedditClient(subreddits=config.subreddits)
    market_client = MarketClient(config)
    sentiment_analyzer = EnhancedSentimentAnalyzer(symbols=config.symbols)

    # Use trading params from config file
    model = ImprovedTradingModel(
        enter_threshold=config.enter_threshold,
        exit_threshold=config.exit_threshold,
        min_confidence=config.min_confidence,
    )
    logger.info(f"Model thresholds: enter={config.enter_threshold}, exit={config.exit_threshold}, min_conf={config.min_confidence}")

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

    last_daily_reset = datetime.now(timezone.utc).date()
    logger.info(f"Risk Management initialized: {RISK_CONFIG}")

    # --- 1. SYNC & PREPARE DATA ---
    logger.info("--- DATA SYNCHRONIZATION ---")
    
    # Load rolling window (Master Record)
    data_path = Path("data/master_reddit.csv")
    if data_path.exists():
        master_reddit = pd.read_csv(data_path)
        master_reddit['created_utc'] = pd.to_datetime(master_reddit['created_utc'])
    else:
        master_reddit = pd.DataFrame()

    # Fetch fresh data to fill gaps
    fresh_reddit = reddit_client.fetch_historical(days=30)
    if not fresh_reddit.empty:
        append_to_archive(fresh_reddit)
        if not master_reddit.empty:
            master_reddit = pd.concat([master_reddit, fresh_reddit]).drop_duplicates(subset=['id'])
        else:
            master_reddit = fresh_reddit
    
    # Save updated rolling window
    master_reddit.to_csv(data_path, index=False)


    # --- 2. INITIAL TRAINING (ARCHIVE + TUNING) ---
    logger.info("--- INITIAL TRAINING (ARCHIVE + TUNING) ---")
    
    # Load entire archive for Deep Tuning
    archive_df = load_archive()
    if not archive_df.empty:
        archive_df['created_utc'] = pd.to_datetime(archive_df['created_utc'])
        # Merge archive with current master to get maximum data
        full_training_data = pd.concat([archive_df, master_reddit]).drop_duplicates(subset=['id'])
        logger.info(f"Training on combined dataset: {len(full_training_data)} posts")
    else:
        logger.warning("Archive empty. Training on master_reddit only.")
        full_training_data = master_reddit

    # Determine date range for market data
    if not full_training_data.empty:
        min_date = full_training_data['created_utc'].min()
        days_needed = (datetime.now(timezone.utc) - min_date).days + 5
        logger.info(f"Fetching {days_needed} days of market data for deep training...")
        
        full_market = market_client.fetch_ohlcv(lookback_days=days_needed)
        full_sentiment = sentiment_analyzer.analyze(full_training_data)
        
        train_df = model.prepare_features(full_market, full_sentiment, is_inference=False)
        
        if not train_df.empty:
            logger.info(">> STARTING HYPERPARAMETER TUNING & FEATURE SELECTION <<")
            # Tune = True: Optimize hyperparameters using Optuna
            # Feature Selection = True: Drop noisy features
            model.train(train_df, tune=True, feature_selection=True)
            logger.info("Initial Tuning Complete.")
            logger.info(f"Selected Features: {model.features}")
            logger.info(f"Best Params: {model.best_params}")
        else:
            logger.error("Initial training failed: Empty features.")
    else:
        logger.error("No data available for training. Exiting.")
        return

    last_train_time = datetime.now()
    RETRAIN_HOURS = 24

    # --- 3. LIVE LOOP ---
    while True:
        try:
            logger.info("--- STARTING LIVE CYCLE ---")
            current_time = datetime.now(timezone.utc)

            # Daily Reset
            today = current_time.date()
            if today > last_daily_reset:
                risk_budget.reset_daily()
                last_daily_reset = today

            # Check Risk Halt
            if not risk_budget.can_trade():
                logger.warning(f"TRADING HALTED: {risk_budget.get_status()['halt_reasons']}")
                time.sleep(LOOP_INTERVAL)
                continue

            # A. Fetch Live Data
            live_reddit = reddit_client.fetch_live(limit=500)
            if not live_reddit.empty:
                append_to_archive(live_reddit)
                master_reddit = pd.concat([master_reddit, live_reddit]).drop_duplicates(subset=['id'])
                
                # Maintain rolling window (e.g., keep last 30 days for live retrains)
                cutoff = datetime.now(timezone.utc) - timedelta(days=30)
                master_reddit = master_reddit[master_reddit['created_utc'] > cutoff]
                master_reddit.to_csv(data_path, index=False)

            # Fetch just enough market data for inference (and small retrains)
            live_market = market_client.fetch_ohlcv(lookback_days=35) # 30 days + buffer

            # B. Analyze & Predict
            # Note: We analyze master_reddit (rolling window) for sentiment context
            live_sentiment = sentiment_analyzer.analyze(master_reddit)
            
            # Generate features for INFERENCE (is_inference=True)
            pred_df = model.prepare_features(live_market, live_sentiment, is_inference=True)

            if pred_df.empty:
                logger.warning("No features generated. Waiting...")
                time.sleep(60)
                continue

            # Filter to latest timestamp
            latest_ts = pred_df['timestamp'].max()
            latest_features = pred_df[pred_df['timestamp'] == latest_ts]
            
            # Predict
            signals = model.predict(latest_features)

            # C. Update Prices & Portfolio
            symbols_to_price = set(signals.keys()) | set(portfolio.state.positions.keys())
            price_map = {}
            for sym in symbols_to_price:
                try:
                    price_map[sym] = executor.get_price(sym)
                except Exception:
                    pass

            current_equity = portfolio.equity(price_map)
            position_sizer.update_account_value(current_equity)
            risk_budget.update(current_equity)

            # D. Manage Stops (Highest Priority)
            stop_actions = stop_manager.check_stops(price_map, pd.Timestamp(current_time))
            for symbol, action in stop_actions.items():
                if portfolio.has_position(symbol):
                    price = price_map.get(symbol)
                    # EXECUTE STOP
                    if DRY_RUN:
                        trade = portfolio.close_position(symbol, price, reason=action, exit_time=pd.Timestamp(current_time))
                        if trade: risk_budget.record_trade(trade['pnl'])
                        stop_manager.remove_stop(symbol)
                    else:
                        if executor.execute_order(symbol, "SELL", None):
                            trade = portfolio.close_position(symbol, price, reason=action)
                            if trade: risk_budget.record_trade(trade['pnl'])
                            stop_manager.remove_stop(symbol)

            # E. Execute Signals
            for symbol, signal in signals.items():
                action = signal['action']
                confidence = signal['confidence']
                price = price_map.get(symbol)
                
                if not price: continue

                is_holding = portfolio.has_position(symbol)

                if action == "BUY" and not is_holding:
                    # Risk Checks
                    in_cooldown, _ = portfolio.is_in_cooldown(symbol, RISK_CONFIG['cooldown_hours'], pd.Timestamp(current_time))
                    if in_cooldown: continue
                    
                    if not portfolio.can_open_position(symbol, 100, price_map)[0]: continue

                    # Sizing
                    # Get volatility from features if available
                    vol = 0.02
                    atr = None
                    sym_feats = latest_features[latest_features['symbol'] == symbol]
                    if not sym_feats.empty and 'atr_14_pct' in sym_feats.columns:
                        vol = sym_feats['atr_14_pct'].iloc[0]
                        atr = sym_feats['atr_14'].iloc[0]

                    pos_calc = position_sizer.calculate_position(
                        symbol, price, confidence, vol, 
                        {s: p.value for s, p in portfolio.state.positions.items()}
                    )
                    
                    if pos_calc['size_usd'] > 0:
                        qty = pos_calc['quantity']
                        # EXECUTE BUY
                        if DRY_RUN:
                            if portfolio.open_position(symbol, pos_calc['size_usd'], price, pd.Timestamp(current_time)):
                                logger.info(f"[PAPER] BUY {symbol} ${pos_calc['size_usd']:.2f} @ {price}")
                                if atr: stop_manager.create_atr_stop(symbol, price, qty, atr)
                                else: stop_manager.create_fixed_stop(symbol, price, qty)
                        else:
                            if executor.execute_order(symbol, "BUY", round(qty, 5)):
                                if atr: stop_manager.create_atr_stop(symbol, price, round(qty, 5), atr)
                                else: stop_manager.create_fixed_stop(symbol, price, round(qty, 5))

                elif action == "SELL" and is_holding:
                    # Exit Management
                    pos = portfolio.get_position(symbol)
                    decision = exit_manager.should_exit(
                        symbol, pos.entry_price, price, pos.entry_time, 
                        True, pd.Timestamp(current_time)
                    )
                    
                    if decision.can_exit:
                        # EXECUTE SELL
                        if DRY_RUN:
                            trade = portfolio.close_position(symbol, price, reason=decision.reason.value, exit_time=pd.Timestamp(current_time))
                            if trade: 
                                logger.info(f"[PAPER] SELL {symbol} PnL={trade['pnl']:.2f}")
                                risk_budget.record_trade(trade['pnl'])
                            stop_manager.remove_stop(symbol)
                        else:
                            if executor.execute_order(symbol, "SELL", None):
                                trade = portfolio.close_position(symbol, price, reason=decision.reason.value)
                                if trade: risk_budget.record_trade(trade['pnl'])
                                stop_manager.remove_stop(symbol)

            # F. Log Status
            summary = portfolio.get_summary(price_map)
            logger.info(f"[STATUS] Equity=${summary['total_value']:.2f} | Positions={summary['n_positions']}")

            # G. Retrain Cycle (ROLLING WINDOW ONLY)
            if (datetime.now() - last_train_time).total_seconds() / 3600 >= RETRAIN_HOURS:
                logger.info("--- RETRAINING MODEL (ROLLING WINDOW) ---")
                
                # 1. Use ONLY master_reddit (the rolling window 30 days)
                train_sentiment = sentiment_analyzer.analyze(master_reddit)
                
                # 2. Prepare features
                # We reuse live_market which covers the last ~35 days
                train_df = model.prepare_features(live_market, train_sentiment, is_inference=False)
                
                if not train_df.empty:
                    # 3. Train WITHOUT tuning or feature selection
                    # This keeps the hyperparameters and feature set from the Archive Tuning
                    # but updates the weights for the current market regime.
                    model.train(
                        train_df, 
                        tune=False,              # <--- Keep params from archive
                        feature_selection=False  # <--- Keep feature set from archive
                    )
                    last_train_time = datetime.now()
                    logger.info("Retraining complete.")
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