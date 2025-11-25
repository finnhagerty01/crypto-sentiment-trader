import time
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Add src to path so imports work
sys.path.append(str(Path(__file__).parent / 'src'))

# Import your new sleek modules
from src.utils.config import TradingConfig
from src.data.reddit_client import RedditClient
from src.data.market_client import MarketClient
from src.analysis.sentiment import SentimentAnalyzer
from src.analysis.models import TradingModel
from src.execution.live import BinanceExecutor

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trade_bot.log"),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
logger = logging.getLogger("MainOrchestrator")

# --- CONFIGURATION ---
DRY_RUN = True  # <--- SET TO FALSE ONLY WHEN READY TO LOSE MONEY
LOOP_INTERVAL = 3600  # Run every 1 hour (3600 seconds)

def main():
    logger.info("Starting Crypto Sentiment Trading Bot...")
    
    # 1. Load Configuration
    config = TradingConfig.from_yaml("configs/data.yaml")
    logger.info(f"Loaded config for symbols: {config.symbols}")

    # 2. Initialize Components
    reddit_client = RedditClient(subreddits=config.subreddits)
    market_client = MarketClient(config)
    sentiment_analyzer = SentimentAnalyzer(symbols=config.symbols)
    model = TradingModel(enter_threshold=config.enter_threshold)
    executor = BinanceExecutor()

    # 3. Training Phase (Run once on startup)
    logger.info("--- STARTING INITIAL TRAINING ---")
    
    # A. Fetch Historical Data (e.g., last 30 days)
    logger.info("Fetching historical Reddit data...")
    hist_reddit = reddit_client.fetch_historical(days=30)
    
    logger.info("Fetching historical Market data...")
    hist_market = market_client.fetch_ohlcv(lookback_days=30)

    if hist_reddit.empty or hist_market.empty:
        logger.critical("Insufficient historical data to train. Exiting.")
        return

    # B. Analyze Historical Sentiment
    logger.info("Analyzing historical sentiment...")
    hist_sentiment = sentiment_analyzer.analyze(hist_reddit)
    
    # C. Prepare Training Features
    logger.info("Building feature set...")
    train_df = model.prepare_features(hist_market, hist_sentiment)
    
    # D. Train Model
    logger.info("Training Random Forest model...")
    model.train(train_df)
    logger.info("--- TRAINING COMPLETE ---")

    # 4. Live Trading Loop
    while True:
        try:
            logger.info("--- STARTING LIVE CYCLE ---")
            
            # A. Fetch Recent Data
            # INCREASED limit to 500 to ensure we capture at least 3-4 hours of history for lags
            logger.info("Fetching live data...")
            live_reddit = reddit_client.fetch_live(limit=500) 
            live_market = market_client.fetch_ohlcv(lookback_days=2)

            if live_reddit.empty:
                logger.warning("No live Reddit data found. Skipping cycle.")
                time.sleep(60)
                continue

            # B. Analyze Live Sentiment
            live_sentiment = sentiment_analyzer.analyze(live_reddit)
            
            # C. Prepare Prediction Features
            # We assume the last row of the dataframe is the "current" hour we want to predict for
            pred_df = model.prepare_features(live_market, live_sentiment)
            
            if pred_df.empty:
                logger.warning("Feature DataFrame empty. Waiting for more data.")
                time.sleep(60)
                continue

            # Get the very latest row of data to predict on
            latest_ts = pred_df['timestamp'].max()
            
            # Select ALL symbols that have data for that timestamp
            latest_features = pred_df[pred_df['timestamp'] == latest_ts]
            
            logger.info(f"Predicting on {len(latest_features)} symbols from: {latest_ts}")

            # D. Generate Signals
            signals = model.predict(latest_features)
            
            # E. Execute Trades
            for symbol, action in signals.items():
                logger.info(f"SIGNAL for {symbol}: {action}")
                
                if action == "BUY":
                    if DRY_RUN:
                        logger.info(f"[DRY RUN] Would BUY {symbol}")
                    else:
                        # Calculate quantity (e.g., $100 worth)
                        price = executor.get_price(symbol)
                        quantity = round(100 / price, 5) # HARDCODED $100 POSITION SIZE
                        executor.execute_order(symbol, "BUY", quantity)
                        
                elif action == "HOLD":
                    logger.info(f"{symbol}: No strong signal. Holding.")

            logger.info(f"Cycle complete. Sleeping for {LOOP_INTERVAL} seconds...")
            time.sleep(LOOP_INTERVAL)

        except KeyboardInterrupt:
            logger.info("Bot stopped by user.")
            break
        except Exception as e:
            logger.error(f"Unexpected error in loop: {e}", exc_info=True)
            time.sleep(60) # Wait a bit before retrying

if __name__ == "__main__":
    main()