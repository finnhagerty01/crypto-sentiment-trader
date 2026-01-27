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
from src.analysis.sentiment import SentimentAnalyzer
from src.analysis.models import TradingModel
from src.execution.live import BinanceExecutor

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

TRADE_NOTIONAL_USDT = 100.0  # consistent baseline sizing for paper trades

class PaperLedger:
    def __init__(self, starting_usdt: float, fee_per_side: float, slippage_per_side: float):
        self.usdt = float(starting_usdt)
        self.fee_per_side = float(fee_per_side)
        self.slippage_per_side = float(slippage_per_side)
        # positions: symbol -> {"qty": float, "avg_entry": float, "entry_ts": datetime}
        self.positions = {} 
        # last_exit: symbol -> datetime (timestamp of last sell)
        self.last_exit = {} 
        self.realized_pnl = 0.0
        self.fees_paid = 0.0

    def _apply_buy_price(self, mid_price: float) -> float:
        return mid_price * (1.0 + self.slippage_per_side)

    def _apply_sell_price(self, mid_price: float) -> float:
        return mid_price * (1.0 - self.slippage_per_side)

    def buy(self, symbol: str, mid_price: float, notional_usdt: float, current_time: datetime) -> bool:
        if notional_usdt <= 0: return False

        fill_price = self._apply_buy_price(mid_price)
        fee = notional_usdt * self.fee_per_side
        total_cost = notional_usdt + fee

        if self.usdt < total_cost: return False

        qty = notional_usdt / fill_price

        # Existing position logic
        pos = self.positions.get(symbol, {"qty": 0.0, "avg_entry": 0.0, "entry_ts": current_time})
        
        # Weighted average entry price
        old_qty = pos["qty"]
        new_qty = old_qty + qty
        if new_qty > 0:
            new_avg = (old_qty * pos["avg_entry"] + qty * fill_price) / new_qty
        else:
            new_avg = 0.0

        pos["qty"] = new_qty
        pos["avg_entry"] = new_avg
        # Only update entry_ts if this is a fresh position (qty was 0)
        if old_qty == 0:
            pos["entry_ts"] = current_time
            
        self.positions[symbol] = pos
        self.usdt -= total_cost
        self.fees_paid += fee
        return True

    def sell_all(self, symbol: str, mid_price: float, current_time: datetime) -> bool:
        pos = self.positions.get(symbol, {"qty": 0.0, "avg_entry": 0.0})
        qty = pos["qty"]
        if qty <= 0: return False

        fill_price = self._apply_sell_price(mid_price)
        gross = qty * fill_price
        fee = gross * self.fee_per_side
        net = gross - fee

        cost_basis = qty * pos["avg_entry"]
        self.realized_pnl += (net - cost_basis)

        self.usdt += net
        self.fees_paid += fee
        
        # Record exit time for cooldown logic
        self.last_exit[symbol] = current_time

        # Clear position
        self.positions[symbol] = {"qty": 0.0, "avg_entry": 0.0, "entry_ts": None}
        return True
    
    # ... equity and snapshot methods remain the same ...
    def equity(self, price_map: dict) -> float:
        eq = self.usdt
        for sym, pos in self.positions.items():
            qty = pos.get("qty", 0.0)
            if qty > 0 and sym in price_map:
                eq += qty * price_map[sym]
        return eq

    def snapshot(self, price_map: dict) -> dict:
        return {
            "usdt": self.usdt,
            "equity": self.equity(price_map),
            "realized_pnl": self.realized_pnl,
            "fees_paid": self.fees_paid,
            "positions": {k: v for k, v in self.positions.items() if v.get("qty", 0.0) > 0}
        }

def main():
    logger.info(f"Starting Sleek Bot... DRY_RUN={DRY_RUN}")
    
    config = TradingConfig.from_yaml("configs/data.yaml")
    reddit_client = RedditClient(subreddits=config.subreddits)
    market_client = MarketClient(config)
    sentiment_analyzer = SentimentAnalyzer(symbols=config.symbols)
    model = TradingModel()
    executor = BinanceExecutor()
    COOLDOWN_HOURS = 4  # Wait 4 hours after selling before buying same symbol
    MIN_HOLD_HOURS = 2  # Minimum hold time for a position

    ledger = PaperLedger(
    starting_usdt=1000.0,
    fee_per_side=config.fee_per_side,
    slippage_per_side=config.slippage_per_side
    )

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
    
    # C. Merge
    if not master_reddit.empty:
        # Combine old + new
        master_reddit = pd.concat([master_reddit, hist_reddit]).drop_duplicates(subset=['id'])
    else:
        master_reddit = hist_reddit
        
    # D. Save merged version immediately
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
            
            # A. Fetch Live Data
            live_reddit = reddit_client.fetch_live(limit=500)
            if not live_reddit.empty:
                master_reddit = pd.concat([master_reddit, live_reddit]).drop_duplicates(subset=['id'])
                # Keep last 30 days rolling
                cutoff = datetime.now(timezone.utc) - timedelta(days=30)
                master_reddit = master_reddit[master_reddit['created_utc'] > cutoff]
                
                master_reddit.to_csv(data_path, index=False)
                logger.info(f"Database updated: {len(master_reddit)} posts.")

            # Lookback increased to 5 days to support 48h lags
            live_market = market_client.fetch_ohlcv(lookback_days=5) 

            # B. Analyze & Predict
            live_sentiment = sentiment_analyzer.analyze(master_reddit)
            pred_df = model.prepare_features(live_market, live_sentiment, is_inference=True)
            
            # PASS is_inference=True TO PREVENT DROPPING THE LIVE ROW
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
            
            # C. Execution (Paper vs Real)
            # Build a small price map for equity marking (only need symbols we hold or act on)
            current_time = datetime.now(timezone.utc) # UTC-aware for calc
            
            # Build price map
            symbols_to_price = set(signals.keys()) | set(ledger.positions.keys())
            price_map = {}
            for sym in symbols_to_price:
                try:
                    price_map[sym] = executor.get_price(sym)
                except Exception:
                    pass

            for symbol, action in signals.items():
                price = price_map.get(symbol)
                if price is None:
                    continue
                
                # Check current holding state
                pos = ledger.positions.get(symbol, {})
                current_qty = pos.get("qty", 0.0)
                is_holding = current_qty > 0

                if action == "BUY":
                    # GATE 1: State Awareness - Don't double buy
                    if is_holding:
                        # logger.info(f"Skipping BUY {symbol}: Already holding.")
                        continue
                        
                    # GATE 2: Cooldown Check
                    last_exit = ledger.last_exit.get(symbol)
                    if last_exit:
                        # Ensure timezone awareness compatibility
                        if last_exit.tzinfo is None:
                            last_exit = last_exit.replace(tzinfo=timezone.utc)
                            
                        hours_since_exit = (current_time - last_exit).total_seconds() / 3600
                        if hours_since_exit < COOLDOWN_HOURS:
                            logger.info(f"COOLDOWN BLOCKED BUY {symbol}: Exited {hours_since_exit:.1f}h ago (<{COOLDOWN_HOURS}h)")
                            continue

                    if DRY_RUN:
                        ok = ledger.buy(symbol, mid_price=price, notional_usdt=TRADE_NOTIONAL_USDT, current_time=current_time)
                        if ok:
                            logger.info(f"[PAPER] BUY {symbol} notional=${TRADE_NOTIONAL_USDT:.2f} mid={price:.6f}")
                        else:
                            logger.info(f"[PAPER] BUY skipped (insufficient cash) {symbol}")
                    else:
                        # Real execution code...
                        quantity = round(TRADE_NOTIONAL_USDT / price, 5)
                        executor.execute_order(symbol, "BUY", quantity)

                elif action == "SELL":
                    # GATE 1: State Awareness - Can't sell what you don't have
                    if not is_holding:
                        continue
                        
                    # GATE 3: Minimum Hold Time
                    entry_ts = pos.get("entry_ts")
                    if entry_ts:
                         # Ensure timezone awareness
                        if entry_ts.tzinfo is None:
                            entry_ts = entry_ts.replace(tzinfo=timezone.utc)
                            
                        hours_held = (current_time - entry_ts).total_seconds() / 3600
                        if hours_held < MIN_HOLD_HOURS:
                            logger.info(f"MIN HOLD BLOCKED SELL {symbol}: Held {hours_held:.1f}h (<{MIN_HOLD_HOURS}h)")
                            continue

                    if DRY_RUN:
                        ok = ledger.sell_all(symbol, mid_price=price, current_time=current_time)
                        if ok:
                            logger.info(f"[PAPER] SELL_ALL {symbol} mid={price:.6f}")
                    else:
                        executor.execute_order(symbol, "SELL", quantity=None)

            # Log ledger mark-to-market status each cycle
            snap = ledger.snapshot(price_map)
            logger.info(
                f"[PAPER_LEDGER] USDT=${snap['usdt']:.2f} | Equity=${snap['equity']:.2f} | "
                f"RealizedPnL=${snap['realized_pnl']:.2f} | Fees=${snap['fees_paid']:.2f} | "
                f"OpenPositions={len(snap['positions'])}"
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