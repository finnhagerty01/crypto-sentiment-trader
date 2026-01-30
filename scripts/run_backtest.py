#!/usr/bin/env python3
# scripts/run_backtest.py
"""
Example script to run a backtest of the crypto sentiment trading strategy.

Usage:
    python scripts/run_backtest.py --days 90 --tune --capital 10000
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest import (
    BacktestConfig,
    BacktestEngine,
    BacktestReport,
    buy_and_hold_benchmark,
    equal_weight_benchmark,
)
from src.backtest.benchmark import compare_strategies, print_comparison
from src.data.archive import load_archive, get_archive_stats
from src.data.market_client import MarketClient
from src.features.sentiment_advanced import EnhancedSentimentAnalyzer
from src.models.trading_model import ImprovedTradingModel
from src.utils.config import TradingConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_signal_generator(
    model: ImprovedTradingModel,
    feature_df: pd.DataFrame,
):
    timestamps = feature_df["timestamp"].unique()
    ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}

    def generate_signals(data: pd.DataFrame, idx: int) -> Dict[str, str]:
        if idx >= len(timestamps):
            return {}

        current_ts = timestamps[idx]
        current_data = feature_df[feature_df["timestamp"] == current_ts]

        if current_data.empty:
            return {}

        signals_dict = model.predict(current_data)

        return {
            symbol: sig["action"]
            for symbol, sig in signals_dict.items()
        }

    return generate_signals


def main():
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--days", type=int, default=90, help="Days of history")
    parser.add_argument("--capital", type=float, default=10000, help="Initial capital")
    parser.add_argument("--train-pct", type=float, default=0.6, help="Train split %")
    parser.add_argument("--no-plots", action="store_true", help="Skip plots")
    parser.add_argument("--use-archive", action="store_true", help="Use full archive")
    
    # --- NEW ARGUMENT ---
    parser.add_argument(
        "--tune", 
        action="store_true", 
        help="Run hyperparameter tuning (Optuna) and feature selection before backtest."
    )
    
    args = parser.parse_args()

    # 1. Load configuration
    logger.info("Loading configuration...")
    config = TradingConfig.from_yaml("configs/data.yaml")

    # 2. Load historical data
    logger.info(f"Loading {args.days} days of historical market data...")
    market_client = MarketClient(config)
    market_df = market_client.fetch_ohlcv(lookback_days=args.days)

    if market_df.empty:
        logger.error("No market data loaded. Exiting.")
        return

    # 3. Load sentiment data
    logger.info("Loading sentiment data...")
    data_path = Path("data/master_reddit.csv")

    if args.use_archive:
        stats = get_archive_stats()
        if stats.get("exists"):
            logger.info(f"Using archive: {stats['total_posts']} posts")
            reddit_df = load_archive()
            reddit_df["created_utc"] = pd.to_datetime(reddit_df["created_utc"])
        else:
            logger.warning("No archive found, falling back to master_reddit.csv")
            if data_path.exists():
                reddit_df = pd.read_csv(data_path)
                reddit_df["created_utc"] = pd.to_datetime(reddit_df["created_utc"])
            else:
                reddit_df = pd.DataFrame()
    elif data_path.exists():
        reddit_df = pd.read_csv(data_path)
        reddit_df["created_utc"] = pd.to_datetime(reddit_df["created_utc"])
    else:
        reddit_df = pd.DataFrame()

    if not reddit_df.empty:
        sentiment_analyzer = EnhancedSentimentAnalyzer(symbols=config.symbols)
        sentiment_df = sentiment_analyzer.analyze(reddit_df)
    else:
        sentiment_df = pd.DataFrame()

    # 4. Prepare features and train model
    logger.info("Preparing features...")
    model = ImprovedTradingModel(
        enter_threshold=config.enter_threshold,
        exit_threshold=config.exit_threshold,
    )
    feature_df = model.prepare_features(market_df, sentiment_df, is_inference=False)

    if feature_df.empty:
        logger.error("Feature prep failed.")
        return

    timestamps = sorted(feature_df["timestamp"].unique())
    n_timestamps = len(timestamps)
    train_end_idx = int(n_timestamps * args.train_pct)
    train_end_ts = timestamps[train_end_idx]

    train_df = feature_df[feature_df["timestamp"] < train_end_ts]
    
    logger.info(f"Training on {len(train_df)} samples...")

    # --- UPDATED TRAINING CALL ---
    if args.tune:
        logger.info(">> TUNING MODE ENABLED: Running Optuna and Feature Selection <<")
    
    train_result = model.train(
        train_df, 
        validate=True,
        tune=args.tune,                 # Only tune if flag is set
        feature_selection=args.tune     # Only drop features if tuning is requested
    )
    
    logger.info(f"Training result: {train_result.get('status')}")
    if 'best_params' in train_result:
        logger.info(f"Best Parameters Found: {train_result['best_params']}")
    if 'n_features' in train_result:
        logger.info(f"Features after selection: {train_result['n_features']}")

    if train_result.get("validation"):
        val = train_result["validation"]
        logger.info(f"Validation F1: {val.get('f1', 0):.3f}")

    # 5. Create signal generator
    signal_generator = create_signal_generator(model, feature_df)

    # 6. Run backtest
    logger.info("Running backtest...")
    backtest_config = BacktestConfig(
        initial_capital=args.capital,
        fee_rate=config.fee_per_side,
        slippage_rate=config.slippage_per_side,
        max_positions=5,
        max_position_pct=0.15,
        max_exposure=0.50,
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
        enable_shorting=True,
    )

    engine = BacktestEngine(backtest_config)
    results = engine.run(
        feature_df,
        signal_generator,
        start_idx=train_end_idx,
    )

    # 7. Generate report
    report = BacktestReport(results)
    report.print_summary()

    # 8. Calculate benchmarks & plots
    logger.info("Calculating benchmarks...")
    symbols = feature_df["symbol"].unique().tolist()
    btc_benchmark = buy_and_hold_benchmark(feature_df, "BTCUSDT", args.capital)
    eth_benchmark = buy_and_hold_benchmark(feature_df, "ETHUSDT", args.capital)
    equal_weight = equal_weight_benchmark(feature_df, symbols, args.capital)

    comparison = compare_strategies(results, [btc_benchmark, eth_benchmark, equal_weight])
    print_comparison(comparison)

    if not args.no_plots:
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        report.generate_plots(save_dir=reports_dir)

    # Save detailed results
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    report.to_dataframe().to_csv(reports_dir / "backtest_trades.csv", index=False)
    comparison.to_csv(reports_dir / "strategy_comparison.csv", index=False)

    logger.info("Backtest complete!")

if __name__ == "__main__":
    main()