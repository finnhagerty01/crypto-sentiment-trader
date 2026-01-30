#!/usr/bin/env python3
# scripts/run_backtest.py
"""
Example script to run a backtest of the crypto sentiment trading strategy.

This script demonstrates:
1. Loading historical market and sentiment data
2. Preparing features and training the model
3. Running walk-forward backtest
4. Generating performance reports
5. Comparing against benchmarks

Usage:
    python scripts/run_backtest.py [--days DAYS] [--capital CAPITAL] [--no-plots]
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
from src.data.reddit_client import RedditClient
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
    """
    Create a signal generator function for the backtest engine.

    The generator uses the trained model to produce BUY/SELL/HOLD signals
    for each symbol at each timestamp.

    Args:
        model: Trained ImprovedTradingModel instance
        feature_df: Full feature DataFrame for lookups

    Returns:
        Signal generator function compatible with BacktestEngine.run()
    """
    # Pre-index data by timestamp for faster lookups
    timestamps = feature_df["timestamp"].unique()
    ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}

    def generate_signals(data: pd.DataFrame, idx: int) -> Dict[str, str]:
        """Generate signals for current timestamp."""
        if idx >= len(timestamps):
            return {}

        current_ts = timestamps[idx]
        current_data = feature_df[feature_df["timestamp"] == current_ts]

        if current_data.empty:
            return {}

        # Get model predictions
        signals_dict = model.predict(current_data)

        # Convert to simple action strings
        return {
            symbol: sig["action"]
            for symbol, sig in signals_dict.items()
        }

    return generate_signals


def main():
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of days of historical data (default: 90)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10000,
        help="Initial capital in USD (default: 10000)",
    )
    parser.add_argument(
        "--train-pct",
        type=float,
        default=0.6,
        help="Percentage of data for training (default: 0.6)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots",
    )
    parser.add_argument(
        "--use-archive",
        action="store_true",
        help="Use full historical archive instead of rolling 30-day data",
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

    logger.info(f"Loaded {len(market_df)} market data rows")

    # 3. Load sentiment data
    logger.info("Loading sentiment data...")
    data_path = Path("data/master_reddit.csv")

    if args.use_archive:
        # Use full historical archive for backtesting
        stats = get_archive_stats()
        if stats.get("exists"):
            logger.info(
                f"Using archive: {stats['total_posts']} posts, "
                f"{stats['file_size_mb']:.1f} MB, range: {stats.get('date_range', 'unknown')}"
            )
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
        logger.info(f"Loaded {len(reddit_df)} Reddit posts from rolling window")
    else:
        reddit_df = pd.DataFrame()

    if not reddit_df.empty:
        logger.info(f"Loaded {len(reddit_df)} Reddit posts")
        sentiment_analyzer = EnhancedSentimentAnalyzer(symbols=config.symbols)
        sentiment_df = sentiment_analyzer.analyze(reddit_df)
    else:
        logger.warning("No sentiment data found, proceeding without sentiment features")
        sentiment_df = pd.DataFrame()

    # 4. Prepare features and train model
    logger.info("Preparing features...")
    model = ImprovedTradingModel(
        enter_threshold=config.enter_threshold,
        exit_threshold=config.exit_threshold,
    )
    feature_df = model.prepare_features(market_df, sentiment_df, is_inference=False)

    if feature_df.empty:
        logger.error("Feature preparation returned empty DataFrame. Exiting.")
        return

    logger.info(f"Prepared {len(feature_df)} feature rows")

    # Split into train/test BY TIMESTAMP (not by row index)
    # This ensures proper temporal separation
    timestamps = sorted(feature_df["timestamp"].unique())
    n_timestamps = len(timestamps)
    train_end_idx = int(n_timestamps * args.train_pct)
    train_end_ts = timestamps[train_end_idx]

    train_df = feature_df[feature_df["timestamp"] < train_end_ts]
    test_df = feature_df[feature_df["timestamp"] >= train_end_ts]

    logger.info(f"Timestamp split: {n_timestamps} unique timestamps")
    logger.info(f"Train period: {timestamps[0]} to {train_end_ts}")
    logger.info(f"Test period:  {train_end_ts} to {timestamps[-1]}")
    logger.info(f"Training on {len(train_df)} samples ({args.train_pct:.0%} of timestamps)...")
    train_result = model.train(train_df, validate=True)
    logger.info(f"Training result: {train_result.get('status')}")

    if train_result.get("validation"):
        val = train_result["validation"]
        logger.info(
            f"Validation: precision={val.get('precision', 0):.3f}, "
            f"recall={val.get('recall', 0):.3f}, "
            f"f1={val.get('f1', 0):.3f}"
        )

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
    )

    engine = BacktestEngine(backtest_config)
    results = engine.run(
        feature_df,
        signal_generator,
        start_idx=train_end_idx,  # Start at test period (timestamp index)
    )

    # 7. Generate report
    report = BacktestReport(results)
    report.print_summary()

    # 8. Calculate benchmarks
    logger.info("Calculating benchmarks...")
    symbols = feature_df["symbol"].unique().tolist()

    # Buy & Hold BTC
    btc_benchmark = buy_and_hold_benchmark(
        feature_df,
        "BTCUSDT",
        args.capital,
        fee_rate=config.fee_per_side,
    )

    # Buy & Hold ETH
    eth_benchmark = buy_and_hold_benchmark(
        feature_df,
        "ETHUSDT",
        args.capital,
        fee_rate=config.fee_per_side,
    )

    # Equal weight portfolio
    equal_weight = equal_weight_benchmark(
        feature_df,
        symbols,
        args.capital,
        fee_rate=config.fee_per_side,
    )

    # Compare strategies
    comparison = compare_strategies(
        results,
        [btc_benchmark, eth_benchmark, equal_weight],
    )
    print_comparison(comparison)

    # 9. Generate plots
    if not args.no_plots:
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        report.generate_plots(save_dir=reports_dir)

    # 10. Save detailed results
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    trades_df = report.to_dataframe()
    if not trades_df.empty:
        trades_path = reports_dir / "backtest_trades.csv"
        trades_df.to_csv(trades_path, index=False)
        logger.info(f"Saved trades to {trades_path}")

        # Symbol breakdown
        symbol_breakdown = report.get_symbol_breakdown()
        if not symbol_breakdown.empty:
            symbol_path = reports_dir / "backtest_by_symbol.csv"
            symbol_breakdown.to_csv(symbol_path)
            logger.info(f"Saved symbol breakdown to {symbol_path}")

        # Exit reason breakdown
        exit_breakdown = report.get_exit_reason_breakdown()
        if not exit_breakdown.empty:
            exit_path = reports_dir / "backtest_by_exit.csv"
            exit_breakdown.to_csv(exit_path)
            logger.info(f"Saved exit breakdown to {exit_path}")

    # Save comparison
    comparison_path = reports_dir / "strategy_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    logger.info(f"Saved comparison to {comparison_path}")

    logger.info("Backtest complete!")


if __name__ == "__main__":
    main()
