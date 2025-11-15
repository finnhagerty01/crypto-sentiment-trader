# main.py
"""
Main orchestrator for the crypto sentiment trading system.
Coordinates data collection, feature engineering, model training, and trading.
"""

import os
import sys
import logging
import argparse
import warnings
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import our modules
from src.utils.config import TradingConfig
from src.data.reddit_ingestion import RedditSentimentCollector
from src.data.market_ingestion import MarketDataCollector
from src.features.feature_builder import FeaturePipeline
from src.models.trading_models import TradingModelPipeline, TradingBacktester

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/crypto_trader_{datetime.now():%Y%m%d}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class CryptoTradingSystem:
    """
    Main system orchestrator that coordinates all components.
    
    Workflow:
    1. Collect Reddit sentiment data
    2. Collect market data
    3. Build features
    4. Train models
    5. Generate predictions
    6. Run backtests
    7. Generate trading signals
    """
    
    def __init__(self, config_path: str = "configs/data.yaml"):
        """Initialize the trading system."""
        logger.info("Initializing Crypto Trading System")
        
        # Load configuration
        self.config = TradingConfig.from_yaml(config_path)
        
        # Initialize components
        self.reddit_collector = RedditSentimentCollector(self.config)
        self.market_collector = MarketDataCollector(self.config)
        self.feature_pipeline = FeaturePipeline(self.config)
        self.model_pipeline = None
        self.backtester = TradingBacktester(self.config)
        
        # Ensure directories exist
        Path('logs').mkdir(exist_ok=True)
        
    def collect_data(self, 
                lookback_days: int = 30,
                use_pushshift: bool = True) -> Dict:
        """
        Collect fresh data with ALIGNED lookback periods.
        
        Args:
            lookback_days: Days of history to fetch (applies to both market and Reddit)
            use_pushshift: Whether to use Pushshift for historical Reddit data
        
        Returns:
            Dictionary with data statistics
        """
        logger.info(f"Starting data collection with {lookback_days} days lookback")
        stats = {}
        
        # Collect Reddit data
        try:
            if use_pushshift and lookback_days > 3:
                # Use hybrid: Pushshift for historical + Reddit API for recent
                reddit_df = self.reddit_collector.fetch_hybrid_data(
                    lookback_hours=24,  # Last day via API
                    historical_days=lookback_days  # Full history via Pushshift
                )
            else:
                # Use Reddit API only (limited to ~3 days realistically)
                lookback_hours = lookback_days * 24
                reddit_df = self.reddit_collector.fetch_posts(
                    lookback_hours=lookback_hours
                )
            
            if not reddit_df.empty:
                self.reddit_collector.save_data(reddit_df)
                stats['reddit_posts'] = len(reddit_df)
                stats['reddit_date_range'] = {
                    'start': reddit_df['created_utc'].min(),
                    'end': reddit_df['created_utc'].max()
                }
                logger.info(f"Collected {len(reddit_df)} Reddit posts")
            else:
                logger.warning("No Reddit posts collected")
                stats['reddit_posts'] = 0
        except Exception as e:
            logger.error(f"Error collecting Reddit data: {e}")
            stats['reddit_posts'] = 0
        
        # Collect market data (same lookback period)
        try:
            market_data = self.market_collector.fetch_multiple_symbols(
                lookback_days=lookback_days
            )
            if market_data:
                self.market_collector.save_data(market_data)
                total_candles = sum(len(df) for df in market_data.values())
                stats['market_candles'] = total_candles
                market_start = min(df['timestamp'].min() for df in market_data.values())
                market_end = max(df['timestamp'].max() for df in market_data.values())
                stats['market_date_range'] = {'start': market_start, 'end': market_end}
                logger.info(f"Collected {total_candles} market candles")
            else:
                logger.warning("No market data collected")
                stats['market_candles'] = 0
        except Exception as e:
            logger.error(f"Error collecting market data: {e}")
            stats['market_candles'] = 0
        
        # Provide quick alignment check between Reddit and market timelines
        reddit_range = stats.get('reddit_date_range')
        market_range = stats.get('market_date_range')
        if reddit_range and market_range:
            overlap_start = max(reddit_range['start'], market_range['start'])
            overlap_end = min(reddit_range['end'], market_range['end'])
            stats['timeline_overlap_hours'] = max(
                0,
                (overlap_end - overlap_start).total_seconds() / 3600
            )

        return stats
    
    def build_features(self, 
                      interval: str = '1h',
                      lookback_days: int = 30) -> pd.DataFrame:
        """
        Build feature dataset from collected data.
        
        Args:
            interval: Time interval for features
            lookback_days: Days of history to process
        
        Returns:
            Feature dataset
        """
        logger.info(f"Building features for {interval} interval")
        
        dataset = self.feature_pipeline.create_dataset(
            interval=interval,
            lookback_days=lookback_days
        )
        
        if not dataset.empty:
            self.feature_pipeline.save_dataset(dataset, interval)
            logger.info(f"Created dataset with {len(dataset)} samples")
        else:
            logger.warning("Failed to create dataset")
        
        return dataset
    
    def train_model(self,
                   dataset: pd.DataFrame,
                   model_type: str = 'ensemble') -> Dict:
        """
        Train trading model on dataset.
        
        Args:
            dataset: Feature dataset
            model_type: Type of model to train
        
        Returns:
            Training metrics
        """
        if dataset.empty:
            logger.error("Cannot train on empty dataset")
            return {}
        
        logger.info(f"Training {model_type} model")
        
        # Initialize model pipeline
        self.model_pipeline = TradingModelPipeline(self.config, model_type)
        
        # Prepare features and labels
        feature_cols = [col for col in dataset.columns 
                       if col not in ['timestamp', 'symbol', 'interval', 
                                     'forward_return_1', 'forward_return_3',
                                     'forward_return_6', 'label_direction',
                                     'label_multiclass']]
        
        X = dataset[feature_cols]
        y = dataset['label_direction']
        
        # Train model
        metrics = self.model_pipeline.train(
            X,
            y,
            feature_cols=feature_cols,
            timestamps=dataset['timestamp'],
        )
        
        # Save model
        self.model_pipeline.save_model()
        
        # Log results
        logger.info(f"Training complete:")
        logger.info(f"  Average AUC: {metrics.get('avg_auc', 0):.3f}")
        logger.info(f"  Average Accuracy: {metrics.get('avg_accuracy', 0):.3f}")
        logger.info(f"  Average F1: {metrics.get('avg_f1', 0):.3f}")
        
        # Get feature importance
        importance = self.model_pipeline.get_feature_importance()
        if not importance.empty:
            logger.info("\nTop 10 Most Important Features:")
            for idx, row in importance.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance_normalized']:.3f}")
        
        return metrics
    
    def generate_predictions(self,
                           dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for dataset.
        
        Args:
            dataset: Feature dataset
        
        Returns:
            Dataset with predictions
        """
        if self.model_pipeline is None:
            logger.error("No model trained yet")
            return dataset
        
        feature_cols = self.model_pipeline.feature_names
        X = dataset[feature_cols]
        
        predictions, probabilities = self.model_pipeline.predict(X)
        
        dataset['prediction'] = predictions
        dataset['probability'] = probabilities
        
        # Smooth probabilities to reduce noise
        dataset['probability_smooth'] = (
            dataset.groupby('symbol')['probability']
            .transform(lambda x: x.rolling(3, min_periods=1).mean())
        )
        
        return dataset
    
    def run_backtest(self,
                    dataset: pd.DataFrame,
                    strategy: str = 'threshold') -> Dict:
        """
        Run backtest on predictions.
        
        Args:
            dataset: Dataset with predictions
            strategy: Trading strategy to use
        
        Returns:
            Backtest results
        """
        if 'probability' not in dataset.columns:
            logger.error("Dataset needs predictions before backtesting")
            return {}
        
        logger.info(f"Running backtest with {strategy} strategy")
        
        results = self.backtester.backtest(
            dataset,
            probability_col='probability_smooth',
            strategy=strategy
        )
        
        # Print report
        report = self.backtester.generate_report()
        logger.info("\n" + report)
        
        # Save results
        results_path = self.config.processed_dir / f"backtest_results_{datetime.now():%Y%m%d_%H%M%S}.csv"
        results['data'].to_csv(results_path, index=False)
        logger.info(f"Saved backtest results to {results_path}")
        
        return results
    
    def generate_signals(self, 
                        interval: str = '1h') -> pd.DataFrame:
        """
        Generate current trading signals.
        
        Args:
            interval: Time interval
        
        Returns:
            DataFrame with trading signals
        """
        logger.info("Generating trading signals")
        
        # Get latest data (last 24 hours)
        dataset = self.build_features(interval=interval, lookback_days=1)
        
        if dataset.empty:
            logger.warning("No recent data for signal generation")
            return pd.DataFrame()
        
        # Generate predictions
        dataset = self.generate_predictions(dataset)
        
        # Get latest signals per symbol
        latest = dataset.groupby('symbol').last()
        
        # Determine action based on probability
        def get_action(prob):
            if prob > self.config.enter_threshold:
                return 'BUY'
            elif prob < self.config.exit_threshold:
                return 'SELL'
            else:
                return 'HOLD'
        
        latest['action'] = latest['probability_smooth'].apply(get_action)
        latest['confidence'] = abs(latest['probability_smooth'] - 0.5) * 2
        
        # Add context
        signals = latest[['action', 'confidence', 'probability_smooth']].copy()
        
        # Get current prices
        current_prices = self.market_collector.get_latest_prices()
        if not current_prices.empty:
            signals = signals.merge(
                current_prices[['symbol', 'price', 'change_24h']],
                left_index=True,
                right_on='symbol',
                how='left'
            )
        
        # Sort by confidence
        signals = signals.sort_values('confidence', ascending=False)
        for idx, row in signals.iterrows():
            symbol = row.get('symbol') or idx
        
        logger.info("\nCurrent Trading Signals:")
        logger.info("-" * 60)

        for symbol, row in signals.iterrows():
            logger.info(f"{symbol}: {row['action']} "
                       f"(Confidence: {row['confidence']:.1%}, "
                       f"Prob: {row['probability_smooth']:.3f})")
        
        return signals
    
    def run_engagement_validation(self, market_df: pd.DataFrame) -> Dict:
        """
        Run engagement validation analysis.
        
        This analyzes how post engagement correlates with price movement
        WITHOUT using engagement for prediction (avoiding temporal leakage).
        
        Args:
            market_df: Market data
        
        Returns:
            Validation results
        """
        from src.analysis.engagement_validation import EngagementValidator
        
        logger.info("Running engagement validation analysis")
        
        validator = EngagementValidator(self.config)
        results = validator.run_comprehensive_validation(market_df)
        
        # Generate plots for each symbol
        for symbol in self.config.symbols:
            plot_path = self.config.processed_dir / f'engagement_analysis_{symbol}.png'
            validator.plot_engagement_analysis(symbol, market_df, plot_path)
        
        return results
    
    def run_pipeline(self,
                    interval: str = '1h',
                    model_type: str = 'ensemble',
                    collect_fresh: bool = True,
                    lookback_days: int = 30,
                    validate_engagement: bool = True):
        """
        Run the complete trading pipeline.
        
        Args:
            interval: Time interval
            model_type: Model type to use
            collect_fresh: Whether to collect fresh data
            lookback_days: Days of history to use
        """
        logger.info("=" * 60)
        logger.info("Starting Complete Trading Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Data Collection
        if collect_fresh:
            stats = self.collect_data(lookback_days=lookback_days)
            logger.info(f"Data collection complete: {stats}")
        
        # Step 2: Feature Engineering
        dataset = self.build_features(
            interval=interval,
            lookback_days=lookback_days
        )
        
        if dataset.empty:
            logger.error("Failed to build features")
            return
        
        # Step 3: Model Training
        train_metrics = self.train_model(dataset, model_type)
        
        # Step 4: Generate Predictions
        dataset = self.generate_predictions(dataset)
        
        # Step 5: Backtest
        backtest_results = self.run_backtest(dataset)
        
        # Step 6: Generate Current Signals
        signals = self.generate_signals(interval)

        logger.info("=" * 60)
        logger.info("Pipeline Complete!")
        logger.info("=" * 60)

        if validate_engagement:
            logger.info("Running engagement validation analysis")
            validation_results = self.run_engagement_validation(dataset)
            
            # Log key findings
            optimal = validation_results['overall_findings'].get('optimal_lag', {})
            logger.info(f"Optimal engagement lag: {optimal.get('lag', 'N/A')}")
            logger.info(f"Combined correlation: {optimal.get('correlation', 0):.3f}")
        
        return {
            'dataset': dataset,
            'train_metrics': train_metrics,
            'backtest_results': backtest_results,
            'signals': signals
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Crypto Sentiment Trading System')
    
    parser.add_argument('--mode', type=str, default='full',
                       choices=['collect', 'train', 'predict', 'backtest', 'signals', 'full'],
                       help='Execution mode')
    parser.add_argument('--interval', type=str, default='1h',
                       choices=['1h', '4h'],
                       help='Time interval')
    parser.add_argument('--model', type=str, default='ensemble',
                       choices=['logistic', 'rf', 'gb', 'xgb', 'lgb', 'ensemble'],
                       help='Model type')
    parser.add_argument('--lookback', type=int, default=30,
                       help='Days of historical data')
    parser.add_argument('--no-collect', action='store_true',
                       help='Skip data collection')
    
    args = parser.parse_args()
    
    # Initialize system
    system = CryptoTradingSystem()
    
    # Execute based on mode
    if args.mode == 'collect':
        system.collect_data(lookback_days=args.lookback)
        
    elif args.mode == 'train':
        dataset = system.build_features(args.interval, args.lookback)
        system.train_model(dataset, args.model)
        
    elif args.mode == 'predict':
        dataset = system.build_features(args.interval, args.lookback)
        system.train_model(dataset, args.model)
        dataset = system.generate_predictions(dataset)
        
    elif args.mode == 'backtest':
        dataset = system.build_features(args.interval, args.lookback)
        system.train_model(dataset, args.model)
        dataset = system.generate_predictions(dataset)
        system.run_backtest(dataset)
        
    elif args.mode == 'signals':
        system.generate_signals(args.interval)
        
    elif args.mode == 'full':
        system.run_pipeline(
            interval=args.interval,
            model_type=args.model,
            collect_fresh=True,
            lookback_days=args.lookback
        )

if __name__ == "__main__":
    main()