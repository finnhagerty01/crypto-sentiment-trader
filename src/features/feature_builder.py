# src/features/feature_builder.py
"""
Feature engineering pipeline that combines sentiment and market data
into ML-ready features with proper time alignment.
"""

import re
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Pattern
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SentimentFeatureBuilder:
    """
    Processes Reddit sentiment data into time-aligned features.
    
    This class:
    1. Maps posts to cryptocurrency symbols using keywords
    2. Aggregates sentiment by time windows
    3. Creates weighted sentiment features
    4. Handles temporal alignment for trading
    """
    
    def __init__(self, config):
        """Initialize with trading configuration."""
        self.config = config
        self.symbol_patterns = self._build_symbol_patterns()
        
    def _build_symbol_patterns(self) -> Dict[str, Pattern]:
        """Build regex patterns for symbol detection in text."""
        patterns = {}
        keyword_map = self.config.get_symbol_keywords()
        
        for symbol, keywords in keyword_map.items():
            # Create pattern that matches keywords or cashtags
            # e.g., matches "bitcoin", "btc", "$BTC"
            keyword_pattern = '|'.join([re.escape(kw) for kw in keywords])
            pattern = rf'\b(\$?({keyword_pattern}))\b'
            patterns[symbol] = re.compile(pattern, re.IGNORECASE)
            
        return patterns
    
    def process_reddit_data(self, 
                           reddit_df: pd.DataFrame,
                           interval: str = '1h') -> pd.DataFrame:
        """
        Process Reddit posts into time-aligned sentiment features.
        
        Args:
            reddit_df: DataFrame with Reddit posts
            interval: Time interval for aggregation
        
        Returns:
            DataFrame with sentiment features per symbol and time window
        """
        if reddit_df.empty:
            return pd.DataFrame()
        
        # Map posts to symbols
        symbol_posts = self._map_posts_to_symbols(reddit_df)
        
        if symbol_posts.empty:
            logger.warning("No posts matched any symbols")
            return pd.DataFrame()
        
        # Aggregate by time windows
        features = self._aggregate_sentiment_features(symbol_posts, interval)
        
        # Add derived features
        features = self._add_derived_features(features)
        
        return features
    
    def _map_posts_to_symbols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map each post to relevant cryptocurrency symbols."""
        mapped_rows = []
        
        for idx, row in df.iterrows():
            text = row.get('full_text', '')
            if not text:
                continue
            
            # Check which symbols are mentioned
            for symbol, pattern in self.symbol_patterns.items():
                if pattern.search(text):
                    mapped_row = row.to_dict()
                    mapped_row['symbol'] = symbol
                    
                    # Calculate mention strength (how many times mentioned)
                    mentions = len(pattern.findall(text))
                    mapped_row['mention_strength'] = min(mentions, 5) / 5.0
                    
                    mapped_rows.append(mapped_row)
        
        result = pd.DataFrame(mapped_rows)
        logger.info(f"Mapped {len(df)} posts to {len(result)} symbol-post pairs")
        return result
    
    def _aggregate_sentiment_features(self,
                                     df: pd.DataFrame,
                                     interval: str) -> pd.DataFrame:
        """
        Aggregate sentiment features by symbol and time window.
        
        Uses exponential time decay and engagement weighting.
        """
        # Convert interval to pandas frequency
        freq = self._interval_to_freq(interval)
        
        # Set timestamp index
        df = df.set_index('created_utc')
        
        aggregated = []
        
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol]
            
            # Resample to intervals
            resampled = symbol_df.resample(freq)
            
            for timestamp, group in resampled:
                if group.empty:
                    continue
                
                features = self._compute_weighted_features(group, timestamp)
                features['symbol'] = symbol
                features['timestamp'] = timestamp
                features['interval'] = interval
                
                aggregated.append(features)
        
        result = pd.DataFrame(aggregated)
        return result.sort_values(['symbol', 'timestamp'])
    
    def _compute_weighted_features(self,
                                  group: pd.DataFrame,
                                  bar_end: pd.Timestamp) -> Dict:
        """
        Compute weighted sentiment features for a time window.
        
        Weights based on:
        1. Time decay (more recent = higher weight)
        2. Engagement (higher score/comments = higher weight)
        3. Mention strength
        """
        if group.empty:
            return {}
        
        # Calculate time weights (exponential decay)
        half_life = pd.Timedelta(self.config.sentiment_half_life)
        age = (bar_end - group.index).total_seconds()
        time_weights = np.exp(-np.log(2) * age / half_life.total_seconds())
        
        # Calculate engagement weights
        score_cap = 500  # Cap to reduce single viral post dominance
        scores = group['score'].fillna(0).clip(upper=score_cap)
        engagement_weights = 1.0 + np.log1p(scores) / 10
        
        # Combine weights
        weights = time_weights * engagement_weights * group.get('mention_strength', 1.0)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(group)) / len(group)
        
        # Compute weighted features
        features = {
            # Count features
            'post_count': len(group),
            'unique_authors': group['author'].nunique(),
            
            # Engagement features
            'avg_score': float(scores.mean()),
            'max_score': float(scores.max()),
            'total_comments': int(group['num_comments'].sum()),
            'avg_engagement_rate': float(group.get('engagement_rate', 0).mean()),
            
            # Weighted sentiment features
            'sentiment_mean': float(np.sum(weights * group['vader_compound'])),
            'sentiment_std': float(group['vader_compound'].std()),
            'positive_ratio': float(np.sum(weights * group['sentiment_positive'])),
            'negative_ratio': float(np.sum(weights * group['sentiment_negative'])),
            
            # VADER components (weighted)
            'vader_pos': float(np.sum(weights * group['vader_pos'])),
            'vader_neg': float(np.sum(weights * group['vader_neg'])),
            'vader_neu': float(np.sum(weights * group['vader_neu'])),
            
            # Controversy indicator
            'controversy_score': float(group.get('controversy', 0).mean()),
        }
        
        return features
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features like momentum and z-scores.
        """

        if df.empty:
            return df
        
        df = df.sort_values(['symbol', 'timestamp']).copy()
        
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            
            # Sentiment momentum (change over time)
            df.loc[mask, 'sentiment_momentum_3'] = (
                df.loc[mask, 'sentiment_mean'].diff(3).fillna(0)
            )
            
            # Z-scores for anomaly detection
            for col in ['post_count', 'sentiment_mean', 'avg_engagement_rate']:
                if col in df.columns:
                    rolling = df.loc[mask, col].rolling(24, min_periods=6)
                    mean = rolling.mean()
                    std = rolling.std()
                    df.loc[mask, f'{col}_zscore'] = (df.loc[mask, col] - mean) / (std + 1e-9)
            
            # Sentiment volatility
            df.loc[mask, 'sentiment_volatility'] = (
                df.loc[mask, 'sentiment_mean'].rolling(6).std()
            )
        
        return df
    
    def _interval_to_freq(self, interval: str) -> str:
        """Convert interval string to pandas frequency."""
        mapping = {
            '1m': '1T', '5m': '5T', '15m': '15T', '30m': '30T',
            '1h': '1H', '4h': '4H', '1d': '1D'
        }
        return mapping.get(interval, interval.upper())


class MarketFeatureBuilder:
    """
    Creates market-based features for trading models.
    
    Includes:
    1. Price-based technical indicators
    2. Volume analysis
    3. Cross-asset correlations
    4. Market microstructure features
    """
    
    def __init__(self, config):
        """Initialize with trading configuration."""
        self.config = config
        
    def process_market_data(self,
                          market_data: Dict[Tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
        """
        Process market data into features.
        
        Args:
            market_data: Dictionary of (symbol, interval) to DataFrame
        
        Returns:
            DataFrame with market features
        """
        all_features = []
        
        for (symbol, interval), df in market_data.items():
            if df.empty:
                continue
            
            features = self._create_market_features(df, symbol, interval)
            all_features.append(features)
        
        if not all_features:
            return pd.DataFrame()
        
        result = pd.concat(all_features, ignore_index=True)
        
        # Add cross-asset features
        result = self._add_cross_asset_features(result)
        
        return result
    
    def _create_market_features(self,
                               df: pd.DataFrame,
                               symbol: str,
                               interval: str) -> pd.DataFrame:
        """Create technical features from market data."""
        df = df.copy()
        
        # Ensure we have required columns
        required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            logger.warning(f"Missing required columns for {symbol}")
            return pd.DataFrame()
        
        # Price returns at different horizons
        for lag in [1, 3, 6, 12, 24]:
            df[f'return_{lag}'] = np.log(df['close'] / df['close'].shift(lag))
        
        # Volatility measures
        df['volatility_6'] = df['return_1'].rolling(6).std()
        df['volatility_24'] = df['return_1'].rolling(24).std()
        df['volatility_ratio'] = df['volatility_6'] / (df['volatility_24'] + 1e-9)
        
        # Price momentum indicators
        df['momentum_12'] = df['close'] / df['close'].shift(12) - 1
        df['momentum_24'] = df['close'] / df['close'].shift(24) - 1
        
        # RSI (Relative Strength Index)
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        # Volume features
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(24).mean()
        df['volume_momentum'] = df['volume'].pct_change(6)
        
        # Price efficiency (how close to high/low)
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)
        df['price_efficiency'] = 2 * df['price_position'] - 1  # Scale to [-1, 1]
        
        # Microstructure
        df['spread'] = (df['high'] - df['low']) / df['close']
        df['overnight_gap'] = np.log(df['open'] / df['close'].shift(1))
        
        # Add metadata
        df['symbol'] = symbol
        df['interval'] = interval
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _add_cross_asset_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features based on cross-asset relationships."""
        if df.empty or 'BTCUSDT' not in df['symbol'].values:
            return df
        
        # Get BTC data as market benchmark
        btc_data = df[df['symbol'] == 'BTCUSDT'][['timestamp', 'return_1', 'volatility_6']].copy()
        btc_data.columns = ['timestamp', 'btc_return', 'btc_volatility']
        
        # Merge BTC features to all assets
        df = df.merge(btc_data, on='timestamp', how='left')
        
        # Calculate beta to BTC
        for symbol in df['symbol'].unique():
            if symbol == 'BTCUSDT':
                df.loc[df['symbol'] == symbol, 'beta_to_btc'] = 1.0
            else:
                mask = df['symbol'] == symbol
                
                # Rolling correlation
                df.loc[mask, 'correlation_to_btc'] = (
                    df.loc[mask, 'return_1'].rolling(24).corr(df.loc[mask, 'btc_return'])
                )
                
                # Rolling beta
                cov = df.loc[mask, 'return_1'].rolling(24).cov(df.loc[mask, 'btc_return'])
                var = df.loc[mask, 'btc_return'].rolling(24).var()
                df.loc[mask, 'beta_to_btc'] = cov / (var + 1e-9)
        
        return df


class FeaturePipeline:
    """
    Main feature pipeline that orchestrates sentiment and market feature creation.
    """
    
    def __init__(self, config):
        """Initialize feature pipeline."""
        self.config = config
        self.sentiment_builder = SentimentFeatureBuilder(config)
        self.market_builder = MarketFeatureBuilder(config)
        
    def create_dataset(self,
                      interval: str = '1h',
                      lookback_days: int = 30) -> pd.DataFrame:
        """
        Create complete dataset with all features.
        
        Args:
            interval: Time interval for features
            lookback_days: Days of history to process
        
        Returns:
            DataFrame with aligned features and labels
        """
        logger.info(f"Creating dataset for {interval} interval")
        
        # Load Reddit data
        reddit_df = self._load_reddit_data(lookback_days)
        
        # Load market data
        market_data = self._load_market_data(interval, lookback_days)
        
        # Process sentiment features
        sentiment_features = self.sentiment_builder.process_reddit_data(reddit_df, interval)
        
        # Process market features
        market_features = self.market_builder.process_market_data(market_data)
        
        # Align and merge features
        dataset = self._merge_features(sentiment_features, market_features, interval)
        
        # Create labels
        dataset = self._create_labels(dataset)
        
        # Add final features
        dataset = self._add_final_features(dataset)
        
        # Clean and validate
        dataset = self._clean_dataset(dataset)
        
        logger.info(f"Created dataset with {len(dataset)} samples and {len(dataset.columns)} features")
        
        return dataset
    
    def _load_reddit_data(self, lookback_days: int) -> pd.DataFrame:
        """Load Reddit data from disk."""
        reddit_files = list(self.config.raw_reddit_dir.glob("reddit_*.parquet"))
        
        if not reddit_files:
            logger.warning("No Reddit data files found")
            return pd.DataFrame()
        
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        
        dfs = []
        for file in reddit_files:
            df = pd.read_parquet(file)
            df['created_utc'] = pd.to_datetime(df['created_utc'])
            df = df[df['created_utc'] >= cutoff]
            dfs.append(df)
        
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    def _load_market_data(self, 
                         interval: str,
                         lookback_days: int) -> Dict[Tuple[str, str], pd.DataFrame]:
        """Load market data from disk."""
        market_data = {}
        
        for symbol in self.config.symbols:
            file_path = self.config.raw_market_dir / f"{symbol}_{interval}.parquet"
            
            if not file_path.exists():
                logger.warning(f"Market data not found: {file_path}")
                continue
            
            df = pd.read_parquet(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
            df = df[df['timestamp'] >= cutoff]
            
            market_data[(symbol, interval)] = df
        
        return market_data
    
    def _merge_features(self,
                       sentiment_df: pd.DataFrame,
                       market_df: pd.DataFrame,
                       interval: str) -> pd.DataFrame:
        """Merge sentiment and market features with proper alignment."""
        if sentiment_df.empty or market_df.empty:
            logger.warning("Cannot merge empty dataframes")
            return pd.DataFrame()
        
        # Ensure consistent types
        sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])
        market_df['timestamp'] = pd.to_datetime(market_df['timestamp'])
        
        # Merge on timestamp, symbol, and interval
        merged = pd.merge(
            market_df,
            sentiment_df,
            on=['timestamp', 'symbol', 'interval'],
            how='left'  # Keep all market data, sentiment may be sparse
        )
        
        # Forward fill sentiment features (sentiment persists)
        sentiment_cols = [col for col in sentiment_df.columns 
                         if col not in ['timestamp', 'symbol', 'interval']]
        
        for col in sentiment_cols:
            merged[col] = merged.groupby('symbol')[col].fillna(method='ffill', limit=6)
        
        return merged
    
    def _create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create prediction labels for supervised learning."""
        if df.empty:
            return df
        
        df = df.sort_values(['symbol', 'timestamp']).copy()
        
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            
            # Forward returns for different horizons
            for horizon in [1, 3, 6]:
                df.loc[mask, f'forward_return_{horizon}'] = (
                    df.loc[mask, 'return_1'].shift(-horizon)
                )
            
            # Binary labels (up/down)
            df.loc[mask, 'label_direction'] = (
                df.loc[mask, 'forward_return_1'] > 0
            ).astype(int)
            
            # Multi-class labels (strong down, down, neutral, up, strong up)
            returns = df.loc[mask, 'forward_return_1']
            percentiles = returns.quantile([0.2, 0.4, 0.6, 0.8])
            
            conditions = [
                returns <= percentiles[0.2],
                (returns > percentiles[0.2]) & (returns <= percentiles[0.4]),
                (returns > percentiles[0.4]) & (returns <= percentiles[0.6]),
                (returns > percentiles[0.6]) & (returns <= percentiles[0.8]),
                returns > percentiles[0.8]
            ]
            choices = [0, 1, 2, 3, 4]  # 0=strong down, 4=strong up
            
            df.loc[mask, 'label_multiclass'] = np.select(conditions, choices, default=2)
        
        return df
    
    def _add_final_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add final engineered features."""
        if df.empty:
            return df
        
        # Interaction features
        if 'sentiment_mean' in df.columns and 'momentum_12' in df.columns:
            df['sentiment_momentum_interaction'] = df['sentiment_mean'] * df['momentum_12']
        
        if 'volume_ratio' in df.columns and 'volatility_6' in df.columns:
            df['volume_volatility_interaction'] = df['volume_ratio'] * df['volatility_6']
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Cyclical encoding for time
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def _clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the final dataset."""
        if df.empty:
            return df
        
        # Remove rows with no labels
        df = df.dropna(subset=['forward_return_1', 'label_direction'])
        
        # Handle infinities
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # Fill remaining NaNs with appropriate values
        for col in numeric_cols:
            if 'sentiment' in col or 'post_count' in col:
                # Sentiment features: fill with neutral (0)
                df[col] = df[col].fillna(0)
            else:
                # Other features: forward fill then backward fill
                df[col] = df.groupby('symbol')[col].fillna(method='ffill').fillna(method='bfill')
        
        # Drop any remaining rows with too many NaNs
        max_nan_ratio = 0.3
        df = df[df.isnull().sum(axis=1) / len(df.columns) < max_nan_ratio]
        
        # Remove the last row per symbol (no forward return)
        df = df.groupby('symbol').apply(lambda x: x.iloc[:-1]).reset_index(drop=True)
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, interval: str):
        """Save processed dataset."""
        if df.empty:
            logger.warning("Cannot save empty dataset")
            return
        
        filepath = self.config.processed_dir / f"ml_dataset_{interval}.parquet"
        df.to_parquet(filepath, index=False, compression='snappy')
        logger.info(f"Saved dataset to {filepath}")
        
        # Also save feature importance reference
        self._save_feature_metadata(df, interval)
    
    def _save_feature_metadata(self, df: pd.DataFrame, interval: str):
        """Save metadata about features for reference."""
        metadata = {
            'n_samples': len(df),
            'n_features': len(df.columns),
            'symbols': df['symbol'].unique().tolist(),
            'date_range': {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat()
            },
            'features': {
                'numeric': df.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical': df.select_dtypes(exclude=[np.number]).columns.tolist()
            },
            'label_distribution': df['label_direction'].value_counts().to_dict()
        }
        
        import json
        filepath = self.config.processed_dir / f"dataset_metadata_{interval}.json"
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved metadata to {filepath}")