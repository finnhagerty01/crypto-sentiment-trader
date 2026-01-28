# Advanced Sentiment Features Implementation Guide

## Overview
This document extends the existing VADER-based sentiment analysis with advanced features that capture sentiment dynamics, engagement patterns, and source quality signals.

---

## Current State Analysis

### What Exists (`src/analysis/sentiment.py`)
```python
# Current output per hour:
# - sentiment_mean: Average VADER compound score
# - post_volume: Number of posts
# - comment_volume: Total comments
```

### What's Missing
1. Sentiment velocity (rate of change)
2. Sentiment dispersion (agreement/disagreement)
3. Engagement-weighted sentiment
4. Source-specific sentiment
5. Entity-specific sentiment extraction
6. Sentiment extremes tracking
7. Topic-based sentiment clustering

---

## File: `src/features/sentiment_advanced.py`

```python
"""
Advanced sentiment features for crypto trading.

Extends basic VADER sentiment with dynamics, engagement weighting,
and source quality signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class AdvancedSentimentAnalyzer:
    """
    Enhanced sentiment analysis with dynamics and weighting.
    
    Usage:
        analyzer = AdvancedSentimentAnalyzer(symbols=['BTCUSDT', 'ETHUSDT'])
        features = analyzer.calculate_all_features(posts_df, sentiment_df)
    """
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.keywords = self._build_keywords()
    
    def _build_keywords(self) -> Dict[str, List[str]]:
        """Build keyword mappings for each symbol."""
        base_map = {
            "BTC": ["btc", "bitcoin", "₿"],
            "ETH": ["eth", "ethereum", "ether"],
            "SOL": ["sol", "solana"],
            "BNB": ["bnb", "binance coin"],
            "XRP": ["xrp", "ripple"],
            "ADA": ["ada", "cardano"],
            "DOGE": ["doge", "dogecoin", "🐕"],
            "AVAX": ["avax", "avalanche"],
            "LINK": ["link", "chainlink"],
            "MATIC": ["matic", "polygon"],
            "DOT": ["dot", "polkadot"],
            "ATOM": ["atom", "cosmos"]
        }
        return {s: base_map.get(s.replace("USDT", ""), []) for s in self.symbols}
    
    def calculate_all_features(self, 
                               posts_df: pd.DataFrame,
                               hourly_sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all advanced sentiment features.
        
        Args:
            posts_df: Raw posts with individual sentiment scores
            hourly_sentiment_df: Aggregated hourly sentiment (from basic analyzer)
        
        Returns:
            DataFrame with all sentiment features
        """
        features = hourly_sentiment_df.copy()
        
        # 1. Sentiment Dynamics
        features = self._add_sentiment_dynamics(features)
        
        # 2. Sentiment Distribution Stats
        if 'score' in posts_df.columns:
            distribution_features = self._calculate_distribution_features(posts_df)
            features = features.merge(distribution_features, on=['timestamp', 'symbol'], how='left')
        
        # 3. Engagement-Weighted Sentiment
        if all(col in posts_df.columns for col in ['score', 'num_comments']):
            engagement_features = self._calculate_engagement_weighted(posts_df)
            features = features.merge(engagement_features, on=['timestamp', 'symbol'], how='left')
        
        # 4. Source-Specific Sentiment
        if 'subreddit' in posts_df.columns:
            source_features = self._calculate_source_sentiment(posts_df)
            features = features.merge(source_features, on=['timestamp', 'symbol'], how='left')
        
        # 5. Sentiment Regime
        features = self._add_sentiment_regime(features)
        
        # 6. Cross-Symbol Sentiment
        features = self._add_cross_symbol_features(features)
        
        return features
    
    # ==================== SENTIMENT DYNAMICS ====================
    
    def _add_sentiment_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add sentiment velocity, acceleration, and momentum features.
        
        These capture HOW sentiment is changing, not just the level.
        """
        df = df.copy()
        
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            sentiment = df.loc[mask, 'sentiment_mean']
            
            # Velocity (1st derivative) - rate of change
            df.loc[mask, 'sentiment_velocity'] = sentiment.diff()
            
            # Acceleration (2nd derivative) - change in rate of change
            df.loc[mask, 'sentiment_acceleration'] = df.loc[mask, 'sentiment_velocity'].diff()
            
            # Moving average crossovers
            for window in [3, 6, 12, 24]:
                sma = sentiment.rolling(window).mean()
                df.loc[mask, f'sentiment_sma_{window}'] = sma
                
                # Momentum: current vs MA
                df.loc[mask, f'sentiment_momentum_{window}'] = sentiment - sma
            
            # MACD-style sentiment indicator
            ema_fast = sentiment.ewm(span=6, adjust=False).mean()
            ema_slow = sentiment.ewm(span=12, adjust=False).mean()
            df.loc[mask, 'sentiment_macd'] = ema_fast - ema_slow
            df.loc[mask, 'sentiment_macd_signal'] = df.loc[mask, 'sentiment_macd'].ewm(span=4).mean()
            
            # RSI-style sentiment (bounded momentum)
            delta = sentiment.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            df.loc[mask, 'sentiment_rsi'] = 100 - (100 / (1 + rs))
            
            # Sentiment reversal detection
            df.loc[mask, 'sentiment_reversal'] = self._detect_reversals(sentiment)
        
        return df
    
    def _detect_reversals(self, sentiment: pd.Series, threshold: float = 0.2) -> pd.Series:
        """
        Detect sentiment reversals (sudden shifts).
        
        Returns:
            1 for bullish reversal, -1 for bearish reversal, 0 otherwise
        """
        # Calculate rolling min/max
        rolling_min = sentiment.rolling(6).min()
        rolling_max = sentiment.rolling(6).max()
        
        # Detect when sentiment crosses from low to high or vice versa
        reversal = pd.Series(0, index=sentiment.index)
        
        # Bullish reversal: was near low, now rising significantly
        bullish_condition = (
            (sentiment.shift(3) <= rolling_min.shift(3) * 1.1) &
            (sentiment > sentiment.shift(3) + threshold)
        )
        reversal[bullish_condition] = 1
        
        # Bearish reversal: was near high, now falling significantly  
        bearish_condition = (
            (sentiment.shift(3) >= rolling_max.shift(3) * 0.9) &
            (sentiment < sentiment.shift(3) - threshold)
        )
        reversal[bearish_condition] = -1
        
        return reversal
    
    # ==================== DISTRIBUTION FEATURES ====================
    
    def _calculate_distribution_features(self, posts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate sentiment distribution statistics per hour.
        
        Captures agreement/disagreement among posts.
        """
        posts_df = posts_df.copy()
        posts_df['hour'] = posts_df['created_utc'].dt.floor('h')
        
        results = []
        
        for (hour, symbol), group in posts_df.groupby(['hour', 'symbol']):
            scores = group['score']
            
            if len(scores) < 2:
                continue
            
            features = {
                'timestamp': hour,
                'symbol': symbol,
                
                # Basic distribution stats
                'sentiment_std': scores.std(),
                'sentiment_median': scores.median(),
                'sentiment_skew': stats.skew(scores) if len(scores) >= 3 else 0,
                'sentiment_kurtosis': stats.kurtosis(scores) if len(scores) >= 4 else 0,
                
                # Range and IQR
                'sentiment_range': scores.max() - scores.min(),
                'sentiment_iqr': scores.quantile(0.75) - scores.quantile(0.25),
                
                # Extremes
                'sentiment_max': scores.max(),
                'sentiment_min': scores.min(),
                
                # Extreme ratios
                'extreme_bullish_ratio': (scores > 0.5).mean(),
                'extreme_bearish_ratio': (scores < -0.5).mean(),
                'neutral_ratio': ((scores >= -0.1) & (scores <= 0.1)).mean(),
                
                # Polarity (how split is sentiment)
                'sentiment_polarity': abs(scores.mean()) / (scores.std() + 0.01),
                
                # Consensus score (1 = full agreement, 0 = maximum disagreement)
                'sentiment_consensus': 1 - min(scores.std() / 0.5, 1)
            }
            
            results.append(features)
        
        return pd.DataFrame(results)
    
    # ==================== ENGAGEMENT WEIGHTED ====================
    
    def _calculate_engagement_weighted(self, posts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate engagement-weighted sentiment.
        
        Hypothesis: High-engagement posts have more market impact.
        """
        posts_df = posts_df.copy()
        posts_df['hour'] = posts_df['created_utc'].dt.floor('h')
        
        # Calculate engagement score
        posts_df['engagement'] = (
            posts_df['score'].clip(lower=0) +  # Upvotes (clip negative)
            posts_df['num_comments'] * 2       # Comments weighted more
        )
        
        # Log transform to reduce outlier impact
        posts_df['engagement_log'] = np.log1p(posts_df['engagement'])
        
        results = []
        
        for (hour, symbol), group in posts_df.groupby(['hour', 'symbol']):
            if len(group) == 0:
                continue
            
            # Normalize weights within hour
            total_engagement = group['engagement_log'].sum()
            if total_engagement > 0:
                weights = group['engagement_log'] / total_engagement
            else:
                weights = 1 / len(group)
            
            weighted_sentiment = (group['sentiment_score'] * weights).sum()
            
            # High engagement posts only (top 20%)
            high_engagement_threshold = group['engagement'].quantile(0.8)
            high_engagement_posts = group[group['engagement'] >= high_engagement_threshold]
            
            features = {
                'timestamp': hour,
                'symbol': symbol,
                'sentiment_engagement_weighted': weighted_sentiment,
                'sentiment_high_engagement': high_engagement_posts['sentiment_score'].mean() if len(high_engagement_posts) > 0 else np.nan,
                
                # Divergence between weighted and unweighted
                'sentiment_engagement_divergence': weighted_sentiment - group['sentiment_score'].mean(),
                
                # Engagement metrics
                'total_engagement': group['engagement'].sum(),
                'avg_engagement': group['engagement'].mean(),
                'max_engagement': group['engagement'].max(),
                'engagement_std': group['engagement'].std()
            }
            
            results.append(features)
        
        return pd.DataFrame(results)
    
    # ==================== SOURCE-SPECIFIC ====================
    
    def _calculate_source_sentiment(self, posts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate per-subreddit sentiment.
        
        Different subreddits may have different signal quality.
        """
        posts_df = posts_df.copy()
        posts_df['hour'] = posts_df['created_utc'].dt.floor('h')
        
        # Define subreddit categories
        subreddit_weights = {
            # Coin-specific (higher signal for that coin)
            'Bitcoin': {'BTCUSDT': 1.5},
            'Ethereum': {'ETHUSDT': 1.5},
            'solana': {'SOLUSDT': 1.5},
            
            # General (equal weight)
            'CryptoCurrency': {},
            'CryptoMarkets': {},
            
            # Lower signal (more noise)
            'altcoin': {'_default': 0.8},
            'SatoshiStreetBets': {'_default': 0.7}
        }
        
        results = []
        
        for (hour, symbol), group in posts_df.groupby(['hour', 'symbol']):
            features = {
                'timestamp': hour,
                'symbol': symbol
            }
            
            # Per-subreddit sentiment
            for sub in ['Bitcoin', 'Ethereum', 'CryptoCurrency', 'CryptoMarkets']:
                sub_posts = group[group['subreddit'].str.lower() == sub.lower()]
                if len(sub_posts) > 0:
                    features[f'sentiment_{sub.lower()}'] = sub_posts['sentiment_score'].mean()
                else:
                    features[f'sentiment_{sub.lower()}'] = np.nan
            
            # Cross-subreddit agreement
            sub_means = group.groupby('subreddit')['sentiment_score'].mean()
            features['subreddit_agreement'] = 1 - sub_means.std() if len(sub_means) > 1 else 1
            
            results.append(features)
        
        return pd.DataFrame(results)
    
    # ==================== SENTIMENT REGIME ====================
    
    def _add_sentiment_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify sentiment into discrete regimes.
        
        Useful for regime-switching models.
        """
        df = df.copy()
        
        # Level-based regime
        df['sentiment_regime_level'] = pd.cut(
            df['sentiment_mean'],
            bins=[-1, -0.3, -0.1, 0.1, 0.3, 1],
            labels=['very_bearish', 'bearish', 'neutral', 'bullish', 'very_bullish']
        )
        
        # Encode as numeric for ML
        regime_map = {
            'very_bearish': -2,
            'bearish': -1,
            'neutral': 0,
            'bullish': 1,
            'very_bullish': 2
        }
        df['sentiment_regime_numeric'] = df['sentiment_regime_level'].map(regime_map)
        
        # Momentum-based regime
        if 'sentiment_velocity' in df.columns:
            df['sentiment_momentum_regime'] = pd.cut(
                df['sentiment_velocity'].fillna(0),
                bins=[-np.inf, -0.05, -0.01, 0.01, 0.05, np.inf],
                labels=['strong_decline', 'declining', 'stable', 'rising', 'strong_rise']
            )
        
        # Combined regime (level + momentum)
        # E.g., "bullish_rising" vs "bullish_declining"
        
        return df
    
    # ==================== CROSS-SYMBOL ====================
    
    def _add_cross_symbol_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add cross-symbol sentiment features.
        
        Captures market-wide sentiment and relative sentiment.
        """
        df = df.copy()
        
        # Pivot to get all symbols as columns
        sentiment_pivot = df.pivot_table(
            index='timestamp',
            columns='symbol',
            values='sentiment_mean',
            aggfunc='first'
        )
        
        # Market-wide sentiment
        market_sentiment = sentiment_pivot.mean(axis=1)
        market_sentiment_std = sentiment_pivot.std(axis=1)
        
        # Merge back
        market_features = pd.DataFrame({
            'timestamp': market_sentiment.index,
            'market_sentiment': market_sentiment.values,
            'market_sentiment_std': market_sentiment_std.values,
            'market_sentiment_breadth': (sentiment_pivot > 0).mean(axis=1).values
        })
        
        df = df.merge(market_features, on='timestamp', how='left')
        
        # Relative sentiment (symbol vs market)
        df['sentiment_vs_market'] = df['sentiment_mean'] - df['market_sentiment']
        
        # Z-score relative to market
        df['sentiment_z_score'] = (
            (df['sentiment_mean'] - df['market_sentiment']) / 
            df['market_sentiment_std'].replace(0, np.nan)
        )
        
        return df


# ==================== UPDATED SENTIMENT ANALYZER ====================

class EnhancedSentimentAnalyzer:
    """
    Drop-in replacement for existing SentimentAnalyzer with advanced features.
    """
    
    def __init__(self, symbols: List[str]):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.vader = SentimentIntensityAnalyzer()
        self.symbols = symbols
        self.advanced = AdvancedSentimentAnalyzer(symbols)
        self.keywords = self._build_keywords()
    
    def _build_keywords(self) -> Dict[str, List[str]]:
        """Build keyword mappings."""
        base_map = {
            "BTC": ["btc", "bitcoin"],
            "ETH": ["eth", "ethereum"],
            "SOL": ["sol", "solana"],
            "DOGE": ["doge", "dogecoin"],
            "ADA": ["ada", "cardano"],
            "XRP": ["xrp", "ripple"],
            "BNB": ["bnb", "binance coin"],
            "AVAX": ["avax", "avalanche"],
            "DOT": ["dot", "polkadot"],
            "LINK": ["link", "chainlink"],
            "MATIC": ["matic", "polygon"],
            "ATOM": ["atom", "cosmos"]
        }
        return {s: base_map.get(s.replace("USDT", ""), []) for s in self.symbols}
    
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze posts and return comprehensive sentiment features.
        
        This is a drop-in replacement for the existing analyze() method.
        """
        if df.empty:
            return pd.DataFrame()
        
        df = df.copy()
        
        # 1. Basic text processing
        df['text'] = (df['title'].fillna('') + ' ' + df['selftext'].fillna('')).str.lower()
        
        # 2. VADER sentiment scores
        unique_text = df['text'].unique()
        scores = {t: self.vader.polarity_scores(t) for t in unique_text}
        
        df['sentiment_score'] = df['text'].map(lambda x: scores[x]['compound'])
        df['sentiment_pos'] = df['text'].map(lambda x: scores[x]['pos'])
        df['sentiment_neg'] = df['text'].map(lambda x: scores[x]['neg'])
        df['sentiment_neu'] = df['text'].map(lambda x: scores[x]['neu'])
        
        # 3. Assign to symbols based on keywords
        results = []
        for symbol, keywords in self.keywords.items():
            if not keywords:
                continue
            
            pattern = r'(?<!\w)(?:' + '|'.join(keywords) + r')(?!\w)'
            mask = df['text'].str.contains(pattern, regex=True)
            subset = df[mask].copy()
            subset['symbol'] = symbol
            results.append(subset)
        
        if not results:
            return pd.DataFrame()
        
        posts_with_symbols = pd.concat(results)
        
        # 4. Hourly aggregation (basic)
        hourly = posts_with_symbols.set_index('created_utc').groupby(
            ['symbol', pd.Grouper(freq='1h')]
        ).agg({
            'sentiment_score': ['mean', 'std', 'min', 'max', 'count'],
            'num_comments': 'sum',
            'score': 'sum'
        })
        
        hourly.columns = [
            'sentiment_mean', 'sentiment_std', 'sentiment_min', 'sentiment_max',
            'post_volume', 'comment_volume', 'total_score'
        ]
        hourly = hourly.reset_index().rename(columns={'created_utc': 'timestamp'})
        
        # 5. Add advanced features
        hourly = self.advanced.calculate_all_features(posts_with_symbols, hourly)
        
        return hourly
```

---

## Integration Example

### Modify `main.py` to use enhanced analyzer:

```python
# Change import
from src.features.sentiment_advanced import EnhancedSentimentAnalyzer

# In main():
sentiment_analyzer = EnhancedSentimentAnalyzer(symbols=config.symbols)
```

---

## Feature Summary

| Feature | Description | Expected Signal |
|---------|-------------|-----------------|
| sentiment_velocity | Rate of sentiment change | Leading indicator |
| sentiment_acceleration | Acceleration of sentiment | Momentum shifts |
| sentiment_std | Disagreement among posts | High = uncertainty |
| sentiment_engagement_weighted | Engagement-weighted mean | Quality signal |
| sentiment_vs_market | Symbol vs market sentiment | Relative strength |
| sentiment_consensus | Agreement level | High = conviction |
| extreme_bullish_ratio | % of very positive posts | Extreme = reversal risk |
| sentiment_reversal | Detected reversal points | Timing signal |
| subreddit_agreement | Cross-source agreement | Confirmation |

---

## Testing

```python
# tests/test_sentiment_advanced.py

import pytest
import pandas as pd
import numpy as np
from src.features.sentiment_advanced import AdvancedSentimentAnalyzer

@pytest.fixture
def sample_posts():
    """Create sample post data."""
    np.random.seed(42)
    n = 100
    
    return pd.DataFrame({
        'created_utc': pd.date_range('2024-01-01', periods=n, freq='30min'),
        'title': ['Bitcoin is great'] * n,
        'selftext': [''] * n,
        'subreddit': np.random.choice(['Bitcoin', 'CryptoCurrency'], n),
        'score': np.random.randint(1, 100, n),
        'num_comments': np.random.randint(0, 50, n),
        'sentiment_score': np.random.uniform(-1, 1, n),
        'symbol': 'BTCUSDT'
    })

def test_sentiment_velocity(sample_posts):
    """Test that velocity is calculated correctly."""
    analyzer = AdvancedSentimentAnalyzer(['BTCUSDT'])
    
    # Create hourly sentiment
    hourly = sample_posts.set_index('created_utc').resample('1h').agg({
        'sentiment_score': 'mean',
        'score': 'count'
    }).reset_index()
    hourly.columns = ['timestamp', 'sentiment_mean', 'post_volume']
    hourly['symbol'] = 'BTCUSDT'
    
    result = analyzer._add_sentiment_dynamics(hourly)
    
    assert 'sentiment_velocity' in result.columns
    assert result['sentiment_velocity'].notna().sum() > 0

def test_engagement_weighting(sample_posts):
    """Test engagement-weighted sentiment."""
    analyzer = AdvancedSentimentAnalyzer(['BTCUSDT'])
    
    result = analyzer._calculate_engagement_weighted(sample_posts)
    
    assert 'sentiment_engagement_weighted' in result.columns
    # Weighted should differ from unweighted
    assert result['sentiment_engagement_divergence'].std() > 0
```
