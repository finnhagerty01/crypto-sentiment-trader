"""
Advanced sentiment features for crypto trading.

Extends basic VADER sentiment with dynamics, engagement weighting,
and source quality signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def _calculate_skewness(arr: np.ndarray) -> float:
    """Calculate skewness without scipy."""
    n = len(arr)
    if n < 3:
        return 0.0
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    if std == 0:
        return 0.0
    return (n / ((n - 1) * (n - 2))) * np.sum(((arr - mean) / std) ** 3)


def _calculate_kurtosis(arr: np.ndarray) -> float:
    """Calculate excess kurtosis without scipy."""
    n = len(arr)
    if n < 4:
        return 0.0
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    if std == 0:
        return 0.0
    m4 = np.mean((arr - mean) ** 4)
    m2 = np.mean((arr - mean) ** 2)
    if m2 == 0:
        return 0.0
    return (m4 / (m2 ** 2)) - 3  # Excess kurtosis


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
            "BTC": ["btc", "bitcoin"],
            "ETH": ["eth", "ethereum", "ether"],
            "SOL": ["sol", "solana"],
            "BNB": ["bnb", "binance coin"],
            "XRP": ["xrp", "ripple"],
            "ADA": ["ada", "cardano"],
            "DOGE": ["doge", "dogecoin"],
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
        if hourly_sentiment_df.empty:
            return hourly_sentiment_df

        features = hourly_sentiment_df.copy()

        # 1. Sentiment Dynamics
        features = self._add_sentiment_dynamics(features)

        # 2. Sentiment Distribution Stats
        score_col = 'score' if 'score' in posts_df.columns else 'sentiment_score'
        if score_col in posts_df.columns and not posts_df.empty:
            distribution_features = self._calculate_distribution_features(posts_df, score_col)
            if not distribution_features.empty:
                features = features.merge(distribution_features, on=['timestamp', 'symbol'], how='left')

        # 3. Engagement-Weighted Sentiment
        if all(col in posts_df.columns for col in [score_col, 'num_comments']) and not posts_df.empty:
            engagement_features = self._calculate_engagement_weighted(posts_df, score_col)
            if not engagement_features.empty:
                features = features.merge(engagement_features, on=['timestamp', 'symbol'], how='left')

        # 4. Source-Specific Sentiment
        if 'subreddit' in posts_df.columns and not posts_df.empty:
            source_features = self._calculate_source_sentiment(posts_df, score_col)
            if not source_features.empty:
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
                sma = sentiment.rolling(window, min_periods=1).mean()
                df.loc[mask, f'sentiment_sma_{window}'] = sma

                # Momentum: current vs MA
                df.loc[mask, f'sentiment_momentum_{window}'] = sentiment - sma

            # MACD-style sentiment indicator
            ema_fast = sentiment.ewm(span=6, adjust=False, min_periods=1).mean()
            ema_slow = sentiment.ewm(span=12, adjust=False, min_periods=1).mean()
            df.loc[mask, 'sentiment_macd'] = ema_fast - ema_slow
            macd_values = df.loc[mask, 'sentiment_macd']
            df.loc[mask, 'sentiment_macd_signal'] = macd_values.ewm(span=4, adjust=False, min_periods=1).mean()

            # RSI-style sentiment (bounded momentum)
            delta = sentiment.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14, min_periods=1).mean()
            avg_loss = loss.rolling(14, min_periods=1).mean()
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
        if len(sentiment) < 6:
            return pd.Series(0, index=sentiment.index)

        # Calculate rolling min/max
        rolling_min = sentiment.rolling(6, min_periods=1).min()
        rolling_max = sentiment.rolling(6, min_periods=1).max()

        # Detect when sentiment crosses from low to high or vice versa
        reversal = pd.Series(0, index=sentiment.index)

        # Bullish reversal: was near low, now rising significantly
        sentiment_shifted = sentiment.shift(3)
        rolling_min_shifted = rolling_min.shift(3)
        rolling_max_shifted = rolling_max.shift(3)

        bullish_condition = (
            (sentiment_shifted <= rolling_min_shifted * 1.1) &
            (sentiment > sentiment_shifted + threshold)
        )
        reversal = reversal.mask(bullish_condition, 1)

        # Bearish reversal: was near high, now falling significantly
        bearish_condition = (
            (sentiment_shifted >= rolling_max_shifted * 0.9) &
            (sentiment < sentiment_shifted - threshold)
        )
        reversal = reversal.mask(bearish_condition, -1)

        return reversal

    # ==================== DISTRIBUTION FEATURES ====================

    def _calculate_distribution_features(self, posts_df: pd.DataFrame, score_col: str) -> pd.DataFrame:
        """
        Calculate sentiment distribution statistics per hour.

        Captures agreement/disagreement among posts.
        """
        posts_df = posts_df.copy()
        posts_df['hour'] = pd.to_datetime(posts_df['created_utc']).dt.floor('h')

        results = []

        for (hour, symbol), group in posts_df.groupby(['hour', 'symbol']):
            scores = group[score_col].dropna()

            if len(scores) < 2:
                continue

            scores_arr = scores.values

            features = {
                'timestamp': hour,
                'symbol': symbol,

                # Basic distribution stats
                'sentiment_std': scores.std(),
                'sentiment_median': scores.median(),
                'sentiment_skew': _calculate_skewness(scores_arr),
                'sentiment_kurtosis': _calculate_kurtosis(scores_arr),

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

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results)

    # ==================== ENGAGEMENT WEIGHTED ====================

    def _calculate_engagement_weighted(self, posts_df: pd.DataFrame, score_col: str) -> pd.DataFrame:
        """
        Calculate engagement-weighted sentiment.

        Hypothesis: High-engagement posts have more market impact.
        """
        posts_df = posts_df.copy()
        posts_df['hour'] = pd.to_datetime(posts_df['created_utc']).dt.floor('h')

        # Calculate engagement score (using upvote score, not sentiment score)
        upvote_col = 'upvote_score' if 'upvote_score' in posts_df.columns else 'score'
        if upvote_col == 'score' and upvote_col == score_col:
            # If 'score' is the sentiment score, we need to check for an alternative
            upvote_col = 'ups' if 'ups' in posts_df.columns else None

        if upvote_col and upvote_col in posts_df.columns:
            posts_df['engagement'] = (
                posts_df[upvote_col].clip(lower=0) +
                posts_df['num_comments'].fillna(0) * 2
            )
        else:
            # Just use comments for engagement
            posts_df['engagement'] = posts_df['num_comments'].fillna(0) * 2

        # Log transform to reduce outlier impact
        posts_df['engagement_log'] = np.log1p(posts_df['engagement'])

        results = []

        for (hour, symbol), group in posts_df.groupby(['hour', 'symbol']):
            if len(group) == 0:
                continue

            sentiment_scores = group[score_col].dropna()
            if len(sentiment_scores) == 0:
                continue

            # Normalize weights within hour
            total_engagement = group['engagement_log'].sum()
            if total_engagement > 0:
                weights = group['engagement_log'] / total_engagement
            else:
                weights = pd.Series(1 / len(group), index=group.index)

            weighted_sentiment = (group[score_col].fillna(0) * weights).sum()

            # High engagement posts only (top 20%)
            high_engagement_threshold = group['engagement'].quantile(0.8)
            high_engagement_posts = group[group['engagement'] >= high_engagement_threshold]

            high_engagement_sentiment = np.nan
            if len(high_engagement_posts) > 0:
                high_scores = high_engagement_posts[score_col].dropna()
                if len(high_scores) > 0:
                    high_engagement_sentiment = high_scores.mean()

            features = {
                'timestamp': hour,
                'symbol': symbol,
                'sentiment_engagement_weighted': weighted_sentiment,
                'sentiment_high_engagement': high_engagement_sentiment,

                # Divergence between weighted and unweighted
                'sentiment_engagement_divergence': weighted_sentiment - sentiment_scores.mean(),

                # Engagement metrics
                'total_engagement': group['engagement'].sum(),
                'avg_engagement': group['engagement'].mean(),
                'max_engagement': group['engagement'].max(),
                'engagement_std': group['engagement'].std() if len(group) > 1 else 0.0
            }

            results.append(features)

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results)

    # ==================== SOURCE-SPECIFIC ====================

    def _calculate_source_sentiment(self, posts_df: pd.DataFrame, score_col: str) -> pd.DataFrame:
        """
        Calculate per-subreddit sentiment.

        Different subreddits may have different signal quality.
        """
        posts_df = posts_df.copy()
        posts_df['hour'] = pd.to_datetime(posts_df['created_utc']).dt.floor('h')

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
                    sub_scores = sub_posts[score_col].dropna()
                    features[f'sentiment_{sub.lower()}'] = sub_scores.mean() if len(sub_scores) > 0 else np.nan
                else:
                    features[f'sentiment_{sub.lower()}'] = np.nan

            # Cross-subreddit agreement
            sub_means = group.groupby('subreddit')[score_col].mean()
            features['subreddit_agreement'] = 1 - sub_means.std() if len(sub_means) > 1 else 1.0

            results.append(features)

        if not results:
            return pd.DataFrame()

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
            bins=[-np.inf, -0.3, -0.1, 0.1, 0.3, np.inf],
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

        return df

    # ==================== CROSS-SYMBOL ====================

    def _add_cross_symbol_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add cross-symbol sentiment features.

        Captures market-wide sentiment and relative sentiment.
        """
        df = df.copy()

        if len(df['symbol'].unique()) < 2:
            # Single symbol - use its own sentiment as market
            df['market_sentiment'] = df['sentiment_mean']
            df['market_sentiment_std'] = 0.0
            df['market_sentiment_breadth'] = (df['sentiment_mean'] > 0).astype(float)
            df['sentiment_vs_market'] = 0.0
            df['sentiment_z_score'] = 0.0
            return df

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


# ==================== ENHANCED SENTIMENT ANALYZER ====================

class EnhancedSentimentAnalyzer:
    """
    Drop-in replacement for existing SentimentAnalyzer with advanced features.

    This class wraps VADER for basic scoring and AdvancedSentimentAnalyzer
    for advanced features. The analyze() method returns a superset of the
    original columns.
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
        Returns all original columns plus advanced features.

        Args:
            df: DataFrame with columns ['title', 'selftext', 'created_utc', 'num_comments']

        Returns:
            DataFrame with columns:
                - timestamp, symbol (index columns)
                - sentiment_mean, post_volume, comment_volume (original)
                - sentiment_velocity, sentiment_acceleration, sentiment_macd, etc. (advanced)
        """
        if df.empty:
            return pd.DataFrame()

        df = df.copy()

        # 1. Basic text processing
        df['text'] = (df['title'].fillna('') + ' ' + df['selftext'].fillna('')).str.lower()

        # 2. VADER sentiment scores
        unique_text = df['text'].unique()
        scores_cache = {t: self.vader.polarity_scores(t) for t in unique_text}

        df['score'] = df['text'].map(lambda x: scores_cache[x]['compound'])
        df['sentiment_pos'] = df['text'].map(lambda x: scores_cache[x]['pos'])
        df['sentiment_neg'] = df['text'].map(lambda x: scores_cache[x]['neg'])
        df['sentiment_neu'] = df['text'].map(lambda x: scores_cache[x]['neu'])

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

        # 4. Hourly aggregation (basic - matches original output)
        hourly = posts_with_symbols.set_index('created_utc').groupby(
            ['symbol', pd.Grouper(freq='1h')]
        ).agg({
            'score': ['mean', 'count'],
            'num_comments': 'sum'
        })

        hourly.columns = ['sentiment_mean', 'post_volume', 'comment_volume']
        hourly = hourly.reset_index().rename(columns={'created_utc': 'timestamp'})

        # 5. Add advanced features
        hourly = self.advanced.calculate_all_features(posts_with_symbols, hourly)

        return hourly
