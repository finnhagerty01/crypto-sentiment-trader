"""Tests for advanced sentiment features module."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.sentiment_advanced import (
    AdvancedSentimentAnalyzer,
    EnhancedSentimentAnalyzer,
    _calculate_skewness,
    _calculate_kurtosis
)


@pytest.fixture
def sample_posts():
    """Create sample post data."""
    np.random.seed(42)
    n = 100

    return pd.DataFrame({
        'created_utc': pd.date_range('2024-01-01', periods=n, freq='30min'),
        'title': ['Bitcoin is great'] * n,
        'selftext': [''] * n,
        'subreddit': np.random.choice(['Bitcoin', 'CryptoCurrency', 'CryptoMarkets'], n),
        'score': np.random.uniform(-1, 1, n),
        'num_comments': np.random.randint(0, 50, n),
        'symbol': 'BTCUSDT'
    })


@pytest.fixture
def sample_hourly_sentiment():
    """Create sample hourly sentiment data."""
    np.random.seed(42)
    n = 50

    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='1h'),
        'symbol': 'BTCUSDT',
        'sentiment_mean': np.random.uniform(-0.5, 0.5, n),
        'post_volume': np.random.randint(1, 20, n),
        'comment_volume': np.random.randint(0, 100, n)
    })


@pytest.fixture
def multi_symbol_hourly():
    """Create multi-symbol hourly sentiment data."""
    np.random.seed(42)
    n = 30

    dfs = []
    for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='1h'),
            'symbol': symbol,
            'sentiment_mean': np.random.uniform(-0.5, 0.5, n),
            'post_volume': np.random.randint(1, 20, n),
            'comment_volume': np.random.randint(0, 100, n)
        })
        dfs.append(df)

    return pd.concat(dfs)


@pytest.fixture
def raw_reddit_posts():
    """Create raw Reddit posts for EnhancedSentimentAnalyzer testing."""
    np.random.seed(42)
    n = 200

    titles = [
        'Bitcoin to the moon!',
        'BTC looks bullish today',
        'Ethereum update is amazing',
        'ETH gas fees dropping',
        'Crypto market analysis',
        'Why I sold my bitcoin',
        'BTC vs ETH comparison'
    ]

    return pd.DataFrame({
        'id': [f'post_{i}' for i in range(n)],
        'created_utc': pd.date_range('2024-01-01', periods=n, freq='15min'),
        'title': np.random.choice(titles, n),
        'selftext': [''] * n,
        'subreddit': np.random.choice(['Bitcoin', 'CryptoCurrency', 'Ethereum'], n),
        'score': np.random.randint(1, 100, n),
        'num_comments': np.random.randint(0, 50, n)
    })


class TestSkewnessKurtosis:
    """Tests for statistical helper functions."""

    def test_skewness_symmetric(self):
        """Symmetric distribution should have ~0 skewness."""
        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        skew = _calculate_skewness(arr)
        assert abs(skew) < 0.5

    def test_skewness_right_skewed(self):
        """Right-skewed distribution should have positive skewness."""
        arr = np.array([1, 1, 1, 2, 2, 3, 10, 20, 50])
        skew = _calculate_skewness(arr)
        assert skew > 0

    def test_skewness_insufficient_data(self):
        """Should return 0 for insufficient data."""
        assert _calculate_skewness(np.array([1, 2])) == 0.0

    def test_kurtosis_normal_like(self):
        """Normal-like distribution should have ~0 excess kurtosis."""
        np.random.seed(42)
        arr = np.random.randn(1000)
        kurt = _calculate_kurtosis(arr)
        assert abs(kurt) < 0.5

    def test_kurtosis_insufficient_data(self):
        """Should return 0 for insufficient data."""
        assert _calculate_kurtosis(np.array([1, 2, 3])) == 0.0


class TestSentimentDynamics:
    """Tests for sentiment dynamics features."""

    def test_velocity_calculated(self, sample_hourly_sentiment):
        """Test that velocity is calculated correctly."""
        analyzer = AdvancedSentimentAnalyzer(['BTCUSDT'])
        result = analyzer._add_sentiment_dynamics(sample_hourly_sentiment)

        assert 'sentiment_velocity' in result.columns
        # Velocity should be diff of sentiment_mean
        expected_velocity = sample_hourly_sentiment['sentiment_mean'].diff()
        pd.testing.assert_series_equal(
            result['sentiment_velocity'].reset_index(drop=True),
            expected_velocity.reset_index(drop=True),
            check_names=False
        )

    def test_acceleration_calculated(self, sample_hourly_sentiment):
        """Test that acceleration is calculated correctly."""
        analyzer = AdvancedSentimentAnalyzer(['BTCUSDT'])
        result = analyzer._add_sentiment_dynamics(sample_hourly_sentiment)

        assert 'sentiment_acceleration' in result.columns
        # Acceleration should be diff of velocity
        assert result['sentiment_acceleration'].notna().sum() > 0

    def test_macd_calculated(self, sample_hourly_sentiment):
        """Test MACD-style sentiment indicator."""
        analyzer = AdvancedSentimentAnalyzer(['BTCUSDT'])
        result = analyzer._add_sentiment_dynamics(sample_hourly_sentiment)

        assert 'sentiment_macd' in result.columns
        assert 'sentiment_macd_signal' in result.columns
        assert result['sentiment_macd'].notna().sum() > 0

    def test_rsi_bounded(self, sample_hourly_sentiment):
        """Test RSI is bounded between 0 and 100."""
        analyzer = AdvancedSentimentAnalyzer(['BTCUSDT'])
        result = analyzer._add_sentiment_dynamics(sample_hourly_sentiment)

        assert 'sentiment_rsi' in result.columns
        rsi_valid = result['sentiment_rsi'].dropna()
        if len(rsi_valid) > 0:
            assert (rsi_valid >= 0).all() and (rsi_valid <= 100).all()

    def test_reversal_values(self, sample_hourly_sentiment):
        """Test reversal detection values are in {-1, 0, 1}."""
        analyzer = AdvancedSentimentAnalyzer(['BTCUSDT'])
        result = analyzer._add_sentiment_dynamics(sample_hourly_sentiment)

        assert 'sentiment_reversal' in result.columns
        assert set(result['sentiment_reversal'].dropna().unique()).issubset({-1, 0, 1})

    def test_sma_windows(self, sample_hourly_sentiment):
        """Test that all SMA windows are calculated."""
        analyzer = AdvancedSentimentAnalyzer(['BTCUSDT'])
        result = analyzer._add_sentiment_dynamics(sample_hourly_sentiment)

        for window in [3, 6, 12, 24]:
            assert f'sentiment_sma_{window}' in result.columns
            assert f'sentiment_momentum_{window}' in result.columns


class TestDistributionFeatures:
    """Tests for distribution features."""

    def test_distribution_features_calculated(self, sample_posts):
        """Test that distribution features are calculated."""
        analyzer = AdvancedSentimentAnalyzer(['BTCUSDT'])
        result = analyzer._calculate_distribution_features(sample_posts, 'score')

        expected_cols = [
            'sentiment_std', 'sentiment_median', 'sentiment_skew',
            'sentiment_kurtosis', 'sentiment_range', 'sentiment_iqr',
            'sentiment_max', 'sentiment_min', 'extreme_bullish_ratio',
            'extreme_bearish_ratio', 'neutral_ratio', 'sentiment_polarity',
            'sentiment_consensus'
        ]

        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_extreme_ratios_bounded(self, sample_posts):
        """Test that extreme ratios are between 0 and 1."""
        analyzer = AdvancedSentimentAnalyzer(['BTCUSDT'])
        result = analyzer._calculate_distribution_features(sample_posts, 'score')

        assert (result['extreme_bullish_ratio'] >= 0).all()
        assert (result['extreme_bullish_ratio'] <= 1).all()
        assert (result['extreme_bearish_ratio'] >= 0).all()
        assert (result['extreme_bearish_ratio'] <= 1).all()

    def test_consensus_bounded(self, sample_posts):
        """Test that consensus is between 0 and 1."""
        analyzer = AdvancedSentimentAnalyzer(['BTCUSDT'])
        result = analyzer._calculate_distribution_features(sample_posts, 'score')

        assert (result['sentiment_consensus'] >= 0).all()
        assert (result['sentiment_consensus'] <= 1).all()

    def test_empty_posts_handled(self):
        """Test handling of empty posts DataFrame."""
        analyzer = AdvancedSentimentAnalyzer(['BTCUSDT'])
        empty_df = pd.DataFrame(columns=['created_utc', 'symbol', 'score'])
        result = analyzer._calculate_distribution_features(empty_df, 'score')

        assert result.empty


class TestEngagementWeighted:
    """Tests for engagement-weighted features."""

    def test_engagement_weighted_calculated(self, sample_posts):
        """Test that engagement-weighted sentiment is calculated."""
        analyzer = AdvancedSentimentAnalyzer(['BTCUSDT'])
        result = analyzer._calculate_engagement_weighted(sample_posts, 'score')

        expected_cols = [
            'sentiment_engagement_weighted', 'sentiment_high_engagement',
            'sentiment_engagement_divergence', 'total_engagement',
            'avg_engagement', 'max_engagement', 'engagement_std'
        ]

        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_total_engagement_positive(self, sample_posts):
        """Test that total engagement is non-negative."""
        analyzer = AdvancedSentimentAnalyzer(['BTCUSDT'])
        result = analyzer._calculate_engagement_weighted(sample_posts, 'score')

        assert (result['total_engagement'] >= 0).all()


class TestSentimentRegime:
    """Tests for sentiment regime classification."""

    def test_regime_levels(self, sample_hourly_sentiment):
        """Test regime level classification."""
        analyzer = AdvancedSentimentAnalyzer(['BTCUSDT'])
        result = analyzer._add_sentiment_regime(sample_hourly_sentiment)

        assert 'sentiment_regime_level' in result.columns
        assert 'sentiment_regime_numeric' in result.columns

        # Check numeric values
        valid_regimes = {-2, -1, 0, 1, 2}
        numeric_values = set(result['sentiment_regime_numeric'].dropna().unique())
        assert numeric_values.issubset(valid_regimes)

    def test_regime_consistency(self, sample_hourly_sentiment):
        """Test that level and numeric regimes are consistent."""
        analyzer = AdvancedSentimentAnalyzer(['BTCUSDT'])

        # Set specific sentiment values
        sample_hourly_sentiment = sample_hourly_sentiment.copy()
        sample_hourly_sentiment.loc[0, 'sentiment_mean'] = -0.5  # very_bearish
        sample_hourly_sentiment.loc[1, 'sentiment_mean'] = 0.5   # very_bullish
        sample_hourly_sentiment.loc[2, 'sentiment_mean'] = 0.0   # neutral

        result = analyzer._add_sentiment_regime(sample_hourly_sentiment)

        assert result.loc[0, 'sentiment_regime_numeric'] == -2
        assert result.loc[1, 'sentiment_regime_numeric'] == 2
        assert result.loc[2, 'sentiment_regime_numeric'] == 0


class TestCrossSymbolFeatures:
    """Tests for cross-symbol features."""

    def test_market_sentiment_calculated(self, multi_symbol_hourly):
        """Test market-wide sentiment calculation."""
        analyzer = AdvancedSentimentAnalyzer(['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])
        result = analyzer._add_cross_symbol_features(multi_symbol_hourly)

        assert 'market_sentiment' in result.columns
        assert 'market_sentiment_std' in result.columns
        assert 'market_sentiment_breadth' in result.columns

    def test_sentiment_vs_market(self, multi_symbol_hourly):
        """Test relative sentiment calculation."""
        analyzer = AdvancedSentimentAnalyzer(['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])
        result = analyzer._add_cross_symbol_features(multi_symbol_hourly)

        assert 'sentiment_vs_market' in result.columns
        assert 'sentiment_z_score' in result.columns

    def test_single_symbol_handled(self, sample_hourly_sentiment):
        """Test handling of single symbol data."""
        analyzer = AdvancedSentimentAnalyzer(['BTCUSDT'])
        result = analyzer._add_cross_symbol_features(sample_hourly_sentiment)

        # Should still have columns but with sensible defaults
        assert 'market_sentiment' in result.columns
        assert 'sentiment_vs_market' in result.columns

        # For single symbol, sentiment_vs_market should be 0
        assert (result['sentiment_vs_market'] == 0).all()


class TestEnhancedSentimentAnalyzer:
    """Tests for the drop-in replacement analyzer."""

    def test_analyze_returns_basic_columns(self, raw_reddit_posts):
        """Test that analyze() returns original columns."""
        analyzer = EnhancedSentimentAnalyzer(['BTCUSDT', 'ETHUSDT'])
        result = analyzer.analyze(raw_reddit_posts)

        # Should have original columns
        assert 'timestamp' in result.columns
        assert 'symbol' in result.columns
        assert 'sentiment_mean' in result.columns
        assert 'post_volume' in result.columns
        assert 'comment_volume' in result.columns

    def test_analyze_returns_advanced_columns(self, raw_reddit_posts):
        """Test that analyze() returns advanced columns."""
        analyzer = EnhancedSentimentAnalyzer(['BTCUSDT', 'ETHUSDT'])
        result = analyzer.analyze(raw_reddit_posts)

        # Should have advanced columns
        advanced_cols = [
            'sentiment_velocity', 'sentiment_acceleration',
            'sentiment_macd', 'sentiment_rsi',
            'market_sentiment', 'sentiment_vs_market',
            'sentiment_regime_numeric'
        ]

        for col in advanced_cols:
            assert col in result.columns, f"Missing advanced column: {col}"

    def test_analyze_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        analyzer = EnhancedSentimentAnalyzer(['BTCUSDT'])
        result = analyzer.analyze(pd.DataFrame())

        assert result.empty

    def test_sentiment_bounds(self, raw_reddit_posts):
        """Test that sentiment values are within expected bounds."""
        analyzer = EnhancedSentimentAnalyzer(['BTCUSDT', 'ETHUSDT'])
        result = analyzer.analyze(raw_reddit_posts)

        # VADER compound scores are between -1 and 1
        assert (result['sentiment_mean'] >= -1).all()
        assert (result['sentiment_mean'] <= 1).all()

    def test_post_volume_positive(self, raw_reddit_posts):
        """Test that post volume is positive."""
        analyzer = EnhancedSentimentAnalyzer(['BTCUSDT', 'ETHUSDT'])
        result = analyzer.analyze(raw_reddit_posts)

        assert (result['post_volume'] > 0).all()


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_post_per_hour(self):
        """Test handling of single post per hour."""
        posts = pd.DataFrame({
            'created_utc': pd.to_datetime(['2024-01-01 10:00']),
            'title': ['Bitcoin test'],
            'selftext': [''],
            'subreddit': ['Bitcoin'],
            'score': [0.5],
            'num_comments': [5],
            'symbol': 'BTCUSDT'
        })

        analyzer = AdvancedSentimentAnalyzer(['BTCUSDT'])

        # Distribution features should handle single post
        dist = analyzer._calculate_distribution_features(posts, 'score')
        assert dist.empty  # Need at least 2 posts for distribution

    def test_all_nan_scores(self):
        """Test handling of all-NaN scores."""
        posts = pd.DataFrame({
            'created_utc': pd.date_range('2024-01-01', periods=10, freq='30min'),
            'score': [np.nan] * 10,
            'num_comments': [0] * 10,
            'symbol': 'BTCUSDT'
        })

        analyzer = AdvancedSentimentAnalyzer(['BTCUSDT'])
        result = analyzer._calculate_engagement_weighted(posts, 'score')

        # Should handle gracefully
        assert isinstance(result, pd.DataFrame)

    def test_no_matching_keywords(self):
        """Test when no posts match keywords."""
        posts = pd.DataFrame({
            'id': ['1', '2'],
            'created_utc': pd.date_range('2024-01-01', periods=2, freq='1h'),
            'title': ['Random post', 'Another random'],
            'selftext': ['', ''],
            'subreddit': ['test', 'test'],
            'score': [10, 20],
            'num_comments': [1, 2]
        })

        analyzer = EnhancedSentimentAnalyzer(['BTCUSDT'])
        result = analyzer.analyze(posts)

        assert result.empty


class TestCalculateAllFeatures:
    """Tests for the main calculate_all_features method."""

    def test_all_feature_categories(self, sample_posts, sample_hourly_sentiment):
        """Test that all feature categories are included."""
        analyzer = AdvancedSentimentAnalyzer(['BTCUSDT'])
        result = analyzer.calculate_all_features(sample_posts, sample_hourly_sentiment)

        # Dynamics
        assert 'sentiment_velocity' in result.columns

        # Regime
        assert 'sentiment_regime_numeric' in result.columns

        # Cross-symbol
        assert 'market_sentiment' in result.columns

    def test_empty_hourly_returns_empty(self, sample_posts):
        """Test that empty hourly data returns empty result."""
        analyzer = AdvancedSentimentAnalyzer(['BTCUSDT'])
        empty_hourly = pd.DataFrame()
        result = analyzer.calculate_all_features(sample_posts, empty_hourly)

        assert result.empty


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
