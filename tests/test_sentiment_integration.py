"""Integration tests for advanced sentiment with the full pipeline."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.sentiment_advanced import EnhancedSentimentAnalyzer
from src.analysis.models import TradingModel


@pytest.fixture
def realistic_reddit_posts():
    """Create realistic Reddit post data."""
    np.random.seed(42)
    n = 500

    # Realistic titles
    btc_titles = [
        'Bitcoin breaks $50k! Moon incoming!',
        'BTC analysis: what to expect this week',
        'Why I am bullish on bitcoin',
        'Just bought more BTC, feeling good',
        'Bitcoin price prediction thread',
        'Sold all my bitcoin, here is why',
        'BTC looking bearish, be careful',
    ]

    eth_titles = [
        'Ethereum 2.0 update looks promising',
        'ETH gas fees finally dropping',
        'Why Ethereum will flip Bitcoin',
        'Just staked my ETH',
        'Ethereum analysis: key levels to watch',
    ]

    all_titles = btc_titles + eth_titles + ['General crypto discussion'] * 3

    timestamps = pd.date_range('2024-01-01', periods=n, freq='10min')

    return pd.DataFrame({
        'id': [f'post_{i}' for i in range(n)],
        'created_utc': timestamps,
        'title': np.random.choice(all_titles, n),
        'selftext': [''] * n,
        'subreddit': np.random.choice(
            ['Bitcoin', 'CryptoCurrency', 'Ethereum', 'CryptoMarkets'],
            n,
            p=[0.3, 0.4, 0.2, 0.1]
        ),
        'score': np.random.randint(1, 200, n),
        'num_comments': np.random.randint(0, 100, n)
    })


@pytest.fixture
def realistic_market_data():
    """Create realistic market data for testing."""
    np.random.seed(42)
    n = 100

    dfs = []
    for symbol, base_price in [('BTCUSDT', 50000), ('ETHUSDT', 3000)]:
        returns = np.random.randn(n) * 0.02
        close = base_price * np.cumprod(1 + returns)

        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='1h'),
            'symbol': symbol,
            'open': close * (1 + np.random.randn(n) * 0.001),
            'high': close * (1 + abs(np.random.randn(n) * 0.01)),
            'low': close * (1 - abs(np.random.randn(n) * 0.01)),
            'close': close,
            'volume': np.random.uniform(100, 1000, n)
        })
        dfs.append(df)

    return pd.concat(dfs)


class TestEnhancedAnalyzerIntegration:
    """Integration tests for EnhancedSentimentAnalyzer."""

    def test_full_analysis_pipeline(self, realistic_reddit_posts):
        """Test complete analysis from raw posts to features."""
        analyzer = EnhancedSentimentAnalyzer(['BTCUSDT', 'ETHUSDT'])
        result = analyzer.analyze(realistic_reddit_posts)

        # Should have results for both symbols
        symbols = result['symbol'].unique()
        assert 'BTCUSDT' in symbols
        assert 'ETHUSDT' in symbols

        # Should have multiple hours of data
        assert len(result['timestamp'].unique()) > 10

        # Should have both basic and advanced columns
        basic_cols = ['sentiment_mean', 'post_volume', 'comment_volume']
        for col in basic_cols:
            assert col in result.columns

        advanced_cols = ['sentiment_velocity', 'sentiment_macd', 'market_sentiment']
        for col in advanced_cols:
            assert col in result.columns

    def test_backward_compatibility(self, realistic_reddit_posts):
        """Test that output is backward compatible with original analyzer."""
        analyzer = EnhancedSentimentAnalyzer(['BTCUSDT', 'ETHUSDT'])
        result = analyzer.analyze(realistic_reddit_posts)

        # Original columns must exist
        required_original = ['timestamp', 'symbol', 'sentiment_mean', 'post_volume', 'comment_volume']
        for col in required_original:
            assert col in result.columns

        # Timestamp should be datetime
        assert pd.api.types.is_datetime64_any_dtype(result['timestamp'])

        # sentiment_mean should be numeric between -1 and 1
        assert result['sentiment_mean'].dtype in [np.float64, np.float32]
        assert (result['sentiment_mean'] >= -1).all()
        assert (result['sentiment_mean'] <= 1).all()

    def test_no_lookahead_bias(self, realistic_reddit_posts):
        """Ensure advanced features don't use future data."""
        analyzer = EnhancedSentimentAnalyzer(['BTCUSDT'])
        result = analyzer.analyze(realistic_reddit_posts)

        if len(result) < 10:
            pytest.skip("Not enough data for lookahead test")

        # Velocity at row 0 should be NaN (no previous data)
        btc_data = result[result['symbol'] == 'BTCUSDT'].sort_values('timestamp')
        if len(btc_data) > 0:
            assert pd.isna(btc_data.iloc[0]['sentiment_velocity'])


class TestModelIntegration:
    """Integration tests with TradingModel."""

    def test_sentiment_with_model_prepare(self, realistic_reddit_posts, realistic_market_data):
        """Test that enhanced sentiment works with model.prepare_features()."""
        # Get sentiment features
        analyzer = EnhancedSentimentAnalyzer(['BTCUSDT', 'ETHUSDT'])
        sentiment_df = analyzer.analyze(realistic_reddit_posts)

        # Prepare features with model
        model = TradingModel()
        features_df = model.prepare_features(realistic_market_data, sentiment_df, is_inference=False)

        # Should have data
        assert not features_df.empty

        # Should have sentiment-derived lag features
        assert 'sent_lag_1' in features_df.columns

    def test_model_train_with_enhanced_sentiment(self, realistic_reddit_posts, realistic_market_data):
        """Test that model can train with enhanced sentiment data."""
        # Get sentiment features
        analyzer = EnhancedSentimentAnalyzer(['BTCUSDT', 'ETHUSDT'])
        sentiment_df = analyzer.analyze(realistic_reddit_posts)

        # Prepare and train
        model = TradingModel()
        train_df = model.prepare_features(realistic_market_data, sentiment_df, is_inference=False)

        if train_df.empty:
            pytest.skip("Not enough data for training test")

        # Should train without error
        model.train(train_df)

        # Model should be fitted
        assert hasattr(model.model, 'n_features_in_')

    def test_model_predict_with_enhanced_sentiment(self, realistic_reddit_posts, realistic_market_data):
        """Test that model can predict with enhanced sentiment data."""
        analyzer = EnhancedSentimentAnalyzer(['BTCUSDT', 'ETHUSDT'])
        sentiment_df = analyzer.analyze(realistic_reddit_posts)

        model = TradingModel()
        train_df = model.prepare_features(realistic_market_data, sentiment_df, is_inference=False)

        if len(train_df) < 10:
            pytest.skip("Not enough data for prediction test")

        model.train(train_df)

        # Prepare inference data
        inference_df = model.prepare_features(realistic_market_data, sentiment_df, is_inference=True)

        if inference_df.empty:
            pytest.skip("No inference data available")

        # Get latest rows for prediction
        latest_ts = inference_df['timestamp'].max()
        latest = inference_df[inference_df['timestamp'] == latest_ts]

        # Should predict without error
        signals = model.predict(latest)

        # Should return dict with symbols
        assert isinstance(signals, dict)


class TestFeatureQuality:
    """Tests for feature quality and consistency."""

    def test_no_all_nan_columns(self, realistic_reddit_posts):
        """Advanced columns should not be all NaN after warmup."""
        analyzer = EnhancedSentimentAnalyzer(['BTCUSDT'])
        result = analyzer.analyze(realistic_reddit_posts)

        if len(result) < 30:
            pytest.skip("Not enough data for NaN test")

        # Skip warmup period (first 20 rows)
        result_warmup = result.iloc[20:]

        key_columns = [
            'sentiment_mean', 'sentiment_velocity', 'sentiment_macd',
            'sentiment_regime_numeric', 'market_sentiment'
        ]

        for col in key_columns:
            if col in result_warmup.columns:
                assert not result_warmup[col].isna().all(), f"Column {col} is all NaN after warmup"

    def test_numeric_columns_are_numeric(self, realistic_reddit_posts):
        """All numeric columns should have numeric dtype."""
        analyzer = EnhancedSentimentAnalyzer(['BTCUSDT'])
        result = analyzer.analyze(realistic_reddit_posts)

        numeric_cols = [
            'sentiment_mean', 'post_volume', 'comment_volume',
            'sentiment_velocity', 'sentiment_acceleration',
            'sentiment_macd', 'sentiment_rsi',
            'market_sentiment', 'sentiment_vs_market'
        ]

        for col in numeric_cols:
            if col in result.columns:
                assert pd.api.types.is_numeric_dtype(result[col]), f"Column {col} is not numeric"

    def test_cross_symbol_consistency(self, realistic_reddit_posts):
        """Cross-symbol features should be consistent across symbols."""
        analyzer = EnhancedSentimentAnalyzer(['BTCUSDT', 'ETHUSDT'])
        result = analyzer.analyze(realistic_reddit_posts)

        # Market sentiment should be the same for all symbols at each timestamp
        for ts in result['timestamp'].unique():
            ts_data = result[result['timestamp'] == ts]
            if len(ts_data) > 1:
                market_vals = ts_data['market_sentiment'].dropna().unique()
                if len(market_vals) > 0:
                    assert len(market_vals) == 1, f"Market sentiment varies across symbols at {ts}"


class TestRobustness:
    """Robustness tests for edge cases."""

    def test_sparse_data(self):
        """Test with very sparse data (few posts)."""
        posts = pd.DataFrame({
            'id': ['1', '2', '3'],
            'created_utc': pd.to_datetime(['2024-01-01 10:00', '2024-01-01 12:00', '2024-01-01 14:00']),
            'title': ['btc test', 'bitcoin test', 'BTC again'],
            'selftext': ['', '', ''],
            'subreddit': ['Bitcoin', 'Bitcoin', 'CryptoCurrency'],
            'score': [10, 20, 30],
            'num_comments': [1, 2, 3]
        })

        analyzer = EnhancedSentimentAnalyzer(['BTCUSDT'])
        result = analyzer.analyze(posts)

        # Should handle sparse data without error
        assert isinstance(result, pd.DataFrame)

    def test_single_symbol_only(self, realistic_reddit_posts):
        """Test with single symbol configuration."""
        analyzer = EnhancedSentimentAnalyzer(['BTCUSDT'])
        result = analyzer.analyze(realistic_reddit_posts)

        # Should work with single symbol
        assert not result.empty
        assert (result['symbol'] == 'BTCUSDT').all()

        # Cross-symbol features should have sensible defaults
        assert 'market_sentiment' in result.columns
        assert 'sentiment_vs_market' in result.columns

    def test_extreme_sentiment_values(self):
        """Test with extreme sentiment-generating text."""
        posts = pd.DataFrame({
            'id': [f'post_{i}' for i in range(100)],
            'created_utc': pd.date_range('2024-01-01', periods=100, freq='10min'),
            'title': ['Bitcoin is absolutely amazing incredible wonderful!'] * 50 +
                     ['Bitcoin is terrible horrible awful disaster!'] * 50,
            'selftext': [''] * 100,
            'subreddit': ['Bitcoin'] * 100,
            'score': [10] * 100,
            'num_comments': [5] * 100
        })

        analyzer = EnhancedSentimentAnalyzer(['BTCUSDT'])
        result = analyzer.analyze(posts)

        # Should capture sentiment extremes
        assert not result.empty
        assert result['sentiment_mean'].std() > 0.1  # Should have variance


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
