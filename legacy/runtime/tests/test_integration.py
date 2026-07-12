# tests/test_integration.py
"""Integration tests for the trading model with technical indicators."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.models import TradingModel


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    np.random.seed(42)
    n = 200  # Need enough data for lag features (48h) + warmup

    dfs = []
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        returns = np.random.randn(n) * 0.02
        close = 50000 * np.cumprod(1 + returns) if symbol == 'BTCUSDT' else 3000 * np.cumprod(1 + returns)

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


@pytest.fixture
def sample_sentiment_data():
    """Create sample sentiment data."""
    np.random.seed(42)
    n = 150

    dfs = []
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-02', periods=n, freq='1h'),
            'symbol': symbol,
            'sentiment_mean': np.random.uniform(-0.5, 0.5, n),
            'post_volume': np.random.randint(1, 50, n),
            'comment_volume': np.random.randint(10, 500, n)
        })
        dfs.append(df)

    return pd.concat(dfs)


class TestModelIntegration:
    """Test the full model pipeline with technical indicators."""

    def test_prepare_features_creates_technical_indicators(self, sample_market_data, sample_sentiment_data):
        """Verify technical indicators are added to features."""
        model = TradingModel()
        df = model.prepare_features(sample_market_data, sample_sentiment_data, is_inference=False)

        # Check that technical features are in the feature list
        technical_in_features = [f for f in model.features if 'rsi' in f or 'macd' in f or 'bb_' in f]
        assert len(technical_in_features) > 0, "No technical indicators in feature list"

        # Check that lag features are also present
        lag_in_features = [f for f in model.features if 'lag' in f]
        assert len(lag_in_features) > 0, "No lag features in feature list"

    def test_prepare_features_has_expected_count(self, sample_market_data, sample_sentiment_data):
        """Verify we have the expected number of features."""
        model = TradingModel()
        df = model.prepare_features(sample_market_data, sample_sentiment_data, is_inference=False)

        # 8 lags * 3 types (sent, vol, ret) = 24 lag features
        # Plus ~20 technical features
        assert len(model.features) >= 40, f"Expected at least 40 features, got {len(model.features)}"

    def test_train_with_technical_indicators(self, sample_market_data, sample_sentiment_data):
        """Test that model can train with technical indicators."""
        model = TradingModel()
        df = model.prepare_features(sample_market_data, sample_sentiment_data, is_inference=False)

        assert not df.empty, "Feature preparation returned empty DataFrame"

        # Train should work
        result = model.train(df)
        assert result is not None, "Training failed"

    def test_predict_with_technical_indicators(self, sample_market_data, sample_sentiment_data):
        """Test full train + predict cycle."""
        model = TradingModel()

        # Train
        train_df = model.prepare_features(sample_market_data, sample_sentiment_data, is_inference=False)
        model.train(train_df)

        # Predict (using last few rows)
        pred_df = model.prepare_features(sample_market_data, sample_sentiment_data, is_inference=True)
        latest = pred_df[pred_df['timestamp'] == pred_df['timestamp'].max()]

        signals = model.predict(latest)

        # Should have signals for both symbols
        assert len(signals) > 0, "No predictions generated"
        assert all(s in ['BUY', 'SELL', 'HOLD'] for s in signals.values()), "Invalid signal values"

    def test_no_nan_in_features_after_warmup(self, sample_market_data, sample_sentiment_data):
        """After warmup period, there should be no NaN in features."""
        model = TradingModel()
        df = model.prepare_features(sample_market_data, sample_sentiment_data, is_inference=False)

        # After dropna in prepare_features, there should be no NaN in feature columns
        for feature in model.features:
            assert not df[feature].isna().any(), f"NaN found in {feature} after preparation"

    def test_empty_sentiment_still_works(self, sample_market_data):
        """Model should work even with empty sentiment data."""
        model = TradingModel()
        df = model.prepare_features(sample_market_data, pd.DataFrame(), is_inference=False)

        assert not df.empty, "Should handle empty sentiment data"
        assert 'sentiment_mean' in df.columns
        assert (df['sentiment_mean'] == 0.0).all(), "Sentiment should be zero-filled"


class TestFeatureImportance:
    """Test feature importance after training."""

    def test_can_get_feature_importances(self, sample_market_data, sample_sentiment_data):
        """Verify we can extract feature importances."""
        model = TradingModel()
        df = model.prepare_features(sample_market_data, sample_sentiment_data, is_inference=False)
        model.train(df)

        importances = model.model.feature_importances_
        assert len(importances) == len(model.features)
        assert all(i >= 0 for i in importances), "Importances should be non-negative"

    def test_feature_importance_ranking(self, sample_market_data, sample_sentiment_data):
        """Check feature importances can be ranked."""
        model = TradingModel()
        df = model.prepare_features(sample_market_data, sample_sentiment_data, is_inference=False)
        model.train(df)

        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': model.features,
            'importance': model.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Features by Importance:")
        print(importance_df.head(10).to_string())

        # Importances should sum to 1
        assert abs(importance_df['importance'].sum() - 1.0) < 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
