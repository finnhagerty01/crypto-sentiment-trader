# tests/test_technical.py
"""Tests for technical indicators module."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.technical import TechnicalIndicators, add_technical_indicators


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    np.random.seed(42)
    n = 100

    # Generate realistic-ish price data
    returns = np.random.randn(n) * 0.02
    close = 50000 * np.cumprod(1 + returns)

    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='1h'),
        'symbol': 'BTCUSDT',
        'open': close * (1 + np.random.randn(n) * 0.001),
        'high': close * (1 + abs(np.random.randn(n) * 0.01)),
        'low': close * (1 - abs(np.random.randn(n) * 0.01)),
        'close': close,
        'volume': np.random.uniform(100, 1000, n)
    })

    return df


@pytest.fixture
def multi_symbol_data():
    """Create sample data with multiple symbols."""
    np.random.seed(42)
    n = 50

    dfs = []
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        returns = np.random.randn(n) * 0.02
        close = 50000 * np.cumprod(1 + returns)

        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='1h'),
            'symbol': symbol,
            'open': close,
            'high': close * 1.01,
            'low': close * 0.99,
            'close': close,
            'volume': 100.0
        })
        dfs.append(df)

    return pd.concat(dfs)


class TestRSI:
    """Tests for RSI indicator."""

    def test_rsi_bounds(self, sample_market_data):
        """RSI should always be between 0 and 100."""
        ti = TechnicalIndicators()
        result = ti.add_all_indicators(sample_market_data)

        assert result['rsi_14'].dropna().between(0, 100).all()
        assert result['rsi_6'].dropna().between(0, 100).all()

    def test_rsi_columns_exist(self, sample_market_data):
        """Check all RSI-related columns are created."""
        ti = TechnicalIndicators()
        result = ti.add_all_indicators(sample_market_data)

        expected_cols = ['rsi_6', 'rsi_14', 'rsi_oversold', 'rsi_overbought', 'rsi_divergence']
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_rsi_signals_binary(self, sample_market_data):
        """RSI signals should be binary (0 or 1)."""
        ti = TechnicalIndicators()
        result = ti.add_all_indicators(sample_market_data)

        assert set(result['rsi_oversold'].unique()).issubset({0, 1})
        assert set(result['rsi_overbought'].unique()).issubset({0, 1})


class TestMACD:
    """Tests for MACD indicator."""

    def test_macd_columns_exist(self, sample_market_data):
        """Check all MACD-related columns are created."""
        ti = TechnicalIndicators()
        result = ti.add_all_indicators(sample_market_data)

        expected_cols = ['macd_line', 'macd_signal', 'macd_histogram',
                         'macd_crossover', 'macd_histogram_increasing']
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_macd_histogram_calculation(self, sample_market_data):
        """Histogram should equal MACD line minus signal line."""
        ti = TechnicalIndicators()
        result = ti.add_all_indicators(sample_market_data)

        calculated_histogram = result['macd_line'] - result['macd_signal']
        pd.testing.assert_series_equal(
            result['macd_histogram'],
            calculated_histogram,
            check_names=False
        )

    def test_macd_crossover_values(self, sample_market_data):
        """MACD crossover should only be -1, 0, or 1."""
        ti = TechnicalIndicators()
        result = ti.add_all_indicators(sample_market_data)

        assert set(result['macd_crossover'].unique()).issubset({-1, 0, 1})


class TestBollingerBands:
    """Tests for Bollinger Bands."""

    def test_bollinger_columns_exist(self, sample_market_data):
        """Check all Bollinger Band columns are created."""
        ti = TechnicalIndicators()
        result = ti.add_all_indicators(sample_market_data)

        expected_cols = ['bb_percent_b', 'bb_bandwidth', 'bb_squeeze',
                         'bb_above_upper', 'bb_below_lower']
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_bollinger_percent_b_range(self, sample_market_data):
        """Bollinger Band %B should be around 0-1 for normal conditions."""
        ti = TechnicalIndicators()
        result = ti.add_all_indicators(sample_market_data)

        bb = result['bb_percent_b'].dropna()
        # Most values should be between -0.5 and 1.5
        assert (bb.between(-0.5, 1.5)).mean() > 0.9


class TestATR:
    """Tests for ATR indicator."""

    def test_atr_positive(self, sample_market_data):
        """ATR should always be positive."""
        ti = TechnicalIndicators()
        result = ti.add_all_indicators(sample_market_data)

        assert (result['atr_14'].dropna() >= 0).all()
        assert (result['atr_7'].dropna() >= 0).all()

    def test_atr_pct_reasonable(self, sample_market_data):
        """ATR as % of price should be reasonable (< 50%)."""
        ti = TechnicalIndicators()
        result = ti.add_all_indicators(sample_market_data)

        assert (result['atr_14_pct'].dropna() < 0.5).all()


class TestMultiSymbol:
    """Tests for multi-symbol processing."""

    def test_all_symbols_processed(self, multi_symbol_data):
        """Test multi-symbol processing."""
        result = add_technical_indicators(multi_symbol_data)

        assert set(result['symbol'].unique()) == {'BTCUSDT', 'ETHUSDT'}
        assert len(result) == len(multi_symbol_data)

    def test_no_cross_contamination(self, multi_symbol_data):
        """Ensure symbols don't affect each other's indicators."""
        result = add_technical_indicators(multi_symbol_data)

        # Each symbol should have its own RSI calculation
        btc_rsi = result[result['symbol'] == 'BTCUSDT']['rsi_14'].dropna()
        eth_rsi = result[result['symbol'] == 'ETHUSDT']['rsi_14'].dropna()

        # They should have the same length (same number of rows per symbol)
        assert len(btc_rsi) == len(eth_rsi)


class TestNoLookahead:
    """Tests to ensure no lookahead bias."""

    def test_no_lookahead_bias(self, sample_market_data):
        """Ensure no future data leaks into current row."""
        ti = TechnicalIndicators()
        result = ti.add_all_indicators(sample_market_data)

        # First row should be NaN for return
        assert 'return_1h' in result.columns
        assert pd.isna(result['return_1h'].iloc[0])

    def test_returns_calculated_correctly(self, sample_market_data):
        """Verify returns use past data only."""
        ti = TechnicalIndicators()
        result = ti.add_all_indicators(sample_market_data)

        # Manual calculation of return at index 1
        expected_return = (result['close'].iloc[1] - result['close'].iloc[0]) / result['close'].iloc[0]
        actual_return = result['return_1h'].iloc[1]

        np.testing.assert_almost_equal(actual_return, expected_return, decimal=10)


class TestAllIndicators:
    """Tests for the full indicator set."""

    def test_all_indicator_columns(self, sample_market_data):
        """Verify all expected indicator columns are created."""
        ti = TechnicalIndicators()
        result = ti.add_all_indicators(sample_market_data)

        # Key indicators that should exist
        key_indicators = [
            # RSI
            'rsi_6', 'rsi_14', 'rsi_oversold', 'rsi_overbought', 'rsi_divergence',
            # MACD
            'macd_line', 'macd_signal', 'macd_histogram', 'macd_crossover',
            # Stochastic
            'stoch_k', 'stoch_d',
            # Bollinger
            'bb_percent_b', 'bb_bandwidth', 'bb_squeeze',
            # ATR
            'atr_7', 'atr_14', 'atr_14_pct',
            # ADX
            'adx', 'plus_di', 'minus_di', 'trend_strength',
            # Moving Averages
            'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26', 'ma_spread',
            # Volume
            'volume_ratio', 'volume_spike', 'obv', 'mfi',
            # Price Action
            'return_1h', 'return_4h', 'return_24h',
            'candle_body_pct', 'range_pct',
            'dist_from_24h_high', 'dist_from_24h_low'
        ]

        for col in key_indicators:
            assert col in result.columns, f"Missing indicator column: {col}"

    def test_no_all_nan_columns(self, sample_market_data):
        """No indicator column should be all NaN (after warmup period)."""
        ti = TechnicalIndicators()
        result = ti.add_all_indicators(sample_market_data)

        # Skip first 50 rows (warmup) and check no column is all NaN
        result_after_warmup = result.iloc[50:]

        for col in result.columns:
            if col not in ['timestamp', 'symbol']:
                assert not result_after_warmup[col].isna().all(), f"Column {col} is all NaN after warmup"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
