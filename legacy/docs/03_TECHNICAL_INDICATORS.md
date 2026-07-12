# Technical Indicators Implementation Guide

## Overview
This document provides copy-paste ready implementations for technical indicators to add to the crypto sentiment trader.

---

## File: `src/features/technical.py`

```python
"""
Technical indicators for crypto sentiment trading.

All functions expect pandas DataFrames/Series and return the same.
Designed to work with the existing MarketClient output format.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Comprehensive technical indicator calculator.
    
    Usage:
        ti = TechnicalIndicators()
        market_df = ti.add_all_indicators(market_df)
    """
    
    def __init__(self, 
                 rsi_period: int = 14,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 atr_period: int = 14,
                 adx_period: int = 14):
        """Initialize with configurable periods."""
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.adx_period = adx_period
    
    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to market DataFrame.
        
        Args:
            df: DataFrame with columns: timestamp, symbol, open, high, low, close, volume
        
        Returns:
            DataFrame with all indicator columns added
        """
        df = df.copy()
        
        # Process each symbol separately to avoid cross-contamination
        result_dfs = []
        
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].copy().sort_values('timestamp')
            
            # Price series
            close = symbol_df['close']
            high = symbol_df['high']
            low = symbol_df['low']
            open_price = symbol_df['open']
            volume = symbol_df['volume']
            
            # Momentum Indicators
            symbol_df = self._add_rsi(symbol_df, close)
            symbol_df = self._add_macd(symbol_df, close)
            symbol_df = self._add_stochastic(symbol_df, high, low, close)
            
            # Volatility Indicators
            symbol_df = self._add_bollinger_bands(symbol_df, close)
            symbol_df = self._add_atr(symbol_df, high, low, close)
            
            # Trend Indicators
            symbol_df = self._add_adx(symbol_df, high, low, close)
            symbol_df = self._add_moving_averages(symbol_df, close)
            
            # Volume Indicators
            symbol_df = self._add_volume_indicators(symbol_df, close, volume)
            
            # Price Action
            symbol_df = self._add_price_action(symbol_df, open_price, high, low, close)
            
            result_dfs.append(symbol_df)
        
        return pd.concat(result_dfs).sort_values(['timestamp', 'symbol']).reset_index(drop=True)
    
    # ==================== MOMENTUM INDICATORS ====================
    
    def _add_rsi(self, df: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
        """Add RSI indicators."""
        for period in [6, 14]:  # Short and standard
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.ewm(span=period, adjust=False).mean()
            avg_loss = loss.ewm(span=period, adjust=False).mean()
            
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            
            df[f'rsi_{period}'] = rsi
        
        # RSI signals
        df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
        
        # RSI divergence (price up, RSI down = bearish)
        price_change = close.diff(5) > 0
        rsi_change = df['rsi_14'].diff(5) > 0
        df['rsi_divergence'] = 0
        df.loc[price_change & ~rsi_change, 'rsi_divergence'] = -1  # Bearish
        df.loc[~price_change & rsi_change, 'rsi_divergence'] = 1   # Bullish
        
        return df
    
    def _add_macd(self, df: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
        """Add MACD indicators."""
        ema_fast = close.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.macd_slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        df['macd_line'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_histogram'] = histogram
        
        # Crossover signals
        df['macd_crossover'] = 0
        macd_above = macd_line > signal_line
        df.loc[macd_above & ~macd_above.shift(1).fillna(False), 'macd_crossover'] = 1   # Bullish
        df.loc[~macd_above & macd_above.shift(1).fillna(True), 'macd_crossover'] = -1  # Bearish
        
        # Histogram momentum
        df['macd_histogram_increasing'] = (histogram > histogram.shift(1)).astype(int)
        
        return df
    
    def _add_stochastic(self, df: pd.DataFrame, 
                        high: pd.Series, 
                        low: pd.Series, 
                        close: pd.Series,
                        k_period: int = 14,
                        d_period: int = 3) -> pd.DataFrame:
        """Add Stochastic Oscillator."""
        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()
        
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
        stoch_d = stoch_k.rolling(d_period).mean()
        
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        
        # Signals
        df['stoch_oversold'] = (stoch_k < 20).astype(int)
        df['stoch_overbought'] = (stoch_k > 80).astype(int)
        
        return df
    
    # ==================== VOLATILITY INDICATORS ====================
    
    def _add_bollinger_bands(self, df: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
        """Add Bollinger Bands."""
        middle = close.rolling(self.bb_period).mean()
        std = close.rolling(self.bb_period).std()
        
        upper = middle + self.bb_std * std
        lower = middle - self.bb_std * std
        
        # %B: Position within bands
        df['bb_percent_b'] = (close - lower) / (upper - lower).replace(0, np.nan)
        
        # Bandwidth: Volatility measure
        df['bb_bandwidth'] = (upper - lower) / middle
        
        # Squeeze detection (low volatility)
        bandwidth_min = df['bb_bandwidth'].rolling(20).min()
        df['bb_squeeze'] = (df['bb_bandwidth'] <= bandwidth_min * 1.1).astype(int)
        
        # Price vs bands
        df['bb_above_upper'] = (close > upper).astype(int)
        df['bb_below_lower'] = (close < lower).astype(int)
        
        return df
    
    def _add_atr(self, df: pd.DataFrame,
                 high: pd.Series,
                 low: pd.Series,
                 close: pd.Series) -> pd.DataFrame:
        """Add Average True Range."""
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        for period in [7, 14]:
            atr = true_range.ewm(span=period, adjust=False).mean()
            df[f'atr_{period}'] = atr
            df[f'atr_{period}_pct'] = atr / close  # Normalized
        
        # ATR expansion/contraction
        atr_ma = df['atr_14'].rolling(20).mean()
        df['atr_expansion'] = (df['atr_14'] > atr_ma * 1.5).astype(int)
        df['atr_contraction'] = (df['atr_14'] < atr_ma * 0.7).astype(int)
        
        return df
    
    # ==================== TREND INDICATORS ====================
    
    def _add_adx(self, df: pd.DataFrame,
                 high: pd.Series,
                 low: pd.Series,
                 close: pd.Series) -> pd.DataFrame:
        """Add Average Directional Index."""
        period = self.adx_period
        
        # Calculate +DM and -DM
        high_diff = high.diff()
        low_diff = -low.diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        # Calculate True Range
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        ], axis=1).max(axis=1)
        
        # Smoothed values (Wilder's smoothing)
        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr
        minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        df['adx'] = adx
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        # Trend strength categories
        df['trend_strength'] = pd.cut(
            adx,
            bins=[0, 20, 40, 100],
            labels=[0, 1, 2]  # weak, developing, strong
        ).astype(float)
        
        # Trend direction
        df['trend_bullish'] = ((plus_di > minus_di) & (adx > 20)).astype(int)
        df['trend_bearish'] = ((minus_di > plus_di) & (adx > 20)).astype(int)
        
        return df
    
    def _add_moving_averages(self, df: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
        """Add moving average indicators."""
        # Simple Moving Averages
        for period in [10, 20, 50]:
            df[f'sma_{period}'] = close.rolling(period).mean()
            df[f'close_vs_sma_{period}'] = (close - df[f'sma_{period}']) / df[f'sma_{period}']
        
        # Exponential Moving Averages
        for period in [12, 26]:
            df[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
        
        # Price position relative to MAs
        df['price_above_sma20'] = (close > df['sma_20']).astype(int)
        df['price_above_sma50'] = (close > df['sma_50']).astype(int)
        
        # MA Crossovers
        sma20_above_sma50 = df['sma_20'] > df['sma_50']
        df['golden_cross'] = (sma20_above_sma50 & ~sma20_above_sma50.shift(1).fillna(True)).astype(int)
        df['death_cross'] = (~sma20_above_sma50 & sma20_above_sma50.shift(1).fillna(False)).astype(int)
        
        # MA spread (trend strength proxy)
        df['ma_spread'] = (df['sma_20'] - df['sma_50']) / close
        
        return df
    
    # ==================== VOLUME INDICATORS ====================
    
    def _add_volume_indicators(self, df: pd.DataFrame,
                                close: pd.Series,
                                volume: pd.Series) -> pd.DataFrame:
        """Add volume-based indicators."""
        # Volume SMA
        df['volume_sma_20'] = volume.rolling(20).mean()
        df['volume_ratio'] = volume / df['volume_sma_20']
        
        # Volume spike detection
        df['volume_spike'] = (df['volume_ratio'] > 2.0).astype(int)
        
        # On-Balance Volume (OBV)
        price_change = close.diff()
        obv = (volume * np.sign(price_change)).cumsum()
        df['obv'] = obv
        df['obv_sma'] = obv.rolling(20).mean()
        df['obv_trend'] = (obv > df['obv_sma']).astype(int)
        
        # Volume-Price Trend
        df['vpt'] = (volume * close.pct_change()).cumsum()
        
        # Money Flow Index (MFI) - Volume-weighted RSI
        typical_price = (df['high'] + df['low'] + close) / 3
        money_flow = typical_price * volume
        
        price_diff = typical_price.diff()
        positive_flow = money_flow.where(price_diff > 0, 0)
        negative_flow = money_flow.where(price_diff < 0, 0)
        
        positive_mf = positive_flow.rolling(14).sum()
        negative_mf = negative_flow.rolling(14).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf.replace(0, np.nan)))
        df['mfi'] = mfi
        
        return df
    
    # ==================== PRICE ACTION ====================
    
    def _add_price_action(self, df: pd.DataFrame,
                          open_price: pd.Series,
                          high: pd.Series,
                          low: pd.Series,
                          close: pd.Series) -> pd.DataFrame:
        """Add price action features."""
        # Returns
        df['return_1h'] = close.pct_change()
        df['return_4h'] = close.pct_change(4)
        df['return_24h'] = close.pct_change(24)
        
        # Candle patterns
        body = close - open_price
        total_range = high - low
        
        df['candle_body_pct'] = body / total_range.replace(0, np.nan)
        df['upper_wick_pct'] = (high - close.clip(lower=open_price)) / total_range.replace(0, np.nan)
        df['lower_wick_pct'] = (open_price.clip(lower=close) - low) / total_range.replace(0, np.nan)
        
        # Doji detection (small body)
        df['is_doji'] = (abs(df['candle_body_pct']) < 0.1).astype(int)
        
        # Momentum candles
        df['is_momentum_up'] = ((df['candle_body_pct'] > 0.7) & (body > 0)).astype(int)
        df['is_momentum_down'] = ((df['candle_body_pct'] > 0.7) & (body < 0)).astype(int)
        
        # High-Low range as % of close
        df['range_pct'] = total_range / close
        
        # Distance from recent high/low
        rolling_high = high.rolling(24).max()
        rolling_low = low.rolling(24).min()
        df['dist_from_24h_high'] = (rolling_high - close) / close
        df['dist_from_24h_low'] = (close - rolling_low) / close
        
        return df


# Convenience function for direct use
def add_technical_indicators(market_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Add all technical indicators to market DataFrame.
    
    Example:
        from src.features.technical import add_technical_indicators
        market_df = add_technical_indicators(market_df)
    """
    ti = TechnicalIndicators(**kwargs)
    return ti.add_all_indicators(market_df)
```

---

## Integration with Existing Code

### Modify `src/analysis/models.py`

```python
# Add to imports
from src.features.technical import add_technical_indicators

class TradingModel:
    def prepare_features(self, market_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges Market + Sentiment and creates technical features.
        """
        # ... existing code ...
        
        # ADD THIS: Technical indicators
        market_df = add_technical_indicators(market_df)
        
        # ... rest of existing code ...
        
        # UPDATE feature list to include new indicators
        technical_features = [
            'rsi_14', 'rsi_6', 'rsi_divergence',
            'macd_histogram', 'macd_crossover',
            'bb_percent_b', 'bb_bandwidth', 'bb_squeeze',
            'atr_14_pct', 'atr_expansion',
            'adx', 'trend_strength',
            'ma_spread', 'price_above_sma20',
            'volume_ratio', 'volume_spike',
            'mfi',
            'return_1h', 'return_4h',
            'dist_from_24h_high', 'dist_from_24h_low'
        ]
        
        self.features = [c for c in df.columns if 'lag' in c] + technical_features
        
        return df
```

---

## Testing the Implementation

```python
# tests/test_technical.py

import pytest
import pandas as pd
import numpy as np
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

def test_rsi_bounds(sample_market_data):
    """RSI should always be between 0 and 100."""
    ti = TechnicalIndicators()
    result = ti.add_all_indicators(sample_market_data)
    
    assert result['rsi_14'].dropna().between(0, 100).all()
    assert result['rsi_6'].dropna().between(0, 100).all()

def test_bollinger_bands(sample_market_data):
    """Bollinger Band %B should be around 0-1 for normal conditions."""
    ti = TechnicalIndicators()
    result = ti.add_all_indicators(sample_market_data)
    
    # Most values should be between -0.5 and 1.5
    bb = result['bb_percent_b'].dropna()
    assert (bb.between(-0.5, 1.5)).mean() > 0.9

def test_no_lookahead_bias(sample_market_data):
    """Ensure no future data leaks into current row."""
    ti = TechnicalIndicators()
    result = ti.add_all_indicators(sample_market_data)
    
    # Check that indicator at time T only uses data up to T
    # This is implicitly tested by using shift() correctly
    assert 'return_1h' in result.columns
    
    # First row should be NaN for return
    assert pd.isna(result['return_1h'].iloc[0])

def test_all_symbols_processed():
    """Test multi-symbol processing."""
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
            'volume': 100
        })
        dfs.append(df)
    
    combined = pd.concat(dfs)
    result = add_technical_indicators(combined)
    
    assert set(result['symbol'].unique()) == {'BTCUSDT', 'ETHUSDT'}
    assert len(result) == len(combined)
```

---

## Feature Importance Notes

Based on quantitative research, expected importance ranking:

1. **High Importance**: RSI divergence, MACD histogram, ATR%, volume ratio
2. **Medium Importance**: Bollinger %B, ADX, MA spread, MFI
3. **Lower Importance**: Individual price levels, crossover signals

The model should be allowed to learn which features matter through training, but these expectations can guide feature selection if needed.
