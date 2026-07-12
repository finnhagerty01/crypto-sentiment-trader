# Feature Engineering Specifications

## Overview
This document specifies 50+ features to add to the crypto sentiment trader. Features are organized by category with implementation details and expected predictive value.

---

## Category 1: Technical Indicators (15 Features)

### 1.1 Momentum Indicators

#### RSI (Relative Strength Index)
```python
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI = 100 - (100 / (1 + RS))
    where RS = avg gain / avg loss over period
    
    Trading signals:
    - RSI < 30: Oversold (potential buy)
    - RSI > 70: Overbought (potential sell)
    - Divergences: Price makes new high, RSI doesn't = bearish
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Features to create:
# - rsi_14: Standard 14-period RSI
# - rsi_6: Short-term RSI (more sensitive)
# - rsi_divergence: 1 if price up but RSI down (bearish), -1 if opposite
```

#### MACD (Moving Average Convergence Divergence)
```python
def calculate_macd(prices: pd.Series, 
                   fast: int = 12, 
                   slow: int = 26, 
                   signal: int = 9) -> tuple:
    """
    MACD Line = EMA(12) - EMA(26)
    Signal Line = EMA(9) of MACD Line
    Histogram = MACD Line - Signal Line
    
    Trading signals:
    - MACD crosses above signal: Bullish
    - MACD crosses below signal: Bearish
    - Histogram expansion: Trend strengthening
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

# Features to create:
# - macd_line: Raw MACD value
# - macd_signal: Signal line
# - macd_histogram: Histogram (momentum)
# - macd_crossover: 1 if bullish cross, -1 if bearish, 0 otherwise
```

#### Stochastic Oscillator
```python
def calculate_stochastic(high: pd.Series, 
                         low: pd.Series, 
                         close: pd.Series,
                         k_period: int = 14,
                         d_period: int = 3) -> tuple:
    """
    %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
    %D = SMA of %K
    
    Signals:
    - %K < 20: Oversold
    - %K > 80: Overbought
    - %K crosses %D: Signal
    """
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(d_period).mean()
    
    return k, d

# Features:
# - stoch_k, stoch_d
# - stoch_oversold: 1 if K < 20
# - stoch_overbought: 1 if K > 80
```

### 1.2 Volatility Indicators

#### Bollinger Bands
```python
def calculate_bollinger_bands(prices: pd.Series, 
                               period: int = 20,
                               std_dev: float = 2.0) -> tuple:
    """
    Middle Band = SMA(20)
    Upper Band = SMA + 2 * std
    Lower Band = SMA - 2 * std
    
    Features:
    - %B = (Price - Lower) / (Upper - Lower)
    - Bandwidth = (Upper - Lower) / Middle
    """
    middle = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    
    percent_b = (prices - lower) / (upper - lower)
    bandwidth = (upper - lower) / middle
    
    return middle, upper, lower, percent_b, bandwidth

# Features:
# - bb_percent_b: Position within bands (0-1 normal, >1 above, <0 below)
# - bb_bandwidth: Volatility measure (squeeze = low bandwidth)
# - bb_squeeze: 1 if bandwidth < 20-day min bandwidth
```

#### ATR (Average True Range)
```python
def calculate_atr(high: pd.Series, 
                  low: pd.Series, 
                  close: pd.Series,
                  period: int = 14) -> pd.Series:
    """
    True Range = max(H-L, |H-Prev Close|, |L-Prev Close|)
    ATR = EMA of True Range
    
    Uses:
    - Stop loss placement (e.g., 2 * ATR)
    - Position sizing (inverse relationship)
    - Volatility regime detection
    """
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(span=period, adjust=False).mean()
    
    return atr

# Features:
# - atr_14: Standard ATR
# - atr_percent: ATR / Close (normalized)
# - atr_expansion: 1 if ATR > 1.5 * 20-day avg ATR
```

### 1.3 Trend Indicators

#### ADX (Average Directional Index)
```python
def calculate_adx(high: pd.Series, 
                  low: pd.Series, 
                  close: pd.Series,
                  period: int = 14) -> tuple:
    """
    Measures trend strength (not direction)
    
    ADX < 20: Weak trend (range-bound)
    ADX 20-40: Developing trend
    ADX > 40: Strong trend
    
    +DI above -DI: Uptrend
    -DI above +DI: Downtrend
    """
    # Implementation involves +DM, -DM, smoothed values
    # ... (full implementation in technical.py)
    pass

# Features:
# - adx: Trend strength
# - plus_di, minus_di: Directional indicators
# - trend_strength: Categorical (weak/medium/strong)
```

#### Moving Average Crossovers
```python
# Features:
# - sma_20, sma_50: Simple moving averages
# - ema_12, ema_26: Exponential moving averages
# - price_above_sma20: Binary
# - golden_cross: 1 if SMA20 crosses above SMA50
# - death_cross: 1 if SMA20 crosses below SMA50
# - ma_spread: (SMA20 - SMA50) / Close (trend strength)
```

---

## Category 2: Enhanced Sentiment Features (12 Features)

### 2.1 Sentiment Dynamics

```python
# Current: Only sentiment_mean per hour
# Add these derived features:

def calculate_sentiment_features(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced sentiment features beyond simple mean.
    """
    features = pd.DataFrame()
    
    # 1. Sentiment Velocity (rate of change)
    features['sentiment_velocity'] = sentiment_df['sentiment_mean'].diff()
    features['sentiment_acceleration'] = features['sentiment_velocity'].diff()
    
    # 2. Sentiment Dispersion (disagreement)
    # Requires storing individual post scores, not just mean
    features['sentiment_std'] = sentiment_df['sentiment_std']  # Need to add to analyzer
    features['sentiment_range'] = sentiment_df['sentiment_max'] - sentiment_df['sentiment_min']
    
    # 3. Sentiment Extremes
    features['sentiment_skew'] = sentiment_df['sentiment_skew']  # Asymmetry
    features['extreme_bullish_ratio'] = sentiment_df['posts_above_0.5'] / sentiment_df['post_volume']
    features['extreme_bearish_ratio'] = sentiment_df['posts_below_-0.5'] / sentiment_df['post_volume']
    
    # 4. Sentiment Momentum (multi-period)
    for window in [3, 6, 12, 24]:
        features[f'sentiment_sma_{window}'] = sentiment_df['sentiment_mean'].rolling(window).mean()
        features[f'sentiment_momentum_{window}'] = (
            sentiment_df['sentiment_mean'] - features[f'sentiment_sma_{window}']
        )
    
    # 5. Sentiment Regime
    features['sentiment_regime'] = pd.cut(
        sentiment_df['sentiment_mean'],
        bins=[-1, -0.3, -0.1, 0.1, 0.3, 1],
        labels=['very_bearish', 'bearish', 'neutral', 'bullish', 'very_bullish']
    )
    
    return features
```

### 2.2 Engagement-Weighted Sentiment

```python
def calculate_engagement_weighted_sentiment(posts_df: pd.DataFrame) -> pd.Series:
    """
    Weight sentiment by engagement metrics.
    
    Hypothesis: Posts with more engagement have more market impact.
    """
    # Engagement score
    posts_df['engagement'] = (
        posts_df['score'] + 
        posts_df['num_comments'] * 2 +  # Comments weighted more
        posts_df.get('awards', 0) * 5    # Awards = high quality signal
    )
    
    # Normalize engagement within time window
    posts_df['engagement_weight'] = (
        posts_df['engagement'] / 
        posts_df.groupby('hour')['engagement'].transform('sum')
    )
    
    # Weighted sentiment
    posts_df['weighted_sentiment'] = (
        posts_df['sentiment_score'] * posts_df['engagement_weight']
    )
    
    return posts_df.groupby('hour')['weighted_sentiment'].sum()

# Features:
# - sentiment_engagement_weighted: Engagement-weighted mean
# - high_engagement_sentiment: Sentiment of top 10% engaged posts only
# - engagement_sentiment_divergence: Difference between weighted and unweighted
```

### 2.3 Source-Specific Sentiment

```python
# Different subreddits may have different signal quality

def calculate_subreddit_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-subreddit sentiment tracking.
    
    Hypothesis: r/Bitcoin sentiment leads BTC price, 
                r/CryptoCurrency is more diversified
    """
    features = pd.DataFrame()
    
    subreddit_groups = df.groupby(['hour', 'subreddit'])['sentiment_score']
    
    for sub in ['Bitcoin', 'Ethereum', 'CryptoCurrency', 'CryptoMarkets']:
        sub_sentiment = subreddit_groups.get_group(sub) if sub in subreddit_groups.groups else 0
        features[f'sentiment_{sub.lower()}'] = sub_sentiment.mean()
    
    # Cross-subreddit agreement
    features['subreddit_sentiment_std'] = subreddit_groups.mean().groupby('hour').std()
    
    return features
```

---

## Category 3: Volatility Features (8 Features)

```python
def calculate_volatility_features(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Comprehensive volatility feature set.
    """
    features = pd.DataFrame()
    
    # 1. Realized Volatility (multiple windows)
    for window in [6, 12, 24, 72]:  # 6h, 12h, 1d, 3d
        features[f'realized_vol_{window}h'] = (
            market_df['hourly_return'].rolling(window).std() * np.sqrt(24 * 365)  # Annualized
        )
    
    # 2. Volatility Ratio (short vs long term)
    features['vol_ratio'] = features['realized_vol_6h'] / features['realized_vol_72h']
    # > 1 means increasing volatility, < 1 means decreasing
    
    # 3. Parkinson Volatility (uses high-low, more efficient)
    features['parkinson_vol'] = np.sqrt(
        (1 / (4 * np.log(2))) * 
        (np.log(market_df['high'] / market_df['low']) ** 2).rolling(24).mean()
    ) * np.sqrt(24 * 365)
    
    # 4. Garman-Klass Volatility (uses OHLC, most efficient)
    features['gk_vol'] = np.sqrt(
        0.5 * (np.log(market_df['high'] / market_df['low']) ** 2) -
        (2 * np.log(2) - 1) * (np.log(market_df['close'] / market_df['open']) ** 2)
    ).rolling(24).mean() * np.sqrt(24 * 365)
    
    # 5. Volatility Regime Detection
    vol_percentile = features['realized_vol_24h'].rolling(168).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    )
    features['vol_regime'] = pd.cut(
        vol_percentile,
        bins=[0, 0.25, 0.75, 1.0],
        labels=['low', 'normal', 'high']
    )
    
    # 6. Intraday Volatility Pattern
    features['hour_of_day'] = market_df['timestamp'].dt.hour
    avg_vol_by_hour = features.groupby('hour_of_day')['realized_vol_6h'].transform('mean')
    features['vol_vs_typical'] = features['realized_vol_6h'] / avg_vol_by_hour
    
    return features
```

---

## Category 4: Time-Based Features (6 Features)

```python
def calculate_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crypto markets have distinct temporal patterns.
    """
    features = pd.DataFrame()
    
    # 1. Hour of day (cyclical encoding)
    hour = df['timestamp'].dt.hour
    features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    
    # 2. Day of week (cyclical encoding)
    dow = df['timestamp'].dt.dayofweek
    features['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    features['dow_cos'] = np.cos(2 * np.pi * dow / 7)
    
    # 3. Trading Sessions
    # UTC times for major sessions
    features['session'] = pd.cut(
        hour,
        bins=[-1, 6, 14, 22, 24],
        labels=['asia', 'europe', 'americas', 'asia_late']
    )
    
    # 4. Weekend Effect
    features['is_weekend'] = (dow >= 5).astype(int)
    
    # 5. Month-end Effect (rebalancing)
    features['days_to_month_end'] = (
        df['timestamp'].dt.days_in_month - df['timestamp'].dt.day
    )
    features['is_month_end'] = (features['days_to_month_end'] <= 2).astype(int)
    
    return features
```

---

## Category 5: Cross-Asset Features (6 Features)

```python
def calculate_cross_asset_features(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-asset relationships and relative performance.
    """
    features = pd.DataFrame()
    
    # Pivot to get all symbols as columns
    returns = market_df.pivot(
        index='timestamp', 
        columns='symbol', 
        values='hourly_return'
    )
    
    # 1. BTC Dominance Proxy (BTC return vs others)
    btc_return = returns.get('BTCUSDT', 0)
    alt_return = returns.drop('BTCUSDT', axis=1, errors='ignore').mean(axis=1)
    features['btc_vs_alts'] = btc_return - alt_return
    
    # 2. Rolling Correlation with BTC
    for symbol in ['ETHUSDT', 'SOLUSDT', 'BNBUSDT']:
        if symbol in returns.columns:
            features[f'{symbol}_btc_corr'] = (
                returns[symbol].rolling(24).corr(btc_return)
            )
    
    # 3. Sector Momentum (average of related coins)
    # Define sectors
    defi_coins = ['ETHUSDT', 'LINKUSDT', 'AVAXUSDT']
    meme_coins = ['DOGEUSDT']
    
    for name, coins in [('defi', defi_coins), ('meme', meme_coins)]:
        available = [c for c in coins if c in returns.columns]
        if available:
            features[f'{name}_sector_momentum'] = returns[available].mean(axis=1).rolling(6).mean()
    
    # 4. Market-Wide Momentum
    features['market_momentum'] = returns.mean(axis=1).rolling(6).mean()
    features['market_breadth'] = (returns > 0).mean(axis=1)  # % of coins positive
    
    return features
```

---

## Feature Summary Table

| Category | Feature Name | Description | Expected Value |
|----------|--------------|-------------|----------------|
| Technical | rsi_14 | 14-period RSI | High (classic indicator) |
| Technical | macd_histogram | MACD momentum | Medium-High |
| Technical | bb_percent_b | Bollinger Band position | Medium |
| Technical | atr_percent | Normalized volatility | High (risk-adjusted) |
| Sentiment | sentiment_velocity | Rate of sentiment change | High (leading indicator) |
| Sentiment | sentiment_std | Disagreement measure | Medium |
| Sentiment | sentiment_engagement_weighted | Engagement-weighted | High |
| Volatility | realized_vol_24h | 24h realized volatility | High (regime) |
| Volatility | vol_ratio | Short/long vol ratio | Medium |
| Time | hour_sin, hour_cos | Cyclical hour | Medium |
| Cross-Asset | btc_vs_alts | BTC relative strength | Medium-High |
| Cross-Asset | market_breadth | % coins positive | Medium |

---

## Implementation Priority

### Week 1: Core Technical + Volatility
1. RSI (14, 6 period)
2. MACD (line, signal, histogram)
3. Bollinger Bands (percent_b, bandwidth)
4. ATR (14 period, percent)
5. Realized volatility (6h, 24h, 72h)

### Week 2: Enhanced Sentiment
1. Sentiment velocity and acceleration
2. Sentiment standard deviation
3. Engagement-weighted sentiment
4. Extreme ratio (bullish/bearish)

### Week 3: Time + Cross-Asset
1. Hour/day cyclical encoding
2. Trading session features
3. BTC correlation features
4. Market breadth

### Week 4: Integration + Tuning
1. Feature selection (remove low-importance)
2. Feature normalization
3. Hyperparameter tuning
4. Walk-forward validation
