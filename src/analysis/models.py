import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import logging

from src.features.technical import add_technical_indicators

logger = logging.getLogger(__name__)

class TradingModel:
    def __init__(self, enter_threshold: float = 0.60, sell_threshold: float = 0.60, use_proba: bool = False):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            min_samples_leaf=10,
            min_samples_split=2,
            max_features=0.5,
            bootstrap=True,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
        self.features = []
        self.enter_threshold = enter_threshold
        self.sell_threshold = sell_threshold
        self.use_proba = use_proba

    def prepare_features(self, market_df: pd.DataFrame, sentiment_df: pd.DataFrame, is_inference: bool = False) -> pd.DataFrame:
        if market_df is None or market_df.empty:
            logger.error("Market DataFrame is empty")
            return pd.DataFrame()

        if sentiment_df is None:
            sentiment_df = pd.DataFrame()

        market_df = market_df.copy()
        sentiment_df = sentiment_df.copy()

        # Align timestamps to hourly bins (UTC-naive)
        market_df['timestamp'] = pd.to_datetime(market_df['timestamp']).dt.floor('h')

        # Add technical indicators BEFORE merging with sentiment
        logger.info("Adding technical indicators...")
        market_df = add_technical_indicators(market_df)

        if not sentiment_df.empty:
            if 'timestamp' not in sentiment_df.columns:
                logger.error("Sentiment DataFrame missing 'timestamp'")
                return pd.DataFrame()
            sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp']).dt.tz_localize(None).dt.floor('h')
            # LEFT JOIN: keep market hours even if no reddit posts
            df = pd.merge(market_df, sentiment_df, on=['timestamp', 'symbol'], how='left')
        else:
            # No sentiment data - just use market data
            df = market_df.copy()

        # Zero-fill reddit features (quiet hour is valid state)
        if 'sentiment_mean' not in df.columns:
            df['sentiment_mean'] = 0.0
        else:
            df['sentiment_mean'] = df['sentiment_mean'].fillna(0.0)

        if 'post_volume' not in df.columns:
            df['post_volume'] = 0.0
        else:
            df['post_volume'] = df['post_volume'].fillna(0.0)

        # Price return (may already exist from technical indicators as return_1h)
        if 'hourly_return' not in df.columns:
            df['hourly_return'] = df.groupby('symbol')['close'].pct_change()

        # Multi-lag features for sentiment (deterministic feature list)
        lags = [1, 2, 3, 6, 12, 24, 36, 48]
        lag_cols = []
        for lag in lags:
            s = f'sent_lag_{lag}'
            v = f'vol_lag_{lag}'
            r = f'ret_lag_{lag}'

            df[s] = df.groupby('symbol')['sentiment_mean'].shift(lag)
            df[v] = df.groupby('symbol')['post_volume'].shift(lag)
            df[r] = df.groupby('symbol')['hourly_return'].shift(lag)

            lag_cols.extend([s, v, r])

        # Technical indicator features to use in model
        technical_features = [
            # RSI
            'rsi_14', 'rsi_6', 'rsi_divergence',
            # MACD
            'macd_histogram', 'macd_crossover',
            # Bollinger Bands
            'bb_percent_b', 'bb_bandwidth', 'bb_squeeze',
            # ATR
            'atr_14_pct', 'atr_expansion',
            # ADX / Trend
            'adx', 'trend_strength',
            # Moving Averages
            'ma_spread', 'price_above_sma20',
            # Volume
            'volume_ratio', 'volume_spike',
            'mfi',
            # Price Action
            'return_1h', 'return_4h',
            'dist_from_24h_high', 'dist_from_24h_low'
        ]

        # Only include technical features that exist in the DataFrame
        available_technical = [f for f in technical_features if f in df.columns]

        self.features = lag_cols + available_technical
        logger.info(f"Using {len(self.features)} features: {len(lag_cols)} lag + {len(available_technical)} technical")

        if not is_inference:
            # Training targets
            df['target_return'] = df.groupby('symbol')['hourly_return'].shift(-1)

            conditions = [
                (df['target_return'] > 0.005),
                (df['target_return'] < -0.005)
            ]
            df['target'] = np.select(conditions, [1, -1], default=0)

            # Drop rows that are not fully defined
            df = df.dropna(subset=self.features + ['target_return', 'target'])
        else:
            # Inference: only drop rows missing lag history (NOT because volume==0)
            df = df.dropna(subset=self.features)

        return df

    def train(self, df: pd.DataFrame):
        if df.empty: return None
        self.model.fit(df[self.features], df['target'])
        logger.info(f"Model Trained on {len(df)} rows using {len(self.features)} features.")
        return self.model

    def predict(self, recent_data: pd.DataFrame) -> dict:
        """
        Uses ArgMax Logic (Winner Takes All).
        """
        if recent_data.empty: return {}
        
        missing = [f for f in self.features if f not in recent_data.columns]
        if missing:
            logger.warning(f"Missing features: {missing}")
            return {}

        X = recent_data[self.features]
        preds = self.model.predict(X)
        
        signals = {}
        for i, symbol in enumerate(recent_data['symbol']):
            prediction = preds[i]
            if prediction == 1:
                signals[symbol] = "BUY"
            elif prediction == -1:
                signals[symbol] = "SELL"
            else:
                signals[symbol] = "HOLD"
                
        return signals