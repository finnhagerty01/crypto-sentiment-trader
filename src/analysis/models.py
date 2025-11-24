import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
import logging

logger = logging.getLogger(__name__)

class TradingModel:
    def __init__(self, enter_threshold: float = 0.52):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
        self.features = []
        self.enter_threshold = enter_threshold

    def prepare_features(self, market_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges Market + Sentiment and creates technical features.
        """
        # Defensive Check
        if 'timestamp' not in sentiment_df.columns:
            logger.error(f"Sentiment DataFrame missing 'timestamp' column. Columns found: {sentiment_df.columns}")
            return pd.DataFrame()

        # Ensure timestamps are aligned (floor to hour)
        market_df = market_df.copy()
        sentiment_df = sentiment_df.copy()
        
        market_df['timestamp'] = pd.to_datetime(market_df['timestamp']).dt.floor('h')
        sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp']).dt.tz_localize(None).dt.floor('h')

        # Merge
        df = pd.merge(market_df, sentiment_df, on=['timestamp', 'symbol'], how='left')
        
        # Fill missing sentiment
        df['sentiment_mean'] = df['sentiment_mean'].fillna(0)
        df['post_volume'] = df['post_volume'].fillna(0)
        df['comment_volume'] = df['comment_volume'].fillna(0)

        # --- FEATURE ENGINEERING ---
        
        # 1. Returns (Current Hour)
        # Calculate per symbol to avoid cross-contamination
        df['hourly_return'] = df.groupby('symbol')['close'].pct_change()
        
        # 2. Lags (Use PAST data to predict FUTURE)
        for lag in [1, 2, 3]:
            df[f'sentiment_lag_{lag}'] = df.groupby('symbol')['sentiment_mean'].shift(lag)
            df[f'return_lag_{lag}'] = df.groupby('symbol')['hourly_return'].shift(lag)
            df[f'volume_lag_{lag}'] = df.groupby('symbol')['post_volume'].shift(lag)
        
        # 3. Target (Next hour return)
        # We take the hourly_return we just calculated and shift it BACKWARDS by 1
        # Row T now contains the return that happened at T+1
        df['target_return'] = df.groupby('symbol')['hourly_return'].shift(-1)
        
        # 4. Label (1 if next hour return > 0.5%)
        df['target'] = (df['target_return'] > 0.005).astype(int)

        df = df.dropna()
        
        # Define features (exclude target, timestamp, symbol, and the raw forward-looking target_return)
        self.features = [c for c in df.columns if 'lag' in c]
        
        return df

    def train(self, df: pd.DataFrame):
        if df.empty:
            logger.warning("Training DataFrame is empty!")
            return None

        X = df[self.features]
        y = df['target']
        
        # Simple time-based split
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        if len(X_train) < 10:
            logger.warning("Not enough data to train.")
            return None

        self.model.fit(X_train, y_train)
        
        preds = self.model.predict(X_test)
        precision = precision_score(y_test, preds, zero_division=0)
        logger.info(f"Model Trained. Precision on Test Set: {precision:.2f}")
        
        return self.model

    def predict(self, recent_data: pd.DataFrame) -> dict:
        """
        Generates dictionary of {symbol: 'BUY'/'HOLD'}
        """
        if recent_data.empty: return {}
        
        # Ensure we have the same features as training
        missing_feats = [f for f in self.features if f not in recent_data.columns]
        if missing_feats:
            logger.warning(f"Prediction data missing features: {missing_feats}")
            return {}

        X = recent_data[self.features]
        probs = self.model.predict_proba(X)
        
        # Get probability of Class 1 (Buy)
        buy_probs = probs[:, 1]
        
        signals = {}
        for i, symbol in enumerate(recent_data['symbol']):
            prob = buy_probs[i]
            if prob > self.enter_threshold: 
                signals[symbol] = "BUY"
                logger.info(f"Buy Signal for {symbol} (Conf: {prob:.2f})")
            else:
                signals[symbol] = "HOLD"
        return signals