# src/models/trading_model.py
"""
Production-ready trading model with ensemble, walk-forward validation,
feature importance tracking, AND hyperparameter tuning.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

from src.models.validation import WalkForwardValidator
from src.models.ensemble import TradingEnsemble
from src.features.technical import add_technical_indicators
from src.models.tuning import TradingModelTuner  # <--- NEW IMPORT

logger = logging.getLogger(__name__)


class ImprovedTradingModel:
    def __init__(
        self,
        enter_threshold: float = 0.55,
        exit_threshold: float = 0.45,
        min_confidence: float = 0.55,
    ):
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.min_confidence = min_confidence

        # Initialize with default params initially
        self.ensemble = TradingEnsemble(voting="soft", calibrate=True)
        
        self.validator = WalkForwardValidator(
            min_train_size=500,
            test_size=168,
            step_size=24,
        )

        self.features: List[str] = []
        self.feature_importances: Optional[pd.Series] = None
        self.validation_results: Optional[Dict] = None
        self.is_fitted = False
        
        # New: Store best params found via tuning
        self.best_params = {}

    def prepare_features(
        self,
        market_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        is_inference: bool = False,
    ) -> pd.DataFrame:
        # ... (Keep your existing prepare_features logic exactly as is) ...
        # [Copy the entire prepare_features method from your original file here]
        # For brevity in this response, I am assuming the existing logic resides here.
        if market_df is None or market_df.empty:
            logger.error("Market DataFrame is empty")
            return pd.DataFrame()

        if sentiment_df is None:
            sentiment_df = pd.DataFrame()

        market_df = market_df.copy()
        sentiment_df = sentiment_df.copy()

        # Align timestamps to hourly bins (UTC-naive)
        market_df["timestamp"] = pd.to_datetime(market_df["timestamp"]).dt.floor("h")
        market_df = add_technical_indicators(market_df)

        if not sentiment_df.empty:
            sentiment_df["timestamp"] = pd.to_datetime(sentiment_df["timestamp"]).dt.tz_localize(None).dt.floor("h")
            df = pd.merge(market_df, sentiment_df, on=["timestamp", "symbol"], how="left")
        else:
            df = market_df.copy()

        cols_to_fill_0 = ["sentiment_mean", "post_volume"]
        for c in cols_to_fill_0:
            if c in df.columns: df[c] = df[c].fillna(0.0)
            else: df[c] = 0.0

        if "hourly_return" not in df.columns:
            df["hourly_return"] = df.groupby("symbol")["close"].pct_change().fillna(0.0)

        # Lags
        lags = [1, 2, 3, 6, 12, 24, 36, 48]
        lag_cols = []
        for lag in lags:
            for base in ["sentiment_mean", "post_volume", "hourly_return"]:
                col_name = f"{base}_lag_{lag}"
                # Rename for consistency with your original code mapping
                if base == "sentiment_mean": col_name = f"sent_lag_{lag}"
                if base == "post_volume": col_name = f"vol_lag_{lag}"
                if base == "hourly_return": col_name = f"ret_lag_{lag}"
                
                df[col_name] = df.groupby("symbol")[base].shift(lag).fillna(0.0)
                lag_cols.append(col_name)

        # Feature Lists (Same as your original)
        technical_features = [
            "rsi_14", "rsi_6", "rsi_divergence", "macd_histogram", "macd_crossover",
            "bb_percent_b", "bb_bandwidth", "bb_squeeze", "atr_14_pct", "atr_expansion",
            "adx", "trend_strength", "ma_spread", "price_above_sma20",
            "volume_ratio", "volume_spike", "mfi", "return_1h", "return_4h",
            "dist_from_24h_high", "dist_from_24h_low",
        ]
        sentiment_features = [
            "sentiment_velocity", "sentiment_acceleration", "sentiment_macd", 
            "sentiment_rsi", "sentiment_reversal", "sentiment_std", "sentiment_consensus",
            "extreme_bullish_ratio", "extreme_bearish_ratio", "sentiment_engagement_weighted", 
            "sentiment_high_engagement", "total_engagement", "market_sentiment", 
            "sentiment_vs_market", "sentiment_z_score", "sentiment_regime_numeric",
        ]

        # Combine and Fill
        available_technical = [f for f in technical_features if f in df.columns]
        available_sentiment = [f for f in sentiment_features if f in df.columns]
        
        # If we have already selected "best features" via feature selection, use only those
        # Otherwise, use everything available
        if self.features and is_inference:
            # During inference, we must respect the trained feature set
            current_features = self.features
        else:
            current_features = lag_cols + available_technical + available_sentiment

        # Fill NaNs
        for col in available_technical:
             if col in df.columns: df[col] = df.groupby("symbol")[col].ffill().bfill()
        
        for col in lag_cols + available_sentiment:
            if col in df.columns: df[col] = df[col].fillna(0.0)

        if not is_inference:
            df["target_return"] = df.groupby("symbol")["hourly_return"].shift(-1)
            df["target"] = (df["target_return"] > 0.001).astype(int)
            df = df.dropna(subset=["target_return", "target", "close"])
        else:
            df = df.dropna(subset=["close"])
            
        # If we are in training mode, we update self.features here to capture everything initially
        if not is_inference and not self.is_fitted:
            self.features = current_features

        return df

    # ------------------------------------------------------------------ #
    #  NEW: Hyperparameter Tuning                                        #
    # ------------------------------------------------------------------ #

    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray):
        """Run Optuna tuning for XGBoost and LightGBM."""
        logger.info("Starting hyperparameter tuning (this may take a while)...")
        tuner = TradingModelTuner(n_trials=20, validation_metric="f1", timeout=600)
        
        # Tune XGBoost
        logger.info("Tuning XGBoost...")
        xgb_result = tuner.tune_xgboost(X, y, self.validator, feature_names=self.features)
        self.best_params['xgb'] = xgb_result['best_params']
        
        # Tune LightGBM
        logger.info("Tuning LightGBM...")
        lgb_result = tuner.tune_lightgbm(X, y, self.validator, feature_names=self.features)
        self.best_params['lgb'] = lgb_result['best_params']
        
        # Re-initialize ensemble with optimized params
        self.ensemble = TradingEnsemble(
            voting="soft", 
            calibrate=True,
            xgb_params=self.best_params['xgb'],
            lgb_params=self.best_params['lgb']
        )
        logger.info("Ensemble updated with optimized parameters.")

    # ------------------------------------------------------------------ #
    #  NEW: Feature Selection                                            #
    # ------------------------------------------------------------------ #

    def perform_feature_selection(self, top_n_percent: float = 0.5) -> List[str]:
        """Drop weak features to reduce noise."""
        if self.feature_importances is None:
            logger.warning("No feature importances found. Skipping selection.")
            return self.features

        # Identify top features
        n_keep = int(len(self.features) * top_n_percent)
        n_keep = max(n_keep, 10)  # Keep at least 10 features
        
        top_features = self.feature_importances.head(n_keep).index.tolist()
        dropped = set(self.features) - set(top_features)
        
        logger.info(f"FEATURE SELECTION: Keeping top {n_keep} features. Dropped {len(dropped)} noise features.")
        logger.debug(f"Dropped: {dropped}")
        
        return top_features

    # ------------------------------------------------------------------ #
    #  Updated Training Logic                                            #
    # ------------------------------------------------------------------ #

    def train(
        self,
        df: pd.DataFrame,
        tune: bool = False,
        feature_selection: bool = True,
        validate: bool = True,
        validation_symbol: str = "BTCUSDT",
    ) -> Dict:
        if df.empty or len(df) < 100:
            return {"status": "insufficient_data"}

        # --- FIX: FILTER FOR TUNING ---
        # Only tune on the validation symbol (e.g., BTC) to ensure 
        # the time-steps (24h step, 168h test) are accurate.
        if tune:
            logger.info(f"Filtering tuning data to {validation_symbol} to preserve time structure...")
            tune_df = df[df["symbol"] == validation_symbol].copy().sort_values("timestamp")
            
            if len(tune_df) > self.validator.min_train_size:
                X_tune = tune_df[self.features].values
                y_tune = tune_df["target"].values
                self.optimize_hyperparameters(X_tune, y_tune)
            else:
                logger.warning(f"Not enough data for {validation_symbol} to tune. Using defaults.")

        # --- NORMAL TRAINING FLOW ---
        # We still fit the final model on ALL data (all symbols) to capture general patterns
        X = df[self.features].values
        y = df["target"].values

        # 3. Initial Fit (to get feature importances)
        logger.info("Running initial fit for feature importance...")
        self.ensemble.fit(X, y, feature_names=self.features)
        self.feature_importances = self.ensemble.get_feature_importances(top_n=len(self.features))
        self.is_fitted = True

        # 4. Feature Selection
        if feature_selection:
            logger.info("Refining feature set to remove noise...")
            refined_features = self.perform_feature_selection(top_n_percent=0.5)
            self.features = refined_features
            
            # Re-select X based on new feature list
            X = df[self.features].values
            
            # Final Fit
            logger.info(f"Retraining on {len(self.features)} best features...")
            self.ensemble.fit(X, y, feature_names=self.features)
            self.feature_importances = self.ensemble.get_feature_importances()

        # 5. Validation
        if validate:
            val_df = df[df["symbol"] == validation_symbol].copy().sort_values("timestamp")
            if len(val_df) > self.validator.min_train_size:
                self.validation_results = self.validator.validate(
                    self.ensemble,
                    val_df[self.features + ["target"]],
                    self.features,
                )
                metrics = self.validation_results["overall"]
                logger.info(f"Final Validation: Precision={metrics['precision']:.2f}, F1={metrics['f1']:.2f}")

        return {
            "status": "success",
            "n_features": len(self.features),
            "top_features": self.feature_importances.head(10).to_dict(),
            "best_params": self.best_params if tune else "default"
        }

    # ... (Keep predict() and get_model_summary() as is) ...
    def predict(self, recent_df: pd.DataFrame) -> Dict[str, Dict]:
        # Use existing predict logic
        if not self.is_fitted: return {}
        if recent_df.empty: return {}
        
        # Ensure we only use the selected features
        missing = [f for f in self.features if f not in recent_df.columns]
        if missing:
            # Handle missing by filling 0 or warning
            for m in missing: recent_df[m] = 0.0
            
        X = recent_df[self.features].values
        probas = self.ensemble.predict_proba(X)
        buy_probs = probas[:, 1]
        if len(buy_probs) > 0:
            logger.info(f"Max Buy Prob: {buy_probs.max():.4f} (Threshold: {self.enter_threshold})")
            
        signals = {}
        for i, (_, row) in enumerate(recent_df.iterrows()):
            symbol = row["symbol"]
            prob = buy_probs[i]
            
            # Logic same as your original file
            if prob >= self.enter_threshold:
                action, conf = "BUY", prob
            elif prob <= self.exit_threshold:
                action, conf = "SELL", 1 - prob
            else:
                action, conf = "HOLD", max(prob, 1-prob)

            if conf < self.min_confidence: action = "HOLD"

            signals[symbol] = {
                "action": action, "confidence": float(conf),
                "buy_probability": float(prob), "features_used": len(self.features)
            }
        return signals

    def get_model_summary(self) -> Dict:
        return {
            "is_fitted": self.is_fitted,
            "n_features": len(self.features),
            "validation": self.validation_results.get("overall") if self.validation_results else None
        }