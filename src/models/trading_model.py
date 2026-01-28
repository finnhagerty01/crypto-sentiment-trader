# src/models/trading_model.py
"""
Production-ready trading model with ensemble, walk-forward validation,
and feature importance tracking.

This is a drop-in replacement for ``TradingModel`` in
``src/analysis/models.py``.  It uses binary classification (0/1) with
probability thresholds to produce BUY/SELL/HOLD signals with confidence
scores, rather than the ternary (-1/0/1) scheme in the original model.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

from src.models.validation import WalkForwardValidator
from src.models.ensemble import TradingEnsemble
from src.features.technical import add_technical_indicators

logger = logging.getLogger(__name__)


class ImprovedTradingModel:
    """
    Production-ready trading model with:

    - Ensemble of RF + XGBoost + LightGBM + Logistic Regression
    - Walk-forward validation
    - Feature importance tracking
    - Threshold-based BUY/SELL/HOLD signals with confidence scores

    Output format (different from old ``TradingModel``)::

        {
            "BTCUSDT": {
                "action": "BUY",
                "confidence": 0.72,
                "buy_probability": 0.72,
                "features_used": 45,
            },
            ...
        }

    Typical usage::

        model = ImprovedTradingModel()
        df = model.prepare_features(market_df, sentiment_df)
        result = model.train(df)
        signals = model.predict(latest_df)
    """

    def __init__(
        self,
        enter_threshold: float = 0.55,
        exit_threshold: float = 0.45,
        min_confidence: float = 0.6,
    ):
        """
        Args:
            enter_threshold: Buy probability >= this triggers BUY.
            exit_threshold: Buy probability <= this triggers SELL.
            min_confidence: Minimum confidence for any non-HOLD action.
        """
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.min_confidence = min_confidence

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

    # ------------------------------------------------------------------ #
    #  Feature preparation                                                #
    # ------------------------------------------------------------------ #

    def prepare_features(
        self,
        market_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        is_inference: bool = False,
    ) -> pd.DataFrame:
        """
        Prepare all features for training or prediction.

        Replicates the feature logic from ``src/analysis/models.py`` but
        produces a **binary** target (0/1) instead of ternary (-1/0/1).

        Data leakage prevention:
        - Technical indicators use only past data (rolling windows, shifts).
        - Lags use ``.shift(lag)`` which only looks backward.
        - Target uses ``.shift(-1)`` (next-hour return) — only created when
          ``is_inference=False``.
        - ``'target'`` and ``'target_return'`` are excluded from features.

        Args:
            market_df: OHLCV data with columns
                ``[timestamp, symbol, open, high, low, close, volume]``.
            sentiment_df: Hourly sentiment data with columns
                ``[timestamp, symbol, sentiment_mean, post_volume, ...]``.
            is_inference: If True, skip target creation and don't drop
                rows missing the target.

        Returns:
            DataFrame ready for ``train()`` or ``predict()``.
        """
        if market_df is None or market_df.empty:
            logger.error("Market DataFrame is empty")
            return pd.DataFrame()

        if sentiment_df is None:
            sentiment_df = pd.DataFrame()

        market_df = market_df.copy()
        sentiment_df = sentiment_df.copy()

        # Align timestamps to hourly bins (UTC-naive)
        market_df["timestamp"] = pd.to_datetime(
            market_df["timestamp"]
        ).dt.floor("h")

        # Add technical indicators BEFORE merging with sentiment
        logger.info("Adding technical indicators...")
        market_df = add_technical_indicators(market_df)

        # Merge with sentiment
        if not sentiment_df.empty:
            if "timestamp" not in sentiment_df.columns:
                logger.error("Sentiment DataFrame missing 'timestamp'")
                return pd.DataFrame()
            sentiment_df["timestamp"] = (
                pd.to_datetime(sentiment_df["timestamp"])
                .dt.tz_localize(None)
                .dt.floor("h")
            )
            df = pd.merge(
                market_df,
                sentiment_df,
                on=["timestamp", "symbol"],
                how="left",
            )
        else:
            df = market_df.copy()

        # Zero-fill sentiment (quiet hour is a valid state)
        if "sentiment_mean" not in df.columns:
            df["sentiment_mean"] = 0.0
        else:
            df["sentiment_mean"] = df["sentiment_mean"].fillna(0.0)

        if "post_volume" not in df.columns:
            df["post_volume"] = 0.0
        else:
            df["post_volume"] = df["post_volume"].fillna(0.0)

        # Price return
        if "hourly_return" not in df.columns:
            df["hourly_return"] = df.groupby("symbol")["close"].pct_change()

        # Multi-lag features (SAME 8 lag periods as current model)
        lags = [1, 2, 3, 6, 12, 24, 36, 48]
        lag_cols: List[str] = []
        for lag in lags:
            s = f"sent_lag_{lag}"
            v = f"vol_lag_{lag}"
            r = f"ret_lag_{lag}"
            df[s] = df.groupby("symbol")["sentiment_mean"].shift(lag)
            df[v] = df.groupby("symbol")["post_volume"].shift(lag)
            df[r] = df.groupby("symbol")["hourly_return"].shift(lag)
            lag_cols.extend([s, v, r])

        # Technical feature names (SAME as current model)
        technical_features = [
            "rsi_14", "rsi_6", "rsi_divergence",
            "macd_histogram", "macd_crossover",
            "bb_percent_b", "bb_bandwidth", "bb_squeeze",
            "atr_14_pct", "atr_expansion",
            "adx", "trend_strength",
            "ma_spread", "price_above_sma20",
            "volume_ratio", "volume_spike", "mfi",
            "return_1h", "return_4h",
            "dist_from_24h_high", "dist_from_24h_low",
        ]

        # Advanced sentiment features (SAME as current model)
        sentiment_features = [
            "sentiment_velocity", "sentiment_acceleration",
            "sentiment_macd", "sentiment_rsi", "sentiment_reversal",
            "sentiment_std", "sentiment_consensus",
            "extreme_bullish_ratio", "extreme_bearish_ratio",
            "sentiment_engagement_weighted", "sentiment_high_engagement",
            "total_engagement",
            "market_sentiment", "sentiment_vs_market", "sentiment_z_score",
            "sentiment_regime_numeric",
        ]

        available_technical = [f for f in technical_features if f in df.columns]
        available_sentiment = [f for f in sentiment_features if f in df.columns]

        self.features = lag_cols + available_technical + available_sentiment
        logger.info(
            "Using %d features: %d lag + %d technical + %d sentiment",
            len(self.features),
            len(lag_cols),
            len(available_technical),
            len(available_sentiment),
        )

        if not is_inference:
            # BINARY target: >0.5% next-hour return = 1, else 0
            df["target_return"] = df.groupby("symbol")[
                "hourly_return"
            ].shift(-1)
            df["target"] = (df["target_return"] > 0.005).astype(int)
            df = df.dropna(subset=self.features + ["target_return", "target"])
        else:
            df = df.dropna(subset=self.features)

        return df

    # ------------------------------------------------------------------ #
    #  Training                                                           #
    # ------------------------------------------------------------------ #

    def train(
        self,
        df: pd.DataFrame,
        validate: bool = True,
    ) -> Dict:
        """
        Train the ensemble model.

        Args:
            df: Output of ``prepare_features(is_inference=False)``.
            validate: If True, run walk-forward validation first.

        Returns:
            Dictionary with training results including status, sample
            counts, validation metrics, and top feature importances.
        """
        if df.empty or len(df) < 100:
            logger.warning("Insufficient data for training")
            return {"status": "insufficient_data"}

        X = df[self.features].values
        y = df["target"].values

        logger.info("Training on %d samples", len(X))
        logger.info(
            "Class distribution: %d neg, %d pos",
            (y == 0).sum(),
            (y == 1).sum(),
        )

        # Walk-forward validation
        if validate:
            logger.info("Running walk-forward validation...")
            self.validation_results = self.validator.validate(
                self.ensemble,
                df[self.features + ["target"]],
                self.features,
            )
            logger.info(
                "Validation: precision=%.3f, recall=%.3f, f1=%.3f",
                self.validation_results["overall"]["precision"],
                self.validation_results["overall"]["recall"],
                self.validation_results["overall"]["f1"],
            )

        # Final fit on all data
        self.ensemble.fit(X, y, feature_names=self.features)
        self.feature_importances = self.ensemble.get_feature_importances()
        self.is_fitted = True

        return {
            "status": "success",
            "n_samples": len(X),
            "n_features": len(self.features),
            "validation": (
                self.validation_results["overall"] if validate else None
            ),
            "top_features": self.feature_importances.head(10).to_dict(),
        }

    # ------------------------------------------------------------------ #
    #  Prediction                                                         #
    # ------------------------------------------------------------------ #

    def predict(self, recent_df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Generate trading signals for recent data.

        Args:
            recent_df: Output of ``prepare_features(is_inference=True)``,
                filtered to the latest timestamp.

        Returns:
            Dict mapping symbol to signal dict::

                {
                    "BTCUSDT": {
                        "action": "BUY",
                        "confidence": 0.72,
                        "buy_probability": 0.72,
                        "features_used": 45,
                    }
                }
        """
        if not self.is_fitted:
            logger.error("Model not fitted. Call train() first.")
            return {}

        if recent_df.empty:
            return {}

        missing = [f for f in self.features if f not in recent_df.columns]
        if missing:
            logger.warning("Missing features: %s", missing[:5])
            return {}

        X = recent_df[self.features].values
        probas = self.ensemble.predict_proba(X)
        buy_probs = probas[:, 1]

        signals: Dict[str, Dict] = {}
        for i, (_, row) in enumerate(recent_df.iterrows()):
            symbol = row["symbol"]
            prob = buy_probs[i]

            if prob >= self.enter_threshold:
                action = "BUY"
                confidence = prob
            elif prob <= self.exit_threshold:
                action = "SELL"
                confidence = 1 - prob
            else:
                action = "HOLD"
                confidence = max(prob, 1 - prob)

            if confidence < self.min_confidence:
                action = "HOLD"

            signals[symbol] = {
                "action": action,
                "confidence": float(confidence),
                "buy_probability": float(prob),
                "features_used": len(self.features),
            }

            if action != "HOLD":
                logger.info(
                    "%s: %s (conf=%.2f, prob=%.2f)",
                    symbol,
                    action,
                    confidence,
                    prob,
                )

        return signals

    # ------------------------------------------------------------------ #
    #  Summary                                                            #
    # ------------------------------------------------------------------ #

    def get_model_summary(self) -> Dict:
        """Return model metadata and validation results."""
        return {
            "is_fitted": self.is_fitted,
            "n_features": len(self.features),
            "thresholds": {
                "enter": self.enter_threshold,
                "exit": self.exit_threshold,
                "min_confidence": self.min_confidence,
            },
            "validation": (
                self.validation_results["overall"]
                if self.validation_results
                else None
            ),
            "top_features": (
                self.feature_importances.head(10).to_dict()
                if self.feature_importances is not None
                else None
            ),
        }
