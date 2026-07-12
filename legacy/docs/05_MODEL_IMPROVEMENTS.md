# Model Improvements Guide

## Current State Analysis

### Existing Model (`src/analysis/models.py`)
```python
# Current: Single Random Forest with basic setup
self.model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=5, 
    random_state=42, 
    class_weight='balanced'
)

# Issues:
# 1. Simple train/test split (not walk-forward)
# 2. No hyperparameter tuning
# 3. Single model (no ensemble)
# 4. Binary classification only (Buy/Hold)
# 5. Fixed 80/20 split regardless of data size
# 6. No feature importance tracking
```

---

## Improvement 1: Proper Walk-Forward Validation

### Why It Matters
Time series data requires special handling to prevent lookahead bias. Standard cross-validation randomly shuffles data, causing future data to leak into training.

### Implementation

```python
# src/models/validation.py
"""
Walk-forward validation for time series trading models.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Generator, Dict, Any
from sklearn.base import BaseEstimator
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import logging

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Walk-forward validation for trading strategies.
    
    Unlike standard CV, this:
    1. Always trains on past, tests on future
    2. Can use expanding or rolling windows
    3. Simulates live trading conditions
    """
    
    def __init__(self,
                 min_train_size: int = 500,
                 test_size: int = 168,  # 1 week of hourly data
                 step_size: int = 24,   # Step forward 1 day
                 expanding: bool = True):
        """
        Args:
            min_train_size: Minimum rows for initial training
            test_size: Rows to test on each fold
            step_size: How many rows to advance each fold
            expanding: If True, train on all past data; if False, use rolling window
        """
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.step_size = step_size
        self.expanding = expanding
    
    def split(self, df: pd.DataFrame) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices for walk-forward validation.
        
        Args:
            df: DataFrame sorted by time
        
        Yields:
            Tuple of (train_indices, test_indices)
        """
        n = len(df)
        
        # Start after we have enough training data
        train_end = self.min_train_size
        
        while train_end + self.test_size <= n:
            if self.expanding:
                train_start = 0
            else:
                train_start = max(0, train_end - self.min_train_size)
            
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(train_end, min(train_end + self.test_size, n))
            
            yield train_idx, test_idx
            
            train_end += self.step_size
    
    def validate(self,
                 model: BaseEstimator,
                 df: pd.DataFrame,
                 features: List[str],
                 target: str = 'target') -> Dict[str, Any]:
        """
        Run walk-forward validation and return metrics.
        
        Args:
            model: Sklearn-compatible model
            df: Feature DataFrame
            features: List of feature column names
            target: Target column name
        
        Returns:
            Dictionary with validation results
        """
        all_predictions = []
        all_actuals = []
        all_probas = []
        fold_metrics = []
        
        X = df[features].values
        y = df[target].values
        
        for fold_idx, (train_idx, test_idx) in enumerate(self.split(df)):
            # Train
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            
            model.fit(X_train, y_train)
            
            # Predict
            predictions = model.predict(X_test)
            probas = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else predictions
            
            # Store
            all_predictions.extend(predictions)
            all_actuals.extend(y_test)
            all_probas.extend(probas)
            
            # Fold metrics
            fold_metrics.append({
                'fold': fold_idx,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'precision': precision_score(y_test, predictions, zero_division=0),
                'recall': recall_score(y_test, predictions, zero_division=0),
                'f1': f1_score(y_test, predictions, zero_division=0),
                'accuracy': accuracy_score(y_test, predictions)
            })
            
            logger.info(f"Fold {fold_idx}: Precision={fold_metrics[-1]['precision']:.3f}, "
                       f"Recall={fold_metrics[-1]['recall']:.3f}")
        
        # Aggregate metrics
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)
        
        return {
            'overall': {
                'precision': precision_score(all_actuals, all_predictions, zero_division=0),
                'recall': recall_score(all_actuals, all_predictions, zero_division=0),
                'f1': f1_score(all_actuals, all_predictions, zero_division=0),
                'accuracy': accuracy_score(all_actuals, all_predictions),
                'n_samples': len(all_actuals),
                'n_folds': len(fold_metrics)
            },
            'per_fold': fold_metrics,
            'predictions': all_predictions,
            'actuals': all_actuals,
            'probabilities': np.array(all_probas)
        }


class PurgedKFold:
    """
    K-Fold with purging to prevent lookahead bias.
    
    Adds a gap between train and test to account for:
    1. Label overlap (e.g., 24h forward returns)
    2. Feature leakage from lagged features
    """
    
    def __init__(self, 
                 n_splits: int = 5,
                 purge_gap: int = 24):  # Hours
        self.n_splits = n_splits
        self.purge_gap = purge_gap
    
    def split(self, df: pd.DataFrame) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate purged train/test splits."""
        n = len(df)
        fold_size = n // self.n_splits
        
        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n
            
            # Train indices: everything except test + purge gap
            train_idx = np.concatenate([
                np.arange(0, max(0, test_start - self.purge_gap)),
                np.arange(test_end + self.purge_gap, n)
            ])
            
            test_idx = np.arange(test_start, test_end)
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx.astype(int), test_idx.astype(int)
```

---

## Improvement 2: Model Ensemble

### Why It Matters
Different models capture different patterns. Ensembling reduces variance and improves robustness.

### Implementation

```python
# src/models/ensemble.py
"""
Ensemble models for crypto trading.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
import logging

logger = logging.getLogger(__name__)


class TradingEnsemble:
    """
    Ensemble of multiple models with calibrated probabilities.
    
    Components:
    1. Random Forest - Good with noisy features
    2. XGBoost - Strong gradient boosting
    3. LightGBM - Fast, handles large feature sets
    4. Logistic Regression - Simple baseline, regularized
    """
    
    def __init__(self,
                 voting: str = 'soft',
                 calibrate: bool = True,
                 rf_params: Optional[Dict] = None,
                 xgb_params: Optional[Dict] = None,
                 lgb_params: Optional[Dict] = None):
        """
        Args:
            voting: 'soft' (average probabilities) or 'hard' (majority vote)
            calibrate: Whether to calibrate probabilities
        """
        self.voting = voting
        self.calibrate = calibrate
        
        # Default parameters (tuned for crypto hourly data)
        self.rf_params = rf_params or {
            'n_estimators': 200,
            'max_depth': 8,
            'min_samples_leaf': 20,
            'max_features': 'sqrt',
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.xgb_params = xgb_params or {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 1,  # Will be set dynamically
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        }
        
        self.lgb_params = lgb_params or {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        self.models = {}
        self.ensemble = None
        self.feature_importances_ = None
    
    def _create_models(self, scale_pos_weight: float = 1.0) -> List[Tuple[str, any]]:
        """Create model instances."""
        xgb_params = self.xgb_params.copy()
        xgb_params['scale_pos_weight'] = scale_pos_weight
        
        models = [
            ('rf', RandomForestClassifier(**self.rf_params)),
            ('xgb', xgb.XGBClassifier(**xgb_params)),
            ('lgb', lgb.LGBMClassifier(**self.lgb_params)),
            ('lr', LogisticRegression(
                C=0.1, 
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            ))
        ]
        
        return models
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: Optional[List[str]] = None) -> 'TradingEnsemble':
        """
        Fit the ensemble.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Optional feature names for importance tracking
        """
        # Calculate class imbalance for XGBoost
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        scale_pos_weight = n_neg / max(n_pos, 1)
        
        logger.info(f"Class distribution: {n_neg} negative, {n_pos} positive")
        logger.info(f"Scale pos weight: {scale_pos_weight:.2f}")
        
        # Create models
        model_list = self._create_models(scale_pos_weight)
        
        # Calibrate if requested
        if self.calibrate:
            model_list = [
                (name, CalibratedClassifierCV(model, cv=3, method='isotonic'))
                for name, model in model_list
            ]
        
        # Create voting ensemble
        self.ensemble = VotingClassifier(
            estimators=model_list,
            voting=self.voting,
            n_jobs=-1
        )
        
        # Fit
        logger.info("Fitting ensemble...")
        self.ensemble.fit(X, y)
        
        # Store individual models and feature importances
        self._store_feature_importances(X, y, feature_names)
        
        return self
    
    def _store_feature_importances(self, X: np.ndarray, y: np.ndarray,
                                    feature_names: Optional[List[str]]):
        """Extract and average feature importances."""
        importances = []
        
        # Fit individual models to get importances
        for name, (_, model) in zip(['rf', 'xgb', 'lgb'], self._create_models()[:3]):
            try:
                model.fit(X, y)
                if hasattr(model, 'feature_importances_'):
                    importances.append(model.feature_importances_)
            except Exception as e:
                logger.warning(f"Could not get importance from {name}: {e}")
        
        if importances:
            # Average importances across models
            avg_importance = np.mean(importances, axis=0)
            
            if feature_names:
                self.feature_importances_ = pd.Series(
                    avg_importance, 
                    index=feature_names
                ).sort_values(ascending=False)
            else:
                self.feature_importances_ = avg_importance
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.ensemble.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.ensemble.predict_proba(X)
    
    def get_feature_importances(self, top_n: int = 20) -> pd.Series:
        """Get top N most important features."""
        if self.feature_importances_ is None:
            raise ValueError("Model not fitted yet")
        
        if isinstance(self.feature_importances_, pd.Series):
            return self.feature_importances_.head(top_n)
        return self.feature_importances_[:top_n]


class MultiTargetEnsemble:
    """
    Ensemble that predicts multiple targets:
    1. Direction (up/down)
    2. Magnitude (small/medium/large move)
    3. Confidence (how sure are we)
    """
    
    def __init__(self):
        self.direction_model = TradingEnsemble()
        self.magnitude_model = TradingEnsemble()
    
    def fit(self, X: np.ndarray, y_direction: np.ndarray, 
            y_magnitude: np.ndarray) -> 'MultiTargetEnsemble':
        """Fit both models."""
        self.direction_model.fit(X, y_direction)
        self.magnitude_model.fit(X, y_magnitude)
        return self
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict direction and magnitude."""
        return {
            'direction': self.direction_model.predict(X),
            'direction_proba': self.direction_model.predict_proba(X),
            'magnitude': self.magnitude_model.predict(X),
            'magnitude_proba': self.magnitude_model.predict_proba(X)
        }
```

---

## Improvement 3: Hyperparameter Tuning

### Implementation

```python
# src/models/tuning.py
"""
Hyperparameter tuning for trading models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable
from sklearn.model_selection import ParameterGrid
import optuna
from optuna.samplers import TPESampler
import logging

logger = logging.getLogger(__name__)


class TradingModelTuner:
    """
    Hyperparameter tuning with walk-forward validation.
    
    Uses Optuna for Bayesian optimization.
    """
    
    def __init__(self, 
                 n_trials: int = 50,
                 validation_metric: str = 'precision'):
        """
        Args:
            n_trials: Number of Optuna trials
            validation_metric: Metric to optimize ('precision', 'f1', 'sharpe')
        """
        self.n_trials = n_trials
        self.validation_metric = validation_metric
    
    def tune_xgboost(self,
                     X: np.ndarray,
                     y: np.ndarray,
                     validator) -> Dict:
        """
        Tune XGBoost hyperparameters.
        
        Args:
            X: Feature matrix
            y: Target labels
            validator: WalkForwardValidator instance
        
        Returns:
            Best parameters
        """
        import xgboost as xgb
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 1),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
                'random_state': 42,
                'n_jobs': -1,
                'eval_metric': 'logloss'
            }
            
            # Create DataFrame for validator
            df = pd.DataFrame(X)
            df['target'] = y
            features = [c for c in df.columns if c != 'target']
            
            model = xgb.XGBClassifier(**params)
            results = validator.validate(model, df, features)
            
            return results['overall'][self.validation_metric]
        
        # Run optimization
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        logger.info(f"Best {self.validation_metric}: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        
        return study.best_params
    
    def tune_lightgbm(self, X: np.ndarray, y: np.ndarray, validator) -> Dict:
        """Tune LightGBM hyperparameters."""
        import lightgbm as lgb
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            
            df = pd.DataFrame(X)
            df['target'] = y
            features = [c for c in df.columns if c != 'target']
            
            model = lgb.LGBMClassifier(**params)
            results = validator.validate(model, df, features)
            
            return results['overall'][self.validation_metric]
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        return study.best_params


class GridSearchWalkForward:
    """
    Simple grid search with walk-forward validation.
    
    Less sophisticated than Optuna but more interpretable.
    """
    
    def __init__(self, param_grid: Dict[str, List]):
        self.param_grid = param_grid
    
    def search(self, 
               model_class,
               X: np.ndarray,
               y: np.ndarray,
               validator,
               metric: str = 'precision') -> Dict:
        """
        Perform grid search.
        
        Returns:
            Best parameters and results
        """
        df = pd.DataFrame(X)
        df['target'] = y
        features = [c for c in df.columns if c != 'target']
        
        best_score = -np.inf
        best_params = None
        all_results = []
        
        for params in ParameterGrid(self.param_grid):
            model = model_class(**params)
            results = validator.validate(model, df, features)
            score = results['overall'][metric]
            
            all_results.append({
                'params': params,
                'score': score,
                'metrics': results['overall']
            })
            
            if score > best_score:
                best_score = score
                best_params = params
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': all_results
        }
```

---

## Improvement 4: Updated TradingModel Class

```python
# src/models/trading_model.py
"""
Improved trading model with all enhancements.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from src.models.validation import WalkForwardValidator
from src.models.ensemble import TradingEnsemble
from src.features.technical import add_technical_indicators
from src.features.sentiment_advanced import AdvancedSentimentAnalyzer

logger = logging.getLogger(__name__)


class ImprovedTradingModel:
    """
    Production-ready trading model with:
    - Ensemble of multiple algorithms
    - Walk-forward validation
    - Feature importance tracking
    - Multiple signal outputs
    """
    
    def __init__(self,
                 enter_threshold: float = 0.55,
                 exit_threshold: float = 0.45,
                 min_confidence: float = 0.6):
        """
        Args:
            enter_threshold: Probability threshold to enter position
            exit_threshold: Probability threshold to exit position
            min_confidence: Minimum confidence for any action
        """
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.min_confidence = min_confidence
        
        self.ensemble = TradingEnsemble(voting='soft', calibrate=True)
        self.validator = WalkForwardValidator(
            min_train_size=500,
            test_size=168,
            step_size=24
        )
        
        self.features: List[str] = []
        self.feature_importances: Optional[pd.Series] = None
        self.validation_results: Optional[Dict] = None
        self.is_fitted = False
    
    def prepare_features(self,
                         market_df: pd.DataFrame,
                         sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare all features for model training/prediction.
        """
        # Ensure timestamps aligned
        market_df = market_df.copy()
        sentiment_df = sentiment_df.copy()
        
        market_df['timestamp'] = pd.to_datetime(market_df['timestamp']).dt.floor('h')
        
        if 'timestamp' in sentiment_df.columns:
            sentiment_df['timestamp'] = pd.to_datetime(
                sentiment_df['timestamp']
            ).dt.tz_localize(None).dt.floor('h')
        
        # Add technical indicators
        market_df = add_technical_indicators(market_df)
        
        # Merge with sentiment
        df = pd.merge(market_df, sentiment_df, on=['timestamp', 'symbol'], how='left')
        
        # Fill missing sentiment with neutral
        sentiment_cols = [c for c in df.columns if 'sentiment' in c.lower()]
        for col in sentiment_cols:
            df[col] = df[col].fillna(0)
        
        # Create lags for all features
        df = self._create_lags(df)
        
        # Create target
        df['future_return'] = df.groupby('symbol')['close'].pct_change().shift(-1)
        df['target'] = (df['future_return'] > 0.005).astype(int)  # >0.5% = buy
        
        # Define feature columns (exclude targets, identifiers, raw prices)
        exclude_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 
                       'volume', 'target', 'future_return']
        self.features = [c for c in df.columns if c not in exclude_cols and not df[c].isna().all()]
        
        # Drop rows with NaN features
        df = df.dropna(subset=self.features + ['target'])
        
        logger.info(f"Prepared {len(df)} samples with {len(self.features)} features")
        
        return df
    
    def _create_lags(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3, 6, 12]) -> pd.DataFrame:
        """Create lagged features."""
        df = df.copy()
        
        # Columns to lag
        lag_cols = [c for c in df.columns if any(x in c.lower() for x in 
                   ['sentiment', 'return', 'rsi', 'macd', 'volume_ratio', 'atr'])]
        
        for col in lag_cols:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df.groupby('symbol')[col].shift(lag)
        
        return df
    
    def train(self, df: pd.DataFrame, validate: bool = True) -> Dict:
        """
        Train the ensemble model.
        
        Args:
            df: Prepared feature DataFrame
            validate: Whether to run walk-forward validation
        
        Returns:
            Training results
        """
        if df.empty or len(df) < 100:
            logger.warning("Insufficient data for training")
            return {'status': 'insufficient_data'}
        
        X = df[self.features].values
        y = df['target'].values
        
        logger.info(f"Training on {len(X)} samples")
        logger.info(f"Class distribution: {(y==0).sum()} neg, {(y==1).sum()} pos")
        
        # Validation (if requested)
        if validate:
            logger.info("Running walk-forward validation...")
            self.validation_results = self.validator.validate(
                self.ensemble,
                df[self.features + ['target']],
                self.features
            )
            logger.info(f"Validation precision: {self.validation_results['overall']['precision']:.3f}")
        
        # Final fit on all data
        self.ensemble.fit(X, y, feature_names=self.features)
        self.feature_importances = self.ensemble.get_feature_importances()
        self.is_fitted = True
        
        return {
            'status': 'success',
            'n_samples': len(X),
            'n_features': len(self.features),
            'validation': self.validation_results['overall'] if validate else None,
            'top_features': self.feature_importances.head(10).to_dict()
        }
    
    def predict(self, recent_df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Generate trading signals for recent data.
        
        Args:
            recent_df: Recent feature DataFrame
        
        Returns:
            Dictionary of {symbol: {action, confidence, probability}}
        """
        if not self.is_fitted:
            logger.error("Model not fitted. Call train() first.")
            return {}
        
        if recent_df.empty:
            return {}
        
        # Check for missing features
        missing = [f for f in self.features if f not in recent_df.columns]
        if missing:
            logger.warning(f"Missing features: {missing[:5]}...")
            return {}
        
        X = recent_df[self.features].values
        
        # Get probabilities
        probas = self.ensemble.predict_proba(X)
        buy_probs = probas[:, 1]
        
        signals = {}
        
        for i, (_, row) in enumerate(recent_df.iterrows()):
            symbol = row['symbol']
            prob = buy_probs[i]
            
            # Determine action
            if prob >= self.enter_threshold:
                action = "BUY"
                confidence = prob
            elif prob <= self.exit_threshold:
                action = "SELL"
                confidence = 1 - prob
            else:
                action = "HOLD"
                confidence = max(prob, 1 - prob)
            
            # Only signal if confident enough
            if confidence < self.min_confidence:
                action = "HOLD"
            
            signals[symbol] = {
                'action': action,
                'confidence': float(confidence),
                'buy_probability': float(prob),
                'features_used': len(self.features)
            }
            
            if action != "HOLD":
                logger.info(f"{symbol}: {action} (conf={confidence:.2f}, prob={prob:.2f})")
        
        return signals
    
    def get_model_summary(self) -> Dict:
        """Get model summary statistics."""
        return {
            'is_fitted': self.is_fitted,
            'n_features': len(self.features),
            'thresholds': {
                'enter': self.enter_threshold,
                'exit': self.exit_threshold,
                'min_confidence': self.min_confidence
            },
            'validation': self.validation_results['overall'] if self.validation_results else None,
            'top_features': self.feature_importances.head(10).to_dict() if self.feature_importances is not None else None
        }
```

---

## Summary of Model Improvements

| Aspect | Before | After |
|--------|--------|-------|
| Algorithm | Single Random Forest | Ensemble (RF + XGB + LGB + LR) |
| Validation | Simple 80/20 split | Walk-forward with purging |
| Hyperparameters | Fixed defaults | Optuna optimization |
| Probability Calibration | None | Isotonic calibration |
| Feature Importance | Not tracked | Averaged across models |
| Signal Output | Binary (Buy/Hold) | Buy/Sell/Hold with confidence |
| Class Imbalance | class_weight='balanced' | Multiple strategies + calibration |
