# Model Improvements — Implementation Plan (Steps 3-5)

## Status

Steps 1-2 are **complete and tested** (64 tests, all passing).

| Step | Module | Status | Tests |
|------|--------|--------|-------|
| 1 | `src/models/validation.py` — `WalkForwardValidator`, `PurgedKFold` | DONE | 34 pass |
| 2 | `src/models/ensemble.py` — `TradingEnsemble`, `MultiTargetEnsemble` | DONE | 30 pass |
| 3 | `src/models/tuning.py` — `TradingModelTuner`, `GridSearchWalkForward` | TODO | — |
| 4 | `src/models/trading_model.py` — `ImprovedTradingModel` | TODO | — |
| 5 | Integration into `main.py` + `src/models/__init__.py` | TODO | — |

---

## Context — Read These Files First

Before writing any code, read these files to understand the project:

### Specs
1. `docs/01_project_overview.md` — architecture and roadmap
2. `docs/02_feature_engineering.md` — 50+ feature specs
3. `docs/05_model_improvements.md` — the original spec this work implements

### Completed code (Steps 1-2) — MUST READ
4. `src/models/__init__.py` — currently just a docstring, needs updating in Step 5
5. `src/models/validation.py` — `WalkForwardValidator` and `PurgedKFold` (Step 1)
6. `src/models/ensemble.py` — `TradingEnsemble` and `MultiTargetEnsemble` (Step 2)

### Existing codebase — MUST READ
7. `src/analysis/models.py` — current `TradingModel` class (to be replaced in Step 5)
8. `main.py` — orchestrator that uses `TradingModel` (to be updated in Step 5)
9. `src/features/technical.py` — `TechnicalIndicators`, `add_technical_indicators`
10. `src/features/sentiment_advanced.py` — `EnhancedSentimentAnalyzer`, `AdvancedSentimentAnalyzer`
11. `src/features/__init__.py` — feature module exports

### Test patterns
12. `tests/test_validation.py` — test style reference for Step 1
13. `tests/test_ensemble.py` — test style reference for Step 2
14. `tests/test_technical.py` — test style reference for the existing codebase

### Dependencies
15. `requirements.txt` — `optuna` must be added here in Step 3

---

## Key Design Decisions (Already Made)

These decisions were made during Steps 1-2 and must be followed:

1. **Binary classification with threshold-based signals.** The new model uses binary classification (0/1) with probability thresholds for BUY/SELL/HOLD, replacing the current ternary (1/0/-1) scheme in `src/analysis/models.py`.

2. **Signal format change.** The current `TradingModel.predict()` returns `{symbol: "BUY"}`. The new `ImprovedTradingModel.predict()` returns `{symbol: {"action": "BUY", "confidence": 0.72, "buy_probability": 0.72, "features_used": 45}}`. `main.py` must be updated to extract `action` from the nested dict.

3. **Feature preparation reuse.** The new `ImprovedTradingModel.prepare_features()` should reuse the existing lag logic from `src/analysis/models.py` (8 lag periods: 1, 2, 3, 6, 12, 24, 36, 48 for sentiment, volume, return), plus the technical and sentiment feature lists.

4. **Backward compatibility.** `src/analysis/models.py` stays untouched. If anything breaks, reverting `main.py`'s imports is sufficient.

5. **sklearn clone() compatibility.** `TradingEnsemble` inherits from `BaseEstimator` + `ClassifierMixin` and works with `sklearn.base.clone()`, which `WalkForwardValidator.validate()` uses to create independent fold models.

6. **Data leakage prevention.**
   - `WalkForwardValidator.validate()` asserts `train_idx.max() < test_idx.min()` every fold.
   - Folds with a single class in training data are skipped with a warning.
   - `purge_gap` parameter creates a buffer between train and test to prevent label leakage from forward-looking targets.

---

## Step 3: Hyperparameter Tuning

### File: `src/models/tuning.py`

### Classes to implement

#### `TradingModelTuner`

Bayesian hyperparameter optimisation using Optuna with walk-forward validation as the objective.

```python
# src/models/tuning.py
"""
Hyperparameter tuning for trading models.
Uses Optuna for Bayesian optimization with walk-forward validation
as the scoring method — no data leakage.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.model_selection import ParameterGrid
import logging

logger = logging.getLogger(__name__)


class TradingModelTuner:
    """
    Hyperparameter tuning with walk-forward validation.
    Uses Optuna for Bayesian optimization.
    """

    def __init__(
        self,
        n_trials: int = 50,
        validation_metric: str = "precision",
        timeout: Optional[int] = None,
    ):
        """
        Args:
            n_trials: Number of Optuna trials.
            validation_metric: Metric to optimize.
                One of 'precision', 'recall', 'f1', 'accuracy'.
            timeout: Optional time limit in seconds for the study.
        """
        self.n_trials = n_trials
        self.validation_metric = validation_metric
        self.timeout = timeout

    def tune_xgboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validator,     # WalkForwardValidator instance
        feature_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        Tune XGBoost hyperparameters.

        IMPORTANT: The validator handles data splitting. This method must
        NOT shuffle or re-split the data — doing so would break temporal
        ordering and introduce leakage.

        Args:
            X: Feature matrix (chronologically ordered).
            y: Target labels.
            validator: A WalkForwardValidator instance.
            feature_names: Optional column names.

        Returns:
            Dictionary with 'best_params' and 'best_score'.
        """
        import optuna
        from optuna.samplers import TPESampler
        import xgboost as xgb

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': 42,
                'n_jobs': -1,
                'eval_metric': 'logloss',
            }

            cols = feature_names or [f"f{i}" for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=cols)
            df['target'] = y

            model = xgb.XGBClassifier(**params)
            results = validator.validate(model, df, cols)

            return results['overall'][self.validation_metric]

        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
        )
        # Suppress Optuna's verbose logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
        )

        logger.info(
            "XGBoost tuning complete: best %s=%.4f",
            self.validation_metric,
            study.best_value,
        )
        logger.info("Best params: %s", study.best_params)

        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'study': study,
        }

    def tune_lightgbm(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validator,
        feature_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        Tune LightGBM hyperparameters.

        Same contract as tune_xgboost — validator handles splits, no shuffling.
        """
        import optuna
        from optuna.samplers import TPESampler
        import lightgbm as lgb

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1,
            }

            cols = feature_names or [f"f{i}" for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=cols)
            df['target'] = y

            model = lgb.LGBMClassifier(**params)
            results = validator.validate(model, df, cols)

            return results['overall'][self.validation_metric]

        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
        )

        logger.info(
            "LightGBM tuning complete: best %s=%.4f",
            self.validation_metric,
            study.best_value,
        )

        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'study': study,
        }
```

#### `GridSearchWalkForward`

Simpler grid search alternative.

```python
class GridSearchWalkForward:
    """
    Exhaustive grid search scored with walk-forward validation.

    Simpler and more interpretable than Optuna but slower for large grids.
    """

    def __init__(self, param_grid: Dict[str, List]):
        """
        Args:
            param_grid: Dictionary mapping parameter names to lists of values.
        """
        self.param_grid = param_grid

    def search(
        self,
        model_class,
        X: np.ndarray,
        y: np.ndarray,
        validator,    # WalkForwardValidator
        metric: str = "precision",
        feature_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        Evaluate every combination in the grid.

        Args:
            model_class: Sklearn-compatible class (e.g. XGBClassifier).
            X: Feature matrix (chronologically ordered).
            y: Target labels.
            validator: WalkForwardValidator instance.
            metric: Which metric to maximize.
            feature_names: Optional column names.

        Returns:
            Dict with 'best_params', 'best_score', 'all_results'.
        """
        cols = feature_names or [f"f{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=cols)
        df['target'] = y

        best_score = -np.inf
        best_params = None
        all_results = []

        for params in ParameterGrid(self.param_grid):
            model = model_class(**params)
            results = validator.validate(model, df, cols)
            score = results['overall'][metric]

            all_results.append({
                'params': params,
                'score': score,
                'metrics': results['overall'],
            })

            if score > best_score:
                best_score = score
                best_params = params

            logger.info("Grid: %s -> %s=%.4f", params, metric, score)

        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': all_results,
        }
```

### Test file: `tests/test_tuning.py`

Key tests to write:

```
TestTradingModelTuner:
    test_tune_xgboost_returns_best_params       — verify keys in return dict
    test_tune_xgboost_best_score_is_float        — score in [0, 1]
    test_tune_xgboost_no_data_leakage            — verify validator is used (mock)
    test_tune_lightgbm_returns_best_params       — same for LightGBM
    test_tune_with_timeout                        — timeout stops early
    test_tune_with_small_n_trials                 — n_trials=2 still works

TestGridSearchWalkForward:
    test_search_returns_best_params               — verify return dict structure
    test_search_all_results_length                — equals len(ParameterGrid)
    test_search_best_score_is_maximum             — best >= all others
    test_search_no_data_leakage                   — train_end < test_start for all folds
    test_search_with_single_param_combination     — 1-element grid

TestParameterValidation:
    test_invalid_metric_handled_gracefully        — bad metric name
```

**Important implementation notes:**
- Use small `n_trials=3` and `min_train_size=50` in tests to keep runtime fast.
- Use small synthetic datasets (200 rows, 5 features).
- `optuna` must be added to `requirements.txt`.
- Suppress optuna logging in tests with `optuna.logging.set_verbosity(optuna.logging.WARNING)`.

### Dependency update

Add to `requirements.txt`:
```
optuna
```

---

## Step 4: Improved Trading Model

### File: `src/models/trading_model.py`

This is the top-level class that orchestrates everything. It replaces `src/analysis/models.py`'s `TradingModel`.

```python
# src/models/trading_model.py
"""
Production-ready trading model with ensemble, walk-forward validation,
and feature importance tracking.
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

    This is a drop-in replacement for TradingModel in src/analysis/models.py.
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

        self.ensemble = TradingEnsemble(voting='soft', calibrate=True)
        self.validator = WalkForwardValidator(
            min_train_size=500,
            test_size=168,
            step_size=24,
        )

        self.features: List[str] = []
        self.feature_importances: Optional[pd.Series] = None
        self.validation_results: Optional[Dict] = None
        self.is_fitted = False

    def prepare_features(
        self,
        market_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        is_inference: bool = False,
    ) -> pd.DataFrame:
        """
        Prepare all features for training or prediction.

        CRITICAL DATA LEAKAGE NOTES:
        - Technical indicators use only past data (rolling windows, shifts).
        - Lags use .shift(lag) which only looks backward.
        - Target uses .shift(-1) (next-hour return) — only present when
          is_inference=False, and these rows are excluded from feature
          NaN-dropping.
        - The target threshold (>0.5% = BUY) is applied to the NEXT hour's
          return, creating the binary label.

        This method replicates the feature logic from src/analysis/models.py
        but produces a BINARY target (0/1) instead of ternary (-1/0/1).

        Args:
            market_df: OHLCV data with columns
                [timestamp, symbol, open, high, low, close, volume].
            sentiment_df: Hourly sentiment data with columns
                [timestamp, symbol, sentiment_mean, post_volume, ...].
            is_inference: If True, skip target creation and don't drop
                rows missing the target.

        Returns:
            DataFrame ready for train() or predict().
        """
        # --- IMPLEMENTATION NOTES ---
        # 1. Copy the full feature prep logic from src/analysis/models.py
        #    lines 28-151, with these changes:
        #
        # 2. CHANGE: Binary target instead of ternary.
        #    Replace:
        #        conditions = [(df['target_return'] > 0.005),
        #                      (df['target_return'] < -0.005)]
        #        df['target'] = np.select(conditions, [1, -1], default=0)
        #    With:
        #        df['target'] = (df['target_return'] > 0.005).astype(int)
        #
        # 3. KEEP: All 8 lag periods (1, 2, 3, 6, 12, 24, 36, 48)
        # 4. KEEP: The technical_features and sentiment_features lists
        # 5. KEEP: The is_inference branch that skips target creation
        # 6. KEEP: Zero-filling of missing sentiment columns

        if market_df is None or market_df.empty:
            logger.error("Market DataFrame is empty")
            return pd.DataFrame()

        if sentiment_df is None:
            sentiment_df = pd.DataFrame()

        market_df = market_df.copy()
        sentiment_df = sentiment_df.copy()

        # Align timestamps
        market_df['timestamp'] = pd.to_datetime(
            market_df['timestamp']
        ).dt.floor('h')

        # Add technical indicators
        logger.info("Adding technical indicators...")
        market_df = add_technical_indicators(market_df)

        # Merge with sentiment
        if not sentiment_df.empty:
            if 'timestamp' not in sentiment_df.columns:
                logger.error("Sentiment DataFrame missing 'timestamp'")
                return pd.DataFrame()
            sentiment_df['timestamp'] = pd.to_datetime(
                sentiment_df['timestamp']
            ).dt.tz_localize(None).dt.floor('h')
            df = pd.merge(
                market_df, sentiment_df,
                on=['timestamp', 'symbol'], how='left',
            )
        else:
            df = market_df.copy()

        # Zero-fill sentiment
        if 'sentiment_mean' not in df.columns:
            df['sentiment_mean'] = 0.0
        else:
            df['sentiment_mean'] = df['sentiment_mean'].fillna(0.0)

        if 'post_volume' not in df.columns:
            df['post_volume'] = 0.0
        else:
            df['post_volume'] = df['post_volume'].fillna(0.0)

        # Hourly return
        if 'hourly_return' not in df.columns:
            df['hourly_return'] = df.groupby('symbol')['close'].pct_change()

        # Lags (SAME as current model)
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

        # Technical feature names (SAME as current model)
        technical_features = [
            'rsi_14', 'rsi_6', 'rsi_divergence',
            'macd_histogram', 'macd_crossover',
            'bb_percent_b', 'bb_bandwidth', 'bb_squeeze',
            'atr_14_pct', 'atr_expansion',
            'adx', 'trend_strength',
            'ma_spread', 'price_above_sma20',
            'volume_ratio', 'volume_spike', 'mfi',
            'return_1h', 'return_4h',
            'dist_from_24h_high', 'dist_from_24h_low',
        ]

        # Sentiment feature names (SAME as current model)
        sentiment_features = [
            'sentiment_velocity', 'sentiment_acceleration',
            'sentiment_macd', 'sentiment_rsi', 'sentiment_reversal',
            'sentiment_std', 'sentiment_consensus',
            'extreme_bullish_ratio', 'extreme_bearish_ratio',
            'sentiment_engagement_weighted', 'sentiment_high_engagement',
            'total_engagement',
            'market_sentiment', 'sentiment_vs_market', 'sentiment_z_score',
            'sentiment_regime_numeric',
        ]

        available_technical = [
            f for f in technical_features if f in df.columns
        ]
        available_sentiment = [
            f for f in sentiment_features if f in df.columns
        ]

        self.features = lag_cols + available_technical + available_sentiment
        logger.info(
            "Using %d features: %d lag + %d technical + %d sentiment",
            len(self.features), len(lag_cols),
            len(available_technical), len(available_sentiment),
        )

        if not is_inference:
            # BINARY target: >0.5% next-hour return = 1, else 0
            df['target_return'] = df.groupby('symbol')[
                'hourly_return'
            ].shift(-1)
            df['target'] = (df['target_return'] > 0.005).astype(int)
            df = df.dropna(subset=self.features + ['target_return', 'target'])
        else:
            df = df.dropna(subset=self.features)

        return df

    def train(
        self,
        df: pd.DataFrame,
        validate: bool = True,
    ) -> Dict:
        """
        Train the ensemble.

        Args:
            df: Output of prepare_features(is_inference=False).
            validate: If True, run walk-forward validation first.

        Returns:
            Dictionary with training results.
        """
        if df.empty or len(df) < 100:
            logger.warning("Insufficient data for training")
            return {'status': 'insufficient_data'}

        X = df[self.features].values
        y = df['target'].values

        logger.info("Training on %d samples", len(X))
        logger.info(
            "Class distribution: %d neg, %d pos",
            (y == 0).sum(), (y == 1).sum(),
        )

        # Walk-forward validation
        if validate:
            logger.info("Running walk-forward validation...")
            self.validation_results = self.validator.validate(
                self.ensemble,
                df[self.features + ['target']],
                self.features,
            )
            logger.info(
                "Validation: precision=%.3f, recall=%.3f, f1=%.3f",
                self.validation_results['overall']['precision'],
                self.validation_results['overall']['recall'],
                self.validation_results['overall']['f1'],
            )

        # Final fit on all data
        self.ensemble.fit(X, y, feature_names=self.features)
        self.feature_importances = self.ensemble.get_feature_importances()
        self.is_fitted = True

        return {
            'status': 'success',
            'n_samples': len(X),
            'n_features': len(self.features),
            'validation': (
                self.validation_results['overall'] if validate else None
            ),
            'top_features': self.feature_importances.head(10).to_dict(),
        }

    def predict(self, recent_df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Generate trading signals.

        OUTPUT FORMAT (different from old TradingModel):
            {
                "BTCUSDT": {
                    "action": "BUY",
                    "confidence": 0.72,
                    "buy_probability": 0.72,
                    "features_used": 45,
                },
                ...
            }

        Args:
            recent_df: Output of prepare_features(is_inference=True),
                filtered to the latest timestamp.

        Returns:
            Dict mapping symbol to signal dict.
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

        signals = {}
        for i, (_, row) in enumerate(recent_df.iterrows()):
            symbol = row['symbol']
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
                'action': action,
                'confidence': float(confidence),
                'buy_probability': float(prob),
                'features_used': len(self.features),
            }

            if action != "HOLD":
                logger.info(
                    "%s: %s (conf=%.2f, prob=%.2f)",
                    symbol, action, confidence, prob,
                )

        return signals

    def get_model_summary(self) -> Dict:
        """Return model metadata."""
        return {
            'is_fitted': self.is_fitted,
            'n_features': len(self.features),
            'thresholds': {
                'enter': self.enter_threshold,
                'exit': self.exit_threshold,
                'min_confidence': self.min_confidence,
            },
            'validation': (
                self.validation_results['overall']
                if self.validation_results else None
            ),
            'top_features': (
                self.feature_importances.head(10).to_dict()
                if self.feature_importances is not None else None
            ),
        }
```

### Test file: `tests/test_trading_model.py`

Key tests to write:

```
TestPrepareFeatures:
    test_prepare_features_returns_dataframe       — non-empty df
    test_prepare_features_has_target_column        — 'target' in columns
    test_prepare_features_binary_target            — target values are 0 or 1 only
    test_prepare_features_no_target_in_inference   — is_inference=True has no 'target'
    test_prepare_features_lag_columns_present      — all 24 lag cols exist
    test_prepare_features_technical_features       — technical cols present
    test_prepare_features_no_future_data_in_lags   — lag_1 at row i equals value at row i-1
    test_prepare_features_empty_market_returns_empty
    test_prepare_features_empty_sentiment_ok       — should still work with no sentiment

TestTrain:
    test_train_returns_success                     — status='success'
    test_train_sets_is_fitted                      — is_fitted becomes True
    test_train_produces_feature_importances        — importances not None
    test_train_validation_results_present          — validation dict populated
    test_train_skip_validation                     — validate=False still works
    test_train_insufficient_data                   — <100 rows returns status

TestPredict:
    test_predict_returns_signals_dict              — {symbol: {action, ...}}
    test_predict_actions_are_valid                 — action in {BUY, SELL, HOLD}
    test_predict_confidence_bounded                — 0 <= confidence <= 1
    test_predict_buy_probability_bounded            — 0 <= prob <= 1
    test_predict_before_fit_returns_empty
    test_predict_threshold_logic                   — prob > enter → BUY, prob < exit → SELL

TestNoDataLeakage:
    test_target_uses_future_return                 — shift(-1) means next hour
    test_features_only_use_past                    — all lags are positive shifts
    test_no_target_leaks_into_features             — 'target' not in self.features
```

**Implementation notes:**
- For tests needing market data, create a fixture with realistic OHLCV (100+ rows, 1-2 symbols).
- For sentiment data, create a matching fixture with `timestamp`, `symbol`, `sentiment_mean`, `post_volume`.
- The `add_technical_indicators` function needs real OHLCV data (open/high/low/close/volume).

---

## Step 5: Integration

### 5a. Update `src/models/__init__.py`

```python
# src/models/__init__.py
"""ML model modules for crypto sentiment trading."""

from src.models.validation import WalkForwardValidator, PurgedKFold
from src.models.ensemble import TradingEnsemble, MultiTargetEnsemble
from src.models.tuning import TradingModelTuner, GridSearchWalkForward
from src.models.trading_model import ImprovedTradingModel

__all__ = [
    'WalkForwardValidator',
    'PurgedKFold',
    'TradingEnsemble',
    'MultiTargetEnsemble',
    'TradingModelTuner',
    'GridSearchWalkForward',
    'ImprovedTradingModel',
]
```

### 5b. Update `main.py`

Two changes needed:

#### Change 1: Import (line 16)

Replace:
```python
from src.analysis.models import TradingModel
```
With:
```python
from src.models.trading_model import ImprovedTradingModel
```

#### Change 2: Model instantiation (line 135)

Replace:
```python
model = TradingModel()
```
With:
```python
model = ImprovedTradingModel()
```

#### Change 3: Signal consumption (lines 252-313)

The current code does:
```python
for symbol, action in signals.items():
    ...
    if action == "BUY":
        ...
    elif action == "SELL":
        ...
```

Replace with:
```python
for symbol, signal in signals.items():
    action = signal['action']
    confidence = signal['confidence']
    price = price_map.get(symbol)
    if price is None:
        continue

    # Check current holding state
    pos = ledger.positions.get(symbol, {})
    current_qty = pos.get("qty", 0.0)
    is_holding = current_qty > 0

    if action == "BUY":
        # GATE 1: Don't double buy
        if is_holding:
            continue

        # GATE 2: Cooldown Check
        last_exit = ledger.last_exit.get(symbol)
        if last_exit:
            if last_exit.tzinfo is None:
                last_exit = last_exit.replace(tzinfo=timezone.utc)
            hours_since_exit = (current_time - last_exit).total_seconds() / 3600
            if hours_since_exit < COOLDOWN_HOURS:
                logger.info(
                    f"COOLDOWN BLOCKED BUY {symbol}: "
                    f"Exited {hours_since_exit:.1f}h ago (<{COOLDOWN_HOURS}h)"
                )
                continue

        if DRY_RUN:
            ok = ledger.buy(
                symbol, mid_price=price,
                notional_usdt=TRADE_NOTIONAL_USDT,
                current_time=current_time,
            )
            if ok:
                logger.info(
                    f"[PAPER] BUY {symbol} notional=${TRADE_NOTIONAL_USDT:.2f} "
                    f"mid={price:.6f} conf={confidence:.2f}"
                )
            else:
                logger.info(f"[PAPER] BUY skipped (insufficient cash) {symbol}")
        else:
            quantity = round(TRADE_NOTIONAL_USDT / price, 5)
            executor.execute_order(symbol, "BUY", quantity)

    elif action == "SELL":
        # GATE 1: Can't sell what you don't have
        if not is_holding:
            continue

        # GATE 3: Minimum Hold Time
        entry_ts = pos.get("entry_ts")
        if entry_ts:
            if entry_ts.tzinfo is None:
                entry_ts = entry_ts.replace(tzinfo=timezone.utc)
            hours_held = (current_time - entry_ts).total_seconds() / 3600
            if hours_held < MIN_HOLD_HOURS:
                logger.info(
                    f"MIN HOLD BLOCKED SELL {symbol}: "
                    f"Held {hours_held:.1f}h (<{MIN_HOLD_HOURS}h)"
                )
                continue

        if DRY_RUN:
            ok = ledger.sell_all(symbol, mid_price=price, current_time=current_time)
            if ok:
                logger.info(
                    f"[PAPER] SELL_ALL {symbol} mid={price:.6f} conf={confidence:.2f}"
                )
        else:
            executor.execute_order(symbol, "SELL", quantity=None)
```

### 5c. Run the full test suite

After all changes:
```bash
python -m pytest tests/ -v
```

This should run all tests:
- `tests/test_validation.py` (34 tests)
- `tests/test_ensemble.py` (30 tests)
- `tests/test_tuning.py` (new)
- `tests/test_trading_model.py` (new)
- `tests/test_technical.py` (existing)
- `tests/test_sentiment_advanced.py` (existing)
- `tests/test_integration.py` (existing)

---

## Critical: Data Leakage Checklist

Before marking any step complete, verify:

- [ ] `WalkForwardValidator` asserts `train_max < test_min` on every fold
- [ ] Purge gap is configured when targets use forward returns
- [ ] All `.shift()` calls in feature prep use **positive** lag values (look backward)
- [ ] Target creation uses `.shift(-1)` (look one step forward) and is ONLY created when `is_inference=False`
- [ ] `'target'` and `'target_return'` are NOT in `self.features`
- [ ] `'timestamp'`, `'symbol'`, `'open'`, `'high'`, `'low'`, `'close'`, `'volume'` are excluded from features
- [ ] No `shuffle=True` anywhere in the pipeline
- [ ] `TradingEnsemble.fit()` rejects single-class training data
- [ ] `WalkForwardValidator.validate()` skips single-class folds
- [ ] Tuning uses walk-forward validation as objective (not random CV)

---

## Environment Notes

- Python 3.11.5 (anaconda)
- xgboost 2.1.4 (installed; version 3.x had OpenMP issues on macOS)
- lightgbm 4.6.0
- `optuna` needs to be installed: `pip install optuna`
- Tests use `pytest` with verbose output: `python -m pytest tests/ -v`
