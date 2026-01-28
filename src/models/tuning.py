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

    Uses Optuna for Bayesian optimization. The walk-forward validator
    handles all data splitting, ensuring strict temporal ordering and
    no data leakage.

    Typical usage::

        tuner = TradingModelTuner(n_trials=50, validation_metric='precision')
        result = tuner.tune_xgboost(X, y, validator)
        print(result['best_params'], result['best_score'])
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
        validator,
        feature_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        Tune XGBoost hyperparameters using Optuna with walk-forward validation.

        The validator handles data splitting. This method does NOT shuffle
        or re-split the data — doing so would break temporal ordering and
        introduce leakage.

        Args:
            X: Feature matrix (chronologically ordered).
            y: Target labels.
            validator: A WalkForwardValidator instance.
            feature_names: Optional column names.

        Returns:
            Dictionary with 'best_params', 'best_score', and 'study'.
        """
        import optuna
        from optuna.samplers import TPESampler
        import xgboost as xgb

        metric = self.validation_metric

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.6, 1.0
                ),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", 1e-8, 10.0, log=True
                ),
                "random_state": 42,
                "n_jobs": -1,
                "eval_metric": "logloss",
            }

            cols = feature_names or [f"f{i}" for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=cols)
            df["target"] = y

            model = xgb.XGBClassifier(**params)
            results = validator.validate(model, df, cols)

            return results["overall"][metric]

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
        )

        logger.info(
            "XGBoost tuning complete: best %s=%.4f",
            metric,
            study.best_value,
        )
        logger.info("Best params: %s", study.best_params)

        return {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "study": study,
        }

    def tune_lightgbm(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validator,
        feature_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        Tune LightGBM hyperparameters using Optuna with walk-forward validation.

        Same contract as tune_xgboost — validator handles splits, no shuffling.

        Args:
            X: Feature matrix (chronologically ordered).
            y: Target labels.
            validator: A WalkForwardValidator instance.
            feature_names: Optional column names.

        Returns:
            Dictionary with 'best_params', 'best_score', and 'study'.
        """
        import optuna
        from optuna.samplers import TPESampler
        import lightgbm as lgb

        metric = self.validation_metric

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.6, 1.0
                ),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", 1e-8, 10.0, log=True
                ),
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            }

            cols = feature_names or [f"f{i}" for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=cols)
            df["target"] = y

            model = lgb.LGBMClassifier(**params)
            results = validator.validate(model, df, cols)

            return results["overall"][metric]

        study = optuna.create_study(
            direction="maximize",
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
            metric,
            study.best_value,
        )

        return {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "study": study,
        }


class GridSearchWalkForward:
    """
    Exhaustive grid search scored with walk-forward validation.

    Simpler and more interpretable than Optuna but slower for large grids.

    Typical usage::

        grid = GridSearchWalkForward({'n_estimators': [50, 100], 'max_depth': [3, 5]})
        result = grid.search(XGBClassifier, X, y, validator)
        print(result['best_params'], result['best_score'])
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
        validator,
        metric: str = "precision",
        feature_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        Evaluate every combination in the grid using walk-forward validation.

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
        df["target"] = y

        best_score = -np.inf
        best_params = None
        all_results = []

        for params in ParameterGrid(self.param_grid):
            model = model_class(**params)
            results = validator.validate(model, df, cols)
            score = results["overall"][metric]

            all_results.append(
                {
                    "params": params,
                    "score": score,
                    "metrics": results["overall"],
                }
            )

            if score > best_score:
                best_score = score
                best_params = params

            logger.info("Grid: %s -> %s=%.4f", params, metric, score)

        return {
            "best_params": best_params,
            "best_score": best_score,
            "all_results": all_results,
        }
