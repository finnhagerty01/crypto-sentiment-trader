# src/models/ensemble.py
"""
Ensemble models for crypto trading.

Combines multiple classifiers (Random Forest, XGBoost, LightGBM, Logistic
Regression) into a voting ensemble with optional probability calibration.

Design rationale:
- Different algorithms capture different signal types.  RF handles noisy
  features well; gradient-boosted trees (XGB/LGB) model interactions;
  Logistic Regression provides a simple regularised baseline.
- Soft voting (averaging calibrated probabilities) produces smoother
  confidence scores than hard voting (majority label).
- Isotonic calibration maps raw model outputs to well-calibrated
  probabilities so the ensemble's 0.6 actually means "60 % chance".
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
import xgboost as xgb
import lightgbm as lgb
import logging

logger = logging.getLogger(__name__)


class TradingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Ensemble of multiple classifiers with calibrated probabilities.

    Components:
        1. Random Forest — good with noisy, correlated features
        2. XGBoost — strong gradient boosting with regularisation
        3. LightGBM — fast, handles large feature sets
        4. Logistic Regression — regularised linear baseline

    Typical usage::

        ens = TradingEnsemble(voting='soft', calibrate=True)
        ens.fit(X_train, y_train, feature_names=feature_cols)
        probas = ens.predict_proba(X_test)
        importances = ens.get_feature_importances(top_n=15)
    """

    def __init__(
        self,
        voting: str = "soft",
        calibrate: bool = True,
        rf_params: Optional[Dict] = None,
        xgb_params: Optional[Dict] = None,
        lgb_params: Optional[Dict] = None,
        lr_params: Optional[Dict] = None,
    ):
        """
        Args:
            voting: ``'soft'`` (average probabilities) or ``'hard'``
                (majority vote).
            calibrate: If True, wrap each base learner in
                ``CalibratedClassifierCV`` (isotonic) before ensembling.
            rf_params: Override default Random Forest hyperparameters.
            xgb_params: Override default XGBoost hyperparameters.
            lgb_params: Override default LightGBM hyperparameters.
            lr_params: Override default Logistic Regression hyperparameters.
        """
        self.voting = voting
        self.calibrate = calibrate

        self.rf_params = rf_params or {
            "n_estimators": 200,
            "max_depth": 8,
            "min_samples_leaf": 20,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }

        self.xgb_params = xgb_params or {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 1,  # set dynamically in fit()
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss",
        }

        self.lgb_params = lgb_params or {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

        self.lr_params = lr_params or {
            "C": 0.1,
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": 42,
        }

        self.ensemble_: Optional[VotingClassifier] = None
        self.feature_importances_: Optional[pd.Series] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None
        self._feature_names: Optional[List[str]] = None

    # ------------------------------------------------------------------ #
    #  Model construction helpers                                         #
    # ------------------------------------------------------------------ #

    def _build_base_estimators(
        self, scale_pos_weight: float = 1.0
    ) -> List[Tuple[str, Any]]:
        """Create fresh (unfitted) instances of each base learner."""
        xgb_params = self.xgb_params.copy()
        xgb_params["scale_pos_weight"] = scale_pos_weight

        estimators = [
            ("rf", RandomForestClassifier(**self.rf_params)),
            ("xgb", xgb.XGBClassifier(**xgb_params)),
            ("lgb", lgb.LGBMClassifier(**self.lgb_params)),
            ("lr", LogisticRegression(**self.lr_params)),
        ]
        return estimators

    def _maybe_calibrate(
        self, estimators: List[Tuple[str, Any]]
    ) -> List[Tuple[str, Any]]:
        """Optionally wrap each estimator in CalibratedClassifierCV."""
        if not self.calibrate:
            return estimators
        return [
            (
                name,
                CalibratedClassifierCV(model, cv=3, method="isotonic"),
            )
            for name, model in estimators
        ]

    # ------------------------------------------------------------------ #
    #  fit / predict / predict_proba                                      #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "TradingEnsemble":
        """
        Fit the voting ensemble on training data.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Target labels of shape ``(n_samples,)``.
            feature_names: Optional list of feature names for importance
                tracking.

        Returns:
            self

        Raises:
            ValueError: If fewer than 2 classes are present in ``y``.
        """
        self.n_features_in_ = X.shape[1]
        self._feature_names = feature_names
        self.classes_ = np.unique(y)

        if len(self.classes_) < 2:
            raise ValueError(
                f"TradingEnsemble requires at least 2 classes in training "
                f"data, got {len(self.classes_)}: {self.classes_.tolist()}"
            )

        # Dynamic class-imbalance weight for XGBoost
        n_neg = int((y == 0).sum())
        n_pos = int((y == 1).sum())
        scale_pos_weight = n_neg / max(n_pos, 1) if n_pos < n_neg else 1.0

        logger.info(
            "Class distribution: %d negative, %d positive (scale_pos_weight=%.2f)",
            n_neg,
            n_pos,
            scale_pos_weight,
        )

        estimators = self._build_base_estimators(scale_pos_weight)
        estimators = self._maybe_calibrate(estimators)

        self.ensemble_ = VotingClassifier(
            estimators=estimators,
            voting=self.voting,
            n_jobs=-1,
        )

        logger.info("Fitting ensemble (%s voting, calibrate=%s) ...", self.voting, self.calibrate)
        self.ensemble_.fit(X, y)

        self._compute_feature_importances(X, y, scale_pos_weight)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Array of predicted labels.
        """
        self._check_is_fitted()
        return self.ensemble_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Array of shape ``(n_samples, n_classes)`` with predicted
            probabilities.
        """
        self._check_is_fitted()
        return self.ensemble_.predict_proba(X)

    # ------------------------------------------------------------------ #
    #  Feature importances                                                #
    # ------------------------------------------------------------------ #

    def _compute_feature_importances(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scale_pos_weight: float,
    ) -> None:
        """
        Fit lightweight copies of tree-based models to extract and average
        feature importances.

        We deliberately fit *separate* copies rather than reaching into the
        VotingClassifier's internals, because calibration wrappers hide the
        underlying estimator's ``feature_importances_`` attribute.
        """
        importance_arrays: List[np.ndarray] = []

        for name, model in self._build_base_estimators(scale_pos_weight)[:3]:
            try:
                model.fit(X, y)
                if hasattr(model, "feature_importances_"):
                    raw = model.feature_importances_
                    # Normalize so each model's importances sum to 1.
                    # RF/XGB use gain-based (already ~1), but LightGBM
                    # defaults to split-count which can be >> 1.
                    total = raw.sum()
                    normed = raw / total if total > 0 else raw
                    importance_arrays.append(normed)
            except Exception as exc:
                logger.warning("Could not extract importances from %s: %s", name, exc)

        if not importance_arrays:
            self.feature_importances_ = None
            return

        avg = np.mean(importance_arrays, axis=0)

        if self._feature_names and len(self._feature_names) == len(avg):
            self.feature_importances_ = pd.Series(
                avg, index=self._feature_names
            ).sort_values(ascending=False)
        else:
            self.feature_importances_ = pd.Series(avg).sort_values(ascending=False)

    def get_feature_importances(self, top_n: int = 20) -> pd.Series:
        """Return the *top_n* most important features.

        Args:
            top_n: Number of features to return.

        Returns:
            Sorted ``pd.Series`` of feature importances.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if self.feature_importances_ is None:
            raise ValueError(
                "Feature importances not available. Call fit() first."
            )
        return self.feature_importances_.head(top_n)

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _check_is_fitted(self) -> None:
        if self.ensemble_ is None:
            raise RuntimeError(
                "TradingEnsemble is not fitted. Call fit() first."
            )


class MultiTargetEnsemble:
    """
    Predict multiple targets with separate ensembles:

    1. **Direction** — will the price go up? (binary)
    2. **Magnitude** — will the move be small / medium / large? (multi-class)

    Combining both gives richer signals: "high probability of a *large* up
    move" is much more actionable than a bare "BUY".

    Typical usage::

        mte = MultiTargetEnsemble()
        mte.fit(X, y_direction, y_magnitude)
        signals = mte.predict(X_new)
        print(signals['direction'], signals['direction_proba'])
    """

    def __init__(
        self,
        direction_params: Optional[Dict] = None,
        magnitude_params: Optional[Dict] = None,
    ):
        """
        Args:
            direction_params: Kwargs forwarded to the direction
                ``TradingEnsemble``.
            magnitude_params: Kwargs forwarded to the magnitude
                ``TradingEnsemble``.
        """
        self.direction_model = TradingEnsemble(**(direction_params or {}))
        self.magnitude_model = TradingEnsemble(**(magnitude_params or {}))
        self.is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y_direction: np.ndarray,
        y_magnitude: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "MultiTargetEnsemble":
        """
        Fit both direction and magnitude ensembles.

        Args:
            X: Feature matrix.
            y_direction: Binary labels (0 = down/flat, 1 = up).
            y_magnitude: Multi-class labels (e.g. 0 = small, 1 = medium,
                2 = large).
            feature_names: Optional feature names for importance tracking.

        Returns:
            self
        """
        logger.info("Fitting direction model ...")
        self.direction_model.fit(X, y_direction, feature_names=feature_names)

        logger.info("Fitting magnitude model ...")
        self.magnitude_model.fit(X, y_magnitude, feature_names=feature_names)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict direction and magnitude.

        Args:
            X: Feature matrix.

        Returns:
            Dictionary with keys ``direction``, ``direction_proba``,
            ``magnitude``, ``magnitude_proba``.
        """
        if not self.is_fitted:
            raise RuntimeError("MultiTargetEnsemble is not fitted. Call fit() first.")

        return {
            "direction": self.direction_model.predict(X),
            "direction_proba": self.direction_model.predict_proba(X),
            "magnitude": self.magnitude_model.predict(X),
            "magnitude_proba": self.magnitude_model.predict_proba(X),
        }
