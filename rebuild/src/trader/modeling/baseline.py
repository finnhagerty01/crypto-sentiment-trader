"""Regularized Logistic Regression baseline model."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from trader.config import TraderConfig
from trader.features.market import MODEL_FEATURE_COLUMNS, model_feature_columns
from trader.features.target import TARGET_COLUMN


RANDOM_STATE = 42


class ModelTrainingError(ValueError):
    """Raised when the baseline model cannot be fit safely."""


class FeatureCompatibilityError(ValueError):
    """Raised when model inputs do not match the fitted feature contract."""


class BaselineLogisticModel:
    """Small sklearn pipeline wrapper for the market-only baseline."""

    def __init__(
        self,
        config: TraderConfig,
        *,
        feature_names: tuple[str, ...] | None = None,
        pipeline: Pipeline | None = None,
    ) -> None:
        self.config = config
        self._feature_names = (
            model_feature_columns(config)
            if feature_names is None
            else tuple(feature_names)
        )
        self.pipeline = pipeline if pipeline is not None else _build_pipeline(config)
        self._is_fit = pipeline is not None

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    def fit(
        self,
        data: pd.DataFrame,
        *,
        target_column: str = TARGET_COLUMN,
    ) -> "BaselineLogisticModel":
        """Fit the pipeline on labeled rows from a Phase 05 feature dataset."""

        training_data = _labeled_rows(data, target_column=target_column)
        y = training_data[target_column].astype("int8")
        if y.nunique(dropna=True) != 2:
            raise ModelTrainingError(
                "baseline logistic model requires both target classes in training data"
            )

        x = self._feature_frame(training_data)
        self.pipeline.fit(x, y)
        self._is_fit = True
        return self

    def predict_positive_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Return positive-class probabilities for the supplied rows."""

        self._require_fit()
        probabilities = self.pipeline.predict_proba(self._feature_frame(data))[:, 1]
        if np.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise RuntimeError("model produced probabilities outside [0, 1]")
        return probabilities

    def predict_signals(self, data: pd.DataFrame) -> np.ndarray:
        """Return 1/0 long-or-cash entry signals from configured threshold."""

        probabilities = self.predict_positive_proba(data)
        return (probabilities >= self.config.model.probability_threshold).astype("int8")

    def metadata(self) -> dict[str, Any]:
        """Return model-level metadata that does not depend on a saved dataset."""

        self._require_fit()
        classifier = self.pipeline.named_steps["classifier"]
        return {
            "model_type": "LogisticRegression",
            "feature_names": list(self._feature_names),
            "probability_threshold": self.config.model.probability_threshold,
            "regularization_c": self.config.model.regularization_c,
            "random_state": RANDOM_STATE,
            "pipeline_steps": list(self.pipeline.named_steps),
            "config": asdict(self.config),
            "library_versions": _library_versions(),
            "classifier": {
                "classes": classifier.classes_.tolist(),
                "n_iter": classifier.n_iter_.tolist(),
            },
        }

    def scaler_mean(self) -> list[float]:
        """Return fitted scaler means for tests and diagnostics."""

        self._require_fit()
        return self.pipeline.named_steps["scaler"].mean_.tolist()

    def _feature_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in self._feature_names if column not in data.columns]
        if missing:
            raise FeatureCompatibilityError(
                "missing model feature column(s): " + ", ".join(missing)
            )
        return data.loc[:, self._feature_names].astype("float64")

    def _require_fit(self) -> None:
        if not self._is_fit:
            raise ModelTrainingError("baseline logistic model has not been fit")


def _build_pipeline(config: TraderConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=config.model.regularization_c,
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                    max_iter=1000,
                ),
            ),
        ]
    )


def _labeled_rows(data: pd.DataFrame, *, target_column: str) -> pd.DataFrame:
    if target_column not in data.columns:
        raise ModelTrainingError(f"missing target column: {target_column}")
    labeled = data.loc[data[target_column].notna()].copy()
    if labeled.empty:
        raise ModelTrainingError("no labeled rows are available for training")
    return labeled


def _library_versions() -> dict[str, str]:
    import joblib
    import sklearn

    return {
        "joblib": joblib.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
