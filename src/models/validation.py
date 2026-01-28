# src/models/validation.py
"""
Walk-forward validation for time series trading models.

Standard cross-validation shuffles data randomly, which causes future data
to leak into training sets. These validators enforce strict temporal ordering:
training data is ALWAYS before test data, with optional purge gaps to prevent
label overlap from forward-looking targets (e.g., 24h future returns).
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Generator, Dict, Any, Optional
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
import logging

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Walk-forward validation for trading strategies.

    Unlike standard CV, this:
    1. Always trains on past, tests on future (no temporal leakage)
    2. Supports expanding or rolling training windows
    3. Simulates live trading conditions where the model is periodically retrained

    Typical usage::

        validator = WalkForwardValidator(min_train_size=500, test_size=168)
        results = validator.validate(model, df, feature_cols)
        print(results['overall']['precision'])
    """

    def __init__(
        self,
        min_train_size: int = 500,
        test_size: int = 168,
        step_size: int = 24,
        expanding: bool = True,
        purge_gap: int = 0,
    ):
        """
        Args:
            min_train_size: Minimum rows for the initial training window.
            test_size: Number of rows in each test fold.
            step_size: How many rows to advance the window each fold.
            expanding: If True, training window grows over time (anchored at
                row 0). If False, uses a fixed-size rolling window.
            purge_gap: Number of rows to drop between the end of training and
                the start of testing. Prevents label leakage when the target
                is computed from future prices (e.g., a 24-hour forward return
                means the last 24 training rows overlap with the first test
                rows' label horizon).
        """
        if min_train_size < 1:
            raise ValueError("min_train_size must be >= 1")
        if test_size < 1:
            raise ValueError("test_size must be >= 1")
        if step_size < 1:
            raise ValueError("step_size must be >= 1")
        if purge_gap < 0:
            raise ValueError("purge_gap must be >= 0")

        self.min_train_size = min_train_size
        self.test_size = test_size
        self.step_size = step_size
        self.expanding = expanding
        self.purge_gap = purge_gap

    def split(
        self, n_samples: int
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test index arrays for walk-forward validation.

        The data MUST already be sorted in chronological order before calling
        this method.  The caller is responsible for ensuring temporal ordering.

        Args:
            n_samples: Total number of rows in the dataset.

        Yields:
            (train_indices, test_indices) — both are 1-D int arrays.
        """
        if n_samples < self.min_train_size + self.purge_gap + self.test_size:
            logger.warning(
                "Dataset too small for even one fold: n_samples=%d, "
                "need >= %d (min_train + purge_gap + test_size)",
                n_samples,
                self.min_train_size + self.purge_gap + self.test_size,
            )
            return

        train_end = self.min_train_size

        while train_end + self.purge_gap + self.test_size <= n_samples:
            # --- training indices ---
            if self.expanding:
                train_start = 0
            else:
                train_start = max(0, train_end - self.min_train_size)

            train_idx = np.arange(train_start, train_end)

            # --- test indices (after purge gap) ---
            test_start = train_end + self.purge_gap
            test_end = min(test_start + self.test_size, n_samples)
            test_idx = np.arange(test_start, test_end)

            yield train_idx, test_idx

            train_end += self.step_size

    def validate(
        self,
        model: BaseEstimator,
        df: pd.DataFrame,
        features: List[str],
        target: str = "target",
    ) -> Dict[str, Any]:
        """
        Run walk-forward validation and return aggregate + per-fold metrics.

        The model is **cloned** before each fold so that no state leaks across
        folds.

        Args:
            model: Any scikit-learn-compatible estimator (must implement
                ``fit``, ``predict``, and optionally ``predict_proba``).
            df: DataFrame that is **already sorted chronologically**.
            features: Column names to use as input features.
            target: Column name for the classification target.

        Returns:
            Dictionary with keys:
                - ``overall``: aggregate precision / recall / f1 / accuracy
                - ``per_fold``: list of per-fold metric dicts
                - ``predictions``: array of all test predictions
                - ``actuals``: array of all test labels
                - ``probabilities``: array of predicted probabilities (if
                  the model supports ``predict_proba``), else same as
                  predictions
        """
        missing_cols = [c for c in features if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")
        if target not in df.columns:
            raise ValueError(f"Missing target column: {target}")

        X = df[features].values
        y = df[target].values
        n_samples = len(df)

        all_predictions: List[np.ndarray] = []
        all_actuals: List[np.ndarray] = []
        all_probas: List[np.ndarray] = []
        fold_metrics: List[Dict[str, Any]] = []

        for fold_idx, (train_idx, test_idx) in enumerate(self.split(n_samples)):
            # -- Leakage guard: assert train is strictly before test --
            assert train_idx.max() < test_idx.min(), (
                f"Fold {fold_idx}: training data overlaps with test data! "
                f"train_max={train_idx.max()}, test_min={test_idx.min()}"
            )

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Skip folds where training data has fewer than 2 classes.
            # Classifiers cannot learn from a single class, and this can
            # legitimately happen in time series data (e.g. flat price
            # windows where all returns are in the same bucket).
            if len(np.unique(y_train)) < 2:
                logger.warning(
                    "Fold %d skipped: training data has only %d class(es)",
                    fold_idx,
                    len(np.unique(y_train)),
                )
                continue

            # Clone model so folds are independent
            fold_model = clone(model)
            fold_model.fit(X_train, y_train)

            predictions = fold_model.predict(X_test)

            if hasattr(fold_model, "predict_proba"):
                probas = fold_model.predict_proba(X_test)
                # Store the positive-class probability if binary
                if probas.ndim == 2 and probas.shape[1] == 2:
                    probas = probas[:, 1]
            else:
                probas = predictions.astype(float)

            all_predictions.append(predictions)
            all_actuals.append(y_test)
            all_probas.append(probas)

            fold_result = {
                "fold": fold_idx,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "train_end_idx": int(train_idx[-1]),
                "test_start_idx": int(test_idx[0]),
                "precision": precision_score(
                    y_test, predictions, zero_division=0, average="weighted"
                ),
                "recall": recall_score(
                    y_test, predictions, zero_division=0, average="weighted"
                ),
                "f1": f1_score(
                    y_test, predictions, zero_division=0, average="weighted"
                ),
                "accuracy": accuracy_score(y_test, predictions),
            }
            fold_metrics.append(fold_result)

            logger.info(
                "Fold %d: train[0:%d] -> test[%d:%d]  "
                "precision=%.3f  recall=%.3f  f1=%.3f",
                fold_idx,
                fold_result["train_end_idx"],
                fold_result["test_start_idx"],
                fold_result["test_start_idx"] + fold_result["test_size"],
                fold_result["precision"],
                fold_result["recall"],
                fold_result["f1"],
            )

        if not fold_metrics:
            logger.warning("No folds were generated — dataset may be too small.")
            return {
                "overall": {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "accuracy": 0.0,
                    "n_samples": 0,
                    "n_folds": 0,
                },
                "per_fold": [],
                "predictions": np.array([]),
                "actuals": np.array([]),
                "probabilities": np.array([]),
            }

        concat_predictions = np.concatenate(all_predictions)
        concat_actuals = np.concatenate(all_actuals)
        concat_probas = np.concatenate(all_probas)

        return {
            "overall": {
                "precision": precision_score(
                    concat_actuals,
                    concat_predictions,
                    zero_division=0,
                    average="weighted",
                ),
                "recall": recall_score(
                    concat_actuals,
                    concat_predictions,
                    zero_division=0,
                    average="weighted",
                ),
                "f1": f1_score(
                    concat_actuals,
                    concat_predictions,
                    zero_division=0,
                    average="weighted",
                ),
                "accuracy": accuracy_score(concat_actuals, concat_predictions),
                "n_samples": len(concat_actuals),
                "n_folds": len(fold_metrics),
            },
            "per_fold": fold_metrics,
            "predictions": concat_predictions,
            "actuals": concat_actuals,
            "probabilities": concat_probas,
        }


class PurgedKFold:
    """
    K-Fold cross-validation with purge gaps to prevent lookahead bias.

    Standard KFold assigns contiguous blocks to each fold, but features
    built from lagged data (e.g., 24-hour rolling averages) and labels built
    from future data (e.g., next-hour return) can leak across fold boundaries.

    PurgedKFold drops ``purge_gap`` rows on **both sides** of each test fold
    so that:
    - Training rows immediately *before* the test fold cannot have labels
      that depend on prices inside the test fold.
    - Training rows immediately *after* the test fold cannot have features
      that depend on data inside the test fold.

    Typical usage::

        pkf = PurgedKFold(n_splits=5, purge_gap=24)
        for train_idx, test_idx in pkf.split(n_samples):
            model.fit(X[train_idx], y[train_idx])
            score = model.score(X[test_idx], y[test_idx])
    """

    def __init__(self, n_splits: int = 5, purge_gap: int = 24):
        """
        Args:
            n_splits: Number of folds.
            purge_gap: Number of rows to purge on each side of the test fold.
        """
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if purge_gap < 0:
            raise ValueError("purge_gap must be >= 0")

        self.n_splits = n_splits
        self.purge_gap = purge_gap

    def split(
        self, n_samples: int
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate purged train/test index arrays.

        Args:
            n_samples: Total number of rows (must be sorted chronologically).

        Yields:
            (train_indices, test_indices)
        """
        if n_samples < self.n_splits:
            raise ValueError(
                f"n_samples ({n_samples}) must be >= n_splits ({self.n_splits})"
            )

        fold_size = n_samples // self.n_splits

        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples

            # Training = everything outside (test ± purge_gap)
            purge_start = max(0, test_start - self.purge_gap)
            purge_end = min(n_samples, test_end + self.purge_gap)

            train_before = np.arange(0, purge_start)
            train_after = np.arange(purge_end, n_samples)
            train_idx = np.concatenate([train_before, train_after])

            test_idx = np.arange(test_start, test_end)

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx.astype(int), test_idx.astype(int)
