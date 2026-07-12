"""Chronological validation utilities for the baseline model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from trader.config import TraderConfig, ValidationConfig
from trader.features.target import TARGET_COLUMN
from trader.modeling.baseline import BaselineLogisticModel, ModelTrainingError


@dataclass(frozen=True, slots=True)
class HoldoutSplit:
    development: pd.DataFrame
    holdout: pd.DataFrame
    holdout_start_position: int


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    fold_number: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    @property
    def train_positions(self) -> range:
        return range(self.train_start, self.train_end)

    @property
    def test_positions(self) -> range:
        return range(self.test_start, self.test_end)


def split_final_holdout(
    data: pd.DataFrame,
    validation_config: ValidationConfig,
) -> HoldoutSplit:
    """Reserve the final configured fraction before development validation."""

    if data.empty:
        raise ValueError("cannot split an empty dataset")

    holdout_rows = int(np.ceil(len(data) * validation_config.final_holdout_fraction))
    holdout_rows = min(max(holdout_rows, 1), len(data) - 1)
    holdout_start = len(data) - holdout_rows
    return HoldoutSplit(
        development=data.iloc[:holdout_start].copy(),
        holdout=data.iloc[holdout_start:].copy(),
        holdout_start_position=holdout_start,
    )


def expanding_walk_forward_folds(
    row_count: int,
    validation_config: ValidationConfig,
) -> list[WalkForwardFold]:
    """Return expanding train windows and fixed-size chronological test windows."""

    if row_count <= validation_config.minimum_train_bars:
        return []

    folds: list[WalkForwardFold] = []
    train_end = validation_config.minimum_train_bars
    fold_number = 1
    while train_end < row_count:
        test_end = min(train_end + validation_config.test_bars, row_count)
        if test_end <= train_end:
            break
        folds.append(
            WalkForwardFold(
                fold_number=fold_number,
                train_start=0,
                train_end=train_end,
                test_start=train_end,
                test_end=test_end,
            )
        )
        train_end += validation_config.step_bars
        fold_number += 1
    return folds


def walk_forward_validate(
    data: pd.DataFrame,
    config: TraderConfig,
    *,
    feature_names: tuple[str, ...],
    target_column: str = TARGET_COLUMN,
    skip_single_class: bool = True,
) -> list[dict[str, Any]]:
    """Fit a fresh baseline model for each development fold and report metrics."""

    split = split_final_holdout(data, config.validation)
    results: list[dict[str, Any]] = []
    for fold in expanding_walk_forward_folds(len(split.development), config.validation):
        train = split.development.iloc[list(fold.train_positions)]
        test = split.development.iloc[list(fold.test_positions)]
        model = BaselineLogisticModel(config, feature_names=feature_names)

        try:
            model.fit(train, target_column=target_column)
        except ModelTrainingError as exc:
            if not skip_single_class:
                raise
            results.append(_skipped_fold_result(fold, str(exc)))
            continue

        labeled_test = test.loc[test[target_column].notna()].copy()
        if labeled_test.empty:
            results.append(_skipped_fold_result(fold, "no labeled validation rows"))
            continue

        probabilities = model.predict_positive_proba(labeled_test)
        predictions = (
            probabilities >= config.model.probability_threshold
        ).astype("int8")
        y_true = labeled_test[target_column].astype("int8").to_numpy()
        results.append(
            {
                **_fold_bounds(fold, train, test),
                "status": "ok",
                "metrics": _classification_metrics(y_true, predictions),
                "probability_diagnostics": _probability_diagnostics(probabilities),
                "train_class_counts": _class_counts(train[target_column]),
                "validation_class_counts": _class_counts(labeled_test[target_column]),
                "scaler_mean": model.scaler_mean(),
            }
        )
    return results


def _classification_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }


def _probability_diagnostics(probabilities: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(probabilities)),
        "max": float(np.max(probabilities)),
        "mean": float(np.mean(probabilities)),
        "std": float(np.std(probabilities)),
    }


def _class_counts(values: pd.Series) -> dict[str, int]:
    counts = values.dropna().astype("int8").value_counts().to_dict()
    return {str(class_label): int(counts.get(class_label, 0)) for class_label in (0, 1)}


def _fold_bounds(
    fold: WalkForwardFold,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "fold": fold.fold_number,
        "train_start_position": fold.train_start,
        "train_end_position": fold.train_end - 1,
        "validation_start_position": fold.test_start,
        "validation_end_position": fold.test_end - 1,
        "train_start_timestamp": _timestamp_or_none(train, 0),
        "train_end_timestamp": _timestamp_or_none(train, -1),
        "validation_start_timestamp": _timestamp_or_none(test, 0),
        "validation_end_timestamp": _timestamp_or_none(test, -1),
    }


def _skipped_fold_result(fold: WalkForwardFold, reason: str) -> dict[str, Any]:
    return {
        "fold": fold.fold_number,
        "train_start_position": fold.train_start,
        "train_end_position": fold.train_end - 1,
        "validation_start_position": fold.test_start,
        "validation_end_position": fold.test_end - 1,
        "status": "skipped",
        "reason": reason,
    }


def _timestamp_or_none(data: pd.DataFrame, position: int) -> str | None:
    if data.empty or "timestamp" not in data.columns:
        return None
    timestamp = pd.Timestamp(data["timestamp"].iloc[position])
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat().replace("+00:00", "Z")
