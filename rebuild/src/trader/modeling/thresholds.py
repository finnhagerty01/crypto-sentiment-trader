"""Threshold sweep diagnostics for the baseline Logistic Regression model."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from trader.backtest.engine import run_long_cash_backtest
from trader.backtest.metrics import calculate_backtest_metrics
from trader.config import TraderConfig, ValidationConfig
from trader.features.market import model_feature_columns
from trader.features.target import TARGET_COLUMN
from trader.modeling.baseline import BaselineLogisticModel, ModelTrainingError
from trader.modeling.validation import (
    WalkForwardFold,
    expanding_walk_forward_folds,
    split_final_holdout,
)


DEFAULT_THRESHOLDS: tuple[float, ...] = (
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
)
TrainWindowPolicy = Literal[
    "expanding",
    "rolling_1000",
    "rolling_1500",
    "rolling_2000",
    "rolling_2500",
]
TRAIN_WINDOW_POLICIES: tuple[TrainWindowPolicy, ...] = (
    "expanding",
    "rolling_1000",
    "rolling_1500",
    "rolling_2000",
    "rolling_2500",
)


class ThresholdModel(Protocol):
    def fit(self, data: pd.DataFrame, *, target_column: str) -> Any:
        ...

    def predict_positive_proba(self, data: pd.DataFrame) -> np.ndarray:
        ...


class ModelFactory(Protocol):
    def __call__(
        self,
        config: TraderConfig,
        *,
        feature_names: tuple[str, ...],
    ) -> ThresholdModel:
        ...


@dataclass(frozen=True, slots=True)
class ThresholdSweepResult:
    """Complete threshold diagnostics for development folds and holdout."""

    fold_metrics: pd.DataFrame
    summary: pd.DataFrame
    selected_threshold: float | None
    holdout_metrics: dict[str, Any] | None


def run_threshold_sweep(
    data: pd.DataFrame,
    config: TraderConfig,
    *,
    feature_names: tuple[str, ...] | None = None,
    target_column: str = TARGET_COLUMN,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    model_factory: ModelFactory = BaselineLogisticModel,
    skip_single_class: bool = True,
    train_window_policy: TrainWindowPolicy = "expanding",
) -> ThresholdSweepResult:
    """Evaluate fixed probability thresholds on development folds.

    The fitted model and validation probability vector are created once per
    chronological development fold. Every threshold is then evaluated against
    that same probability vector.
    """

    _require_thresholds(thresholds)
    _train_window_bars(train_window_policy)
    selected_feature_names = (
        model_feature_columns(config) if feature_names is None else feature_names
    )
    split = split_final_holdout(data, config.validation)
    fold_rows: list[dict[str, Any]] = []

    for fold in train_window_folds(
        len(split.development),
        config.validation,
        train_window_policy=train_window_policy,
    ):
        train = split.development.iloc[list(fold.train_positions)]
        validation = split.development.iloc[list(fold.test_positions)]
        model = model_factory(config, feature_names=selected_feature_names)
        try:
            model.fit(train, target_column=target_column)
        except ModelTrainingError as exc:
            if not skip_single_class:
                raise
            fold_rows.extend(_skipped_threshold_rows(fold, thresholds, str(exc)))
            continue

        labeled_validation = validation.loc[validation[target_column].notna()].copy()
        if len(labeled_validation) < 2:
            fold_rows.extend(
                _skipped_threshold_rows(
                    fold, thresholds, "fewer than two labeled validation rows"
                )
            )
            continue

        probabilities = model.predict_positive_proba(labeled_validation)
        fold_rows.extend(
            _evaluate_thresholds_for_period(
                fold=fold,
                period="development",
                market_data=labeled_validation,
                probabilities=probabilities,
                thresholds=thresholds,
                config=config,
                target_column=target_column,
            )
        )

    fold_metrics = pd.DataFrame(fold_rows)
    summary = summarize_thresholds(fold_metrics, thresholds=thresholds)
    selected_threshold = select_threshold(summary)
    holdout_metrics = None
    if selected_threshold is not None:
        holdout_metrics = _evaluate_holdout(
            split.development,
            split.holdout,
            config,
            feature_names=selected_feature_names,
            target_column=target_column,
            threshold=selected_threshold,
            model_factory=model_factory,
            train_window_policy=train_window_policy,
        )

    return ThresholdSweepResult(
        fold_metrics=fold_metrics,
        summary=summary,
        selected_threshold=selected_threshold,
        holdout_metrics=holdout_metrics,
    )


def train_window_folds(
    row_count: int,
    validation_config: ValidationConfig,
    *,
    train_window_policy: TrainWindowPolicy = "expanding",
) -> list[WalkForwardFold]:
    """Return development folds for expanding or recent rolling train windows."""

    window_bars = _train_window_bars(train_window_policy)
    folds = expanding_walk_forward_folds(row_count, validation_config)
    if window_bars is None:
        return folds
    return [
        WalkForwardFold(
            fold_number=fold.fold_number,
            train_start=max(0, fold.train_end - window_bars),
            train_end=fold.train_end,
            test_start=fold.test_start,
            test_end=fold.test_end,
        )
        for fold in folds
    ]


def probabilities_to_signals(
    probabilities: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Convert probabilities to long/cash signals using inclusive thresholding."""

    return (np.asarray(probabilities, dtype="float64") >= threshold).astype("int8")


def summarize_thresholds(
    fold_metrics: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    """Aggregate development-fold metrics by threshold for selection."""

    rows: list[dict[str, Any]] = []
    ok_metrics = fold_metrics.loc[fold_metrics.get("status") == "ok"].copy()
    for threshold in thresholds:
        threshold_rows = ok_metrics.loc[ok_metrics["threshold"] == threshold]
        trade_count = int(threshold_rows["trade_count"].sum()) if not threshold_rows.empty else 0
        median_total_return = _median_or_nan(threshold_rows, "total_return")
        median_cash_return = _median_or_nan(threshold_rows, "cash_total_return")
        median_max_drawdown = _median_or_nan(threshold_rows, "max_drawdown")
        median_turnover = _median_or_nan(threshold_rows, "turnover")
        rows.append(
            {
                "threshold": threshold,
                "fold_count": int(len(threshold_rows)),
                "trade_count": trade_count,
                "median_total_return": median_total_return,
                "median_cash_total_return": median_cash_return,
                "median_max_drawdown": median_max_drawdown,
                "median_turnover": median_turnover,
                "mean_precision": _mean_or_nan(threshold_rows, "precision"),
                "mean_recall": _mean_or_nan(threshold_rows, "recall"),
                "mean_f1": _mean_or_nan(threshold_rows, "f1"),
                "passes_trade_filter": trade_count > 0,
                "passes_return_filter": _finite(median_total_return)
                and median_total_return >= 0.0,
                "passes_cash_filter": _finite(median_total_return)
                and _finite(median_cash_return)
                and median_total_return >= median_cash_return,
            }
        )
    summary = pd.DataFrame(rows)
    summary["eligible"] = (
        summary["passes_trade_filter"]
        & summary["passes_return_filter"]
        & summary["passes_cash_filter"]
    )
    return summary


def select_threshold(summary: pd.DataFrame) -> float | None:
    """Select a threshold using development-fold metrics only."""

    eligible = summary.loc[summary["eligible"]].copy()
    if eligible.empty:
        return None
    eligible["drawdown_magnitude"] = eligible["median_max_drawdown"].abs()
    eligible = eligible.sort_values(
        by=[
            "median_total_return",
            "drawdown_magnitude",
            "median_turnover",
            "threshold",
        ],
        ascending=[False, True, True, True],
        kind="mergesort",
    )
    return float(eligible.iloc[0]["threshold"])


def write_threshold_sweep_artifacts(
    result: ThresholdSweepResult,
    output_dir: str | Path,
) -> Path:
    """Write the threshold diagnostics artifact bundle."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    result.fold_metrics.to_csv(destination / "threshold_fold_metrics.csv", index=False)
    result.summary.to_csv(destination / "threshold_summary.csv", index=False)
    _write_json(
        destination / "selected_threshold.json",
        {
            "selected_threshold": result.selected_threshold,
            "threshold_candidates": list(DEFAULT_THRESHOLDS),
            "selection_policy": [
                "exclude zero-trade thresholds",
                "exclude negative median development total return",
                "exclude thresholds below median cash total return",
                "maximize median development total return",
                "tie-break on lower drawdown magnitude",
                "tie-break on lower turnover",
            ],
        },
    )
    _write_json(
        destination / "holdout_threshold_metrics.json",
        result.holdout_metrics or {"selected_threshold": None, "status": "not_evaluated"},
    )
    return destination


def _evaluate_holdout(
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    config: TraderConfig,
    *,
    feature_names: tuple[str, ...],
    target_column: str,
    threshold: float,
    model_factory: ModelFactory,
    train_window_policy: TrainWindowPolicy,
) -> dict[str, Any]:
    train = _holdout_train_window(development, train_window_policy=train_window_policy)
    model = model_factory(config, feature_names=feature_names)
    model.fit(train, target_column=target_column)
    probabilities = model.predict_positive_proba(holdout)
    metrics = _evaluate_period(
        market_data=holdout,
        probabilities=probabilities,
        threshold=threshold,
        config=config,
        target_column=target_column,
    )
    return {
        "status": "ok",
        "selected_threshold": threshold,
        **metrics,
    }


def _evaluate_thresholds_for_period(
    *,
    fold: WalkForwardFold,
    period: str,
    market_data: pd.DataFrame,
    probabilities: np.ndarray,
    thresholds: tuple[float, ...],
    config: TraderConfig,
    target_column: str,
) -> list[dict[str, Any]]:
    rows = []
    for threshold in thresholds:
        rows.append(
            {
                "fold": fold.fold_number,
                "period": period,
                "threshold": threshold,
                "status": "ok",
                **_evaluate_period(
                    market_data=market_data,
                    probabilities=probabilities,
                    threshold=threshold,
                    config=config,
                    target_column=target_column,
                ),
            }
        )
    return rows


def _evaluate_period(
    *,
    market_data: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    config: TraderConfig,
    target_column: str,
) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype="float64")
    if len(probabilities) != len(market_data):
        raise ValueError("probability count must match market rows")
    signals = probabilities_to_signals(probabilities, threshold)
    predictions = pd.DataFrame(
        {
            "timestamp": market_data["timestamp"],
            "signal": signals,
            "probability": probabilities,
        }
    )
    backtest = run_long_cash_backtest(
        market_data,
        predictions,
        backtest_config=config.backtest,
        costs_config=config.costs,
    )
    metrics = calculate_backtest_metrics(backtest.equity, backtest.trades)
    cash = run_long_cash_backtest(
        market_data,
        pd.DataFrame({"timestamp": market_data["timestamp"], "signal": 0}),
        backtest_config=config.backtest,
        costs_config=config.costs,
    )
    labeled_mask = market_data[target_column].notna().to_numpy()
    classification = _classification_metrics(
        market_data.loc[labeled_mask, target_column].astype("int8").to_numpy(),
        signals[labeled_mask],
    )
    return {
        **classification,
        "probability_min": float(np.min(probabilities)),
        "probability_max": float(np.max(probabilities)),
        "probability_mean": float(np.mean(probabilities)),
        "probability_std": float(np.std(probabilities)),
        "signal_count": int(signals.sum()),
        "cash_total_return": calculate_backtest_metrics(cash.equity, cash.trades)[
            "total_return"
        ],
        **metrics,
    }


def _classification_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    if len(y_true) == 0:
        return {
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
        }
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }


def _skipped_threshold_rows(
    fold: WalkForwardFold,
    thresholds: tuple[float, ...],
    reason: str,
) -> list[dict[str, Any]]:
    return [
        {
            "fold": fold.fold_number,
            "period": "development",
            "threshold": threshold,
            "status": "skipped",
            "reason": reason,
        }
        for threshold in thresholds
    ]


def _require_thresholds(thresholds: tuple[float, ...]) -> None:
    if thresholds != DEFAULT_THRESHOLDS:
        raise ValueError("threshold sweep must evaluate the fixed default candidates")


def _holdout_train_window(
    development: pd.DataFrame,
    *,
    train_window_policy: TrainWindowPolicy,
) -> pd.DataFrame:
    window_bars = _train_window_bars(train_window_policy)
    if window_bars is None:
        return development
    train_start = max(0, len(development) - window_bars)
    return development.iloc[train_start:].copy()


def _train_window_bars(train_window_policy: TrainWindowPolicy) -> int | None:
    if train_window_policy == "expanding":
        return None
    prefix = "rolling_"
    if (
        not isinstance(train_window_policy, str)
        or not train_window_policy.startswith(prefix)
    ):
        raise ValueError(f"invalid train-window policy: {train_window_policy}")
    try:
        window_bars = int(train_window_policy.removeprefix(prefix))
    except ValueError as exc:
        raise ValueError(
            f"invalid train-window policy: {train_window_policy}"
        ) from exc
    if train_window_policy not in TRAIN_WINDOW_POLICIES:
        raise ValueError(f"invalid train-window policy: {train_window_policy}")
    return window_bars


def _median_or_nan(data: pd.DataFrame, column: str) -> float:
    if data.empty or column not in data:
        return float("nan")
    return float(data[column].median())


def _mean_or_nan(data: pd.DataFrame, column: str) -> float:
    if data.empty or column not in data:
        return float("nan")
    return float(data[column].mean())


def _finite(value: float) -> bool:
    return bool(np.isfinite(value))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
