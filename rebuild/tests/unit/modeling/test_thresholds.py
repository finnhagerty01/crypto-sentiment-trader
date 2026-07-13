from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from trader.config import (
    BacktestConfig,
    CostsConfig,
    DataConfig,
    FeaturesConfig,
    ModelConfig,
    TargetConfig,
    TraderConfig,
    ValidationConfig,
)
from trader.features.market import MODEL_FEATURE_COLUMNS
from trader.modeling.thresholds import (
    DEFAULT_THRESHOLDS,
    probabilities_to_signals,
    run_threshold_sweep,
    select_threshold,
    summarize_thresholds,
)

from .testing_data import synthetic_feature_dataset


def test_thresholds_are_evaluated_from_same_probability_vector() -> None:
    calls = {"predict": 0}

    class FakeModel:
        def __init__(self, config: TraderConfig, *, feature_names: tuple[str, ...]) -> None:
            self.config = config
            self.feature_names = feature_names

        def fit(self, data: pd.DataFrame, *, target_column: str) -> "FakeModel":
            return self

        def predict_positive_proba(self, data: pd.DataFrame) -> np.ndarray:
            calls["predict"] += 1
            return np.linspace(0.10, 0.55, len(data))

    result = run_threshold_sweep(
        _declining_dataset(50),
        _config(),
        model_factory=FakeModel,
    )

    ok_rows = result.fold_metrics.loc[result.fold_metrics["status"] == "ok"]
    assert calls["predict"] == ok_rows["fold"].nunique()
    for _, fold_rows in ok_rows.groupby("fold"):
        assert fold_rows["probability_min"].nunique() == 1
        assert fold_rows["probability_max"].nunique() == 1
        assert fold_rows["probability_mean"].nunique() == 1


def test_threshold_comparison_is_deterministic() -> None:
    summary = pd.DataFrame(
        [
            _summary_row(0.20, total_return=0.05, max_drawdown=-0.20, turnover=2.0),
            _summary_row(0.30, total_return=0.05, max_drawdown=-0.05, turnover=3.0),
            _summary_row(0.40, total_return=0.05, max_drawdown=-0.05, turnover=1.0),
        ]
    )

    assert select_threshold(summary) == pytest.approx(0.40)


def test_zero_trade_thresholds_are_excluded_from_selection() -> None:
    fold_metrics = pd.DataFrame(
        [
            _fold_row(0.10, total_return=0.20, trade_count=0, turnover=0.0),
            _fold_row(0.15, total_return=0.10, trade_count=2, turnover=1.0),
        ]
    )

    summary = summarize_thresholds(fold_metrics)

    assert summary.loc[summary["threshold"] == 0.10, "eligible"].item() is False
    assert select_threshold(summary) == pytest.approx(0.15)


def test_final_holdout_is_not_used_during_threshold_selection() -> None:
    config = replace(
        _config(),
        costs=CostsConfig(fee_per_side=0.000001, slippage_per_side=0.000001),
    )
    result = run_threshold_sweep(
        _selection_dataset(),
        config,
        model_factory=DevelopmentOnlySignalModel,
    )

    assert result.selected_threshold == pytest.approx(0.15)
    assert result.holdout_metrics is not None
    assert result.holdout_metrics["selected_threshold"] == pytest.approx(0.15)


def test_rolling_holdout_fit_uses_recent_development_rows_only() -> None:
    fit_windows: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []

    class RecordingModel:
        def __init__(self, config: TraderConfig, *, feature_names: tuple[str, ...]) -> None:
            self.config = config
            self.feature_names = feature_names

        def fit(self, data: pd.DataFrame, *, target_column: str) -> "RecordingModel":
            fit_windows.append(
                (
                    pd.Timestamp(data["timestamp"].iloc[0]),
                    pd.Timestamp(data["timestamp"].iloc[-1]),
                    len(data),
                )
            )
            return self

        def predict_positive_proba(self, data: pd.DataFrame) -> np.ndarray:
            return np.full(len(data), 0.55)

    data = synthetic_feature_dataset(1400)
    config = replace(
        _config(),
        costs=CostsConfig(fee_per_side=0.000001, slippage_per_side=0.000001),
    )

    result = run_threshold_sweep(
        data,
        config,
        model_factory=RecordingModel,
        train_window_policy="rolling_1000",
    )

    assert result.selected_threshold is not None
    holdout_fit = fit_windows[-1]
    assert holdout_fit[0] == pd.Timestamp(data["timestamp"].iloc[120])
    assert holdout_fit[1] == pd.Timestamp(data["timestamp"].iloc[1119])
    assert holdout_fit[2] == 1000


def test_selected_threshold_is_none_when_all_thresholds_fail_policy() -> None:
    result = run_threshold_sweep(
        _declining_dataset(50),
        _config(),
        model_factory=AlwaysLongModel,
    )

    assert result.selected_threshold is None
    assert result.holdout_metrics is None


def test_backtest_signals_use_inclusive_threshold() -> None:
    probabilities = np.array([0.49, 0.50, 0.51])

    assert probabilities_to_signals(probabilities, 0.50).tolist() == [0, 1, 1]


def _config() -> TraderConfig:
    return TraderConfig(
        data=DataConfig(symbol="BTCUSDT", interval="1h"),
        features=FeaturesConfig(
            volatility_window=24,
            volume_window=24,
            rsi_window=14,
            clipping_window=168,
            clipping_mad_multiplier=8.0,
        ),
        target=TargetConfig(
            horizon_bars=1,
            cost_buffer="round_trip",
            volatility_multiplier=0.10,
        ),
        model=ModelConfig(probability_threshold=0.55, regularization_c=1.0),
        validation=ValidationConfig(
            minimum_train_bars=20,
            test_bars=10,
            step_bars=10,
            final_holdout_fraction=0.20,
        ),
        costs=CostsConfig(fee_per_side=0.001, slippage_per_side=0.001),
        backtest=BacktestConfig(initial_capital=10000.0),
    )


def _declining_dataset(row_count: int) -> pd.DataFrame:
    data = synthetic_feature_dataset(row_count)
    prices = 100.0 - np.arange(row_count, dtype=float)
    data["open"] = prices
    data["high"] = prices + 1.0
    data["low"] = prices - 1.0
    data["close"] = prices
    return data


def _selection_dataset() -> pd.DataFrame:
    data = synthetic_feature_dataset(80)
    pattern = np.array([100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 105.0, 100.0, 95.0, 90.0])
    prices = np.resize(pattern, len(data))
    prices[64:] = np.linspace(100.0, 120.0, len(data) - 64)
    data["open"] = prices
    data["close"] = prices
    data["high"] = data["close"] + 1.0
    data["low"] = data["close"] - 1.0
    return data


def _fold_row(
    threshold: float,
    *,
    total_return: float,
    trade_count: int,
    turnover: float,
) -> dict[str, Any]:
    return {
        "fold": 1,
        "period": "development",
        "threshold": threshold,
        "status": "ok",
        "total_return": total_return,
        "cash_total_return": 0.0,
        "max_drawdown": -0.01,
        "turnover": turnover,
        "trade_count": trade_count,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }


def _summary_row(
    threshold: float,
    *,
    total_return: float,
    max_drawdown: float,
    turnover: float,
) -> dict[str, Any]:
    return {
        "threshold": threshold,
        "eligible": True,
        "median_total_return": total_return,
        "median_max_drawdown": max_drawdown,
        "median_turnover": turnover,
    }


class AlwaysLongModel:
    def __init__(self, config: TraderConfig, *, feature_names: tuple[str, ...]) -> None:
        self.config = config
        self.feature_names = feature_names

    def fit(self, data: pd.DataFrame, *, target_column: str) -> "AlwaysLongModel":
        return self

    def predict_positive_proba(self, data: pd.DataFrame) -> np.ndarray:
        return np.full(len(data), 0.55)


class DevelopmentOnlySignalModel:
    def __init__(self, config: TraderConfig, *, feature_names: tuple[str, ...]) -> None:
        self.config = config
        self.feature_names = feature_names

    def fit(self, data: pd.DataFrame, *, target_column: str) -> "DevelopmentOnlySignalModel":
        return self

    def predict_positive_proba(self, data: pd.DataFrame) -> np.ndarray:
        timestamps = pd.to_datetime(data["timestamp"], utc=True)
        probabilities = np.full(len(data), 0.10)
        positions = (
            (timestamps - pd.Timestamp("2026-01-01T00:00:00Z")).dt.total_seconds()
            // 3600
        ).astype(int)
        development_window = positions < 64
        early_block = positions % 10 < 5
        probabilities[(development_window & early_block).to_numpy()] = 0.50
        return probabilities
