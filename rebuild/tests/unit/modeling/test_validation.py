from __future__ import annotations

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
from trader.modeling.validation import (
    expanding_walk_forward_folds,
    split_final_holdout,
    walk_forward_validate,
)

from .testing_data import synthetic_feature_dataset


def test_training_data_always_precedes_validation_data() -> None:
    config = _config()
    folds = expanding_walk_forward_folds(70, config.validation)

    assert folds
    for fold in folds:
        assert fold.train_start == 0
        assert fold.train_end <= fold.test_start
        assert fold.test_start < fold.test_end


def test_holdout_rows_never_appear_in_development_folds() -> None:
    config = _config()
    data = synthetic_feature_dataset(100)
    split = split_final_holdout(data, config.validation)
    folds = expanding_walk_forward_folds(len(split.development), config.validation)

    assert len(split.holdout) == 20
    assert split.holdout["timestamp"].min() > split.development["timestamp"].max()
    for fold in folds:
        assert fold.test_end <= split.holdout_start_position


def test_scaler_statistics_are_fitted_per_training_fold() -> None:
    config = _config()
    data = synthetic_feature_dataset(90)

    results = walk_forward_validate(
        data,
        config,
        feature_names=MODEL_FEATURE_COLUMNS,
    )
    ok_results = [result for result in results if result["status"] == "ok"]

    assert len(ok_results) >= 2
    assert ok_results[0]["scaler_mean"] != ok_results[1]["scaler_mean"]


def test_walk_forward_reports_metrics_and_probability_diagnostics() -> None:
    config = _config()
    data = synthetic_feature_dataset(80)

    results = walk_forward_validate(
        data,
        config,
        feature_names=MODEL_FEATURE_COLUMNS,
    )

    assert any(result["status"] == "ok" for result in results)
    first_ok = next(result for result in results if result["status"] == "ok")
    assert set(first_ok["metrics"]) == {"accuracy", "precision", "recall", "f1"}
    assert set(first_ok["probability_diagnostics"]) == {"min", "max", "mean", "std"}
    assert 0 <= first_ok["probability_diagnostics"]["min"] <= 1
    assert 0 <= first_ok["probability_diagnostics"]["max"] <= 1


def test_single_class_fold_can_be_skipped_explicitly() -> None:
    config = _config()
    data = synthetic_feature_dataset(60)
    data.loc[data.index[: config.validation.minimum_train_bars], "target"] = 0

    results = walk_forward_validate(
        data,
        config,
        feature_names=MODEL_FEATURE_COLUMNS,
        skip_single_class=True,
    )

    assert results[0]["status"] == "skipped"
    assert "requires both target classes" in results[0]["reason"]


def test_single_class_fold_can_fail_explicitly() -> None:
    config = _config()
    data = synthetic_feature_dataset(60)
    data.loc[data.index[: config.validation.minimum_train_bars], "target"] = 0

    with pytest.raises(ValueError, match="requires both target classes"):
        walk_forward_validate(
            data,
            config,
            feature_names=MODEL_FEATURE_COLUMNS,
            skip_single_class=False,
        )


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
        costs=CostsConfig(fee_per_side=0.001, slippage_per_side=0.0005),
        backtest=BacktestConfig(initial_capital=10000.0),
    )
