from __future__ import annotations

from dataclasses import replace

import numpy as np
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
from trader.modeling.baseline import BaselineLogisticModel, ModelTrainingError

from .testing_data import synthetic_feature_dataset


def test_probability_outputs_are_bounded() -> None:
    config = _config()
    data = synthetic_feature_dataset(80)
    model = BaselineLogisticModel(config).fit(data)

    probabilities = model.predict_positive_proba(data.iloc[:10])
    signals = model.predict_signals(data.iloc[:10])

    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    assert set(signals.tolist()) <= {0, 1}


def test_single_class_training_fails_clearly() -> None:
    data = synthetic_feature_dataset(40)
    data["target"] = 0

    with pytest.raises(ModelTrainingError, match="requires both target classes"):
        BaselineLogisticModel(_config()).fit(data)


def test_repeated_training_is_deterministic() -> None:
    config = _config()
    data = synthetic_feature_dataset(100)

    first = BaselineLogisticModel(config).fit(data)
    second = BaselineLogisticModel(config).fit(data)

    np.testing.assert_allclose(
        first.predict_positive_proba(data.iloc[10:40]),
        second.predict_positive_proba(data.iloc[10:40]),
    )


def test_model_exposes_feature_names_and_metadata() -> None:
    config = _config()
    model = BaselineLogisticModel(config).fit(synthetic_feature_dataset(60))

    metadata = model.metadata()

    assert model.feature_names == MODEL_FEATURE_COLUMNS
    assert metadata["model_type"] == "LogisticRegression"
    assert metadata["probability_threshold"] == config.model.probability_threshold
    assert metadata["feature_names"] == list(MODEL_FEATURE_COLUMNS)


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
        target=TargetConfig(horizon_bars=1, volatility_multiplier=0.10),
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


def test_threshold_changes_signal_generation() -> None:
    data = synthetic_feature_dataset(80)
    loose = BaselineLogisticModel(
        replace(_config(), model=ModelConfig(probability_threshold=0.10, regularization_c=1.0))
    ).fit(data)
    strict = BaselineLogisticModel(
        replace(_config(), model=ModelConfig(probability_threshold=0.90, regularization_c=1.0))
    ).fit(data)

    assert loose.predict_signals(data).sum() >= strict.predict_signals(data).sum()
