from __future__ import annotations

import json

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
from trader.modeling.artifacts import (
    MODEL_ARTIFACT_SCHEMA_VERSION,
    load_model_artifact,
    metadata_path,
    save_model_artifact,
)
from trader.modeling.baseline import BaselineLogisticModel, FeatureCompatibilityError

from .testing_data import synthetic_feature_dataset


def test_saved_and_loaded_models_produce_identical_probabilities(tmp_path) -> None:
    config = _config()
    data = synthetic_feature_dataset(100)
    model = BaselineLogisticModel(config).fit(data)
    artifact_path = tmp_path / "baseline.joblib"

    metadata = save_model_artifact(
        model,
        artifact_path,
        training_data=data,
        dataset_content_hash="abc123",
        validation_fold_metrics=[{"fold": 1, "status": "ok"}],
        target_definition="1 when next_return exceeds noise_band",
    )
    loaded = load_model_artifact(
        artifact_path,
        config=config,
        expected_feature_names=MODEL_FEATURE_COLUMNS,
    )

    np.testing.assert_allclose(
        model.predict_positive_proba(data.iloc[20:40]),
        loaded.predict_positive_proba(data.iloc[20:40]),
    )
    assert metadata["schema_version"] == MODEL_ARTIFACT_SCHEMA_VERSION
    assert metadata["dataset_content_hash"] == "abc123"
    assert metadata["feature_names"] == list(MODEL_FEATURE_COLUMNS)
    assert metadata["target_definition"] == "1 when next_return exceeds noise_band"


def test_feature_order_mismatch_is_rejected(tmp_path) -> None:
    config = _config()
    data = synthetic_feature_dataset(80)
    model = BaselineLogisticModel(config).fit(data)
    artifact_path = tmp_path / "baseline.joblib"
    save_model_artifact(
        model,
        artifact_path,
        training_data=data,
        dataset_content_hash="abc123",
        validation_fold_metrics=[],
        target_definition="target",
    )

    with pytest.raises(FeatureCompatibilityError, match="feature order mismatch"):
        load_model_artifact(
            artifact_path,
            config=config,
            expected_feature_names=tuple(reversed(MODEL_FEATURE_COLUMNS)),
        )


def test_metadata_sidecar_contains_required_fields(tmp_path) -> None:
    config = _config()
    data = synthetic_feature_dataset(80)
    model = BaselineLogisticModel(config).fit(data)
    artifact_path = tmp_path / "baseline.joblib"

    save_model_artifact(
        model,
        artifact_path,
        training_data=data,
        dataset_content_hash="abc123",
        validation_fold_metrics=[],
        target_definition="target",
    )
    sidecar = json.loads(metadata_path(artifact_path).read_text(encoding="utf-8"))

    assert sidecar["schema_version"] == MODEL_ARTIFACT_SCHEMA_VERSION
    assert "created_at" in sidecar
    assert sidecar["training_date_range"]["start"].endswith("Z")
    assert sidecar["training_date_range"]["end"].endswith("Z")
    assert "library_versions" in sidecar
    assert sidecar["config"]["data"]["symbol"] == "BTCUSDT"


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
