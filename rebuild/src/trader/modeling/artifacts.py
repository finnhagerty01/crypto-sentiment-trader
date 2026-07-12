"""Model artifact persistence for the baseline pipeline."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from trader.config import TraderConfig
from trader.features.target import TARGET_COLUMN
from trader.modeling.baseline import (
    BaselineLogisticModel,
    FeatureCompatibilityError,
)


MODEL_ARTIFACT_SCHEMA_VERSION = "model-artifact-v1"


def save_model_artifact(
    model: BaselineLogisticModel,
    path: str | Path,
    *,
    training_data: pd.DataFrame,
    dataset_content_hash: str,
    validation_fold_metrics: list[dict[str, Any]],
    target_definition: str,
) -> dict[str, Any]:
    """Save the fitted pipeline with JSON metadata and return that metadata."""

    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = build_model_artifact_metadata(
        model,
        training_data=training_data,
        dataset_content_hash=dataset_content_hash,
        validation_fold_metrics=validation_fold_metrics,
        target_definition=target_definition,
    )
    joblib.dump(model.pipeline, artifact_path)
    metadata_path(artifact_path).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def load_model_artifact(
    path: str | Path,
    *,
    config: TraderConfig,
    expected_feature_names: tuple[str, ...],
) -> BaselineLogisticModel:
    """Load a model artifact and validate exact feature compatibility."""

    artifact_path = Path(path)
    metadata = json.loads(metadata_path(artifact_path).read_text(encoding="utf-8"))
    feature_names = tuple(metadata.get("feature_names", ()))
    if feature_names != tuple(expected_feature_names):
        raise FeatureCompatibilityError(
            "artifact feature order mismatch; expected "
            + ", ".join(expected_feature_names)
        )
    pipeline = joblib.load(artifact_path)
    return BaselineLogisticModel(config, feature_names=feature_names, pipeline=pipeline)


def build_model_artifact_metadata(
    model: BaselineLogisticModel,
    *,
    training_data: pd.DataFrame,
    dataset_content_hash: str,
    validation_fold_metrics: list[dict[str, Any]],
    target_definition: str,
) -> dict[str, Any]:
    labeled = training_data.loc[training_data[TARGET_COLUMN].notna()]
    if labeled.empty:
        raise ValueError("cannot build artifact metadata without labeled training rows")

    return {
        "schema_version": MODEL_ARTIFACT_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "training_date_range": {
            "start": _timestamp_string(labeled["timestamp"].iloc[0]),
            "end": _timestamp_string(labeled["timestamp"].iloc[-1]),
        },
        "dataset_content_hash": dataset_content_hash,
        "feature_names": list(model.feature_names),
        "config": asdict(model.config),
        "library_versions": model.metadata()["library_versions"],
        "target_definition": target_definition,
        "probability_threshold": model.config.model.probability_threshold,
        "validation_fold_metrics": validation_fold_metrics,
    }


def metadata_path(path: str | Path) -> Path:
    artifact_path = Path(path)
    return artifact_path.with_suffix(artifact_path.suffix + ".metadata.json")


def _timestamp_string(value: object) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat().replace("+00:00", "Z")
