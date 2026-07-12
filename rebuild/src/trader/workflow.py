"""End-to-end workflow helpers for the reproducible baseline CLI."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from trader.backtest.benchmarks import run_benchmarks
from trader.backtest.engine import BacktestResult, run_long_cash_backtest
from trader.backtest.metrics import calculate_backtest_metrics
from trader.config import TraderConfig
from trader.data.storage import (
    build_metadata,
    metadata_path as dataset_metadata_path,
    read_market_dataset,
)
from trader.features.market import MODEL_FEATURE_COLUMNS, build_feature_dataset
from trader.modeling.artifacts import (
    load_model_artifact,
    metadata_path as model_metadata_path,
    save_model_artifact,
)
from trader.modeling.baseline import BaselineLogisticModel
from trader.modeling.validation import split_final_holdout, walk_forward_validate
from trader.reporting.writer import write_backtest_report


FEATURE_DATASET_SCHEMA_VERSION = "feature-dataset-v1"
TARGET_DEFINITION = (
    "target=1 when next_return > 2*(fee_per_side+slippage_per_side) "
    "+ volatility_multiplier*realized_volatility_24h; else 0; unlabeled when "
    "future close or volatility is unavailable"
)


class WorkflowError(ValueError):
    """Raised when a CLI workflow cannot produce a trustworthy artifact."""


def build_dataset_artifact(
    *,
    market_dataset_path: str | Path,
    output_dir: str | Path,
    config: TraderConfig,
) -> tuple[Path, dict[str, Any]]:
    """Build and persist a feature dataset from a saved market dataset."""

    market_path = Path(market_dataset_path)
    market_data = read_market_dataset(
        market_path,
        symbol=config.data.symbol,
        interval=config.data.interval,
    )
    raw_metadata = _read_json(dataset_metadata_path(market_path))
    features = build_feature_dataset(market_data, config)
    feature_metadata = _feature_metadata(
        features,
        raw_metadata=raw_metadata,
        config=config,
    )
    output_path = Path(output_dir) / _feature_dataset_name(feature_metadata)
    _require_absent(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    _write_json(_feature_metadata_path(output_path), feature_metadata)
    _write_yaml(output_path.with_suffix(output_path.suffix + ".config.yaml"), asdict(config))
    return output_path, feature_metadata


def train_model_artifact(
    *,
    feature_dataset_path: str | Path,
    output_dir: str | Path,
    config: TraderConfig,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    """Run development validation and save the final fitted model artifact."""

    feature_path = Path(feature_dataset_path)
    features = pd.read_parquet(feature_path)
    feature_metadata = _read_json(_feature_metadata_path(feature_path))
    split = split_final_holdout(features, config.validation)
    validation_metrics = walk_forward_validate(
        features,
        config,
        feature_names=MODEL_FEATURE_COLUMNS,
    )
    model = BaselineLogisticModel(config, feature_names=MODEL_FEATURE_COLUMNS)
    model.fit(split.development)
    model_path = Path(output_dir) / _model_name(feature_metadata)
    _require_absent(model_path)
    model_metadata = save_model_artifact(
        model,
        model_path,
        training_data=split.development,
        dataset_content_hash=str(feature_metadata["content_hash"]),
        validation_fold_metrics=validation_metrics,
        target_definition=TARGET_DEFINITION,
    )
    return model_path, model_metadata, validation_metrics


def run_backtest_artifact(
    *,
    feature_dataset_path: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    config: TraderConfig,
    run_id: str | None = None,
) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]], BacktestResult]:
    """Run the saved model on final holdout data and write a report bundle."""

    feature_path = Path(feature_dataset_path)
    model_artifact_path = Path(model_path)
    features = pd.read_parquet(feature_path)
    feature_metadata = _read_json(_feature_metadata_path(feature_path))
    model_metadata = _read_json(model_metadata_path(model_artifact_path))
    validation_metrics = list(model_metadata.get("validation_fold_metrics", []))
    split = split_final_holdout(features, config.validation)
    holdout = split.holdout.copy()
    model = load_model_artifact(
        model_artifact_path,
        config=config,
        expected_feature_names=MODEL_FEATURE_COLUMNS,
    )
    probabilities = model.predict_positive_proba(holdout)
    predictions = pd.DataFrame(
        {
            "timestamp": holdout["timestamp"],
            "signal": (probabilities >= config.model.probability_threshold).astype("int8"),
            "probability": probabilities,
        }
    )
    backtest = run_long_cash_backtest(
        holdout,
        predictions,
        backtest_config=config.backtest,
        costs_config=config.costs,
    )
    metrics = calculate_backtest_metrics(backtest.equity, backtest.trades)
    benchmark_results = run_benchmarks(
        holdout,
        backtest_config=config.backtest,
        costs_config=config.costs,
    )
    benchmark_metrics = {
        name: result.metrics for name, result in benchmark_results.items()
    }
    report_run_id = run_id or _report_run_id(feature_metadata, model_metadata)
    report_path = Path(output_dir) / report_run_id
    _require_absent(report_path)
    report_dir = write_backtest_report(
        output_dir,
        config=config,
        dataset_metadata=feature_metadata,
        model_metadata=model_metadata,
        fold_metrics=validation_metrics,
        backtest=backtest,
        metrics=metrics,
        benchmark_metrics=benchmark_metrics,
        run_id=report_run_id,
    )
    return report_dir, metrics, benchmark_metrics, backtest


def run_baseline_artifacts(
    *,
    market_dataset_path: str | Path,
    dataset_output_dir: str | Path,
    model_output_dir: str | Path,
    report_output_dir: str | Path,
    config: TraderConfig,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Build, train, and backtest from an already saved market dataset."""

    feature_path, feature_metadata = build_dataset_artifact(
        market_dataset_path=market_dataset_path,
        output_dir=dataset_output_dir,
        config=config,
    )
    model_path, model_metadata, validation_metrics = train_model_artifact(
        feature_dataset_path=feature_path,
        output_dir=model_output_dir,
        config=config,
    )
    report_dir, metrics, benchmark_metrics, _ = run_backtest_artifact(
        feature_dataset_path=feature_path,
        model_path=model_path,
        output_dir=report_output_dir,
        config=config,
        run_id=run_id,
    )
    return {
        "feature_dataset": feature_path,
        "feature_metadata": feature_metadata,
        "model": model_path,
        "model_metadata": model_metadata,
        "validation_fold_metrics": validation_metrics,
        "report": report_dir,
        "metrics": metrics,
        "benchmark_metrics": benchmark_metrics,
    }


def _feature_metadata(
    features: pd.DataFrame,
    *,
    raw_metadata: dict[str, Any],
    config: TraderConfig,
) -> dict[str, Any]:
    labeled_rows = int(features["target"].notna().sum()) if "target" in features else 0
    metadata = build_metadata(
        features.loc[:, ["timestamp", "symbol", "open", "high", "low", "close", "volume"]],
        source="feature-builder",
        symbol=config.data.symbol,
        interval=config.data.interval,
    )
    return {
        "schema_version": FEATURE_DATASET_SCHEMA_VERSION,
        "symbol": metadata.symbol,
        "interval": metadata.interval,
        "row_count": metadata.row_count,
        "labeled_row_count": labeled_rows,
        "start": metadata.start,
        "end": metadata.end,
        "source": "feature-builder",
        "raw_dataset": raw_metadata,
        "raw_dataset_content_hash": raw_metadata.get("content_hash"),
        "content_hash": _dataframe_hash(features),
        "model_feature_columns": list(MODEL_FEATURE_COLUMNS),
        "target_definition": TARGET_DEFINITION,
        "config_hash": _config_hash(config),
    }


def _feature_dataset_name(metadata: dict[str, Any]) -> str:
    return (
        f"{metadata['symbol'].lower()}_{metadata['interval']}_features_"
        f"{str(metadata['content_hash'])[:12]}.parquet"
    )


def _model_name(feature_metadata: dict[str, Any]) -> str:
    return (
        f"{str(feature_metadata['symbol']).lower()}_"
        f"{feature_metadata['interval']}_logistic_"
        f"{str(feature_metadata['content_hash'])[:12]}.joblib"
    )


def _report_run_id(
    feature_metadata: dict[str, Any],
    model_metadata: dict[str, Any],
) -> str:
    created_at = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    model_hash = _stable_json_hash(model_metadata)[:12]
    return (
        f"baseline_{str(feature_metadata['content_hash'])[:12]}_"
        f"{model_hash}_{created_at}"
    )


def _feature_metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".metadata.json")


def _dataframe_hash(data: pd.DataFrame) -> str:
    normalized = data.copy()
    for column in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = pd.to_datetime(normalized[column], utc=True).dt.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
    payload = normalized.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _config_hash(config: TraderConfig) -> str:
    return hashlib.sha256(
        yaml.safe_dump(asdict(config), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _stable_json_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=True), encoding="utf-8")


def _require_absent(path: Path) -> None:
    if path.exists():
        raise WorkflowError(f"refusing to overwrite existing artifact: {path}")
