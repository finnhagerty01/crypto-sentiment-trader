"""Named model-refinement experiment orchestration."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from trader.backtest.benchmarks import run_benchmarks
from trader.config import ConfigError, TraderConfig, _build_dataclass, _validate, load_config
from trader.data.storage import read_market_dataset
from trader.features.market import build_feature_dataset, model_feature_columns
from trader.features.target import summarize_target_distribution
from trader.modeling.thresholds import run_threshold_sweep
from trader.modeling.validation import split_final_holdout


VARIANT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class ExperimentConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


def run_experiment_grid(
    *,
    market_dataset_path: str | Path,
    experiment_config_path: str | Path,
    output_dir: str | Path,
    run_id: str,
) -> dict[str, Any]:
    """Run named model-refinement variants from a saved market dataset."""

    experiment_path = Path(experiment_config_path)
    raw_experiment = _read_experiment_config(experiment_path)
    variants = _parse_variants(raw_experiment, experiment_path=experiment_path)
    resolved_variants = [
        (variant, _resolve_variant_config(variant, experiment_path=experiment_path))
        for variant in variants
    ]

    run_dir = Path(output_dir) / run_id
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing experiment run: {run_dir}")
    run_dir.mkdir(parents=True)
    variants_dir = run_dir / "variants"
    variants_dir.mkdir()

    _write_yaml(run_dir / "experiment_config.yaml", raw_experiment)

    summary_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    threshold_rows: list[pd.DataFrame] = []
    fold_rows: list[pd.DataFrame] = []
    holdout_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []

    market_cache: dict[tuple[str, str], pd.DataFrame] = {}
    for index, (variant, config) in enumerate(resolved_variants):
        variant_dir = variants_dir / variant["name"]
        variant_dir.mkdir()
        _write_yaml(variant_dir / "resolved_config.yaml", asdict(config))

        cache_key = (config.data.symbol, config.data.interval)
        if cache_key not in market_cache:
            market_cache[cache_key] = read_market_dataset(
                market_dataset_path,
                symbol=config.data.symbol,
                interval=config.data.interval,
            )

        features = build_feature_dataset(market_cache[cache_key], config)
        feature_names = model_feature_columns(config)
        target_distribution = _target_distribution_dict(features)
        target_rows.append({"variant_name": variant["name"], **target_distribution})

        threshold_result = run_threshold_sweep(
            features,
            config,
            feature_names=feature_names,
        )
        threshold_summary = threshold_result.summary.copy()
        threshold_summary.insert(0, "variant_name", variant["name"])
        threshold_rows.append(threshold_summary)

        fold_metrics = threshold_result.fold_metrics.copy()
        fold_metrics.insert(0, "variant_name", variant["name"])
        fold_rows.append(fold_metrics)

        holdout_metrics = threshold_result.holdout_metrics or {
            "status": "not_evaluated",
            "selected_threshold": None,
        }
        holdout_rows.append({"variant_name": variant["name"], **holdout_metrics})

        split = split_final_holdout(features, config.validation)
        for benchmark_name, benchmark in run_benchmarks(
            split.holdout,
            backtest_config=config.backtest,
            costs_config=config.costs,
        ).items():
            benchmark_rows.append(
                {
                    "variant_name": variant["name"],
                    "benchmark": benchmark_name,
                    **benchmark.metrics,
                }
            )

        summary_rows.append(
            _variant_summary_row(
                variant_name=variant["name"],
                variant_order=index,
                config=config,
                feature_names=feature_names,
                target_distribution=target_distribution,
                threshold_result=threshold_result,
            )
        )

    variant_summary = _rank_variants(pd.DataFrame(summary_rows))
    target_distribution = pd.DataFrame(target_rows)
    threshold_summary = _concat_frames(threshold_rows)
    fold_metrics = _concat_frames(fold_rows)
    holdout_metrics = pd.DataFrame(holdout_rows)
    benchmark_metrics = pd.DataFrame(benchmark_rows)

    _write_csv(variant_summary, run_dir / "variant_summary.csv")
    _write_csv(target_distribution, run_dir / "target_distribution.csv")
    _write_csv(threshold_summary, run_dir / "threshold_summary.csv")
    _write_csv(fold_metrics, run_dir / "fold_metrics.csv")
    _write_csv(holdout_metrics, run_dir / "holdout_metrics.csv")
    _write_csv(benchmark_metrics, run_dir / "benchmark_metrics.csv")

    selected = variant_summary.loc[variant_summary["rank"] == 1]
    selected_variant = selected.iloc[0].to_dict() if not selected.empty else None
    return {
        "run_dir": run_dir,
        "variant_summary": variant_summary,
        "selected_variant": selected_variant,
    }


def _read_experiment_config(path: Path) -> Mapping[str, Any]:
    try:
        with path.open(encoding="utf-8") as config_file:
            raw = yaml.safe_load(config_file)
    except OSError as exc:
        raise ExperimentConfigError(f"cannot read experiment config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ExperimentConfigError(f"invalid YAML in experiment config {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ExperimentConfigError("experiment config must be a mapping")
    return raw


def _parse_variants(
    raw_experiment: Mapping[str, Any],
    *,
    experiment_path: Path,
) -> list[dict[str, Any]]:
    raw_variants = raw_experiment.get("variants")
    if not isinstance(raw_variants, list) or not raw_variants:
        raise ExperimentConfigError("experiment config must define a non-empty variants list")

    names: set[str] = set()
    variants: list[dict[str, Any]] = []
    for position, raw_variant in enumerate(raw_variants, start=1):
        if not isinstance(raw_variant, Mapping):
            raise ExperimentConfigError(f"variants[{position}]: expected a mapping")
        name = raw_variant.get("name")
        if not isinstance(name, str) or not name:
            raise ExperimentConfigError(f"variants[{position}].name: expected non-empty string")
        if not VARIANT_NAME_PATTERN.fullmatch(name):
            raise ExperimentConfigError(
                f"variant name is not path-safe: {name!r}; use letters, digits, _, -, or ."
            )
        if name in names:
            raise ExperimentConfigError(f"duplicate variant name: {name}")
        names.add(name)

        config_path = raw_variant.get("config")
        if not isinstance(config_path, str) or not config_path:
            raise ExperimentConfigError(f"variant {name}: config must be a non-empty string")

        overrides = raw_variant.get("overrides", {})
        if overrides is None:
            overrides = {}
        if not isinstance(overrides, Mapping):
            raise ExperimentConfigError(f"variant {name}: overrides must be a mapping")
        for path in overrides:
            if not isinstance(path, str) or not path:
                raise ExperimentConfigError(f"variant {name}: override paths must be strings")

        variants.append(
            {
                "name": name,
                "config": config_path,
                "overrides": dict(overrides),
                "experiment_path": experiment_path,
            }
        )
    return variants


def _resolve_variant_config(
    variant: Mapping[str, Any],
    *,
    experiment_path: Path,
) -> TraderConfig:
    base_path = _resolve_config_path(str(variant["config"]), experiment_path)
    config = load_config(base_path)
    values = asdict(config)
    for override_path, override_value in dict(variant.get("overrides", {})).items():
        _apply_override(values, override_path, override_value, variant_name=str(variant["name"]))
    try:
        resolved = _build_dataclass(TraderConfig, values, "config")
        _validate(resolved)
    except ConfigError as exc:
        raise ExperimentConfigError(
            f"variant {variant['name']}: resolved config is invalid: {exc}"
        ) from exc
    return resolved


def _resolve_config_path(value: str, experiment_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return experiment_path.parent / path


def _apply_override(
    values: dict[str, Any],
    path: str,
    value: Any,
    *,
    variant_name: str,
) -> None:
    parts = path.split(".")
    if any(part == "" for part in parts):
        raise ExperimentConfigError(f"variant {variant_name}: invalid override path: {path}")
    current: Any = values
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            raise ExperimentConfigError(f"variant {variant_name}: unknown override path: {path}")
        current = current[part]
    leaf = parts[-1]
    if not isinstance(current, dict) or leaf not in current:
        raise ExperimentConfigError(f"variant {variant_name}: unknown override path: {path}")
    current[leaf] = value


def _variant_summary_row(
    *,
    variant_name: str,
    variant_order: int,
    config: TraderConfig,
    feature_names: tuple[str, ...],
    target_distribution: dict[str, Any],
    threshold_result: Any,
) -> dict[str, Any]:
    selected_threshold = threshold_result.selected_threshold
    selected_row: dict[str, Any] = {}
    if selected_threshold is not None:
        matching = threshold_result.summary.loc[
            threshold_result.summary["threshold"] == selected_threshold
        ]
        if not matching.empty:
            selected_row = matching.iloc[0].to_dict()

    holdout_metrics = threshold_result.holdout_metrics or {}
    return {
        "variant_name": variant_name,
        "rank": None,
        "eligible": False,
        "variant_order": variant_order,
        "horizon_bars": config.target.horizon_bars,
        "cost_buffer": config.target.cost_buffer,
        "volatility_multiplier": config.target.volatility_multiplier,
        "enabled_groups": ",".join(config.features.enabled_groups),
        "feature_count": len(feature_names),
        "selected_threshold": selected_threshold,
        "selected_threshold_trade_count": selected_row.get("trade_count"),
        "selected_median_total_return": selected_row.get("median_total_return"),
        "selected_median_cash_total_return": selected_row.get("median_cash_total_return"),
        "selected_median_max_drawdown": selected_row.get("median_max_drawdown"),
        "selected_median_turnover": selected_row.get("median_turnover"),
        "selected_mean_precision": selected_row.get("mean_precision"),
        "selected_mean_recall": selected_row.get("mean_recall"),
        "selected_mean_f1": selected_row.get("mean_f1"),
        "holdout_status": holdout_metrics.get("status", "not_evaluated"),
        "holdout_total_return": holdout_metrics.get("total_return"),
        "holdout_cash_total_return": holdout_metrics.get("cash_total_return"),
        "holdout_max_drawdown": holdout_metrics.get("max_drawdown"),
        "holdout_trade_count": holdout_metrics.get("trade_count"),
        "holdout_turnover": holdout_metrics.get("turnover"),
        "holdout_exposure_percentage": holdout_metrics.get("exposure_percentage"),
        **target_distribution,
    }


def _rank_variants(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    ranked = summary.copy()
    selected = ranked["selected_threshold"].notna()
    positive_return = ranked["selected_median_total_return"].fillna(-np.inf) >= 0.0
    beats_cash = (
        ranked["selected_median_total_return"].fillna(-np.inf)
        > ranked["selected_median_cash_total_return"].fillna(np.inf)
    )
    ranked["eligible"] = selected & positive_return & beats_cash

    eligible = ranked.loc[ranked["eligible"]].copy()
    if not eligible.empty:
        eligible["drawdown_magnitude"] = eligible["selected_median_max_drawdown"].abs()
        eligible = eligible.sort_values(
            by=[
                "selected_median_total_return",
                "drawdown_magnitude",
                "selected_median_turnover",
                "variant_order",
            ],
            ascending=[False, True, True, True],
            kind="mergesort",
        )
        for rank, index in enumerate(eligible.index, start=1):
            ranked.loc[index, "rank"] = rank

    ranked["rank_sort"] = ranked["rank"].fillna(np.inf)
    ranked = ranked.sort_values(
        by=["rank_sort", "variant_order"],
        ascending=[True, True],
        kind="mergesort",
    ).drop(columns=["rank_sort"])
    return ranked


def _target_distribution_dict(features: pd.DataFrame) -> dict[str, Any]:
    distribution = summarize_target_distribution(features)
    return {
        "row_count": distribution.row_count,
        "labeled_row_count": distribution.labeled_row_count,
        "positive_count": distribution.positive_count,
        "negative_count": distribution.negative_count,
        "positive_rate": distribution.positive_rate,
        "unlabeled_count": distribution.unlabeled_count,
        "first_labeled_timestamp": _timestamp_or_none(distribution.first_labeled_timestamp),
        "last_labeled_timestamp": _timestamp_or_none(distribution.last_labeled_timestamp),
    }


def _timestamp_or_none(value: pd.Timestamp | None) -> str | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat().replace("+00:00", "Z")


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _write_csv(data: pd.DataFrame, path: Path) -> None:
    data.to_csv(path, index=False, lineterminator="\n")


def _write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=True), encoding="utf-8")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
