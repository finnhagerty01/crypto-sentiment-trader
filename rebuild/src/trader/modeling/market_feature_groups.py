"""Market feature group experiments for the baseline Logistic Regression model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from trader.config import FeaturesConfig, TargetConfig, TraderConfig
from trader.data.storage import read_market_dataset
from trader.features.market import (
    build_feature_dataset,
    model_feature_columns,
)
from trader.modeling.thresholds import (
    ThresholdSweepResult,
    run_threshold_sweep,
    write_threshold_sweep_artifacts,
)


TARGET_HORIZON_BARS = 12
TARGET_COST_BUFFER = "none"
TARGET_VOLATILITY_MULTIPLIER = 0.10
FEATURE_GROUP_CANDIDATES: tuple[tuple[str, ...], ...] = (
    ("baseline",),
    ("baseline", "trend"),
    ("baseline", "volatility"),
    ("baseline", "volume"),
    ("baseline", "calendar"),
    ("baseline", "momentum_reversal"),
)


@dataclass(frozen=True, slots=True)
class FeatureGroupResult:
    """Aggregated result for one enabled feature-group candidate."""

    enabled_groups: tuple[str, ...]
    summary_row: dict[str, Any]
    threshold_result: ThresholdSweepResult
    feature_dataset_path: Path
    artifact_dir: Path


@dataclass(frozen=True, slots=True)
class FeatureGroupExperimentResult:
    """Aggregated output for the market feature-group experiment."""

    results: tuple[FeatureGroupResult, ...]
    summary: pd.DataFrame
    selected_group: dict[str, Any] | None
    output_dir: Path


def feature_group_id(groups: tuple[str, ...]) -> str:
    """Return a stable artifact id for an enabled-group tuple."""

    return "__".join(groups)


def target_grid_winner_config(config: TraderConfig) -> TraderConfig:
    """Return a config copy using the selected Model Refinement 02 target."""

    return replace(
        config,
        target=TargetConfig(
            horizon_bars=TARGET_HORIZON_BARS,
            cost_buffer=TARGET_COST_BUFFER,
            volatility_multiplier=TARGET_VOLATILITY_MULTIPLIER,
        ),
    )


def config_for_feature_groups(
    config: TraderConfig,
    enabled_groups: tuple[str, ...],
) -> TraderConfig:
    """Return a config copy with only enabled feature groups changed."""

    return replace(
        config,
        features=FeaturesConfig(
            volatility_window=config.features.volatility_window,
            volume_window=config.features.volume_window,
            rsi_window=config.features.rsi_window,
            clipping_window=config.features.clipping_window,
            clipping_mad_multiplier=config.features.clipping_mad_multiplier,
            enabled_groups=enabled_groups,
        ),
    )


def run_market_feature_group_experiment(
    *,
    market_dataset_path: str | Path,
    config: TraderConfig,
    output_dir: str | Path,
    overwrite: bool = False,
) -> FeatureGroupExperimentResult:
    """Run baseline-plus-one feature-group experiments and write artifacts."""

    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing feature-group artifacts: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)

    market_data = read_market_dataset(
        market_dataset_path,
        symbol=config.data.symbol,
        interval=config.data.interval,
    )
    experiment_config = target_grid_winner_config(config)
    rows: list[dict[str, Any]] = []
    results: list[FeatureGroupResult] = []

    for enabled_groups in FEATURE_GROUP_CANDIDATES:
        candidate_config = config_for_feature_groups(
            experiment_config,
            enabled_groups,
        )
        group_id = feature_group_id(enabled_groups)
        artifact_dir = destination / group_id
        artifact_dir.mkdir(parents=True, exist_ok=overwrite)

        features = build_feature_dataset(market_data, candidate_config)
        feature_dataset_path = artifact_dir / "feature_dataset.parquet"
        if feature_dataset_path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite {feature_dataset_path}")
        features.to_parquet(feature_dataset_path, index=False)
        _write_yaml(artifact_dir / "config.yaml", asdict(candidate_config))

        feature_names = model_feature_columns(candidate_config)
        _write_json(
            artifact_dir / "feature_names.json",
            {
                "enabled_groups": list(enabled_groups),
                "feature_names": list(feature_names),
                "feature_count": len(feature_names),
            },
        )

        threshold_result = run_threshold_sweep(
            features,
            candidate_config,
            feature_names=feature_names,
        )
        write_threshold_sweep_artifacts(threshold_result, artifact_dir)

        row = summarize_feature_group_result(
            enabled_groups=enabled_groups,
            feature_names=feature_names,
            threshold_result=threshold_result,
        )
        rows.append(row)
        results.append(
            FeatureGroupResult(
                enabled_groups=enabled_groups,
                summary_row=row,
                threshold_result=threshold_result,
                feature_dataset_path=feature_dataset_path,
                artifact_dir=artifact_dir,
            )
        )

    summary = pd.DataFrame(rows)
    selected_group = select_feature_group(summary)
    summary.to_csv(destination / "feature_group_results.csv", index=False)
    _write_json(destination / "feature_group_results.json", {"groups": rows})
    _write_json(
        destination / "selected_feature_group.json",
        selected_group or {"selected_feature_group": None},
    )
    (destination / "market_feature_group_report.md").write_text(
        build_feature_group_report(summary, selected_group=selected_group),
        encoding="utf-8",
    )
    return FeatureGroupExperimentResult(
        results=tuple(results),
        summary=summary,
        selected_group=selected_group,
        output_dir=destination,
    )


def summarize_feature_group_result(
    *,
    enabled_groups: tuple[str, ...],
    feature_names: tuple[str, ...],
    threshold_result: ThresholdSweepResult,
) -> dict[str, Any]:
    """Build one top-level summary row for a feature-group candidate."""

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
        "feature_group_id": feature_group_id(enabled_groups),
        "enabled_groups": list(enabled_groups),
        "feature_count": len(feature_names),
        "feature_names": list(feature_names),
        "selected_threshold": selected_threshold,
        "selected_threshold_trade_count": selected_row.get("trade_count"),
        "selected_median_total_return": selected_row.get("median_total_return"),
        "selected_median_cash_total_return": selected_row.get(
            "median_cash_total_return"
        ),
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
    }


def select_feature_group(summary: pd.DataFrame) -> dict[str, Any] | None:
    """Select a feature-group candidate using development metrics only."""

    if summary.empty:
        return None
    eligible = summary.loc[summary["selected_threshold"].notna()].copy()
    eligible = eligible.loc[
        eligible["selected_median_total_return"].notna()
        & eligible["selected_median_max_drawdown"].notna()
        & eligible["selected_median_turnover"].notna()
    ]
    if eligible.empty:
        return None

    eligible["drawdown_magnitude"] = eligible[
        "selected_median_max_drawdown"
    ].abs()
    eligible = eligible.sort_values(
        by=[
            "selected_median_total_return",
            "drawdown_magnitude",
            "selected_median_turnover",
            "feature_count",
            "feature_group_id",
        ],
        ascending=[False, True, True, True, True],
        kind="mergesort",
    )
    return _json_ready(eligible.iloc[0].drop(labels=["drawdown_magnitude"]).to_dict())


def build_feature_group_report(
    summary: pd.DataFrame,
    *,
    selected_group: dict[str, Any] | None,
) -> str:
    """Build a concise Markdown report for the feature-group experiment."""

    lines = [
        "# Market Feature Group Experiment",
        "",
        "Selection uses development-fold metrics only. Holdout metrics are confirmation only.",
        "",
        "## Selected Development Group",
        "",
    ]
    if selected_group is None:
        lines.extend(["No feature-group candidate produced a selectable threshold.", ""])
    else:
        lines.extend(
            [
                f"- Feature group: `{selected_group['feature_group_id']}`",
                f"- Feature count: `{selected_group['feature_count']}`",
                f"- Selected threshold: `{selected_group['selected_threshold']}`",
                "- Development median return: "
                f"`{_format_percent(selected_group['selected_median_total_return'])}`",
                "- Development median max drawdown: "
                f"`{_format_percent(selected_group['selected_median_max_drawdown'])}`",
                "- Holdout return: "
                f"`{_format_percent(selected_group['holdout_total_return'])}`",
                "- Holdout max drawdown: "
                f"`{_format_percent(selected_group['holdout_max_drawdown'])}`",
                "- Holdout exposure: "
                f"`{_format_percent(selected_group['holdout_exposure_percentage'])}`",
                "",
            ]
        )

    lines.extend(["## Group Comparison", ""])
    if summary.empty:
        lines.extend(["No feature-group results were produced.", ""])
    else:
        lines.append(
            "| group | threshold | features | dev return | dev drawdown | dev turnover | holdout return | holdout drawdown | exposure |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in summary.to_dict(orient="records"):
            lines.append(
                "| "
                f"`{row['feature_group_id']}` | "
                f"{_format_value(row['selected_threshold'])} | "
                f"{row['feature_count']} | "
                f"{_format_percent(row['selected_median_total_return'])} | "
                f"{_format_percent(row['selected_median_max_drawdown'])} | "
                f"{_format_number(row['selected_median_turnover'])} | "
                f"{_format_percent(row['holdout_total_return'])} | "
                f"{_format_percent(row['holdout_max_drawdown'])} | "
                f"{_format_percent(row['holdout_exposure_percentage'])} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=True), encoding="utf-8")


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_ready(value: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, default=_json_default))


def _format_percent(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _format_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.4f}"


def _format_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)
