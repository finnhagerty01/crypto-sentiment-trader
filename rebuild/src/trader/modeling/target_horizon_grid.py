"""Target strictness and horizon grid experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from trader.config import TargetConfig, TraderConfig
from trader.data.storage import read_market_dataset
from trader.features.market import build_feature_dataset
from trader.features.target import summarize_target_distribution
from trader.modeling.thresholds import (
    ThresholdSweepResult,
    run_threshold_sweep,
    write_threshold_sweep_artifacts,
)


HORIZON_CANDIDATES: tuple[int, ...] = (1, 3, 6, 12, 24)
COST_BUFFER_CANDIDATES: tuple[str, ...] = ("none", "one_way", "round_trip")
VOLATILITY_MULTIPLIER_CANDIDATES: tuple[float, ...] = (0.00, 0.05, 0.10)


@dataclass(frozen=True, slots=True)
class TargetHorizonCandidate:
    """One target/horizon experiment candidate."""

    horizon_bars: int
    cost_buffer: str
    volatility_multiplier: float

    @property
    def candidate_id(self) -> str:
        volatility = f"{self.volatility_multiplier:.2f}".replace(".", "p")
        return (
            f"h{self.horizon_bars}_"
            f"{self.cost_buffer}_"
            f"vol{volatility}"
        )


@dataclass(frozen=True, slots=True)
class CandidateResult:
    """Aggregated result for one target/horizon candidate."""

    candidate: TargetHorizonCandidate
    summary_row: dict[str, Any]
    threshold_result: ThresholdSweepResult
    feature_dataset_path: Path
    artifact_dir: Path


@dataclass(frozen=True, slots=True)
class GridSearchResult:
    """Aggregated result for the target/horizon experiment grid."""

    results: tuple[CandidateResult, ...]
    summary: pd.DataFrame
    selected_candidate: dict[str, Any] | None
    output_dir: Path


def target_horizon_grid() -> tuple[TargetHorizonCandidate, ...]:
    """Return the fixed target/horizon grid from the refinement plan."""

    return tuple(
        TargetHorizonCandidate(
            horizon_bars=horizon,
            cost_buffer=cost_buffer,
            volatility_multiplier=volatility_multiplier,
        )
        for horizon in HORIZON_CANDIDATES
        for cost_buffer in COST_BUFFER_CANDIDATES
        for volatility_multiplier in VOLATILITY_MULTIPLIER_CANDIDATES
    )


def config_for_candidate(
    config: TraderConfig,
    candidate: TargetHorizonCandidate,
) -> TraderConfig:
    """Return a config copy with only the target candidate changed."""

    return replace(
        config,
        target=TargetConfig(
            horizon_bars=candidate.horizon_bars,
            cost_buffer=candidate.cost_buffer,
            volatility_multiplier=candidate.volatility_multiplier,
        ),
    )


def summarize_candidate_result(
    *,
    candidate: TargetHorizonCandidate,
    target_distribution: dict[str, Any],
    threshold_result: ThresholdSweepResult,
) -> dict[str, Any]:
    """Build the top-level grid row for one candidate."""

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
        "candidate_id": candidate.candidate_id,
        "horizon_bars": candidate.horizon_bars,
        "cost_buffer": candidate.cost_buffer,
        "volatility_multiplier": candidate.volatility_multiplier,
        **target_distribution,
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


def select_grid_candidate(summary: pd.DataFrame) -> dict[str, Any] | None:
    """Select the best target candidate from development metrics only."""

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
            "horizon_bars",
            "cost_buffer",
            "volatility_multiplier",
        ],
        ascending=[False, True, True, True, True, True],
        kind="mergesort",
    )
    return _json_ready(eligible.iloc[0].drop(labels=["drawdown_magnitude"]).to_dict())


def run_target_horizon_grid(
    *,
    market_dataset_path: str | Path,
    config: TraderConfig,
    output_dir: str | Path,
    overwrite: bool = False,
) -> GridSearchResult:
    """Run the full target/horizon grid and write reproducible artifacts."""

    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing grid artifacts: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)

    market_data = read_market_dataset(
        market_dataset_path,
        symbol=config.data.symbol,
        interval=config.data.interval,
    )
    results: list[CandidateResult] = []
    rows: list[dict[str, Any]] = []

    for candidate in target_horizon_grid():
        candidate_config = config_for_candidate(config, candidate)
        artifact_dir = destination / candidate.candidate_id
        artifact_dir.mkdir(parents=True, exist_ok=overwrite)

        features = build_feature_dataset(market_data, candidate_config)
        feature_dataset_path = artifact_dir / "feature_dataset.parquet"
        if feature_dataset_path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite {feature_dataset_path}")
        features.to_parquet(feature_dataset_path, index=False)
        _write_yaml(artifact_dir / "config.yaml", asdict(candidate_config))

        target_distribution = _target_distribution_dict(features)
        _write_json(artifact_dir / "target_distribution.json", target_distribution)

        threshold_result = run_threshold_sweep(features, candidate_config)
        write_threshold_sweep_artifacts(threshold_result, artifact_dir)

        row = summarize_candidate_result(
            candidate=candidate,
            target_distribution=target_distribution,
            threshold_result=threshold_result,
        )
        rows.append(row)
        results.append(
            CandidateResult(
                candidate=candidate,
                summary_row=row,
                threshold_result=threshold_result,
                feature_dataset_path=feature_dataset_path,
                artifact_dir=artifact_dir,
            )
        )

    summary = pd.DataFrame(rows)
    selected_candidate = select_grid_candidate(summary)
    summary.to_csv(destination / "grid_results.csv", index=False)
    _write_json(destination / "grid_results.json", {"candidates": rows})
    _write_json(
        destination / "selected_candidate.json",
        selected_candidate or {"selected_candidate": None},
    )
    (destination / "target_horizon_grid_report.md").write_text(
        build_grid_report(summary, selected_candidate=selected_candidate),
        encoding="utf-8",
    )
    return GridSearchResult(
        results=tuple(results),
        summary=summary,
        selected_candidate=selected_candidate,
        output_dir=destination,
    )


def build_grid_report(
    summary: pd.DataFrame,
    *,
    selected_candidate: dict[str, Any] | None,
) -> str:
    """Build a concise Markdown report for the grid result."""

    lines = [
        "# Target and Horizon Grid Search",
        "",
        "Selection uses development-fold metrics only. Holdout metrics are confirmation only.",
        "",
        "## Selected Candidate",
        "",
    ]
    if selected_candidate is None:
        lines.extend(
            [
                "No candidate survived the development-fold threshold policy.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"- Candidate: `{selected_candidate['candidate_id']}`",
                f"- Horizon bars: `{selected_candidate['horizon_bars']}`",
                f"- Cost buffer: `{selected_candidate['cost_buffer']}`",
                f"- Volatility multiplier: `{selected_candidate['volatility_multiplier']}`",
                f"- Selected threshold: `{selected_candidate['selected_threshold']}`",
                "- Development median return: "
                f"`{_format_percent(selected_candidate['selected_median_total_return'])}`",
                "- Development median max drawdown: "
                f"`{_format_percent(selected_candidate['selected_median_max_drawdown'])}`",
                "- Holdout return: "
                f"`{_format_percent(selected_candidate['holdout_total_return'])}`",
                "- Holdout max drawdown: "
                f"`{_format_percent(selected_candidate['holdout_max_drawdown'])}`",
                "",
            ]
        )

    lines.extend(["## Top Development Candidates", ""])
    top = _top_development_rows(summary, limit=10)
    if top.empty:
        lines.extend(["No eligible development candidates.", ""])
    else:
        lines.append(
            "| candidate | threshold | dev return | dev drawdown | dev turnover | holdout return | positive rate |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in top.to_dict(orient="records"):
            lines.append(
                "| "
                f"`{row['candidate_id']}` | "
                f"{row['selected_threshold']} | "
                f"{_format_percent(row['selected_median_total_return'])} | "
                f"{_format_percent(row['selected_median_max_drawdown'])} | "
                f"{_format_number(row['selected_median_turnover'])} | "
                f"{_format_percent(row['holdout_total_return'])} | "
                f"{_format_percent(row['positive_rate'])} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Dataset Caveat",
            "",
            "This run uses the currently available five-month BTCUSDT 1h dataset. "
            "Treat the result as a current-data refinement signal, not a robust "
            "paper-trading gate.",
            "",
        ]
    )
    return "\n".join(lines)


def _top_development_rows(summary: pd.DataFrame, *, limit: int) -> pd.DataFrame:
    eligible = summary.loc[summary["selected_threshold"].notna()].copy()
    if eligible.empty:
        return eligible
    eligible["drawdown_magnitude"] = eligible[
        "selected_median_max_drawdown"
    ].abs()
    return eligible.sort_values(
        by=[
            "selected_median_total_return",
            "drawdown_magnitude",
            "selected_median_turnover",
            "candidate_id",
        ],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).head(limit)


def _target_distribution_dict(features: pd.DataFrame) -> dict[str, Any]:
    distribution = summarize_target_distribution(features)
    values = asdict(distribution)
    return _json_ready(values)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_yaml(path: Path, value: Any) -> None:
    path.write_text(yaml.safe_dump(_json_ready(value), sort_keys=True), encoding="utf-8")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _format_percent(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _format_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.2f}"
