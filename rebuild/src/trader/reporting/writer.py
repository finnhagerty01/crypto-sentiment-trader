"""Deterministic backtest report writer."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from trader.backtest.engine import BacktestResult
from trader.config import TraderConfig


REPORT_SCHEMA_VERSION = "backtest-report-v1"


def write_backtest_report(
    output_dir: str | Path,
    *,
    config: TraderConfig,
    dataset_metadata: Mapping[str, Any] | object,
    model_metadata: Mapping[str, Any],
    fold_metrics: list[dict[str, Any]],
    backtest: BacktestResult,
    metrics: Mapping[str, Any],
    benchmark_metrics: Mapping[str, Mapping[str, Any]],
    run_id: str,
) -> Path:
    """Write all report artifacts into a deterministic versioned directory."""

    report_dir = Path(output_dir) / run_id
    report_dir.mkdir(parents=True, exist_ok=True)

    _write_yaml(report_dir / "config.yaml", asdict(config))
    _write_json(report_dir / "dataset_metadata.json", _mapping(dataset_metadata))
    _write_json(report_dir / "model_metadata.json", dict(model_metadata))
    pd.DataFrame(fold_metrics).to_csv(report_dir / "fold_metrics.csv", index=False)
    backtest.predictions.to_csv(report_dir / "predictions.csv", index=False)
    backtest.trades.to_csv(report_dir / "trades.csv", index=False)
    backtest.equity.to_csv(report_dir / "equity.csv", index=False)
    _write_json(report_dir / "metrics.json", dict(metrics))
    _write_json(report_dir / "benchmark_metrics.json", dict(benchmark_metrics))
    _write_summary(
        report_dir / "summary.md",
        run_id=run_id,
        dataset_metadata=_mapping(dataset_metadata),
        metrics=metrics,
        benchmark_metrics=benchmark_metrics,
    )
    return report_dir


def _mapping(value: Mapping[str, Any] | object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        if hasattr(value, "as_dict"):
            return value.as_dict()
        return asdict(value)
    raise TypeError("metadata must be a mapping or dataclass")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=True), encoding="utf-8")


def _write_summary(
    path: Path,
    *,
    run_id: str,
    dataset_metadata: Mapping[str, Any],
    metrics: Mapping[str, Any],
    benchmark_metrics: Mapping[str, Mapping[str, Any]],
) -> None:
    lines = [
        "# Backtest Summary",
        "",
        f"- schema_version: {REPORT_SCHEMA_VERSION}",
        f"- run_id: {run_id}",
        f"- symbol: {dataset_metadata.get('symbol', 'unknown')}",
        f"- start: {dataset_metadata.get('start', 'unknown')}",
        f"- end: {dataset_metadata.get('end', 'unknown')}",
        f"- total_return: {_format_metric(metrics.get('total_return'))}",
        f"- max_drawdown: {_format_metric(metrics.get('max_drawdown'))}",
        f"- trade_count: {metrics.get('trade_count', 0)}",
        "",
        "## Benchmarks",
        "",
    ]
    for name in sorted(benchmark_metrics):
        values = benchmark_metrics[name]
        lines.append(
            f"- {name}: total_return={_format_metric(values.get('total_return'))}, "
            f"max_drawdown={_format_metric(values.get('max_drawdown'))}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.8f}"
    return str(value)
