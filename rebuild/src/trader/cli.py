"""Command-line entry point for the staged rebuild."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
import sys

from trader.config import ConfigError, load_config
from trader.data.market import SOURCE_NAME, MarketCollectionError, collect_market_data
from trader.data.storage import write_market_dataset
from trader.workflow import (
    WorkflowError,
    build_dataset_artifact,
    run_backtest_artifact,
    run_baseline_artifacts,
    train_model_artifact,
)


COMMANDS = (
    "collect-market",
    "build-dataset",
    "train",
    "backtest",
    "run-baseline",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crypto-trader",
        description="Clean crypto research and backtesting pipeline (rebuild in progress).",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    collect = subparsers.add_parser(
        "collect-market",
        help="collect explicit BTCUSDT 1h market candles",
        description="Collect explicit BTCUSDT 1h market candles and save a raw dataset.",
    )
    collect.add_argument("--config", default="configs/baseline.yaml")
    collect.add_argument("--start", required=True, help="UTC start timestamp, inclusive")
    collect.add_argument("--end", required=True, help="UTC end timestamp, exclusive")
    collect.add_argument(
        "--output-dir",
        default="artifacts/datasets",
        help="directory for the saved raw Parquet dataset",
    )
    collect.add_argument(
        "--source",
        default=SOURCE_NAME,
        choices=(SOURCE_NAME,),
        help="market data source",
    )

    build_dataset = subparsers.add_parser(
        "build-dataset",
        help="build causal features and target from a saved market dataset",
        description="Build a feature dataset from an existing market dataset. Never fetches data.",
    )
    build_dataset.add_argument("--config", default="configs/baseline.yaml")
    build_dataset.add_argument("--input", required=True, help="saved raw market Parquet dataset")
    build_dataset.add_argument(
        "--output-dir",
        default="artifacts/datasets",
        help="directory for the saved feature dataset",
    )

    train = subparsers.add_parser(
        "train",
        help="run walk-forward validation and save a fitted baseline model",
        description="Train the Logistic Regression baseline from a saved feature dataset.",
    )
    train.add_argument("--config", default="configs/baseline.yaml")
    train.add_argument("--dataset", required=True, help="saved feature Parquet dataset")
    train.add_argument(
        "--output-dir",
        default="artifacts/models",
        help="directory for the saved model artifact",
    )

    backtest = subparsers.add_parser(
        "backtest",
        help="evaluate a saved model on the final holdout and write a report",
        description="Backtest a saved model artifact against the feature dataset final holdout.",
    )
    backtest.add_argument("--config", default="configs/baseline.yaml")
    backtest.add_argument("--dataset", required=True, help="saved feature Parquet dataset")
    backtest.add_argument("--model", required=True, help="saved model artifact")
    backtest.add_argument(
        "--output-dir",
        default="artifacts/backtests",
        help="directory for the report bundle",
    )
    backtest.add_argument("--run-id", help="explicit report directory name")

    run_baseline = subparsers.add_parser(
        "run-baseline",
        help="build, train, and backtest from an already saved market dataset",
        description=(
            "Run the offline baseline from a saved market dataset. This command never "
            "fetches market data."
        ),
    )
    run_baseline.add_argument("--config", default="configs/baseline.yaml")
    run_baseline.add_argument(
        "--market-data",
        required=True,
        help="saved raw market Parquet dataset with metadata sidecar",
    )
    run_baseline.add_argument(
        "--dataset-dir",
        default="artifacts/datasets",
        help="directory for the saved feature dataset",
    )
    run_baseline.add_argument(
        "--model-dir",
        default="artifacts/models",
        help="directory for the saved model artifact",
    )
    run_baseline.add_argument(
        "--report-dir",
        default="artifacts/backtests",
        help="directory for the report bundle",
    )
    run_baseline.add_argument("--run-id", help="explicit report directory name")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command is None:
        build_parser().print_help()
        return 0

    if args.command == "collect-market":
        return _collect_market(args)
    if args.command == "build-dataset":
        return _build_dataset(args)
    if args.command == "train":
        return _train(args)
    if args.command == "backtest":
        return _backtest(args)
    if args.command == "run-baseline":
        return _run_baseline(args)

    raise AssertionError(f"unhandled command: {args.command}")


def _collect_market(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
        data = collect_market_data(
            start=args.start,
            end=args.end,
            symbol=config.data.symbol,
            interval=config.data.interval,
        )
        output_dir = Path(args.output_dir)
        dataset_path = output_dir / _raw_dataset_name(
            symbol=config.data.symbol,
            interval=config.data.interval,
            start=args.start,
            end=args.end,
        )
        metadata = write_market_dataset(
            data,
            dataset_path,
            source=args.source,
            symbol=config.data.symbol,
            interval=config.data.interval,
        )
    except (ConfigError, MarketCollectionError, OSError, ValueError) as exc:
        print(f"error: collect-market failed: {exc}", file=sys.stderr)
        return 2

    print(f"saved {metadata.row_count} rows to {dataset_path}")
    print(f"metadata content_hash={metadata.content_hash}")
    return 0


def _build_dataset(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
        path, metadata = build_dataset_artifact(
            market_dataset_path=args.input,
            output_dir=args.output_dir,
            config=config,
        )
    except (ConfigError, OSError, ValueError, WorkflowError) as exc:
        print(f"error: build-dataset failed: {exc}", file=sys.stderr)
        return 2

    print(f"saved feature dataset to {path}")
    print(f"metadata content_hash={metadata['content_hash']}")
    return 0


def _train(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
        path, metadata, fold_metrics = train_model_artifact(
            feature_dataset_path=args.dataset,
            output_dir=args.output_dir,
            config=config,
        )
    except (ConfigError, OSError, ValueError, WorkflowError) as exc:
        print(f"error: train failed: {exc}", file=sys.stderr)
        return 2

    ok_folds = sum(1 for fold in fold_metrics if fold.get("status") == "ok")
    print(f"saved model artifact to {path}")
    print(f"validated folds={ok_folds}/{len(fold_metrics)}")
    print(f"dataset content_hash={metadata['dataset_content_hash']}")
    return 0


def _backtest(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
        report_dir, metrics, benchmark_metrics, _ = run_backtest_artifact(
            feature_dataset_path=args.dataset,
            model_path=args.model,
            output_dir=args.output_dir,
            config=config,
            run_id=args.run_id,
        )
    except (ConfigError, OSError, ValueError, WorkflowError) as exc:
        print(f"error: backtest failed: {exc}", file=sys.stderr)
        return 2

    print(f"saved report bundle to {report_dir}")
    _print_metric_comparison(metrics, benchmark_metrics)
    return 0


def _run_baseline(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
        result = run_baseline_artifacts(
            market_dataset_path=args.market_data,
            dataset_output_dir=args.dataset_dir,
            model_output_dir=args.model_dir,
            report_output_dir=args.report_dir,
            config=config,
            run_id=args.run_id,
        )
    except (ConfigError, OSError, ValueError, WorkflowError) as exc:
        print(f"error: run-baseline failed: {exc}", file=sys.stderr)
        return 2

    print(f"saved feature dataset to {result['feature_dataset']}")
    print(f"saved model artifact to {result['model']}")
    print(f"saved report bundle to {result['report']}")
    _print_metric_comparison(result["metrics"], result["benchmark_metrics"])
    return 0


def _raw_dataset_name(*, symbol: str, interval: str, start: str, end: str) -> str:
    collected_at = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    start_label = _timestamp_label(start)
    end_label = _timestamp_label(end)
    return f"{symbol.lower()}_{interval}_raw_{start_label}_{end_label}_{collected_at}.parquet"


def _timestamp_label(value: str) -> str:
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _print_metric_comparison(
    metrics: dict[str, object],
    benchmark_metrics: dict[str, dict[str, object]],
) -> None:
    print(
        "strategy "
        f"total_return={_format_metric(metrics.get('total_return'))} "
        f"max_drawdown={_format_metric(metrics.get('max_drawdown'))} "
        f"trades={metrics.get('trade_count')}"
    )
    for name in sorted(benchmark_metrics):
        values = benchmark_metrics[name]
        print(
            f"{name} "
            f"total_return={_format_metric(values.get('total_return'))} "
            f"max_drawdown={_format_metric(values.get('max_drawdown'))}"
        )


def _format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
