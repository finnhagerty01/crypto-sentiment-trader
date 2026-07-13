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
from trader.modeling.candle_intervals import (
    CandleIntervalComparisonError,
    run_candle_interval_comparison,
)
from trader.modeling.experiments import ExperimentConfigError, run_experiment_grid
from trader.modeling.model_class_comparison import (
    ModelClassComparisonError,
    run_model_class_comparison,
)
from trader.modeling.regime_specialists import (
    RegimeSpecialistComparisonError,
    run_regime_specialist_comparison,
)
from trader.modeling.symbol_interval_grid import (
    DEFAULT_END as SYMBOL_GRID_DEFAULT_END,
    DEFAULT_INTERVALS as SYMBOL_GRID_DEFAULT_INTERVALS,
    DEFAULT_START as SYMBOL_GRID_DEFAULT_START,
    DEFAULT_SYMBOLS as SYMBOL_GRID_DEFAULT_SYMBOLS,
    SymbolIntervalGridError,
    run_symbol_interval_grid,
)
from trader.sentiment.gate import SentimentGateError, run_sentiment_gate
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
    "run-experiment-grid",
    "run-sentiment-gate",
    "run-model-class-comparison",
    "run-candle-interval-comparison",
    "run-regime-specialist-comparison",
    "run-symbol-interval-grid",
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

    experiment_grid = subparsers.add_parser(
        "run-experiment-grid",
        help="compare named model-refinement variants from a saved market dataset",
        description=(
            "Run a deterministic named-variant experiment grid from a saved market "
            "dataset. This command never fetches external data."
        ),
    )
    experiment_grid.add_argument(
        "--market-data",
        required=True,
        help="saved raw market Parquet dataset with metadata sidecar",
    )
    experiment_grid.add_argument(
        "--experiment-config",
        required=True,
        help="YAML file defining named variants and dot-path overrides",
    )
    experiment_grid.add_argument(
        "--output-dir",
        required=True,
        help="directory under which the run_id experiment directory is written",
    )
    experiment_grid.add_argument(
        "--run-id",
        required=True,
        help="deterministic experiment run directory name",
    )

    sentiment_gate = subparsers.add_parser(
        "run-sentiment-gate",
        help="reevaluate saved hourly sentiment against the fixed market-only setup",
        description=(
            "Run the Model Refinement 05 sentiment reevaluation gate from saved "
            "market and hourly sentiment datasets. This command never fetches "
            "Reddit, Binance, or external data."
        ),
    )
    sentiment_gate.add_argument(
        "--market-data",
        required=True,
        help="saved raw market Parquet dataset with metadata sidecar",
    )
    sentiment_gate.add_argument(
        "--hourly-sentiment",
        required=True,
        help="saved hourly sentiment Parquet dataset with metadata sidecar",
    )
    sentiment_gate.add_argument(
        "--output-dir",
        required=True,
        help="directory under which the run_id sentiment gate directory is written",
    )
    sentiment_gate.add_argument(
        "--run-id",
        required=True,
        help="deterministic sentiment gate run directory name",
    )

    model_class = subparsers.add_parser(
        "run-model-class-comparison",
        help="compare fixed model classes from a saved market dataset",
        description=(
            "Run the Model Refinement 06 model-class comparison from a saved "
            "market dataset. This command never fetches Reddit, Binance, or "
            "external data."
        ),
    )
    model_class.add_argument("--config", default="configs/baseline.yaml")
    model_class.add_argument(
        "--market-data",
        required=True,
        help="saved raw market Parquet dataset with metadata sidecar",
    )
    model_class.add_argument(
        "--output-dir",
        required=True,
        help="directory under which the run_id model comparison directory is written",
    )
    model_class.add_argument(
        "--run-id",
        required=True,
        help="deterministic model comparison run directory name",
    )
    model_class.add_argument(
        "--enable-xgboost",
        action="store_true",
        help="evaluate XGBoost only when the optional dependency is installed",
    )

    candle_intervals = subparsers.add_parser(
        "run-candle-interval-comparison",
        help="compare resampled BTCUSDT candle intervals from saved 1h data",
        description=(
            "Run an offline candle interval diagnostic from a saved BTCUSDT 1h "
            "market dataset. This command never fetches market or external data."
        ),
    )
    candle_intervals.add_argument("--config", default="configs/baseline.yaml")
    candle_intervals.add_argument(
        "--market-data",
        required=True,
        help="saved raw 1h market Parquet dataset with metadata sidecar",
    )
    candle_intervals.add_argument(
        "--output-dir",
        required=True,
        help="directory under which the run_id interval comparison directory is written",
    )
    candle_intervals.add_argument(
        "--run-id",
        required=True,
        help="deterministic candle interval run directory name",
    )
    candle_intervals.add_argument(
        "--intervals",
        default="1h,4h,12h,1d",
        help="comma-separated intervals to compare; defaults to 1h,4h,12h,1d",
    )
    candle_intervals.add_argument(
        "--max-development-exposure",
        type=float,
        default=0.80,
        help="maximum median development exposure for threshold eligibility",
    )

    regime_specialists = subparsers.add_parser(
        "run-regime-specialist-comparison",
        help="compare causal bull/bear regime specialist candidates",
        description=(
            "Run an offline rule-regime specialist diagnostic from a saved BTCUSDT "
            "1h market dataset. This command never fetches market or external data."
        ),
    )
    regime_specialists.add_argument("--config", default="configs/baseline.yaml")
    regime_specialists.add_argument(
        "--market-data",
        required=True,
        help="saved raw 1h market Parquet dataset with metadata sidecar",
    )
    regime_specialists.add_argument(
        "--output-dir",
        required=True,
        help="directory under which the run_id regime comparison directory is written",
    )
    regime_specialists.add_argument(
        "--run-id",
        required=True,
        help="deterministic regime comparison run directory name",
    )
    regime_specialists.add_argument(
        "--lookback-bars",
        type=int,
        default=168,
        help="trailing bars used for causal bull/bear regime labels",
    )

    symbol_interval_grid = subparsers.add_parser(
        "run-symbol-interval-grid",
        help="collect/reuse liquid symbols and compare 1h through 14h intervals",
        description=(
            "Run a cross-symbol market-only interval diagnostic. Binance.US "
            "availability is checked at runtime; unavailable symbols are "
            "recorded in diagnostics and skipped. This command never starts "
            "paper trading."
        ),
    )
    symbol_interval_grid.add_argument("--config", default="configs/baseline.yaml")
    symbol_interval_grid.add_argument(
        "--output-dir",
        required=True,
        help="directory under which the run_id symbol interval directory is written",
    )
    symbol_interval_grid.add_argument(
        "--run-id",
        required=True,
        help="deterministic symbol interval run directory name",
    )
    symbol_interval_grid.add_argument(
        "--symbols",
        default=",".join(SYMBOL_GRID_DEFAULT_SYMBOLS),
        help="comma-separated Binance.US spot symbols to evaluate",
    )
    symbol_interval_grid.add_argument(
        "--intervals",
        default=",".join(SYMBOL_GRID_DEFAULT_INTERVALS),
        help="comma-separated intervals to compare; defaults to 1h through 14h",
    )
    symbol_interval_grid.add_argument(
        "--start",
        default=SYMBOL_GRID_DEFAULT_START,
        help="UTC start timestamp for 1h collection, inclusive",
    )
    symbol_interval_grid.add_argument(
        "--end",
        default=SYMBOL_GRID_DEFAULT_END,
        help="UTC end timestamp for 1h collection, exclusive",
    )
    symbol_interval_grid.add_argument(
        "--max-development-exposure",
        type=float,
        default=0.80,
        help="maximum median development exposure for threshold eligibility",
    )
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
    if args.command == "run-experiment-grid":
        return _run_experiment_grid(args)
    if args.command == "run-sentiment-gate":
        return _run_sentiment_gate(args)
    if args.command == "run-model-class-comparison":
        return _run_model_class_comparison(args)
    if args.command == "run-candle-interval-comparison":
        return _run_candle_interval_comparison(args)
    if args.command == "run-regime-specialist-comparison":
        return _run_regime_specialist_comparison(args)
    if args.command == "run-symbol-interval-grid":
        return _run_symbol_interval_grid(args)

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


def _run_experiment_grid(args: argparse.Namespace) -> int:
    try:
        result = run_experiment_grid(
            market_dataset_path=args.market_data,
            experiment_config_path=args.experiment_config,
            output_dir=args.output_dir,
            run_id=args.run_id,
        )
    except (ConfigError, ExperimentConfigError, OSError, ValueError, WorkflowError) as exc:
        print(f"error: run-experiment-grid failed: {exc}", file=sys.stderr)
        return 2

    print(f"saved experiment artifacts to {result['run_dir']}")
    selected = result["selected_variant"]
    if selected is None:
        print("selected_variant=None")
    else:
        print(
            "selected_variant="
            f"{selected['variant_name']} "
            f"threshold={_format_metric(selected.get('selected_threshold'))} "
            f"dev_total_return={_format_metric(selected.get('selected_median_total_return'))} "
            f"holdout_total_return={_format_metric(selected.get('holdout_total_return'))}"
        )
    return 0


def _run_sentiment_gate(args: argparse.Namespace) -> int:
    try:
        config = load_config("configs/baseline.yaml")
        result = run_sentiment_gate(
            market_dataset_path=args.market_data,
            hourly_sentiment_path=args.hourly_sentiment,
            output_dir=args.output_dir,
            run_id=args.run_id,
            config=config,
        )
    except (
        ConfigError,
        ExperimentConfigError,
        SentimentGateError,
        OSError,
        FileExistsError,
        ValueError,
        WorkflowError,
    ) as exc:
        print(f"error: run-sentiment-gate failed: {exc}", file=sys.stderr)
        return 2

    decision = result["decision"]
    print(f"saved sentiment gate to {result['run_dir']}")
    print(f"decision={decision['decision']}")
    print(f"phase11_remains_blocked={decision['phase11_remains_blocked']}")
    return 0


def _run_model_class_comparison(args: argparse.Namespace) -> int:
    try:
        result = run_model_class_comparison(
            market_dataset_path=args.market_data,
            output_dir=args.output_dir,
            run_id=args.run_id,
            config_path=args.config,
            enable_xgboost=args.enable_xgboost,
        )
    except (
        ConfigError,
        ModelClassComparisonError,
        OSError,
        FileExistsError,
        ValueError,
        WorkflowError,
    ) as exc:
        print(f"error: run-model-class-comparison failed: {exc}", file=sys.stderr)
        return 2

    decision = result["decision"]
    print(f"saved model-class comparison to {result['run_dir']}")
    print(f"decision={decision['decision']}")
    print(f"phase_11_status={decision['phase_11_status']}")
    return 0


def _run_candle_interval_comparison(args: argparse.Namespace) -> int:
    try:
        intervals = tuple(
            interval.strip()
            for interval in str(args.intervals).split(",")
            if interval.strip()
        )
        result = run_candle_interval_comparison(
            market_dataset_path=args.market_data,
            output_dir=args.output_dir,
            run_id=args.run_id,
            config_path=args.config,
            intervals=intervals,
            max_development_exposure=args.max_development_exposure,
        )
        print(f"wrote candle interval comparison to {result['run_dir']}")
        return 0
    except (
        CandleIntervalComparisonError,
        ConfigError,
        FileExistsError,
        OSError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _run_regime_specialist_comparison(args: argparse.Namespace) -> int:
    try:
        result = run_regime_specialist_comparison(
            market_dataset_path=args.market_data,
            output_dir=args.output_dir,
            run_id=args.run_id,
            config_path=args.config,
            lookback_bars=args.lookback_bars,
        )
    except (
        RegimeSpecialistComparisonError,
        ConfigError,
        FileExistsError,
        OSError,
        ValueError,
    ) as exc:
        print(f"error: run-regime-specialist-comparison failed: {exc}", file=sys.stderr)
        return 2

    decision = result["decision"]
    print(f"saved regime specialist comparison to {result['run_dir']}")
    print(f"decision={decision['decision']}")
    print(f"phase_11_status={decision['phase_11_status']}")
    return 0


def _run_symbol_interval_grid(args: argparse.Namespace) -> int:
    try:
        result = run_symbol_interval_grid(
            output_dir=args.output_dir,
            run_id=args.run_id,
            config_path=args.config,
            symbols=_parse_csv_tuple(args.symbols),
            intervals=_parse_csv_tuple(args.intervals),
            start=args.start,
            end=args.end,
            max_development_exposure=args.max_development_exposure,
        )
    except (
        ConfigError,
        SymbolIntervalGridError,
        MarketCollectionError,
        FileExistsError,
        OSError,
        ValueError,
    ) as exc:
        print(f"error: run-symbol-interval-grid failed: {exc}", file=sys.stderr)
        return 2

    decision = result["decision"]
    print(f"saved symbol interval grid to {result['run_dir']}")
    print(
        "selected_development_ranked_symbol_interval="
        f"{decision['selected_development_ranked_symbol_interval']}"
    )
    print(f"holdout_confirmation_result={decision['holdout_confirmation_result']}")
    print(f"phase_11_status={decision['phase_11_status']}")
    return 0


def _parse_csv_tuple(value: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not values:
        raise ValueError("at least one value is required")
    return values


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
