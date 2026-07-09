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

    for command in COMMANDS[1:]:
        subparsers.add_parser(
            command,
            help="reserved for a later rebuild phase",
            description=f"{command} is reserved for a later rebuild phase.",
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command is None:
        build_parser().print_help()
        return 0

    if args.command == "collect-market":
        return _collect_market(args)

    print(
        f"error: command '{args.command}' is not implemented in the current phase",
        file=sys.stderr,
    )
    return 2


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


if __name__ == "__main__":
    raise SystemExit(main())
