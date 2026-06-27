"""Command-line entry point for the staged rebuild."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import sys


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
    for command in COMMANDS:
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

    print(
        f"error: command '{args.command}' is not implemented in Phase 03",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
