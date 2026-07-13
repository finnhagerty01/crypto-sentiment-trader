"""Run Model Refinement 03 market feature-group diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from trader.config import load_config
from trader.modeling.market_feature_groups import (
    run_market_feature_group_experiment,
)


DEFAULT_MARKET_DATA = Path(
    "artifacts/datasets/"
    "btcusdt_1h_raw_20260101T000000Z_20260601T000000Z_20260712T174420Z.parquet"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/model_refinement/market_feature_groups")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run controlled market feature-group experiments."
    )
    parser.add_argument(
        "--market-data",
        type=Path,
        default=DEFAULT_MARKET_DATA,
        help=f"Raw market parquet path. Default: {DEFAULT_MARKET_DATA}",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.yaml"),
        help="Baseline config path. Default: configs/baseline.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing feature-group artifacts.",
    )
    args = parser.parse_args()

    if not args.market_data.exists():
        parser.error(f"market dataset does not exist: {args.market_data}")
    if not args.config.exists():
        parser.error(f"config does not exist: {args.config}")

    result = run_market_feature_group_experiment(
        market_dataset_path=args.market_data,
        config=load_config(args.config),
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )

    print(f"group_count: {len(result.results)}")
    if result.selected_group is None:
        print("selected_feature_group: None")
    else:
        print(f"selected_feature_group: {result.selected_group['feature_group_id']}")
        print(f"selected_threshold: {result.selected_group['selected_threshold']}")
    print(f"output_dir: {result.output_dir}")
    print(f"results_csv: {result.output_dir / 'feature_group_results.csv'}")
    print(f"selected_json: {result.output_dir / 'selected_feature_group.json'}")
    print(f"report: {result.output_dir / 'market_feature_group_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
