"""Run Model Refinement 02 target and horizon grid diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from trader.config import load_config
from trader.modeling.target_horizon_grid import run_target_horizon_grid


DEFAULT_MARKET_DATA = Path(
    "artifacts/datasets/"
    "btcusdt_1h_raw_20260101T000000Z_20260601T000000Z_20260712T174420Z.parquet"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/model_refinement/target_horizon_grid")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run target strictness and horizon experiments."
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
        help="Allow overwriting existing target/horizon grid artifacts.",
    )
    args = parser.parse_args()

    if not args.market_data.exists():
        parser.error(f"market dataset does not exist: {args.market_data}")
    if not args.config.exists():
        parser.error(f"config does not exist: {args.config}")

    config = load_config(args.config)
    result = run_target_horizon_grid(
        market_dataset_path=args.market_data,
        config=config,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )

    print(f"candidate_count: {len(result.results)}")
    if result.selected_candidate is None:
        print("selected_candidate: None")
    else:
        print(f"selected_candidate: {result.selected_candidate['candidate_id']}")
        print(f"selected_threshold: {result.selected_candidate['selected_threshold']}")
    print(f"output_dir: {result.output_dir}")
    print(f"results_csv: {result.output_dir / 'grid_results.csv'}")
    print(f"selected_json: {result.output_dir / 'selected_candidate.json'}")
    print(f"report: {result.output_dir / 'target_horizon_grid_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
