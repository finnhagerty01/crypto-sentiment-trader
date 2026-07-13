"""Run Model Refinement 01 threshold diagnostics from a feature dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from trader.config import load_config
from trader.modeling.thresholds import (
    run_threshold_sweep,
    write_threshold_sweep_artifacts,
)


DEFAULT_FEATURES = Path("artifacts/datasets/btcusdt_1h_features_e5ddf65b05a6.parquet")
DEFAULT_OUTPUT_DIR = Path("artifacts/model_refinement/threshold_sweep_01")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run threshold sweep diagnostics on a saved feature dataset."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES,
        help=f"Feature parquet path. Default: {DEFAULT_FEATURES}",
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
    args = parser.parse_args()

    if not args.features.exists():
        parser.error(f"feature dataset does not exist: {args.features}")
    if not args.config.exists():
        parser.error(f"config does not exist: {args.config}")

    config = load_config(args.config)
    data = pd.read_parquet(args.features)
    result = run_threshold_sweep(data, config)
    write_threshold_sweep_artifacts(result, args.output_dir)

    print(f"selected_threshold: {result.selected_threshold}")
    print(f"output_dir: {args.output_dir}")
    print(f"summary_csv: {args.output_dir / 'threshold_summary.csv'}")
    print(f"fold_metrics_csv: {args.output_dir / 'threshold_fold_metrics.csv'}")
    print(f"selected_json: {args.output_dir / 'selected_threshold.json'}")
    print(f"holdout_json: {args.output_dir / 'holdout_threshold_metrics.json'}")
    print()
    print(result.summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
