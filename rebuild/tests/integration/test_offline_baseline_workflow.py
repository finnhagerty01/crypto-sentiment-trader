from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trader.cli import main
from trader.data.storage import write_market_dataset


PROJECT_ROOT = Path(__file__).parents[2]
FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "btcusdt_1h.csv"


def test_run_baseline_offline_workflow_is_reproducible(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_dataset = tmp_path / "raw" / "btcusdt_1h.parquet"
    write_market_dataset(_expanded_fixture(), raw_dataset, source="fixture")
    config_path = tmp_path / "offline-config.yaml"
    config_path.write_text(_offline_config(), encoding="utf-8")

    def fail_if_network_collection_is_used(**_: object) -> None:
        raise AssertionError("run-baseline must not collect market data")

    monkeypatch.setattr(
        "trader.cli.collect_market_data",
        fail_if_network_collection_is_used,
    )

    first = _run_baseline(tmp_path / "first", raw_dataset, config_path)
    second = _run_baseline(tmp_path / "second", raw_dataset, config_path)

    assert first["dataset"].exists()
    assert first["model"].exists()
    assert first["report"].is_dir()
    for filename in (
        "predictions.csv",
        "metrics.json",
        "benchmark_metrics.json",
        "summary.md",
    ):
        assert (first["report"] / filename).exists()

    first_predictions = pd.read_csv(first["report"] / "predictions.csv")
    second_predictions = pd.read_csv(second["report"] / "predictions.csv")
    pd.testing.assert_frame_equal(first_predictions, second_predictions)

    first_metrics = json.loads((first["report"] / "metrics.json").read_text())
    second_metrics = json.loads((second["report"] / "metrics.json").read_text())
    assert first_metrics == second_metrics
    assert json.loads((first["report"] / "benchmark_metrics.json").read_text())


def _run_baseline(root: Path, raw_dataset: Path, config_path: Path) -> dict[str, Path]:
    dataset_dir = root / "datasets"
    model_dir = root / "models"
    report_dir = root / "reports"
    result = main(
        [
            "run-baseline",
            "--config",
            str(config_path),
            "--market-data",
            str(raw_dataset),
            "--dataset-dir",
            str(dataset_dir),
            "--model-dir",
            str(model_dir),
            "--report-dir",
            str(report_dir),
            "--run-id",
            "offline-demo",
        ]
    )

    assert result == 0
    datasets = sorted(dataset_dir.glob("*.parquet"))
    models = sorted(model_dir.glob("*.joblib"))
    assert len(datasets) == 1
    assert len(models) == 1
    return {
        "dataset": datasets[0],
        "model": models[0],
        "report": report_dir / "offline-demo",
    }


def _expanded_fixture() -> pd.DataFrame:
    base = pd.read_csv(FIXTURE)
    rows = []
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    for block in range(20):
        part = base.copy()
        timestamps = start + pd.to_timedelta(
            range(block * len(base), (block + 1) * len(base)),
            unit="h",
        )
        part["timestamp"] = timestamps.strftime("%Y-%m-%dT%H:%M:%SZ")
        price_scale = 1.0 + block * 0.001
        part[["open", "high", "low", "close"]] = (
            part[["open", "high", "low", "close"]] * price_scale
        )
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def _offline_config() -> str:
    return """\
data:
  symbol: BTCUSDT
  interval: 1h
features:
  volatility_window: 2
  volume_window: 2
  rsi_window: 2
  clipping_window: 4
  clipping_mad_multiplier: 8.0
target:
  horizon_bars: 1
  volatility_multiplier: 0.0
model:
  probability_threshold: 0.5
  regularization_c: 1.0
validation:
  minimum_train_bars: 40
  test_bars: 10
  step_bars: 10
  final_holdout_fraction: 0.2
costs:
  fee_per_side: 0.000001
  slippage_per_side: 0.000001
backtest:
  initial_capital: 1000.0
"""
