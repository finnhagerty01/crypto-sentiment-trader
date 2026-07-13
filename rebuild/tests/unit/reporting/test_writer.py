from __future__ import annotations

from pathlib import Path

import pandas as pd

from trader.backtest.benchmarks import run_benchmarks
from trader.backtest.engine import run_long_cash_backtest
from trader.backtest.metrics import calculate_backtest_metrics
from trader.config import (
    BacktestConfig,
    CostsConfig,
    DataConfig,
    FeaturesConfig,
    ModelConfig,
    TargetConfig,
    TraderConfig,
    ValidationConfig,
)
from trader.data.storage import build_metadata
from trader.reporting.writer import write_backtest_report


def test_report_output_is_complete_and_deterministic(tmp_path: Path) -> None:
    config = _config()
    market = pd.read_csv(Path(__file__).parents[2] / "fixtures" / "btcusdt_1h.csv")
    predictions = pd.DataFrame(
        {
            "timestamp": market["timestamp"],
            "signal": [0, 1, 1, 0, 0, 1],
            "probability": [0.1, 0.8, 0.7, 0.2, 0.2, 0.9],
        }
    )
    result = run_long_cash_backtest(
        market,
        predictions,
        backtest_config=config.backtest,
        costs_config=config.costs,
    )
    metrics = calculate_backtest_metrics(result.equity, result.trades)
    benchmark_results = run_benchmarks(
        market,
        backtest_config=config.backtest,
        costs_config=config.costs,
        momentum_lookback_bars=3,
    )
    benchmark_metrics = {
        name: benchmark.metrics for name, benchmark in benchmark_results.items()
    }

    report_dir = write_backtest_report(
        tmp_path,
        config=config,
        dataset_metadata=build_metadata(market, source="unit-test"),
        model_metadata={"model_type": "unit-test"},
        fold_metrics=[{"fold": 1, "status": "ok"}],
        backtest=result,
        metrics=metrics,
        benchmark_metrics=benchmark_metrics,
        run_id="deterministic-run",
    )
    first_snapshot = {
        path.name: path.read_bytes() for path in sorted(report_dir.iterdir())
    }

    second_dir = write_backtest_report(
        tmp_path,
        config=config,
        dataset_metadata=build_metadata(market, source="unit-test"),
        model_metadata={"model_type": "unit-test"},
        fold_metrics=[{"fold": 1, "status": "ok"}],
        backtest=result,
        metrics=metrics,
        benchmark_metrics=benchmark_metrics,
        run_id="deterministic-run",
    )
    second_snapshot = {
        path.name: path.read_bytes() for path in sorted(second_dir.iterdir())
    }

    assert report_dir == second_dir
    assert set(first_snapshot) == {
        "config.yaml",
        "dataset_metadata.json",
        "model_metadata.json",
        "fold_metrics.csv",
        "predictions.csv",
        "trades.csv",
        "equity.csv",
        "metrics.json",
        "benchmark_metrics.json",
        "summary.md",
    }
    assert first_snapshot == second_snapshot


def _config() -> TraderConfig:
    return TraderConfig(
        data=DataConfig(symbol="BTCUSDT", interval="1h"),
        features=FeaturesConfig(
            volatility_window=24,
            volume_window=24,
            rsi_window=14,
            clipping_window=168,
            clipping_mad_multiplier=8.0,
        ),
        target=TargetConfig(
            horizon_bars=1,
            cost_buffer="round_trip",
            volatility_multiplier=0.1,
        ),
        model=ModelConfig(probability_threshold=0.55, regularization_c=1.0),
        validation=ValidationConfig(
            minimum_train_bars=10,
            test_bars=5,
            step_bars=5,
            final_holdout_fraction=0.2,
        ),
        costs=CostsConfig(fee_per_side=0.001, slippage_per_side=0.001),
        backtest=BacktestConfig(initial_capital=1000.0),
    )
