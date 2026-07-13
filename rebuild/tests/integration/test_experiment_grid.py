from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trader.cli import main
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
from trader.data.storage import write_market_dataset
from trader.modeling.experiments import _rank_variants
from trader.sentiment.storage import write_hourly_sentiment_dataset


def test_run_experiment_grid_from_saved_market_dataset_is_deterministic(
    tmp_path: Path,
    monkeypatch,
) -> None:
    market_path = _write_market_fixture(tmp_path)
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        _experiment_config(tmp_path / "base.yaml"),
        encoding="utf-8",
    )

    def fail_if_network_collection_is_used(**_: object) -> None:
        raise AssertionError("run-experiment-grid must not collect market data")

    monkeypatch.setattr(
        "trader.cli.collect_market_data",
        fail_if_network_collection_is_used,
    )

    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    assert _run_grid(market_path, config_path, first_output) == 0
    assert _run_grid(market_path, config_path, second_output) == 0

    first_run = first_output / "offline-grid"
    second_run = second_output / "offline-grid"
    for relative in (
        "experiment_config.yaml",
        "variant_summary.csv",
        "target_distribution.csv",
        "threshold_summary.csv",
        "fold_metrics.csv",
        "holdout_metrics.csv",
        "benchmark_metrics.csv",
        "variants/baseline_variant/resolved_config.yaml",
        "variants/trend_variant/resolved_config.yaml",
    ):
        assert (first_run / relative).exists()
        assert (first_run / relative).read_text(encoding="utf-8") == (
            second_run / relative
        ).read_text(encoding="utf-8")

    summary = pd.read_csv(first_run / "variant_summary.csv")
    holdout = pd.read_csv(first_run / "holdout_metrics.csv")
    benchmarks = pd.read_csv(first_run / "benchmark_metrics.csv")
    assert {"variant_name", "rank", "selected_median_total_return"}.issubset(summary.columns)
    assert {"variant_name", "total_return", "max_drawdown"}.issubset(holdout.columns)
    assert set(benchmarks["benchmark"]) >= {"cash", "buy_and_hold", "momentum_24h"}
    assert "holdout_total_return" in summary.columns


def test_run_experiment_grid_rejects_unknown_override_path(
    tmp_path: Path,
    capsys,
) -> None:
    market_path = _write_market_fixture(tmp_path)
    base_config = tmp_path / "base.yaml"
    base_config.write_text(_base_config(), encoding="utf-8")
    experiment_config = tmp_path / "bad-override.yaml"
    experiment_config.write_text(
        f"""\
variants:
  - name: bad_override
    config: {base_config}
    overrides:
      target.unknown_field: 1
""",
        encoding="utf-8",
    )

    result = _run_grid(market_path, experiment_config, tmp_path / "out")

    assert result == 2
    assert "unknown override path" in capsys.readouterr().err


def test_run_experiment_grid_rejects_duplicate_variant_names(
    tmp_path: Path,
    capsys,
) -> None:
    market_path = _write_market_fixture(tmp_path)
    base_config = tmp_path / "base.yaml"
    base_config.write_text(_base_config(), encoding="utf-8")
    experiment_config = tmp_path / "duplicate.yaml"
    experiment_config.write_text(
        f"""\
variants:
  - name: duplicate
    config: {base_config}
  - name: duplicate
    config: {base_config}
""",
        encoding="utf-8",
    )

    result = _run_grid(market_path, experiment_config, tmp_path / "out")

    assert result == 2
    assert "duplicate variant name" in capsys.readouterr().err


def test_run_sentiment_gate_cli_from_saved_fixtures_is_offline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    market_path = _write_market_fixture(tmp_path)
    sentiment_path = _write_hourly_sentiment_fixture(tmp_path)

    def fail_if_network_collection_is_used(**_: object) -> None:
        raise AssertionError("run-sentiment-gate must not collect external data")

    monkeypatch.setattr(
        "trader.cli.collect_market_data",
        fail_if_network_collection_is_used,
    )
    monkeypatch.setattr("trader.cli.load_config", lambda _: _sentiment_gate_config())

    result = main(
        [
            "run-sentiment-gate",
            "--market-data",
            str(market_path),
            "--hourly-sentiment",
            str(sentiment_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--run-id",
            "sentiment-gate",
        ]
    )

    run_dir = tmp_path / "out" / "sentiment-gate"
    assert result == 0
    assert (run_dir / "sentiment_gate_decision.json").exists()
    summary = pd.read_csv(run_dir / "sentiment_gate_summary.csv")
    holdout = pd.read_csv(run_dir / "sentiment_holdout_metrics.csv")
    assert "market_only" in set(summary["variant_name"])
    assert "sentiment_only" in set(summary["variant_name"])
    assert {"variant_name", "total_return", "max_drawdown"}.issubset(holdout.columns)


def test_run_model_class_comparison_cli_from_saved_fixture_is_offline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    market_path = _write_market_fixture(tmp_path)
    config_path = tmp_path / "base.yaml"
    config_path.write_text(_base_config(), encoding="utf-8")

    def fail_if_network_collection_is_used(**_: object) -> None:
        raise AssertionError("run-model-class-comparison must not collect external data")

    monkeypatch.setattr(
        "trader.cli.collect_market_data",
        fail_if_network_collection_is_used,
    )

    result = main(
        [
            "run-model-class-comparison",
            "--config",
            str(config_path),
            "--market-data",
            str(market_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--run-id",
            "model-classes",
        ]
    )

    run_dir = tmp_path / "out" / "model-classes"
    assert result == 0
    assert (run_dir / "model_class_decision.json").exists()
    summary = pd.read_csv(run_dir / "model_class_summary.csv")
    holdout = pd.read_csv(run_dir / "holdout_metrics.csv")
    diagnostics = pd.read_csv(run_dir / "model_diagnostics.csv")
    assert {"logistic_regression", "random_forest", "gradient_boosting", "xgboost"} == set(
        summary["candidate_name"]
    )
    assert {"candidate_name", "total_return", "max_drawdown"}.issubset(holdout.columns)
    xgboost = diagnostics.loc[diagnostics["candidate_name"] == "xgboost"].iloc[0]
    assert xgboost["status"] == "skipped"


def test_run_candle_interval_comparison_cli_from_saved_fixture_is_offline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    market_path = _write_long_market_fixture(tmp_path)
    config_path = tmp_path / "base.yaml"
    config_path.write_text(_base_config(), encoding="utf-8")

    def fail_if_network_collection_is_used(**_: object) -> None:
        raise AssertionError("run-candle-interval-comparison must not collect external data")

    monkeypatch.setattr(
        "trader.cli.collect_market_data",
        fail_if_network_collection_is_used,
    )

    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    args = [
        "run-candle-interval-comparison",
        "--config",
        str(config_path),
        "--market-data",
        str(market_path),
        "--intervals",
        "1h,4h",
        "--max-development-exposure",
        "0.8",
    ]

    assert main([*args, "--output-dir", str(first_output), "--run-id", "candles"]) == 0
    assert main([*args, "--output-dir", str(second_output), "--run-id", "candles"]) == 0

    first_run = first_output / "candles"
    second_run = second_output / "candles"
    for relative in (
        "interval_summary.csv",
        "threshold_summary.csv",
        "fold_metrics.csv",
        "holdout_metrics.csv",
        "benchmark_metrics.csv",
        "dataset_diagnostics.csv",
        "candle_interval_decision.json",
        "intervals/1h/resolved_config.yaml",
        "intervals/1h/resampled_market_metadata.json",
        "intervals/1h/feature_columns.json",
        "intervals/4h/resolved_config.yaml",
        "intervals/4h/resampled_market_metadata.json",
        "intervals/4h/feature_columns.json",
    ):
        assert (first_run / relative).exists()
        assert (first_run / relative).read_text(encoding="utf-8") == (
            second_run / relative
        ).read_text(encoding="utf-8")

    summary = pd.read_csv(first_run / "interval_summary.csv")
    thresholds = pd.read_csv(first_run / "threshold_summary.csv")
    diagnostics = pd.read_csv(first_run / "dataset_diagnostics.csv")
    decision = json.loads(
        (first_run / "candle_interval_decision.json").read_text(encoding="utf-8")
    )
    assert set(summary["interval"]) == {"1h", "4h"}
    assert "median_buy_hold_total_return" in thresholds.columns
    assert diagnostics.set_index("interval").loc["4h", "minimum_train_bars"] == 250
    assert diagnostics.set_index("interval").loc["4h", "target_horizon_bars"] == 1
    assert decision["holdout_used_for_ranking"] is False
    assert decision["phase_11_status"] == "blocked"


def test_run_regime_specialist_comparison_cli_from_saved_fixture_is_offline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    market_path = _write_regime_market_fixture(tmp_path)
    config_path = tmp_path / "base.yaml"
    config_path.write_text(_base_config(), encoding="utf-8")

    def fail_if_network_collection_is_used(**_: object) -> None:
        raise AssertionError("run-regime-specialist-comparison must not collect external data")

    monkeypatch.setattr(
        "trader.cli.collect_market_data",
        fail_if_network_collection_is_used,
    )

    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    args = [
        "run-regime-specialist-comparison",
        "--config",
        str(config_path),
        "--market-data",
        str(market_path),
        "--lookback-bars",
        "24",
    ]

    assert main([*args, "--output-dir", str(first_output), "--run-id", "regimes"]) == 0
    assert main([*args, "--output-dir", str(second_output), "--run-id", "regimes"]) == 0

    first_run = first_output / "regimes"
    second_run = second_output / "regimes"
    for relative in (
        "regime_summary.csv",
        "threshold_summary.csv",
        "fold_metrics.csv",
        "holdout_metrics.csv",
        "benchmark_metrics.csv",
        "regime_diagnostics.csv",
        "regime_specialist_decision.json",
        "candidates/global_baseline/resolved_config.yaml",
        "candidates/global_baseline/feature_columns.json",
        "candidates/regime_specialists/resolved_config.yaml",
        "candidates/regime_specialists/feature_columns.json",
        "candidates/regime_cash_filter/resolved_config.yaml",
        "candidates/regime_cash_filter/feature_columns.json",
    ):
        assert (first_run / relative).exists()
        assert (first_run / relative).read_text(encoding="utf-8") == (
            second_run / relative
        ).read_text(encoding="utf-8")

    summary = pd.read_csv(first_run / "regime_summary.csv")
    thresholds = pd.read_csv(first_run / "threshold_summary.csv")
    diagnostics = pd.read_csv(first_run / "regime_diagnostics.csv")
    decision = json.loads(
        (first_run / "regime_specialist_decision.json").read_text(encoding="utf-8")
    )
    assert set(summary["candidate_name"]) == {
        "global_baseline",
        "regime_specialists",
        "regime_cash_filter",
    }
    assert "median_buy_hold_total_return" in thresholds.columns
    assert set(diagnostics["period"]) == {"all", "development", "holdout"}
    assert decision["holdout_used_for_ranking"] is False
    assert decision["phase_11_status"] == "blocked"


def test_run_symbol_interval_grid_cli_from_local_fixtures_records_diagnostics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "base.yaml"
    config_path.write_text(_base_config(), encoding="utf-8")
    market_by_symbol = {
        "BTCUSDT": _market_frame("BTCUSDT", rows=1600, price_offset=0.0),
        "ETHUSDT": _market_frame("ETHUSDT", rows=1600, price_offset=20.0),
    }

    class FakeClient:
        def available_symbols(self) -> set[str]:
            return {"BTCUSDT", "ETHUSDT"}

    def fake_collect_market_data(**kwargs: object) -> pd.DataFrame:
        symbol = str(kwargs["symbol"])
        return market_by_symbol[symbol]

    monkeypatch.setattr(
        "trader.modeling.symbol_interval_grid.BinanceUsSpotKlineClient",
        lambda: FakeClient(),
    )
    monkeypatch.setattr(
        "trader.modeling.symbol_interval_grid.collect_market_data",
        fake_collect_market_data,
    )

    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    args = [
        "run-symbol-interval-grid",
        "--config",
        str(config_path),
        "--symbols",
        "BTCUSDT,ETHUSDT,UNIUSDT",
        "--intervals",
        "5h,7h,12h,13h",
        "--start",
        "2026-01-01T00:00:00Z",
        "--end",
        "2026-06-01T00:00:00Z",
    ]

    assert main([*args, "--output-dir", str(first_output), "--run-id", "symbols"]) == 0
    assert main([*args, "--output-dir", str(second_output), "--run-id", "symbols"]) == 0

    first_run = first_output / "symbols"
    second_run = second_output / "symbols"
    for relative in (
        "symbol_interval_summary.csv",
        "threshold_summary.csv",
        "fold_metrics.csv",
        "holdout_metrics.csv",
        "benchmark_metrics.csv",
        "dataset_diagnostics.csv",
        "symbol_interval_decision.json",
        "sentiment_provenance_audit.json",
        "symbols/BTCUSDT/intervals/5h/resolved_config.yaml",
        "symbols/BTCUSDT/intervals/5h/resampled_market_metadata.json",
        "symbols/BTCUSDT/intervals/5h/feature_columns.json",
        "symbols/ETHUSDT/intervals/13h/resolved_config.yaml",
        "symbols/ETHUSDT/intervals/13h/resampled_market_metadata.json",
        "symbols/ETHUSDT/intervals/13h/feature_columns.json",
    ):
        assert (first_run / relative).exists()
        assert (first_run / relative).read_text(encoding="utf-8") == (
            second_run / relative
        ).read_text(encoding="utf-8")

    summary = pd.read_csv(first_run / "symbol_interval_summary.csv")
    diagnostics = pd.read_csv(first_run / "dataset_diagnostics.csv")
    decision = json.loads(
        (first_run / "symbol_interval_decision.json").read_text(encoding="utf-8")
    )
    audit = json.loads(
        (first_run / "sentiment_provenance_audit.json").read_text(encoding="utf-8")
    )
    assert set(summary["symbol"]) == {"BTCUSDT", "ETHUSDT"}
    assert set(summary["interval"]) == {"5h", "7h", "12h", "13h"}
    assert diagnostics.loc[
        diagnostics["collection_status"] == "skipped_unavailable", "symbol"
    ].tolist() == ["UNIUSDT"]
    assert diagnostics.set_index(["symbol", "interval"]).loc[
        ("ETHUSDT", "13h"), "minimum_train_bars"
    ] == 77
    assert diagnostics.set_index(["symbol", "interval"]).loc[
        ("ETHUSDT", "13h"), "target_horizon_bars"
    ] == 1
    assert decision["holdout_used_for_ranking"] is False
    assert decision["phase_11_status"] == "blocked"
    assert "any_candidate_confirms_on_holdout" in decision
    assert (
        audit["prior_sentiment_gate_status"]
        == "inconclusive_for_bitcoin_specific_sentiment"
    )


def test_ranking_uses_development_metrics_not_holdout() -> None:
    summary = pd.DataFrame(
        [
            {
                "variant_name": "dev_winner_bad_holdout",
                "rank": None,
                "eligible": False,
                "variant_order": 0,
                "selected_threshold": 0.3,
                "selected_median_total_return": 0.05,
                "selected_median_cash_total_return": 0.0,
                "selected_median_max_drawdown": -0.02,
                "selected_median_turnover": 1.0,
                "holdout_total_return": -0.50,
            },
            {
                "variant_name": "dev_loser_good_holdout",
                "rank": None,
                "eligible": False,
                "variant_order": 1,
                "selected_threshold": 0.3,
                "selected_median_total_return": 0.02,
                "selected_median_cash_total_return": 0.0,
                "selected_median_max_drawdown": -0.01,
                "selected_median_turnover": 0.5,
                "holdout_total_return": 0.50,
            },
        ]
    )

    ranked = _rank_variants(summary)

    selected = ranked.loc[ranked["rank"] == 1].iloc[0]
    assert selected["variant_name"] == "dev_winner_bad_holdout"


def _run_grid(market_path: Path, config_path: Path, output_dir: Path) -> int:
    return main(
        [
            "run-experiment-grid",
            "--market-data",
            str(market_path),
            "--experiment-config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--run-id",
            "offline-grid",
        ]
    )


def _write_market_fixture(tmp_path: Path) -> Path:
    rows = 240
    timestamps = pd.date_range(
        "2026-01-01T00:00:00Z",
        periods=rows,
        freq="h",
    )
    index = np.arange(rows, dtype="float64")
    close = 100.0 + 0.04 * index + 2.0 * np.sin(index / 4.0)
    open_ = close + 0.15 * np.sin(index / 3.0)
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    volume = 100.0 + 10.0 * np.cos(index / 5.0)
    data = pd.DataFrame(
        {
            "timestamp": timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "symbol": "BTCUSDT",
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    market_path = tmp_path / "market" / "btcusdt_1h.parquet"
    write_market_dataset(data, market_path, source="fixture")
    return market_path


def _write_long_market_fixture(tmp_path: Path) -> Path:
    rows = 1500
    timestamps = pd.date_range(
        "2026-01-01T00:00:00Z",
        periods=rows,
        freq="h",
    )
    index = np.arange(rows, dtype="float64")
    close = 100.0 + 0.02 * index + 2.0 * np.sin(index / 12.0)
    open_ = close + 0.10 * np.sin(index / 5.0)
    high = np.maximum(open_, close) + 0.4
    low = np.minimum(open_, close) - 0.4
    volume = 100.0 + 10.0 * np.cos(index / 7.0)
    data = pd.DataFrame(
        {
            "timestamp": timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "symbol": "BTCUSDT",
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    market_path = tmp_path / "market" / "btcusdt_1h_long.parquet"
    write_market_dataset(data, market_path, source="fixture")
    return market_path


def _write_regime_market_fixture(tmp_path: Path) -> Path:
    rows = 1700
    timestamps = pd.date_range(
        "2026-01-01T00:00:00Z",
        periods=rows,
        freq="h",
    )
    index = np.arange(rows, dtype="float64")
    trend = 0.02 * index
    cycle = 12.0 * np.sin(index / 48.0)
    close = 120.0 + trend + cycle
    open_ = close + 0.15 * np.sin(index / 5.0)
    high = np.maximum(open_, close) + 0.6
    low = np.minimum(open_, close) - 0.6
    volume = 100.0 + 15.0 * np.cos(index / 9.0)
    data = pd.DataFrame(
        {
            "timestamp": timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "symbol": "BTCUSDT",
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    market_path = tmp_path / "market" / "btcusdt_1h_regime.parquet"
    write_market_dataset(data, market_path, source="fixture")
    return market_path


def _market_frame(symbol: str, *, rows: int, price_offset: float) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-01-01T00:00:00Z",
        periods=rows,
        freq="h",
    )
    index = np.arange(rows, dtype="float64")
    close = 100.0 + price_offset + 0.02 * index + 2.0 * np.sin(index / 12.0)
    open_ = close + 0.10 * np.sin(index / 5.0)
    high = np.maximum(open_, close) + 0.4
    low = np.minimum(open_, close) - 0.4
    volume = 100.0 + 10.0 * np.cos(index / 7.0)
    return pd.DataFrame(
        {
            "timestamp": timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "symbol": symbol,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _write_hourly_sentiment_fixture(tmp_path: Path) -> Path:
    rows = 240
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=rows, freq="h")
    index = np.arange(rows, dtype="float64")
    missing = (index.astype(int) % 13 == 0).astype("int8")
    mean = 0.03 * np.cos(index / 9.0)
    filled = np.where(missing == 1, 0.0, mean)
    ewma_6h = pd.Series(filled).ewm(span=6, adjust=False).mean()
    ewma_24h = pd.Series(filled).ewm(span=24, adjust=False).mean()
    observation_count = np.where(missing == 1, 0, 2)
    reliability = observation_count / (observation_count + 5.0)
    hourly = pd.DataFrame(
        {
            "timestamp": timestamps,
            "submission_sentiment_mean": np.where(missing == 1, np.nan, mean),
            "comment_sentiment_mean": np.where(missing == 1, np.nan, -mean),
            "combined_sentiment_mean": np.where(missing == 1, np.nan, 0.0),
            "submission_count": np.where(missing == 1, 0, 1),
            "comment_count": np.where(missing == 1, 0, 1),
            "subreddit_count": np.where(missing == 1, 0, 1),
            "combined_observation_count": observation_count,
            "sentiment_missing": missing,
            "sentiment_reliability": reliability,
            "sentiment_reliability_shrunk": filled * reliability,
            "sentiment_ewma_6h": ewma_6h,
            "sentiment_ewma_24h": ewma_24h,
            "sentiment_ewma_fast_minus_slow": ewma_6h - ewma_24h,
            "sentiment_lag_1h": pd.Series(filled).shift(1),
            "sentiment_kalman": pd.Series(filled).ewm(alpha=0.2, adjust=False).mean(),
        }
    )
    path, _ = write_hourly_sentiment_dataset(
        hourly,
        tmp_path / "sentiment",
        dataset_id="hourly-fixture",
        source_dataset_id="raw-fixture",
    )
    return path


def _sentiment_gate_config() -> TraderConfig:
    return TraderConfig(
        data=DataConfig(symbol="BTCUSDT", interval="1h"),
        features=FeaturesConfig(
            enabled_groups=("baseline",),
            volatility_window=8,
            volume_window=8,
            rsi_window=8,
            clipping_window=24,
            clipping_mad_multiplier=8.0,
        ),
        target=TargetConfig(
            horizon_bars=1,
            cost_buffer="round_trip",
            volatility_multiplier=0.10,
        ),
        model=ModelConfig(probability_threshold=0.55, regularization_c=1.0),
        validation=ValidationConfig(
            minimum_train_bars=60,
            test_bars=24,
            step_bars=24,
            final_holdout_fraction=0.20,
        ),
        costs=CostsConfig(fee_per_side=0.000001, slippage_per_side=0.000001),
        backtest=BacktestConfig(initial_capital=1000.0),
    )


def _experiment_config(base_config: Path) -> str:
    base_config.write_text(_base_config(), encoding="utf-8")
    return f"""\
variants:
  - name: baseline_variant
    config: {base_config}
    overrides:
      target.horizon_bars: 3
      target.cost_buffer: none
      target.volatility_multiplier: 0.0
      features.enabled_groups:
        - baseline
  - name: trend_variant
    config: {base_config}
    overrides:
      target.horizon_bars: 3
      target.cost_buffer: none
      target.volatility_multiplier: 0.0
      features.enabled_groups:
        - baseline
        - trend
"""


def _base_config() -> str:
    return """\
data:
  symbol: BTCUSDT
  interval: 1h
features:
  enabled_groups:
    - baseline
  volatility_window: 4
  volume_window: 4
  rsi_window: 4
  clipping_window: 12
  clipping_mad_multiplier: 8.0
target:
  horizon_bars: 1
  cost_buffer: round_trip
  volatility_multiplier: 0.0
model:
  probability_threshold: 0.5
  regularization_c: 1.0
validation:
  minimum_train_bars: 80
  test_bars: 20
  step_bars: 20
  final_holdout_fraction: 0.2
costs:
  fee_per_side: 0.000001
  slippage_per_side: 0.000001
backtest:
  initial_capital: 1000.0
"""
