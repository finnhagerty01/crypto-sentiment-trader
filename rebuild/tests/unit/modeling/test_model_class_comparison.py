from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

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
from trader.modeling.model_class_comparison import (
    _candidate_specs,
    _decision,
    model_class_comparison_config,
    run_model_class_comparison,
)


def test_model_class_comparison_writes_required_artifacts_and_fixed_controls(
    tmp_path: Path,
) -> None:
    market_path = _write_market_fixture(tmp_path)
    config_path = _write_config(tmp_path)

    result = run_model_class_comparison(
        market_dataset_path=market_path,
        output_dir=tmp_path / "out",
        run_id="model-classes",
        config_path=config_path,
    )

    run_dir = result["run_dir"]
    for relative in (
        "model_class_summary.csv",
        "threshold_summary.csv",
        "fold_metrics.csv",
        "holdout_metrics.csv",
        "benchmark_metrics.csv",
        "model_diagnostics.csv",
        "model_class_decision.json",
        "candidates/logistic_regression/resolved_config.yaml",
        "candidates/logistic_regression/feature_columns.json",
        "candidates/logistic_regression/model_metadata.json",
        "candidates/random_forest/resolved_config.yaml",
        "candidates/random_forest/feature_columns.json",
        "candidates/random_forest/model_metadata.json",
        "candidates/gradient_boosting/resolved_config.yaml",
        "candidates/gradient_boosting/feature_columns.json",
        "candidates/gradient_boosting/model_metadata.json",
        "candidates/xgboost/model_metadata.json",
    ):
        assert (run_dir / relative).exists(), relative

    summary = pd.read_csv(run_dir / "model_class_summary.csv")
    diagnostics = pd.read_csv(run_dir / "model_diagnostics.csv")
    decision = json.loads((run_dir / "model_class_decision.json").read_text())
    assert set(summary["candidate_name"]) == {
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "xgboost",
    }
    assert set(summary["horizon_bars"].dropna()) == {12.0}
    assert set(summary["cost_buffer"].dropna()) == {"none"}
    assert set(summary["enabled_groups"].dropna()) == {"baseline"}
    assert set(diagnostics["probability_calibration"]) == {"uncalibrated_predict_proba"}
    assert decision["phase_11_status"] == "blocked"
    assert decision["holdout_used_for_ranking"] is False


def test_model_classes_share_identical_config_feature_data_and_splits(
    tmp_path: Path,
) -> None:
    market_path = _write_market_fixture(tmp_path)
    config_path = _write_config(tmp_path)

    result = run_model_class_comparison(
        market_dataset_path=market_path,
        output_dir=tmp_path / "out",
        run_id="model-classes",
        config_path=config_path,
    )

    run_dir = result["run_dir"]
    configs = [
        (run_dir / "candidates" / name / "resolved_config.yaml").read_text()
        for name in ("logistic_regression", "random_forest", "gradient_boosting")
    ]
    assert len(set(configs)) == 1

    features = [
        json.loads(
            (run_dir / "candidates" / name / "feature_columns.json").read_text()
        )
        for name in ("logistic_regression", "random_forest", "gradient_boosting")
    ]
    assert len({item["feature_data_sha256"] for item in features}) == 1
    assert len({item["fold_signature_sha256"] for item in features}) == 1
    assert len({tuple(item["feature_columns"]) for item in features}) == 1


def test_threshold_and_holdout_artifacts_keep_scopes_separate(tmp_path: Path) -> None:
    market_path = _write_market_fixture(tmp_path)
    config_path = _write_config(tmp_path)

    result = run_model_class_comparison(
        market_dataset_path=market_path,
        output_dir=tmp_path / "out",
        run_id="model-classes",
        config_path=config_path,
    )

    run_dir = result["run_dir"]
    threshold = pd.read_csv(run_dir / "threshold_summary.csv")
    fold_metrics = pd.read_csv(run_dir / "fold_metrics.csv")
    holdout = pd.read_csv(run_dir / "holdout_metrics.csv")
    summary = pd.read_csv(run_dir / "model_class_summary.csv")

    assert set(fold_metrics["period"].dropna()) == {"development"}
    assert {"candidate_name", "grid_index", "threshold", "median_total_return"}.issubset(
        threshold.columns
    )
    assert {"candidate_name", "grid_index", "total_return", "max_drawdown"}.issubset(
        holdout.columns
    )
    assert "rank" in summary.columns
    assert "holdout_total_return" in summary.columns


def test_results_are_deterministic_for_fixed_seed(tmp_path: Path) -> None:
    market_path = _write_market_fixture(tmp_path)
    config_path = _write_config(tmp_path)

    first = run_model_class_comparison(
        market_dataset_path=market_path,
        output_dir=tmp_path / "first",
        run_id="model-classes",
        config_path=config_path,
    )["run_dir"]
    second = run_model_class_comparison(
        market_dataset_path=market_path,
        output_dir=tmp_path / "second",
        run_id="model-classes",
        config_path=config_path,
    )["run_dir"]

    for relative in (
        "model_class_summary.csv",
        "threshold_summary.csv",
        "fold_metrics.csv",
        "holdout_metrics.csv",
        "benchmark_metrics.csv",
        "model_diagnostics.csv",
        "model_class_decision.json",
    ):
        assert (first / relative).read_text() == (second / relative).read_text()


def test_xgboost_skipped_by_default_and_cleanly_when_enabled_but_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    default_specs = _candidate_specs(enable_xgboost=False)
    assert default_specs[-1].name == "xgboost"
    assert "disabled" in str(default_specs[-1].skipped_reason)

    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    enabled_specs = _candidate_specs(enable_xgboost=True)
    assert enabled_specs[-1].name == "xgboost"
    assert "unavailable" in str(enabled_specs[-1].skipped_reason)


def test_decision_rejects_when_model_class_change_does_not_help() -> None:
    summary = pd.DataFrame(
        [
            {
                "candidate_name": "logistic_regression",
                "model_class": "logistic_regression",
                "status": "ok",
                "candidate_order": 0,
                "selected_threshold": 0.30,
                "selected_median_total_return": 0.02,
                "selected_median_cash_total_return": 0.0,
                "selected_median_max_drawdown": -0.03,
                "selected_median_turnover": 2.0,
                "holdout_total_return": -0.01,
                "holdout_max_drawdown": -0.04,
                "holdout_turnover": 2.0,
            },
            {
                "candidate_name": "random_forest",
                "model_class": "random_forest",
                "status": "ok",
                "candidate_order": 1,
                "selected_threshold": 0.30,
                "selected_median_total_return": 0.01,
                "selected_median_cash_total_return": 0.0,
                "selected_median_max_drawdown": -0.03,
                "selected_median_turnover": 2.0,
                "holdout_total_return": 0.05,
                "holdout_max_drawdown": -0.04,
                "holdout_turnover": 2.0,
            },
        ]
    )

    decision = _decision(summary)

    assert decision["decision"] == "model_class_change_rejected"
    assert decision["selected_model_class"] == "logistic_regression"
    assert decision["phase_11_status"] == "blocked"


def test_fixed_comparison_config_only_changes_refinement_controls() -> None:
    base = _test_config()

    fixed = model_class_comparison_config(base)

    assert fixed.target.horizon_bars == 12
    assert fixed.target.cost_buffer == "none"
    assert fixed.target.volatility_multiplier == 0.10
    assert fixed.features.enabled_groups == ("baseline",)
    assert fixed.validation == base.validation
    assert fixed.costs == base.costs
    assert fixed.backtest == base.backtest


def _write_market_fixture(tmp_path: Path) -> Path:
    rows = 240
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=rows, freq="h")
    index = np.arange(rows, dtype="float64")
    close = 100.0 + 0.03 * index + 2.5 * np.sin(index / 5.0)
    open_ = close + 0.1 * np.cos(index / 3.0)
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    volume = 100.0 + 12.0 * np.cos(index / 6.0)
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


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        """\
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
""",
        encoding="utf-8",
    )
    return path


def _test_config() -> TraderConfig:
    return TraderConfig(
        data=DataConfig(symbol="BTCUSDT", interval="1h"),
        features=FeaturesConfig(
            enabled_groups=("baseline", "trend"),
            volatility_window=4,
            volume_window=4,
            rsi_window=4,
            clipping_window=12,
            clipping_mad_multiplier=8.0,
        ),
        target=TargetConfig(
            horizon_bars=1,
            cost_buffer="round_trip",
            volatility_multiplier=0.0,
        ),
        model=ModelConfig(probability_threshold=0.5, regularization_c=1.0),
        validation=ValidationConfig(
            minimum_train_bars=80,
            test_bars=20,
            step_bars=20,
            final_holdout_fraction=0.2,
        ),
        costs=CostsConfig(fee_per_side=0.000001, slippage_per_side=0.000001),
        backtest=BacktestConfig(initial_capital=1000.0),
    )
