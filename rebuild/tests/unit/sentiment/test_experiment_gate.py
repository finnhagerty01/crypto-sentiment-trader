from __future__ import annotations

import json
from pathlib import Path

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
from trader.features.market import build_feature_dataset, model_feature_columns
from trader.sentiment.features import SENTIMENT_VARIANT_ORDER, sentiment_feature_columns
from trader.sentiment.gate import (
    MARKET_ONLY_VARIANT,
    SENTIMENT_ONLY_VARIANT,
    fixed_market_only_config,
    join_hourly_sentiment,
    run_sentiment_gate,
)
from trader.sentiment.storage import write_hourly_sentiment_dataset


def test_sentiment_gate_writes_research_only_rejection_path(tmp_path: Path) -> None:
    market_path = _write_market_fixture(tmp_path)
    sentiment_path = _write_hourly_sentiment_fixture(tmp_path, rows=240)

    result = run_sentiment_gate(
        market_dataset_path=market_path,
        hourly_sentiment_path=sentiment_path,
        output_dir=tmp_path / "out",
        run_id="gate",
        config=_config(),
    )

    run_dir = result["run_dir"]
    for relative in (
        "sentiment_gate_summary.csv",
        "sentiment_target_distribution.csv",
        "sentiment_threshold_summary.csv",
        "sentiment_fold_metrics.csv",
        "sentiment_holdout_metrics.csv",
        "sentiment_benchmark_metrics.csv",
        "sentiment_feature_diagnostics.csv",
        "sentiment_gate_decision.json",
        "variants/market_only/resolved_config.yaml",
        "variants/market_only/feature_columns.json",
        "variants/hourly_mean/resolved_config.yaml",
        "variants/sentiment_only/feature_columns.json",
    ):
        assert (run_dir / relative).exists()

    summary = pd.read_csv(run_dir / "sentiment_gate_summary.csv")
    decision = json.loads((run_dir / "sentiment_gate_decision.json").read_text())
    assert summary["variant_name"].tolist() == [
        MARKET_ONLY_VARIANT,
        *SENTIMENT_VARIANT_ORDER,
        SENTIMENT_ONLY_VARIANT,
    ]
    assert decision["sentiment_kept_for_phase11"] is False
    assert decision["phase11_remains_blocked"] is True
    assert decision["sentiment_only_phase11_eligible"] is False
    sentiment_only = summary.loc[summary["variant_name"].eq(SENTIMENT_ONLY_VARIANT)].iloc[0]
    assert bool(sentiment_only["research_only"]) is True
    assert bool(sentiment_only["phase11_candidate"]) is False


def test_sentiment_variants_keep_identical_market_controls(tmp_path: Path) -> None:
    market_path = _write_market_fixture(tmp_path)
    sentiment_path = _write_hourly_sentiment_fixture(tmp_path, rows=240)
    result = run_sentiment_gate(
        market_dataset_path=market_path,
        hourly_sentiment_path=sentiment_path,
        output_dir=tmp_path / "out",
        run_id="gate",
        config=_config(),
    )

    run_dir = result["run_dir"]
    baseline_columns = _feature_columns(run_dir, MARKET_ONLY_VARIANT)
    hourly_columns = _feature_columns(run_dir, "hourly_mean")
    ewma_columns = _feature_columns(run_dir, "ewma_6h")

    assert hourly_columns["market_feature_columns"] == baseline_columns["market_feature_columns"]
    assert ewma_columns["market_feature_columns"] == baseline_columns["market_feature_columns"]
    assert hourly_columns["sentiment_feature_columns"] == list(
        sentiment_feature_columns("hourly_mean")
    )
    assert ewma_columns["sentiment_feature_columns"] == list(
        sentiment_feature_columns("ewma_6h")
    )
    assert (
        hourly_columns["feature_columns"][: len(baseline_columns["market_feature_columns"])]
        == baseline_columns["market_feature_columns"]
    )


def test_fixed_gate_config_forces_selected_market_only_controls() -> None:
    config = _config(
        enabled_groups=("baseline", "trend"),
        horizon_bars=1,
        cost_buffer="round_trip",
        probability_threshold=0.55,
    )

    fixed = fixed_market_only_config(config)

    assert fixed.features.enabled_groups == ("baseline",)
    assert fixed.target.horizon_bars == 12
    assert fixed.target.cost_buffer == "none"
    assert fixed.target.volatility_multiplier == 0.10
    assert fixed.model.probability_threshold == 0.30


def test_sentiment_join_is_timestamp_only_causal_and_keeps_missing_explicit() -> None:
    config = fixed_market_only_config(_config())
    market = _market_fixture(rows=40)
    features = build_feature_dataset(market, config)
    hourly = _hourly_sentiment_frame(rows=40).drop(index=[5]).reset_index(drop=True)
    baseline = join_hourly_sentiment(features, hourly)

    mutated = hourly.copy()
    mutated.loc[mutated.index[-1], "combined_sentiment_mean"] = 0.99
    mutated.loc[mutated.index[-1], "sentiment_ewma_6h"] = 0.99
    changed = join_hourly_sentiment(features, mutated)

    pd.testing.assert_frame_equal(
        baseline.loc[:10, ["timestamp", "combined_sentiment_mean", "sentiment_ewma_6h"]],
        changed.loc[:10, ["timestamp", "combined_sentiment_mean", "sentiment_ewma_6h"]],
    )
    assert baseline.loc[5, "sentiment_join_missing"] == 1
    assert baseline.loc[5, "sentiment_missing"] == 1
    assert baseline.loc[5, "combined_observation_count"] == 0


def test_sentiment_gate_output_is_deterministic_for_fixed_run_id(tmp_path: Path) -> None:
    market_path = _write_market_fixture(tmp_path)
    sentiment_path = _write_hourly_sentiment_fixture(tmp_path, rows=240)

    first = run_sentiment_gate(
        market_dataset_path=market_path,
        hourly_sentiment_path=sentiment_path,
        output_dir=tmp_path / "first",
        run_id="gate",
        config=_config(),
    )["run_dir"]
    second = run_sentiment_gate(
        market_dataset_path=market_path,
        hourly_sentiment_path=sentiment_path,
        output_dir=tmp_path / "second",
        run_id="gate",
        config=_config(),
    )["run_dir"]

    for relative in (
        "sentiment_gate_summary.csv",
        "sentiment_target_distribution.csv",
        "sentiment_threshold_summary.csv",
        "sentiment_fold_metrics.csv",
        "sentiment_holdout_metrics.csv",
        "sentiment_benchmark_metrics.csv",
        "sentiment_feature_diagnostics.csv",
        "sentiment_gate_decision.json",
        "variants/kalman_filtered/feature_columns.json",
    ):
        assert (first / relative).read_text(encoding="utf-8") == (
            second / relative
        ).read_text(encoding="utf-8")


def test_model_refinement_06_plan_exists_and_is_referenced() -> None:
    root = Path(__file__).parents[3]
    plan = root / "codex_rebuild" / "model_refinement" / "06_MODEL_CLASS_COMPARISON.md"
    readme = root / "codex_rebuild" / "model_refinement" / "README.md"

    assert plan.exists()
    assert "Logistic Regression" in plan.read_text(encoding="utf-8")
    assert "06_MODEL_CLASS_COMPARISON.md" in readme.read_text(encoding="utf-8")


def _feature_columns(run_dir: Path, variant: str) -> dict[str, object]:
    return json.loads(
        (run_dir / "variants" / variant / "feature_columns.json").read_text(
            encoding="utf-8"
        )
    )


def _config(
    *,
    enabled_groups: tuple[str, ...] = ("baseline",),
    horizon_bars: int = 1,
    cost_buffer: str = "round_trip",
    probability_threshold: float = 0.55,
) -> TraderConfig:
    return TraderConfig(
        data=DataConfig(symbol="BTCUSDT", interval="1h"),
        features=FeaturesConfig(
            enabled_groups=enabled_groups,
            volatility_window=8,
            volume_window=8,
            rsi_window=8,
            clipping_window=24,
            clipping_mad_multiplier=8.0,
        ),
        target=TargetConfig(
            horizon_bars=horizon_bars,
            cost_buffer=cost_buffer,
            volatility_multiplier=0.10,
        ),
        model=ModelConfig(probability_threshold=probability_threshold, regularization_c=1.0),
        validation=ValidationConfig(
            minimum_train_bars=60,
            test_bars=24,
            step_bars=24,
            final_holdout_fraction=0.20,
        ),
        costs=CostsConfig(fee_per_side=0.000001, slippage_per_side=0.000001),
        backtest=BacktestConfig(initial_capital=1000.0),
    )


def _write_market_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "market" / "btcusdt_1h.parquet"
    write_market_dataset(_market_fixture(rows=240), path, source="fixture")
    return path


def _market_fixture(*, rows: int) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=rows, freq="h")
    index = np.arange(rows, dtype="float64")
    close = 100.0 + 0.03 * index + 1.8 * np.sin(index / 4.0)
    open_ = close + 0.05 * np.cos(index / 3.0)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": "BTCUSDT",
            "open": open_,
            "high": np.maximum(open_, close) + 0.4,
            "low": np.minimum(open_, close) - 0.4,
            "close": close,
            "volume": 100.0 + 8.0 * np.cos(index / 5.0),
        }
    )


def _write_hourly_sentiment_fixture(tmp_path: Path, *, rows: int) -> Path:
    path, _ = write_hourly_sentiment_dataset(
        _hourly_sentiment_frame(rows=rows),
        tmp_path / "sentiment",
        dataset_id="hourly-fixture",
        source_dataset_id="raw-fixture",
        variant="all_features",
    )
    return path


def _hourly_sentiment_frame(*, rows: int) -> pd.DataFrame:
    timestamp = pd.date_range("2026-01-01T00:00:00Z", periods=rows, freq="h")
    index = np.arange(rows, dtype="float64")
    mean = 0.05 * np.sin(index / 7.0)
    missing = (index.astype(int) % 11 == 0).astype("int8")
    observation_count = np.where(missing == 1, 0, 2)
    reliability = observation_count / (observation_count + 5.0)
    filled = np.where(missing == 1, 0.0, mean)
    ewma_6h = pd.Series(filled).ewm(span=6, adjust=False).mean()
    ewma_24h = pd.Series(filled).ewm(span=24, adjust=False).mean()
    return pd.DataFrame(
        {
            "timestamp": timestamp,
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
