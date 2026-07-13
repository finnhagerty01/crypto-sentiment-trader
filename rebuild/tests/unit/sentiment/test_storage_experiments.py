from __future__ import annotations

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
from trader.features.market import MODEL_FEATURE_COLUMNS
from trader.sentiment.experiments import run_sentiment_ablation
from trader.sentiment.storage import (
    read_hourly_sentiment_dataset,
    read_sentiment_dataset,
    write_hourly_sentiment_dataset,
    write_sentiment_dataset,
)


def test_sentiment_storage_round_trip_preserves_separate_tables(tmp_path) -> None:
    submissions = pd.DataFrame(
        [
            {
                "submission_id": "s1",
                "subreddit": "Bitcoin",
                "created_utc": "2026-01-01T00:00:00Z",
                "title": "bullish",
                "selftext": "",
                "score": 10,
                "num_comments": 1,
                "author": "a",
                "permalink": "/s1",
                "collection_status": "collected",
                "collected_at": "2026-01-02T00:00:00Z",
                "source": "fixture",
            }
        ]
    )
    comments = pd.DataFrame(
        [
            {
                "comment_id": "c1",
                "submission_id": "s1",
                "parent_id": "s1",
                "subreddit": "Bitcoin",
                "created_utc": "2026-01-01T00:05:00Z",
                "body": "good",
                "score": 3,
                "author": None,
                "depth": 0,
                "collected_at": "2026-01-02T00:00:00Z",
                "source": "fixture",
            }
        ]
    )

    path, metadata = write_sentiment_dataset(
        submissions,
        comments,
        tmp_path,
        dataset_id="fixture-sentiment",
        source="fixture",
        extra_metadata={"max_comments_per_submission": 10},
    )
    loaded_submissions, loaded_comments, loaded_metadata = read_sentiment_dataset(path)

    assert metadata["submission_count"] == 1
    assert metadata["comment_count"] == 1
    assert loaded_metadata["extra"]["max_comments_per_submission"] == 10
    assert loaded_submissions.loc[0, "score"] == 10
    assert loaded_comments.loc[0, "parent_id"] == "s1"


def test_hourly_sentiment_storage_is_separate_from_raw_dataset(tmp_path) -> None:
    hourly = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
            "combined_sentiment_mean": [0.2, None],
            "sentiment_missing": [0, 1],
        }
    )

    path, metadata = write_hourly_sentiment_dataset(
        hourly,
        tmp_path,
        dataset_id="hourly-fixture",
        source_dataset_id="raw-fixture",
        variant="hourly_mean",
    )
    loaded, loaded_metadata = read_hourly_sentiment_dataset(path)

    assert path.name == "hourly-fixture.parquet"
    assert metadata["source_dataset_id"] == "raw-fixture"
    assert loaded_metadata["variant"] == "hourly_mean"
    pd.testing.assert_frame_equal(loaded, hourly)


def test_sentiment_ablation_keeps_market_baseline_and_adds_variants() -> None:
    config = _config()
    features = _synthetic_feature_dataset(80)
    hourly_sentiment = pd.DataFrame(
        {
            "timestamp": features["timestamp"],
            "submission_sentiment_mean": 0.1,
            "comment_sentiment_mean": 0.0,
            "combined_sentiment_mean": 0.05,
            "submission_count": 1,
            "comment_count": 1,
            "subreddit_count": 1,
            "combined_observation_count": 2,
            "sentiment_missing": 0,
            "sentiment_reliability": 2 / 7,
            "sentiment_reliability_shrunk": 0.05 * 2 / 7,
            "sentiment_ewma_6h": 0.05,
            "sentiment_ewma_24h": 0.05,
            "sentiment_ewma_fast_minus_slow": 0.0,
            "sentiment_lag_1h": 0.05,
            "sentiment_kalman": 0.05,
        }
    )

    result = run_sentiment_ablation(
        features,
        hourly_sentiment,
        config,
        variants=("hourly_mean", "counts_missingness"),
    )

    assert result.ablation_table["variant"].tolist() == [
        "market_only",
        "hourly_mean",
        "counts_missingness",
    ]
    assert result.ablation_table["feature_count"].is_monotonic_increasing
    assert "total_return_delta_vs_market" in result.ablation_table
    assert set(result.fold_metrics) == {"market_only", "hourly_mean", "counts_missingness"}


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
            volatility_multiplier=0.10,
        ),
        model=ModelConfig(probability_threshold=0.55, regularization_c=1.0),
        validation=ValidationConfig(
            minimum_train_bars=20,
            test_bars=10,
            step_bars=10,
            final_holdout_fraction=0.20,
        ),
        costs=CostsConfig(fee_per_side=0.001, slippage_per_side=0.0005),
        backtest=BacktestConfig(initial_capital=10000.0),
    )


def _synthetic_feature_dataset(row_count: int) -> pd.DataFrame:
    import numpy as np

    timestamp = pd.date_range("2026-01-01", periods=row_count, freq="h", tz="UTC")
    trend = np.arange(row_count, dtype=float)
    data = pd.DataFrame(
        {
            "timestamp": timestamp,
            "symbol": "BTCUSDT",
            "open": 100.0 + trend,
            "high": 101.0 + trend,
            "low": 99.0 + trend,
            "close": 100.5 + trend,
            "volume": 1000.0 + 10.0 * trend,
            "target": ((trend.astype(int) % 4) < 2).astype("int8"),
        }
    )
    for index, column in enumerate(MODEL_FEATURE_COLUMNS):
        if column.endswith("_missing"):
            data[column] = 0
        else:
            data[column] = np.sin(trend / (index + 2)) + trend * 0.001
    data.loc[data.index[0], MODEL_FEATURE_COLUMNS[0]] = np.nan
    return data
