from __future__ import annotations

import pandas as pd
import pytest

from trader.sentiment.features import (
    build_hourly_sentiment_features,
    sentiment_feature_columns,
)
from trader.sentiment.scoring import LexiconSentimentScorer, score_reddit_records


def _raw_records() -> tuple[pd.DataFrame, pd.DataFrame]:
    submissions = pd.DataFrame(
        [
            {
                "submission_id": "s1",
                "subreddit": "Bitcoin",
                "created_utc": "2026-01-01T00:05:00Z",
                "title": "bullish breakout",
                "selftext": "good rally",
                "score": 42,
                "num_comments": 2,
            },
            {
                "submission_id": "s2",
                "subreddit": "CryptoCurrency",
                "created_utc": "2026-01-01T02:05:00Z",
                "title": "bearish risk",
                "selftext": "bad crash",
                "score": 7,
                "num_comments": 0,
            },
        ]
    )
    comments = pd.DataFrame(
        [
            {
                "comment_id": "c1",
                "submission_id": "s1",
                "parent_id": "s1",
                "subreddit": "Bitcoin",
                "created_utc": "2026-01-01T00:30:00Z",
                "body": "moon gains",
                "score": 5,
                "depth": 0,
            }
        ]
    )
    return submissions, comments


def test_scoring_preserves_engagement_fields_separately() -> None:
    submissions, comments = _raw_records()

    scored_submissions, scored_comments = score_reddit_records(
        submissions,
        comments,
        LexiconSentimentScorer(),
    )

    assert scored_submissions["score"].tolist() == [42, 7]
    assert scored_comments["score"].tolist() == [5]
    assert scored_submissions.loc[0, "sentiment_score"] > 0
    assert scored_submissions.loc[1, "sentiment_score"] < 0
    assert "sentiment_score" in scored_comments


def test_hourly_features_separate_submission_comment_and_missingness() -> None:
    submissions, comments = score_reddit_records(*_raw_records(), LexiconSentimentScorer())

    hourly = build_hourly_sentiment_features(
        submissions,
        comments,
        start="2026-01-01T00:00:00Z",
        end="2026-01-01T02:00:00Z",
    )

    assert hourly["timestamp"].tolist() == list(
        pd.date_range("2026-01-01T00:00:00Z", periods=3, freq="h", tz="UTC")
    )
    assert hourly.loc[0, "submission_count"] == 1
    assert hourly.loc[0, "comment_count"] == 1
    assert hourly.loc[0, "subreddit_count"] == 1
    assert hourly.loc[1, "combined_observation_count"] == 0
    assert hourly.loc[1, "sentiment_missing"] == 1
    assert pd.isna(hourly.loc[1, "combined_sentiment_mean"])
    assert hourly.loc[2, "submission_sentiment_mean"] < 0


def test_sentiment_transforms_are_causal() -> None:
    submissions, comments = score_reddit_records(*_raw_records(), LexiconSentimentScorer())
    baseline = build_hourly_sentiment_features(
        submissions,
        comments,
        start="2026-01-01T00:00:00Z",
        end="2026-01-01T02:00:00Z",
    )
    mutated = submissions.copy()
    mutated.loc[mutated["submission_id"].eq("s2"), "sentiment_score"] = 1.0

    changed = build_hourly_sentiment_features(
        mutated,
        comments,
        start="2026-01-01T00:00:00Z",
        end="2026-01-01T02:00:00Z",
    )

    causal_columns = [
        "sentiment_ewma_6h",
        "sentiment_ewma_24h",
        "sentiment_ewma_fast_minus_slow",
        "sentiment_lag_1h",
        "sentiment_kalman",
    ]
    pd.testing.assert_frame_equal(
        baseline.loc[:1, ["timestamp", *causal_columns]],
        changed.loc[:1, ["timestamp", *causal_columns]],
    )


def test_variant_columns_are_cumulative_and_gated() -> None:
    assert "sentiment_ewma_24h" not in sentiment_feature_columns("ewma_6h")
    assert "sentiment_ewma_24h" in sentiment_feature_columns("ewma_24h")
    assert "sentiment_kalman" in sentiment_feature_columns("kalman_filtered")
    with pytest.raises(ValueError, match="unknown sentiment variant"):
        sentiment_feature_columns("all_at_once")
