from __future__ import annotations

import pandas as pd
import pytest

from trader.sentiment.collector import (
    CommentCollectionStatus,
    SentimentCollectionConfig,
    collect_reddit_records,
)


class FakeClient:
    def __init__(self) -> None:
        self.comments_by_submission = {
            "s1": [
                {
                    "comment_id": "c1",
                    "submission_id": "s1",
                    "parent_id": "s1",
                    "subreddit": "Bitcoin",
                    "created_utc": "2026-01-01T00:30:00Z",
                    "body": "bullish comment",
                    "score": 3,
                    "author": "commenter",
                    "depth": 0,
                },
                {
                    "comment_id": "c2",
                    "submission_id": "s1",
                    "parent_id": "c1",
                    "subreddit": "Bitcoin",
                    "created_utc": "2026-01-01T00:40:00Z",
                    "body": "bearish reply",
                    "score": -1,
                    "author": None,
                    "depth": 1,
                },
            ],
            "s2": [],
        }

    def iter_submissions(self, subreddit: str, *, limit: int):
        assert subreddit == "Bitcoin"
        assert limit == 2
        return [
            {
                "submission_id": "s1",
                "subreddit": subreddit,
                "created_utc": "2026-01-01T00:05:00Z",
                "title": "BTC breakout",
                "selftext": "good setup",
                "score": 10,
                "num_comments": 2,
                "author": "poster",
                "permalink": "/r/Bitcoin/s1",
            },
            {
                "submission_id": "s2",
                "subreddit": subreddit,
                "created_utc": "2026-01-01T01:05:00Z",
                "title": "Quiet hour",
                "selftext": "",
                "score": 1,
                "num_comments": 0,
            },
        ]

    def iter_comments(self, submission_id: str, *, max_comments: int, max_depth: int):
        assert max_comments == 2
        assert max_depth == 1
        return self.comments_by_submission[submission_id][:max_comments]


def test_collects_submissions_and_comments_as_separate_raw_frames() -> None:
    config = SentimentCollectionConfig(
        subreddits=("Bitcoin",),
        max_submissions_per_subreddit=2,
        max_comments_per_submission=2,
        max_comment_depth=1,
    )

    submissions, comments = collect_reddit_records(
        FakeClient(),
        config,
        collected_at="2026-01-02T00:00:00Z",
    )

    assert submissions["submission_id"].tolist() == ["s1", "s2"]
    assert comments["comment_id"].tolist() == ["c1", "c2"]
    assert set(["submission_id", "score", "num_comments", "collection_status"]) <= set(
        submissions.columns
    )
    assert set(["comment_id", "submission_id", "parent_id", "score"]) <= set(
        comments.columns
    )
    assert submissions.loc[0, "score"] == 10
    assert submissions.loc[0, "collection_status"] == CommentCollectionStatus.CAPPED
    assert submissions.loc[1, "collection_status"] == CommentCollectionStatus.ZERO_COMMENTS
    assert str(submissions["created_utc"].dt.tz) == "UTC"


def test_records_not_requested_when_comments_disabled() -> None:
    config = SentimentCollectionConfig(
        subreddits=("Bitcoin",),
        max_submissions_per_subreddit=2,
        collect_comments=False,
    )

    submissions, comments = collect_reddit_records(FakeClient(), config)

    assert comments.empty
    assert set(submissions["collection_status"]) == {CommentCollectionStatus.NOT_REQUESTED}


def test_rejects_unbounded_collection_config() -> None:
    with pytest.raises(ValueError, match="max_comments_per_submission"):
        SentimentCollectionConfig(
            subreddits=("Bitcoin",),
            max_submissions_per_subreddit=1,
            max_comments_per_submission=0,
        )
