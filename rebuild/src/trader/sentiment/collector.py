"""Fake-client-friendly Reddit submission and comment collection."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC
from enum import StrEnum
from typing import Any, Protocol

import pandas as pd


class CommentCollectionStatus(StrEnum):
    """Explicit comment collection outcome for a submission."""

    ZERO_COMMENTS = "zero_comments"
    COLLECTED = "collected"
    CAPPED = "capped"
    FAILED = "failed"
    NOT_REQUESTED = "not_requested"


@dataclass(frozen=True, slots=True)
class SentimentCollectionConfig:
    """Bounded Reddit collection settings for deterministic experiments."""

    subreddits: tuple[str, ...]
    max_submissions_per_subreddit: int
    collect_comments: bool = True
    max_comments_per_submission: int = 100
    max_comment_depth: int = 3
    request_timeout_seconds: float = 10.0
    max_retries: int = 2
    source: str = "reddit"

    def __post_init__(self) -> None:
        if not self.subreddits:
            raise ValueError("at least one subreddit is required")
        _require_positive_int(self.max_submissions_per_subreddit, "max_submissions_per_subreddit")
        _require_positive_int(self.max_comments_per_submission, "max_comments_per_submission")
        if self.max_comment_depth < 0:
            raise ValueError("max_comment_depth must be non-negative")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")


@dataclass(frozen=True, slots=True)
class RedditSubmissionRecord:
    submission_id: str
    subreddit: str
    created_utc: pd.Timestamp
    title: str
    selftext: str
    score: int
    num_comments: int
    author: str | None
    permalink: str | None
    collection_status: str
    collected_at: pd.Timestamp
    source: str


@dataclass(frozen=True, slots=True)
class RedditCommentRecord:
    comment_id: str
    submission_id: str
    parent_id: str
    subreddit: str
    created_utc: pd.Timestamp
    body: str
    score: int
    author: str | None
    depth: int
    collected_at: pd.Timestamp
    source: str


class RedditClient(Protocol):
    """Minimal client boundary implemented by tests or an optional PRAW adapter."""

    def iter_submissions(
        self,
        subreddit: str,
        *,
        limit: int,
    ) -> Iterable[MappingLike]:
        ...

    def iter_comments(
        self,
        submission_id: str,
        *,
        max_comments: int,
        max_depth: int,
    ) -> Iterable[MappingLike]:
        ...


MappingLike = dict[str, Any] | Any


def collect_reddit_records(
    client: RedditClient,
    config: SentimentCollectionConfig,
    *,
    collected_at: pd.Timestamp | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect submissions and bounded comments into separate raw frames.

    The client is intentionally tiny so unit tests can provide local fixtures
    and production adapters can live outside imports until sentiment is used.
    """

    collection_time = _utc_timestamp(collected_at or pd.Timestamp.now(tz=UTC))
    submissions: list[RedditSubmissionRecord] = []
    comments: list[RedditCommentRecord] = []

    for subreddit in config.subreddits:
        for raw_submission in client.iter_submissions(
            subreddit,
            limit=config.max_submissions_per_subreddit,
        ):
            submission = _submission_record(
                raw_submission,
                subreddit=subreddit,
                collected_at=collection_time,
                source=config.source,
            )
            status = CommentCollectionStatus.NOT_REQUESTED
            collected_comments: list[RedditCommentRecord] = []
            if config.collect_comments:
                try:
                    collected_comments = [
                        _comment_record(
                            raw_comment,
                            fallback_submission_id=submission.submission_id,
                            fallback_subreddit=subreddit,
                            collected_at=collection_time,
                            source=config.source,
                        )
                        for raw_comment in client.iter_comments(
                            submission.submission_id,
                            max_comments=config.max_comments_per_submission,
                            max_depth=config.max_comment_depth,
                        )
                    ]
                except Exception:
                    status = CommentCollectionStatus.FAILED
                else:
                    if not collected_comments:
                        status = CommentCollectionStatus.ZERO_COMMENTS
                    elif len(collected_comments) >= config.max_comments_per_submission:
                        status = CommentCollectionStatus.CAPPED
                    else:
                        status = CommentCollectionStatus.COLLECTED

            submissions.append(
                RedditSubmissionRecord(
                    **{
                        **asdict(submission),
                        "collection_status": str(status),
                    }
                )
            )
            comments.extend(collected_comments)

    return _submission_frame(submissions), _comment_frame(comments)


def _submission_record(
    raw: MappingLike,
    *,
    subreddit: str,
    collected_at: pd.Timestamp,
    source: str,
) -> RedditSubmissionRecord:
    return RedditSubmissionRecord(
        submission_id=str(_get(raw, "submission_id", _get(raw, "id"))),
        subreddit=str(_get(raw, "subreddit", subreddit)),
        created_utc=_utc_timestamp(_get(raw, "created_utc")),
        title=str(_get(raw, "title", "")),
        selftext=str(_get(raw, "selftext", "")),
        score=int(_get(raw, "score", 0)),
        num_comments=int(_get(raw, "num_comments", 0)),
        author=_optional_str(_get(raw, "author", None)),
        permalink=_optional_str(_get(raw, "permalink", None)),
        collection_status=str(CommentCollectionStatus.NOT_REQUESTED),
        collected_at=collected_at,
        source=source,
    )


def _comment_record(
    raw: MappingLike,
    *,
    fallback_submission_id: str,
    fallback_subreddit: str,
    collected_at: pd.Timestamp,
    source: str,
) -> RedditCommentRecord:
    return RedditCommentRecord(
        comment_id=str(_get(raw, "comment_id", _get(raw, "id"))),
        submission_id=str(_get(raw, "submission_id", fallback_submission_id)),
        parent_id=str(_get(raw, "parent_id", fallback_submission_id)),
        subreddit=str(_get(raw, "subreddit", fallback_subreddit)),
        created_utc=_utc_timestamp(_get(raw, "created_utc")),
        body=str(_get(raw, "body", "")),
        score=int(_get(raw, "score", 0)),
        author=_optional_str(_get(raw, "author", None)),
        depth=int(_get(raw, "depth", 0)),
        collected_at=collected_at,
        source=source,
    )


def _submission_frame(records: list[RedditSubmissionRecord]) -> pd.DataFrame:
    columns = [
        "submission_id",
        "subreddit",
        "created_utc",
        "title",
        "selftext",
        "score",
        "num_comments",
        "author",
        "permalink",
        "collection_status",
        "collected_at",
        "source",
    ]
    return pd.DataFrame([asdict(record) for record in records], columns=columns)


def _comment_frame(records: list[RedditCommentRecord]) -> pd.DataFrame:
    columns = [
        "comment_id",
        "submission_id",
        "parent_id",
        "subreddit",
        "created_utc",
        "body",
        "score",
        "author",
        "depth",
        "collected_at",
        "source",
    ]
    return pd.DataFrame([asdict(record) for record in records], columns=columns)


def _get(raw: MappingLike, name: str, default: Any = None) -> Any:
    if isinstance(raw, dict):
        return raw.get(name, default)
    return getattr(raw, name, default)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _utc_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, utc=True)
    if pd.isna(timestamp):
        raise ValueError("timestamp values must be valid")
    return pd.Timestamp(timestamp).tz_convert("UTC")


def _require_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
