"""Versioned storage for raw and derived sentiment datasets."""

from __future__ import annotations

from datetime import UTC
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


SENTIMENT_DATASET_SCHEMA_VERSION = "sentiment-dataset-v1"
SUBMISSION_COLUMNS = (
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
)
COMMENT_COLUMNS = (
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
)


def write_sentiment_dataset(
    submissions: pd.DataFrame,
    comments: pd.DataFrame,
    output_dir: str | Path,
    *,
    dataset_id: str,
    source: str,
    extra_metadata: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Write submissions and comments as separate Parquet files with metadata."""

    safe_id = _safe_dataset_id(dataset_id)
    normalized_submissions = normalize_submissions(submissions)
    normalized_comments = normalize_comments(comments)
    dataset_dir = Path(output_dir) / safe_id
    if dataset_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing sentiment dataset: {dataset_dir}")
    dataset_dir.mkdir(parents=True)
    submissions_path = dataset_dir / "submissions.parquet"
    comments_path = dataset_dir / "comments.parquet"
    normalized_submissions.to_parquet(submissions_path, index=False)
    normalized_comments.to_parquet(comments_path, index=False)
    metadata = _metadata(
        normalized_submissions,
        normalized_comments,
        dataset_id=safe_id,
        source=source,
        extra_metadata=extra_metadata or {},
    )
    metadata_path(dataset_dir).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return dataset_dir, metadata


def read_sentiment_dataset(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Read a versioned sentiment dataset directory."""

    dataset_dir = Path(path)
    submissions = normalize_submissions(pd.read_parquet(dataset_dir / "submissions.parquet"))
    comments = normalize_comments(pd.read_parquet(dataset_dir / "comments.parquet"))
    metadata = json.loads(metadata_path(dataset_dir).read_text(encoding="utf-8"))
    return submissions, comments, metadata


def write_hourly_sentiment_dataset(
    hourly_sentiment: pd.DataFrame,
    output_dir: str | Path,
    *,
    dataset_id: str,
    source_dataset_id: str,
    variant: str = "all_features",
) -> tuple[Path, dict[str, Any]]:
    """Write derived hourly sentiment features as a separate dataset."""

    safe_id = _safe_dataset_id(dataset_id)
    normalized = normalize_hourly_sentiment(hourly_sentiment)
    output_path = Path(output_dir) / f"{safe_id}.parquet"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing sentiment dataset: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_parquet(output_path, index=False)
    metadata = {
        "schema_version": "hourly-sentiment-dataset-v1",
        "dataset_id": safe_id,
        "source_dataset_id": source_dataset_id,
        "variant": variant,
        "created_at": pd.Timestamp.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "row_count": int(len(normalized)),
        "start": _format_timestamp(normalized["timestamp"].min())
        if not normalized.empty
        else None,
        "end": _format_timestamp(normalized["timestamp"].max())
        if not normalized.empty
        else None,
        "content_hash": content_hash(normalized),
    }
    hourly_metadata_path(output_path).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path, metadata


def read_hourly_sentiment_dataset(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read a derived hourly sentiment dataset and its metadata."""

    dataset_path = Path(path)
    data = normalize_hourly_sentiment(pd.read_parquet(dataset_path))
    metadata = json.loads(hourly_metadata_path(dataset_path).read_text(encoding="utf-8"))
    return data, metadata


def normalize_submissions(data: pd.DataFrame) -> pd.DataFrame:
    return _normalize(data, SUBMISSION_COLUMNS, id_columns=("submission_id",))


def normalize_comments(data: pd.DataFrame) -> pd.DataFrame:
    return _normalize(data, COMMENT_COLUMNS, id_columns=("comment_id",))


def metadata_path(dataset_dir: str | Path) -> Path:
    return Path(dataset_dir) / "metadata.json"


def hourly_metadata_path(path: str | Path) -> Path:
    dataset_path = Path(path)
    return dataset_path.with_suffix(dataset_path.suffix + ".metadata.json")


def normalize_hourly_sentiment(data: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in data.columns:
        raise ValueError("hourly sentiment dataset must include timestamp")
    normalized = data.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True)
    if normalized["timestamp"].isna().any():
        raise ValueError("timestamp contains invalid values")
    if normalized["timestamp"].duplicated().any():
        raise ValueError("hourly sentiment timestamps must be unique")
    return normalized.sort_values("timestamp").reset_index(drop=True)


def content_hash(data: pd.DataFrame) -> str:
    normalized = data.copy()
    for column in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = pd.to_datetime(normalized[column], utc=True).dt.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
    payload = normalized.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize(
    data: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    id_columns: tuple[str, ...],
) -> pd.DataFrame:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError("missing sentiment column(s): " + ", ".join(missing))
    normalized = data.loc[:, columns].copy()
    for column in ("created_utc", "collected_at"):
        if column in normalized:
            normalized[column] = pd.to_datetime(normalized[column], utc=True)
            if normalized[column].isna().any():
                raise ValueError(f"{column} contains invalid timestamps")
    for column in id_columns:
        if normalized[column].astype("string").str.len().eq(0).any():
            raise ValueError(f"{column} values must be non-empty")
    if normalized.duplicated(list(id_columns)).any():
        raise ValueError(f"duplicate {', '.join(id_columns)} values")
    sort_columns = [column for column in ("created_utc", *id_columns) if column in normalized]
    return normalized.sort_values(sort_columns).reset_index(drop=True)


def _metadata(
    submissions: pd.DataFrame,
    comments: pd.DataFrame,
    *,
    dataset_id: str,
    source: str,
    extra_metadata: dict[str, Any],
) -> dict[str, Any]:
    combined_times = pd.concat(
        [submissions["created_utc"], comments["created_utc"]],
        ignore_index=True,
    )
    if combined_times.empty:
        start = end = None
    else:
        start = _format_timestamp(combined_times.min())
        end = _format_timestamp(combined_times.max())
    return {
        "schema_version": SENTIMENT_DATASET_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "source": source,
        "created_at": pd.Timestamp.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "submission_count": int(len(submissions)),
        "comment_count": int(len(comments)),
        "subreddit_count": int(
            submissions["subreddit"].nunique(dropna=True)
            if "subreddit" in submissions
            else 0
        ),
        "start": start,
        "end": end,
        "submissions_content_hash": content_hash(submissions),
        "comments_content_hash": content_hash(comments),
        "extra": dict(extra_metadata),
    }


def _format_timestamp(value: Any) -> str:
    return pd.Timestamp(value).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_dataset_id(value: str) -> str:
    if not value or any(character in value for character in "/\\"):
        raise ValueError("dataset_id must be a non-empty path segment")
    return value
