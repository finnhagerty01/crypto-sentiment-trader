# src/data/archive.py
"""
Reddit data archival utilities.

Provides append-only storage for Reddit posts to preserve historical data
while allowing live trading to use a rolling window. This ensures that
sentiment data is never lost, enabling long-term backtesting and analysis.

Usage:
    from src.data.archive import append_to_archive, load_archive, get_archive_stats

    # Archive new posts (automatically deduplicates)
    append_to_archive(reddit_df)

    # Load archived data for backtesting
    df = load_archive(min_date=datetime(2024, 1, 1))

    # Check archive status
    stats = get_archive_stats()
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Default archive location
ARCHIVE_PATH = Path("data/reddit_archive.csv")


def append_to_archive(
    new_df: pd.DataFrame,
    archive_path: Optional[Path] = None,
) -> int:
    """
    Append new Reddit posts to the archive (deduplicated by 'id').

    This function is safe to call repeatedly with overlapping data - it will
    only add posts that don't already exist in the archive.

    Args:
        new_df: DataFrame with Reddit posts. Must have 'id' column.
        archive_path: Optional custom archive path. Defaults to ARCHIVE_PATH.

    Returns:
        Number of new posts added to the archive.

    Raises:
        ValueError: If new_df is missing required 'id' column.
    """
    if new_df.empty:
        return 0

    if "id" not in new_df.columns:
        raise ValueError("DataFrame must have 'id' column for deduplication")

    path = archive_path or ARCHIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure consistent dtypes for id column
    new_df = new_df.copy()
    new_df["id"] = new_df["id"].astype(str)

    if path.exists():
        existing = pd.read_csv(path, dtype={"id": str})
        existing_ids = set(existing["id"])
        new_posts = new_df[~new_df["id"].isin(existing_ids)]

        if not new_posts.empty:
            combined = pd.concat([existing, new_posts], ignore_index=True)
            combined.to_csv(path, index=False)
            logger.info(
                f"Archived {len(new_posts)} new posts (total: {len(combined)})"
            )
            return len(new_posts)
        else:
            logger.debug("No new posts to archive (all duplicates)")
            return 0
    else:
        new_df.to_csv(path, index=False)
        logger.info(f"Created archive with {len(new_df)} posts at {path}")
        return len(new_df)


def load_archive(
    min_date: Optional[datetime] = None,
    max_date: Optional[datetime] = None,
    archive_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load archived Reddit data with optional date filtering.

    Args:
        min_date: Only load posts created after this date (inclusive).
        max_date: Only load posts created before this date (inclusive).
        archive_path: Optional custom archive path. Defaults to ARCHIVE_PATH.

    Returns:
        DataFrame of Reddit posts, or empty DataFrame if archive doesn't exist.
    """
    path = archive_path or ARCHIVE_PATH

    if not path.exists():
        logger.warning(f"No archive found at {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)

    if "created_utc" in df.columns:
        df["created_utc"] = pd.to_datetime(df["created_utc"])

        if min_date:
            min_date = pd.to_datetime(min_date)
            df = df[df["created_utc"] >= min_date]

        if max_date:
            max_date = pd.to_datetime(max_date)
            df = df[df["created_utc"] <= max_date]

    logger.info(f"Loaded {len(df)} posts from archive")
    return df


def get_archive_stats(archive_path: Optional[Path] = None) -> dict:
    """
    Return statistics about the archive.

    Args:
        archive_path: Optional custom archive path. Defaults to ARCHIVE_PATH.

    Returns:
        Dictionary with archive statistics:
        - exists: bool - whether archive file exists
        - total_posts: int - number of posts in archive
        - date_range: tuple - (min_date, max_date) of posts
        - file_size_mb: float - file size in megabytes
        - subreddits: list - unique subreddits in archive (if column exists)
    """
    path = archive_path or ARCHIVE_PATH

    if not path.exists():
        return {"exists": False}

    df = pd.read_csv(path)

    stats = {
        "exists": True,
        "total_posts": len(df),
        "file_size_mb": round(path.stat().st_size / (1024 * 1024), 2),
    }

    if "created_utc" in df.columns:
        df["created_utc"] = pd.to_datetime(df["created_utc"])
        stats["date_range"] = (
            df["created_utc"].min(),
            df["created_utc"].max(),
        )

    if "subreddit" in df.columns:
        stats["subreddits"] = df["subreddit"].unique().tolist()

    return stats


def get_archive_path() -> Path:
    """Return the default archive path."""
    return ARCHIVE_PATH
