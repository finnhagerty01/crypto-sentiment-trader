# src/data/archive.py
"""
Reddit data archival utilities using SQLite for atomic, reliable storage.

Provides append-only storage for Reddit posts to preserve historical data
while allowing live trading to use a rolling window. This ensures that
sentiment data is never lost, enabling long-term backtesting and analysis.

SQLite provides:
- Atomic transactions (no partial writes on crash)
- Efficient querying with indexes
- Concurrent read safety
- No corruption from interrupted writes

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
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

DB_PATH = Path("data/reddit_archive.db")
# Default archive location
ARCHIVE_DIR = Path("data")
ARCHIVE_DB_PATH = ARCHIVE_DIR / "reddit_archive.db"

# Legacy CSV path for migration
LEGACY_CSV_PATH = ARCHIVE_DIR / "reddit_archive.csv"

# SQLite schema
POSTS_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS posts (
    id TEXT PRIMARY KEY,
    created_utc TIMESTAMP NOT NULL,
    title TEXT,
    selftext TEXT,
    subreddit TEXT,
    score INTEGER DEFAULT 0,
    num_comments INTEGER DEFAULT 0,
    source TEXT,
    archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

POSTS_INDEX_SCHEMA = """
CREATE INDEX IF NOT EXISTS idx_posts_created_utc ON posts(created_utc);
CREATE INDEX IF NOT EXISTS idx_posts_subreddit ON posts(subreddit);
"""

SENTIMENT_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS sentiment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP NOT NULL,
    symbol TEXT NOT NULL,
    sentiment_mean REAL,
    post_volume INTEGER,
    comment_volume INTEGER,
    UNIQUE(timestamp, symbol)
);
"""

SENTIMENT_INDEX_SCHEMA = """
CREATE INDEX IF NOT EXISTS idx_sentiment_timestamp ON sentiment(timestamp);
CREATE INDEX IF NOT EXISTS idx_sentiment_symbol ON sentiment(symbol);
"""


@contextmanager
def get_connection(db_path: Optional[Path] = None) -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager for SQLite connections with proper cleanup.

    Args:
        db_path: Path to database file. Defaults to ARCHIVE_DB_PATH.

    Yields:
        SQLite connection with row factory set to sqlite3.Row.
    """
    path = db_path or ARCHIVE_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        yield conn
    finally:
        conn.close()


def init_database(db_path: Optional[Path] = None) -> None:
    """
    Initialize the database schema.

    Creates tables and indexes if they don't exist.

    Args:
        db_path: Path to database file. Defaults to ARCHIVE_DB_PATH.
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # Create tables
        cursor.execute(POSTS_TABLE_SCHEMA)
        cursor.executescript(POSTS_INDEX_SCHEMA)
        cursor.execute(SENTIMENT_TABLE_SCHEMA)
        cursor.executescript(SENTIMENT_INDEX_SCHEMA)

        conn.commit()
        logger.debug("Database schema initialized")


def migrate_from_csv(
    csv_path: Optional[Path] = None,
    db_path: Optional[Path] = None,
) -> int:
    """
    Migrate data from legacy CSV archive to SQLite.

    This is a one-time migration. After successful migration, the CSV
    can be archived or deleted.

    Args:
        csv_path: Path to CSV file. Defaults to LEGACY_CSV_PATH.
        db_path: Path to database file. Defaults to ARCHIVE_DB_PATH.

    Returns:
        Number of posts migrated.
    """
    csv_file = csv_path or LEGACY_CSV_PATH

    if not csv_file.exists():
        logger.debug("No legacy CSV to migrate")
        return 0

    logger.info(f"Migrating from CSV: {csv_file}")

    df = pd.read_csv(csv_file)
    if df.empty:
        return 0

    # Ensure id column is string
    if "id" in df.columns:
        df["id"] = df["id"].astype(str)

    # Convert timestamps
    if "created_utc" in df.columns:
        df["created_utc"] = pd.to_datetime(df["created_utc"])

    count = append_to_archive(df, db_path=db_path)
    logger.info(f"Migrated {count} posts from CSV to SQLite")

    return count


def append_to_archive(
    new_df: pd.DataFrame,
    db_path: Optional[Path] = None,
) -> int:
    """
    Append new Reddit posts to the archive (deduplicated by 'id').

    This function is safe to call repeatedly with overlapping data - it will
    only add posts that don't already exist in the archive. Uses SQLite
    transactions for atomic writes.

    Args:
        new_df: DataFrame with Reddit posts. Must have 'id' column.
        db_path: Optional custom database path. Defaults to ARCHIVE_DB_PATH.

    Returns:
        Number of new posts added to the archive.

    Raises:
        ValueError: If new_df is missing required 'id' column.
    """
    if new_df.empty:
        return 0

    if "id" not in new_df.columns:
        raise ValueError("DataFrame must have 'id' column for deduplication")

    # Initialize database if needed
    init_database(db_path)

    # Prepare data
    new_df = new_df.copy()
    new_df["id"] = new_df["id"].astype(str)

    # Convert timestamps to datetime
    if "created_utc" in new_df.columns:
        new_df["created_utc"] = pd.to_datetime(new_df["created_utc"])

    # Define columns to insert (match schema)
    columns = ["id", "created_utc", "title", "selftext", "subreddit", "score", "num_comments", "source"]
    available_columns = [c for c in columns if c in new_df.columns]

    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # Use INSERT OR IGNORE for deduplication (id is PRIMARY KEY)
        placeholders = ", ".join(["?" for _ in available_columns])
        column_names = ", ".join(available_columns)

        insert_sql = f"INSERT OR IGNORE INTO posts ({column_names}) VALUES ({placeholders})"

        # Prepare rows for insertion
        rows = []
        for _, row in new_df.iterrows():
            values = []
            for col in available_columns:
                val = row.get(col)
                # Convert pandas Timestamp to Python datetime for SQLite
                if isinstance(val, pd.Timestamp):
                    val = val.to_pydatetime()
                elif pd.isna(val):
                    val = None
                values.append(val)
            rows.append(tuple(values))

        # Execute in a transaction
        try:
            cursor.executemany(insert_sql, rows)
            inserted = cursor.rowcount
            conn.commit()

            if inserted > 0:
                logger.info(f"Archived {inserted} new posts")
            else:
                logger.debug("No new posts to archive (all duplicates)")

            return inserted

        except sqlite3.Error as e:
            conn.rollback()
            logger.error(f"Failed to archive posts: {e}")
            raise


def append_sentiment(
    sentiment_df: pd.DataFrame,
    db_path: Optional[Path] = None,
) -> int:
    """
    Append sentiment data to the archive.

    Uses INSERT OR REPLACE to update existing entries for the same
    timestamp/symbol combination.

    Args:
        sentiment_df: DataFrame with columns [timestamp, symbol, sentiment_mean, ...]
        db_path: Optional custom database path. Defaults to ARCHIVE_DB_PATH.

    Returns:
        Number of rows inserted/updated.
    """
    if sentiment_df.empty:
        return 0

    # Initialize database if needed
    init_database(db_path)

    # Prepare data
    df = sentiment_df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    columns = ["timestamp", "symbol", "sentiment_mean", "post_volume", "comment_volume"]
    available_columns = [c for c in columns if c in df.columns]

    if "timestamp" not in available_columns or "symbol" not in available_columns:
        logger.warning("Sentiment data missing required columns (timestamp, symbol)")
        return 0

    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        placeholders = ", ".join(["?" for _ in available_columns])
        column_names = ", ".join(available_columns)

        insert_sql = f"INSERT OR REPLACE INTO sentiment ({column_names}) VALUES ({placeholders})"

        rows = []
        for _, row in df.iterrows():
            values = []
            for col in available_columns:
                val = row.get(col)
                if isinstance(val, pd.Timestamp):
                    val = val.to_pydatetime()
                elif pd.isna(val):
                    val = None
                values.append(val)
            rows.append(tuple(values))

        try:
            cursor.executemany(insert_sql, rows)
            count = cursor.rowcount
            conn.commit()

            logger.debug(f"Archived {count} sentiment records")
            return count

        except sqlite3.Error as e:
            conn.rollback()
            logger.error(f"Failed to archive sentiment: {e}")
            raise


def load_archive(
    min_date: Optional[datetime] = None,
    max_date: Optional[datetime] = None,
    subreddits: Optional[List[str]] = None,
    limit: Optional[int] = None,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load archived Reddit data with optional filtering.

    Args:
        min_date: Only load posts created after this date (inclusive).
        max_date: Only load posts created before this date (inclusive).
        subreddits: Filter to specific subreddits.
        limit: Maximum number of posts to return.
        db_path: Optional custom database path. Defaults to ARCHIVE_DB_PATH.

    Returns:
        DataFrame of Reddit posts, or empty DataFrame if archive doesn't exist.
    """
    path = db_path or ARCHIVE_DB_PATH

    if not path.exists():
        logger.warning(f"No archive found at {path}")
        return pd.DataFrame()

    # Build query with filters
    query = "SELECT * FROM posts WHERE 1=1"
    params: List = []

    if min_date:
        query += " AND created_utc >= ?"
        params.append(min_date)

    if max_date:
        query += " AND created_utc <= ?"
        params.append(max_date)

    if subreddits:
        placeholders = ", ".join(["?" for _ in subreddits])
        query += f" AND subreddit IN ({placeholders})"
        params.extend(subreddits)

    query += " ORDER BY created_utc DESC"

    if limit:
        query += " LIMIT ?"
        params.append(limit)

    with get_connection(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params, parse_dates=["created_utc"])

    if "created_utc" in df.columns:
        df["created_utc"] = pd.to_datetime(df["created_utc"])

    logger.info(f"Loaded {len(df)} posts from archive")
    return df


def load_sentiment(
    min_date: Optional[datetime] = None,
    max_date: Optional[datetime] = None,
    symbols: Optional[List[str]] = None,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load archived sentiment data with optional filtering.

    Args:
        min_date: Only load data after this date (inclusive).
        max_date: Only load data before this date (inclusive).
        symbols: Filter to specific symbols.
        db_path: Optional custom database path. Defaults to ARCHIVE_DB_PATH.

    Returns:
        DataFrame of sentiment data, or empty DataFrame if none exists.
    """
    path = db_path or ARCHIVE_DB_PATH

    if not path.exists():
        return pd.DataFrame()

    query = "SELECT * FROM sentiment WHERE 1=1"
    params: List = []

    if min_date:
        query += " AND timestamp >= ?"
        params.append(min_date)

    if max_date:
        query += " AND timestamp <= ?"
        params.append(max_date)

    if symbols:
        placeholders = ", ".join(["?" for _ in symbols])
        query += f" AND symbol IN ({placeholders})"
        params.extend(symbols)

    query += " ORDER BY timestamp DESC"

    with get_connection(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params, parse_dates=["created_utc"])

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


def get_archive_stats(db_path: Optional[Path] = None) -> dict:
    """
    Return statistics about the archive.

    Args:
        db_path: Optional custom database path. Defaults to ARCHIVE_DB_PATH.

    Returns:
        Dictionary with archive statistics:
        - exists: bool - whether archive file exists
        - total_posts: int - number of posts in archive
        - total_sentiment: int - number of sentiment records
        - date_range: tuple - (min_date, max_date) of posts
        - file_size_mb: float - file size in megabytes
        - subreddits: list - unique subreddits in archive
    """
    path = db_path or ARCHIVE_DB_PATH

    if not path.exists():
        return {"exists": False}

    stats = {
        "exists": True,
        "file_size_mb": round(path.stat().st_size / (1024 * 1024), 2),
    }

    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # Post count
        cursor.execute("SELECT COUNT(*) FROM posts")
        stats["total_posts"] = cursor.fetchone()[0]

        # Sentiment count
        cursor.execute("SELECT COUNT(*) FROM sentiment")
        stats["total_sentiment"] = cursor.fetchone()[0]

        # Date range
        cursor.execute("SELECT MIN(created_utc), MAX(created_utc) FROM posts")
        row = cursor.fetchone()
        if row[0] and row[1]:
            stats["date_range"] = (
                pd.to_datetime(row[0]),
                pd.to_datetime(row[1]),
            )

        # Subreddits
        cursor.execute("SELECT DISTINCT subreddit FROM posts WHERE subreddit IS NOT NULL")
        stats["subreddits"] = [row[0] for row in cursor.fetchall()]

    return stats


def get_archive_path() -> Path:
    """Return the default archive database path."""
    return ARCHIVE_DB_PATH


def vacuum_database(db_path: Optional[Path] = None) -> None:
    """
    Optimize the database by reclaiming unused space.

    Run periodically to keep the database file size efficient.

    Args:
        db_path: Optional custom database path. Defaults to ARCHIVE_DB_PATH.
    """
    path = db_path or ARCHIVE_DB_PATH

    if not path.exists():
        return

    with get_connection(db_path) as conn:
        conn.execute("VACUUM")
        logger.info("Database vacuumed")
