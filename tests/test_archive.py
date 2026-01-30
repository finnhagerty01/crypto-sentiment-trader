# tests/test_archive.py
"""
Unit tests for the Reddit data archive module.

Tests:
- append_to_archive: deduplication, file creation, append behavior
- load_archive: date filtering, empty archive handling
- get_archive_stats: statistics calculation
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.archive import (
    append_to_archive,
    load_archive,
    get_archive_stats,
)


@pytest.fixture
def temp_archive(tmp_path):
    """Create a temporary archive path for testing."""
    return tmp_path / "test_archive.csv"


@pytest.fixture
def sample_reddit_df():
    """Create sample Reddit data for testing."""
    return pd.DataFrame({
        "id": ["post1", "post2", "post3"],
        "created_utc": pd.to_datetime([
            "2024-01-01 10:00:00",
            "2024-01-02 11:00:00",
            "2024-01-03 12:00:00",
        ]),
        "title": ["Title 1", "Title 2", "Title 3"],
        "subreddit": ["Bitcoin", "CryptoCurrency", "Bitcoin"],
    })


class TestAppendToArchive:
    """Tests for append_to_archive function."""

    def test_creates_new_archive(self, temp_archive, sample_reddit_df):
        """Should create a new archive file if none exists."""
        assert not temp_archive.exists()

        added = append_to_archive(sample_reddit_df, archive_path=temp_archive)

        assert temp_archive.exists()
        assert added == 3

    def test_appends_new_posts(self, temp_archive, sample_reddit_df):
        """Should append new posts to existing archive."""
        append_to_archive(sample_reddit_df, archive_path=temp_archive)

        new_posts = pd.DataFrame({
            "id": ["post4", "post5"],
            "created_utc": pd.to_datetime(["2024-01-04", "2024-01-05"]),
            "title": ["Title 4", "Title 5"],
            "subreddit": ["Bitcoin", "Ethereum"],
        })

        added = append_to_archive(new_posts, archive_path=temp_archive)

        assert added == 2
        df = pd.read_csv(temp_archive)
        assert len(df) == 5

    def test_deduplicates_by_id(self, temp_archive, sample_reddit_df):
        """Should not add posts with duplicate IDs."""
        append_to_archive(sample_reddit_df, archive_path=temp_archive)

        # Try to add duplicates
        duplicate_posts = pd.DataFrame({
            "id": ["post1", "post2", "new_post"],
            "created_utc": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-06"]),
            "title": ["Dup 1", "Dup 2", "New"],
            "subreddit": ["Bitcoin", "Bitcoin", "Bitcoin"],
        })

        added = append_to_archive(duplicate_posts, archive_path=temp_archive)

        assert added == 1  # Only new_post should be added
        df = pd.read_csv(temp_archive)
        assert len(df) == 4

    def test_returns_zero_for_empty_df(self, temp_archive):
        """Should return 0 when given empty DataFrame."""
        added = append_to_archive(pd.DataFrame(), archive_path=temp_archive)
        assert added == 0

    def test_returns_zero_for_all_duplicates(self, temp_archive, sample_reddit_df):
        """Should return 0 when all posts are duplicates."""
        append_to_archive(sample_reddit_df, archive_path=temp_archive)
        added = append_to_archive(sample_reddit_df, archive_path=temp_archive)
        assert added == 0

    def test_raises_without_id_column(self, temp_archive):
        """Should raise ValueError if 'id' column is missing."""
        bad_df = pd.DataFrame({"title": ["Test"]})
        with pytest.raises(ValueError, match="must have 'id' column"):
            append_to_archive(bad_df, archive_path=temp_archive)


class TestLoadArchive:
    """Tests for load_archive function."""

    def test_loads_all_data(self, temp_archive, sample_reddit_df):
        """Should load all archived data."""
        append_to_archive(sample_reddit_df, archive_path=temp_archive)

        df = load_archive(archive_path=temp_archive)

        assert len(df) == 3
        assert "created_utc" in df.columns

    def test_filters_by_min_date(self, temp_archive, sample_reddit_df):
        """Should filter posts after min_date."""
        append_to_archive(sample_reddit_df, archive_path=temp_archive)

        df = load_archive(
            min_date=datetime(2024, 1, 2),
            archive_path=temp_archive,
        )

        assert len(df) == 2  # Only posts on 2024-01-02 and 2024-01-03

    def test_filters_by_max_date(self, temp_archive, sample_reddit_df):
        """Should filter posts before max_date."""
        append_to_archive(sample_reddit_df, archive_path=temp_archive)

        df = load_archive(
            max_date=datetime(2024, 1, 2, 23, 59, 59),  # End of day
            archive_path=temp_archive,
        )

        assert len(df) == 2  # Only posts on 2024-01-01 and 2024-01-02

    def test_filters_by_date_range(self, temp_archive, sample_reddit_df):
        """Should filter posts within date range."""
        append_to_archive(sample_reddit_df, archive_path=temp_archive)

        df = load_archive(
            min_date=datetime(2024, 1, 2),
            max_date=datetime(2024, 1, 2, 23, 59, 59),
            archive_path=temp_archive,
        )

        assert len(df) == 1

    def test_returns_empty_if_no_archive(self, temp_archive):
        """Should return empty DataFrame if archive doesn't exist."""
        df = load_archive(archive_path=temp_archive)
        assert df.empty


class TestGetArchiveStats:
    """Tests for get_archive_stats function."""

    def test_returns_exists_false_if_no_archive(self, temp_archive):
        """Should return exists=False if archive doesn't exist."""
        stats = get_archive_stats(archive_path=temp_archive)
        assert stats == {"exists": False}

    def test_returns_stats_if_archive_exists(self, temp_archive, sample_reddit_df):
        """Should return statistics for existing archive."""
        append_to_archive(sample_reddit_df, archive_path=temp_archive)

        stats = get_archive_stats(archive_path=temp_archive)

        assert stats["exists"] is True
        assert stats["total_posts"] == 3
        assert "date_range" in stats
        assert "file_size_mb" in stats
        assert stats["subreddits"] == ["Bitcoin", "CryptoCurrency"]

    def test_date_range_is_correct(self, temp_archive, sample_reddit_df):
        """Should return correct date range."""
        append_to_archive(sample_reddit_df, archive_path=temp_archive)

        stats = get_archive_stats(archive_path=temp_archive)
        min_date, max_date = stats["date_range"]

        assert min_date.date() == datetime(2024, 1, 1).date()
        assert max_date.date() == datetime(2024, 1, 3).date()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
