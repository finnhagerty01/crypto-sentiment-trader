"""Data ingestion modules for the crypto sentiment trading system."""

from src.data.archive import (
    append_to_archive,
    load_archive,
    get_archive_stats,
    get_archive_path,
)

__all__ = [
    "append_to_archive",
    "load_archive",
    "get_archive_stats",
    "get_archive_path",
]
