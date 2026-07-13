from __future__ import annotations

import json

import pandas as pd
import pytest

from trader.data.storage import (
    MarketStorageError,
    build_metadata,
    content_hash,
    metadata_path,
    read_market_dataset,
    write_market_dataset,
)


def rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"],
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [10.0, 11.0],
        }
    )


def test_writes_reads_and_verifies_metadata(tmp_path) -> None:
    path = tmp_path / "dataset.parquet"

    metadata = write_market_dataset(rows(), path, source="unit-test")
    loaded = read_market_dataset(path)

    assert len(loaded) == 2
    assert metadata.row_count == 2
    assert metadata.source == "unit-test"
    assert metadata.content_hash == content_hash(loaded)
    assert json.loads(metadata_path(path).read_text(encoding="utf-8")) == metadata.as_dict()


def test_metadata_hash_is_stable_for_unsorted_input() -> None:
    first = rows()
    second = rows().iloc[::-1].reset_index(drop=True)

    assert content_hash(first) == content_hash(second)
    assert build_metadata(first, source="a").content_hash == build_metadata(
        second, source="a"
    ).content_hash


def test_content_hash_includes_symbol() -> None:
    btc = rows()
    eth = rows().assign(symbol="ETHUSDT")

    assert content_hash(btc) != content_hash(eth)


def test_rejects_empty_dataset(tmp_path) -> None:
    with pytest.raises(MarketStorageError, match="empty"):
        write_market_dataset(rows().iloc[0:0], tmp_path / "empty.parquet", source="test")
