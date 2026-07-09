from __future__ import annotations

from pathlib import Path

import pandas as pd

from trader.data.schemas import normalize_ohlcv
from trader.data.storage import (
    content_hash,
    metadata_path,
    read_market_dataset,
    write_market_dataset,
)


PROJECT_ROOT = Path(__file__).parents[2]
FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "btcusdt_1h.csv"


def test_fixture_round_trips_through_parquet_with_stable_metadata(tmp_path: Path) -> None:
    fixture = normalize_ohlcv(pd.read_csv(FIXTURE), require_continuity=True)
    path = tmp_path / "btcusdt_1h.parquet"

    metadata = write_market_dataset(fixture, path, source="fixture")
    reloaded = read_market_dataset(path)

    assert reloaded.equals(fixture)
    assert metadata.row_count == len(fixture)
    assert metadata.start == "2026-01-01T00:00:00Z"
    assert metadata.end == "2026-01-01T05:00:00Z"
    assert metadata.content_hash == content_hash(fixture)
    assert metadata_path(path).read_text(encoding="utf-8").endswith("\n")
