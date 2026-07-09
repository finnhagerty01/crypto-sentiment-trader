from __future__ import annotations

import pandas as pd
import pytest

from trader.data.schemas import CANONICAL_COLUMNS, MarketDataError, normalize_ohlcv


def valid_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": ["2026-01-01T01:00:00Z", "2026-01-01T00:00:00Z"],
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "open": ["2", "1"],
            "high": ["3", "2"],
            "low": ["1.5", "0.5"],
            "close": ["2.5", "1.5"],
            "volume": ["10", "0"],
        }
    )


def test_normalizes_canonical_columns_and_sorts() -> None:
    data = normalize_ohlcv(valid_rows(), require_continuity=True)

    assert tuple(data.columns) == CANONICAL_COLUMNS
    assert data["timestamp"].dt.tz is not None
    assert data["timestamp"].tolist() == sorted(data["timestamp"].tolist())
    assert data["open"].tolist() == [1.0, 2.0]


def test_rejects_missing_required_columns() -> None:
    with pytest.raises(MarketDataError, match="missing required column"):
        normalize_ohlcv(valid_rows().drop(columns=["volume"]))


def test_rejects_duplicate_timestamp_symbol_rows() -> None:
    rows = valid_rows()
    rows.loc[1, "timestamp"] = rows.loc[0, "timestamp"]

    with pytest.raises(MarketDataError, match="duplicate"):
        normalize_ohlcv(rows)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("open", "0", "open must be greater than zero"),
        ("high", "-1", "high must be greater than zero"),
        ("low", "0", "low must be greater than zero"),
        ("close", "0", "close must be greater than zero"),
        ("volume", "-0.1", "volume must be greater than or equal to zero"),
    ],
)
def test_rejects_invalid_prices_and_volume(
    column: str, value: str, message: str
) -> None:
    rows = valid_rows()
    rows.loc[0, column] = value

    with pytest.raises(MarketDataError, match=message):
        normalize_ohlcv(rows)


def test_rejects_hourly_gaps_when_continuity_required() -> None:
    rows = valid_rows()
    rows.loc[0, "timestamp"] = "2026-01-01T03:00:00Z"

    with pytest.raises(MarketDataError, match="not continuous"):
        normalize_ohlcv(rows, require_continuity=True)
