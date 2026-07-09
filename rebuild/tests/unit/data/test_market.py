from __future__ import annotations

from dataclasses import dataclass

import pytest

from trader.data.market import (
    MarketCollectionError,
    collect_market_data,
)


@dataclass
class FakeKlineClient:
    payload: list[list[object]]
    calls: int = 0

    def fetch_klines(
        self,
        *,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        limit: int = 1000,
    ) -> list[list[object]]:
        self.calls += 1
        return self.payload


def kline(open_ms: int, close_ms: int, close: str = "101") -> list[object]:
    return [
        open_ms,
        "100",
        "102",
        "99",
        close,
        "12.5",
        close_ms,
        "0",
        0,
        "0",
        "0",
        "0",
    ]


def test_collects_closed_candles_with_fake_client() -> None:
    client = FakeKlineClient(
        [
            kline(1767225600000, 1767229199999),
            kline(1767229200000, 1767232799999, "102"),
        ]
    )

    data = collect_market_data(
        start="2026-01-01T00:00:00Z",
        end="2026-01-01T02:00:00Z",
        client=client,
        now="2026-01-01T02:00:00Z",
    )

    assert len(data) == 2
    assert data["symbol"].unique().tolist() == ["BTCUSDT"]
    assert data["close"].tolist() == [101.0, 102.0]
    assert client.calls == 1


def test_drops_incomplete_candles() -> None:
    client = FakeKlineClient(
        [
            kline(1767225600000, 1767229199999),
            kline(1767229200000, 1767232799999, "102"),
        ]
    )

    data = collect_market_data(
        start="2026-01-01T00:00:00Z",
        end="2026-01-01T02:00:00Z",
        client=client,
        now="2026-01-01T01:30:00Z",
    )

    assert len(data) == 1
    assert data["close"].tolist() == [101.0]


def test_rejects_unsupported_symbol_and_interval() -> None:
    with pytest.raises(MarketCollectionError, match="supports only BTCUSDT"):
        collect_market_data(start="2026-01-01", end="2026-01-02", symbol="ETHUSDT")

    with pytest.raises(MarketCollectionError, match="supports only 1h"):
        collect_market_data(start="2026-01-01", end="2026-01-02", interval="5m")
