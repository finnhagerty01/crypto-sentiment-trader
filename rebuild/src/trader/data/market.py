"""Explicit BTCUSDT 1h market-data collection."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from trader.data.schemas import normalize_ohlcv


BINANCE_US_KLINES_URL = "https://api.binance.us/api/v3/klines"
SOURCE_NAME = "binance-us-spot-klines"


class MarketCollectionError(RuntimeError):
    """Raised when explicit market collection fails."""


class HttpClient(Protocol):
    def get_json(self, url: str, params: dict[str, object], *, timeout: float) -> object:
        """Return decoded JSON for the requested URL and query parameters."""


@dataclass(frozen=True, slots=True)
class UrlLibHttpClient:
    """Small stdlib HTTP client used by the explicit collector command."""

    user_agent: str = "crypto-sentiment-trader-rebuild/0.1"

    def get_json(self, url: str, params: dict[str, object], *, timeout: float) -> object:
        query = urlencode(params)
        request = Request(
            f"{url}?{query}",
            headers={"User-Agent": self.user_agent},
        )
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))


@dataclass(frozen=True, slots=True)
class BinanceUsSpotKlineClient:
    """Client for the Binance US spot kline endpoint."""

    http_client: HttpClient = UrlLibHttpClient()
    base_url: str = BINANCE_US_KLINES_URL
    timeout_seconds: float = 10.0
    max_retries: int = 3

    def fetch_klines(
        self,
        *,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        limit: int = 1000,
    ) -> list[list[object]]:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": limit,
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                payload = self.http_client.get_json(
                    self.base_url, params, timeout=self.timeout_seconds
                )
                if not isinstance(payload, list):
                    raise MarketCollectionError("kline response was not a JSON list")
                return payload
            except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(0.25 * (2**attempt))
        raise MarketCollectionError(f"failed to fetch klines: {last_error}") from last_error


def collect_market_data(
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    client: BinanceUsSpotKlineClient | None = None,
    now: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Fetch closed BTCUSDT 1h candles and return canonical OHLCV rows."""

    if symbol != "BTCUSDT":
        raise MarketCollectionError("baseline collector supports only BTCUSDT")
    if interval != "1h":
        raise MarketCollectionError("baseline collector supports only 1h interval")

    start_ts = _to_utc_timestamp(start, "start")
    end_ts = _to_utc_timestamp(end, "end")
    if end_ts <= start_ts:
        raise MarketCollectionError("end must be after start")

    effective_now = (
        pd.Timestamp.now(tz="UTC") if now is None else _to_utc_timestamp(now, "now")
    )
    cutoff = min(end_ts, effective_now)
    if cutoff <= start_ts:
        raise MarketCollectionError("requested range contains no closed candles")

    active_client = client or BinanceUsSpotKlineClient()
    rows: list[dict[str, object]] = []
    cursor = start_ts
    while cursor < cutoff:
        payload = active_client.fetch_klines(
            symbol=symbol,
            interval=interval,
            start_ms=_to_milliseconds(cursor),
            end_ms=_to_milliseconds(cutoff),
        )
        if not payload:
            break

        latest_open: pd.Timestamp | None = None
        for candle in payload:
            row = _parse_kline(candle, symbol=symbol)
            if row["close_time"] <= cutoff:
                rows.append(
                    {
                        "timestamp": row["timestamp"],
                        "symbol": symbol,
                        "open": row["open"],
                        "high": row["high"],
                        "low": row["low"],
                        "close": row["close"],
                        "volume": row["volume"],
                    }
                )
                latest_open = row["timestamp"]

        if latest_open is None:
            break
        cursor = latest_open + pd.Timedelta(hours=1)
        if len(payload) < 1000:
            break

    if not rows:
        raise MarketCollectionError("collector returned no closed candles")
    return normalize_ohlcv(pd.DataFrame(rows), symbol=symbol, interval=interval)


def _parse_kline(candle: object, *, symbol: str) -> dict[str, object]:
    if not isinstance(candle, list) or len(candle) < 7:
        raise MarketCollectionError("kline row has an unexpected shape")
    open_time = pd.to_datetime(int(candle[0]), unit="ms", utc=True)
    close_time = pd.to_datetime(int(candle[6]), unit="ms", utc=True)
    return {
        "timestamp": open_time,
        "symbol": symbol,
        "open": candle[1],
        "high": candle[2],
        "low": candle[3],
        "close": candle[4],
        "volume": candle[5],
        "close_time": close_time,
    }


def _to_utc_timestamp(value: str | pd.Timestamp, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    if pd.isna(timestamp):
        raise MarketCollectionError(f"{name} is not a valid timestamp")
    return timestamp


def _to_milliseconds(timestamp: pd.Timestamp) -> int:
    return int(timestamp.timestamp() * 1000)
