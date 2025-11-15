"""Market data collection utilities (Binance)."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BINANCE_BASE_URL = "https://api.binance.com"


class MarketDataCollector:
    """Fetch OHLCV data for Binance symbols."""

    def __init__(self, config) -> None:
        self.config = config

    # ------------------------------------------------------------------
    def fetch_symbol(
        self,
        symbol: str,
        interval: str,
        lookback_days: int,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a single symbol/interval pair."""

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback_days)
        start_ms = int(start.timestamp() * 1000)
        result_frames = []
        url = f"{BINANCE_BASE_URL}/api/v3/klines"

        total_fetched = 0
        while True:
            remaining = self.config.market_data_limit - total_fetched
            if remaining <= 0:
                break

            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_ms,
                "limit": min(limit, remaining),
            }
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
            except requests.RequestException as exc:  # pragma: no cover - network failure path
                logger.error("Binance request failed for %s %s: %s", symbol, interval, exc)
                break

            data = response.json()
            if not data:
                break

            frame = self._klines_to_dataframe(data, symbol, interval)
            result_frames.append(frame)
            total_fetched += len(frame)

            last_open_time = data[-1][0]
            next_start = last_open_time + self._interval_to_milliseconds(interval)
            if next_start >= int(end.timestamp() * 1000):
                break
            start_ms = next_start

        if not result_frames:
            return pd.DataFrame()

        combined = pd.concat(result_frames, ignore_index=True)
        combined.drop_duplicates(subset=["timestamp"], inplace=True)
        combined.sort_values("timestamp", inplace=True)
        return combined

    # ------------------------------------------------------------------
    def fetch_multiple_symbols(self, lookback_days: int) -> Dict[Tuple[str, str], pd.DataFrame]:
        market_data: Dict[Tuple[str, str], pd.DataFrame] = {}
        for symbol in self.config.symbols:
            for interval in self.config.intervals:
                logger.info("Fetching Binance data for %s %s", symbol, interval)
                df = self.fetch_symbol(symbol, interval, lookback_days)
                if df.empty:
                    logger.warning("No market data returned for %s %s", symbol, interval)
                    continue
                market_data[(symbol, interval)] = df
        return market_data

    # ------------------------------------------------------------------
    def save_data(self, market_data: Dict[Tuple[str, str], pd.DataFrame]) -> None:
        if not market_data:
            logger.warning("No market data to save")
            return

        for (symbol, interval), df in market_data.items():
            path = self.config.raw_market_dir / f"{symbol}_{interval}.parquet"
            df.to_parquet(path, index=False)
            logger.info("Saved %s bars for %s %s to %s", len(df), symbol, interval, path)

    # ------------------------------------------------------------------
    def get_latest_prices(self) -> pd.DataFrame:
        url = f"{BINANCE_BASE_URL}/api/v3/ticker/24hr"
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network failure path
            logger.error("Failed to fetch latest prices: %s", exc)
            return pd.DataFrame()

        data = response.json()
        records = []
        interested = set(self.config.symbols)
        for item in data:
            symbol = item.get("symbol")
            if symbol not in interested:
                continue
            records.append(
                {
                    "symbol": symbol,
                    "price": float(item.get("lastPrice", 0)),
                    "change_24h": float(item.get("priceChangePercent", 0)) / 100.0,
                    "volume": float(item.get("volume", 0)),
                    "quote_volume": float(item.get("quoteVolume", 0)),
                    "timestamp": datetime.now(timezone.utc),
                }
            )

        if not records:
            return pd.DataFrame()

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    @staticmethod
    def _klines_to_dataframe(data: Iterable, symbol: str, interval: str) -> pd.DataFrame:
        columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ]
        frame = pd.DataFrame(data, columns=columns)
        frame["timestamp"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
        numeric_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base",
            "taker_buy_quote",
        ]
        for col in numeric_cols:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame["symbol"] = symbol
        frame["interval"] = interval
        return frame[[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "symbol",
            "interval",
        ]]

    # ------------------------------------------------------------------
    @staticmethod
    def _interval_to_milliseconds(interval: str) -> int:
        unit = interval[-1]
        value = int(interval[:-1])
        mapping = {"m": 60, "h": 60 * 60, "d": 60 * 60 * 24}
        return value * mapping[unit] * 1000


__all__ = ["MarketDataCollector"]
