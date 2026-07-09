"""Canonical OHLCV schema validation for market datasets."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


SCHEMA_VERSION = "ohlcv.v1"
CANONICAL_COLUMNS = ("timestamp", "symbol", "open", "high", "low", "close", "volume")
PRICE_COLUMNS = ("open", "high", "low", "close")
NUMERIC_COLUMNS = PRICE_COLUMNS + ("volume",)


class MarketDataError(ValueError):
    """Raised when market data cannot be normalized to the canonical schema."""


@dataclass(frozen=True, slots=True)
class OhlcvSchema:
    version: str = SCHEMA_VERSION
    columns: tuple[str, ...] = CANONICAL_COLUMNS


def normalize_ohlcv(
    rows: pd.DataFrame,
    *,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    require_continuity: bool = False,
) -> pd.DataFrame:
    """Validate and normalize OHLCV rows to the canonical schema.

    The function is pure: it returns a new frame and never mutates the input.
    Timestamps are converted to UTC and output rows are sorted by timestamp and
    symbol.
    """

    if not isinstance(rows, pd.DataFrame):
        raise MarketDataError("market data must be a pandas DataFrame")

    missing = [column for column in CANONICAL_COLUMNS if column not in rows.columns]
    if missing:
        raise MarketDataError(f"missing required column(s): {', '.join(missing)}")

    data = rows.loc[:, CANONICAL_COLUMNS].copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    if data["timestamp"].isna().any():
        raise MarketDataError("timestamp contains invalid or missing values")

    data["symbol"] = data["symbol"].astype("string")
    if data["symbol"].isna().any() or (data["symbol"].str.len() == 0).any():
        raise MarketDataError("symbol contains invalid or missing values")
    if (data["symbol"] != symbol).any():
        raise MarketDataError(f"all rows must use symbol {symbol}")

    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        if data[column].isna().any():
            raise MarketDataError(f"{column} contains non-numeric or missing values")

    for column in PRICE_COLUMNS:
        if (data[column] <= 0).any():
            raise MarketDataError(f"{column} must be greater than zero")
    if (data["volume"] < 0).any():
        raise MarketDataError("volume must be greater than or equal to zero")

    if data.duplicated(["timestamp", "symbol"]).any():
        raise MarketDataError("duplicate rows for timestamp and symbol")

    data = data.sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)
    if require_continuity:
        _validate_hourly_continuity(data, interval)

    return data


def _validate_hourly_continuity(data: pd.DataFrame, interval: str) -> None:
    if interval != "1h":
        raise MarketDataError("continuity validation supports only 1h interval")
    if len(data) < 2:
        return

    expected_delta = pd.Timedelta(hours=1)
    for symbol, group in data.groupby("symbol", sort=False):
        deltas = group["timestamp"].diff().dropna()
        if not (deltas == expected_delta).all():
            raise MarketDataError(f"{symbol} rows are not continuous at 1h interval")
