"""Parquet storage and deterministic metadata for market datasets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from trader.data.schemas import CANONICAL_COLUMNS, SCHEMA_VERSION, normalize_ohlcv


class MarketStorageError(ValueError):
    """Raised when a stored market dataset is invalid or incomplete."""


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    symbol: str
    interval: str
    row_count: int
    start: str
    end: str
    schema_version: str
    source: str
    content_hash: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "row_count": self.row_count,
            "start": self.start,
            "end": self.end,
            "schema_version": self.schema_version,
            "source": self.source,
            "content_hash": self.content_hash,
        }


def write_market_dataset(
    data: pd.DataFrame,
    path: str | Path,
    *,
    source: str,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
) -> DatasetMetadata:
    """Validate and write a market dataset plus a JSON metadata sidecar."""

    normalized = normalize_ohlcv(data, symbol=symbol, interval=interval)
    if normalized.empty:
        raise MarketStorageError("cannot write an empty market dataset")

    dataset_path = Path(path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_parquet(dataset_path, index=False)

    metadata = build_metadata(
        normalized,
        source=source,
        symbol=symbol,
        interval=interval,
    )
    metadata_path(dataset_path).write_text(
        json.dumps(metadata.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def read_market_dataset(
    path: str | Path,
    *,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    require_metadata: bool = True,
) -> pd.DataFrame:
    """Load a Parquet market dataset and validate it against its metadata."""

    dataset_path = Path(path)
    data = normalize_ohlcv(
        pd.read_parquet(dataset_path),
        symbol=symbol,
        interval=interval,
    )

    sidecar = metadata_path(dataset_path)
    if require_metadata:
        if not sidecar.exists():
            raise MarketStorageError(f"missing metadata sidecar: {sidecar}")
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        actual_hash = content_hash(data)
        if metadata.get("content_hash") != actual_hash:
            raise MarketStorageError("metadata content_hash does not match dataset")
        if metadata.get("row_count") != len(data):
            raise MarketStorageError("metadata row_count does not match dataset")
    return data


def build_metadata(
    data: pd.DataFrame,
    *,
    source: str,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
) -> DatasetMetadata:
    normalized = normalize_ohlcv(data, symbol=symbol, interval=interval)
    if normalized.empty:
        raise MarketStorageError("cannot build metadata for an empty market dataset")

    return DatasetMetadata(
        symbol=symbol,
        interval=interval,
        row_count=len(normalized),
        start=_timestamp_to_string(normalized["timestamp"].iloc[0]),
        end=_timestamp_to_string(normalized["timestamp"].iloc[-1]),
        schema_version=SCHEMA_VERSION,
        source=source,
        content_hash=content_hash(normalized),
    )


def content_hash(data: pd.DataFrame) -> str:
    """Return a deterministic hash of canonical dataset content."""

    normalized = normalize_ohlcv(data)
    lines = ["timestamp,symbol,open,high,low,close,volume"]
    for row in normalized.itertuples(index=False):
        lines.append(
            ",".join(
                [
                    _timestamp_to_string(row.timestamp),
                    str(row.symbol),
                    _format_number(row.open),
                    _format_number(row.high),
                    _format_number(row.low),
                    _format_number(row.close),
                    _format_number(row.volume),
                ]
            )
        )
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def metadata_path(dataset_path: str | Path) -> Path:
    path = Path(dataset_path)
    return path.with_suffix(path.suffix + ".metadata.json")


def _timestamp_to_string(value: object) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat().replace("+00:00", "Z")


def _format_number(value: object) -> str:
    return f"{float(value):.12g}"
