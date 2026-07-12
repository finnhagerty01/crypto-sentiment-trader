from __future__ import annotations

import numpy as np
import pandas as pd

from trader.features.market import MODEL_FEATURE_COLUMNS


def synthetic_feature_dataset(row_count: int) -> pd.DataFrame:
    timestamp = pd.date_range("2026-01-01", periods=row_count, freq="h", tz="UTC")
    trend = np.arange(row_count, dtype=float)
    wave = np.sin(trend / 3.0)
    data = pd.DataFrame(
        {
            "timestamp": timestamp,
            "symbol": "BTCUSDT",
            "open": 100.0 + trend,
            "high": 101.0 + trend,
            "low": 99.0 + trend,
            "close": 100.5 + trend,
            "volume": 1000.0 + 10.0 * trend,
            "target": ((trend.astype(int) % 4) < 2).astype("int8"),
        }
    )
    feature_values = {
        "return_1h_clipped": wave + trend * 0.01,
        "return_6h_clipped": np.cos(trend / 5.0),
        "return_24h_clipped": trend / max(row_count, 1),
        "realized_volatility_24h_clipped": 0.01 + (trend % 7) * 0.001,
        "volume_ratio_24h_clipped": 1.0 + wave * 0.05,
        "rsi_14_clipped": 50.0 + wave * 10.0,
    }
    for column, values in feature_values.items():
        data[column] = values
    for column in MODEL_FEATURE_COLUMNS:
        if column.endswith("_missing"):
            data[column] = 0
    data.loc[data.index[0], "return_1h_clipped"] = np.nan
    return data
