"""Volatility and cost aware target construction."""

from __future__ import annotations

import pandas as pd

from trader.config import CostsConfig, TargetConfig


NEXT_RETURN_COLUMN = "next_return"
NOISE_BAND_COLUMN = "noise_band"
TARGET_COLUMN = "target"


def add_target_columns(
    data: pd.DataFrame,
    *,
    target_config: TargetConfig,
    costs_config: CostsConfig,
    volatility_column: str = "realized_volatility_24h",
) -> pd.DataFrame:
    """Add next return, noise band, and binary target columns.

    ``realized_volatility_24h`` is an hourly return standard deviation measured
    over the trailing configured window. It is not annualized and is not scaled
    by the target horizon, so it has the same return units as ``next_return``.
    Rows without a future close or a volatility estimate keep ``target`` as
    ``NA`` and can be excluded from training while still serving inference.
    """

    horizon = target_config.horizon_bars
    if horizon <= 0:
        raise ValueError("target horizon must be greater than zero")

    result = data.copy()
    result[NEXT_RETURN_COLUMN] = result["close"].shift(-horizon) / result["close"] - 1.0

    round_trip_cost = 2.0 * (
        float(costs_config.fee_per_side) + float(costs_config.slippage_per_side)
    )
    result[NOISE_BAND_COLUMN] = (
        round_trip_cost
        + float(target_config.volatility_multiplier) * result[volatility_column]
    )

    target = pd.Series(pd.NA, index=result.index, dtype="Int8")
    valid = result[NEXT_RETURN_COLUMN].notna() & result[NOISE_BAND_COLUMN].notna()
    target.loc[valid] = (
        result.loc[valid, NEXT_RETURN_COLUMN] > result.loc[valid, NOISE_BAND_COLUMN]
    ).astype("int8")
    result[TARGET_COLUMN] = target
    return result
