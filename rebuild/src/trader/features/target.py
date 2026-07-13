"""Volatility and cost aware target construction."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trader.config import CostsConfig, TargetConfig


NEXT_RETURN_COLUMN = "next_return"
NOISE_BAND_COLUMN = "noise_band"
TARGET_COLUMN = "target"


@dataclass(frozen=True, slots=True)
class TargetDistribution:
    """Label availability and class-balance diagnostics for a target frame."""

    row_count: int
    labeled_row_count: int
    positive_count: int
    negative_count: int
    positive_rate: float | None
    unlabeled_count: int
    first_labeled_timestamp: pd.Timestamp | None
    last_labeled_timestamp: pd.Timestamp | None


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

    cost_buffer = _cost_buffer(target_config, costs_config)
    result[NOISE_BAND_COLUMN] = (
        cost_buffer
        + float(target_config.volatility_multiplier) * result[volatility_column]
    )

    target = pd.Series(pd.NA, index=result.index, dtype="Int8")
    valid = result[NEXT_RETURN_COLUMN].notna() & result[NOISE_BAND_COLUMN].notna()
    target.loc[valid] = (
        result.loc[valid, NEXT_RETURN_COLUMN] > result.loc[valid, NOISE_BAND_COLUMN]
    ).astype("int8")
    result[TARGET_COLUMN] = target
    return result


def summarize_target_distribution(
    data: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
    timestamp_column: str = "timestamp",
) -> TargetDistribution:
    """Summarize target label availability and class balance."""

    row_count = int(len(data))
    labeled = data.loc[data[target_column].notna()]
    labeled_row_count = int(len(labeled))
    positive_count = int((labeled[target_column] == 1).sum())
    negative_count = int((labeled[target_column] == 0).sum())
    positive_rate = (
        positive_count / labeled_row_count if labeled_row_count > 0 else None
    )
    first_labeled_timestamp = None
    last_labeled_timestamp = None
    if labeled_row_count > 0:
        labeled_timestamps = pd.to_datetime(labeled[timestamp_column], utc=True)
        first_labeled_timestamp = labeled_timestamps.iloc[0]
        last_labeled_timestamp = labeled_timestamps.iloc[-1]

    return TargetDistribution(
        row_count=row_count,
        labeled_row_count=labeled_row_count,
        positive_count=positive_count,
        negative_count=negative_count,
        positive_rate=positive_rate,
        unlabeled_count=row_count - labeled_row_count,
        first_labeled_timestamp=first_labeled_timestamp,
        last_labeled_timestamp=last_labeled_timestamp,
    )


def _cost_buffer(target_config: TargetConfig, costs_config: CostsConfig) -> float:
    one_way_cost = float(costs_config.fee_per_side) + float(
        costs_config.slippage_per_side
    )
    if target_config.cost_buffer == "none":
        return 0.0
    if target_config.cost_buffer == "one_way":
        return one_way_cost
    if target_config.cost_buffer == "round_trip":
        return 2.0 * one_way_cost
    raise ValueError(
        "target cost_buffer must be one of none, one_way, round_trip"
    )
