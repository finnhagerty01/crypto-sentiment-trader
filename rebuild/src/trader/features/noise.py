"""Causal noise handling for model features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def missingness_flag_name(column: str) -> str:
    """Return the missingness flag column name for a raw feature."""

    return f"{column}_missing"


def clipped_feature_name(column: str) -> str:
    """Return the clipped model feature column name for a raw feature."""

    return f"{column}_clipped"


def add_missingness_flags(data: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Add 0/1 missingness flags based on the input columns before imputation."""

    result = data.copy()
    for column in columns:
        result[missingness_flag_name(column)] = result[column].isna().astype("int8")
    return result


def causal_mad_clip(
    data: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    window: int,
    mad_multiplier: float,
) -> pd.DataFrame:
    """Clip columns using rolling median/MAD from observations strictly before t.

    The rolling statistics are calculated on ``series.shift(1)`` so the current
    observation never contributes to its own clipping limits. Rows without a
    complete prior window, and rows where prior MAD is zero, are left unchanged.
    This fallback keeps warm-up behavior explicit and avoids infinite limits.
    """

    if window <= 0:
        raise ValueError("window must be greater than zero")
    if not np.isfinite(mad_multiplier) or mad_multiplier <= 0:
        raise ValueError("mad_multiplier must be finite and greater than zero")

    result = data.copy()
    for column in columns:
        prior = result[column].shift(1)
        median = prior.rolling(window=window, min_periods=window).median()
        mad = prior.rolling(window=window, min_periods=window).apply(
            _median_absolute_deviation,
            raw=True,
        )
        lower = median - mad_multiplier * mad
        upper = median + mad_multiplier * mad
        valid_limits = mad.notna() & (mad > 0)

        clipped = result[column].copy()
        clipped.loc[valid_limits] = clipped.loc[valid_limits].clip(
            lower=lower.loc[valid_limits],
            upper=upper.loc[valid_limits],
        )
        result[clipped_feature_name(column)] = clipped
    return result


def _median_absolute_deviation(values: np.ndarray) -> float:
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan")
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))
