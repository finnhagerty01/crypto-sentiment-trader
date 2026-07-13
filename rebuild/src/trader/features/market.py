"""Causal market feature generation for the BTCUSDT baseline."""

from __future__ import annotations

import pandas as pd
import numpy as np

from trader.config import TraderConfig
from trader.data.schemas import CANONICAL_COLUMNS
from trader.features.noise import (
    add_missingness_flags,
    causal_mad_clip,
    clipped_feature_name,
    missingness_flag_name,
)
from trader.features.target import add_target_columns


EXECUTION_COLUMNS = CANONICAL_COLUMNS
BASELINE_RAW_FEATURE_COLUMNS = (
    "return_1h",
    "return_6h",
    "return_24h",
    "realized_volatility_24h",
    "volume_ratio_24h",
    "rsi_14",
)
FEATURE_GROUP_RAW_COLUMNS: dict[str, tuple[str, ...]] = {
    "baseline": BASELINE_RAW_FEATURE_COLUMNS,
    "trend": (
        "sma_24_distance",
        "sma_168_distance",
        "ema_12_26_distance",
    ),
    "volatility": (
        "high_low_range_1h",
        "realized_volatility_change_24h",
        "volatility_percentile_168h",
    ),
    "volume": (
        "volume_zscore_24h",
        "volume_zscore_168h",
        "dollar_volume_zscore_168h",
    ),
    "calendar": (
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "weekend_flag",
    ),
    "momentum_reversal": (
        "return_3h",
        "return_12h",
        "return_72h",
        "drawdown_from_24h_high",
        "distance_from_24h_low",
    ),
}
CALENDAR_FEATURE_COLUMNS = FEATURE_GROUP_RAW_COLUMNS["calendar"]
RAW_FEATURE_COLUMNS = BASELINE_RAW_FEATURE_COLUMNS
CLIPPED_FEATURE_COLUMNS = tuple(
    clipped_feature_name(column) for column in RAW_FEATURE_COLUMNS
)
MISSINGNESS_FLAG_COLUMNS = tuple(
    missingness_flag_name(column) for column in RAW_FEATURE_COLUMNS
)
MODEL_FEATURE_COLUMNS = CLIPPED_FEATURE_COLUMNS + MISSINGNESS_FLAG_COLUMNS


def raw_feature_columns_for_groups(groups: tuple[str, ...]) -> tuple[str, ...]:
    """Return raw feature columns for enabled groups in configured order."""

    return tuple(
        column
        for group in groups
        for column in FEATURE_GROUP_RAW_COLUMNS[group]
    )


def non_calendar_feature_columns_for_groups(groups: tuple[str, ...]) -> tuple[str, ...]:
    """Return enabled raw features that need clipping and missingness flags."""

    calendar = set(CALENDAR_FEATURE_COLUMNS)
    return tuple(
        column
        for column in raw_feature_columns_for_groups(groups)
        if column not in calendar
    )


def model_feature_columns_for_groups(groups: tuple[str, ...]) -> tuple[str, ...]:
    """Return model-ready columns for the supplied feature groups."""

    non_calendar = non_calendar_feature_columns_for_groups(groups)
    calendar = tuple(
        column
        for column in raw_feature_columns_for_groups(groups)
        if column in CALENDAR_FEATURE_COLUMNS
    )
    return (
        tuple(clipped_feature_name(column) for column in non_calendar)
        + tuple(missingness_flag_name(column) for column in non_calendar)
        + calendar
    )


def model_feature_columns(config: TraderConfig) -> tuple[str, ...]:
    """Return model-ready columns for the current feature config."""

    return model_feature_columns_for_groups(config.features.enabled_groups)


def build_market_features(data: pd.DataFrame, config: TraderConfig) -> pd.DataFrame:
    """Return OHLCV plus causal market features for enabled groups.

    The input is expected to already satisfy the canonical OHLCV schema. Feature
    values at row ``t`` use only data available after bar ``t`` has closed.
    """

    result = data.loc[:, EXECUTION_COLUMNS].copy()
    close = result["close"]
    high = result["high"]
    low = result["low"]
    volume = result["volume"]

    groups = set(config.features.enabled_groups)
    if "baseline" in groups or "volatility" in groups:
        result["return_1h"] = close.pct_change(1)
        result["realized_volatility_24h"] = result["return_1h"].rolling(
            window=config.features.volatility_window,
            min_periods=config.features.volatility_window,
        ).std(ddof=0)

    if "baseline" in groups:
        result["return_6h"] = close.pct_change(6)
        result["return_24h"] = close.pct_change(24)
        average_volume = volume.rolling(
            window=config.features.volume_window,
            min_periods=config.features.volume_window,
        ).mean()
        result["volume_ratio_24h"] = volume / average_volume
        result["rsi_14"] = _rsi(close, window=config.features.rsi_window)

    if "trend" in groups:
        sma_24 = close.rolling(window=24, min_periods=24).mean()
        sma_168 = close.rolling(window=168, min_periods=168).mean()
        ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
        ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
        result["sma_24_distance"] = close / sma_24 - 1.0
        result["sma_168_distance"] = close / sma_168 - 1.0
        result["ema_12_26_distance"] = ema_12 / ema_26 - 1.0

    if "volatility" in groups:
        result["high_low_range_1h"] = (high - low) / close
        result["realized_volatility_change_24h"] = (
            result["realized_volatility_24h"]
            / result["realized_volatility_24h"].shift(24)
            - 1.0
        )
        result["volatility_percentile_168h"] = result[
            "realized_volatility_24h"
        ].rolling(window=168, min_periods=168).apply(_last_value_percentile, raw=True)

    if "volume" in groups:
        result["volume_zscore_24h"] = _rolling_zscore(volume, window=24)
        result["volume_zscore_168h"] = _rolling_zscore(volume, window=168)
        dollar_volume = close * volume
        result["dollar_volume_zscore_168h"] = _rolling_zscore(
            dollar_volume,
            window=168,
        )

    if "calendar" in groups:
        timestamp = pd.to_datetime(result["timestamp"], utc=True)
        hour = timestamp.dt.hour.astype("float64")
        day_of_week = timestamp.dt.dayofweek.astype("float64")
        result["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        result["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        result["day_of_week_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0)
        result["day_of_week_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0)
        result["weekend_flag"] = (day_of_week >= 5.0).astype("int8")

    if "momentum_reversal" in groups:
        result["return_3h"] = close.pct_change(3)
        result["return_12h"] = close.pct_change(12)
        result["return_72h"] = close.pct_change(72)
        rolling_high = high.rolling(window=24, min_periods=24).max()
        rolling_low = low.rolling(window=24, min_periods=24).min()
        result["drawdown_from_24h_high"] = close / rolling_high - 1.0
        result["distance_from_24h_low"] = close / rolling_low - 1.0

    return result


def build_feature_dataset(data: pd.DataFrame, config: TraderConfig) -> pd.DataFrame:
    """Build execution columns, model features, missingness flags, and target.

    This function is pure: it fetches nothing, writes nothing, and returns a new
    deterministic frame for the supplied OHLCV rows and configuration.
    """

    result = build_market_features(data, config)
    raw_feature_columns = non_calendar_feature_columns_for_groups(
        config.features.enabled_groups
    )
    result = add_missingness_flags(result, raw_feature_columns)
    result = causal_mad_clip(
        result,
        columns=raw_feature_columns,
        window=config.features.clipping_window,
        mad_multiplier=config.features.clipping_mad_multiplier,
    )
    result = add_target_columns(
        result,
        target_config=config.target,
        costs_config=config.costs,
    )
    return result


def _rolling_zscore(series: pd.Series, *, window: int) -> pd.Series:
    mean = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std(ddof=0)
    return (series - mean) / std.replace(0.0, np.nan)


def _last_value_percentile(values: np.ndarray) -> float:
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan")
    current = values[-1]
    return float(np.mean(values <= current))


def _rsi(close: pd.Series, *, window: int) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    average_gain = gains.rolling(window=window, min_periods=window).mean()
    average_loss = losses.rolling(window=window, min_periods=window).mean()

    relative_strength = average_gain / average_loss
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))

    no_losses = average_loss == 0
    no_gains = average_gain == 0
    rsi = rsi.mask(no_losses & ~no_gains, 100.0)
    rsi = rsi.mask(no_losses & no_gains, 50.0)
    return rsi.clip(lower=0.0, upper=100.0)
