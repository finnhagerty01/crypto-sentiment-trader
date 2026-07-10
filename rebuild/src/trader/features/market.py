"""Causal market feature generation for the BTCUSDT baseline."""

from __future__ import annotations

import pandas as pd

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
RAW_FEATURE_COLUMNS = (
    "return_1h",
    "return_6h",
    "return_24h",
    "realized_volatility_24h",
    "volume_ratio_24h",
    "rsi_14",
)
CLIPPED_FEATURE_COLUMNS = tuple(
    clipped_feature_name(column) for column in RAW_FEATURE_COLUMNS
)
MISSINGNESS_FLAG_COLUMNS = tuple(
    missingness_flag_name(column) for column in RAW_FEATURE_COLUMNS
)
MODEL_FEATURE_COLUMNS = CLIPPED_FEATURE_COLUMNS + MISSINGNESS_FLAG_COLUMNS


def build_market_features(data: pd.DataFrame, config: TraderConfig) -> pd.DataFrame:
    """Return OHLCV plus the six causal market features.

    The input is expected to already satisfy the canonical OHLCV schema. Feature
    values at row ``t`` use only data available after bar ``t`` has closed.
    """

    result = data.loc[:, EXECUTION_COLUMNS].copy()
    close = result["close"]
    volume = result["volume"]

    result["return_1h"] = close.pct_change(1)
    result["return_6h"] = close.pct_change(6)
    result["return_24h"] = close.pct_change(24)
    result["realized_volatility_24h"] = result["return_1h"].rolling(
        window=config.features.volatility_window,
        min_periods=config.features.volatility_window,
    ).std(ddof=0)
    average_volume = volume.rolling(
        window=config.features.volume_window,
        min_periods=config.features.volume_window,
    ).mean()
    result["volume_ratio_24h"] = volume / average_volume
    result["rsi_14"] = _rsi(close, window=config.features.rsi_window)

    return result


def build_feature_dataset(data: pd.DataFrame, config: TraderConfig) -> pd.DataFrame:
    """Build execution columns, model features, missingness flags, and target.

    This function is pure: it fetches nothing, writes nothing, and returns a new
    deterministic frame for the supplied OHLCV rows and configuration.
    """

    result = build_market_features(data, config)
    result = add_missingness_flags(result, RAW_FEATURE_COLUMNS)
    result = causal_mad_clip(
        result,
        columns=RAW_FEATURE_COLUMNS,
        window=config.features.clipping_window,
        mad_multiplier=config.features.clipping_mad_multiplier,
    )
    result = add_target_columns(
        result,
        target_config=config.target,
        costs_config=config.costs,
    )
    return result


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
