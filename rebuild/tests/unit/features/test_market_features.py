from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from trader.config import TraderConfig, load_config
from trader.data.schemas import normalize_ohlcv
from trader.features.market import (
    CALENDAR_FEATURE_COLUMNS,
    FEATURE_GROUP_RAW_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    RAW_FEATURE_COLUMNS,
    build_feature_dataset,
    build_market_features,
    model_feature_columns,
    model_feature_columns_for_groups,
)
from trader.modeling.baseline import BaselineLogisticModel


PROJECT_ROOT = Path(__file__).parents[3]
BASELINE_PATH = PROJECT_ROOT / "configs" / "baseline.yaml"


@pytest.fixture
def config() -> TraderConfig:
    return load_config(BASELINE_PATH)


def ohlcv(
    *,
    closes: list[float],
    volumes: list[float] | None = None,
) -> pd.DataFrame:
    if volumes is None:
        volumes = [100.0] * len(closes)
    rows = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01", periods=len(closes), freq="h", tz="UTC"
            ),
            "symbol": ["BTCUSDT"] * len(closes),
            "open": closes,
            "high": [value + 1.0 for value in closes],
            "low": [value - 1.0 for value in closes],
            "close": closes,
            "volume": volumes,
        }
    )
    return normalize_ohlcv(rows, require_continuity=True)


def test_known_returns_from_hand_calculated_sequence(config: TraderConfig) -> None:
    data = ohlcv(closes=[100.0 + i for i in range(30)])

    features = build_market_features(data, config)

    assert features.loc[1, "return_1h"] == pytest.approx(0.01)
    assert features.loc[6, "return_6h"] == pytest.approx(0.06)
    assert features.loc[24, "return_24h"] == pytest.approx(0.24)


def test_rsi_bounds_and_warmup_behavior(config: TraderConfig) -> None:
    data = ohlcv(closes=[100, 101, 100, 102, 101, 103, 102, 104, 103, 105, 104, 106, 105, 107, 106, 108])

    features = build_market_features(data, config)
    rsi = features["rsi_14"]

    assert rsi.iloc[:14].isna().all()
    assert rsi.iloc[14:].between(0.0, 100.0).all()


def test_volume_ratio_is_one_for_constant_volume(config: TraderConfig) -> None:
    data = ohlcv(closes=[100.0 + i for i in range(30)], volumes=[250.0] * 30)

    features = build_market_features(data, config)

    assert features.loc[23, "volume_ratio_24h"] == pytest.approx(1.0)
    assert features.loc[29, "volume_ratio_24h"] == pytest.approx(1.0)


def test_realized_volatility_is_hourly_standard_deviation(config: TraderConfig) -> None:
    closes = [100.0]
    returns = [0.01, -0.01] * 12
    for hourly_return in returns:
        closes.append(closes[-1] * (1.0 + hourly_return))
    data = ohlcv(closes=closes)

    features = build_market_features(data, config)

    assert features.loc[24, "realized_volatility_24h"] == pytest.approx(0.01)


def test_future_mutation_does_not_change_past_features(config: TraderConfig) -> None:
    data = ohlcv(closes=[100.0 + i for i in range(220)])
    mutated = data.copy()
    mutated.loc[180, "close"] = 10_000.0

    original = build_feature_dataset(data, config)
    changed = build_feature_dataset(mutated, config)

    pd.testing.assert_frame_equal(original.iloc[:120], changed.iloc[:120])


def test_feature_list_excludes_labels_and_raw_future_values() -> None:
    disallowed = {"target", "next_return", "noise_band", "close"}

    assert disallowed.isdisjoint(MODEL_FEATURE_COLUMNS)
    assert all(column.endswith("_clipped") or column.endswith("_missing") for column in MODEL_FEATURE_COLUMNS)
    assert set(RAW_FEATURE_COLUMNS).isdisjoint(MODEL_FEATURE_COLUMNS)


def test_feature_dataset_is_deterministic(config: TraderConfig) -> None:
    data = ohlcv(closes=[100.0 + i for i in range(220)])

    first = build_feature_dataset(data, config)
    second = build_feature_dataset(data, config)

    pd.testing.assert_frame_equal(first, second)


def test_baseline_only_config_preserves_current_model_columns(
    config: TraderConfig,
) -> None:
    assert config.features.enabled_groups == ("baseline",)
    assert model_feature_columns(config) == MODEL_FEATURE_COLUMNS
    assert model_feature_columns_for_groups(("baseline",)) == MODEL_FEATURE_COLUMNS


@pytest.mark.parametrize(
    "group",
    ["trend", "volatility", "volume", "calendar", "momentum_reversal"],
)
def test_each_group_can_be_enabled_with_baseline(
    config: TraderConfig,
    group: str,
) -> None:
    grouped_config = _with_groups(config, ("baseline", group))
    data = ohlcv(closes=[100.0 + i for i in range(240)], volumes=[100.0 + i for i in range(240)])

    features = build_feature_dataset(data, grouped_config)

    for raw_column in FEATURE_GROUP_RAW_COLUMNS[group]:
        assert raw_column in features.columns
    for feature_column in model_feature_columns(grouped_config):
        assert feature_column in features.columns


def test_future_mutation_does_not_change_past_group_features(
    config: TraderConfig,
) -> None:
    grouped_config = _with_groups(
        config,
        ("baseline", "trend", "volatility", "volume", "calendar", "momentum_reversal"),
    )
    data = ohlcv(closes=[100.0 + i for i in range(260)], volumes=[100.0 + i for i in range(260)])
    mutated = data.copy()
    mutated.loc[230, "close"] = 50_000.0
    mutated.loc[230, "volume"] = 50_000.0

    original = build_feature_dataset(data, grouped_config)
    changed = build_feature_dataset(mutated, grouped_config)

    pd.testing.assert_frame_equal(original.iloc[:180], changed.iloc[:180])


def test_warmup_missingness_flags_are_set_before_imputation(
    config: TraderConfig,
) -> None:
    grouped_config = _with_groups(config, ("baseline", "trend"))
    data = ohlcv(closes=[100.0 + i for i in range(220)])

    features = build_feature_dataset(data, grouped_config)

    assert features.loc[0, "return_24h_missing"] == 1
    assert features.loc[0, "sma_168_distance_missing"] == 1
    assert features.loc[200, "sma_168_distance_missing"] == 0


def test_calendar_encodings_are_bounded_and_deterministic(
    config: TraderConfig,
) -> None:
    grouped_config = _with_groups(config, ("baseline", "calendar"))
    data = ohlcv(closes=[100.0 + i for i in range(72)])

    first = build_feature_dataset(data, grouped_config)
    second = build_feature_dataset(data, grouped_config)

    pd.testing.assert_frame_equal(
        first.loc[:, CALENDAR_FEATURE_COLUMNS],
        second.loc[:, CALENDAR_FEATURE_COLUMNS],
    )
    assert first["hour_sin"].between(-1.0, 1.0).all()
    assert first["hour_cos"].between(-1.0, 1.0).all()
    assert first["day_of_week_sin"].between(-1.0, 1.0).all()
    assert first["day_of_week_cos"].between(-1.0, 1.0).all()
    assert set(first["weekend_flag"].unique()) <= {0, 1}
    assert not any(
        column.startswith("hour_") and column.endswith("_clipped")
        for column in first.columns
    )


def test_model_metadata_records_exact_enabled_group_feature_names(
    config: TraderConfig,
) -> None:
    grouped_config = _with_groups(config, ("baseline", "calendar"))
    data = build_feature_dataset(
        ohlcv(closes=[100.0 + i for i in range(240)], volumes=[100.0 + i for i in range(240)]),
        grouped_config,
    )

    model = BaselineLogisticModel(grouped_config).fit(data)
    metadata = model.metadata()

    assert model.feature_names == model_feature_columns(grouped_config)
    assert metadata["feature_names"] == list(model_feature_columns(grouped_config))


def _with_groups(
    config: TraderConfig,
    groups: tuple[str, ...],
) -> TraderConfig:
    return replace(
        config,
        features=replace(config.features, enabled_groups=groups),
    )
