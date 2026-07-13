from __future__ import annotations

import pandas as pd
import pytest

from trader.config import CostsConfig, TargetConfig
from trader.features.target import add_target_columns, summarize_target_distribution


@pytest.mark.parametrize(
    ("cost_buffer", "expected_noise_band", "expected_targets"),
    [
        ("none", 0.001, [1, 1, 0, pd.NA]),
        ("one_way", 0.0025, [1, 0, 0, pd.NA]),
        ("round_trip", 0.004, [0, 0, 0, pd.NA]),
    ],
)
def test_target_labels_are_hand_calculated_for_each_cost_buffer(
    cost_buffer: str,
    expected_noise_band: float,
    expected_targets: list[object],
) -> None:
    data = pd.DataFrame(
        {
            "close": [100.0, 100.3, 100.55, 100.65],
            "realized_volatility_24h": [0.01, 0.01, 0.01, 0.01],
        }
    )

    result = add_target_columns(
        data,
        target_config=TargetConfig(
            horizon_bars=1,
            cost_buffer=cost_buffer,
            volatility_multiplier=0.10,
        ),
        costs_config=CostsConfig(fee_per_side=0.001, slippage_per_side=0.0005),
    )

    assert result.loc[0, "next_return"] == pytest.approx(0.003)
    assert result.loc[0, "noise_band"] == pytest.approx(expected_noise_band)
    assert result["target"].tolist() == expected_targets


def test_horizon_uses_the_configured_future_close() -> None:
    data = pd.DataFrame(
        {
            "close": [100.0, 101.0, 103.0, 104.0, 105.0],
            "realized_volatility_24h": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )

    result = add_target_columns(
        data,
        target_config=TargetConfig(
            horizon_bars=3,
            cost_buffer="none",
            volatility_multiplier=0.0,
        ),
        costs_config=CostsConfig(fee_per_side=0.001, slippage_per_side=0.0005),
    )

    assert result.loc[0, "next_return"] == pytest.approx(0.04)
    assert result.loc[1, "next_return"] == pytest.approx(105.0 / 101.0 - 1.0)
    assert result.loc[0, "target"] == 1
    assert result.loc[1, "target"] == 1
    assert result["target"].isna().tolist() == [False, False, True, True, True]


def test_missing_volatility_keeps_target_unlabeled() -> None:
    data = pd.DataFrame(
        {
            "close": [100.0, 102.0],
            "realized_volatility_24h": [None, None],
        }
    )

    result = add_target_columns(
        data,
        target_config=TargetConfig(
            horizon_bars=1,
            cost_buffer="round_trip",
            volatility_multiplier=0.10,
        ),
        costs_config=CostsConfig(fee_per_side=0.001, slippage_per_side=0.0005),
    )

    assert result["target"].isna().all()


def test_positive_rate_diagnostics_are_correct() -> None:
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC"),
            "target": pd.Series([1, 0, 1, pd.NA, 0], dtype="Int8"),
        }
    )

    diagnostics = summarize_target_distribution(data)

    assert diagnostics.row_count == 5
    assert diagnostics.labeled_row_count == 4
    assert diagnostics.positive_count == 2
    assert diagnostics.negative_count == 2
    assert diagnostics.positive_rate == pytest.approx(0.5)
    assert diagnostics.unlabeled_count == 1
    assert diagnostics.first_labeled_timestamp == pd.Timestamp(
        "2026-01-01T00:00:00Z"
    )
    assert diagnostics.last_labeled_timestamp == pd.Timestamp(
        "2026-01-01T04:00:00Z"
    )
