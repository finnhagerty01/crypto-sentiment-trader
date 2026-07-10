from __future__ import annotations

import pandas as pd
import pytest

from trader.config import CostsConfig, TargetConfig
from trader.features.target import add_target_columns


def test_target_responds_to_costs_and_volatility() -> None:
    data = pd.DataFrame(
        {
            "close": [100.0, 101.0, 101.2],
            "realized_volatility_24h": [0.01, 0.01, 0.01],
        }
    )

    result = add_target_columns(
        data,
        target_config=TargetConfig(horizon_bars=1, volatility_multiplier=0.10),
        costs_config=CostsConfig(fee_per_side=0.001, slippage_per_side=0.0005),
    )

    assert result.loc[0, "next_return"] == pytest.approx(0.01)
    assert result.loc[0, "noise_band"] == pytest.approx(0.004)
    assert result.loc[0, "target"] == 1
    assert result.loc[1, "target"] == 0
    assert pd.isna(result.loc[2, "target"])


def test_missing_volatility_keeps_target_unlabeled() -> None:
    data = pd.DataFrame(
        {
            "close": [100.0, 102.0],
            "realized_volatility_24h": [None, None],
        }
    )

    result = add_target_columns(
        data,
        target_config=TargetConfig(horizon_bars=1, volatility_multiplier=0.10),
        costs_config=CostsConfig(fee_per_side=0.001, slippage_per_side=0.0005),
    )

    assert result["target"].isna().all()
