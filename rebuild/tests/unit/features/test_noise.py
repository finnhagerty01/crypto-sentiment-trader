from __future__ import annotations

import pandas as pd
import pytest

from trader.features.noise import add_missingness_flags, causal_mad_clip


def test_current_extreme_observation_does_not_set_its_own_threshold() -> None:
    data = pd.DataFrame({"return_1h": [0.0, 1.0, 2.0, 100.0]})

    clipped = causal_mad_clip(
        data,
        columns=("return_1h",),
        window=3,
        mad_multiplier=1.0,
    )

    assert clipped.loc[3, "return_1h_clipped"] == pytest.approx(2.0)


def test_future_mutation_does_not_change_past_clipped_values() -> None:
    data = pd.DataFrame({"return_1h": [float(i) for i in range(20)]})
    mutated = data.copy()
    mutated.loc[19, "return_1h"] = 10_000.0

    original = causal_mad_clip(
        data,
        columns=("return_1h",),
        window=5,
        mad_multiplier=2.0,
    )
    changed = causal_mad_clip(
        mutated,
        columns=("return_1h",),
        window=5,
        mad_multiplier=2.0,
    )

    pd.testing.assert_series_equal(
        original.loc[:18, "return_1h_clipped"],
        changed.loc[:18, "return_1h_clipped"],
    )


def test_zero_mad_fallback_leaves_value_unchanged() -> None:
    data = pd.DataFrame({"return_1h": [1.0, 1.0, 1.0, 5.0]})

    clipped = causal_mad_clip(
        data,
        columns=("return_1h",),
        window=3,
        mad_multiplier=1.0,
    )

    assert clipped.loc[3, "return_1h_clipped"] == pytest.approx(5.0)


def test_missingness_flags_are_set_before_later_imputation() -> None:
    data = pd.DataFrame({"return_1h": [1.0, None, 2.0]})

    flagged = add_missingness_flags(data, ("return_1h",))

    assert flagged["return_1h_missing"].tolist() == [0, 1, 0]
