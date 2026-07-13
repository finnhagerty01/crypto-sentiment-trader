from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from trader.modeling.market_feature_groups import (
    config_for_feature_groups,
    feature_group_id,
    select_feature_group,
    summarize_feature_group_result,
    target_grid_winner_config,
)

from .test_baseline import _config


def test_feature_group_id_is_stable() -> None:
    assert feature_group_id(("baseline", "trend")) == "baseline__trend"


def test_feature_group_config_changes_only_enabled_groups() -> None:
    config = target_grid_winner_config(_config())

    grouped = config_for_feature_groups(config, ("baseline", "calendar"))

    assert grouped.features.enabled_groups == ("baseline", "calendar")
    assert grouped.target.horizon_bars == 12
    assert grouped.target.cost_buffer == "none"
    assert grouped.target.volatility_multiplier == pytest.approx(0.10)


def test_feature_group_selection_ignores_holdout_metrics() -> None:
    summary = pd.DataFrame(
        [
            _group_row(
                "baseline",
                development_return=0.01,
                drawdown=-0.02,
                turnover=1.0,
                holdout_return=0.50,
            ),
            _group_row(
                "baseline__trend",
                development_return=0.03,
                drawdown=-0.04,
                turnover=2.0,
                holdout_return=-0.50,
            ),
        ]
    )

    selected = select_feature_group(summary)

    assert selected is not None
    assert selected["feature_group_id"] == "baseline__trend"


def test_summary_row_records_feature_names_and_holdout_metrics() -> None:
    result = SimpleNamespace(
        selected_threshold=0.30,
        summary=pd.DataFrame(
            [
                {
                    "threshold": 0.30,
                    "trade_count": 4,
                    "median_total_return": 0.04,
                    "median_cash_total_return": 0.01,
                    "median_max_drawdown": -0.03,
                    "median_turnover": 1.5,
                    "mean_precision": 0.6,
                    "mean_recall": 0.4,
                    "mean_f1": 0.48,
                }
            ]
        ),
        holdout_metrics={
            "status": "ok",
            "total_return": -0.02,
            "cash_total_return": 0.0,
            "max_drawdown": -0.05,
            "trade_count": 2,
            "turnover": 0.7,
            "exposure_percentage": 0.95,
        },
    )

    row = summarize_feature_group_result(
        enabled_groups=("baseline", "calendar"),
        feature_names=("return_1h_clipped", "hour_sin"),
        threshold_result=result,
    )

    assert row["feature_group_id"] == "baseline__calendar"
    assert row["feature_names"] == ["return_1h_clipped", "hour_sin"]
    assert row["selected_threshold"] == pytest.approx(0.30)
    assert row["selected_median_total_return"] == pytest.approx(0.04)
    assert row["holdout_total_return"] == pytest.approx(-0.02)
    assert row["holdout_exposure_percentage"] == pytest.approx(0.95)


def _group_row(
    feature_group_id: str,
    *,
    development_return: float,
    drawdown: float,
    turnover: float,
    holdout_return: float,
) -> dict[str, object]:
    return {
        "feature_group_id": feature_group_id,
        "selected_threshold": 0.10,
        "selected_median_total_return": development_return,
        "selected_median_max_drawdown": drawdown,
        "selected_median_turnover": turnover,
        "feature_count": 10,
        "holdout_total_return": holdout_return,
    }
