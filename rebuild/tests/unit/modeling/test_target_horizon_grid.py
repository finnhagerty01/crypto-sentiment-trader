from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from trader.config import (
    BacktestConfig,
    CostsConfig,
    DataConfig,
    FeaturesConfig,
    ModelConfig,
    TargetConfig,
    TraderConfig,
    ValidationConfig,
)
from trader.modeling.target_horizon_grid import (
    TargetHorizonCandidate,
    config_for_candidate,
    select_grid_candidate,
    summarize_candidate_result,
    target_horizon_grid,
)


def test_grid_generation_matches_refinement_plan() -> None:
    grid = target_horizon_grid()

    assert len(grid) == 45
    assert grid[0] == TargetHorizonCandidate(
        horizon_bars=1,
        cost_buffer="none",
        volatility_multiplier=0.0,
    )
    assert grid[-1] == TargetHorizonCandidate(
        horizon_bars=24,
        cost_buffer="round_trip",
        volatility_multiplier=0.10,
    )


def test_candidate_id_is_deterministic() -> None:
    candidate = TargetHorizonCandidate(
        horizon_bars=12,
        cost_buffer="one_way",
        volatility_multiplier=0.05,
    )

    assert candidate.candidate_id == "h12_one_way_vol0p05"


def test_config_for_candidate_does_not_mutate_baseline_config() -> None:
    config = _config()
    candidate = TargetHorizonCandidate(
        horizon_bars=6,
        cost_buffer="none",
        volatility_multiplier=0.0,
    )

    candidate_config = config_for_candidate(config, candidate)

    assert config.target.cost_buffer == "round_trip"
    assert config.target.horizon_bars == 1
    assert candidate_config.target.cost_buffer == "none"
    assert candidate_config.target.horizon_bars == 6
    assert candidate_config.costs == config.costs


def test_candidate_summary_includes_target_and_selected_threshold_metrics() -> None:
    candidate = TargetHorizonCandidate(
        horizon_bars=3,
        cost_buffer="one_way",
        volatility_multiplier=0.05,
    )
    result = _threshold_result(
        selected_threshold=0.15,
        holdout_total_return=-0.03,
    )

    summary = summarize_candidate_result(
        candidate=candidate,
        target_distribution={
            "row_count": 100,
            "labeled_row_count": 90,
            "positive_count": 30,
            "negative_count": 60,
            "positive_rate": 1 / 3,
            "unlabeled_count": 10,
            "first_labeled_timestamp": "2026-01-01T00:00:00+00:00",
            "last_labeled_timestamp": "2026-01-04T17:00:00+00:00",
        },
        threshold_result=result,
    )

    assert summary["candidate_id"] == "h3_one_way_vol0p05"
    assert summary["positive_rate"] == pytest.approx(1 / 3)
    assert summary["selected_threshold"] == pytest.approx(0.15)
    assert summary["selected_median_total_return"] == pytest.approx(0.02)
    assert summary["holdout_total_return"] == pytest.approx(-0.03)


def test_grid_selection_ignores_holdout_metrics() -> None:
    summary = pd.DataFrame(
        [
            _summary_row(
                "h1_none_vol0p00",
                horizon_bars=1,
                development_return=0.01,
                drawdown=-0.02,
                turnover=1.0,
                holdout_return=0.50,
            ),
            _summary_row(
                "h6_round_trip_vol0p05",
                horizon_bars=6,
                development_return=0.03,
                drawdown=-0.04,
                turnover=2.0,
                holdout_return=-0.50,
            ),
        ]
    )

    selected = select_grid_candidate(summary)

    assert selected is not None
    assert selected["candidate_id"] == "h6_round_trip_vol0p05"


def _config() -> TraderConfig:
    return TraderConfig(
        data=DataConfig(symbol="BTCUSDT", interval="1h"),
        features=FeaturesConfig(
            volatility_window=24,
            volume_window=24,
            rsi_window=14,
            clipping_window=168,
            clipping_mad_multiplier=8.0,
        ),
        target=TargetConfig(
            horizon_bars=1,
            cost_buffer="round_trip",
            volatility_multiplier=0.10,
        ),
        model=ModelConfig(probability_threshold=0.55, regularization_c=1.0),
        validation=ValidationConfig(
            minimum_train_bars=20,
            test_bars=10,
            step_bars=10,
            final_holdout_fraction=0.20,
        ),
        costs=CostsConfig(fee_per_side=0.001, slippage_per_side=0.001),
        backtest=BacktestConfig(initial_capital=10000.0),
    )


def _threshold_result(*, selected_threshold: float, holdout_total_return: float):
    return SimpleNamespace(
        selected_threshold=selected_threshold,
        summary=pd.DataFrame(
            [
                {
                    "threshold": 0.10,
                    "median_total_return": 0.01,
                    "median_cash_total_return": 0.0,
                    "median_max_drawdown": -0.03,
                    "median_turnover": 3.0,
                    "trade_count": 3,
                    "mean_precision": 0.4,
                    "mean_recall": 0.5,
                    "mean_f1": 0.45,
                },
                {
                    "threshold": 0.15,
                    "median_total_return": 0.02,
                    "median_cash_total_return": 0.0,
                    "median_max_drawdown": -0.02,
                    "median_turnover": 2.0,
                    "trade_count": 2,
                    "mean_precision": 0.5,
                    "mean_recall": 0.6,
                    "mean_f1": 0.55,
                },
            ]
        ),
        holdout_metrics={
            "status": "ok",
            "total_return": holdout_total_return,
            "cash_total_return": 0.0,
            "max_drawdown": -0.01,
            "trade_count": 1,
            "turnover": 1.0,
            "exposure": 0.4,
        },
    )


def _summary_row(
    candidate_id: str,
    *,
    horizon_bars: int,
    development_return: float,
    drawdown: float,
    turnover: float,
    holdout_return: float,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "horizon_bars": horizon_bars,
        "cost_buffer": "none",
        "volatility_multiplier": 0.0,
        "selected_threshold": 0.10,
        "selected_median_total_return": development_return,
        "selected_median_max_drawdown": drawdown,
        "selected_median_turnover": turnover,
        "holdout_total_return": holdout_return,
    }
