from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from trader.config import ValidationConfig
from trader.modeling.thresholds import train_window_folds
from trader.modeling.validation import expanding_walk_forward_folds

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "run_recency_window_diagnostic.py"
)
_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "run_recency_window_diagnostic", _SCRIPT_PATH
)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
_SCRIPT = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(_SCRIPT)
select_recency_window = _SCRIPT.select_recency_window
summarize_window_result = _SCRIPT.summarize_window_result


def test_expanding_fold_positions_match_current_implementation() -> None:
    validation = _validation_config()

    actual = train_window_folds(
        100,
        validation,
        train_window_policy="expanding",
    )
    expected = expanding_walk_forward_folds(100, validation)

    assert actual == expected


def test_rolling_windows_cap_train_start_and_exclude_validation_rows() -> None:
    folds = train_window_folds(
        100,
        _validation_config(),
        train_window_policy="rolling_1000",
    )

    assert folds[0].train_start == 0
    assert folds[0].train_end == 20
    assert folds[0].test_start == 20
    assert folds[0].test_end == 30
    assert all(fold.train_end <= fold.test_start for fold in folds)
    assert all(fold.test_end <= 100 for fold in folds)


def test_rolling_windows_cap_to_most_recent_rows() -> None:
    validation = ValidationConfig(
        minimum_train_bars=1200,
        test_bars=100,
        step_bars=300,
        final_holdout_fraction=0.20,
    )

    folds = train_window_folds(
        2500,
        validation,
        train_window_policy="rolling_1000",
    )

    assert folds[0].train_start == 200
    assert folds[0].train_end == 1200
    assert folds[1].train_start == 500
    assert folds[1].train_end == 1500


def test_rolling_windows_preserve_chronological_order() -> None:
    folds = train_window_folds(
        2500,
        ValidationConfig(
            minimum_train_bars=1200,
            test_bars=100,
            step_bars=300,
            final_holdout_fraction=0.20,
        ),
        train_window_policy="rolling_1000",
    )

    assert all(fold.train_start < fold.train_end for fold in folds)
    assert all(fold.train_end == fold.test_start for fold in folds)
    assert all(fold.test_start < fold.test_end for fold in folds)


def test_invalid_train_window_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid train-window policy"):
        train_window_folds(
            100,
            _validation_config(),
            train_window_policy="rolling_999",  # type: ignore[arg-type]
        )


def test_recency_window_selection_ignores_holdout_metrics() -> None:
    summary = pd.DataFrame(
        [
            _window_row(
                "expanding",
                development_return=0.01,
                drawdown=-0.02,
                turnover=1.0,
                holdout_return=0.50,
            ),
            _window_row(
                "rolling_1000",
                development_return=0.03,
                drawdown=-0.04,
                turnover=2.0,
                holdout_return=-0.50,
            ),
        ]
    )

    selected = select_recency_window(summary)

    assert selected is not None
    assert selected["train_window_policy"] == "rolling_1000"


def test_script_summary_rows_include_development_and_holdout_metrics() -> None:
    result = SimpleNamespace(
        selected_threshold=0.15,
        summary=pd.DataFrame(
            [
                {
                    "threshold": 0.15,
                    "trade_count": 3,
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
            "total_return": 0.02,
            "cash_total_return": 0.0,
            "max_drawdown": -0.01,
            "trade_count": 2,
            "turnover": 0.7,
            "exposure_percentage": 0.25,
            "probability_min": 0.1,
            "probability_mean": 0.2,
            "probability_max": 0.3,
            "probability_std": 0.05,
        },
    )

    row = summarize_window_result(
        train_window_policy="rolling_1500",
        threshold_result=result,
    )

    assert row["selected_threshold"] == pytest.approx(0.15)
    assert row["selected_median_total_return"] == pytest.approx(0.04)
    assert row["selected_median_max_drawdown"] == pytest.approx(-0.03)
    assert row["holdout_total_return"] == pytest.approx(0.02)
    assert row["holdout_turnover"] == pytest.approx(0.7)
    assert row["holdout_probability_mean"] == pytest.approx(0.2)


def _validation_config() -> ValidationConfig:
    return ValidationConfig(
        minimum_train_bars=20,
        test_bars=10,
        step_bars=10,
        final_holdout_fraction=0.20,
    )


def _window_row(
    train_window_policy: str,
    *,
    development_return: float,
    drawdown: float,
    turnover: float,
    holdout_return: float,
) -> dict[str, object]:
    return {
        "train_window_policy": train_window_policy,
        "selected_threshold": 0.10,
        "selected_median_total_return": development_return,
        "selected_median_max_drawdown": drawdown,
        "selected_median_turnover": turnover,
        "holdout_total_return": holdout_return,
    }
