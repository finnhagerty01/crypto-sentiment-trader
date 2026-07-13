from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

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
from trader.modeling.candle_intervals import (
    candle_interval_decision,
    interval_comparison_config,
    rank_intervals,
    resample_ohlcv,
    select_interval_threshold,
    summarize_interval_thresholds,
)


def test_resample_ohlcv_aggregates_4h_12h_and_1d() -> None:
    source = _market_rows(48)

    four_hour, four_diag = resample_ohlcv(source, "4h")
    twelve_hour, _ = resample_ohlcv(source, "12h")
    daily, _ = resample_ohlcv(source, "1d")

    assert len(four_hour) == 12
    assert len(twelve_hour) == 4
    assert len(daily) == 2
    first = four_hour.iloc[0]
    assert first["timestamp"] == pd.Timestamp("2026-01-01T00:00:00Z")
    assert first["open"] == source.iloc[0]["open"]
    assert first["high"] == source.iloc[:4]["high"].max()
    assert first["low"] == source.iloc[:4]["low"].min()
    assert first["close"] == source.iloc[3]["close"]
    assert first["volume"] == source.iloc[:4]["volume"].sum()
    assert four_diag["dropped_source_row_count"] == 0


def test_incomplete_resample_buckets_are_dropped_and_recorded() -> None:
    source = _market_rows(10, start="2026-01-01T01:00:00Z")

    resampled, diagnostics = resample_ohlcv(source, "4h")

    assert list(resampled["timestamp"]) == [pd.Timestamp("2026-01-01T04:00:00Z")]
    assert diagnostics["dropped_incomplete_bucket_count"] == 2
    assert diagnostics["dropped_source_row_count"] == 6


def test_resample_ohlcv_supports_non_day_divisor_intervals_and_alt_symbols() -> None:
    source = _market_rows(28).assign(symbol="ETHUSDT")

    five_hour, five_diag = resample_ohlcv(source, "5h")
    seven_hour, _ = resample_ohlcv(source, "7h")
    thirteen_hour, _ = resample_ohlcv(source, "13h")

    assert five_hour["symbol"].unique().tolist() == ["ETHUSDT"]
    assert len(five_hour) == 4
    assert len(seven_hour) == 4
    assert len(thirteen_hour) == 1
    assert five_diag["expected_source_rows_per_bucket"] == 5


def test_validation_windows_scale_by_interval_hours_and_target_is_one_candle() -> None:
    config = interval_comparison_config(_config(), interval="12h", interval_hours=12)

    assert config.validation.minimum_train_bars == 84
    assert config.validation.test_bars == 14
    assert config.validation.step_bars == 14
    assert config.target.horizon_bars == 1
    assert config.data.interval == "12h"


def test_threshold_selection_uses_development_exposure_and_buy_hold_gates() -> None:
    folds = pd.DataFrame(
        [
            _fold_row(0.30, 0.04, 0.0, 0.01, 0.70, 2),
            _fold_row(0.35, 0.10, 0.0, 0.02, 0.95, 2),
            _fold_row(0.40, 0.03, 0.0, 0.05, 0.40, 2),
        ]
    )

    summary = summarize_interval_thresholds(folds, max_development_exposure=0.80)

    assert select_interval_threshold(summary) == 0.30
    near_always_long = summary.loc[summary["threshold"] == 0.35].iloc[0]
    assert not bool(near_always_long["passes_exposure_filter"])
    buy_hold_loser = summary.loc[summary["threshold"] == 0.40].iloc[0]
    assert not bool(buy_hold_loser["passes_buy_hold_filter"])


def test_holdout_columns_do_not_affect_interval_ranking() -> None:
    summary = pd.DataFrame(
        [
            {
                "interval": "4h",
                "interval_order": 1,
                "rank": None,
                "selected_threshold": 0.3,
                "selected_median_total_return": 0.05,
                "selected_median_max_drawdown": -0.02,
                "selected_median_turnover": 1.0,
                "selected_median_exposure_percentage": 0.5,
                "holdout_total_return": -0.5,
            },
            {
                "interval": "12h",
                "interval_order": 2,
                "rank": None,
                "selected_threshold": 0.3,
                "selected_median_total_return": 0.02,
                "selected_median_max_drawdown": -0.01,
                "selected_median_turnover": 0.5,
                "selected_median_exposure_percentage": 0.4,
                "holdout_total_return": 0.5,
            },
        ]
    )

    ranked = rank_intervals(summary)

    assert ranked.loc[ranked["rank"] == 1].iloc[0]["interval"] == "4h"


def test_decision_writes_interval_change_does_not_help_and_phase_11_blocked(
    tmp_path: Path,
) -> None:
    summary = pd.DataFrame(
        [
            {
                "interval": "1h",
                "rank": 1,
                "selected_threshold": 0.3,
                "holdout_total_return": 0.01,
                "holdout_cash_total_return": 0.0,
                "holdout_buy_hold_total_return": 0.02,
                "selected_threshold_trade_count": 2,
            }
        ]
    )

    decision = candle_interval_decision(summary)
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(decision), encoding="utf-8")

    assert decision["decision"] == "interval_change_does_not_help"
    assert decision["interval_change_helps"] is False
    assert decision["phase_11_status"] == "blocked"
    assert json.loads(path.read_text(encoding="utf-8"))["holdout_used_for_ranking"] is False


def _fold_row(
    threshold: float,
    total_return: float,
    cash_return: float,
    buy_hold_return: float,
    exposure: float,
    trade_count: int,
) -> dict[str, object]:
    return {
        "fold": 1,
        "period": "development",
        "threshold": threshold,
        "status": "ok",
        "trade_count": trade_count,
        "total_return": total_return,
        "cash_total_return": cash_return,
        "buy_hold_total_return": buy_hold_return,
        "max_drawdown": -0.01,
        "turnover": 1.0,
        "exposure_percentage": exposure,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }


def _market_rows(rows: int, *, start: str = "2026-01-01T00:00:00Z") -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=rows, freq="h")
    index = np.arange(rows, dtype="float64")
    close = 100.0 + index
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": "BTCUSDT",
            "open": close - 0.25,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 10.0 + index,
        }
    )


def _config() -> TraderConfig:
    return TraderConfig(
        data=DataConfig(symbol="BTCUSDT", interval="1h"),
        features=FeaturesConfig(
            enabled_groups=("baseline",),
            volatility_window=4,
            volume_window=4,
            rsi_window=4,
            clipping_window=12,
            clipping_mad_multiplier=8.0,
        ),
        target=TargetConfig(
            horizon_bars=3,
            cost_buffer="round_trip",
            volatility_multiplier=0.10,
        ),
        model=ModelConfig(probability_threshold=0.5, regularization_c=1.0),
        validation=ValidationConfig(
            minimum_train_bars=80,
            test_bars=20,
            step_bars=20,
            final_holdout_fraction=0.2,
        ),
        costs=CostsConfig(fee_per_side=0.000001, slippage_per_side=0.000001),
        backtest=BacktestConfig(initial_capital=1000.0),
    )
