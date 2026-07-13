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
from trader.modeling.regime_specialists import (
    REGIME_COLUMN,
    add_regime_labels,
    evaluate_candidate_holdout,
    rank_regime_candidates,
    regime_specialist_config,
    regime_specialist_decision,
    select_regime_threshold,
    summarize_regime_thresholds,
)


def test_regime_labels_are_causal_and_drop_initial_lookback_rows() -> None:
    data = _market_like_rows(6)

    labeled = add_regime_labels(data, lookback_bars=3)

    assert labeled[REGIME_COLUMN].iloc[:3].isna().all()
    assert labeled["past_week_return"].iloc[3] == data["close"].iloc[3] / data["close"].iloc[0] - 1.0
    assert labeled[REGIME_COLUMN].iloc[3] == "bull"
    assert labeled[REGIME_COLUMN].iloc[5] == "bear"


def test_regime_specialist_config_uses_fixed_h12_market_only_controls() -> None:
    config = regime_specialist_config(_config())

    assert config.target.horizon_bars == 12
    assert config.target.cost_buffer == "none"
    assert config.target.volatility_multiplier == 0.10
    assert config.features.enabled_groups == ("baseline",)


def test_threshold_selection_uses_development_folds_and_buy_hold_gate() -> None:
    folds = pd.DataFrame(
        [
            _fold_row(0.20, 0.04, 0.0, 0.01, 0.50, 2),
            _fold_row(0.25, 0.10, 0.0, 0.12, 0.40, 2),
            _fold_row(0.30, -0.01, 0.0, -0.02, 0.20, 2),
        ]
    )

    summary = summarize_regime_thresholds(folds)

    assert select_regime_threshold(summary) == 0.20
    assert not bool(summary.loc[summary["threshold"] == 0.25].iloc[0]["passes_buy_hold_filter"])
    assert not bool(summary.loc[summary["threshold"] == 0.30].iloc[0]["passes_cash_filter"])


def test_holdout_columns_do_not_affect_regime_ranking() -> None:
    summary = pd.DataFrame(
        [
            {
                "candidate_name": "regime_specialists",
                "candidate_order": 1,
                "rank": None,
                "selected_threshold": 0.2,
                "selected_median_total_return": 0.05,
                "selected_median_max_drawdown": -0.02,
                "selected_median_turnover": 1.0,
                "selected_median_exposure_percentage": 0.5,
                "holdout_total_return": -0.5,
            },
            {
                "candidate_name": "regime_cash_filter",
                "candidate_order": 2,
                "rank": None,
                "selected_threshold": 0.2,
                "selected_median_total_return": 0.02,
                "selected_median_max_drawdown": -0.01,
                "selected_median_turnover": 0.5,
                "selected_median_exposure_percentage": 0.4,
                "holdout_total_return": 0.5,
            },
        ]
    )

    ranked = rank_regime_candidates(summary)

    assert ranked.loc[ranked["rank"] == 1].iloc[0]["candidate_name"] == "regime_specialists"


def test_regime_cash_filter_forces_bear_rows_to_cash(monkeypatch) -> None:
    train = _feature_rows(
        timestamps=pd.date_range("2026-01-01T00:00:00Z", periods=6, freq="h"),
        regimes=["bull", "bear", "bull", "bear", "bull", "bear"],
    )
    holdout = _feature_rows(
        timestamps=pd.date_range("2026-01-02T00:00:00Z", periods=4, freq="h"),
        regimes=["bull", "bear", "bull", "bear"],
    )

    class FakeModel:
        def fit(self, data: pd.DataFrame, *, target_column: str) -> "FakeModel":
            return self

        def predict_positive_proba(self, data: pd.DataFrame) -> np.ndarray:
            return np.full(len(data), 0.9)

    monkeypatch.setattr(
        "trader.modeling.regime_specialists.BaselineLogisticModel",
        lambda config, *, feature_names: FakeModel(),
    )

    metrics = evaluate_candidate_holdout(
        train,
        holdout,
        _config(),
        candidate_name="regime_cash_filter",
        feature_names=("feature",),
        threshold=0.5,
    )

    assert metrics["signal_count"] == 2
    assert metrics["bear_rows_forced_cash"] == 2


def test_specialist_models_train_only_on_their_own_regime(monkeypatch) -> None:
    train = _feature_rows(
        timestamps=pd.date_range("2026-01-01T00:00:00Z", periods=8, freq="h"),
        regimes=["bull", "bull", "bull", "bull", "bear", "bear", "bear", "bear"],
    )
    holdout = _feature_rows(
        timestamps=pd.date_range("2026-01-02T00:00:00Z", periods=4, freq="h"),
        regimes=["bull", "bear", "bull", "bear"],
    )
    seen_regimes: list[set[str]] = []

    class FakeModel:
        def fit(self, data: pd.DataFrame, *, target_column: str) -> "FakeModel":
            seen_regimes.append(set(data[REGIME_COLUMN]))
            return self

        def predict_positive_proba(self, data: pd.DataFrame) -> np.ndarray:
            return np.full(len(data), 0.9)

    monkeypatch.setattr(
        "trader.modeling.regime_specialists.BaselineLogisticModel",
        lambda config, *, feature_names: FakeModel(),
    )

    metrics = evaluate_candidate_holdout(
        train,
        holdout,
        _config(),
        candidate_name="regime_specialists",
        feature_names=("feature",),
        threshold=0.5,
    )

    assert seen_regimes == [{"bull"}, {"bear"}]
    assert metrics["bull_evaluation_row_count"] == 2
    assert metrics["bear_evaluation_row_count"] == 2


def test_decision_writes_regime_does_not_help_and_phase_11_blocked(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        [
            {
                "candidate_name": "global_baseline",
                "rank": 1,
                "selected_threshold": 0.3,
                "selected_median_total_return": 0.04,
                "holdout_total_return": -0.01,
                "holdout_cash_total_return": 0.0,
                "holdout_buy_hold_total_return": 0.02,
            },
            {
                "candidate_name": "regime_specialists",
                "rank": 2,
                "selected_threshold": 0.3,
                "selected_median_total_return": 0.02,
                "holdout_total_return": 0.10,
                "holdout_cash_total_return": 0.0,
                "holdout_buy_hold_total_return": 0.02,
            },
        ]
    )

    decision = regime_specialist_decision(summary)
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(decision), encoding="utf-8")

    assert decision["decision"] == "regime_specialization_does_not_help"
    assert decision["holdout_used_for_ranking"] is False
    assert decision["phase_11_status"] == "blocked"
    assert json.loads(path.read_text(encoding="utf-8"))["regime_specialization_helps"] is False


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


def _market_like_rows(rows: int) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=rows, freq="h")
    close = pd.Series([100.0, 101.0, 102.0, 103.0, 100.0, 99.0])
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": "BTCUSDT",
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 10.0,
        }
    )


def _feature_rows(
    *,
    timestamps: pd.DatetimeIndex,
    regimes: list[str],
) -> pd.DataFrame:
    rows = len(regimes)
    target = [0, 1] * ((rows + 1) // 2)
    close = 100.0 + np.arange(rows, dtype="float64")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": "BTCUSDT",
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 10.0,
            "feature": np.arange(rows, dtype="float64"),
            "target": target[:rows],
            REGIME_COLUMN: regimes,
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
            horizon_bars=12,
            cost_buffer="none",
            volatility_multiplier=0.10,
        ),
        model=ModelConfig(probability_threshold=0.5, regularization_c=1.0),
        validation=ValidationConfig(
            minimum_train_bars=8,
            test_bars=4,
            step_bars=4,
            final_holdout_fraction=0.2,
        ),
        costs=CostsConfig(fee_per_side=0.000001, slippage_per_side=0.000001),
        backtest=BacktestConfig(initial_capital=1000.0),
    )
