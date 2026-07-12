from __future__ import annotations

import pytest
import pandas as pd

from trader.backtest.engine import run_long_cash_backtest
from trader.config import BacktestConfig, CostsConfig


def test_signal_at_t_fills_at_next_bar_open() -> None:
    result = run_long_cash_backtest(
        _market([100.0, 110.0, 120.0]),
        _signals([1, 0, 0]),
        backtest_config=BacktestConfig(initial_capital=1000.0),
        costs_config=CostsConfig(fee_per_side=0.001, slippage_per_side=0.001),
    )

    first_trade = result.trades.iloc[0]
    assert first_trade["side"] == "buy"
    assert first_trade["signal_timestamp"] == pd.Timestamp("2026-01-01T00:00:00Z")
    assert first_trade["fill_timestamp"] == pd.Timestamp("2026-01-01T01:00:00Z")
    assert first_trade["raw_price"] == pytest.approx(110.0)


def test_buy_and_sell_costs_are_hand_verifiable() -> None:
    costs = CostsConfig(fee_per_side=0.002, slippage_per_side=0.001)
    result = run_long_cash_backtest(
        _market([100.0, 110.0, 120.0, 130.0]),
        _signals([1, 1, 0, 0]),
        backtest_config=BacktestConfig(initial_capital=10000.0),
        costs_config=costs,
    )

    buy = result.trades.iloc[0]
    buy_fill = 110.0 * 1.001
    quantity = 10000.0 / (buy_fill * 1.002)
    assert buy["fill_price"] == pytest.approx(buy_fill)
    assert buy["quantity"] == pytest.approx(quantity)
    assert buy["notional"] == pytest.approx(quantity * buy_fill)
    assert buy["fee"] == pytest.approx(quantity * buy_fill * 0.002)
    assert buy["slippage"] == pytest.approx(quantity * 110.0 * 0.001)
    assert buy["cash_after"] == pytest.approx(0.0)

    sell = result.trades.iloc[1]
    sell_fill = 130.0 * 0.999
    assert sell["fill_price"] == pytest.approx(sell_fill)
    assert sell["quantity"] == pytest.approx(quantity)
    assert sell["fee"] == pytest.approx(quantity * sell_fill * 0.002)
    assert sell["slippage"] == pytest.approx(quantity * 130.0 * 0.001)


def test_no_short_position_is_possible_and_cash_never_negative() -> None:
    result = run_long_cash_backtest(
        _market([100.0, 95.0, 90.0, 85.0]),
        _signals([0, -1, 0, 0]),
        backtest_config=BacktestConfig(initial_capital=500.0),
        costs_config=CostsConfig(fee_per_side=0.001, slippage_per_side=0.001),
    )

    assert result.trades.empty
    assert (result.equity["btc_quantity"] >= 0.0).all()
    assert (result.equity["cash"] >= 0.0).all()
    assert result.equity["equity"].tolist() == [500.0, 500.0, 500.0, 500.0]


def test_final_position_is_closed_at_final_close_after_costs() -> None:
    result = run_long_cash_backtest(
        _market([100.0, 100.0, 150.0]),
        _signals([1, 1, 1]),
        backtest_config=BacktestConfig(initial_capital=1000.0),
        costs_config=CostsConfig(fee_per_side=0.001, slippage_per_side=0.001),
    )

    assert result.trades.iloc[-1]["reason"] == "final_close"
    assert result.trades.iloc[-1]["fill_timestamp"] == pd.Timestamp("2026-01-01T02:00:00Z")
    assert result.trades.iloc[-1]["raw_price"] == pytest.approx(150.0)
    assert result.equity.iloc[-1]["btc_quantity"] == pytest.approx(0.0)
    assert result.equity.iloc[-1]["cash"] == pytest.approx(result.equity.iloc[-1]["equity"])


def test_signal_without_next_bar_is_ignored() -> None:
    result = run_long_cash_backtest(
        _market([100.0, 101.0]),
        _signals([0, 1]),
        backtest_config=BacktestConfig(initial_capital=1000.0),
        costs_config=CostsConfig(fee_per_side=0.001, slippage_per_side=0.001),
    )

    assert result.trades.empty
    assert result.equity.iloc[-1]["equity"] == pytest.approx(1000.0)


def _market(closes: list[float]) -> pd.DataFrame:
    timestamp = pd.date_range("2026-01-01", periods=len(closes), freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "symbol": "BTCUSDT",
            "open": closes,
            "high": [value + 1.0 for value in closes],
            "low": [value - 1.0 for value in closes],
            "close": closes,
            "volume": 100.0,
        }
    )


def _signals(signals: list[int]) -> pd.DataFrame:
    timestamp = pd.date_range("2026-01-01", periods=len(signals), freq="h", tz="UTC")
    return pd.DataFrame({"timestamp": timestamp, "signal": signals})
