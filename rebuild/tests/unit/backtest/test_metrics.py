from __future__ import annotations

import pytest
import pandas as pd

from trader.backtest.metrics import calculate_backtest_metrics, max_drawdown


def test_drawdown_matches_known_equity_sequence() -> None:
    equity = pd.Series([100.0, 120.0, 90.0, 108.0, 80.0])

    assert max_drawdown(equity) == pytest.approx(-1.0 / 3.0)


def test_zero_trade_metrics_are_valid() -> None:
    equity = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"),
            "equity": [1000.0, 1000.0, 1000.0],
            "exposure": [0.0, 0.0, 0.0],
        }
    )
    metrics = calculate_backtest_metrics(equity, pd.DataFrame())

    assert metrics["total_return"] == pytest.approx(0.0)
    assert metrics["annualized_return"] is None
    assert metrics["sharpe_ratio"] is None
    assert metrics["trade_count"] == 0
    assert metrics["win_rate"] is None
    assert metrics["average_trade_return"] is None
    assert metrics["profit_factor"] is None
    assert metrics["turnover"] == pytest.approx(0.0)
    assert metrics["total_fees"] == pytest.approx(0.0)
    assert metrics["total_slippage"] == pytest.approx(0.0)
    assert metrics["exposure_percentage"] == pytest.approx(0.0)


def test_trade_metrics_pair_round_trips() -> None:
    equity = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=30, freq="h", tz="UTC"),
            "equity": [1000.0 + value for value in range(30)],
            "exposure": [1.0] * 30,
        }
    )
    trades = pd.DataFrame(
        [
            {"side": "buy", "notional": 1000.0, "fee": 1.0, "slippage": 0.5},
            {"side": "sell", "notional": 1100.0, "fee": 1.1, "slippage": 0.6},
            {"side": "buy", "notional": 1000.0, "fee": 1.0, "slippage": 0.5},
            {"side": "sell", "notional": 900.0, "fee": 0.9, "slippage": 0.4},
        ]
    )

    metrics = calculate_backtest_metrics(equity, trades)

    expected_win = (1100.0 - 1.1) / (1000.0 + 1.0) - 1.0
    expected_loss = (900.0 - 0.9) / (1000.0 + 1.0) - 1.0
    assert metrics["win_rate"] == pytest.approx(0.5)
    assert metrics["average_trade_return"] == pytest.approx(
        (expected_win + expected_loss) / 2.0
    )
    assert metrics["profit_factor"] == pytest.approx(expected_win / abs(expected_loss))
    assert metrics["turnover"] == pytest.approx(4000.0 / 1000.0)


def test_profit_factor_avoids_infinity_when_no_losing_trades() -> None:
    equity = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=30, freq="h", tz="UTC"),
            "equity": [1000.0 + value for value in range(30)],
            "exposure": [1.0] * 30,
        }
    )
    trades = pd.DataFrame(
        [
            {"side": "buy", "notional": 1000.0, "fee": 1.0, "slippage": 0.5},
            {"side": "sell", "notional": 1100.0, "fee": 1.1, "slippage": 0.6},
        ]
    )

    metrics = calculate_backtest_metrics(equity, trades)

    assert metrics["profit_factor"] is None
