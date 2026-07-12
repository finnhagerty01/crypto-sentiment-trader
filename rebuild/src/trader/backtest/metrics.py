"""Backtest metric calculations."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


HOURS_PER_YEAR = 365.25 * 24.0
MIN_ANNUALIZATION_HOURS = 24.0


def calculate_backtest_metrics(
    equity: pd.DataFrame,
    trades: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate strategy metrics with hourly annualization assumptions."""

    if equity.empty:
        raise ValueError("equity is empty")
    equity_values = equity["equity"].astype("float64")
    initial_equity = float(equity_values.iloc[0])
    final_equity = float(equity_values.iloc[-1])
    total_return = _safe_return(final_equity, initial_equity)
    hourly_returns = equity_values.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    return {
        "total_return": total_return,
        "annualized_return": _annualized_return(equity, total_return),
        "max_drawdown": max_drawdown(equity_values),
        "sharpe_ratio": _sharpe_ratio(hourly_returns),
        "trade_count": int(len(trades)),
        "win_rate": _win_rate(trades),
        "average_trade_return": _average_trade_return(trades),
        "profit_factor": _profit_factor(trades),
        "turnover": _turnover(trades, initial_equity),
        "total_fees": _sum_column(trades, "fee"),
        "total_slippage": _sum_column(trades, "slippage"),
        "exposure_percentage": float(equity["exposure"].mean()) if "exposure" in equity else 0.0,
    }


def max_drawdown(equity_values: pd.Series) -> float:
    """Return the most negative peak-to-trough drawdown in an equity sequence."""

    values = equity_values.astype("float64")
    running_max = values.cummax()
    drawdowns = values / running_max - 1.0
    return float(drawdowns.min())


def _annualized_return(equity: pd.DataFrame, total_return: float) -> float | None:
    timestamps = pd.to_datetime(equity["timestamp"], utc=True)
    elapsed_hours = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds() / 3600.0
    if elapsed_hours < MIN_ANNUALIZATION_HOURS or total_return <= -1.0:
        return None
    return float((1.0 + total_return) ** (HOURS_PER_YEAR / elapsed_hours) - 1.0)


def _sharpe_ratio(hourly_returns: pd.Series) -> float | None:
    if len(hourly_returns) < 2:
        return None
    std = float(hourly_returns.std(ddof=1))
    if std == 0.0 or not math.isfinite(std):
        return None
    return float(hourly_returns.mean() / std * math.sqrt(HOURS_PER_YEAR))


def _safe_return(final_value: float, initial_value: float) -> float:
    if initial_value <= 0.0:
        raise ValueError("initial equity must be positive")
    return final_value / initial_value - 1.0


def _round_trip_returns(trades: pd.DataFrame) -> list[float]:
    if trades.empty:
        return []
    returns: list[float] = []
    open_buy_cost: float | None = None
    for trade in trades.itertuples(index=False):
        if trade.side == "buy":
            open_buy_cost = float(trade.notional) + float(trade.fee)
        elif trade.side == "sell" and open_buy_cost is not None:
            sell_value = float(trade.notional) - float(trade.fee)
            returns.append(_safe_return(sell_value, open_buy_cost))
            open_buy_cost = None
    return returns


def _win_rate(trades: pd.DataFrame) -> float | None:
    returns = _round_trip_returns(trades)
    if not returns:
        return None
    return float(sum(value > 0.0 for value in returns) / len(returns))


def _average_trade_return(trades: pd.DataFrame) -> float | None:
    returns = _round_trip_returns(trades)
    if not returns:
        return None
    return float(np.mean(returns))


def _profit_factor(trades: pd.DataFrame) -> float | None:
    returns = _round_trip_returns(trades)
    if not returns:
        return None
    gross_profit = sum(value for value in returns if value > 0.0)
    gross_loss = -sum(value for value in returns if value < 0.0)
    if gross_loss == 0.0:
        return None
    return float(gross_profit / gross_loss)


def _turnover(trades: pd.DataFrame, initial_equity: float) -> float:
    if trades.empty:
        return 0.0
    return float(trades["notional"].astype("float64").sum() / initial_equity)


def _sum_column(trades: pd.DataFrame, column: str) -> float:
    if trades.empty or column not in trades:
        return 0.0
    return float(trades[column].astype("float64").sum())
