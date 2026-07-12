"""Comparable benchmark strategies for the Phase 07 backtest."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trader.backtest.engine import BacktestResult, run_long_cash_backtest
from trader.backtest.metrics import calculate_backtest_metrics
from trader.config import BacktestConfig, CostsConfig


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    name: str
    backtest: BacktestResult
    metrics: dict[str, object]


def run_benchmarks(
    market_data: pd.DataFrame,
    *,
    backtest_config: BacktestConfig,
    costs_config: CostsConfig,
    momentum_lookback_bars: int = 24,
) -> dict[str, BenchmarkResult]:
    """Run cash, buy-and-hold, and causal momentum over one identical period."""

    market = market_data.copy()
    market["timestamp"] = pd.to_datetime(market["timestamp"], utc=True)
    cash = _benchmark(
        "cash",
        market,
        signals=pd.Series(0, index=market.index),
        backtest_config=backtest_config,
        costs_config=costs_config,
    )
    buy_and_hold = _benchmark(
        "buy_and_hold",
        market,
        signals=pd.Series(1, index=market.index),
        backtest_config=backtest_config,
        costs_config=costs_config,
    )
    momentum_returns = market["close"].pct_change(momentum_lookback_bars)
    momentum_signals = (momentum_returns > 0.0).fillna(False).astype("int8")
    momentum = _benchmark(
        f"momentum_{momentum_lookback_bars}h",
        market,
        signals=momentum_signals,
        backtest_config=backtest_config,
        costs_config=costs_config,
    )
    return {
        cash.name: cash,
        buy_and_hold.name: buy_and_hold,
        momentum.name: momentum,
    }


def _benchmark(
    name: str,
    market: pd.DataFrame,
    *,
    signals: pd.Series,
    backtest_config: BacktestConfig,
    costs_config: CostsConfig,
) -> BenchmarkResult:
    predictions = pd.DataFrame(
        {
            "timestamp": market["timestamp"],
            "signal": signals.astype("int8").to_numpy(),
        }
    )
    result = run_long_cash_backtest(
        market,
        predictions,
        backtest_config=backtest_config,
        costs_config=costs_config,
    )
    return BenchmarkResult(
        name=name,
        backtest=result,
        metrics=calculate_backtest_metrics(result.equity, result.trades),
    )
