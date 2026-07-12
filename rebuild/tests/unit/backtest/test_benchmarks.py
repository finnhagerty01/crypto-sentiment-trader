from __future__ import annotations

import pandas as pd

from trader.backtest.benchmarks import run_benchmarks
from trader.config import BacktestConfig, CostsConfig


def test_benchmarks_use_identical_periods() -> None:
    market = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=30, freq="h", tz="UTC"),
            "symbol": "BTCUSDT",
            "open": [100.0 + value for value in range(30)],
            "high": [101.0 + value for value in range(30)],
            "low": [99.0 + value for value in range(30)],
            "close": [100.5 + value for value in range(30)],
            "volume": 100.0,
        }
    )

    results = run_benchmarks(
        market,
        backtest_config=BacktestConfig(initial_capital=1000.0),
        costs_config=CostsConfig(fee_per_side=0.001, slippage_per_side=0.001),
        momentum_lookback_bars=3,
    )

    starts = {result.backtest.equity["timestamp"].iloc[0] for result in results.values()}
    ends = {result.backtest.equity["timestamp"].iloc[-1] for result in results.values()}
    lengths = {len(result.backtest.equity) for result in results.values()}
    assert starts == {market["timestamp"].iloc[0]}
    assert ends == {market["timestamp"].iloc[-1]}
    assert lengths == {len(market)}
    assert set(results) == {"cash", "buy_and_hold", "momentum_3h"}
