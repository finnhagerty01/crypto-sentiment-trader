# src/backtest/__init__.py
"""
Backtesting framework for crypto trading strategies.

This module provides walk-forward backtesting with realistic execution simulation,
comprehensive performance metrics, and visualization tools.

Example usage:
    from src.backtest import BacktestEngine, BacktestConfig, BacktestReport

    config = BacktestConfig(initial_capital=10000)
    engine = BacktestEngine(config)
    results = engine.run(data, signal_generator)

    report = BacktestReport(results)
    report.print_summary()
"""

from src.backtest.engine import (
    OrderType,
    Trade,
    BacktestConfig,
    BacktestEngine,
    run_walk_forward_backtest,
)
from src.backtest.report import BacktestReport
from src.backtest.benchmark import (
    buy_and_hold_benchmark,
    equal_weight_benchmark,
    compare_strategies,
    print_comparison,
)

__all__ = [
    "OrderType",
    "Trade",
    "BacktestConfig",
    "BacktestEngine",
    "BacktestReport",
    "buy_and_hold_benchmark",
    "equal_weight_benchmark",
    "compare_strategies",
    "print_comparison",
    "run_walk_forward_backtest",
]
