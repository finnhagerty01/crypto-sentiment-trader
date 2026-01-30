# src/backtest/benchmark.py
"""
Benchmark strategies for backtest comparison.

Provides simple baseline strategies to compare against the trading model:
- Buy and hold: Buy once and hold throughout the period
- Equal weight: Buy and hold equal amounts of all symbols
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def buy_and_hold_benchmark(
    data: pd.DataFrame,
    symbol: str,
    initial_capital: float,
    fee_rate: float = 0.001,
) -> Dict:
    """
    Calculate buy and hold returns for a single symbol.

    Simulates buying at the first available price and selling at the last,
    with fee deduction on both sides.

    Args:
        data: DataFrame with columns [timestamp, symbol, close]
        symbol: Symbol to benchmark (e.g., 'BTCUSDT')
        initial_capital: Starting capital in USD
        fee_rate: Fee per side as decimal (default 0.1%)

    Returns:
        Dictionary with benchmark metrics:
        - strategy: Strategy name
        - total_return: Decimal return
        - total_return_pct: Percentage return
        - sharpe_ratio: Annualized Sharpe ratio
        - max_drawdown: Maximum drawdown as decimal
        - max_drawdown_pct: Maximum drawdown as percentage
        - final_value: Ending portfolio value
    """
    symbol_data = data[data["symbol"] == symbol].sort_values("timestamp")

    if symbol_data.empty:
        logger.warning(f"No data for symbol {symbol}")
        return {
            "strategy": f"Buy & Hold {symbol}",
            "total_return": 0.0,
            "total_return_pct": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "final_value": initial_capital,
        }

    # Entry and exit prices
    initial_price = symbol_data["close"].iloc[0]
    final_price = symbol_data["close"].iloc[-1]

    # Simulate buy with fees
    buy_cost = initial_capital
    buy_fee = buy_cost * fee_rate
    position_value = buy_cost - buy_fee
    shares = position_value / initial_price

    # Simulate sell with fees
    gross_proceeds = shares * final_price
    sell_fee = gross_proceeds * fee_rate
    final_value = gross_proceeds - sell_fee

    total_return = (final_value - initial_capital) / initial_capital

    # Calculate Sharpe ratio from price returns
    returns = symbol_data["close"].pct_change().dropna()
    hours_per_year = 252 * 24
    if len(returns) > 0 and returns.std() > 0:
        sharpe = np.sqrt(hours_per_year) * returns.mean() / returns.std()
    else:
        sharpe = 0.0

    # Calculate max drawdown
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdowns = (peak - cumulative) / peak
    max_drawdown = drawdowns.max() if len(drawdowns) > 0 else 0.0

    return {
        "strategy": f"Buy & Hold {symbol}",
        "total_return": total_return,
        "total_return_pct": total_return * 100,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown * 100,
        "final_value": final_value,
    }


def equal_weight_benchmark(
    data: pd.DataFrame,
    symbols: List[str],
    initial_capital: float,
    fee_rate: float = 0.001,
) -> Dict:
    """
    Calculate equal-weight portfolio returns.

    Simulates buying equal amounts of all symbols at the start and
    holding until the end.

    Args:
        data: DataFrame with columns [timestamp, symbol, close]
        symbols: List of symbols to include
        initial_capital: Starting capital in USD
        fee_rate: Fee per side as decimal (default 0.1%)

    Returns:
        Dictionary with benchmark metrics (same format as buy_and_hold_benchmark)
    """
    if not symbols:
        return {
            "strategy": "Equal Weight Portfolio",
            "total_return": 0.0,
            "total_return_pct": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "final_value": initial_capital,
        }

    capital_per_symbol = initial_capital / len(symbols)
    total_final_value = 0.0
    all_returns = []
    included_symbols = []

    for symbol in symbols:
        symbol_data = data[data["symbol"] == symbol].sort_values("timestamp")
        if symbol_data.empty:
            logger.warning(f"No data for symbol {symbol}, skipping")
            continue

        included_symbols.append(symbol)

        # Entry and exit
        initial_price = symbol_data["close"].iloc[0]
        final_price = symbol_data["close"].iloc[-1]

        # Buy with fees
        buy_cost = capital_per_symbol
        buy_fee = buy_cost * fee_rate
        position_value = buy_cost - buy_fee
        shares = position_value / initial_price

        # Sell with fees
        gross_proceeds = shares * final_price
        sell_fee = gross_proceeds * fee_rate
        final_value = gross_proceeds - sell_fee
        total_final_value += final_value

        # Collect returns for Sharpe calculation
        returns = symbol_data["close"].pct_change().dropna()
        if len(returns) > 0:
            all_returns.append(returns.reset_index(drop=True))

    if not included_symbols:
        return {
            "strategy": "Equal Weight Portfolio",
            "total_return": 0.0,
            "total_return_pct": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "final_value": initial_capital,
        }

    # Calculate portfolio returns (equal-weighted average)
    if all_returns:
        # Align returns by index and average
        combined = pd.concat(all_returns, axis=1)
        portfolio_returns = combined.mean(axis=1)

        hours_per_year = 252 * 24
        if len(portfolio_returns) > 0 and portfolio_returns.std() > 0:
            sharpe = (
                np.sqrt(hours_per_year)
                * portfolio_returns.mean()
                / portfolio_returns.std()
            )
        else:
            sharpe = 0.0

        # Portfolio cumulative and drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        peak = cumulative.cummax()
        drawdowns = (peak - cumulative) / peak
        max_drawdown = drawdowns.max() if len(drawdowns) > 0 else 0.0
    else:
        sharpe = 0.0
        max_drawdown = 0.0

    total_return = (total_final_value - initial_capital) / initial_capital

    return {
        "strategy": "Equal Weight Portfolio",
        "total_return": total_return,
        "total_return_pct": total_return * 100,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown * 100,
        "final_value": total_final_value,
        "symbols_included": included_symbols,
    }


def compare_strategies(
    strategy_results: Dict,
    benchmarks: List[Dict],
) -> pd.DataFrame:
    """
    Create comparison table of strategy vs benchmarks.

    Args:
        strategy_results: Results from BacktestEngine.run()
        benchmarks: List of benchmark results from buy_and_hold or equal_weight

    Returns:
        DataFrame comparing all strategies
    """
    rows = []

    # Strategy results
    if strategy_results.get("status") == "success":
        metrics = strategy_results.get("metrics", {})
        rows.append(
            {
                "Strategy": "Trading Model",
                "Total Return (%)": metrics.get("total_return_pct", 0),
                "Sharpe Ratio": metrics.get("sharpe_ratio", 0),
                "Max Drawdown (%)": metrics.get("max_drawdown_pct", 0),
                "Win Rate (%)": metrics.get("win_rate", 0) * 100,
                "Profit Factor": metrics.get("profit_factor", 0),
                "Final Value ($)": metrics.get("final_value", 0),
            }
        )

    # Benchmark results
    for bm in benchmarks:
        rows.append(
            {
                "Strategy": bm.get("strategy", "Unknown"),
                "Total Return (%)": bm.get("total_return_pct", 0),
                "Sharpe Ratio": bm.get("sharpe_ratio", 0),
                "Max Drawdown (%)": bm.get("max_drawdown_pct", 0),
                "Win Rate (%)": "-",
                "Profit Factor": "-",
                "Final Value ($)": bm.get("final_value", 0),
            }
        )

    df = pd.DataFrame(rows)
    return df


def print_comparison(comparison_df: pd.DataFrame) -> None:
    """
    Print formatted comparison table to console.

    Args:
        comparison_df: DataFrame from compare_strategies()
    """
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON")
    print("=" * 80)

    # Format numeric columns
    formatted = comparison_df.copy()
    for col in formatted.columns:
        if col in ["Total Return (%)", "Max Drawdown (%)", "Win Rate (%)"]:
            formatted[col] = formatted[col].apply(
                lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x
            )
        elif col == "Final Value ($)":
            formatted[col] = formatted[col].apply(
                lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else x
            )
        elif col in ["Sharpe Ratio", "Profit Factor"]:
            formatted[col] = formatted[col].apply(
                lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x
            )

    print(formatted.to_string(index=False))
    print("=" * 80 + "\n")
