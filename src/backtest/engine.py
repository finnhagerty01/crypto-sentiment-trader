# src/backtest/engine.py
"""
Walk-forward backtesting engine for crypto trading strategies.

Provides realistic trade simulation with:
- Fee and slippage modeling
- Position sizing and risk limits
- Stop-loss and take-profit execution
- Detailed trade and performance tracking
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order execution types."""

    MARKET = "market"
    LIMIT = "limit"


@dataclass
class Trade:
    """
    Represents a completed trade with full P&L accounting.

    Attributes:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        entry_time: Timestamp of entry
        exit_time: Timestamp of exit
        entry_price: Price at entry (after slippage)
        exit_price: Price at exit (after slippage)
        quantity: Number of units traded
        side: Trade direction ('long' or 'short')
        entry_signal: What triggered entry (e.g., 'model', 'manual')
        exit_reason: What triggered exit (e.g., 'signal', 'stop_loss', 'take_profit')
        fees: Total fees paid (entry + exit)
        slippage: Total slippage cost (entry + exit)
    """

    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    side: str
    entry_signal: str
    exit_reason: str
    fees: float
    slippage: float

    @property
    def gross_pnl(self) -> float:
        """Gross profit/loss before fees and slippage."""
        if self.side == "long":
            return (self.exit_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.exit_price) * self.quantity

    @property
    def net_pnl(self) -> float:
        """Net profit/loss after fees and slippage."""
        return self.gross_pnl - self.fees - self.slippage

    @property
    def return_pct(self) -> float:
        """Return as percentage of position cost."""
        cost = self.entry_price * self.quantity
        return self.net_pnl / cost if cost > 0 else 0.0

    @property
    def hold_time_hours(self) -> float:
        """Duration of the trade in hours."""
        return (self.exit_time - self.entry_time).total_seconds() / 3600


@dataclass
class BacktestConfig:
    """
    Configuration parameters for backtesting.

    Attributes:
        initial_capital: Starting portfolio value in USD
        fee_rate: Trading fee as decimal (e.g., 0.001 = 0.1% per side)
        slippage_rate: Slippage as decimal (e.g., 0.0005 = 0.05% per side)
        max_positions: Maximum simultaneous positions
        max_position_pct: Maximum single position as % of portfolio
        max_exposure: Maximum total market exposure as % of portfolio
        stop_loss_pct: Default stop-loss percentage
        take_profit_pct: Default take-profit percentage
        enable_shorting: Whether to allow short positions
        use_confidence_sizing: Scale position size by signal confidence
        min_confidence: Minimum confidence to open a position
    """

    initial_capital: float = 10000.0
    fee_rate: float = 0.001
    slippage_rate: float = 0.0005
    max_positions: int = 5
    max_position_pct: float = 0.15
    max_exposure: float = 0.50
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    enable_shorting: bool = False
    use_confidence_sizing: bool = False
    min_confidence: float = 0.55


class BacktestEngine:
    """
    Walk-forward backtesting engine.

    Simulates trading on historical data with realistic execution costs,
    position limits, and risk controls.

    Example:
        config = BacktestConfig(initial_capital=10000)
        engine = BacktestEngine(config)

        def signal_gen(data, idx):
            # Your signal logic here
            return {'BTCUSDT': 'BUY'}

        results = engine.run(data, signal_gen)
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize the backtesting engine.

        Args:
            config: BacktestConfig with simulation parameters
        """
        self.config = config
        self.reset()

    def reset(self) -> None:
        """Reset engine state for a new backtest run."""
        self.cash: float = self.config.initial_capital
        self.positions: Dict[str, Dict] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        self.current_time: Optional[pd.Timestamp] = None

    def run(
        self,
        data: pd.DataFrame,
        signal_generator: Callable[
            [pd.DataFrame, int], Dict[str, Union[str, Dict]]
        ],
        start_idx: int = 0,
    ) -> Dict:
        """
        Run backtest on historical data.

        Args:
            data: DataFrame with columns [timestamp, symbol, open, high, low, close, volume]
                  plus any features needed for signals. Must be sorted by timestamp.
            signal_generator: Function that takes (data, current_idx) and returns
                             {symbol: 'BUY'/'SELL'/'HOLD'} or
                             {symbol: {'action': 'BUY', 'confidence': 0.72}}
            start_idx: Index to start from (for walk-forward, skip training period)

        Returns:
            Dictionary with backtest results including metrics, trades, and equity curve
        """
        self.reset()

        # Ensure data is sorted by timestamp
        data = data.sort_values("timestamp").reset_index(drop=True)
        timestamps = data["timestamp"].unique()

        if start_idx >= len(timestamps):
            logger.error(
                f"start_idx ({start_idx}) >= number of timestamps ({len(timestamps)})"
            )
            return {"status": "invalid_start_idx"}

        logger.info(
            f"Running backtest from {timestamps[start_idx]} to {timestamps[-1]}"
        )
        logger.info(f"Initial capital: ${self.config.initial_capital:,.2f}")

        for i in range(start_idx, len(timestamps)):
            ts = timestamps[i]
            self.current_time = pd.Timestamp(ts)

            # Get current bar data for all symbols
            current_data = data[data["timestamp"] == ts]

            # 1. Update position values with current prices
            self._update_positions(current_data)

            # 2. Check and execute stop-loss / take-profit
            self._check_stops(current_data)

            # 3. Generate signals using the provided function
            signals = signal_generator(data, i)

            # 4. Execute trades based on signals
            for symbol, signal in signals.items():
                self._process_signal(symbol, signal, current_data)

            # 5. Record equity curve data point
            self._record_equity(current_data)

        # Close all remaining positions at the end
        final_data = data[data["timestamp"] == timestamps[-1]]
        self._close_all_positions(final_data)

        return self._calculate_results()

    def _update_positions(self, current_data: pd.DataFrame) -> None:
        """
        Update position values with current prices.

        Args:
            current_data: DataFrame with current bar data for all symbols
        """
        for symbol, pos in self.positions.items():
            symbol_data = current_data[current_data["symbol"] == symbol]
            if not symbol_data.empty:
                pos["current_price"] = symbol_data["close"].iloc[0]
                pos["current_value"] = pos["quantity"] * pos["current_price"]

    def _check_stops(self, current_data: pd.DataFrame) -> None:
        """
        Check and execute stop-loss and take-profit orders.

        Uses high/low prices to simulate intrabar stop execution.

        Args:
            current_data: DataFrame with current bar data for all symbols
        """
        positions_to_close = []

        for symbol, pos in self.positions.items():
            symbol_data = current_data[current_data["symbol"] == symbol]
            if symbol_data.empty:
                continue

            high = symbol_data["high"].iloc[0]
            low = symbol_data["low"].iloc[0]

            if pos["side"] == "long":
                # Stop-loss: triggered if low touches stop price
                if low <= pos["stop_price"]:
                    positions_to_close.append((symbol, pos["stop_price"], "stop_loss"))
                # Take-profit: triggered if high touches take-profit price
                elif high >= pos["take_profit_price"]:
                    positions_to_close.append(
                        (symbol, pos["take_profit_price"], "take_profit")
                    )
            else:  # short position
                # Stop-loss: triggered if high touches stop price
                if high >= pos["stop_price"]:
                    positions_to_close.append((symbol, pos["stop_price"], "stop_loss"))
                # Take-profit: triggered if low touches take-profit price
                elif low <= pos["take_profit_price"]:
                    positions_to_close.append(
                        (symbol, pos["take_profit_price"], "take_profit")
                    )

        for symbol, price, reason in positions_to_close:
            self._close_position(symbol, price, reason)

    def _process_signal(
        self,
        symbol: str,
        signal: Union[str, Dict],
        current_data: pd.DataFrame,
    ) -> None:
        """
        Process a trading signal and execute if appropriate.

        Supports both simple string signals and dict signals with confidence:
        - String: 'BUY', 'SELL', 'HOLD'
        - Dict: {'action': 'BUY', 'confidence': 0.72, ...}

        Args:
            symbol: Trading symbol
            signal: Signal string or dict with action and optional confidence
            current_data: DataFrame with current bar data
        """
        symbol_data = current_data[current_data["symbol"] == symbol]
        if symbol_data.empty:
            return

        price = symbol_data["close"].iloc[0]

        # Parse signal - support both string and dict formats
        if isinstance(signal, dict):
            action = signal.get("action", "HOLD")
            confidence = signal.get("confidence", 1.0)
        else:
            action = signal
            confidence = 1.0

        # Check minimum confidence
        if confidence < self.config.min_confidence:
            return

        if action == "BUY":
            # BUY signal: close short position (exit short) or open long position
            if symbol in self.positions and self.positions[symbol]["side"] == "short":
                self._close_position(symbol, price, "signal")  # Exit short
            elif symbol not in self.positions:
                self._open_position(symbol, price, "long", confidence)  # Enter long

        elif action == "SELL":
            # SELL signal: close long position (exit long) or open short position
            if symbol in self.positions and self.positions[symbol]["side"] == "long":
                self._close_position(symbol, price, "signal")  # Exit long
            elif symbol not in self.positions and self.config.enable_shorting:
                self._open_position(symbol, price, "short", confidence)  # Enter short

    def _open_position(
        self,
        symbol: str,
        price: float,
        side: str,
        confidence: float = 1.0,
    ) -> None:
        """
        Open a new position with fee and slippage simulation.

        Args:
            symbol: Trading symbol
            price: Current market price (mid)
            side: Trade direction ('long' or 'short')
            confidence: Signal confidence (0-1) for position sizing
        """
        # Check position count limit
        if len(self.positions) >= self.config.max_positions:
            logger.debug(f"Max positions reached, skipping {symbol}")
            return

        total_value = self._get_total_value()
        current_exposure = self._get_current_exposure()

        # Check exposure limit
        if current_exposure >= self.config.max_exposure * total_value:
            logger.debug(f"Max exposure reached, skipping {symbol}")
            return

        # Calculate position size
        max_position_value = total_value * self.config.max_position_pct
        available_for_exposure = (
            total_value * self.config.max_exposure - current_exposure
        )
        position_value = min(
            max_position_value, available_for_exposure, self.cash * 0.95
        )

        # Scale by confidence if enabled
        if self.config.use_confidence_sizing:
            # Scale linearly from min_confidence to 1.0
            # At min_confidence, use 50% of position; at 1.0, use 100%
            min_conf = self.config.min_confidence
            confidence_scale = 0.5 + 0.5 * (confidence - min_conf) / (1.0 - min_conf)
            confidence_scale = max(0.5, min(1.0, confidence_scale))
            position_value *= confidence_scale

        if position_value <= 0:
            return

        # Apply slippage to entry price
        if side == "long":
            execution_price = price * (1 + self.config.slippage_rate)
        else:
            execution_price = price * (1 - self.config.slippage_rate)

        quantity = position_value / execution_price
        cost = quantity * execution_price
        fees = cost * self.config.fee_rate

        # Check cash availability
        if cost + fees > self.cash:
            logger.debug(f"Insufficient cash for {symbol}")
            return

        # Execute entry
        self.cash -= cost + fees

        # Calculate stop-loss and take-profit prices
        if side == "long":
            stop_price = execution_price * (1 - self.config.stop_loss_pct)
            take_profit_price = execution_price * (1 + self.config.take_profit_pct)
        else:
            stop_price = execution_price * (1 + self.config.stop_loss_pct)
            take_profit_price = execution_price * (1 - self.config.take_profit_pct)

        self.positions[symbol] = {
            "quantity": quantity,
            "entry_price": execution_price,
            "entry_time": self.current_time,
            "current_price": execution_price,
            "current_value": position_value,
            "side": side,
            "entry_fees": fees,
            "slippage": position_value * self.config.slippage_rate,
            "stop_price": stop_price,
            "take_profit_price": take_profit_price,
            "entry_signal": "model",
        }

        logger.debug(
            f"Opened {side} {symbol}: {quantity:.6f} @ {execution_price:.2f}"
        )

    def _close_position(self, symbol: str, price: float, reason: str) -> None:
        """
        Close an existing position with fee and slippage simulation.

        Args:
            symbol: Trading symbol
            price: Exit price (before slippage)
            reason: Reason for closing ('signal', 'stop_loss', 'take_profit', etc.)
        """
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]

        # Apply slippage to exit price
        if pos["side"] == "long":
            execution_price = price * (1 - self.config.slippage_rate)
        else:
            execution_price = price * (1 + self.config.slippage_rate)

        proceeds = pos["quantity"] * execution_price
        exit_fees = proceeds * self.config.fee_rate
        exit_slippage = proceeds * self.config.slippage_rate

        # Record trade
        trade = Trade(
            symbol=symbol,
            entry_time=pos["entry_time"],
            exit_time=self.current_time,
            entry_price=pos["entry_price"],
            exit_price=execution_price,
            quantity=pos["quantity"],
            side=pos["side"],
            entry_signal=pos["entry_signal"],
            exit_reason=reason,
            fees=pos["entry_fees"] + exit_fees,
            slippage=pos["slippage"] + exit_slippage,
        )
        self.trades.append(trade)

        # Update cash
        self.cash += proceeds - exit_fees

        # Remove position
        del self.positions[symbol]

        logger.debug(f"Closed {symbol}: PnL={trade.net_pnl:.2f} ({reason})")

    def _close_all_positions(self, current_data: pd.DataFrame) -> None:
        """
        Close all remaining positions at market.

        Called at end of backtest to realize all P&L.

        Args:
            current_data: DataFrame with final bar data
        """
        for symbol in list(self.positions.keys()):
            symbol_data = current_data[current_data["symbol"] == symbol]
            
            if not symbol_data.empty:
                price = symbol_data["close"].iloc[0]
            else:
                # FALLBACK: Use the current_price tracked in the position (last known price)
                price = self.positions[symbol]["current_price"]
                
            self._close_position(symbol, price, "end_of_backtest")

    def _get_total_value(self) -> float:
        """Get total portfolio value (cash + positions)."""
        positions_value = sum(p["current_value"] for p in self.positions.values())
        return self.cash + positions_value

    def _get_current_exposure(self) -> float:
        """Get current market exposure in USD."""
        return sum(p["current_value"] for p in self.positions.values())

    def _record_equity(self, current_data: pd.DataFrame) -> None:
        """
        Record equity curve data point.

        Args:
            current_data: DataFrame with current bar data (unused, kept for consistency)
        """
        self.equity_curve.append(
            {
                "timestamp": self.current_time,
                "total_value": self._get_total_value(),
                "cash": self.cash,
                "positions_value": self._get_current_exposure(),
                "n_positions": len(self.positions),
            }
        )

    def _calculate_results(self) -> Dict:
        """
        Calculate comprehensive backtest results and performance metrics.

        Returns:
            Dictionary containing status, metrics, trades, and equity curve
        """
        if not self.trades:
            logger.warning("Backtest completed with no trades")
            return {"status": "no_trades", "equity_curve": self.equity_curve}

        # Create trade DataFrame for analysis
        trade_df = pd.DataFrame(
            [
                {
                    "pnl": t.net_pnl,
                    "return": t.return_pct,
                    "hold_time": t.hold_time_hours,
                    "exit_reason": t.exit_reason,
                }
                for t in self.trades
            ]
        )

        # Create equity DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df["returns"] = equity_df["total_value"].pct_change()

        # Calculate metrics
        metrics = self._calculate_metrics(trade_df, equity_df)

        return {
            "status": "success",
            "metrics": metrics,
            "trades": self.trades,
            "equity_curve": equity_df.to_dict("records"),
            "trade_summary": trade_df.describe().to_dict(),
        }

    def _calculate_metrics(
        self, trade_df: pd.DataFrame, equity_df: pd.DataFrame
    ) -> Dict:
        """
        Calculate comprehensive performance metrics.

        Args:
            trade_df: DataFrame of trade-level statistics
            equity_df: DataFrame of equity curve

        Returns:
            Dictionary of performance metrics
        """
        returns = equity_df["returns"].dropna()

        # Basic statistics
        total_return = (
            equity_df["total_value"].iloc[-1] / self.config.initial_capital
        ) - 1
        n_trades = len(self.trades)
        n_winners = (trade_df["pnl"] > 0).sum()
        n_losers = (trade_df["pnl"] <= 0).sum()

        # Risk metrics (annualized from hourly data)
        hours_per_year = 252 * 24
        if len(returns) > 0 and returns.std() > 0:
            sharpe = np.sqrt(hours_per_year) * returns.mean() / returns.std()
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0 and negative_returns.std() > 0:
                sortino = (
                    np.sqrt(hours_per_year) * returns.mean() / negative_returns.std()
                )
            else:
                sortino = np.inf
        else:
            sharpe = 0.0
            sortino = 0.0

        # Drawdown calculation
        equity_df["peak"] = equity_df["total_value"].cummax()
        equity_df["drawdown"] = (
            equity_df["peak"] - equity_df["total_value"]
        ) / equity_df["peak"]
        max_drawdown = equity_df["drawdown"].max()

        # Win/loss analysis
        winners = trade_df[trade_df["pnl"] > 0]["pnl"]
        losers = trade_df[trade_df["pnl"] <= 0]["pnl"]
        avg_win = winners.mean() if len(winners) > 0 else 0.0
        avg_loss = losers.mean() if len(losers) > 0 else 0.0

        # Profit factor
        total_wins = winners.sum() if len(winners) > 0 else 0.0
        total_losses = abs(losers.sum()) if len(losers) > 0 else 0.0
        profit_factor = total_wins / total_losses if total_losses > 0 else np.inf

        # Calmar ratio (annualized return / max drawdown)
        calmar = total_return / max_drawdown if max_drawdown > 0 else np.inf

        return {
            # Return metrics
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            # Risk metrics
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown * 100,
            "calmar_ratio": calmar,
            # Trade statistics
            "n_trades": n_trades,
            "n_winners": n_winners,
            "n_losers": n_losers,
            "win_rate": n_winners / n_trades if n_trades > 0 else 0.0,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "avg_trade": trade_df["pnl"].mean(),
            "avg_return_per_trade": trade_df["return"].mean(),
            # Time metrics
            "avg_hold_time_hours": trade_df["hold_time"].mean(),
            # Cost metrics
            "total_fees": sum(t.fees for t in self.trades),
            "total_slippage": sum(t.slippage for t in self.trades),
            # Portfolio metrics
            "final_value": equity_df["total_value"].iloc[-1],
            "initial_capital": self.config.initial_capital,
        }


def run_walk_forward_backtest(
    data: pd.DataFrame,
    model_trainer: Callable[[pd.DataFrame], None],
    signal_generator: Callable[[pd.DataFrame, int], Dict[str, Union[str, Dict]]],
    config: BacktestConfig,
    n_splits: int = 5,
    train_pct: float = 0.7,
) -> Dict:
    """
    Run walk-forward backtesting with multiple train/test periods.

    This function divides the data into n_splits periods and for each:
    1. Trains the model on the training portion
    2. Runs backtest on the test portion
    3. Aggregates results across all periods

    Args:
        data: Full historical DataFrame with features
        model_trainer: Function that trains the model on a DataFrame
        signal_generator: Function that generates signals (passed to BacktestEngine)
        config: BacktestConfig for the backtest
        n_splits: Number of walk-forward periods
        train_pct: Fraction of each period used for training

    Returns:
        Dictionary with aggregated results:
        - per_period: List of results for each period
        - aggregated: Combined metrics across all periods
        - all_trades: All trades from all periods
    """
    timestamps = data["timestamp"].unique()
    n_timestamps = len(timestamps)

    # Calculate period boundaries
    period_size = n_timestamps // n_splits
    if period_size < 100:
        logger.warning(
            f"Period size ({period_size}) is small. Consider using fewer splits."
        )

    all_results = []
    all_trades = []
    all_equity = []

    cumulative_capital = config.initial_capital

    for i in range(n_splits):
        period_start = i * period_size
        period_end = min((i + 1) * period_size, n_timestamps)

        # Get period data
        period_timestamps = timestamps[period_start:period_end]
        period_data = data[data["timestamp"].isin(period_timestamps)]

        # Split into train/test
        train_end_idx = int(len(period_timestamps) * train_pct)
        train_timestamps = period_timestamps[:train_end_idx]
        train_data = period_data[period_data["timestamp"].isin(train_timestamps)]

        logger.info(
            f"Period {i + 1}/{n_splits}: "
            f"Train {period_timestamps[0]} to {train_timestamps[-1]}, "
            f"Test {train_timestamps[-1]} to {period_timestamps[-1]}"
        )

        # Train model on this period's training data
        model_trainer(train_data)

        # Run backtest on test portion
        # Use cumulative capital from previous periods
        period_config = BacktestConfig(
            initial_capital=cumulative_capital,
            fee_rate=config.fee_rate,
            slippage_rate=config.slippage_rate,
            max_positions=config.max_positions,
            max_position_pct=config.max_position_pct,
            max_exposure=config.max_exposure,
            stop_loss_pct=config.stop_loss_pct,
            take_profit_pct=config.take_profit_pct,
            enable_shorting=config.enable_shorting,
            use_confidence_sizing=config.use_confidence_sizing,
            min_confidence=config.min_confidence,
        )

        engine = BacktestEngine(period_config)
        results = engine.run(period_data, signal_generator, start_idx=train_end_idx)

        all_results.append(
            {
                "period": i + 1,
                "train_start": str(period_timestamps[0]),
                "train_end": str(train_timestamps[-1]),
                "test_start": str(train_timestamps[-1]),
                "test_end": str(period_timestamps[-1]),
                "results": results,
            }
        )

        if results.get("status") == "success":
            all_trades.extend(results["trades"])
            all_equity.extend(results["equity_curve"])
            cumulative_capital = results["metrics"]["final_value"]

    # Aggregate metrics
    successful_periods = [
        r for r in all_results if r["results"].get("status") == "success"
    ]

    if not successful_periods:
        return {
            "status": "no_successful_periods",
            "per_period": all_results,
        }

    # Calculate aggregated metrics
    total_return = (cumulative_capital / config.initial_capital) - 1
    total_trades = len(all_trades)

    if total_trades > 0:
        winners = sum(1 for t in all_trades if t.net_pnl > 0)
        win_rate = winners / total_trades
        avg_trade = sum(t.net_pnl for t in all_trades) / total_trades
        total_fees = sum(t.fees for t in all_trades)
        total_slippage = sum(t.slippage for t in all_trades)
    else:
        win_rate = 0.0
        avg_trade = 0.0
        total_fees = 0.0
        total_slippage = 0.0

    # Aggregate Sharpe from equity curve
    if all_equity:
        equity_df = pd.DataFrame(all_equity)
        returns = equity_df["total_value"].pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            sharpe = np.sqrt(252 * 24) * returns.mean() / returns.std()
        else:
            sharpe = 0.0

        # Max drawdown
        equity_df["peak"] = equity_df["total_value"].cummax()
        equity_df["drawdown"] = (
            equity_df["peak"] - equity_df["total_value"]
        ) / equity_df["peak"]
        max_drawdown = equity_df["drawdown"].max()
    else:
        sharpe = 0.0
        max_drawdown = 0.0

    return {
        "status": "success",
        "per_period": all_results,
        "aggregated": {
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown * 100,
            "n_trades": total_trades,
            "win_rate": win_rate,
            "avg_trade": avg_trade,
            "total_fees": total_fees,
            "total_slippage": total_slippage,
            "final_value": cumulative_capital,
            "initial_capital": config.initial_capital,
        },
        "all_trades": all_trades,
        "equity_curve": all_equity,
    }
