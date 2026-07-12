# Backtesting Framework Design

## Overview
A proper backtesting framework is essential for validating strategy performance before live deployment. This document outlines the design and implementation.

---

## Current State
The project currently has **no backtesting framework**. Training happens on historical data, but there's no simulation of actual trading performance with realistic assumptions.

---

## Backtesting Requirements

### Must-Have Features
1. **Realistic execution simulation** - Fees, slippage, partial fills
2. **Walk-forward testing** - Out-of-sample validation
3. **Performance metrics** - Sharpe, Sortino, max drawdown, win rate
4. **Trade-level analysis** - Entry/exit analysis, holding periods
5. **Comparison to benchmarks** - Buy & hold, equal weight

### Nice-to-Have Features
1. **Monte Carlo simulation** - Robustness testing
2. **Parameter sensitivity** - How stable is performance
3. **Regime analysis** - Performance in different market conditions
4. **Custom metrics** - Calmar ratio, profit factor

---

## Implementation

```python
# src/backtest/engine.py
"""
Walk-forward backtesting engine for crypto trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


@dataclass
class Trade:
    """Represents a completed trade."""
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    side: str  # 'long' or 'short'
    entry_signal: str
    exit_reason: str
    fees: float
    slippage: float
    
    @property
    def gross_pnl(self) -> float:
        if self.side == 'long':
            return (self.exit_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.exit_price) * self.quantity
    
    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.fees - self.slippage
    
    @property
    def return_pct(self) -> float:
        cost = self.entry_price * self.quantity
        return self.net_pnl / cost if cost > 0 else 0
    
    @property
    def hold_time_hours(self) -> float:
        return (self.exit_time - self.entry_time).total_seconds() / 3600


@dataclass
class BacktestConfig:
    """Configuration for backtest."""
    initial_capital: float = 10000
    fee_rate: float = 0.001  # 0.1% per side
    slippage_rate: float = 0.0005  # 0.05% per side
    max_positions: int = 5
    max_position_pct: float = 0.15
    max_exposure: float = 0.50
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    enable_shorting: bool = False


class BacktestEngine:
    """
    Walk-forward backtesting engine.
    
    Features:
    - Realistic fee and slippage modeling
    - Position sizing and risk limits
    - Stop-loss and take-profit execution
    - Detailed trade and performance tracking
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.reset()
    
    def reset(self):
        """Reset engine state."""
        self.cash = self.config.initial_capital
        self.positions: Dict[str, Dict] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        self.current_time: Optional[pd.Timestamp] = None
    
    def run(self,
            data: pd.DataFrame,
            signal_generator: Callable[[pd.DataFrame, int], Dict[str, str]],
            start_idx: int = 0) -> Dict:
        """
        Run backtest on historical data.
        
        Args:
            data: DataFrame with columns [timestamp, symbol, open, high, low, close, volume]
                  plus any features needed for signals
            signal_generator: Function that takes (data, current_idx) and returns
                             {symbol: 'BUY'/'SELL'/'HOLD'}
            start_idx: Index to start from (for walk-forward)
        
        Returns:
            Backtest results dictionary
        """
        self.reset()
        
        # Ensure data is sorted
        data = data.sort_values('timestamp').reset_index(drop=True)
        timestamps = data['timestamp'].unique()
        
        logger.info(f"Running backtest from {timestamps[start_idx]} to {timestamps[-1]}")
        
        for i, ts in enumerate(timestamps[start_idx:], start=start_idx):
            self.current_time = ts
            
            # Get current bar data
            current_data = data[data['timestamp'] == ts]
            
            # Update position values and check stops
            self._update_positions(current_data)
            self._check_stops(current_data)
            
            # Generate signals
            signals = signal_generator(data, i)
            
            # Execute trades
            for symbol, signal in signals.items():
                self._process_signal(symbol, signal, current_data)
            
            # Record equity
            self._record_equity(current_data)
        
        # Close all remaining positions at end
        self._close_all_positions(data[data['timestamp'] == timestamps[-1]])
        
        return self._calculate_results()
    
    def _update_positions(self, current_data: pd.DataFrame):
        """Update position values with current prices."""
        for symbol, pos in self.positions.items():
            symbol_data = current_data[current_data['symbol'] == symbol]
            if not symbol_data.empty:
                pos['current_price'] = symbol_data['close'].iloc[0]
                pos['current_value'] = pos['quantity'] * pos['current_price']
    
    def _check_stops(self, current_data: pd.DataFrame):
        """Check and execute stop-loss and take-profit orders."""
        positions_to_close = []
        
        for symbol, pos in self.positions.items():
            symbol_data = current_data[current_data['symbol'] == symbol]
            if symbol_data.empty:
                continue
            
            high = symbol_data['high'].iloc[0]
            low = symbol_data['low'].iloc[0]
            
            # Check stop-loss (use low for long positions)
            if pos['side'] == 'long' and low <= pos['stop_price']:
                positions_to_close.append((symbol, pos['stop_price'], 'stop_loss'))
            
            # Check take-profit (use high for long positions)
            elif pos['side'] == 'long' and high >= pos['take_profit_price']:
                positions_to_close.append((symbol, pos['take_profit_price'], 'take_profit'))
        
        for symbol, price, reason in positions_to_close:
            self._close_position(symbol, price, reason)
    
    def _process_signal(self, symbol: str, signal: str, current_data: pd.DataFrame):
        """Process a trading signal."""
        symbol_data = current_data[current_data['symbol'] == symbol]
        if symbol_data.empty:
            return
        
        price = symbol_data['close'].iloc[0]
        
        if signal == 'BUY' and symbol not in self.positions:
            self._open_position(symbol, price, 'long')
        
        elif signal == 'SELL' and symbol in self.positions:
            self._close_position(symbol, price, 'signal')
        
        elif signal == 'SELL' and symbol not in self.positions and self.config.enable_shorting:
            self._open_position(symbol, price, 'short')
    
    def _open_position(self, symbol: str, price: float, side: str):
        """Open a new position."""
        # Check limits
        if len(self.positions) >= self.config.max_positions:
            return
        
        total_value = self._get_total_value()
        current_exposure = self._get_current_exposure()
        
        if current_exposure >= self.config.max_exposure * total_value:
            return
        
        # Calculate position size
        max_position_value = total_value * self.config.max_position_pct
        available_for_exposure = total_value * self.config.max_exposure - current_exposure
        position_value = min(max_position_value, available_for_exposure, self.cash * 0.95)
        
        if position_value <= 0:
            return
        
        # Apply slippage to entry
        if side == 'long':
            execution_price = price * (1 + self.config.slippage_rate)
        else:
            execution_price = price * (1 - self.config.slippage_rate)
        
        quantity = position_value / execution_price
        cost = quantity * execution_price
        fees = cost * self.config.fee_rate
        
        # Check cash
        if cost + fees > self.cash:
            return
        
        # Execute
        self.cash -= (cost + fees)
        
        # Calculate stops
        if side == 'long':
            stop_price = execution_price * (1 - self.config.stop_loss_pct)
            take_profit_price = execution_price * (1 + self.config.take_profit_pct)
        else:
            stop_price = execution_price * (1 + self.config.stop_loss_pct)
            take_profit_price = execution_price * (1 - self.config.take_profit_pct)
        
        self.positions[symbol] = {
            'quantity': quantity,
            'entry_price': execution_price,
            'entry_time': self.current_time,
            'current_price': execution_price,
            'current_value': position_value,
            'side': side,
            'entry_fees': fees,
            'slippage': position_value * self.config.slippage_rate,
            'stop_price': stop_price,
            'take_profit_price': take_profit_price,
            'entry_signal': 'model'
        }
        
        logger.debug(f"Opened {side} {symbol}: {quantity:.6f} @ {execution_price:.2f}")
    
    def _close_position(self, symbol: str, price: float, reason: str):
        """Close an existing position."""
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        
        # Apply slippage to exit
        if pos['side'] == 'long':
            execution_price = price * (1 - self.config.slippage_rate)
        else:
            execution_price = price * (1 + self.config.slippage_rate)
        
        proceeds = pos['quantity'] * execution_price
        exit_fees = proceeds * self.config.fee_rate
        exit_slippage = proceeds * self.config.slippage_rate
        
        # Record trade
        trade = Trade(
            symbol=symbol,
            entry_time=pos['entry_time'],
            exit_time=self.current_time,
            entry_price=pos['entry_price'],
            exit_price=execution_price,
            quantity=pos['quantity'],
            side=pos['side'],
            entry_signal=pos['entry_signal'],
            exit_reason=reason,
            fees=pos['entry_fees'] + exit_fees,
            slippage=pos['slippage'] + exit_slippage
        )
        self.trades.append(trade)
        
        # Update cash
        self.cash += (proceeds - exit_fees)
        
        # Remove position
        del self.positions[symbol]
        
        logger.debug(f"Closed {symbol}: PnL={trade.net_pnl:.2f} ({reason})")
    
    def _close_all_positions(self, current_data: pd.DataFrame):
        """Close all remaining positions at market."""
        for symbol in list(self.positions.keys()):
            symbol_data = current_data[current_data['symbol'] == symbol]
            if not symbol_data.empty:
                price = symbol_data['close'].iloc[0]
                self._close_position(symbol, price, 'end_of_backtest')
    
    def _get_total_value(self) -> float:
        """Get total portfolio value."""
        positions_value = sum(p['current_value'] for p in self.positions.values())
        return self.cash + positions_value
    
    def _get_current_exposure(self) -> float:
        """Get current market exposure."""
        return sum(p['current_value'] for p in self.positions.values())
    
    def _record_equity(self, current_data: pd.DataFrame):
        """Record equity curve data point."""
        self.equity_curve.append({
            'timestamp': self.current_time,
            'total_value': self._get_total_value(),
            'cash': self.cash,
            'positions_value': self._get_current_exposure(),
            'n_positions': len(self.positions)
        })
    
    def _calculate_results(self) -> Dict:
        """Calculate backtest results and metrics."""
        if not self.trades:
            return {'status': 'no_trades', 'equity_curve': self.equity_curve}
        
        # Trade statistics
        trade_df = pd.DataFrame([{
            'pnl': t.net_pnl,
            'return': t.return_pct,
            'hold_time': t.hold_time_hours,
            'exit_reason': t.exit_reason
        } for t in self.trades])
        
        # Equity curve
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['returns'] = equity_df['total_value'].pct_change()
        
        # Calculate metrics
        metrics = self._calculate_metrics(trade_df, equity_df)
        
        return {
            'status': 'success',
            'metrics': metrics,
            'trades': self.trades,
            'equity_curve': equity_df.to_dict('records'),
            'trade_summary': trade_df.describe().to_dict()
        }
    
    def _calculate_metrics(self, trade_df: pd.DataFrame, equity_df: pd.DataFrame) -> Dict:
        """Calculate performance metrics."""
        returns = equity_df['returns'].dropna()
        
        # Basic stats
        total_return = (equity_df['total_value'].iloc[-1] / self.config.initial_capital) - 1
        n_trades = len(self.trades)
        n_winners = (trade_df['pnl'] > 0).sum()
        n_losers = (trade_df['pnl'] <= 0).sum()
        
        # Risk metrics
        if len(returns) > 0 and returns.std() > 0:
            sharpe = np.sqrt(252 * 24) * returns.mean() / returns.std()  # Hourly to annual
            sortino = np.sqrt(252 * 24) * returns.mean() / returns[returns < 0].std() if (returns < 0).any() else np.inf
        else:
            sharpe = 0
            sortino = 0
        
        # Drawdown
        equity_df['peak'] = equity_df['total_value'].cummax()
        equity_df['drawdown'] = (equity_df['peak'] - equity_df['total_value']) / equity_df['peak']
        max_drawdown = equity_df['drawdown'].max()
        
        # Win/loss analysis
        avg_win = trade_df[trade_df['pnl'] > 0]['pnl'].mean() if n_winners > 0 else 0
        avg_loss = trade_df[trade_df['pnl'] <= 0]['pnl'].mean() if n_losers > 0 else 0
        
        profit_factor = abs(trade_df[trade_df['pnl'] > 0]['pnl'].sum() / 
                          trade_df[trade_df['pnl'] <= 0]['pnl'].sum()) if n_losers > 0 else np.inf
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'calmar_ratio': total_return / max_drawdown if max_drawdown > 0 else np.inf,
            
            'n_trades': n_trades,
            'n_winners': n_winners,
            'n_losers': n_losers,
            'win_rate': n_winners / n_trades if n_trades > 0 else 0,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_trade': trade_df['pnl'].mean(),
            'avg_return_per_trade': trade_df['return'].mean(),
            
            'avg_hold_time_hours': trade_df['hold_time'].mean(),
            'total_fees': sum(t.fees for t in self.trades),
            'total_slippage': sum(t.slippage for t in self.trades),
            
            'final_value': equity_df['total_value'].iloc[-1],
            'initial_capital': self.config.initial_capital
        }
```

---

## Performance Reporting

```python
# src/backtest/report.py
"""
Backtest performance reporting and visualization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class BacktestReport:
    """Generate comprehensive backtest reports."""
    
    def __init__(self, results: Dict):
        self.results = results
        self.metrics = results.get('metrics', {})
        self.trades = results.get('trades', [])
        self.equity_curve = pd.DataFrame(results.get('equity_curve', []))
    
    def print_summary(self):
        """Print summary to console."""
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        
        print(f"\n--- PERFORMANCE ---")
        print(f"Total Return: {self.metrics.get('total_return_pct', 0):.2f}%")
        print(f"Sharpe Ratio: {self.metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Sortino Ratio: {self.metrics.get('sortino_ratio', 0):.2f}")
        print(f"Max Drawdown: {self.metrics.get('max_drawdown_pct', 0):.2f}%")
        print(f"Calmar Ratio: {self.metrics.get('calmar_ratio', 0):.2f}")
        
        print(f"\n--- TRADES ---")
        print(f"Total Trades: {self.metrics.get('n_trades', 0)}")
        print(f"Win Rate: {self.metrics.get('win_rate', 0)*100:.1f}%")
        print(f"Profit Factor: {self.metrics.get('profit_factor', 0):.2f}")
        print(f"Avg Win: ${self.metrics.get('avg_win', 0):.2f}")
        print(f"Avg Loss: ${self.metrics.get('avg_loss', 0):.2f}")
        print(f"Avg Hold Time: {self.metrics.get('avg_hold_time_hours', 0):.1f}h")
        
        print(f"\n--- COSTS ---")
        print(f"Total Fees: ${self.metrics.get('total_fees', 0):.2f}")
        print(f"Total Slippage: ${self.metrics.get('total_slippage', 0):.2f}")
        
        print(f"\n--- FINAL ---")
        print(f"Initial Capital: ${self.metrics.get('initial_capital', 0):,.2f}")
        print(f"Final Value: ${self.metrics.get('final_value', 0):,.2f}")
        print("="*60 + "\n")
    
    def generate_plots(self, save_dir: Optional[Path] = None) -> plt.Figure:
        """Generate performance visualization plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Backtest Performance Analysis', fontsize=14, fontweight='bold')
        
        # 1. Equity Curve
        ax = axes[0, 0]
        if not self.equity_curve.empty:
            ax.plot(self.equity_curve['timestamp'], self.equity_curve['total_value'], 
                   linewidth=1.5, color='blue')
            ax.axhline(y=self.metrics.get('initial_capital', 10000), 
                      color='gray', linestyle='--', alpha=0.7, label='Initial Capital')
            ax.fill_between(self.equity_curve['timestamp'], 
                           self.equity_curve['total_value'],
                           self.metrics.get('initial_capital', 10000),
                           alpha=0.3,
                           color='green' if self.metrics.get('total_return', 0) > 0 else 'red')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_title('Equity Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Drawdown
        ax = axes[0, 1]
        if not self.equity_curve.empty:
            self.equity_curve['peak'] = self.equity_curve['total_value'].cummax()
            self.equity_curve['drawdown'] = (self.equity_curve['peak'] - self.equity_curve['total_value']) / self.equity_curve['peak']
            ax.fill_between(self.equity_curve['timestamp'], 
                           self.equity_curve['drawdown'] * 100,
                           0, color='red', alpha=0.5)
            ax.axhline(y=self.metrics.get('max_drawdown_pct', 0), 
                      color='darkred', linestyle='--', label=f"Max DD: {self.metrics.get('max_drawdown_pct', 0):.1f}%")
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.set_title('Drawdown')
        ax.legend()
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        
        # 3. Trade P&L Distribution
        ax = axes[1, 0]
        if self.trades:
            pnls = [t.net_pnl for t in self.trades]
            colors = ['green' if p > 0 else 'red' for p in pnls]
            ax.bar(range(len(pnls)), pnls, color=colors, alpha=0.7, width=1)
            ax.axhline(y=0, color='black', linewidth=0.5)
            ax.axhline(y=np.mean(pnls), color='blue', linestyle='--', 
                      label=f'Avg: ${np.mean(pnls):.2f}')
        ax.set_xlabel('Trade #')
        ax.set_ylabel('P&L ($)')
        ax.set_title('Trade P&L')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Monthly Returns Heatmap
        ax = axes[1, 1]
        if not self.equity_curve.empty and 'timestamp' in self.equity_curve.columns:
            self.equity_curve['month'] = pd.to_datetime(self.equity_curve['timestamp']).dt.to_period('M')
            monthly_returns = self.equity_curve.groupby('month')['total_value'].last().pct_change() * 100
            
            if len(monthly_returns) > 1:
                # Simple bar chart of monthly returns
                ax.bar(range(len(monthly_returns)), monthly_returns.values, 
                      color=['green' if r > 0 else 'red' for r in monthly_returns.values],
                      alpha=0.7)
                ax.set_xticks(range(len(monthly_returns)))
                ax.set_xticklabels([str(m) for m in monthly_returns.index], rotation=45, ha='right')
                ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_xlabel('Month')
        ax.set_ylabel('Return (%)')
        ax.set_title('Monthly Returns')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            filepath = save_dir / 'backtest_report.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {filepath}")
        
        return fig
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame for analysis."""
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame([{
            'symbol': t.symbol,
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'quantity': t.quantity,
            'side': t.side,
            'gross_pnl': t.gross_pnl,
            'net_pnl': t.net_pnl,
            'return_pct': t.return_pct,
            'fees': t.fees,
            'slippage': t.slippage,
            'hold_time_hours': t.hold_time_hours,
            'exit_reason': t.exit_reason
        } for t in self.trades])
```

---

## Running a Backtest

```python
# scripts/run_backtest.py
"""
Example script to run a backtest.
"""

import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.backtest.engine import BacktestEngine, BacktestConfig
from src.backtest.report import BacktestReport
from src.models.trading_model import ImprovedTradingModel
from src.data.market_client import MarketClient
from src.analysis.sentiment import SentimentAnalyzer
from src.utils.config import TradingConfig


def main():
    # 1. Load configuration
    config = TradingConfig.from_yaml("configs/data.yaml")
    
    # 2. Load historical data
    print("Loading historical data...")
    market_client = MarketClient(config)
    market_df = market_client.fetch_ohlcv(lookback_days=90)  # 3 months
    
    # Load sentiment (would need historical Reddit data)
    # For now, we'll use a simplified version
    sentiment_df = pd.DataFrame()  # Placeholder
    
    # 3. Prepare features and train model
    print("Preparing features...")
    model = ImprovedTradingModel(enter_threshold=0.55)
    feature_df = model.prepare_features(market_df, sentiment_df)
    
    # Train on first 60% of data
    train_end = int(len(feature_df) * 0.6)
    train_df = feature_df.iloc[:train_end]
    
    print(f"Training on {len(train_df)} samples...")
    model.train(train_df, validate=False)
    
    # 4. Create signal generator
    def generate_signals(data: pd.DataFrame, idx: int) -> Dict[str, str]:
        """Generate signals for current timestamp."""
        current_ts = data['timestamp'].iloc[idx]
        current_data = data[data['timestamp'] == current_ts]
        
        if current_data.empty:
            return {}
        
        signals = model.predict(current_data)
        return {symbol: sig['action'] for symbol, sig in signals.items()}
    
    # 5. Run backtest
    print("Running backtest...")
    backtest_config = BacktestConfig(
        initial_capital=10000,
        fee_rate=0.001,
        slippage_rate=0.0005,
        max_positions=5,
        max_position_pct=0.15,
        stop_loss_pct=0.02,
        take_profit_pct=0.04
    )
    
    engine = BacktestEngine(backtest_config)
    results = engine.run(
        feature_df,
        generate_signals,
        start_idx=train_end  # Start after training period
    )
    
    # 6. Generate report
    report = BacktestReport(results)
    report.print_summary()
    report.generate_plots(save_dir=Path('reports'))
    
    # 7. Save detailed results
    trades_df = report.to_dataframe()
    trades_df.to_csv('reports/backtest_trades.csv', index=False)
    print("Results saved to reports/")


if __name__ == "__main__":
    main()
```

---

## Benchmark Comparison

```python
# src/backtest/benchmark.py
"""
Benchmark strategies for comparison.
"""

import pandas as pd
import numpy as np
from typing import Dict


def buy_and_hold_benchmark(data: pd.DataFrame, 
                           symbol: str,
                           initial_capital: float) -> Dict:
    """
    Calculate buy and hold returns for comparison.
    """
    symbol_data = data[data['symbol'] == symbol].sort_values('timestamp')
    
    if symbol_data.empty:
        return {}
    
    initial_price = symbol_data['close'].iloc[0]
    final_price = symbol_data['close'].iloc[-1]
    
    shares = initial_capital / initial_price
    final_value = shares * final_price
    total_return = (final_value - initial_capital) / initial_capital
    
    # Calculate metrics
    returns = symbol_data['close'].pct_change().dropna()
    sharpe = np.sqrt(252 * 24) * returns.mean() / returns.std() if returns.std() > 0 else 0
    
    max_price = symbol_data['close'].cummax()
    drawdowns = (max_price - symbol_data['close']) / max_price
    max_drawdown = drawdowns.max()
    
    return {
        'strategy': f'Buy & Hold {symbol}',
        'total_return': total_return,
        'total_return_pct': total_return * 100,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown * 100,
        'final_value': final_value
    }


def equal_weight_benchmark(data: pd.DataFrame,
                          symbols: list,
                          initial_capital: float) -> Dict:
    """
    Equal-weight portfolio of all symbols.
    """
    capital_per_symbol = initial_capital / len(symbols)
    
    total_final_value = 0
    all_returns = []
    
    for symbol in symbols:
        symbol_data = data[data['symbol'] == symbol].sort_values('timestamp')
        if symbol_data.empty:
            continue
        
        initial_price = symbol_data['close'].iloc[0]
        final_price = symbol_data['close'].iloc[-1]
        shares = capital_per_symbol / initial_price
        final_value = shares * final_price
        total_final_value += final_value
        
        returns = symbol_data['close'].pct_change().dropna()
        all_returns.append(returns)
    
    # Portfolio returns
    if all_returns:
        portfolio_returns = pd.concat(all_returns, axis=1).mean(axis=1)
        sharpe = np.sqrt(252 * 24) * portfolio_returns.mean() / portfolio_returns.std()
        
        cumulative = (1 + portfolio_returns).cumprod()
        max_dd = ((cumulative.cummax() - cumulative) / cumulative.cummax()).max()
    else:
        sharpe = 0
        max_dd = 0
    
    total_return = (total_final_value - initial_capital) / initial_capital
    
    return {
        'strategy': 'Equal Weight Portfolio',
        'total_return': total_return,
        'total_return_pct': total_return * 100,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'max_drawdown_pct': max_dd * 100,
        'final_value': total_final_value
    }
```

---

## Summary

| Component | Purpose |
|-----------|---------|
| BacktestEngine | Simulates trading with realistic costs |
| BacktestConfig | Centralizes backtest parameters |
| BacktestReport | Generates performance reports |
| Benchmarks | Compare strategy vs buy-and-hold |
