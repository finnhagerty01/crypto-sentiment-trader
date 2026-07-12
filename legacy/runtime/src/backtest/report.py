# src/backtest/report.py
"""
Backtest performance reporting and visualization.

Generates comprehensive reports including:
- Console summary output
- Equity curve plots
- Drawdown analysis
- Trade P&L distribution
- Monthly returns heatmap
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from src.backtest.engine import Trade

logger = logging.getLogger(__name__)


class BacktestReport:
    """
    Generate comprehensive backtest reports and visualizations.

    Example:
        results = engine.run(data, signal_gen)
        report = BacktestReport(results)
        report.print_summary()
        report.generate_plots(save_dir=Path('reports'))
    """

    def __init__(self, results: Dict):
        """
        Initialize report with backtest results.

        Args:
            results: Dictionary returned by BacktestEngine.run()
        """
        self.results = results
        self.metrics = results.get("metrics", {})
        self.trades: List[Trade] = results.get("trades", [])
        self.equity_curve = pd.DataFrame(results.get("equity_curve", []))

    def print_summary(self) -> None:
        """Print formatted summary to console."""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)

        if self.results.get("status") != "success":
            print(f"Status: {self.results.get('status', 'unknown')}")
            print("=" * 60 + "\n")
            return

        print(f"\n--- PERFORMANCE ---")
        print(f"Total Return: {self.metrics.get('total_return_pct', 0):.2f}%")
        print(f"Sharpe Ratio: {self.metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Sortino Ratio: {self.metrics.get('sortino_ratio', 0):.2f}")
        print(f"Max Drawdown: {self.metrics.get('max_drawdown_pct', 0):.2f}%")
        print(f"Calmar Ratio: {self.metrics.get('calmar_ratio', 0):.2f}")

        print(f"\n--- TRADES ---")
        print(f"Total Trades: {self.metrics.get('n_trades', 0)}")
        print(f"Win Rate: {self.metrics.get('win_rate', 0) * 100:.1f}%")
        print(f"Profit Factor: {self.metrics.get('profit_factor', 0):.2f}")
        print(f"Avg Win: ${self.metrics.get('avg_win', 0):.2f}")
        print(f"Avg Loss: ${self.metrics.get('avg_loss', 0):.2f}")
        print(f"Avg Trade: ${self.metrics.get('avg_trade', 0):.2f}")
        print(f"Avg Hold Time: {self.metrics.get('avg_hold_time_hours', 0):.1f}h")

        print(f"\n--- COSTS ---")
        print(f"Total Fees: ${self.metrics.get('total_fees', 0):.2f}")
        print(f"Total Slippage: ${self.metrics.get('total_slippage', 0):.2f}")

        print(f"\n--- FINAL ---")
        print(f"Initial Capital: ${self.metrics.get('initial_capital', 0):,.2f}")
        print(f"Final Value: ${self.metrics.get('final_value', 0):,.2f}")
        print("=" * 60 + "\n")

    def generate_plots(self, save_dir: Optional[Path] = None) -> Optional["plt.Figure"]:
        """
        Generate performance visualization plots.

        Creates a 2x2 grid with:
        1. Equity curve with initial capital line
        2. Drawdown chart
        3. Trade P&L bar chart
        4. Monthly returns bar chart

        Args:
            save_dir: Optional directory to save the plot. If None, displays interactively.

        Returns:
            matplotlib Figure object, or None if matplotlib unavailable
        """
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib not available, skipping plots")
            return None

        if self.equity_curve.empty:
            logger.warning("No equity curve data, skipping plots")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Backtest Performance Analysis", fontsize=14, fontweight="bold")

        # 1. Equity Curve
        ax = axes[0, 0]
        self._plot_equity_curve(ax)

        # 2. Drawdown
        ax = axes[0, 1]
        self._plot_drawdown(ax)

        # 3. Trade P&L Distribution
        ax = axes[1, 0]
        self._plot_trade_pnl(ax)

        # 4. Monthly Returns
        ax = axes[1, 1]
        self._plot_monthly_returns(ax)

        plt.tight_layout()

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            filepath = save_dir / "backtest_report.png"
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            logger.info(f"Saved plot to {filepath}")

        return fig

    def _plot_equity_curve(self, ax: "plt.Axes") -> None:
        """Plot equity curve with initial capital reference line."""
        initial_capital = self.metrics.get("initial_capital", 10000)
        total_return = self.metrics.get("total_return", 0)

        ax.plot(
            self.equity_curve["timestamp"],
            self.equity_curve["total_value"],
            linewidth=1.5,
            color="blue",
            label="Portfolio Value",
        )
        ax.axhline(
            y=initial_capital,
            color="gray",
            linestyle="--",
            alpha=0.7,
            label="Initial Capital",
        )

        # Fill area between equity and initial capital
        ax.fill_between(
            self.equity_curve["timestamp"],
            self.equity_curve["total_value"],
            initial_capital,
            alpha=0.3,
            color="green" if total_return > 0 else "red",
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.set_title("Equity Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Format y-axis with comma separator
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f"${x:,.0f}")
        )

    def _plot_drawdown(self, ax: "plt.Axes") -> None:
        """Plot drawdown chart."""
        # Calculate drawdown if not already in equity curve
        equity = self.equity_curve.copy()
        equity["peak"] = equity["total_value"].cummax()
        equity["drawdown"] = (equity["peak"] - equity["total_value"]) / equity["peak"]

        ax.fill_between(
            equity["timestamp"],
            equity["drawdown"] * 100,
            0,
            color="red",
            alpha=0.5,
        )

        max_dd = self.metrics.get("max_drawdown_pct", 0)
        ax.axhline(
            y=max_dd,
            color="darkred",
            linestyle="--",
            label=f"Max DD: {max_dd:.1f}%",
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.set_title("Drawdown")
        ax.legend()
        ax.invert_yaxis()  # Drawdown is negative, so invert
        ax.grid(True, alpha=0.3)

    def _plot_trade_pnl(self, ax: "plt.Axes") -> None:
        """Plot trade P&L distribution as bar chart."""
        if not self.trades:
            ax.text(
                0.5,
                0.5,
                "No trades",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Trade P&L")
            return

        pnls = [t.net_pnl for t in self.trades]
        colors = ["green" if p > 0 else "red" for p in pnls]

        ax.bar(range(len(pnls)), pnls, color=colors, alpha=0.7, width=1)
        ax.axhline(y=0, color="black", linewidth=0.5)

        avg_pnl = np.mean(pnls)
        ax.axhline(
            y=avg_pnl,
            color="blue",
            linestyle="--",
            label=f"Avg: ${avg_pnl:.2f}",
        )

        ax.set_xlabel("Trade #")
        ax.set_ylabel("P&L ($)")
        ax.set_title("Trade P&L")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_monthly_returns(self, ax: "plt.Axes") -> None:
        """Plot monthly returns as bar chart."""
        if self.equity_curve.empty or "timestamp" not in self.equity_curve.columns:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Monthly Returns")
            return

        equity = self.equity_curve.copy()
        equity["timestamp"] = pd.to_datetime(equity["timestamp"])
        equity["month"] = equity["timestamp"].dt.to_period("M")

        # Get last value of each month and calculate returns
        monthly_values = equity.groupby("month")["total_value"].last()
        monthly_returns = monthly_values.pct_change() * 100

        if len(monthly_returns) > 1:
            monthly_returns = monthly_returns.dropna()
            colors = ["green" if r > 0 else "red" for r in monthly_returns.values]

            ax.bar(
                range(len(monthly_returns)),
                monthly_returns.values,
                color=colors,
                alpha=0.7,
            )

            # X-axis labels
            labels = [str(m) for m in monthly_returns.index]
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.axhline(y=0, color="black", linewidth=0.5)
        else:
            ax.text(
                0.5,
                0.5,
                "Insufficient data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.set_xlabel("Month")
        ax.set_ylabel("Return (%)")
        ax.set_title("Monthly Returns")
        ax.grid(True, alpha=0.3)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert trade results to DataFrame for further analysis.

        Returns:
            DataFrame with one row per trade and all trade attributes
        """
        if not self.trades:
            return pd.DataFrame()

        return pd.DataFrame(
            [
                {
                    "symbol": t.symbol,
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "quantity": t.quantity,
                    "side": t.side,
                    "gross_pnl": t.gross_pnl,
                    "net_pnl": t.net_pnl,
                    "return_pct": t.return_pct * 100,
                    "fees": t.fees,
                    "slippage": t.slippage,
                    "hold_time_hours": t.hold_time_hours,
                    "entry_signal": t.entry_signal,
                    "exit_reason": t.exit_reason,
                }
                for t in self.trades
            ]
        )

    def get_exit_reason_breakdown(self) -> pd.DataFrame:
        """
        Get breakdown of exits by reason.

        Returns:
            DataFrame with exit reason statistics
        """
        if not self.trades:
            return pd.DataFrame()

        trades_df = self.to_dataframe()
        breakdown = (
            trades_df.groupby("exit_reason")
            .agg(
                {
                    "net_pnl": ["count", "sum", "mean"],
                    "return_pct": "mean",
                    "hold_time_hours": "mean",
                }
            )
            .round(2)
        )
        breakdown.columns = [
            "count",
            "total_pnl",
            "avg_pnl",
            "avg_return_pct",
            "avg_hold_hours",
        ]
        return breakdown

    def get_symbol_breakdown(self) -> pd.DataFrame:
        """
        Get performance breakdown by symbol.

        Returns:
            DataFrame with per-symbol statistics
        """
        if not self.trades:
            return pd.DataFrame()

        trades_df = self.to_dataframe()
        breakdown = (
            trades_df.groupby("symbol")
            .agg(
                {
                    "net_pnl": ["count", "sum", "mean"],
                    "return_pct": "mean",
                }
            )
            .round(2)
        )
        breakdown.columns = ["n_trades", "total_pnl", "avg_pnl", "avg_return_pct"]
        breakdown = breakdown.sort_values("total_pnl", ascending=False)
        return breakdown

    def to_dict(self) -> Dict:
        """
        Export full results as dictionary for serialization.

        Returns:
            Dictionary with all results data
        """
        return {
            "status": self.results.get("status"),
            "metrics": self.metrics,
            "trades": self.to_dataframe().to_dict("records") if self.trades else [],
            "equity_curve": self.equity_curve.to_dict("records"),
            "exit_breakdown": (
                self.get_exit_reason_breakdown().to_dict()
                if self.trades
                else {}
            ),
            "symbol_breakdown": (
                self.get_symbol_breakdown().to_dict() if self.trades else {}
            ),
        }
