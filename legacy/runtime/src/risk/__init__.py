"""
Risk management module for crypto sentiment trading.

This module provides:
- Position sizing algorithms (fixed fractional, volatility-adjusted, Kelly criterion)
- Stop-loss management (fixed, ATR-based, trailing)
- Exit management with minimum hold time and risk overrides
- Portfolio-level risk controls (exposure limits, drawdown, sector limits)
"""

from src.risk.position_sizing import PositionSizer, RiskBudget
from src.risk.stop_loss import StopLossManager, StopType, StopLossOrder
from src.risk.exit_manager import ExitManager, ExitReason, ExitDecision
from src.risk.portfolio import PortfolioRiskManager, Position, PortfolioState

__all__ = [
    # Position sizing
    'PositionSizer',
    'RiskBudget',
    # Stop-loss
    'StopLossManager',
    'StopType',
    'StopLossOrder',
    # Exit management
    'ExitManager',
    'ExitReason',
    'ExitDecision',
    # Portfolio
    'PortfolioRiskManager',
    'Position',
    'PortfolioState',
]
