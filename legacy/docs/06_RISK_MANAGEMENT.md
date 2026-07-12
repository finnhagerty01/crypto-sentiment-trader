# Risk Management Implementation Guide

## Overview
This document covers essential risk management components for the crypto sentiment trader. Without proper risk management, even a profitable strategy can lead to ruin.

---

## Current State Analysis

### What Exists (`main.py` and `src/execution/live.py`)
```python
# Current issues:
# 1. Fixed $100 position size regardless of account size
quantity = round(100 / price, 5)  # HARDCODED $100 POSITION SIZE

# 2. No stop-loss logic
# 3. No position limits
# 4. No drawdown controls
# 5. No portfolio-level risk management
```

---

## Risk Management Components

### 1. Position Sizing

```python
# src/risk/position_sizing.py
"""
Position sizing algorithms for crypto trading.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PositionSizer:
    """
    Calculate optimal position sizes based on various methods.
    """
    
    def __init__(self,
                 account_value: float,
                 max_position_pct: float = 0.10,
                 max_total_exposure: float = 0.50):
        """
        Args:
            account_value: Total account value in USD
            max_position_pct: Maximum % of account for single position
            max_total_exposure: Maximum % of account exposed to market
        """
        self.account_value = account_value
        self.max_position_pct = max_position_pct
        self.max_total_exposure = max_total_exposure
    
    def fixed_fractional(self, risk_per_trade: float = 0.02) -> float:
        """
        Fixed fractional: Risk fixed % of account per trade.
        
        Args:
            risk_per_trade: Percentage of account to risk (default 2%)
        
        Returns:
            Position size in USD
        """
        position_size = self.account_value * risk_per_trade
        return min(position_size, self.account_value * self.max_position_pct)
    
    def volatility_adjusted(self,
                           current_volatility: float,
                           target_volatility: float = 0.02,
                           base_position: Optional[float] = None) -> float:
        """
        Adjust position size based on current volatility.
        
        Higher volatility = smaller position.
        
        Args:
            current_volatility: Current asset volatility (e.g., 24h realized vol)
            target_volatility: Target volatility for position
            base_position: Base position size (default: 5% of account)
        
        Returns:
            Adjusted position size in USD
        """
        if base_position is None:
            base_position = self.account_value * 0.05
        
        if current_volatility <= 0:
            return base_position
        
        # Scale inversely with volatility
        adjustment = target_volatility / current_volatility
        adjusted_size = base_position * adjustment
        
        # Apply limits
        return min(adjusted_size, self.account_value * self.max_position_pct)
    
    def kelly_criterion(self,
                       win_rate: float,
                       avg_win: float,
                       avg_loss: float,
                       kelly_fraction: float = 0.25) -> float:
        """
        Kelly Criterion: Mathematically optimal bet sizing.
        
        Full Kelly is aggressive; use fractional Kelly (typically 25%).
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive number)
            kelly_fraction: Fraction of Kelly to use (default 0.25)
        
        Returns:
            Position size in USD
        """
        if avg_loss <= 0 or avg_win <= 0:
            return self.fixed_fractional(0.01)  # Fallback to conservative
        
        # Kelly formula: f* = (p * b - q) / b
        # where p = win rate, q = loss rate, b = win/loss ratio
        b = avg_win / avg_loss
        q = 1 - win_rate
        
        kelly_pct = (win_rate * b - q) / b
        
        # Apply fraction and limits
        kelly_pct = max(0, kelly_pct * kelly_fraction)
        kelly_pct = min(kelly_pct, self.max_position_pct)
        
        return self.account_value * kelly_pct
    
    def signal_confidence_weighted(self,
                                   confidence: float,
                                   min_confidence: float = 0.55,
                                   max_confidence: float = 0.80) -> float:
        """
        Scale position based on model confidence.
        
        Args:
            confidence: Model confidence (0-1)
            min_confidence: Minimum confidence for any position
            max_confidence: Confidence at which max position is used
        
        Returns:
            Position size in USD
        """
        if confidence < min_confidence:
            return 0
        
        # Linear scaling between min and max confidence
        confidence_range = max_confidence - min_confidence
        confidence_pct = (confidence - min_confidence) / confidence_range
        confidence_pct = min(confidence_pct, 1.0)  # Cap at 1.0
        
        # Scale from 25% to 100% of max position
        min_size_pct = 0.25
        size_pct = min_size_pct + (1 - min_size_pct) * confidence_pct
        
        max_position = self.account_value * self.max_position_pct
        return max_position * size_pct
    
    def calculate_position(self,
                          symbol: str,
                          price: float,
                          confidence: float,
                          volatility: float,
                          win_rate: Optional[float] = None,
                          avg_win: Optional[float] = None,
                          avg_loss: Optional[float] = None,
                          current_positions: Optional[Dict[str, float]] = None) -> Dict:
        """
        Calculate final position size combining multiple factors.
        
        Args:
            symbol: Trading symbol
            price: Current price
            confidence: Model confidence
            volatility: Current volatility (e.g., ATR %)
            win_rate: Historical win rate (optional)
            avg_win: Average win (optional)
            avg_loss: Average loss (optional)
            current_positions: Dict of current positions {symbol: value}
        
        Returns:
            Dictionary with position details
        """
        # Check total exposure
        if current_positions:
            current_exposure = sum(current_positions.values())
            remaining_capacity = (self.account_value * self.max_total_exposure) - current_exposure
            
            if remaining_capacity <= 0:
                logger.warning("Max exposure reached. No new positions allowed.")
                return {'size_usd': 0, 'quantity': 0, 'reason': 'max_exposure'}
        else:
            remaining_capacity = self.account_value * self.max_total_exposure
        
        # Calculate sizes from different methods
        sizes = []
        
        # 1. Confidence-weighted
        confidence_size = self.signal_confidence_weighted(confidence)
        sizes.append(('confidence', confidence_size))
        
        # 2. Volatility-adjusted
        vol_size = self.volatility_adjusted(volatility)
        sizes.append(('volatility', vol_size))
        
        # 3. Kelly (if we have the data)
        if all([win_rate, avg_win, avg_loss]):
            kelly_size = self.kelly_criterion(win_rate, avg_win, avg_loss)
            sizes.append(('kelly', kelly_size))
        
        # Take minimum of all methods (conservative)
        final_size_usd = min([s[1] for s in sizes])
        
        # Apply remaining capacity limit
        final_size_usd = min(final_size_usd, remaining_capacity)
        
        # Calculate quantity
        quantity = final_size_usd / price
        
        return {
            'symbol': symbol,
            'size_usd': final_size_usd,
            'quantity': quantity,
            'price': price,
            'pct_of_account': final_size_usd / self.account_value,
            'method_sizes': dict(sizes),
            'limiting_factor': min(sizes, key=lambda x: x[1])[0]
        }


class RiskBudget:
    """
    Portfolio-level risk budgeting.
    """
    
    def __init__(self,
                 account_value: float,
                 max_daily_loss: float = 0.05,
                 max_drawdown: float = 0.15):
        """
        Args:
            account_value: Total account value
            max_daily_loss: Maximum daily loss as % of account
            max_drawdown: Maximum drawdown from peak
        """
        self.account_value = account_value
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        
        self.daily_pnl = 0
        self.peak_value = account_value
        self.is_trading_halted = False
    
    def update(self, current_value: float) -> Dict:
        """
        Update risk metrics and check if trading should halt.
        
        Args:
            current_value: Current portfolio value
        
        Returns:
            Risk status dictionary
        """
        # Update peak
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        # Calculate drawdown
        drawdown = (self.peak_value - current_value) / self.peak_value
        
        # Check limits
        daily_loss_pct = -self.daily_pnl / self.account_value if self.daily_pnl < 0 else 0
        
        halt_reasons = []
        
        if daily_loss_pct >= self.max_daily_loss:
            halt_reasons.append(f"Daily loss limit hit: {daily_loss_pct:.1%}")
        
        if drawdown >= self.max_drawdown:
            halt_reasons.append(f"Max drawdown hit: {drawdown:.1%}")
        
        self.is_trading_halted = len(halt_reasons) > 0
        
        return {
            'current_value': current_value,
            'peak_value': self.peak_value,
            'drawdown': drawdown,
            'daily_pnl': self.daily_pnl,
            'is_halted': self.is_trading_halted,
            'halt_reasons': halt_reasons
        }
    
    def record_trade(self, pnl: float):
        """Record a trade's P&L."""
        self.daily_pnl += pnl
    
    def reset_daily(self):
        """Reset daily P&L tracking (call at start of each day)."""
        self.daily_pnl = 0
```

### 2. Stop-Loss Management

```python
# src/risk/stop_loss.py
"""
Stop-loss and take-profit management.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class StopType(Enum):
    FIXED = "fixed"
    ATR = "atr"
    TRAILING = "trailing"
    TIME = "time"


@dataclass
class StopLossOrder:
    """Represents a stop-loss configuration."""
    symbol: str
    entry_price: float
    stop_price: float
    take_profit_price: Optional[float]
    stop_type: StopType
    quantity: float
    entry_time: pd.Timestamp
    max_hold_hours: Optional[int] = None


class StopLossManager:
    """
    Manages stop-loss and take-profit orders.
    """
    
    def __init__(self,
                 default_stop_pct: float = 0.02,
                 default_take_profit_pct: float = 0.04,
                 atr_multiplier: float = 2.0,
                 trailing_activation_pct: float = 0.01):
        """
        Args:
            default_stop_pct: Default stop-loss percentage
            default_take_profit_pct: Default take-profit percentage
            atr_multiplier: Multiplier for ATR-based stops
            trailing_activation_pct: Profit % to activate trailing stop
        """
        self.default_stop_pct = default_stop_pct
        self.default_take_profit_pct = default_take_profit_pct
        self.atr_multiplier = atr_multiplier
        self.trailing_activation_pct = trailing_activation_pct
        
        self.active_stops: Dict[str, StopLossOrder] = {}
    
    def create_fixed_stop(self,
                         symbol: str,
                         entry_price: float,
                         quantity: float,
                         stop_pct: Optional[float] = None,
                         take_profit_pct: Optional[float] = None) -> StopLossOrder:
        """
        Create a fixed percentage stop-loss.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            quantity: Position quantity
            stop_pct: Stop-loss percentage (optional, uses default)
            take_profit_pct: Take-profit percentage (optional, uses default)
        
        Returns:
            StopLossOrder
        """
        stop_pct = stop_pct or self.default_stop_pct
        take_profit_pct = take_profit_pct or self.default_take_profit_pct
        
        stop = StopLossOrder(
            symbol=symbol,
            entry_price=entry_price,
            stop_price=entry_price * (1 - stop_pct),
            take_profit_price=entry_price * (1 + take_profit_pct),
            stop_type=StopType.FIXED,
            quantity=quantity,
            entry_time=pd.Timestamp.now('UTC')
        )
        
        self.active_stops[symbol] = stop
        
        logger.info(f"Created fixed stop for {symbol}: "
                   f"Entry={entry_price:.2f}, Stop={stop.stop_price:.2f}, "
                   f"TP={stop.take_profit_price:.2f}")
        
        return stop
    
    def create_atr_stop(self,
                       symbol: str,
                       entry_price: float,
                       quantity: float,
                       atr: float,
                       atr_multiplier: Optional[float] = None) -> StopLossOrder:
        """
        Create ATR-based stop-loss (volatility-adjusted).
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            quantity: Position quantity
            atr: Current ATR value
            atr_multiplier: Multiplier for ATR (optional, uses default)
        
        Returns:
            StopLossOrder
        """
        multiplier = atr_multiplier or self.atr_multiplier
        
        stop_distance = atr * multiplier
        take_profit_distance = atr * multiplier * 2  # 2:1 reward-risk
        
        stop = StopLossOrder(
            symbol=symbol,
            entry_price=entry_price,
            stop_price=entry_price - stop_distance,
            take_profit_price=entry_price + take_profit_distance,
            stop_type=StopType.ATR,
            quantity=quantity,
            entry_time=pd.Timestamp.now('UTC')
        )
        
        self.active_stops[symbol] = stop
        
        logger.info(f"Created ATR stop for {symbol}: "
                   f"Entry={entry_price:.2f}, Stop={stop.stop_price:.2f} "
                   f"({stop_distance:.2f} = {multiplier}x ATR)")
        
        return stop
    
    def create_time_stop(self,
                        symbol: str,
                        entry_price: float,
                        quantity: float,
                        max_hold_hours: int = 24) -> StopLossOrder:
        """
        Create time-based stop (exit after N hours regardless of P&L).
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            quantity: Position quantity
            max_hold_hours: Maximum hours to hold position
        
        Returns:
            StopLossOrder
        """
        stop = StopLossOrder(
            symbol=symbol,
            entry_price=entry_price,
            stop_price=entry_price * (1 - self.default_stop_pct),
            take_profit_price=entry_price * (1 + self.default_take_profit_pct),
            stop_type=StopType.TIME,
            quantity=quantity,
            entry_time=pd.Timestamp.now('UTC'),
            max_hold_hours=max_hold_hours
        )
        
        self.active_stops[symbol] = stop
        
        return stop
    
    def update_trailing_stop(self,
                            symbol: str,
                            current_price: float,
                            trailing_pct: float = 0.02) -> Optional[float]:
        """
        Update trailing stop if in profit.
        
        Args:
            symbol: Trading symbol
            current_price: Current price
            trailing_pct: Trailing stop percentage
        
        Returns:
            New stop price if updated, None otherwise
        """
        if symbol not in self.active_stops:
            return None
        
        stop = self.active_stops[symbol]
        
        # Calculate current profit
        profit_pct = (current_price - stop.entry_price) / stop.entry_price
        
        # Only trail if we're in profit beyond activation threshold
        if profit_pct < self.trailing_activation_pct:
            return None
        
        # Calculate new trailing stop
        new_stop = current_price * (1 - trailing_pct)
        
        # Only update if new stop is higher
        if new_stop > stop.stop_price:
            stop.stop_price = new_stop
            stop.stop_type = StopType.TRAILING
            
            logger.info(f"Trailing stop updated for {symbol}: "
                       f"New stop={new_stop:.2f}, Profit={profit_pct:.1%}")
            
            return new_stop
        
        return None
    
    def check_stops(self,
                   current_prices: Dict[str, float],
                   current_time: Optional[pd.Timestamp] = None) -> Dict[str, str]:
        """
        Check all active stops against current prices.
        
        Args:
            current_prices: Dictionary of {symbol: price}
            current_time: Current timestamp (for time stops)
        
        Returns:
            Dictionary of {symbol: action} where action is 'stop', 'take_profit', 
            'time_stop', or None
        """
        if current_time is None:
            current_time = pd.Timestamp.now('UTC')
        
        actions = {}
        
        for symbol, stop in list(self.active_stops.items()):
            if symbol not in current_prices:
                continue
            
            price = current_prices[symbol]
            
            # Check stop-loss
            if price <= stop.stop_price:
                actions[symbol] = 'stop_loss'
                logger.warning(f"STOP LOSS triggered for {symbol} at {price:.2f}")
                continue
            
            # Check take-profit
            if stop.take_profit_price and price >= stop.take_profit_price:
                actions[symbol] = 'take_profit'
                logger.info(f"TAKE PROFIT triggered for {symbol} at {price:.2f}")
                continue
            
            # Check time stop
            if stop.max_hold_hours:
                hours_held = (current_time - stop.entry_time).total_seconds() / 3600
                if hours_held >= stop.max_hold_hours:
                    actions[symbol] = 'time_stop'
                    logger.info(f"TIME STOP triggered for {symbol} after {hours_held:.1f}h")
                    continue
        
        return actions
    
    def remove_stop(self, symbol: str):
        """Remove a stop from active tracking."""
        if symbol in self.active_stops:
            del self.active_stops[symbol]
    
    def get_all_stops(self) -> Dict[str, Dict]:
        """Get summary of all active stops."""
        return {
            symbol: {
                'entry_price': stop.entry_price,
                'stop_price': stop.stop_price,
                'take_profit': stop.take_profit_price,
                'type': stop.stop_type.value,
                'quantity': stop.quantity,
                'hours_held': (pd.Timestamp.now('UTC') - stop.entry_time).total_seconds() / 3600
            }
            for symbol, stop in self.active_stops.items()
        }
```

### 2b. Minimum Hold Time with Risk Overrides

**Problem:** A fixed minimum hold time (e.g., 2 hours) prevents whipsawing on noise, but conflicts with stop-loss logic. If you buy and the price drops 8% in 20 minutes, a strict hold time forces you to watch losses deepen.

**Solution:** Hold time should prevent *signal-based* exits, not *risk-based* exits. Stop-loss and take-profit can always override the hold requirement.

```python
# src/risk/exit_manager.py
"""
Coordinates exit decisions with minimum hold time and risk overrides.
"""

import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ExitReason(Enum):
    """Reason for exit decision."""
    HOLD = "hold"                    # Must continue holding
    SIGNAL = "signal"                # Model signal says sell
    STOP_LOSS = "stop_loss"          # Risk override - price dropped
    TAKE_PROFIT = "take_profit"      # Risk override - price target hit
    TIME_LIMIT = "time_limit"        # Maximum hold time exceeded
    MANUAL = "manual"                # Manual override


@dataclass
class ExitDecision:
    """Result of exit evaluation."""
    can_exit: bool
    reason: ExitReason
    details: str


class ExitManager:
    """
    Manages exit decisions, coordinating hold time with risk overrides.
    
    Priority order for exits:
    1. Stop-loss (ALWAYS allowed - capital preservation)
    2. Take-profit (ALWAYS allowed - lock in gains)
    3. Maximum hold time (exit regardless of signal)
    4. Signal-based exit (only after minimum hold time)
    """
    
    def __init__(self,
                 min_hold_hours: float = 2.0,
                 max_hold_hours: float = 48.0,
                 stop_loss_pct: float = 0.03,
                 take_profit_pct: float = 0.05):
        """
        Args:
            min_hold_hours: Minimum time before signal-based exits allowed
            max_hold_hours: Maximum time before forced exit
            stop_loss_pct: Stop-loss percentage (can exit anytime)
            take_profit_pct: Take-profit percentage (can exit anytime)
        """
        self.min_hold_hours = min_hold_hours
        self.max_hold_hours = max_hold_hours
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
    
    def evaluate_exit(self,
                      entry_price: float,
                      current_price: float,
                      entry_time: pd.Timestamp,
                      current_time: pd.Timestamp,
                      signal_says_sell: bool) -> ExitDecision:
        """
        Evaluate whether position should be exited.
        
        Args:
            entry_price: Price at entry
            current_price: Current price
            entry_time: When position was opened
            current_time: Current timestamp
            signal_says_sell: Whether model/signal recommends selling
        
        Returns:
            ExitDecision with can_exit, reason, and details
        """
        hours_held = (current_time - entry_time).total_seconds() / 3600
        pnl_pct = (current_price - entry_price) / entry_price
        
        # PRIORITY 1: Stop-loss - ALWAYS allowed (capital preservation)
        if pnl_pct <= -self.stop_loss_pct:
            return ExitDecision(
                can_exit=True,
                reason=ExitReason.STOP_LOSS,
                details=f"Stop-loss triggered: {pnl_pct:.1%} loss exceeds -{self.stop_loss_pct:.1%} threshold"
            )
        
        # PRIORITY 2: Take-profit - ALWAYS allowed (lock in gains)
        if pnl_pct >= self.take_profit_pct:
            return ExitDecision(
                can_exit=True,
                reason=ExitReason.TAKE_PROFIT,
                details=f"Take-profit triggered: {pnl_pct:.1%} gain exceeds +{self.take_profit_pct:.1%} threshold"
            )
        
        # PRIORITY 3: Maximum hold time - forced exit
        if hours_held >= self.max_hold_hours:
            return ExitDecision(
                can_exit=True,
                reason=ExitReason.TIME_LIMIT,
                details=f"Max hold time exceeded: {hours_held:.1f}h >= {self.max_hold_hours}h"
            )
        
        # PRIORITY 4: Signal-based exit - only if minimum hold time satisfied
        if signal_says_sell:
            if hours_held >= self.min_hold_hours:
                return ExitDecision(
                    can_exit=True,
                    reason=ExitReason.SIGNAL,
                    details=f"Signal exit allowed: held {hours_held:.1f}h >= {self.min_hold_hours}h minimum"
                )
            else:
                remaining = self.min_hold_hours - hours_held
                return ExitDecision(
                    can_exit=False,
                    reason=ExitReason.HOLD,
                    details=f"Signal says sell but hold time not met: {remaining:.1f}h remaining"
                )
        
        # No exit conditions met
        return ExitDecision(
            can_exit=False,
            reason=ExitReason.HOLD,
            details=f"Holding: {hours_held:.1f}h elapsed, P&L={pnl_pct:+.1%}"
        )
    
    def check_all_positions(self,
                            positions: Dict[str, dict],
                            current_prices: Dict[str, float],
                            signals: Dict[str, str],
                            current_time: Optional[pd.Timestamp] = None) -> Dict[str, ExitDecision]:
        """
        Check exit conditions for all positions.
        
        Args:
            positions: Dict of {symbol: {'entry_price': float, 'entry_time': Timestamp}}
            current_prices: Dict of {symbol: current_price}
            signals: Dict of {symbol: 'BUY'|'SELL'|'HOLD'}
            current_time: Current timestamp (default: now)
        
        Returns:
            Dict of {symbol: ExitDecision}
        """
        if current_time is None:
            current_time = pd.Timestamp.now('UTC')
        
        decisions = {}
        
        for symbol, pos in positions.items():
            if symbol not in current_prices:
                continue
            
            signal_says_sell = signals.get(symbol) == 'SELL'
            
            decisions[symbol] = self.evaluate_exit(
                entry_price=pos['entry_price'],
                current_price=current_prices[symbol],
                entry_time=pos['entry_time'],
                current_time=current_time,
                signal_says_sell=signal_says_sell
            )
            
            if decisions[symbol].can_exit:
                logger.info(f"{symbol}: EXIT - {decisions[symbol].reason.value} - {decisions[symbol].details}")
            else:
                logger.debug(f"{symbol}: {decisions[symbol].details}")
        
        return decisions
```

**Integration with your existing hold time logic:**

Replace your current 2-hour hold check in `main.py` with the ExitManager:

```python
# OLD CODE (in main.py):
# if time_held < timedelta(hours=2):
#     continue  # Don't sell yet

# NEW CODE:
from src.risk.exit_manager import ExitManager, ExitReason

exit_manager = ExitManager(
    min_hold_hours=2.0,      # Your current 2-hour minimum
    max_hold_hours=48.0,     # Force exit after 48 hours
    stop_loss_pct=0.03,      # Exit immediately if down 3%
    take_profit_pct=0.05     # Exit immediately if up 5%
)

# In your main loop:
for symbol in open_positions:
    decision = exit_manager.evaluate_exit(
        entry_price=position.entry_price,
        current_price=current_price,
        entry_time=position.entry_time,
        current_time=pd.Timestamp.now('UTC'),
        signal_says_sell=(signal == 'SELL')
    )
    
    if decision.can_exit:
        execute_sell(symbol, reason=decision.reason.value)
    # If can_exit is False, position is held regardless of signal
```

**Key behavior:**
- Stop-loss at -3%: Exits immediately even if only 20 minutes in
- Take-profit at +5%: Exits immediately to lock in gains  
- Model says SELL at 1 hour: Blocked (minimum hold not met)
- Model says SELL at 2.5 hours: Allowed (minimum hold satisfied)
- Position at 48 hours: Forced exit regardless of signal or P&L

**Tuning the parameters:**

| Parameter | Conservative | Moderate | Aggressive |
|-----------|-------------|----------|------------|
| `min_hold_hours` | 4.0 | 2.0 | 0.5 |
| `max_hold_hours` | 72.0 | 48.0 | 24.0 |
| `stop_loss_pct` | 0.02 | 0.03 | 0.05 |
| `take_profit_pct` | 0.03 | 0.05 | 0.10 |

For volatile crypto markets, consider using ATR-based thresholds instead of fixed percentages (integrate with `StopLossManager.create_atr_stop()`).

### 3. Portfolio Risk Manager

```python
# src/risk/portfolio.py
"""
Portfolio-level risk management.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: pd.Timestamp
    current_price: float = 0.0
    
    @property
    def value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def cost_basis(self) -> float:
        return self.quantity * self.entry_price
    
    @property
    def unrealized_pnl(self) -> float:
        return self.value - self.cost_basis
    
    @property
    def unrealized_pnl_pct(self) -> float:
        if self.cost_basis == 0:
            return 0
        return self.unrealized_pnl / self.cost_basis


@dataclass
class PortfolioState:
    """Current portfolio state."""
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    
    @property
    def total_value(self) -> float:
        positions_value = sum(p.value for p in self.positions.values())
        return self.cash + positions_value
    
    @property
    def exposure(self) -> float:
        """Total market exposure as fraction of portfolio."""
        if self.total_value == 0:
            return 0
        return sum(p.value for p in self.positions.values()) / self.total_value
    
    @property
    def n_positions(self) -> int:
        return len(self.positions)


class PortfolioRiskManager:
    """
    Comprehensive portfolio risk management.
    
    Handles:
    - Position limits
    - Correlation-based exposure limits
    - Sector limits
    - Drawdown monitoring
    """
    
    def __init__(self,
                 initial_capital: float,
                 max_positions: int = 5,
                 max_exposure: float = 0.50,
                 max_single_position: float = 0.15,
                 max_correlated_exposure: float = 0.25,
                 max_sector_exposure: float = 0.30):
        """
        Args:
            initial_capital: Starting capital
            max_positions: Maximum number of simultaneous positions
            max_exposure: Maximum total market exposure
            max_single_position: Maximum single position as % of portfolio
            max_correlated_exposure: Max exposure to correlated assets
            max_sector_exposure: Max exposure to single sector
        """
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.max_exposure = max_exposure
        self.max_single_position = max_single_position
        self.max_correlated_exposure = max_correlated_exposure
        self.max_sector_exposure = max_sector_exposure
        
        self.state = PortfolioState(cash=initial_capital)
        
        # Track history for analytics
        self.value_history: List[Dict] = []
        self.trade_history: List[Dict] = []
        
        # Sector mappings
        self.sectors = {
            'BTCUSDT': 'store_of_value',
            'ETHUSDT': 'smart_contract',
            'SOLUSDT': 'smart_contract',
            'BNBUSDT': 'exchange',
            'XRPUSDT': 'payments',
            'ADAUSDT': 'smart_contract',
            'AVAXUSDT': 'smart_contract',
            'DOGEUSDT': 'meme',
            'LINKUSDT': 'oracle',
            'MATICUSDT': 'layer2',
            'DOTUSDT': 'interoperability',
            'ATOMUSDT': 'interoperability'
        }
    
    def can_open_position(self,
                         symbol: str,
                         size_usd: float,
                         current_prices: Dict[str, float]) -> Tuple[bool, str]:
        """
        Check if a new position can be opened.
        
        Args:
            symbol: Symbol to trade
            size_usd: Proposed position size in USD
            current_prices: Current prices for all symbols
        
        Returns:
            Tuple of (can_open, reason)
        """
        # Update current prices
        self._update_prices(current_prices)
        
        # 1. Check position count
        if self.state.n_positions >= self.max_positions:
            return False, f"Max positions reached ({self.max_positions})"
        
        # 2. Check if already have position in this symbol
        if symbol in self.state.positions:
            return False, f"Already have position in {symbol}"
        
        # 3. Check total exposure
        new_exposure = (self.state.exposure * self.state.total_value + size_usd) / self.state.total_value
        if new_exposure > self.max_exposure:
            return False, f"Would exceed max exposure ({new_exposure:.1%} > {self.max_exposure:.1%})"
        
        # 4. Check single position limit
        position_pct = size_usd / self.state.total_value
        if position_pct > self.max_single_position:
            return False, f"Position too large ({position_pct:.1%} > {self.max_single_position:.1%})"
        
        # 5. Check sector exposure
        sector = self.sectors.get(symbol, 'other')
        sector_exposure = self._get_sector_exposure(sector)
        if (sector_exposure + size_usd) / self.state.total_value > self.max_sector_exposure:
            return False, f"Would exceed {sector} sector limit"
        
        # 6. Check correlated exposure (BTC + correlated alts)
        if symbol != 'BTCUSDT':
            correlated_exposure = self._get_btc_correlated_exposure(size_usd)
            if correlated_exposure / self.state.total_value > self.max_correlated_exposure:
                return False, "Would exceed BTC-correlated exposure limit"
        
        return True, "OK"
    
    def _update_prices(self, prices: Dict[str, float]):
        """Update position prices."""
        for symbol, position in self.state.positions.items():
            if symbol in prices:
                position.current_price = prices[symbol]
    
    def _get_sector_exposure(self, sector: str) -> float:
        """Get current exposure to a sector."""
        exposure = 0
        for symbol, position in self.state.positions.items():
            if self.sectors.get(symbol) == sector:
                exposure += position.value
        return exposure
    
    def _get_btc_correlated_exposure(self, new_size: float) -> float:
        """
        Get exposure to BTC-correlated assets.
        
        Most alts have high correlation with BTC.
        """
        # BTC position
        btc_exposure = 0
        if 'BTCUSDT' in self.state.positions:
            btc_exposure = self.state.positions['BTCUSDT'].value
        
        # Other crypto (assume ~0.7 correlation with BTC)
        correlated_exposure = btc_exposure
        for symbol, position in self.state.positions.items():
            if symbol != 'BTCUSDT':
                correlated_exposure += position.value * 0.7
        
        # Add new position (assume correlated)
        correlated_exposure += new_size * 0.7
        
        return correlated_exposure
    
    def open_position(self,
                     symbol: str,
                     quantity: float,
                     price: float) -> bool:
        """
        Record opening a new position.
        
        Returns:
            True if successful
        """
        cost = quantity * price
        
        if cost > self.state.cash:
            logger.error(f"Insufficient cash for {symbol} position")
            return False
        
        self.state.cash -= cost
        self.state.positions[symbol] = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=price,
            entry_time=pd.Timestamp.now('UTC'),
            current_price=price
        )
        
        self.trade_history.append({
            'timestamp': pd.Timestamp.now('UTC'),
            'symbol': symbol,
            'action': 'BUY',
            'quantity': quantity,
            'price': price,
            'value': cost
        })
        
        logger.info(f"Opened position: {symbol} {quantity} @ {price}")
        return True
    
    def close_position(self,
                      symbol: str,
                      price: float,
                      reason: str = 'signal') -> Optional[Dict]:
        """
        Close an existing position.
        
        Returns:
            Trade summary or None if no position
        """
        if symbol not in self.state.positions:
            return None
        
        position = self.state.positions[symbol]
        proceeds = position.quantity * price
        pnl = proceeds - position.cost_basis
        pnl_pct = pnl / position.cost_basis
        
        # Update cash
        self.state.cash += proceeds
        
        # Record trade
        trade = {
            'timestamp': pd.Timestamp.now('UTC'),
            'symbol': symbol,
            'action': 'SELL',
            'quantity': position.quantity,
            'price': price,
            'entry_price': position.entry_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'hold_time_hours': (pd.Timestamp.now('UTC') - position.entry_time).total_seconds() / 3600
        }
        self.trade_history.append(trade)
        
        # Remove position
        del self.state.positions[symbol]
        
        logger.info(f"Closed {symbol}: PnL={pnl:.2f} ({pnl_pct:.1%}), Reason={reason}")
        
        return trade
    
    def record_snapshot(self, prices: Dict[str, float]):
        """Record portfolio value snapshot."""
        self._update_prices(prices)
        
        self.value_history.append({
            'timestamp': pd.Timestamp.now('UTC'),
            'total_value': self.state.total_value,
            'cash': self.state.cash,
            'positions_value': sum(p.value for p in self.state.positions.values()),
            'exposure': self.state.exposure,
            'n_positions': self.state.n_positions
        })
    
    def get_summary(self, prices: Dict[str, float]) -> Dict:
        """Get portfolio summary."""
        self._update_prices(prices)
        
        positions_summary = {}
        for symbol, pos in self.state.positions.items():
            positions_summary[symbol] = {
                'quantity': pos.quantity,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'value': pos.value,
                'unrealized_pnl': pos.unrealized_pnl,
                'unrealized_pnl_pct': pos.unrealized_pnl_pct
            }
        
        # Calculate metrics
        total_unrealized_pnl = sum(p.unrealized_pnl for p in self.state.positions.values())
        realized_pnl = sum(t.get('pnl', 0) for t in self.trade_history if 'pnl' in t)
        
        # Calculate drawdown if we have history
        if self.value_history:
            values = pd.Series([h['total_value'] for h in self.value_history])
            peak = values.cummax()
            drawdown = ((peak - values) / peak).max()
        else:
            drawdown = 0
        
        return {
            'total_value': self.state.total_value,
            'cash': self.state.cash,
            'exposure': self.state.exposure,
            'n_positions': self.state.n_positions,
            'positions': positions_summary,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_realized_pnl': realized_pnl,
            'total_pnl': total_unrealized_pnl + realized_pnl,
            'return_pct': (self.state.total_value - self.initial_capital) / self.initial_capital,
            'max_drawdown': drawdown,
            'n_trades': len([t for t in self.trade_history if t['action'] == 'SELL'])
        }
```

---

## Integration with Main Loop

```python
# Updated main.py snippet

from src.risk.position_sizing import PositionSizer
from src.risk.stop_loss import StopLossManager
from src.risk.portfolio import PortfolioRiskManager

def main():
    # ... existing initialization ...
    
    # Initialize risk management
    ACCOUNT_VALUE = 10000  # Set your actual account value
    
    position_sizer = PositionSizer(
        account_value=ACCOUNT_VALUE,
        max_position_pct=0.10,
        max_total_exposure=0.50
    )
    
    stop_manager = StopLossManager(
        default_stop_pct=0.02,
        default_take_profit_pct=0.04,
        atr_multiplier=2.0
    )
    
    portfolio = PortfolioRiskManager(
        initial_capital=ACCOUNT_VALUE,
        max_positions=5,
        max_exposure=0.50
    )
    
    # ... in the trading loop ...
    
    for symbol, signal in signals.items():
        if signal['action'] == "BUY":
            # 1. Check portfolio risk limits
            current_prices = {s: executor.get_price(s) for s in config.symbols}
            price = current_prices[symbol]
            
            # 2. Calculate position size
            position = position_sizer.calculate_position(
                symbol=symbol,
                price=price,
                confidence=signal['confidence'],
                volatility=latest_features.loc[latest_features['symbol'] == symbol, 'atr_14_pct'].iloc[0],
                current_positions={s: p.value for s, p in portfolio.state.positions.items()}
            )
            
            if position['size_usd'] <= 0:
                logger.info(f"Skipping {symbol}: {position.get('reason', 'size too small')}")
                continue
            
            # 3. Check portfolio limits
            can_open, reason = portfolio.can_open_position(symbol, position['size_usd'], current_prices)
            if not can_open:
                logger.info(f"Skipping {symbol}: {reason}")
                continue
            
            # 4. Execute trade
            if DRY_RUN:
                logger.info(f"[DRY RUN] Would BUY {position['quantity']:.6f} {symbol}")
            else:
                order = executor.execute_order(symbol, "BUY", position['quantity'])
                if order:
                    portfolio.open_position(symbol, position['quantity'], price)
                    
                    # Create stop-loss
                    atr = latest_features.loc[latest_features['symbol'] == symbol, 'atr_14'].iloc[0]
                    stop_manager.create_atr_stop(symbol, price, position['quantity'], atr)
    
    # Check stops
    stop_actions = stop_manager.check_stops(current_prices)
    for symbol, action in stop_actions.items():
        if action:
            price = current_prices[symbol]
            if DRY_RUN:
                logger.info(f"[DRY RUN] {action.upper()} triggered for {symbol}")
            else:
                portfolio.close_position(symbol, price, reason=action)
                executor.execute_order(symbol, "SELL", stop_manager.active_stops[symbol].quantity)
                stop_manager.remove_stop(symbol)
```

---

## Risk Management Summary

| Component | Purpose | Key Parameters |
|-----------|---------|----------------|
| PositionSizer | Calculate optimal position size | max_position=10%, volatility-adjusted |
| StopLossManager | Manage exits | ATR-based stops, trailing stops |
| PortfolioRiskManager | Portfolio-level limits | max_exposure=50%, max_positions=5 |
| RiskBudget | Daily/drawdown limits | max_daily_loss=5%, max_drawdown=15% |
