"""
Stop-loss and take-profit management.

Provides multiple stop-loss strategies:
- Fixed percentage: Simple percentage-based stops
- ATR-based: Volatility-adjusted stops using Average True Range
- Trailing: Locks in profits by moving stop up as price rises
- Time-based: Exit after maximum hold time regardless of P&L
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class StopType(Enum):
    """Type of stop-loss order."""
    FIXED = "fixed"
    ATR = "atr"
    TRAILING = "trailing"
    TIME = "time"


@dataclass
class StopLossOrder:
    """Represents a stop-loss configuration for a position."""
    symbol: str
    entry_price: float
    stop_price: float
    take_profit_price: Optional[float]
    stop_type: StopType
    quantity: float
    entry_time: pd.Timestamp
    max_hold_hours: Optional[int] = None
    highest_price: Optional[float] = None  # For trailing stops

    def __post_init__(self):
        """Initialize highest price for trailing stop tracking."""
        if self.highest_price is None:
            self.highest_price = self.entry_price


class StopLossManager:
    """
    Manages stop-loss and take-profit orders for all positions.

    Supports multiple stop types and automatically tracks which
    stops should be triggered based on current prices.
    """

    def __init__(
        self,
        default_stop_pct: float = 0.02,
        default_take_profit_pct: float = 0.04,
        atr_multiplier: float = 2.0,
        trailing_activation_pct: float = 0.01,
        trailing_distance_pct: float = 0.02
    ):
        """
        Initialize the stop-loss manager.

        Args:
            default_stop_pct: Default stop-loss percentage (default 2%)
            default_take_profit_pct: Default take-profit percentage (default 4%)
            atr_multiplier: Multiplier for ATR-based stops (default 2.0x ATR)
            trailing_activation_pct: Profit % to activate trailing stop (default 1%)
            trailing_distance_pct: Distance for trailing stop (default 2%)
        """
        self.default_stop_pct = default_stop_pct
        self.default_take_profit_pct = default_take_profit_pct
        self.atr_multiplier = atr_multiplier
        self.trailing_activation_pct = trailing_activation_pct
        self.trailing_distance_pct = trailing_distance_pct

        self.active_stops: Dict[str, StopLossOrder] = {}

    def create_fixed_stop(
        self,
        symbol: str,
        entry_price: float,
        quantity: float,
        stop_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None
    ) -> StopLossOrder:
        """
        Create a fixed percentage stop-loss.

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            quantity: Position quantity
            stop_pct: Stop-loss percentage (uses default if not provided)
            take_profit_pct: Take-profit percentage (uses default if not provided)

        Returns:
            StopLossOrder
        """
        stop_pct = stop_pct if stop_pct is not None else self.default_stop_pct
        take_profit_pct = take_profit_pct if take_profit_pct is not None else self.default_take_profit_pct

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

        logger.info(
            f"Created FIXED stop for {symbol}: "
            f"Entry=${entry_price:.4f}, Stop=${stop.stop_price:.4f} (-{stop_pct:.1%}), "
            f"TP=${stop.take_profit_price:.4f} (+{take_profit_pct:.1%})"
        )

        return stop

    def create_atr_stop(
        self,
        symbol: str,
        entry_price: float,
        quantity: float,
        atr: float,
        atr_multiplier: Optional[float] = None,
        reward_risk_ratio: float = 2.0
    ) -> StopLossOrder:
        """
        Create ATR-based stop-loss (volatility-adjusted).

        ATR stops adapt to market volatility - wider stops in volatile
        markets, tighter stops in calm markets.

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            quantity: Position quantity
            atr: Current ATR value (in price units, not percentage)
            atr_multiplier: Multiplier for ATR (uses default if not provided)
            reward_risk_ratio: Take-profit distance as multiple of stop distance

        Returns:
            StopLossOrder
        """
        multiplier = atr_multiplier if atr_multiplier is not None else self.atr_multiplier

        stop_distance = atr * multiplier
        take_profit_distance = stop_distance * reward_risk_ratio

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

        stop_pct = stop_distance / entry_price
        tp_pct = take_profit_distance / entry_price

        logger.info(
            f"Created ATR stop for {symbol}: "
            f"Entry=${entry_price:.4f}, Stop=${stop.stop_price:.4f} (-{stop_pct:.1%}), "
            f"TP=${stop.take_profit_price:.4f} (+{tp_pct:.1%}), "
            f"ATR=${atr:.4f}, Multiplier={multiplier}x"
        )

        return stop

    def create_time_stop(
        self,
        symbol: str,
        entry_price: float,
        quantity: float,
        max_hold_hours: int = 24,
        stop_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None
    ) -> StopLossOrder:
        """
        Create time-based stop (exit after N hours regardless of P&L).

        Also includes fixed stop-loss and take-profit levels.

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            quantity: Position quantity
            max_hold_hours: Maximum hours to hold position (default 24)
            stop_pct: Stop-loss percentage (uses default if not provided)
            take_profit_pct: Take-profit percentage (uses default if not provided)

        Returns:
            StopLossOrder
        """
        stop_pct = stop_pct if stop_pct is not None else self.default_stop_pct
        take_profit_pct = take_profit_pct if take_profit_pct is not None else self.default_take_profit_pct

        stop = StopLossOrder(
            symbol=symbol,
            entry_price=entry_price,
            stop_price=entry_price * (1 - stop_pct),
            take_profit_price=entry_price * (1 + take_profit_pct),
            stop_type=StopType.TIME,
            quantity=quantity,
            entry_time=pd.Timestamp.now('UTC'),
            max_hold_hours=max_hold_hours
        )

        self.active_stops[symbol] = stop

        logger.info(
            f"Created TIME stop for {symbol}: "
            f"Entry=${entry_price:.4f}, MaxHold={max_hold_hours}h, "
            f"Stop=${stop.stop_price:.4f}, TP=${stop.take_profit_price:.4f}"
        )

        return stop

    def update_trailing_stop(
        self,
        symbol: str,
        current_price: float,
        trailing_pct: Optional[float] = None
    ) -> Optional[float]:
        """
        Update trailing stop if position is in profit.

        Trailing stops lock in profits by moving the stop price up
        as the price rises. The stop only moves up, never down.

        Args:
            symbol: Trading symbol
            current_price: Current price
            trailing_pct: Trailing stop percentage (uses default if not provided)

        Returns:
            New stop price if updated, None otherwise
        """
        if symbol not in self.active_stops:
            return None

        stop = self.active_stops[symbol]
        trailing_pct = trailing_pct if trailing_pct is not None else self.trailing_distance_pct

        # Update highest price seen
        if current_price > stop.highest_price:
            stop.highest_price = current_price

        # Calculate current profit
        profit_pct = (current_price - stop.entry_price) / stop.entry_price

        # Only trail if we're in profit beyond activation threshold
        if profit_pct < self.trailing_activation_pct:
            return None

        # Calculate new trailing stop
        new_stop = stop.highest_price * (1 - trailing_pct)

        # Only update if new stop is higher than current
        if new_stop > stop.stop_price:
            old_stop = stop.stop_price
            stop.stop_price = new_stop
            stop.stop_type = StopType.TRAILING

            logger.info(
                f"Trailing stop updated for {symbol}: "
                f"${old_stop:.4f} -> ${new_stop:.4f}, "
                f"Profit={profit_pct:.1%}, High=${stop.highest_price:.4f}"
            )

            return new_stop

        return None

    def check_stops(
        self,
        current_prices: Dict[str, float],
        current_time: Optional[pd.Timestamp] = None
    ) -> Dict[str, str]:
        """
        Check all active stops against current prices.

        Args:
            current_prices: Dictionary of {symbol: current_price}
            current_time: Current timestamp (for time stops, default: now)

        Returns:
            Dictionary of {symbol: action} where action is one of:
                - 'stop_loss': Stop-loss triggered
                - 'take_profit': Take-profit triggered
                - 'time_stop': Time limit exceeded
                - None: No action needed
        """
        if current_time is None:
            current_time = pd.Timestamp.now('UTC')

        actions = {}

        for symbol, stop in list(self.active_stops.items()):
            if symbol not in current_prices:
                continue

            price = current_prices[symbol]

            # First, update trailing stop if applicable
            self.update_trailing_stop(symbol, price)

            # Check stop-loss
            if price <= stop.stop_price:
                actions[symbol] = 'stop_loss'
                pnl_pct = (price - stop.entry_price) / stop.entry_price
                logger.warning(
                    f"STOP LOSS triggered for {symbol}: "
                    f"Price=${price:.4f} <= Stop=${stop.stop_price:.4f}, "
                    f"P&L={pnl_pct:.1%}"
                )
                continue

            # Check take-profit
            if stop.take_profit_price and price >= stop.take_profit_price:
                actions[symbol] = 'take_profit'
                pnl_pct = (price - stop.entry_price) / stop.entry_price
                logger.info(
                    f"TAKE PROFIT triggered for {symbol}: "
                    f"Price=${price:.4f} >= TP=${stop.take_profit_price:.4f}, "
                    f"P&L={pnl_pct:.1%}"
                )
                continue

            # Check time stop
            if stop.max_hold_hours is not None:
                hours_held = (current_time - stop.entry_time).total_seconds() / 3600
                if hours_held >= stop.max_hold_hours:
                    actions[symbol] = 'time_stop'
                    pnl_pct = (price - stop.entry_price) / stop.entry_price
                    logger.info(
                        f"TIME STOP triggered for {symbol}: "
                        f"Held {hours_held:.1f}h >= {stop.max_hold_hours}h, "
                        f"P&L={pnl_pct:.1%}"
                    )
                    continue

        return actions

    def remove_stop(self, symbol: str) -> Optional[StopLossOrder]:
        """
        Remove a stop from active tracking.

        Args:
            symbol: Trading symbol

        Returns:
            The removed StopLossOrder, or None if not found
        """
        if symbol in self.active_stops:
            stop = self.active_stops.pop(symbol)
            logger.debug(f"Removed stop for {symbol}")
            return stop
        return None

    def get_stop(self, symbol: str) -> Optional[StopLossOrder]:
        """Get the stop-loss order for a symbol."""
        return self.active_stops.get(symbol)

    def get_all_stops(self) -> Dict[str, Dict]:
        """
        Get summary of all active stops.

        Returns:
            Dictionary of {symbol: stop_details}
        """
        now = pd.Timestamp.now('UTC')

        return {
            symbol: {
                'entry_price': stop.entry_price,
                'stop_price': stop.stop_price,
                'take_profit': stop.take_profit_price,
                'type': stop.stop_type.value,
                'quantity': stop.quantity,
                'hours_held': (now - stop.entry_time).total_seconds() / 3600,
                'max_hold_hours': stop.max_hold_hours,
                'highest_price': stop.highest_price,
                'current_stop_pct': (stop.entry_price - stop.stop_price) / stop.entry_price
            }
            for symbol, stop in self.active_stops.items()
        }

    def calculate_risk(self, symbol: str, current_price: float) -> Optional[Dict]:
        """
        Calculate current risk metrics for a position.

        Args:
            symbol: Trading symbol
            current_price: Current price

        Returns:
            Risk metrics dictionary or None if no stop exists
        """
        stop = self.active_stops.get(symbol)
        if not stop:
            return None

        # Current P&L
        pnl_pct = (current_price - stop.entry_price) / stop.entry_price
        pnl_usd = (current_price - stop.entry_price) * stop.quantity

        # Risk to stop
        risk_to_stop_pct = (current_price - stop.stop_price) / current_price
        risk_to_stop_usd = (current_price - stop.stop_price) * stop.quantity

        # Potential to take-profit
        if stop.take_profit_price:
            potential_pct = (stop.take_profit_price - current_price) / current_price
            potential_usd = (stop.take_profit_price - current_price) * stop.quantity
        else:
            potential_pct = None
            potential_usd = None

        return {
            'symbol': symbol,
            'entry_price': stop.entry_price,
            'current_price': current_price,
            'stop_price': stop.stop_price,
            'take_profit': stop.take_profit_price,
            'pnl_pct': pnl_pct,
            'pnl_usd': pnl_usd,
            'risk_to_stop_pct': risk_to_stop_pct,
            'risk_to_stop_usd': risk_to_stop_usd,
            'potential_pct': potential_pct,
            'potential_usd': potential_usd,
            'reward_risk_ratio': potential_pct / risk_to_stop_pct if potential_pct and risk_to_stop_pct > 0 else None
        }
