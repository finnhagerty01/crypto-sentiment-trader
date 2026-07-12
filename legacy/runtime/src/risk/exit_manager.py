"""
Exit management with minimum hold time and risk overrides.

Coordinates exit decisions by prioritizing:
1. Stop-loss (ALWAYS allowed - capital preservation)
2. Take-profit (ALWAYS allowed - lock in gains)
3. Maximum hold time (forced exit)
4. Signal-based exit (only after minimum hold time)

This prevents whipsawing on noise while still protecting capital.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import pandas as pd

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
    pnl_pct: float = 0.0
    hours_held: float = 0.0


class ExitManager:
    """
    Manages exit decisions, coordinating hold time with risk overrides.

    The key insight is that minimum hold time should prevent *signal-based*
    exits, but NOT *risk-based* exits. If price drops 8% in 20 minutes,
    stop-loss should trigger immediately regardless of hold time.

    Priority order for exits:
    1. Stop-loss (ALWAYS allowed - capital preservation)
    2. Take-profit (ALWAYS allowed - lock in gains)
    3. Maximum hold time (exit regardless of signal)
    4. Signal-based exit (only after minimum hold time)
    """

    def __init__(
        self,
        min_hold_hours: float = 2.0,
        max_hold_hours: float = 48.0,
        stop_loss_pct: float = 0.03,
        take_profit_pct: float = 0.05
    ):
        """
        Initialize the exit manager.

        Args:
            min_hold_hours: Minimum time before signal-based exits allowed (default 2h)
            max_hold_hours: Maximum time before forced exit (default 48h)
            stop_loss_pct: Stop-loss percentage - can exit anytime (default 3%)
            take_profit_pct: Take-profit percentage - can exit anytime (default 5%)
        """
        self.min_hold_hours = min_hold_hours
        self.max_hold_hours = max_hold_hours
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def evaluate_exit(
        self,
        entry_price: float,
        current_price: float,
        entry_time: pd.Timestamp,
        current_time: pd.Timestamp,
        signal_says_sell: bool
    ) -> ExitDecision:
        """
        Evaluate whether position should be exited.

        Args:
            entry_price: Price at entry
            current_price: Current price
            entry_time: When position was opened
            current_time: Current timestamp
            signal_says_sell: Whether model/signal recommends selling

        Returns:
            ExitDecision with can_exit, reason, details, pnl_pct, and hours_held
        """
        # Ensure timezone awareness
        if entry_time.tzinfo is None:
            entry_time = entry_time.tz_localize('UTC')
        if current_time.tzinfo is None:
            current_time = current_time.tz_localize('UTC')

        hours_held = (current_time - entry_time).total_seconds() / 3600
        pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0

        # PRIORITY 1: Stop-loss - ALWAYS allowed (capital preservation)
        if pnl_pct <= -self.stop_loss_pct:
            return ExitDecision(
                can_exit=True,
                reason=ExitReason.STOP_LOSS,
                details=f"Stop-loss triggered: {pnl_pct:.1%} loss exceeds -{self.stop_loss_pct:.1%} threshold",
                pnl_pct=pnl_pct,
                hours_held=hours_held
            )

        # PRIORITY 2: Take-profit - ALWAYS allowed (lock in gains)
        if pnl_pct >= self.take_profit_pct:
            return ExitDecision(
                can_exit=True,
                reason=ExitReason.TAKE_PROFIT,
                details=f"Take-profit triggered: {pnl_pct:.1%} gain exceeds +{self.take_profit_pct:.1%} threshold",
                pnl_pct=pnl_pct,
                hours_held=hours_held
            )

        # PRIORITY 3: Maximum hold time - forced exit
        if hours_held >= self.max_hold_hours:
            return ExitDecision(
                can_exit=True,
                reason=ExitReason.TIME_LIMIT,
                details=f"Max hold time exceeded: {hours_held:.1f}h >= {self.max_hold_hours}h",
                pnl_pct=pnl_pct,
                hours_held=hours_held
            )

        # PRIORITY 4: Signal-based exit - only if minimum hold time satisfied
        if signal_says_sell:
            if hours_held >= self.min_hold_hours:
                return ExitDecision(
                    can_exit=True,
                    reason=ExitReason.SIGNAL,
                    details=f"Signal exit allowed: held {hours_held:.1f}h >= {self.min_hold_hours}h minimum",
                    pnl_pct=pnl_pct,
                    hours_held=hours_held
                )
            else:
                remaining = self.min_hold_hours - hours_held
                return ExitDecision(
                    can_exit=False,
                    reason=ExitReason.HOLD,
                    details=f"Signal says sell but min hold not met: {remaining:.1f}h remaining ({hours_held:.1f}h/{self.min_hold_hours}h)",
                    pnl_pct=pnl_pct,
                    hours_held=hours_held
                )

        # No exit conditions met - continue holding
        return ExitDecision(
            can_exit=False,
            reason=ExitReason.HOLD,
            details=f"Holding: {hours_held:.1f}h elapsed, P&L={pnl_pct:+.1%}",
            pnl_pct=pnl_pct,
            hours_held=hours_held
        )

    def check_all_positions(
        self,
        positions: Dict[str, dict],
        current_prices: Dict[str, float],
        signals: Dict[str, str],
        current_time: Optional[pd.Timestamp] = None
    ) -> Dict[str, ExitDecision]:
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
                logger.warning(f"No price available for {symbol}, skipping exit check")
                continue

            entry_price = pos.get('entry_price', 0)
            entry_time = pos.get('entry_time')

            if entry_price <= 0 or entry_time is None:
                logger.warning(f"Invalid position data for {symbol}: entry_price={entry_price}, entry_time={entry_time}")
                continue

            signal_says_sell = signals.get(symbol, '').upper() == 'SELL'

            decision = self.evaluate_exit(
                entry_price=entry_price,
                current_price=current_prices[symbol],
                entry_time=entry_time,
                current_time=current_time,
                signal_says_sell=signal_says_sell
            )

            decisions[symbol] = decision

            if decision.can_exit:
                logger.info(f"{symbol}: EXIT - {decision.reason.value} - {decision.details}")
            else:
                logger.debug(f"{symbol}: {decision.details}")

        return decisions

    def should_exit(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        entry_time: pd.Timestamp,
        signal_says_sell: bool,
        current_time: Optional[pd.Timestamp] = None
    ) -> ExitDecision:
        """
        Convenience method for checking a single position.

        Args:
            symbol: Trading symbol (for logging)
            entry_price: Price at entry
            current_price: Current price
            entry_time: When position was opened
            signal_says_sell: Whether model recommends selling
            current_time: Current timestamp (default: now)

        Returns:
            ExitDecision
        """
        if current_time is None:
            current_time = pd.Timestamp.now('UTC')

        decision = self.evaluate_exit(
            entry_price=entry_price,
            current_price=current_price,
            entry_time=entry_time,
            current_time=current_time,
            signal_says_sell=signal_says_sell
        )

        if decision.can_exit:
            logger.info(f"{symbol}: {decision.reason.value} - {decision.details}")

        return decision

    def get_config(self) -> Dict:
        """Get current configuration."""
        return {
            'min_hold_hours': self.min_hold_hours,
            'max_hold_hours': self.max_hold_hours,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct
        }

    def update_config(
        self,
        min_hold_hours: Optional[float] = None,
        max_hold_hours: Optional[float] = None,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None
    ) -> None:
        """
        Update configuration parameters.

        Args:
            min_hold_hours: New minimum hold time
            max_hold_hours: New maximum hold time
            stop_loss_pct: New stop-loss percentage
            take_profit_pct: New take-profit percentage
        """
        if min_hold_hours is not None:
            self.min_hold_hours = min_hold_hours
        if max_hold_hours is not None:
            self.max_hold_hours = max_hold_hours
        if stop_loss_pct is not None:
            self.stop_loss_pct = stop_loss_pct
        if take_profit_pct is not None:
            self.take_profit_pct = take_profit_pct

        logger.info(f"ExitManager config updated: {self.get_config()}")
