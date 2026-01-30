"""
Position sizing algorithms for crypto trading.

Provides multiple methods for calculating optimal position sizes:
- Fixed fractional: Risk fixed % of account per trade
- Volatility-adjusted: Scale inversely with volatility
- Kelly criterion: Mathematically optimal bet sizing
- Signal confidence-weighted: Scale based on model confidence
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PositionSizer:
    """
    Calculate optimal position sizes based on various methods.

    The final position size is the minimum of all applicable methods,
    providing a conservative approach to position sizing.
    """

    def __init__(
        self,
        account_value: float,
        max_position_pct: float = 0.10,
        max_total_exposure: float = 0.50
    ):
        """
        Initialize the position sizer.

        Args:
            account_value: Total account value in USD
            max_position_pct: Maximum % of account for single position (default 10%)
            max_total_exposure: Maximum % of account exposed to market (default 50%)
        """
        self.account_value = account_value
        self.max_position_pct = max_position_pct
        self.max_total_exposure = max_total_exposure

    def update_account_value(self, new_value: float) -> None:
        """Update the account value (e.g., after P&L changes)."""
        self.account_value = new_value

    def fixed_fractional(self, risk_per_trade: float = 0.02) -> float:
        """
        Fixed fractional: Risk fixed % of account per trade.

        This is the simplest position sizing method. It ensures consistent
        risk exposure regardless of the trade setup.

        Args:
            risk_per_trade: Percentage of account to risk (default 2%)

        Returns:
            Position size in USD
        """
        position_size = self.account_value * risk_per_trade
        max_size = self.account_value * self.max_position_pct
        return min(position_size, max_size)

    def volatility_adjusted(
        self,
        current_volatility: float,
        target_volatility: float = 0.02,
        base_position: Optional[float] = None
    ) -> float:
        """
        Adjust position size based on current volatility.

        Higher volatility = smaller position to maintain consistent risk.
        Uses the formula: size = base_size * (target_vol / current_vol)

        Args:
            current_volatility: Current asset volatility (e.g., ATR % or 24h realized vol)
            target_volatility: Target volatility for position (default 2%)
            base_position: Base position size in USD (default: 5% of account)

        Returns:
            Adjusted position size in USD
        """
        if base_position is None:
            base_position = self.account_value * 0.05

        if current_volatility <= 0:
            logger.warning("Invalid volatility (<= 0), using base position")
            return min(base_position, self.account_value * self.max_position_pct)

        # Scale inversely with volatility
        adjustment = target_volatility / current_volatility
        adjusted_size = base_position * adjustment

        # Clamp to reasonable bounds
        min_size = self.account_value * 0.01  # At least 1% of account
        max_size = self.account_value * self.max_position_pct

        return max(min_size, min(adjusted_size, max_size))

    def kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        kelly_fraction: float = 0.25
    ) -> float:
        """
        Kelly Criterion: Mathematically optimal bet sizing.

        Full Kelly is aggressive and can lead to large drawdowns.
        Using fractional Kelly (typically 25%) is recommended.

        Kelly formula: f* = (p * b - q) / b
        where:
            p = win rate
            q = loss rate (1 - p)
            b = win/loss ratio

        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade return (as decimal, e.g., 0.05 for 5%)
            avg_loss: Average losing trade return (positive number, e.g., 0.02 for 2%)
            kelly_fraction: Fraction of Kelly to use (default 0.25 = quarter Kelly)

        Returns:
            Position size in USD
        """
        # Validate inputs
        if not (0 < win_rate < 1):
            logger.warning(f"Invalid win_rate {win_rate}, using conservative sizing")
            return self.fixed_fractional(0.01)

        if avg_loss <= 0 or avg_win <= 0:
            logger.warning("Invalid avg_win or avg_loss, using conservative sizing")
            return self.fixed_fractional(0.01)

        # Kelly formula
        b = avg_win / avg_loss  # Win/loss ratio
        q = 1 - win_rate

        kelly_pct = (win_rate * b - q) / b

        # Negative Kelly means don't bet (expected value is negative)
        if kelly_pct <= 0:
            logger.info("Negative Kelly - strategy has negative expected value")
            return 0.0

        # Apply fraction and limits
        kelly_pct = kelly_pct * kelly_fraction
        kelly_pct = min(kelly_pct, self.max_position_pct)

        return self.account_value * kelly_pct

    def signal_confidence_weighted(
        self,
        confidence: float,
        min_confidence: float = 0.55,
        max_confidence: float = 0.80
    ) -> float:
        """
        Scale position based on model confidence.

        Low confidence = small position, high confidence = full position.
        Below min_confidence, no position is taken.

        Args:
            confidence: Model confidence (0-1)
            min_confidence: Minimum confidence for any position (default 0.55)
            max_confidence: Confidence at which max position is used (default 0.80)

        Returns:
            Position size in USD (0 if below min_confidence)
        """
        if confidence < min_confidence:
            logger.debug(f"Confidence {confidence:.2%} below minimum {min_confidence:.2%}")
            return 0.0

        # Linear scaling between min and max confidence
        confidence_range = max_confidence - min_confidence
        if confidence_range <= 0:
            confidence_pct = 1.0
        else:
            confidence_pct = (confidence - min_confidence) / confidence_range
            confidence_pct = min(confidence_pct, 1.0)  # Cap at 1.0

        # Scale from 25% to 100% of max position
        min_size_pct = 0.25
        size_pct = min_size_pct + (1 - min_size_pct) * confidence_pct

        max_position = self.account_value * self.max_position_pct
        return max_position * size_pct

    def calculate_position(
        self,
        symbol: str,
        price: float,
        confidence: float,
        volatility: float,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
        current_positions: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Calculate final position size combining multiple factors.

        Takes the minimum of all applicable sizing methods for a
        conservative approach, then applies exposure limits.

        Args:
            symbol: Trading symbol
            price: Current price
            confidence: Model confidence (0-1)
            volatility: Current volatility (e.g., ATR % as decimal)
            win_rate: Historical win rate (optional, for Kelly)
            avg_win: Average win return (optional, for Kelly)
            avg_loss: Average loss return (optional, for Kelly)
            current_positions: Dict of current positions {symbol: value_usd}

        Returns:
            Dictionary with position details:
                - symbol: Trading symbol
                - size_usd: Final position size in USD
                - quantity: Position size in asset units
                - price: Entry price used
                - pct_of_account: Position as % of account
                - method_sizes: Dict of sizes from each method
                - limiting_factor: Which method produced the smallest size
                - reason: Reason if size is 0
        """
        # Check total exposure capacity
        if current_positions:
            current_exposure = sum(current_positions.values())
            remaining_capacity = (self.account_value * self.max_total_exposure) - current_exposure

            if remaining_capacity <= 0:
                logger.warning(f"Max exposure reached. No new positions allowed.")
                return {
                    'symbol': symbol,
                    'size_usd': 0.0,
                    'quantity': 0.0,
                    'price': price,
                    'pct_of_account': 0.0,
                    'method_sizes': {},
                    'limiting_factor': 'max_exposure',
                    'reason': 'max_exposure_reached'
                }
        else:
            remaining_capacity = self.account_value * self.max_total_exposure

        # Calculate sizes from different methods
        sizes = []

        # 1. Confidence-weighted (primary filter)
        confidence_size = self.signal_confidence_weighted(confidence)
        if confidence_size <= 0:
            return {
                'symbol': symbol,
                'size_usd': 0.0,
                'quantity': 0.0,
                'price': price,
                'pct_of_account': 0.0,
                'method_sizes': {'confidence': 0.0},
                'limiting_factor': 'confidence',
                'reason': 'confidence_too_low'
            }
        sizes.append(('confidence', confidence_size))

        # 2. Volatility-adjusted
        if volatility > 0:
            vol_size = self.volatility_adjusted(volatility)
            sizes.append(('volatility', vol_size))

        # 3. Kelly criterion (if we have the data)
        if all([win_rate, avg_win, avg_loss]):
            kelly_size = self.kelly_criterion(win_rate, avg_win, avg_loss)
            if kelly_size > 0:
                sizes.append(('kelly', kelly_size))

        # Take minimum of all methods (conservative)
        final_size_usd = min(s[1] for s in sizes)

        # Apply remaining capacity limit
        final_size_usd = min(final_size_usd, remaining_capacity)

        # Calculate quantity
        quantity = final_size_usd / price if price > 0 else 0.0

        # Find limiting factor
        limiting = min(sizes, key=lambda x: x[1])

        return {
            'symbol': symbol,
            'size_usd': final_size_usd,
            'quantity': quantity,
            'price': price,
            'pct_of_account': final_size_usd / self.account_value if self.account_value > 0 else 0.0,
            'method_sizes': dict(sizes),
            'limiting_factor': limiting[0],
            'reason': None
        }


class RiskBudget:
    """
    Portfolio-level risk budgeting.

    Tracks daily P&L and drawdown, halting trading when limits are hit.
    This provides a circuit breaker to prevent catastrophic losses.
    """

    def __init__(
        self,
        account_value: float,
        max_daily_loss: float = 0.05,
        max_drawdown: float = 0.15
    ):
        """
        Initialize risk budget tracker.

        Args:
            account_value: Total account value
            max_daily_loss: Maximum daily loss as % of account (default 5%)
            max_drawdown: Maximum drawdown from peak (default 15%)
        """
        self.initial_account_value = account_value
        self.account_value = account_value
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown

        self.daily_pnl = 0.0
        self.peak_value = account_value
        self.is_trading_halted = False
        self._halt_reasons = []

    def update(self, current_value: float) -> Dict:
        """
        Update risk metrics and check if trading should halt.

        Args:
            current_value: Current portfolio value

        Returns:
            Risk status dictionary with:
                - current_value: Current portfolio value
                - peak_value: Highest value seen
                - drawdown: Current drawdown from peak
                - daily_pnl: Today's P&L
                - is_halted: Whether trading is halted
                - halt_reasons: List of reasons for halt
        """
        self.account_value = current_value

        # Update peak
        if current_value > self.peak_value:
            self.peak_value = current_value

        # Calculate drawdown from peak
        drawdown = (self.peak_value - current_value) / self.peak_value if self.peak_value > 0 else 0.0

        # Calculate daily loss percentage
        daily_loss_pct = -self.daily_pnl / self.initial_account_value if self.daily_pnl < 0 else 0.0

        # Check limits
        halt_reasons = []

        if daily_loss_pct >= self.max_daily_loss:
            halt_reasons.append(f"Daily loss limit hit: {daily_loss_pct:.1%} >= {self.max_daily_loss:.1%}")

        if drawdown >= self.max_drawdown:
            halt_reasons.append(f"Max drawdown hit: {drawdown:.1%} >= {self.max_drawdown:.1%}")

        self._halt_reasons = halt_reasons
        self.is_trading_halted = len(halt_reasons) > 0

        if self.is_trading_halted:
            for reason in halt_reasons:
                logger.warning(f"TRADING HALTED: {reason}")

        return {
            'current_value': current_value,
            'peak_value': self.peak_value,
            'drawdown': drawdown,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': self.daily_pnl / self.initial_account_value if self.initial_account_value > 0 else 0.0,
            'is_halted': self.is_trading_halted,
            'halt_reasons': halt_reasons
        }

    def record_trade(self, pnl: float) -> None:
        """
        Record a trade's P&L.

        Args:
            pnl: Trade P&L in USD (positive for profit, negative for loss)
        """
        self.daily_pnl += pnl
        logger.debug(f"Trade P&L recorded: ${pnl:.2f}, Daily total: ${self.daily_pnl:.2f}")

    def reset_daily(self) -> None:
        """Reset daily P&L tracking (call at start of each day)."""
        logger.info(f"Resetting daily P&L. Previous: ${self.daily_pnl:.2f}")
        self.daily_pnl = 0.0

        # Also reset halt if it was due to daily loss (but not drawdown)
        if self.is_trading_halted and not any("drawdown" in r.lower() for r in self._halt_reasons):
            self.is_trading_halted = False
            self._halt_reasons = []
            logger.info("Daily loss halt lifted")

    def can_trade(self) -> bool:
        """Check if trading is allowed."""
        return not self.is_trading_halted

    def get_status(self) -> Dict:
        """Get current risk budget status."""
        drawdown = (self.peak_value - self.account_value) / self.peak_value if self.peak_value > 0 else 0.0

        return {
            'account_value': self.account_value,
            'initial_value': self.initial_account_value,
            'peak_value': self.peak_value,
            'drawdown': drawdown,
            'drawdown_remaining': self.max_drawdown - drawdown,
            'daily_pnl': self.daily_pnl,
            'daily_loss_remaining': (self.initial_account_value * self.max_daily_loss) + self.daily_pnl,
            'is_halted': self.is_trading_halted,
            'halt_reasons': self._halt_reasons
        }
