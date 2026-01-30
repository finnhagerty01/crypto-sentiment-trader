"""
Portfolio-level risk management.

Provides comprehensive portfolio controls:
- Position count limits
- Total exposure limits
- Single position size limits
- Sector concentration limits
- Correlation-based exposure limits
- Trade history tracking
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

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
        """Current market value of the position."""
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        """Original cost of the position."""
        return self.quantity * self.entry_price

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss in USD."""
        return self.value - self.cost_basis

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized profit/loss as percentage."""
        if self.cost_basis == 0:
            return 0.0
        return self.unrealized_pnl / self.cost_basis

    def update_price(self, price: float) -> None:
        """Update the current price."""
        self.current_price = price


@dataclass
class PortfolioState:
    """Current portfolio state."""
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)

    @property
    def total_value(self) -> float:
        """Total portfolio value (cash + positions)."""
        positions_value = sum(p.value for p in self.positions.values())
        return self.cash + positions_value

    @property
    def positions_value(self) -> float:
        """Total value of all positions."""
        return sum(p.value for p in self.positions.values())

    @property
    def exposure(self) -> float:
        """Total market exposure as fraction of portfolio."""
        if self.total_value == 0:
            return 0.0
        return self.positions_value / self.total_value

    @property
    def n_positions(self) -> int:
        """Number of open positions."""
        return len(self.positions)


class PortfolioRiskManager:
    """
    Comprehensive portfolio risk management.

    Handles:
    - Position limits (max count, max single size)
    - Exposure limits (total, per sector)
    - Correlation-based exposure limits
    - Trade history and performance tracking
    """

    # Crypto sector mappings
    DEFAULT_SECTORS = {
        'BTCUSDT': 'store_of_value',
        'ETHUSDT': 'smart_contract',
        'SOLUSDT': 'smart_contract',
        'BNBUSDT': 'exchange',
        'XRPUSDT': 'payments',
        'ADAUSDT': 'smart_contract',
        'AVAXUSDT': 'smart_contract',
        'DOGEUSDT': 'meme',
        'SHIBUSDT': 'meme',
        'LINKUSDT': 'oracle',
        'MATICUSDT': 'layer2',
        'DOTUSDT': 'interoperability',
        'ATOMUSDT': 'interoperability',
        'LTCUSDT': 'store_of_value',
        'TRXUSDT': 'smart_contract',
        'UNIUSDT': 'defi',
        'AAVEUSDT': 'defi',
    }

    def __init__(
        self,
        initial_capital: float,
        max_positions: int = 5,
        max_exposure: float = 0.50,
        max_single_position: float = 0.15,
        max_correlated_exposure: float = 0.30,
        max_sector_exposure: float = 0.30,
        fee_per_side: float = 0.001,
        slippage_per_side: float = 0.0005,
        sectors: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the portfolio risk manager.

        Args:
            initial_capital: Starting capital in USD
            max_positions: Maximum number of simultaneous positions (default 5)
            max_exposure: Maximum total market exposure (default 50%)
            max_single_position: Maximum single position as % of portfolio (default 15%)
            max_correlated_exposure: Max exposure to BTC-correlated assets (default 30%)
            max_sector_exposure: Max exposure to single sector (default 30%)
            fee_per_side: Trading fee per side as decimal (default 0.1%)
            slippage_per_side: Slippage per side as decimal (default 0.05%)
            sectors: Custom sector mappings (uses defaults if not provided)
        """
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.max_exposure = max_exposure
        self.max_single_position = max_single_position
        self.max_correlated_exposure = max_correlated_exposure
        self.max_sector_exposure = max_sector_exposure
        self.fee_per_side = fee_per_side
        self.slippage_per_side = slippage_per_side

        self.sectors = sectors if sectors is not None else self.DEFAULT_SECTORS.copy()

        self.state = PortfolioState(cash=initial_capital)

        # Track history for analytics
        self.value_history: List[Dict] = []
        self.trade_history: List[Dict] = []

        # Track last exit time per symbol for cooldown logic
        self.last_exit: Dict[str, pd.Timestamp] = {}

        # Track total fees paid
        self.total_fees_paid: float = 0.0

    def can_open_position(
        self,
        symbol: str,
        size_usd: float,
        current_prices: Dict[str, float]
    ) -> Tuple[bool, str]:
        """
        Check if a new position can be opened.

        Validates against all risk limits.

        Args:
            symbol: Symbol to trade
            size_usd: Proposed position size in USD
            current_prices: Current prices for all symbols

        Returns:
            Tuple of (can_open: bool, reason: str)
        """
        # Update current prices first
        self._update_prices(current_prices)
        total_value = self.state.total_value

        # 1. Check position count
        if self.state.n_positions >= self.max_positions:
            return False, f"Max positions reached ({self.max_positions})"

        # 2. Check if already have position in this symbol
        if symbol in self.state.positions:
            return False, f"Already have position in {symbol}"

        # 3. Check total exposure
        current_exposure_value = self.state.positions_value
        new_exposure = (current_exposure_value + size_usd) / total_value if total_value > 0 else 0
        if new_exposure > self.max_exposure:
            return False, f"Would exceed max exposure ({new_exposure:.1%} > {self.max_exposure:.1%})"

        # 4. Check single position limit
        position_pct = size_usd / total_value if total_value > 0 else 0
        if position_pct > self.max_single_position:
            return False, f"Position too large ({position_pct:.1%} > {self.max_single_position:.1%})"

        # 5. Check sector exposure
        sector = self.sectors.get(symbol, 'other')
        sector_exposure = self._get_sector_exposure(sector)
        new_sector_exposure = (sector_exposure + size_usd) / total_value if total_value > 0 else 0
        if new_sector_exposure > self.max_sector_exposure:
            return False, f"Would exceed {sector} sector limit ({new_sector_exposure:.1%} > {self.max_sector_exposure:.1%})"

        # 6. Check correlated exposure (if not BTC itself)
        if symbol != 'BTCUSDT':
            correlated_exposure = self._get_btc_correlated_exposure(size_usd)
            correlated_pct = correlated_exposure / total_value if total_value > 0 else 0
            if correlated_pct > self.max_correlated_exposure:
                return False, f"Would exceed BTC-correlated limit ({correlated_pct:.1%} > {self.max_correlated_exposure:.1%})"

        return True, "OK"

    def _update_prices(self, prices: Dict[str, float]) -> None:
        """Update position prices."""
        for symbol, position in self.state.positions.items():
            if symbol in prices:
                position.update_price(prices[symbol])

    def _get_sector_exposure(self, sector: str) -> float:
        """Get current USD exposure to a sector."""
        exposure = 0.0
        for symbol, position in self.state.positions.items():
            if self.sectors.get(symbol) == sector:
                exposure += position.value
        return exposure

    def _get_btc_correlated_exposure(self, new_size: float) -> float:
        """
        Get exposure to BTC-correlated assets.

        Most alts have high correlation with BTC, so we track
        aggregate exposure to avoid concentration risk.
        """
        # Direct BTC exposure
        btc_exposure = 0.0
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

    def open_position(
        self,
        symbol: str,
        notional_usd: float,
        mid_price: float,
        entry_time: Optional[pd.Timestamp] = None
    ) -> bool:
        """
        Open a new position with fee and slippage simulation.

        Args:
            symbol: Trading symbol
            notional_usd: Notional value to buy in USD (before fees)
            mid_price: Mid-market price
            entry_time: Entry timestamp (default: now)

        Returns:
            True if successful, False if insufficient cash
        """
        if notional_usd <= 0:
            return False

        # Apply slippage to get fill price (buy at higher price)
        fill_price = mid_price * (1.0 + self.slippage_per_side)

        # Calculate fee
        fee = notional_usd * self.fee_per_side
        total_cost = notional_usd + fee

        if total_cost > self.state.cash:
            logger.error(f"Insufficient cash for {symbol}: need ${total_cost:.2f}, have ${self.state.cash:.2f}")
            return False

        # Calculate quantity after slippage
        quantity = notional_usd / fill_price

        if entry_time is None:
            entry_time = pd.Timestamp.now('UTC')

        # Deduct cash and record position
        self.state.cash -= total_cost
        self.total_fees_paid += fee

        self.state.positions[symbol] = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=fill_price,
            entry_time=entry_time,
            current_price=fill_price
        )

        self.trade_history.append({
            'timestamp': entry_time,
            'symbol': symbol,
            'action': 'BUY',
            'quantity': quantity,
            'price': fill_price,
            'mid_price': mid_price,
            'notional': notional_usd,
            'fee': fee,
            'total_cost': total_cost
        })

        logger.info(
            f"Opened position: {symbol} {quantity:.6f} @ ${fill_price:.4f} "
            f"(mid=${mid_price:.4f}, fee=${fee:.2f})"
        )
        return True

    def close_position(
        self,
        symbol: str,
        mid_price: float,
        reason: str = 'signal',
        exit_time: Optional[pd.Timestamp] = None
    ) -> Optional[Dict]:
        """
        Close an existing position with fee and slippage simulation.

        Args:
            symbol: Trading symbol
            mid_price: Mid-market exit price
            reason: Reason for closing (for tracking)
            exit_time: Exit timestamp (default: now)

        Returns:
            Trade summary dictionary, or None if no position exists
        """
        if symbol not in self.state.positions:
            logger.warning(f"No position to close for {symbol}")
            return None

        position = self.state.positions[symbol]

        if exit_time is None:
            exit_time = pd.Timestamp.now('UTC')

        # Apply slippage to get fill price (sell at lower price)
        fill_price = mid_price * (1.0 - self.slippage_per_side)

        # Calculate proceeds and fee
        gross_proceeds = position.quantity * fill_price
        fee = gross_proceeds * self.fee_per_side
        net_proceeds = gross_proceeds - fee

        # Calculate P&L (net of fees on both sides)
        pnl = net_proceeds - position.cost_basis
        pnl_pct = pnl / position.cost_basis if position.cost_basis > 0 else 0

        # Update cash and fees
        self.state.cash += net_proceeds
        self.total_fees_paid += fee

        # Record last exit time for cooldown
        self.last_exit[symbol] = exit_time

        # Calculate hold time
        hold_time_hours = (exit_time - position.entry_time).total_seconds() / 3600

        # Record trade
        trade = {
            'timestamp': exit_time,
            'symbol': symbol,
            'action': 'SELL',
            'quantity': position.quantity,
            'price': fill_price,
            'mid_price': mid_price,
            'entry_price': position.entry_price,
            'gross_proceeds': gross_proceeds,
            'fee': fee,
            'net_proceeds': net_proceeds,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'hold_time_hours': hold_time_hours
        }
        self.trade_history.append(trade)

        # Remove position
        del self.state.positions[symbol]

        logger.info(
            f"Closed {symbol}: {position.quantity:.6f} @ ${fill_price:.4f} "
            f"(mid=${mid_price:.4f}), P&L=${pnl:.2f} ({pnl_pct:.1%}), "
            f"Held={hold_time_hours:.1f}h, Reason={reason}"
        )

        return trade

    def is_in_cooldown(
        self,
        symbol: str,
        cooldown_hours: float,
        current_time: Optional[pd.Timestamp] = None
    ) -> Tuple[bool, float]:
        """
        Check if a symbol is in cooldown period after last exit.

        Args:
            symbol: Trading symbol
            cooldown_hours: Required cooldown period in hours
            current_time: Current time (default: now)

        Returns:
            Tuple of (is_in_cooldown: bool, hours_remaining: float)
        """
        if symbol not in self.last_exit:
            return False, 0.0

        if current_time is None:
            current_time = pd.Timestamp.now('UTC')

        last_exit_time = self.last_exit[symbol]

        # Ensure timezone awareness
        if last_exit_time.tzinfo is None:
            last_exit_time = last_exit_time.tz_localize('UTC')
        if current_time.tzinfo is None:
            current_time = current_time.tz_localize('UTC')

        hours_since_exit = (current_time - last_exit_time).total_seconds() / 3600

        if hours_since_exit < cooldown_hours:
            return True, cooldown_hours - hours_since_exit
        return False, 0.0

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get a position by symbol."""
        return self.state.positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        """Check if we have an open position in a symbol."""
        pos = self.state.positions.get(symbol)
        return pos is not None and pos.quantity > 0

    def equity(self, prices: Dict[str, float]) -> float:
        """
        Calculate current equity (total portfolio value).

        Args:
            prices: Current prices for all symbols

        Returns:
            Total equity in USD
        """
        self._update_prices(prices)
        return self.state.total_value

    def record_snapshot(self, prices: Dict[str, float]) -> None:
        """Record a portfolio value snapshot for tracking."""
        self._update_prices(prices)

        self.value_history.append({
            'timestamp': pd.Timestamp.now('UTC'),
            'total_value': self.state.total_value,
            'cash': self.state.cash,
            'positions_value': self.state.positions_value,
            'exposure': self.state.exposure,
            'n_positions': self.state.n_positions,
            'total_fees_paid': self.total_fees_paid
        })

    def get_summary(self, prices: Dict[str, float]) -> Dict:
        """
        Get comprehensive portfolio summary.

        Args:
            prices: Current prices for all symbols

        Returns:
            Portfolio summary dictionary
        """
        self._update_prices(prices)

        # Position details
        positions_summary = {}
        for symbol, pos in self.state.positions.items():
            positions_summary[symbol] = {
                'quantity': pos.quantity,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'value': pos.value,
                'cost_basis': pos.cost_basis,
                'unrealized_pnl': pos.unrealized_pnl,
                'unrealized_pnl_pct': pos.unrealized_pnl_pct,
                'hours_held': (pd.Timestamp.now('UTC') - pos.entry_time).total_seconds() / 3600
            }

        # Calculate aggregate metrics
        total_unrealized_pnl = sum(p.unrealized_pnl for p in self.state.positions.values())
        realized_trades = [t for t in self.trade_history if 'pnl' in t]
        realized_pnl = sum(t['pnl'] for t in realized_trades)

        # Calculate max drawdown from history
        if self.value_history:
            values = pd.Series([h['total_value'] for h in self.value_history])
            peak = values.cummax()
            drawdowns = (peak - values) / peak
            max_drawdown = drawdowns.max()
        else:
            max_drawdown = 0.0

        # Win rate
        if realized_trades:
            wins = sum(1 for t in realized_trades if t['pnl'] > 0)
            win_rate = wins / len(realized_trades)
            avg_win = sum(t['pnl'] for t in realized_trades if t['pnl'] > 0) / max(1, wins)
            losses = len(realized_trades) - wins
            avg_loss = abs(sum(t['pnl'] for t in realized_trades if t['pnl'] < 0)) / max(1, losses)
        else:
            win_rate = 0.0
            avg_win = 0.0
            avg_loss = 0.0

        # Sector exposure
        sector_exposure = {}
        for sector in set(self.sectors.values()):
            exposure = self._get_sector_exposure(sector)
            if exposure > 0:
                sector_exposure[sector] = exposure

        return {
            'total_value': self.state.total_value,
            'cash': self.state.cash,
            'positions_value': self.state.positions_value,
            'exposure': self.state.exposure,
            'n_positions': self.state.n_positions,
            'positions': positions_summary,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_realized_pnl': realized_pnl,
            'total_pnl': total_unrealized_pnl + realized_pnl,
            'total_fees_paid': self.total_fees_paid,
            'return_pct': (self.state.total_value - self.initial_capital) / self.initial_capital if self.initial_capital > 0 else 0,
            'max_drawdown': max_drawdown,
            'n_trades': len(realized_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sector_exposure': sector_exposure
        }

    def get_trade_history(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Get trade history, optionally filtered by symbol.

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            List of trade dictionaries
        """
        if symbol:
            return [t for t in self.trade_history if t['symbol'] == symbol]
        return self.trade_history.copy()

    def get_performance_stats(self) -> Dict:
        """
        Calculate detailed performance statistics.

        Returns:
            Performance statistics dictionary
        """
        realized_trades = [t for t in self.trade_history if 'pnl' in t]

        if not realized_trades:
            return {
                'n_trades': 0,
                'win_rate': 0.0,
                'avg_pnl': 0.0,
                'total_pnl': 0.0,
                'avg_hold_hours': 0.0,
                'best_trade': 0.0,
                'worst_trade': 0.0,
                'profit_factor': 0.0
            }

        pnls = [t['pnl'] for t in realized_trades]
        hold_times = [t.get('hold_time_hours', 0) for t in realized_trades]

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0

        return {
            'n_trades': len(realized_trades),
            'win_rate': len(wins) / len(realized_trades) if realized_trades else 0,
            'avg_pnl': sum(pnls) / len(pnls) if pnls else 0,
            'total_pnl': sum(pnls),
            'avg_hold_hours': sum(hold_times) / len(hold_times) if hold_times else 0,
            'best_trade': max(pnls) if pnls else 0,
            'worst_trade': min(pnls) if pnls else 0,
            'profit_factor': total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0,
            'avg_win': sum(wins) / len(wins) if wins else 0,
            'avg_loss': abs(sum(losses)) / len(losses) if losses else 0
        }

    def reset(self, new_capital: Optional[float] = None) -> None:
        """
        Reset the portfolio to initial state.

        Args:
            new_capital: New starting capital (uses initial if not provided)
        """
        capital = new_capital if new_capital is not None else self.initial_capital
        self.state = PortfolioState(cash=capital)
        self.value_history = []
        self.trade_history = []
        self.last_exit = {}
        self.total_fees_paid = 0.0
        self.initial_capital = capital
        logger.info(f"Portfolio reset with capital=${capital:.2f}")
