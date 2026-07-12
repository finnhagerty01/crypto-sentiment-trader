"""
Unit tests for risk management module.

Tests:
- Position sizing algorithms
- Stop-loss management
- Exit management with hold time
- Portfolio risk controls
"""

import pytest
import pandas as pd
from datetime import timedelta

from src.risk.position_sizing import PositionSizer, RiskBudget
from src.risk.stop_loss import StopLossManager, StopType, StopLossOrder
from src.risk.exit_manager import ExitManager, ExitReason, ExitDecision
from src.risk.portfolio import PortfolioRiskManager, Position, PortfolioState


# ============================================================================
# PositionSizer Tests
# ============================================================================

class TestPositionSizer:
    """Tests for PositionSizer class."""

    def test_initialization(self):
        """Test basic initialization."""
        sizer = PositionSizer(account_value=10000, max_position_pct=0.10)
        assert sizer.account_value == 10000
        assert sizer.max_position_pct == 0.10
        assert sizer.max_total_exposure == 0.50

    def test_fixed_fractional_basic(self):
        """Test fixed fractional position sizing."""
        sizer = PositionSizer(account_value=10000, max_position_pct=0.10)

        # 2% risk on $10000 = $200
        size = sizer.fixed_fractional(risk_per_trade=0.02)
        assert size == 200.0

    def test_fixed_fractional_respects_max(self):
        """Test that fixed fractional respects max position limit."""
        sizer = PositionSizer(account_value=10000, max_position_pct=0.05)

        # 10% risk would be $1000, but max is 5% = $500
        size = sizer.fixed_fractional(risk_per_trade=0.10)
        assert size == 500.0

    def test_volatility_adjusted_normal(self):
        """Test volatility-adjusted sizing with normal volatility."""
        sizer = PositionSizer(account_value=10000)

        # Target 2%, current 2% -> no adjustment
        size = sizer.volatility_adjusted(current_volatility=0.02, target_volatility=0.02)
        assert size == 500.0  # 5% base position

    def test_volatility_adjusted_high_vol(self):
        """Test volatility-adjusted sizing with high volatility."""
        sizer = PositionSizer(account_value=10000)

        # Target 2%, current 4% -> halve the position
        size = sizer.volatility_adjusted(current_volatility=0.04, target_volatility=0.02)
        assert size == 250.0

    def test_volatility_adjusted_low_vol(self):
        """Test volatility-adjusted sizing with low volatility."""
        sizer = PositionSizer(account_value=10000, max_position_pct=0.10)

        # Target 2%, current 1% -> double the position (capped at max)
        size = sizer.volatility_adjusted(current_volatility=0.01, target_volatility=0.02)
        assert size == 1000.0  # Capped at 10% max

    def test_volatility_adjusted_zero_vol(self):
        """Test volatility-adjusted sizing with zero/invalid volatility."""
        sizer = PositionSizer(account_value=10000, max_position_pct=0.10)

        size = sizer.volatility_adjusted(current_volatility=0)
        assert size == 500.0  # Falls back to base position

    def test_kelly_criterion_positive_edge(self):
        """Test Kelly criterion with positive edge."""
        sizer = PositionSizer(account_value=10000, max_position_pct=0.20)

        # 60% win rate, 1:1 risk/reward -> Kelly = 0.20 (20%)
        # Quarter Kelly = 5%
        size = sizer.kelly_criterion(
            win_rate=0.60,
            avg_win=0.03,
            avg_loss=0.03,
            kelly_fraction=0.25
        )
        assert size > 0

    def test_kelly_criterion_negative_edge(self):
        """Test Kelly criterion with negative expected value."""
        sizer = PositionSizer(account_value=10000)

        # 30% win rate with 1:1 ratio -> negative edge
        size = sizer.kelly_criterion(
            win_rate=0.30,
            avg_win=0.02,
            avg_loss=0.02
        )
        assert size == 0.0

    def test_kelly_criterion_invalid_inputs(self):
        """Test Kelly criterion with invalid inputs."""
        sizer = PositionSizer(account_value=10000)

        # Invalid win rate
        size = sizer.kelly_criterion(win_rate=1.5, avg_win=0.02, avg_loss=0.02)
        assert size == sizer.fixed_fractional(0.01)

        # Invalid avg_loss
        size = sizer.kelly_criterion(win_rate=0.5, avg_win=0.02, avg_loss=0)
        assert size == sizer.fixed_fractional(0.01)

    def test_signal_confidence_weighted_below_min(self):
        """Test confidence-weighted sizing below minimum confidence."""
        sizer = PositionSizer(account_value=10000)

        size = sizer.signal_confidence_weighted(confidence=0.50, min_confidence=0.55)
        assert size == 0.0

    def test_signal_confidence_weighted_at_min(self):
        """Test confidence-weighted sizing at minimum confidence."""
        sizer = PositionSizer(account_value=10000, max_position_pct=0.10)

        # At min confidence, get 25% of max position
        size = sizer.signal_confidence_weighted(confidence=0.55, min_confidence=0.55)
        assert size == 250.0  # 25% of $1000

    def test_signal_confidence_weighted_at_max(self):
        """Test confidence-weighted sizing at maximum confidence."""
        sizer = PositionSizer(account_value=10000, max_position_pct=0.10)

        # At max confidence, get 100% of max position
        size = sizer.signal_confidence_weighted(confidence=0.80, max_confidence=0.80)
        assert size == 1000.0

    def test_calculate_position_comprehensive(self):
        """Test comprehensive position calculation."""
        sizer = PositionSizer(account_value=10000, max_position_pct=0.10)

        result = sizer.calculate_position(
            symbol='BTCUSDT',
            price=50000.0,
            confidence=0.70,
            volatility=0.03,
            current_positions={}
        )

        assert result['symbol'] == 'BTCUSDT'
        assert result['size_usd'] > 0
        assert result['quantity'] > 0
        assert result['price'] == 50000.0
        assert 'limiting_factor' in result
        assert 'method_sizes' in result

    def test_calculate_position_max_exposure(self):
        """Test position calculation respects max exposure."""
        sizer = PositionSizer(account_value=10000, max_total_exposure=0.50)

        # Already have 50% exposure
        result = sizer.calculate_position(
            symbol='ETHUSDT',
            price=3000.0,
            confidence=0.70,
            volatility=0.03,
            current_positions={'BTCUSDT': 5000.0}
        )

        assert result['size_usd'] == 0.0
        assert result['reason'] == 'max_exposure_reached'

    def test_calculate_position_low_confidence(self):
        """Test position calculation rejects low confidence."""
        sizer = PositionSizer(account_value=10000)

        result = sizer.calculate_position(
            symbol='BTCUSDT',
            price=50000.0,
            confidence=0.40,
            volatility=0.03
        )

        assert result['size_usd'] == 0.0
        assert result['reason'] == 'confidence_too_low'


# ============================================================================
# RiskBudget Tests
# ============================================================================

class TestRiskBudget:
    """Tests for RiskBudget class."""

    def test_initialization(self):
        """Test basic initialization."""
        budget = RiskBudget(account_value=10000, max_daily_loss=0.05, max_drawdown=0.15)
        assert budget.account_value == 10000
        assert budget.max_daily_loss == 0.05
        assert budget.max_drawdown == 0.15
        assert not budget.is_trading_halted

    def test_record_trade_positive(self):
        """Test recording positive P&L."""
        budget = RiskBudget(account_value=10000)

        budget.record_trade(pnl=100)
        assert budget.daily_pnl == 100

    def test_record_trade_negative(self):
        """Test recording negative P&L."""
        budget = RiskBudget(account_value=10000)

        budget.record_trade(pnl=-100)
        assert budget.daily_pnl == -100

    def test_daily_loss_halt(self):
        """Test trading halts on daily loss limit."""
        budget = RiskBudget(account_value=10000, max_daily_loss=0.05)

        # Lose 5% = $500
        budget.record_trade(pnl=-500)
        status = budget.update(current_value=9500)

        assert budget.is_trading_halted
        assert 'daily loss' in status['halt_reasons'][0].lower()

    def test_drawdown_halt(self):
        """Test trading halts on max drawdown."""
        budget = RiskBudget(account_value=10000, max_drawdown=0.15)

        # Peak was 10000, now at 8500 = 15% drawdown
        status = budget.update(current_value=8500)

        assert budget.is_trading_halted
        assert 'drawdown' in status['halt_reasons'][0].lower()

    def test_peak_tracking(self):
        """Test peak value tracking."""
        budget = RiskBudget(account_value=10000)

        budget.update(current_value=11000)
        assert budget.peak_value == 11000

        budget.update(current_value=10500)
        assert budget.peak_value == 11000  # Peak shouldn't decrease

    def test_reset_daily(self):
        """Test daily reset."""
        budget = RiskBudget(account_value=10000, max_daily_loss=0.05)

        budget.record_trade(pnl=-400)
        budget.update(current_value=9600)
        assert budget.daily_pnl == -400

        budget.reset_daily()
        assert budget.daily_pnl == 0
        assert not budget.is_trading_halted  # Daily halt should be lifted

    def test_can_trade(self):
        """Test can_trade convenience method."""
        budget = RiskBudget(account_value=10000, max_daily_loss=0.05)

        assert budget.can_trade()

        budget.record_trade(pnl=-600)
        budget.update(current_value=9400)

        assert not budget.can_trade()


# ============================================================================
# StopLossManager Tests
# ============================================================================

class TestStopLossManager:
    """Tests for StopLossManager class."""

    def test_initialization(self):
        """Test basic initialization."""
        manager = StopLossManager(default_stop_pct=0.02)
        assert manager.default_stop_pct == 0.02
        assert len(manager.active_stops) == 0

    def test_create_fixed_stop(self):
        """Test creating fixed percentage stop."""
        manager = StopLossManager(default_stop_pct=0.02, default_take_profit_pct=0.04)

        stop = manager.create_fixed_stop(
            symbol='BTCUSDT',
            entry_price=50000,
            quantity=0.1
        )

        assert stop.symbol == 'BTCUSDT'
        assert stop.entry_price == 50000
        assert stop.stop_price == 49000  # 50000 * (1 - 0.02)
        assert stop.take_profit_price == 52000  # 50000 * (1 + 0.04)
        assert stop.stop_type == StopType.FIXED
        assert 'BTCUSDT' in manager.active_stops

    def test_create_atr_stop(self):
        """Test creating ATR-based stop."""
        manager = StopLossManager(atr_multiplier=2.0)

        stop = manager.create_atr_stop(
            symbol='BTCUSDT',
            entry_price=50000,
            quantity=0.1,
            atr=500  # $500 ATR
        )

        assert stop.stop_price == 49000  # 50000 - (500 * 2)
        assert stop.take_profit_price == 52000  # 50000 + (500 * 2 * 2)
        assert stop.stop_type == StopType.ATR

    def test_create_time_stop(self):
        """Test creating time-based stop."""
        manager = StopLossManager()

        stop = manager.create_time_stop(
            symbol='BTCUSDT',
            entry_price=50000,
            quantity=0.1,
            max_hold_hours=24
        )

        assert stop.max_hold_hours == 24
        assert stop.stop_type == StopType.TIME

    def test_check_stops_stop_loss_triggered(self):
        """Test stop-loss trigger detection."""
        manager = StopLossManager(default_stop_pct=0.02)
        manager.create_fixed_stop('BTCUSDT', entry_price=50000, quantity=0.1)

        actions = manager.check_stops({'BTCUSDT': 48000})

        assert actions['BTCUSDT'] == 'stop_loss'

    def test_check_stops_take_profit_triggered(self):
        """Test take-profit trigger detection."""
        manager = StopLossManager(default_take_profit_pct=0.04)
        manager.create_fixed_stop('BTCUSDT', entry_price=50000, quantity=0.1)

        actions = manager.check_stops({'BTCUSDT': 53000})

        assert actions['BTCUSDT'] == 'take_profit'

    def test_check_stops_time_stop_triggered(self):
        """Test time stop trigger detection."""
        manager = StopLossManager()
        manager.create_time_stop('BTCUSDT', entry_price=50000, quantity=0.1, max_hold_hours=1)

        # Manually set entry time to 2 hours ago
        manager.active_stops['BTCUSDT'].entry_time = pd.Timestamp.now('UTC') - timedelta(hours=2)

        actions = manager.check_stops({'BTCUSDT': 50000})

        assert actions['BTCUSDT'] == 'time_stop'

    def test_trailing_stop_update(self):
        """Test trailing stop updates."""
        manager = StopLossManager(
            default_stop_pct=0.02,
            trailing_activation_pct=0.01,
            trailing_distance_pct=0.02
        )
        manager.create_fixed_stop('BTCUSDT', entry_price=50000, quantity=0.1)

        # Price rises to 51000 (2% profit) - should activate trailing
        new_stop = manager.update_trailing_stop('BTCUSDT', current_price=51000)

        assert new_stop is not None
        assert new_stop == 51000 * 0.98  # 2% trailing distance
        assert manager.active_stops['BTCUSDT'].stop_type == StopType.TRAILING

    def test_trailing_stop_only_moves_up(self):
        """Test trailing stop only moves up, never down."""
        manager = StopLossManager(trailing_activation_pct=0.01)
        manager.create_fixed_stop('BTCUSDT', entry_price=50000, quantity=0.1)

        # Activate trailing at 51000
        manager.update_trailing_stop('BTCUSDT', current_price=51000)
        stop1 = manager.active_stops['BTCUSDT'].stop_price

        # Price drops - stop should not move down
        manager.update_trailing_stop('BTCUSDT', current_price=50500)
        stop2 = manager.active_stops['BTCUSDT'].stop_price

        assert stop2 == stop1  # Stop didn't move down

    def test_remove_stop(self):
        """Test removing a stop."""
        manager = StopLossManager()
        manager.create_fixed_stop('BTCUSDT', entry_price=50000, quantity=0.1)

        assert 'BTCUSDT' in manager.active_stops

        removed = manager.remove_stop('BTCUSDT')

        assert removed is not None
        assert 'BTCUSDT' not in manager.active_stops


# ============================================================================
# ExitManager Tests
# ============================================================================

class TestExitManager:
    """Tests for ExitManager class."""

    def test_initialization(self):
        """Test basic initialization."""
        manager = ExitManager(min_hold_hours=2.0, stop_loss_pct=0.03)
        assert manager.min_hold_hours == 2.0
        assert manager.stop_loss_pct == 0.03

    def test_stop_loss_overrides_hold_time(self):
        """Test that stop-loss exits immediately regardless of hold time."""
        manager = ExitManager(min_hold_hours=2.0, stop_loss_pct=0.03)

        entry_time = pd.Timestamp.now('UTC')
        current_time = entry_time + timedelta(minutes=20)  # Only 20 minutes

        decision = manager.evaluate_exit(
            entry_price=50000,
            current_price=48000,  # 4% loss
            entry_time=entry_time,
            current_time=current_time,
            signal_says_sell=False
        )

        assert decision.can_exit
        assert decision.reason == ExitReason.STOP_LOSS

    def test_take_profit_overrides_hold_time(self):
        """Test that take-profit exits immediately regardless of hold time."""
        manager = ExitManager(min_hold_hours=2.0, take_profit_pct=0.05)

        entry_time = pd.Timestamp.now('UTC')
        current_time = entry_time + timedelta(minutes=20)  # Only 20 minutes

        decision = manager.evaluate_exit(
            entry_price=50000,
            current_price=53000,  # 6% profit
            entry_time=entry_time,
            current_time=current_time,
            signal_says_sell=False
        )

        assert decision.can_exit
        assert decision.reason == ExitReason.TAKE_PROFIT

    def test_signal_blocked_before_min_hold(self):
        """Test that signal-based exits are blocked before minimum hold."""
        manager = ExitManager(min_hold_hours=2.0)

        entry_time = pd.Timestamp.now('UTC')
        current_time = entry_time + timedelta(hours=1)  # 1 hour < 2 hour min

        decision = manager.evaluate_exit(
            entry_price=50000,
            current_price=50500,  # Small profit, no stop triggered
            entry_time=entry_time,
            current_time=current_time,
            signal_says_sell=True
        )

        assert not decision.can_exit
        assert decision.reason == ExitReason.HOLD

    def test_signal_allowed_after_min_hold(self):
        """Test that signal-based exits are allowed after minimum hold."""
        manager = ExitManager(min_hold_hours=2.0)

        entry_time = pd.Timestamp.now('UTC')
        current_time = entry_time + timedelta(hours=3)  # 3 hours > 2 hour min

        decision = manager.evaluate_exit(
            entry_price=50000,
            current_price=50500,  # Small profit, no stop triggered
            entry_time=entry_time,
            current_time=current_time,
            signal_says_sell=True
        )

        assert decision.can_exit
        assert decision.reason == ExitReason.SIGNAL

    def test_max_hold_time_forces_exit(self):
        """Test that max hold time forces exit."""
        manager = ExitManager(max_hold_hours=48.0)

        entry_time = pd.Timestamp.now('UTC')
        current_time = entry_time + timedelta(hours=50)  # 50 hours > 48 max

        decision = manager.evaluate_exit(
            entry_price=50000,
            current_price=50500,
            entry_time=entry_time,
            current_time=current_time,
            signal_says_sell=False  # Signal says hold
        )

        assert decision.can_exit
        assert decision.reason == ExitReason.TIME_LIMIT

    def test_hold_when_no_exit_condition(self):
        """Test hold decision when no exit condition met."""
        manager = ExitManager(min_hold_hours=2.0, stop_loss_pct=0.05, take_profit_pct=0.10)

        entry_time = pd.Timestamp.now('UTC')
        current_time = entry_time + timedelta(hours=1)

        decision = manager.evaluate_exit(
            entry_price=50000,
            current_price=50500,  # 1% profit - no thresholds hit
            entry_time=entry_time,
            current_time=current_time,
            signal_says_sell=False
        )

        assert not decision.can_exit
        assert decision.reason == ExitReason.HOLD

    def test_check_all_positions(self):
        """Test checking multiple positions at once."""
        manager = ExitManager(min_hold_hours=2.0, stop_loss_pct=0.05)

        now = pd.Timestamp.now('UTC')

        positions = {
            'BTCUSDT': {'entry_price': 50000, 'entry_time': now - timedelta(hours=3)},
            'ETHUSDT': {'entry_price': 3000, 'entry_time': now - timedelta(hours=1)},
        }

        prices = {'BTCUSDT': 50500, 'ETHUSDT': 3100}
        signals = {'BTCUSDT': 'SELL', 'ETHUSDT': 'SELL'}

        decisions = manager.check_all_positions(positions, prices, signals, now)

        # BTC: past min hold, signal says sell -> exit allowed
        assert decisions['BTCUSDT'].can_exit
        assert decisions['BTCUSDT'].reason == ExitReason.SIGNAL

        # ETH: before min hold, signal says sell but blocked
        assert not decisions['ETHUSDT'].can_exit
        assert decisions['ETHUSDT'].reason == ExitReason.HOLD


# ============================================================================
# PortfolioRiskManager Tests
# ============================================================================

class TestPortfolioRiskManager:
    """Tests for PortfolioRiskManager class."""

    def test_initialization(self):
        """Test basic initialization."""
        manager = PortfolioRiskManager(initial_capital=10000, max_positions=5)
        assert manager.initial_capital == 10000
        assert manager.max_positions == 5
        assert manager.state.cash == 10000
        assert manager.state.n_positions == 0

    def test_can_open_position_success(self):
        """Test successful position opening check."""
        manager = PortfolioRiskManager(initial_capital=10000)

        can_open, reason = manager.can_open_position(
            symbol='BTCUSDT',
            size_usd=1000,
            current_prices={'BTCUSDT': 50000}
        )

        assert can_open
        assert reason == "OK"

    def test_can_open_position_max_positions(self):
        """Test position count limit."""
        manager = PortfolioRiskManager(initial_capital=10000, max_positions=2, fee_per_side=0, slippage_per_side=0)

        # Open two positions (notional_usd, mid_price)
        manager.open_position('BTCUSDT', notional_usd=500, mid_price=50000)
        manager.open_position('ETHUSDT', notional_usd=300, mid_price=3000)

        can_open, reason = manager.can_open_position(
            symbol='SOLUSDT',
            size_usd=500,
            current_prices={'BTCUSDT': 50000, 'ETHUSDT': 3000, 'SOLUSDT': 100}
        )

        assert not can_open
        assert "max positions" in reason.lower()

    def test_can_open_position_duplicate(self):
        """Test blocking duplicate positions."""
        manager = PortfolioRiskManager(initial_capital=10000, fee_per_side=0, slippage_per_side=0)

        manager.open_position('BTCUSDT', notional_usd=500, mid_price=50000)

        can_open, reason = manager.can_open_position(
            symbol='BTCUSDT',
            size_usd=500,
            current_prices={'BTCUSDT': 50000}
        )

        assert not can_open
        assert "already have position" in reason.lower()

    def test_can_open_position_max_exposure(self):
        """Test total exposure limit."""
        manager = PortfolioRiskManager(initial_capital=10000, max_exposure=0.30, fee_per_side=0, slippage_per_side=0)

        manager.open_position('BTCUSDT', notional_usd=2500, mid_price=50000)  # $2500 = 25%

        can_open, reason = manager.can_open_position(
            symbol='ETHUSDT',
            size_usd=1000,  # Would be 35% total > 30%
            current_prices={'BTCUSDT': 50000, 'ETHUSDT': 3000}
        )

        assert not can_open
        assert "max exposure" in reason.lower()

    def test_can_open_position_max_single(self):
        """Test single position size limit."""
        manager = PortfolioRiskManager(initial_capital=10000, max_single_position=0.10)

        can_open, reason = manager.can_open_position(
            symbol='BTCUSDT',
            size_usd=1500,  # 15% > 10% max
            current_prices={'BTCUSDT': 50000}
        )

        assert not can_open
        assert "too large" in reason.lower()

    def test_open_position(self):
        """Test opening a position."""
        manager = PortfolioRiskManager(initial_capital=10000, fee_per_side=0, slippage_per_side=0)

        success = manager.open_position('BTCUSDT', notional_usd=1000, mid_price=50000)

        assert success
        assert 'BTCUSDT' in manager.state.positions
        assert manager.state.positions['BTCUSDT'].quantity == 0.02  # 1000 / 50000
        assert manager.state.cash == 9000  # 10000 - 1000

    def test_open_position_insufficient_cash(self):
        """Test opening position with insufficient cash."""
        manager = PortfolioRiskManager(initial_capital=1000, fee_per_side=0, slippage_per_side=0)

        success = manager.open_position('BTCUSDT', notional_usd=5000, mid_price=50000)  # $5000

        assert not success
        assert 'BTCUSDT' not in manager.state.positions
        assert manager.state.cash == 1000

    def test_close_position(self):
        """Test closing a position."""
        manager = PortfolioRiskManager(initial_capital=10000, fee_per_side=0, slippage_per_side=0)

        manager.open_position('BTCUSDT', notional_usd=1000, mid_price=50000)  # Cost: $1000
        trade = manager.close_position('BTCUSDT', mid_price=52000, reason='take_profit')  # Proceeds: $1040

        assert trade is not None
        assert trade['pnl'] == 40  # $1040 - $1000
        assert 'BTCUSDT' not in manager.state.positions
        assert manager.state.cash == 10040

    def test_close_position_not_found(self):
        """Test closing non-existent position."""
        manager = PortfolioRiskManager(initial_capital=10000)

        trade = manager.close_position('BTCUSDT', mid_price=50000)

        assert trade is None

    def test_get_summary(self):
        """Test portfolio summary."""
        manager = PortfolioRiskManager(initial_capital=10000, fee_per_side=0, slippage_per_side=0)

        manager.open_position('BTCUSDT', notional_usd=1000, mid_price=50000)  # qty = 0.02

        summary = manager.get_summary({'BTCUSDT': 52000})

        assert 'total_value' in summary
        assert 'positions' in summary
        assert 'BTCUSDT' in summary['positions']
        assert summary['positions']['BTCUSDT']['unrealized_pnl'] == 40  # 0.02 * (52000 - 50000)

    def test_sector_exposure(self):
        """Test sector exposure tracking."""
        manager = PortfolioRiskManager(initial_capital=10000, max_sector_exposure=0.20, fee_per_side=0, slippage_per_side=0)

        # ETHUSDT and SOLUSDT are both 'smart_contract' sector
        manager.open_position('ETHUSDT', notional_usd=1500, mid_price=3000)  # $1500 = 15%

        can_open, reason = manager.can_open_position(
            symbol='SOLUSDT',
            size_usd=1000,  # Would be 25% smart_contract > 20%
            current_prices={'ETHUSDT': 3000, 'SOLUSDT': 100}
        )

        assert not can_open
        assert "sector" in reason.lower()

    def test_position_properties(self):
        """Test Position dataclass properties."""
        pos = Position(
            symbol='BTCUSDT',
            quantity=0.1,
            entry_price=50000,
            entry_time=pd.Timestamp.now('UTC'),
            current_price=52000
        )

        assert pos.cost_basis == 5000
        assert pos.value == 5200
        assert pos.unrealized_pnl == 200
        assert pos.unrealized_pnl_pct == 0.04

    def test_portfolio_state_properties(self):
        """Test PortfolioState properties."""
        state = PortfolioState(cash=5000)
        state.positions['BTCUSDT'] = Position(
            symbol='BTCUSDT',
            quantity=0.1,
            entry_price=50000,
            entry_time=pd.Timestamp.now('UTC'),
            current_price=50000
        )

        assert state.total_value == 10000
        assert state.positions_value == 5000
        assert state.exposure == 0.5
        assert state.n_positions == 1

    def test_cooldown_tracking(self):
        """Test cooldown tracking after position close."""
        manager = PortfolioRiskManager(initial_capital=10000, fee_per_side=0, slippage_per_side=0)

        # Open and close a position
        manager.open_position('BTCUSDT', notional_usd=1000, mid_price=50000)
        manager.close_position('BTCUSDT', mid_price=51000)

        # Check cooldown
        in_cooldown, hours_remaining = manager.is_in_cooldown('BTCUSDT', cooldown_hours=4)
        assert in_cooldown
        assert hours_remaining > 3.9  # Should be close to 4 hours

    def test_fees_and_slippage(self):
        """Test that fees and slippage are applied correctly."""
        manager = PortfolioRiskManager(
            initial_capital=10000,
            fee_per_side=0.001,  # 0.1%
            slippage_per_side=0.0005  # 0.05%
        )

        # Open position - should pay slippage + fee
        manager.open_position('BTCUSDT', notional_usd=1000, mid_price=50000)

        # Cash should be less than 9000 due to fee
        assert manager.state.cash < 9000
        assert manager.total_fees_paid > 0

        # Close position
        manager.close_position('BTCUSDT', mid_price=50000)  # Same price

        # Should have less than original due to fees + slippage on both sides
        assert manager.state.total_value < 10000


# ============================================================================
# Integration Tests
# ============================================================================

class TestRiskManagementIntegration:
    """Integration tests for risk management components working together."""

    def test_full_trade_lifecycle(self):
        """Test complete trade lifecycle with all risk components."""
        # Initialize components
        sizer = PositionSizer(account_value=10000, max_position_pct=0.10)
        stop_manager = StopLossManager(default_stop_pct=0.02, default_take_profit_pct=0.04)
        exit_manager = ExitManager(min_hold_hours=2.0, stop_loss_pct=0.03, take_profit_pct=0.05)
        portfolio = PortfolioRiskManager(initial_capital=10000, fee_per_side=0, slippage_per_side=0)

        # Calculate position
        position = sizer.calculate_position(
            symbol='BTCUSDT',
            price=50000,
            confidence=0.70,
            volatility=0.02
        )

        assert position['size_usd'] > 0

        # Check portfolio limits
        can_open, _ = portfolio.can_open_position(
            'BTCUSDT',
            position['size_usd'],
            {'BTCUSDT': 50000}
        )

        assert can_open

        # Open position using new API
        portfolio.open_position('BTCUSDT', notional_usd=position['size_usd'], mid_price=50000)
        quantity = portfolio.state.positions['BTCUSDT'].quantity

        # Create stop
        stop_manager.create_fixed_stop('BTCUSDT', 50000, quantity)

        # Simulate price movement
        new_price = 49000  # 2% drop

        # Check stops
        stop_actions = stop_manager.check_stops({'BTCUSDT': new_price})

        # Check exit manager
        pos = portfolio.state.positions['BTCUSDT']
        exit_decision = exit_manager.should_exit(
            symbol='BTCUSDT',
            entry_price=pos.entry_price,
            current_price=new_price,
            entry_time=pos.entry_time,
            signal_says_sell=False
        )

        # Either stop manager or exit manager should trigger exit
        assert 'BTCUSDT' in stop_actions or exit_decision.can_exit


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
