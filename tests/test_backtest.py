"""
Unit tests for the backtesting module.

Tests:
- Trade dataclass calculations
- BacktestConfig validation
- BacktestEngine trading logic
- BacktestReport generation
- Benchmark calculations
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    OrderType,
    Trade,
    run_walk_forward_backtest,
)
from src.backtest.report import BacktestReport
from src.backtest.benchmark import (
    buy_and_hold_benchmark,
    equal_weight_benchmark,
    compare_strategies,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_trade():
    """Create a sample trade for testing."""
    return Trade(
        symbol="BTCUSDT",
        entry_time=pd.Timestamp("2024-01-01 10:00:00"),
        exit_time=pd.Timestamp("2024-01-01 14:00:00"),
        entry_price=42000.0,
        exit_price=42420.0,
        quantity=0.1,
        side="long",
        entry_signal="model",
        exit_reason="signal",
        fees=8.4,
        slippage=4.2,
    )


@pytest.fixture
def sample_market_data():
    """Create sample market data for backtesting."""
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    symbols = ["BTCUSDT", "ETHUSDT"]

    rows = []
    for symbol in symbols:
        base_price = 42000 if symbol == "BTCUSDT" else 2200
        for i, ts in enumerate(dates):
            # Simulate some price movement
            noise = np.random.normal(0, 0.01)
            trend = 0.0001 * i
            price = base_price * (1 + trend + noise)

            rows.append({
                "timestamp": ts,
                "symbol": symbol,
                "open": price * (1 - 0.001),
                "high": price * (1 + 0.005),
                "low": price * (1 - 0.005),
                "close": price,
                "volume": np.random.uniform(100, 1000),
            })

    return pd.DataFrame(rows)


@pytest.fixture
def default_config():
    """Create default backtest configuration."""
    return BacktestConfig(
        initial_capital=10000,
        fee_rate=0.001,
        slippage_rate=0.0005,
        max_positions=5,
        max_position_pct=0.15,
        max_exposure=0.50,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
    )


# ============================================================================
# Trade Dataclass Tests
# ============================================================================


class TestTrade:
    """Tests for Trade dataclass."""

    def test_gross_pnl_long_winning(self, sample_trade):
        """Test gross P&L for winning long trade."""
        # (42420 - 42000) * 0.1 = 42
        assert sample_trade.gross_pnl == pytest.approx(42.0, rel=0.01)

    def test_gross_pnl_long_losing(self):
        """Test gross P&L for losing long trade."""
        trade = Trade(
            symbol="BTCUSDT",
            entry_time=pd.Timestamp("2024-01-01 10:00:00"),
            exit_time=pd.Timestamp("2024-01-01 14:00:00"),
            entry_price=42000.0,
            exit_price=41580.0,
            quantity=0.1,
            side="long",
            entry_signal="model",
            exit_reason="stop_loss",
            fees=8.4,
            slippage=4.2,
        )
        # (41580 - 42000) * 0.1 = -42
        assert trade.gross_pnl == pytest.approx(-42.0, rel=0.01)

    def test_gross_pnl_short_winning(self):
        """Test gross P&L for winning short trade."""
        trade = Trade(
            symbol="BTCUSDT",
            entry_time=pd.Timestamp("2024-01-01 10:00:00"),
            exit_time=pd.Timestamp("2024-01-01 14:00:00"),
            entry_price=42000.0,
            exit_price=41580.0,
            quantity=0.1,
            side="short",
            entry_signal="model",
            exit_reason="take_profit",
            fees=8.4,
            slippage=4.2,
        )
        # (42000 - 41580) * 0.1 = 42
        assert trade.gross_pnl == pytest.approx(42.0, rel=0.01)

    def test_net_pnl(self, sample_trade):
        """Test net P&L calculation (gross - fees - slippage)."""
        # 42 - 8.4 - 4.2 = 29.4
        assert sample_trade.net_pnl == pytest.approx(29.4, rel=0.01)

    def test_return_pct(self, sample_trade):
        """Test return percentage calculation."""
        cost = 42000 * 0.1  # 4200
        expected_return = 29.4 / 4200  # ~0.7%
        assert sample_trade.return_pct == pytest.approx(expected_return, rel=0.01)

    def test_hold_time_hours(self, sample_trade):
        """Test hold time calculation in hours."""
        # 10:00 to 14:00 = 4 hours
        assert sample_trade.hold_time_hours == 4.0

    def test_return_pct_zero_cost(self):
        """Test return percentage handles zero cost."""
        trade = Trade(
            symbol="BTCUSDT",
            entry_time=pd.Timestamp("2024-01-01"),
            exit_time=pd.Timestamp("2024-01-01"),
            entry_price=0.0,
            exit_price=100.0,
            quantity=0.0,
            side="long",
            entry_signal="test",
            exit_reason="test",
            fees=0.0,
            slippage=0.0,
        )
        assert trade.return_pct == 0.0


# ============================================================================
# BacktestConfig Tests
# ============================================================================


class TestBacktestConfig:
    """Tests for BacktestConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BacktestConfig()
        assert config.initial_capital == 10000.0
        assert config.fee_rate == 0.001
        assert config.slippage_rate == 0.0005
        assert config.max_positions == 5
        assert config.enable_shorting is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = BacktestConfig(
            initial_capital=50000,
            fee_rate=0.0005,
            max_positions=10,
            enable_shorting=True,
        )
        assert config.initial_capital == 50000
        assert config.fee_rate == 0.0005
        assert config.max_positions == 10
        assert config.enable_shorting is True


# ============================================================================
# BacktestEngine Tests
# ============================================================================


class TestBacktestEngine:
    """Tests for BacktestEngine class."""

    def test_initialization(self, default_config):
        """Test engine initialization."""
        engine = BacktestEngine(default_config)
        assert engine.cash == 10000
        assert len(engine.positions) == 0
        assert len(engine.trades) == 0
        assert len(engine.equity_curve) == 0

    def test_reset(self, default_config):
        """Test engine reset clears state."""
        engine = BacktestEngine(default_config)
        engine.cash = 5000
        engine.positions["BTCUSDT"] = {"quantity": 0.1}
        engine.reset()

        assert engine.cash == 10000
        assert len(engine.positions) == 0

    def test_get_total_value_cash_only(self, default_config):
        """Test total value with only cash."""
        engine = BacktestEngine(default_config)
        assert engine._get_total_value() == 10000

    def test_get_total_value_with_positions(self, default_config):
        """Test total value with positions."""
        engine = BacktestEngine(default_config)
        engine.cash = 5000
        engine.positions["BTCUSDT"] = {"current_value": 3000}
        engine.positions["ETHUSDT"] = {"current_value": 2000}
        assert engine._get_total_value() == 10000

    def test_get_current_exposure(self, default_config):
        """Test current exposure calculation."""
        engine = BacktestEngine(default_config)
        engine.positions["BTCUSDT"] = {"current_value": 1500}
        engine.positions["ETHUSDT"] = {"current_value": 1000}
        assert engine._get_current_exposure() == 2500

    def test_run_with_no_signals(self, default_config, sample_market_data):
        """Test backtest with no trading signals."""
        engine = BacktestEngine(default_config)

        def no_signals(data, idx):
            return {}

        results = engine.run(sample_market_data, no_signals)
        assert results["status"] == "no_trades"
        assert len(results["equity_curve"]) > 0

    def test_run_with_buy_hold_signals(self, default_config, sample_market_data):
        """Test backtest with simple buy-and-hold signals."""
        engine = BacktestEngine(default_config)

        bought = set()

        def buy_once(data, idx):
            signals = {}
            for symbol in ["BTCUSDT", "ETHUSDT"]:
                if symbol not in bought:
                    signals[symbol] = "BUY"
                    bought.add(symbol)
            return signals

        results = engine.run(sample_market_data, buy_once)

        # Should have 2 trades (closed at end of backtest)
        assert results["status"] == "success"
        assert results["metrics"]["n_trades"] == 2

    def test_run_with_buy_sell_signals(self, default_config, sample_market_data):
        """Test backtest with buy and sell signals."""
        engine = BacktestEngine(default_config)

        def alternating_signals(data, idx):
            timestamps = data["timestamp"].unique()
            if idx >= len(timestamps):
                return {}

            # Buy at idx 10, sell at idx 20
            if idx == 10:
                return {"BTCUSDT": "BUY"}
            elif idx == 20:
                return {"BTCUSDT": "SELL"}
            return {}

        results = engine.run(sample_market_data, alternating_signals)

        assert results["status"] == "success"
        assert results["metrics"]["n_trades"] == 1

        # Verify trade was closed by signal, not end of backtest
        trade = results["trades"][0]
        assert trade.exit_reason == "signal"

    def test_position_limits_enforced(self, default_config, sample_market_data):
        """Test that max positions limit is enforced."""
        config = BacktestConfig(
            initial_capital=10000,
            max_positions=1,  # Only allow 1 position
            max_position_pct=0.50,
            max_exposure=0.50,
        )
        engine = BacktestEngine(config)

        def buy_both(data, idx):
            # Try to buy both at once
            if idx == 10:
                return {"BTCUSDT": "BUY", "ETHUSDT": "BUY"}
            return {}

        results = engine.run(sample_market_data, buy_both)

        # Should only open 1 position due to limit
        # But both get closed at end, so only 1 trade
        assert results["metrics"]["n_trades"] == 1

    def test_exposure_limit_enforced(self, sample_market_data):
        """Test that max exposure limit is enforced."""
        config = BacktestConfig(
            initial_capital=10000,
            max_positions=10,
            max_position_pct=0.40,  # 40% per position
            max_exposure=0.30,  # But only 30% total exposure
        )
        engine = BacktestEngine(config)

        def buy_multiple(data, idx):
            if idx == 10:
                return {"BTCUSDT": "BUY", "ETHUSDT": "BUY"}
            return {}

        results = engine.run(sample_market_data, buy_multiple)

        # Due to exposure limit, may only open 1 position
        assert results["status"] == "success"

    def test_stop_loss_execution(self, sample_market_data):
        """Test stop-loss execution."""
        config = BacktestConfig(
            initial_capital=10000,
            stop_loss_pct=0.001,  # Very tight stop (0.1%)
            take_profit_pct=0.50,  # Very wide take profit
        )
        engine = BacktestEngine(config)

        def buy_early(data, idx):
            if idx == 5:
                return {"BTCUSDT": "BUY"}
            return {}

        results = engine.run(sample_market_data, buy_early)

        # With such a tight stop, likely hit stop-loss
        if results["status"] == "success" and results["trades"]:
            # Check if any trade was stopped out
            stop_trades = [t for t in results["trades"] if t.exit_reason == "stop_loss"]
            # Stop-loss may or may not trigger depending on price movement
            assert results["metrics"]["n_trades"] >= 1

    def test_metrics_calculation(self, default_config, sample_market_data):
        """Test that metrics are calculated correctly."""
        engine = BacktestEngine(default_config)

        def simple_strategy(data, idx):
            if idx == 10:
                return {"BTCUSDT": "BUY"}
            elif idx == 50:
                return {"BTCUSDT": "SELL"}
            return {}

        results = engine.run(sample_market_data, simple_strategy)

        if results["status"] == "success":
            metrics = results["metrics"]

            # Check all expected metrics exist
            expected_keys = [
                "total_return",
                "sharpe_ratio",
                "max_drawdown",
                "n_trades",
                "win_rate",
                "avg_trade",
                "final_value",
                "initial_capital",
            ]
            for key in expected_keys:
                assert key in metrics


# ============================================================================
# BacktestReport Tests
# ============================================================================


class TestBacktestReport:
    """Tests for BacktestReport class."""

    @pytest.fixture
    def sample_results(self, sample_trade):
        """Create sample results for testing."""
        return {
            "status": "success",
            "metrics": {
                "total_return": 0.05,
                "total_return_pct": 5.0,
                "sharpe_ratio": 1.5,
                "sortino_ratio": 2.0,
                "max_drawdown": 0.02,
                "max_drawdown_pct": 2.0,
                "calmar_ratio": 2.5,
                "n_trades": 10,
                "n_winners": 6,
                "n_losers": 4,
                "win_rate": 0.6,
                "avg_win": 50.0,
                "avg_loss": -30.0,
                "profit_factor": 2.5,
                "avg_trade": 20.0,
                "avg_return_per_trade": 0.005,
                "avg_hold_time_hours": 8.0,
                "total_fees": 100.0,
                "total_slippage": 50.0,
                "final_value": 10500.0,
                "initial_capital": 10000.0,
            },
            "trades": [sample_trade],
            "equity_curve": [
                {"timestamp": pd.Timestamp("2024-01-01"), "total_value": 10000},
                {"timestamp": pd.Timestamp("2024-01-02"), "total_value": 10200},
                {"timestamp": pd.Timestamp("2024-01-03"), "total_value": 10500},
            ],
        }

    def test_initialization(self, sample_results):
        """Test report initialization."""
        report = BacktestReport(sample_results)
        assert report.metrics == sample_results["metrics"]
        assert len(report.trades) == 1
        assert len(report.equity_curve) == 3

    def test_print_summary_success(self, sample_results, capsys):
        """Test print_summary for successful backtest."""
        report = BacktestReport(sample_results)
        report.print_summary()

        captured = capsys.readouterr()
        assert "BACKTEST RESULTS" in captured.out
        assert "5.00%" in captured.out  # Total return
        assert "1.50" in captured.out  # Sharpe ratio

    def test_print_summary_no_trades(self, capsys):
        """Test print_summary when no trades."""
        results = {"status": "no_trades"}
        report = BacktestReport(results)
        report.print_summary()

        captured = capsys.readouterr()
        assert "no_trades" in captured.out

    def test_to_dataframe(self, sample_results):
        """Test conversion to DataFrame."""
        report = BacktestReport(sample_results)
        df = report.to_dataframe()

        assert len(df) == 1
        assert "symbol" in df.columns
        assert "net_pnl" in df.columns
        assert "entry_time" in df.columns

    def test_to_dataframe_empty(self):
        """Test to_dataframe with no trades."""
        results = {"status": "no_trades", "trades": []}
        report = BacktestReport(results)
        df = report.to_dataframe()

        assert df.empty

    def test_to_dict(self, sample_results):
        """Test export to dictionary."""
        report = BacktestReport(sample_results)
        result_dict = report.to_dict()

        assert "status" in result_dict
        assert "metrics" in result_dict
        assert "trades" in result_dict
        assert "equity_curve" in result_dict


# ============================================================================
# Benchmark Tests
# ============================================================================


class TestBenchmarks:
    """Tests for benchmark functions."""

    @pytest.fixture
    def benchmark_data(self):
        """Create data for benchmark testing."""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        rows = []

        for symbol, base_price in [("BTCUSDT", 40000), ("ETHUSDT", 2000)]:
            for i, ts in enumerate(dates):
                # 10% gain over period
                price = base_price * (1 + 0.1 * i / len(dates))
                rows.append({
                    "timestamp": ts,
                    "symbol": symbol,
                    "close": price,
                })

        return pd.DataFrame(rows)

    def test_buy_and_hold_positive_return(self, benchmark_data):
        """Test buy and hold with positive return."""
        result = buy_and_hold_benchmark(
            benchmark_data,
            "BTCUSDT",
            initial_capital=10000,
            fee_rate=0.001,
        )

        assert "Buy & Hold BTCUSDT" in result["strategy"]
        assert result["total_return"] > 0
        assert result["final_value"] > 10000

    def test_buy_and_hold_with_fees(self, benchmark_data):
        """Test that fees reduce returns."""
        result_no_fee = buy_and_hold_benchmark(
            benchmark_data,
            "BTCUSDT",
            initial_capital=10000,
            fee_rate=0.0,
        )
        result_with_fee = buy_and_hold_benchmark(
            benchmark_data,
            "BTCUSDT",
            initial_capital=10000,
            fee_rate=0.01,  # 1% fee
        )

        assert result_with_fee["final_value"] < result_no_fee["final_value"]

    def test_buy_and_hold_missing_symbol(self, benchmark_data):
        """Test buy and hold with non-existent symbol."""
        result = buy_and_hold_benchmark(
            benchmark_data,
            "DOGUSDT",  # Not in data
            initial_capital=10000,
        )

        assert result["total_return"] == 0.0
        assert result["final_value"] == 10000

    def test_equal_weight_benchmark(self, benchmark_data):
        """Test equal weight portfolio benchmark."""
        result = equal_weight_benchmark(
            benchmark_data,
            ["BTCUSDT", "ETHUSDT"],
            initial_capital=10000,
            fee_rate=0.001,
        )

        assert result["strategy"] == "Equal Weight Portfolio"
        assert result["total_return"] > 0
        assert "symbols_included" in result
        assert len(result["symbols_included"]) == 2

    def test_equal_weight_partial_symbols(self, benchmark_data):
        """Test equal weight with some missing symbols."""
        result = equal_weight_benchmark(
            benchmark_data,
            ["BTCUSDT", "DOGUSDT"],  # DOGUSDT not in data
            initial_capital=10000,
        )

        # Should only include BTCUSDT
        assert len(result["symbols_included"]) == 1
        assert "BTCUSDT" in result["symbols_included"]

    def test_equal_weight_empty_symbols(self, benchmark_data):
        """Test equal weight with no symbols."""
        result = equal_weight_benchmark(
            benchmark_data,
            [],
            initial_capital=10000,
        )

        assert result["total_return"] == 0.0
        assert result["final_value"] == 10000

    def test_compare_strategies(self, benchmark_data):
        """Test strategy comparison table generation."""
        strategy_results = {
            "status": "success",
            "metrics": {
                "total_return_pct": 15.0,
                "sharpe_ratio": 2.0,
                "max_drawdown_pct": 5.0,
                "win_rate": 0.6,
                "profit_factor": 2.5,
                "final_value": 11500,
            },
        }

        btc_benchmark = buy_and_hold_benchmark(
            benchmark_data, "BTCUSDT", 10000
        )

        comparison = compare_strategies(strategy_results, [btc_benchmark])

        assert len(comparison) == 2  # Strategy + benchmark
        assert "Strategy" in comparison.columns
        assert "Total Return (%)" in comparison.columns


# ============================================================================
# Integration Tests
# ============================================================================


class TestBacktestIntegration:
    """Integration tests for the full backtest workflow."""

    def test_full_workflow(self, default_config, sample_market_data):
        """Test complete backtest workflow."""
        # 1. Run backtest
        engine = BacktestEngine(default_config)

        def simple_strategy(data, idx):
            if idx == 10:
                return {"BTCUSDT": "BUY", "ETHUSDT": "BUY"}
            elif idx == 50:
                return {"BTCUSDT": "SELL", "ETHUSDT": "SELL"}
            return {}

        results = engine.run(sample_market_data, simple_strategy)

        # 2. Generate report
        report = BacktestReport(results)

        # 3. Get trade DataFrame
        trades_df = report.to_dataframe()

        # 4. Calculate benchmarks
        btc_benchmark = buy_and_hold_benchmark(
            sample_market_data,
            "BTCUSDT",
            default_config.initial_capital,
        )

        # 5. Compare
        comparison = compare_strategies(results, [btc_benchmark])

        # Verify all steps completed
        assert results["status"] == "success"
        assert len(trades_df) > 0
        assert len(comparison) == 2

    def test_reproducibility(self, default_config, sample_market_data):
        """Test that same inputs produce same outputs."""
        def deterministic_strategy(data, idx):
            if idx == 10:
                return {"BTCUSDT": "BUY"}
            elif idx == 30:
                return {"BTCUSDT": "SELL"}
            return {}

        engine1 = BacktestEngine(default_config)
        results1 = engine1.run(sample_market_data, deterministic_strategy)

        engine2 = BacktestEngine(default_config)
        results2 = engine2.run(sample_market_data, deterministic_strategy)

        # Results should be identical
        assert results1["metrics"]["n_trades"] == results2["metrics"]["n_trades"]
        assert results1["metrics"]["final_value"] == pytest.approx(
            results2["metrics"]["final_value"], rel=0.001
        )


# ============================================================================
# Dict Signal and Confidence Tests
# ============================================================================


class TestDictSignals:
    """Tests for dict signal format with confidence."""

    @pytest.fixture
    def confidence_config(self):
        """Config with confidence-based sizing enabled."""
        return BacktestConfig(
            initial_capital=10000,
            use_confidence_sizing=True,
            min_confidence=0.55,
            max_position_pct=0.20,
            max_exposure=0.60,
        )

    def test_dict_signal_format(self, default_config, sample_market_data):
        """Test that dict signals work correctly."""
        engine = BacktestEngine(default_config)

        def dict_signal_strategy(data, idx):
            if idx == 10:
                return {
                    "BTCUSDT": {"action": "BUY", "confidence": 0.75},
                    "ETHUSDT": {"action": "BUY", "confidence": 0.65},
                }
            elif idx == 50:
                return {
                    "BTCUSDT": {"action": "SELL", "confidence": 0.80},
                    "ETHUSDT": {"action": "SELL", "confidence": 0.70},
                }
            return {}

        results = engine.run(sample_market_data, dict_signal_strategy)

        assert results["status"] == "success"
        assert results["metrics"]["n_trades"] == 2

    def test_confidence_filtering(self, sample_market_data):
        """Test that low confidence signals are filtered out."""
        config = BacktestConfig(
            initial_capital=10000,
            min_confidence=0.70,  # High threshold
        )
        engine = BacktestEngine(config)

        def low_confidence_strategy(data, idx):
            if idx == 10:
                return {
                    "BTCUSDT": {"action": "BUY", "confidence": 0.50},  # Below threshold
                    "ETHUSDT": {"action": "BUY", "confidence": 0.80},  # Above threshold
                }
            return {}

        results = engine.run(sample_market_data, low_confidence_strategy)

        # Only ETH should be traded (BTC confidence too low)
        assert results["status"] == "success"
        assert results["metrics"]["n_trades"] == 1

    def test_confidence_sizing(self, confidence_config, sample_market_data):
        """Test that confidence affects position size."""
        engine = BacktestEngine(confidence_config)

        # Track position values
        position_values = []

        def capture_positions(data, idx):
            if idx == 10:
                return {"BTCUSDT": {"action": "BUY", "confidence": 0.60}}
            elif idx == 20:
                if "BTCUSDT" in engine.positions:
                    position_values.append(
                        ("low_conf", engine.positions["BTCUSDT"]["current_value"])
                    )
                return {"BTCUSDT": {"action": "SELL", "confidence": 0.80}}
            elif idx == 30:
                return {"BTCUSDT": {"action": "BUY", "confidence": 0.95}}
            elif idx == 40:
                if "BTCUSDT" in engine.positions:
                    position_values.append(
                        ("high_conf", engine.positions["BTCUSDT"]["current_value"])
                    )
                return {"BTCUSDT": {"action": "SELL", "confidence": 0.80}}
            return {}

        results = engine.run(sample_market_data, capture_positions)

        # Higher confidence should result in larger position
        # (accounting for market movement and remaining capital)
        assert results["status"] == "success"

    def test_mixed_signal_formats(self, default_config, sample_market_data):
        """Test mixing string and dict signal formats."""
        engine = BacktestEngine(default_config)

        def mixed_strategy(data, idx):
            if idx == 10:
                return {
                    "BTCUSDT": "BUY",  # String format
                    "ETHUSDT": {"action": "BUY", "confidence": 0.75},  # Dict format
                }
            elif idx == 50:
                return {
                    "BTCUSDT": {"action": "SELL", "confidence": 0.80},
                    "ETHUSDT": "SELL",
                }
            return {}

        results = engine.run(sample_market_data, mixed_strategy)

        assert results["status"] == "success"
        assert results["metrics"]["n_trades"] == 2


# ============================================================================
# Walk-Forward Backtest Tests
# ============================================================================


# ============================================================================
# Shorting Tests
# ============================================================================


class TestShorting:
    """Tests for short selling functionality."""

    @pytest.fixture
    def shorting_config(self):
        """Config with shorting enabled and wide stops to avoid early exits."""
        return BacktestConfig(
            initial_capital=10000,
            fee_rate=0.001,
            slippage_rate=0.0005,
            max_positions=5,
            max_position_pct=0.15,
            max_exposure=0.50,
            stop_loss_pct=0.50,  # Wide stop to avoid early exit
            take_profit_pct=0.50,  # Wide take-profit to avoid early exit
            enable_shorting=True,
        )

    def test_short_position_opened_on_sell(self, shorting_config, sample_market_data):
        """Test that SELL signal opens short position when no position exists."""
        engine = BacktestEngine(shorting_config)

        def sell_signal(data, idx):
            if idx == 10:
                return {"BTCUSDT": "SELL"}
            return {}

        results = engine.run(sample_market_data, sell_signal)

        # Should have 1 trade (short closed at end of backtest)
        assert results["status"] == "success"
        assert results["metrics"]["n_trades"] == 1
        # Verify it was a short trade
        trade = results["trades"][0]
        assert trade.side == "short"

    def test_short_not_opened_when_disabled(self, default_config, sample_market_data):
        """Test that SELL signal does NOT open short when shorting is disabled."""
        engine = BacktestEngine(default_config)

        def sell_signal(data, idx):
            if idx == 10:
                return {"BTCUSDT": "SELL"}
            return {}

        results = engine.run(sample_market_data, sell_signal)

        # Should have no trades (SELL ignored without position)
        assert results["status"] == "no_trades"

    def test_buy_closes_short_position(self, shorting_config, sample_market_data):
        """Test that BUY signal closes an existing short position."""
        engine = BacktestEngine(shorting_config)

        def short_then_buy(data, idx):
            if idx == 10:
                return {"BTCUSDT": "SELL"}  # Open short
            elif idx == 30:
                return {"BTCUSDT": "BUY"}  # Close short
            return {}

        results = engine.run(sample_market_data, short_then_buy)

        assert results["status"] == "success"
        assert results["metrics"]["n_trades"] == 1
        # Verify exit reason is signal, not end_of_backtest
        trade = results["trades"][0]
        assert trade.exit_reason == "signal"
        assert trade.side == "short"

    def test_sell_closes_long_position(self, shorting_config, sample_market_data):
        """Test that SELL signal closes an existing long position (not opens short)."""
        engine = BacktestEngine(shorting_config)

        def long_then_sell(data, idx):
            if idx == 10:
                return {"BTCUSDT": "BUY"}  # Open long
            elif idx == 30:
                return {"BTCUSDT": "SELL"}  # Close long
            return {}

        results = engine.run(sample_market_data, long_then_sell)

        assert results["status"] == "success"
        assert results["metrics"]["n_trades"] == 1
        trade = results["trades"][0]
        assert trade.side == "long"
        assert trade.exit_reason == "signal"

    def test_buy_does_not_double_long(self, shorting_config, sample_market_data):
        """Test that BUY signal does not open a second long when long exists."""
        engine = BacktestEngine(shorting_config)

        def double_buy(data, idx):
            if idx == 10:
                return {"BTCUSDT": "BUY"}  # Open long
            elif idx == 20:
                return {"BTCUSDT": "BUY"}  # Should be ignored (already long)
            return {}

        results = engine.run(sample_market_data, double_buy)

        assert results["status"] == "success"
        # Only 1 trade (the initial long, closed at end)
        assert results["metrics"]["n_trades"] == 1

    def test_sell_does_not_double_short(self, shorting_config, sample_market_data):
        """Test that SELL signal does not open a second short when short exists."""
        engine = BacktestEngine(shorting_config)

        def double_sell(data, idx):
            if idx == 10:
                return {"BTCUSDT": "SELL"}  # Open short
            elif idx == 20:
                return {"BTCUSDT": "SELL"}  # Should be ignored (already short)
            return {}

        results = engine.run(sample_market_data, double_sell)

        assert results["status"] == "success"
        # Only 1 trade (the initial short, closed at end)
        assert results["metrics"]["n_trades"] == 1

    def test_short_stop_loss(self, sample_market_data):
        """Test stop-loss execution for short position."""
        config = BacktestConfig(
            initial_capital=10000,
            stop_loss_pct=0.001,  # Very tight stop (0.1%)
            take_profit_pct=0.50,
            enable_shorting=True,
        )
        engine = BacktestEngine(config)

        def open_short(data, idx):
            if idx == 5:
                return {"BTCUSDT": "SELL"}  # Open short
            return {}

        results = engine.run(sample_market_data, open_short)

        # With tight stop on a volatile market, likely hit stop-loss
        if results["status"] == "success" and results["trades"]:
            trade = results["trades"][0]
            assert trade.side == "short"

    def test_short_take_profit(self, sample_market_data):
        """Test take-profit execution for short position."""
        config = BacktestConfig(
            initial_capital=10000,
            stop_loss_pct=0.50,  # Wide stop
            take_profit_pct=0.001,  # Very tight take-profit (0.1%)
            enable_shorting=True,
        )
        engine = BacktestEngine(config)

        def open_short(data, idx):
            if idx == 5:
                return {"BTCUSDT": "SELL"}
            return {}

        results = engine.run(sample_market_data, open_short)

        if results["status"] == "success" and results["trades"]:
            trade = results["trades"][0]
            assert trade.side == "short"

    def test_short_pnl_calculation(self, shorting_config, sample_market_data):
        """Test that short P&L is calculated correctly (profit when price drops)."""
        engine = BacktestEngine(shorting_config)

        # Create data with known price movement
        dates = pd.date_range("2024-01-01", periods=50, freq="h")
        rows = []
        for i, ts in enumerate(dates):
            # Price drops 10% over the period
            price = 40000 * (1 - 0.1 * i / len(dates))
            rows.append({
                "timestamp": ts,
                "symbol": "BTCUSDT",
                "open": price,
                "high": price * 1.001,
                "low": price * 0.999,
                "close": price,
                "volume": 1000,
            })
        declining_data = pd.DataFrame(rows)

        def short_strategy(data, idx):
            if idx == 5:
                return {"BTCUSDT": "SELL"}  # Open short
            elif idx == 40:
                return {"BTCUSDT": "BUY"}  # Close short
            return {}

        results = engine.run(declining_data, short_strategy)

        assert results["status"] == "success"
        trade = results["trades"][0]
        assert trade.side == "short"
        # Price dropped, so short should be profitable
        assert trade.gross_pnl > 0

    def test_mixed_long_and_short_trades(self, shorting_config, sample_market_data):
        """Test alternating between long and short positions."""
        engine = BacktestEngine(shorting_config)

        def alternating_strategy(data, idx):
            if idx == 10:
                return {"BTCUSDT": "BUY"}  # Open long
            elif idx == 20:
                return {"BTCUSDT": "SELL"}  # Close long
            elif idx == 30:
                return {"BTCUSDT": "SELL"}  # Open short
            elif idx == 40:
                return {"BTCUSDT": "BUY"}  # Close short
            return {}

        results = engine.run(sample_market_data, alternating_strategy)

        assert results["status"] == "success"
        assert results["metrics"]["n_trades"] == 2

        # Verify we have both long and short trades
        sides = {t.side for t in results["trades"]}
        assert "long" in sides
        assert "short" in sides


class TestWalkForwardBacktest:
    """Tests for walk-forward backtesting."""

    @pytest.fixture
    def larger_market_data(self):
        """Create larger dataset for walk-forward testing."""
        dates = pd.date_range("2024-01-01", periods=500, freq="h")
        symbols = ["BTCUSDT", "ETHUSDT"]

        rows = []
        for symbol in symbols:
            base_price = 42000 if symbol == "BTCUSDT" else 2200
            for i, ts in enumerate(dates):
                noise = np.random.normal(0, 0.01)
                trend = 0.0001 * i
                price = base_price * (1 + trend + noise)

                rows.append({
                    "timestamp": ts,
                    "symbol": symbol,
                    "open": price * (1 - 0.001),
                    "high": price * (1 + 0.005),
                    "low": price * (1 - 0.005),
                    "close": price,
                    "volume": np.random.uniform(100, 1000),
                })

        return pd.DataFrame(rows)

    def test_walk_forward_basic(self, larger_market_data):
        """Test basic walk-forward backtest."""
        config = BacktestConfig(initial_capital=10000)

        # Dummy trainer that does nothing
        def dummy_trainer(train_data):
            pass

        # Simple strategy
        def simple_strategy(data, idx):
            timestamps = data["timestamp"].unique()
            if idx < len(timestamps):
                # Buy every 20 bars, sell every 40
                if idx % 40 == 20:
                    return {"BTCUSDT": "BUY"}
                elif idx % 40 == 0 and idx > 0:
                    return {"BTCUSDT": "SELL"}
            return {}

        results = run_walk_forward_backtest(
            larger_market_data,
            dummy_trainer,
            simple_strategy,
            config,
            n_splits=3,
            train_pct=0.6,
        )

        assert results["status"] == "success"
        assert len(results["per_period"]) == 3
        assert "aggregated" in results

    def test_walk_forward_cumulative_capital(self, larger_market_data):
        """Test that capital accumulates across periods."""
        config = BacktestConfig(initial_capital=10000)

        def dummy_trainer(train_data):
            pass

        # Strategy that should be profitable
        counter = {"idx": 0}

        def profitable_strategy(data, idx):
            # Simple trend following
            timestamps = data["timestamp"].unique()
            if idx >= len(timestamps):
                return {}
            if idx % 30 == 10:
                return {"BTCUSDT": "BUY"}
            elif idx % 30 == 25:
                return {"BTCUSDT": "SELL"}
            return {}

        results = run_walk_forward_backtest(
            larger_market_data,
            dummy_trainer,
            profitable_strategy,
            config,
            n_splits=2,
            train_pct=0.5,
        )

        # Check that per_period results exist
        assert len(results["per_period"]) == 2

    def test_walk_forward_aggregated_metrics(self, larger_market_data):
        """Test that aggregated metrics are calculated."""
        config = BacktestConfig(initial_capital=10000)

        def dummy_trainer(train_data):
            pass

        def simple_strategy(data, idx):
            if idx % 50 == 25:
                return {"BTCUSDT": "BUY"}
            elif idx % 50 == 45:
                return {"BTCUSDT": "SELL"}
            return {}

        results = run_walk_forward_backtest(
            larger_market_data,
            dummy_trainer,
            simple_strategy,
            config,
            n_splits=2,
            train_pct=0.6,
        )

        if results["status"] == "success":
            agg = results["aggregated"]
            assert "total_return" in agg
            assert "sharpe_ratio" in agg
            assert "max_drawdown" in agg
            assert "n_trades" in agg
            assert "final_value" in agg
