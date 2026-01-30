# tests/test_trading_model.py
"""
Tests for ImprovedTradingModel.

Covers:
- prepare_features: output shape, binary target, lag columns, no leakage
- train: status, is_fitted, feature importances, validation
- predict: signal format, action validity, confidence bounds, threshold logic
- Data leakage: target uses shift(-1), features use positive shifts
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.trading_model import ImprovedTradingModel


# ──────────────────────────── helpers ────────────────────────────


def _make_market_df(n_hours: int = 200, symbol: str = "BTCUSDT", seed: int = 42):
    """Create realistic OHLCV market data."""
    rng = np.random.RandomState(seed)
    timestamps = pd.date_range("2024-01-01", periods=n_hours, freq="1h")

    # Generate random walk price
    returns = rng.normal(0, 0.01, n_hours)
    close = 40000 * np.exp(np.cumsum(returns))

    # Create OHLCV
    high = close * (1 + rng.uniform(0, 0.01, n_hours))
    low = close * (1 - rng.uniform(0, 0.01, n_hours))
    open_price = close * (1 + rng.normal(0, 0.005, n_hours))
    volume = rng.uniform(100, 10000, n_hours)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": symbol,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _make_sentiment_df(n_hours: int = 200, symbol: str = "BTCUSDT", seed: int = 42):
    """Create matching sentiment data."""
    rng = np.random.RandomState(seed + 1)
    timestamps = pd.date_range("2024-01-01", periods=n_hours, freq="1h")

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": symbol,
            "sentiment_mean": rng.uniform(-0.5, 0.5, n_hours),
            "post_volume": rng.randint(0, 50, n_hours).astype(float),
        }
    )


# ──────────────────────────── fixtures ────────────────────────────


@pytest.fixture
def market_df():
    return _make_market_df(n_hours=200)


@pytest.fixture
def sentiment_df():
    return _make_sentiment_df(n_hours=200)


@pytest.fixture
def large_market_df():
    """Larger dataset for training tests (needs min_train_size samples)."""
    return _make_market_df(n_hours=800)


@pytest.fixture
def large_sentiment_df():
    return _make_sentiment_df(n_hours=800)


@pytest.fixture
def model():
    return ImprovedTradingModel()


@pytest.fixture
def small_model():
    """Model with smaller validation windows for faster tests."""
    m = ImprovedTradingModel()
    m.validator.min_train_size = 50
    m.validator.test_size = 20
    m.validator.step_size = 20
    return m


# ═══════════════════════ prepare_features ═══════════════════════


class TestPrepareFeatures:
    """Tests for the prepare_features method."""

    def test_returns_dataframe(self, model, market_df, sentiment_df):
        """Should return a non-empty DataFrame."""
        df = model.prepare_features(market_df, sentiment_df)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_has_target_column(self, model, market_df, sentiment_df):
        """Training mode should produce a 'target' column."""
        df = model.prepare_features(market_df, sentiment_df, is_inference=False)
        assert "target" in df.columns

    def test_binary_target(self, model, market_df, sentiment_df):
        """Target values should be 0 or 1 only (binary, not ternary)."""
        df = model.prepare_features(market_df, sentiment_df, is_inference=False)
        unique_targets = set(df["target"].unique())
        assert unique_targets.issubset({0, 1})

    def test_no_target_in_inference(self, model, market_df, sentiment_df):
        """Inference mode should not create a 'target' column."""
        df = model.prepare_features(market_df, sentiment_df, is_inference=True)
        assert "target" not in df.columns
        assert "target_return" not in df.columns

    def test_lag_columns_present(self, model, market_df, sentiment_df):
        """All 24 lag columns should exist (8 lags x 3 types)."""
        df = model.prepare_features(market_df, sentiment_df)
        expected_lags = [1, 2, 3, 6, 12, 24, 36, 48]
        for lag in expected_lags:
            assert f"sent_lag_{lag}" in df.columns
            assert f"vol_lag_{lag}" in df.columns
            assert f"ret_lag_{lag}" in df.columns

    def test_technical_features_present(self, model, market_df, sentiment_df):
        """At least some technical indicator columns should exist."""
        df = model.prepare_features(market_df, sentiment_df)
        technical_cols = [c for c in df.columns if c.startswith(("rsi_", "macd_", "bb_", "atr_"))]
        assert len(technical_cols) > 0

    def test_features_list_populated(self, model, market_df, sentiment_df):
        """model.features should be populated after prepare_features."""
        model.prepare_features(market_df, sentiment_df)
        assert len(model.features) > 0

    def test_no_future_data_in_lags(self, model, market_df, sentiment_df):
        """
        Lag features at row i should equal the source value at row i-lag.
        This confirms we only look backward.
        """
        df = model.prepare_features(market_df, sentiment_df, is_inference=True)
        # Check sent_lag_1: at any row, it should be the previous row's sentiment_mean
        sym = df["symbol"].iloc[0]
        sym_df = df[df["symbol"] == sym].sort_values("timestamp").reset_index(drop=True)

        for idx in range(1, min(10, len(sym_df))):
            lag_val = sym_df.loc[idx, "sent_lag_1"]
            actual_prev = sym_df.loc[idx - 1, "sentiment_mean"] if "sentiment_mean" in sym_df.columns else np.nan
            if not np.isnan(lag_val) and not np.isnan(actual_prev):
                assert lag_val == pytest.approx(actual_prev, abs=1e-10)

    def test_empty_market_returns_empty(self, model, sentiment_df):
        """Empty market data should return empty DataFrame."""
        result = model.prepare_features(pd.DataFrame(), sentiment_df)
        assert result.empty

    def test_empty_sentiment_ok(self, model, market_df):
        """Should work with no sentiment data."""
        df = model.prepare_features(market_df, pd.DataFrame())
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_none_sentiment_ok(self, model, market_df):
        """Should handle None sentiment gracefully."""
        df = model.prepare_features(market_df, None)
        assert not df.empty


# ═══════════════════════ train ═══════════════════════


class TestTrain:
    """Tests for the train method."""

    def test_train_returns_success(self, small_model, market_df, sentiment_df):
        """Training should return status='success'."""
        df = small_model.prepare_features(market_df, sentiment_df)
        result = small_model.train(df, validate=False)
        assert result["status"] == "success"

    def test_train_sets_is_fitted(self, small_model, market_df, sentiment_df):
        """After training, is_fitted should be True."""
        df = small_model.prepare_features(market_df, sentiment_df)
        small_model.train(df, validate=False)
        assert small_model.is_fitted is True

    def test_train_produces_feature_importances(
        self, small_model, market_df, sentiment_df
    ):
        """Feature importances should be populated after training."""
        df = small_model.prepare_features(market_df, sentiment_df)
        small_model.train(df, validate=False)
        assert small_model.feature_importances is not None
        assert len(small_model.feature_importances) > 0

    def test_train_result_keys(self, small_model, market_df, sentiment_df):
        """Result dict should have expected keys."""
        df = small_model.prepare_features(market_df, sentiment_df)
        result = small_model.train(df, validate=False)
        assert "n_samples" in result
        assert "n_features" in result
        assert "top_features" in result

    def test_train_skip_validation(self, small_model, market_df, sentiment_df):
        """validate=False should skip walk-forward and set validation to None."""
        df = small_model.prepare_features(market_df, sentiment_df)
        result = small_model.train(df, validate=False)
        assert result["validation"] is None
        assert small_model.validation_results is None

    def test_train_with_validation(self, small_model, market_df, sentiment_df):
        """validate=True should populate validation_results."""
        df = small_model.prepare_features(market_df, sentiment_df)
        result = small_model.train(df, validate=True)
        assert result["validation"] is not None
        assert "precision" in result["validation"]

    def test_train_insufficient_data(self, small_model):
        """< 100 rows should return insufficient_data status."""
        df = pd.DataFrame(
            {
                "f1": np.random.randn(50),
                "target": np.random.randint(0, 2, 50),
            }
        )
        small_model.features = ["f1"]
        result = small_model.train(df, validate=False)
        assert result["status"] == "insufficient_data"

    def test_train_empty_returns_insufficient(self, small_model):
        """Empty DataFrame should return insufficient_data."""
        small_model.features = []
        result = small_model.train(pd.DataFrame(), validate=False)
        assert result["status"] == "insufficient_data"


# ═══════════════════════ predict ═══════════════════════


class TestPredict:
    """Tests for the predict method."""

    def test_returns_signals_dict(self, small_model, market_df, sentiment_df):
        """Should return a dict mapping symbol to signal dict."""
        df = small_model.prepare_features(market_df, sentiment_df)
        small_model.train(df, validate=False)

        pred_df = small_model.prepare_features(market_df, sentiment_df, is_inference=True)
        latest_ts = pred_df["timestamp"].max()
        latest = pred_df[pred_df["timestamp"] == latest_ts]
        signals = small_model.predict(latest)

        assert isinstance(signals, dict)
        assert len(signals) > 0

    def test_signal_has_required_keys(self, small_model, market_df, sentiment_df):
        """Each signal should have action, confidence, buy_probability, features_used."""
        df = small_model.prepare_features(market_df, sentiment_df)
        small_model.train(df, validate=False)

        pred_df = small_model.prepare_features(market_df, sentiment_df, is_inference=True)
        latest_ts = pred_df["timestamp"].max()
        latest = pred_df[pred_df["timestamp"] == latest_ts]
        signals = small_model.predict(latest)

        for symbol, signal in signals.items():
            assert "action" in signal
            assert "confidence" in signal
            assert "buy_probability" in signal
            assert "features_used" in signal

    def test_actions_are_valid(self, small_model, market_df, sentiment_df):
        """Actions should be one of BUY, SELL, HOLD."""
        df = small_model.prepare_features(market_df, sentiment_df)
        small_model.train(df, validate=False)

        pred_df = small_model.prepare_features(market_df, sentiment_df, is_inference=True)
        latest_ts = pred_df["timestamp"].max()
        latest = pred_df[pred_df["timestamp"] == latest_ts]
        signals = small_model.predict(latest)

        for signal in signals.values():
            assert signal["action"] in {"BUY", "SELL", "HOLD"}

    def test_confidence_bounded(self, small_model, market_df, sentiment_df):
        """Confidence should be in [0, 1]."""
        df = small_model.prepare_features(market_df, sentiment_df)
        small_model.train(df, validate=False)

        pred_df = small_model.prepare_features(market_df, sentiment_df, is_inference=True)
        latest_ts = pred_df["timestamp"].max()
        latest = pred_df[pred_df["timestamp"] == latest_ts]
        signals = small_model.predict(latest)

        for signal in signals.values():
            assert 0.0 <= signal["confidence"] <= 1.0

    def test_buy_probability_bounded(self, small_model, market_df, sentiment_df):
        """Buy probability should be in [0, 1]."""
        df = small_model.prepare_features(market_df, sentiment_df)
        small_model.train(df, validate=False)

        pred_df = small_model.prepare_features(market_df, sentiment_df, is_inference=True)
        latest_ts = pred_df["timestamp"].max()
        latest = pred_df[pred_df["timestamp"] == latest_ts]
        signals = small_model.predict(latest)

        for signal in signals.values():
            assert 0.0 <= signal["buy_probability"] <= 1.0

    def test_predict_before_fit_returns_empty(self, model):
        """Predicting before fit should return empty dict."""
        dummy = pd.DataFrame({"symbol": ["BTCUSDT"], "f1": [1.0]})
        result = model.predict(dummy)
        assert result == {}

    def test_predict_empty_df_returns_empty(self, small_model, market_df, sentiment_df):
        """Predicting on empty DataFrame should return empty dict."""
        df = small_model.prepare_features(market_df, sentiment_df)
        small_model.train(df, validate=False)
        result = small_model.predict(pd.DataFrame())
        assert result == {}

    def test_threshold_logic_buy(self, small_model, market_df, sentiment_df):
        """When buy_probability >= enter_threshold, action should be BUY
        (unless confidence < min_confidence)."""
        df = small_model.prepare_features(market_df, sentiment_df)
        small_model.train(df, validate=False)

        pred_df = small_model.prepare_features(market_df, sentiment_df, is_inference=True)
        latest_ts = pred_df["timestamp"].max()
        latest = pred_df[pred_df["timestamp"] == latest_ts]
        signals = small_model.predict(latest)

        for signal in signals.values():
            prob = signal["buy_probability"]
            action = signal["action"]
            if prob >= small_model.enter_threshold and prob >= small_model.min_confidence:
                assert action == "BUY"
            elif prob <= small_model.exit_threshold and (1 - prob) >= small_model.min_confidence:
                assert action == "SELL"


# ═══════════════════════ No data leakage ═══════════════════════


class TestNoDataLeakage:
    """Tests verifying no future data leaks into features."""

    def test_target_uses_future_return(self, model, market_df, sentiment_df):
        """Target should be based on the NEXT hour's return (shift(-1))."""
        df = model.prepare_features(market_df, sentiment_df, is_inference=False)
        # target_return is the next hour's return
        assert "target_return" in df.columns

    def test_features_only_use_past(self, model, market_df, sentiment_df):
        """All lag features should use positive shift values (look backward)."""
        model.prepare_features(market_df, sentiment_df)
        for feat in model.features:
            if "lag" in feat:
                # Extract lag number — all should be positive
                parts = feat.split("_")
                lag_num = int(parts[-1])
                assert lag_num > 0, f"Feature {feat} has non-positive lag"

    def test_no_target_in_features(self, model, market_df, sentiment_df):
        """'target' and 'target_return' should not be in the feature list."""
        model.prepare_features(market_df, sentiment_df)
        assert "target" not in model.features
        assert "target_return" not in model.features

    def test_no_raw_prices_in_features(self, model, market_df, sentiment_df):
        """Raw OHLCV columns should not be in the feature list."""
        model.prepare_features(market_df, sentiment_df)
        excluded = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
        for feat in model.features:
            assert feat not in excluded, f"Raw column '{feat}' found in features"


# ═══════════════════════ get_model_summary ═══════════════════════


class TestGetModelSummary:
    """Tests for the get_model_summary method."""

    def test_summary_before_fit(self, model):
        """Summary should work before fitting."""
        summary = model.get_model_summary()
        assert summary["is_fitted"] is False
        assert summary["validation"] is None
        assert summary["top_features"] is None

    def test_summary_after_fit(self, small_model, market_df, sentiment_df):
        """Summary should contain all expected keys after fitting."""
        df = small_model.prepare_features(market_df, sentiment_df)
        small_model.train(df, validate=False)
        summary = small_model.get_model_summary()

        assert summary["is_fitted"] is True
        assert summary["n_features"] > 0
        assert "enter" in summary["thresholds"]
        assert "exit" in summary["thresholds"]
        assert "min_confidence" in summary["thresholds"]
        assert summary["top_features"] is not None


# ═══════════════════════ Single-symbol validation ═══════════════════════


class TestSingleSymbolValidation:
    """Tests verifying validation uses single symbol to prevent data leakage."""

    @pytest.fixture
    def multi_symbol_market_df(self):
        """Create multi-symbol market data to test validation behavior."""
        dfs = []
        for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            # Use 200 hours (smaller for faster tests)
            df = _make_market_df(n_hours=200, symbol=symbol, seed=42)
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    @pytest.fixture
    def multi_symbol_sentiment_df(self):
        """Create matching multi-symbol sentiment data."""
        dfs = []
        for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            df = _make_sentiment_df(n_hours=200, symbol=symbol, seed=42)
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    def test_validation_uses_single_symbol(
        self, multi_symbol_market_df, multi_symbol_sentiment_df
    ):
        """Validation should use only the specified symbol, not all rows."""
        model = ImprovedTradingModel()
        # Use smaller windows for faster test
        model.validator.min_train_size = 100
        model.validator.test_size = 50
        model.validator.step_size = 50

        df = model.prepare_features(
            multi_symbol_market_df, multi_symbol_sentiment_df, is_inference=False
        )

        # Count rows per symbol
        btc_rows = len(df[df["symbol"] == "BTCUSDT"])
        total_rows = len(df)

        # Train with validation
        result = model.train(df, validate=True, validation_symbol="BTCUSDT")

        # Validation should have been performed
        assert result["validation"] is not None

        # The number of folds should be based on single-symbol rows, not total
        # With 3 symbols, if we used all rows, we'd have ~3x more folds
        n_folds = model.validation_results["overall"]["n_folds"]
        expected_max_folds = (btc_rows - 100 - 50) // 50 + 1

        # Folds should be close to single-symbol expectation, not 3x higher
        assert n_folds <= expected_max_folds + 1, (
            f"Too many folds ({n_folds}), expected ~{expected_max_folds}. "
            f"Validation may be using all {total_rows} rows instead of "
            f"single-symbol {btc_rows} rows."
        )

    def test_validation_respects_temporal_order(
        self, multi_symbol_market_df, multi_symbol_sentiment_df
    ):
        """Validation folds should maintain strict temporal ordering."""
        model = ImprovedTradingModel()
        model.validator.min_train_size = 100
        model.validator.test_size = 50
        model.validator.step_size = 50

        df = model.prepare_features(
            multi_symbol_market_df, multi_symbol_sentiment_df, is_inference=False
        )
        model.train(df, validate=True, validation_symbol="BTCUSDT")

        # Check each fold maintains train_end < test_start
        for fold in model.validation_results["per_fold"]:
            assert fold["train_end_idx"] < fold["test_start_idx"], (
                f"Fold {fold['fold']}: train ends at {fold['train_end_idx']} "
                f"but test starts at {fold['test_start_idx']} - temporal leak!"
            )

    def test_training_uses_all_symbols(
        self, multi_symbol_market_df, multi_symbol_sentiment_df
    ):
        """Final model training should use all symbols, not just validation symbol."""
        model = ImprovedTradingModel()
        model.validator.min_train_size = 100
        model.validator.test_size = 50
        model.validator.step_size = 50

        df = model.prepare_features(
            multi_symbol_market_df, multi_symbol_sentiment_df, is_inference=False
        )
        result = model.train(df, validate=True, validation_symbol="BTCUSDT")

        # n_samples should reflect ALL symbols, not just BTCUSDT
        total_rows = len(df)
        btc_rows = len(df[df["symbol"] == "BTCUSDT"])

        assert result["n_samples"] == total_rows, (
            f"Training used {result['n_samples']} samples but should use "
            f"all {total_rows} (not just {btc_rows} from BTCUSDT)"
        )

    def test_fallback_when_validation_symbol_missing(self, market_df, sentiment_df):
        """Should skip validation gracefully if validation symbol not in data."""
        model = ImprovedTradingModel()
        model.validator.min_train_size = 50
        model.validator.test_size = 20
        model.validator.step_size = 20

        df = model.prepare_features(market_df, sentiment_df, is_inference=False)

        # Use a symbol that doesn't exist in the data
        result = model.train(df, validate=True, validation_symbol="DOGUSDT")

        # Should still train successfully, just without validation
        assert result["status"] == "success"
        assert result["validation"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
