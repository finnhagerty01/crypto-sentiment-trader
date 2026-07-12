# tests/test_tuning.py
"""
Tests for hyperparameter tuning modules.

Covers:
- TradingModelTuner (Optuna-based Bayesian optimization)
- GridSearchWalkForward (exhaustive grid search)
- Return structure validation
- Score bounds
- No data leakage (walk-forward validator is used as objective)
- Timeout and small n_trials edge cases
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.tuning import TradingModelTuner, GridSearchWalkForward
from src.models.validation import WalkForwardValidator


# ──────────────────────────── helpers ────────────────────────────


def _make_binary_dataset(
    n_samples: int = 300,
    n_features: int = 5,
    seed: int = 42,
):
    """Create a synthetic binary classification dataset.

    Both classes are distributed throughout the array so walk-forward
    splits always see both classes in their training window.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    prob = 1 / (1 + np.exp(-0.5 * X[:, 0]))
    y = (rng.rand(n_samples) < prob).astype(int)
    feature_names = [f"f{i}" for i in range(n_features)]
    return X, y, feature_names


# ──────────────────────────── fixtures ────────────────────────────


@pytest.fixture
def binary_data():
    return _make_binary_dataset()


@pytest.fixture
def small_validator():
    """Walk-forward validator with small windows for fast tests."""
    return WalkForwardValidator(
        min_train_size=50,
        test_size=30,
        step_size=30,
    )


# ═══════════════════════ TradingModelTuner — XGBoost ═══════════════════════


class TestTuneXGBoost:
    """Tests for tune_xgboost method."""

    def test_returns_best_params(self, binary_data, small_validator):
        """Return dict should contain 'best_params' and 'best_score'."""
        X, y, names = binary_data
        tuner = TradingModelTuner(n_trials=3, validation_metric="precision")
        result = tuner.tune_xgboost(X, y, small_validator, feature_names=names)

        assert "best_params" in result
        assert "best_score" in result
        assert "study" in result
        assert isinstance(result["best_params"], dict)

    def test_best_score_is_float_in_range(self, binary_data, small_validator):
        """Best score should be a float in [0, 1]."""
        X, y, names = binary_data
        tuner = TradingModelTuner(n_trials=3, validation_metric="precision")
        result = tuner.tune_xgboost(X, y, small_validator, feature_names=names)

        assert isinstance(result["best_score"], float)
        assert 0.0 <= result["best_score"] <= 1.0

    def test_small_n_trials_still_works(self, binary_data, small_validator):
        """Even n_trials=2 should produce a valid result."""
        X, y, names = binary_data
        tuner = TradingModelTuner(n_trials=2, validation_metric="f1")
        result = tuner.tune_xgboost(X, y, small_validator, feature_names=names)

        assert result["best_params"] is not None
        assert result["best_score"] >= 0.0

    def test_no_data_leakage(self, binary_data, small_validator):
        """Verify the walk-forward validator is actually called (not bypassed)."""
        X, y, names = binary_data
        tuner = TradingModelTuner(n_trials=2, validation_metric="precision")

        with patch.object(
            small_validator, "validate", wraps=small_validator.validate
        ) as mock_validate:
            tuner.tune_xgboost(X, y, small_validator, feature_names=names)
            assert mock_validate.call_count >= 2  # called once per trial

    def test_without_feature_names(self, binary_data, small_validator):
        """Should work when feature_names is None."""
        X, y, _ = binary_data
        tuner = TradingModelTuner(n_trials=2, validation_metric="accuracy")
        result = tuner.tune_xgboost(X, y, small_validator)

        assert result["best_params"] is not None


# ═══════════════════════ TradingModelTuner — LightGBM ═══════════════════════


class TestTuneLightGBM:
    """Tests for tune_lightgbm method."""

    def test_returns_best_params(self, binary_data, small_validator):
        X, y, names = binary_data
        tuner = TradingModelTuner(n_trials=3, validation_metric="precision")
        result = tuner.tune_lightgbm(X, y, small_validator, feature_names=names)

        assert "best_params" in result
        assert "best_score" in result
        assert "study" in result

    def test_best_score_is_float_in_range(self, binary_data, small_validator):
        X, y, names = binary_data
        tuner = TradingModelTuner(n_trials=3, validation_metric="recall")
        result = tuner.tune_lightgbm(X, y, small_validator, feature_names=names)

        assert isinstance(result["best_score"], float)
        assert 0.0 <= result["best_score"] <= 1.0

    def test_no_data_leakage(self, binary_data, small_validator):
        X, y, names = binary_data
        tuner = TradingModelTuner(n_trials=2, validation_metric="precision")

        with patch.object(
            small_validator, "validate", wraps=small_validator.validate
        ) as mock_validate:
            tuner.tune_lightgbm(X, y, small_validator, feature_names=names)
            assert mock_validate.call_count >= 2


# ═══════════════════════ TradingModelTuner — Timeout ═══════════════════════


class TestTunerTimeout:
    """Tests for timeout behavior."""

    def test_timeout_limits_study(self, binary_data, small_validator):
        """With a very short timeout, the study should stop early."""
        X, y, names = binary_data
        tuner = TradingModelTuner(
            n_trials=1000,  # high trial count
            validation_metric="precision",
            timeout=5,  # but only 5 seconds
        )
        result = tuner.tune_xgboost(X, y, small_validator, feature_names=names)

        # Should still return valid results even if not all trials ran
        assert result["best_params"] is not None
        assert result["best_score"] >= 0.0
        # The study should have fewer trials than requested
        assert len(result["study"].trials) < 1000


# ═══════════════════════ GridSearchWalkForward ═══════════════════════


class TestGridSearchWalkForward:
    """Tests for grid search with walk-forward validation."""

    def test_returns_expected_keys(self, binary_data, small_validator):
        """Return dict should have 'best_params', 'best_score', 'all_results'."""
        from sklearn.ensemble import RandomForestClassifier

        X, y, names = binary_data
        grid = GridSearchWalkForward(
            {"n_estimators": [10, 20], "max_depth": [2, 3], "random_state": [42]}
        )
        result = grid.search(
            RandomForestClassifier, X, y, small_validator,
            metric="precision", feature_names=names,
        )

        assert "best_params" in result
        assert "best_score" in result
        assert "all_results" in result

    def test_all_results_length_matches_grid(self, binary_data, small_validator):
        """Number of results should equal the grid size."""
        from sklearn.ensemble import RandomForestClassifier

        X, y, names = binary_data
        param_grid = {
            "n_estimators": [10, 20],
            "max_depth": [2, 3],
            "random_state": [42],
        }
        grid = GridSearchWalkForward(param_grid)
        result = grid.search(
            RandomForestClassifier, X, y, small_validator,
            metric="precision", feature_names=names,
        )

        expected_count = 2 * 2 * 1  # 4 combinations
        assert len(result["all_results"]) == expected_count

    def test_best_score_is_maximum(self, binary_data, small_validator):
        """Best score should be >= all other scores."""
        from sklearn.ensemble import RandomForestClassifier

        X, y, names = binary_data
        grid = GridSearchWalkForward(
            {"n_estimators": [10, 50], "max_depth": [2, 4], "random_state": [42]}
        )
        result = grid.search(
            RandomForestClassifier, X, y, small_validator,
            metric="precision", feature_names=names,
        )

        all_scores = [r["score"] for r in result["all_results"]]
        assert result["best_score"] >= max(all_scores) - 1e-9

    def test_no_data_leakage(self, binary_data, small_validator):
        """Verify the walk-forward validator is used for every grid point."""
        from sklearn.ensemble import RandomForestClassifier

        X, y, names = binary_data
        grid = GridSearchWalkForward(
            {"n_estimators": [10, 20], "random_state": [42]}
        )

        with patch.object(
            small_validator, "validate", wraps=small_validator.validate
        ) as mock_validate:
            grid.search(
                RandomForestClassifier, X, y, small_validator,
                metric="precision", feature_names=names,
            )
            assert mock_validate.call_count == 2  # 2 param combos

    def test_single_param_combination(self, binary_data, small_validator):
        """Grid with a single combination should still work."""
        from sklearn.ensemble import RandomForestClassifier

        X, y, names = binary_data
        grid = GridSearchWalkForward(
            {"n_estimators": [10], "random_state": [42]}
        )
        result = grid.search(
            RandomForestClassifier, X, y, small_validator,
            metric="f1", feature_names=names,
        )

        assert len(result["all_results"]) == 1
        assert result["best_params"] is not None

    def test_without_feature_names(self, binary_data, small_validator):
        """Should work when feature_names is None."""
        from sklearn.ensemble import RandomForestClassifier

        X, y, _ = binary_data
        grid = GridSearchWalkForward(
            {"n_estimators": [10], "random_state": [42]}
        )
        result = grid.search(
            RandomForestClassifier, X, y, small_validator,
            metric="accuracy",
        )

        assert result["best_params"] is not None


# ═══════════════════════ Parameter validation ═══════════════════════


class TestParameterValidation:
    """Edge-case and constructor tests."""

    def test_tuner_accepts_all_valid_metrics(self):
        """All four supported metrics should be accepted."""
        for metric in ["precision", "recall", "f1", "accuracy"]:
            tuner = TradingModelTuner(n_trials=1, validation_metric=metric)
            assert tuner.validation_metric == metric

    def test_tuner_default_values(self):
        tuner = TradingModelTuner()
        assert tuner.n_trials == 50
        assert tuner.validation_metric == "precision"
        assert tuner.timeout is None

    def test_grid_search_empty_grid_runs_default(self, binary_data, small_validator):
        """An empty param grid yields one combo (default params)."""
        from sklearn.ensemble import RandomForestClassifier

        X, y, names = binary_data
        grid = GridSearchWalkForward({})
        result = grid.search(
            RandomForestClassifier, X, y, small_validator,
            metric="precision", feature_names=names,
        )

        # ParameterGrid({}) yields [{}] — one combination with no overrides
        assert result["best_params"] == {}
        assert len(result["all_results"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
