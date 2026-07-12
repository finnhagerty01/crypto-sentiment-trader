# tests/test_ensemble.py
"""
Tests for the TradingEnsemble and MultiTargetEnsemble classes.

Covers:
- Basic fit / predict / predict_proba contracts
- Probability calibration
- Soft vs hard voting
- Feature importance extraction
- Class-imbalance handling (scale_pos_weight)
- MultiTargetEnsemble direction + magnitude
- Edge cases (single class, minimal data)
- Integration with WalkForwardValidator to verify no leakage through ensemble
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.ensemble import TradingEnsemble, MultiTargetEnsemble
from src.models.validation import WalkForwardValidator


# ──────────────────────────── helpers ────────────────────────────


def _make_binary_dataset(
    n_samples: int = 500,
    n_features: int = 10,
    pos_frac: float = 0.3,
    seed: int = 42,
):
    """Create a synthetic binary classification dataset.

    Both classes are distributed evenly throughout the array so that
    walk-forward splits always see both classes in their training window.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    # Make target weakly correlated with first feature so models learn something
    prob = 1 / (1 + np.exp(-0.5 * X[:, 0]))
    y = (rng.rand(n_samples) < prob).astype(int)
    # Force approximate pos_frac by flipping RANDOM positives (not the first
    # N — doing that would starve early walk-forward folds of the minority
    # class and cause single-class training errors).
    current_frac = y.mean()
    if current_frac > pos_frac + 0.1:
        flip_n = int((current_frac - pos_frac) * n_samples)
        ones = np.where(y == 1)[0]
        chosen = rng.choice(ones, size=min(flip_n, len(ones)), replace=False)
        y[chosen] = 0
    feature_names = [f"feat_{i}" for i in range(n_features)]
    return X, y, feature_names


def _make_multiclass_dataset(
    n_samples: int = 500,
    n_features: int = 10,
    n_classes: int = 3,
    seed: int = 42,
):
    """Create a synthetic multi-class dataset."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    y = rng.randint(0, n_classes, n_samples)
    feature_names = [f"feat_{i}" for i in range(n_features)]
    return X, y, feature_names


# ──────────────────────────── fixtures ────────────────────────────


@pytest.fixture
def binary_data():
    return _make_binary_dataset()


@pytest.fixture
def imbalanced_data():
    """Heavily imbalanced: ~10 % positive."""
    return _make_binary_dataset(pos_frac=0.10)


@pytest.fixture
def multiclass_data():
    return _make_multiclass_dataset()


# ═══════════════════════ TradingEnsemble — fit / predict ═══════════════════════


class TestTradingEnsembleFitPredict:
    """Core API contract tests."""

    def test_fit_returns_self(self, binary_data):
        X, y, names = binary_data
        ens = TradingEnsemble(calibrate=False)
        result = ens.fit(X, y, feature_names=names)
        assert result is ens

    def test_predict_shape(self, binary_data):
        X, y, names = binary_data
        ens = TradingEnsemble(calibrate=False)
        ens.fit(X, y, feature_names=names)
        preds = ens.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_proba_shape(self, binary_data):
        X, y, names = binary_data
        ens = TradingEnsemble(calibrate=False)
        ens.fit(X, y, feature_names=names)
        probas = ens.predict_proba(X)
        assert probas.shape == (len(X), 2)

    def test_predict_proba_sums_to_one(self, binary_data):
        X, y, names = binary_data
        ens = TradingEnsemble(calibrate=False)
        ens.fit(X, y, feature_names=names)
        probas = ens.predict_proba(X)
        row_sums = probas.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_predict_proba_bounded_zero_one(self, binary_data):
        X, y, names = binary_data
        ens = TradingEnsemble(calibrate=False)
        ens.fit(X, y, feature_names=names)
        probas = ens.predict_proba(X)
        assert (probas >= 0.0).all()
        assert (probas <= 1.0).all()

    def test_predict_labels_match_classes(self, binary_data):
        X, y, names = binary_data
        ens = TradingEnsemble(calibrate=False)
        ens.fit(X, y, feature_names=names)
        preds = ens.predict(X)
        assert set(np.unique(preds)).issubset(set(np.unique(y)))

    def test_predict_before_fit_raises(self):
        ens = TradingEnsemble()
        with pytest.raises(RuntimeError, match="not fitted"):
            ens.predict(np.zeros((5, 3)))

    def test_predict_proba_before_fit_raises(self):
        ens = TradingEnsemble()
        with pytest.raises(RuntimeError, match="not fitted"):
            ens.predict_proba(np.zeros((5, 3)))


# ═══════════════════════ Calibration ═══════════════════════


class TestCalibration:
    """Tests specific to probability calibration."""

    def test_calibrated_probas_valid(self, binary_data):
        """Calibrated ensemble should still produce valid probabilities."""
        X, y, names = binary_data
        ens = TradingEnsemble(calibrate=True)
        ens.fit(X, y, feature_names=names)
        probas = ens.predict_proba(X)
        row_sums = probas.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)
        assert (probas >= 0.0).all()
        assert (probas <= 1.0).all()

    def test_calibrated_vs_uncalibrated_differ(self, binary_data):
        """
        Calibration should change the probability distribution somewhat.
        They shouldn't be identical (unless by extreme coincidence).
        """
        X, y, names = binary_data
        cal = TradingEnsemble(calibrate=True)
        cal.fit(X, y, feature_names=names)

        uncal = TradingEnsemble(calibrate=False)
        uncal.fit(X, y, feature_names=names)

        p_cal = cal.predict_proba(X)[:, 1]
        p_uncal = uncal.predict_proba(X)[:, 1]

        # They should NOT be identical
        assert not np.allclose(p_cal, p_uncal, atol=1e-4)


# ═══════════════════════ Voting modes ═══════════════════════


class TestVotingModes:
    """Tests for soft vs hard voting behaviour."""

    def test_soft_voting_produces_probas(self, binary_data):
        X, y, names = binary_data
        ens = TradingEnsemble(voting="soft", calibrate=False)
        ens.fit(X, y, feature_names=names)
        # Soft voting should produce continuous probabilities
        probas = ens.predict_proba(X)[:, 1]
        unique_vals = np.unique(np.round(probas, 4))
        # Should have many distinct probability values
        assert len(unique_vals) > 5

    def test_hard_voting_produces_predictions(self, binary_data):
        X, y, names = binary_data
        ens = TradingEnsemble(voting="hard", calibrate=False)
        ens.fit(X, y, feature_names=names)
        preds = ens.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})


# ═══════════════════════ Feature importances ═══════════════════════


class TestFeatureImportances:

    def test_importances_returned_as_series(self, binary_data):
        X, y, names = binary_data
        ens = TradingEnsemble(calibrate=False)
        ens.fit(X, y, feature_names=names)
        imp = ens.get_feature_importances(top_n=5)
        assert isinstance(imp, pd.Series)
        assert len(imp) == 5

    def test_importances_sorted_descending(self, binary_data):
        X, y, names = binary_data
        ens = TradingEnsemble(calibrate=False)
        ens.fit(X, y, feature_names=names)
        imp = ens.get_feature_importances()
        values = imp.values
        assert (values[:-1] >= values[1:]).all()

    def test_importances_sum_roughly_to_one(self, binary_data):
        """After per-model normalisation, averaged importances should sum ≈ 1."""
        X, y, names = binary_data
        ens = TradingEnsemble(calibrate=False)
        ens.fit(X, y, feature_names=names)
        total = ens.feature_importances_.sum()
        # Each model's importances are normalised to sum=1 before averaging,
        # so the average should also sum to ~1.
        assert 0.9 < total < 1.1

    def test_importances_have_correct_feature_names(self, binary_data):
        X, y, names = binary_data
        ens = TradingEnsemble(calibrate=False)
        ens.fit(X, y, feature_names=names)
        imp = ens.get_feature_importances()
        assert set(imp.index).issubset(set(names))

    def test_importances_before_fit_raises(self):
        ens = TradingEnsemble()
        with pytest.raises(ValueError, match="not available"):
            ens.get_feature_importances()

    def test_importances_without_names(self, binary_data):
        """Importances should still work without feature_names."""
        X, y, _ = binary_data
        ens = TradingEnsemble(calibrate=False)
        ens.fit(X, y)  # no feature_names
        imp = ens.get_feature_importances()
        assert isinstance(imp, pd.Series)
        assert len(imp) > 0


# ═══════════════════════ Class imbalance ═══════════════════════


class TestClassImbalance:

    def test_imbalanced_data_trains_without_error(self, imbalanced_data):
        X, y, names = imbalanced_data
        ens = TradingEnsemble(calibrate=False)
        ens.fit(X, y, feature_names=names)
        preds = ens.predict(X)
        assert len(preds) == len(X)

    def test_imbalanced_predicts_minority_class(self, imbalanced_data):
        """
        With balanced class weights, the ensemble should still predict some
        positive labels even on heavily imbalanced data.
        """
        X, y, names = imbalanced_data
        ens = TradingEnsemble(calibrate=False)
        ens.fit(X, y, feature_names=names)
        preds = ens.predict(X)
        # Should predict at least a few positives
        assert preds.sum() > 0

    def test_scale_pos_weight_set_dynamically(self, imbalanced_data):
        """Verify XGBoost's scale_pos_weight adapts to class distribution."""
        X, y, names = imbalanced_data
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        expected_weight = n_neg / max(n_pos, 1)

        ens = TradingEnsemble(calibrate=False)
        ens.fit(X, y, feature_names=names)

        # Access the XGBoost estimator from the voting classifier
        xgb_estimator = None
        for name, est in ens.ensemble_.named_estimators_.items():
            if name == "xgb":
                xgb_estimator = est
                break

        assert xgb_estimator is not None
        assert xgb_estimator.scale_pos_weight == pytest.approx(
            expected_weight, rel=0.01
        )


# ═══════════════════════ Single-class guard ═══════════════════════


class TestSingleClassGuard:

    def test_single_class_raises_valueerror(self):
        """Fitting with only one class should raise a clear error."""
        X = np.random.randn(100, 5)
        y = np.zeros(100, dtype=int)  # all class 0
        ens = TradingEnsemble(calibrate=False)
        with pytest.raises(ValueError, match="at least 2 classes"):
            ens.fit(X, y)


# ═══════════════════════ MultiTargetEnsemble ═══════════════════════


class TestMultiTargetEnsemble:

    def test_fit_and_predict(self, binary_data, multiclass_data):
        X_dir, y_dir, names = binary_data
        _, y_mag, _ = multiclass_data
        # Ensure same number of samples
        n = min(len(y_dir), len(y_mag))
        X = X_dir[:n]
        y_d = y_dir[:n]
        y_m = y_mag[:n]

        mte = MultiTargetEnsemble(
            direction_params={"calibrate": False},
            magnitude_params={"calibrate": False},
        )
        mte.fit(X, y_d, y_m, feature_names=names)
        result = mte.predict(X)

        assert "direction" in result
        assert "direction_proba" in result
        assert "magnitude" in result
        assert "magnitude_proba" in result
        assert result["direction"].shape == (n,)
        assert result["magnitude"].shape == (n,)

    def test_predict_before_fit_raises(self):
        mte = MultiTargetEnsemble()
        with pytest.raises(RuntimeError, match="not fitted"):
            mte.predict(np.zeros((5, 3)))

    def test_direction_is_binary(self, binary_data):
        X, y, names = binary_data
        # Use same y for magnitude as a placeholder (3-class)
        y_mag = (y + np.random.RandomState(0).randint(0, 2, len(y))) % 3

        mte = MultiTargetEnsemble(
            direction_params={"calibrate": False},
            magnitude_params={"calibrate": False},
        )
        mte.fit(X, y, y_mag, feature_names=names)
        result = mte.predict(X)

        assert set(np.unique(result["direction"])).issubset({0, 1})


# ═══════════════════════ Integration: Ensemble + WalkForward ═══════════════════════


class TestEnsembleWithWalkForward:
    """
    Verify the ensemble works correctly with walk-forward validation,
    and that no data leaks through the pipeline.
    """

    def test_walkforward_with_ensemble_no_leakage(self, binary_data):
        """
        Run walk-forward validation using TradingEnsemble and confirm that
        every fold's train indices are strictly before its test indices.
        """
        X, y, names = binary_data
        df = pd.DataFrame(X, columns=names)
        df["target"] = y

        wfv = WalkForwardValidator(
            min_train_size=200, test_size=50, step_size=50
        )
        ens = TradingEnsemble(calibrate=False)
        results = wfv.validate(ens, df, names)

        assert results["overall"]["n_folds"] > 0
        for fold in results["per_fold"]:
            assert fold["train_end_idx"] < fold["test_start_idx"]

    def test_walkforward_with_purge_gap(self, binary_data):
        """Purge gap should be respected through the ensemble pipeline."""
        X, y, names = binary_data
        df = pd.DataFrame(X, columns=names)
        df["target"] = y

        gap = 12
        wfv = WalkForwardValidator(
            min_train_size=200, test_size=50, step_size=50, purge_gap=gap
        )
        ens = TradingEnsemble(calibrate=False)
        results = wfv.validate(ens, df, names)

        for fold in results["per_fold"]:
            distance = fold["test_start_idx"] - fold["train_end_idx"]
            assert distance > gap

    def test_walkforward_returns_valid_probabilities(self, binary_data):
        """Walk-forward probabilities should be in [0, 1]."""
        X, y, names = binary_data
        df = pd.DataFrame(X, columns=names)
        df["target"] = y

        wfv = WalkForwardValidator(
            min_train_size=200, test_size=50, step_size=50
        )
        ens = TradingEnsemble(calibrate=False)
        results = wfv.validate(ens, df, names)

        probas = results["probabilities"]
        assert (probas >= 0.0).all()
        assert (probas <= 1.0).all()


# ═══════════════════════ Custom parameters ═══════════════════════


class TestCustomParameters:
    """Verify custom hyperparameters are forwarded correctly."""

    def test_custom_rf_params(self, binary_data):
        X, y, names = binary_data
        custom_rf = {
            "n_estimators": 10,
            "max_depth": 3,
            "random_state": 42,
            "n_jobs": 1,
        }
        ens = TradingEnsemble(rf_params=custom_rf, calibrate=False)
        ens.fit(X, y, feature_names=names)
        preds = ens.predict(X)
        assert len(preds) == len(X)

    def test_custom_xgb_params(self, binary_data):
        X, y, names = binary_data
        custom_xgb = {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "scale_pos_weight": 1,
            "random_state": 42,
            "n_jobs": 1,
            "eval_metric": "logloss",
        }
        ens = TradingEnsemble(xgb_params=custom_xgb, calibrate=False)
        ens.fit(X, y, feature_names=names)
        preds = ens.predict(X)
        assert len(preds) == len(X)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
