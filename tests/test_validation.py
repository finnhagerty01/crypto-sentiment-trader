# tests/test_validation.py
"""
Tests for walk-forward validation and purged k-fold.

Heavy emphasis on temporal ordering and data-leakage prevention:
every test that touches indices asserts train_max < test_min.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.models.validation import WalkForwardValidator, PurgedKFold


# ──────────────────────────── fixtures ────────────────────────────


@pytest.fixture
def simple_df():
    """700-row DataFrame with monotonic timestamps and a binary target."""
    np.random.seed(42)
    n = 700
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
            "f3": np.random.randn(n),
            "target": np.random.randint(0, 2, n),
        }
    )
    return df


@pytest.fixture
def small_df():
    """50-row DataFrame — too small for the default min_train_size=500."""
    np.random.seed(99)
    n = 50
    return pd.DataFrame(
        {
            "f1": np.random.randn(n),
            "target": np.random.randint(0, 2, n),
        }
    )


@pytest.fixture
def features():
    return ["f1", "f2", "f3"]


# ═══════════════════════ WalkForwardValidator.split ═══════════════════════


class TestWalkForwardSplit:
    """Tests for the split generator."""

    def test_basic_split_yields_folds(self):
        """split() should yield at least one fold for adequate data."""
        wfv = WalkForwardValidator(
            min_train_size=100, test_size=50, step_size=50
        )
        folds = list(wfv.split(300))
        assert len(folds) >= 1

    def test_train_always_before_test(self):
        """Core leakage guard: every training index must be < every test index."""
        wfv = WalkForwardValidator(
            min_train_size=100, test_size=50, step_size=25
        )
        for train_idx, test_idx in wfv.split(500):
            assert train_idx.max() < test_idx.min(), (
                f"Leakage: train_max={train_idx.max()} >= test_min={test_idx.min()}"
            )

    def test_purge_gap_separates_train_test(self):
        """With purge_gap > 0, there must be a gap between train and test."""
        gap = 10
        wfv = WalkForwardValidator(
            min_train_size=100, test_size=50, step_size=25, purge_gap=gap
        )
        for train_idx, test_idx in wfv.split(500):
            distance = test_idx.min() - train_idx.max()
            assert distance > gap, (
                f"Gap too small: distance={distance}, expected > {gap}"
            )

    def test_expanding_window_starts_at_zero(self):
        """In expanding mode, train always starts at index 0."""
        wfv = WalkForwardValidator(
            min_train_size=100, test_size=50, step_size=50, expanding=True
        )
        for train_idx, _ in wfv.split(500):
            assert train_idx[0] == 0

    def test_rolling_window_fixed_size(self):
        """In rolling mode, train size should equal min_train_size."""
        train_size = 100
        wfv = WalkForwardValidator(
            min_train_size=train_size,
            test_size=50,
            step_size=50,
            expanding=False,
        )
        for train_idx, _ in wfv.split(500):
            assert len(train_idx) == train_size

    def test_expanding_window_grows(self):
        """In expanding mode, later folds should have more training data."""
        wfv = WalkForwardValidator(
            min_train_size=100, test_size=50, step_size=50, expanding=True
        )
        folds = list(wfv.split(500))
        assert len(folds) >= 2
        assert len(folds[-1][0]) > len(folds[0][0])

    def test_no_overlap_between_train_and_test(self):
        """Train and test index sets must be completely disjoint."""
        wfv = WalkForwardValidator(
            min_train_size=100, test_size=50, step_size=25
        )
        for train_idx, test_idx in wfv.split(500):
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0, f"Overlap found: {overlap}"

    def test_test_size_respected(self):
        """Each test fold should have exactly test_size rows (or fewer at end)."""
        wfv = WalkForwardValidator(
            min_train_size=100, test_size=50, step_size=50
        )
        for _, test_idx in wfv.split(300):
            assert len(test_idx) <= 50

    def test_small_dataset_yields_no_folds(self):
        """Dataset smaller than min_train + test should yield nothing."""
        wfv = WalkForwardValidator(
            min_train_size=100, test_size=50, step_size=25
        )
        folds = list(wfv.split(100))  # need >= 150
        assert len(folds) == 0

    def test_all_test_indices_in_range(self):
        """No test index should exceed n_samples - 1."""
        n = 400
        wfv = WalkForwardValidator(
            min_train_size=100, test_size=50, step_size=25
        )
        for _, test_idx in wfv.split(n):
            assert test_idx.max() < n

    def test_step_size_advances_folds(self):
        """Consecutive test folds should start step_size apart."""
        step = 30
        wfv = WalkForwardValidator(
            min_train_size=100, test_size=50, step_size=step
        )
        folds = list(wfv.split(500))
        if len(folds) >= 2:
            for i in range(1, len(folds)):
                prev_test_start = folds[i - 1][1][0]
                curr_test_start = folds[i][1][0]
                assert curr_test_start - prev_test_start == step


# ═══════════════════════ WalkForwardValidator.validate ═══════════════════════


class TestWalkForwardValidate:
    """Tests for the validate() method with real sklearn models."""

    def test_validate_returns_expected_keys(self, simple_df, features):
        wfv = WalkForwardValidator(
            min_train_size=200, test_size=100, step_size=100
        )
        model = LogisticRegression(max_iter=200, random_state=42)
        results = wfv.validate(model, simple_df, features)

        assert "overall" in results
        assert "per_fold" in results
        assert "predictions" in results
        assert "actuals" in results
        assert "probabilities" in results

    def test_overall_metrics_are_floats(self, simple_df, features):
        wfv = WalkForwardValidator(
            min_train_size=200, test_size=100, step_size=100
        )
        model = LogisticRegression(max_iter=200, random_state=42)
        results = wfv.validate(model, simple_df, features)

        for key in ["precision", "recall", "f1", "accuracy"]:
            assert isinstance(results["overall"][key], float)
            assert 0.0 <= results["overall"][key] <= 1.0

    def test_predictions_length_matches_actuals(self, simple_df, features):
        wfv = WalkForwardValidator(
            min_train_size=200, test_size=100, step_size=100
        )
        model = RandomForestClassifier(
            n_estimators=10, max_depth=3, random_state=42
        )
        results = wfv.validate(model, simple_df, features)

        assert len(results["predictions"]) == len(results["actuals"])
        assert len(results["probabilities"]) == len(results["actuals"])
        assert results["overall"]["n_samples"] == len(results["actuals"])

    def test_validate_no_data_leakage_in_folds(self, simple_df, features):
        """
        Verify that for every fold the model was trained on indices strictly
        before the test indices by checking per_fold metadata.
        """
        wfv = WalkForwardValidator(
            min_train_size=200, test_size=100, step_size=50
        )
        model = LogisticRegression(max_iter=200, random_state=42)
        results = wfv.validate(model, simple_df, features)

        for fold in results["per_fold"]:
            assert fold["train_end_idx"] < fold["test_start_idx"], (
                f"Fold {fold['fold']}: train leaked into test. "
                f"train_end={fold['train_end_idx']}, "
                f"test_start={fold['test_start_idx']}"
            )

    def test_validate_with_purge_gap(self, simple_df, features):
        """Purge gap should separate train_end from test_start."""
        gap = 12
        wfv = WalkForwardValidator(
            min_train_size=200, test_size=100, step_size=50, purge_gap=gap
        )
        model = LogisticRegression(max_iter=200, random_state=42)
        results = wfv.validate(model, simple_df, features)

        for fold in results["per_fold"]:
            distance = fold["test_start_idx"] - fold["train_end_idx"]
            assert distance > gap, (
                f"Fold {fold['fold']}: purge gap not enforced. "
                f"distance={distance}, expected > {gap}"
            )

    def test_validate_small_data_returns_empty(self, small_df):
        """With insufficient data, validate should return zero-fold results."""
        wfv = WalkForwardValidator(min_train_size=500, test_size=168)
        model = LogisticRegression(max_iter=200)
        results = wfv.validate(model, small_df, ["f1"])

        assert results["overall"]["n_folds"] == 0
        assert results["overall"]["n_samples"] == 0
        assert len(results["predictions"]) == 0

    def test_validate_missing_feature_raises(self, simple_df):
        wfv = WalkForwardValidator(min_train_size=200, test_size=100)
        model = LogisticRegression()
        with pytest.raises(ValueError, match="Missing feature"):
            wfv.validate(model, simple_df, ["nonexistent_col"])

    def test_validate_missing_target_raises(self, simple_df, features):
        wfv = WalkForwardValidator(min_train_size=200, test_size=100)
        model = LogisticRegression()
        with pytest.raises(ValueError, match="Missing target"):
            wfv.validate(model, simple_df, features, target="nonexistent")

    def test_folds_are_independent(self, simple_df, features):
        """
        Running validate twice with the same inputs should give identical
        results, proving that model state does not leak across folds (we
        clone the model).
        """
        wfv = WalkForwardValidator(
            min_train_size=200, test_size=100, step_size=100
        )
        model = RandomForestClassifier(
            n_estimators=10, max_depth=3, random_state=42
        )
        r1 = wfv.validate(model, simple_df, features)
        r2 = wfv.validate(model, simple_df, features)

        np.testing.assert_array_equal(r1["predictions"], r2["predictions"])
        np.testing.assert_array_equal(r1["actuals"], r2["actuals"])


# ═══════════════════════ PurgedKFold ═══════════════════════


class TestPurgedKFold:
    """Tests for PurgedKFold."""

    def test_correct_number_of_splits(self):
        pkf = PurgedKFold(n_splits=5, purge_gap=0)
        folds = list(pkf.split(500))
        assert len(folds) == 5

    def test_no_overlap_between_train_and_test(self):
        pkf = PurgedKFold(n_splits=5, purge_gap=10)
        for train_idx, test_idx in pkf.split(500):
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0

    def test_purge_gap_removes_neighbors(self):
        """
        No training index should be within purge_gap rows of any test index.
        """
        gap = 24
        pkf = PurgedKFold(n_splits=5, purge_gap=gap)
        for train_idx, test_idx in pkf.split(500):
            test_min, test_max = test_idx.min(), test_idx.max()

            # No train index within [test_min - gap, test_max + gap]
            forbidden_zone = set(
                range(max(0, test_min - gap), min(500, test_max + gap + 1))
            )
            leaked = set(train_idx.tolist()) & forbidden_zone
            # The only allowed indices in the forbidden zone are the test
            # indices themselves (which aren't in train), so leaked should
            # be empty.
            assert len(leaked) == 0, (
                f"Training indices within purge zone: {sorted(leaked)[:5]}..."
            )

    def test_all_indices_covered(self):
        """Every index [0, n) should appear in at least one test fold."""
        n = 500
        pkf = PurgedKFold(n_splits=5, purge_gap=0)
        all_test = set()
        for _, test_idx in pkf.split(n):
            all_test.update(test_idx.tolist())
        assert all_test == set(range(n))

    def test_test_folds_are_disjoint(self):
        """Test folds should not overlap with each other."""
        pkf = PurgedKFold(n_splits=5, purge_gap=10)
        test_sets = []
        for _, test_idx in pkf.split(500):
            test_sets.append(set(test_idx.tolist()))

        for i in range(len(test_sets)):
            for j in range(i + 1, len(test_sets)):
                assert test_sets[i].isdisjoint(test_sets[j])

    def test_purge_gap_zero_behaves_like_kfold(self):
        """With purge_gap=0, train + test should cover all indices per fold."""
        n = 100
        pkf = PurgedKFold(n_splits=5, purge_gap=0)
        for train_idx, test_idx in pkf.split(n):
            combined = set(train_idx.tolist()) | set(test_idx.tolist())
            assert combined == set(range(n))

    def test_purge_gap_larger_than_fold_still_works(self):
        """A very large purge gap should still produce valid (smaller) folds."""
        pkf = PurgedKFold(n_splits=3, purge_gap=100)
        folds = list(pkf.split(500))
        for train_idx, test_idx in folds:
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0

    def test_invalid_n_splits_raises(self):
        with pytest.raises(ValueError):
            PurgedKFold(n_splits=1)

    def test_invalid_purge_gap_raises(self):
        with pytest.raises(ValueError):
            PurgedKFold(n_splits=5, purge_gap=-1)

    def test_too_few_samples_raises(self):
        pkf = PurgedKFold(n_splits=10)
        with pytest.raises(ValueError):
            list(pkf.split(5))


# ═══════════════════════ Parameter validation ═══════════════════════


class TestParameterValidation:
    """Edge-case tests for constructor arguments."""

    def test_wfv_min_train_size_zero_raises(self):
        with pytest.raises(ValueError):
            WalkForwardValidator(min_train_size=0)

    def test_wfv_test_size_zero_raises(self):
        with pytest.raises(ValueError):
            WalkForwardValidator(test_size=0)

    def test_wfv_step_size_zero_raises(self):
        with pytest.raises(ValueError):
            WalkForwardValidator(step_size=0)

    def test_wfv_negative_purge_gap_raises(self):
        with pytest.raises(ValueError):
            WalkForwardValidator(purge_gap=-1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
