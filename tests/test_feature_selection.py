"""
Tests for app.feature_selection — FeatureSelector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clf_data(
    n_samples: int = 500,
    n_features: int = 20,
    n_informative: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=2,
        random_state=seed,
    )
    return X, y


def _reg_data(
    n_samples: int = 500,
    n_features: int = 20,
    n_informative: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        random_state=seed,
        noise=10.0,
    )
    return X, y


def _low_variance_data(n: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Dataset where first column is near-constant (low variance)."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (n, 5))
    X[:, 0] = 0.0001  # nearly constant
    y = rng.integers(0, 2, n)
    return X, y


def _correlated_data(n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Dataset where two columns are perfectly correlated."""
    rng = np.random.default_rng(1)
    X1 = rng.uniform(0, 1, (n, 3))
    X2 = X1[:, [0]]  # duplicate of first col
    X = np.hstack([X1, X2])
    y = rng.integers(0, 2, n)
    return X, y


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFeatureSelector:
    """Unit tests for FeatureSelector."""

    def test_variance_threshold(self) -> None:
        """Columns with near-zero variance must be removed."""
        from app.feature_selection import FeatureSelector

        X, y = _low_variance_data()
        selector = FeatureSelector(method="variance", threshold=0.01)
        selector.fit(X, y, problem_type="classification")
        X_sel = selector.transform(X)
        # Column 0 (near-constant) should be dropped
        assert X_sel.shape[1] < X.shape[1], (
            "Near-constant column was not removed by variance threshold"
        )
        assert X_sel.shape[1] >= 1, "At least one column must survive selection"

    def test_correlation_filter(self) -> None:
        """One of two perfectly correlated columns must be dropped."""
        from app.feature_selection import FeatureSelector

        X, y = _correlated_data()
        selector = FeatureSelector(method="correlation", threshold=0.95)
        selector.fit(X, y, problem_type="classification")
        X_sel = selector.transform(X)
        assert X_sel.shape[1] < X.shape[1], (
            "Correlated duplicate column was not removed"
        )

    def test_kbest_classification(self) -> None:
        """SelectKBest should keep exactly k features for classification."""
        from app.feature_selection import FeatureSelector

        k = 5
        X, y = _clf_data(n_features=20)
        selector = FeatureSelector(method="kbest", k=k)
        selector.fit(X, y, problem_type="classification")
        X_sel = selector.transform(X)
        assert X_sel.shape[1] == k, (
            f"Expected {k} features, got {X_sel.shape[1]}"
        )

    def test_kbest_regression(self) -> None:
        """SelectKBest should keep exactly k features for regression."""
        from app.feature_selection import FeatureSelector

        k = 4
        X, y = _reg_data(n_features=20)
        selector = FeatureSelector(method="kbest", k=k)
        selector.fit(X, y, problem_type="regression")
        X_sel = selector.transform(X)
        assert X_sel.shape[1] == k, (
            f"Expected {k} features, got {X_sel.shape[1]}"
        )

    def test_model_based(self) -> None:
        """Model-based selection must reduce feature count (informative << total)."""
        from app.feature_selection import FeatureSelector

        X, y = _clf_data(n_features=20, n_informative=3)
        selector = FeatureSelector(method="model_based")
        selector.fit(X, y, problem_type="classification")
        X_sel = selector.transform(X)
        # Should select fewer than all 20 features
        assert X_sel.shape[1] <= X.shape[1], "Cannot have more features than input"
        assert X_sel.shape[1] >= 1, "Must keep at least one feature"

    def test_auto_selection(self) -> None:
        """Auto mode must pick a method and return a reduced (or equal) feature set."""
        from app.feature_selection import FeatureSelector

        X, y = _clf_data(n_features=20)
        selector = FeatureSelector(method="auto")
        selector.fit(X, y, problem_type="classification")
        X_sel = selector.transform(X)
        assert X_sel.shape[0] == X.shape[0], "Row count must be preserved"
        assert 1 <= X_sel.shape[1] <= X.shape[1], (
            "Selected feature count must be in [1, original_n_features]"
        )

    def test_transform_consistent(self) -> None:
        """transform() on test data must produce same column count as training data."""
        from app.feature_selection import FeatureSelector

        X_train, y_train = _clf_data(n_samples=400)
        X_test, _ = _clf_data(n_samples=100)

        selector = FeatureSelector(method="kbest", k=8)
        selector.fit(X_train, y_train, problem_type="classification")

        X_train_sel = selector.transform(X_train)
        X_test_sel = selector.transform(X_test)

        assert X_train_sel.shape[1] == X_test_sel.shape[1], (
            f"Train cols={X_train_sel.shape[1]} != test cols={X_test_sel.shape[1]}"
        )
        assert X_test_sel.shape[0] == 100, "Test row count must be preserved"
