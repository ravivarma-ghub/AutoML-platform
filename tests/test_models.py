"""
Tests for app.models — ClassificationLibrary, RegressionLibrary, HyperparameterTuner.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _optuna_available() -> bool:
    try:
        import optuna  # noqa: F401
        return True
    except ImportError:
        return False


def _clf_arrays(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    return make_classification(
        n_samples=n, n_features=10, n_informative=5, random_state=seed
    )


def _reg_arrays(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    return make_regression(
        n_samples=n, n_features=10, n_informative=5, random_state=seed, noise=5.0
    )


# ---------------------------------------------------------------------------
# Classification library tests
# ---------------------------------------------------------------------------

class TestClassificationLibrary:
    """Tests for ClassificationLibrary."""

    def test_all_models_available(self) -> None:
        """Library must expose at least the standard classification models."""
        from app.models import ClassificationLibrary

        lib = ClassificationLibrary()
        models = lib.get_models()
        expected_keys = {
            "logistic_regression",
            "random_forest",
            "gradient_boosting",
        }
        available = set(models.keys())
        missing = expected_keys - available
        assert not missing, (
            f"Missing classification models: {missing}. Available: {available}"
        )

    def test_hyperparameter_grids_exist(self) -> None:
        """Every registered model must have a non-empty hyperparameter grid."""
        from app.models import ClassificationLibrary

        lib = ClassificationLibrary()
        grids = lib.get_param_grids()
        models = lib.get_models()
        for name in models:
            assert name in grids, f"No param grid for model '{name}'"
            assert len(grids[name]) > 0, f"Param grid for '{name}' is empty"

    def test_model_trains_on_data(self) -> None:
        """Each classification model must fit without error and produce predictions."""
        from app.models import ClassificationLibrary

        X, y = _clf_arrays()
        lib = ClassificationLibrary()
        models = lib.get_models()

        for name, model in models.items():
            model.fit(X, y)
            preds = model.predict(X)
            assert len(preds) == len(y), (
                f"Model '{name}' returned wrong number of predictions"
            )
            assert set(preds).issubset({0, 1}), (
                f"Model '{name}' produced invalid class labels"
            )


# ---------------------------------------------------------------------------
# Regression library tests
# ---------------------------------------------------------------------------

class TestRegressionLibrary:
    """Tests for RegressionLibrary."""

    def test_all_models_available(self) -> None:
        """Library must expose at least the standard regression models."""
        from app.models import RegressionLibrary

        lib = RegressionLibrary()
        models = lib.get_models()
        expected_keys = {
            "linear_regression",
            "random_forest",
            "gradient_boosting",
        }
        available = set(models.keys())
        missing = expected_keys - available
        assert not missing, (
            f"Missing regression models: {missing}. Available: {available}"
        )

    def test_model_trains_on_data(self) -> None:
        """Each regression model must fit without error and produce predictions."""
        from app.models import RegressionLibrary

        X, y = _reg_arrays()
        lib = RegressionLibrary()
        models = lib.get_models()

        for name, model in models.items():
            model.fit(X, y)
            preds = model.predict(X)
            assert len(preds) == len(y), (
                f"Regression model '{name}' returned wrong number of predictions"
            )
            assert not np.any(np.isnan(preds)), (
                f"Regression model '{name}' returned NaN predictions"
            )


# ---------------------------------------------------------------------------
# HyperparameterTuner tests
# ---------------------------------------------------------------------------

class TestHyperparameterTuner:
    """Tests for HyperparameterTuner — random search, grid search, optuna."""

    def test_random_search(self) -> None:
        """RandomizedSearchCV must return best params and a fitted estimator."""
        from app.models import ClassificationLibrary, HyperparameterTuner
        from sklearn.linear_model import LogisticRegression

        X, y = _clf_arrays(n=300)
        model = LogisticRegression(max_iter=200)
        param_grid = {"C": [0.01, 0.1, 1.0, 10.0]}
        tuner = HyperparameterTuner(method="random", n_iter=4, cv=2)
        best_model, best_params = tuner.tune(model, param_grid, X, y)

        assert best_model is not None, "Tuner must return a fitted model"
        assert isinstance(best_params, dict), "best_params must be a dict"
        assert "C" in best_params, "best_params must contain the tuned parameter 'C'"
        # Verify model can predict
        preds = best_model.predict(X)
        assert len(preds) == len(y)

    def test_grid_search(self) -> None:
        """GridSearchCV must exhaustively search and return the best estimator."""
        from app.models import HyperparameterTuner
        from sklearn.tree import DecisionTreeClassifier

        X, y = _clf_arrays(n=200)
        model = DecisionTreeClassifier()
        param_grid = {"max_depth": [2, 4, 6], "min_samples_split": [2, 5]}
        tuner = HyperparameterTuner(method="grid", cv=2)
        best_model, best_params = tuner.tune(model, param_grid, X, y)

        assert best_model is not None
        assert isinstance(best_params, dict)
        assert "max_depth" in best_params
        assert best_params["max_depth"] in [2, 4, 6]

    @pytest.mark.skipif(
        not _optuna_available(),
        reason="optuna not installed",
    )
    def test_optuna_search(self) -> None:
        """Optuna-based tuning must return a fitted model and best params."""
        from app.models import HyperparameterTuner
        from sklearn.linear_model import Ridge

        X, y = _reg_arrays(n=300)
        param_grid = {"alpha": {"type": "float", "low": 0.001, "high": 10.0, "log": True}}
        tuner = HyperparameterTuner(method="optuna", n_iter=10, cv=2)
        best_model, best_params = tuner.tune(Ridge(), param_grid, X, y)

        assert best_model is not None
        assert isinstance(best_params, dict)
        assert "alpha" in best_params
        preds = best_model.predict(X)
        assert len(preds) == len(y)


# ---------------------------------------------------------------------------
# Module-level helper for the optuna skip marker
# ---------------------------------------------------------------------------

def _optuna_available() -> bool:
    """Return True if optuna is importable."""
    try:
        import optuna  # noqa: F401
        return True
    except ImportError:
        return False
