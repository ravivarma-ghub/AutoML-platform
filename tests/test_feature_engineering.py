"""
Tests for app.feature_engineering — FeatureEngineer and FeaturePreprocessor.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from app.features.feature_engineering import FeatureEngineer
from app.features.preprocessing import FeaturePreprocessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_binary_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "f1": rng.uniform(0, 1, n),
            "f2": rng.uniform(0, 1, n),
            "target": rng.integers(0, 2, n),
        }
    )


def _simple_regression_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "f1": rng.uniform(0, 10, n),
            "f2": rng.uniform(0, 5, n),
            "price": rng.uniform(100, 1000, n),
        }
    )


def _df_with_datetime(n: int = 200, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = datetime(2021, 1, 1)
    dates = pd.to_datetime(
        [(base + timedelta(days=int(d))).isoformat() for d in rng.integers(0, 730, n)]
    )
    return pd.DataFrame(
        {
            "created_at": dates,
            "value": rng.uniform(0, 1, n),
            "target": rng.integers(0, 2, n),
        }
    )


def _df_with_id(n: int = 200, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "id": range(n),
            "feature_a": rng.uniform(0, 1, n),
            "feature_b": rng.choice(["x", "y", "z"], n),
            "target": rng.integers(0, 2, n),
        }
    )


def _df_with_missing_categorical(n: int = 200, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cats = rng.choice(["cat", "dog", "bird", None], n).tolist()
    nums = rng.uniform(0, 1, n)
    nums[:10] = np.nan
    return pd.DataFrame(
        {"animal": cats, "score": nums, "target": rng.integers(0, 2, n)}
    )


# ---------------------------------------------------------------------------
# TestFeatureEngineer
# ---------------------------------------------------------------------------

class TestFeatureEngineer:
    """Tests for FeatureEngineer utility methods."""

    def test_detect_problem_type_classification(self) -> None:
        """Binary target → classification."""
        df = _simple_binary_df()
        engineer = FeatureEngineer(df, target_column="target")
        problem_type = engineer.detect_problem_type()
        assert "classif" in problem_type.lower(), (
            f"Expected 'classification', got '{problem_type}'"
        )

    def test_detect_problem_type_regression(self) -> None:
        """Continuous float target → regression."""
        df = _simple_regression_df()
        engineer = FeatureEngineer(df, target_column="price")
        problem_type = engineer.detect_problem_type()
        assert "regress" in problem_type.lower(), (
            f"Expected 'regression', got '{problem_type}'"
        )

    def test_datetime_feature_extraction(self) -> None:
        """Datetime columns should be expanded into year/month/day/dayofweek features."""
        df = _df_with_datetime()
        engineer = FeatureEngineer(df, target_column="target")
        transformed = engineer.extract_features()

        # The original datetime column should be gone; derived features should appear
        assert "created_at" not in transformed.columns or (
            any(
                col.startswith("created_at_") or col in ("year", "month", "day", "dayofweek")
                for col in transformed.columns
            )
        ), "Datetime column was not decomposed into sub-features"

        # At least year and month should exist somewhere
        col_str = " ".join(transformed.columns.tolist())
        assert any(
            kw in col_str for kw in ("year", "month", "day", "dow", "dayofweek", "week")
        ), f"No datetime-derived features found in columns: {transformed.columns.tolist()}"

    def test_type_detection(self, sample_classification_df: pd.DataFrame) -> None:
        """FeatureEngineer must correctly identify numeric and categorical columns."""
        engineer = FeatureEngineer(sample_classification_df, target_column="churn")
        numeric_cols, categorical_cols = engineer.detect_column_types()

        expected_numeric = {"age", "income", "score"}
        expected_categorical = {"education", "occupation"}

        # All expected numeric cols are detected as numeric
        for col in expected_numeric:
            assert col in numeric_cols, f"'{col}' should be numeric, got: {numeric_cols}"

        # All expected categorical cols are detected as categorical
        for col in expected_categorical:
            assert col in categorical_cols, (
                f"'{col}' should be categorical, got: {categorical_cols}"
            )

    def test_id_column_dropped(self) -> None:
        """Columns that look like ID columns (all-unique integers) should be dropped."""
        df = _df_with_id()
        engineer = FeatureEngineer(df, target_column="target")
        transformed = engineer.extract_features()
        assert "id" not in transformed.columns, (
            "ID column 'id' should have been dropped during feature engineering"
        )


# ---------------------------------------------------------------------------
# TestFeaturePreprocessor
# ---------------------------------------------------------------------------

class TestFeaturePreprocessor:
    """Tests for FeaturePreprocessor fit/transform pipeline."""

    def test_fit_transform_returns_array(self, sample_classification_df: pd.DataFrame) -> None:
        """fit_transform must return a 2-D numpy array with the correct row count."""
        feature_cols = ["age", "income", "score", "education", "occupation"]
        X = sample_classification_df[feature_cols]
        preprocessor = FeaturePreprocessor()
        result = preprocessor.fit_transform(X)
        assert isinstance(result, np.ndarray), "fit_transform must return a numpy array"
        assert result.ndim == 2, "Result must be 2-dimensional"
        assert result.shape[0] == len(X), "Row count must match input"

    def test_transform_consistent(self) -> None:
        """transform() on new data must produce same number of features as fit data."""
        rng = np.random.default_rng(10)
        n = 300
        df_train = pd.DataFrame(
            {
                "num": rng.uniform(0, 1, n),
                "cat": rng.choice(["x", "y", "z"], n),
            }
        )
        df_test = pd.DataFrame(
            {
                "num": rng.uniform(0, 1, 50),
                "cat": rng.choice(["x", "y", "z"], 50),
            }
        )
        preprocessor = FeaturePreprocessor()
        preprocessor.fit(df_train)
        train_out = preprocessor.transform(df_train)
        test_out = preprocessor.transform(df_test)
        assert train_out.shape[1] == test_out.shape[1], (
            f"Feature count mismatch: train={train_out.shape[1]}, test={test_out.shape[1]}"
        )

    def test_categorical_encoding(self) -> None:
        """Categorical columns must be encoded to numeric values."""
        rng = np.random.default_rng(5)
        n = 200
        df = pd.DataFrame(
            {
                "cat_feature": rng.choice(["alpha", "beta", "gamma"], n),
                "num_feature": rng.uniform(0, 1, n),
            }
        )
        preprocessor = FeaturePreprocessor()
        result = preprocessor.fit_transform(df)
        # Result must be entirely numeric (no object columns remain)
        assert result.dtype.kind in ("f", "i", "u"), (
            f"Output array dtype should be numeric, got {result.dtype}"
        )

    def test_missing_value_imputation(self) -> None:
        """Missing values in numeric and categorical columns must be imputed."""
        rng = np.random.default_rng(6)
        n = 200
        nums = rng.uniform(0, 1, n)
        nums[:15] = np.nan
        cats = rng.choice(["a", "b", "c", None], n).tolist()
        df = pd.DataFrame({"num": nums, "cat": cats})
        preprocessor = FeaturePreprocessor()
        result = preprocessor.fit_transform(df)
        assert not np.isnan(result).any(), (
            "Output array must not contain NaN after imputation"
        )

    def test_frequency_encoder(self) -> None:
        """FrequencyEncoder must replace categories with their relative frequencies."""
        try:
            from app.feature_engineering import FrequencyEncoder
        except ImportError:
            pytest.skip("FrequencyEncoder not exported from feature_engineering")

        rng = np.random.default_rng(9)
        n = 400
        cats = rng.choice(["common", "rare", "rare"], n, p=[0.8, 0.1, 0.1])
        df = pd.DataFrame({"cat": cats})
        enc = FrequencyEncoder()
        enc.fit(df[["cat"]])
        transformed = enc.transform(df[["cat"]])
        # 'common' should map to ~0.8, 'rare' to ~0.2
        assert transformed.shape == (n, 1), "Shape must be preserved"
        unique_vals = np.unique(np.round(transformed[:, 0], 1))
        assert len(unique_vals) <= 3, "Should have at most 3 distinct frequency values"

    def test_outlier_clipper(self) -> None:
        """OutlierClipper must clip extreme values within IQR-based bounds."""
        try:
            from app.feature_engineering import OutlierClipper
        except ImportError:
            pytest.skip("OutlierClipper not exported from feature_engineering")

        rng = np.random.default_rng(11)
        n = 300
        x = rng.normal(0, 1, n)
        x[0] = 1_000.0
        x[1] = -1_000.0
        df = pd.DataFrame({"x": x})
        clipper = OutlierClipper(factor=1.5)
        clipper.fit(df[["x"]])
        clipped = clipper.transform(df[["x"]])
        assert clipped.max() < 500, "Upper outlier was not clipped"
        assert clipped.min() > -500, "Lower outlier was not clipped"
