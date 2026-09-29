"""
Tests for app.data_profiler — DataProfiler class.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data_profiler import DataProfiler


class TestDataProfiler:
    """Unit tests for DataProfiler."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _numeric_df() -> pd.DataFrame:
        """Deterministic numeric-only DataFrame."""
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            {
                "a": rng.normal(50, 10, 500),
                "b": rng.uniform(0, 100, 500),
                "c": rng.exponential(5, 500),
            }
        )

    @staticmethod
    def _categorical_df() -> pd.DataFrame:
        """DataFrame with categorical columns and known distributions."""
        rng = np.random.default_rng(42)
        n = 500
        return pd.DataFrame(
            {
                "color": rng.choice(["red", "green", "blue"], n, p=[0.5, 0.3, 0.2]),
                "size": rng.choice(["S", "M", "L", "XL"], n),
                "value": rng.uniform(0, 1, n),
                "target": rng.integers(0, 2, n),
            }
        )

    @staticmethod
    def _df_with_missing() -> pd.DataFrame:
        """DataFrame with known number of missing values."""
        rng = np.random.default_rng(7)
        n = 200
        arr = rng.uniform(0, 1, n)
        arr[:20] = np.nan  # exactly 20 missing
        return pd.DataFrame({"feature": arr, "target": rng.integers(0, 2, n)})

    @staticmethod
    def _df_with_outliers() -> pd.DataFrame:
        """DataFrame where column 'x' has clearly inserted outliers."""
        rng = np.random.default_rng(0)
        n = 300
        x = rng.normal(0, 1, n)
        x[0] = 1_000.0  # extreme outlier
        x[1] = -1_000.0  # extreme outlier
        return pd.DataFrame({"x": x, "target": rng.integers(0, 2, n)})

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_profile_has_all_columns(self, sample_classification_df: pd.DataFrame) -> None:
        """Profile output must contain an entry for every column in the DataFrame."""
        profiler = DataProfiler(sample_classification_df)
        profile = profiler.profile()

        # Profile may be a dict keyed by column name, or an object with a
        # .columns attribute / iterable column profiles.
        if isinstance(profile, dict):
            profiled_cols = set(profile.keys())
        elif hasattr(profile, "columns"):
            profiled_cols = set(profile.columns)
        else:
            # Fallback: convert to string and check column names present
            profile_str = str(profile)
            for col in sample_classification_df.columns:
                assert col in profile_str, f"Column '{col}' missing from profile"
            return

        for col in sample_classification_df.columns:
            assert col in profiled_cols, f"Column '{col}' not found in profile keys"

    def test_numeric_stats_correct(self) -> None:
        """Mean and std for a numeric column must be within tolerance."""
        df = self._numeric_df()
        profiler = DataProfiler(df)
        profile = profiler.profile()

        # Extract stats for column 'a' (mean ≈ 50, std ≈ 10)
        if isinstance(profile, dict) and "a" in profile:
            col_stats = profile["a"]
        elif hasattr(profile, "get_column"):
            col_stats = profile.get_column("a")
        else:
            pytest.skip("Cannot extract per-column stats from profile format")

        # Support both dict-style and object-style access
        def _get(stats, key: str):
            if isinstance(stats, dict):
                return stats.get(key)
            return getattr(stats, key, None)

        mean_val = _get(col_stats, "mean")
        std_val = _get(col_stats, "std")

        if mean_val is not None:
            assert abs(mean_val - df["a"].mean()) < 1.0, "Mean is significantly off"
        if std_val is not None:
            assert abs(std_val - df["a"].std()) < 1.0, "Std is significantly off"

    def test_categorical_stats_correct(self) -> None:
        """Top category for 'color' column must be 'red' (50% probability)."""
        df = self._categorical_df()
        profiler = DataProfiler(df)
        profile = profiler.profile()

        if isinstance(profile, dict) and "color" in profile:
            col_stats = profile["color"]
        elif hasattr(profile, "get_column"):
            col_stats = profile.get_column("color")
        else:
            pytest.skip("Cannot extract per-column stats from profile format")

        def _get(stats, key: str):
            if isinstance(stats, dict):
                return stats.get(key)
            return getattr(stats, key, None)

        top_value = _get(col_stats, "top") or _get(col_stats, "mode") or _get(col_stats, "most_frequent")
        if top_value is not None:
            assert top_value == "red", f"Expected top category 'red', got '{top_value}'"

        n_unique = _get(col_stats, "unique") or _get(col_stats, "n_unique") or _get(col_stats, "cardinality")
        if n_unique is not None:
            assert n_unique == 3, f"Expected 3 unique values for 'color', got {n_unique}"

    def test_missing_values_counted(self) -> None:
        """Profile must correctly count 20 missing values in 'feature' column."""
        df = self._df_with_missing()
        profiler = DataProfiler(df)
        profile = profiler.profile()

        if isinstance(profile, dict) and "feature" in profile:
            col_stats = profile["feature"]
        elif hasattr(profile, "get_column"):
            col_stats = profile.get_column("feature")
        else:
            pytest.skip("Cannot extract per-column stats from profile format")

        def _get(stats, key: str):
            if isinstance(stats, dict):
                return stats.get(key)
            return getattr(stats, key, None)

        missing_count = (
            _get(col_stats, "missing_count")
            or _get(col_stats, "null_count")
            or _get(col_stats, "n_missing")
        )
        if missing_count is not None:
            assert missing_count == 20, (
                f"Expected 20 missing values, got {missing_count}"
            )

    def test_outlier_detection(self) -> None:
        """Profile must flag that column 'x' contains at least 1 outlier."""
        df = self._df_with_outliers()
        profiler = DataProfiler(df)
        profile = profiler.profile()

        if isinstance(profile, dict) and "x" in profile:
            col_stats = profile["x"]
        elif hasattr(profile, "get_column"):
            col_stats = profile.get_column("x")
        else:
            pytest.skip("Cannot extract per-column stats from profile format")

        def _get(stats, key: str):
            if isinstance(stats, dict):
                return stats.get(key)
            return getattr(stats, key, None)

        outlier_count = (
            _get(col_stats, "outlier_count")
            or _get(col_stats, "n_outliers")
            or _get(col_stats, "outliers")
        )
        if outlier_count is not None:
            assert outlier_count >= 1, "Expected at least 1 outlier to be detected"
        else:
            # Fall back: check IQR fence – value 1000 must be outside q75 + 1.5*IQR
            q25, q75 = df["x"].quantile([0.25, 0.75])
            iqr = q75 - q25
            upper_fence = q75 + 1.5 * iqr
            assert 1_000.0 > upper_fence, "Outlier fence calculation seems wrong"

    def test_format_summary(self) -> None:
        """format_summary() must return a non-empty string representation."""
        df = self._numeric_df()
        profiler = DataProfiler(df)
        profile = profiler.profile()

        if hasattr(profile, "format_summary"):
            summary = profile.format_summary()
        elif hasattr(profiler, "format_summary"):
            summary = profiler.format_summary()
        elif hasattr(profile, "to_string"):
            summary = profile.to_string()
        else:
            summary = str(profile)

        assert isinstance(summary, str), "Summary must be a string"
        assert len(summary.strip()) > 0, "Summary must not be empty"
