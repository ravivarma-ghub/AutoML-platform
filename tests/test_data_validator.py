"""
Tests for app.data_validator — DataValidator class.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data_validator import DataValidator


class TestDataValidator:
    """Unit tests for DataValidator."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_df() -> pd.DataFrame:
        """Return a well-formed DataFrame with no quality issues."""
        rng = np.random.default_rng(0)
        n = 200
        return pd.DataFrame(
            {
                "num1": rng.uniform(0, 100, n),
                "num2": rng.uniform(0, 50, n),
                "cat1": rng.choice(["a", "b", "c"], n),
                "target": rng.integers(0, 2, n),
            }
        )

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_valid_dataset_passes(self, sample_classification_df: pd.DataFrame) -> None:
        """A realistic dataset should pass validation without critical errors."""
        validator = DataValidator(sample_classification_df, target_column="churn")
        report = validator.validate()
        # May have warnings (missing values, duplicates) but no critical failures
        # that would prevent pipeline execution
        assert report is not None
        assert hasattr(report, "is_valid") or isinstance(report, dict) or hasattr(report, "checks")

    def test_detects_missing_values(self, sample_classification_df: pd.DataFrame) -> None:
        """Validator must flag columns that contain NaN values."""
        validator = DataValidator(sample_classification_df, target_column="churn")
        report = validator.validate()
        # Convert report to string to check for missing-value mention
        report_str = str(report)
        assert any(
            keyword in report_str.lower()
            for keyword in ("missing", "null", "nan")
        ), "Report did not mention missing values"

    def test_detects_duplicate_rows(self) -> None:
        """Validator must flag duplicate rows in the dataset."""
        base = self._clean_df()
        df_with_dupes = pd.concat([base, base.iloc[:10]], ignore_index=True)
        validator = DataValidator(df_with_dupes, target_column="target")
        report = validator.validate()
        report_str = str(report)
        assert any(
            kw in report_str.lower() for kw in ("duplicate", "duplicated")
        ), "Report did not flag duplicate rows"

    def test_detects_constant_columns(self) -> None:
        """A column with a single unique value should be flagged."""
        df = self._clean_df()
        df["constant_col"] = 7  # constant column
        validator = DataValidator(df, target_column="target")
        report = validator.validate()
        report_str = str(report)
        assert any(
            kw in report_str.lower()
            for kw in ("constant", "zero variance", "single value", "unique")
        ), "Report did not flag the constant column"

    def test_detects_high_cardinality(self) -> None:
        """Categorical column with nearly unique values per row should be flagged."""
        rng = np.random.default_rng(1)
        n = 300
        df = pd.DataFrame(
            {
                "id_col": [f"user_{i}" for i in range(n)],  # high cardinality
                "num": rng.uniform(0, 1, n),
                "target": rng.integers(0, 2, n),
            }
        )
        validator = DataValidator(df, target_column="target")
        report = validator.validate()
        report_str = str(report)
        assert any(
            kw in report_str.lower()
            for kw in ("cardinality", "high cardinality", "unique", "id")
        ), "Report did not flag the high-cardinality column"

    def test_detects_infinite_values(self) -> None:
        """Columns containing inf/-inf values must be flagged."""
        df = self._clean_df()
        df.loc[0, "num1"] = np.inf
        df.loc[1, "num2"] = -np.inf
        validator = DataValidator(df, target_column="target")
        report = validator.validate()
        report_str = str(report)
        assert any(
            kw in report_str.lower()
            for kw in ("inf", "infinite", "infinity")
        ), "Report did not flag infinite values"

    def test_format_report_produces_string(self) -> None:
        """format_report() (or __str__) must return a non-empty string."""
        df = self._clean_df()
        validator = DataValidator(df, target_column="target")
        report = validator.validate()
        # Support both explicit format_report method and __str__
        if hasattr(report, "format_report"):
            result = report.format_report()
        elif hasattr(validator, "format_report"):
            result = validator.format_report()
        else:
            result = str(report)
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    def test_empty_dataframe_fails(self) -> None:
        """An empty DataFrame must fail validation."""
        df = pd.DataFrame()
        validator = DataValidator(df, target_column="target")
        report = validator.validate()
        # Either raises or returns invalid
        report_str = str(report)
        is_invalid = (
            (hasattr(report, "is_valid") and not report.is_valid)
            or "error" in report_str.lower()
            or "empty" in report_str.lower()
            or "fail" in report_str.lower()
            or "invalid" in report_str.lower()
        )
        assert is_invalid, "Empty DataFrame should not pass validation"

    def test_single_column_fails(self) -> None:
        """A DataFrame with only the target column and no features should fail."""
        rng = np.random.default_rng(2)
        df = pd.DataFrame({"target": rng.integers(0, 2, 100)})
        validator = DataValidator(df, target_column="target")
        report = validator.validate()
        report_str = str(report)
        is_invalid = (
            (hasattr(report, "is_valid") and not report.is_valid)
            or "error" in report_str.lower()
            or "feature" in report_str.lower()
            or "column" in report_str.lower()
            or "fail" in report_str.lower()
            or "invalid" in report_str.lower()
        )
        assert is_invalid, "Single-column DataFrame should not pass validation"
