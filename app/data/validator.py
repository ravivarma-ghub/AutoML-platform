"""
Data validation module for the Intelligent AutoML Platform.

Produces a comprehensive ValidationResult with blocking errors, non-blocking
warnings, and dataset statistics.  A human-readable ASCII report can be
generated via ``DataValidator.format_report``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ValidationResult:
    """Holds the outcome of a full dataset validation run."""

    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def merge_stats(self, new_stats: dict) -> None:
        self.stats.update(new_stats)

    def __str__(self) -> str:
        lines = [f"ValidationResult(is_valid={self.is_valid})"]
        if self.errors:
            lines.append("Errors:")
            lines.extend(f"  - {e}" for e in self.errors)
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  - {w}" for w in self.warnings)
        if self.stats:
            lines.append(f"Stats: {self.stats}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Validator
# ──────────────────────────────────────────────────────────────────────────────


class DataValidator:
    """
    Validates a pandas DataFrame before it enters the AutoML pipeline.

    Configuration keys (``config`` dict)
    -------------------------------------
    min_rows : int  (default 10)
        Minimum row count before raising a blocking error.
    min_cols : int  (default 2)
        Minimum column count (including target).
    max_missing_pct : float  (default 0.9)
        Columns with more missing values than this fraction raise an error.
    high_cardinality_threshold : int  (default 50)
        Unique value count above which a categorical column is flagged.
    imbalance_threshold : float  (default 0.1)
        Minority class fraction below this value triggers an imbalance warning.
    leakage_date_keywords : list[str]
        Extra date-like column name keywords to check for target leakage.
    """

    # Default configuration
    _DEFAULTS: Dict[str, Any] = {
        "min_rows": 10,
        "min_cols": 2,
        "max_missing_pct": 0.90,
        "high_cardinality_threshold": 50,
        "imbalance_threshold": 0.10,
        "leakage_date_keywords": [
            "closed", "churn", "exit", "end", "cancel", "term",
            "default", "leave", "stop", "expire",
        ],
    }

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        target_column: Optional[str] = None,
        target_col: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._df = df
        self._target_col = target_column or target_col
        self._cfg: Dict[str, Any] = {**self._DEFAULTS, **(config or {})}
        self._last_result: Optional[ValidationResult] = None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def validate(
        self,
        df: Optional[pd.DataFrame] = None,
        target_col: Optional[str] = None,
        target_column: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Run all validation checks and return a ``ValidationResult``.
        """
        if df is None:
            df = self._df
        if target_col is None:
            target_col = target_column or self._target_col

        self._cfg = {**self._DEFAULTS, **(config or {})}
        result = ValidationResult()

        if df is None:
            result.add_error("No DataFrame provided for validation.")
            self._last_result = result
            return result

        if len(df) == 0:
            result.add_error("Dataset is empty (0 rows).")
            self._last_result = result
            return result

        if len(df.columns) < 2:
            result.add_error(f"Dataset must have at least 2 columns; found {len(df.columns)}.")
            self._last_result = result
            return result

        if target_col not in df.columns:
            result.add_error(
                f"Target column {target_col!r} not found in dataset. "
                f"Available columns: {list(df.columns)}"
            )
            self._last_result = result
            return result

        checks = [
            self._check_shape,
            self._check_missing_values,
            self._check_infinite_values,
            self._check_duplicates,
            self._check_constant_columns,
            self._check_unique_identifiers,
            self._check_high_cardinality,
            self._check_data_types,
        ]
        target_checks = [
            self._check_target_leakage,
            self._check_target_distribution,
        ]

        for check in checks:
            errs, warns, stats = check(df)
            for e in errs:
                result.add_error(e)
            for w in warns:
                result.add_warning(w)
            result.merge_stats(stats)

        for check in target_checks:
            errs, warns, stats = check(df, target_col)
            for e in errs:
                result.add_error(e)
            for w in warns:
                result.add_warning(w)
            result.merge_stats(stats)

        logger.info(
            "Validation finished — valid=%s  errors=%d  warnings=%d",
            result.is_valid,
            len(result.errors),
            len(result.warnings),
        )
        return result

    # ------------------------------------------------------------------ #
    #  Individual checks                                                   #
    # ------------------------------------------------------------------ #

    def _check_shape(
        self, df: pd.DataFrame
    ) -> Tuple[List[str], List[str], dict]:
        errors, warnings, stats = [], [], {}
        n_rows, n_cols = df.shape
        stats["n_rows"] = n_rows
        stats["n_cols"] = n_cols

        if n_rows < self._cfg["min_rows"]:
            errors.append(
                f"Dataset has only {n_rows} rows; minimum required is {self._cfg['min_rows']}."
            )
        if n_cols < self._cfg["min_cols"]:
            errors.append(
                f"Dataset has only {n_cols} columns; minimum required is {self._cfg['min_cols']}."
            )
        if n_rows < 100:
            warnings.append(
                f"Only {n_rows} rows — model reliability may be low. "
                "Consider collecting more data."
            )
        elif n_rows < 1_000:
            warnings.append(
                f"Dataset is small ({n_rows} rows). Results may have high variance."
            )
        return errors, warnings, stats

    def _check_missing_values(
        self, df: pd.DataFrame
    ) -> Tuple[List[str], List[str], dict]:
        errors, warnings, stats = [], [], {}
        n_rows = len(df)
        missing_pcts: Dict[str, float] = {}

        for col in df.columns:
            n_miss = int(df[col].isna().sum())
            if n_miss == 0:
                continue
            pct = n_miss / n_rows
            missing_pcts[col] = round(pct, 6)
            if pct > self._cfg["max_missing_pct"]:
                errors.append(
                    f"Column {col!r} has {pct:.1%} missing values "
                    f"(threshold: {self._cfg['max_missing_pct']:.0%}). "
                    "Consider dropping this column."
                )
            elif pct > 0.30:
                warnings.append(
                    f"Column {col!r} has {pct:.1%} missing values."
                )

        stats["missing_pcts"] = missing_pcts
        stats["missing_col_count"] = len(missing_pcts)
        return errors, warnings, stats

    def _check_duplicates(
        self, df: pd.DataFrame
    ) -> Tuple[List[str], List[str], dict]:
        errors, warnings, stats = [], [], {}
        n_dup = int(df.duplicated().sum())
        stats["duplicate_rows"] = n_dup

        if n_dup > 0:
            pct = n_dup / len(df)
            msg = f"Dataset contains {n_dup:,} duplicate rows ({pct:.1%})."
            if pct > 0.20:
                errors.append(msg + " This is unusually high.")
            else:
                warnings.append(msg)
        return errors, warnings, stats

    def _check_constant_columns(
        self, df: pd.DataFrame
    ) -> Tuple[List[str], List[str], dict]:
        errors, warnings, stats = [], [], {}
        constant_cols: List[str] = []

        for col in df.columns:
            n_unique = df[col].nunique(dropna=True)
            if n_unique <= 1:
                constant_cols.append(col)
                errors.append(
                    f"Column {col!r} has only {n_unique} unique value(s) "
                    "(constant/near-constant) — should be dropped."
                )

        stats["constant_columns"] = constant_cols
        return errors, warnings, stats

    def _check_high_cardinality(
        self, df: pd.DataFrame
    ) -> Tuple[List[str], List[str], dict]:
        errors, warnings, stats = [], [], {}
        threshold = self._cfg["high_cardinality_threshold"]
        high_card: Dict[str, int] = {}

        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            n_unique = df[col].nunique(dropna=True)
            if n_unique > threshold:
                high_card[col] = n_unique
                warnings.append(
                    f"Column {col!r} has {n_unique:,} unique values "
                    "(high cardinality) — may need encoding or grouping."
                )

        stats["high_cardinality_columns"] = high_card
        return errors, warnings, stats

    def _check_data_types(
        self, df: pd.DataFrame
    ) -> Tuple[List[str], List[str], dict]:
        errors, warnings, stats = [], [], {}
        type_map: Dict[str, str] = {col: str(df[col].dtype) for col in df.columns}
        stats["column_dtypes"] = type_map

        # Flag object columns that might be numeric
        for col in df.select_dtypes(include="object").columns:
            converted = pd.to_numeric(df[col], errors="coerce")
            valid_ratio = converted.notna().sum() / max(len(df), 1)
            if valid_ratio > 0.80:
                warnings.append(
                    f"Column {col!r} is stored as object but {valid_ratio:.0%} of its "
                    "values look numeric. Consider casting to a numeric type."
                )

        return errors, warnings, stats

    def _check_infinite_values(
        self, df: pd.DataFrame
    ) -> Tuple[List[str], List[str], dict]:
        errors, warnings, stats = [], [], {}
        num_df = df.select_dtypes(include=[np.number])
        inf_count = int(np.isinf(num_df.values).sum())
        stats["infinite_values"] = inf_count

        if inf_count > 0:
            inf_cols = [
                col for col in num_df.columns if np.isinf(num_df[col]).any()
            ]
            warnings.append(
                f"Found {inf_count} infinite value(s) in columns: "
                f"{inf_cols}. Replace with NaN or clip before training."
            )
        return errors, warnings, stats

    def _check_target_leakage(
        self, df: pd.DataFrame, target_col: str
    ) -> Tuple[List[str], List[str], dict]:
        """
        Flag date/time columns whose names hint at post-target information
        (e.g., 'account_closed_date', 'churn_timestamp').
        """
        errors, warnings, stats = [], [], {}
        leakage_keywords = self._cfg["leakage_date_keywords"]
        date_keywords = {"date", "time", "timestamp", "dt", "created", "closed"}

        potential_leaks: List[str] = []
        for col in df.columns:
            if col == target_col:
                continue
            col_lower = col.lower()
            has_date_hint = any(kw in col_lower for kw in date_keywords)
            has_leak_hint = any(kw in col_lower for kw in leakage_keywords)
            is_datetime = pd.api.types.is_datetime64_any_dtype(df[col])

            if (has_date_hint and has_leak_hint) or (is_datetime and has_leak_hint):
                potential_leaks.append(col)
                warnings.append(
                    f"Column {col!r} may represent post-target event information "
                    "(potential target leakage). Review carefully."
                )

        stats["potential_leakage_columns"] = potential_leaks
        return errors, warnings, stats

    def _check_target_distribution(
        self, df: pd.DataFrame, target_col: str
    ) -> Tuple[List[str], List[str], dict]:
        errors, warnings, stats = [], [], {}
        series = df[target_col].dropna()

        if series.empty:
            errors.append(f"Target column {target_col!r} has no non-null values.")
            return errors, warnings, stats

        # Classification: check class balance
        if series.dtype == "object" or series.nunique() <= 20:
            value_counts = series.value_counts(normalize=True)
            dist = {str(k): round(float(v), 6) for k, v in value_counts.items()}
            stats["target_distribution"] = dist
            stats["target_type"] = "classification"

            minority_frac = float(value_counts.min())
            if minority_frac < self._cfg["imbalance_threshold"]:
                warnings.append(
                    f"Target {target_col!r} is highly imbalanced: "
                    f"minority class = {minority_frac:.1%}. "
                    "Consider resampling or using class weights."
                )
            if series.nunique() == 1:
                errors.append(
                    f"Target column {target_col!r} has only one unique class value."
                )
        else:
            # Regression: check for extreme skewness
            try:
                skew = float(series.skew())
                stats["target_skewness"] = round(skew, 4)
                stats["target_type"] = "regression"
                if abs(skew) > 2.0:
                    warnings.append(
                        f"Target {target_col!r} has high skewness ({skew:.2f}). "
                        "Consider a log or Box-Cox transformation."
                    )
            except Exception:
                pass

        return errors, warnings, stats

    def _check_unique_identifiers(
        self, df: pd.DataFrame
    ) -> Tuple[List[str], List[str], dict]:
        """
        Detect columns that look like unique identifiers (>= 95% unique values).
        These are typically useless features and should be dropped.
        """
        errors, warnings, stats = [], [], {}
        id_like: List[str] = []
        n_rows = len(df)

        id_keywords = {"id", "uuid", "guid", "key", "hash", "index", "serial", "ref"}

        for col in df.columns:
            col_lower = col.lower()
            n_unique = df[col].nunique(dropna=True)
            unique_ratio = n_unique / max(n_rows, 1)
            has_id_keyword = any(kw in col_lower for kw in id_keywords)

            if unique_ratio >= 0.95 or (has_id_keyword and unique_ratio >= 0.50):
                id_like.append(col)
                warnings.append(
                    f"Column {col!r} appears to be a unique identifier "
                    f"({n_unique:,} unique / {n_rows:,} rows = {unique_ratio:.0%}). "
                    "Should likely be excluded from features."
                )

        stats["id_like_columns"] = id_like
        return errors, warnings, stats

    # ------------------------------------------------------------------ #
    #  Report formatting                                                   #
    # ------------------------------------------------------------------ #

    def format_report(self, result: Optional[ValidationResult] = None) -> str:
        """
        Produce a human-readable ASCII validation report.
        """
        if result is None:
            result = self._last_result
        if result is None:
            return "No validation result available."

        W = 54  # total line width
        HEAVY = "═" * W
        LIGHT = "─" * W

        def section(title: str) -> str:
            return f"\n{title}\n{LIGHT}"

        def row(label: str, value: str, symbol: str = "") -> str:
            sym_part = f"  {symbol}" if symbol else ""
            label_w = W - 2 - len(str(value)) - len(sym_part)
            return f"  {label:<{label_w}}{value}{sym_part}"

        lines: List[str] = []
        stats = result.stats

        # ── Header ────────────────────────────────────────────────────
        lines.append("\nDATA VALIDATION REPORT")
        lines.append(HEAVY)
        lines.append(
            row("Rows", f"{stats.get('n_rows', '?'):,}")
        )
        lines.append(
            row("Columns", f"{stats.get('n_cols', '?'):,}")
        )

        dup = stats.get("duplicate_rows", 0)
        lines.append(
            row("Duplicate Rows", f"{dup:,}", "⚠" if dup else "✓")
        )

        inf_v = stats.get("infinite_values", 0)
        lines.append(
            row("Infinite Values", f"{inf_v:,}", "⚠" if inf_v else "✓")
        )

        miss_c = stats.get("missing_col_count", 0)
        lines.append(
            row("Missing Value Columns", f"{miss_c:,}", "⚠" if miss_c else "✓")
        )

        # ── Missing values ─────────────────────────────────────────────
        missing_pcts: Dict[str, float] = stats.get("missing_pcts", {})
        if missing_pcts:
            lines.append(section("MISSING VALUES"))
            for col, pct in sorted(missing_pcts.items(), key=lambda x: -x[1]):
                sym = "⚠" if pct > 0.05 else "✓"
                lines.append(row(f"  {col}", f"{pct:.1%}", sym))

        # ── Constant columns ──────────────────────────────────────────
        const_cols: List[str] = stats.get("constant_columns", [])
        if const_cols:
            lines.append(section("CONSTANT COLUMNS"))
            for col in const_cols:
                lines.append(row(f"  {col}", "", "✗ REMOVED"))

        # ── High cardinality ─────────────────────────────────────────
        hc: Dict[str, int] = stats.get("high_cardinality_columns", {})
        if hc:
            lines.append(section("HIGH CARDINALITY COLUMNS"))
            for col, n_uniq in hc.items():
                lines.append(row(f"  {col}", f"{n_uniq:,} unique values", "⚠"))

        # ── ID-like columns ──────────────────────────────────────────
        id_cols: List[str] = stats.get("id_like_columns", [])
        if id_cols:
            lines.append(section("POTENTIAL IDENTIFIER COLUMNS"))
            for col in id_cols:
                lines.append(row(f"  {col}", "", "⚠ REVIEW"))

        # ── Potential leakage ────────────────────────────────────────
        leak_cols: List[str] = stats.get("potential_leakage_columns", [])
        if leak_cols:
            lines.append(section("POTENTIAL LEAKAGE COLUMNS"))
            for col in leak_cols:
                lines.append(row(f"  {col}", "", "⚠ REVIEW"))

        # ── Target distribution ──────────────────────────────────────
        target_dist: Dict[str, float] = stats.get("target_distribution", {})
        if target_dist:
            target_type = stats.get("target_type", "classification")
            lines.append(section(f"TARGET DISTRIBUTION"))
            for cls_label, frac in target_dist.items():
                lines.append(row(f"  {cls_label}", f"{frac:.1%}"))
            minority = min(target_dist.values())
            if minority < self._cfg["imbalance_threshold"]:
                lines.append(row("  Class Imbalance", "", "⚠"))

        skew = stats.get("target_skewness")
        if skew is not None:
            lines.append(section("TARGET DISTRIBUTION (Regression)"))
            lines.append(row("  Skewness", f"{skew:.4f}", "⚠" if abs(skew) > 2 else "✓"))

        # ── Overall status ───────────────────────────────────────────
        lines.append(f"\n{HEAVY}")
        if not result.is_valid:
            status = "✗ ERRORS FOUND - Fix before proceeding"
        elif result.warnings:
            status = "⚠ WARNINGS - Proceed with caution"
        else:
            status = "✓ PASSED - No issues detected"
        lines.append(f"OVERALL STATUS: {status}")
        lines.append(HEAVY)

        # ── Errors & Warnings detail ──────────────────────────────────
        if result.errors:
            lines.append("\nERRORS")
            lines.append(LIGHT)
            for i, err in enumerate(result.errors, 1):
                lines.append(f"  [{i}] ✗ {err}")

        if result.warnings:
            lines.append("\nWARNINGS")
            lines.append(LIGHT)
            for i, warn in enumerate(result.warnings, 1):
                lines.append(f"  [{i}] ⚠ {warn}")

        lines.append("")
        return "\n".join(lines)
