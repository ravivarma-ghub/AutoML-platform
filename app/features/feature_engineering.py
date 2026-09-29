"""
app/features/feature_engineering.py

Smart feature engineering and column-type detection for the AutoML platform.
Handles type inference, datetime expansion, ID-column detection, and
problem-type (classification vs regression) determination.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.features.preprocessing import FrequencyEncoder, OutlierClipper

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
_UNIQUE_RATIO_ID_THRESHOLD = 1.0        # 100% unique → likely ID column
_NUMERIC_CAT_MAX_UNIQUE = 20            # numeric with ≤ this many unique values
_REGRESSION_UNIQUE_MIN = 20             # unique count above which we assume regression
_DATE_SAMPLE_SIZE = 200                 # rows sampled when testing date parseability


class FeatureEngineer:
    """
    Analyses a raw DataFrame and engineers useful features.

    Responsibilities
    ----------------
    * Detect column types (numeric, categorical, datetime, ID/drop).
    * Extract rich temporal features from datetime columns.
    * Determine whether the ML task is classification or regression.
    * Return structured metadata about all transformations applied.
    """

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        target_column: Optional[str] = None,
        target_col: Optional[str] = None,
    ) -> None:
        self._df = df
        self._target_col = target_column or target_col
        self._metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def engineer(
        self,
        df: Optional[pd.DataFrame] = None,
        target_col: Optional[str] = None,
        target_column: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run the full feature-engineering pipeline.
        """
        if df is None:
            df = self._df
        if target_col is None:
            target_col = target_column or self._target_col
        if df is None or target_col is None:
            raise ValueError("Both df and target_col must be provided.")

        df = df.copy()

        # 1. Detect column types
        type_info = self._detect_column_types(df, target_col)
        numeric_cols: List[str] = type_info["numeric_cols"]
        categorical_cols: List[str] = type_info["categorical_cols"]
        datetime_cols: List[str] = type_info["datetime_cols"]
        dropped_cols: List[str] = type_info["dropped_cols"]

        # 2. Drop ID/useless columns
        df.drop(columns=dropped_cols, errors="ignore", inplace=True)

        # 3. Expand datetime columns
        engineered_cols: List[str] = []
        for col in datetime_cols:
            if col not in df.columns:
                continue
            expanded = self._extract_datetime_features(df, col)
            new_cols = [c for c in expanded.columns if c not in df.columns]
            df = pd.concat([df.drop(columns=[col]), expanded[new_cols]], axis=1)
            engineered_cols.extend(new_cols)

        # 4. Detect problem type
        problem_type = self.detect_problem_type(df, target_col)

        # 5. Persist metadata
        self._metadata = {
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "datetime_cols": datetime_cols,
            "dropped_cols": dropped_cols,
            "engineered_cols": engineered_cols,
            "problem_type": problem_type,
        }

        logger.info(
            "Feature engineering complete. "
            "numeric=%d, categorical=%d, datetime=%d, dropped=%d, engineered=%d. "
            "Problem type: %s.",
            len(numeric_cols),
            len(categorical_cols),
            len(datetime_cols),
            len(dropped_cols),
            len(engineered_cols),
            problem_type,
        )
        return df, self._metadata

    # ------------------------------------------------------------------

    def extract_features(
        self,
        df: Optional[pd.DataFrame] = None,
        target_col: Optional[str] = None,
        target_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Convenience method returning just the transformed DataFrame."""
        transformed_df, _ = self.engineer(df=df, target_col=target_col, target_column=target_column)
        return transformed_df

    # ------------------------------------------------------------------

    def detect_problem_type(
        self,
        df: Optional[pd.DataFrame] = None,
        target_col: Optional[str] = None,
        target_column: Optional[str] = None,
    ) -> str:
        """
        Determine whether the task is *classification* or *regression*.
        """
        if df is None:
            df = self._df
        if target_col is None:
            target_col = target_column or self._target_col
        if df is None or target_col is None:
            raise ValueError("DataFrame and target_col must be provided.")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
        problem_type = self._detect_target_type(df[target_col])
        logger.info("Detected problem type: %s (target='%s').", problem_type, target_col)
        return problem_type

    # ------------------------------------------------------------------

    def detect_column_types(
        self,
        df: Optional[pd.DataFrame] = None,
        target_col: Optional[str] = None,
        target_column: Optional[str] = None,
    ) -> Tuple[List[str], List[str]]:
        """Public helper returning (numeric_cols, categorical_cols)."""
        if df is None:
            df = self._df
        if target_col is None:
            target_col = target_column or self._target_col
        if df is None:
            raise ValueError("DataFrame must be provided.")
        info = self._detect_column_types(df, target_col or "")
        return info["numeric_cols"], info["categorical_cols"]

    # ------------------------------------------------------------------

    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata produced by the last call to :meth:`engineer`."""
        return dict(self._metadata)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_column_types(
        self, df: pd.DataFrame, target_col: str
    ) -> Dict[str, List[str]]:
        """
        Heuristically classify every non-target column into one of:
        numeric, categorical, datetime, or drop (ID-like columns).

        Detection logic
        ---------------
        1. Already a datetime dtype → datetime.
        2. Object/string column that parses successfully as dates → datetime.
        3. Numeric with unique count == row count (100% unique) → drop (ID).
        4. Numeric with ≤ ``_NUMERIC_CAT_MAX_UNIQUE`` unique values → categorical.
        5. All other numerics → numeric.
        6. Remaining object/category dtypes → categorical.

        Parameters
        ----------
        df : pd.DataFrame
        target_col : str

        Returns
        -------
        dict with keys: numeric_cols, categorical_cols, datetime_cols, dropped_cols.
        """
        numeric_cols: List[str] = []
        categorical_cols: List[str] = []
        datetime_cols: List[str] = []
        dropped_cols: List[str] = []

        feature_cols = [c for c in df.columns if c != target_col]

        for col in feature_cols:
            series = df[col]
            n_rows = len(series)
            n_unique = series.nunique(dropna=True)

            # -- Already datetime ------------------------------------------
            if pd.api.types.is_datetime64_any_dtype(series):
                datetime_cols.append(col)
                logger.debug("Column '%s' → datetime (native dtype).", col)
                continue

            # -- Object/string column: attempt date parsing ----------------
            if series.dtype == object or str(series.dtype) == "string":
                if self._is_parseable_as_datetime(series):
                    datetime_cols.append(col)
                    logger.debug("Column '%s' → datetime (parseable string).", col)
                    continue
                # Otherwise treat as categorical
                categorical_cols.append(col)
                logger.debug("Column '%s' → categorical (object dtype).", col)
                continue

            # -- Category dtype -------------------------------------------
            if hasattr(series, "cat") or str(series.dtype) == "category":
                categorical_cols.append(col)
                logger.debug("Column '%s' → categorical (category dtype).", col)
                continue

            # -- Numeric columns ------------------------------------------
            if pd.api.types.is_numeric_dtype(series):
                unique_ratio = n_unique / n_rows if n_rows > 0 else 0.0

                # 100% unique integers → likely an ID column
                if (
                    unique_ratio >= _UNIQUE_RATIO_ID_THRESHOLD
                    and pd.api.types.is_integer_dtype(series)
                    and n_rows > 1
                ):
                    dropped_cols.append(col)
                    logger.debug(
                        "Column '%s' → dropped (ID-like, %d/%d unique).",
                        col, n_unique, n_rows,
                    )
                    continue

                # Low cardinality numeric → treat as categorical
                if n_unique <= _NUMERIC_CAT_MAX_UNIQUE:
                    categorical_cols.append(col)
                    logger.debug(
                        "Column '%s' → categorical (numeric with %d unique values).",
                        col, n_unique,
                    )
                    continue

                numeric_cols.append(col)
                logger.debug("Column '%s' → numeric.", col)
                continue

            # -- Fallback --------------------------------------------------
            categorical_cols.append(col)
            logger.debug("Column '%s' → categorical (fallback).", col)

        return {
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "datetime_cols": datetime_cols,
            "dropped_cols": dropped_cols,
        }

    # ------------------------------------------------------------------

    def _extract_datetime_features(
        self, df: pd.DataFrame, col: str
    ) -> pd.DataFrame:
        """
        Expand a single datetime column into multiple numeric features.

        Extracted features
        ------------------
        ``{col}_year``, ``{col}_month``, ``{col}_day``,
        ``{col}_dayofweek``, ``{col}_quarter``,
        ``{col}_hour``  *(only if any non-zero hour exists)*,
        ``{col}_is_weekend``, ``{col}_days_since_min``

        Parameters
        ----------
        df : pd.DataFrame – must contain *col*.
        col : str

        Returns
        -------
        pd.DataFrame
            A copy of *df* with the new feature columns appended (original
            *col* is **not** dropped here — caller is responsible).
        """
        out = df.copy()

        # Coerce to datetime if necessary
        if not pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = pd.to_datetime(out[col], infer_datetime_format=True, errors="coerce")

        dt = out[col]
        col_min = dt.min()

        out[f"{col}_year"] = dt.dt.year.astype("float32")
        out[f"{col}_month"] = dt.dt.month.astype("float32")
        out[f"{col}_day"] = dt.dt.day.astype("float32")
        out[f"{col}_dayofweek"] = dt.dt.dayofweek.astype("float32")
        out[f"{col}_quarter"] = dt.dt.quarter.astype("float32")
        out[f"{col}_is_weekend"] = (dt.dt.dayofweek >= 5).astype("float32")
        out[f"{col}_days_since_min"] = (dt - col_min).dt.days.astype("float32")

        # Hour only if meaningful
        if (dt.dt.hour != 0).any():
            out[f"{col}_hour"] = dt.dt.hour.astype("float32")

        logger.debug("Extracted datetime features from column '%s'.", col)
        return out

    # ------------------------------------------------------------------

    def _detect_target_type(self, series: pd.Series) -> str:
        """
        Infer whether the target column is for *classification* or *regression*.

        Rules (in order)
        ----------------
        1. Non-numeric dtype → classification.
        2. Binary (exactly 2 unique values) → classification.
        3. Numeric with ≤ ``_REGRESSION_UNIQUE_MIN`` unique values → classification.
        4. Otherwise → regression.

        Parameters
        ----------
        series : pd.Series – the target column.

        Returns
        -------
        ``'classification'`` or ``'regression'``
        """
        n_unique = series.nunique(dropna=True)

        # Non-numeric → always classification
        if not pd.api.types.is_numeric_dtype(series):
            logger.debug(
                "Target type: classification (non-numeric, %d unique values).", n_unique
            )
            return "classification"

        # Binary target
        if n_unique <= 2:
            logger.debug(
                "Target type: classification (binary, %d unique values).", n_unique
            )
            return "classification"

        # Low-cardinality numeric → treat as classification
        if n_unique <= _REGRESSION_UNIQUE_MIN:
            logger.debug(
                "Target type: classification (numeric, %d unique values ≤ %d).",
                n_unique, _REGRESSION_UNIQUE_MIN,
            )
            return "classification"

        # High-cardinality numeric → regression
        logger.debug(
            "Target type: regression (numeric, %d unique values > %d).",
            n_unique, _REGRESSION_UNIQUE_MIN,
        )
        return "regression"

    # ------------------------------------------------------------------

    @staticmethod
    def _is_parseable_as_datetime(series: pd.Series) -> bool:
        """
        Return True if the majority of non-null values in *series* can be
        parsed as dates.

        Samples up to ``_DATE_SAMPLE_SIZE`` non-null values and attempts
        ``pd.to_datetime`` with ``errors='coerce'``, accepting a column as
        datetime when ≥ 80 % of the sample parses successfully.
        """
        non_null = series.dropna()
        if len(non_null) == 0:
            return False

        sample = non_null.sample(
            n=min(_DATE_SAMPLE_SIZE, len(non_null)), random_state=42
        )
        try:
            parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
            success_rate = parsed.notna().mean()
            return bool(success_rate >= 0.80)
        except Exception:
            return False
