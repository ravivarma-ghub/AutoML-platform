"""
app/features/preprocessing.py

Automated sklearn preprocessing pipeline builder for the AutoML platform.
Handles numeric, categorical, and datetime features with smart encoding strategies.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from app.core.config import Settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom transformers
# ---------------------------------------------------------------------------


class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Clips numeric values to the IQR fence: [Q1 - 1.5*IQR, Q3 + 1.5*IQR].

    Fitted lower/upper bounds are stored per column so they can be reused
    consistently during inference.
    """

    def __init__(self, factor: float = 1.5) -> None:
        self.factor = factor
        self.lower_bounds_: Optional[np.ndarray] = None
        self.upper_bounds_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "OutlierClipper":
        """Compute per-column IQR fences on the training data."""
        X = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_bounds_ = q1 - self.factor * iqr
        self.upper_bounds_ = q3 + self.factor * iqr
        logger.debug("OutlierClipper fitted on %d columns.", X.shape[1])
        return self

    # ------------------------------------------------------------------
    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Clip values to the fitted IQR fences column-wise."""
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise RuntimeError("OutlierClipper must be fitted before calling transform.")
        X = np.asarray(X, dtype=float).copy()
        for col_idx in range(X.shape[1]):
            X[:, col_idx] = np.clip(
                X[:, col_idx],
                self.lower_bounds_[col_idx],
                self.upper_bounds_[col_idx],
            )
        return X


# ---------------------------------------------------------------------------


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Replaces each category with its relative frequency observed during training.

    Unseen categories (at inference time) are assigned frequency 0.0.
    Handles multi-column arrays by processing each column independently.
    """

    def __init__(self) -> None:
        self.freq_maps_: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "FrequencyEncoder":
        """Build per-column frequency maps."""
        X = np.asarray(X, dtype=object)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.freq_maps_ = []
        for col_idx in range(X.shape[1]):
            col = X[:, col_idx]
            n = len(col)
            unique, counts = np.unique(col, return_counts=True)
            freq_map: Dict[str, float] = {
                str(val): int(cnt) / n for val, cnt in zip(unique, counts)
            }
            self.freq_maps_.append(freq_map)
        logger.debug("FrequencyEncoder fitted on %d columns.", X.shape[1])
        return self

    # ------------------------------------------------------------------
    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Map categories to their training frequencies."""
        if not self.freq_maps_:
            raise RuntimeError("FrequencyEncoder must be fitted before calling transform.")
        X = np.asarray(X, dtype=object)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        out = np.zeros((X.shape[0], X.shape[1]), dtype=float)
        for col_idx, freq_map in enumerate(self.freq_maps_):
            col = X[:, col_idx]
            out[:, col_idx] = np.array(
                [freq_map.get(str(val), 0.0) for val in col], dtype=float
            )
        return out

    # ------------------------------------------------------------------
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        """Return output feature names (same as input)."""
        n_cols = len(self.freq_maps_)
        if input_features is not None:
            return np.asarray(input_features[:n_cols])
        return np.array([f"freq_enc_{i}" for i in range(n_cols)])


# ---------------------------------------------------------------------------
# Main preprocessor
# ---------------------------------------------------------------------------


class FeaturePreprocessor:
    """
    Automatically builds and applies sklearn preprocessing pipelines.

    Supports:
    - Numeric columns  : median imputation → IQR clipping → standard scaling
    - Low-cardinality categoricals (≤10 unique)  : OneHotEncoder
    - Mid-cardinality categoricals (11–50 unique) : OrdinalEncoder
    - High-cardinality categoricals (>50 unique)  : FrequencyEncoder
    - Datetime columns : feature extraction (year, month, day, …) before fitting
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        self.config = config
        self._column_transformer: Optional[ColumnTransformer] = None
        self._feature_names: List[str] = []
        self._numeric_cols: List[str] = []
        self._categorical_cols: List[str] = []
        self._datetime_cols: List[str] = []
        self._cardinality_map: Dict[str, int] = {}
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        numeric_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        datetime_cols: Optional[List[str]] = None,
    ) -> "FeaturePreprocessor":
        """Fit preprocessor pipelines."""
        self.fit_transform(
            df=df,
            target_col=target_col,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            datetime_cols=datetime_cols,
        )
        return self

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        numeric_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        datetime_cols: Optional[List[str]] = None,
    ) -> Any:
        """
        Build pipelines, fit on *df* and return transformed matrix.
        """
        df = df.copy()

        # If columns not specified, auto-detect
        if numeric_cols is None and categorical_cols is None and datetime_cols is None:
            cols = [c for c in df.columns if c != target_col]
            dt_cols = []
            num_cols = []
            cat_cols = []
            for c in cols:
                if pd.api.types.is_datetime64_any_dtype(df[c]):
                    dt_cols.append(c)
                elif pd.api.types.is_numeric_dtype(df[c]):
                    num_cols.append(c)
                else:
                    cat_cols.append(c)
            datetime_cols = dt_cols
            numeric_cols = num_cols
            categorical_cols = cat_cols
        else:
            datetime_cols = datetime_cols or []
            numeric_cols = numeric_cols or []
            categorical_cols = categorical_cols or []

        # 1. Extract datetime features → add new columns, remove originals
        df = self._build_datetime_features(df, datetime_cols)

        # Store column groups (after datetime expansion)
        self._datetime_cols = datetime_cols
        self._numeric_cols = [c for c in numeric_cols if c != target_col]
        self._categorical_cols = [c for c in categorical_cols if c != target_col]

        # 2. Compute cardinalities for categorical columns
        self._cardinality_map = {
            col: int(df[col].nunique()) for col in self._categorical_cols
        }

        # 3. Build ColumnTransformer
        transformers = self._build_transformers()
        self._column_transformer = ColumnTransformer(
            transformers=transformers, remainder="drop", sparse_threshold=0
        )

        # 4. Prepare feature matrix (drop target)
        if target_col and target_col in df.columns:
            feature_df = df.drop(columns=[target_col], errors="ignore")
        else:
            feature_df = df
        X_transformed = self._column_transformer.fit_transform(feature_df)

        # 5. Collect output feature names
        self._feature_names = self._collect_feature_names()
        self._fitted = True

        logger.info(
            "FeaturePreprocessor fitted. Input columns: %d, Output features: %d",
            len(feature_df.columns),
            X_transformed.shape[1],
        )
        if target_col is not None:
            return X_transformed, self._feature_names
        return X_transformed

    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Apply the fitted preprocessing pipeline to new data.

        Parameters
        ----------
        df : pd.DataFrame
            Raw data (must contain the same columns as training data).

        Returns
        -------
        np.ndarray – transformed feature matrix.
        """
        if not self._fitted or self._column_transformer is None:
            raise RuntimeError("FeaturePreprocessor must be fitted before calling transform.")

        df = df.copy()
        df = self._build_datetime_features(df, self._datetime_cols)
        return self._column_transformer.transform(df)

    # ------------------------------------------------------------------
    # Pipeline builders
    # ------------------------------------------------------------------

    def _build_numeric_pipeline(self) -> Pipeline:
        """
        Build numeric preprocessing pipeline:
          SimpleImputer(median) → OutlierClipper (IQR) → StandardScaler
        """
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("outlier_clipper", OutlierClipper(factor=1.5)),
                ("scaler", StandardScaler()),
            ]
        )

    # ------------------------------------------------------------------

    def _build_categorical_pipeline(self, cardinality_map: Dict[str, int]) -> Dict[str, Pipeline]:
        """
        Build per-column categorical pipelines based on cardinality:
          ≤10  → OneHotEncoder (with handle_unknown='ignore')
          11–50 → OrdinalEncoder
          >50   → FrequencyEncoder

        Returns
        -------
        dict mapping column name → fitted Pipeline (unfitted at this stage).
        """
        pipelines: Dict[str, Pipeline] = {}
        for col, card in cardinality_map.items():
            if card <= 10:
                encoder: BaseEstimator = OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False
                )
                label = "ohe"
            elif card <= 50:
                encoder = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )
                label = "ordinal"
            else:
                encoder = FrequencyEncoder()
                label = "freq"

            pipelines[col] = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),
                    (label, encoder),
                ]
            )
        return pipelines

    # ------------------------------------------------------------------

    def _build_datetime_features(
        self, df: pd.DataFrame, datetime_cols: List[str]
    ) -> pd.DataFrame:
        """
        Extract temporal features from datetime columns and drop the originals.

        Extracted features per column *col*:
          col_year, col_month, col_day, col_dayofweek, col_quarter,
          col_hour  (only if time component is non-zero for any row),
          col_is_weekend, col_days_since_min
        """
        for col in datetime_cols:
            if col not in df.columns:
                logger.warning("Datetime column '%s' not found in DataFrame, skipping.", col)
                continue

            # Attempt conversion if not already datetime
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                except Exception:
                    df[col] = pd.to_datetime(df[col], errors="coerce")

            dt = df[col]
            col_min = dt.min()

            df[f"{col}_year"] = dt.dt.year.astype("float32")
            df[f"{col}_month"] = dt.dt.month.astype("float32")
            df[f"{col}_day"] = dt.dt.day.astype("float32")
            df[f"{col}_dayofweek"] = dt.dt.dayofweek.astype("float32")
            df[f"{col}_quarter"] = dt.dt.quarter.astype("float32")
            df[f"{col}_is_weekend"] = (dt.dt.dayofweek >= 5).astype("float32")
            df[f"{col}_days_since_min"] = (dt - col_min).dt.days.astype("float32")

            # Include hour only if there is meaningful time information
            if (dt.dt.hour != 0).any():
                df[f"{col}_hour"] = dt.dt.hour.astype("float32")

            df.drop(columns=[col], inplace=True)
            logger.debug("Extracted datetime features from column '%s'.", col)

        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_transformers(self) -> List[Tuple]:
        """
        Assemble the list of (name, transformer, columns) tuples for
        ColumnTransformer based on numeric and categorical columns.
        """
        transformers: List[Tuple] = []

        # Numeric transformer (single pipeline for all numeric cols)
        if self._numeric_cols:
            transformers.append(
                ("numeric", self._build_numeric_pipeline(), self._numeric_cols)
            )

        # Categorical transformers (one per column group by encoder type)
        cat_pipelines = self._build_categorical_pipeline(self._cardinality_map)

        ohe_cols = [c for c, card in self._cardinality_map.items() if card <= 10]
        ordinal_cols = [
            c for c, card in self._cardinality_map.items() if 11 <= card <= 50
        ]
        freq_cols = [c for c, card in self._cardinality_map.items() if card > 50]

        if ohe_cols:
            transformers.append(
                (
                    "cat_ohe",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "ohe",
                                OneHotEncoder(
                                    handle_unknown="ignore", sparse_output=False
                                ),
                            ),
                        ]
                    ),
                    ohe_cols,
                )
            )

        if ordinal_cols:
            transformers.append(
                (
                    "cat_ordinal",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "ordinal",
                                OrdinalEncoder(
                                    handle_unknown="use_encoded_value",
                                    unknown_value=-1,
                                ),
                            ),
                        ]
                    ),
                    ordinal_cols,
                )
            )

        if freq_cols:
            transformers.append(
                (
                    "cat_freq",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("freq", FrequencyEncoder()),
                        ]
                    ),
                    freq_cols,
                )
            )

        return transformers

    # ------------------------------------------------------------------

    def _collect_feature_names(self) -> List[str]:
        """
        Gather output feature names from the fitted ColumnTransformer.

        Falls back to generic names if the transformer does not expose
        ``get_feature_names_out``.
        """
        if self._column_transformer is None:
            return []

        names: List[str] = []
        for name, transformer, columns in self._column_transformer.transformers_:
            if name == "remainder":
                continue
            try:
                step_names = transformer.get_feature_names_out(columns)
                names.extend([str(n) for n in step_names])
            except AttributeError:
                # Fallback: derive names from column list
                if name == "numeric":
                    names.extend(columns)
                else:
                    names.extend(columns)
        return names

    # ------------------------------------------------------------------

    def get_feature_names(self) -> List[str]:
        """Return the list of output feature names after fitting."""
        return list(self._feature_names)

    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Persist the fitted preprocessor to disk using pickle.

        Parameters
        ----------
        path : str
            File path (e.g. ``"artifacts/preprocessor.pkl"``).
        """
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted FeaturePreprocessor.")
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "column_transformer": self._column_transformer,
            "feature_names": self._feature_names,
            "numeric_cols": self._numeric_cols,
            "categorical_cols": self._categorical_cols,
            "datetime_cols": self._datetime_cols,
            "cardinality_map": self._cardinality_map,
        }
        with open(save_path, "wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("FeaturePreprocessor saved to '%s'.", save_path)

    # ------------------------------------------------------------------

    def load(self, path: str) -> None:
        """
        Restore a previously saved preprocessor from disk.

        Parameters
        ----------
        path : str
            File path produced by :meth:`save`.
        """
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Preprocessor file not found: {load_path}")
        with open(load_path, "rb") as fh:
            state: dict = pickle.load(fh)
        self._column_transformer = state["column_transformer"]
        self._feature_names = state["feature_names"]
        self._numeric_cols = state["numeric_cols"]
        self._categorical_cols = state["categorical_cols"]
        self._datetime_cols = state["datetime_cols"]
        self._cardinality_map = state["cardinality_map"]
        self._fitted = True
        logger.info("FeaturePreprocessor loaded from '%s'.", load_path)
