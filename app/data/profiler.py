"""
Data profiling module for the Intelligent AutoML Platform.

Produces rich per-column and dataset-level statistics including numeric
descriptive stats, categorical frequency tables, datetime range info,
outlier counts (IQR method), and pairwise Pearson correlations.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Data-classes
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ColumnProfile:
    """Statistical profile for a single DataFrame column."""

    name: str
    dtype: str
    missing_count: int
    missing_pct: float
    unique_count: int
    unique_pct: float

    # Numeric stats (None for non-numeric columns)
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    outlier_count: Optional[int] = None  # IQR-based

    # Categorical stats (None for non-categorical columns)
    top_categories: Optional[Dict[str, int]] = None  # {category: count}
    mode: Optional[str] = None

    # Datetime stats (None for non-datetime columns)
    dt_min: Optional[str] = None
    dt_max: Optional[str] = None
    dt_range_days: Optional[float] = None

    # Column kind: "numeric" | "categorical" | "datetime" | "boolean"
    kind: str = "unknown"


@dataclass
class DatasetProfile:
    """Full profile for an entire DataFrame."""

    dataset_name: str
    num_rows: int
    num_columns: int
    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    datetime_columns: List[str] = field(default_factory=list)
    boolean_columns: List[str] = field(default_factory=list)
    memory_usage_mb: float = 0.0
    column_profiles: List[ColumnProfile] = field(default_factory=list)
    correlation_matrix: Optional[Dict[str, float]] = None  # top correlated pairs
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    @property
    def columns(self) -> List[str]:
        return [cp.name for cp in self.column_profiles]

    def get_column(self, name: str) -> Optional[ColumnProfile]:
        for cp in self.column_profiles:
            if cp.name == name:
                return cp
        return None

    def __getitem__(self, name: str) -> ColumnProfile:
        cp = self.get_column(name)
        if cp is None:
            raise KeyError(name)
        return cp

    def __contains__(self, name: str) -> bool:
        return self.get_column(name) is not None


# ──────────────────────────────────────────────────────────────────────────────
# Profiler
# ──────────────────────────────────────────────────────────────────────────────


class DataProfiler:
    """
    Generates comprehensive statistical profiles for pandas DataFrames.

    Usage
    -----
    >>> profiler = DataProfiler()
    >>> profile = profiler.profile(df, dataset_name="my_dataset")
    >>> print(profiler.format_summary(profile))
    >>> profile_dict = profiler.to_dict(profile)
    """

    # Maximum number of top categories to store per categorical column
    TOP_K_CATEGORIES: int = 20
    # Maximum number of top correlated pairs to store
    TOP_K_CORRELATIONS: int = 30

    def __init__(self, df: Optional[pd.DataFrame] = None) -> None:
        self._df = df
        self._last_profile: Optional[DatasetProfile] = None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def profile(
        self, df: Optional[pd.DataFrame] = None, dataset_name: str = "dataset"
    ) -> DatasetProfile:
        """
        Profile every column of ``df`` and return a ``DatasetProfile``.
        """
        if df is None:
            df = self._df
        if df is None:
            raise ValueError("No DataFrame provided to profile.")
        if df.empty:
            raise ValueError("Cannot profile an empty DataFrame.")

        n_rows, n_cols = df.shape
        mem_mb = df.memory_usage(deep=True).sum() / 1_048_576

        col_types = self._detect_column_types(df)
        logger.info(
            "Profiling '%s' — %d rows × %d cols | numeric=%d cat=%d dt=%d bool=%d",
            dataset_name, n_rows, n_cols,
            len(col_types["numeric"]), len(col_types["categorical"]),
            len(col_types["datetime"]), len(col_types["boolean"]),
        )

        column_profiles: List[ColumnProfile] = []
        for col in df.columns:
            series = df[col]
            if col in col_types["numeric"]:
                cp = self._profile_numeric_col(col, series)
            elif col in col_types["datetime"]:
                cp = self._profile_datetime_col(col, series)
            elif col in col_types["boolean"]:
                cp = self._profile_boolean_col(col, series)
            else:
                cp = self._profile_categorical_col(col, series)
            column_profiles.append(cp)

        correlations = self._compute_correlations(df)

        res = DatasetProfile(
            dataset_name=dataset_name,
            num_rows=n_rows,
            num_columns=n_cols,
            numeric_columns=col_types["numeric"],
            categorical_columns=col_types["categorical"],
            datetime_columns=col_types["datetime"],
            boolean_columns=col_types["boolean"],
            memory_usage_mb=round(mem_mb, 4),
            column_profiles=column_profiles,
            correlation_matrix=correlations,
        )
        self._last_profile = res
        return res

    # ------------------------------------------------------------------ #
    #  Column-type detection                                               #
    # ------------------------------------------------------------------ #

    def _detect_column_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Classify each column into one of: numeric, categorical, datetime, boolean.

        Returns
        -------
        dict
            Keys: ``"numeric"``, ``"categorical"``, ``"datetime"``, ``"boolean"``
        """
        result: Dict[str, List[str]] = {
            "numeric": [], "categorical": [], "datetime": [], "boolean": []
        }

        for col in df.columns:
            series = df[col]
            dtype = series.dtype

            if pd.api.types.is_bool_dtype(dtype):
                result["boolean"].append(col)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                result["datetime"].append(col)
            elif pd.api.types.is_numeric_dtype(dtype):
                # Check if it behaves like a boolean (0/1 only)
                uniques = series.dropna().unique()
                if set(uniques).issubset({0, 1, 0.0, 1.0}):
                    result["boolean"].append(col)
                else:
                    result["numeric"].append(col)
            elif pd.api.types.is_categorical_dtype(dtype):
                result["categorical"].append(col)
            else:  # object / string / etc.
                # Attempt datetime parse
                sample = series.dropna().head(50)
                parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
                if parsed.notna().mean() > 0.80:
                    result["datetime"].append(col)
                else:
                    result["categorical"].append(col)

        return result

    # ------------------------------------------------------------------ #
    #  Per-column profiling                                                #
    # ------------------------------------------------------------------ #

    def _profile_numeric(self, series: pd.Series) -> dict:
        """
        Compute numeric descriptive statistics for a Series.

        Returns
        -------
        dict
            Keys: mean, median, std, min, max, skewness, kurtosis,
            q25, q75, outlier_count.
        """
        clean = series.dropna().astype(float)
        if clean.empty:
            return {}

        q25 = float(clean.quantile(0.25))
        q75 = float(clean.quantile(0.75))
        iqr = q75 - q25
        lower = q25 - 1.5 * iqr
        upper = q75 + 1.5 * iqr
        outlier_count = int(((clean < lower) | (clean > upper)).sum())

        skewness: Optional[float] = None
        kurtosis_val: Optional[float] = None
        try:
            skewness = float(clean.skew())
            kurtosis_val = float(clean.kurtosis())
        except Exception:
            pass

        return {
            "mean": round(float(clean.mean()), 6),
            "median": round(float(clean.median()), 6),
            "std": round(float(clean.std()), 6),
            "min": round(float(clean.min()), 6),
            "max": round(float(clean.max()), 6),
            "skewness": round(skewness, 6) if skewness is not None else None,
            "kurtosis": round(kurtosis_val, 6) if kurtosis_val is not None else None,
            "q25": round(q25, 6),
            "q75": round(q75, 6),
            "outlier_count": outlier_count,
        }

    def _profile_categorical(self, series: pd.Series) -> dict:
        """
        Compute categorical statistics for a Series.

        Returns
        -------
        dict
            Keys: top_categories, mode.
        """
        clean = series.dropna()
        if clean.empty:
            return {}

        value_counts = clean.value_counts()
        top_k = value_counts.head(self.TOP_K_CATEGORIES)
        top_categories = {str(k): int(v) for k, v in top_k.items()}

        mode_val: Optional[str] = None
        try:
            mode_series = clean.mode()
            if not mode_series.empty:
                mode_val = str(mode_series.iloc[0])
        except Exception:
            pass

        return {"top_categories": top_categories, "mode": mode_val}

    def _profile_datetime(self, series: pd.Series) -> dict:
        """
        Compute datetime range statistics for a Series.

        Returns
        -------
        dict
            Keys: dt_min, dt_max, dt_range_days.
        """
        parsed = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
        clean = parsed.dropna()
        if clean.empty:
            return {}

        dt_min = clean.min()
        dt_max = clean.max()
        range_days = (dt_max - dt_min).total_seconds() / 86_400

        return {
            "dt_min": dt_min.isoformat(),
            "dt_max": dt_max.isoformat(),
            "dt_range_days": round(range_days, 2),
        }

    # ── Typed profile builders ─────────────────────────────────────────

    def _base_profile(self, col: str, series: pd.Series) -> ColumnProfile:
        """Create a ColumnProfile with common fields pre-filled."""
        n = len(series)
        n_miss = int(series.isna().sum())
        n_uniq = int(series.nunique(dropna=True))
        return ColumnProfile(
            name=col,
            dtype=str(series.dtype),
            missing_count=n_miss,
            missing_pct=round(n_miss / max(n, 1), 6),
            unique_count=n_uniq,
            unique_pct=round(n_uniq / max(n, 1), 6),
        )

    def _profile_numeric_col(self, col: str, series: pd.Series) -> ColumnProfile:
        cp = self._base_profile(col, series)
        cp.kind = "numeric"
        stats = self._profile_numeric(series)
        for key, val in stats.items():
            setattr(cp, key, val)
        return cp

    def _profile_categorical_col(self, col: str, series: pd.Series) -> ColumnProfile:
        cp = self._base_profile(col, series)
        cp.kind = "categorical"
        stats = self._profile_categorical(series)
        for key, val in stats.items():
            setattr(cp, key, val)
        return cp

    def _profile_datetime_col(self, col: str, series: pd.Series) -> ColumnProfile:
        cp = self._base_profile(col, series)
        cp.kind = "datetime"
        stats = self._profile_datetime(series)
        for key, val in stats.items():
            setattr(cp, key, val)
        return cp

    def _profile_boolean_col(self, col: str, series: pd.Series) -> ColumnProfile:
        cp = self._base_profile(col, series)
        cp.kind = "boolean"
        # Boolean treated like categorical for top categories
        stats = self._profile_categorical(series.astype(str))
        for key, val in stats.items():
            setattr(cp, key, val)
        return cp

    # ------------------------------------------------------------------ #
    #  Correlations                                                        #
    # ------------------------------------------------------------------ #

    def _compute_correlations(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute Pearson correlations between all numeric column pairs and
        return the top-K most-correlated (by absolute value) pairs.

        Returns
        -------
        dict
            ``{"ColA|ColB": correlation_value, …}``
        """
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return {}

        try:
            corr_matrix = numeric_df.corr(method="pearson", numeric_only=True)
        except Exception as exc:
            logger.warning("Correlation computation failed: %s", exc)
            return {}

        pairs: List[Tuple[str, str, float]] = []
        cols = corr_matrix.columns.tolist()
        for i, c1 in enumerate(cols):
            for j, c2 in enumerate(cols):
                if j <= i:
                    continue
                val = corr_matrix.loc[c1, c2]
                if not math.isnan(val):
                    pairs.append((c1, c2, round(float(val), 6)))

        # Sort by absolute correlation descending
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        top_pairs = pairs[: self.TOP_K_CORRELATIONS]

        return {f"{c1}|{c2}": corr for c1, c2, corr in top_pairs}

    # ------------------------------------------------------------------ #
    #  Serialisation                                                       #
    # ------------------------------------------------------------------ #

    def to_dict(self, profile: DatasetProfile) -> dict:
        """
        Serialize a ``DatasetProfile`` to a plain Python dictionary
        (safe for JSON serialization).

        Parameters
        ----------
        profile : DatasetProfile

        Returns
        -------
        dict
        """
        def _col_to_dict(cp: ColumnProfile) -> dict:
            return {
                "name": cp.name,
                "dtype": cp.dtype,
                "kind": cp.kind,
                "missing_count": cp.missing_count,
                "missing_pct": cp.missing_pct,
                "unique_count": cp.unique_count,
                "unique_pct": cp.unique_pct,
                # numeric
                "mean": cp.mean,
                "median": cp.median,
                "std": cp.std,
                "min": cp.min,
                "max": cp.max,
                "skewness": cp.skewness,
                "kurtosis": cp.kurtosis,
                "q25": cp.q25,
                "q75": cp.q75,
                "outlier_count": cp.outlier_count,
                # categorical
                "top_categories": cp.top_categories,
                "mode": cp.mode,
                # datetime
                "dt_min": cp.dt_min,
                "dt_max": cp.dt_max,
                "dt_range_days": cp.dt_range_days,
            }

        return {
            "dataset_name": profile.dataset_name,
            "num_rows": profile.num_rows,
            "num_columns": profile.num_columns,
            "numeric_columns": profile.numeric_columns,
            "categorical_columns": profile.categorical_columns,
            "datetime_columns": profile.datetime_columns,
            "boolean_columns": profile.boolean_columns,
            "memory_usage_mb": profile.memory_usage_mb,
            "generated_at": profile.generated_at,
            "column_profiles": [_col_to_dict(cp) for cp in profile.column_profiles],
            "correlation_matrix": profile.correlation_matrix or {},
        }

    # ------------------------------------------------------------------ #
    #  ASCII summary                                                       #
    # ------------------------------------------------------------------ #

    def format_summary(self, profile: Optional[DatasetProfile] = None) -> str:
        """
        Render a compact ASCII table summary of the dataset profile.
        """
        if profile is None:
            profile = self._last_profile
        if profile is None:
            return "No profile available."
        W = 80
        HEAVY = "═" * W
        LIGHT = "─" * W
        COL_W = [28, 10, 10, 10, 8, 8, 8]
        HEADERS = ["Column", "Type", "Missing%", "Unique%", "Mean/Top", "Std/Cnt", "Outliers"]

        def fmt_row(*cells: str) -> str:
            parts = []
            for i, cell in enumerate(cells):
                w = COL_W[i] if i < len(COL_W) else 10
                parts.append(str(cell)[:w].ljust(w))
            return "  " + "  ".join(parts)

        lines: List[str] = []

        # ── Header block ──────────────────────────────────────────────
        lines.append(f"\nDATASET PROFILE: {profile.dataset_name}")
        lines.append(HEAVY)
        lines.append(f"  Generated : {profile.generated_at}")
        lines.append(f"  Shape     : {profile.num_rows:,} rows × {profile.num_columns:,} columns")
        lines.append(f"  Memory    : {profile.memory_usage_mb:.2f} MB")
        lines.append(
            f"  Numeric   : {len(profile.numeric_columns)}  |  "
            f"Categorical: {len(profile.categorical_columns)}  |  "
            f"Datetime: {len(profile.datetime_columns)}  |  "
            f"Boolean: {len(profile.boolean_columns)}"
        )

        # ── Column table ──────────────────────────────────────────────
        lines.append(f"\nCOLUMN STATISTICS")
        lines.append(LIGHT)
        lines.append(fmt_row(*HEADERS))
        lines.append(LIGHT)

        for cp in profile.column_profiles:
            miss_str = f"{cp.missing_pct:.1%}"
            uniq_str = f"{cp.unique_pct:.1%}"

            if cp.kind == "numeric":
                mean_str = f"{cp.mean:.4g}" if cp.mean is not None else "—"
                std_str = f"{cp.std:.4g}" if cp.std is not None else "—"
                out_str = str(cp.outlier_count) if cp.outlier_count is not None else "—"
            elif cp.kind in ("categorical", "boolean"):
                # Show most common category
                if cp.top_categories:
                    top_cat, top_cnt = next(iter(cp.top_categories.items()))
                    mean_str = top_cat[:10]
                    std_str = str(top_cnt)
                else:
                    mean_str, std_str = "—", "—"
                out_str = "—"
            elif cp.kind == "datetime":
                mean_str = (cp.dt_min or "")[:10]
                std_str = f"{cp.dt_range_days:.0f}d" if cp.dt_range_days is not None else "—"
                out_str = "—"
            else:
                mean_str = std_str = out_str = "—"

            lines.append(
                fmt_row(cp.name, cp.kind, miss_str, uniq_str, mean_str, std_str, out_str)
            )

        lines.append(LIGHT)

        # ── Top correlations ─────────────────────────────────────────
        if profile.correlation_matrix:
            lines.append(f"\nTOP CORRELATED PAIRS (Pearson |r|)")
            lines.append(LIGHT)
            shown = 0
            for pair, corr in profile.correlation_matrix.items():
                if shown >= 10:
                    break
                c1, c2 = pair.split("|", 1)
                bar_len = int(abs(corr) * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                lines.append(f"  {c1[:20]:<20}  {c2[:20]:<20}  {corr:+.4f}  [{bar}]")
                shown += 1
            lines.append(LIGHT)

        lines.append("")
        return "\n".join(lines)
