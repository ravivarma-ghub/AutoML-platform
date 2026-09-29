"""
app/features/selection.py

Feature selection module for the AutoML platform.
Implements multiple selection strategies (variance, correlation, k-best, model-based)
and an 'auto' mode that chains them for a robust final feature set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    SelectFromModel,
    VarianceThreshold,
    f_classif,
    f_regression,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class FeatureSelectorResult:
    """Encapsulates the outcome of a feature selection pass."""

    selected_features: List[str]
    dropped_features: List[str]
    method_used: str
    feature_scores: Dict[str, float] = field(default_factory=dict)
    report: str = ""

    # Convenience properties
    @property
    def n_selected(self) -> int:
        return len(self.selected_features)

    @property
    def n_dropped(self) -> int:
        return len(self.dropped_features)


# ---------------------------------------------------------------------------
# Main selector
# ---------------------------------------------------------------------------


class FeatureSelector:
    """
    Multi-strategy feature selector.

    Parameters
    ----------
    method : str
        One of ``'auto'``, ``'variance'``, ``'correlation'``,
        ``'kbest'``, or ``'model'``.
    max_features : int, optional
        Maximum number of features to keep (applied as a post-filter when set).
    """

    _VALID_METHODS = {"auto", "variance", "correlation", "kbest", "model", "model_based"}

    def __init__(
        self,
        method: str = "auto",
        max_features: Optional[int] = None,
        threshold: float = 0.01,
        k: Optional[int] = None,
    ) -> None:
        if method not in self._VALID_METHODS:
            raise ValueError(
                f"Unknown method '{method}'. Choose from {self._VALID_METHODS}."
            )
        if method == "model_based":
            method = "model"
        self.method = method
        self.k = k
        self.max_features = k if k is not None else max_features
        self.threshold = threshold

        # State populated during fit_transform
        self._selected_mask: Optional[np.ndarray] = None
        self._selected_features: List[str] = []
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        problem_type: str = "classification",
    ) -> "FeatureSelector":
        """Fit selector to X and y."""
        if feature_names is None:
            feature_names = [f"f_{i}" for i in range(X.shape[1])]
        if y is None:
            y = np.zeros(X.shape[0])
        self.fit_transform(X, y, feature_names, problem_type)
        return self

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        problem_type: str,
    ) -> FeatureSelectorResult:
        """
        Select features from *X* and return a :class:`FeatureSelectorResult`.

        Parameters
        ----------
        X : np.ndarray  shape (n_samples, n_features)
        y : np.ndarray  shape (n_samples,)
        feature_names : List[str]  – names aligned with X columns
        problem_type : str  – ``'classification'`` or ``'regression'``

        Returns
        -------
        FeatureSelectorResult
        """
        if X.shape[1] != len(feature_names):
            raise ValueError(
                f"X has {X.shape[1]} columns but {len(feature_names)} feature names were provided."
            )

        dispatch = {
            "variance": lambda: self._variance_threshold(X, feature_names, threshold=self.threshold),
            "correlation": lambda: self._correlation_filter(X, feature_names),
            "kbest": lambda: self._kbest_selection(X, y, feature_names, problem_type, k=self.k or "all"),
            "model": lambda: self._model_based_selection(X, y, feature_names, problem_type),
            "auto": lambda: self._auto_select(X, y, feature_names, problem_type),
        }

        result: FeatureSelectorResult = dispatch[self.method]()

        # Apply max_features cap
        if self.max_features is not None and result.n_selected > self.max_features:
            result = self._apply_max_features(result)

        # Persist state for transform()
        self._selected_features = result.selected_features
        self._selected_mask = np.array(
            [f in set(result.selected_features) for f in feature_names], dtype=bool
        )
        self._fitted = True

        logger.info(
            "Feature selection (%s): kept %d / %d features.",
            result.method_used,
            result.n_selected,
            len(feature_names),
        )
        return result

    # ------------------------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the fitted selection mask to new data.

        Parameters
        ----------
        X : np.ndarray  shape (n_samples, n_original_features)

        Returns
        -------
        np.ndarray  shape (n_samples, n_selected_features)
        """
        if not self._fitted or self._selected_mask is None:
            raise RuntimeError("FeatureSelector must be fitted before calling transform.")
        return X[:, self._selected_mask]

    # ------------------------------------------------------------------

    def get_selected_mask(self) -> np.ndarray:
        """Return a boolean array indicating which features were selected."""
        if self._selected_mask is None:
            raise RuntimeError("FeatureSelector has not been fitted yet.")
        return self._selected_mask.copy()

    # ------------------------------------------------------------------
    # Selection strategies
    # ------------------------------------------------------------------

    def _variance_threshold(
        self,
        X: np.ndarray,
        feature_names: List[str],
        threshold: float = 0.01,
    ) -> FeatureSelectorResult:
        """
        Remove features with near-zero variance.

        Parameters
        ----------
        threshold : float
            Features with variance below this value are dropped.
        """
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)
        mask: np.ndarray = selector.get_support()

        variances = selector.variances_
        scores = {name: float(variances[i]) for i, name in enumerate(feature_names)}

        selected = [n for n, m in zip(feature_names, mask) if m]
        dropped = [n for n, m in zip(feature_names, mask) if not m]

        report = (
            f"Variance threshold (threshold={threshold}):\n"
            f"  Input features  : {len(feature_names)}\n"
            f"  Selected        : {len(selected)}\n"
            f"  Dropped         : {len(dropped)}\n"
            f"  Dropped columns : {dropped}"
        )

        return FeatureSelectorResult(
            selected_features=selected,
            dropped_features=dropped,
            method_used="variance",
            feature_scores=scores,
            report=report,
        )

    # ------------------------------------------------------------------

    def _correlation_filter(
        self,
        X: np.ndarray,
        feature_names: List[str],
        threshold: float = 0.95,
    ) -> FeatureSelectorResult:
        """
        Remove one feature from each highly correlated pair (|r| ≥ threshold).

        The feature that appears *later* in the column order is dropped so
        that the method is deterministic.
        """
        corr_matrix = np.corrcoef(X, rowvar=False)
        n = len(feature_names)
        to_drop: set = set()

        for i in range(n):
            if i in to_drop:
                continue
            for j in range(i + 1, n):
                if j in to_drop:
                    continue
                if abs(corr_matrix[i, j]) >= threshold:
                    to_drop.add(j)  # drop the later-occurring feature

        mask = np.array([i not in to_drop for i in range(n)], dtype=bool)
        selected = [n_ for n_, m in zip(feature_names, mask) if m]
        dropped = [n_ for n_, m in zip(feature_names, mask) if not m]

        # Scores: max absolute correlation with any other retained feature
        scores: Dict[str, float] = {}
        for i, name in enumerate(feature_names):
            if mask[i]:
                scores[name] = float(np.max(np.abs(corr_matrix[i, mask])))
            else:
                scores[name] = float(np.max(np.abs(corr_matrix[i])))

        report = (
            f"Correlation filter (threshold={threshold}):\n"
            f"  Input features  : {len(feature_names)}\n"
            f"  Selected        : {len(selected)}\n"
            f"  Dropped         : {len(dropped)}\n"
            f"  Dropped columns : {dropped}"
        )

        return FeatureSelectorResult(
            selected_features=selected,
            dropped_features=dropped,
            method_used="correlation",
            feature_scores=scores,
            report=report,
        )

    # ------------------------------------------------------------------

    def _kbest_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        problem_type: str,
        k: int | str = "all",
    ) -> FeatureSelectorResult:
        """
        Select features using univariate statistical tests.

        Uses :func:`sklearn.feature_selection.f_classif` for classification
        and :func:`sklearn.feature_selection.f_regression` for regression.

        Parameters
        ----------
        k : int or ``'all'``
            Number of top features to keep.
        """
        from sklearn.feature_selection import SelectKBest

        score_func = f_classif if problem_type == "classification" else f_regression

        # Replace NaN in X (some score functions cannot handle NaN)
        X_clean = np.nan_to_num(X, nan=0.0)

        n_features = X_clean.shape[1]
        effective_k = n_features if k == "all" else min(int(k), n_features)

        selector = SelectKBest(score_func=score_func, k=effective_k)
        selector.fit(X_clean, y)
        mask: np.ndarray = selector.get_support()

        raw_scores = selector.scores_
        raw_scores = np.nan_to_num(raw_scores, nan=0.0)
        scores = {name: float(raw_scores[i]) for i, name in enumerate(feature_names)}

        selected = [n for n, m in zip(feature_names, mask) if m]
        dropped = [n for n, m in zip(feature_names, mask) if not m]

        report = (
            f"K-Best selection (k={effective_k}, score_func={score_func.__name__}):\n"
            f"  Input features  : {len(feature_names)}\n"
            f"  Selected        : {len(selected)}\n"
            f"  Dropped         : {len(dropped)}\n"
            f"  Top 5 by score  : "
            + str(
                sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
            )
        )

        return FeatureSelectorResult(
            selected_features=selected,
            dropped_features=dropped,
            method_used="kbest",
            feature_scores=scores,
            report=report,
        )

    # ------------------------------------------------------------------

    def _model_based_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        problem_type: str,
        n_estimators: int = 100,
    ) -> FeatureSelectorResult:
        """
        Select features using a Random Forest's feature importances via
        :class:`sklearn.feature_selection.SelectFromModel`.

        A ``RandomForestClassifier`` is used for classification tasks and a
        ``RandomForestRegressor`` for regression tasks.

        Parameters
        ----------
        n_estimators : int
            Number of trees in the Random Forest.
        """
        X_clean = np.nan_to_num(X, nan=0.0)

        if problem_type == "classification":
            estimator = RandomForestClassifier(
                n_estimators=n_estimators,
                n_jobs=-1,
                random_state=42,
            )
        else:
            estimator = RandomForestRegressor(
                n_estimators=n_estimators,
                n_jobs=-1,
                random_state=42,
            )

        # Fit the underlying estimator first so we can read importances
        estimator.fit(X_clean, y)
        importances = estimator.feature_importances_
        scores = {name: float(importances[i]) for i, name in enumerate(feature_names)}

        # SelectFromModel uses mean importance as the default threshold
        sfm = SelectFromModel(estimator=estimator, prefit=True)
        mask: np.ndarray = sfm.get_support()

        # Guarantee at least one feature is selected
        if not mask.any():
            best_idx = int(np.argmax(importances))
            mask[best_idx] = True

        selected = [n for n, m in zip(feature_names, mask) if m]
        dropped = [n for n, m in zip(feature_names, mask) if not m]

        top5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]

        report = (
            f"Model-based selection (RandomForest, n_estimators={n_estimators}):\n"
            f"  Input features  : {len(feature_names)}\n"
            f"  Selected        : {len(selected)}\n"
            f"  Dropped         : {len(dropped)}\n"
            f"  Top 5 importances: {top5}"
        )

        return FeatureSelectorResult(
            selected_features=selected,
            dropped_features=dropped,
            method_used="model",
            feature_scores=scores,
            report=report,
        )

    # ------------------------------------------------------------------

    def _auto_select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        problem_type: str,
    ) -> FeatureSelectorResult:
        """
        Chain selection methods for robust feature reduction:

        1. **Variance threshold** – remove near-zero variance features.
        2. **Correlation filter** – remove highly collinear features.
        3. **Model-based selection** – keep features above mean importance.

        Intermediate results are logged; the final result carries
        ``method_used='auto'`` and an aggregated report.
        """
        reports: List[str] = []
        current_X = X
        current_names = list(feature_names)
        all_dropped: List[str] = []
        all_scores: Dict[str, float] = {}

        # --- Step 1: Variance ------------------------------------------
        if current_X.shape[1] > 1:
            result1 = self._variance_threshold(current_X, current_names)
            mask1 = np.array(
                [n in set(result1.selected_features) for n in current_names], dtype=bool
            )
            current_X = current_X[:, mask1]
            all_dropped.extend(result1.dropped_features)
            all_scores.update(result1.feature_scores)
            current_names = result1.selected_features
            reports.append(f"[Step 1 – Variance]\n{result1.report}")
            logger.debug(
                "Auto-select step 1 (variance): %d → %d features.",
                len(feature_names), len(current_names),
            )

        # --- Step 2: Correlation ----------------------------------------
        if current_X.shape[1] > 1:
            result2 = self._correlation_filter(current_X, current_names)
            mask2 = np.array(
                [n in set(result2.selected_features) for n in current_names], dtype=bool
            )
            current_X = current_X[:, mask2]
            all_dropped.extend(result2.dropped_features)
            all_scores.update(result2.feature_scores)
            current_names = result2.selected_features
            reports.append(f"[Step 2 – Correlation]\n{result2.report}")
            logger.debug(
                "Auto-select step 2 (correlation): → %d features.",
                len(current_names),
            )

        # --- Step 3: Model-based ----------------------------------------
        if current_X.shape[1] > 1:
            result3 = self._model_based_selection(
                current_X, y, current_names, problem_type
            )
            all_dropped.extend(result3.dropped_features)
            all_scores.update(result3.feature_scores)
            current_names = result3.selected_features
            reports.append(f"[Step 3 – Model-based]\n{result3.report}")
            logger.debug(
                "Auto-select step 3 (model): → %d features.",
                len(current_names),
            )

        full_report = (
            "=== Auto Feature Selection ===\n"
            f"  Started with    : {len(feature_names)} features\n"
            f"  Final selection : {len(current_names)} features\n"
            f"  Total dropped   : {len(all_dropped)}\n\n"
        ) + "\n\n".join(reports)

        return FeatureSelectorResult(
            selected_features=current_names,
            dropped_features=all_dropped,
            method_used="auto",
            feature_scores=all_scores,
            report=full_report,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_max_features(self, result: FeatureSelectorResult) -> FeatureSelectorResult:
        """
        Trim *result* to at most ``self.max_features`` features, keeping those
        with the highest scores when scores are available.

        Parameters
        ----------
        result : FeatureSelectorResult

        Returns
        -------
        FeatureSelectorResult  with a capped feature list.
        """
        assert self.max_features is not None

        if result.feature_scores:
            # Sort selected features by score descending
            ranked = sorted(
                result.selected_features,
                key=lambda f: result.feature_scores.get(f, 0.0),
                reverse=True,
            )
        else:
            ranked = list(result.selected_features)

        kept = ranked[: self.max_features]
        extra_dropped = [f for f in result.selected_features if f not in set(kept)]

        trimmed_report = (
            result.report
            + f"\n\n[max_features cap={self.max_features}] "
            f"Further dropped {len(extra_dropped)} features: {extra_dropped}"
        )

        return FeatureSelectorResult(
            selected_features=kept,
            dropped_features=result.dropped_features + extra_dropped,
            method_used=result.method_used,
            feature_scores=result.feature_scores,
            report=trimmed_report,
        )
