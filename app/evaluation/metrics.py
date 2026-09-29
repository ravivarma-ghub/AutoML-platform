"""
app/evaluation/metrics.py
=========================
Model evaluation, cross-validation, benchmarking and champion selection
for the Intelligent AutoML Platform.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """Holds all evaluation metrics for a single model."""

    model_name: str
    problem_type: str  # "classification" | "regression"

    # ── Classification metrics ──────────────────────────────────────────────
    accuracy: Optional[float] = None
    f1_weighted: Optional[float] = None
    f1_macro: Optional[float] = None
    precision_weighted: Optional[float] = None
    recall_weighted: Optional[float] = None
    roc_auc: Optional[float] = None           # one-vs-rest for multiclass
    confusion_matrix: Optional[list] = None
    classification_report: Optional[str] = None

    # ── Regression metrics ──────────────────────────────────────────────────
    r2: Optional[float] = None
    mae: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None              # mean absolute percentage error

    # ── Cross-validation ────────────────────────────────────────────────────
    cv_scores: Optional[List[float]] = field(default=None)
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None

    # ── Training info ───────────────────────────────────────────────────────
    training_time: float = 0.0
    prediction_time_ms: float = 0.0

    # ── Stored reference (not serialised) ──────────────────────────────────
    model: object = field(default=None, repr=False, compare=False)


# ---------------------------------------------------------------------------
# ModelEvaluator
# ---------------------------------------------------------------------------


class ModelEvaluator:
    """
    Evaluates scikit-learn-compatible models on classification and regression
    tasks. Provides cross-validation, benchmarking, champion selection, and
    pretty-printed result tables.

    Parameters
    ----------
    cv_folds : int
        Number of cross-validation folds (default 5).
    random_state : int
        Random seed for reproducibility (default 42).
    """

    _CLASSIFICATION_PRIMARY = "f1_weighted"
    _REGRESSION_PRIMARY = "r2"

    def __init__(self, cv_folds: int = 5, random_state: int = 42) -> None:
        self.cv_folds = cv_folds
        self.random_state = random_state
        logger.info(
            "ModelEvaluator initialised — cv_folds=%d, random_state=%d",
            cv_folds,
            random_state,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        problem_type: str,
        model_name: str = "",
    ) -> EvaluationResult:
        """
        Fully evaluate a *trained* model on a held-out test set and perform
        cross-validation on the training data.

        Parameters
        ----------
        model :
            A fitted scikit-learn estimator.
        X_test, y_test :
            Hold-out test features and labels.
        X_train, y_train :
            Training features and labels used for cross-validation.
        problem_type : str
            ``"classification"`` or ``"regression"``.
        model_name : str
            Human-readable name stored in the result.

        Returns
        -------
        EvaluationResult
        """
        problem_type = problem_type.lower()
        logger.info("Evaluating model '%s' (%s) …", model_name, problem_type)

        result = EvaluationResult(
            model_name=model_name,
            problem_type=problem_type,
            model=model,
        )

        # ── Predict on test set ─────────────────────────────────────────────
        t0 = time.perf_counter()
        y_pred = model.predict(X_test)
        pred_elapsed_ms = (time.perf_counter() - t0) * 1_000
        result.prediction_time_ms = round(pred_elapsed_ms, 3)

        # ── Problem-specific metrics ────────────────────────────────────────
        if problem_type == "classification":
            self._compute_classification_metrics(result, model, X_test, y_test, y_pred)
        elif problem_type == "regression":
            self._compute_regression_metrics(result, y_test, y_pred)
        else:
            raise ValueError(
                f"Unsupported problem_type '{problem_type}'. "
                "Choose 'classification' or 'regression'."
            )

        # ── Cross-validation on training data ───────────────────────────────
        cv_info = self.cross_validate_model(model, X_train, y_train, problem_type)
        result.cv_scores = cv_info["cv_scores"]
        result.cv_mean = cv_info["cv_mean"]
        result.cv_std = cv_info["cv_std"]

        logger.info(
            "Evaluation done for '%s' — CV %.4f ± %.4f | pred %.2f ms",
            model_name,
            result.cv_mean or 0,
            result.cv_std or 0,
            result.prediction_time_ms,
        )
        return result

    def cross_validate_model(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        problem_type: str,
    ) -> Dict[str, object]:
        """
        Perform stratified (classification) or plain (regression)
        k-fold cross-validation.

        Parameters
        ----------
        model :
            A scikit-learn estimator (will be cloned internally).
        X, y :
            Feature matrix and target vector.
        problem_type : str
            ``"classification"`` or ``"regression"``.

        Returns
        -------
        dict
            Keys: ``cv_scores`` (list[float]), ``cv_mean`` (float),
            ``cv_std`` (float).
        """
        from sklearn.base import clone

        problem_type = problem_type.lower()
        estimator = clone(model)

        if problem_type == "classification":
            scoring = "f1_weighted"
            cv = StratifiedKFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_state,
            )
        else:
            scoring = "r2"
            cv = KFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_state,
            )

        try:
            scores = cross_val_score(
                estimator,
                X,
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                error_score="raise",
            )
        except Exception as exc:
            logger.warning("Cross-validation failed: %s", exc)
            scores = np.array([np.nan] * self.cv_folds)

        return {
            "cv_scores": scores.tolist(),
            "cv_mean": float(np.nanmean(scores)),
            "cv_std": float(np.nanstd(scores)),
        }

    def benchmark_models(
        self,
        models: Dict[str, object],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        problem_type: str,
    ) -> List[EvaluationResult]:
        """
        Train and evaluate every model in *models*, returning results sorted
        best-first by the primary metric (f1_weighted / r2).

        Parameters
        ----------
        models : dict
            Mapping ``{model_name: unfitted_estimator}``.
        X_train, y_train :
            Training data.
        X_test, y_test :
            Hold-out test data.
        problem_type : str
            ``"classification"`` or ``"regression"``.

        Returns
        -------
        List[EvaluationResult]
            Sorted best → worst by CV primary metric.
        """
        problem_type = problem_type.lower()
        results: List[EvaluationResult] = []

        for name, estimator in models.items():
            logger.info("Benchmarking: %s …", name)
            try:
                t0 = time.perf_counter()
                estimator.fit(X_train, y_train)
                training_time = time.perf_counter() - t0

                result = self.evaluate(
                    model=estimator,
                    X_test=X_test,
                    y_test=y_test,
                    X_train=X_train,
                    y_train=y_train,
                    problem_type=problem_type,
                    model_name=name,
                )
                result.training_time = round(training_time, 3)
                results.append(result)

            except Exception as exc:
                logger.error("Benchmarking failed for '%s': %s", name, exc, exc_info=True)

        # Sort best → worst by primary CV metric
        primary_attr = (
            "cv_mean" if True else None
        )  # always sort by cv_mean (reflects primary scoring)

        results.sort(
            key=lambda r: (r.cv_mean if r.cv_mean is not None else -np.inf),
            reverse=True,
        )
        logger.info(
            "Benchmarking complete — %d models evaluated, best: %s",
            len(results),
            results[0].model_name if results else "N/A",
        )
        return results

    def select_champion(
        self,
        results: List[EvaluationResult],
        problem_type: str,
    ) -> EvaluationResult:
        """
        Select the best model from a list of evaluation results.

        Primary metric
        --------------
        - Classification → ``f1_weighted`` (falls back to ``cv_mean``)
        - Regression     → ``r2``           (falls back to ``cv_mean``)

        Parameters
        ----------
        results : List[EvaluationResult]
            Non-empty list of evaluation results (need not be sorted).
        problem_type : str
            ``"classification"`` or ``"regression"``.

        Returns
        -------
        EvaluationResult
            The champion result.
        """
        if not results:
            raise ValueError("Cannot select champion from an empty results list.")

        problem_type = problem_type.lower()

        def _score(r: EvaluationResult) -> float:
            if problem_type == "classification":
                return r.f1_weighted if r.f1_weighted is not None else (r.cv_mean or -np.inf)
            else:
                return r.r2 if r.r2 is not None else (r.cv_mean or -np.inf)

        champion = max(results, key=_score)
        logger.info(
            "Champion selected: '%s' (score=%.4f)",
            champion.model_name,
            _score(champion),
        )
        return champion

    def format_benchmark_table(
        self,
        results: List[EvaluationResult],
        problem_type: str,
    ) -> str:
        """
        Render a beautiful ASCII benchmark table.

        Parameters
        ----------
        results : List[EvaluationResult]
            Sorted list of evaluation results (best first).
        problem_type : str
            ``"classification"`` or ``"regression"``.

        Returns
        -------
        str
            Multi-line ASCII table string ready to print or log.
        """
        if not results:
            return "No benchmark results to display."

        problem_type = problem_type.lower()
        is_cls = problem_type == "classification"

        # Column headers differ by problem type
        test_metric_label = "Test F1 " if is_cls else "Test R²  "

        # Determine column widths dynamically
        max_name_len = max(len(r.model_name) for r in results)
        max_name_len = max(max_name_len, len("Model"))
        col_model = max_name_len + 2

        separator_width = 14 + col_model + 10 + 9 + 10 + 9
        sep_line = "═" * separator_width

        lines: List[str] = []
        lines.append("")
        lines.append("  MODEL BENCHMARKING RESULTS")
        lines.append(f"  {sep_line}")

        header = (
            f"  {'Rank':<6}"
            f"{'Model':<{col_model}}"
            f"{'CV Mean':>9}"
            f"{'CV Std':>9}"
            f"{test_metric_label:>10}"
            f"{'Time(s)':>9}"
        )
        lines.append(header)
        lines.append(f"  {sep_line}")

        champion = self.select_champion(results, problem_type)

        for rank, r in enumerate(results, start=1):
            if is_cls:
                test_val = r.f1_weighted
            else:
                test_val = r.r2

            cv_mean_str = f"{r.cv_mean:.3f}" if r.cv_mean is not None else "   N/A"
            cv_std_str = f"{r.cv_std:.3f}" if r.cv_std is not None else "  N/A"
            test_str = f"{test_val:.3f}" if test_val is not None else "   N/A"
            time_str = f"{r.training_time:.1f}"

            marker = " ★" if r.model_name == champion.model_name else "  "
            row = (
                f"  {rank:<6}"
                f"{r.model_name:<{col_model}}"
                f"{cv_mean_str:>9}"
                f"{cv_std_str:>9}"
                f"{test_str:>10}"
                f"{time_str:>9}"
                f"{marker}"
            )
            lines.append(row)

        lines.append(f"  {sep_line}")

        # Champion summary line
        if is_cls:
            champ_metric = "CV F1"
        else:
            champ_metric = "CV R²"

        cv_mean_disp = f"{champion.cv_mean:.3f}" if champion.cv_mean is not None else "N/A"
        cv_std_disp = f"{champion.cv_std:.3f}" if champion.cv_std is not None else "N/A"

        lines.append(
            f"  Champion: {champion.model_name}  "
            f"({champ_metric}: {cv_mean_disp} ± {cv_std_disp})"
        )
        lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_classification_metrics(
        self,
        result: EvaluationResult,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        """Populate *result* with all classification metrics."""
        result.accuracy = float(accuracy_score(y_test, y_pred))
        result.f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        result.f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        result.precision_weighted = float(
            precision_score(y_test, y_pred, average="weighted", zero_division=0)
        )
        result.recall_weighted = float(
            recall_score(y_test, y_pred, average="weighted", zero_division=0)
        )

        # Confusion matrix as plain Python list
        cm = confusion_matrix(y_test, y_pred)
        result.confusion_matrix = cm.tolist()

        # Full classification report string
        result.classification_report = classification_report(
            y_test, y_pred, zero_division=0
        )

        # ROC-AUC (one-vs-rest; requires probability support)
        classes = np.unique(y_test)
        n_classes = len(classes)
        try:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)
                if n_classes == 2:
                    result.roc_auc = float(roc_auc_score(y_test, y_prob[:, 1]))
                else:
                    result.roc_auc = float(
                        roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
                    )
            elif hasattr(model, "decision_function"):
                y_df = model.decision_function(X_test)
                if n_classes == 2:
                    result.roc_auc = float(roc_auc_score(y_test, y_df))
                else:
                    result.roc_auc = float(
                        roc_auc_score(y_test, y_df, multi_class="ovr", average="weighted")
                    )
        except Exception as exc:
            logger.debug("ROC-AUC computation skipped: %s", exc)
            result.roc_auc = None

    def _compute_regression_metrics(
        self,
        result: EvaluationResult,
        y_test: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        """Populate *result* with all regression metrics."""
        result.r2 = float(r2_score(y_test, y_pred))
        result.mae = float(mean_absolute_error(y_test, y_pred))
        result.mse = float(mean_squared_error(y_test, y_pred))
        result.rmse = float(np.sqrt(result.mse))

        # MAPE — guard against zero targets
        try:
            result.mape = float(mean_absolute_percentage_error(y_test, y_pred))
        except Exception:
            # Fallback manual computation that avoids division by zero
            mask = y_test != 0
            if mask.sum() > 0:
                result.mape = float(
                    np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
                )
            else:
                result.mape = None
