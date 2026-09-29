"""
app/evaluation/reports.py
=========================
Professional HTML, JSON, and plot-based report generation for the
Intelligent AutoML Platform.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavyweight imports – graceful degradation if matplotlib / shap
# are not installed.
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    logger.warning("matplotlib not found – plots will be skipped.")

try:
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc as sk_auc
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    import shap as shap_lib
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False
    logger.debug("shap not installed – SHAP plots will be skipped.")


# ---------------------------------------------------------------------------
# Local imports – use TYPE_CHECKING guard to avoid circular imports at runtime
# ---------------------------------------------------------------------------
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.evaluation.metrics import EvaluationResult
    from app.data.profiler import DatasetProfile
    from app.data.validator import ValidationResult
    from app.features.selector import FeatureSelectorResult
    from app.models.registry import ModelMetadata


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------


class ReportGenerator:
    """
    Generates training reports (HTML + JSON + plots) after a full AutoML run.

    Parameters
    ----------
    reports_path : str
        Base directory where reports and plots are written.
    """

    # Colour palette for consistent chart styling
    _PALETTE = [
        "#4C72B0", "#DD8452", "#55A868", "#C44E52",
        "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
        "#CCB974", "#64B5CD",
    ]
    _BG = "#FAFAFA"
    _GRID = "#E0E0E0"

    def __init__(self, reports_path: str) -> None:
        self.reports_path = Path(reports_path)
        self.reports_path.mkdir(parents=True, exist_ok=True)
        logger.info("ReportGenerator ready — reports_path=%s", self.reports_path)

    # ------------------------------------------------------------------
    # Primary public API
    # ------------------------------------------------------------------

    def generate_training_report(
        self,
        profile: "DatasetProfile",
        validation_result: "ValidationResult",
        selector_result: "FeatureSelectorResult",
        benchmark_results: List["EvaluationResult"],
        champion: "EvaluationResult",
        champion_metadata: "ModelMetadata",
        shap_values: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Orchestrate the full report generation pipeline.

        Returns a dict with keys ``report_path`` (HTML), ``json_path``,
        and ``plots_dir``.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{timestamp}"
        run_dir = self.reports_path / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating training report — run_id=%s", run_id)

        # ── Assemble structured report data ─────────────────────────────────
        report_data = self._assemble_report_data(
            profile=profile,
            validation_result=validation_result,
            selector_result=selector_result,
            benchmark_results=benchmark_results,
            champion=champion,
            champion_metadata=champion_metadata,
            shap_values=shap_values,
            run_id=run_id,
            timestamp=timestamp,
        )

        # ── Plots ────────────────────────────────────────────────────────────
        # Attempt to generate plots; skip gracefully if data unavailable
        plot_paths: List[str] = []
        if _HAS_MPL:
            try:
                plot_paths = self._generate_plots(
                    champion_result=champion,
                    X_test=getattr(champion_metadata, "X_test", None),
                    y_test=getattr(champion_metadata, "y_test", None),
                    model=champion.model,
                    feature_names=getattr(selector_result, "selected_features", []),
                    shap_values=shap_values,
                    plots_dir=str(plots_dir),
                )
            except Exception as exc:
                logger.warning("Plot generation encountered an error: %s", exc, exc_info=True)

        report_data["plots"] = plot_paths

        # ── JSON report ──────────────────────────────────────────────────────
        json_path = str(run_dir / "report.json")
        self.generate_json_report(report_data, json_path)

        # ── HTML report ──────────────────────────────────────────────────────
        html_path = str(run_dir / "report.html")
        self.generate_html_report(report_data, html_path)

        logger.info("Reports written to %s", run_dir)
        return {
            "report_path": html_path,
            "json_path": json_path,
            "plots_dir": str(plots_dir),
            "run_id": run_id,
        }

    def _generate_plots(
        self,
        champion_result: "EvaluationResult",
        X_test,
        y_test,
        model,
        feature_names: List[str],
        shap_values: Optional[Dict[str, Any]] = None,
        plots_dir: Optional[str] = None,
    ) -> List[str]:
        """
        Generate all diagnostic plots for the champion model.

        Returns a list of absolute paths to saved PNG files.
        """
        if not _HAS_MPL:
            return []

        pd = Path(plots_dir) if plots_dir else self.reports_path / "plots"
        pd.mkdir(parents=True, exist_ok=True)
        paths: List[str] = []

        problem_type = champion_result.problem_type.lower()

        # ── Benchmark CV scores ──────────────────────────────────────────────
        # We pass a list here; caller should supply all benchmark results.
        # When called from _generate_plots directly, we only have champion.
        try:
            p = self._plot_cv_scores([champion_result], str(pd / "cv_scores.png"))
            if p:
                paths.append(p)
        except Exception as exc:
            logger.debug("cv_scores plot failed: %s", exc)

        if problem_type == "classification":
            # Confusion matrix
            if champion_result.confusion_matrix is not None:
                try:
                    p = self._plot_confusion_matrix(
                        cm=np.array(champion_result.confusion_matrix),
                        class_names=list(
                            map(str, range(len(champion_result.confusion_matrix)))
                        ),
                        save_path=str(pd / "confusion_matrix.png"),
                    )
                    if p:
                        paths.append(p)
                except Exception as exc:
                    logger.debug("confusion_matrix plot failed: %s", exc)

            # ROC curve
            if model is not None and X_test is not None and y_test is not None:
                try:
                    p = self._plot_roc_curve(
                        model=model,
                        X_test=X_test,
                        y_test=y_test,
                        save_path=str(pd / "roc_curve.png"),
                    )
                    if p:
                        paths.append(p)
                except Exception as exc:
                    logger.debug("roc_curve plot failed: %s", exc)

        else:  # regression
            if model is not None and X_test is not None and y_test is not None:
                y_pred = model.predict(X_test)
                try:
                    p = self._plot_residuals(y_test, y_pred, str(pd / "residuals.png"))
                    if p:
                        paths.append(p)
                except Exception as exc:
                    logger.debug("residuals plot failed: %s", exc)

                try:
                    p = self._plot_actual_vs_predicted(
                        y_test, y_pred, str(pd / "actual_vs_predicted.png")
                    )
                    if p:
                        paths.append(p)
                except Exception as exc:
                    logger.debug("actual_vs_predicted plot failed: %s", exc)

        # Feature importance
        if model is not None and feature_names:
            try:
                p = self._plot_feature_importance(
                    model=model,
                    feature_names=feature_names,
                    save_path=str(pd / "feature_importance.png"),
                )
                if p:
                    paths.append(p)
            except Exception as exc:
                logger.debug("feature_importance plot failed: %s", exc)

        # SHAP summary
        if shap_values is not None and feature_names and _HAS_SHAP:
            try:
                p = self._plot_shap_summary(
                    shap_values=shap_values,
                    feature_names=feature_names,
                    save_path=str(pd / "shap_summary.png"),
                )
                if p:
                    paths.append(p)
            except Exception as exc:
                logger.debug("shap_summary plot failed: %s", exc)

        logger.info("Generated %d plots in %s", len(paths), pd)
        return paths

    # ------------------------------------------------------------------
    # Individual plot methods
    # ------------------------------------------------------------------

    def _plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        save_path: str,
    ) -> str:
        """Plot and save a normalised confusion matrix heatmap."""
        if not _HAS_MPL:
            return ""

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor(self._BG)

        for ax, (data, title) in zip(
            axes,
            [
                (cm, "Counts"),
                (
                    cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1),
                    "Normalised",
                ),
            ],
        ):
            im = ax.imshow(data, interpolation="nearest", cmap="Blues")
            ax.set_facecolor(self._BG)
            plt.colorbar(im, ax=ax)
            ax.set_title(f"Confusion Matrix — {title}", fontsize=13, fontweight="bold", pad=12)
            tick_marks = np.arange(len(class_names))
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
            ax.set_yticklabels(class_names, fontsize=9)
            ax.set_ylabel("True label", fontsize=11)
            ax.set_xlabel("Predicted label", fontsize=11)

            thresh = data.max() / 2.0
            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    val = f"{data[i, j]:.2f}" if title == "Normalised" else str(int(data[i, j]))
                    ax.text(
                        j, i, val,
                        ha="center", va="center",
                        color="white" if data[i, j] > thresh else "black",
                        fontsize=9,
                    )

        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=self._BG)
        plt.close(fig)
        logger.debug("Saved confusion_matrix → %s", save_path)
        return save_path

    def _plot_roc_curve(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_path: str,
    ) -> str:
        """Plot ROC curve(s); handles binary and multiclass (OvR)."""
        if not _HAS_MPL or not _HAS_SKLEARN:
            return ""

        classes = np.unique(y_test)
        n_classes = len(classes)

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor(self._BG)
        ax.set_facecolor(self._BG)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
            if y_prob.ndim == 1:
                y_prob = np.column_stack([-y_prob, y_prob])
        else:
            ax.text(0.5, 0.5, "Model does not support probability output",
                    ha="center", va="center", transform=ax.transAxes)
            plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=self._BG)
            plt.close(fig)
            return save_path

        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            roc_auc = sk_auc(fpr, tpr)
            ax.plot(fpr, tpr, color=self._PALETTE[0], lw=2,
                    label=f"ROC (AUC = {roc_auc:.3f})")
        else:
            y_bin = label_binarize(y_test, classes=classes)
            for i, cls in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                roc_auc = sk_auc(fpr, tpr)
                ax.plot(fpr, tpr, color=self._PALETTE[i % len(self._PALETTE)],
                        lw=1.5, label=f"Class {cls} (AUC = {roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Random (AUC = 0.500)")
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, color=self._GRID, linestyle="--", linewidth=0.7)

        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=self._BG)
        plt.close(fig)
        logger.debug("Saved roc_curve → %s", save_path)
        return save_path

    def _plot_feature_importance(
        self,
        model,
        feature_names: List[str],
        save_path: str,
        top_n: int = 20,
    ) -> str:
        """
        Plot a horizontal bar chart of feature importances.
        Supports ``feature_importances_`` and ``coef_`` attributes.
        """
        if not _HAS_MPL:
            return ""

        importances: Optional[np.ndarray] = None

        if hasattr(model, "feature_importances_"):
            importances = np.asarray(model.feature_importances_)
        elif hasattr(model, "coef_"):
            coef = np.asarray(model.coef_)
            importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)

        if importances is None or len(importances) == 0:
            logger.debug("No feature importances available for this model.")
            return ""

        # Align with feature_names length
        n = min(len(importances), len(feature_names))
        importances = importances[:n]
        names = feature_names[:n]

        # Select top-N
        idx = np.argsort(importances)[-top_n:]
        top_importances = importances[idx]
        top_names = [names[i] for i in idx]

        fig, ax = plt.subplots(figsize=(10, max(4, len(top_names) * 0.35 + 1)))
        fig.patch.set_facecolor(self._BG)
        ax.set_facecolor(self._BG)

        colours = [self._PALETTE[i % len(self._PALETTE)] for i in range(len(top_names))]
        bars = ax.barh(range(len(top_names)), top_importances, color=colours, edgecolor="white")
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names, fontsize=9)
        ax.set_xlabel("Importance", fontsize=11)
        ax.set_title(f"Feature Importance (Top {len(top_names)})", fontsize=13, fontweight="bold")
        ax.grid(True, axis="x", color=self._GRID, linestyle="--", linewidth=0.7)

        for bar, val in zip(bars, top_importances):
            ax.text(
                bar.get_width() + top_importances.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center", ha="left", fontsize=8,
            )

        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=self._BG)
        plt.close(fig)
        logger.debug("Saved feature_importance → %s", save_path)
        return save_path

    def _plot_residuals(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        save_path: str,
    ) -> str:
        """Residual plot for regression models."""
        if not _HAS_MPL:
            return ""

        residuals = np.asarray(y_test) - np.asarray(y_pred)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor(self._BG)

        # Residuals vs predicted
        ax = axes[0]
        ax.set_facecolor(self._BG)
        ax.scatter(y_pred, residuals, alpha=0.5, s=20, color=self._PALETTE[0], edgecolors="none")
        ax.axhline(0, color="#C44E52", linewidth=1.5, linestyle="--")
        ax.set_xlabel("Predicted Values", fontsize=11)
        ax.set_ylabel("Residuals", fontsize=11)
        ax.set_title("Residuals vs Predicted", fontsize=13, fontweight="bold")
        ax.grid(True, color=self._GRID, linestyle="--", linewidth=0.7)

        # Residual histogram
        ax2 = axes[1]
        ax2.set_facecolor(self._BG)
        ax2.hist(residuals, bins=40, color=self._PALETTE[1], edgecolor="white", linewidth=0.5)
        ax2.axvline(0, color="#C44E52", linewidth=1.5, linestyle="--")
        ax2.set_xlabel("Residual", fontsize=11)
        ax2.set_ylabel("Count", fontsize=11)
        ax2.set_title("Residual Distribution", fontsize=13, fontweight="bold")
        ax2.grid(True, color=self._GRID, linestyle="--", linewidth=0.7)

        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=self._BG)
        plt.close(fig)
        logger.debug("Saved residuals → %s", save_path)
        return save_path

    def _plot_actual_vs_predicted(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        save_path: str,
    ) -> str:
        """Actual vs Predicted scatter plot for regression models."""
        if not _HAS_MPL:
            return ""

        y_test_arr = np.asarray(y_test)
        y_pred_arr = np.asarray(y_pred)

        fig, ax = plt.subplots(figsize=(7, 6))
        fig.patch.set_facecolor(self._BG)
        ax.set_facecolor(self._BG)

        ax.scatter(y_test_arr, y_pred_arr, alpha=0.5, s=20,
                   color=self._PALETTE[0], edgecolors="none")

        lo = min(y_test_arr.min(), y_pred_arr.min())
        hi = max(y_test_arr.max(), y_pred_arr.max())
        pad = (hi - lo) * 0.05
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                "r--", lw=1.5, label="Perfect fit")

        ax.set_xlabel("Actual Values", fontsize=11)
        ax.set_ylabel("Predicted Values", fontsize=11)
        ax.set_title("Actual vs Predicted", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, color=self._GRID, linestyle="--", linewidth=0.7)

        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=self._BG)
        plt.close(fig)
        logger.debug("Saved actual_vs_predicted → %s", save_path)
        return save_path

    def _plot_cv_scores(
        self,
        results: List["EvaluationResult"],
        save_path: str,
    ) -> str:
        """
        Bar chart of CV mean scores with error bars (std) for all benchmarked models.
        """
        if not _HAS_MPL:
            return ""

        valid = [r for r in results if r.cv_mean is not None]
        if not valid:
            return ""

        names = [r.model_name for r in valid]
        means = [r.cv_mean for r in valid]
        stds = [r.cv_std or 0 for r in valid]

        # Sort by mean descending
        order = np.argsort(means)[::-1]
        names = [names[i] for i in order]
        means = [means[i] for i in order]
        stds = [stds[i] for i in order]

        fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))
        fig.patch.set_facecolor(self._BG)
        ax.set_facecolor(self._BG)

        x = np.arange(len(names))
        colours = [self._PALETTE[i % len(self._PALETTE)] for i in range(len(names))]
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colours,
                      edgecolor="white", linewidth=0.8, error_kw={"elinewidth": 1.5})

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("CV Score", fontsize=11)
        ax.set_title("Cross-Validation Scores by Model", fontsize=13, fontweight="bold")
        ax.grid(True, axis="y", color=self._GRID, linestyle="--", linewidth=0.7)

        for bar, mean, std in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.002,
                f"{mean:.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=self._BG)
        plt.close(fig)
        logger.debug("Saved cv_scores → %s", save_path)
        return save_path

    def _plot_shap_summary(
        self,
        shap_values: Dict[str, Any],
        feature_names: List[str],
        save_path: str,
    ) -> str:
        """
        Generate a SHAP beeswarm / summary plot.

        *shap_values* dict must contain key ``"values"`` (np.ndarray) and
        optionally ``"data"`` (background data for beeswarm).
        """
        if not _HAS_MPL or not _HAS_SHAP:
            return ""

        sv = shap_values.get("values")
        data = shap_values.get("data")

        if sv is None:
            return ""

        sv_arr = np.asarray(sv)
        # For multiclass, take mean absolute across classes
        if sv_arr.ndim == 3:
            sv_arr = sv_arr.mean(axis=2)

        fig, ax = plt.subplots(figsize=(10, max(5, len(feature_names) * 0.35 + 1)))
        fig.patch.set_facecolor(self._BG)

        try:
            if data is not None:
                shap_lib.summary_plot(
                    sv_arr,
                    features=data,
                    feature_names=feature_names,
                    show=False,
                    plot_type="dot",
                )
            else:
                shap_lib.summary_plot(
                    sv_arr,
                    feature_names=feature_names,
                    show=False,
                    plot_type="bar",
                )
        except Exception as exc:
            logger.warning("SHAP summary_plot failed: %s", exc)
            plt.close(fig)
            return ""

        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=self._BG)
        plt.close("all")
        logger.debug("Saved shap_summary → %s", save_path)
        return save_path

    # ------------------------------------------------------------------
    # Report serialisation
    # ------------------------------------------------------------------

    def generate_html_report(
        self,
        report_data: Dict[str, Any],
        output_path: str,
    ) -> str:
        """
        Generate a professional, standalone HTML report with all plots embedded
        as base64 data URIs.

        Parameters
        ----------
        report_data : dict
            Structured report data from ``_assemble_report_data``.
        output_path : str
            Destination HTML file path.

        Returns
        -------
        str
            Absolute path to the written HTML file.
        """
        plots = report_data.get("plots", [])
        embedded: Dict[str, str] = {}
        for p in plots:
            if p and Path(p).exists():
                key = Path(p).stem
                embedded[key] = self._embed_plot_base64(p)

        html = self._build_html(report_data, embedded)
        output_path = str(output_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(html)

        logger.info("HTML report written → %s", output_path)
        return output_path

    def generate_json_report(
        self,
        report_data: Dict[str, Any],
        output_path: str,
    ) -> str:
        """
        Serialise report_data to a machine-readable JSON file.

        Parameters
        ----------
        report_data : dict
            Structured report data.
        output_path : str
            Destination JSON file path.

        Returns
        -------
        str
            Absolute path to the written JSON file.
        """
        output_path = str(output_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Make serialisable — remove non-serialisable objects (model instances etc.)
        safe_data = _make_json_safe(report_data)

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(safe_data, fh, indent=2, default=str)

        logger.info("JSON report written → %s", output_path)
        return output_path

    def _embed_plot_base64(self, plot_path: str) -> str:
        """
        Read a PNG file and return a ``data:image/png;base64,...`` URI string.
        """
        try:
            with open(plot_path, "rb") as fh:
                encoded = base64.b64encode(fh.read()).decode("utf-8")
            return f"data:image/png;base64,{encoded}"
        except Exception as exc:
            logger.warning("Failed to embed plot %s: %s", plot_path, exc)
            return ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assemble_report_data(
        self,
        profile,
        validation_result,
        selector_result,
        benchmark_results: List["EvaluationResult"],
        champion: "EvaluationResult",
        champion_metadata,
        shap_values: Optional[Dict[str, Any]],
        run_id: str,
        timestamp: str,
    ) -> Dict[str, Any]:
        """
        Build the canonical report_data dictionary from all pipeline outputs.
        """
        # ── Dataset overview ─────────────────────────────────────────────────
        dataset_overview = {}
        if profile is not None:
            dataset_overview = {
                "n_rows": getattr(profile, "n_rows", None),
                "n_cols": getattr(profile, "n_cols", None),
                "target_column": getattr(profile, "target_column", None),
                "problem_type": getattr(profile, "problem_type", None),
                "memory_mb": getattr(profile, "memory_mb", None),
                "n_numeric": getattr(profile, "n_numeric", None),
                "n_categorical": getattr(profile, "n_categorical", None),
                "class_distribution": getattr(profile, "class_distribution", None),
            }

        # ── Data quality ─────────────────────────────────────────────────────
        data_quality = {}
        if validation_result is not None:
            data_quality = {
                "is_valid": getattr(validation_result, "is_valid", None),
                "warnings": getattr(validation_result, "warnings", []),
                "errors": getattr(validation_result, "errors", []),
                "missing_pct": getattr(validation_result, "missing_pct", None),
                "duplicate_rows": getattr(validation_result, "duplicate_rows", None),
                "outlier_cols": getattr(validation_result, "outlier_cols", []),
            }

        # ── Feature engineering ──────────────────────────────────────────────
        feature_summary = {}
        if selector_result is not None:
            feature_summary = {
                "n_original": getattr(selector_result, "n_original_features", None),
                "n_selected": getattr(selector_result, "n_selected_features", None),
                "selected_features": getattr(selector_result, "selected_features", []),
                "dropped_features": getattr(selector_result, "dropped_features", []),
                "method": getattr(selector_result, "method", None),
                "importance_scores": getattr(selector_result, "importance_scores", {}),
            }

        # ── Benchmark results ────────────────────────────────────────────────
        benchmark_list = []
        for r in benchmark_results:
            benchmark_list.append(
                {
                    "model_name": r.model_name,
                    "problem_type": r.problem_type,
                    "cv_mean": r.cv_mean,
                    "cv_std": r.cv_std,
                    "cv_scores": r.cv_scores,
                    "accuracy": r.accuracy,
                    "f1_weighted": r.f1_weighted,
                    "f1_macro": r.f1_macro,
                    "precision_weighted": r.precision_weighted,
                    "recall_weighted": r.recall_weighted,
                    "roc_auc": r.roc_auc,
                    "r2": r.r2,
                    "mae": r.mae,
                    "mse": r.mse,
                    "rmse": r.rmse,
                    "mape": r.mape,
                    "training_time": r.training_time,
                    "prediction_time_ms": r.prediction_time_ms,
                }
            )

        # ── Champion details ─────────────────────────────────────────────────
        champion_info = {
            "model_name": champion.model_name,
            "problem_type": champion.problem_type,
            "cv_mean": champion.cv_mean,
            "cv_std": champion.cv_std,
            "accuracy": champion.accuracy,
            "f1_weighted": champion.f1_weighted,
            "f1_macro": champion.f1_macro,
            "precision_weighted": champion.precision_weighted,
            "recall_weighted": champion.recall_weighted,
            "roc_auc": champion.roc_auc,
            "r2": champion.r2,
            "mae": champion.mae,
            "mse": champion.mse,
            "rmse": champion.rmse,
            "mape": champion.mape,
            "training_time": champion.training_time,
            "prediction_time_ms": champion.prediction_time_ms,
            "confusion_matrix": champion.confusion_matrix,
            "classification_report": champion.classification_report,
        }

        metadata_info = {}
        if champion_metadata is not None:
            metadata_info = {
                "model_id": getattr(champion_metadata, "model_id", None),
                "version": getattr(champion_metadata, "version", None),
                "created_at": str(getattr(champion_metadata, "created_at", "")),
                "framework": getattr(champion_metadata, "framework", None),
                "hyperparameters": getattr(champion_metadata, "hyperparameters", {}),
                "model_path": getattr(champion_metadata, "model_path", None),
            }

        shap_info = None
        if shap_values is not None:
            shap_info = {
                "feature_names": shap_values.get("feature_names", []),
                "mean_abs_shap": (
                    np.abs(np.asarray(shap_values["values"])).mean(axis=0).tolist()
                    if "values" in shap_values
                    else None
                ),
            }

        return {
            "run_id": run_id,
            "timestamp": timestamp,
            "generated_at": datetime.now().isoformat(),
            "platform": "Intelligent AutoML Platform",
            "dataset_overview": dataset_overview,
            "data_quality": data_quality,
            "feature_summary": feature_summary,
            "benchmark_results": benchmark_list,
            "champion": champion_info,
            "champion_metadata": metadata_info,
            "shap_analysis": shap_info,
            "plots": [],   # filled by caller after plot generation
        }

    # ------------------------------------------------------------------
    # HTML builder
    # ------------------------------------------------------------------

    def _build_html(
        self,
        data: Dict[str, Any],
        embedded_plots: Dict[str, str],
    ) -> str:
        """
        Compose a self-contained, responsive HTML report string.
        """
        champion = data.get("champion", {})
        ds = data.get("dataset_overview", {})
        dq = data.get("data_quality", {})
        feat = data.get("feature_summary", {})
        bench = data.get("benchmark_results", [])
        shap_data = data.get("shap_analysis")
        meta = data.get("champion_metadata", {})

        # ── Helper formatters ────────────────────────────────────────────────
        def _v(val, fmt=".4f") -> str:
            if val is None:
                return "—"
            try:
                return format(float(val), fmt)
            except (TypeError, ValueError):
                return str(val)

        def _pct(val) -> str:
            if val is None:
                return "—"
            return f"{float(val):.1f}%"

        def _badge(ok: bool) -> str:
            colour = "#27ae60" if ok else "#e74c3c"
            label = "PASS" if ok else "FAIL"
            return f'<span style="background:{colour};color:#fff;padding:2px 8px;border-radius:4px;font-size:0.8rem;">{label}</span>'

        def _img_section(key: str, title: str) -> str:
            uri = embedded_plots.get(key, "")
            if not uri:
                return ""
            return (
                f'<div class="plot-card">'
                f'<h3 class="plot-title">{title}</h3>'
                f'<img src="{uri}" alt="{title}" style="max-width:100%;border-radius:6px;" />'
                f'</div>'
            )

        # ── Benchmark table rows ─────────────────────────────────────────────
        is_cls = champion.get("problem_type", "classification") == "classification"

        bench_rows = ""
        for i, r in enumerate(bench, start=1):
            test_m = _v(r.get("f1_weighted") if is_cls else r.get("r2"))
            highlight = 'style="background:#eaf7ea;"' if i == 1 else ""
            bench_rows += (
                f"<tr {highlight}>"
                f"<td>{i}</td>"
                f"<td><strong>{r.get('model_name','')}</strong></td>"
                f"<td>{_v(r.get('cv_mean'))}</td>"
                f"<td>{_v(r.get('cv_std'))}</td>"
                f"<td>{test_m}</td>"
                f"<td>{_v(r.get('training_time'), '.1f')} s</td>"
                f"<td>{_v(r.get('prediction_time_ms'), '.1f')} ms</td>"
                f"</tr>"
            )

        test_metric_col = "Test F1" if is_cls else "Test R²"

        # ── Classification metrics panel ─────────────────────────────────────
        if is_cls:
            champ_metrics_html = f"""
            <div class="metric-grid">
              <div class="metric-card"><div class="metric-val">{_v(champion.get('accuracy'), '.4f')}</div><div class="metric-lbl">Accuracy</div></div>
              <div class="metric-card"><div class="metric-val">{_v(champion.get('f1_weighted'), '.4f')}</div><div class="metric-lbl">F1 Weighted</div></div>
              <div class="metric-card"><div class="metric-val">{_v(champion.get('f1_macro'), '.4f')}</div><div class="metric-lbl">F1 Macro</div></div>
              <div class="metric-card"><div class="metric-val">{_v(champion.get('precision_weighted'), '.4f')}</div><div class="metric-lbl">Precision (W)</div></div>
              <div class="metric-card"><div class="metric-val">{_v(champion.get('recall_weighted'), '.4f')}</div><div class="metric-lbl">Recall (W)</div></div>
              <div class="metric-card"><div class="metric-val">{_v(champion.get('roc_auc'), '.4f')}</div><div class="metric-lbl">ROC-AUC</div></div>
              <div class="metric-card"><div class="metric-val">{_v(champion.get('cv_mean'), '.4f')}</div><div class="metric-lbl">CV Mean F1</div></div>
              <div class="metric-card"><div class="metric-val">{_v(champion.get('cv_std'), '.4f')}</div><div class="metric-lbl">CV Std</div></div>
            </div>"""
        else:
            champ_metrics_html = f"""
            <div class="metric-grid">
              <div class="metric-card"><div class="metric-val">{_v(champion.get('r2'), '.4f')}</div><div class="metric-lbl">R²</div></div>
              <div class="metric-card"><div class="metric-val">{_v(champion.get('mae'))}</div><div class="metric-lbl">MAE</div></div>
              <div class="metric-card"><div class="metric-val">{_v(champion.get('mse'))}</div><div class="metric-lbl">MSE</div></div>
              <div class="metric-card"><div class="metric-val">{_v(champion.get('rmse'))}</div><div class="metric-lbl">RMSE</div></div>
              <div class="metric-card"><div class="metric-val">{_pct(champion.get('mape'))}</div><div class="metric-lbl">MAPE</div></div>
              <div class="metric-card"><div class="metric-val">{_v(champion.get('cv_mean'), '.4f')}</div><div class="metric-lbl">CV Mean R²</div></div>
              <div class="metric-card"><div class="metric-val">{_v(champion.get('cv_std'), '.4f')}</div><div class="metric-lbl">CV Std</div></div>
            </div>"""

        # ── Classification report pre block ──────────────────────────────────
        cls_report_section = ""
        if champion.get("classification_report"):
            cls_report_section = f"""
            <div class="section">
              <h2>📋 Classification Report</h2>
              <pre class="mono-block">{champion['classification_report']}</pre>
            </div>"""

        # ── SHAP section ─────────────────────────────────────────────────────
        shap_section = ""
        if shap_data:
            feat_names = shap_data.get("feature_names", [])
            mean_abs = shap_data.get("mean_abs_shap", [])
            shap_rows = ""
            if feat_names and mean_abs:
                pairs = sorted(zip(feat_names, mean_abs), key=lambda x: -x[1])[:15]
                for fn, mv in pairs:
                    shap_rows += f"<tr><td>{fn}</td><td>{mv:.5f}</td></tr>"

            shap_plot_html = _img_section("shap_summary", "SHAP Summary Plot")
            shap_section = f"""
            <div class="section">
              <h2>🔬 SHAP Analysis</h2>
              <p>SHAP (SHapley Additive exPlanations) values quantify each feature's
              contribution to individual predictions.</p>
              {f'<table class="data-table"><thead><tr><th>Feature</th><th>Mean |SHAP|</th></tr></thead><tbody>{shap_rows}</tbody></table>' if shap_rows else ''}
              {shap_plot_html}
            </div>"""

        # ── Feature list ─────────────────────────────────────────────────────
        selected_feats = feat.get("selected_features", [])
        feat_list_html = ""
        if selected_feats:
            items = "".join(f"<span class='feat-badge'>{f}</span>" for f in selected_feats[:50])
            feat_list_html = f"<div class='feat-list'>{items}</div>"
            if len(selected_feats) > 50:
                feat_list_html += f"<p><em>…and {len(selected_feats)-50} more.</em></p>"

        # ── Warnings / errors ─────────────────────────────────────────────────
        warnings = dq.get("warnings", []) or []
        errors = dq.get("errors", []) or []
        warn_html = "".join(f"<li class='warn-item'>⚠️ {w}</li>" for w in warnings)
        err_html = "".join(f"<li class='err-item'>❌ {e}</li>" for e in errors)
        quality_list = f"<ul>{warn_html}{err_html}</ul>" if (warn_html or err_html) else "<p>No issues detected.</p>"

        # ── Hyperparameters table ─────────────────────────────────────────────
        hp = meta.get("hyperparameters", {}) or {}
        hp_rows = "".join(
            f"<tr><td>{k}</td><td><code>{v}</code></td></tr>"
            for k, v in hp.items()
        )
        hp_section = (
            f'<table class="data-table"><thead><tr><th>Parameter</th><th>Value</th></tr></thead>'
            f'<tbody>{hp_rows}</tbody></table>'
            if hp_rows
            else "<p>No hyperparameter information available.</p>"
        )

        # ── Full HTML ─────────────────────────────────────────────────────────
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AutoML Report — {data.get('run_id', '')}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, sans-serif;
      background: #f0f2f5;
      color: #1a1a2e;
      line-height: 1.6;
    }}
    header {{
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
      color: #fff;
      padding: 40px 48px;
    }}
    header h1 {{ font-size: 2rem; font-weight: 700; letter-spacing: 0.5px; }}
    header p {{ opacity: 0.75; font-size: 0.95rem; margin-top: 6px; }}
    .badge {{
      display: inline-block;
      background: #e94560;
      color: #fff;
      padding: 4px 12px;
      border-radius: 20px;
      font-size: 0.8rem;
      margin-top: 10px;
      font-weight: 600;
    }}
    main {{ max-width: 1200px; margin: 0 auto; padding: 32px 24px; }}
    .section {{
      background: #fff;
      border-radius: 12px;
      padding: 28px 32px;
      margin-bottom: 28px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    }}
    .section h2 {{
      font-size: 1.25rem;
      font-weight: 700;
      margin-bottom: 18px;
      padding-bottom: 10px;
      border-bottom: 2px solid #f0f2f5;
      color: #0f3460;
    }}
    .info-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
      gap: 16px;
    }}
    .info-card {{
      background: #f7f8fc;
      border-radius: 8px;
      padding: 14px 16px;
    }}
    .info-card .label {{ font-size: 0.78rem; color: #666; text-transform: uppercase; letter-spacing: 0.4px; }}
    .info-card .value {{ font-size: 1.15rem; font-weight: 700; margin-top: 4px; color: #1a1a2e; }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
      gap: 14px;
      margin-top: 6px;
    }}
    .metric-card {{
      background: linear-gradient(135deg, #0f3460, #16213e);
      color: #fff;
      border-radius: 10px;
      padding: 18px 14px;
      text-align: center;
    }}
    .metric-val {{ font-size: 1.5rem; font-weight: 800; letter-spacing: 0.5px; }}
    .metric-lbl {{ font-size: 0.75rem; opacity: 0.8; margin-top: 4px; }}
    .champion-banner {{
      background: linear-gradient(135deg, #27ae60, #219a52);
      color: #fff;
      border-radius: 10px;
      padding: 16px 20px;
      margin-bottom: 20px;
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .champion-banner .trophy {{ font-size: 2rem; }}
    .champion-banner .name {{ font-size: 1.3rem; font-weight: 700; }}
    .champion-banner .sub {{ font-size: 0.85rem; opacity: 0.85; }}
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.88rem;
    }}
    .data-table thead tr {{ background: #0f3460; color: #fff; }}
    .data-table th {{ padding: 10px 14px; text-align: left; font-weight: 600; }}
    .data-table td {{ padding: 9px 14px; border-bottom: 1px solid #f0f2f5; }}
    .data-table tbody tr:hover {{ background: #f7f8fc; }}
    .data-table tbody tr:first-child td {{ font-weight: 600; }}
    .plot-card {{ margin-top: 20px; }}
    .plot-title {{ font-size: 1rem; font-weight: 600; color: #444; margin-bottom: 10px; }}
    .plot-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
      gap: 20px;
      margin-top: 10px;
    }}
    .feat-list {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }}
    .feat-badge {{
      background: #edf2ff;
      color: #364fc7;
      border-radius: 6px;
      padding: 3px 10px;
      font-size: 0.8rem;
      font-weight: 500;
    }}
    .mono-block {{
      background: #1a1a2e;
      color: #a8ff78;
      padding: 16px 20px;
      border-radius: 8px;
      font-size: 0.82rem;
      overflow-x: auto;
      white-space: pre;
      font-family: 'Consolas', 'Courier New', monospace;
    }}
    .warn-item {{ color: #b7791f; list-style: none; padding: 4px 0; }}
    .err-item  {{ color: #c53030; list-style: none; padding: 4px 0; }}
    footer {{
      text-align: center;
      padding: 28px;
      color: #888;
      font-size: 0.82rem;
    }}
  </style>
</head>
<body>

<header>
  <h1>🤖 Intelligent AutoML Platform</h1>
  <p>Training Report · Run ID: <strong>{data.get('run_id', 'N/A')}</strong></p>
  <p>Generated: {data.get('generated_at', '')}</p>
  <span class="badge">{champion.get('problem_type', '').upper()}</span>
</header>

<main>

  <!-- Dataset Overview -->
  <div class="section">
    <h2>📊 Dataset Overview</h2>
    <div class="info-grid">
      <div class="info-card"><div class="label">Rows</div><div class="value">{ds.get('n_rows', '—')}</div></div>
      <div class="info-card"><div class="label">Columns</div><div class="value">{ds.get('n_cols', '—')}</div></div>
      <div class="info-card"><div class="label">Target</div><div class="value">{ds.get('target_column', '—')}</div></div>
      <div class="info-card"><div class="label">Problem Type</div><div class="value">{ds.get('problem_type', '—')}</div></div>
      <div class="info-card"><div class="label">Memory</div><div class="value">{_v(ds.get('memory_mb'), '.1f')} MB</div></div>
      <div class="info-card"><div class="label">Numeric Cols</div><div class="value">{ds.get('n_numeric', '—')}</div></div>
      <div class="info-card"><div class="label">Categorical Cols</div><div class="value">{ds.get('n_categorical', '—')}</div></div>
    </div>
  </div>

  <!-- Data Quality -->
  <div class="section">
    <h2>🔍 Data Quality</h2>
    <div class="info-grid" style="margin-bottom:16px;">
      <div class="info-card"><div class="label">Validation</div><div class="value">{_badge(dq.get('is_valid', True))}</div></div>
      <div class="info-card"><div class="label">Missing %</div><div class="value">{_pct(dq.get('missing_pct'))}</div></div>
      <div class="info-card"><div class="label">Duplicate Rows</div><div class="value">{dq.get('duplicate_rows', '—')}</div></div>
    </div>
    {quality_list}
  </div>

  <!-- Feature Engineering -->
  <div class="section">
    <h2>⚙️ Feature Engineering Summary</h2>
    <div class="info-grid" style="margin-bottom:16px;">
      <div class="info-card"><div class="label">Original Features</div><div class="value">{feat.get('n_original', '—')}</div></div>
      <div class="info-card"><div class="label">Selected Features</div><div class="value">{feat.get('n_selected', '—')}</div></div>
      <div class="info-card"><div class="label">Selection Method</div><div class="value">{feat.get('method', '—')}</div></div>
    </div>
    {feat_list_html}
  </div>

  <!-- Benchmark -->
  <div class="section">
    <h2>🏆 Model Benchmarking</h2>
    <table class="data-table">
      <thead>
        <tr>
          <th>Rank</th><th>Model</th>
          <th>CV Mean</th><th>CV Std</th>
          <th>{test_metric_col}</th>
          <th>Train Time</th><th>Pred Time</th>
        </tr>
      </thead>
      <tbody>{bench_rows}</tbody>
    </table>
    {_img_section('cv_scores', 'Cross-Validation Score Comparison')}
  </div>

  <!-- Champion -->
  <div class="section">
    <h2>🥇 Champion Model</h2>
    <div class="champion-banner">
      <div class="trophy">🏆</div>
      <div>
        <div class="name">{champion.get('model_name', '—')}</div>
        <div class="sub">
          CV Score: {_v(champion.get('cv_mean'))} ± {_v(champion.get('cv_std'))} &nbsp;|&nbsp;
          Train Time: {_v(champion.get('training_time'), '.1f')} s &nbsp;|&nbsp;
          Pred Time: {_v(champion.get('prediction_time_ms'), '.1f')} ms
        </div>
      </div>
    </div>

    <h3 style="margin:16px 0 10px;font-size:1rem;color:#555;">Evaluation Metrics</h3>
    {champ_metrics_html}

    <h3 style="margin:22px 0 10px;font-size:1rem;color:#555;">Hyperparameters</h3>
    {hp_section}
  </div>

  {cls_report_section}

  <!-- Plots -->
  <div class="section">
    <h2>📈 Evaluation Plots</h2>
    <div class="plot-grid">
      {_img_section('confusion_matrix', 'Confusion Matrix')}
      {_img_section('roc_curve', 'ROC Curve')}
      {_img_section('feature_importance', 'Feature Importance')}
      {_img_section('residuals', 'Residual Analysis')}
      {_img_section('actual_vs_predicted', 'Actual vs Predicted')}
    </div>
  </div>

  {shap_section}

</main>

<footer>
  <p>Intelligent AutoML Platform &nbsp;·&nbsp; Report generated {data.get('generated_at', '')} &nbsp;·&nbsp; Run ID: {data.get('run_id', '')}</p>
</footer>

</body>
</html>"""
        return html


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _make_json_safe(obj: Any) -> Any:
    """
    Recursively convert an object to a JSON-serialisable form.
    Drops non-serialisable values (replaces them with ``None``).
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items() if isinstance(k, str)}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    # Attempt str coercion for datetime etc.
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return None
