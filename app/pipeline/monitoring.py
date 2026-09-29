"""
app/pipeline/monitoring.py
--------------------------
Production-grade model monitoring for the Intelligent AutoML Platform.

The :class:`ModelMonitor` detects feature and prediction drift using two
complementary statistics:

* **Population Stability Index (PSI)** — measures distributional shift across
  buckets; interpretable thresholds: <0.1 stable, 0.1–0.2 slight drift,
  >0.2 significant drift.
* **Kolmogorov–Smirnov test** — non-parametric two-sample test; the p-value
  indicates the probability of observing the data if the distributions were
  identical.

When drift exceeds configurable thresholds the monitor can automatically
trigger a re-training run via :class:`~app.pipeline.orchestrator.AutoMLOrchestrator`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from app.pipeline.orchestrator import AutoMLOrchestrator, TrainingResult, TrainingConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PSI thresholds
# ---------------------------------------------------------------------------
_PSI_STABLE = 0.10
_PSI_SLIGHT = 0.20  # > this → significant

# KS test p-value threshold (significance level)
_KS_ALPHA = 0.05


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class FeatureDriftDetail:
    """Drift statistics for a single feature."""

    psi: float = 0.0
    ks_stat: float = 0.0
    ks_pvalue: float = 1.0
    status: str = "OK"  # OK | WARNING | CRITICAL


@dataclass
class DriftReport:
    """
    Aggregated drift analysis for a registered model.

    Attributes
    ----------
    model_id :
        Identifier of the monitored model.
    reference_date :
        ISO-8601 timestamp of the reference dataset.
    current_date :
        ISO-8601 timestamp of the current dataset.
    feature_drift :
        Per-feature drift details keyed by feature name.
    prediction_drift :
        Summary statistics for prediction-level drift.
    overall_status :
        ``OK`` | ``WARNING`` | ``CRITICAL``
    recommendation :
        Human-readable action: ``MONITOR`` | ``RETRAIN`` | ``URGENT_RETRAIN``
    """

    model_id: str = ""
    reference_date: str = ""
    current_date: str = ""
    feature_drift: Dict[str, FeatureDriftDetail] = field(default_factory=dict)
    prediction_drift: Dict[str, Any] = field(default_factory=dict)
    overall_status: str = "OK"
    recommendation: str = "MONITOR"


# ---------------------------------------------------------------------------
# ModelMonitor
# ---------------------------------------------------------------------------


class ModelMonitor:
    """
    Detects data drift and prediction drift for production ML models.

    Parameters
    ----------
    registry_path : str
        Path to the model registry directory (``Settings.MODEL_REGISTRY_PATH``).
    db_session : optional
        SQLAlchemy session for persisting drift reports.  When ``None`` all
        results stay in memory.
    """

    def __init__(
        self,
        registry_path: str = "./models",
        db_session: Any = None,
        registry: Any = None,
    ) -> None:
        if registry is not None:
            self.registry = registry
            self.registry_path = getattr(registry, "registry_path", str(registry_path))
        else:
            self.registry_path = str(registry_path)
            self.registry = None
        self.db_session = db_session
        self._drift_history: List[DriftReport] = []
        logger.info("ModelMonitor initialised — registry: %s", self.registry_path)

    # ------------------------------------------------------------------
    # Core statistical primitives
    # ------------------------------------------------------------------

    def compute_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        buckets: int = 10,
    ) -> float:
        """
        Compute the **Population Stability Index** between two distributions.

        PSI = Σ (actual_pct − expected_pct) × ln(actual_pct / expected_pct)

        Interpretation
        --------------
        * PSI < 0.10  → no significant drift
        * PSI 0.10–0.20 → slight drift (monitor)
        * PSI > 0.20  → significant drift (consider retraining)

        Parameters
        ----------
        expected : np.ndarray
            Reference / training distribution values.
        actual : np.ndarray
            Current / production distribution values.
        buckets : int
            Number of equal-width histogram bins.

        Returns
        -------
        float
            PSI value (≥ 0).
        """
        expected = np.array(expected, dtype=float)
        actual = np.array(actual, dtype=float)

        # Remove NaNs
        expected = expected[~np.isnan(expected)]
        actual = actual[~np.isnan(actual)]

        if len(expected) == 0 or len(actual) == 0:
            logger.warning("PSI computation skipped — empty array(s).")
            return 0.0

        # Build bin edges on the combined range
        global_min = min(expected.min(), actual.min())
        global_max = max(expected.max(), actual.max())

        if global_min == global_max:
            # Constant feature — no drift possible
            return 0.0

        bin_edges = np.linspace(global_min, global_max, buckets + 1)
        # Ensure boundary points are included
        bin_edges[0] -= 1e-9
        bin_edges[-1] += 1e-9

        expected_counts, _ = np.histogram(expected, bins=bin_edges)
        actual_counts, _ = np.histogram(actual, bins=bin_edges)

        # Convert to percentages; add tiny epsilon to avoid log(0)
        eps = 1e-9
        expected_pct = expected_counts / (len(expected) + eps)
        actual_pct = actual_counts / (len(actual) + eps)

        # Replace zeros
        expected_pct = np.where(expected_pct == 0, eps, expected_pct)
        actual_pct = np.where(actual_pct == 0, eps, actual_pct)

        psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
        return round(psi, 6)

    def compute_ks_test(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Perform a two-sample **Kolmogorov–Smirnov test**.

        Parameters
        ----------
        reference : np.ndarray
            Reference distribution values.
        current : np.ndarray
            Current distribution values.

        Returns
        -------
        tuple[float, float]
            ``(statistic, p_value)``
        """
        ref = np.array(reference, dtype=float)
        cur = np.array(current, dtype=float)

        ref = ref[~np.isnan(ref)]
        cur = cur[~np.isnan(cur)]

        if len(ref) < 2 or len(cur) < 2:
            logger.warning("KS test skipped — insufficient non-null samples.")
            return 0.0, 1.0

        ks_stat, p_value = stats.ks_2samp(ref, cur)
        return float(ks_stat), float(p_value)

    # ------------------------------------------------------------------
    # High-level drift checks
    # ------------------------------------------------------------------

    def check_data_drift(
        self,
        model_id: str,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_names: List[str],
    ) -> DriftReport:
        """
        Compute feature-level drift between a reference and current dataset.

        For each numeric feature both PSI and the KS-test are computed.
        For categorical features only PSI is computed (via frequency encoding).

        Parameters
        ----------
        model_id : str
            Identifier of the model being monitored.
        reference_df : pd.DataFrame
            Historical / training data used as the reference distribution.
        current_df : pd.DataFrame
            Recent production data to compare against.
        feature_names : list[str]
            Features to analyse (must be present in both DataFrames).

        Returns
        -------
        DriftReport
            Populated drift report with per-feature statistics and an overall
            status.
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        report = DriftReport(
            model_id=model_id,
            reference_date=now,
            current_date=now,
        )

        n_critical = 0
        n_warning = 0

        for feat in feature_names:
            if feat not in reference_df.columns or feat not in current_df.columns:
                logger.debug("Feature '%s' missing from one of the datasets; skipping.", feat)
                continue

            ref_vals = reference_df[feat]
            cur_vals = current_df[feat]

            # For categorical: convert to numeric frequency rank
            if not pd.api.types.is_numeric_dtype(ref_vals):
                ref_vals = self._encode_categorical_for_drift(ref_vals)
                cur_vals = self._encode_categorical_for_drift(cur_vals)

            ref_arr = ref_vals.dropna().values.astype(float)
            cur_arr = cur_vals.dropna().values.astype(float)

            psi = self.compute_psi(ref_arr, cur_arr)
            ks_stat, ks_pvalue = self.compute_ks_test(ref_arr, cur_arr)

            # Determine per-feature status
            if psi > _PSI_SLIGHT or (ks_pvalue < _KS_ALPHA and ks_stat > 0.3):
                status = "CRITICAL"
                n_critical += 1
            elif psi > _PSI_STABLE or ks_pvalue < _KS_ALPHA:
                status = "WARNING"
                n_warning += 1
            else:
                status = "OK"

            report.feature_drift[feat] = FeatureDriftDetail(
                psi=psi,
                ks_stat=ks_stat,
                ks_pvalue=ks_pvalue,
                status=status,
            )

        # ── Overall status ──────────────────────────────────────────────
        total_features = len(report.feature_drift)
        critical_frac = n_critical / max(total_features, 1)
        warning_frac = n_warning / max(total_features, 1)

        if n_critical > 0 and critical_frac >= 0.3:
            report.overall_status = "CRITICAL"
            report.recommendation = "URGENT_RETRAIN"
        elif n_critical > 0 or warning_frac >= 0.3:
            report.overall_status = "WARNING"
            report.recommendation = "RETRAIN"
        else:
            report.overall_status = "OK"
            report.recommendation = "MONITOR"

        self._drift_history.append(report)
        logger.info(
            "Data drift check — model=%s, features=%d, critical=%d, warning=%d, status=%s",
            model_id, total_features, n_critical, n_warning, report.overall_status,
        )
        return report

    def check_prediction_drift(
        self,
        model_id: str,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Detect distributional shift in model predictions.

        Computes PSI on the prediction scores/labels and, for binary
        classifiers, tracks positive-class rate change.

        Parameters
        ----------
        model_id : str
            Model identifier (for logging).
        reference_predictions : np.ndarray
            Predictions on the reference / historical split.
        current_predictions : np.ndarray
            Predictions from the recent production window.

        Returns
        -------
        dict with keys:
            ``psi``, ``ks_stat``, ``ks_pvalue``,
            ``before_positive_rate``, ``current_positive_rate``, ``status``
        """
        ref = np.array(reference_predictions, dtype=float)
        cur = np.array(current_predictions, dtype=float)

        psi = self.compute_psi(ref, cur)
        ks_stat, ks_pvalue = self.compute_ks_test(ref, cur)

        # Binary classification positive rate
        ref_unique = np.unique(ref)
        is_binary = len(ref_unique) <= 2 and set(ref_unique).issubset({0, 1, 0.0, 1.0})
        before_positive_rate = float(np.mean(ref)) if is_binary else float(np.nan)
        current_positive_rate = float(np.mean(cur)) if is_binary else float(np.nan)

        if psi > _PSI_SLIGHT or (ks_pvalue < _KS_ALPHA and ks_stat > 0.3):
            status = "CRITICAL"
        elif psi > _PSI_STABLE or ks_pvalue < _KS_ALPHA:
            status = "WARNING"
        else:
            status = "OK"

        result = {
            "psi": psi,
            "ks_stat": ks_stat,
            "ks_pvalue": ks_pvalue,
            "before_positive_rate": before_positive_rate,
            "current_positive_rate": current_positive_rate,
            "status": status,
        }
        logger.info(
            "Prediction drift check — model=%s, PSI=%.4f, KS=%.4f (p=%.4f), status=%s",
            model_id, psi, ks_stat, ks_pvalue, status,
        )
        return result

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def format_drift_report(self, report: DriftReport) -> str:
        """
        Render *report* as a human-readable ASCII table.

        Returns
        -------
        str
            Multi-line ASCII report suitable for logging or CLI display.
        """
        lines: List[str] = []
        sep = "=" * 70

        lines.append(sep)
        lines.append("  MODEL DRIFT REPORT")
        lines.append(sep)
        lines.append(f"  Model ID   : {report.model_id}")
        lines.append(f"  Checked at : {report.current_date}")
        lines.append(f"  Status     : {report.overall_status}")
        lines.append(f"  Action     : {report.recommendation}")
        lines.append(sep)

        if report.feature_drift:
            lines.append(
                f"  {'FEATURE':<30} {'PSI':>8}  {'KS-STAT':>8}  {'KS-PVAL':>8}  STATUS"
            )
            lines.append("-" * 70)
            for feat, detail in sorted(
                report.feature_drift.items(),
                key=lambda x: x[1].psi,
                reverse=True,
            ):
                icon = "🔴" if detail.status == "CRITICAL" else ("🟡" if detail.status == "WARNING" else "🟢")
                lines.append(
                    f"  {feat:<30} {detail.psi:>8.4f}  {detail.ks_stat:>8.4f}  "
                    f"{detail.ks_pvalue:>8.4f}  {icon} {detail.status}"
                )

        if report.prediction_drift:
            lines.append(sep)
            lines.append("  PREDICTION DRIFT")
            lines.append("-" * 70)
            pd_info = report.prediction_drift
            lines.append(f"  PSI              : {pd_info.get('psi', 'N/A')}")
            lines.append(f"  KS stat          : {pd_info.get('ks_stat', 'N/A')}")
            lines.append(f"  KS p-value       : {pd_info.get('ks_pvalue', 'N/A')}")
            lines.append(f"  Status           : {pd_info.get('status', 'N/A')}")
            if not np.isnan(pd_info.get("before_positive_rate", float("nan"))):
                lines.append(f"  Positive rate Δ  : "
                             f"{pd_info['before_positive_rate']:.3f} → {pd_info['current_positive_rate']:.3f}")

        lines.append(sep)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Retraining logic
    # ------------------------------------------------------------------

    def should_retrain(self, report: DriftReport) -> bool:
        """
        Return ``True`` when the drift report recommends immediate retraining.

        Triggers retraining when:
        * Overall status is ``CRITICAL``, or
        * Recommendation is ``RETRAIN`` or ``URGENT_RETRAIN``.

        Parameters
        ----------
        report : DriftReport

        Returns
        -------
        bool
        """
        return report.recommendation in ("RETRAIN", "URGENT_RETRAIN")

    def trigger_retraining(
        self,
        model_id: str,
        new_data_path: str,
        orchestrator: "AutoMLOrchestrator",
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> "TrainingResult":
        """
        Launch a new AutoML training run using the drifted model's metadata as
        a starting point.

        Steps
        -----
        1. Load the metadata of the drifted model from the registry.
        2. Build a :class:`~app.pipeline.orchestrator.TrainingConfig` that
           reuses the original problem type and target column.
        3. Delegate to :meth:`~app.pipeline.orchestrator.AutoMLOrchestrator.train`.

        Parameters
        ----------
        model_id : str
            Registry identifier of the model that triggered retraining.
        new_data_path : str
            Path to the fresh dataset to retrain on.
        orchestrator : AutoMLOrchestrator
            The orchestrator instance to use for training.
        extra_config : dict, optional
            Any additional :class:`~app.pipeline.orchestrator.TrainingConfig`
            fields to override.

        Returns
        -------
        TrainingResult
            Result of the new training run.
        """
        from app.pipeline.orchestrator import TrainingConfig  # local import avoids circular

        logger.info(
            "Triggering retraining for model_id=%s with data=%s", model_id, new_data_path
        )

        # Load original metadata to inherit problem_type, target, etc.
        import json
        from pathlib import Path

        model_dir = Path(self.registry_path) / model_id
        metadata: Dict[str, Any] = {}
        meta_path = model_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as fh:
                metadata = json.load(fh)

        problem_type: Optional[str] = metadata.get("problem_type")
        target_column: str = metadata.get("target_column", "target")
        experiment_name: str = f"retrain_{model_id[:8]}"

        config_kwargs: Dict[str, Any] = {
            "dataset_path": new_data_path,
            "target_column": target_column,
            "problem_type": problem_type,
            "experiment_name": experiment_name,
        }
        if extra_config:
            config_kwargs.update(extra_config)

        training_config = TrainingConfig(**config_kwargs)

        def _progress(step: str, pct: int, msg: str) -> None:
            logger.info("[retrain][%d%%] %s — %s", pct, step, msg)

        result = orchestrator.train(training_config, progress_callback=_progress)
        logger.info(
            "Retraining complete — new model_id=%s, status=%s",
            result.champion_model_id, result.status,
        )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode_categorical_for_drift(self, series: pd.Series) -> pd.Series:
        """
        Convert a categorical series to numeric frequency ranks for PSI/KS.

        Each unique value is replaced by its relative frequency (proportion)
        in the series, giving a [0, 1]-bounded numeric encoding that reflects
        class imbalance.
        """
        freq = series.value_counts(normalize=True)
        return series.map(freq).fillna(0.0)

    def get_drift_history(self, model_id: Optional[str] = None) -> List[DriftReport]:
        """
        Return historical drift reports.

        Parameters
        ----------
        model_id : str, optional
            Filter to a specific model.  If ``None`` all reports are returned.

        Returns
        -------
        list[DriftReport]
        """
        if model_id is None:
            return list(self._drift_history)
        return [r for r in self._drift_history if r.model_id == model_id]

    def summarise_drift_trend(self, model_id: str) -> Dict[str, Any]:
        """
        Compute a trend summary across all historical reports for *model_id*.

        Returns a dict with:
        * ``n_checks`` — total number of drift checks performed.
        * ``n_critical`` — number of CRITICAL incidents.
        * ``n_warning`` — number of WARNING incidents.
        * ``mean_psi_per_feature`` — averaged PSI per feature across checks.
        * ``latest_status`` — most recent overall status.
        * ``latest_recommendation`` — most recent recommendation.
        """
        history = self.get_drift_history(model_id)
        if not history:
            return {"n_checks": 0, "model_id": model_id}

        n_critical = sum(1 for r in history if r.overall_status == "CRITICAL")
        n_warning = sum(1 for r in history if r.overall_status == "WARNING")

        # Accumulate per-feature PSI across reports
        psi_accumulator: Dict[str, List[float]] = {}
        for report in history:
            for feat, detail in report.feature_drift.items():
                psi_accumulator.setdefault(feat, []).append(detail.psi)

        mean_psi = {f: float(np.mean(vals)) for f, vals in psi_accumulator.items()}

        latest = history[-1]
        return {
            "model_id": model_id,
            "n_checks": len(history),
            "n_critical": n_critical,
            "n_warning": n_warning,
            "mean_psi_per_feature": mean_psi,
            "latest_status": latest.overall_status,
            "latest_recommendation": latest.recommendation,
        }
