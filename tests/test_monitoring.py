"""
Unit and integration tests for ModelMonitor (data drift and prediction drift detection).
"""
import pytest
import numpy as np
import pandas as pd
from app.pipeline.monitoring import ModelMonitor, DriftReport
from app.models.registry import ModelRegistry


@pytest.fixture
def mock_registry(tmp_path):
    return ModelRegistry(registry_path=str(tmp_path / "models"))


@pytest.fixture
def monitor(mock_registry):
    return ModelMonitor(registry=mock_registry)


class TestModelMonitor:
    def test_psi_identical_distributions(self, monitor):
        """PSI of identical distributions should be virtually 0 (< 0.05)."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        psi = monitor.compute_psi(data, data)
        assert psi < 0.05

    def test_psi_significant_drift(self, monitor):
        """Shifted distributions should trigger high PSI (> 0.2)."""
        np.random.seed(42)
        ref = np.random.normal(0, 1, 1000)
        curr = np.random.normal(3, 1, 1000)  # Significant shift
        psi = monitor.compute_psi(ref, curr)
        assert psi > 0.2

    def test_ks_test_no_drift(self, monitor):
        """KS test on same distribution should yield high p-value."""
        np.random.seed(42)
        ref = np.random.normal(50, 10, 500)
        curr = np.random.normal(50, 10, 500)
        stat, p_val = monitor.compute_ks_test(ref, curr)
        assert p_val > 0.01

    def test_ks_test_with_drift(self, monitor):
        """KS test on different distributions should yield low p-value (< 0.05)."""
        np.random.seed(42)
        ref = np.random.normal(50, 10, 500)
        curr = np.random.normal(65, 10, 500)
        stat, p_val = monitor.compute_ks_test(ref, curr)
        assert p_val < 0.05

    def test_check_data_drift_report(self, monitor):
        """Test generating a full DriftReport from reference and current DataFrames."""
        np.random.seed(42)
        ref_df = pd.DataFrame({
            "age": np.random.normal(35, 10, 500),
            "income": np.random.normal(60000, 15000, 500),
        })
        curr_df = pd.DataFrame({
            "age": np.random.normal(36, 10, 500),         # Minimal drift
            "income": np.random.normal(90000, 15000, 500), # Heavy drift
        })
        
        report = monitor.check_data_drift(
            model_id="test_model_v1",
            reference_df=ref_df,
            current_df=curr_df,
            feature_names=["age", "income"]
        )
        
        assert isinstance(report, DriftReport)
        assert "income" in report.feature_drift
        assert "age" in report.feature_drift
        assert report.overall_status in ["OK", "WARNING", "CRITICAL"]

    def test_prediction_drift_check(self, monitor):
        """Test checking distribution changes between reference and current predictions."""
        ref_preds = np.array([0] * 80 + [1] * 20)  # 20% positive
        curr_preds = np.array([0] * 50 + [1] * 50) # 50% positive
        
        pred_drift = monitor.check_prediction_drift(
            model_id="test_model_v1",
            reference_predictions=ref_preds,
            current_predictions=curr_preds
        )
        
        assert "psi" in pred_drift
        assert pred_drift["before_positive_rate"] == pytest.approx(0.2, abs=0.01)
        assert pred_drift["current_positive_rate"] == pytest.approx(0.5, abs=0.01)

    def test_format_drift_report(self, monitor):
        """Ensure formatting produces a readable ASCII summary string."""
        np.random.seed(42)
        ref_df = pd.DataFrame({"num": np.random.normal(0, 1, 100)})
        curr_df = pd.DataFrame({"num": np.random.normal(2, 1, 100)})
        report = monitor.check_data_drift("m1", ref_df, curr_df, ["num"])
        formatted = monitor.format_drift_report(report)
        assert isinstance(formatted, str)
        assert "DRIFT" in formatted.upper()
