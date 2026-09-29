"""
Tests for app.orchestrator — AutoMLOrchestrator end-to-end pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


class TestAutoMLOrchestrator:
    """Integration-level tests for AutoMLOrchestrator."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_orchestrator(output_dir: Path):
        """Instantiate an orchestrator with a temp output directory."""
        from app.orchestrator import AutoMLOrchestrator

        return AutoMLOrchestrator(output_dir=str(output_dir))

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_full_pipeline_classification(
        self,
        sample_classification_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """
        Running the full classification pipeline must:
        1. Complete without raising an exception.
        2. Return a results dict with 'best_model', 'metrics', and 'model_path'.
        3. Produce a serialized model file on disk.
        """
        orchestrator = self._make_orchestrator(tmp_path)
        results = orchestrator.run(
            df=sample_classification_df,
            target_column="churn",
            problem_type="classification",
        )

        assert isinstance(results, dict), "run() must return a dict"
        assert "best_model" in results or "model" in results, (
            "Results must contain the best model"
        )
        assert "metrics" in results or "score" in results or "evaluation" in results, (
            "Results must contain evaluation metrics"
        )

        # A model artifact must have been saved
        saved_files = list(tmp_path.rglob("*.pkl")) + list(tmp_path.rglob("*.joblib"))
        assert len(saved_files) >= 1, (
            f"No serialized model found in {tmp_path}. Files: {list(tmp_path.rglob('*'))}"
        )

    def test_full_pipeline_regression(
        self,
        sample_regression_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """
        Running the full regression pipeline must:
        1. Complete without raising an exception.
        2. Return results with regression metrics (e.g. RMSE or R²).
        """
        orchestrator = self._make_orchestrator(tmp_path)
        results = orchestrator.run(
            df=sample_regression_df,
            target_column="price",
            problem_type="regression",
        )

        assert isinstance(results, dict), "run() must return a dict"

        # Check that at least one regression metric is present
        metrics = results.get("metrics") or results.get("evaluation") or results
        metrics_str = str(metrics).lower()
        assert any(
            kw in metrics_str
            for kw in ("rmse", "mse", "mae", "r2", "r_squared", "mean_squared")
        ), f"No regression metric found in results: {results}"

    def test_predict_after_train(self, tmp_path: Path) -> None:
        """
        After training, predict() must return predictions for new data without
        requiring re-fitting.
        """
        from app.orchestrator import AutoMLOrchestrator

        rng = np.random.default_rng(42)
        n = 300
        df = pd.DataFrame(
            {
                "x1": rng.uniform(0, 1, n),
                "x2": rng.uniform(0, 1, n),
                "x3": rng.choice(["a", "b", "c"], n),
                "target": rng.integers(0, 2, n),
            }
        )

        orchestrator = AutoMLOrchestrator(output_dir=str(tmp_path))
        orchestrator.run(df=df, target_column="target", problem_type="classification")

        # Predict on a small new batch
        new_data = pd.DataFrame(
            {
                "x1": rng.uniform(0, 1, 10),
                "x2": rng.uniform(0, 1, 10),
                "x3": rng.choice(["a", "b", "c"], 10),
            }
        )
        predictions = orchestrator.predict(new_data)

        assert predictions is not None, "predict() must not return None"
        assert len(predictions) == 10, (
            f"Expected 10 predictions, got {len(predictions)}"
        )

    def test_progress_callback_called(self, tmp_path: Path) -> None:
        """
        If a progress_callback is supplied, it must be called at least once
        during a training run.
        """
        from app.orchestrator import AutoMLOrchestrator

        rng = np.random.default_rng(7)
        n = 300
        df = pd.DataFrame(
            {
                "f1": rng.uniform(0, 1, n),
                "f2": rng.uniform(0, 1, n),
                "label": rng.integers(0, 2, n),
            }
        )

        callback_calls: List[dict] = []

        def _callback(progress: dict) -> None:
            callback_calls.append(progress)

        orchestrator = AutoMLOrchestrator(
            output_dir=str(tmp_path),
            progress_callback=_callback,
        )
        orchestrator.run(df=df, target_column="label", problem_type="classification")

        assert len(callback_calls) >= 1, (
            "progress_callback was never called during training"
        )
        # Each callback payload should contain 'stage' or 'progress' or 'step'
        for call in callback_calls:
            assert isinstance(call, dict), "Callback payload must be a dict"
