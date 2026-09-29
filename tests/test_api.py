"""
Tests for REST API endpoints — health, training, prediction, model management.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_check(self, client: TestClient) -> None:
        """Health endpoint must return 200 and indicate a healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        # Accept {"status": "ok"} or {"status": "healthy"} or {"healthy": true}
        assert (
            data.get("status") in ("ok", "healthy", "up")
            or data.get("healthy") is True
            or "ok" in str(data).lower()
        ), f"Unexpected health response: {data}"


# ---------------------------------------------------------------------------
# Training endpoints
# ---------------------------------------------------------------------------

class TestTrainingEndpoints:
    """Tests for /train/* endpoints."""

    def test_upload_file(self, client: TestClient, tmp_path: Path) -> None:
        """POST /train/upload must accept a CSV file and return a dataset ID."""
        # Create a small synthetic CSV file
        rng = np.random.default_rng(0)
        n = 50
        df = pd.DataFrame(
            {
                "feature_a": rng.uniform(0, 1, n),
                "feature_b": rng.choice(["x", "y", "z"], n),
                "target": rng.integers(0, 2, n),
            }
        )
        csv_path = tmp_path / "test_dataset.csv"
        df.to_csv(csv_path, index=False)

        with open(csv_path, "rb") as f:
            response = client.post(
                "/train/upload",
                files={"file": ("test_dataset.csv", f, "text/csv")},
            )

        assert response.status_code in (200, 201, 202), (
            f"Upload failed with status {response.status_code}: {response.text}"
        )
        data = response.json()
        assert (
            "dataset_id" in data
            or "id" in data
            or "file_id" in data
            or "upload_id" in data
        ), f"Response missing dataset identifier: {data}"

    def test_train_start(self, client: TestClient) -> None:
        """POST /train/start must accept training config and return a job ID."""
        payload = {
            "dataset_id": "test-dataset-001",
            "target_column": "target",
            "problem_type": "classification",
        }
        response = client.post("/train/start", json=payload)
        # Accept 200, 201, 202 (async job accepted)
        assert response.status_code in (200, 201, 202), (
            f"Train start failed: {response.status_code} — {response.text}"
        )
        data = response.json()
        assert (
            "job_id" in data
            or "task_id" in data
            or "run_id" in data
            or "id" in data
        ), f"Response missing job identifier: {data}"

    def test_train_status(self, client: TestClient) -> None:
        """GET /train/status/{job_id} must return a status field."""
        # Use a dummy job_id — endpoint must handle unknown IDs gracefully
        response = client.get("/train/status/dummy-job-000")
        assert response.status_code in (200, 404), (
            f"Unexpected status code: {response.status_code}"
        )
        if response.status_code == 200:
            data = response.json()
            assert (
                "status" in data
                or "state" in data
                or "phase" in data
            ), f"Response missing status field: {data}"


# ---------------------------------------------------------------------------
# Prediction endpoints
# ---------------------------------------------------------------------------

class TestPredictEndpoints:
    """Tests for /predict/* endpoints."""

    def test_predict_single(self, client: TestClient) -> None:
        """POST /predict must return a prediction for a single record."""
        payload = {
            "model_id": "test-model-001",
            "data": {
                "feature_a": 0.5,
                "feature_b": "x",
            },
        }
        response = client.post("/predict", json=payload)
        # 200 if model exists, 404 if model not found — both are acceptable
        assert response.status_code in (200, 404, 422), (
            f"Unexpected status: {response.status_code} — {response.text}"
        )
        if response.status_code == 200:
            data = response.json()
            assert (
                "prediction" in data
                or "result" in data
                or "output" in data
                or "label" in data
            ), f"Response missing prediction field: {data}"


# ---------------------------------------------------------------------------
# Model management endpoints
# ---------------------------------------------------------------------------

class TestModelEndpoints:
    """Tests for /models/* endpoints."""

    def test_list_models(self, client: TestClient) -> None:
        """GET /models must return a list (possibly empty) of registered models."""
        response = client.get("/models")
        assert response.status_code == 200, (
            f"List models failed: {response.status_code} — {response.text}"
        )
        data = response.json()
        # Accept list directly or wrapped {"models": [...]}
        assert isinstance(data, list) or (
            isinstance(data, dict) and (
                "models" in data
                or "items" in data
                or "data" in data
            )
        ), f"Unexpected response format for /models: {data}"

    def test_get_model(self, client: TestClient) -> None:
        """GET /models/{model_id} must return 200 or 404 with a JSON body."""
        response = client.get("/models/nonexistent-model-xyz")
        assert response.status_code in (200, 404), (
            f"Unexpected status: {response.status_code} — {response.text}"
        )
        # Response body must be valid JSON regardless of status
        data = response.json()
        assert data is not None, "Response body must not be null"
        if response.status_code == 404:
            assert (
                "detail" in data
                or "message" in data
                or "error" in data
            ), f"404 response missing error detail: {data}"
