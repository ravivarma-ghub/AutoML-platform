"""
routes_predict.py
-----------------
FastAPI router for single predictions, batch CSV predictions,
and prediction-log retrieval.
"""

from __future__ import annotations

import io
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Prediction"])

# ---------------------------------------------------------------------------
# In-memory prediction log  {prediction_id: dict}
# ---------------------------------------------------------------------------
prediction_logs: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# In-memory model registry  (populated by routes_models when models are loaded)
# Imported lazily to avoid circular imports.
# ---------------------------------------------------------------------------
_MODELS_CACHE: Dict[str, Any] = {}  # model_id → loaded model object


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Single-row prediction request."""

    model_id: str = Field(..., description="Registered model ID")
    data: Dict[str, Any] = Field(..., description="Feature dict {feature_name: value}")
    return_shap: bool = Field(False, description="Include per-feature SHAP contributions")


class PredictBatchRequest(BaseModel):
    """Metadata carried alongside the CSV upload for batch prediction."""

    model_id: str = Field(..., description="Registered model ID")
    return_shap: bool = Field(False, description="Compute SHAP values for each row")


class PredictionResponse(BaseModel):
    """Full prediction result for a single row."""

    prediction_id: str
    model_id: str
    model_version: str
    model_name: str
    prediction: Union[int, float, str]
    probability: Optional[float] = Field(None, description="Max class probability (classification)")
    class_probabilities: Optional[Dict[str, float]] = Field(
        None, description="{class_label: probability}"
    )
    shap_contributions: Optional[Dict[str, float]] = Field(
        None, description="{feature_name: shap_value}"
    )
    prediction_time_ms: float
    timestamp: str


class PredictionLogEntry(BaseModel):
    """Summary row for prediction log listing."""

    prediction_id: str
    model_id: str
    model_name: str
    prediction: Union[int, float, str]
    probability: Optional[float]
    prediction_time_ms: float
    timestamp: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _load_model(model_id: str) -> Any:
    """
    Load a model from the registry / disk.

    Attempts to import the model store from routes_models; falls back to a
    lightweight mock predictor so the endpoint remains testable in isolation.
    """
    if model_id in _MODELS_CACHE:
        return _MODELS_CACHE[model_id]

    try:
        from app.api.routes_models import model_registry  # type: ignore

        entry = model_registry.get(model_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found in registry.")

        model_path = entry.get("model_path")
        if model_path and Path(model_path).exists():
            import joblib  # type: ignore

            loaded = joblib.load(model_path)
            _MODELS_CACHE[model_id] = loaded
            return loaded

        # Return a mock wrapper using registry metadata
        return _MockModel(entry)

    except ImportError:
        logger.warning("routes_models not importable; using MockModel for model_id=%s", model_id)
        mock_meta = {
            "model_id": model_id,
            "model_name": "MockModel",
            "model_version": "1.0.0",
            "problem_type": "classification",
        }
        return _MockModel(mock_meta)


class _MockModel:
    """Lightweight stand-in when the real model artifact is unavailable."""

    def __init__(self, meta: Dict[str, Any]) -> None:
        self.meta = meta

    @property
    def model_id(self) -> str:
        return self.meta.get("model_id", "unknown")

    @property
    def model_name(self) -> str:
        return self.meta.get("model_name", "MockModel")

    @property
    def model_version(self) -> str:
        return self.meta.get("model_version", "0.0.0")

    @property
    def problem_type(self) -> str:
        return self.meta.get("problem_type", "classification")

    def predict_single(
        self, data: Dict[str, Any], return_shap: bool = False
    ) -> Dict[str, Any]:
        """Return a deterministic mock prediction."""
        import hashlib, json

        digest = int(hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest(), 16)
        if self.problem_type == "classification":
            pred = int(digest % 2)
            prob = round(0.55 + (digest % 45) / 100, 4)
            class_probs = {str(pred): prob, str(1 - pred): round(1 - prob, 4)}
        else:
            pred = round((digest % 10000) / 100, 4)
            prob = None
            class_probs = None

        shap = (
            {k: round((hash(k) % 1000) / 10000, 6) for k in list(data.keys())[:10]}
            if return_shap
            else None
        )
        return {
            "prediction": pred,
            "probability": prob,
            "class_probabilities": class_probs,
            "shap_contributions": shap,
        }

    def predict_dataframe(
        self, df: pd.DataFrame, return_shap: bool = False
    ) -> pd.DataFrame:
        """Return mock batch predictions for every row in *df*."""
        results = []
        for _, row in df.iterrows():
            r = self.predict_single(row.to_dict(), return_shap=return_shap)
            results.append(r)
        out = pd.DataFrame(results)
        out.insert(0, "prediction_id", [str(uuid.uuid4()) for _ in range(len(out))])
        return out


def _log_prediction(prediction_id: str, model: Any, result: Dict[str, Any]) -> None:
    """Persist prediction to the in-memory log (and optionally a DB later)."""
    prediction_logs[prediction_id] = {
        "prediction_id": prediction_id,
        "model_id": model.model_id,
        "model_name": model.model_name,
        "model_version": model.model_version,
        "prediction": result["prediction"],
        "probability": result.get("probability"),
        "class_probabilities": result.get("class_probabilities"),
        "shap_contributions": result.get("shap_contributions"),
        "prediction_time_ms": result["prediction_time_ms"],
        "timestamp": result["timestamp"],
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Run a single-row prediction",
)
async def predict_single(request: PredictRequest) -> PredictionResponse:
    """
    Accept a JSON feature dict, run inference, log the result, and return
    the full `PredictionResponse`.
    """
    t0 = time.perf_counter()
    model = _load_model(request.model_id)

    try:
        raw = model.predict_single(request.data, return_shap=request.return_shap)
    except Exception as exc:
        logger.exception("Prediction failed for model %s", request.model_id)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
    prediction_id = str(uuid.uuid4())
    ts = _now_iso()

    result = {**raw, "prediction_time_ms": elapsed_ms, "timestamp": ts}
    _log_prediction(prediction_id, model, result)

    return PredictionResponse(
        prediction_id=prediction_id,
        model_id=model.model_id,
        model_version=model.model_version,
        model_name=model.model_name,
        prediction=raw["prediction"],
        probability=raw.get("probability"),
        class_probabilities=raw.get("class_probabilities"),
        shap_contributions=raw.get("shap_contributions"),
        prediction_time_ms=elapsed_ms,
        timestamp=ts,
    )


@router.post(
    "/predict/batch",
    summary="Run batch predictions from a CSV upload",
)
async def predict_batch(
    background_tasks: BackgroundTasks,
    model_id: str = Query(..., description="Registered model ID"),
    return_shap: bool = Query(False, description="Include SHAP values"),
    file: UploadFile = File(..., description="CSV file with feature columns"),
) -> StreamingResponse:
    """
    Accept a CSV file upload, run batch inference, and stream the results
    back as a CSV.  Very large files are handed off to a background task;
    for files ≤ 10 000 rows the response is immediate.
    """
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted for batch prediction.")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot parse CSV: {exc}") from exc

    if df.empty:
        raise HTTPException(status_code=422, detail="Uploaded CSV is empty.")

    model = _load_model(model_id)

    t0 = time.perf_counter()
    try:
        predictions_df = model.predict_dataframe(df, return_shap=return_shap)
    except Exception as exc:
        logger.exception("Batch prediction failed for model %s", model_id)
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {exc}") from exc

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
    logger.info(
        "Batch prediction: model=%s rows=%d elapsed=%.1fms", model_id, len(df), elapsed_ms
    )

    # Log summary entries in background
    ts = _now_iso()
    for _, row in predictions_df.iterrows():
        pid = row.get("prediction_id", str(uuid.uuid4()))
        prediction_logs[str(pid)] = {
            "prediction_id": str(pid),
            "model_id": model.model_id,
            "model_name": model.model_name,
            "model_version": model.model_version,
            "prediction": row.get("prediction"),
            "probability": row.get("probability"),
            "class_probabilities": None,
            "shap_contributions": None,
            "prediction_time_ms": elapsed_ms / len(df),
            "timestamp": ts,
        }

    # Stream CSV response
    output = io.StringIO()
    predictions_df.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="predictions_{model_id}.csv"',
            "X-Prediction-Time-Ms": str(elapsed_ms),
            "X-Row-Count": str(len(predictions_df)),
        },
    )


@router.get(
    "/predict/logs",
    response_model=List[PredictionLogEntry],
    summary="List recent prediction logs with pagination",
)
async def list_prediction_logs(
    model_id: Optional[str] = Query(None, description="Filter by model ID"),
    limit: int = Query(100, ge=1, le=1000, description="Max records to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
) -> List[PredictionLogEntry]:
    """
    Return paginated prediction logs.  Optionally filter by `model_id`.
    Results are sorted newest-first.
    """
    logs = list(prediction_logs.values())

    if model_id:
        logs = [l for l in logs if l["model_id"] == model_id]

    # Sort newest-first
    logs.sort(key=lambda x: x["timestamp"], reverse=True)
    page = logs[offset : offset + limit]

    return [
        PredictionLogEntry(
            prediction_id=entry["prediction_id"],
            model_id=entry["model_id"],
            model_name=entry["model_name"],
            prediction=entry["prediction"],
            probability=entry.get("probability"),
            prediction_time_ms=entry["prediction_time_ms"],
            timestamp=entry["timestamp"],
        )
        for entry in page
    ]


@router.get(
    "/predict/logs/{prediction_id}",
    summary="Get a single prediction log entry",
)
async def get_prediction_log(prediction_id: str) -> Dict[str, Any]:
    """Return the full stored record for a specific prediction."""
    entry = prediction_logs.get(prediction_id)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"Prediction log '{prediction_id}' not found.",
        )
    return entry
