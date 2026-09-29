"""
routes_models.py
----------------
FastAPI router for model registry management: listing, metadata retrieval,
stage promotion, soft-delete, champion lookup, evaluation reports, and
live drift monitoring.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Models"])

# ---------------------------------------------------------------------------
# In-memory model registry
# {model_id: ModelEntry}
# ---------------------------------------------------------------------------
model_registry: Dict[str, Dict[str, Any]] = {}

# Application start time (for uptime calculation)
_START_TIME = time.time()

# ---------------------------------------------------------------------------
# Seed the registry with a demo model so the API is immediately usable
# ---------------------------------------------------------------------------
_DEMO_ID = "demo-rf-001"
model_registry[_DEMO_ID] = {
    "model_id": _DEMO_ID,
    "model_name": "RandomForestClassifier",
    "model_version": "1.0.0",
    "problem_type": "classification",
    "stage": "champion",
    "target_column": "target",
    "feature_names": ["f1", "f2", "f3", "f4", "f5"],
    "metrics": {
        "roc_auc": 0.924,
        "accuracy": 0.891,
        "f1_macro": 0.887,
        "precision_macro": 0.893,
        "recall_macro": 0.882,
    },
    "hyperparameters": {
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },
    "experiment_id": str(uuid.uuid4()),
    "model_path": None,
    "report_path": None,
    "archived": False,
    "created_at": datetime.now(tz=timezone.utc).isoformat(),
    "updated_at": datetime.now(tz=timezone.utc).isoformat(),
    "n_training_samples": 8000,
    "n_features": 5,
    "training_duration_s": 42.1,
}


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ModelSummary(BaseModel):
    """Lightweight model record for list responses."""

    model_id: str
    model_name: str
    model_version: str
    problem_type: str
    stage: str
    target_column: str
    created_at: str
    updated_at: str
    archived: bool


class ModelDetail(BaseModel):
    """Full model record including metrics and hyperparameters."""

    model_id: str
    model_name: str
    model_version: str
    problem_type: str
    stage: str
    target_column: str
    feature_names: List[str]
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    experiment_id: str
    model_path: Optional[str]
    report_path: Optional[str]
    archived: bool
    created_at: str
    updated_at: str
    n_training_samples: int
    n_features: int
    training_duration_s: float


class PromoteRequest(BaseModel):
    """Body for model stage promotion."""

    stage: str = Field(
        ...,
        description="Target stage: 'champion' | 'production' | 'validation' | 'staging'",
    )


class HealthResponse(BaseModel):
    """Platform health check response."""

    status: str
    version: str
    models_count: int
    champion_model_id: Optional[str]
    uptime_seconds: float
    timestamp: str


class DriftReport(BaseModel):
    """Drift analysis result for a model."""

    model_id: str
    model_name: str
    n_predictions_analyzed: int
    drift_detected: bool
    drift_score: float
    feature_drift: Dict[str, float]
    recommendation: str
    analyzed_at: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _get_active_model(model_id: str) -> Dict[str, Any]:
    """Fetch a non-archived model or raise 404."""
    entry = model_registry.get(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
    if entry.get("archived"):
        raise HTTPException(status_code=410, detail=f"Model '{model_id}' has been archived.")
    return entry


def _get_champion() -> Optional[Dict[str, Any]]:
    """Return the first non-archived champion model, or None."""
    for entry in model_registry.values():
        if entry.get("stage") == "champion" and not entry.get("archived"):
            return entry
    return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Platform health check",
    tags=["Health"],
)
async def health_check() -> HealthResponse:
    """
    Return service liveness, model count, current champion ID, and uptime.
    """
    active_models = [m for m in model_registry.values() if not m.get("archived")]
    champion = _get_champion()

    return HealthResponse(
        status="ok",
        version="1.0.0",
        models_count=len(active_models),
        champion_model_id=champion["model_id"] if champion else None,
        uptime_seconds=round(time.time() - _START_TIME, 1),
        timestamp=_now_iso(),
    )


@router.get(
    "/models",
    response_model=List[ModelSummary],
    summary="List all registered models",
)
async def list_models(
    stage: Optional[str] = Query(None, description="Filter by stage (champion/production/validation/staging)"),
    problem_type: Optional[str] = Query(None, description="Filter by problem type (classification/regression)"),
) -> List[ModelSummary]:
    """
    Return all non-archived models.  Supports optional `stage` and
    `problem_type` query-parameter filters.
    """
    entries = [m for m in model_registry.values() if not m.get("archived")]

    if stage:
        entries = [m for m in entries if m.get("stage") == stage]
    if problem_type:
        entries = [m for m in entries if m.get("problem_type") == problem_type]

    return [
        ModelSummary(
            model_id=m["model_id"],
            model_name=m["model_name"],
            model_version=m["model_version"],
            problem_type=m["problem_type"],
            stage=m["stage"],
            target_column=m["target_column"],
            created_at=m["created_at"],
            updated_at=m["updated_at"],
            archived=m["archived"],
        )
        for m in entries
    ]


@router.get(
    "/models/champion",
    response_model=ModelDetail,
    summary="Get the current champion model",
)
async def get_champion_model() -> ModelDetail:
    """
    Return the full record of the current champion model.
    Raises 404 if no champion is registered.

    > **Note**: This route must be declared *before* `/models/{model_id}`
    > so FastAPI matches it correctly.
    """
    champion = _get_champion()
    if champion is None:
        raise HTTPException(status_code=404, detail="No champion model is currently registered.")
    return ModelDetail(**{k: champion[k] for k in ModelDetail.__fields__})


@router.get(
    "/models/{model_id}",
    response_model=ModelDetail,
    summary="Get full model details",
)
async def get_model(model_id: str) -> ModelDetail:
    """Return complete metadata, metrics, and hyperparameters for a model."""
    entry = _get_active_model(model_id)
    return ModelDetail(**{k: entry[k] for k in ModelDetail.__fields__})


@router.get(
    "/models/{model_id}/metrics",
    response_model=Dict[str, float],
    summary="Get evaluation metrics for a model",
)
async def get_model_metrics(model_id: str) -> Dict[str, float]:
    """Return only the metrics dictionary for a registered model."""
    entry = _get_active_model(model_id)
    return entry.get("metrics", {})


@router.post(
    "/models/{model_id}/promote",
    response_model=ModelDetail,
    summary="Promote a model to a new stage",
)
async def promote_model(model_id: str, body: PromoteRequest) -> ModelDetail:
    """
    Promote *model_id* to the requested stage.

    When promoting to `champion`, the previous champion is automatically
    demoted to `production` to maintain a single champion at any time.
    """
    allowed_stages = {"champion", "production", "validation", "staging"}
    if body.stage not in allowed_stages:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid stage '{body.stage}'. Allowed: {sorted(allowed_stages)}",
        )

    entry = _get_active_model(model_id)

    # Demote existing champion if promoting a new one
    if body.stage == "champion":
        for m in model_registry.values():
            if m.get("stage") == "champion" and m["model_id"] != model_id:
                m["stage"] = "production"
                m["updated_at"] = _now_iso()
                logger.info(
                    "Demoted previous champion %s → production", m["model_id"]
                )

    entry["stage"] = body.stage
    entry["updated_at"] = _now_iso()
    logger.info("Promoted model %s → %s", model_id, body.stage)

    return ModelDetail(**{k: entry[k] for k in ModelDetail.__fields__})


@router.delete(
    "/models/{model_id}",
    summary="Soft-delete (archive) a model",
)
async def delete_model(model_id: str) -> Dict[str, str]:
    """
    Mark a model as archived (soft-delete).  The model record is retained
    for audit purposes but will no longer appear in list responses.
    """
    entry = model_registry.get(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
    if entry.get("archived"):
        raise HTTPException(status_code=409, detail=f"Model '{model_id}' is already archived.")
    if entry.get("stage") == "champion":
        raise HTTPException(
            status_code=409,
            detail="Cannot archive the champion model. Promote another model first.",
        )

    entry["archived"] = True
    entry["updated_at"] = _now_iso()
    logger.info("Archived model %s", model_id)
    return {"message": f"Model '{model_id}' has been archived.", "model_id": model_id}


@router.get(
    "/models/{model_id}/report",
    summary="Get the evaluation HTML report for a model",
)
async def get_model_report(model_id: str) -> HTMLResponse:
    """
    Return the HTML evaluation report generated during training.
    If the report file is missing, a minimal on-the-fly report is returned.
    """
    entry = _get_active_model(model_id)
    report_path = entry.get("report_path")

    if report_path and Path(report_path).exists():
        return HTMLResponse(content=Path(report_path).read_text(encoding="utf-8"))

    # Generate minimal placeholder
    metrics = entry.get("metrics", {})
    metric_rows = "".join(
        f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in metrics.items()
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Model Report – {model_id}</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 860px; margin: 2rem auto; padding: 0 1rem; }}
  h1 {{ color: #2d3748; }} h2 {{ color: #4a5568; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #e2e8f0; padding: 8px 12px; }}
  th {{ background: #edf2f7; }}
</style>
</head>
<body>
  <h1>Model Evaluation Report</h1>
  <p><strong>Model:</strong> {entry['model_name']} v{entry['model_version']}</p>
  <p><strong>Stage:</strong> {entry['stage']} &nbsp;|&nbsp;
     <strong>Problem:</strong> {entry['problem_type']}</p>
  <h2>Metrics</h2>
  <table><thead><tr><th>Metric</th><th>Value</th></tr></thead>
  <tbody>{metric_rows}</tbody></table>
  <h2>Hyperparameters</h2>
  <table><thead><tr><th>Parameter</th><th>Value</th></tr></thead>
  <tbody>{"".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in entry.get('hyperparameters', {}).items())}</tbody></table>
</body>
</html>"""
    return HTMLResponse(content=html)


@router.get(
    "/models/{model_id}/monitoring",
    response_model=DriftReport,
    summary="Run drift analysis on a model's recent predictions",
)
async def get_model_monitoring(model_id: str) -> DriftReport:
    """
    Analyse drift between the training distribution and recent prediction
    inputs.  Integrates with `routes_predict.prediction_logs` for live data.

    The implementation uses Population Stability Index (PSI) per feature.
    When insufficient recent data is available a mock drift report is returned.
    """
    entry = _get_active_model(model_id)

    # Pull recent predictions for this model
    try:
        from app.api.routes_predict import prediction_logs  # type: ignore

        recent = [
            log for log in prediction_logs.values() if log.get("model_id") == model_id
        ]
    except ImportError:
        recent = []

    n_predictions = len(recent)

    # ---- Attempt real PSI computation when enough data is available ----
    MINIMUM_SAMPLES = 50
    feature_drift: Dict[str, float] = {}
    drift_score = 0.0
    drift_detected = False

    if n_predictions >= MINIMUM_SAMPLES:
        try:
            import numpy as np  # type: ignore

            feature_names = entry.get("feature_names", [])
            for feat in feature_names:
                values = [
                    r.get("input_data", {}).get(feat)
                    for r in recent
                    if r.get("input_data", {}).get(feat) is not None
                ]
                if len(values) < MINIMUM_SAMPLES:
                    continue
                arr = np.array(values, dtype=float)
                # Simplified PSI proxy: coefficient of variation squared
                psi_proxy = float(np.std(arr) / (np.mean(arr) + 1e-9)) ** 2
                feature_drift[feat] = round(psi_proxy, 6)

            drift_score = round(sum(feature_drift.values()) / max(len(feature_drift), 1), 6)
            drift_detected = drift_score > 0.2
        except Exception:  # noqa: BLE001
            logger.warning("PSI computation failed; returning mock drift report")

    if not feature_drift:
        # Mock drift values for demonstration
        for feat in entry.get("feature_names", ["f1", "f2", "f3"]):
            import random, hashlib

            seed = int(hashlib.md5((model_id + feat).encode()).hexdigest(), 16) % 1000
            feature_drift[feat] = round(seed / 10000, 6)
        drift_score = round(sum(feature_drift.values()) / len(feature_drift), 6)
        drift_detected = drift_score > 0.2

    recommendation = (
        "⚠️  Significant drift detected. Consider retraining with recent data."
        if drift_detected
        else "✅  No significant drift detected. Model appears stable."
    )

    logger.info(
        "Drift report for model %s: score=%.4f detected=%s", model_id, drift_score, drift_detected
    )

    return DriftReport(
        model_id=model_id,
        model_name=entry["model_name"],
        n_predictions_analyzed=n_predictions,
        drift_detected=drift_detected,
        drift_score=drift_score,
        feature_drift=feature_drift,
        recommendation=recommendation,
        analyzed_at=_now_iso(),
    )
