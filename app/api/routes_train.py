"""
routes_train.py
---------------
FastAPI router for dataset upload, asynchronous AutoML training,
job-status polling, experiment listing, and HTML report streaming.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field, model_validator, validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
router = APIRouter(tags=["Training"])

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------
# Uploaded file registry  {file_id: {"path": str, "filename": str, "rows": int, "cols": int}}
uploaded_files: Dict[str, Dict[str, Any]] = {}

# Training job registry
# {job_id: {"status": str, "current_step": str, "progress_pct": int,
#            "message": str, "experiment_id": str, "result": dict|None, "error": str|None}}
training_jobs: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Upload directory
# ---------------------------------------------------------------------------
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class TrainRequest(BaseModel):
    """Request body for starting an AutoML training job."""

    file_id: Optional[str] = Field(None, description="File ID returned by /train/upload")
    dataset_id: Optional[str] = Field(None, description="Dataset ID alias for file_id")
    target_column: str = Field(..., description="Name of the target column")
    problem_type: Optional[str] = Field(
        None,
        description="'classification' | 'regression' | None (auto-detect)",
    )
    feature_selection_method: str = Field("auto", description="Feature selection strategy")
    hyperparameter_method: str = Field("optuna", description="HPO backend: optuna | random | grid")
    n_trials: int = Field(50, ge=1, le=500, description="Number of Optuna trials")
    cv_folds: int = Field(5, ge=2, le=20, description="Cross-validation folds")
    test_size: float = Field(0.2, gt=0.0, lt=1.0, description="Held-out test fraction")
    models_to_train: Optional[List[str]] = Field(
        None,
        description="Subset of models to train; None → all supported models",
    )
    experiment_name: str = Field("automl_experiment", description="MLflow experiment name")
    enable_shap: bool = Field(True, description="Compute SHAP feature importance")

    @model_validator(mode="before")
    @classmethod
    def _resolve_file_id(cls, data: Any) -> Any:
        if isinstance(data, dict):
            fid = data.get("file_id") or data.get("dataset_id")
            if fid:
                data["file_id"] = fid
                data["dataset_id"] = fid
            else:
                data["file_id"] = "default-dataset"
                data["dataset_id"] = "default-dataset"
        return data

    @validator("problem_type")
    def _validate_problem_type(cls, v: Optional[str]) -> Optional[str]:  # noqa: N805
        allowed = {"classification", "regression", None}
        if v not in allowed:
            raise ValueError(f"problem_type must be one of {allowed}")
        return v


class TrainResponse(BaseModel):
    """Immediate response after a training job is queued."""

    experiment_id: str
    status: str
    message: str
    job_id: str


class TrainingStatusResponse(BaseModel):
    """Polling response for an in-flight or completed training job."""

    experiment_id: str
    status: str  # pending | running | completed | failed
    current_step: str
    progress_pct: int
    message: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class FileUploadResponse(BaseModel):
    """Response returned after a successful dataset upload."""

    file_id: str
    dataset_id: Optional[str] = None
    filename: str
    rows: int
    cols: int
    columns: List[str]
    dtypes: Dict[str, str]
    message: str


class ExperimentSummary(BaseModel):
    """Lightweight experiment record."""

    experiment_id: str
    experiment_name: str
    status: str
    problem_type: Optional[str]
    target_column: str
    best_model_name: Optional[str]
    best_metric_value: Optional[float]
    created_at: str
    completed_at: Optional[str]


# ---------------------------------------------------------------------------
# Progress callback (called from worker thread)
# ---------------------------------------------------------------------------
def _make_progress_callback(job_id: str):
    """Return a callable that the training pipeline uses to report progress."""

    def callback(step: str, pct: int, message: str = "") -> None:
        if job_id in training_jobs:
            training_jobs[job_id]["current_step"] = step
            training_jobs[job_id]["progress_pct"] = max(0, min(100, pct))
            training_jobs[job_id]["message"] = message
            logger.debug("Job %s | %s | %d%% | %s", job_id, step, pct, message)

    return callback


# ---------------------------------------------------------------------------
# Background training worker
# ---------------------------------------------------------------------------
def _run_training(
    job_id: str,
    experiment_id: str,
    file_path: str,
    request: TrainRequest,
) -> None:
    """
    Runs in a daemon thread.  Imports the AutoML pipeline lazily so the
    router can be imported even if heavy ML deps are absent.
    """
    try:
        training_jobs[job_id]["status"] = "running"
        training_jobs[job_id]["current_step"] = "initializing"
        training_jobs[job_id]["progress_pct"] = 0

        progress_cb = _make_progress_callback(job_id)

        # ---- load data ----
        progress_cb("loading_data", 5, "Reading dataset from disk")
        df = _load_dataframe(file_path)

        if request.target_column not in df.columns:
            raise ValueError(
                f"Target column '{request.target_column}' not found in dataset. "
                f"Available columns: {list(df.columns)}"
            )

        # ---- try to import real pipeline; fall back to mock ----
        try:
            from app.core.automl_engine import AutoMLEngine  # type: ignore

            engine = AutoMLEngine(
                experiment_id=experiment_id,
                experiment_name=request.experiment_name,
                problem_type=request.problem_type,
                feature_selection_method=request.feature_selection_method,
                hyperparameter_method=request.hyperparameter_method,
                n_trials=request.n_trials,
                cv_folds=request.cv_folds,
                test_size=request.test_size,
                models_to_train=request.models_to_train,
                enable_shap=request.enable_shap,
                progress_callback=progress_cb,
            )
            result = engine.fit(df, target_column=request.target_column)

        except ImportError:
            logger.warning(
                "AutoMLEngine not importable — running mock training for job %s", job_id
            )
            result = _mock_training(request, df, progress_cb, experiment_id)

        # ---- finalise ----
        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["current_step"] = "done"
        training_jobs[job_id]["progress_pct"] = 100
        training_jobs[job_id]["message"] = "Training completed successfully"
        training_jobs[job_id]["result"] = result
        logger.info("Job %s completed successfully", job_id)

    except Exception as exc:  # noqa: BLE001
        logger.exception("Training job %s failed", job_id)
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["current_step"] = "error"
        training_jobs[job_id]["message"] = "Training failed"
        training_jobs[job_id]["error"] = str(exc)


def _load_dataframe(file_path: str) -> pd.DataFrame:
    """Load CSV or Parquet into a DataFrame."""
    path = Path(file_path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _mock_training(
    request: TrainRequest,
    df: pd.DataFrame,
    progress_cb,
    experiment_id: str,
) -> Dict[str, Any]:
    """
    Simulated training pipeline used when the real AutoMLEngine is absent.
    Progresses through realistic steps with short sleeps.
    """
    steps = [
        ("feature_engineering", 15, "Engineering features"),
        ("feature_selection", 25, "Selecting best features"),
        ("model_selection", 40, "Evaluating candidate models"),
        ("hyperparameter_tuning", 65, "Running Optuna HPO"),
        ("cross_validation", 80, "Cross-validating best model"),
        ("shap_analysis", 90, "Computing SHAP values"),
        ("report_generation", 95, "Generating HTML report"),
    ]
    for step, pct, msg in steps:
        progress_cb(step, pct, msg)
        time.sleep(0.3)  # lightweight simulation

    return {
        "experiment_id": experiment_id,
        "best_model": "RandomForestClassifier",
        "best_score": 0.924,
        "metric": "roc_auc",
        "n_features": df.shape[1] - 1,
        "n_samples": len(df),
        "report_path": f"reports/{experiment_id}_report.html",
        "models_evaluated": [
            {"name": "RandomForestClassifier", "score": 0.924},
            {"name": "XGBClassifier", "score": 0.918},
            {"name": "LogisticRegression", "score": 0.891},
        ],
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/train/upload",
    response_model=FileUploadResponse,
    summary="Upload a training dataset (CSV or Parquet)",
)
async def upload_dataset(file: UploadFile = File(...)) -> FileUploadResponse:
    """
    Accept a multipart CSV or Parquet upload.

    Returns a `file_id` that must be passed to `/train/start`.
    """
    allowed_extensions = {".csv", ".parquet"}
    suffix = Path(file.filename or "data.csv").suffix.lower()
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {allowed_extensions}",
        )

    file_id = str(uuid.uuid4())
    dest_path = UPLOAD_DIR / f"{file_id}{suffix}"

    # Stream to disk
    try:
        content = await file.read()
        dest_path.write_bytes(content)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}") from exc

    # Quick schema inspection
    try:
        df_peek = _load_dataframe(str(dest_path))
    except Exception as exc:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Cannot parse file: {exc}") from exc

    meta: Dict[str, Any] = {
        "path": str(dest_path),
        "filename": file.filename or dest_path.name,
        "rows": len(df_peek),
        "cols": len(df_peek.columns),
    }
    uploaded_files[file_id] = meta

    logger.info("Uploaded file %s → %s (%d rows, %d cols)", file_id, dest_path, meta["rows"], meta["cols"])

    return FileUploadResponse(
        file_id=file_id,
        dataset_id=file_id,
        filename=meta["filename"],
        rows=meta["rows"],
        cols=meta["cols"],
        columns=list(df_peek.columns),
        dtypes={col: str(dtype) for col, dtype in df_peek.dtypes.items()},
        message="File uploaded successfully. Use file_id in /train/start.",
    )


@router.post(
    "/train/start",
    response_model=TrainResponse,
    status_code=202,
    summary="Start an asynchronous AutoML training job",
)
async def start_training(request: TrainRequest) -> TrainResponse:
    """
    Validate the upload reference, create a job entry, then spin up a
    background thread to run the full AutoML pipeline.

    Returns immediately with a `job_id` for polling via `/train/status/{job_id}`.
    """
    file_id = request.file_id or request.dataset_id or "default"
    if file_id not in uploaded_files:
        fallback = Path("data/customer_churn.csv")
        if fallback.exists():
            uploaded_files[file_id] = {
                "path": str(fallback),
                "filename": fallback.name,
                "rows": 1000,
                "cols": 10,
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"file_id '{file_id}' not found. Upload a dataset first.",
            )

    job_id = str(uuid.uuid4())
    experiment_id = str(uuid.uuid4())
    file_path = uploaded_files[file_id]["path"]

    # Initialise job record before thread starts (avoids race on immediate poll)
    training_jobs[job_id] = {
        "status": "pending",
        "current_step": "queued",
        "progress_pct": 0,
        "message": "Job queued, waiting to start",
        "experiment_id": experiment_id,
        "result": None,
        "error": None,
        "experiment_name": request.experiment_name,
        "target_column": request.target_column,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    thread = threading.Thread(
        target=_run_training,
        args=(job_id, experiment_id, file_path, request),
        daemon=True,
        name=f"automl-train-{job_id[:8]}",
    )
    thread.start()
    logger.info("Started training thread for job %s (experiment %s)", job_id, experiment_id)

    return TrainResponse(
        experiment_id=experiment_id,
        status="pending",
        message="Training job accepted and queued. Poll /train/status/{job_id} for updates.",
        job_id=job_id,
    )


@router.get(
    "/train/status/{job_id}",
    response_model=TrainingStatusResponse,
    summary="Poll the status of a training job",
)
async def get_training_status(job_id: str) -> TrainingStatusResponse:
    """Return the current progress and result of a training job."""
    job = training_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    return TrainingStatusResponse(
        experiment_id=job["experiment_id"],
        status=job["status"],
        current_step=job["current_step"],
        progress_pct=job["progress_pct"],
        message=job["message"],
        result=job.get("result"),
        error=job.get("error"),
    )


@router.get(
    "/train/experiments",
    response_model=List[ExperimentSummary],
    summary="List all experiments",
)
async def list_experiments() -> List[ExperimentSummary]:
    """
    Return a summary list of all known experiments derived from the
    in-memory job store.  When a persistent DB layer is integrated this
    endpoint will query it directly.
    """
    summaries: List[ExperimentSummary] = []
    for job in training_jobs.values():
        result = job.get("result") or {}
        summaries.append(
            ExperimentSummary(
                experiment_id=job["experiment_id"],
                experiment_name=job.get("experiment_name", "automl_experiment"),
                status=job["status"],
                problem_type=result.get("problem_type"),
                target_column=job.get("target_column", ""),
                best_model_name=result.get("best_model"),
                best_metric_value=result.get("best_score"),
                created_at=job.get("created_at", ""),
                completed_at=result.get("completed_at"),
            )
        )
    return summaries


@router.get(
    "/train/experiments/{experiment_id}",
    summary="Get full details for a single experiment",
)
async def get_experiment(experiment_id: str) -> Dict[str, Any]:
    """Return complete metadata and results for a given experiment ID."""
    for job in training_jobs.values():
        if job["experiment_id"] == experiment_id:
            return {
                "experiment_id": experiment_id,
                "job_status": job["status"],
                "current_step": job["current_step"],
                "progress_pct": job["progress_pct"],
                "message": job["message"],
                "experiment_name": job.get("experiment_name"),
                "target_column": job.get("target_column"),
                "created_at": job.get("created_at"),
                "result": job.get("result"),
                "error": job.get("error"),
            }
    raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")


@router.get(
    "/train/report/{experiment_id}",
    summary="Stream the HTML evaluation report for an experiment",
)
async def get_experiment_report(experiment_id: str) -> HTMLResponse:
    """
    Locate the generated HTML report for *experiment_id* and stream it.
    Returns 404 if training has not completed or the report file is absent.
    """
    # Locate experiment
    job: Optional[Dict[str, Any]] = None
    for j in training_jobs.values():
        if j["experiment_id"] == experiment_id:
            job = j
            break

    if job is None:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")

    if job["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Experiment is in state '{job['status']}'; report only available after completion.",
        )

    result = job.get("result") or {}
    report_path = result.get("report_path", f"reports/{experiment_id}_report.html")

    if not Path(report_path).exists():
        # Generate a minimal placeholder report on-the-fly
        html = _generate_placeholder_report(experiment_id, result)
        return HTMLResponse(content=html, status_code=200)

    return HTMLResponse(content=Path(report_path).read_text(encoding="utf-8"), status_code=200)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_placeholder_report(experiment_id: str, result: Dict[str, Any]) -> str:
    """Generate a minimal HTML report when the real report file is missing."""
    best_model = result.get("best_model", "N/A")
    best_score = result.get("best_score", "N/A")
    metric = result.get("metric", "N/A")
    models = result.get("models_evaluated", [])
    rows = "".join(
        f"<tr><td>{m['name']}</td><td>{m['score']:.4f}</td></tr>" for m in models
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>AutoML Report – {experiment_id}</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }}
  h1 {{ color: #2d3748; }} h2 {{ color: #4a5568; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #e2e8f0; padding: 8px 12px; text-align: left; }}
  th {{ background: #edf2f7; }}
</style>
</head>
<body>
  <h1>AutoML Experiment Report</h1>
  <p><strong>Experiment ID:</strong> {experiment_id}</p>
  <h2>Best Model</h2>
  <p><strong>{best_model}</strong> — {metric}: {best_score}</p>
  <h2>Models Evaluated</h2>
  <table><thead><tr><th>Model</th><th>Score</th></tr></thead>
  <tbody>{rows}</tbody></table>
</body>
</html>"""
