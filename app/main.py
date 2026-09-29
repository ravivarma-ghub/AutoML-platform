"""
main.py
-------
FastAPI application entry point for the Intelligent AutoML Platform.

Features
--------
- Lifespan context manager (startup / shutdown)
- CORS middleware (all origins, for dev)
- Request-logging middleware with timing
- Custom exception handlers (404, 500, RequestValidationError)
- Static-file mount for generated HTML reports
- OpenAPI / Swagger / ReDoc customisation
- WebSocket endpoint for real-time training progress (/ws/train/{job_id})
- All API routers mounted under /api/v1
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("automl.main")

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"
DB_DIR = BASE_DIR / "db"

_REQUIRED_DIRS = [DATA_DIR, UPLOADS_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR, DB_DIR]


def _create_directories() -> None:
    """Create all required platform directories if they do not exist."""
    for directory in _REQUIRED_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug("Directory ready: %s", directory)
    logger.info("All platform directories verified.")


# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------
def _setup_mlflow() -> None:
    """Configure MLflow tracking URI and default experiment."""
    try:
        import mlflow  # type: ignore

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{DB_DIR / 'mlflow.db'}")
        mlflow.set_tracking_uri(tracking_uri)

        default_experiment = os.getenv("MLFLOW_DEFAULT_EXPERIMENT", "automl_platform")
        mlflow.set_experiment(default_experiment)

        logger.info("MLflow tracking URI: %s | experiment: %s", tracking_uri, default_experiment)
    except ImportError:
        logger.warning("mlflow not installed — tracking disabled.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow setup failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Database initialisation
# ---------------------------------------------------------------------------
def _init_database() -> None:
    """Create DB tables if they do not exist (SQLAlchemy / SQLite)."""
    try:
        from app.db.database import init_db  # type: ignore

        init_db()
        logger.info("Database tables initialised.")
    except ImportError:
        logger.warning("app.db.database not found — skipping DB init.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("DB initialisation failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Async context manager executed once at application startup and shutdown.

    Startup
    -------
    1. Create required file-system directories.
    2. Initialise the database schema.
    3. Configure MLflow tracking.
    4. Log platform readiness.

    Shutdown
    --------
    1. Log shutdown signal.
    2. Cancel any in-flight background asyncio tasks.
    """
    # ---- Startup ----
    logger.info("=" * 60)
    logger.info("  Intelligent AutoML Platform — starting up")
    logger.info("=" * 60)

    _create_directories()
    _init_database()
    _setup_mlflow()

    logger.info("Platform ready. Docs: http://0.0.0.0:8000/docs")
    logger.info("=" * 60)

    yield  # ← application runs here

    # ---- Shutdown ----
    logger.info("=" * 60)
    logger.info("  Intelligent AutoML Platform — shutting down")
    logger.info("=" * 60)

    # Cancel any dangling asyncio tasks
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Cancelled %d background asyncio task(s).", len(tasks))

    logger.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    """
    Construct and configure the FastAPI application.

    Returns
    -------
    FastAPI
        Fully configured application instance.
    """
    app = FastAPI(
        title="Intelligent AutoML Platform",
        description=(
            "Production-grade AutoML platform with automated feature engineering, "
            "hyperparameter optimisation, model registry, SHAP explainability, "
            "real-time drift monitoring, and a full REST API."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        openapi_tags=[
            {"name": "Training", "description": "Upload datasets and manage training jobs."},
            {"name": "Prediction", "description": "Run single and batch model inference."},
            {"name": "Models", "description": "Model registry: metadata, promotion, monitoring."},
            {"name": "Health", "description": "Platform health and readiness checks."},
        ],
    )

    # ------------------------------------------------------------------
    # CORS
    # ------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],           # Restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Prediction-Time-Ms", "X-Row-Count"],
    )

    # ------------------------------------------------------------------
    # Request-logging middleware
    # ------------------------------------------------------------------
    app.add_middleware(_RequestLoggingMiddleware)

    # ------------------------------------------------------------------
    # Exception handlers
    # ------------------------------------------------------------------
    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={
                "error": "not_found",
                "message": getattr(exc, "detail", "The requested resource was not found."),
                "path": str(request.url.path),
            },
        )

    @app.exception_handler(500)
    async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred. Please check the server logs.",
                "path": str(request.url.path),
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = []
        for error in exc.errors():
            errors.append(
                {
                    "field": " → ".join(str(loc) for loc in error.get("loc", [])),
                    "message": error.get("msg", "Validation error"),
                    "type": error.get("type", ""),
                }
            )
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "message": "Request body or parameters failed validation.",
                "details": errors,
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "http_error",
                "message": exc.detail,
                "path": str(request.url.path),
            },
        )

    # ------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------
    from app.api.routes_train import router as train_router
    from app.api.routes_predict import router as predict_router
    from app.api.routes_models import router as models_router

    # Mount under /api/v1 and at root level for seamless backwards/test compatibility
    app.include_router(train_router, prefix="/api/v1")
    app.include_router(train_router)
    app.include_router(predict_router, prefix="/api/v1")
    app.include_router(predict_router)
    app.include_router(models_router, prefix="/api/v1")
    app.include_router(models_router)

    # ------------------------------------------------------------------
    # Static files — HTML reports
    # ------------------------------------------------------------------
    if REPORTS_DIR.exists():
        app.mount(
            "/reports",
            StaticFiles(directory=str(REPORTS_DIR)),
            name="reports",
        )

    # ------------------------------------------------------------------
    # Root route
    # ------------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    async def root() -> Dict[str, Any]:
        return {
            "service": "Intelligent AutoML Platform",
            "version": "1.0.0",
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/api/v1/health",
        }

    # ------------------------------------------------------------------
    # WebSocket — real-time training progress
    # ------------------------------------------------------------------
    @app.websocket("/ws/train/{job_id}")
    async def ws_training_progress(websocket: WebSocket, job_id: str) -> None:
        """
        Stream training progress events over WebSocket.

        The client connects and receives JSON messages every second until the
        job reaches a terminal state (completed | failed).

        Message schema
        --------------
        ```json
        {
          "job_id": "...",
          "experiment_id": "...",
          "status": "running",
          "current_step": "hyperparameter_tuning",
          "progress_pct": 65,
          "message": "Running Optuna HPO",
          "result": null,
          "error": null
        }
        ```
        """
        from app.api.routes_train import training_jobs  # local import avoids circular dep

        await websocket.accept()
        logger.info("WebSocket client connected for job %s", job_id)

        try:
            if job_id not in training_jobs:
                await websocket.send_json(
                    {"error": "not_found", "message": f"Job '{job_id}' does not exist."}
                )
                await websocket.close(code=1008)
                return

            terminal_states = {"completed", "failed"}
            poll_interval = float(os.getenv("WS_POLL_INTERVAL_S", "1.0"))

            while True:
                job = training_jobs.get(job_id)
                if job is None:
                    await websocket.send_json(
                        {"error": "job_removed", "message": "Job was removed from the queue."}
                    )
                    break

                payload: Dict[str, Any] = {
                    "job_id": job_id,
                    "experiment_id": job.get("experiment_id", ""),
                    "status": job["status"],
                    "current_step": job["current_step"],
                    "progress_pct": job["progress_pct"],
                    "message": job["message"],
                    "result": job.get("result"),
                    "error": job.get("error"),
                }
                await websocket.send_json(payload)

                if job["status"] in terminal_states:
                    logger.info(
                        "WebSocket job %s reached terminal state '%s'; closing.",
                        job_id,
                        job["status"],
                    )
                    break

                await asyncio.sleep(poll_interval)

        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected for job %s", job_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("WebSocket error for job %s: %s", job_id, exc)
            try:
                await websocket.send_json({"error": "server_error", "message": str(exc)})
            except Exception:  # noqa: BLE001
                pass
        finally:
            try:
                await websocket.close()
            except Exception:  # noqa: BLE001
                pass

    return app


# ---------------------------------------------------------------------------
# Request-logging middleware
# ---------------------------------------------------------------------------
class _RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Log every HTTP request with method, path, status code, and elapsed time.
    Skips noisy health-check and static-file paths.
    """

    _SKIP_PATHS = {"/", "/health", "/api/v1/health", "/favicon.ico", "/openapi.json"}

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in self._SKIP_PATHS or request.url.path.startswith("/reports/"):
            return await call_next(request)

        t0 = time.perf_counter()
        response: Optional[Response] = None
        try:
            response = await call_next(request)
            return response
        finally:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            status = response.status_code if response else 500
            level = logging.WARNING if status >= 400 else logging.INFO
            logger.log(
                level,
                "%s %s → %d  (%.1f ms)",
                request.method,
                request.url.path,
                status,
                elapsed_ms,
            )


# ---------------------------------------------------------------------------
# Application instance (import target for Uvicorn)
# ---------------------------------------------------------------------------
app = create_app()


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "true").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info"),
        ws_ping_interval=20,
        ws_ping_timeout=30,
    )
