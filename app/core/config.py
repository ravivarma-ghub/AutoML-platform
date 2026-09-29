"""
app/core/config.py
------------------
Pydantic Settings-based configuration for the Intelligent AutoML Platform.
All fields can be overridden via environment variables or a .env file.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration class for the AutoML Platform.

    All settings can be overridden by environment variables with the same name
    (case-insensitive).  A `.env` file in the working directory is also supported.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Application metadata                                                 #
    # ------------------------------------------------------------------ #
    APP_NAME: str = Field(
        default="Intelligent AutoML Platform",
        description="Human-readable application name.",
    )
    APP_VERSION: str = Field(
        default="1.0.0",
        description="Semantic version string.",
    )
    DEBUG: bool = Field(
        default=False,
        description="Enable debug mode (verbose logging, hot-reload, etc.).",
    )

    # ------------------------------------------------------------------ #
    # API server                                                           #
    # ------------------------------------------------------------------ #
    API_HOST: str = Field(
        default="0.0.0.0",
        description="Host address the Uvicorn server will bind to.",
    )
    API_PORT: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="TCP port the Uvicorn server will listen on.",
    )

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #
    DATABASE_URL: str = Field(
        default="sqlite:///./automl.db",
        description="SQLAlchemy-compatible database connection URL.",
    )
    MLFLOW_TRACKING_URI: str = Field(
        default="./mlruns",
        description="MLflow tracking server URI (local directory or remote URL).",
    )
    MODEL_REGISTRY_PATH: str = Field(
        default="./models",
        description="Local filesystem path where serialised model artefacts are stored.",
    )
    REPORTS_PATH: str = Field(
        default="./reports",
        description="Directory for generated HTML / PDF reports.",
    )
    LOGS_PATH: str = Field(
        default="./logs",
        description="Directory for rotating log files.",
    )

    # ------------------------------------------------------------------ #
    # Data constraints                                                     #
    # ------------------------------------------------------------------ #
    MAX_ROWS: int = Field(
        default=1_000_000,
        gt=0,
        description="Maximum number of rows accepted in an uploaded dataset.",
    )
    MAX_COLUMNS: int = Field(
        default=500,
        gt=0,
        description="Maximum number of columns accepted in an uploaded dataset.",
    )

    # ------------------------------------------------------------------ #
    # Machine-learning defaults                                            #
    # ------------------------------------------------------------------ #
    CV_FOLDS: int = Field(
        default=5,
        ge=2,
        description="Number of cross-validation folds used during model training.",
    )
    RANDOM_STATE: int = Field(
        default=42,
        description="Global random seed for reproducibility.",
    )
    TEST_SIZE: float = Field(
        default=0.2,
        gt=0.0,
        lt=1.0,
        description="Fraction of data reserved for the hold-out test set.",
    )
    N_JOBS: int = Field(
        default=-1,
        description="Number of parallel workers (-1 = use all available CPU cores).",
    )

    # ------------------------------------------------------------------ #
    # Hyper-parameter optimisation                                         #
    # ------------------------------------------------------------------ #
    OPTUNA_TRIALS: int = Field(
        default=50,
        gt=0,
        description="Number of Optuna trials per hyper-parameter search.",
    )

    # ------------------------------------------------------------------ #
    # Explainability                                                       #
    # ------------------------------------------------------------------ #
    SHAP_MAX_SAMPLES: int = Field(
        default=100,
        gt=0,
        description="Maximum background samples passed to the SHAP explainer.",
    )

    # ------------------------------------------------------------------ #
    # Validators                                                           #
    # ------------------------------------------------------------------ #
    @field_validator("LOGS_PATH", "MODEL_REGISTRY_PATH", "REPORTS_PATH", mode="before")
    @classmethod
    def _ensure_directory_exists(cls, value: str) -> str:
        """Create the directory if it does not exist yet."""
        os.makedirs(value, exist_ok=True)
        return value

    @model_validator(mode="after")
    def _validate_derived_paths(self) -> "Settings":
        """Ensure MLflow tracking directory exists when a local path is given."""
        if not self.MLFLOW_TRACKING_URI.startswith(("http://", "https://")):
            os.makedirs(self.MLFLOW_TRACKING_URI, exist_ok=True)
        return self

    # ------------------------------------------------------------------ #
    # Convenience helpers                                                  #
    # ------------------------------------------------------------------ #
    def as_dict(self) -> dict[str, Any]:
        """Return all settings as a plain Python dictionary."""
        return self.model_dump()

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Settings app={self.APP_NAME!r} version={self.APP_VERSION!r} "
            f"debug={self.DEBUG} port={self.API_PORT}>"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton :class:`Settings` instance.

    The result is cached after the first call so that environment variables and
    the ``.env`` file are only parsed once per process lifetime.

    Returns
    -------
    Settings
        The application-wide configuration object.

    Example
    -------
    >>> from app.core.config import get_settings
    >>> cfg = get_settings()
    >>> print(cfg.APP_NAME)
    Intelligent AutoML Platform
    """
    return Settings()
