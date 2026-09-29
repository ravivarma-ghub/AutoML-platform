"""
app/core/database.py
--------------------
SQLAlchemy 2.x database setup and ORM model definitions for the
Intelligent AutoML Platform.

Tables
------
* ``experiments``      – one record per AutoML training run.
* ``model_versions``   – versioned model artefacts linked to an experiment.
* ``prediction_logs``  – individual inference records for monitoring.
* ``dataset_profiles`` – cached data-profiling reports.

Usage
-----
FastAPI dependency injection::

    from app.core.database import get_db

    @router.get("/experiments")
    def list_experiments(db: Session = Depends(get_db)):
        return db.query(Experiment).all()

Application startup::

    from app.core.database import create_tables
    create_tables()
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Generator, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Engine & session factory
# ---------------------------------------------------------------------------
_settings = get_settings()

connect_args: dict[str, Any] = {}
if _settings.DATABASE_URL.startswith("sqlite"):
    # Required for SQLite when used with FastAPI (multi-threaded environment)
    connect_args["check_same_thread"] = False

engine = create_engine(
    _settings.DATABASE_URL,
    connect_args=connect_args,
    echo=_settings.DEBUG,           # SQL echo only in debug mode
    future=True,                     # SQLAlchemy 2.x style
    pool_pre_ping=True,              # detect stale connections
)

SessionLocal: sessionmaker[Session] = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,          # safer for async/detached access patterns
)


# Enable WAL mode for SQLite to improve concurrent read performance
@event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_connection: Any, _: Any) -> None:  # noqa: ANN401
    if _settings.DATABASE_URL.startswith("sqlite"):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.close()


# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _new_uuid() -> str:
    """Return a new UUID4 as a plain string (SQLite-compatible primary key)."""
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------
class Experiment(Base):
    """
    Records a single AutoML training run.

    Each row captures the dataset metadata, the winning model, its
    cross-validation and hold-out scores, and the full hyper-parameter
    configuration used to reproduce the result.
    """

    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=_new_uuid,
        doc="UUID primary key.",
    )
    dataset_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        doc="Human-readable name or filename of the training dataset.",
    )
    target_column: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        doc="Name of the target / label column.",
    )
    problem_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        doc="One of: classification, regression, clustering.",
    )
    model_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        doc="Class name of the selected estimator (e.g. RandomForestClassifier).",
    )
    cv_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        doc="Mean cross-validation score on the primary metric.",
    )
    test_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        doc="Hold-out test-set score on the primary metric.",
    )
    training_time_seconds: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        doc="Wall-clock time (seconds) for the full training pipeline.",
    )
    features_used: Mapped[Optional[Any]] = mapped_column(
        JSON,
        nullable=True,
        doc="JSON list of feature names used by the final model.",
    )
    hyperparameters: Mapped[Optional[Any]] = mapped_column(
        JSON,
        nullable=True,
        doc="JSON dict of hyper-parameters for the final estimator.",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        nullable=False,
        doc="UTC timestamp of record creation.",
    )

    # Relationships
    model_versions: Mapped[list["ModelVersion"]] = relationship(
        "ModelVersion",
        back_populates="experiment",
        cascade="all, delete-orphan",
        lazy="select",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Experiment id={self.id!r} model={self.model_name!r} "
            f"cv={self.cv_score}>"
        )


class ModelVersion(Base):
    """
    Represents a versioned, deployable model artefact.

    Supports a simple promotion workflow via the *stage* field:
    ``dev`` → ``validation`` → ``champion`` → ``production``.
    """

    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=_new_uuid,
        doc="UUID primary key.",
    )
    experiment_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Foreign key to the parent Experiment.",
    )
    model_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        doc="Class name of the estimator.",
    )
    version: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="1",
        doc="Monotonically increasing version string within an experiment.",
    )
    stage: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="dev",
        index=True,
        doc="Deployment stage: dev | validation | champion | production.",
    )

    # ---- Classification metrics ----
    accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    f1_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precision_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    recall_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # ---- Regression metrics ----
    r2_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mae: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, doc="Mean Absolute Error."
    )
    mse: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, doc="Mean Squared Error."
    )
    rmse: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, doc="Root Mean Squared Error."
    )

    # ---- Artefact paths ----
    model_path: Mapped[Optional[str]] = mapped_column(
        String(1024),
        nullable=True,
        doc="Filesystem path to the serialised model file (pickle / joblib).",
    )
    metadata_path: Mapped[Optional[str]] = mapped_column(
        String(1024),
        nullable=True,
        doc="Filesystem path to the JSON metadata sidecar file.",
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        doc="Soft-delete flag; False = archived.",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        nullable=False,
    )

    # Relationships
    experiment: Mapped["Experiment"] = relationship(
        "Experiment",
        back_populates="model_versions",
        lazy="select",
    )
    prediction_logs: Mapped[list["PredictionLog"]] = relationship(
        "PredictionLog",
        back_populates="model_version",
        cascade="all, delete-orphan",
        lazy="select",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<ModelVersion id={self.id!r} model={self.model_name!r} "
            f"v={self.version!r} stage={self.stage!r}>"
        )


class PredictionLog(Base):
    """
    Records an individual inference request and its result.

    Used for online monitoring, drift detection, and audit trails.
    """

    __tablename__ = "prediction_logs"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=_new_uuid,
    )
    model_version_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("model_versions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    input_data: Mapped[Optional[Any]] = mapped_column(
        JSON,
        nullable=True,
        doc="JSON-serialised feature dict passed to the model.",
    )
    prediction: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        doc="Model output (class label or regression value) as a string.",
    )
    probability: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        doc="Confidence probability for the predicted class (classification only).",
    )
    prediction_time_ms: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        doc="Inference latency in milliseconds.",
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        nullable=False,
        index=True,
    )

    # Relationships
    model_version: Mapped["ModelVersion"] = relationship(
        "ModelVersion",
        back_populates="prediction_logs",
        lazy="select",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<PredictionLog id={self.id!r} prediction={self.prediction!r} "
            f"ms={self.prediction_time_ms}>"
        )


class DatasetProfile(Base):
    """
    Caches the result of a data-profiling run for a given dataset.

    ``profile_json`` stores the full ydata-profiling report as JSON text so
    that repeated calls for the same dataset can serve the cached version
    without re-running the expensive profiling step.
    """

    __tablename__ = "dataset_profiles"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=_new_uuid,
    )
    dataset_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        doc="Human-readable name or file hash of the dataset.",
    )
    num_rows: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="Number of rows in the dataset at profiling time.",
    )
    num_columns: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="Number of columns in the dataset at profiling time.",
    )
    profile_json: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Full profiling report serialised as a JSON string.",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        nullable=False,
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<DatasetProfile id={self.id!r} dataset={self.dataset_name!r} "
            f"rows={self.num_rows} cols={self.num_columns}>"
        )


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------
def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a database :class:`~sqlalchemy.orm.Session`.

    The session is automatically closed (and any open transaction rolled back)
    when the request context exits — even if an exception is raised.

    Yields
    ------
    Session
        An active SQLAlchemy ORM session.

    Example
    -------
    ::

        from fastapi import Depends
        from sqlalchemy.orm import Session
        from app.core.database import get_db

        @router.get("/experiments/{exp_id}")
        def get_experiment(exp_id: str, db: Session = Depends(get_db)):
            return db.get(Experiment, exp_id)
    """
    db: Session = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def create_tables() -> None:
    """
    Create all ORM-mapped tables in the configured database.

    This is a lightweight, idempotent operation: existing tables are left
    untouched.  For schema migrations in production environments, use Alembic
    instead.

    Raises
    ------
    sqlalchemy.exc.SQLAlchemyError
        If the database is unreachable or the DDL statements fail.
    """
    log.info("Creating database tables (if not exist) at: %s", _settings.DATABASE_URL)
    Base.metadata.create_all(bind=engine)
    log.info(
        "Tables ready: %s",
        ", ".join(Base.metadata.tables.keys()),
    )
