"""
Alembic migration environment.

Supports both offline (SQL dump) and online (live DB connection) modes.
The DATABASE_URL is read from the DATABASE_URL environment variable,
falling back to the value in alembic.ini so the CLI works out-of-the-box
without Docker.

Usage:
    alembic upgrade head          # apply all pending migrations
    alembic downgrade -1          # revert the last migration
    alembic revision --autogenerate -m "add users table"
    alembic history --verbose
"""

from __future__ import annotations

import logging
import os
from logging.config import fileConfig
from typing import Any

from alembic import context
from sqlalchemy import engine_from_config, pool, text

# ---------------------------------------------------------------------------
# Import all SQLAlchemy models so Alembic can detect schema changes.
# The Base.metadata must be populated before autogenerate runs.
# ---------------------------------------------------------------------------
# We do a conditional import so that running `alembic` without the full
# application installed does not blow up with an ImportError.
try:
    from app.database import Base  # noqa: F401 — populates Base.metadata
    # Import every model module here so the mappers register themselves:
    import app.models.experiment   # noqa: F401
    import app.models.dataset      # noqa: F401
    import app.models.pipeline     # noqa: F401
    import app.models.prediction   # noqa: F401

    target_metadata = Base.metadata
except ImportError:
    # Fallback: no autogenerate support, but offline/online still works
    logging.getLogger("alembic.env").warning(
        "Could not import app models — autogenerate will not detect changes. "
        "Make sure `app` is on PYTHONPATH."
    )
    target_metadata = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Alembic Config object (gives access to alembic.ini values)
# ---------------------------------------------------------------------------
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

logger = logging.getLogger("alembic.env")

# ---------------------------------------------------------------------------
# Override sqlalchemy.url with the DATABASE_URL environment variable
# (takes precedence over alembic.ini so Docker / CI always wins).
# ---------------------------------------------------------------------------
_db_url: str | None = os.getenv("DATABASE_URL") or config.get_main_option(
    "sqlalchemy.url"
)
if not _db_url:
    raise RuntimeError(
        "No database URL found. Set the DATABASE_URL environment variable "
        "or configure sqlalchemy.url in alembic.ini."
    )

# Patch asyncpg/psycopg3 async DSNs to their sync equivalents when running
# migrations (Alembic's standard engine is synchronous).
_sync_url = (
    _db_url.replace("postgresql+asyncpg://", "postgresql://")
    .replace("postgresql+psycopg://", "postgresql://")
)
config.set_main_option("sqlalchemy.url", _sync_url)

logger.info("Alembic using DB: %s", _sync_url.split("@")[-1])  # hide credentials


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _include_object(
    obj: Any,
    name: str | None,
    type_: str,
    reflected: bool,
    compare_to: Any,
) -> bool:
    """
    Filter hook used by autogenerate.

    Returns False for tables that belong to external schemas (e.g. MLflow's
    own tables) so Alembic doesn't try to drop them.
    """
    # Exclude tables whose names start with the mlflow prefix
    if type_ == "table" and name and name.startswith("mlflow_"):
        return False
    return True


def _run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    In this mode we don't need an actual DB connection — Alembic emits the
    SQL to stdout (or a file) instead.  Useful for generating migration
    scripts for review before applying.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=_include_object,
        compare_type=True,
        compare_server_default=True,
        render_as_batch=False,
    )

    with context.begin_transaction():
        context.run_migrations()


def _run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    Creates a real SQLAlchemy connection, then applies pending migrations
    inside a transaction.  Rolls back automatically if anything fails.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # no pool needed for one-shot migration runs
    )

    with connectable.connect() as connection:
        # Verify connectivity
        connection.execute(text("SELECT 1"))

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=_include_object,
            compare_type=True,
            compare_server_default=True,
            render_as_batch=False,
        )

        logger.info("Running migrations…")
        with context.begin_transaction():
            context.run_migrations()
        logger.info("Migrations complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if context.is_offline_mode():
    logger.info("Running Alembic in offline mode.")
    _run_migrations_offline()
else:
    logger.info("Running Alembic in online mode.")
    _run_migrations_online()
