"""
app/core/logging.py
-------------------
Centralised logging configuration for the Intelligent AutoML Platform.

Features
--------
* Console handler with ANSI colour support (via ``colorlog`` when installed,
  falling back gracefully to the standard library).
* Rotating file handler (10 MB per file, 5 backups).
* Optional structured JSON log records for log-aggregation pipelines.
* Thread-safe singleton initialisation guard so ``setup_logging`` can be
  called multiple times without duplicating handlers.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import threading
from datetime import datetime, timezone
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_setup_lock = threading.Lock()
_logging_configured = False

# Default log format (also used as fallback when colorlog is unavailable)
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Mapping of level names → colorlog colour codes
_LOG_COLORS: dict[str, str] = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold_red",
}


# ---------------------------------------------------------------------------
# Structured (JSON) formatter
# ---------------------------------------------------------------------------
class _JsonFormatter(logging.Formatter):
    """
    Emit each log record as a single-line JSON object.

    The resulting JSON includes the following keys:
    ``timestamp``, ``level``, ``logger``, ``message``, ``module``,
    ``funcName``, ``lineno``, and — when present — ``exc_info``.
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineno": record.lineno,
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)
        return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _build_console_handler(log_level: int) -> logging.Handler:
    """
    Return a StreamHandler that writes colourised output to *stderr*.

    When ``colorlog`` is importable, its :class:`~colorlog.ColoredFormatter`
    is used.  Otherwise a plain :class:`logging.Formatter` is used as a
    drop-in fallback so the rest of the application is unaffected.

    Parameters
    ----------
    log_level:
        Numeric logging level (e.g. ``logging.INFO``).

    Returns
    -------
    logging.Handler
        Configured console handler.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(log_level)

    try:
        import colorlog  # type: ignore[import-untyped]

        fmt = colorlog.ColoredFormatter(
            fmt="%(log_color)s" + _LOG_FORMAT,
            datefmt=_DATE_FORMAT,
            log_colors=_LOG_COLORS,
            reset=True,
            style="%",
        )
    except ImportError:
        fmt = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    handler.setFormatter(fmt)
    return handler


def _build_file_handler(
    log_file: str,
    log_level: int,
    use_json: bool = False,
) -> logging.Handler:
    """
    Return a :class:`~logging.handlers.RotatingFileHandler`.

    The log directory is created automatically if it does not exist.

    Parameters
    ----------
    log_file:
        Absolute or relative path to the log file.
    log_level:
        Numeric logging level.
    use_json:
        When *True*, records are serialised as JSON instead of the default
        human-readable format.

    Returns
    -------
    logging.Handler
        Configured rotating-file handler.
    """
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    handler.setLevel(log_level)

    if use_json:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)
        )

    return handler


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def setup_logging(
    log_level: str = "INFO",
    log_file: str = "logs/automl.log",
    use_json_file: bool = False,
    logger_name: Optional[str] = None,
) -> logging.Logger:
    """
    Configure and return the root (or named) logger for the platform.

    The function is **idempotent** — subsequent calls with the same arguments
    are no-ops; the cached logger is returned instead.  Pass a different
    *logger_name* to obtain a child logger with independent configuration.

    Parameters
    ----------
    log_level:
        String log level: ``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
        ``"ERROR"``, or ``"CRITICAL"``.  Case-insensitive.
    log_file:
        Path to the rotating log file (relative to the working directory
        or absolute).  Parent directories are created as needed.
    use_json_file:
        When *True*, file records are formatted as JSON objects (suitable for
        Elasticsearch / Loki / Datadog ingestion).  Console output is always
        human-readable.
    logger_name:
        When *None* (default) the root logger is configured.  Pass a
        non-empty string to configure a named hierarchy instead.

    Returns
    -------
    logging.Logger
        The configured logger instance.

    Example
    -------
    >>> from app.core.logging import setup_logging
    >>> logger = setup_logging(log_level="DEBUG")
    >>> logger.info("Platform starting up …")
    """
    global _logging_configured  # noqa: PLW0603

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    with _setup_lock:
        if _logging_configured and logger_name is None:
            return logging.getLogger(logger_name)

        target_logger = logging.getLogger(logger_name)
        target_logger.setLevel(numeric_level)

        # Avoid duplicating handlers when re-configuring a named logger
        if target_logger.handlers:
            target_logger.handlers.clear()

        # Prevent propagation to root when configuring a named logger so that
        # we don't get duplicate records on the root's console handler.
        target_logger.propagate = logger_name is None

        # --- Console ---
        target_logger.addHandler(_build_console_handler(numeric_level))

        # --- File (rotating) ---
        target_logger.addHandler(
            _build_file_handler(log_file, numeric_level, use_json=use_json_file)
        )

        # Silence noisy third-party loggers at WARNING level
        _quiet_third_party()

        if logger_name is None:
            _logging_configured = True

        target_logger.debug(
            "Logging initialised | level=%s | file=%s | json=%s",
            log_level.upper(),
            log_file,
            use_json_file,
        )

    return target_logger


def get_logger(name: str) -> logging.Logger:
    """
    Return a standard :class:`logging.Logger` for *name*.

    This is a thin convenience wrapper around :func:`logging.getLogger`.
    If :func:`setup_logging` has not been called yet, a minimal default
    configuration (WARNING level, stderr only) is applied automatically.

    Parameters
    ----------
    name:
        Typically ``__name__`` from the calling module so that the logger
        hierarchy mirrors the package structure.

    Returns
    -------
    logging.Logger
        Logger instance for *name*.

    Example
    -------
    >>> from app.core.logging import get_logger
    >>> log = get_logger(__name__)
    >>> log.info("Component initialised.")
    """
    logger = logging.getLogger(name)

    # Lazy bootstrap: if nobody has called setup_logging() yet, ensure at
    # least a StreamHandler exists so records are not silently dropped.
    if not logging.root.handlers and not logger.handlers:
        logging.basicConfig(
            level=logging.WARNING,
            format=_LOG_FORMAT,
            datefmt=_DATE_FORMAT,
            stream=sys.stderr,
        )

    return logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _quiet_third_party() -> None:
    """
    Suppress overly verbose log output from common third-party libraries.

    Levels are set to WARNING so that genuine errors still surface while
    routine INFO / DEBUG chatter is hidden.
    """
    noisy_loggers = [
        "uvicorn.access",
        "uvicorn.error",
        "multipart",
        "httpx",
        "httpcore",
        "optuna",
        "matplotlib",
        "PIL",
        "numexpr",
        "filelock",
        "mlflow",
    ]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)
