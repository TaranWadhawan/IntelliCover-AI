"""
Structured logging configuration using structlog.

Industry standard: every log entry is a JSON object with:
  - timestamp, level, logger, event
  - Contextual fields: user_id, job_id, stage, duration_ms, etc.

This allows log aggregation systems (Datadog, CloudWatch, Loki)
to query logs with filters like: { "job_id": "abc123", "level": "error" }
"""

import logging
import logging.config
import os
import sys
from pathlib import Path
from typing import Any, Dict

import structlog

from app.core.config import settings


def _ensure_log_dir() -> None:
    """Create the log directory if it doesn't exist."""
    Path(settings.LOG_DIR).mkdir(parents=True, exist_ok=True)


def _get_processors_for_format() -> list:
    """Return the correct processor chain for the configured format."""
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]

    if settings.LOG_FORMAT == "json":
        return shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Pretty (human-readable) format for local development
        return shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]


def setup_logging() -> None:
    """
    Configure structlog + standard library logging.

    Call this ONCE at application startup (in main.py lifespan).
    After this, get loggers with: `structlog.get_logger(__name__)`
    """
    _ensure_log_dir()

    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    # ── Standard library handler configuration ────────────────────────────────
    handlers: Dict[str, Any] = {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "plain",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": os.path.join(settings.LOG_DIR, "app.log"),
            "maxBytes": 10 * 1024 * 1024,  # 10 MB per file
            "backupCount": 5,
            "encoding": "utf-8",
            "formatter": "plain",
        },
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": os.path.join(settings.LOG_DIR, "errors.log"),
            "maxBytes": 10 * 1024 * 1024,
            "backupCount": 5,
            "encoding": "utf-8",
            "formatter": "plain",
            "level": "ERROR",
        },
    }

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "plain": {
                    "()": structlog.stdlib.ProcessorFormatter,
                    "processors": [
                        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                        structlog.processors.JSONRenderer()
                        if settings.LOG_FORMAT == "json"
                        else structlog.dev.ConsoleRenderer(colors=False),
                    ],
                },
            },
            "handlers": handlers,
            "loggers": {
                "": {  # Root logger
                    "handlers": ["console", "file", "error_file"],
                    "level": log_level,
                    "propagate": True,
                },
                "uvicorn": {"level": "INFO"},
                "uvicorn.error": {"level": "ERROR"},
                "uvicorn.access": {"level": "WARNING"},
                "celery": {"level": "INFO"},
                "sqlalchemy.engine": {
                    "level": "WARNING"
                },  # Set to DEBUG for SQL query logging
            },
        }
    )

    # ── structlog configuration ───────────────────────────────────────────────
    structlog.configure(
        processors=_get_processors_for_format(),
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Convenience wrapper.
    Usage:
        from app.core.logging_config import get_logger
        logger = get_logger(__name__)
        logger.info("resume_parsed", job_id=job_id, num_skills=5)
    """
    return structlog.get_logger(name)
