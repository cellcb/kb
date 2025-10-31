"""Central logging configuration helpers.

This module centralises logging setup and runtime controls so that every
component (API, services, workers, third-party integrations) flows through a
single well-defined pipeline.  It keeps code changes small while delivering:

* One root handler (stdout) with either JSON or human-readable formatting
* Level control through environment variables and runtime updates
* Helper accessors for request logging middleware (levels, truncation, redaction)
* Uvicorn logger alignment to avoid duplicate/conflicting handlers
"""

from __future__ import annotations

import json
import logging
import os
from typing import Iterable, Optional, Sequence, Set


DEFAULT_HUMAN_FORMAT = "%(asctime)s %(levelname)s %(name)s - %(message)s"
DEFAULT_HUMAN_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _coerce_level(value: Optional[str], fallback: int) -> int:
    if value is None:
        return fallback
    if isinstance(value, int):
        return value
    lookup = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "warn": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    return lookup.get(str(value).strip().lower(), fallback)


class JsonFormatter(logging.Formatter):
    """Simple JSON line formatter for structured logs."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "lvl": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Attach request-level correlation metadata when present on the record.
        for key in ("request_id", "tenant_id", "session_id", "task_id", "document_id"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    """Initialise the root logger once per process."""

    root = logging.getLogger()

    # Remove any previous handlers (e.g. during uvicorn reload) to avoid duplicate output.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    root_level = _coerce_level(os.getenv("LOG_LEVEL"), logging.INFO)
    log_json = _env_flag("APP_LOG_JSON", default=False)

    handler = logging.StreamHandler()
    if log_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(DEFAULT_HUMAN_FORMAT, DEFAULT_HUMAN_DATEFMT))

    root.addHandler(handler)
    root.setLevel(root_level)

    # Align uvicorn loggers so everything flows through the root handler exactly once.
    uvicorn_level = _coerce_level(os.getenv("APP_ACCESS_LOG_LEVEL"), root_level)
    for name in ("uvicorn", "uvicorn.error"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True
        logger.setLevel(root_level)

    access_logger = logging.getLogger("uvicorn.access")
    access_logger.handlers = []
    access_logger.propagate = True
    access_logger.setLevel(uvicorn_level)


def set_log_level(logger_name: Optional[str], level: str) -> None:
    """Adjust the logging level for a given logger (or root when name is falsy)."""

    target = logging.getLogger(logger_name) if logger_name else logging.getLogger()
    target.setLevel(_coerce_level(level, target.level))


def get_request_log_level() -> int:
    """Return the configured level for request body logging."""

    return _coerce_level(os.getenv("APP_REQUEST_LOG_LEVEL"), logging.INFO)


def get_request_truncate_bytes(default: int = 4096) -> int:
    """Return the maximum number of bytes to log for request bodies."""

    raw = os.getenv("APP_REQUEST_LOG_MAX_BYTES")
    if raw is None:
        return default
    try:
        value = int(raw)
        return default if value <= 0 else value
    except ValueError:
        return default


def get_request_exclude_paths() -> Set[str]:
    """Paths (prefix match) that should never emit request-body logs."""

    raw = os.getenv("APP_REQUEST_LOG_EXCLUDE_PATHS", "")
    items = {"/docs", "/redoc", "/openapi.json", "/favicon.ico"}
    extra = {piece.strip() for piece in raw.split(",") if piece.strip()}
    return items | extra


def get_redact_keys() -> Set[str]:
    """Keys that must be redacted when logging structured payloads."""

    raw = os.getenv("APP_LOG_REDACT_KEYS", "")
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def should_pretty_print_request_body() -> bool:
    """Pretty-print JSON bodies when requested via environment."""

    return _env_flag("APP_REQUEST_LOG_PRETTY", default=False)


def iter_logger_levels(logger_names: Sequence[str]) -> Iterable[tuple[str, str]]:
    """Utility used by admin endpoints to collect current levels."""

    for name in logger_names:
        logger = logging.getLogger(name) if name else logging.getLogger()
        yield (name or "<root>", logging.getLevelName(logger.level))

