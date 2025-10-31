"""Administrative endpoints for runtime diagnostics and controls."""

from __future__ import annotations

import logging
from typing import Dict

from fastapi import APIRouter, HTTPException

from shared.logging_config import iter_logger_levels, set_log_level


router = APIRouter()


DEFAULT_LOGGER_NAMES = (
    "",
    "api.request",
    "api.routes.chat",
    "api.routes.documents",
    "api.routes.tasks",
    "knowledge",
    "services.task_manager",
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
)


@router.get("/admin/loggers", summary="查看日志级别")
async def list_loggers() -> Dict[str, str]:
    """Return the effective level for the common logger namespaces."""

    return {name: level for name, level in iter_logger_levels(DEFAULT_LOGGER_NAMES)}


@router.put("/admin/loggers/{logger_name}", summary="设置日志级别")
async def update_logger_level(logger_name: str, level: str) -> Dict[str, str]:
    """Update the level for a logger; use 'root' to target the root logger."""

    try:
        target_name = None if logger_name in {"root", "<root>", ""} else logger_name
        set_log_level(target_name, level)
        logger = logging.getLogger(target_name) if target_name else logging.getLogger()
        return {"logger": logger_name or "root", "level": logging.getLevelName(logger.level)}
    except Exception as exc:  # pragma: no cover - defensive guardrail
        raise HTTPException(status_code=400, detail=f"无法设置日志级别: {exc}")

