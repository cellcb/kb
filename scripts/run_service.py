#!/usr/bin/env python3
"""Executable entrypoint for packaging the FastAPI service."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn


def _ensure_import_paths() -> None:
    """Include the project ``src`` directory when running unfrozen."""

    if getattr(sys, "frozen", False):
        # PyInstaller bootstraps sys.path for us when frozen.
        return

    src_dir = Path(__file__).resolve().parents[1] / "src"
    if src_dir.exists():
        src_str = str(src_dir)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)


_ensure_import_paths()


def _resolve_app_import_path() -> str:
    """Return the ASGI application import path.

    We keep this separate so the path can be overridden via env when needed.
    """

    return os.environ.get("APP_MODULE", "api.main:app")


def _configure_runtime_env() -> None:
    """Adjust runtime environment for frozen executables.

    When the binary is packaged with PyInstaller, resources such as the data
    directory are expected next to the executable. This helper switches the
    working directory accordingly so relative paths continue to work.
    """

    if getattr(sys, "frozen", False):  # Running from PyInstaller bundle
        exe_dir = os.path.dirname(sys.executable)
        os.chdir(exe_dir)


def main() -> None:
    """Start the FastAPI service via Uvicorn."""

    _configure_runtime_env()

    host = os.environ.get("APP_HOST", "0.0.0.0")
    port = int(os.environ.get("APP_PORT", "8000"))
    log_level = os.environ.get("APP_LOG_LEVEL", "info")

    print("Starting KB service ...")

    try:
        uvicorn.run(
            _resolve_app_import_path(),
            host=host,
            port=port,
            reload=False,
            log_level=log_level,
        )
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        print(
            f"Failed to import module '{missing}'. "
            "Ensure PyInstaller includes project packages via hidden imports."
        )
        raise


if __name__ == "__main__":
    main()
