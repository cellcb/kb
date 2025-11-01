#!/usr/bin/env python3
"""Executable entrypoint requiring a TOML config file (-c/--config).

This entrypoint is intended for packaged binaries (PyInstaller) and for
operational runs. It enforces explicit configuration and exports logging
knobs to the environment so the API's import-time logging setup reflects
the user's preferences.
"""

from __future__ import annotations

import argparse
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

# Delayed imports until sys.path is adjusted
from shared.config_loader import apply_env_from_config, load_config, set_current_config  # noqa: E402


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KB API Service")
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to TOML configuration file (relative paths resolved from file dir)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable Uvicorn autoreload (development mode)",
    )
    return parser.parse_args()


def main() -> None:
    """Start the FastAPI service via Uvicorn with explicit config."""

    _configure_runtime_env()
    args = _parse_args()

    conf = load_config(args.config)
    set_current_config(conf)
    apply_env_from_config(conf)

    print(f"Starting KB service with config: {conf._config_path}")

    try:
        uvicorn.run(
            _resolve_app_import_path(),
            host=conf.server.host,
            port=conf.server.port,
            reload=bool(args.reload),
            log_level=str(conf.logging.level).lower(),
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
