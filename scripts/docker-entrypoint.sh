#!/usr/bin/env bash
set -euo pipefail

# Allow skipping uv sync via environment.
if [[ "${SKIP_UV_SYNC:-0}" != "1" ]]; then
    SYNC_FLAGS=()
    [[ -f uv.lock ]] && SYNC_FLAGS+=(--frozen)
    [[ "${INSTALL_DEV_DEPS:-0}" == "1" ]] && SYNC_FLAGS+=(--dev)

    echo "Running uv sync ${SYNC_FLAGS[*]} ..."
    uv sync "${SYNC_FLAGS[@]}"
else
    echo "Skipping uv sync (SKIP_UV_SYNC=${SKIP_UV_SYNC})"
    export UV_NO_SYNC=1
fi

if [[ "${INSTALL_EDITABLE:-0}" == "1" ]]; then
    if [[ "${SKIP_UV_SYNC:-0}" == "1" && ! -d .venv ]]; then
        echo "Requested editable install but no .venv exists; run with SKIP_UV_SYNC=0 or create the venv first."
    else
        echo "Installing project in editable mode..."
        uv pip install --editable /app
    fi
fi

exec "$@"
