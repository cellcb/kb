"""Request logging middleware capturing raw JSON/form payloads.

The middleware focuses on observability while respecting safety:

* JSON / x-www-form-urlencoded bodies are logged verbatim (with redaction)
* multipart uploads are summarised instead of dumping binary content
* payloads are truncated by byte length to avoid log flooding
* noisy paths (docs/openapi) are automatically skipped
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict
from urllib.parse import parse_qsl, urlencode

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from .logging_config import (
    get_redact_keys,
    get_request_exclude_paths,
    get_request_log_level,
    get_request_truncate_bytes,
    should_pretty_print_request_body,
)


LOG = logging.getLogger("api.request")


def _should_skip(path: str) -> bool:
    exclusions = get_request_exclude_paths()
    return any(path.startswith(prefix) for prefix in exclusions)


def _redact_json(value: Any, redact_keys: set[str]) -> Any:
    if isinstance(value, dict):
        return {
            key: ("****" if key.lower() in redact_keys else _redact_json(val, redact_keys))
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [_redact_json(item, redact_keys) for item in value]
    return value


def _redact_form(encoded: str, redact_keys: set[str]) -> str:
    pairs = parse_qsl(encoded, keep_blank_values=True)
    sanitized = []
    for key, value in pairs:
        sanitized.append((key, "****" if key.lower() in redact_keys else value))
    return urlencode(sanitized)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs raw request bodies in a safe, readable way."""

    def __init__(self, app):
        super().__init__(app)
        self.log_level = get_request_log_level()
        self.truncate_bytes = get_request_truncate_bytes()
        self.pretty = should_pretty_print_request_body()
        self.redact_keys = get_redact_keys()

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        if _should_skip(request.url.path):
            return await call_next(request)

        content_type = (request.headers.get("content-type") or "").lower()

        try:
            body = await request.body()

            if not body:
                return await call_next(request)

            message = self._build_message(content_type, body, request)

            if message is not None:
                LOG.log(
                    self.log_level,
                    message,
                    extra={
                        "request_id": request.headers.get("x-request-id"),
                        "tenant_id": request.headers.get("x-tenant-id"),
                    },
                )

        except asyncio.CancelledError:
            raise
        except Exception:
            # Never break the request pipeline because of logging issues.
            pass

        return await call_next(request)

    def _build_message(self, content_type: str, body: bytes, request: Request) -> str | None:
        method = request.method.upper()
        path = request.url.path

        if content_type.startswith("application/json"):
            payload = self._format_json_body(body)
            return f"[json] {method} {path} body={payload}"

        if content_type.startswith("application/x-www-form-urlencoded"):
            encoded = body.decode("utf-8", errors="replace")
            sanitized = _redact_form(encoded, self.redact_keys) if self.redact_keys else encoded
            truncated = self._truncate(sanitized)
            return f"[form] {method} {path} body={truncated}"

        if "multipart/form-data" in content_type:
            return f"[multipart] {method} {path} size={len(body)} bytes"

        # Ignore other content types (binary, plain text, etc.) to avoid noise.
        return None

    def _format_json_body(self, body: bytes) -> str:
        text = body.decode("utf-8", errors="replace")
        try:
            data = json.loads(text)
            if self.redact_keys:
                data = _redact_json(data, self.redact_keys)
            if self.pretty:
                formatted = json.dumps(data, ensure_ascii=False, indent=2)
            else:
                formatted = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        except json.JSONDecodeError:
            formatted = text
        return self._truncate(formatted)

    def _truncate(self, content: str) -> str:
        encoded = content.encode("utf-8")
        if len(encoded) <= self.truncate_bytes:
            return content
        cut = encoded[: self.truncate_bytes]
        return cut.decode("utf-8", errors="replace") + "...<truncated>"
