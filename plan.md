# Logging Unification Plan

**Overall Progress:** `100%`

## Tasks

- [x] 🟩 Finalize logger naming (no `kb.` prefix)
  - Use: `api.*`, `knowledge`, `services.task_manager`, `uvicorn.*`.

- [x] 🟩 Add a single root logging config
  - One root handler; JSON or pretty format via env; no per-module handlers.

- [x] 🟩 Align `uvicorn.*` with root
  - Remove their handlers, set `propagate=True`, level synchronized via env.

- [x] 🟩 Add request-body logging middleware
  - Log raw JSON/form bodies; summarize multipart; apply truncation and exclusions.

- [x] 🟩 Add redaction + truncation controls
  - Env: `APP_REQUEST_LOG_MAX_BYTES`, `APP_REQUEST_LOG_EXCLUDE_PATHS`, `APP_LOG_REDACT_KEYS`.

- [x] 🟩 Provide dynamic log-level control
  - Lightweight admin endpoint or CLI hook to set levels for specific loggers.

- [x] 🟩 Instrument SSE stream and callbacks
  - Log start/complete with `session_id/tenant/chunks/bytes/duration`; callback url/status/bytes.

- [x] 🟩 Update docs and examples
  - Add `/api/documents/upload` curl; document `response_mode=compact`; env knobs for logs.

- [x] 🟩 Add smoke tests (minimal)
  - Verify request logging behavior, SSE happy-path, callback logging and error path.
