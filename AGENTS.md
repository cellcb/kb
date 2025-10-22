# Repository Guidelines

## Project Structure & Module Organization
- `src/kb/api`: FastAPI app wiring, route handlers, and request models; `main.py` boots the service and composes routers.
- `src/kb/core`: Retrieval pipeline primitives such as `AsyncRAGEngine` and `TaskManager`.
- `scripts/`: Helper utilities and maintenance entry points.
- `data/` and `storage/`: Default ingest corpus and local caches (vector data now lives in Elasticsearch); keep large or sensitive inputs out of Git.
- `test_api.py`: Async smoke tests that probe the HTTP surface once the server is running.

## Build, Test, and Development Commands
- `make install` / `make install-dev`: Synchronize dependencies via `uv`; run the dev target when you need test and lint extras.
- `make start` or `make start-uv`: Launch the API locally (`start-uv` reloads through `uv run uvicorn`).
- `make test`: Execute `uv run python test_api.py`, which exercises the live API and requires the server to be listening on `:8000`.
- `make format` / `make lint`: Auto-format with Black and isort, or run check-only verification.
- `make clean`, `make build`, `make docker[-run|-compose]`: Housekeeping, packaging, and container workflows.

## Coding Style & Naming Conventions
- Python code is formatted with Black (line length 100) and sorted with isort (Black profile); always run `make format` before review.
- Use 4-space indentation, `snake_case` for functions and variables, and `PascalCase` for classes to match existing modules.
- Keep FastAPI routes small; push heavy lifting into `core` services or helpers under `scripts/`.

## Testing Guidelines
- Prefer pytest-style coroutines when expanding coverage; colocate new tests near `test_api.py` or mirror module paths under `tests/`.
- Name new test files `test_*.py` and spell async helpers clearly (`async def test_*`).
- Document any external services the tests expect (e.g., running server, configured embeddings) in the docstring so CI jobs can replicate.

## Commit & Pull Request Guidelines
- Existing commits use descriptive, sentence-style summaries (e.g., “Update dependencies...”); follow that tone, leading with the primary change.
- Reference issues in the body (`Refs #123`) and enumerate functional or docs impacts in short bullet lists when opening a PR.
- PRs should describe how you validated the change (`make test`, manual checks) and include screenshots for UI or API contract alterations.

## Environment & Secrets
- Copy `env.example` to `.env` and fill provider keys needed by LlamaIndex backends; never commit real credentials.
- Keep default `data/` inputs lightweight; point to external storage for large corpora via environment variables or volume mounts.
