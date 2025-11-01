import logging
from datetime import datetime
from typing import Any, Dict

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import dependencies
from api.routes import chat
from api.models.documents import DocumentStatus, TaskStatus
from services.task_manager import TaskData, TaskManager
from shared.logging_config import configure_logging
from shared.request_logging import RequestLoggingMiddleware


configure_logging()


@pytest.fixture
def anyio_backend():
    return "asyncio"

class _StubConversationService:
    async def rag_query(self, message: str, search_params=None, tenant_id: str | None = None):
        return {"answer": "Hello world", "sources": [], "no_results": False}
def test_request_logging_json(monkeypatch, caplog):
    monkeypatch.setenv("APP_REQUEST_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("APP_LOG_REDACT_KEYS", "token")
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)

    @app.post("/echo")
    async def echo(payload: Dict[str, Any]):
        return payload

    client = TestClient(app)

    with caplog.at_level(logging.DEBUG, logger="api.request"):
        client.post("/echo", json={"token": "secret", "value": 1})

    messages = [record.message for record in caplog.records if record.name == "api.request"]
    assert any("****" in message for message in messages)
    assert any("[json] POST /echo" in message for message in messages)


def test_sse_logging(monkeypatch, caplog):
    monkeypatch.setattr(dependencies, "_conversation_service", _StubConversationService())
    app = FastAPI()
    app.include_router(chat.router, prefix="/api")

    client = TestClient(app)

    with caplog.at_level(logging.INFO, logger="api.routes.chat"):
        with client.stream(
            "POST",
            "/api/chat/stream",
            json={"message": "hi", "session_id": "sess"},
            headers={"X-Tenant-ID": "tenant-test"},
        ) as response:
            list(response.iter_text())  # consume stream to trigger completion

    messages = [record.message for record in caplog.records if record.name == "api.routes.chat"]
    assert any("chat_rag_stream.start" in message for message in messages)
    assert any("chat_rag_stream.complete" in message for message in messages)


@pytest.mark.anyio("asyncio")
async def test_task_manager_callback_logging_success(monkeypatch, caplog):
    class DummyResponse:
        status_code = 200

        def raise_for_status(self):
            return None

    class DummyClient:
        def __init__(self, *args, **kwargs):
            self._response = DummyResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, content=None, headers=None):
            self.last_call = {"url": url, "content": content, "headers": headers}
            return self._response

    monkeypatch.setattr("services.task_manager.httpx.AsyncClient", DummyClient)

    manager = TaskManager()
    task_data = TaskData(
        task_id="task123",
        documents=[],
        options={},
        status=TaskStatus.COMPLETED,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        progress={},
        results=[
            {
                "document_id": "docA",
                "filename": "a.txt",
                "upload_time": datetime.utcnow(),
                "file_size": 10,
                "status": DocumentStatus.INDEXED,
                "char_count": 5,
                "processing_time": "0.1s",
                "error": None,
            }
        ],
        tenant_id="tenantA",
        callback_url="https://example.com/callback",
    )

    with caplog.at_level(logging.INFO, logger="services.task_manager"):
        await manager._notify_callback(task_data)

    messages = [record.message for record in caplog.records if record.name == "services.task_manager"]
    assert any("callback.start" in message for message in messages)
    assert any("callback.success" in message for message in messages)


@pytest.mark.anyio("asyncio")
async def test_task_manager_callback_logging_error(monkeypatch, caplog):
    class FailingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, content=None, headers=None):  # pragma: no cover - simple stub
            raise RuntimeError("boom")

    monkeypatch.setattr("services.task_manager.httpx.AsyncClient", FailingClient)

    manager = TaskManager()
    task_data = TaskData(
        task_id="task456",
        documents=[],
        options={},
        status=TaskStatus.COMPLETED,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        progress={},
        results=[],
        tenant_id="tenantB",
        callback_url="https://example.com/callback",
    )

    with caplog.at_level(logging.ERROR, logger="services.task_manager"):
        await manager._notify_callback(task_data)

    messages = [record.message for record in caplog.records if record.name == "services.task_manager"]
    assert any("callback.error" in message for message in messages)
