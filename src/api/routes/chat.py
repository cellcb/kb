"""Chat API Routes"""

import asyncio
import json
import logging
import time
import uuid
from typing import AsyncGenerator, Dict, Iterable, List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ..dependencies import get_conversation_service, get_tenant_id
from ..models.chat import ChatRequest, ChatResponse, SourceInfo

router = APIRouter()
logger = logging.getLogger(__name__)


def _log_request(label: str, payload: Dict[str, object]):
    """统一格式化请求日志"""
    logger.info("%s %s", label, json.dumps(payload, ensure_ascii=False, indent=2))


def _format_sources(raw_sources: Iterable[Dict[str, object]]) -> List[SourceInfo]:
    """将引擎返回的来源信息转换为 Pydantic 对象"""
    return [
        SourceInfo(
            filename=source.get("filename", "未知文档"),
            content_preview=source.get("content_preview", ""),
            document_id=source.get("document_id"),
            score=source.get("score"),
        )
        for source in raw_sources or []
    ]


def _sse_payload(data: Dict[str, object], event: str | None = None) -> str:
    """构造 SSE 数据帧"""
    payload = json.dumps(data, ensure_ascii=False)
    if event:
        return f"event: {event}\ndata: {payload}\n\n"
    return f"data: {payload}\n\n"


def _iter_answer_chunks(answer: str, chunk_size: int = 60) -> Iterable[str]:
    """将回答内容拆分为小段以便 SSE 逐步发送"""
    answer = answer or ""
    if not answer:
        return
    for idx in range(0, len(answer), chunk_size):
        yield answer[idx : idx + chunk_size]


@router.post("/chat", response_model=ChatResponse, summary="RAG 对话查询")
async def chat_rag(request: ChatRequest, tenant_id: str = Depends(get_tenant_id)):
    """基于向量检索的对话查询（非流式）"""
    try:
        conversation_service = get_conversation_service()
        start_time = time.time()
        payload = request.model_dump(exclude_none=False)
        payload["tenant_id"] = tenant_id
        _log_request("chat_rag", payload)

        result = await conversation_service.rag_query(
            request.message,
            search_params=request.search_params,
            tenant_id=tenant_id,
        )

        response_time = f"{time.time() - start_time:.2f}s"
        session_id = request.session_id or f"session_{uuid.uuid4().hex[:8]}"
        raw_sources = result.get("sources", [])
        sources = list(_format_sources(raw_sources))
        answer = result.get("answer", "")
        if result.get("no_results") or not answer.strip():
            answer = "未找到相关内容。"

        return ChatResponse(
            answer=answer,
            sources=sources,
            session_id=session_id,
            response_time=response_time,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG 查询处理失败: {exc}")


@router.post("/chat/stream", summary="RAG 对话查询（SSE）")
async def chat_rag_stream(request: ChatRequest, tenant_id: str = Depends(get_tenant_id)):
    """基于向量检索的流式对话查询，使用 SSE 输出"""
    conversation_service = get_conversation_service()
    session_id = request.session_id or f"session_{uuid.uuid4().hex[:8]}"
    start_time = time.time()
    payload = request.model_dump(exclude_none=False)
    payload["tenant_id"] = tenant_id
    payload["session_id_generated"] = session_id
    _log_request("chat_rag_stream", payload)

    logger.info(
        "chat_rag_stream.start tenant=%s session=%s", tenant_id, session_id, extra={"tenant_id": tenant_id, "session_id": session_id}
    )

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            result = await conversation_service.rag_query(
                request.message,
                search_params=request.search_params,
                tenant_id=tenant_id,
            )

            answer = result.get("answer", "")
            raw_sources = result.get("sources", [])
            sources = list(_format_sources(raw_sources))
            if result.get("no_results") or not answer.strip():
                answer = "未找到相关内容。"

            chunk_count = 0
            byte_count = 0

            for chunk in _iter_answer_chunks(answer):
                yield _sse_payload(
                    {
                        "type": "answer",
                        "session_id": session_id,
                        "content": chunk,
                    }
                )
                await asyncio.sleep(0)
                chunk_count += 1
                byte_count += len(chunk.encode("utf-8"))

            metadata = {
                "type": "metadata",
                "session_id": session_id,
                "response_time": f"{time.time() - start_time:.2f}s",
                "sources": [src.model_dump() for src in sources],
                "no_results": bool(result.get("no_results")),
            }
            yield _sse_payload(metadata, event="complete")

            logger.info(
                "chat_rag_stream.complete tenant=%s session=%s chunks=%d bytes=%d duration_ms=%d",
                tenant_id,
                session_id,
                chunk_count,
                byte_count,
                int((time.time() - start_time) * 1000),
                extra={
                    "tenant_id": tenant_id,
                    "session_id": session_id,
                },
            )

        except Exception as exc:  # pragma: no cover - SSE 错误直接写入流
            error_payload = {"type": "error", "message": str(exc)}
            yield _sse_payload(error_payload, event="error")
            logger.error(
                "chat_rag_stream.error tenant=%s session=%s exc=%s",
                tenant_id,
                session_id,
                exc,
                extra={"tenant_id": tenant_id, "session_id": session_id},
            )

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


@router.post("/search", response_model=ChatResponse, summary="Elasticsearch 关键字查询")
async def chat_es(request: ChatRequest, tenant_id: str = Depends(get_tenant_id)):
    """基于 Elasticsearch 倒排索引的关键字检索"""
    try:
        conversation_service = get_conversation_service()
        start_time = time.time()
        payload = request.model_dump(exclude_none=False)
        payload["tenant_id"] = tenant_id
        _log_request("chat_es", payload)

        result = await conversation_service.keyword_query(
            request.message,
            search_params=request.search_params,
            tenant_id=tenant_id,
        )

        response_time = f"{time.time() - start_time:.2f}s"
        session_id = request.session_id or f"session_{uuid.uuid4().hex[:8]}"
        sources = list(_format_sources(result.get("sources", [])))

        return ChatResponse(
            answer=result.get("answer", ""),
            sources=sources,
            session_id=session_id,
            response_time=response_time,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Elasticsearch 查询处理失败: {exc}")


@router.get("/chat/sessions/{session_id}/history", summary="获取会话历史")
async def get_chat_history(session_id: str):
    """
    获取指定会话的对话历史

    注意：当前版本暂不支持会话历史存储，返回空列表
    """
    _log_request("get_chat_history", {"session_id": session_id})
    # TODO: 实现会话历史存储和检索
    return {"session_id": session_id, "history": [], "message": "会话历史功能将在后续版本中实现"}
