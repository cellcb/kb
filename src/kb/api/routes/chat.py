"""Chat API Routes"""

import asyncio
import json
import time
import uuid
from typing import AsyncGenerator, Dict, Iterable, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..models.chat import ChatRequest, ChatResponse, SourceInfo
from ..dependencies import get_rag_engine


router = APIRouter()


def _format_sources(raw_sources: Iterable[Dict[str, object]]) -> List[SourceInfo]:
    """将引擎返回的来源信息转换为 Pydantic 对象"""
    return [
        SourceInfo(
            filename=source.get("filename", "未知文档"),
            content_preview=source.get("content_preview", ""),
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


@router.post("/query", response_model=ChatResponse, summary="RAG 对话查询")
async def chat_rag(request: ChatRequest):
    """基于向量检索的对话查询（非流式）"""
    try:
        rag_engine = get_rag_engine()
        start_time = time.time()

        top_k = None
        response_mode = "compact"
        if request.search_params:
            top_k = request.search_params.get("top_k") or request.search_params.get("similarity_top_k")
            response_mode = request.search_params.get("response_mode", response_mode)

        result = await rag_engine.rag_search_async(
            request.message,
            similarity_top_k=top_k,
            response_mode=response_mode,
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
        raise HTTPException(status_code=500, detail=f"RAG 查询处理失败: {exc}")


@router.post("/query/stream", summary="RAG 对话查询（SSE）")
async def chat_rag_stream(request: ChatRequest):
    """基于向量检索的流式对话查询，使用 SSE 输出"""
    rag_engine = get_rag_engine()
    session_id = request.session_id or f"session_{uuid.uuid4().hex[:8]}"
    start_time = time.time()

    similarity_top_k = None
    response_mode = "compact"
    if request.search_params:
        similarity_top_k = request.search_params.get("top_k") or request.search_params.get("similarity_top_k")
        response_mode = request.search_params.get("response_mode", response_mode)

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            result = await rag_engine.rag_search_async(
                request.message,
                similarity_top_k=similarity_top_k,
                response_mode=response_mode,
            )

            answer = result.get("answer", "")
            sources = list(_format_sources(result.get("sources", [])))

            for chunk in _iter_answer_chunks(answer):
                yield _sse_payload({
                    "type": "answer",
                    "session_id": session_id,
                    "content": chunk,
                })
                await asyncio.sleep(0)

            metadata = {
                "type": "metadata",
                "session_id": session_id,
                "response_time": f"{time.time() - start_time:.2f}s",
                "sources": [src.model_dump() for src in sources],
            }
            yield _sse_payload(metadata, event="complete")

        except Exception as exc:  # pragma: no cover - SSE 错误直接写入流
            error_payload = {"type": "error", "message": str(exc)}
            yield _sse_payload(error_payload, event="error")

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


@router.post("/search", response_model=ChatResponse, summary="Elasticsearch 关键字查询")
async def chat_es(request: ChatRequest):
    """基于 Elasticsearch 倒排索引的关键字检索"""
    try:
        rag_engine = get_rag_engine()
        start_time = time.time()

        top_k = 5
        min_score = None
        if request.search_params:
            top_k = request.search_params.get("top_k") or request.search_params.get("keyword_top_k") or top_k
            min_score = request.search_params.get("min_score")

        result = await rag_engine.keyword_search_async(
            request.message,
            top_k=top_k,
            min_score=min_score,
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
    # TODO: 实现会话历史存储和检索
    return {
        "session_id": session_id,
        "history": [],
        "message": "会话历史功能将在后续版本中实现"
    }
