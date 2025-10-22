"""
Chat API Routes
"""

import time
import uuid
from fastapi import APIRouter, HTTPException, Depends

from ..models.chat import ChatRequest, ChatResponse, SourceInfo
from ..dependencies import get_rag_engine


router = APIRouter()


@router.post("/chat", response_model=ChatResponse, summary="对话查询")
async def chat(request: ChatRequest):
    """
    向RAG系统提问并获取答案
    
    - **message**: 用户问题
    - **session_id**: 可选的会话ID，用于追踪对话上下文
    - **search_params**: 搜索参数配置
    """
    try:
        rag_engine = get_rag_engine()
        
        start_time = time.time()
        
        # 执行查询
        result = await rag_engine.query_async(request.message)
        
        response_time = f"{time.time() - start_time:.2f}s"
        
        # 生成或使用提供的session_id
        session_id = request.session_id or f"session_{uuid.uuid4().hex[:8]}"
        
        # 构建来源信息
        sources = [
            SourceInfo(
                filename=source['filename'],
                content_preview=source['content_preview'],
                score=source['score']
            )
            for source in result['sources']
        ]
        
        return ChatResponse(
            answer=result['answer'],
            sources=sources,
            session_id=session_id,
            response_time=response_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"查询处理失败: {str(e)}"
        )


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
