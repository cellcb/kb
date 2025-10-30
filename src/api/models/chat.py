"""
Chat API Models
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """聊天请求模型"""

    message: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    session_id: Optional[str] = Field(None, description="会话ID")
    search_params: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {"top_k": 5, "min_score": 0.7},
        description="搜索参数，如 top_k、min_score、response_mode 等",
    )


class SourceInfo(BaseModel):
    """来源文档信息"""

    filename: str = Field(..., description="文件名")
    content_preview: str = Field(..., description="内容预览")
    document_id: Optional[str] = Field(None, description="文档ID")
    score: Optional[float] = Field(None, description="相关度分数")


class ChatResponse(BaseModel):
    """聊天响应模型"""

    answer: str = Field(..., description="回答内容")
    sources: List[SourceInfo] = Field(default_factory=list, description="来源文档")
    session_id: str = Field(..., description="会话ID")
    response_time: Optional[str] = Field(None, description="响应时间")


class ErrorResponse(BaseModel):
    """错误响应模型"""

    error: str = Field(..., description="错误信息")
    detail: Optional[str] = Field(None, description="详细错误信息")
    code: Optional[str] = Field(None, description="错误代码")
