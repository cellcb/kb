"""
Document API Models
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class DocumentStatus(str, Enum):
    """文档状态枚举"""

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class TaskStatus(str, Enum):
    """任务状态枚举"""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingOptions(BaseModel):
    """文档处理选项"""

    parallel_workers: Optional[int] = Field(4, ge=1, le=16, description="并行工作线程数")
    enable_batch_processing: bool = Field(True, description="启用批量处理")
    priority: str = Field("normal", description="处理优先级")


class DocumentInfo(BaseModel):
    """文档信息"""

    document_id: str = Field(..., description="文档ID")
    filename: str = Field(..., description="文件名")
    upload_time: datetime = Field(..., description="上传时间")
    file_size: int = Field(..., description="文件大小(字节)")
    status: DocumentStatus = Field(..., description="文档状态")
    char_count: Optional[int] = Field(None, description="字符数")
    processing_time: Optional[str] = Field(None, description="处理时间")
    error: Optional[str] = Field(None, description="错误信息")


class UploadResponse(BaseModel):
    """文档上传响应"""

    success: bool = Field(..., description="上传是否成功")
    message: str = Field(..., description="响应消息")
    task_id: str = Field(..., description="任务ID")
    files: List[DocumentInfo] = Field(default_factory=list, description="文件信息列表")
    estimated_time: Optional[str] = Field(None, description="预估处理时间")
    parallel_workers: int = Field(..., description="并行工作线程数")


class DocumentListResponse(BaseModel):
    """文档列表响应"""

    documents: List[DocumentInfo] = Field(default_factory=list, description="文档列表")
    total_count: int = Field(..., description="文档总数")
    total_size: int = Field(..., description="总文件大小")


class TaskProgress(BaseModel):
    """任务进度信息"""

    total_files: int = Field(..., description="总文件数")
    processed: int = Field(..., description="已处理文件数")
    failed: int = Field(..., description="失败文件数")
    current_file: Optional[str] = Field(None, description="当前处理文件")
    percentage: float = Field(..., description="完成百分比")


class TaskInfo(BaseModel):
    """任务信息"""

    task_id: str = Field(..., description="任务ID")
    status: TaskStatus = Field(..., description="任务状态")
    progress: TaskProgress = Field(..., description="进度信息")
    results: List[DocumentInfo] = Field(default_factory=list, description="处理结果")
    parallel_workers: int = Field(..., description="并行工作线程数")
    start_time: datetime = Field(..., description="开始时间")
    end_time: Optional[datetime] = Field(None, description="结束时间")
    error: Optional[str] = Field(None, description="错误信息")


class IndexRebuildRequest(BaseModel):
    """索引重建请求"""

    parallel_workers: Optional[int] = Field(4, ge=1, le=16, description="并行工作线程数")
    force: bool = Field(False, description="强制重建")


class SystemStatus(BaseModel):
    """系统状态"""

    index_status: str = Field(..., description="索引状态")
    documents_count: int = Field(..., description="文档数量")
    active_tasks: int = Field(..., description="活跃任务数")
    parallel_capacity: int = Field(..., description="并行处理能力")
    memory_usage: Optional[str] = Field(None, description="内存使用情况")
    available_workers: int = Field(..., description="可用工作线程数")


class ConfigUpdate(BaseModel):
    """配置更新"""

    max_parallel_workers: Optional[int] = Field(None, ge=1, le=32, description="最大并行工作线程数")
    embedding_model: Optional[str] = Field(None, description="Embedding模型")
    chunk_size: Optional[int] = Field(None, ge=100, le=4096, description="文本分块大小")


class IngestDocumentChunk(BaseModel):
    """外部预处理的文档切片"""

    content: str = Field(..., min_length=1, description="文本内容")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="切片级元数据")


class IngestDocument(BaseModel):
    """外部服务提交的文档"""

    document_id: str = Field(..., description="文档唯一ID")
    filename: Optional[str] = Field(None, description="原始文件名")
    content: Optional[str] = Field(None, description="文档完整文本内容")
    chunks: Optional[List[IngestDocumentChunk]] = Field(
        None,
        description="预切好的文本切片；若提供则优先生效",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="文档级元数据")

    @model_validator(mode="after")
    def validate_content_or_chunks(cls, values):
        content = values.content
        chunks = values.chunks
        if (content is None or not content.strip()) and not chunks:
            raise ValueError("content 或 chunks 至少提供一个")
        return values


class IngestRequest(BaseModel):
    """文档摄取请求"""

    documents: List[IngestDocument] = Field(..., min_items=1, description="待索引的文档列表")
    reset_index: bool = Field(False, description="是否重建现有索引")


class IngestResponse(BaseModel):
    """文档摄取响应"""

    success: bool = Field(..., description="摄取是否成功")
    indexed_documents: int = Field(..., description="处理的文档数量")
    indexed_nodes: int = Field(..., description="写入向量索引的节点数")
    keyword_chunks: int = Field(..., description="写入关键字索引的文本块数")
    message: str = Field(..., description="提示信息")
