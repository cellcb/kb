"""
Health Check Routes
"""

import psutil
from datetime import datetime
from fastapi import APIRouter, Depends

from ..models.documents import SystemStatus
from ..dependencies import get_rag_engine, get_task_manager


router = APIRouter()


@router.get("/health", summary="健康检查")
async def health_check():
    """基础健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "RAG Demo API"
    }


@router.get("/status", response_model=SystemStatus, summary="系统状态")
async def get_system_status():
    """获取详细系统状态"""
    try:
        rag_engine = get_rag_engine()
        task_manager = get_task_manager()
        
        # 获取系统信息
        system_stats = task_manager.get_system_stats()
        rag_status = rag_engine.get_status()
        
        # 获取内存使用情况
        memory = psutil.virtual_memory()
        memory_usage = f"{memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB"
        
        return SystemStatus(
            index_status="ready" if rag_status['index_ready'] else "not_ready",
            documents_count=rag_status.get('cached_files', 0),
            active_tasks=system_stats['active_tasks'],
            parallel_capacity=system_stats['max_concurrent'],
            memory_usage=memory_usage,
            available_workers=system_stats['available_workers']
        )
        
    except Exception as e:
        return SystemStatus(
            index_status="error",
            documents_count=0,
            active_tasks=0,
            parallel_capacity=0,
            memory_usage="unknown",
            available_workers=0
        )
