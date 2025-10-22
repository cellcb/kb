"""
Dependencies for FastAPI
全局对象管理和依赖注入
"""

from typing import Optional
from fastapi import HTTPException

from ..core.rag_engine import AsyncRAGEngine
from ..core.task_manager import TaskManager


# 全局变量
_rag_engine: Optional[AsyncRAGEngine] = None
_task_manager: Optional[TaskManager] = None


def set_rag_engine(engine: AsyncRAGEngine):
    """设置RAG引擎实例"""
    global _rag_engine
    _rag_engine = engine


def set_task_manager(manager: TaskManager):
    """设置任务管理器实例"""
    global _task_manager
    _task_manager = manager


def get_rag_engine() -> AsyncRAGEngine:
    """获取RAG引擎实例"""
    if _rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG引擎未初始化")
    return _rag_engine


def get_task_manager() -> TaskManager:
    """获取任务管理器实例"""
    if _task_manager is None:
        raise HTTPException(status_code=503, detail="任务管理器未初始化")
    return _task_manager


