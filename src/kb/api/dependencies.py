"""
Dependencies for FastAPI
全局对象管理和依赖注入
"""

import re
from typing import Optional
from fastapi import HTTPException, Header

from ..core.rag_engine import AsyncRAGEngine
from ..core.task_manager import TaskManager


# 全局变量
_rag_engine: Optional[AsyncRAGEngine] = None
_task_manager: Optional[TaskManager] = None
_TENANT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


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


def get_tenant_id(x_tenant_id: str = Header(default=None, alias="X-Tenant-ID")) -> str:
    """
    从请求头中解析租户标识。

    允许显式传入 `X-Tenant-ID`，若缺失或为空则返回 HTTP 400。
    """
    if not x_tenant_id:
        raise HTTPException(status_code=400, detail="缺少租户标识头 X-Tenant-ID")
    tenant_id = x_tenant_id.strip()
    if not tenant_id:
        raise HTTPException(status_code=400, detail="租户标识不能为空")
    if not _TENANT_ID_PATTERN.fullmatch(tenant_id):
        raise HTTPException(status_code=400, detail="租户标识只能包含字母、数字、下划线或连字符")
    return tenant_id
