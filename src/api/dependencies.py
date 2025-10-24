"""
Dependencies for FastAPI
全局对象管理和依赖注入
"""

import re
from typing import Optional

from fastapi import HTTPException, Header

from knowledge import KnowledgeService
from services.conversation_service import ConversationService
from services.task_manager import TaskManager


_knowledge_service: Optional[KnowledgeService] = None
_conversation_service: Optional[ConversationService] = None
_task_manager: Optional[TaskManager] = None
_TENANT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def set_knowledge_service(service: KnowledgeService):
    """注册全局知识服务实例"""
    global _knowledge_service
    _knowledge_service = service


def set_conversation_service(service: ConversationService):
    """注册对话服务实例"""
    global _conversation_service
    _conversation_service = service


def set_task_manager(manager: TaskManager):
    """设置任务管理器实例"""
    global _task_manager
    _task_manager = manager


def get_knowledge_service() -> KnowledgeService:
    """获取知识服务实例"""
    if _knowledge_service is None:
        raise HTTPException(status_code=503, detail="知识服务未初始化")
    return _knowledge_service


def get_conversation_service() -> ConversationService:
    """获取对话服务实例"""
    if _conversation_service is None:
        raise HTTPException(status_code=503, detail="对话服务未初始化")
    return _conversation_service


def get_task_manager() -> TaskManager:
    """获取任务管理器实例"""
    if _task_manager is None:
        raise HTTPException(status_code=503, detail="任务管理器未初始化")
    return _task_manager


# 暂时保留兼容接口，供旧代码过渡使用
def get_rag_engine() -> KnowledgeService:
    """兼容旧接口"""
    return get_knowledge_service()


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
