"""
Task Management API Routes
"""

from typing import List
from fastapi import APIRouter, HTTPException

from ..models.documents import TaskInfo
from ..dependencies import get_task_manager


router = APIRouter()


@router.get("/tasks/{task_id}", response_model=TaskInfo, summary="获取任务状态")
async def get_task_status(task_id: str):
    """
    获取指定任务的状态信息
    
    - **task_id**: 任务ID
    """
    try:
        task_manager = get_task_manager()
        task_info = task_manager.get_task_status(task_id)
        
        if task_info is None:
            raise HTTPException(
                status_code=404,
                detail=f"任务 {task_id} 不存在"
            )
        
        return task_info
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取任务状态失败: {str(e)}"
        )


@router.get("/tasks", summary="获取所有活跃任务")
async def list_active_tasks():
    """
    获取所有活跃任务的列表
    """
    try:
        task_manager = get_task_manager()
        active_task_ids = task_manager.list_active_tasks()
        
        # 获取每个任务的详细信息
        tasks = []
        for task_id in active_task_ids:
            task_info = task_manager.get_task_status(task_id)
            if task_info:
                tasks.append(task_info)
        
        return {
            "active_tasks": tasks,
            "total_count": len(tasks)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取任务列表失败: {str(e)}"
        )


@router.delete("/tasks/{task_id}", summary="取消任务")
async def cancel_task(task_id: str):
    """
    取消指定的任务
    
    注意：当前版本暂不支持任务取消，返回提示信息
    """
    # TODO: 实现任务取消功能
    return {
        "message": f"任务取消功能将在后续版本中实现",
        "task_id": task_id
    }


@router.post("/tasks/cleanup", summary="清理已完成任务")
async def cleanup_completed_tasks(max_age_hours: int = 24):
    """
    清理指定时间之前的已完成任务
    
    - **max_age_hours**: 任务最大保留时间（小时）
    """
    try:
        task_manager = get_task_manager()
        await task_manager.cleanup_completed_tasks(max_age_hours)
        
        return {
            "message": f"已清理超过 {max_age_hours} 小时的已完成任务"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"清理任务失败: {str(e)}"
        )
