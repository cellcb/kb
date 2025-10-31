"""
Task Manager for Async Document Processing
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import httpx

from api.models.documents import DocumentInfo, DocumentStatus, TaskInfo, TaskProgress, TaskStatus

if TYPE_CHECKING:
    from knowledge import KnowledgeService


@dataclass
class QueuedDocument:
    """待处理的文档描述"""

    path: Path
    document_id: str
    filename: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskData:
    """任务数据"""

    task_id: str
    documents: List[QueuedDocument] = field(default_factory=list)
    options: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.QUEUED
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    progress: Dict[str, Any] = field(default_factory=dict)
    results: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    tenant_id: Optional[str] = None
    callback_url: Optional[str] = None


class TaskManager:
    """异步任务管理器"""

    def __init__(
        self, max_concurrent_tasks: int = 3, knowledge_service: Optional["KnowledgeService"] = None
    ):
        self.active_tasks: Dict[str, TaskData] = {}
        self.task_queue = asyncio.Queue()
        self.max_concurrent_tasks = max_concurrent_tasks
        self.logger = self._setup_logger()
        self._workers_running = False
        self._worker_tasks: List[asyncio.Task] = []
        self.knowledge_service = knowledge_service

    def _setup_logger(self) -> logging.Logger:
        """配置日志记录器"""
        logger = logging.getLogger(f"{__name__}.TaskManager")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _generate_task_id(self) -> str:
        """生成唯一任务ID"""
        return f"task_{uuid.uuid4().hex[:8]}"

    async def start_workers(self):
        """启动工作线程"""
        if self._workers_running:
            return

        self._workers_running = True
        self.logger.info(f"启动 {self.max_concurrent_tasks} 个任务工作线程")

        for i in range(self.max_concurrent_tasks):
            worker_task = asyncio.create_task(self._worker(f"worker-{i}"))
            self._worker_tasks.append(worker_task)

    async def stop_workers(self):
        """停止工作线程"""
        if not self._workers_running:
            return

        self._workers_running = False

        # 取消所有工作任务
        for worker_task in self._worker_tasks:
            worker_task.cancel()

        # 等待所有任务完成
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()

        self.logger.info("所有任务工作线程已停止")

    async def _worker(self, worker_name: str):
        """工作线程处理任务"""
        self.logger.info(f"工作线程 {worker_name} 启动")

        try:
            while self._workers_running:
                try:
                    # 等待任务，如果5秒内没有任务就继续循环
                    task_data = await asyncio.wait_for(self.task_queue.get(), timeout=5.0)

                    await self._process_task(task_data, worker_name)

                except asyncio.TimeoutError:
                    # 没有任务，继续等待
                    continue
                except Exception as e:
                    self.logger.error(f"工作线程 {worker_name} 处理任务时出错: {e}")

        except asyncio.CancelledError:
            self.logger.info(f"工作线程 {worker_name} 被取消")
        except Exception as e:
            self.logger.error(f"工作线程 {worker_name} 异常退出: {e}")

    async def _process_task(self, task_data: TaskData, worker_name: str):
        """处理单个任务"""
        task_id = task_data.task_id
        self.logger.info(f"工作线程 {worker_name} 开始处理任务 {task_id}")

        try:
            # 更新任务状态
            task_data.status = TaskStatus.PROCESSING
            task_data.start_time = datetime.now()
            task_data.progress = {
                "total_files": len(task_data.documents),
                "processed": 0,
                "failed": 0,
                "current_file": None,
                "percentage": 0,
            }

            # 这里应该调用实际的文档处理逻辑
            await self._process_documents(task_data)

            # 任务完成
            task_data.status = TaskStatus.COMPLETED
            task_data.end_time = datetime.now()
            task_data.progress["percentage"] = 100

            self.logger.info(f"任务 {task_id} 处理完成")

        except Exception as e:
            task_data.status = TaskStatus.FAILED
            task_data.end_time = datetime.now()
            task_data.error = str(e)
            self.logger.error(f"任务 {task_id} 处理失败: {e}")
        finally:
            await self._notify_callback(task_data)

    async def _process_documents(self, task_data: TaskData):
        """真实的文档处理与索引构建逻辑"""
        if not self.knowledge_service:
            raise RuntimeError("任务管理器未配置知识服务，无法处理上传的文档")

        documents = task_data.documents
        total_files = len(documents)
        if total_files == 0:
            return

        tenant_id = task_data.tenant_id or task_data.options.get("tenant_id")

        for index, queued in enumerate(documents):
            start_time = time.time()
            task_data.progress["current_file"] = queued.filename

            upload_time = datetime.now()
            if queued.metadata.get("upload_time"):
                try:
                    upload_time = datetime.fromisoformat(str(queued.metadata["upload_time"]))
                except ValueError:
                    upload_time = datetime.now()

            file_size = queued.metadata.get("file_size")
            if file_size is None:
                try:
                    file_size = queued.path.stat().st_size
                except OSError:
                    file_size = 0

            status = DocumentStatus.INDEXED
            error_message: Optional[str] = None
            char_count: Optional[int] = None

            try:
                if not queued.path.exists():
                    raise FileNotFoundError(f"文件不存在: {queued.path}")

                file_bytes = queued.path.read_bytes()
                if not file_bytes:
                    raise ValueError("文件内容为空")

                documents_from_file = await self.knowledge_service.documents_from_uploads(
                    [{"filename": queued.filename, "data": file_bytes}]
                )
                if not documents_from_file:
                    raise ValueError("未能从文件中提取有效内容")

                document = documents_from_file[0]
                char_count = len(document.text or "")

                metadata = dict(document.metadata or {})
                metadata.update(queued.metadata or {})
                metadata.setdefault("filename", queued.filename)
                metadata.setdefault("file_path", str(queued.path))
                metadata.setdefault("file_size", file_size)
                metadata.setdefault("char_count", char_count)
                metadata["document_id"] = queued.document_id

                payload = [
                    {
                        "document_id": queued.document_id,
                        "filename": queued.filename,
                        "content": document.text,
                        "metadata": metadata,
                    }
                ]

                await self.knowledge_service.ingest_documents_async(
                    payload,
                    reset_existing=False,
                    tenant_id=tenant_id,
                )

            except Exception as exc:
                status = DocumentStatus.FAILED
                error_message = str(exc)
                task_data.progress["failed"] += 1

            processing_time = time.time() - start_time
            task_data.progress["processed"] = index + 1
            task_data.progress["percentage"] = (
                (index + 1) / total_files * 100 if total_files else 100
            )

            task_data.results.append(
                {
                    "document_id": queued.document_id,
                    "filename": queued.filename,
                    "upload_time": upload_time,
                    "file_size": file_size,
                    "status": status,
                    "char_count": char_count,
                    "processing_time": f"{processing_time:.2f}s",
                    "error": error_message,
                }
            )

    async def _notify_callback(self, task_data: TaskData):
        """任务完成后通知回调地址"""
        if not task_data.callback_url:
            return

        payload = self._build_callback_payload(task_data)
        try:
            simplified_payload = {
                "task_id": payload.get("task_id"),
                "status": payload.get("status"),
                "start_time": payload.get("start_time"),
                "end_time": payload.get("end_time"),
                "error": payload.get("error"),
                "tenant_id": payload.get("tenant_id"),
                "document_ids": [
                    result.get("document_id") for result in payload.get("results", [])
                ],
            }
            payload_json = json.dumps(simplified_payload, ensure_ascii=False)
            payload_bytes = len(payload_json.encode("utf-8"))
            results_count = len(payload.get("results", []))
            self.logger.info(
                "任务 %s 准备发送回调: url=%s, status=%s, results=%d, payload_bytes=%d, tenant=%s",
                task_data.task_id,
                task_data.callback_url,
                payload.get("status"),
                results_count,
                payload_bytes,
                task_data.tenant_id,
            )
            self.logger.debug(
                "任务 %s 回调请求体: %s",
                task_data.task_id,
                payload_json,
            )
            headers = {
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "*/*",
                "User-Agent": "curl/8.7.1 (TaskManager)",
                "Connection": "close",
            }
            payload_bytes_data = payload_json.encode("utf-8")
            async with httpx.AsyncClient(timeout=10.0, http2=False) as client:
                response = await client.post(
                    task_data.callback_url,
                    content=payload_bytes_data,
                    headers=headers,
                )
                response.raise_for_status()
            self.logger.info(
                "任务 %s 回调通知已发送至 %s (status=%s)",
                task_data.task_id,
                task_data.callback_url,
                payload["status"],
            )
        except Exception as exc:
            self.logger.error(
                "任务 %s 回调通知失败: %s",
                task_data.task_id,
                exc,
            )

    def _build_callback_payload(self, task_data: TaskData) -> Dict[str, Any]:
        """构建回调通知负载"""
        progress = {
            "total_files": task_data.progress.get("total_files", 0),
            "processed": task_data.progress.get("processed", 0),
            "failed": task_data.progress.get("failed", 0),
            "current_file": task_data.progress.get("current_file"),
            "percentage": task_data.progress.get("percentage", 0),
        }

        results: List[Dict[str, Any]] = []
        for result in task_data.results:
            status = result.get("status")
            if isinstance(status, DocumentStatus):
                status_value = status.value
            else:
                status_value = status
            upload_time = result.get("upload_time")
            if isinstance(upload_time, datetime):
                upload_time_value = upload_time.isoformat()
            else:
                upload_time_value = upload_time

            results.append(
                {
                    "document_id": result.get("document_id"),
                    "filename": result.get("filename"),
                    "upload_time": upload_time_value,
                    "file_size": result.get("file_size"),
                    "status": status_value,
                    "char_count": result.get("char_count"),
                    "processing_time": result.get("processing_time"),
                    "error": result.get("error"),
                }
            )

        start_time = (
            task_data.start_time.isoformat() if isinstance(task_data.start_time, datetime) else None
        )
        end_time = (
            task_data.end_time.isoformat() if isinstance(task_data.end_time, datetime) else None
        )

        status_value = (
            task_data.status.value if isinstance(task_data.status, TaskStatus) else task_data.status
        )

        return {
            "task_id": task_data.task_id,
            "status": status_value,
            "start_time": start_time,
            "end_time": end_time,
            "progress": progress,
            "results": results,
            "error": task_data.error,
            "options": task_data.options,
            "tenant_id": task_data.tenant_id,
        }

    async def submit_task(
        self,
        documents: List[QueuedDocument],
        options: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
        callback_url: Optional[str] = None,
    ) -> str:
        """提交文档处理任务"""
        task_id = self._generate_task_id()

        task_data = TaskData(
            task_id=task_id,
            documents=documents,
            options=options or {},
            status=TaskStatus.QUEUED,
            tenant_id=tenant_id,
            callback_url=callback_url,
        )

        self.active_tasks[task_id] = task_data
        await self.task_queue.put(task_data)

        self.logger.info(f"任务 {task_id} 已提交，包含 {len(documents)} 个文件")
        return task_id

    def get_task_status(self, task_id: str) -> Optional[TaskInfo]:
        """获取任务状态"""
        task_data = self.active_tasks.get(task_id)
        if not task_data:
            return None

        # 构建进度信息
        progress = TaskProgress(
            total_files=task_data.progress.get("total_files", 0),
            processed=task_data.progress.get("processed", 0),
            failed=task_data.progress.get("failed", 0),
            current_file=task_data.progress.get("current_file"),
            percentage=task_data.progress.get("percentage", 0),
        )

        # 构建文档信息列表
        results = []
        for result_data in task_data.results:
            doc_info = DocumentInfo(
                document_id=result_data["document_id"],
                filename=result_data["filename"],
                upload_time=result_data["upload_time"],
                file_size=result_data["file_size"],
                status=result_data["status"],
                char_count=result_data.get("char_count"),
                processing_time=result_data.get("processing_time"),
                error=result_data.get("error"),
            )
            results.append(doc_info)

        return TaskInfo(
            task_id=task_id,
            status=task_data.status,
            progress=progress,
            results=results,
            parallel_workers=task_data.options.get("parallel_workers", 4),
            start_time=task_data.start_time or datetime.now(),
            end_time=task_data.end_time,
            error=task_data.error,
        )

    def list_active_tasks(self) -> List[str]:
        """列出所有活跃任务ID"""
        return list(self.active_tasks.keys())

    def get_active_task_count(self) -> int:
        """获取活跃任务数量"""
        return len(
            [
                task
                for task in self.active_tasks.values()
                if task.status in [TaskStatus.QUEUED, TaskStatus.PROCESSING]
            ]
        )

    async def cleanup_completed_tasks(self, max_age_hours: int = 24):
        """清理已完成的旧任务"""
        current_time = datetime.now()
        tasks_to_remove = []

        for task_id, task_data in self.active_tasks.items():
            if (
                task_data.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
                and task_data.end_time
                and (current_time - task_data.end_time).total_seconds() > max_age_hours * 3600
            ):
                tasks_to_remove.append(task_id)

        for task_id in tasks_to_remove:
            del self.active_tasks[task_id]
            self.logger.info(f"清理旧任务: {task_id}")

    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        active_count = self.get_active_task_count()
        completed_count = len(
            [task for task in self.active_tasks.values() if task.status == TaskStatus.COMPLETED]
        )
        failed_count = len(
            [task for task in self.active_tasks.values() if task.status == TaskStatus.FAILED]
        )

        return {
            "total_tasks": len(self.active_tasks),
            "active_tasks": active_count,
            "completed_tasks": completed_count,
            "failed_tasks": failed_count,
            "max_concurrent": self.max_concurrent_tasks,
            "available_workers": max(0, self.max_concurrent_tasks - active_count),
        }
