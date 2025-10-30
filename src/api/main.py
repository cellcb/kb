"""
FastAPI entrypoint wiring services and routers.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from knowledge import KnowledgeService
from services.conversation_service import ConversationService
from services.task_manager import TaskManager

from . import dependencies
from .models.chat import ErrorResponse
from .routes import chat, documents, health, tasks


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    logging.info("启动RAG API服务...")

    # 初始化知识服务
    knowledge_service = KnowledgeService(
        data_dir="data",
        persist_dir="storage",
        embedding_model="BAAI/bge-small-zh-v1.5",
        enable_parallel=True,
        max_workers=4,
        auto_ingest_local_data=False,
    )
    logging.info(
        "Elasticsearch vector store -> %s (index template: %s)",
        knowledge_service.es_url,
        knowledge_service.es_index_template,
    )

    conversation_service = ConversationService(knowledge_service=knowledge_service)

    # 初始化任务管理器
    task_manager = TaskManager(max_concurrent_tasks=3, knowledge_service=knowledge_service)
    await task_manager.start_workers()

    # 设置全局依赖
    dependencies.set_knowledge_service(knowledge_service)
    dependencies.set_conversation_service(conversation_service)
    dependencies.set_task_manager(task_manager)

    # 尝试加载现有索引
    try:
        await knowledge_service.get_or_create_index_async()
        logging.info("默认租户索引加载成功")
    except Exception as e:
        logging.warning(f"RAG索引加载失败: {e}")

    logging.info("RAG API服务启动完成")

    yield

    # 关闭时清理
    logging.info("关闭RAG API服务...")
    await task_manager.stop_workers()
    logging.info("RAG API服务已关闭")


# 创建FastAPI应用
app = FastAPI(
    title="RAG Demo API",
    description="基于LlamaIndex的RAG（检索增强生成）API服务",
    version="1.0.0",
    lifespan=lifespan,
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 全局异常处理器
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理"""
    logging.error(f"未处理异常: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error="Internal Server Error", detail=str(exc)).dict(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP异常处理"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=exc.detail, code=str(exc.status_code)).dict(),
    )


# 注册路由
app.include_router(health.router, prefix="/api", tags=["健康检查"])
app.include_router(chat.router, prefix="/api", tags=["对话"])
app.include_router(documents.router, prefix="/api", tags=["文档管理"])
app.include_router(tasks.router, prefix="/api", tags=["任务管理"])


@app.get("/", summary="根路径")
async def root():
    """根路径响应"""
    return {
        "message": "KB API Service",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }


# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main():
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    import uvicorn

    main()
