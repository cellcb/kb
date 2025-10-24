"""
FastAPI Main Application
RAG Demo Web API Service
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..core.rag_engine import AsyncRAGEngine
from ..core.task_manager import TaskManager
from . import dependencies
from .routes import chat, documents, tasks, health
from .models.chat import ErrorResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    logging.info("启动RAG API服务...")
    
    # 初始化RAG引擎
    rag_engine = AsyncRAGEngine(
        data_dir="data",
        persist_dir="storage",
        embedding_model="BAAI/bge-small-zh-v1.5",
        enable_parallel=True,
        max_workers=4
    )
    logging.info(
        "Elasticsearch vector store -> %s (index template: %s)",
        rag_engine.es_url,
        rag_engine.es_index_template,
    )
    
    # 初始化任务管理器
    task_manager = TaskManager(max_concurrent_tasks=3, rag_engine=rag_engine)
    await task_manager.start_workers()
    
    # 设置全局依赖
    dependencies.set_rag_engine(rag_engine)
    dependencies.set_task_manager(task_manager)
    
    # 尝试加载现有索引
    try:
        await rag_engine.get_or_create_index_async()
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
    lifespan=lifespan
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
        content=ErrorResponse(
            error="Internal Server Error",
            detail=str(exc)
        ).dict()
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP异常处理"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            code=str(exc.status_code)
        ).dict()
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
        "health": "/api/health"
    }





# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.kb.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
