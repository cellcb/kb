"""FastAPI entrypoint wiring services and routers.

When launched via ``scripts/run_service.py -c config.toml``, the loader
publishes a process-wide configuration accessible here. In development
without a TOML file, the legacy defaults (.env fallback) remain in effect.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from knowledge import KnowledgeService
from services.conversation_service import ConversationService
from services.task_manager import TaskManager

from shared.config_loader import get_current_config, load_config, set_current_config
from shared.logging_config import configure_logging
from shared.request_logging import RequestLoggingMiddleware

from . import dependencies
from .models.chat import ErrorResponse
from .routes import admin, chat, documents, health, tasks


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    logging.info("启动RAG API服务...")

    # 初始化知识服务（优先使用外部配置）
    conf = get_current_config()
    if conf is None:
        config_path = os.getenv("KB_CONFIG_PATH")
        if config_path:
            try:
                conf = load_config(config_path)
                set_current_config(conf)
                logging.info("Reload worker loaded config from %s", config_path)
            except Exception as exc:  # pragma: no cover - defensive logging
                logging.warning("Failed to reload config from %s: %s", config_path, exc)

    if conf is not None:
        # Optional: configure LLM from config before KnowledgeService uses Settings
        try:
            if conf.llm and conf.llm.provider == "openai_like":
                from llama_index.core import Settings
                from llama_index.llms.openai_like import OpenAILike

                Settings.llm = OpenAILike(
                    model=conf.llm.model,
                    api_key=conf.llm.api_key,
                    api_base=conf.llm.api_base,
                    is_chat_model=bool(conf.llm.is_chat_model),
                    temperature=float(conf.llm.temperature),
                )
        except Exception as e:  # pragma: no cover - defensive guardrail
            logging.warning(f"LLM 配置初始化失败，将使用默认设置: {e}")

        knowledge_service = KnowledgeService(
            data_dir=conf.knowledge.data_dir,
            persist_dir=conf.knowledge.persist_dir,
            embedding_model=conf.knowledge.embedding_model,
            embedding_cache_dir=conf.knowledge.embedding_cache_dir,
            embedding_local_files_only=conf.knowledge.embedding_local_files_only,
            enable_parallel=bool(conf.knowledge.enable_parallel),
            max_workers=int(conf.knowledge.max_workers),
            auto_ingest_local_data=bool(conf.knowledge.auto_ingest_local_data),
            es_url=conf.elasticsearch.url,
            es_index=conf.elasticsearch.index,
            es_keyword_index=conf.elasticsearch.text_index,
            es_user=conf.elasticsearch.user,
            es_password=conf.elasticsearch.password,
            es_text_analyzer=conf.elasticsearch.text_analyzer,
        )
    else:
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
    max_tasks = conf.tasks.max_concurrent_tasks if conf is not None else 3
    task_manager = TaskManager(max_concurrent_tasks=max_tasks, knowledge_service=knowledge_service)
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


# 统一日志配置（在应用初始化前执行一次）
configure_logging()

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

# 记录原始请求体（JSON/表单），其余类型保持摘要
app.add_middleware(RequestLoggingMiddleware)


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
app.include_router(admin.router, prefix="/api", tags=["管理"])


@app.get("/", summary="根路径")
async def root():
    """根路径响应"""
    return {
        "message": "KB API Service",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }



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
