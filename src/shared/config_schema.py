"""Typed configuration schema for TOML-driven runtime settings.

The config is deliberately minimal and mapped to existing modules so that
it integrates without invasive refactors. All fields have sensible defaults
so that development without a TOML file still works (using .env as fallback).

Priority: TOML (highest) > environment (.env) > internal defaults.
Relative paths in TOML are resolved against the TOML file location.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    host: str = Field(default="0.0.0.0", description="Uvicorn host bind address")
    port: int = Field(default=8000, description="Uvicorn port")


class LoggingConfig(BaseModel):
    level: str = Field(default="info", description="Root log level (debug/info/warning/error)")
    json_logs: bool = Field(default=False, description="Emit JSON logs when true")
    access_level: str | None = Field(default=None, description="uvicorn.access level override")
    request_pretty: bool = Field(default=False, description="Pretty-print JSON request bodies")
    request_max_bytes: int = Field(default=4096, description="Max bytes to log for request body")


class KnowledgeConfig(BaseModel):
    data_dir: str = Field(default="data", description="Input corpus directory")
    persist_dir: str = Field(default="storage", description="Runtime cache/persist root")
    auto_ingest_local_data: bool = Field(default=True, description="Auto-ingest on startup")
    enable_parallel: bool = Field(default=True, description="Enable thread pool for processing")
    max_workers: int = Field(default=4, description="Thread pool size for ingestion")

    # Embeddings
    embedding_model: str = Field(default="BAAI/bge-small-zh-v1.5", description="HF embedding model")
    embedding_cache_dir: str | None = Field(default=None, description="Embedding cache directory")
    embedding_local_files_only: bool | None = Field(
        default=None, description="Only use local HF caches (offline mode)"
    )


class ElasticsearchConfig(BaseModel):
    url: str = Field(default="http://localhost:9200", description="Elasticsearch URL")
    index: str = Field(default="kb-documents", description="Vector index template/name")
    text_index: str = Field(default="kb-documents-text", description="Keyword index template/name")
    user: str | None = Field(default=None, description="Elasticsearch basic auth user")
    password: str | None = Field(default=None, description="Elasticsearch basic auth password")
    verify_certs: bool | None = Field(default=None, description="Verify TLS certificates")
    ca_certs: str | None = Field(default=None, description="Path to CA bundle file")
    timeout: int | None = Field(default=None, description="Request timeout in seconds")
    text_analyzer: str | None = Field(default=None, description="Analyzer for keyword index")


class TasksConfig(BaseModel):
    max_concurrent_tasks: int = Field(default=3, description="Concurrent background tasks")


class LLMConfig(BaseModel):
    provider: str = Field(default="openai_like", description="LLM provider kind")
    api_base: str = Field(default="", description="Provider API base URL")
    api_key: str = Field(default="", description="Provider API key")
    model: str = Field(default="", description="Model identifier")
    temperature: float = Field(default=0.1, description="Sampling temperature")
    is_chat_model: bool = Field(default=True, description="Whether the model is chat-based")


class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    knowledge: KnowledgeConfig = Field(default_factory=KnowledgeConfig)
    elasticsearch: ElasticsearchConfig = Field(default_factory=ElasticsearchConfig)
    tasks: TasksConfig = Field(default_factory=TasksConfig)
    llm: LLMConfig | None = Field(default=None)

    # Internal: absolute path to the TOML file (resolved by loader)
    _config_path: str | None = None
    _config_dir: str | None = None
