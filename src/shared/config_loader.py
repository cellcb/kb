"""Configuration loader for TOML files with .env fallback.

Usage
-----

from shared.config_loader import load_config, set_current_config, get_current_config

conf = load_config("path/to/config.toml")
set_current_config(conf)

Precedence rules
----------------
1. Values explicitly present in TOML have highest priority.
2. Missing values fall back to environment variables (e.g. from a .env file).
3. Remaining values use sane internal defaults.

Path handling
-------------
All relative paths in TOML are resolved against the TOML file directory so the
configuration remains portable when invoked from different working directories.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .config_schema import AppConfig, ElasticsearchConfig, KnowledgeConfig, LLMConfig, LoggingConfig, ServerConfig, TasksConfig


try:  # Python 3.11+
    import tomllib  # type: ignore
except Exception:  # pragma: no cover - fallback for older Pythons
    import tomli as tomllib  # type: ignore


_CURRENT_CONFIG: Optional[AppConfig] = None


def get_current_config() -> Optional[AppConfig]:
    """Return the process-wide configuration if one was set by the entrypoint."""

    return _CURRENT_CONFIG


def set_current_config(conf: AppConfig) -> None:
    """Make the configuration available across the process.

    This allows modules like ``api.main`` to access the loaded configuration
    without CLI argument threading.
    """

    global _CURRENT_CONFIG
    _CURRENT_CONFIG = conf


def _env_bool(name: str, default: Optional[bool] = None) -> Optional[bool]:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _resolve_path(base_dir: Path, value: Optional[str]) -> Optional[str]:
    if value is None or not value:
        return value
    p = Path(value)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return str(p)


def _merge_logging(raw: Dict[str, Any], base: LoggingConfig) -> LoggingConfig:
    env = {
        "level": os.getenv("LOG_LEVEL"),
        "json_logs": _env_bool("APP_LOG_JSON", base.json_logs),
        "access_level": os.getenv("APP_ACCESS_LOG_LEVEL"),
        "request_pretty": _env_bool("APP_REQUEST_LOG_PRETTY", base.request_pretty),
        "request_max_bytes": _env_int("APP_REQUEST_LOG_MAX_BYTES", base.request_max_bytes),
    }
    # Backward compat: map "json" key from TOML to "json_logs"
    raw = dict(raw)
    if "json" in raw and "json_logs" not in raw:
        raw["json_logs"] = raw.pop("json")
    return LoggingConfig(**{**base.model_dump(), **env, **raw})


def _merge_server(raw: Dict[str, Any], base: ServerConfig) -> ServerConfig:
    env = {
        "host": os.getenv("APP_HOST", base.host),
        "port": _env_int("APP_PORT", base.port),
    }
    return ServerConfig(**{**base.model_dump(), **env, **raw})


def _merge_knowledge(raw: Dict[str, Any], base: KnowledgeConfig, base_dir: Path) -> KnowledgeConfig:
    env = {
        "data_dir": os.getenv("DATA_DIR", base.data_dir),
        "persist_dir": os.getenv("PERSIST_DIR", base.persist_dir),
        "auto_ingest_local_data": _env_bool("AUTO_INGEST_LOCAL_DATA", base.auto_ingest_local_data),
        "enable_parallel": _env_bool("ENABLE_PARALLEL", base.enable_parallel),
        "max_workers": _env_int("MAX_WORKERS", base.max_workers),
        "embedding_model": os.getenv("EMBEDDING_MODEL", base.embedding_model),
        "embedding_cache_dir": os.getenv("EMBEDDING_CACHE_DIR", base.embedding_cache_dir or ""),
        "embedding_local_files_only": _env_bool(
            "EMBEDDING_LOCAL_FILES_ONLY", base.embedding_local_files_only
        ),
    }

    merged = {**base.model_dump(), **env, **raw}

    # Normalize paths relative to the config file directory
    merged["data_dir"] = _resolve_path(base_dir, merged.get("data_dir")) or base.data_dir
    merged["persist_dir"] = _resolve_path(base_dir, merged.get("persist_dir")) or base.persist_dir
    if merged.get("embedding_cache_dir"):
        merged["embedding_cache_dir"] = _resolve_path(base_dir, merged.get("embedding_cache_dir"))

    return KnowledgeConfig(**merged)


def _merge_elasticsearch(
    raw: Dict[str, Any], base: ElasticsearchConfig, base_dir: Path
) -> ElasticsearchConfig:
    env = {
        "url": os.getenv("ELASTICSEARCH_URL", base.url),
        "index": os.getenv("ELASTICSEARCH_INDEX", base.index),
        "text_index": os.getenv("ELASTICSEARCH_TEXT_INDEX", base.text_index),
        "user": os.getenv("ELASTICSEARCH_USER", base.user or ""),
        "password": os.getenv("ELASTICSEARCH_PASSWORD", base.password or ""),
        "verify_certs": _env_bool("ELASTICSEARCH_VERIFY_CERTS", base.verify_certs),
        "ca_certs": os.getenv("ELASTICSEARCH_CA_CERTS", base.ca_certs or ""),
        "timeout": _env_int("ELASTICSEARCH_TIMEOUT", base.timeout),
        "text_analyzer": os.getenv("ELASTICSEARCH_TEXT_ANALYZER", base.text_analyzer or ""),
    }
    merged = {**base.model_dump(), **env, **raw}
    if merged.get("ca_certs"):
        merged["ca_certs"] = _resolve_path(base_dir, merged.get("ca_certs"))
    return ElasticsearchConfig(**merged)


def _merge_tasks(raw: Dict[str, Any], base: TasksConfig) -> TasksConfig:
    env = {"max_concurrent_tasks": _env_int("MAX_CONCURRENT_TASKS", base.max_concurrent_tasks)}
    return TasksConfig(**{**base.model_dump(), **env, **raw})


def _merge_llm(raw: Dict[str, Any], base: LLMConfig | None) -> LLMConfig | None:
    if raw is None and base is None:
        # Allow LLM to be omitted in TOML; fallback to env-only if present.
        env_present = any(
            os.getenv(k)
            for k in ("LLM_PROVIDER", "LLM_API_BASE", "LLM_API_KEY", "LLM_MODEL")
        )
        if not env_present:
            return None
        base = LLMConfig()

    base = base or LLMConfig()
    env = {
        "provider": os.getenv("LLM_PROVIDER", base.provider),
        "api_base": os.getenv("LLM_API_BASE", base.api_base),
        "api_key": os.getenv("LLM_API_KEY", base.api_key),
        "model": os.getenv("LLM_MODEL", base.model),
        "temperature": float(os.getenv("LLM_TEMPERATURE", base.temperature)),
        "is_chat_model": _env_bool("LLM_IS_CHAT_MODEL", base.is_chat_model),
    }
    return LLMConfig(**{**base.model_dump(), **env, **(raw or {})})


def load_config(path: str) -> AppConfig:
    """Load configuration from a TOML file and apply .env fallback.

    Relative paths are resolved against the TOML file directory.
    """

    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    base = AppConfig()
    base_dir = config_path.parent

    server_raw = dict(raw.get("server", {}))
    logging_raw = dict(raw.get("logging", {}))
    knowledge_raw = dict(raw.get("knowledge", {}))
    es_raw = dict(raw.get("elasticsearch", {}))
    tasks_raw = dict(raw.get("tasks", {}))
    llm_raw = raw.get("llm")  # may be absent

    conf = AppConfig(
        server=_merge_server(server_raw, base.server),
        logging=_merge_logging(logging_raw, base.logging),
        knowledge=_merge_knowledge(knowledge_raw, base.knowledge, base_dir),
        elasticsearch=_merge_elasticsearch(es_raw, base.elasticsearch, base_dir),
        tasks=_merge_tasks(tasks_raw, base.tasks),
        llm=_merge_llm(llm_raw, base.llm),
    )
    conf._config_path = str(config_path)
    conf._config_dir = str(base_dir)
    return conf


def apply_env_from_config(conf: AppConfig) -> None:
    """Export selected config fields to environment variables for modules that
    read settings from `os.environ` during import-time initialization.

    This is primarily used so that `api.main`'s early `configure_logging()` can
    reflect the TOML logging preferences before the app is imported by Uvicorn.
    """

    os.environ["LOG_LEVEL"] = str(conf.logging.level)
    os.environ["APP_LOG_JSON"] = "1" if conf.logging.json_logs else "0"
    if conf.logging.access_level:
        os.environ["APP_ACCESS_LOG_LEVEL"] = str(conf.logging.access_level)
    os.environ["APP_REQUEST_LOG_PRETTY"] = "1" if conf.logging.request_pretty else "0"
    os.environ["APP_REQUEST_LOG_MAX_BYTES"] = str(conf.logging.request_max_bytes)
    # Uvicorn host/port also exposed for convenience
    os.environ["APP_HOST"] = conf.server.host
    os.environ["APP_PORT"] = str(conf.server.port)
    # LLM configuration exported so child reload process and libraries relying on env can pick it up
    if conf.llm is not None:
        os.environ["LLM_PROVIDER"] = str(conf.llm.provider)
        os.environ["LLM_API_BASE"] = str(conf.llm.api_base)
        os.environ["LLM_API_KEY"] = str(conf.llm.api_key)
        os.environ["LLM_MODEL"] = str(conf.llm.model)
        os.environ["LLM_TEMPERATURE"] = str(conf.llm.temperature)
        os.environ["LLM_IS_CHAT_MODEL"] = "1" if conf.llm.is_chat_model else "0"
