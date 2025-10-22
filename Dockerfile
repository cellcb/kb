# RAG Demo Web API Dockerfile
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖 (包括 curl for health check)
RUN apt-get update && apt-get install -y \
    build-essential \
    libmagic1 \
    libmagic-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv
RUN pip install --no-cache-dir uv

# 复制项目配置文件
COPY pyproject.toml uv.lock* ./

# 安装 Python 依赖 (使用 uv)
RUN uv sync --frozen --no-cache

# 复制项目文件
COPY src/ ./src/
COPY data/ ./data/
COPY README.md ./

# 创建必要的目录
RUN mkdir -p /app/storage /app/data

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# 设置环境变量
ENV PYTHONPATH="/app/src"
ENV EMBEDDING_MODEL="BAAI/bge-small-zh-v1.5"
ENV MAX_WORKERS="4"
ENV ELASTICSEARCH_URL="http://elasticsearch:9200"
ENV ELASTICSEARCH_INDEX="kb-documents"

# 启动命令 (使用 uv run)
CMD ["uv", "run", "uvicorn", "kb.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
