# syntax=docker/dockerfile:1

##############################
# 📦 Build stage: produce wheel
##############################
FROM python:3.11-slim AS builder

WORKDIR /app

# 基础依赖（仅构建阶段需要编译工具链）
RUN apt-get update && apt-get install -y \
    build-essential \
    libmagic-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv 以复现依赖并构建 wheel
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install --no-cache-dir uv

# 复制依赖声明并同步（使用锁文件保障一致性）
COPY pyproject.toml uv.lock* README.md ./
RUN uv sync --frozen --no-cache

# 复制源码与构建所需文件，生成 wheel 包
COPY src/ ./src/
RUN uv build


######################################
# 🛠️ Dev stage: x86_64 development base
######################################
FROM --platform=linux/amd64 python:3.11-slim AS dev

WORKDIR /app

# 使用清华镜像安装 uv，镜像仅保留 Python 与 uv 供开发环境使用
ENV PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple" \
    PIP_TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn" \
    UV_PYPI_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmagic-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install --no-cache-dir uv

RUN mkdir -p /app/config /app/data /app/storage

CMD ["bash"]


#################################
# 🚀 Runtime stage: slim artifact
#################################
FROM python:3.11-slim AS runtime

WORKDIR /app

# 仅安装运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv 与应用 wheel（包含全部 Python 依赖）
COPY --from=builder /app/dist /tmp/dist
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install --no-cache-dir uv \
    && pip install --no-cache-dir /tmp/dist/*.whl \
    && rm -rf /tmp/dist

RUN mkdir -p /app/config /app/data /app/storage

ENV EMBEDDING_MODEL="BAAI/bge-small-zh-v1.5" \
    MAX_WORKERS="4" \
    ELASTICSEARCH_INDEX="kb-documents"

VOLUME ["/app/config", "/app/data", "/app/storage"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# 默认通过 uvicorn 启动 API，应结合卷挂载提供配置与数据
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
