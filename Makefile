# RAG Demo Makefile (使用 uv)
# 常用开发命令的快捷方式

.PHONY: help install dev start start-uv start-uv-noreload start-c start-c-reload test clean format lint build \
	docker docker-native docker-x86_64 docker-dev-x86_64 docker-dev-run-x86_64 docker-push dist dist-native dist-x86_64 \
	package-wheel deploy deploy-wp

DIST_ENTRY ?= scripts/run_service.py
DIST_NAME ?= kb-service
PYINSTALLER_FLAGS ?= --clean --onefile --paths src --hidden-import api.main \
	--collect-submodules transformers --collect-submodules transformers.models \
	--collect-submodules sentence_transformers --collect-submodules sentence_transformers.models \
	--collect-submodules tiktoken --collect-submodules tiktoken_ext \
	--collect-all tiktoken --collect-all tiktoken_ext \
	--collect-data transformers --collect-data sentence_transformers \
	--collect-data tiktoken --collect-data tiktoken_ext
DIST_MODEL_DIR ?= storage/models
DIST_MODEL_PAYLOAD := $(strip $(wildcard $(DIST_MODEL_DIR)))
DIST_MODEL_FLAG := $(if $(DIST_MODEL_PAYLOAD),--add-data $(DIST_MODEL_DIR):storage/models,)
PYINSTALLER ?= uv run pyinstaller

# Config file path (relative to project root by default)
CONFIG ?= ./config.toml

# Docker image information
IMAGE_NAME ?= kb-api
IMAGE_TAG ?= latest
IMAGE_REGISTRY ?=
IMAGE_FULL_NAME := $(if $(IMAGE_REGISTRY),$(IMAGE_REGISTRY)/)$(IMAGE_NAME):$(IMAGE_TAG)

# Dev container image information
DEV_IMAGE_NAME ?= kb-api-dev
DEV_IMAGE_TAG ?= amd64
DEV_IMAGE_REGISTRY ?=$(IMAGE_REGISTRY)
DEV_IMAGE_FULL_NAME := $(if $(DEV_IMAGE_REGISTRY),$(DEV_IMAGE_REGISTRY)/)$(DEV_IMAGE_NAME):$(DEV_IMAGE_TAG)
DEV_CONTAINER_NAME ?= kb-api-dev

# Deployment configuration
DEPLOY_HOST ?= wp
DEPLOY_PATH ?= /opt/water/apps/kb
DEPLOY_SOURCE ?= .
DEPLOY_TARBALL ?= /tmp/kb-deploy.tar.gz
DEPLOY_EXCLUDES ?= --exclude=".git" \
	--exclude=".venv" \
	--exclude="__pycache__" \
	--exclude="*.pyc" \
	--exclude="dist" \
	--exclude="build" \
	--exclude="storage" \
	--exclude="config.toml" \
	--exclude=".mypy_cache" \
	--exclude=".pytest_cache"

# Set X86_64_PYTHON to the interpreter capable of producing x86_64 binaries.
# Example: X86_64_PYTHON="/usr/bin/arch -x86_64 python3"
X86_64_PYTHON ?=

help:  ## 显示帮助信息
	@echo "RAG Demo 项目管理 (使用 uv)"
	@echo "========================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## 安装项目依赖
	@echo "📦 安装项目依赖..."
	uv sync
	@echo "✅ 依赖安装完成"

install-dev:  ## 安装开发依赖
	@echo "🛠️  安装开发依赖..."
	uv sync --dev
	@echo "✅ 开发依赖安装完成"

start:  ## 启动Web服务（加载 config.toml，禁用热重载）
	@echo "🚀 启动Web服务（加载 $(CONFIG)）..."
	TOKENIZERS_PARALLELISM=false UVICORN_LOOP=asyncio uv run python $(DIST_ENTRY) -c $(CONFIG)

start-uv:  ## 使用uv启动Web服务（加载 config.toml 并启用热重载）
	@echo "🚀 使用uv启动Web服务（加载 $(CONFIG) 并启用热重载）..."
	TOKENIZERS_PARALLELISM=false UVICORN_LOOP=asyncio uv run python $(DIST_ENTRY) -c $(CONFIG) --reload

start-uv-noreload:  ## 使用uv启动Web服务（加载 config.toml，禁用热重载）
	@echo "🚀 使用uv启动Web服务（加载 $(CONFIG)，禁用热重载）..."
	TOKENIZERS_PARALLELISM=false UVICORN_LOOP=asyncio uv run python $(DIST_ENTRY) -c $(CONFIG)

start-c:  ## 使用配置文件启动（无热重载）
	@echo "🚀 使用配置文件启动（$(CONFIG)）..."
	TOKENIZERS_PARALLELISM=false UVICORN_LOOP=asyncio uv run python $(DIST_ENTRY) -c $(CONFIG)

start-c-reload:  ## 使用配置文件启动（热重载）
	@echo "🚀 使用配置文件启动（$(CONFIG)，热重载）..."
	TOKENIZERS_PARALLELISM=false UVICORN_LOOP=asyncio uv run python $(DIST_ENTRY) -c $(CONFIG) --reload

test:  ## 运行API测试
	@echo "🧪 运行API测试..."
	uv run python test_api.py

format:  ## 格式化代码
	@echo "🎨 格式化代码..."
	uv run black src/
	uv run isort src/
	@echo "✅ 代码格式化完成"

lint:  ## 检查代码格式
	@echo "🔍 检查代码格式..."
	uv run black --check src/
	uv run isort --check-only src/
	@echo "✅ 代码格式检查完成"

clean:  ## 清理缓存文件
	@echo "🧹 清理缓存文件..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ 缓存清理完成"

build:  ## 构建项目
	@echo "🔨 构建项目..."
	uv build
	@echo "✅ 项目构建完成"

package-wheel:  ## 生成 wheel 包
	@echo "📦 构建 wheel 包..."
	uv build && ls -1 dist/*.whl
	@echo "✅ Wheel 包已生成"

dist: dist-native dist-x86_64  ## 打包成本机和x86_64可执行文件

dist-native:  ## 使用本机架构生成可执行文件
	@echo "📦 打包本机架构可执行文件..."
	rm -f $(DIST_NAME).spec
	uv pip install --quiet --upgrade pyinstaller
	@if [ -z "$(DIST_MODEL_PAYLOAD)" ]; then \
		echo "⚠️  未找到 $(DIST_MODEL_DIR)，跳过模型资源打包。"; \
	fi
	$(PYINSTALLER) $(PYINSTALLER_FLAGS) $(DIST_MODEL_FLAG) --name $(DIST_NAME) $(DIST_ENTRY)
	@echo "✅ 本机架构可执行文件输出于 dist/$(DIST_NAME)"

dist-x86_64:  ## 使用x86_64架构生成可执行文件（需要Rosetta或x86_64 Python）
	@if [ -z "$(X86_64_PYTHON)" ]; then \
		echo "⚠️  未设置 X86_64_PYTHON，跳过 x86_64 构建。"; \
		echo "    请设置 X86_64_PYTHON=\"/usr/bin/arch -x86_64 python3\" 或指向对应解释器后重新执行 make dist。"; \
	else \
		echo "📦 打包 x86_64 可执行文件..."; \
		rm -f $(DIST_NAME)-x86_64.spec; \
		$(X86_64_PYTHON) -m pip install --quiet --upgrade pyinstaller; \
		if [ -z "$(DIST_MODEL_PAYLOAD)" ]; then \
			echo "⚠️  未找到 $(DIST_MODEL_DIR)，跳过模型资源打包。"; \
		fi; \
		$(X86_64_PYTHON) -m PyInstaller $(PYINSTALLER_FLAGS) $(DIST_MODEL_FLAG) --name $(DIST_NAME)-x86_64 $(DIST_ENTRY); \
		echo "✅ x86_64 可执行文件输出于 dist/$(DIST_NAME)-x86_64"; \
	fi

docker:  ## 构建Docker镜像
	@echo "🐳 构建Docker镜像..."
	docker build -t $(IMAGE_FULL_NAME) .
	@echo "✅ Docker镜像构建完成"

docker-run:  ## 运行Docker容器
	@echo "🐳 运行Docker容器..."
	docker run -p 8000:8000 -v $(PWD)/data:/app/data -v $(PWD)/storage:/app/storage $(IMAGE_FULL_NAME)

docker-compose:  ## 使用docker-compose启动
	@echo "🐳 使用docker-compose启动..."
	docker-compose up --build

docker-native:  ## 构建本机架构 Docker 镜像
	@echo "🐳 构建本机架构Docker镜像..."
	docker build --platform $(shell docker info --format '{{.OSType}}/{{.Architecture}}') -t $(IMAGE_FULL_NAME) .
	@echo "✅ 本机架构Docker镜像构建完成"

docker-x86_64:  ## 构建 x86_64 Docker 镜像
	@echo "🐳 构建x86_64架构Docker镜像..."
	docker build --platform linux/amd64 -t $(IMAGE_FULL_NAME)-amd64 .
	@echo "✅ x86_64 Docker镜像构建完成"

docker-dev-x86_64:  ## 构建 x86_64 开发容器镜像
	@echo "🐳 构建x86_64开发镜像..."
	docker build --platform linux/amd64 --target dev -t $(DEV_IMAGE_FULL_NAME) .
	@echo "✅ 开发镜像构建完成: $(DEV_IMAGE_FULL_NAME)"

docker-dev-run-x86_64: docker-dev-x86_64 ## 运行 x86_64 开发容器（依赖安装在容器内，挂载源码）
	@echo "🛠️  启动开发容器..."
	mkdir -p $(PWD)/data $(PWD)/storage
	docker run --rm -it \
		--platform linux/amd64 \
		-p 8000:8000 \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/pyproject.toml:/app/pyproject.toml \
		-v $(PWD)/uv.lock:/app/uv.lock:ro \
		-v $(PWD)/config.toml:/app/config/config.toml:ro \
		-v $(PWD)/env.example:/app/config/.env.example:ro \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/storage:/app/storage \
		--name $(DEV_CONTAINER_NAME) \
		$(DEV_IMAGE_FULL_NAME) bash

docker-push:  ## 推送镜像 (需先构建)
	@if [ -z "$(IMAGE_REGISTRY)" ]; then \
		echo "⚠️  未设置 IMAGE_REGISTRY，跳过推送。"; \
	else \
		echo "🚢 推送镜像到 $(IMAGE_REGISTRY)..."; \
		docker push $(IMAGE_FULL_NAME); \
		echo "✅ 镜像已推送"; \
	fi

deploy:  ## 部署代码到目标主机目录
	@echo "🚚 部署到 $(DEPLOY_HOST):$(DEPLOY_PATH)..."
	@echo "📦 打包部署文件..."
	COPYFILE_DISABLE=1 gtar --format=gnu --no-xattrs --no-acls -czf $(DEPLOY_TARBALL) $(DEPLOY_EXCLUDES) -C $(DEPLOY_SOURCE) .
	@echo "📤 传输到远端..."
	scp $(DEPLOY_TARBALL) $(DEPLOY_HOST):$(DEPLOY_TARBALL)
	@echo "🗂️  解压部署包..."
	ssh $(DEPLOY_HOST) "mkdir -p $(DEPLOY_PATH) && tar xzf $(DEPLOY_TARBALL) -C $(DEPLOY_PATH) && rm -f $(DEPLOY_TARBALL)"
	rm -f $(DEPLOY_TARBALL)
	@echo "✅ 部署完成"

deploy-wp:  ## 部署代码到 wp:/opt/water/apps/kb
	@$(MAKE) deploy DEPLOY_HOST=wp DEPLOY_PATH=/opt/water/apps/kb

info:  ## 显示项目信息
	@echo "📊 项目信息"
	@echo "==========="
	@echo "uv版本: $$(uv --version)"
	@echo "Python版本: $$(uv run python --version)"
	@echo "项目根目录: $$(pwd)"
	@echo "虚拟环境: $$(uv venv --help | head -1)"

# 默认目标
.DEFAULT_GOAL := help
