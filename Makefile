# RAG Demo Makefile (使用 uv)
# 常用开发命令的快捷方式

.PHONY: help install dev start test clean format lint build docker

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

start:  ## 启动Web服务
	@echo "🚀 启动Web服务..."
	python start_web.py

start-uv:  ## 使用uv启动Web服务
	@echo "🚀 使用uv启动Web服务..."
	UVICORN_LOOP=asyncio uv run uvicorn src.kb.api.main:app --host 0.0.0.0 --port 8000 --reload

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

docker:  ## 构建Docker镜像
	@echo "🐳 构建Docker镜像..."
	docker build -t rag-demo-api .
	@echo "✅ Docker镜像构建完成"

docker-run:  ## 运行Docker容器
	@echo "🐳 运行Docker容器..."
	docker run -p 8000:8000 -v $(PWD)/data:/app/data -v $(PWD)/storage:/app/storage rag-demo-api

docker-compose:  ## 使用docker-compose启动
	@echo "🐳 使用docker-compose启动..."
	docker-compose up --build

info:  ## 显示项目信息
	@echo "📊 项目信息"
	@echo "==========="
	@echo "uv版本: $$(uv --version)"
	@echo "Python版本: $$(uv run python --version)"
	@echo "项目根目录: $$(pwd)"
	@echo "虚拟环境: $$(uv venv --help | head -1)"

# 默认目标
.DEFAULT_GOAL := help
