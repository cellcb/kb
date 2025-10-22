#!/bin/bash
"""
RAG Demo 安装脚本 (使用 uv)
一键安装和配置项目
"""

set -e

echo "🚀 RAG Demo 自动安装脚本"
echo "========================="

# 检查 uv 是否安装
if ! command -v uv &> /dev/null; then
    echo "📦 安装 uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "✅ uv 版本: $(uv --version)"

# 同步项目依赖
echo "📚 同步项目依赖..."
uv sync

# 创建必要的目录
echo "📁 创建必要目录..."
mkdir -p data storage

# 检查 Python 模块是否可以导入
echo "🔍 验证安装..."
if uv run python -c "import kb.api.main" 2>/dev/null; then
    echo "✅ 项目安装成功！"
else
    echo "❌ 项目安装验证失败"
    exit 1
fi

echo ""
echo "🎉 安装完成！"
echo ""
echo "📖 使用方法："
echo "   启动Web服务: python start_web.py"
echo "   或者: uv run uvicorn src.kb.api.main:app --host 0.0.0.0 --port 8000 --reload"
echo ""
echo "📚 文档地址: http://localhost:8000/docs"
echo ""
