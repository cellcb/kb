#!/bin/bash
"""
RAG Demo 开发环境脚本 (使用 uv)
快速设置开发环境
"""

set -e

echo "🛠️  RAG Demo 开发环境设置"
echo "========================="

# 同步开发依赖
echo "📚 同步开发依赖..."
uv sync --dev

# 安装项目（可编辑模式）
echo "🔧 安装项目（开发模式）..."
uv pip install -e .

# 检查代码格式
echo "🎨 检查代码格式..."
if uv run black --check src/; then
    echo "✅ 代码格式检查通过"
else
    echo "❌ 代码格式需要修复，运行: uv run black src/"
fi

# 检查导入排序
echo "📦 检查导入排序..."
if uv run isort --check-only src/; then
    echo "✅ 导入排序检查通过"
else
    echo "❌ 导入排序需要修复，运行: uv run isort src/"
fi

# 运行测试 (如果有)
echo "🧪 运行测试..."
if [ -d "tests" ]; then
    uv run pytest tests/
else
    echo "⚠️  没有找到测试目录"
fi

echo ""
echo "🎉 开发环境设置完成！"
echo ""
echo "🔧 开发命令："
echo "   格式化代码: uv run black src/"
echo "   排序导入: uv run isort src/"
echo "   启动服务: python start_web.py"
echo "   运行测试: uv run python test_api.py"
echo ""
