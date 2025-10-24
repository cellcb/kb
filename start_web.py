#!/usr/bin/env python3
"""
RAG Demo Web API 启动脚本
快速启动Web服务的便捷脚本 (使用 uv)
"""

import os
import sys
import subprocess

def main():
    """启动Web API服务"""
    print("🚀 启动RAG Demo Web API服务 (使用 uv)...")
    print("📖 API文档地址: http://localhost:8000/docs")
    print("🔍 健康检查: http://localhost:8000/api/health")
    print("💬 对话API: http://localhost:8000/api/chat")
    print("\n按 Ctrl+C 停止服务\n")
    
    try:
        # 使用 uv run 启动服务
        subprocess.run([
            "uv", "run", "uvicorn",
            "api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
            "--log-level", "info"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 服务已停止")
    except subprocess.CalledProcessError as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ uv 未安装，请先安装 uv:")
        print("   curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
