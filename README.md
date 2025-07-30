# RAG Demo

这是一个使用 LlamaIndex 实现的检索增强生成（RAG）演示项目。

## 功能特性

- 📚 **文档索引**: 自动加载和索引文本文档
- 🔍 **智能检索**: 基于向量相似度的语义搜索
- 💬 **问答系统**: 支持自然语言问答
- 💾 **持久化存储**: 索引可保存和重复使用
- 🎨 **美观界面**: 使用 Rich 库提供彩色命令行界面

## 快速开始

### 1. 环境要求

- Python 3.8+
- OpenAI API Key

### 2. 安装依赖

使用 uv 安装项目依赖：

```bash
# 安装 uv (如果还没有安装)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境并安装依赖
uv sync
```

### 3. 配置 API Key

复制环境变量示例文件并配置你的 OpenAI API Key：

```bash
cp .env.example .env
```

编辑 `.env` 文件，添加你的 OpenAI API Key：

```
OPENAI_API_KEY=your_actual_api_key_here
```

### 4. 运行演示

激活虚拟环境并运行：

```bash
# 激活虚拟环境
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 运行交互式问答
python -m src.rag_demo.main

# 或者直接查询
python -m src.rag_demo.main --query "什么是机器学习？"
```

## 使用方法

### 交互式模式

直接运行程序进入交互式问答模式：

```bash
python -m src.rag_demo.main
```

然后输入问题，例如：
- "什么是深度学习？"
- "机器学习有哪些类型？"
- "解释一下神经网络"

### 命令行查询

直接查询而不进入交互模式：

```bash
python -m src.rag_demo.main --query "什么是自然语言处理？"
```

### 重建索引

如果添加了新文档或想重建索引：

```bash
python -m src.rag_demo.main --rebuild
```

## 项目结构

```
rag-demo/
├── src/
│   └── rag_demo/
│       ├── __init__.py
│       └── main.py          # 主要RAG实现
├── data/                    # 文档数据目录（自动创建）
├── storage/                 # 索引存储目录（自动创建）
├── pyproject.toml          # 项目配置
├── .env.example            # 环境变量示例
└── README.md               # 项目说明
```

## 自定义文档

1. 将你的文本文件（.txt格式）放入 `data/` 目录
2. 运行程序时使用 `--rebuild` 参数重建索引
3. 开始查询你的自定义文档

## 命令行选项

```bash
python -m src.rag_demo.main [选项]

选项:
  --data-dir DIR      指定文档数据目录 (默认: data)
  --persist-dir DIR   指定索引存储目录 (默认: storage)
  --query QUESTION    直接查询问题而不进入交互模式
  --rebuild           强制重建索引
  -h, --help          显示帮助信息
```

## 开发

安装开发依赖：

```bash
uv sync --group dev
```

代码格式化：

```bash
black src/
isort src/
```

## 依赖说明

- **llama-index**: 核心RAG框架
- **llama-index-llms-openai**: OpenAI LLM集成
- **llama-index-embeddings-openai**: OpenAI嵌入模型
- **python-dotenv**: 环境变量管理
- **rich**: 美观的命令行界面

## 许可证

MIT License 