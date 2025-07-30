# RAG Demo

这是一个使用 LlamaIndex 实现的检索增强生成（RAG）演示项目。

## 功能特性

- 📚 **文档索引**: 自动加载和索引文本文档
- 🔍 **智能检索**: 基于向量相似度的语义搜索
- 💬 **问答系统**: 支持自然语言问答
- 💾 **持久化存储**: 索引可保存和重复使用
- 🎨 **美观界面**: 使用 Rich 库提供彩色命令行界面
- 🏠 **离线部署**: 使用本地embedding模型，无需外部API调用

## 快速开始

### 1. 环境要求

- Python 3.9+
- OpenAI API Key (仅用于LLM，embedding模型为本地部署)
- 8GB+ RAM (推荐，用于加载embedding模型)

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
cp env.example .env
```

编辑 `.env` 文件，添加你的 OpenAI API Key：

```
OPENAI_API_KEY=your_actual_api_key_here
```

**注意**: 只有LLM需要OpenAI API，embedding模型完全在本地运行！

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

### Embedding模型选择

查看可用的embedding模型：

```bash
python -m src.rag_demo.main --list-models
```

使用不同的embedding模型：

```bash
# 中文优化模型（默认）
python -m src.rag_demo.main --embedding-model BAAI/bge-small-zh-v1.5

# 英文轻量模型
python -m src.rag_demo.main --embedding-model sentence-transformers/all-MiniLM-L6-v2

# 多语言模型
python -m src.rag_demo.main --embedding-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
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
  --data-dir DIR           指定文档数据目录 (默认: data)
  --persist-dir DIR        指定索引存储目录 (默认: storage)
  --query QUESTION         直接查询问题而不进入交互模式
  --rebuild                强制重建索引
  --embedding-model MODEL  指定embedding模型 (默认: BAAI/bge-small-zh-v1.5)
  --list-models            列出推荐的embedding模型
  -h, --help               显示帮助信息
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

- **llama-index-core**: 核心RAG框架
- **llama-index-llms-openai**: OpenAI LLM集成
- **llama-index-embeddings-huggingface**: Hugging Face embedding集成
- **sentence-transformers**: 本地embedding模型库
- **torch**: PyTorch深度学习框架
- **python-dotenv**: 环境变量管理
- **rich**: 美观的命令行界面

## Embedding模型说明

本项目支持多种本地embedding模型：

| 模型名称 | 特点 | 适用场景 |
|---------|------|----------|
| `BAAI/bge-small-zh-v1.5` | 中文优化，轻量级 | 中文文档，快速部署 |
| `BAAI/bge-base-zh-v1.5` | 中文优化，更高精度 | 中文文档，质量优先 |
| `all-MiniLM-L6-v2` | 英文轻量级 | 英文文档，资源受限 |
| `all-mpnet-base-v2` | 英文高质量 | 英文文档，最佳效果 |
| `paraphrase-multilingual-MiniLM-L12-v2` | 多语言支持 | 混合语言文档 |

**优势**:
- ✅ 完全离线运行，保护数据隐私
- ✅ 无需API费用
- ✅ 响应速度快
- ✅ 支持中英文和多语言

## 许可证

MIT License 