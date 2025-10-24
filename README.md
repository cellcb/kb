# RAG Demo

这是一个使用 LlamaIndex 实现的检索增强生成（RAG）演示项目，支持**命令行界面**和**Web API服务**两种使用方式。

👉 想快速了解项目中的智能体角色与职责？请查看 [AGENTS.md](AGENTS.md)。

## 🌟 功能特性

### 核心功能
- 📚 **多格式文档支持**: 支持 TXT 和 PDF 文档的自动加载和索引
- 🔍 **智能检索**: 基于向量相似度的语义搜索
- 💬 **问答系统**: 支持自然语言问答
- 💾 **持久化存储**: 索引可保存和重复使用
- 🏠 **离线部署**: 使用本地embedding模型，无需外部API调用

### 性能特性
- ⚡ **并行处理**: 多文档并行处理，支持1-16个工作线程
- 💾 **智能缓存**: 避免重复处理相同文件
- 📄 **PDF智能处理**: 自动选择最佳PDF文本提取策略
- 📊 **详细统计**: 显示处理进度和性能指标

### 界面支持
- 🎨 **命令行界面**: 使用 Rich 库提供彩色交互式界面
- 🌐 **Web API服务**: FastAPI驱动的RESTful API服务
- 📖 **自动文档**: Swagger UI 和 ReDoc 自动生成API文档

## 🚀 快速开始

### 方式1: Web API 服务 (推荐)

```bash
# 1. 安装 uv (如果还没有安装)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 安装项目依赖
uv sync

# 3. 启动 Web API 服务
python start_web.py
```

访问 http://localhost:8000/docs 查看 API 文档

### 方式2: 命令行界面

```bash
# 安装依赖 (如果还没有)
uv sync

# 运行交互式问答
uv run python src/kb/main.py

# 或者直接查询
uv run python src/kb/main.py --query "什么是机器学习？"
```

### 方式3: 一键安装脚本

```bash
# 运行自动安装脚本
chmod +x scripts/install.sh
./scripts/install.sh
```

### 方式4: 使用 Makefile

```bash
# 查看所有可用命令
make help

# 安装依赖
make install

# 启动Web服务
make start

# 运行测试
make test
```

### 准备 Elasticsearch 向量存储

- 默认连接地址为 `http://localhost:9200`，可通过环境变量 `ELASTICSEARCH_URL` 调整。
- 如果还未运行 Elasticsearch，可以使用项目自带的 Docker Compose 服务：

```bash
# 启动 API + Elasticsearch
docker-compose up -d

# 或只启动 Elasticsearch（供本地开发/调试）
docker compose up -d elasticsearch
```

- 如果 API 在容器中运行，需要访问宿主机上的 Elasticsearch，可将 `ELASTICSEARCH_URL` 设置为 `http://host.docker.internal:9200`（macOS/Windows）或在 Linux 上使用自定义网络与主机共享端口。

- 已有自定义安装时，仅需保证 API 容器或本地进程能够访问上述地址，并根据需要配置用户名密码。

## 环境要求

- Python 3.9+
- 8GB+ RAM (推荐，用于加载embedding模型)
- Elasticsearch 8.x (向量存储，默认访问 `http://localhost:9200`)
- **注意**: 已使用 DeepSeek API，无需配置 OpenAI API Key

## 📚 使用方法

### Web API 方式 (推荐)

#### 1. 上传文档
```bash
curl -X POST "http://localhost:8000/api/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@document.pdf" \
  -F "parallel_workers=4"
```

#### 2. 查询对话
```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "什么是机器学习？"}'
```

#### 3. 查看文档列表
```bash
curl http://localhost:8000/api/documents
```

#### 4. 检查系统状态
```bash
curl http://localhost:8000/api/status
```

详细 API 文档：[WEB_API_README.md](WEB_API_README.md)

### 命令行方式

#### 交互式模式
```bash
uv run python src/kb/main.py
```

然后输入问题，例如：
- "什么是深度学习？"
- "机器学习有哪些类型？"
- "解释一下神经网络"

#### 直接查询
```bash
uv run python src/kb/main.py --query "什么是自然语言处理？"
```

### 重建索引

如果添加了新文档或想重建索引：

```bash
python -m src.kb.main --rebuild
```

### Embedding模型选择

查看可用的embedding模型：

```bash
python -m src.kb.main --list-models
```

使用不同的embedding模型：

```bash
# 中文优化模型（默认）
python -m src.kb.main --embedding-model BAAI/bge-small-zh-v1.5

# 英文轻量模型
python -m src.kb.main --embedding-model sentence-transformers/all-MiniLM-L6-v2

# 多语言模型
python -m src.kb.main --embedding-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

### 模型离线缓存

- 运行过程中会自动将 HuggingFace 模型缓存到 `storage/models/`（可通过 `EMBEDDING_CACHE_DIR` 自定义目录）。
- 将模型手动下载至该目录后，可设置 `EMBEDDING_LOCAL_FILES_ONLY=1` 或在命令中传入 `--embedding-local-files-only`，即可脱离外网使用。
- CLI 及 Web API 都会复用缓存目录，避免重复下载。

## 📁 项目结构

```
rag-demo/
├── src/
│   └── kb/
│       ├── __init__.py
│       ├── main.py              # 命令行界面入口
│       ├── api/                 # Web API 服务
│       │   ├── main.py         # FastAPI 应用入口
│       │   ├── routes/         # API 路由
│       │   └── models/         # Pydantic 模型
│       └── core/               # 核心业务逻辑
│           ├── rag_engine.py   # 异步 RAG 引擎
│           └── task_manager.py # 任务管理器
├── scripts/                    # 工具脚本
│   ├── install.sh             # 一键安装脚本
│   └── dev.sh                 # 开发环境设置
├── data/                       # 文档数据目录（自动创建）
├── storage/                    # 本地缓存目录（文件缓存、任务状态）
├── start_web.py               # Web 服务启动脚本
├── test_api.py                # API 测试脚本
├── Makefile                   # 常用命令快捷方式
├── Dockerfile                 # Docker 配置
├── docker-compose.yml         # Docker Compose 配置
├── pyproject.toml             # 项目配置 (uv)
├── WEB_API_README.md          # Web API 详细文档
└── README.md                  # 项目说明
```

## 自定义文档

1. 将你的文档文件放入 `data/` 目录
   - 支持格式：`.txt`、`.pdf`
   - PDF文件会自动进行文本提取
   - 系统会智能检测文件类型
2. 运行程序时使用 `--rebuild` 参数重建索引
3. 开始查询你的自定义文档

### PDF文档支持

- ✅ **标准PDF**: 使用 pypdf 快速提取
- ✅ **复杂布局PDF**: 自动降级到 pdfplumber 处理
- ✅ **多页文档**: 支持多页PDF，保留页面信息
- ✅ **错误处理**: 自动跳过损坏或扫描版PDF
- ✅ **缓存机制**: 避免重复处理相同文件

## 命令行选项

```bash
python -m src.kb.main [选项]

选项:
  --data-dir DIR           指定文档数据目录 (默认: data)
  --persist-dir DIR        指定本地缓存目录 (默认: storage)
  --query QUESTION         直接查询问题而不进入交互模式
  --rebuild                强制重建索引（会重置 Elasticsearch 索引）
  --embedding-model MODEL  指定embedding模型 (默认: BAAI/bge-small-zh-v1.5)
  --list-models            列出推荐的embedding模型
  --test-embedding         测试embedding模型功能
  --disable-parallel       禁用并行处理，使用串行模式
  --max-workers N          设置并行处理的最大工作线程数 (默认: 2)
  --es-url URL             Elasticsearch 服务地址（默认读取 ELASTICSEARCH_URL）
  --es-index NAME          Elasticsearch 索引名称 (默认: kb-documents)
  --es-user USER           Elasticsearch 基本认证用户名
  --es-password PASS       Elasticsearch 基本认证密码
  -h, --help               显示帮助信息
```

## 🛠️ 开发

### 开发环境设置
```bash
# 一键设置开发环境
chmod +x scripts/dev.sh
./scripts/dev.sh

# 或手动设置
uv sync --dev
uv pip install -e .
```

### 使用 Makefile 开发
```bash
# 查看所有命令
make help

# 安装开发依赖
make install-dev

# 代码格式化
make format

# 检查代码格式
make lint

# 清理缓存
make clean
```

### 手动开发命令
```bash
# 代码格式化
uv run black src/
uv run isort src/

# 代码检查
uv run black --check src/
uv run isort --check-only src/

# 运行 API 测试
uv run python test_api.py
```

## 依赖说明

- **llama-index-core**: 核心RAG框架
- **llama-index-llms-openai**: OpenAI LLM集成
- **llama-index-embeddings-huggingface**: Hugging Face embedding集成
- **llama-index-vector-stores-elasticsearch**: Elasticsearch 向量存储适配器
- **elasticsearch**: 官方 Elasticsearch Python 客户端
- **sentence-transformers**: 本地embedding模型库
- **torch**: PyTorch深度学习框架
- **python-dotenv**: 环境变量管理
- **rich**: 美观的命令行界面
- **pypdf**: PDF文档快速文本提取
- **pdfplumber**: 复杂PDF布局处理
- **python-magic**: 智能文件类型检测（可选）

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

## PDF支持使用示例

### 基本使用

```bash
# 将PDF文件放入data目录
cp /path/to/your/document.pdf data/

# 重建索引以包含PDF文档
uv run python src/kb/main.py --rebuild

# 查询PDF内容
uv run python src/kb/main.py --query "文档中提到了什么？"
```

### 性能优化选项

```bash
# 使用更多线程并行处理多个PDF文件
uv run python src/kb/main.py --rebuild --max-workers 4

# 禁用并行处理（适用于内存受限环境）
uv run python src/kb/main.py --rebuild --disable-parallel
```

### 处理大量文档

当处理大量PDF文档时，系统会：
- 📊 显示详细的处理进度
- ⚡ 自动并行处理多个文件
- 💾 缓存已处理的文件避免重复工作
- 📈 提供性能统计信息

**示例输出：**
```
发现 20 个文件，开始处理...
使用并行处理 (最大 4 个工作线程)
✓ pypdf成功提取 2,543 字符
✓ 使用缓存内容: document2.pdf (1,234 字符)
  完成 document3.pdf 100% 20/20 0:00:15

文档处理统计
┏━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 项目       ┃   数量 ┃ 说明                   ┃
┡━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 成功处理   │     18 │ ✓ 已加载到向量数据库   │
│ PDF文件    │     15 │ 通过pypdf/pdfplumber │
│ 文本文件   │      3 │ 直接读取              │
│ 跳过文件   │      1 │ 空文件或不支持格式     │
│ 失败文件   │      1 │ ❌ 处理过程中出错     │
│ 总字符数   │ 45,821 │ 提取的文本总长度       │
│ 处理时间   │ 15.2秒 │ 并行处理              │
└────────────┴────────┴────────────────────────┘
📊 处理速度: 1.2 文件/秒, 3,014 字符/秒
💾 缓存中有 10 个文件，下次处理将更快
```

## 许可证

MIT License 
