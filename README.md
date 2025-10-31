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

### 方式2: 一键安装脚本

```bash
# 运行自动安装脚本
chmod +x scripts/install.sh
./scripts/install.sh
```

### 方式3: 使用 Makefile

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
  -H "X-Tenant-ID: default" \
  -F "files=@/path/to/contract.pdf" \
  -F "files=@/path/to/spec.txt" \
  -F "parallel_workers=4" \
  -F "enable_batch_processing=true" \
  -F "priority=normal" \
  -F "callback_url=https://example.com/callback" \
  -F 'document_ids=["contract-2024","spec-1"]'
```

#### 2. 查询对话
```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "什么是机器学习？"}'
```

> `search_params.response_mode` 默认为 `compact`，表示 LlamaIndex 仅返回精简自然语言答案，适合面向终端用户的展示。如果需要更详细的推理过程，可将其改为 `tree_summarize` 或其它模式。

#### 3. 查看文档列表
```bash
curl http://localhost:8000/api/documents
```

#### 4. 检查系统状态
```bash
curl http://localhost:8000/api/status
```

详细 API 文档：[WEB_API_README.md](WEB_API_README.md)

### 模型离线缓存

- 运行过程中会自动将 HuggingFace 模型缓存到 `storage/models/`（可通过 `EMBEDDING_CACHE_DIR` 自定义目录）。
- 将模型手动下载至该目录后，可设置 `EMBEDDING_LOCAL_FILES_ONLY=1` 或在命令中传入 `--embedding-local-files-only`，即可脱离外网使用。
- API 服务会复用缓存目录，避免重复下载。

## 📁 项目结构

```
rag-demo/
├── src/
│   ├── api/                    # FastAPI 应用与路由
│   ├── knowledge/              # 知识服务 (索引、检索、解析)
│   ├── services/               # 对话服务、任务服务等业务编排
│   ├── agents/                 # Agno 智能体运行时（规划中）
│   └── shared/                 # 公共配置与工具
├── scripts/                    # 工具脚本
│   ├── install.sh              # 一键安装脚本
│   └── dev.sh                  # 开发环境设置
├── data/                       # 文档数据目录（自动创建）
├── storage/                    # 本地缓存目录（文件缓存、任务状态）
├── start_web.py                # Web 服务启动脚本
├── test_api.py                 # API 测试脚本
├── Makefile                    # 常用命令快捷方式
├── Dockerfile                  # Docker 配置
├── docker-compose.yml          # Docker Compose 配置
├── pyproject.toml              # 项目配置 (uv)
├── WEB_API_README.md           # Web API 详细文档
└── README.md                   # 项目说明
```

## 自定义文档

1. 调用 `/api/documents/upload` 上传文件（支持 `.txt`、`.pdf`、`.docx`）
2. 通过 `/api/tasks/{task_id}` 查看摄取进度
3. 使用 `/api/chat` 或 `/api/chat/stream` 进行问答

### PDF文档支持

- ✅ **标准PDF**: 使用 pypdf 快速提取
- ✅ **复杂布局PDF**: 自动降级到 pdfplumber 处理
- ✅ **多页文档**: 支持多页PDF，保留页面信息
- ✅ **错误处理**: 自动跳过损坏或扫描版PDF
- ✅ **缓存机制**: 避免重复处理相同文件

## 🔍 日志与监控

- `LOG_LEVEL`：根日志级别，默认 `info`
- `APP_LOG_JSON`：设为 `true` 启用 JSON 行输出
- `APP_REQUEST_LOG_LEVEL`：请求体日志级别（建议 `debug`）
- `APP_REQUEST_LOG_MAX_BYTES`：请求体日志截断长度（默认 4096 字节）
- `APP_REQUEST_LOG_EXCLUDE_PATHS`：逗号分隔的排除路径前缀
- `APP_REQUEST_LOG_PRETTY`：设为 `true` 时美化 JSON 请求体
- `APP_LOG_REDACT_KEYS`：需要脱敏的字段列表（例如 `authorization,token,password`）
- `APP_ACCESS_LOG_LEVEL`：`uvicorn.access` 的日志级别

运行时可通过管理接口调整日志级别：

```bash
curl -X PUT "http://localhost:8000/api/admin/loggers/uvicorn.access?level=DEBUG"
```

上面的命令会即时提高访问日志的详细程度，方便在排障期间观察请求情况。

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

### 上传与查询

```bash
# 1. 上传 PDF 文件
curl -X POST "http://localhost:8000/api/documents/upload" \
  -H "X-Tenant-ID: demo" \
  -F "files=@/path/to/document.pdf"

# 2. 轮询任务状态
curl "http://localhost:8000/api/tasks/<task_id>"

# 3. 发起问答
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: demo" \
  -d '{"message": "文档中提到了什么？"}'
```

### 性能调整

- 上传时设置 `parallel_workers` 控制并行解析线程数（默认 4）。
- 通过 `enable_batch_processing`、`priority` 等参数优化长任务调度。
- Elasticsearch 索引会自动复用缓存，避免重复处理相同文件。

### 批量处理建议

- 分批上传大型文档集，配合任务查询接口追踪进度。
- 将常用文档保存在 `storage/cache`，系统会自动跳过已处理文件。
- 结合 `/api/chat/stream` 获取实时回答片段，提升交互体验。

## 许可证

MIT License 
