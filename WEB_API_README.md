# RAG Demo Web API 服务

本项目已经成功转换为Web API服务，支持并行文档处理和异步查询。

## 🚀 快速开始

### 前置要求

```bash
# 安装 uv (如果尚未安装)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或使用 pip 安装
pip install uv
```

- Elasticsearch 8.x 向量数据库 (默认 http://localhost:9200，可通过 ELASTICSEARCH_URL 覆盖)

### 方式1：使用 uv 直接运行 (推荐)

```bash
# 同步依赖
uv sync

# 启动Web服务
python start_web.py

# 或直接使用 uv run
uv run uvicorn src.kb.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 方式2：传统方式 (兼容)

```bash
# 安装项目
uv pip install -e .

# 使用项目脚本
rag-web
```

### 方式3：Docker部署

```bash
# 构建并启动
docker-compose up --build

# 后台运行
docker-compose up -d --build
```

- 默认 `docker-compose` 会同时启动一个单节点的 Elasticsearch 服务，API 将通过 `http://elasticsearch:9200` 访问它。若你已有自定义集群，可修改环境变量 `ELASTICSEARCH_URL`/`ELASTICSEARCH_INDEX`。

## 📚 API文档

服务启动后访问：
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **健康检查**: http://localhost:8000/api/health

## 🔥 主要功能

### 1. 文档上传和处理
- **支持格式**: PDF、TXT
- **并行处理**: 可配置1-16个工作线程
- **批量上传**: 同时处理多个文件
- **进度追踪**: 实时查看处理状态

```bash
# 示例：上传文档
curl -X POST "http://localhost:8000/api/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@document.pdf" \
  -F "parallel_workers=4"
```

### 2. 智能对话
- **异步查询**: 不阻塞其他请求
- **来源追踪**: 显示答案来源文档
- **会话管理**: 支持会话ID追踪

```bash
# 示例：对话查询
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "什么是机器学习？"}'
```

### 3. 任务管理
- **状态查询**: 查看文档处理进度
- **并发控制**: 智能任务队列管理
- **错误处理**: 详细的错误信息和重试机制

## 📊 API端点总览

| 端点 | 方法 | 功能 | 说明 |
|------|------|------|------|
| `/api/health` | GET | 健康检查 | 服务状态 |
| `/api/status` | GET | 系统状态 | 详细系统信息 |
| `/api/chat` | POST | 对话查询 | RAG问答 |
| `/api/documents/upload` | POST | 文档上传 | 支持多文件并行 |
| `/api/documents` | GET | 文档列表 | 已上传文档 |
| `/api/documents/{id}` | DELETE | 删除文档 | 移除指定文档 |
| `/api/tasks/{task_id}` | GET | 任务状态 | 查看处理进度 |
| `/api/tasks` | GET | 活跃任务 | 所有进行中任务 |
| `/api/index/rebuild` | POST | 重建索引 | 强制重建向量索引 |
| `/api/config` | PUT | 更新配置 | 动态配置调整 |

## 🛠️ 配置选项

### 环境变量
```bash
export EMBEDDING_MODEL="BAAI/bge-small-zh-v1.5"  # Embedding模型
export MAX_WORKERS="4"                            # 最大并行线程
export DATA_DIR="./data"                          # 数据目录
export STORAGE_DIR="./storage"                    # 存储目录
```

### 运行时配置
```bash
# 使用 uv 启动 (推荐)
uv run uvicorn src.kb.api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1 \
  --loop uvloop

# 传统方式
uvicorn src.kb.api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1 \
  --loop uvloop
```

## 🔧 性能优化

### 并行处理配置
- **文档处理**: 支持1-16个并行工作线程
- **多用户并发**: 支持10-20个并发请求
- **智能负载分配**: 根据文件类型自动调整

### 内存管理
- **文件缓存**: 自动缓存已处理文档
- **大文件处理**: 分块处理避免内存溢出
- **资源监控**: 实时监控内存使用

## 🧪 测试

### 运行API测试
```bash
# 确保服务已启动
python start_web.py

# 在另一个终端运行测试
python test_api.py
```

### 测试覆盖
- ✅ 健康检查
- ✅ 系统状态
- ✅ 文档上传
- ✅ 任务状态查询
- ✅ 对话功能
- ✅ 文档管理

## 📈 性能指标

### 处理能力
- **小文件 (< 1MB)**: 1-3秒
- **中等文件 (1-10MB)**: 10-30秒
- **大文件 (10-50MB)**: 30-120秒

### 并发性能
- **单用户**: 4-8个文档并行处理
- **多用户**: 10-20个并发请求
- **响应时间**: API调用 < 100ms

## 🔍 监控和日志

### 系统监控
```bash
# 查看系统状态
curl http://localhost:8000/api/status

# 查看活跃任务
curl http://localhost:8000/api/tasks
```

### 日志级别
- **INFO**: 正常操作信息
- **WARNING**: 非致命错误
- **ERROR**: 严重错误

## 🚨 故障排除

### 常见问题

1. **服务启动失败**
   ```bash
   # 使用 uv 检查依赖
   uv sync
   
   # 或传统方式
   uv pip install -e .
   
   # 检查端口占用
   lsof -i :8000
   
   # 检查 uv 是否安装
   uv --version
   ```

2. **文档处理失败**
   ```bash
   # 检查文件权限
   chmod 644 data/*.pdf
   
   # 查看详细错误
   curl http://localhost:8000/api/tasks/{task_id}
   ```

3. **内存不足**
   ```bash
   # 减少并行线程数
   curl -X PUT http://localhost:8000/api/config \
     -H "Content-Type: application/json" \
     -d '{"max_parallel_workers": 2}'
   ```

## 🎯 后续改进计划

- [ ] 添加用户认证和权限管理
- [ ] 实现分布式部署支持
- [ ] 添加更多文档格式支持
- [ ] 集成监控和告警系统
- [ ] 优化大规模文档处理性能

## 📞 技术支持

如有问题，请检查：
1. API文档 (http://localhost:8000/docs)
2. 健康检查 (http://localhost:8000/api/health)
3. 系统状态 (http://localhost:8000/api/status)
4. 运行测试脚本 (python test_api.py)
