# Intelligence Query — 智能知识库检索系统

基于 RAG（检索增强生成）的企业级知识库问答系统，支持多模态文档解析、向量/图谱混合检索、流式答案生成和多轮对话。

## 架构总览

```
┌─────────────────────────────────────────────────────┐
│                    前端 (chat.html)                  │
│              流式 SSE / 非流式 两种交互模式            │
└─────────────────────┬───────────────────────────────┘
                      │ FastAPI (main.py)
          ┌───────────┴───────────┐
          ▼                       ▼
┌─ 导入流程 ──────────┐   ┌─ 查询流程 ──────────┐
│ POST /upload        │   │ POST /query         │
│   PDF → MD → 切分   │   │   商品确认           │
│   → 向量化 → 入库   │   │   ├─ 向量检索        │
│   → 知识图谱构建     │   │   ├─ HyDE 检索      │
└─────────────────────┘   │   ├─ 知识图谱查询    │
                          │   ├─ 网络搜索 (MCP)  │
                          │   ├─ RRF 融合        │
                          │   ├─ Rerank 重排     │
                          │   └─ LLM 答案生成    │
                          └─────────────────────┘
          │                       │
          ▼                       ▼
┌─────────────────────────────────────────────────────┐
│                    数据层                            │
│  Milvus (向量)  Neo4j (图谱)  MongoDB (会话)  MinIO  │
└─────────────────────────────────────────────────────┘
```

## 技术栈

| 类别 | 技术 |
|------|------|
| Web 框架 | FastAPI + Uvicorn |
| 工作流编排 | LangGraph |
| 向量数据库 | Milvus 2.4+ (BGE-M3 混合检索) |
| 图数据库 | Neo4j |
| 文档数据库 | MongoDB (对话历史) |
| 对象存储 | MinIO |
| LLM | DashScope (Qwen) |
| 网络搜索 | OpenAI Agents SDK + MCP |
| Embedding | BGE-M3 (dense + sparse) |
| Reranker | BGE-Reranker-v2-m3 |
| 前端 | 原生 HTML/CSS/JS (SSE + EventSource) |

## 快速开始

### 0. 前置条件

- **Python 3.10+**
- **Docker & Docker Compose** — 用于运行 MongoDB、Neo4j、Milvus、MinIO 等基础设施
  - [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
  - [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/)
  - [Docker Engine for Linux](https://docs.docker.com/engine/install/)

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/Yxx66/Product-QA-System.git
cd Product-QA-System

# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 2. Docker 部署基础设施

项目需要以下基础设施服务，均已通过 `docker-compose.yml` 编排：

| 服务 | 用途 | 端口 |
|------|------|------|
| **MongoDB** | 对话历史存储 | 27017 |
| **Neo4j** | 知识图谱 | 7474 (HTTP) / 7687 (Bolt) |
| **Milvus** | 向量检索 | 19530 / 9091 |
| **Etcd** | Milvus 元数据协调 | 2379 |
| **MinIO** | 文件对象存储 | 9000 (API) / 9001 (Console) |
| **Attu** | Milvus 可视化管理 (可选) | 8001 |

#### 启动所有服务

```bash
# 启动所有基础设施（后台运行）
docker-compose up -d

# 仅启动指定服务
docker-compose up -d mongodb neo4j minio

# 查看服务运行状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 停止所有服务
docker-compose down

# 停止并清空数据（危险操作）
docker-compose down -v
```

#### MongoDB

```bash
# 默认连接信息
URL:      mongodb://localhost:27017
用户:     admin
密码:     admin123

# 进入 MongoDB Shell
docker exec -it iq-mongodb mongosh -u admin -p admin123
```

#### Neo4j

```bash
# 默认连接信息
Bolt URI: bolt://localhost:7687
HTTP:     http://localhost:7474
用户:     neo4j
密码:     neo4j123456

# 浏览器访问 Neo4j Browser
http://localhost:7474
```

#### MinIO

```bash
# 默认连接信息
API:      http://localhost:9000
Console:  http://localhost:9001
用户:     minioadmin
密码:     minioadmin123

# 浏览器访问 MinIO 管理控制台
http://localhost:9001
```

#### Milvus

```bash
# 默认连接信息
URL:      http://localhost:19530

# Attu 可视化管理界面
http://localhost:8001
```

> **注意**：首次启动 Milvus 会从 Docker Hub 拉取约 1GB 镜像，请确保网络畅通。Milvus 依赖 etcd 和内部 MinIO，会自动一并启动。

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key、数据库地址等
```

### 4. 启动服务

```bash
python main.py
```

访问：
- 导入页面：http://localhost:8000/
- 查询页面：http://localhost:8000/chat

## API 文档

### 导入侧

| 端点 | 方法 | 说明 |
|------|------|------|
| `/upload` | POST | 上传文件，启动导入流水线 |
| `/status/{task_id}` | GET | 查询导入进度 |

### 查询侧

| 端点 | 方法 | 说明 |
|------|------|------|
| `/query` | POST | 提交查询（支持流式/非流式） |
| `/stream/{session_id}` | GET | SSE 流式实时推送 |
| `/history/{session_id}` | GET | 获取历史对话 |
| `/history/{session_id}` | DELETE | 清空历史对话 |
| `/sessions` | GET | 列出所有历史会话 |

## 项目结构

```
intelligence_query/
├── main.py                       # FastAPI 应用入口
├── knowledge/
│   ├── import_file_router.py     # 导入侧路由
│   ├── api/
│   │   └── query_router.py       # 查询侧路由（4个端点）
│   ├── core/
│   │   └── deps.py               # 依赖注入
│   ├── schema/                   # Pydantic 数据模型
│   ├── services/                 # 业务逻辑层
│   │   ├── query_service.py
│   │   ├── file_import_service.py
│   │   └── task_service.py
│   ├── tools/                    # 工具层
│   │   ├── sse_utils.py          # SSE 事件队列
│   │   ├── task_utils.py         # 任务状态追踪
│   │   ├── mongo_history_utils.py # MongoDB 历史管理
│   │   ├── llm_client.py         # LLM 客户端
│   │   ├── BGE3_client.py        # BGE-M3 Embedding
│   │   ├── milvus_client.py      # Milvus 客户端
│   │   └── ...
│   ├── processor/
│   │   ├── import_process/       # 导入流水线（LangGraph）
│   │   └── query_process/        # 查询流水线（LangGraph）
│   │       ├── nodes/            # 8 个查询节点
│   │       ├── state.py          # 状态定义
│   │       └── main_graph.py     # 图编排
│   └── front/
│       ├── chat.html             # 查询聊天界面
│       └── import.html           # 导入界面
├── docker-compose.yml            # 基础设施编排
├── .env.example                  # 环境变量模板
├── requirements.txt
└── .gitignore
```

## 核心设计

- **双模式查询**：流式（SSE 逐字推送）和非流式（同步等待），一个 `is_stream` 参数切换
- **Queue 解耦**：后台线程 push → `queue.Queue` ← SSE Generator pull，生产消费分离
- **职责分离**：Router → Service → Processor 三层架构，导入/查询两侧完全对称
- **历史会话**：MongoDB 持久化多轮对话上下文，支持指代消解（"它"/"那"→具体商品名）
- **item_names 回填**：第二轮确认商品名后自动回填旧消息，带数据库层面 `$or` 三次防御
- **进度双保险**：`FINAL` 事件正常收尾 + `progress(completed)` 兜底防止前端卡死
- **安全规范**：所有配置通过环境变量注入，无硬编码密钥/密码/IP

## License

MIT
