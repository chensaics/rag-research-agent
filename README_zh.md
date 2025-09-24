# LangGraph RAG 研究代理模板

这是一个入门项目，可帮助您使用 [LangGraph](https://github.com/langchain-ai/langgraph) 在 [LangGraph Studio](https://github.com/langchain-ai/langgraph-studio) 中开发 RAG 研究代理。

![Graph](./assets/flowchart.png)

## 功能介绍

本项目提供了三个可运行的图，实现在 `src/graphs/` 下：

* 一个"索引"图 (`src/graphs/index_graph.py`)
* 一个"主"图 (`src/graphs/main_graph.py`)
* 一个"研究员"子图（主图的一部分）(`src/graphs/researcher_graph.py`)

索引图接收文档对象并将其索引。

```json
[{ "page_content": "RAG Research Agent是一种结合检索增强生成(RAG)技术与智能Agent能力的研究型智能体系统。它不仅具备基础的检索与生成能力，还融合了自我反思、工具链式调用及多智能体协同等高级特性，突破了传统大模型'仅会聊天、检索'的局限。" }]
```

## 图关系

系统中的三个图具有以下关系：

1. **索引图**：
   - 用于将文档索引到向量存储中的独立图
   - 不直接连接到其他图
   - 用于填充其他图查询的向量存储

2. **主图**：
   - 主对话代理
   - 包含研究员图作为子图
   - 使用索引图填充的向量存储

3. **研究员图**：
   - 主图的子图
   - 由主图中的 'conduct_research' 节点调用
   - 从同一向量存储生成查询和检索文档

数据流：
```
索引图 --> 向量存储 <-- 研究员图 <-- 主图
```


## 设置

### 设置向量存储

此模板支持以下向量存储：

#### Elasticsearch（本地）

Elasticsearch 是一个分布式、RESTful 搜索引擎，针对速度和相关性进行了优化。对于本地开发，我们建议使用官方 Docker 镜像。

1. 拉取 Elasticsearch Docker 镜像：
   ```bash
   docker pull docker.elastic.co/elasticsearch/elasticsearch:9.1.4
   ```

2. 在 Docker 容器中启动 Elasticsearch：
   ```bash
   docker run \
     -p 127.0.0.1:9200:9200 \
     -d \
     --name elasticsearch \
     -e ELASTIC_PASSWORD=your_password \
     -e "discovery.type=single-node" \
     -e "xpack.security.http.ssl.enabled=false" \
     -e "xpack.license.self_generated.type=trial" \
     docker.elastic.co/elasticsearch/elasticsearch:9.1.4
   ```

3. 设置环境：
   - 如果还没有，请在项目根目录中创建一个 `.env` 文件。
   - 将 Elasticsearch 配置添加到您的 `.env` 文件中：

   ```
   ELASTICSEARCH_USER=elastic
   ELASTICSEARCH_PASSWORD=your_password
   ELASTICSEARCH_URL=http://localhost:9200
   ```

#### MongoDB Atlas

MongoDB Atlas 是一个完全托管的云数据库，包含用于 AI 驱动应用程序的向量搜索功能。

1. 创建一个免费的 Atlas 集群：
- 访问 [MongoDB Atlas 网站](https://www.mongodb.com/cloud/atlas/register) 并注册一个免费账户。
- 登录后，按照屏幕上的指示创建一个免费集群。

2. 创建向量搜索索引
- 按照 [Mongo 文档](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-type/) 中的说明操作
- 默认情况下，我们使用集合 `langgraph_retrieval_agent.default` - 在那里创建索引
- 为路径 `user_id` 添加索引过滤器
- **重要**：创建索引时选择 Atlas Vector Search 而不是 Atlas Search
您的最终 JSON 编辑器配置应如下所示：

```json
{
  "fields": [
    {
      "numDimensions": 1024,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    }
  ]
}
```

如果选择不同的嵌入模型，确切的 numDimensions 可能会有所不同。

2. 设置环境：
- 在 Atlas 仪表板中，单击集群的"连接"。
- 选择"连接您的应用程序"并复制提供的连接字符串。
- 如果还没有，请在项目根目录中创建一个 `.env` 文件。
- 将您的 MongoDB Atlas 连接字符串添加到 `.env` 文件中：

```
MONGODB_URI="mongodb+srv://username:password@your-cluster-url.mongodb.net/?retryWrites=true&w=majority&appName=your-cluster-name"
```

将 `username`、`password`、`your-cluster-url` 和 `your-cluster-name` 替换为您的实际凭据和集群信息。

#### Milvus

Milvus 是一个高性能的开源向量数据库，专为 AI 应用程序和相似性搜索而设计。

1. 使用 Docker 设置 Milvus（用于开发）或使用托管的 Milvus 服务：
   - 对于 Docker 设置，请按照 [官方 Milvus 安装指南](https://milvus.io/docs/install_standalone-docker.md) 操作
   - 对于 Zilliz Cloud（托管的 Milvus 服务），请在 [Zilliz Cloud](https://zilliz.com/cloud) 注册

2. 一旦您的 Milvus 实例运行，请将连接 URI 添加到您的 `.env` 文件中：

```
MILVUS_URI=your_milvus_uri
```

对于本地开发，这通常是 `http://localhost:19530`。

### 设置模型

`llm_model`、`embedding_model` 的默认值如下所示：

```yaml
llm_model: ollama/qwen3:4b
embedding_model: ollama/bge-m3:latest
```

按照以下说明进行设置，或选择其他选项。

#### OpenAI

要使用 OpenAI 的聊天模型：

1. 注册一个 [OpenAI API 密钥](https://platform.openai.com/signup)（如果还没有）。
2. 获取 API 密钥后，将其添加到您的 `.env` 文件中：

```
OPENAI_API_KEY=your-api-key
```

#### Ollama（本地模型）

要通过 Ollama 使用本地模型：

1. 安装 [Ollama](https://ollama.com/download)
2. 拉取您要使用的模型：
   ```bash
   ollama pull qwen3:4b
   ollama pull bge-m3:latest
   ```
3. 本地模型不需要 API 密钥。

### 安装依赖

```bash
pip install -e .
```

对于开发，还请安装开发依赖：

```bash
pip install -e ".[dev]"
```

## 使用方法

### 运行示例

要运行模板中的示例，请使用以下命令：

```bash
python main.py
```

### 可视化图

要可视化模板中的图，请运行：

```bash
python src/visualize_graphs.py
```

这将在终端中显示图，并将它们保存为 `graphs/` 目录中的 PNG 文件。

### 运行测试

要运行测试，请使用以下命令：

```bash
make test
```

要在监视模式下运行测试：

```bash
make test_watch
```

要运行带性能分析的测试：

```bash
make test_profile
```

## 开发

### 格式化

要格式化代码，请使用：

```bash
make format
```

### 代码检查

要进行代码检查，请使用：

```bash
make lint
```

## 项目结构

```
.
├── src
│   ├── crud
│   │   ├── __init__.py
│   │   ├── elasticsearch_crud_manager.py
│   │   ├── milvus_crud_manager.py
│   │   ├── mongodb_crud_manager.py
│   │   └── vector_store_crud_manager.py
│   ├── graphs
│   │   ├── __init__.py
│   │   ├── index_graph.py
│   │   ├── main_graph.py
│   │   └── researcher_graph.py
│   ├── shared
│   │   ├── __init__.py
│   │   ├── configuration_manager.py
│   │   ├── model_manager.py
│   │   ├── prompts.py
│   │   ├── retrieval_manager.py
│   │   ├── state.py
│   │   ├── text_encoder.py
│   │   └── utils.py
│   ├── __init__.py
│   ├── log_util.py
│   ├── sample_docs.json
│   └── visualize_graphs.py
├── tests
│   └── test_crud.py
├── Makefile
├── README.md
├── langgraph.json
├── main.py
└── pyproject.toml
```