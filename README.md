# LangGraph RAG Research Agent Template

This is a starter project to help you get started with developing a RAG research agent using [LangGraph](https://github.com/langchain-ai/langgraph) in [LangGraph Studio](https://github.com/langchain-ai/langgraph-studio).

![Graph](./assets/flowchart.png)

## What it does

This project provides three runnable graphs implemented under `src/graphs/`:

* an "index" graph (`src/graphs/index_graph.py`)
* a "main" graph (`src/graphs/main_graph.py`)
* a "researcher" subgraph (part of the main graph) (`src/graphs/researcher_graph.py`)

The index graph takes in document objects and indexes them.

```json
[{ "page_content": "RAG Research Agent是一种结合检索增强生成(RAG)技术与智能Agent能力的研究型智能体系统。它不仅具备基础的检索与生成能力，还融合了自我反思、工具链式调用及多智能体协同等高级特性，突破了传统大模型'仅会聊天、检索'的局限。" }]
```

## Graph Relationships

The three graphs in this system have the following relationships:

1. **Index Graph**:
   - Independent graph for indexing documents into a vector store
   - Not directly connected to other graphs
   - Used to populate the vector store that other graphs query

2. **Main Graph**:
   - Main conversational agent
   - Contains the Researcher Graph as a subgraph
   - Uses the vector store populated by the Index Graph

3. **Researcher Graph**:
   - Subgraph of the Main Graph
   - Invoked by the 'conduct_research' node in the Main Graph
   - Generates queries and retrieves documents from the same vector store

Data Flow:
```
Index Graph --> Vector Store <-- Researcher Graph <-- Main Graph
```


## Setup

### Setup Vector Store

This template supports the following vector stores:

#### Elasticsearch (Local)

Elasticsearch is a distributed, RESTful search engine optimized for speed and relevance. For local development, we recommend using the official Docker image.

1. Pull the Elasticsearch Docker image:
   ```bash
   docker pull docker.elastic.co/elasticsearch/elasticsearch:9.1.4
   ```

2. Start Elasticsearch in a Docker container:
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

3. Set up your environment:
   - Create a `.env` file in your project root if you haven't already.
   - Add the Elasticsearch configuration to your `.env` file:

   ```
   ELASTICSEARCH_USER=elastic
   ELASTICSEARCH_PASSWORD=your_password
   ELASTICSEARCH_URL=http://localhost:9200
   ```

#### MongoDB Atlas

MongoDB Atlas is a fully-managed cloud database that includes vector search capabilities for AI-powered applications.

1. Create a free Atlas cluster:
- Go to the [MongoDB Atlas website](https://www.mongodb.com/cloud/atlas/register) and sign up for a free account.
- After logging in, create a free cluster by following the on-screen instructions.

2. Create a vector search index
- Follow the instructions at [the Mongo docs](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-type/)
- By default, we use the collection `langgraph_retrieval_agent.default` - create the index there
- Add an indexed filter for path `user_id`
- **IMPORTANT**: select Atlas Vector Search NOT Atlas Search when creating the index
Your final JSON editor configuration should look something like the following:

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

The exact numDimensions may differ if you select a different embedding model.

2. Set up your environment:
- In the Atlas dashboard, click on "Connect" for your cluster.
- Choose "Connect your application" and copy the provided connection string.
- Create a `.env` file in your project root if you haven't already.
- Add your MongoDB Atlas connection string to the `.env` file:

```
MONGODB_URI="mongodb+srv://username:password@your-cluster-url.mongodb.net/?retryWrites=true&w=majority&appName=your-cluster-name"
```

Replace `username`, `password`, `your-cluster-url`, and `your-cluster-name` with your actual credentials and cluster information.

#### Milvus

Milvus is a high-performance, open-source vector database designed for AI applications and similarity search.

1. Set up Milvus using Docker (for development) or use a managed Milvus service:
   - For Docker setup, follow the [official Milvus installation guide](https://milvus.io/docs/install_standalone-docker.md)
   - For Zilliz Cloud (managed Milvus service), sign up at [Zilliz Cloud](https://zilliz.com/cloud)

2. Once you have your Milvus instance running, add the connection URI to your `.env` file:

```
MILVUS_URI=your_milvus_uri
```

For local development, this would typically be `http://localhost:19530`.

### Setup Model

The defaults values for `llm_model`, `embedding_model` are shown below:

```yaml
llm_model: ollama/qwen3:4b
embedding_model: ollama/bge-m3:latest
```

Follow the instructions below to get set up, or pick one of the additional options.

#### OpenAI

To use OpenAI's chat models:

1. Sign up for an [OpenAI API key](https://platform.openai.com/signup).
2. Once you have your API key, add it to your `.env` file:

```
OPENAI_API_KEY=your-api-key
```

#### Ollama (Local Models)

To use local models via Ollama:

1. Install [Ollama](https://ollama.com/download)
2. Pull the models you want to use:
   ```bash
   ollama pull qwen3:4b
   ollama pull bge-m3:latest
   ```
3. No API key is needed for local models.

### Install dependencies

```bash
pip install -e .
```

For development, also install dev dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

### Running examples

To run the examples in the template, use the following command:

```bash
python main.py
```

### Visualizing graphs

To visualize the graphs in the template, run:

```bash
python src/visualize_graphs.py
```

This will display the graphs in your terminal and save them as PNG files in the `graphs/` directory.

### Running tests

To run the tests, use the following command:

```bash
make test
```

To run tests in watch mode:

```bash
make test_watch
```

To run tests with profiling:

```bash
make test_profile
```

## Development

### Formatting

To format the code, use:

```bash
make format
```

### Linting

To lint the code, use:

```bash
make lint
```

## Project Structure

```
.
├── src
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
│   │   ├── retrieval.py
│   │   ├── retrieval_manager.py
│   │   ├── state.py
│   │   ├── text_encoder.py
│   │   └── utils.py
│   ├── __init__.py
│   ├── log_util.py
│   ├── sample_docs.json
│   └── visualize_graphs.py
├── tests
│   ├── integration_tests
│   │   ├── __init__.py
│   │   └── test_graph.py
│   ├── unit_tests
│   │   ├── __init__.py
│   │   └── test_configuration.py
│   └── __init__.py
├── Makefile
├── README.md
├── langgraph.json
├── main.py
└── pyproject.toml
```