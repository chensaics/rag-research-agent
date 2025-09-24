"""Retriever manager for loading and caching vector store retrievers."""

import hashlib
import json
import os
from contextlib import contextmanager
from typing import Dict, Generator, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStoreRetriever

from src.log_util import logger
from src.shared.configuration_manager import BaseConfiguration
from src.shared.text_encoder import make_text_encoder


@contextmanager
def make_elastic_retriever(
    configuration: BaseConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific elastic index."""
    logger.info("Setting up Elasticsearch retriever")
    from langchain_elasticsearch import ElasticsearchStore

    # Set default connection options
    connection_options = {}

    if configuration.retriever_provider == "elastic-local":
        # Local development configuration
        connection_options = {
            "es_user": os.environ.get("ELASTICSEARCH_USER", "elastic"),
            "es_password": os.environ.get("ELASTICSEARCH_PASSWORD", "escs3789"),
        }
        # 尝试多种连接方式，支持不同的本地环境配置
        es_url = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")
        logger.debug("Using local Elasticsearch configuration")
    else:
        # Cloud/production configuration using API key
        if "ELASTICSEARCH_API_KEY" not in os.environ:
            raise ValueError(
                "ELASTICSEARCH_API_KEY environment variable is required for non-local providers"
            )
        connection_options = {"es_api_key": os.environ["ELASTICSEARCH_API_KEY"]}
        es_url = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")
        logger.debug("Using cloud/production Elasticsearch configuration")

    logger.debug(f"Connecting to Elasticsearch at {es_url}")

    vstore = ElasticsearchStore(
        **connection_options,
        es_url=es_url,
        index_name=os.environ.get("ELASTICSEARCH_INDEX", "index"),
        embedding=embedding_model,
    )

    logger.info("Elasticsearch retriever setup complete")
    yield vstore.as_retriever(search_kwargs=configuration.search_kwargs)


@contextmanager
def make_milvus_retriever(
    configuration: BaseConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific milvus index."""
    logger.info("Setting up Milvus retriever")
    from langchain_milvus import Milvus

    vstore = Milvus(
        embedding_function=embedding_model,
        connection_args={
            "uri": os.environ.get("MILVUS_URI", "http://localhost:19530"),
        },
        collection_name="index",
    )

    logger.info("Milvus retriever setup complete")
    yield vstore.as_retriever(search_kwargs=configuration.search_kwargs)


@contextmanager
def make_mongodb_retriever(
    configuration: BaseConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific MongoDB Atlas index & namespaces."""
    logger.info("Setting up MongoDB retriever")
    from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch

    vstore = MongoDBAtlasVectorSearch.from_connection_string(
        os.environ["MONGODB_URI"],
        namespace="langgraph_retrieval_agent.default",
        embedding=embedding_model,
    )

    logger.info("MongoDB retriever setup complete")
    yield vstore.as_retriever(search_kwargs=configuration.search_kwargs)


class RetrieverManager:
    """A singleton class to manage and cache vector store retrievers."""

    _instance: Optional["RetrieverManager"] = None
    _retrievers: Dict[str, VectorStoreRetriever] = {}

    def __new__(cls) -> "RetrieverManager":
        """Create a singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            logger.debug("[RetrieverManager] >>> instance created")
        return cls._instance

    def _generate_config_key(self, config: RunnableConfig) -> str:
        """Generate a unique key for the configuration."""
        configuration = BaseConfiguration.from_runnable_config(config)
        # Create a dictionary with the relevant config parameters
        config_dict = {
            "retriever_provider": configuration.retriever_provider,
            "embedding_model": configuration.embedding_model,
            "search_kwargs": configuration.search_kwargs,
        }
        # Convert to JSON and hash it to create a unique key
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def get_retriever(self, config: RunnableConfig) -> VectorStoreRetriever:
        """Get a retriever, creating it if necessary."""
        config_key = self._generate_config_key(config)

        if config_key not in self._retrievers:
            logger.info(
                f"[RetrieverManager] >>> Creating new retriever for config key: {config_key}"
            )
            retriever = self._create_retriever(config)
            self._retrievers[config_key] = retriever
        # else:
        #     logger.debug(f"[RetrieverManager] >>> Using cached retriever for config key: {config_key}")

        return self._retrievers[config_key]

    def _create_retriever(self, config: RunnableConfig) -> VectorStoreRetriever:
        """Create a new retriever based on the configuration."""
        configuration = BaseConfiguration.from_runnable_config(config)
        logger.info(
            f"[RetrieverManager] >>> Creating retriever with provider: {configuration.retriever_provider}"
        )

        embedding_model = make_text_encoder(configuration.embedding_model)
        logger.debug(
            f"[RetrieverManager] >>> Using embedding model: {configuration.embedding_model}"
        )

        if configuration.retriever_provider in ("elastic", "elastic-local"):
            logger.debug("[RetrieverManager] >>> Using Elasticsearch retriever")
            with make_elastic_retriever(configuration, embedding_model) as retriever:
                return retriever

        elif configuration.retriever_provider == "milvus":
            logger.debug("[RetrieverManager] >>> Using Milvus retriever")
            with make_milvus_retriever(configuration, embedding_model) as retriever:
                return retriever

        elif configuration.retriever_provider == "mongodb":
            logger.debug("[RetrieverManager] >>> Using MongoDB retriever")
            with make_mongodb_retriever(configuration, embedding_model) as retriever:
                return retriever

        else:
            logger.error(
                f"[RetrieverManager] >>> Unrecognized retriever_provider: {configuration.retriever_provider}"
            )
            raise ValueError(
                "Unrecognized retriever_provider in configuration. "
                f"Expected one of: {', '.join(BaseConfiguration.__annotations__['retriever_provider'].__args__)}\n"
                f"Got: {configuration.retriever_provider}"
            )

    def clear_cache(self) -> None:
        """Clear the retriever cache."""
        logger.info("[RetrieverManager] >>> Clearing retriever cache")
        self._retrievers.clear()


retriever_manager = RetrieverManager()


def make_retriever(config: RunnableConfig) -> VectorStoreRetriever:
    """Create a retriever for the agent, based on the current configuration.

    This function uses a retriever manager to cache and reuse retriever instances
    to avoid repeated setup overhead.
    """
    return retriever_manager.get_retriever(config)
