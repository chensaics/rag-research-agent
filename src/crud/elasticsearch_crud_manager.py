#!/usr/bin/env python
"""@Author:     sai.chen
@FileName:   elasticsearch_crud_manager.py
@Date:       2025/09/24
@Description:
-----------------------------------------------------------
Elasticsearch CRUD Manager
-----------------------------------------------------------
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict, Generator, List

from elasticsearch import Elasticsearch
from langchain_core.documents import Document
from langchain_elasticsearch import ElasticsearchStore

from src.shared.configuration_manager import BaseConfiguration
from src.shared.text_encoder import make_text_encoder


class ElasticsearchCRUDManager:
    """CRUD Manager for Elasticsearch."""

    def __init__(self, configuration: BaseConfiguration):
        """Initialize the CRUD manager with a configuration."""
        self.configuration = configuration
        self.embedding_model = make_text_encoder(configuration.embedding_model)
        self._vector_store = None

    def _get_connection_options(self) -> tuple[Dict[str, Any], str]:
        """Get connection options and URL for Elasticsearch."""
        connection_options = {}
        es_url = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")

        if self.configuration.retriever_provider == "elastic-local":
            # Local development configuration
            connection_options = {
                "es_user": os.environ.get("ELASTICSEARCH_USER", "elastic"),
                "es_password": os.environ.get("ELASTICSEARCH_PASSWORD", "escs3789"),
            }
        else:
            # Cloud/production configuration using API key
            if "ELASTICSEARCH_API_KEY" not in os.environ:
                raise ValueError(
                    "ELASTICSEARCH_API_KEY environment variable is required for non-local providers"
                )
            connection_options = {"es_api_key": os.environ["ELASTICSEARCH_API_KEY"]}

        return connection_options, es_url

    def _get_vector_store(self) -> ElasticsearchStore:
        """Get Elasticsearch vector store."""
        if self._vector_store is None:
            connection_options, es_url = self._get_connection_options()

            self._vector_store = ElasticsearchStore(
                **connection_options,
                es_url=es_url,
                index_name=os.environ.get("ELASTICSEARCH_INDEX", "index"),
                embedding=self.embedding_model,
            )

        return self._vector_store

    @contextmanager
    def _get_es_client(self) -> Generator[Elasticsearch, None, None]:
        """Context manager for Elasticsearch client."""
        connection_options, es_url = self._get_connection_options()

        es_client_args = {}
        if self.configuration.retriever_provider == "elastic-local":
            es_client_args = {
                "http_auth": (
                    connection_options["es_user"],
                    connection_options["es_password"],
                )
            }
        else:
            es_client_args = {"api_key": connection_options["es_api_key"]}

        es_client = Elasticsearch(hosts=[es_url], **es_client_args)
        try:
            yield es_client
        finally:
            es_client.close()

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store.

        Args:
            documents: List of Document objects to add

        Returns:
            List of document IDs
        """
        vector_store = self._get_vector_store()
        return vector_store.add_documents(documents)

    def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents by IDs.

        Args:
            ids: List of document IDs to delete

        Returns:
            True if successful
        """
        vector_store = self._get_vector_store()
        vector_store.delete(ids)
        return True

    def search_documents(self, query: str, k: int = 4) -> List[Document]:
        """Search for documents similar to the query.

        Args:
            query: Query text
            k: Number of documents to return

        Returns:
            List of similar documents
        """
        vector_store = self._get_vector_store()
        return vector_store.similarity_search(query, k=k)

    def count_documents(self) -> int:
        """Count the number of documents in the vector store.

        Returns:
            Number of documents
        """
        index_name = os.environ.get("ELASTICSEARCH_INDEX", "index")

        with self._get_es_client() as es_client:
            try:
                count_result = es_client.count(index=index_name)
                return count_result["count"]
            except Exception:
                # If count fails, return 0
                return 0

    def create_index(self) -> None:
        """Create index/collection in the vector store."""
        # Elasticsearch auto-creates indices when adding documents
        # This method exists for consistency
        print(
            f"Index creation not required for {self.configuration.retriever_provider}"
        )

    def delete_index(self) -> None:
        """Delete the entire index/collection."""
        # For Elasticsearch, delete all documents to effectively clear the index
        vector_store = self._get_vector_store()
        vector_store.delete(ids=None)  # Delete all documents

    def clear_documents(self) -> None:
        """Remove all documents from the vector store."""
        index_name = os.environ.get("ELASTICSEARCH_INDEX", "index")

        with self._get_es_client() as es_client:
            # Use Elasticsearch client to delete all documents in the index
            es_client.delete_by_query(
                index=index_name, body={"query": {"match_all": {}}}, refresh=True
            )
