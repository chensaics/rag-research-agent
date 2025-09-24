#!/usr/bin/env python
"""@Author:     sai.chen
@FileName:   milvus_crud_manager.py
@Date:       2025/09/24
@Description:
-----------------------------------------------------------
Milvus CRUD Manager
-----------------------------------------------------------
"""

from __future__ import annotations

import os
from typing import List

from langchain_core.documents import Document
from langchain_milvus import Milvus
from pymilvus import Collection, connections, utility

from src.shared.configuration_manager import BaseConfiguration
from src.shared.text_encoder import make_text_encoder


class MilvusCRUDManager:
    """CRUD Manager for Milvus."""

    def __init__(self, configuration: BaseConfiguration):
        """Initialize the CRUD manager with a configuration."""
        self.configuration = configuration
        self.embedding_model = make_text_encoder(configuration.embedding_model)
        self._vector_store = None

    def _get_milvus_uri(self) -> str:
        """Get Milvus URI from environment variables."""
        return os.environ.get("MILVUS_URI", "http://localhost:19530")

    def _get_collection_name(self) -> str:
        """Get collection name from environment variables."""
        return os.environ.get("MILVUS_COLLECTION", "index")

    def _ensure_connection(self, alias: str = "default") -> None:
        """Ensure Milvus connection exists."""
        if not connections.has_connection(alias):
            connections.connect(
                alias=alias,
                uri=self._get_milvus_uri(),
            )

    def _get_vector_store(self) -> Milvus:
        """Get Milvus vector store."""
        if self._vector_store is None:
            self._vector_store = Milvus(
                embedding_function=self.embedding_model,
                connection_args={
                    "uri": self._get_milvus_uri(),
                },
                collection_name=self._get_collection_name(),
            )

        return self._vector_store

    def create_index(self) -> None:
        """Create index/collection in the vector store."""
        # Milvus auto-creates collections when adding documents
        # This method exists for consistency
        print(
            f"Index creation not required for {self.configuration.retriever_provider}"
        )

    def delete_index(self) -> None:
        """Delete the entire index/collection."""
        # For Milvus, drop the collection
        alias = "default"
        self._ensure_connection(alias)

        collection_name = self._get_collection_name()
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)

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
        raise NotImplementedError(
            f"Delete operation by IDs not implemented for {self.configuration.retriever_provider}"
        )

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
        alias = "default"
        self._ensure_connection(alias)

        collection_name = self._get_collection_name()
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            collection.load()
            count = collection.num_entities
            return count
        else:
            return 0

    def clear_documents(self) -> None:
        """Remove all documents from the vector store."""
        alias = "default"
        self._ensure_connection(alias)

        collection_name = self._get_collection_name()
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            # Delete all entities
            collection.delete(expr="pk >= 0")
