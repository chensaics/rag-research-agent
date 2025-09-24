#!/usr/bin/env python
"""@Author:     sai.chen
@FileName:   mongodb_crud_manager.py
@Date:       2025/09/24
@Description:
-----------------------------------------------------------
MongoDB CRUD Manager
-----------------------------------------------------------
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient

from src.shared.configuration_manager import BaseConfiguration
from src.shared.text_encoder import make_text_encoder


class MongoDBCRUDManager:
    """CRUD Manager for MongoDB."""

    def __init__(self, configuration: BaseConfiguration):
        """Initialize the CRUD manager with a configuration."""
        self.configuration = configuration
        self.embedding_model = make_text_encoder(configuration.embedding_model)
        self._vector_store = None

    def _parse_namespace(self) -> Tuple[str, str]:
        """Parse database and collection names from namespace."""
        namespace = os.environ.get(
            "MONGODB_NAMESPACE", "langgraph_retrieval_agent.default"
        )
        try:
            db_name, collection_name = namespace.split(".", 1)
        except ValueError:
            db_name = "langgraph_retrieval_agent"
            collection_name = "default"
        return db_name, collection_name

    def _get_mongodb_uri(self) -> str:
        """Get MongoDB URI from environment variables."""
        return os.environ.get("MONGODB_URI", "mongodb://localhost:27017/")

    @contextmanager
    def _get_mongo_client(self):
        """Context manager for MongoDB client."""
        client = MongoClient(self._get_mongodb_uri())
        try:
            yield client
        finally:
            client.close()

    def _get_vector_store(self) -> MongoDBAtlasVectorSearch:
        """Get MongoDB Atlas vector store."""
        if self._vector_store is None:
            # Check if we should use local MongoDB or MongoDB Atlas
            if self.configuration.retriever_provider == "mongodb-local":
                # Local MongoDB setup
                # Parse the MongoDB URI to extract connection parameters
                mongodb_uri = self._get_mongodb_uri()
                client = MongoClient(mongodb_uri)

                # Parse database and collection names from namespace or use defaults
                db_name, collection_name = self._parse_namespace()

                db = client[db_name]

                # Create the vector store with direct client connection
                self._vector_store = MongoDBAtlasVectorSearch(
                    collection=db[collection_name],
                    embedding=self.embedding_model,
                    index_name=os.environ.get("MONGODB_INDEX_NAME", "default_index"),
                    text_key="text",
                    embedding_key="embedding",
                )
            else:
                # MongoDB Atlas setup (existing code)
                self._vector_store = MongoDBAtlasVectorSearch.from_connection_string(
                    os.environ["MONGODB_URI"],
                    namespace=os.environ.get(
                        "MONGODB_NAMESPACE", "langgraph_retrieval_agent.default"
                    ),
                    embedding=self.embedding_model,
                )

        return self._vector_store

    def create_index(self) -> None:
        """Create index/collection in the vector store."""
        # MongoDB auto-creates collections when adding documents
        # This method exists for consistency
        print(
            f"Index creation not required for {self.configuration.retriever_provider}"
        )

    def delete_index(self) -> None:
        """Delete the entire index/collection."""
        # For MongoDB, delete all documents in the collection
        db_name, collection_name = self._parse_namespace()

        with self._get_mongo_client() as client:
            db = client[db_name]
            collection = db[collection_name]
            collection.delete_many({})

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
        if hasattr(vector_store, "delete"):
            vector_store.delete(ids)
            return True
        else:
            raise NotImplementedError(
                f"Delete operation not implemented for {self.configuration.retriever_provider}"
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
        db_name, collection_name = self._parse_namespace()

        with self._get_mongo_client() as client:
            db = client[db_name]
            collection = db[collection_name]
            count = collection.count_documents({})
            return count

    def clear_documents(self) -> None:
        """Remove all documents from the vector store."""
        db_name, collection_name = self._parse_namespace()

        with self._get_mongo_client() as client:
            db = client[db_name]
            collection = db[collection_name]
            collection.delete_many({})
