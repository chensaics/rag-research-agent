#!/usr/bin/env python
"""@Author:     sai.chen
@FileName:   test_crud.py
@Date:       2025/09/24
@Description:
-----------------------------------------------------------

-----------------------------------------------------------
"""

import sys
from pathlib import Path

if "__file__" in globals():
    base_path = Path(__file__).resolve().parent
else:
    base_path = Path.cwd()

parent_dir = base_path.parent
grandparent_dir = parent_dir.parent

# 需要添加的目录列表（按优先级从高到低）
dirs_to_add = [parent_dir, grandparent_dir]

# 按优先级顺序添加到 sys.path（避免重复添加）
for dir_path in dirs_to_add:
    if dir_path and str(dir_path) not in sys.path:
        sys.path.insert(1, str(dir_path))  # 插入到索引1位置

from src.crud.elasticsearch_crud_manager import ElasticsearchCRUDManager
from src.shared.configuration_manager import BaseConfiguration

if __name__ == "__main__":
    print("Testing Elasticsearch CRUD operations...")

    config = BaseConfiguration(
        retriever_provider="elastic-local",
        embedding_model="ollama/bge-m3:latest",
    )

    crud_manager = ElasticsearchCRUDManager(config)

    count = crud_manager.count_documents()
    print(f"Number of documents: {count}")

    # Add documents
    from langchain_core.documents import Document

    documents = [
        Document(
            page_content="This is a test document about artificial intelligence",
            metadata={"source": "test"},
        ),
        Document(
            page_content="This is another test document about machine learning",
            metadata={"source": "test"},
        ),
    ]

    ids = crud_manager.add_documents(documents)
    print(f"Added documents with IDs: {ids}")

    # Search documents
    results = crud_manager.search_documents("artificial intelligence")
    print(f"Search results: {results}")

    count = crud_manager.count_documents()
    print(f"Number of documents after adding: {count}")

    crud_manager.clear_documents()
    print("Cleared all documents")

    count = crud_manager.count_documents()
    print(f"Number of documents after clearing: {count}")
