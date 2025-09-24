#!/usr/bin/env python
"""@Author:     sai.chen
@FileName:   index_graph.py
@Date:       2025/09/24
@Description:
-----------------------------------------------------------
用来对文档进行构建索引
-----------------------------------------------------------
"""

import json
from typing import Optional

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from src.log_util import logger
from src.shared.configuration_manager import IndexConfiguration
from src.shared.retrieval_manager import make_retriever
from src.shared.state import IndexState, reduce_docs


async def index_docs(
    state: IndexState, *, config: Optional[RunnableConfig] = None
) -> dict[str, str]:
    """使用检索器异步索引文档。

    从状态中获取文档并添加到检索器的索引中。

    如果状态中未提供文档，将从配置 .docs_file JSON文件中加载。

    Args:
        state (IndexState): 包含文档和检索器的当前状态。
        config (Optional[RunnableConfig]): 索引过程的配置。

    Returns:
        dict[str, str]: 包含从状态中删除文档指令的字典。
    """
    logger.info("开始文档索引过程")

    if not config:
        logger.error("运行 index_docs 需要配置。")
        raise ValueError("运行 index_docs 需要配置。")

    configuration = IndexConfiguration.from_runnable_config(config)
    logger.info(f"配置已加载: {configuration}")
    docs = state.documents

    if not docs:
        logger.info(f"从文件加载文档: {configuration.docs_file}")
        try:
            with open(configuration.docs_file) as f:
                serialized_docs = json.load(f)
                docs = reduce_docs([], serialized_docs)
            logger.debug(f"已加载 {len(docs)} 个文档")
        except FileNotFoundError:
            logger.warning(f"未找到文档文件: {configuration.docs_file}")
            docs = []
        except json.JSONDecodeError as e:
            logger.error(f"解析JSON文档文件时出错: {e}")
            docs = []

    if docs:
        logger.info("将文档添加到向量存储")
        try:
            retriever = make_retriever(config)
            await retriever.aadd_documents(docs)
            logger.info("文档已成功添加到向量存储")
        except Exception as e:
            logger.error(f"将文档添加到向量存储时出错: {e}")
            raise
    else:
        logger.info("没有文档需要索引")

    return {"documents": "delete"}


builder = StateGraph(IndexState, config_schema=IndexConfiguration)
builder.add_node(index_docs)
builder.add_edge(START, "index_docs")
builder.add_edge("index_docs", END)

index_graph = builder.compile()
index_graph.name = "IndexGraph"
