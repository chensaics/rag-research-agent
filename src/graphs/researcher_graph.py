#!/usr/bin/env python
"""@Author:     sai.chen
@FileName:   researcher_graph.py
@Date:       2025/09/24
@Description:
-----------------------------------------------------------
在 Main Graph 检索系统中用作子图。

本模块定义了研究者图的核心结构和功能，负责生成搜索查询和检索相关文档。
-----------------------------------------------------------
"""

from typing import TypedDict, cast

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.log_util import logger
from src.shared.configuration_manager import AgentConfiguration
from src.shared.model_manager import model_manager
from src.shared.retrieval_manager import make_retriever
from src.shared.state import QueryState, ResearcherState


class Response(TypedDict):
    queries: list[str]


async def generate_queries(
    state: ResearcherState, *, config: RunnableConfig
) -> dict[str, list[str]]:
    """基于问题（研究计划中的一个步骤）生成搜索查询。

    该函数使用语言模型生成多样化的搜索查询以帮助回答问题。

    参数:
        state (ResearcherState): 研究者的当前状态，包括用户的问题。
        config (RunnableConfig): 配置，包含用于生成查询的模型。

    返回:
        dict[str, list[str]]: 包含'queries'键的字典，其值为生成的搜索查询列表。
    """
    logger.debug(f"Generating search queries for question: {state.question}")

    configuration = AgentConfiguration.from_runnable_config(config)
    model = model_manager.get_model(configuration.llm_model).with_structured_output(
        Response
    )
    messages = [
        {"role": "system", "content": configuration.generate_queries_system_prompt},
        {"role": "human", "content": state.question},
    ]
    response = cast(Response, await model.ainvoke(messages))

    logger.debug(f"Generated {len(response['queries'])} queries")
    return {"queries": response["queries"]}


async def retrieve_documents(
    state: QueryState, *, config: RunnableConfig
) -> dict[str, list[Document]]:
    """基于给定查询检索文档。

    该函数使用检索器为给定查询获取相关文档。

    参数:
        state (QueryState): 包含查询字符串的当前状态。
        config (RunnableConfig): 配置，包含用于获取文档的检索器。

    返回:
        dict[str, list[Document]]: 包含'documents'键的字典，其值为检索到的文档列表。
    """
    logger.info("Retrieving documents")
    logger.debug(f"Retrieving documents for query: {state.query}")

    retriever = make_retriever(config)
    response = await retriever.ainvoke(state.query, config)
    logger.debug(f"Retrieved {len(response)} documents")
    return {"documents": response}


def retrieve_in_parallel(state: ResearcherState) -> list[Send]:
    """为每个生成的查询创建并行检索任务。

    该函数为研究者状态中的每个查询准备并行文档检索任务。

    参数:
        state (ResearcherState): 研究者的当前状态，包括生成的查询。

    返回:
        Literal["retrieve_documents"]: Send对象列表，每个代表一个文档检索任务。

    行为:
        - 为状态中的每个查询创建一个Send对象。
        - 每个Send对象以相应的查询为目标，指向"retrieve_documents"节点。
    """
    logger.info(f"Preparing parallel retrieval for {len(state.queries)} queries")
    tasks = [
        Send("retrieve_documents", QueryState(query=query)) for query in state.queries
    ]
    logger.debug(f"Created {len(tasks)} parallel retrieval tasks")
    return tasks


# Define the researcher subgraph
builder = StateGraph(ResearcherState, config_schema=AgentConfiguration)
builder.add_node("generate_queries", generate_queries)
builder.add_node("retrieve_documents", retrieve_documents)

builder.add_edge(START, "generate_queries")
builder.add_conditional_edges("generate_queries", retrieve_in_parallel)
builder.add_edge("retrieve_documents", END)

researcher_graph = builder.compile()
researcher_graph.name = "ResearcherGraph"
