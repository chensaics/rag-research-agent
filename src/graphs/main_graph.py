#!/usr/bin/env python
"""@Author:     sai.chen
@FileName:   main_graph.py
@Date:       2025/09/24
@Description:
-----------------------------------------------------------
主对话检索图。

本模块定义了对话检索图的核心结构和功能，
负责路由用户查询、进行研究并生成响应。
-----------------------------------------------------------
"""

import asyncio
import os
from typing import Any, List, Literal, TypedDict, cast

from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.mongodb import AsyncMongoDBSaver
from langgraph.graph import END, START, StateGraph

from src.graphs.researcher_graph import researcher_graph
from src.log_util import logger
from src.shared.configuration_manager import AgentConfiguration
from src.shared.model_manager import model_manager
from src.shared.state import AgentState, InputState, Router


class ResearchPlan(TypedDict):
    """研究计划结构。"""

    steps: List[str]


async def analyze_and_route_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, Router]:
    """分析用户查询并将其路由到适当的处理器。

    该函数使用语言模型将查询分类为以下三类之一：
    - "more-info": 查询需要更多信息才能正确回答。
    - "rag-research": 查询与rag-research相关，应进行研究。
    - "general": 查询可以直接回答，无需研究。

    参数:
        state (AgentState): 代理的当前状态，包括对话历史。
        config (RunnableConfig): 配置，包含用于查询分析的模型。

    返回:
        dict[str, Router]: 包含'router'键的字典，其值为分类结果（分类类型和逻辑）。
    """
    logger.info("Analyzing and routing user query")
    configuration = AgentConfiguration.from_runnable_config(config)
    messages = [
        {"role": "system", "content": configuration.router_system_prompt}
    ] + state.messages

    model = model_manager.get_model(configuration.llm_model)
    response = cast(
        Router, await model.with_structured_output(Router).ainvoke(messages)
    )
    logger.debug(f"Query routed with classification: {response['type']}")
    return {"router": response}


async def ask_for_more_info(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, Any]:
    """当查询不明确时，要求用户提供更多信息。

    该函数根据路由逻辑生成一个响应，要求用户提供更具体的信息。

    参数:
        state (AgentState): 代理的当前状态。
        config (RunnableConfig): 配置，包含用于生成响应的模型。

    返回:
        dict[str, Any]: 包含'messages'键的字典，其值为生成的响应。
    """
    logger.info("Generating request for more information")
    configuration = AgentConfiguration.from_runnable_config(config)
    messages = [
        {"role": "system", "content": configuration.more_info_system_prompt}
    ] + state.messages
    model = model_manager.get_model(configuration.llm_model)
    response = await model.ainvoke(messages)
    return {"messages": [response]}


async def respond_to_general_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, Any]:
    """回应不需要研究的一般查询。

    该函数使用语言模型为一般查询生成直接响应。

    参数:
        state (AgentState): 代理的当前状态。
        config (RunnableConfig): 配置，包含用于生成响应的模型。

    返回:
        dict[str, Any]: 包含'messages'键的字典，其值为生成的响应。
    """
    logger.info("Responding to general query")
    configuration = AgentConfiguration.from_runnable_config(config)
    response_system_prompt = configuration.response_system_prompt
    messages = [{"role": "system", "content": response_system_prompt}] + state.messages
    model = model_manager.get_model(configuration.llm_model)
    response = await model.ainvoke(messages)
    return {"messages": [response]}


async def create_research_plan(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[str]]:
    """为LangChain相关查询创建研究计划。

    该函数为回答LangChain相关查询生成逐步研究计划。

    参数:
        state (AgentState): 代理的当前状态。
        config (RunnableConfig): 配置，包含用于生成研究计划的模型。

    返回:
        dict[str, list[str]]: 包含'steps'键的字典，其值为生成的研究计划。
    """
    logger.info("Creating research plan")
    configuration = AgentConfiguration.from_runnable_config(config)
    messages = [
        {"role": "system", "content": configuration.research_plan_system_prompt}
    ] + state.messages

    model = model_manager.get_model(configuration.llm_model)
    response = cast(
        ResearchPlan, await model.with_structured_output(ResearchPlan).ainvoke(messages)
    )
    logger.debug(f"Research plan created with {len(response['steps'])} steps")
    return {"steps": response["steps"]}


async def conduct_research(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[AnyMessage]]:
    """通过执行研究计划来进行研究。

    该函数通过为每个步骤运行研究子图来执行研究计划。

    参数:
        state (AgentState): 代理的当前状态。
        config (RunnableConfig): 配置，包含用于进行研究的模型。

    返回:
        dict[str, list[AnyMessage]]: 包含'next'键的字典，其值为流程中的下一步。
    """
    logger.info("Conducting research")
    tasks = [
        researcher_graph.ainvoke({"question": step}, config) for step in state.steps
    ]
    results = await asyncio.gather(*tasks)
    documents = [
        document for result in results for document in result.get("documents", [])
    ]
    return {"documents": documents}


async def respond(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[AnyMessage]]:
    """基于研究结果生成最终响应。

    该函数基于检索到的文档生成对用户查询的最终响应。

    参数:
        state (AgentState): 代理的当前状态。
        config (RunnableConfig): 配置，包含用于生成响应的模型。

    返回:
        dict[str, list[AnyMessage]]: 包含'messages'键的字典，其值为生成的响应。
    """
    logger.info("Generating final response")
    configuration = AgentConfiguration.from_runnable_config(config)
    system_prompt = configuration.response_system_prompt
    context = "\n\n".join([doc.page_content for doc in state.documents])
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "system",
            "content": f"<docs>{context}</docs>",
        },
    ] + state.messages
    model = model_manager.get_model(configuration.llm_model)
    response = await model.ainvoke(messages)
    return {"messages": [response]}


def route_query(
    state: AgentState,
) -> Literal["ask_for_more_info", "respond_to_general_query", "create_research_plan"]:
    """根据分类结果路由查询。

    参数:
        state (AgentState): 代理的当前状态。

    返回:
        Literal["ask_for_more_info", "respond_to_general_query", "create_research_plan"]: 要执行的下一个节点。
    """
    logger.debug(f"Routing based on classification: {state.router['type']}")
    mapping = {
        "more-info": "ask_for_more_info",
        "general": "respond_to_general_query",
        "rag-research": "create_research_plan",
    }
    return mapping[state.router["type"]]


def check_finished(state: AgentState) -> Literal["respond"]:
    """检查研究是否完成并路由到响应生成。

    参数:
        state (AgentState): 代理的当前状态。

    返回:
        Literal["respond"]: 总是返回"respond"以继续到响应生成。
    """
    logger.debug("Research finished, proceeding to response generation")
    return "respond"


# Define the Main Graph
builder = StateGraph(AgentState, input=InputState, config_schema=AgentConfiguration)
builder.add_node("analyze_and_route_query", analyze_and_route_query)
builder.add_node("ask_for_more_info", ask_for_more_info)
builder.add_node("respond_to_general_query", respond_to_general_query)
builder.add_node("create_research_plan", create_research_plan)
builder.add_node("conduct_research", conduct_research)
builder.add_node("respond", respond)

builder.add_edge(START, "analyze_and_route_query")
builder.add_conditional_edges("analyze_and_route_query", route_query)
builder.add_edge("create_research_plan", "conduct_research")
builder.add_conditional_edges("conduct_research", check_finished)
builder.add_edge("ask_for_more_info", END)
builder.add_edge("respond_to_general_query", END)
builder.add_edge("respond", END)

if os.environ.get("MONGODB_URI"):
    checkpointer = AsyncMongoDBSaver.from_conn_string(os.environ["MONGODB_URI"])
else:
    from langgraph.checkpoint.memory import MemorySaver

    checkpointer = MemorySaver()

main_graph = builder.compile(checkpointer=checkpointer)
main_graph.name = "MainGraph"
