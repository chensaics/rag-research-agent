#!/usr/bin/env python
"""@Author:     sai.chen
@FileName:   configuration_manager.py
@Date:       2025/09/23
@Description:
-----------------------------------------------------------
用于加载和缓存配置对象的配置管理器。
-----------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Any, Literal, Optional, Type, TypeVar

from langchain_core.runnables import RunnableConfig, ensure_config

from src.shared import prompts


@dataclass(kw_only=True)
class BaseConfiguration:
    """用于索引和检索操作的配置类。

    该类定义了配置索引和检索过程所需的参数，
    包括嵌入模型选择、检索器提供者选择和搜索参数。
    """

    embedding_model: Annotated[
        str,
        {"__template_metadata__": {"kind": "embeddings"}},
    ] = field(
        default="ollama/bge-m3:latest",
        metadata={"description": "要使用的嵌入模型的名称。必须是有效的嵌入模型名称。"},
    )

    retriever_provider: Annotated[
        Literal["elastic-local", "elastic", "pinecone", "mongodb"],
        {"__template_metadata__": {"kind": "retriever"}},
    ] = field(
        default="elastic-local",
        metadata={
            "description": "用于检索的向量存储提供者。选项有'elastic'、'pinecone'或'mongodb'。"
        },
    )

    search_kwargs: dict[str, Any] = field(
        default_factory=dict,
        metadata={"description": "传递给检索器搜索函数的额外关键字参数。"},
    )

    @classmethod
    def from_runnable_config(
        cls: Type[T], config: Optional[RunnableConfig] = None
    ) -> T:
        """从RunnableConfig对象创建IndexConfiguration实例。

        Args:
            cls (Type[T]): 类本身。
            config (Optional[RunnableConfig]): 要使用的配置对象。

        Returns:
            T: 具有指定配置的IndexConfiguration实例。
        """
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})


T = TypeVar("T", bound=BaseConfiguration)


@dataclass(kw_only=True)
class AgentConfiguration(BaseConfiguration):
    """The configuration for the agent."""

    llm_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="ollama/qwen3:4b",
        metadata={
            "description": "用于生成响应的语言模型。应以如下形式：provider/model-name。"
        },
    )

    router_system_prompt: str = field(
        default=prompts.ROUTER_SYSTEM_PROMPT,
        metadata={"description": "用于分类用户问题以将其路由到正确节点的系统提示。"},
    )

    more_info_system_prompt: str = field(
        default=prompts.MORE_INFO_SYSTEM_PROMPT,
        metadata={"description": "用于向用户询问更多信息的系统提示。"},
    )

    general_system_prompt: str = field(
        default=prompts.GENERAL_SYSTEM_PROMPT,
        metadata={"description": "用于响应一般问题的系统提示。"},
    )

    research_plan_system_prompt: str = field(
        default=prompts.RESEARCH_PLAN_SYSTEM_PROMPT,
        metadata={"description": "用于根据用户的问题生成研究计划的系统提示。"},
    )

    generate_queries_system_prompt: str = field(
        default=prompts.GENERATE_QUERIES_SYSTEM_PROMPT,
        metadata={
            "description": "用于根据研究计划中的步骤生成查询的研究者的系统提示。"
        },
    )

    response_system_prompt: str = field(
        default=prompts.RESPONSE_SYSTEM_PROMPT,
        metadata={"description": "用于生成响应的系统提示。"},
    )


class IndexConfiguration(BaseConfiguration):
    """Configuration class for indexing and retrieval operations.

    This class defines the parameters needed for configuring the indexing and
    retrieval processes, including embedding model selection, retriever provider choice, and search parameters.
    """

    docs_file: str = field(
        default="src/sample_docs.json",
        metadata={
            "description": "Path to a JSON file containing default documents to index."
        },
    )
