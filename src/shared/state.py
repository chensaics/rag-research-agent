"""代理的共享状态。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Literal, Sequence, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

from src.shared.utils import reduce_docs


@dataclass
class GraphState:
    """代理的共享状态。"""

    messages: Sequence[AnyMessage]
    documents: Annotated[Sequence[Document], "检索到的文档"] = field(
        default_factory=list
    )
    plan: Sequence[str] = field(default_factory=list)
    num_researched: int = 0
    response: str = ""


@dataclass
class QueryState:
    """研究者图中retrieve_documents节点的私有状态。"""

    query: str


@dataclass
class ResearcherState:
    """研究者图的状态。"""

    question: str
    """由检索器代理生成的研究计划中的一个步骤。"""
    queries: Annotated[Sequence[str], "搜索查询列表"] = field(default_factory=list)
    """研究者基于问题生成的搜索查询列表。"""
    documents: Annotated[Sequence[Document], reduce_docs] = field(default_factory=list)
    """由检索器填充。这是代理可以引用的文档列表。"""


@dataclass
class InputState:
    """代理的输入状态。

    该类定义了输入状态的结构，包括
    用户和代理之间交换的消息。它作为
    完整状态的受限版本，与内部维护的状态相比，
    向外部世界提供更窄的接口。
    """

    messages: Annotated[list[AnyMessage], add_messages]


class Router(TypedDict):
    """对用户查询进行分类。"""

    logic: str
    type: Literal["more-info", "rag-research", "general"]


@dataclass
class AgentState(InputState):
    """检索图/代理的状态。"""

    router: Router = field(default_factory=lambda: Router(type="general", logic=""))
    """路由器对用户查询的分类。"""
    steps: list[str] = field(default_factory=list)
    """研究计划中的步骤列表。"""
    documents: Annotated[list[Document], reduce_docs] = field(default_factory=list)
    """由检索器填充。这是代理可以引用的文档列表。"""


@dataclass
class IndexState:
    """索引图的状态。"""

    messages: Sequence[AnyMessage] = field(default_factory=list)
    documents: Annotated[Sequence[Document], "已索引的文档"] = field(
        default_factory=list
    )
