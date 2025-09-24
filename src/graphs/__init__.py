#!/usr/bin/env python
"""@Author:     sai.chen
@FileName:   __init__.py
@Date:       2025/09/23
@Description:
-----------------------------------------------------------
检索图模块

本模块提供了一个智能对话检索图系统，用于
处理用户查询和主题。

该系统的主要组件包括：

1. 用于处理对话上下文和研究步骤的状态管理系统。
2. 分析和路由机制，用于分类用户查询并确定适当的响应路径。
3. 研究规划器，将复杂查询分解为可管理的步骤。
4. 研究代理，根据研究步骤生成查询并获取相关信息。
5. 响应生成器，使用检索到的文档和对话历史来制定答案。

该图使用 AgentConfiguration 类中定义的可定制参数进行配置，
允许在模型选择、检索方法和系统提示方面具有灵活性。

主要特性：
- 智能查询分类和路由
- 针对复杂查询的多步骤研究规划
- 与各种检索提供者集成（例如，Elastic、Milvus、MongoDB）
- 可定制的语言模型用于查询分析、研究规划和响应生成
- 有状态的对话管理，实现上下文感知的交互

用法：
    使用此系统的主要入口点是从本模块导出的`graph`对象。
    可以调用它来处理用户输入，进行研究，并基于检索到的信息和对话上下文
    生成明智的响应。

-----------------------------------------------------------
"""

from src.graphs.index_graph import index_graph
from src.graphs.main_graph import main_graph
from src.graphs.researcher_graph import researcher_graph

__all__ = ["main_graph", "index_graph", "researcher_graph"]
