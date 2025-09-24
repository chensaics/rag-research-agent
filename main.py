#!/usr/bin/env python
"""@Author:     sai.chen
@FileName:   example_usage.py
@Date:       2025/09/22
@Description:
-----------------------------------------------------------
Example usage of the RAG Research Agent Template.

This example demonstrates how to use the three main graphs in the RAG Research Agent Template:
1. Index Graph - for indexing documents
2. Main Graph - for conversational question answering
3. Researcher Subgraph - for conducting research on specific topics

The example showcases the main design patterns of the system:
- Multi-agent architecture with specialized graphs
- State management using typed dictionaries
- Configurable models and prompts
- Parallel document retrieval
- Research planning and execution
-----------------------------------------------------------
"""

import asyncio
import uuid

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from src.graphs import index_graph, main_graph, researcher_graph
from src.log_util import logger


async def example_indexing():
    """Index documents using the index graph."""
    logger.info("=== Starting Indexing Example ===")

    # Example documents to index
    sample_docs = [
        Document(
            page_content="RAG Research Agent是一种结合检索增强生成(RAG)技术与智能Agent能力的研究型智能体系统。它不仅具备基础的检索与生成能力，还融合了自我反思、工具链式调用及多智能体协同等高级特性，突破了传统大模型'仅会聊天、检索'的局限。",
            metadata={"source": "RAG Research Agent介绍", "category": "concepts"},
        ),
        Document(
            page_content="RAG Research Agent的核心工作原理包含多阶段检索、智能分块处理、深度上下文理解、迭代式研究和工具链集成。它通过向量数据库进行语义级检索，采用语义感知算法将文档分割为300-500字的逻辑段落，将检索结果与用户查询整合为增强型Prompt，并具备自我反思能力，能根据初步结果调整检索策略。",
            metadata={"source": "RAG Research Agent工作原理", "category": "components"},
        ),
        Document(
            page_content="RAG Research Agent的核心优势包括减少幻觉问题、突破知识边界、专业领域适配、可溯源性和自主研究能力。它基于检索到的真实证据生成回答，避免'一本正经胡说八道'，解决大模型知识固定在训练时间点的局限，特别适合需要专业知识的研究场景。",
            metadata={"source": "RAG Research Agent核心优势", "category": "concepts"},
        ),
        Document(
            page_content="RAG Research Agent的典型应用场景包括学术研究辅助、技术深度调研、市场趋势分析、产品竞品研究和复杂问题解决。它能快速检索并整合相关领域文献，针对特定技术问题进行系统性资料收集与分析，整合多来源数据生成行业分析报告，自动收集并对比竞品信息，通过多步骤检索与推理解决专业领域问题。",
            metadata={
                "source": "RAG Research Agent应用场景",
                "category": "applications",
            },
        ),
        Document(
            page_content="与基础RAG相比，RAG Research Agent在检索方式、推理能力、工具使用、自我优化和协作能力等方面都有显著提升。它支持多轮迭代检索而非单次检索，具备复杂问题分解与解决能力而非简单问答，支持多工具链式调用而非有限工具使用，具备自我反思与策略调整能力而非无自我优化，支持多智能体协同作业而非单一系统。",
            metadata={
                "source": "RAG Research Agent与基础RAG的区别",
                "category": "comparison",
            },
        ),
    ]

    config = {
        "configurable": {
            "retriever_provider": "elastic-local",
            "embedding_model": "ollama/bge-m3:latest",
        }
    }

    try:
        # Index the documents
        await index_graph.ainvoke({"documents": sample_docs, "messages": []}, config)
        logger.info(f"Indexed {len(sample_docs)} documents")
        logger.info("Document indexing completed successfully")
    except Exception as e:
        logger.error(f"Error during document indexing: {e}")
        raise
    logger.info("")


async def example_researcher():
    """Conduct research using the researcher subgraph."""
    logger.info("=== Starting Researcher Graph Example ===")

    # Configuration for the researcher (use elastic-local + OpenAI embeddings by default)
    config = {
        "configurable": {
            "llm_model": "ollama/qwen3:4b",
            "retriever_provider": "elastic-local",
            "embedding_model": "ollama/bge-m3:latest",
        }
    }

    question = "RAG Research Agent 是咋样的架构?"
    logger.info(f"Researching question: {question}")

    try:
        # Run the researcher graph
        result = await researcher_graph.ainvoke({"question": question}, config)

        # Display results
        logger.info(f"Generated queries: {result['queries']}")
        logger.info(f"Retrieved {len(result['documents'])} documents")
        logger.info(
            f"Research completed with {len(result['documents'])} documents retrieved"
        )

        for i, doc in enumerate(result["documents"][:2]):  # Show first 2 docs
            logger.info(f"Document {i + 1}: {doc.page_content[:100]}...")
    except Exception as e:
        logger.error(f"Error during research: {e}")
        raise
    logger.info("")


async def example_retrieval():
    """Use the retrieval graph for conversational QA."""
    logger.info("=== Starting Retrieval Graph Example ===")

    # Configuration for retrieval (use elastic-local + OpenAI embeddings by default)
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "llm_model": "ollama/qwen3:4b",
            "retriever_provider": "elastic-local",
            "embedding_model": "ollama/bge-m3:latest",
        }
    }

    try:
        question1 = "RAG Research Agent 有哪些应用场景？"
        logger.info(f"Question 1: {question1}")

        result1 = await main_graph.ainvoke(
            {"messages": [HumanMessage(content=question1)]}, config
        )

        answer1 = result1["messages"][-1].content
        logger.info(f"Answer 1: {answer1}")
        logger.info("===================================")

        # Follow-up question
        question2 = "RAG Research Agent 和传统的 RAG 有什么区别？"
        logger.info(f"Question 2: {question2}")

        result2 = await main_graph.ainvoke(
            {"messages": [HumanMessage(content=question2)]}, config
        )

        answer2 = result2["messages"][-1].content
        logger.info(f"Answer 2: {answer2}")
        logger.info("Answer 2 generated successfully")
        logger.info("===================================")
    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        raise


async def main():
    """Run all examples."""
    logger.info("Starting RAG Research Agent Template Usage Examples")
    logger.info("==========================================")

    try:
        # # Run indexing
        # await example_indexing()

        # # Run researcher
        # await example_researcher()

        # Run retrieval
        await example_retrieval()

        logger.info("All examples completed successfully!")

    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
