#!/usr/bin/env python
"""@Author:     sai.chen
@FileName:   text_encoder.py
@Date:       2025/09/23
@Description:
-----------------------------------------------------------
文本编码器工具。
-----------------------------------------------------------
"""

from langchain_core.embeddings import Embeddings

from src.log_util import logger


def make_text_encoder(model: str) -> Embeddings:
    """连接到配置的文本编码器。"""
    provider, model = model.split("/", maxsplit=1)
    logger.debug(f"从提供者初始化文本编码器: {provider}, 模型: {model}")

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model)
    elif provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=model, base_url="http://localhost:11434")
    else:
        logger.error(f"不支持的嵌入提供者: {provider}")
        raise ValueError(f"不支持的嵌入提供者: {provider}")
