#!/usr/bin/env python
"""@Author:     sai.chen
@FileName:   log_util.py
@Date:       2025/09/23
@Description:
-----------------------------------------------------------

-----------------------------------------------------------
"""

import sys
from pathlib import Path

from loguru import logger as LOG

# 将项目根目录添加到系统路径
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)


fmt = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"


def setup_logger(name):
    LOG.add(
        f"{project_root}/logs/{name}.log",  # 日志文件路径
        rotation="10 MB",  # 每10MB轮转一次
        retention="7 days",  # 保留7天日志
        compression="zip",  # 压缩旧日志
        level="INFO",  # 根据环境变量设置日志级别
        enqueue=True,  # 异步安全
        format=fmt,  # 日志格式
    )
    return LOG


logger = setup_logger("rag_research_agent")
