"""项目中使用的共享工具函数。

函数:
    format_docs: 将文档转换为xml格式的字符串。
"""

import hashlib
import uuid
from typing import Any, Literal, Optional, Union

from langchain_core.documents import Document

from src.log_util import logger


def _format_doc(doc: Document) -> str:
    """将单个文档格式化为XML。

    Args:
        doc (Document): 要格式化的文档。

    Returns:
        str: 格式化后的XML字符串文档。
    """
    logger.debug("格式化文档")
    metadata = doc.metadata or {}
    meta = "".join(f" {k}={v!r}" for k, v in metadata.items())
    if meta:
        meta = f" {meta}"

    return f"<document{meta}>\n{doc.page_content}\n</document>"


def format_docs(docs: Optional[list[Document]]) -> str:
    """将文档列表格式化为XML。

    该函数接收Document对象列表并将其格式化为单个XML字符串。

    Args:
        docs (Optional[list[Document]]): 要格式化的Document对象列表，或None。

    Returns:
        str: 包含XML格式文档的字符串。

    Examples:
        >>> docs = [Document(page_content="Hello"), Document(page_content="World")]
        >>> print(format_docs(docs))
        <documents>
        <document>
        Hello
        </document>
        <document>
        World
        </document>
        </documents>

        >>> print(format_docs(None))
        <documents></documents>
    """
    doc_count = len(docs) if docs else 0
    logger.debug(f"将 {doc_count} 个文档格式化为XML")

    if not docs:
        return "<documents></documents>"
    formatted = "\n".join(_format_doc(doc) for doc in docs)
    result = f"""<documents>
{formatted}
</documents>"""

    logger.debug("文档格式化完成")
    return result


def _generate_uuid(page_content: str) -> str:
    """根据页面内容为文档生成UUID。"""
    md5_hash = hashlib.md5(page_content.encode()).hexdigest()
    return str(uuid.UUID(md5_hash))


def reduce_docs(
    existing: Optional[list[Document]],
    new: Union[
        list[Document],
        list[dict[str, Any]],
        list[str],
        str,
        Literal["delete"],
    ],
) -> list[Document]:
    """根据输入类型减少和处理文档。

    该函数处理各种输入类型并将其转换为Document对象序列。
    它可以删除现有文档，从字符串或字典创建新文档，或返回现有文档。
    它还可以根据文档ID将现有文档与新文档合并。

    Args:
        existing (Optional[Sequence[Document]]): 状态中的现有文档（如果有）。
        new (Union[Sequence[Document], Sequence[dict[str, Any]], Sequence[str], str, Literal["delete"]]):
            要处理的新输入。可以是文档序列、字典序列、字符串序列、单个字符串，
            或字面量"delete"。
    """
    if new == "delete":
        return []

    existing_list = list(existing) if existing else []
    if isinstance(new, str):
        return existing_list + [
            Document(page_content=new, metadata={"uuid": _generate_uuid(new)})
        ]

    new_list = []
    if isinstance(new, list):
        existing_ids = set(doc.metadata.get("uuid") for doc in existing_list)
        for item in new:
            if isinstance(item, str):
                item_id = _generate_uuid(item)
                new_list.append(Document(page_content=item, metadata={"uuid": item_id}))
                existing_ids.add(item_id)

            elif isinstance(item, dict):
                metadata = item.get("metadata", {})
                item_id = metadata.get("uuid") or _generate_uuid(
                    item.get("page_content", "")
                )

                if item_id not in existing_ids:
                    new_list.append(
                        Document(**{**item, "metadata": {**metadata, "uuid": item_id}})
                    )
                    existing_ids.add(item_id)

            elif isinstance(item, Document):
                item_id = item.metadata.get("uuid", "")
                if not item_id:
                    item_id = _generate_uuid(item.page_content)
                    new_item = item.copy(deep=True)
                    new_item.metadata["uuid"] = item_id
                else:
                    new_item = item

                if item_id not in existing_ids:
                    new_list.append(new_item)
                    existing_ids.add(item_id)

    return existing_list + new_list
