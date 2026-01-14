from __future__ import annotations

"""
Markdown 解析器
---------------

使用 langchain 的 MarkdownHeaderTextSplitter 进行结构感知解析。

参考 C8 的实现方式。
"""

from pathlib import Path
from typing import Dict, Any, Optional, Callable
import hashlib
import uuid

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from .parser_base import BaseParser


class MarkdownParser(BaseParser):
    """
    Markdown 解析器，参考 C8 的实现。
    
    支持按标题层级进行结构感知解析。
    """
    
    def __init__(
        self,
        headers_to_split_on: Optional[list[tuple[str, str]]] = None,
        strip_headers: bool = False,
    ):
        """
        初始化 Markdown 解析器。
        
        Args:
            headers_to_split_on: 要分割的标题层级，格式为 [("#", "主标题"), ("##", "二级标题")]
            strip_headers: 是否去除标题标记
        """
        if headers_to_split_on is None:
            # 默认按三级标题分割（参考 C8）
            headers_to_split_on = [
                ("#", "主标题"),
                ("##", "二级标题"),
                ("###", "三级标题"),
            ]
        
        self.headers_to_split_on = headers_to_split_on
        self.strip_headers = strip_headers
        
        # 创建 Markdown 分割器
        self.splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=strip_headers,
        )
    
    def parse(
        self,
        file_path: str | Path,
        encoding: str = "utf-8",
        **kwargs
    ) -> Dict[str, Any]:
        """
        解析 Markdown 文件。
        
        Args:
            file_path: 文件路径
            encoding: 文件编码，默认 utf-8
            **kwargs: 其他参数（如 parent_id 等）
        
        Returns:
            包含文档信息的字典
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 读取文件内容
        with open(file_path, "r", encoding=encoding) as f:
            content = f.read()
        
        # 生成父文档 ID（参考 C8 的实现）
        try:
            relative_path = file_path.resolve().as_posix()
            parent_id = hashlib.md5(relative_path.encode("utf-8")).hexdigest()
        except Exception:
            parent_id = str(uuid.uuid4())
        
        # 使用 Markdown 分割器分割（这会返回 Document 列表）
        chunks = self.splitter.split_text(content)
        
        # 转换为字典格式
        return {
            "content": content,  # 完整内容
            "file_path": str(file_path),
            "file_type": "markdown",
            "file_name": file_path.name,
            "parent_id": parent_id,
            "chunks": chunks,  # 分割后的块（Document 列表）
            "metadata": {
                "parent_id": parent_id,
                "doc_type": "parent",
            },
        }
    
    def parse_to_documents(
        self,
        file_path: str | Path,
        encoding: str = "utf-8",
        enhance_metadata: Optional[Callable] = None,
    ) -> tuple[Document, list[Document]]:
        """
        解析 Markdown 文件，返回父文档和子文档列表（参考 C8 的实现）。
        
        Args:
            file_path: 文件路径
            encoding: 文件编码
            enhance_metadata: 元数据增强函数（可选）
        
        Returns:
            (父文档, 子文档列表)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 读取文件内容
        with open(file_path, "r", encoding=encoding) as f:
            content = f.read()
        
        # 生成父文档 ID
        try:
            relative_path = file_path.resolve().as_posix()
            parent_id = hashlib.md5(relative_path.encode("utf-8")).hexdigest()
        except Exception:
            parent_id = str(uuid.uuid4())
        
        # 创建父文档
        parent_metadata = {
            "source": str(file_path),
            "parent_id": parent_id,
            "doc_id": parent_id, # 补全 doc_id 字段
            "doc_type": "parent",
            "file_name": file_path.name,
            "file_type": "markdown",
        }
        
        # 元数据增强（如果提供了增强函数）
        if enhance_metadata:
            parent_metadata = enhance_metadata(parent_metadata, file_path, content)
        
        parent_doc = Document(
            page_content=content,
            metadata=parent_metadata,
        )
        
        # 使用 Markdown 分割器分割
        child_chunks = self.splitter.split_text(content)
        
        # 为每个子块添加元数据
        child_docs = []
        for i, chunk in enumerate(child_chunks):
            child_id = str(uuid.uuid4())
            
            # 合并父文档元数据和标题元数据
            child_metadata = {
                **parent_metadata,
                **chunk.metadata,  # 标题元数据（来自 MarkdownHeaderTextSplitter）
                "chunk_id": child_id,
                "parent_id": parent_id,
                "doc_type": "child",
                "chunk_index": i,
            }
            
            child_doc = Document(
                page_content=chunk.page_content,
                metadata=child_metadata,
            )
            child_docs.append(child_doc)
        
        return parent_doc, child_docs
    
    @classmethod
    def get_supported_extensions(cls) -> list[str]:
        """获取支持的扩展名"""
        return [".md", ".markdown", ".mdx"]


__all__ = ["MarkdownParser"]

