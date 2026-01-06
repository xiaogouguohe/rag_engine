"""
文档处理模块
-----------

提供文档解析和文本分块功能。

参考 RAGFlow 和 C8 的实现。
"""

from .parser_base import BaseParser
from .parser import TxtParser
from .markdown_parser import MarkdownParser
from .parser_factory import ParserFactory
from .chunker import TextChunker
from .metadata_enhancer import MetadataEnhancer
from .data_preparation import DataPreparationModule

# 为了向后兼容，保留 DocumentParser 作为 ParserFactory 的别名
DocumentParser = ParserFactory

__all__ = [
    "BaseParser",
    "TxtParser",
    "MarkdownParser",
    "ParserFactory",
    "DocumentParser",  # 向后兼容
    "TextChunker",
    "MetadataEnhancer",
    "DataPreparationModule",
]
