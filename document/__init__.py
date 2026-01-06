"""
文档处理模块
-----------

提供文档解析和文本分块功能。
"""

from .parser import DocumentParser
from .chunker import TextChunker

__all__ = ["DocumentParser", "TextChunker"]
