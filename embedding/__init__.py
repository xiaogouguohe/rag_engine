"""
Embedding 模块
-------------

提供文本向量化功能，支持同步和异步调用。
"""

from .client import EmbeddingClient, Vector

__all__ = ["EmbeddingClient", "Vector"]

