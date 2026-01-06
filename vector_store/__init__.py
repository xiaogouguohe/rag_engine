"""
向量存储模块
-----------

提供向量存储和检索功能，使用 Milvus Lite。
"""

from .store import VectorStore, ChunkMetadata

__all__ = ["VectorStore", "ChunkMetadata"]

