from __future__ import annotations

"""
向量存储模块
------------

参考 RAGFlow 的设计和 cloud-edge-milk-tea-agent 的实现，提供向量存储和检索功能。

使用 Milvus Lite 作为向量数据库（轻量、单机、无需 Docker，作为 Python 库直接使用）。

设计要点：
1. 按知识库（kb_id）组织向量集合（collection）
2. 存储向量和对应的元数据（文档ID、chunk文本等）
3. 支持相似度搜索（top-k）
4. 持久化到本地文件系统
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

try:
    from pymilvus import MilvusClient
    _HAS_MILVUS = True
except ImportError:
    _HAS_MILVUS = False
    MilvusClient = None


@dataclass
class ChunkMetadata:
    """Chunk 元数据（参考 RAGFlow 的 chunk 结构）"""
    chunk_id: str
    doc_id: str
    kb_id: str
    text: str
    # 可选字段
    page_num: Optional[int] = None
    position: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class VectorStore:
    """
    向量存储类，参考 RAGFlow 的检索服务设计和 cloud-edge-milk-tea-agent 的 Milvus Lite 实现。
    
    每个知识库（kb_id）对应一个独立的 Milvus collection。
    """
    
    def __init__(self, storage_path: str = "./data/indices"):
        """
        初始化向量存储。
        
        Args:
            storage_path: 数据库文件存储路径（Milvus Lite 会在此路径下创建数据库文件）
        """
        if not _HAS_MILVUS:
            raise ImportError(
                "需要安装 pymilvus。运行: pip install pymilvus[milvus_lite]"
            )
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Milvus Lite 数据库路径
        db_path = str(self.storage_path / "milvus_lite.db")
        
        # 连接 Milvus Lite（参考 cloud-edge-milk-tea-agent 的实现）
        try:
            self.client = MilvusClient(uri=db_path)
        except Exception as e:
            raise ConnectionError(f"无法连接到 Milvus Lite 数据库 {db_path}: {str(e)}")
        
        # 存储每个知识库的集合名称（kb_id -> collection_name）
        self._collections: Dict[str, str] = {}
    
    def _get_collection_name(self, kb_id: str) -> str:
        """获取知识库对应的 collection 名称"""
        if kb_id not in self._collections:
            # 使用 kb_id 作为 collection 名称（Milvus 的 collection 名称需要符合规范）
            # 替换特殊字符，确保符合 Milvus 命名规范
            collection_name = f"kb_{kb_id}".replace("-", "_").replace(".", "_")
            self._collections[kb_id] = collection_name
        return self._collections[kb_id]
    
    def _ensure_collection(self, kb_id: str, vector_dim: int):
        """确保 collection 存在（参考 cloud-edge-milk-tea-agent 的实现）"""
        collection_name = self._get_collection_name(kb_id)
        
        if self.client.has_collection(collection_name):
            return collection_name
        
        # 创建新 collection（使用余弦相似度，参考 cloud-edge-milk-tea-agent）
        self.client.create_collection(
            collection_name=collection_name,
            dimension=vector_dim,
            metric_type="COSINE",  # 使用余弦相似度
            auto_id=True,  # 自动生成 ID
        )
        
        return collection_name
    
    def add_texts(
        self,
        kb_id: str,
        texts: List[str],
        vectors: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        添加文本和向量到指定知识库（参考 RAGFlow 的 insert_es 方法和 cloud-edge-milk-tea-agent 的实现）。
        
        Args:
            kb_id: 知识库 ID
            texts: 文本列表
            vectors: 向量列表（与 texts 一一对应）
            metadatas: 元数据列表（可选，与 texts 一一对应）
        
        Returns:
            chunk_id 列表
        """
        if not texts or not vectors:
            return []
        
        if len(texts) != len(vectors):
            raise ValueError(f"texts 和 vectors 数量不一致: {len(texts)} vs {len(vectors)}")
        
        vector_dim = len(vectors[0])
        if not all(len(v) == vector_dim for v in vectors):
            raise ValueError("所有向量的维度必须一致")
        
        # 确保 collection 存在
        collection_name = self._ensure_collection(kb_id, vector_dim)
        
        # 准备数据（参考 cloud-edge-milk-tea-agent 的格式）
        data = []
        chunk_ids = []
        
        for i, (text, vector) in enumerate(zip(texts, vectors)):
            chunk_id = f"{kb_id}_chunk_{i}"
            chunk_ids.append(chunk_id)
            
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            
            # 将元数据序列化为 JSON 字符串（参考 cloud-edge-milk-tea-agent）
            metadata_dict = {
                "chunk_id": chunk_id,
                "doc_id": metadata.get("doc_id", ""),
                "kb_id": kb_id,
                "page_num": metadata.get("page_num"),
                "position": metadata.get("position"),
                **metadata.get("metadata", {}),
            }
            metadata_str = json.dumps(metadata_dict, ensure_ascii=False)
            
            data.append({
                "vector": vector,
                "text": text,  # 存储文本内容
                "metadata": metadata_str,  # 存储元数据 JSON 字符串
            })
        
        # 插入到 Milvus（参考 cloud-edge-milk-tea-agent）
        try:
            self.client.insert(
                collection_name=collection_name,
                data=data
            )
        except Exception as e:
            raise RuntimeError(f"插入数据到 Milvus 失败: {str(e)}") from e
        
        return chunk_ids
    
    def search(
        self,
        kb_id: str,
        query_vector: List[float],
        top_k: int = 5,
        filter_condition: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[float, ChunkMetadata]]:
        """
        在指定知识库中搜索相似向量（参考 RAGFlow 的 search 方法和 cloud-edge-milk-tea-agent 的实现）。
        
        Args:
            kb_id: 知识库 ID
            query_vector: 查询向量
            top_k: 返回 top-k 个最相似的结果
            filter_condition: 过滤条件（可选，后续可以扩展）
        
        Returns:
            (相似度分数, ChunkMetadata) 列表，按相似度降序排列
            注意：Milvus 使用余弦相似度，返回的距离越小越相似
        """
        collection_name = self._get_collection_name(kb_id)
        
        # 检查 collection 是否存在
        if not self.client.has_collection(collection_name):
            return []
        
        # 执行搜索（参考 cloud-edge-milk-tea-agent）
        try:
            results = self.client.search(
                collection_name=collection_name,
                data=[query_vector],
                limit=top_k,
                output_fields=["text", "metadata"],
            )
        except Exception as e:
            raise RuntimeError(f"Milvus 搜索失败: {str(e)}") from e
        
        # 处理结果（参考 cloud-edge-milk-tea-agent 的格式）
        search_results = []
        
        if results and len(results) > 0:
            for hit in results[0]:
                # Milvus 返回的距离（对于余弦相似度，距离 = 1 - 相似度）
                # 所以相似度 = 1 - 距离（参考 cloud-edge-milk-tea-agent）
                distance = hit.get("distance", 0)
                similarity_score = 1.0 - distance if distance <= 1 else 1.0 / (1.0 + distance)
                
                # 提取文本和元数据
                text = hit.get("text", hit.get("entity", {}).get("text", ""))
                metadata_str = hit.get("metadata", hit.get("entity", {}).get("metadata", "{}"))
                
                # 解析元数据
                try:
                    metadata_dict = json.loads(metadata_str)
                except Exception:
                    metadata_dict = {}
                
                # 应用过滤条件（简单实现，后续可以扩展）
                if filter_condition:
                    if "doc_id" in filter_condition:
                        if metadata_dict.get("doc_id") != filter_condition["doc_id"]:
                            continue
                
                # 构建 ChunkMetadata
                chunk_metadata = ChunkMetadata(
                    chunk_id=metadata_dict.get("chunk_id", ""),
                    doc_id=metadata_dict.get("doc_id", ""),
                    kb_id=metadata_dict.get("kb_id", kb_id),
                    text=text,
                    page_num=metadata_dict.get("page_num"),
                    position=metadata_dict.get("position"),
                    metadata={k: v for k, v in metadata_dict.items() 
                             if k not in ["chunk_id", "doc_id", "kb_id", "page_num", "position"]},
                )
                
                search_results.append((similarity_score, chunk_metadata))
        
        # 按相似度降序排列（相似度越高越靠前）
        search_results.sort(key=lambda x: x[0], reverse=True)
        
        return search_results
    
    def delete_knowledge_base(self, kb_id: str):
        """删除整个知识库的 collection"""
        collection_name = self._get_collection_name(kb_id)
        
        if self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)
        
        if kb_id in self._collections:
            del self._collections[kb_id]
    
    def get_all_chunks(self, kb_id: str, limit: Optional[int] = None) -> List[ChunkMetadata]:
        """
        获取知识库中的所有 chunks（用于测试集生成等场景）。
        
        Args:
            kb_id: 知识库 ID
            limit: 限制返回的 chunks 数量（None 表示不限制）
        
        Returns:
            ChunkMetadata 列表
        """
        collection_name = self._get_collection_name(kb_id)
        
        if not self.client.has_collection(collection_name):
            return []
        
        try:
            # 使用 Milvus 的 query 功能获取所有数据
            # 注意：Milvus Lite 的 query 可能需要指定 limit，如果没有 limit 参数，我们使用一个很大的数字
            query_limit = limit if limit is not None else 100000  # 默认最多 10 万条
            
            results = self.client.query(
                collection_name=collection_name,
                filter="",  # 不过滤，获取所有
                limit=query_limit,
                output_fields=["text", "metadata"],
            )
            
            chunks = []
            for result in results:
                text = result.get("text", "")
                metadata_str = result.get("metadata", "{}")
                
                # 解析元数据
                try:
                    metadata_dict = json.loads(metadata_str)
                except Exception:
                    metadata_dict = {}
                
                chunk_metadata = ChunkMetadata(
                    chunk_id=metadata_dict.get("chunk_id", ""),
                    doc_id=metadata_dict.get("doc_id", ""),
                    kb_id=metadata_dict.get("kb_id", kb_id),
                    text=text,
                    page_num=metadata_dict.get("page_num"),
                    position=metadata_dict.get("position"),
                    metadata={k: v for k, v in metadata_dict.items() 
                             if k not in ["chunk_id", "doc_id", "kb_id", "page_num", "position"]},
                )
                chunks.append(chunk_metadata)
            
            return chunks
        except Exception as e:
            raise RuntimeError(f"从 Milvus 获取所有 chunks 失败: {str(e)}") from e
    
    def list_all_knowledge_bases(self) -> List[Dict[str, Any]]:
        """
        列出向量数据库中的所有知识库。
        
        Returns:
            知识库信息列表，每个元素包含：
            {
                "kb_id": str,  # 从 collection 名称推断
                "collection_name": str,
                "vector_count": int,
                "vector_dim": int,
            }
        """
        try:
            # 获取所有 collection 名称
            collection_names = self.client.list_collections()
            
            kb_list = []
            for collection_name in collection_names:
                # 从 collection 名称推断 kb_id（格式：kb_{kb_id}）
                if collection_name.startswith("kb_"):
                    kb_id = collection_name[3:].replace("_", "-").replace("_", ".")
                else:
                    kb_id = collection_name
                
                # 获取统计信息
                stats = self.get_stats(kb_id)
                kb_list.append(stats)
            
            return kb_list
        except Exception as e:
            raise RuntimeError(f"列出知识库失败: {str(e)}") from e
    
    def get_document_list(self, kb_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取知识库中的文档列表（按 doc_id 分组）。
        
        Args:
            kb_id: 知识库 ID
            limit: 限制返回的文档数量
        
        Returns:
            文档信息列表，每个元素包含：
            {
                "doc_id": str,
                "chunks_count": int,
                "first_chunk_preview": str,  # 第一个 chunk 的前 100 字符
            }
        """
        chunks = self.get_all_chunks(kb_id, limit=limit * 10)  # 获取更多 chunks 以确保有足够的文档
        
        from collections import defaultdict
        docs_by_id = defaultdict(list)
        for chunk in chunks:
            docs_by_id[chunk.doc_id].append(chunk)
        
        doc_list = []
        for doc_id, chunks_list in list(docs_by_id.items())[:limit]:
            # 按 position 排序
            chunks_sorted = sorted(chunks_list, key=lambda c: c.position if c.position is not None else 0)
            first_chunk_text = chunks_sorted[0].text if chunks_sorted else ""
            
            doc_list.append({
                "doc_id": doc_id,
                "chunks_count": len(chunks_list),
                "first_chunk_preview": first_chunk_text[:100] + "..." if len(first_chunk_text) > 100 else first_chunk_text,
            })
        
        return doc_list
    
    def get_stats(self, kb_id: str) -> Dict[str, Any]:
        """获取知识库的统计信息（参考 cloud-edge-milk-tea-agent 的实现）"""
        collection_name = self._get_collection_name(kb_id)
        
        if not self.client.has_collection(collection_name):
            return {
                "kb_id": kb_id,
                "vector_count": 0,
                "collection_name": collection_name,
            }
        
        try:
            # 尝试获取 collection 信息
            stats = self.client.describe_collection(collection_name)
            num_entities = stats.get("num_entities", 0) if isinstance(stats, dict) else 0
            vector_dim = stats.get("dimension", stats.get("fields", [{}])[0].get("params", {}).get("dim", None)) if isinstance(stats, dict) else None
        except Exception:
            num_entities = 0
            vector_dim = None
        
        return {
            "kb_id": kb_id,
            "vector_count": num_entities,
            "vector_dim": vector_dim,
            "collection_name": collection_name,
        }
    
    def __del__(self):
        """清理连接（参考 cloud-edge-milk-tea-agent）"""
        try:
            if hasattr(self, 'client'):
                self.client.close()
        except Exception:
            pass


__all__ = ["VectorStore", "ChunkMetadata"]

