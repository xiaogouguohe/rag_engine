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
import time
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
    parent_id: str
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
    
    def _ensure_collection(self, kb_id: str, vector_dim: int, use_sparse: bool = False):
        """确保 collection 存在，支持多向量字段和标量索引"""
        collection_name = self._get_collection_name(kb_id)
        
        if self.client.has_collection(collection_name):
            # 检查现有 collection 是否包含关键字段
            stats = self.client.describe_collection(collection_name)
            fields = [f.get("name") for f in stats.get("fields", [])]
            
            # 检查架构是否符合预期
            has_sparse = "sparse_vector" in fields
            has_parent_id = "parent_id" in fields
            
            # 如果配置要求使用稀疏向量，但现有集合没有，或者缺少 parent_id 字段，则需要重建
            if (use_sparse and not has_sparse) or (not has_parent_id):
                print(f"     ⚠️  知识库 {kb_id} 架构已更新，正在重建集合...")
                self.client.drop_collection(collection_name)
            else:
                return collection_name
        
        # 使用详细的 Schema 创建支持多向量的 collection
        from pymilvus import DataType
        # 核心改动：auto_id=False，使用自定义的确定性 ID
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        
        # 1. 基础字段
        # 主键改为 VARCHAR，存储 hash(parent_id + position)
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=64, is_primary=True)
        # 独立标量字段：父文档 ID 和 知识库 ID
        schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="kb_id", datatype=DataType.VARCHAR, max_length=64)
        
        # 2. 向量字段
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=vector_dim)
        if use_sparse:
            try:
                schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            except Exception:
                use_sparse = False

        # 3. 文本和元数据
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="metadata", datatype=DataType.VARCHAR, max_length=65535)
        
        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            # 索引参数
            index_params=None # Milvus Lite 会自动创建索引
        )
        
        # 建立索引
        index_params = self.client.prepare_index_params()
        # 密集向量索引
        index_params.add_index(field_name="vector", metric_type="COSINE")
        if use_sparse:
            # Milvus Lite 本地模式对稀疏向量不支持 AUTOINDEX
            # 必须显式指定为 SPARSE_INVERTED_INDEX 或 SPARSE_WAND
            index_params.add_index(
                field_name="sparse_vector", 
                metric_type="IP", 
                index_type="SPARSE_INVERTED_INDEX"
            )
        
        # 核心改动：为标量字段建立索引，加速过滤查询
        index_params.add_index(field_name="parent_id")
        index_params.add_index(field_name="kb_id")
        
        self.client.create_index(collection_name, index_params)
        
        return collection_name

    def add_texts(
        self,
        kb_id: str,
        texts: List[str],
        vectors: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        sparse_vectors: Optional[List[Dict[int, float]]] = None,
        colbert_vectors: Optional[List[Any]] = None,
    ) -> List[str]:
        """
        添加文本和多种向量到指定知识库。支持确定性 ID。
        """
        if not texts or not vectors:
            return []
        
        import hashlib
        vector_dim = len(vectors[0])
        use_sparse = sparse_vectors is not None
        
        # 确保 collection 存在
        collection_name = self._ensure_collection(kb_id, vector_dim, use_sparse=use_sparse)
        
        data = []
        ids = []
        
        for i, (text, vector) in enumerate(zip(texts, vectors)):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            
            # 严谨命名：parent_id 代表原始文档
            parent_id = metadata.get("parent_id") or metadata.get("doc_id") or ""
            position = metadata.get("position") or metadata.get("chunk_index") or 0
            
            # 生成确定性 ID: hash(parent_id + position)
            id_str = f"{parent_id}_{position}"
            deterministic_id = hashlib.md5(id_str.encode()).hexdigest()
            ids.append(deterministic_id)
            
            # 构建元数据 JSON（排除已提拔为独立字段的信息）
            metadata_dict = {
                "chunk_id": metadata.get("chunk_id", deterministic_id),
                **{k: v for k, v in metadata.items() if k not in ["doc_id", "parent_id", "position", "chunk_index", "page_num", "kb_id", "chunk_id"]},
            }
            
            item = {
                "id": deterministic_id,      # 主键
                "parent_id": parent_id,      # 独立标量字段
                "kb_id": kb_id,              # 独立标量字段
                "vector": vector,
                "text": text,
                "metadata": json.dumps(metadata_dict, ensure_ascii=False),
            }
            
            if use_sparse:
                item["sparse_vector"] = sparse_vectors[i]
                
            data.append(item)
        
        try:
            # 使用 upsert 模式，如果 ID 已存在则更新，不存在则插入
            self.client.upsert(collection_name=collection_name, data=data)
        except Exception as e:
            raise RuntimeError(f"写入数据失败: {str(e)}")
        
        return ids
    
    def search(
        self,
        kb_id: str,
        query_vector: List[float],
        top_k: int = 5,
        filter_condition: Optional[Dict[str, Any]] = None,
        query_sparse_vector: Optional[Dict[int, float]] = None,
    ) -> List[Tuple[float, ChunkMetadata]]:
        """
        在指定知识库中搜索相似向量。支持 Dense 和 Sparse 混合检索。
        """
        collection_name = self._get_collection_name(kb_id)
        if not self.client.has_collection(collection_name):
            return []
        
        try:
            # 1. 如果没有稀疏向量，执行常规密集向量搜索
            if query_sparse_vector is None:
                results = self.client.search(
                    collection_name=collection_name,
                    data=[query_vector],
                    limit=top_k,
                    output_fields=["text", "metadata", "parent_id", "kb_id"],
                )
            else:
                # 2. 混合搜索 (Hybrid Search)
                from pymilvus import AnnSearchRequest, RRFRanker
                
                # 密集向量请求
                dense_req = AnnSearchRequest(
                    data=[query_vector],
                    anns_field="vector",
                    param={"metric_type": "COSINE"},
                    limit=top_k * 2 # 扩大范围以便后续融合
                )
                
                # 稀疏向量请求
                sparse_req = AnnSearchRequest(
                    data=[query_sparse_vector],
                    anns_field="sparse_vector",
                    param={"metric_type": "IP"},
                    limit=top_k * 2
                )
                
                # 使用 RRF (Reciprocal Rank Fusion) 进行结果融合
                results = self.client.hybrid_search(
                    collection_name=collection_name,
                    reqs=[dense_req, sparse_req],
                    ranker=RRFRanker(), # 默认 RRF 排名
                    limit=top_k,
                    output_fields=["text", "metadata", "parent_id", "kb_id"]
                )
        except Exception as e:
            # 如果 hybrid_search 报错（可能是某些环境不支持），降级到密集向量搜索
            print(f"     ⚠️  混合检索执行失败，降级到密集检索: {e}")
            results = self.client.search(
                collection_name=collection_name,
                data=[query_vector],
                limit=top_k,
                output_fields=["text", "metadata", "parent_id", "kb_id"],
            )
        
        # 处理结果
        search_results = []
        
        if results and len(results) > 0:
            for hit in results[0]:
                distance = hit.get("distance", 0)
                similarity_score = 1.0 - distance if distance <= 1 else 1.0 / (1.0 + distance)
                
                # 提取字段（优先从 entity 获取，Milvus Lite 2.4+ 的返回结构）
                entity = hit.get("entity", {})
                text = hit.get("text", entity.get("text", ""))
                metadata_str = hit.get("metadata", entity.get("metadata", "{}"))
                parent_id = hit.get("parent_id", entity.get("parent_id", ""))
                hit_kb_id = hit.get("kb_id", entity.get("kb_id", kb_id))
                
                # 解析元数据
                try:
                    metadata_dict = json.loads(metadata_str)
                except Exception:
                    metadata_dict = {}
                
                # 应用过滤条件（标量过滤）
                if filter_condition:
                    if "parent_id" in filter_condition:
                        if parent_id != filter_condition["parent_id"]:
                            continue
                
                # 构建 ChunkMetadata
                chunk_metadata = ChunkMetadata(
                    chunk_id=metadata_dict.get("chunk_id", ""),
                    parent_id=parent_id,
                    kb_id=hit_kb_id,
                    text=text,
                    page_num=metadata_dict.get("page_num"),
                    position=metadata_dict.get("position"),
                    metadata={k: v for k, v in metadata_dict.items() 
                             if k not in ["chunk_id", "parent_id", "kb_id", "page_num", "position"]},
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
            limit: 限制返回的 chunks 数量（None 表示使用最大 limit 16384）
        
        Returns:
            ChunkMetadata 列表
        
        Note:
            Milvus 的 limit 范围是 [1, 16384]。如果数据量超过 16384，需要多次调用此方法。
        """
        collection_name = self._get_collection_name(kb_id)
        
        if not self.client.has_collection(collection_name):
            return []
        
        try:
            # Milvus 的 limit 范围是 [1, 16384]
            MAX_MILVUS_LIMIT = 16384
            
            # 如果指定了 limit，确保不超过 Milvus 的最大值
            # 如果没有指定 limit，使用最大 limit
            query_limit = min(limit, MAX_MILVUS_LIMIT) if limit is not None else MAX_MILVUS_LIMIT
            
            results = self.client.query(
                collection_name=collection_name,
                filter="",  # 不过滤，获取所有
                limit=query_limit,
                output_fields=["text", "metadata", "parent_id", "kb_id"],
            )
            
            chunks = []
            for result in results:
                text = result.get("text", "")
                metadata_str = result.get("metadata", "{}")
                parent_id = result.get("parent_id", "")
                hit_kb_id = result.get("kb_id", kb_id)
                
                # 解析元数据
                try:
                    metadata_dict = json.loads(metadata_str)
                except Exception:
                    metadata_dict = {}
                
                chunk_metadata = ChunkMetadata(
                    chunk_id=metadata_dict.get("chunk_id", ""),
                    parent_id=parent_id,
                    kb_id=hit_kb_id,
                    text=text,
                    page_num=metadata_dict.get("page_num"),
                    position=metadata_dict.get("position"),
                    metadata={k: v for k, v in metadata_dict.items() 
                             if k not in ["chunk_id", "parent_id", "kb_id", "page_num", "position"]},
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
        获取知识库中的文档列表（按 parent_id 分组）。
        
        Args:
            kb_id: 知识库 ID
            limit: 限制返回的文档数量
        
        Returns:
            文档信息列表，每个元素包含：
            {
                "parent_id": str,
                "chunks_count": int,
                "first_chunk_preview": str,  # 第一个 chunk 的前 100 字符
            }
        """
        chunks = self.get_all_chunks(kb_id, limit=limit * 10)  # 获取更多 chunks 以确保有足够的文档
        
        from collections import defaultdict
        docs_by_id = defaultdict(list)
        for chunk in chunks:
            docs_by_id[chunk.parent_id].append(chunk)
        
        doc_list = []
        for parent_id, chunks_list in list(docs_by_id.items())[:limit]:
            # 按 position 排序
            chunks_sorted = sorted(chunks_list, key=lambda c: c.position if c.position is not None else 0)
            first_chunk_text = chunks_sorted[0].text if chunks_sorted else ""
            
            doc_list.append({
                "parent_id": parent_id,
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
            # 在获取统计信息前，先显式 flush 确保数据落盘
            try:
                self.client.flush(collection_name)
            except Exception:
                pass
            
            # 1. 尝试通过查询获取精确数量（最可靠）
            try:
                res = self.client.query(
                    collection_name=collection_name, 
                    filter="", 
                    output_fields=["count(*)"]
                )
                num_entities = res[0]["count(*)"] if res else 0
            except Exception:
                # 2. 如果查询失败，回落到 describe_collection
                stats = self.client.describe_collection(collection_name)
                num_entities = stats.get("num_entities", 0) if isinstance(stats, dict) else 0
            
            # 获取维度
            try:
                stats = self.client.describe_collection(collection_name)
                # 兼容不同版本的返回格式
                vector_dim = stats.get("dimension")
                if vector_dim is None and "fields" in stats:
                    for field in stats["fields"]:
                        if field.get("name") == "vector":
                            vector_dim = field.get("params", {}).get("dim")
                            break
            except Exception:
                vector_dim = None
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

