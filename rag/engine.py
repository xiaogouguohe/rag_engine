from __future__ import annotations

"""
RAG 引擎
--------

整合所有模块，实现完整的 RAG 流程：
1. 文档处理流程：文档 → 解析 → 分块 → 向量化 → 存储
2. 问答流程：问题 → 向量化 → 检索 → 生成回答

参考 RAGFlow 的设计思路，但大幅简化实现。
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid
import numpy as np

from config import AppConfig
from document import (
    ParserFactory,
    TextChunker,
    MetadataEnhancer,
    DataPreparationModule,
)
from embedding import EmbeddingClient
from llm import LLMClient
from vector_store import VectorStore


class RAGEngine:
    """
    RAG 引擎核心类，整合所有模块。
    
    功能：
    - 文档处理：解析、分块、向量化、存储
    - 问答：检索、生成回答
    """
    
    def __init__(
        self,
        kb_id: str,
        config: Optional[AppConfig] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        use_markdown_header_split: bool = True,
        metadata_enhancer: Optional[MetadataEnhancer] = None,
    ):
        """
        初始化 RAG 引擎。
        
        Args:
            kb_id: 知识库 ID
            config: 应用配置（如果不提供，则自动加载）
            chunk_size: 分块大小（字符数）
            chunk_overlap: 分块重叠大小（字符数）
            use_markdown_header_split: 是否对 Markdown 使用标题分割（参考 C8）
            metadata_enhancer: 元数据增强器（可选）
        """
        self.kb_id = kb_id
        
        # 加载配置
        if config is None:
            config = AppConfig.load()
        self.config = config
        
        # 初始化各个组件
        self.data_module = DataPreparationModule(
            metadata_enhancer=metadata_enhancer,
            use_markdown_header_split=use_markdown_header_split,
        )
        self.embedding_client = EmbeddingClient.from_config(config)
        self.llm_client = LLMClient.from_config(config)
        self.vector_store = VectorStore(storage_path=config.storage_path)
        
        # 记录该知识库的特定配置
        self.kb_config = None
        if self.config.knowledge_bases:
            for kb in self.config.knowledge_bases:
                if kb.kb_id == kb_id:
                    self.kb_config = kb
                    break
        
        # 设置默认检索数量
        self.default_top_k = self.kb_config.top_k if self.kb_config else 4
    
    def ingest_document(
        self,
        file_path: str | Path,
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        处理单篇文档：解析 → 分块 → 向量化 → 存储。
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 1. 清理之前的状态，确保只处理当前文档
        self.data_module.documents = []
        self.data_module.chunks = []
        
        # 2. 加载文档
        self.data_module.load_documents([file_path], enhance_metadata=True)
        
        # 3. 进行分块
        if not self.data_module.chunks:
            self.data_module.chunk_documents()
        
        # 4. 获取该文档的块
        parent_doc = None
        for doc in self.data_module.documents:
            if str(file_path) in doc.metadata.get("source", ""):
                parent_doc = doc
                break
        
        if not parent_doc:
            raise ValueError(f"文档加载失败: {file_path}")
        
        parent_id = parent_doc.metadata.get("parent_id")
        doc_chunks = [
            chunk for chunk in self.data_module.chunks
            if chunk.metadata.get("parent_id") == parent_id
        ]
        
        if not doc_chunks:
            raise ValueError(f"文档分块失败: {file_path}")
        
        # 5. 提取文本和元数据
        texts = [chunk.page_content for chunk in doc_chunks]
        metadatas = [chunk.metadata for chunk in doc_chunks]
        
        # 6. 向量化 (根据配置决定是否生成多种向量)
        use_sparse = self.kb_config.use_sparse if self.kb_config else False
        # 存储时不包含 multi_vector
        
        emb_results = self.embedding_client.embed_texts(
            texts, 
            return_sparse=use_sparse,
            return_multi=False
        )
        
        # 7. 存储到向量数据库
        chunk_ids = self.vector_store.add_texts(
            kb_id=self.kb_id,
            texts=texts,
            vectors=emb_results["dense_vecs"],
            metadatas=metadatas,
            sparse_vectors=emb_results.get("sparse_vecs")
        )
        
        return {
            "doc_id": parent_id,
            "chunks_count": len(doc_chunks),
            "chunk_ids": chunk_ids,
            "status": "success",
        }

    def ingest_directory(
        self,
        dir_path: str | Path,
        pattern: str = "*.md",
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        批量处理目录下的文档。
        
        Args:
            dir_path: 目录路径
            pattern: 文件匹配模式
            verbose: 是否打印进度
            
        Returns:
            处理统计结果
        """
        dir_path = Path(dir_path)
        if not dir_path.exists() or not dir_path.is_dir():
            raise FileNotFoundError(f"目录不存在或不是目录: {dir_path}")

        files = sorted(list(dir_path.rglob(pattern)))
        if not files:
            return {
                "total_files": 0,
                "success_count": 0,
                "fail_count": 0,
                "total_chunks": 0,
            }

        if verbose:
            print(f"开始加载目录: {dir_path} (找到 {len(files)} 个文件)")
            print("-" * 40)

        success_count = 0
        fail_count = 0
        total_chunks = 0

        for i, file_path in enumerate(files, 1):
            if verbose:
                try:
                    rel_path = file_path.relative_to(dir_path)
                except ValueError:
                    rel_path = file_path.name
                print(f"[{i}/{len(files)}] 处理: {rel_path}", end=" ... ", flush=True)
            
            try:
                result = self.ingest_document(file_path)
                success_count += 1
                total_chunks += result["chunks_count"]
                if verbose:
                    print(f"✅ ({result['chunks_count']} 块)")
            except Exception as e:
                fail_count += 1
                if verbose:
                    print(f"❌ 失败: {e}")

        return {
            "total_files": len(files),
            "success_count": success_count,
            "fail_count": fail_count,
            "total_chunks": total_chunks,
            "status": "completed"
        }
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        similarity_threshold: float = 0.0,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        问答流程：支持密集、稀疏、多向量检索。
        """
        if not question.strip():
            raise ValueError("问题不能为空")
        
        # 0. 获取配置
        final_top_k = top_k if top_k is not None else self.default_top_k
        use_sparse = self.kb_config.use_sparse if self.kb_config else False
        use_multi = self.kb_config.use_multi_vector if self.kb_config else False
        
        # 1. 问题向量化
        emb_results = self.embedding_client.embed_texts(
            [question], 
            return_sparse=use_sparse,
            return_multi=use_multi
        )
        query_dense = emb_results["dense_vecs"][0]
        query_sparse = emb_results.get("sparse_vecs", [None])[0]
        query_multi = emb_results.get("multi_vecs", [None])[0]
        
        # 2. 检索相关块 (混合检索: Dense + Sparse)
        search_results = self.vector_store.search(
            kb_id=self.kb_id,
            query_vector=query_dense,
            top_k=final_top_k * 3 if use_multi else final_top_k, # 如果有精排，先多捞点
            query_sparse_vector=query_sparse,
        )
        
        # 3. 如果启用多向量精排 (ColBERT Online Rerank)
        if use_multi and query_multi is not None and search_results:
            print(f"     🎯 正在对 {len(search_results)} 个候选片段进行在线多向量精排 (ColBERT)...")
            
            # 获取候选片段的原始文本
            candidate_texts = [res[1].text for res in search_results]
            
            # 现场计算候选片段的多向量 (Online Encoding)
            # 注意：这里只计算几十个片段，速度会很快
            candidate_emb = self.embedding_client.embed_texts(
                candidate_texts, 
                return_dense=False, 
                return_sparse=False, 
                return_multi=True
            )
            candidate_multi_vecs = candidate_emb.get("multi_vecs")
            
            reranked_results = []
            if candidate_multi_vecs is not None:
                for i, (score, metadata) in enumerate(search_results):
                    doc_multi = candidate_multi_vecs[i]
                    
                    # 计算 ColBERT MaxSim 分数
                    # query_multi: [q_len, dim], doc_multi: [d_len, dim]
                    sim_matrix = np.matmul(query_multi, doc_multi.T)
                    max_sim_score = np.mean(np.max(sim_matrix, axis=1))
                    
                    # 融合分数
                    final_score = score * 0.3 + max_sim_score * 0.7
                    reranked_results.append((final_score, metadata))
                
                # 重新排序并取最终 top_k
                reranked_results.sort(key=lambda x: x[0], reverse=True)
                search_results = reranked_results[:final_top_k]
            else:
                search_results = search_results[:final_top_k]
        else:
            # 如果没开精排，直接取 top_k
            search_results = search_results[:final_top_k]
        
        # 4. 过滤低相似度的结果
        filtered_results = [
            (score, metadata) for score, metadata in search_results
            if score >= similarity_threshold
        ]
        
        if not filtered_results:
            return {
                "answer": "抱歉，没有找到相关信息。",
                "sources": [],
                "query": question,
            }
        
        # 4. 构建上下文
        context_chunks = []
        for score, metadata in filtered_results:
            context_chunks.append({
                "text": metadata.text,
                "score": score,
                "doc_id": metadata.doc_id,
                "chunk_id": metadata.chunk_id,
            })
        
        # 5. 拼接上下文和问题
        context = "\n\n".join([
            f"[文档片段 {i+1}]\n{chunk['text']}"
            for i, chunk in enumerate(context_chunks)
        ])
        
        # 6. 构建提示词
        if system_prompt is None:
            system_prompt = """你是一个专业的 AI 助手。请根据提供的文档片段回答问题。
如果文档中没有相关信息，请诚实地说不知道。"""
        
        prompt = f"""基于以下文档片段回答问题：

{context}

问题：{question}

请基于上述文档片段回答问题，如果文档中没有相关信息，请说明。"""
        
        # 7. 生成回答
        answer = self.llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.1,
        )
        
        return {
            "answer": answer,
            "sources": context_chunks,
            # 向后兼容：有些评估脚本或上层会用 chunks 表示检索到的上下文块
            "chunks": context_chunks,
            "query": question,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        return self.vector_store.get_stats(self.kb_id)
    
    def delete_knowledge_base(self):
        """删除整个知识库"""
        self.vector_store.delete_knowledge_base(self.kb_id)


__all__ = ["RAGEngine"]
