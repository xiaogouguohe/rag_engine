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
    
    def ingest_document(
        self,
        file_path: str | Path,
        doc_id: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        处理文档：解析 → 分块 → 向量化 → 存储（参考 C8 的实现）。
        
        Args:
            file_path: 文档文件路径
            doc_id: 文档 ID（如果不提供，则自动生成）
            verbose: 是否显示详细进度
        
        Returns:
            处理结果：
            {
                "doc_id": str,
                "chunks_count": int,
                "status": "success"
            }
        """
        import time
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        if verbose:
            print(f"  📄 文件: {file_path.name} ({file_path.stat().st_size} 字节)")
        
        # 1. 加载文档（会自动解析和分块，如果是 Markdown 且启用标题分割）
        if verbose:
            print(f"  ⏳ 步骤 1/6: 加载和解析文档...")
        start_time = time.time()
        
        self.data_module.load_documents([file_path], enhance_metadata=True)
        
        load_time = time.time() - start_time
        if verbose:
            print(f"  ✅ 步骤 1 完成，耗时: {load_time:.2f} 秒")
        
        # 2. 如果还没有分块，进行分块
        if verbose:
            print(f"  ⏳ 步骤 2/6: 文档分块...")
        start_time = time.time()
        
        if not self.data_module.chunks:
            self.data_module.chunk_documents()
        
        chunk_time = time.time() - start_time
        if verbose:
            print(f"  ✅ 步骤 2 完成，耗时: {chunk_time:.2f} 秒")
        
        # 3. 获取该文档的块（通过 parent_id 匹配）
        if verbose:
            print(f"  ⏳ 步骤 3/6: 提取文档块...")
        start_time = time.time()
        
        # 找到刚加载的文档
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
        
        extract_time = time.time() - start_time
        if verbose:
            print(f"  ✅ 步骤 3 完成，找到 {len(doc_chunks)} 个块，耗时: {extract_time:.2f} 秒")
        
        # 4. 提取文本和元数据
        if verbose:
            print(f"  ⏳ 步骤 4/6: 准备向量化数据...")
        start_time = time.time()
        
        texts = [chunk.page_content for chunk in doc_chunks]
        metadatas = [chunk.metadata for chunk in doc_chunks]
        
        prep_time = time.time() - start_time
        if verbose:
            total_text_len = sum(len(t) for t in texts)
            print(f"  ✅ 步骤 4 完成，总文本长度: {total_text_len} 字符，耗时: {prep_time:.2f} 秒")
        
        # 5. 向量化（最可能卡住的地方）
        if verbose:
            print(f"  ⏳ 步骤 5/6: 调用向量化 API（{len(texts)} 个文本块）...")
            print(f"     这可能需要几秒到几十秒，取决于网络和 API 响应速度...")
        start_time = time.time()
        
        try:
            vectors = self.embedding_client.embed_texts(texts, verbose=verbose)
            embed_time = time.time() - start_time
            if verbose:
                print(f"  ✅ 步骤 5 完成，生成 {len(vectors)} 个向量，耗时: {embed_time:.2f} 秒")
        except Exception as e:
            embed_time = time.time() - start_time
            if verbose:
                print(f"  ❌ 步骤 5 失败，耗时: {embed_time:.2f} 秒")
                print(f"     错误详情: {type(e).__name__}: {e}")
            raise RuntimeError(f"向量化失败: {e}") from e
        
        # 6. 存储到向量数据库
        if verbose:
            print(f"  ⏳ 步骤 6/6: 写入向量数据库（Milvus）...")
        start_time = time.time()
        
        try:
            chunk_ids = self.vector_store.add_texts(
                kb_id=self.kb_id,
                texts=texts,
                vectors=vectors,
                metadatas=metadatas,
            )
            store_time = time.time() - start_time
            if verbose:
                print(f"  ✅ 步骤 6 完成，写入 {len(chunk_ids)} 个向量，耗时: {store_time:.2f} 秒")
        except Exception as e:
            store_time = time.time() - start_time
            if verbose:
                print(f"  ❌ 步骤 6 失败，耗时: {store_time:.2f} 秒")
            raise RuntimeError(f"向量数据库写入失败: {e}") from e
        
        total_time = load_time + chunk_time + extract_time + prep_time + embed_time + store_time
        if verbose:
            print(f"  📊 总耗时: {total_time:.2f} 秒")
        
        return {
            "doc_id": parent_id,
            "chunks_count": len(doc_chunks),
            "chunk_ids": chunk_ids,
            "status": "success",
        }
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        问答流程：问题 → 向量化 → 检索 → 生成回答
        
        Args:
            question: 用户问题
            top_k: 检索 top-k 个相关块
            similarity_threshold: 相似度阈值（低于此值的块将被过滤）
            system_prompt: 系统提示词（可选）
        
        Returns:
            回答结果：
            {
                "answer": str,  # LLM 生成的回答
                "sources": List[Dict],  # 检索到的相关块
                "query": str,  # 原始问题
            }
        """
        if not question.strip():
            raise ValueError("问题不能为空")
        
        # 1. 问题向量化
        query_vectors = self.embedding_client.embed_texts([question])
        query_vector = query_vectors[0]
        
        # 2. 检索相关块
        search_results = self.vector_store.search(
            kb_id=self.kb_id,
            query_vector=query_vector,
            top_k=top_k,
        )
        
        # 3. 过滤低相似度的结果
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

