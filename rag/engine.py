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
from document import DocumentParser, TextChunker
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
    ):
        """
        初始化 RAG 引擎。
        
        Args:
            kb_id: 知识库 ID
            config: 应用配置（如果不提供，则自动加载）
            chunk_size: 分块大小（字符数）
            chunk_overlap: 分块重叠大小（字符数）
        """
        self.kb_id = kb_id
        
        # 加载配置
        if config is None:
            config = AppConfig.load()
        self.config = config
        
        # 初始化各个组件
        self.parser = DocumentParser()
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.embedding_client = EmbeddingClient.from_config(config)
        self.llm_client = LLMClient.from_config(config)
        self.vector_store = VectorStore(storage_path=config.storage_path)
    
    def ingest_document(
        self,
        file_path: str | Path,
        doc_id: Optional[str] = None,
        file_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        处理文档：解析 → 分块 → 向量化 → 存储
        
        Args:
            file_path: 文档文件路径
            doc_id: 文档 ID（如果不提供，则自动生成）
            file_type: 文件类型（可选，自动识别）
        
        Returns:
            处理结果：
            {
                "doc_id": str,
                "chunks_count": int,
                "status": "success"
            }
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 生成文档 ID
        if doc_id is None:
            doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        
        # 1. 解析文档
        doc = self.parser.parse(file_path, file_type=file_type)
        content = doc['content']
        
        if not content.strip():
            raise ValueError(f"文档内容为空: {file_path}")
        
        # 2. 分块
        chunks = self.chunker.chunk_document(
            content,
            metadata={
                "doc_id": doc_id,
                "file_name": doc['file_name'],
                "file_type": doc['file_type'],
            }
        )
        
        if not chunks:
            raise ValueError(f"文档分块失败: {file_path}")
        
        # 3. 提取文本和元数据
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        # 4. 向量化
        vectors = self.embedding_client.embed_texts(texts)
        
        # 5. 存储到向量数据库
        chunk_ids = self.vector_store.add_texts(
            kb_id=self.kb_id,
            texts=texts,
            vectors=vectors,
            metadatas=metadatas,
        )
        
        return {
            "doc_id": doc_id,
            "chunks_count": len(chunks),
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
            "query": question,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        return self.vector_store.get_stats(self.kb_id)
    
    def delete_knowledge_base(self):
        """删除整个知识库"""
        self.vector_store.delete_knowledge_base(self.kb_id)


__all__ = ["RAGEngine"]

