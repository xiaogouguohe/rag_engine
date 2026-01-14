from __future__ import annotations

"""
数据准备模块
-----------

整合文档解析、分块和元数据增强功能。

参考 C8 的 DataPreparationModule 实现。
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import uuid
import hashlib

from langchain_core.documents import Document

from .parser_factory import ParserFactory
from .markdown_parser import MarkdownParser
from .chunker import TextChunker
from .metadata_enhancer import MetadataEnhancer


class DataPreparationModule:
    """
    数据准备模块，参考 C8 的实现。
    
    功能：
    - 加载文档
    - 解析文档（支持多种格式）
    - 分块（支持 Markdown 标题分割）
    - 元数据增强
    - 建立父子文档关系
    """
    
    def __init__(
        self,
        metadata_enhancer: Optional[MetadataEnhancer] = None,
        use_markdown_header_split: bool = True,
    ):
        """
        初始化数据准备模块。
        
        Args:
            metadata_enhancer: 元数据增强器（可选）
            use_markdown_header_split: 是否对 Markdown 使用标题分割（参考 C8）
        """
        self.metadata_enhancer = metadata_enhancer or MetadataEnhancer.create_default()
        self.use_markdown_header_split = use_markdown_header_split
        
        # 存储文档
        self.documents: List[Document] = []  # 父文档列表
        self.chunks: List[Document] = []     # 子文档列表
        self.parent_child_map: Dict[str, str] = {}  # 子块ID -> 父文档ID的映射

    @staticmethod
    def find_files(directory: str | Path, pattern: str = "*.md") -> List[Path]:
        """
        递归查找目录中匹配模式的文件。
        
        Args:
            directory: 目录路径
            pattern: 文件匹配模式（默认：*.md）
        
        Returns:
            文件路径列表
        """
        directory = Path(directory)
        if not directory.exists() or not directory.is_dir():
            return []
        
        files = []
        for file_path in directory.rglob(pattern):
            if file_path.is_file():
                files.append(file_path)
        
        return sorted(files)
    
    def load_documents(
        self,
        file_paths: List[str | Path],
        enhance_metadata: bool = True,
    ) -> List[Document]:
        """
        加载文档列表（参考 C8 的实现）。
        
        Args:
            file_paths: 文件路径列表
            enhance_metadata: 是否增强元数据
        
        Returns:
            父文档列表
        """
        documents = []
        
        for file_path in file_paths:
            file_path = Path(file_path)
            
            if not file_path.exists():
                continue
            
            # 使用解析器工厂解析文档
            parser = ParserFactory.get_parser(file_path)
            if parser is None:
                # 默认使用 TXT 解析器
                from .parser import TxtParser
                parser = TxtParser()
            
            # 如果是 Markdown 且使用标题分割，使用特殊方法
            if isinstance(parser, MarkdownParser) and self.use_markdown_header_split:
                # 定义元数据增强函数
                def enhance_fn(metadata, file_path, content):
                    return self.metadata_enhancer.enhance(metadata, file_path, content)
                
                parent_doc, child_docs = parser.parse_to_documents(
                    file_path,
                    enhance_metadata=enhance_fn if enhance_metadata else None,
                )
                documents.append(parent_doc)
                self.chunks.extend(child_docs)
                
                # 建立父子映射
                parent_id = parent_doc.metadata.get("parent_id")
                for child_doc in child_docs:
                    child_id = child_doc.metadata.get("chunk_id")
                    if child_id and parent_id:
                        self.parent_child_map[child_id] = parent_id
            else:
                # 普通解析方式
                result = parser.parse(file_path)
                
                # 生成父文档 ID
                try:
                    relative_path = file_path.resolve().as_posix()
                    parent_id = hashlib.md5(relative_path.encode("utf-8")).hexdigest()
                except Exception:
                    parent_id = str(uuid.uuid4())
                
                # 创建父文档
                metadata = {
                    "source": str(file_path),
                    "parent_id": parent_id,
                    "doc_type": "parent",
                    "file_name": result["file_name"],
                    "file_type": result["file_type"],
                }
                
                # 元数据增强
                if enhance_metadata:
                    metadata = self.metadata_enhancer.enhance(
                        metadata,
                        file_path,
                        result["content"],
                    )
                
                parent_doc = Document(
                    page_content=result["content"],
                    metadata=metadata,
                )
                documents.append(parent_doc)
        
        self.documents = documents
        return documents
    
    def chunk_documents(
        self,
        use_markdown_header_split: Optional[bool] = None,
    ) -> List[Document]:
        """
        对已加载的文档进行分块（参考 C8 的实现）。
        
        Args:
            use_markdown_header_split: 是否使用 Markdown 标题分割（覆盖初始化时的设置）
        
        Returns:
            子文档列表
        """
        if not self.documents:
            raise ValueError("请先加载文档")
        
        use_header_split = (
            use_markdown_header_split
            if use_markdown_header_split is not None
            else self.use_markdown_header_split
        )
        
        # 如果已经使用 Markdown 标题分割加载，chunks 已经生成
        if use_header_split and self.chunks:
            return self.chunks
        
        # 否则，使用普通分块器
        all_chunks = []
        chunker = TextChunker(
            use_markdown_header_split=use_header_split,
        )
        
        for doc in self.documents:
            file_type = doc.metadata.get("file_type", "txt")
            parent_id = doc.metadata.get("parent_id")
            
            # 分块
            chunks = chunker.chunk_document(
                doc.page_content,
                metadata=doc.metadata,
                file_type=file_type,
            )
            
            # 转换为 Document 对象
            for i, chunk in enumerate(chunks):
                child_id = str(uuid.uuid4())
                
                child_metadata = {
                    **chunk["metadata"],
                    "chunk_id": child_id,
                    "parent_id": parent_id,
                    "doc_type": "child",
                    "chunk_index": i,
                }
                
                child_doc = Document(
                    page_content=chunk["text"],
                    metadata=child_metadata,
                )
                
                all_chunks.append(child_doc)
                
                # 建立父子映射
                if parent_id:
                    self.parent_child_map[child_id] = parent_id
        
        self.chunks = all_chunks
        return all_chunks
    
    def get_parent_documents(self, child_chunks: List[Document]) -> List[Document]:
        """
        根据子块获取对应的父文档（智能去重，参考 C8 的实现）。
        
        Args:
            child_chunks: 检索到的子块列表
        
        Returns:
            对应的父文档列表（去重，按相关性排序）
        """
        # 统计每个父文档被匹配的次数（相关性指标）
        parent_relevance = {}
        parent_docs_map = {}
        
        # 收集所有相关的父文档ID和相关性分数
        for chunk in child_chunks:
            parent_id = chunk.metadata.get("parent_id")
            if parent_id:
                # 增加相关性计数
                parent_relevance[parent_id] = parent_relevance.get(parent_id, 0) + 1
                
                # 缓存父文档（避免重复查找）
                if parent_id not in parent_docs_map:
                    for doc in self.documents:
                        if doc.metadata.get("parent_id") == parent_id:
                            parent_docs_map[parent_id] = doc
                            break
        
        # 按相关性排序（匹配次数多的排在前面）
        sorted_parent_ids = sorted(
            parent_relevance.keys(),
            key=lambda x: parent_relevance[x],
            reverse=True,
        )
        
        # 构建去重后的父文档列表
        parent_docs = []
        for parent_id in sorted_parent_ids:
            if parent_id in parent_docs_map:
                parent_docs.append(parent_docs_map[parent_id])
        
        return parent_docs
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据统计信息（参考 C8）"""
        if not self.documents:
            return {}
        
        # 统计分类和难度（如果存在）
        categories = {}
        difficulties = {}
        
        for doc in self.documents:
            category = doc.metadata.get("category")
            if category:
                categories[category] = categories.get(category, 0) + 1
            
            difficulty = doc.metadata.get("difficulty")
            if difficulty:
                difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
        
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "categories": categories,
            "difficulties": difficulties,
        }


__all__ = ["DataPreparationModule"]

