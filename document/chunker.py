from __future__ import annotations

"""
文本分块器
----------

参考 RAGFlow 的实现，提供文本分块功能。

分块策略：
- 固定大小分块（按字符数或 token 数）
- 支持重叠窗口（overlap）
- 尽量在句子边界处切分
"""

from typing import List, Dict, Any, Optional
import re


class TextChunker:
    """
    文本分块器，参考 RAGFlow 的分块策略。
    
    将长文本分割成固定大小的块，支持重叠窗口。
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
    ):
        """
        初始化分块器。
        
        Args:
            chunk_size: 每个块的大小（字符数）
            chunk_overlap: 块之间的重叠大小（字符数）
            separators: 分隔符列表，用于在合适的位置切分（默认：句号、换行等）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 默认分隔符（参考 RAGFlow 的 delimiter）
        if separators is None:
            # 中英文句号、问号、感叹号、分号、换行符
            separators = ["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";"]
        self.separators = separators
    
    def split_text(self, text: str) -> List[Dict[str, Any]]:
        """
        将文本分割成块。
        
        Args:
            text: 要分割的文本
        
        Returns:
            块列表，每个块包含：
            {
                "text": str,  # 块文本
                "start": int,  # 在原文本中的起始位置
                "end": int,  # 在原文本中的结束位置
            }
        """
        if not text:
            return []
        
        text = text.strip()
        if len(text) <= self.chunk_size:
            return [{
                "text": text,
                "start": 0,
                "end": len(text),
            }]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # 计算当前块的结束位置
            end = min(start + self.chunk_size, len(text))
            
            # 如果不是最后一块，尝试在分隔符处切分
            if end < len(text):
                # 从结束位置向前查找分隔符
                chunk_text = text[start:end]
                
                # 查找最后一个分隔符的位置
                last_sep_pos = -1
                for sep in self.separators:
                    pos = chunk_text.rfind(sep)
                    if pos > last_sep_pos:
                        last_sep_pos = pos
                
                # 如果找到分隔符，在分隔符后切分
                if last_sep_pos > self.chunk_size * 0.5:  # 至少保留一半内容
                    end = start + last_sep_pos + len(self._find_separator(chunk_text, last_sep_pos))
            
            # 提取块文本
            chunk_text = text[start:end].strip()
            
            if chunk_text:  # 只添加非空块
                chunks.append({
                    "text": chunk_text,
                    "start": start,
                    "end": end,
                })
            
            # 计算下一个块的起始位置（考虑重叠）
            if end >= len(text):
                break
            
            # 下一个块的起始位置 = 当前结束位置 - 重叠大小
            start = max(start + 1, end - self.chunk_overlap)
        
        return chunks
    
    def _find_separator(self, text: str, pos: int) -> str:
        """查找位置 pos 处的分隔符"""
        for sep in self.separators:
            if text[pos:pos+len(sep)] == sep:
                return sep
        return ""
    
    def chunk_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        分块文档内容，并附加元数据。
        
        Args:
            content: 文档内容
            metadata: 文档元数据（可选）
        
        Returns:
            块列表，每个块包含：
            {
                "text": str,  # 块文本
                "start": int,  # 在原文本中的起始位置
                "end": int,  # 在原文本中的结束位置
                "metadata": dict,  # 元数据（包含文档信息）
            }
        """
        chunks = self.split_text(content)
        
        # 为每个块添加元数据和索引
        result = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                "chunk_index": i,
                "total_chunks": len(chunks),
                **(metadata or {}),
            }
            
            result.append({
                **chunk,
                "metadata": chunk_metadata,
            })
        
        return result


__all__ = ["TextChunker"]
