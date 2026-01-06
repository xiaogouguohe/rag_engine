from __future__ import annotations

"""
文档解析器（TXT）
----------------

简单的 TXT 文件解析器，使用 Python 标准库。

注意：Markdown 解析已迁移到 markdown_parser.py，使用 langchain 的 MarkdownHeaderTextSplitter。
"""

from pathlib import Path
from typing import Optional, Dict, Any

from .parser_base import BaseParser


class TxtParser(BaseParser):
    """
    文档解析器，参考 RAGFlow 的 parser 实现。
    
    支持多种文档格式的解析，提取纯文本内容。
    """
    
    def parse_txt(self, file_path: str | Path, encoding: str = "utf-8") -> str:
        """
        解析 TXT 文件。
        
        Args:
            file_path: 文件路径
            encoding: 文件编码，默认 utf-8
        
        Returns:
            文档的文本内容
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        try:
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()
            return content
        except UnicodeDecodeError:
            # 尝试其他编码
            for enc in ["gbk", "gb2312", "latin-1"]:
                try:
                    with open(file_path, "r", encoding=enc) as f:
                        content = f.read()
                    return content
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"无法解析文件编码: {file_path}")
    
    def parse(self, file_path: str | Path, encoding: str = "utf-8", **kwargs) -> Dict[str, Any]:
        """
        解析 TXT 文件。
        
        Args:
            file_path: 文件路径
            encoding: 文件编码，默认 utf-8
            **kwargs: 其他参数
        
        Returns:
            包含文档信息的字典
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        content = self.parse_txt(file_path, encoding=encoding)
        
        return {
            "content": content,
            "file_path": str(file_path),
            "file_type": "txt",
            "file_name": file_path.name,
        }
    
    @classmethod
    def get_supported_extensions(cls) -> list[str]:
        """获取支持的扩展名"""
        return [".txt"]


__all__ = ["DocumentParser"]
