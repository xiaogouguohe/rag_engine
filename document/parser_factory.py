from __future__ import annotations

"""
解析器工厂
----------

根据文件类型自动选择合适的解析器。

参考 RAGFlow 的设计思路。
"""

from pathlib import Path
from typing import Dict, Type, Optional

from .parser_base import BaseParser
from .parser import TxtParser
from .markdown_parser import MarkdownParser


class ParserFactory:
    """
    解析器工厂，根据文件类型自动选择解析器。
    """
    
    # 注册的解析器（扩展名 -> 解析器类）
    _parsers: Dict[str, Type[BaseParser]] = {
        ".txt": TxtParser,
        ".md": MarkdownParser,
        ".markdown": MarkdownParser,
        ".mdx": MarkdownParser,
    }
    
    @classmethod
    def register_parser(cls, extension: str, parser_class: Type[BaseParser]):
        """
        注册解析器。
        
        Args:
            extension: 文件扩展名（如 ".html"）
            parser_class: 解析器类
        """
        cls._parsers[extension.lower()] = parser_class
    
    @classmethod
    def get_parser(cls, file_path: str | Path) -> Optional[BaseParser]:
        """
        根据文件路径获取合适的解析器。
        
        Args:
            file_path: 文件路径
        
        Returns:
            解析器实例，如果找不到则返回 None
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        parser_class = cls._parsers.get(suffix)
        if parser_class:
            return parser_class()
        return None
    
    @classmethod
    def parse(cls, file_path: str | Path, **kwargs) -> Dict:
        """
        自动选择解析器并解析文件。
        
        Args:
            file_path: 文件路径
            **kwargs: 传递给解析器的参数
        
        Returns:
            解析结果字典
        """
        parser = cls.get_parser(file_path)
        if parser is None:
            # 默认使用 TXT 解析器
            parser = TxtParser()
        
        return parser.parse(file_path, **kwargs)
    
    @classmethod
    def get_supported_extensions(cls) -> list[str]:
        """获取所有支持的扩展名"""
        return list(cls._parsers.keys())


__all__ = ["ParserFactory"]

