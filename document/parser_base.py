from __future__ import annotations

"""
解析器抽象基类
--------------

定义统一的解析器接口，支持多种文档格式的扩展。

参考 RAGFlow 的设计思路，但提供明确的抽象接口。
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional


class BaseParser(ABC):
    """
    解析器抽象基类。
    
    所有具体的解析器都应该继承这个类并实现 parse 方法。
    """
    
    @abstractmethod
    def parse(self, file_path: str | Path, **kwargs) -> Dict[str, Any]:
        """
        解析文档。
        
        Args:
            file_path: 文件路径
            **kwargs: 其他解析参数（如编码、配置等）
        
        Returns:
            包含文档信息的字典：
            {
                "content": str,      # 文档文本内容
                "file_path": str,    # 文件路径
                "file_type": str,    # 文件类型
                "file_name": str,    # 文件名
                "metadata": dict,    # 额外的元数据（可选）
            }
        """
        pass
    
    @classmethod
    def get_supported_extensions(cls) -> list[str]:
        """
        获取支持的文件扩展名列表。
        
        Returns:
            支持的扩展名列表，如 [".txt", ".md"]
        """
        return []
    
    def can_parse(self, file_path: str | Path) -> bool:
        """
        检查是否可以解析该文件。
        
        Args:
            file_path: 文件路径
        
        Returns:
            是否可以解析
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        return suffix in self.get_supported_extensions()


__all__ = ["BaseParser"]

