from __future__ import annotations

"""
元数据增强器
-----------

参考 C8 的实现，提供元数据增强功能。

可以提取：
- 文件路径信息（分类、文件名等）
- 文档内容信息（难度、关键词等）
- 自定义元数据
"""

from pathlib import Path
from typing import Dict, Any, Callable, Optional
import re


class MetadataEnhancer:
    """
    元数据增强器，参考 C8 的实现。
    
    提供通用的元数据提取和增强功能。
    """
    
    def __init__(
        self,
        category_mapping: Optional[Dict[str, str]] = None,
        difficulty_patterns: Optional[Dict[str, str]] = None,
        custom_extractors: Optional[list[Callable]] = None,
    ):
        """
        初始化元数据增强器。
        
        Args:
            category_mapping: 分类映射（如 {"meat_dish": "荤菜"}）
            difficulty_patterns: 难度模式（如 {"★★★★★": "非常困难"}）
            custom_extractors: 自定义提取函数列表
        """
        self.category_mapping = category_mapping or {}
        self.difficulty_patterns = difficulty_patterns or self._default_difficulty_patterns()
        self.custom_extractors = custom_extractors or []
    
    @staticmethod
    def _default_difficulty_patterns() -> Dict[str, str]:
        """默认难度模式（参考 C8）"""
        return {
            "★★★★★": "非常困难",
            "★★★★": "困难",
            "★★★": "中等",
            "★★": "简单",
            "★": "非常简单",
        }
    
    def enhance(
        self,
        metadata: Dict[str, Any],
        file_path: Path,
        content: str,
    ) -> Dict[str, Any]:
        """
        增强元数据。
        
        Args:
            metadata: 原始元数据
            file_path: 文件路径
            content: 文档内容
        
        Returns:
            增强后的元数据
        """
        enhanced = metadata.copy()
        
        # 1. 提取文件路径信息
        self._extract_path_info(enhanced, file_path)
        
        # 2. 提取内容信息
        self._extract_content_info(enhanced, content)
        
        # 3. 执行自定义提取器
        for extractor in self.custom_extractors:
            try:
                extracted = extractor(file_path, content)
                if extracted:
                    enhanced.update(extracted)
            except Exception as e:
                # 忽略提取器错误，继续处理
                pass
        
        return enhanced
    
    def _extract_path_info(self, metadata: Dict[str, Any], file_path: Path):
        """从文件路径提取信息（参考 C8）"""
        path_parts = file_path.parts
        
        # 提取分类（从路径中）
        if self.category_mapping:
            metadata.setdefault("category", "其他")
            for key, value in self.category_mapping.items():
                if key in path_parts:
                    metadata["category"] = value
                    break
        
        # 提取文件名（不含扩展名）作为文档名称
        if "file_name" not in metadata:
            metadata["file_name"] = file_path.stem
        
        # 提取文件扩展名
        metadata["file_extension"] = file_path.suffix.lower()
    
    def _extract_content_info(self, metadata: Dict[str, Any], content: str):
        """从文档内容提取信息（参考 C8）"""
        # 提取难度（通过星号模式）
        if self.difficulty_patterns:
            metadata.setdefault("difficulty", "未知")
            for pattern, difficulty in sorted(
                self.difficulty_patterns.items(),
                key=lambda x: len(x[0]),
                reverse=True,  # 从长到短匹配
            ):
                if pattern in content:
                    metadata["difficulty"] = difficulty
                    break
        
        # 提取文档长度
        metadata["content_length"] = len(content)
        
        # 提取行数
        metadata["line_count"] = len(content.splitlines())
    
    @classmethod
    def create_default(cls) -> "MetadataEnhancer":
        """创建默认的元数据增强器"""
        return cls()


__all__ = ["MetadataEnhancer"]

