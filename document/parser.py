from __future__ import annotations

"""
文档解析器
----------

参考 RAGFlow 的实现，提供文档解析功能。

支持格式：
- TXT：纯文本文件
- Markdown：Markdown 格式文件

使用 Python 标准库，不依赖第三方 AI 框架。
"""

from pathlib import Path
from typing import Optional, Dict, Any
import re


class DocumentParser:
    """
    文档解析器，参考 RAGFlow 的 parser 实现。
    
    支持多种文档格式的解析，提取纯文本内容。
    """
    
    @staticmethod
    def parse_txt(file_path: str | Path, encoding: str = "utf-8") -> str:
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
    
    @staticmethod
    def parse_markdown(file_path: str | Path, encoding: str = "utf-8") -> str:
        """
        解析 Markdown 文件。
        
        提取 Markdown 的文本内容（去除 Markdown 语法标记）。
        这是一个简化实现，只提取基本文本内容。
        
        Args:
            file_path: 文件路径
            encoding: 文件编码，默认 utf-8
        
        Returns:
            文档的文本内容（去除 Markdown 语法）
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        with open(file_path, "r", encoding=encoding) as f:
            content = f.read()
        
        # 简单的 Markdown 清理：去除常见的 Markdown 语法
        # 1. 去除标题标记（# ## ### 等）
        content = re.sub(r'^#{1,6}\s+', '', content, flags=re.MULTILINE)
        
        # 2. 去除代码块（```...```）
        content = re.sub(r'```[\s\S]*?```', '', content)
        
        # 3. 去除行内代码（`...`）
        content = re.sub(r'`([^`]+)`', r'\1', content)
        
        # 4. 去除链接标记（[text](url) -> text）
        content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
        
        # 5. 去除图片标记（![alt](url) -> alt）
        content = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', r'\1', content)
        
        # 6. 去除粗体和斜体标记（**text** -> text, *text* -> text）
        content = re.sub(r'\*\*([^\*]+)\*\*', r'\1', content)
        content = re.sub(r'\*([^\*]+)\*', r'\1', content)
        
        # 7. 去除列表标记（- * + 等）
        content = re.sub(r'^[\s]*[-*+]\s+', '', content, flags=re.MULTILINE)
        content = re.sub(r'^[\s]*\d+\.\s+', '', content, flags=re.MULTILINE)
        
        # 8. 去除引用标记（>）
        content = re.sub(r'^>\s+', '', content, flags=re.MULTILINE)
        
        # 9. 去除水平线（---）
        content = re.sub(r'^---+$', '', content, flags=re.MULTILINE)
        
        # 10. 清理多余的空行
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        return content.strip()
    
    @classmethod
    def parse(cls, file_path: str | Path, file_type: Optional[str] = None) -> Dict[str, Any]:
        """
        解析文档（自动识别文件类型）。
        
        Args:
            file_path: 文件路径
            file_type: 文件类型（可选，如果不提供则自动识别）
        
        Returns:
            包含文档信息的字典：
            {
                "content": str,  # 文档文本内容
                "file_path": str,  # 文件路径
                "file_type": str,  # 文件类型
                "file_name": str,  # 文件名
            }
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 自动识别文件类型
        if file_type is None:
            suffix = file_path.suffix.lower()
            if suffix == ".txt":
                file_type = "txt"
            elif suffix in [".md", ".markdown"]:
                file_type = "markdown"
            else:
                # 默认尝试作为 TXT 处理
                file_type = "txt"
        
        # 根据文件类型解析
        if file_type == "txt":
            content = cls.parse_txt(file_path)
        elif file_type == "markdown":
            content = cls.parse_markdown(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {file_type}")
        
        return {
            "content": content,
            "file_path": str(file_path),
            "file_type": file_type,
            "file_name": file_path.name,
        }


__all__ = ["DocumentParser"]
