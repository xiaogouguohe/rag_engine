from __future__ import annotations

"""
应用配置模块
---------------

统一管理本地 RAG 引擎中与模型相关的配置：
- LLM：模型名称、API Key、Base URL 等
- Embedding：模型名称、API Key、Base URL 等

支持从 .env 文件或环境变量读取配置（优先使用 .env 文件）。
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional

# 尝试加载 python-dotenv（如果已安装）
try:
    from dotenv import load_dotenv
    _HAS_DOTENV = True
except ImportError:
    _HAS_DOTENV = False


@dataclass
class LLMConfig:
    """LLM 基本配置（默认以 OpenAI 兼容接口为主）"""

    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4.1-mini"
    timeout: float = 60.0


@dataclass
class EmbeddingConfig:
    """Embedding 模型配置"""

    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "text-embedding-3-small"
    timeout: float = 60.0
    mode: str = "api"  # "api" 或 "local"


@dataclass
class KnowledgeBaseConfig:
    """知识库配置"""
    
    kb_id: str
    source_path: str  # 知识库源文件路径（如：../HowToCook/dishes）
    file_pattern: str = "*.md"  # 文件匹配模式（默认：所有 .md 文件）
    use_markdown_header_split: bool = True  # 是否使用 Markdown 标题分割


@dataclass
class AppConfig:
    """应用级配置入口，后续可以在这里挂更多字段（索引路径、日志等）"""

    llm: LLMConfig
    embedding: EmbeddingConfig
    storage_path: str = "./data/indices"  # 向量索引存储路径
    knowledge_bases: Optional[List[KnowledgeBaseConfig]] = None  # 知识库配置列表

    @classmethod
    def load(cls, *, env_prefix: str = "RAG_", env_file: Optional[str] = None) -> "AppConfig":
        """
        从 .env 文件或环境变量加载配置。
        
        优先级：.env 文件 > 系统环境变量
        
        Args:
            env_prefix: 环境变量前缀，默认 "RAG_"
            env_file: .env 文件路径，默认自动查找项目根目录的 .env 文件
        
        环境变量约定（可以与 RAGFlow 的配置思路保持一致）：
        - LLM：
          - RAG_LLM_API_KEY（必需）
          - RAG_LLM_BASE_URL（可选，默认 OpenAI 官方）
            - OpenAI: "https://api.openai.com/v1"
            - 通义千问: "https://dashscope.aliyuncs.com/compatible-mode/v1" (华北2/北京)
            - 通义千问: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1" (新加坡)
            - DeepSeek: "https://api.deepseek.com/v1"
            - 其他兼容 OpenAI 格式的厂商...
          - RAG_LLM_MODEL（可选，默认 "gpt-4.1-mini"）
            - OpenAI: "gpt-4.1-mini", "gpt-4", "gpt-3.5-turbo" 等
            - 通义千问: "qwen-plus", "qwen-turbo", "qwen-max" 等
          - RAG_LLM_TIMEOUT（可选，默认 60 秒）
        - Embedding：
          - RAG_EMBEDDING_API_KEY（可选，默认回落到 RAG_LLM_API_KEY）
          - RAG_EMBEDDING_BASE_URL（可选，默认与 LLM 一致）
          - RAG_EMBEDDING_MODEL（可选，默认 "text-embedding-3-small"）
            - OpenAI: "text-embedding-3-small", "text-embedding-3-large" 等
            - 通义千问: "text-embedding-v3" 等
          - RAG_EMBEDDING_TIMEOUT（可选，默认 60 秒）
        
        配置方式（两种方式任选其一）：
        
        方式一：使用 .env 文件（推荐，不会提交到 git）
        在项目根目录创建 .env 文件：
        ```bash
        RAG_LLM_API_KEY=sk-xxx
        RAG_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
        RAG_LLM_MODEL=qwen-plus
        RAG_EMBEDDING_MODEL=text-embedding-v3
        ```
        
        方式二：使用系统环境变量
        ```bash
        export RAG_LLM_API_KEY="sk-xxx"
        export RAG_LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
        export RAG_LLM_MODEL="qwen-plus"
        export RAG_EMBEDDING_MODEL="text-embedding-v3"
        ```
        """
        
        # 自动加载 .env 文件（如果存在且已安装 python-dotenv）
        if _HAS_DOTENV:
            if env_file:
                # 使用指定的 .env 文件路径
                load_dotenv(env_file, override=False)  # override=False 表示环境变量优先级更高
            else:
                # 自动查找项目根目录的 .env 文件
                # 从当前文件向上查找，直到找到包含 .env 的目录或到达文件系统根目录
                current_file = Path(__file__).resolve()
                project_root = current_file.parent
                env_path = project_root / ".env"
                if env_path.exists():
                    load_dotenv(env_path, override=False)
                else:
                    # 如果当前目录没有，尝试加载（dotenv 会自动查找）
                    load_dotenv(override=False)

        def _get(name: str, default: Optional[str] = None) -> Optional[str]:
            return os.getenv(f"{env_prefix}{name}", default)

        llm_api_key = _get("LLM_API_KEY")
        if not llm_api_key:
            raise RuntimeError(
                "缺少环境变量 RAG_LLM_API_KEY，用于调用 LLM 接口。"
            )

        llm_cfg = LLMConfig(
            api_key=llm_api_key,
            base_url=_get("LLM_BASE_URL", "https://api.openai.com/v1"),
            model=_get("LLM_MODEL", "gpt-4.1-mini"),
            timeout=float(_get("LLM_TIMEOUT", "60")),
        )

        emb_api_key = _get("EMBEDDING_API_KEY", llm_api_key)
        emb_cfg = EmbeddingConfig(
            api_key=emb_api_key,
            base_url=_get("EMBEDDING_BASE_URL", llm_cfg.base_url),
            model=_get("EMBEDDING_MODEL", "text-embedding-3-small"),
            timeout=float(_get("EMBEDDING_TIMEOUT", "60")),
            mode=_get("EMBEDDING_MODE", "api"), # "api" 或 "local"
        )

        storage_path = _get("STORAGE_PATH", "./data/indices")
        
        # 解析知识库配置（可选，格式：KB_ID:SOURCE_PATH:FILE_PATTERN）
        # 例如：RECIPES_KB:../HowToCook/dishes:*.md
        knowledge_bases = None
        kb_config_str = _get("KNOWLEDGE_BASES")
        if kb_config_str:
            kb_configs = []
            for kb_entry in kb_config_str.split(","):
                parts = kb_entry.strip().split(":")
                if len(parts) >= 2:
                    kb_id = parts[0].strip()
                    source_path = parts[1].strip()
                    file_pattern = parts[2].strip() if len(parts) > 2 else "*.md"
                    kb_configs.append(KnowledgeBaseConfig(
                        kb_id=kb_id,
                        source_path=source_path,
                        file_pattern=file_pattern,
                    ))
            if kb_configs:
                knowledge_bases = kb_configs

        return cls(
            llm=llm_cfg,
            embedding=emb_cfg,
            storage_path=storage_path,
            knowledge_bases=knowledge_bases,
        )


__all__ = ["LLMConfig", "EmbeddingConfig", "AppConfig", "KnowledgeBaseConfig"]

