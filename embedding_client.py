from __future__ import annotations

"""
EmbeddingClient
---------------

参考 RAGFlow 的实现方式，使用 OpenAI SDK 作为基础客户端。

与 RAGFlow 中的 embedding_model 类似，这里只关注「给定一批文本 → 返回一批向量」。
"""

from dataclasses import dataclass
from typing import List
import os

from openai import OpenAI, AsyncOpenAI

from config import AppConfig, EmbeddingConfig


Vector = List[float]


@dataclass
class EmbeddingClient:
    """
    Embedding 客户端，参考 RAGFlow 的 EmbeddingModel 实现。
    
    使用 OpenAI SDK，通过 base_url 适配不同厂商的 embedding 接口。
    """
    
    cfg: EmbeddingConfig
    client: OpenAI
    async_client: AsyncOpenAI

    @classmethod
    def from_config(cls, app_cfg: AppConfig) -> "EmbeddingClient":
        """从配置创建客户端（参考 RAGFlow 的初始化方式）"""
        cfg = app_cfg.embedding
        timeout = int(os.environ.get("LLM_TIMEOUT_SECONDS", int(cfg.timeout)))
        
        client = OpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            timeout=timeout,
        )
        async_client = AsyncOpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            timeout=timeout,
        )
        
        return cls(cfg=cfg, client=client, async_client=async_client)

    def embed_texts(self, texts: List[str]) -> List[Vector]:
        """
        将一批文本转换为向量（参考 RAGFlow 的 encode 方法）。

        - texts: 文本列表
        - 返回：与输入顺序一一对应的向量列表
        """
        if not texts:
            return []

        try:
            # 使用 OpenAI SDK 调用（兼容所有支持 OpenAI 格式的厂商）
            response = self.client.embeddings.create(
                model=self.cfg.model,
                input=texts,
            )
            
            # OpenAI 兼容格式：data[i].embedding
            vectors: List[Vector] = [item.embedding for item in response.data]
            
            if len(vectors) != len(texts):
                raise RuntimeError(
                    f"Embedding 数量与输入不一致: {len(vectors)} vs {len(texts)}"
                )
            
            return vectors
            
        except Exception as e:
            raise RuntimeError(f"Embedding 调用失败: {str(e)}") from e

    async def async_embed_texts(self, texts: List[str]) -> List[Vector]:
        """异步向量化"""
        if not texts:
            return []

        try:
            response = await self.async_client.embeddings.create(
                model=self.cfg.model,
                input=texts,
            )
            
            vectors: List[Vector] = [item.embedding for item in response.data]
            
            if len(vectors) != len(texts):
                raise RuntimeError(
                    f"Embedding 数量与输入不一致: {len(vectors)} vs {len(texts)}"
                )
            
            return vectors
            
        except Exception as e:
            raise RuntimeError(f"Embedding 异步调用失败: {str(e)}") from e


__all__ = ["EmbeddingClient", "Vector"]

