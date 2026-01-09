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

    def embed_texts(self, texts: List[str], verbose: bool = False, batch_size: int = 10) -> List[Vector]:
        """
        将一批文本转换为向量（参考 RAGFlow 的 encode 方法）。

        - texts: 文本列表
        - verbose: 是否显示详细日志
        - batch_size: 批处理大小（某些 API 如通义千问限制最多 10 个）
        - 返回：与输入顺序一一对应的向量列表
        """
        if not texts:
            return []

        import time
        
        if verbose:
            total_chars = sum(len(t) for t in texts)
            print(f"     准备调用 API: {len(texts)} 个文本，总长度 {total_chars} 字符")
            print(f"     API: {self.cfg.base_url}")
            print(f"     模型: {self.cfg.model}")
            print(f"     超时设置: {self.cfg.timeout} 秒")
            print(f"     批处理大小: {batch_size}（如果文本数超过此值，将分批处理）")

        try:
            start_time = time.time()
            all_vectors: List[Vector] = []
            
            # 如果文本数量超过 batch_size，需要分批处理
            if len(texts) > batch_size:
                if verbose:
                    num_batches = (len(texts) + batch_size - 1) // batch_size
                    print(f"     ⚠️  文本数量 ({len(texts)}) 超过批处理大小 ({batch_size})，将分 {num_batches} 批处理")
                
                # 分批处理
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_num = i // batch_size + 1
                    total_batches = (len(texts) + batch_size - 1) // batch_size
                    
                    if verbose:
                        print(f"     ⏳ 批次 {batch_num}/{total_batches}: 处理 {len(batch_texts)} 个文本...")
                    
                    batch_start = time.time()
                    response = self.client.embeddings.create(
                        model=self.cfg.model,
                        input=batch_texts,
                    )
                    batch_time = time.time() - batch_start
                    
                    batch_vectors = [item.embedding for item in response.data]
                    all_vectors.extend(batch_vectors)
                    
                    if verbose:
                        print(f"     ✅ 批次 {batch_num} 完成，获得 {len(batch_vectors)} 个向量，耗时: {batch_time:.2f} 秒")
            else:
                # 文本数量不超过 batch_size，直接处理
                if verbose:
                    print(f"     ⏳ 正在发送请求到 API...")
                
                response = self.client.embeddings.create(
                    model=self.cfg.model,
                    input=texts,
                )
                
                if verbose:
                    print(f"     ✅ API 响应成功")
                
                # OpenAI 兼容格式：data[i].embedding
                if verbose:
                    print(f"     ⏳ 解析响应数据...")
                
                all_vectors = [item.embedding for item in response.data]
            
            api_time = time.time() - start_time
            
            if len(all_vectors) != len(texts):
                raise RuntimeError(
                    f"Embedding 数量与输入不一致: {len(all_vectors)} vs {len(texts)}"
                )
            
            if verbose:
                print(f"     ✅ 全部完成，获得 {len(all_vectors)} 个向量，总耗时: {api_time:.2f} 秒")
            
            return all_vectors
            
        except Exception as e:
            api_time = time.time() - start_time if 'start_time' in locals() else 0
            if verbose:
                print(f"     ❌ API 调用失败，耗时: {api_time:.2f} 秒")
                print(f"     错误类型: {type(e).__name__}")
                print(f"     错误信息: {str(e)[:200]}...")
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

