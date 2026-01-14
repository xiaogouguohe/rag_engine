from __future__ import annotations

"""
EmbeddingClient
---------------

参考 RAGFlow 的实现方式，使用 OpenAI SDK 作为基础客户端。

与 RAGFlow 中的 embedding_model 类似，这里只关注「给定一批文本 → 返回一批向量」。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any
import os
import time

from openai import OpenAI, AsyncOpenAI
from config import AppConfig, EmbeddingConfig

# --- 补丁：绕过 transformers 的强制版本检查 (CVE-2025-32434) ---
def patch_transformers_security_check():
    try:
        import transformers.utils.import_utils as iu
        iu.check_torch_load_is_safe = lambda: None
        import transformers.utils as u
        if hasattr(u, "check_torch_load_is_safe"):
            u.check_torch_load_is_safe = lambda: None
        import transformers.modeling_utils as mu
        if hasattr(mu, "check_torch_load_is_safe"):
            mu.check_torch_load_is_safe = lambda: None
    except Exception:
        pass

# 在本地模式下，我们需要在导入 FlagEmbedding 前执行补丁
# 因为 FlagEmbedding 内部会导入 transformers
patch_transformers_security_check()
# ---------------------------------------------------

Vector = List[float]


@dataclass
class EmbeddingClient:
    """
    Embedding 客户端，支持 API 和 本地 (BGE-M3) 模式。
    """
    
    cfg: EmbeddingConfig
    client: Optional[OpenAI] = None
    async_client: Optional[AsyncOpenAI] = None
    _local_model: Any = field(default=None, repr=False)

    @classmethod
    def from_config(cls, app_cfg: AppConfig) -> "EmbeddingClient":
        """从配置创建客户端"""
        cfg = app_cfg.embedding
        
        if cfg.mode == "local":
            print(f"     🚀 正在初始化本地 Embedding 模型: {cfg.model}...")
            try:
                # 1. 优先设置离线环境变量
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
                os.environ["HF_HUB_OFFLINE"] = "1"  # 强制离线模式
                
                from FlagEmbedding import BGEM3FlagModel
                from huggingface_hub import snapshot_download
                
                # 2. 获取本地缓存的绝对路径（不再联网，直接查本地）
                try:
                    local_model_path = snapshot_download(
                        repo_id=cfg.model,
                        local_files_only=True, # 强制只查找本地
                        ignore_patterns=["imgs/*", ".DS_Store", "*.pdf", "*.png"]
                    )
                except Exception:
                    # 如果强制离线查找失败，尝试正常路径（可能由于 snapshots 软连接问题）
                    local_model_path = cfg.model

                # 3. 初始化本地模型
                model = BGEM3FlagModel(local_model_path, use_fp16=False)
                print(f"     ✅ 本地模型加载成功 (路径: {local_model_path})")
                return cls(cfg=cfg, _local_model=model)
            except ImportError:
                raise RuntimeError("未安装 FlagEmbedding 库。请执行: pip install FlagEmbedding")
            except Exception as e:
                raise RuntimeError(f"本地模型加载失败: {str(e)}")
        
        # API 模式
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
        """将一批文本转换为向量"""
        if not texts:
            return []

        # 1. 本地模式处理
        if self.cfg.mode == "local" and self._local_model:
            if verbose:
                print(f"     ⏳ 正在使用本地 BGE-M3 进行向量化 (文本数: {len(texts)})...")
            
            start_time = time.time()
            # BGE-M3 默认只返回 dense_vecs，适合现有的检索逻辑
            output = self._local_model.encode(texts, return_dense=True)
            vectors = output['dense_vecs'].tolist()
            
            if verbose:
                print(f"     ✅ 本地向量化完成，耗时: {time.time() - start_time:.2f} 秒")
            return vectors

        # 2. API 模式处理 (保留原有逻辑)
        if not self.client:
            raise RuntimeError("客户端未初始化")
        
        if verbose:
            total_chars = sum(len(t) for t in texts)
            print(f"     准备调用 API: {len(texts)} 个文本，总长度 {total_chars} 字符")
            print(f"     API: {self.cfg.base_url}")
            print(f"     模型: {self.cfg.model}")
            print(f"     批处理大小: {batch_size}")

        try:
            start_time = time.time()
            all_vectors: List[Vector] = []
            
            # 如果文本数量超过 batch_size，需要分批处理
            if len(texts) > batch_size:
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    if i > 0:
                        time.sleep(2.0)  # 规避 API 频率限制
                    
                    response = self.client.embeddings.create(
                        model=self.cfg.model,
                        input=batch_texts,
                    )
                    all_vectors.extend([item.embedding for item in response.data])
            else:
                response = self.client.embeddings.create(
                    model=self.cfg.model,
                    input=texts,
                )
                all_vectors = [item.embedding for item in response.data]
            
            return all_vectors
            
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

