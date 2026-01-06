from __future__ import annotations

"""
LLMClient
---------

参考 RAGFlow 的实现方式，使用 OpenAI SDK 作为基础客户端。

设计要点（对齐 RAGFlow）：
1. 使用 openai.OpenAI 作为基础客户端（而非直接 HTTP 调用）
2. 通过 base_url + model_name 适配不同厂商（只要兼容 OpenAI 格式）
3. 支持同步和异步调用
4. 统一的错误处理和重试机制

与 cloud-edge-milk-tea-agent 的区别：
- cloud-edge-milk-tea-agent：使用厂商官方 SDK（如 dashscope.Generation.call()）
- rag_engine（本实现）：使用 OpenAI SDK + base_url，更通用和标准化
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import os

from openai import OpenAI, AsyncOpenAI

from config import AppConfig, LLMConfig


Message = Dict[str, str]  # {"role": "user" | "assistant" | "system", "content": "..."}


@dataclass
class LLMClient:
    """
    LLM 客户端，参考 RAGFlow 的 Base 类实现。
    
    使用 OpenAI SDK，通过 base_url 适配不同厂商：
    - OpenAI: base_url="https://api.openai.com/v1"
    - 通义千问: base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    - DeepSeek: base_url="https://api.deepseek.com/v1"
    - 其他兼容 OpenAI 格式的厂商...
    """
    
    cfg: LLMConfig
    client: OpenAI
    async_client: AsyncOpenAI

    @classmethod
    def from_config(cls, app_cfg: AppConfig) -> "LLMClient":
        """从配置创建客户端（参考 RAGFlow 的初始化方式）"""
        cfg = app_cfg.llm
        timeout = int(os.environ.get("LLM_TIMEOUT_SECONDS", int(cfg.timeout)))
        
        # 使用 OpenAI SDK，通过 base_url 适配不同厂商
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

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        history: Optional[List[Message]] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        同步生成一个回答（参考 RAGFlow 的 _chat 方法）。

        - prompt：当前用户问题
        - system_prompt：系统指令（可选）
        - history：历史对话（可选，OpenAI 格式）
        """
        messages: List[Message] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        # 构建生成配置（参考 RAGFlow 的 _clean_conf）
        gen_conf: Dict[str, Any] = {
            "temperature": temperature,
        }
        if max_tokens is not None:
            gen_conf["max_tokens"] = max_tokens

        try:
            # 使用 OpenAI SDK 调用（兼容所有支持 OpenAI 格式的厂商）
            response = self.client.chat.completions.create(
                model=self.cfg.model,
                messages=messages,
                **gen_conf,
            )
            
            if not response.choices or not response.choices[0].message:
                raise RuntimeError(f"LLM 返回格式异常: {response}")
            
            ans = response.choices[0].message.content
            if not ans:
                return ""
            
            # 处理长度限制（参考 RAGFlow）
            if response.choices[0].finish_reason == "length":
                ans += "\n...\n由于上下文窗口限制，回答已被截断。"
            
            return ans.strip()
            
        except Exception as e:
            # 简化错误处理，后续可以扩展为 RAGFlow 的 _classify_error 和重试机制
            raise RuntimeError(f"LLM 调用失败: {str(e)}") from e

    async def async_generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        history: Optional[List[Message]] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> str:
        """异步生成（参考 RAGFlow 的 async_chat 方法）"""
        messages: List[Message] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        gen_conf: Dict[str, Any] = {
            "temperature": temperature,
        }
        if max_tokens is not None:
            gen_conf["max_tokens"] = max_tokens

        try:
            response = await self.async_client.chat.completions.create(
                model=self.cfg.model,
                messages=messages,
                **gen_conf,
            )
            
            if not response.choices or not response.choices[0].message:
                raise RuntimeError(f"LLM 返回格式异常: {response}")
            
            ans = response.choices[0].message.content
            if not ans:
                return ""
            
            if response.choices[0].finish_reason == "length":
                ans += "\n...\n由于上下文窗口限制，回答已被截断。"
            
            return ans.strip()
            
        except Exception as e:
            raise RuntimeError(f"LLM 异步调用失败: {str(e)}") from e


__all__ = ["LLMClient", "Message"]

