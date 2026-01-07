# RAGAS 使用 llm_factory 直接访问 LLM API

## 概述

RAGAS **不需要**使用 LangChain！它提供了 `llm_factory` 函数，可以直接使用 OpenAI SDK 或其他 LLM 客户端。

## 为什么使用 llm_factory？

1. **更简单**：不需要 LangChain 中间层
2. **更直接**：直接使用 OpenAI SDK
3. **更现代**：这是 RAGAS 推荐的方式
4. **更灵活**：支持多种 LLM 提供商

## 使用方法

### 基本用法

```python
from openai import OpenAI
from ragas.llms import llm_factory

# 创建 OpenAI 客户端
client = OpenAI(
    api_key="your-api-key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 使用 llm_factory 创建 RAGAS LLM
llm = llm_factory("qwen3-max", client=client)

# 在评估中使用
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy],
    llm=llm,  # 直接使用 llm_factory 创建的 LLM
    embeddings=embeddings,
)
```

### 与 LangChain 的对比

#### 旧方法（使用 LangChain，已废弃）

```python
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper  # ⚠️ 已废弃

# 需要 LangChain
langchain_llm = ChatOpenAI(
    model="qwen3-max",
    api_key="...",
    base_url="...",
)

# 需要包装
ragas_llm = LangchainLLMWrapper(langchain_llm)  # ⚠️ 已废弃
```

#### 新方法（使用 llm_factory，推荐）

```python
from openai import OpenAI
from ragas.llms import llm_factory

# 直接使用 OpenAI SDK
client = OpenAI(
    api_key="...",
    base_url="...",
)

# 直接创建 RAGAS LLM
ragas_llm = llm_factory("qwen3-max", client=client)  # ✅ 推荐
```

## 支持的 LLM 提供商

### OpenAI（包括兼容 OpenAI API 的服务）

```python
from openai import OpenAI

client = OpenAI(
    api_key="...",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 通义千问
)

llm = llm_factory("qwen3-max", client=client)
```

### Anthropic

```python
from anthropic import Anthropic

client = Anthropic(api_key="...")
llm = llm_factory("claude-3-sonnet", provider="anthropic", client=client)
```

### Google Gemini

```python
from litellm import OpenAI as LiteLLMClient

client = LiteLLMClient(api_key="...", model="gemini-2.0-flash")
llm = llm_factory("gemini-2.0-flash", client=client)
```

### 异步客户端

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key="...", base_url="...")
llm = llm_factory("qwen3-max", client=client)

# 使用异步方法
result = await llm.agenerate(prompt, ResponseModel)
```

## 在 generate_ragas_dataset.py 中的使用

### 当前实现（使用 LangChain）

```python
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper  # ⚠️ 已废弃

generator_llm = ChatOpenAI(...)
ragas_llm = LangchainLLMWrapper(generator_llm)
```

### 推荐实现（使用 llm_factory）

```python
from openai import OpenAI
from ragas.llms import llm_factory

openai_client = OpenAI(
    api_key=app_config.llm.api_key,
    base_url=app_config.llm.base_url,
)

ragas_llm = llm_factory(
    model=app_config.llm.model,
    provider="openai",
    client=openai_client,
)
```

## 优势

1. **更少的依赖**：不需要 LangChain
2. **更快的速度**：减少中间层
3. **更好的兼容性**：直接使用 OpenAI SDK，兼容性更好
4. **更清晰的代码**：代码更简洁

## 注意事项

1. **LangchainLLMWrapper 已废弃**：RAGAS 推荐使用 `llm_factory`
2. **需要 OpenAI SDK**：`pip install openai`
3. **支持缓存**：可以使用 `cache` 参数启用缓存

```python
from ragas.cache import DiskCacheBackend

cache = DiskCacheBackend()
llm = llm_factory("qwen3-max", client=client, cache=cache)
```

## 相关文档

- RAGAS 官方文档：https://docs.ragas.io/en/latest/llm-factory
- 测试脚本：`test_llm_factory.py`

