# 去掉 LangChain 依赖

## 修改内容

已成功去掉评估部分的 LangChain 依赖，改用 RAGAS 原生的接口。

## 修改前后对比

### LLM 初始化（修改前）

```python
# 使用 LangChain（已废弃）
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

generator_llm = ChatOpenAI(
    model=app_config.llm.model,
    api_key=app_config.llm.api_key,
    base_url=app_config.llm.base_url,
    temperature=0.1,
    timeout=120.0,
    max_retries=3,
)
ragas_llm = LangchainLLMWrapper(generator_llm)  # ⚠️ 已废弃
```

### LLM 初始化（修改后）

```python
# 使用 llm_factory（推荐）
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
    temperature=0.1,
)  # ✅ 推荐方式
```

### Embeddings 初始化（修改前）

```python
# 使用 LangChain（已废弃）
from langchain_openai import OpenAIEmbeddings as LangchainOpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

langchain_embeddings = LangchainOpenAIEmbeddings(
    model=app_config.embedding.model,
    api_key=app_config.embedding.api_key,
    base_url=app_config.embedding.base_url,
)
embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)  # ⚠️ 已废弃
```

### Embeddings 初始化（修改后）

```python
# 使用 RAGAS 原生接口（推荐）
from openai import OpenAI, AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings

embedding_client = OpenAI(
    api_key=app_config.embedding.api_key,
    base_url=app_config.embedding.base_url,
)
async_embedding_client = AsyncOpenAI(
    api_key=app_config.embedding.api_key,
    base_url=app_config.embedding.base_url,
)

embeddings = RagasOpenAIEmbeddings(
    client=embedding_client,
    model=app_config.embedding.model,
)
if hasattr(embeddings, 'async_client'):
    embeddings.async_client = async_embedding_client  # ✅ 推荐方式
```

## 优势

1. **更少的依赖**：不需要 LangChain
2. **更快的速度**：减少中间层
3. **更好的兼容性**：直接使用 OpenAI SDK
4. **更清晰的代码**：代码更简洁
5. **符合 RAGAS 推荐**：使用官方推荐的方式

## 注意事项

### 生成测试集部分

生成测试集部分（`generate_ragas_testset` 函数）**仍需要 LangChain**，因为：
- RAGAS 的 `TestsetGenerator.from_langchain()` 需要 LangChain 接口
- `LangchainDocument` 格式是 RAGAS TestsetGenerator 需要的

如果将来 RAGAS 提供了不依赖 LangChain 的 TestsetGenerator，可以进一步去掉依赖。

### 依赖要求

现在评估部分只需要：
- `openai` - OpenAI SDK
- `ragas` - RAGAS 库

不再需要：
- `langchain` - LangChain 核心库
- `langchain-openai` - LangChain OpenAI 集成

## 测试

运行评估测试：

```bash
python3 generate_ragas_dataset.py \
  --evaluate-only ragas_testset_1doc_1q.json \
  --eval-output ragas_eval_results.json
```

应该会看到：
```
✅ LLM 初始化成功（使用 llm_factory，不依赖 LangChain）
✅ Embeddings 初始化成功（使用 RAGAS OpenAIEmbeddings，不依赖 LangChain）
```

## 相关文档

- `docs/ragas_llm_factory.md` - llm_factory 使用说明
- `test_llm_factory.py` - 测试示例

