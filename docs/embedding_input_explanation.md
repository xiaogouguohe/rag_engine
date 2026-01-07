# 向量化模型的输入内容（面试要点）

## 核心答案

**最终被输入到向量化模型的，是切块后的文档片段（纯文本），元数据不参与向量化，而是单独存储。**

## 代码证据

### 1. 向量化时的输入（`rag/engine.py`）

```python
# 4. 提取文本和元数据
texts = [chunk.page_content for chunk in doc_chunks]  # 只提取纯文本
metadatas = [chunk.metadata for chunk in doc_chunks]  # 元数据单独提取

# 5. 向量化（只传入纯文本）
vectors = self.embedding_client.embed_texts(texts)  # 只传入 texts，不包含元数据

# 6. 存储到向量数据库（文本、向量、元数据分别存储）
chunk_ids = self.vector_store.add_texts(
    kb_id=self.kb_id,
    texts=texts,        # 纯文本
    vectors=vectors,    # 向量
    metadatas=metadatas,  # 元数据（单独存储）
)
```

### 2. Embedding Client 的实现（`embedding/client.py`）

```python
def embed_texts(self, texts: List[str]) -> List[Vector]:
    """
    将一批文本转换为向量。
    
    注意：只接收纯文本列表，不包含元数据。
    """
    response = self.client.embeddings.create(
        model=self.cfg.model,
        input=texts,  # 只传入文本，不包含元数据
    )
    vectors = [item.embedding for item in response.data]
    return vectors
```

## 设计原理

### 为什么元数据不参与向量化？

#### 1. **语义相似度 vs 结构化信息**

- **向量化模型的作用**：将文本转换为向量，用于计算**语义相似度**
- **元数据的作用**：提供**结构化信息**（分类、难度、文件路径等）

**示例**：
```python
# 文本内容（参与向量化）
text = "西红柿鸡蛋是一道简单易做的家常菜，需要西红柿、鸡蛋、盐等原料。"

# 元数据（不参与向量化，单独存储）
metadata = {
    "category": "荤菜",
    "difficulty": "简单",
    "file_name": "西红柿鸡蛋",
    "parent_id": "doc_123",
}
```

如果元数据参与向量化：
- ❌ 会污染语义表示（"荤菜"、"简单" 这些标签会影响语义相似度计算）
- ❌ 不同文档的相同内容会因为元数据不同而产生不同的向量
- ❌ 检索时无法利用元数据的结构化特性进行过滤

#### 2. **分离关注点（Separation of Concerns）**

```
向量化模型：专注于语义理解
    ↓
    输入：纯文本
    输出：语义向量

元数据系统：专注于结构化信息管理
    ↓
    存储：JSON 格式
    用途：过滤、统计、溯源
```

#### 3. **检索策略的灵活性**

**纯文本向量化 + 元数据分离存储** 的优势：

```python
# 场景一：纯语义检索
results = vector_store.search(
    query_vector,  # 基于语义相似度
    top_k=10
)

# 场景二：语义检索 + 元数据过滤
results = vector_store.search(
    query_vector,  # 基于语义相似度
    filter={"category": "荤菜", "difficulty": "简单"},  # 元数据过滤
    top_k=10
)

# 场景三：纯元数据过滤（不需要向量检索）
results = vector_store.filter(
    {"category": "荤菜"}  # 只根据元数据过滤
)
```

## 实际存储结构

在 Milvus 向量数据库中的存储格式：

```python
{
    "id": "chunk_uuid",
    "vector": [0.1, 0.2, 0.3, ...],  # 1536 维向量（基于纯文本生成）
    "text": "西红柿鸡蛋是一道简单易做的家常菜...",  # 原始文本
    "metadata": {  # 元数据（JSON 格式，不参与向量化）
        "parent_id": "doc_123",
        "chunk_id": "chunk_uuid",
        "category": "荤菜",
        "difficulty": "简单",
        "file_name": "西红柿鸡蛋",
        "主标题": "西红柿鸡蛋的做法",
    }
}
```

## 对比：如果元数据参与向量化会怎样？

### 方案 A：元数据参与向量化（不推荐）

```python
# 将元数据拼接到文本中
text_with_metadata = f"{text}\n分类: {metadata['category']}\n难度: {metadata['difficulty']}"
vector = embed_texts([text_with_metadata])
```

**问题**：
1. ❌ 相同内容的文档会因为元数据不同而产生不同向量
2. ❌ 检索时无法区分"语义相似"和"元数据匹配"
3. ❌ 元数据的结构化特性无法被充分利用（如过滤、统计）

### 方案 B：元数据不参与向量化（当前方案，推荐）✅

```python
# 只向量化纯文本
vector = embed_texts([text])

# 元数据单独存储
store(text=text, vector=vector, metadata=metadata)
```

**优势**：
1. ✅ 语义相似度计算更准确（不受元数据干扰）
2. ✅ 支持灵活的检索策略（语义检索 + 元数据过滤）
3. ✅ 元数据的结构化特性得到充分利用

## 面试常见问题

### Q1: 为什么元数据不参与向量化？

**A**: 
1. **语义 vs 结构化**：向量化模型专注于语义理解，元数据是结构化信息，两者关注点不同
2. **避免污染**：元数据参与向量化会污染语义表示，影响相似度计算的准确性
3. **灵活性**：分离存储后，可以支持"语义检索 + 元数据过滤"的混合检索策略

### Q2: 如果我想让元数据影响检索结果怎么办？

**A**: 
使用**混合检索策略**：
1. 先用向量相似度检索（基于纯文本）
2. 再用元数据过滤（如：只保留"简单"难度的菜谱）
3. 或者先过滤，再检索（取决于向量数据库的能力）

```python
# 示例：语义检索 + 元数据过滤
results = vector_store.search(
    query_vector,
    filter={"difficulty": "简单"},  # 元数据过滤
    top_k=10
)
```

### Q3: 有没有场景需要将元数据包含在向量化中？

**A**: 
**极少数场景**可以考虑，但需要谨慎：

1. **元数据本身就是语义信息**：
   ```python
   # 例如：文档摘要、标签（这些本身就是语义内容）
   text_with_summary = f"{text}\n摘要: {summary}"
   ```

2. **领域特定的增强**：
   ```python
   # 例如：在医疗领域，将"疾病类型"作为上下文
   text_with_context = f"疾病类型: {disease_type}\n{text}"
   ```

但一般情况下，**推荐分离存储**，因为：
- 更灵活（可以动态组合检索策略）
- 更准确（语义相似度不受元数据干扰）
- 更高效（元数据可以建立索引，支持快速过滤）

## 总结

| 项目 | 内容 | 是否参与向量化 |
|------|------|----------------|
| 文档片段文本 | "西红柿鸡蛋是一道..." | ✅ **是** |
| 元数据 | category, difficulty, file_name 等 | ❌ **否**（单独存储） |
| 标题信息 | "主标题"、"二级标题" | ❌ **否**（作为元数据存储） |

**核心原则**：
- **向量化**：只处理语义内容（纯文本）
- **元数据**：单独存储，用于过滤、统计、溯源
- **检索**：结合使用（语义相似度 + 元数据过滤）

