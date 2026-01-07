# 向量检索后如何找到对应的原始文本切片（面试要点）

## 核心答案

**向量数据库（Milvus）在存储时建立了向量与文本/元数据的关联关系，检索时通过 `output_fields` 参数可以同时返回匹配向量的文本和元数据，无需额外查询。**

## 完整流程

### 1. 存储阶段：建立关联关系

```python
# vector_store/store.py - add_texts 方法
def add_texts(self, kb_id, texts, vectors, metadatas):
    # 准备数据：向量、文本、元数据一起存储
    data = []
    for text, vector, metadata in zip(texts, vectors, metadatas):
        data.append({
            "vector": vector,           # 向量
            "text": text,               # 原始文本切片
            "metadata": json.dumps(metadata),  # 元数据（JSON 字符串）
        })
    
    # 插入到 Milvus（向量、文本、元数据作为一个整体存储）
    self.client.insert(
        collection_name=collection_name,
        data=data
    )
```

**关键点**：
- Milvus 内部为每条记录分配唯一 ID
- 向量、文本、元数据存储在**同一条记录**中
- 建立了**向量 ↔ 文本 ↔ 元数据**的关联关系

### 2. 检索阶段：同时返回文本和元数据

```python
# vector_store/store.py - search 方法
def search(self, kb_id, query_vector, top_k=5):
    # 执行向量相似度搜索
    results = self.client.search(
        collection_name=collection_name,
        data=[query_vector],           # 查询向量
        limit=top_k,                   # 返回 top-k 个结果
        output_fields=["text", "metadata"],  # ⭐ 关键：指定返回的字段
    )
    
    # 处理结果
    search_results = []
    for hit in results[0]:
        # Milvus 返回的结果中包含了：
        # - 向量（用于计算相似度）
        # - text（原始文本切片）⭐
        # - metadata（元数据）⭐
        
        text = hit.get("text")                    # 直接获取文本
        metadata_str = hit.get("metadata")        # 直接获取元数据
        similarity_score = 1.0 - hit.get("distance")  # 相似度分数
        
        search_results.append((similarity_score, text, metadata))
    
    return search_results
```

**关键点**：
- `output_fields=["text", "metadata"]` 指定返回的字段
- Milvus 会自动返回匹配向量对应的文本和元数据
- **无需额外查询**，一次检索即可获得所有信息

### 3. 使用阶段：直接使用检索结果

```python
# rag/engine.py - query 方法
def query(self, question, top_k=5):
    # 1. 问题向量化
    query_vector = self.embedding_client.embed_texts([question])[0]
    
    # 2. 向量检索（返回相似度、文本、元数据）
    search_results = self.vector_store.search(
        kb_id=self.kb_id,
        query_vector=query_vector,
        top_k=top_k,
    )
    
    # 3. 直接使用检索到的文本
    context_chunks = []
    for score, metadata in search_results:
        context_chunks.append({
            "text": metadata.text,      # ⭐ 直接使用文本
            "score": score,
            "doc_id": metadata.doc_id, # ⭐ 直接使用元数据
        })
    
    # 4. 拼接上下文，发送给 LLM
    context = "\n\n".join([chunk['text'] for chunk in context_chunks])
    answer = self.llm_client.generate(context, question)
    
    return {"answer": answer, "sources": context_chunks}
```

## 技术原理

### Milvus 的存储结构

```
Collection (知识库)
├── Record 1
│   ├── ID: auto_generated_1
│   ├── vector: [0.1, 0.2, ...]      # 向量字段（用于检索）
│   ├── text: "西红柿鸡蛋的做法..."  # 文本字段（用于返回）
│   └── metadata: {"category": "荤菜", ...}  # 元数据字段（用于返回）
│
├── Record 2
│   ├── ID: auto_generated_2
│   ├── vector: [0.3, 0.4, ...]
│   ├── text: "麻婆豆腐的做法..."
│   └── metadata: {"category": "素菜", ...}
│
└── ...
```

### 检索过程

```
1. 输入：查询向量 [0.15, 0.25, ...]
   ↓
2. Milvus 计算与所有向量的相似度
   ↓
3. 返回 top-k 个最相似的记录
   ↓
4. 根据 output_fields 返回指定字段
   ↓
5. 输出：
   [
     {
       "distance": 0.1,                    # 距离（越小越相似）
       "text": "西红柿鸡蛋的做法...",     # 原始文本切片 ⭐
       "metadata": {"category": "荤菜"}   # 元数据 ⭐
     },
     ...
   ]
```

## 为什么不需要额外查询？

### 传统关系型数据库的方式（需要额外查询）

```python
# ❌ 低效方式（需要两次查询）
# 1. 向量检索，只返回 ID
vector_ids = vector_db.search(query_vector, top_k=5)  # 返回: [id1, id2, ...]

# 2. 根据 ID 查询文本和元数据
texts = []
for vector_id in vector_ids:
    text = text_db.get_by_id(vector_id)  # 额外查询
    texts.append(text)
```

### 向量数据库的方式（一次查询）

```python
# ✅ 高效方式（一次查询）
# 向量检索，同时返回向量、文本、元数据
results = vector_db.search(
    query_vector,
    top_k=5,
    output_fields=["text", "metadata"]  # 指定返回字段
)
# 直接获得：[(score, text, metadata), ...]
```

**优势**：
- ✅ 减少查询次数（1 次 vs 2 次）
- ✅ 降低延迟（网络往返减少）
- ✅ 保证数据一致性（向量、文本、元数据来自同一条记录）

## 实际代码示例

### 存储时的数据结构

```python
# 存储一条记录
{
    "vector": [0.1, 0.2, 0.3, ...],  # 1536 维向量
    "text": "西红柿鸡蛋是一道简单易做的家常菜，需要西红柿、鸡蛋、盐等原料。",
    "metadata": json.dumps({
        "parent_id": "doc_123",
        "chunk_id": "chunk_456",
        "category": "荤菜",
        "difficulty": "简单",
        "file_name": "西红柿鸡蛋",
    })
}
```

### 检索时的返回结果

```python
# Milvus 返回的结果
[
    {
        "id": "auto_generated_1",
        "distance": 0.15,  # 距离（越小越相似）
        "entity": {
            "vector": [0.1, 0.2, ...],  # 匹配的向量
            "text": "西红柿鸡蛋是一道简单易做的家常菜...",  # ⭐ 原始文本切片
            "metadata": '{"parent_id": "doc_123", "category": "荤菜", ...}'  # ⭐ 元数据
        }
    },
    ...
]
```

### 使用检索结果

```python
# 直接使用文本和元数据
for hit in results[0]:
    text = hit.get("text")                    # ⭐ 直接获取文本
    metadata = json.loads(hit.get("metadata")) # ⭐ 直接获取元数据
    score = 1.0 - hit.get("distance")         # 计算相似度
    
    # 使用文本构建上下文
    context += f"\n{text}"
    
    # 使用元数据溯源
    print(f"来源: {metadata['file_name']}")
```

## 面试常见问题

### Q1: 向量检索后，如何找到对应的原始文本切片？

**A**: 
1. **存储时建立关联**：向量、文本、元数据存储在 Milvus 的**同一条记录**中，建立了关联关系
2. **检索时指定字段**：使用 `output_fields=["text", "metadata"]` 参数，Milvus 会自动返回匹配向量对应的文本和元数据
3. **一次查询完成**：无需额外查询，一次向量检索即可获得文本和元数据

### Q2: 为什么不先检索向量 ID，再根据 ID 查询文本？

**A**: 
- **效率问题**：需要两次查询（向量检索 + 文本查询），增加延迟
- **数据一致性**：向量和文本可能来自不同数据源，存在不一致风险
- **向量数据库优势**：Milvus 等向量数据库支持在检索时同时返回关联字段，一次查询即可

### Q3: 如果向量数据库不支持 output_fields 怎么办？

**A**: 
需要额外查询：
1. 向量检索返回向量 ID 或向量本身
2. 根据 ID 在关系型数据库或文档存储中查询文本和元数据
3. 或者维护一个 ID → 文本/元数据的映射表（内存或缓存）

但这种方式效率较低，不推荐。

### Q4: 检索结果的顺序是什么？

**A**: 
- **按相似度降序排列**：相似度最高的排在前面
- **相似度计算**：`similarity = 1.0 - distance`（对于余弦相似度）
- **距离越小，相似度越高**

## 总结

| 阶段 | 操作 | 关键点 |
|------|------|--------|
| **存储** | 向量、文本、元数据一起存储 | 建立关联关系 |
| **检索** | 使用 `output_fields` 指定返回字段 | 一次查询获得所有信息 |
| **使用** | 直接使用返回的文本和元数据 | 无需额外查询 |

**核心原理**：
- 向量数据库在存储时建立了**向量 ↔ 文本 ↔ 元数据**的关联关系
- 检索时通过 `output_fields` 参数可以**同时返回**匹配向量的文本和元数据
- **一次查询，获得所有信息**，无需额外查询

