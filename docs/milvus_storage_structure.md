# Milvus 向量数据库的存储结构（面试要点）

## 核心答案

**Milvus 采用类似关系数据库的"表-行-列"结构，但专门为向量数据优化。一条记录（Entity）包含：一个向量字段（Vector Field）和多个标量字段（Scalar Fields，用于存储元数据）。**

## Milvus 的存储结构

### 1. 概念对比：关系数据库 vs Milvus

| 关系数据库 | Milvus 向量数据库 |
|-----------|------------------|
| Database（数据库） | Database（数据库）|
| Table（表） | Collection（集合）|
| Row（行） | Entity（实体/记录）|
| Column（列） | Field（字段）|
| Primary Key | Primary Key（主键）|

### 2. Collection（集合）的结构

一个 Collection 类似于关系数据库中的表，包含：

```python
Collection: "kb_recipes_kb"
├── Schema（模式定义）
│   ├── Primary Key: id (自动生成)
│   ├── Vector Field: vector (1536 维)
│   ├── Scalar Field: text (VARCHAR)
│   └── Scalar Field: metadata (VARCHAR/JSON)
│
└── Entities（记录）
    ├── Entity 1
    │   ├── id: 1
    │   ├── vector: [0.1, 0.2, 0.3, ...]
    │   ├── text: "西红柿鸡蛋的做法..."
    │   └── metadata: '{"category": "荤菜", ...}'
    │
    ├── Entity 2
    │   ├── id: 2
    │   ├── vector: [0.4, 0.5, 0.6, ...]
    │   ├── text: "麻婆豆腐的做法..."
    │   └── metadata: '{"category": "素菜", ...}'
    │
    └── ...
```

### 3. 字段类型

#### Vector Field（向量字段）
- **类型**：`FLOAT_VECTOR` 或 `BINARY_VECTOR`
- **维度**：固定维度（如 1536）
- **用途**：存储向量，用于相似度计算
- **索引**：支持多种索引类型（IVF_FLAT、HNSW 等）

#### Scalar Fields（标量字段）
- **类型**：`INT64`、`VARCHAR`、`JSON`、`BOOL` 等
- **用途**：存储元数据（文本、分类、时间戳等）
- **索引**：可以建立标量索引，支持过滤查询

### 4. 实际存储示例

#### 创建 Collection 时的 Schema

```python
# Milvus 的 Schema 定义（简化版）
schema = {
    "fields": [
        {
            "name": "id",
            "type": "INT64",
            "is_primary": True,
            "auto_id": True  # 自动生成 ID
        },
        {
            "name": "vector",
            "type": "FLOAT_VECTOR",
            "dim": 1536,  # 向量维度
            "metric_type": "COSINE"  # 相似度度量方式
        },
        {
            "name": "text",
            "type": "VARCHAR",
            "max_length": 65535  # 最大长度
        },
        {
            "name": "metadata",
            "type": "VARCHAR",
            "max_length": 65535  # JSON 字符串
        }
    ]
}
```

#### 插入数据时的结构

```python
# 我们代码中的插入方式
data = [
    {
        "vector": [0.1, 0.2, 0.3, ...],  # 向量字段
        "text": "西红柿鸡蛋的做法...",     # 标量字段
        "metadata": '{"category": "荤菜", "difficulty": "简单"}'  # 标量字段
    },
    {
        "vector": [0.4, 0.5, 0.6, ...],
        "text": "麻婆豆腐的做法...",
        "metadata": '{"category": "素菜", "difficulty": "中等"}'
    }
]

# Milvus 内部存储结构（概念上）
Entity 1:
  id: 1 (自动生成)
  vector: [0.1, 0.2, 0.3, ...]  # 存储在向量索引中
  text: "西红柿鸡蛋的做法..."      # 存储在标量字段中
  metadata: '{"category": "荤菜", ...}'  # 存储在标量字段中

Entity 2:
  id: 2 (自动生成)
  vector: [0.4, 0.5, 0.6, ...]
  text: "麻婆豆腐的做法..."
  metadata: '{"category": "素菜", ...}'
```

## 底层存储机制

### 1. 向量字段的存储

**向量索引**：
- Milvus 使用专门的向量索引（如 IVF_FLAT、HNSW）来加速相似度搜索
- 向量数据存储在**专门的向量索引结构**中，不是简单的数组存储
- 支持多种相似度度量方式：余弦相似度（COSINE）、欧氏距离（L2）、内积（IP）

**存储位置**：
```
向量数据 → 向量索引（IVF_FLAT/HNSW）→ 磁盘/内存
```

### 2. 标量字段的存储

**存储方式**：
- 标量字段（text、metadata）存储在**列式存储**或**行式存储**中
- 可以建立**标量索引**（如 B-tree）来加速过滤查询
- 支持范围查询、等值查询等

**存储位置**：
```
标量数据 → 列式存储/行式存储 → 磁盘
```

### 3. 数据关联

**关键点**：
- 每个 Entity 有一个**唯一 ID**（主键）
- 向量字段和标量字段通过**同一个 ID** 关联
- 检索时，Milvus 通过 ID 将向量和标量字段**组合返回**

## 代码中的实际使用

### 1. 创建 Collection（隐式 Schema）

```python
# vector_store/store.py - _ensure_collection 方法
self.client.create_collection(
    collection_name=collection_name,
    dimension=vector_dim,      # 向量维度
    metric_type="COSINE",      # 相似度度量
    auto_id=True,              # 自动生成 ID
)
# MilvusClient 会自动创建 Schema，包含：
# - id (INT64, primary key, auto_id)
# - vector (FLOAT_VECTOR, dimension=vector_dim)
# - text (VARCHAR) - 自动推断
# - metadata (VARCHAR) - 自动推断
```

### 2. 插入数据

```python
# vector_store/store.py - add_texts 方法
data = [{
    "vector": vector,           # 向量字段
    "text": text,               # 标量字段
    "metadata": metadata_str,   # 标量字段（JSON 字符串）
}]

self.client.insert(
    collection_name=collection_name,
    data=data
)
```

### 3. 检索数据

```python
# vector_store/store.py - search 方法
results = self.client.search(
    collection_name=collection_name,
    data=[query_vector],        # 查询向量
    limit=top_k,
    output_fields=["text", "metadata"]  # 指定返回的标量字段
)

# Milvus 返回结果：
# [
#   {
#     "id": 1,
#     "distance": 0.15,
#     "entity": {
#       "vector": [0.1, 0.2, ...],  # 向量字段
#       "text": "西红柿鸡蛋的做法...",  # 标量字段
#       "metadata": '{"category": "荤菜"}'  # 标量字段
#     }
#   },
#   ...
# ]
```

## 与关系数据库的对比

### 关系数据库（MySQL/PostgreSQL）

```sql
CREATE TABLE chunks (
    id INT PRIMARY KEY AUTO_INCREMENT,
    vector BLOB,  -- 向量（二进制存储）
    text TEXT,
    metadata JSON
);

-- 存储
INSERT INTO chunks (vector, text, metadata) VALUES (...);

-- 检索（需要自己实现向量相似度计算）
SELECT * FROM chunks WHERE ...;  -- 无法直接做向量相似度搜索
```

**问题**：
- ❌ 没有向量索引，无法高效进行相似度搜索
- ❌ 需要自己实现向量相似度计算（效率低）
- ❌ 无法利用向量索引加速

### Milvus 向量数据库

```python
# 创建 Collection（自动处理向量索引）
collection = create_collection(
    dimension=1536,
    metric_type="COSINE"
)

# 存储（自动建立向量索引）
insert(data=[{"vector": [...], "text": "...", "metadata": "..."}])

# 检索（利用向量索引加速）
search(query_vector, output_fields=["text", "metadata"])
```

**优势**：
- ✅ 自动建立向量索引，高效相似度搜索
- ✅ 支持标量字段过滤（如：`filter={"category": "荤菜"}`）
- ✅ 向量和标量字段统一管理，一次查询返回所有信息

## 存储优化

### 1. 向量索引类型

Milvus 支持多种向量索引类型：

- **FLAT**：暴力搜索，精度最高，速度最慢
- **IVF_FLAT**：倒排索引，平衡速度和精度
- **HNSW**：分层导航小世界图，速度快，精度高
- **IVF_SQ8**：量化索引，节省存储空间

### 2. 标量字段索引

```python
# 可以为标量字段建立索引，加速过滤查询
create_index(
    field_name="category",
    index_type="STL_SORT"  # 标量索引
)
```

### 3. 数据分区

```python
# 可以按标量字段分区，提高查询效率
create_partition(
    partition_name="meat_dishes",
    filter={"category": "荤菜"}
)
```

## 面试常见问题

### Q1: Milvus 如何存储元数据？

**A**: 
1. **标量字段存储**：元数据存储在标量字段（VARCHAR/JSON）中
2. **与向量关联**：通过 Entity ID 将向量和元数据关联
3. **统一管理**：向量和标量字段存储在同一个 Collection 中
4. **一次查询**：检索时可以同时返回向量和元数据

### Q2: 一条记录在 Milvus 中是如何存储的？

**A**: 
一条记录（Entity）包含：
- **ID**：唯一标识符（主键，自动生成）
- **Vector Field**：向量字段（存储在向量索引中）
- **Scalar Fields**：标量字段（text、metadata 等，存储在列式/行式存储中）

所有字段通过**同一个 ID** 关联，检索时 Milvus 会自动组合返回。

### Q3: 为什么不像关系数据库那样存储？

**A**: 
1. **向量索引**：Milvus 有专门的向量索引结构，支持高效的相似度搜索
2. **统一管理**：向量和元数据统一管理，一次查询返回所有信息
3. **性能优化**：针对向量检索场景优化，比关系数据库更高效

### Q4: 元数据可以建立索引吗？

**A**: 
可以！Milvus 支持为标量字段建立索引：
```python
# 为 category 字段建立索引，加速过滤查询
create_index(field_name="category", index_type="STL_SORT")
```

然后可以这样查询：
```python
search(
    query_vector,
    filter={"category": "荤菜"},  # 利用索引加速
    top_k=10
)
```

## 总结

| 项目 | 关系数据库 | Milvus |
|------|-----------|--------|
| **表结构** | Table（表） | Collection（集合）|
| **记录** | Row（行） | Entity（实体）|
| **字段** | Column（列） | Field（字段）|
| **向量存储** | BLOB（二进制） | Vector Field（向量索引）|
| **元数据存储** | JSON/VARCHAR | Scalar Fields（标量字段）|
| **关联方式** | 主键 | Entity ID（主键）|
| **检索方式** | SQL 查询 | 向量相似度搜索 + 标量过滤 |

**核心特点**：
- Milvus 采用**类似关系数据库的表结构**，但专门为向量数据优化
- 一条记录包含**一个向量字段**和**多个标量字段**（元数据）
- 向量和元数据通过**同一个 Entity ID** 关联
- 检索时可以**同时返回**向量和元数据，无需额外查询

