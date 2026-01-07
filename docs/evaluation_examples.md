# 评估数据集生成示例

## 一、问题生成示例

### 策略1：从文件名生成问题（主要方式）

**示例**：

| 文件名 | 生成的问题 |
|--------|-----------|
| `西红柿鸡蛋.md` | `如何做西红柿鸡蛋？` |
| `麻婆豆腐.md` | `如何做麻婆豆腐？` |
| `红烧肉.md` | `如何做红烧肉？` |
| `宫保鸡丁.md` | `如何做宫保鸡丁？` |

**特点**：
- ✅ 快速、无需调用 LLM
- ✅ 问题与文档高度相关
- ✅ 问题格式统一（"如何做..."）

### 策略2：使用 LLM 从内容生成问题（可选，增加多样性）

**示例**（基于菜谱内容）：

假设文档内容：
```markdown
# 西红柿鸡蛋的做法

预估烹饪难度：★★

## 必备原料和工具

* 西红柿
* 鸡蛋
* 盐

## 操作

* 西红柿切成小丁
* 起锅烧油
* 炒鸡蛋
* 加入西红柿
* 调味
```

**LLM 可能生成的问题**：
1. `西红柿鸡蛋需要哪些原料？`
2. `西红柿鸡蛋的难度是多少？`
3. `如何炒西红柿鸡蛋？`
4. `西红柿鸡蛋的制作步骤是什么？`
5. `西红柿鸡蛋需要什么调料？`

**特点**：
- ✅ 问题多样化（原料、难度、步骤等）
- ✅ 更贴近真实用户问题
- ⚠️ 需要调用 LLM，成本较高

## 二、生成的评估数据集示例

### 完整数据集结构

```json
{
  "kb_id": "recipes_kb",
  "source_path": "../HowToCook/dishes",
  "total_samples": 150,
  "samples": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "question": "如何做西红柿鸡蛋？",
      "source_document": "meat_dish/西红柿鸡蛋.md",
      "parent_id": "a1b2c3d4e5f6...",
      "relevant_chunks": ["a1b2c3d4e5f6..."],
      "metadata": {
        "file_name": "西红柿鸡蛋.md",
        "file_path": "/path/to/meat_dish/西红柿鸡蛋.md"
      }
    },
    {
      "id": "660e8400-e29b-41d4-a716-446655440001",
      "question": "西红柿鸡蛋需要哪些原料？",
      "source_document": "meat_dish/西红柿鸡蛋.md",
      "parent_id": "a1b2c3d4e5f6...",
      "relevant_chunks": ["a1b2c3d4e5f6..."],
      "metadata": {
        "file_name": "西红柿鸡蛋.md",
        "file_path": "/path/to/meat_dish/西红柿鸡蛋.md"
      }
    },
    {
      "id": "770e8400-e29b-41d4-a716-446655440002",
      "question": "如何做麻婆豆腐？",
      "source_document": "vegetable_dish/麻婆豆腐.md",
      "parent_id": "b2c3d4e5f6a1...",
      "relevant_chunks": ["b2c3d4e5f6a1..."],
      "metadata": {
        "file_name": "麻婆豆腐.md",
        "file_path": "/path/to/vegetable_dish/麻婆豆腐.md"
      }
    }
  ]
}
```

### 单个样本说明

```json
{
  "id": "样本唯一ID",
  "question": "如何做西红柿鸡蛋？",  // 生成的问题
  "source_document": "meat_dish/西红柿鸡蛋.md",  // 问题来源文档
  "parent_id": "a1b2c3d4e5f6...",  // 文档的唯一ID（MD5哈希）
  "relevant_chunks": ["a1b2c3d4e5f6..."],  // 相关文档块的ID列表（当前是整个文档）
  "metadata": {
    "file_name": "西红柿鸡蛋.md",
    "file_path": "/path/to/meat_dish/西红柿鸡蛋.md"
  }
}
```

## 三、召回率评估示例

### 评估流程

```
问题："如何做西红柿鸡蛋？"
   ↓
相关文档块（ground truth）：
  - parent_id: "a1b2c3d4e5f6..." （整个文档的所有块）
   ↓
向量检索（top_k=5）：
  - 检索到的文档块及其 parent_id
   ↓
计算召回率：
  - 检索到的相关块数量 / 所有相关块数量
```

### 具体计算示例

**场景1：完美检索**

```
问题："如何做西红柿鸡蛋？"
相关文档：parent_id = "a1b2c3d4e5f6..."（包含 5 个块）

检索结果（top_k=5）：
  1. parent_id = "a1b2c3d4e5f6..." (相关) ✅
  2. parent_id = "a1b2c3d4e5f6..." (相关) ✅
  3. parent_id = "a1b2c3d4e5f6..." (相关) ✅
  4. parent_id = "a1b2c3d4e5f6..." (相关) ✅
  5. parent_id = "a1b2c3d4e5f6..." (相关) ✅

召回率 = 5/5 = 1.0 (100%)
```

**场景2：部分检索**

```
问题："如何做西红柿鸡蛋？"
相关文档：parent_id = "a1b2c3d4e5f6..."（包含 5 个块）

检索结果（top_k=5）：
  1. parent_id = "a1b2c3d4e5f6..." (相关) ✅
  2. parent_id = "b2c3d4e5f6a1..." (不相关) ❌
  3. parent_id = "a1b2c3d4e5f6..." (相关) ✅
  4. parent_id = "c3d4e5f6a1b2..." (不相关) ❌
  5. parent_id = "a1b2c3d4e5f6..." (相关) ✅

检索到的相关块：3 个
所有相关块：5 个
召回率 = 3/5 = 0.6 (60%)
```

**场景3：完全未检索到**

```
问题："如何做西红柿鸡蛋？"
相关文档：parent_id = "a1b2c3d4e5f6..."（包含 5 个块）

检索结果（top_k=5）：
  1. parent_id = "b2c3d4e5f6a1..." (不相关) ❌
  2. parent_id = "c3d4e5f6a1b2..." (不相关) ❌
  3. parent_id = "d4e5f6a1b2c3..." (不相关) ❌
  4. parent_id = "e5f6a1b2c3d4..." (不相关) ❌
  5. parent_id = "f6a1b2c3d4e5..." (不相关) ❌

检索到的相关块：0 个
所有相关块：5 个
召回率 = 0/5 = 0.0 (0%)
```

## 四、实际生成的问题示例

假设 HowToCook 知识库中有以下文档：

```
dishes/
├── meat_dish/
│   ├── 西红柿鸡蛋.md
│   ├── 红烧肉.md
│   └── 宫保鸡丁.md
├── vegetable_dish/
│   ├── 麻婆豆腐.md
│   └── 地三鲜.md
└── soup/
    └── 西红柿鸡蛋汤.md
```

**生成的问题示例**：

1. `如何做西红柿鸡蛋？` (来自 `meat_dish/西红柿鸡蛋.md`)
2. `如何做红烧肉？` (来自 `meat_dish/红烧肉.md`)
3. `如何做宫保鸡丁？` (来自 `meat_dish/宫保鸡丁.md`)
4. `如何做麻婆豆腐？` (来自 `vegetable_dish/麻婆豆腐.md`)
5. `如何做地三鲜？` (来自 `vegetable_dish/地三鲜.md`)
6. `如何做西红柿鸡蛋汤？` (来自 `soup/西红柿鸡蛋汤.md`)

**如果使用 LLM 生成更多问题**：

对于 `西红柿鸡蛋.md`，可能生成：
1. `如何做西红柿鸡蛋？` (文件名生成)
2. `西红柿鸡蛋需要哪些原料？` (LLM 生成)
3. `西红柿鸡蛋的难度是多少？` (LLM 生成)
4. `西红柿鸡蛋的制作步骤是什么？` (LLM 生成)

## 五、召回率评估代码逻辑

```python
# 对每个问题
question = "如何做西红柿鸡蛋？"
relevant_parent_ids = ["a1b2c3d4e5f6..."]  # 相关文档的 parent_id

# 进行检索
query_vector = embed(question)
search_results = vector_store.search(query_vector, top_k=5)

# 获取检索到的 parent_id
retrieved_parent_ids = [metadata.parent_id for metadata in search_results]

# 计算召回率
relevant_retrieved = relevant_parent_ids ∩ retrieved_parent_ids

# 简化计算：如果检索到相关文档，召回率 = 1.0（因为整个文档都是相关的）
# 如果未检索到，召回率 = 0.0
if relevant_retrieved:
    recall = 1.0  # 检索到了相关文档
else:
    recall = 0.0  # 未检索到相关文档
```

**注意**：当前实现是简化版本，因为：
- 我们将整个文档的所有块都标记为相关
- 只要检索到该文档的任何一个块，就认为检索到了所有相关块
- 因此召回率要么是 1.0（检索到），要么是 0.0（未检索到）

## 六、改进方向（更精确的召回率计算）

如果要更精确地计算召回率，需要：

1. **标注每个块的相关性**：
```json
{
  "question": "西红柿鸡蛋需要哪些原料？",
  "relevant_chunks": ["chunk_1", "chunk_2"],  // 只有包含原料信息的块是相关的
  "irrelevant_chunks": ["chunk_3", "chunk_4"]  // 其他块不相关
}
```

2. **计算精确的召回率**：
```python
# 相关块总数
total_relevant_chunks = len(relevant_chunks)  # 例如：2 个

# 检索到的相关块
retrieved_relevant_chunks = [chunk for chunk in retrieved_chunks if chunk in relevant_chunks]

# 召回率
recall = len(retrieved_relevant_chunks) / total_relevant_chunks
```

## 七、总结

### 生成的问题类型

1. **主要类型**：`如何做{菜名}？`（从文件名生成）
2. **次要类型**：多样化问题（使用 LLM 生成）
   - 原料相关问题
   - 难度相关问题
   - 步骤相关问题
   - 其他相关问题

### 召回率评估

- **当前实现**：简化版本，只要检索到相关文档，召回率 = 1.0
- **改进方向**：标注每个块的相关性，计算精确的召回率

