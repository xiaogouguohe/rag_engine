# RAG 评估数据集的生成和验证思路

## 一、整体思路

### 核心假设

**如果一个问题是从某个文档中生成的，那么该文档的所有块都应该被认为是"相关"的。**

这个假设的合理性：
- ✅ 问题与文档内容相关（因为问题是从文档生成的）
- ✅ 文档的所有块都可能包含回答问题的信息
- ✅ 简化了标注工作（无需人工标注每个块的相关性）

### 验证流程

```
1. 从知识库中读取文档
   ↓
2. 从文档生成问题（使用 LLM 或文件名）
   ↓
3. 标注相关文档块（整个文档的所有块都是相关的）
   ↓
4. 生成评估数据集（JSON 格式）
   ↓
5. 使用数据集评估检索性能
   ↓
6. 计算评估指标（Recall, Precision, MRR, Hit Rate）
```

## 二、问题生成策略

### 策略1：从文件名生成问题（简单、快速）

```python
# 文件名：西红柿鸡蛋.md
# 生成问题：如何做西红柿鸡蛋？
question = f"如何做{file_name}？"
```

**优点**：
- ✅ 快速、无需调用 LLM
- ✅ 问题与文档高度相关
- ✅ 适合菜谱等结构化文档

**缺点**：
- ❌ 问题类型单一（都是"如何做..."）
- ❌ 可能不够多样化

### 策略2：使用 LLM 从文档内容生成问题（多样化）

```python
prompt = f"""基于以下菜谱内容，生成 {n} 个用户可能问的问题。
要求：
1. 问题应该与菜谱内容相关
2. 问题应该具体、可回答
3. 问题应该多样化（如：做法、原料、难度等）

菜谱内容：
{content_preview}

请生成问题，每行一个问题："""

questions = llm_client.generate(prompt)
```

**优点**：
- ✅ 问题多样化（做法、原料、难度等）
- ✅ 更贴近真实用户问题
- ✅ 可以生成多个问题

**缺点**：
- ❌ 需要调用 LLM，成本较高
- ❌ 生成的问题质量依赖于 LLM
- ❌ 可能生成不相关的问题

### 策略3：混合策略（推荐）

```python
# 1. 从文件名生成一个问题（保证相关性）
questions = [f"如何做{file_name}？"]

# 2. 使用 LLM 生成更多问题（增加多样性）
llm_questions = generate_questions_with_llm(content)
questions.extend(llm_questions)
```

**优点**：
- ✅ 结合两种策略的优点
- ✅ 保证至少有一个高质量问题
- ✅ 增加问题多样性

## 三、相关文档块标注策略

### 当前策略：整个文档都是相关的

```python
# 如果问题来自文档 A，那么文档 A 的所有块都是相关的
relevant_chunks = [parent_id]  # parent_id 代表整个文档
```

**逻辑**：
- 问题是从文档生成的 → 问题与文档相关
- 文档的所有块都可能包含回答问题的信息
- 因此，整个文档的所有块都是相关的

**优点**：
- ✅ 简单、自动化
- ✅ 无需人工标注
- ✅ 适合快速评估

**缺点**：
- ❌ 可能过于宽松（文档中某些块可能不相关）
- ❌ 无法评估细粒度的相关性

### 更精确的策略（可选，需要扩展）

**策略1：基于标题匹配**
```python
# 如果问题是关于"原料"的，只标记包含"原料"标题的块
if "原料" in question:
    relevant_chunks = [chunk for chunk in chunks if "原料" in chunk.metadata.get("二级标题", "")]
```

**策略2：使用 LLM 判断相关性**
```python
# 对每个块，使用 LLM 判断是否与问题相关
for chunk in chunks:
    is_relevant = llm_judge_relevance(question, chunk.text)
    if is_relevant:
        relevant_chunks.append(chunk.id)
```

**策略3：人工标注（最准确）**
```python
# 人工标注每个块的相关性
relevant_chunks = ["chunk_1", "chunk_3", "chunk_5"]  # 人工标注
```

## 四、评估流程

### 1. 检索阶段

```python
# 对每个问题，进行检索
query_vector = embed(question)
search_results = vector_store.search(query_vector, top_k=5)

# 获取检索到的文档块
retrieved_chunks = [result.chunk_id for result in search_results]
```

### 2. 计算指标

```python
# Recall@k：检索到的相关块 / 所有相关块
recall = len(retrieved_chunks ∩ relevant_chunks) / len(relevant_chunks)

# Precision@k：检索到的相关块 / 检索到的所有块
precision = len(retrieved_chunks ∩ relevant_chunks) / len(retrieved_chunks)

# MRR：第一个相关块的倒数排名
mrr = 1.0 / rank_of_first_relevant_chunk

# Hit Rate@k：至少找到一个相关块的查询比例
hit_rate = (至少找到一个相关块的查询数) / (总查询数)
```

### 3. 简化实现（当前）

由于我们使用 `parent_id` 标注相关块，评估时：

```python
# 相关文档的 parent_id
relevant_parent_ids = ["parent_1", "parent_2"]

# 检索结果中的 parent_id
retrieved_parent_ids = [metadata.parent_id for metadata in search_results]

# 如果检索到的 parent_id 在相关列表中，则认为相关
relevant_retrieved = relevant_parent_ids ∩ retrieved_parent_ids

# 计算指标
recall = len(relevant_retrieved) / len(relevant_parent_ids)
precision = len(relevant_retrieved) / len(retrieved_parent_ids)
```

## 五、思路的优缺点

### 优点

1. **自动化程度高**：
   - 无需人工标注
   - 可以快速生成大量评估数据

2. **适合快速评估**：
   - 可以快速了解系统的基本性能
   - 适合迭代开发阶段

3. **成本低**：
   - 主要使用文件名生成问题（无需 LLM）
   - 可选使用 LLM 增加问题多样性

### 缺点

1. **标注可能不够精确**：
   - 整个文档都标记为相关，可能过于宽松
   - 无法评估细粒度的相关性

2. **问题质量依赖生成策略**：
   - 文件名生成的问题类型单一
   - LLM 生成的问题可能不准确

3. **无法评估跨文档相关性**：
   - 只考虑问题来源文档的相关性
   - 无法评估其他文档的相关性

## 六、改进方向

### 1. 更精确的相关性标注

```python
# 使用 LLM 判断每个块的相关性
for chunk in document_chunks:
    relevance_score = llm_judge_relevance(question, chunk.text)
    if relevance_score > threshold:
        relevant_chunks.append(chunk.id)
```

### 2. 支持跨文档相关性

```python
# 使用向量相似度找到其他相关文档
similar_docs = vector_store.search(question, top_k=10)
# 标注所有相关文档的块
```

### 3. 人工验证和修正

```python
# 生成数据集后，人工验证和修正
# 1. 检查问题质量
# 2. 修正相关文档块标注
# 3. 添加遗漏的相关块
```

## 七、使用建议

### 快速评估（当前方案）

```bash
# 1. 生成评估数据集（使用文件名生成问题）
python generate_eval_dataset.py --kb-id recipes_kb --max-questions-per-doc 1

# 2. 评估检索性能
python evaluate_retrieval.py --kb-id recipes_kb --dataset eval_dataset.json
```

**适用场景**：
- 快速了解系统基本性能
- 迭代开发阶段
- 大规模知识库的初步评估

### 精确评估（需要扩展）

```bash
# 1. 生成评估数据集（使用 LLM 生成多样化问题）
python generate_eval_dataset.py --kb-id recipes_kb --max-questions-per-doc 5

# 2. 人工验证和修正数据集（可选）

# 3. 评估检索性能
python evaluate_retrieval.py --kb-id recipes_kb --dataset eval_dataset.json
```

**适用场景**：
- 正式评估
- 论文或报告
- 生产环境部署前的验证

## 八、总结

### 核心思路

1. **问题生成**：从文档自动生成问题（文件名 + LLM）
2. **相关性标注**：问题来源文档的所有块都是相关的（简化策略）
3. **评估指标**：Recall、Precision、MRR、Hit Rate

### 关键假设

- ✅ 问题与来源文档相关（合理）
- ✅ 文档的所有块都可能相关（简化假设，可能过于宽松）

### 适用场景

- ✅ 快速评估系统性能
- ✅ 迭代开发阶段
- ✅ 大规模知识库的初步评估
- ⚠️ 不适合需要精确评估的场景（需要人工标注）

