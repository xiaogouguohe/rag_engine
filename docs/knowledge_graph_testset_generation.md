# 基于知识图谱的测试用例生成

## 概述

RAGAS 支持使用知识图谱（Knowledge Graph）来生成更高质量的测试用例，特别是对于多跳查询（multi-hop queries）。知识图谱可以帮助：

1. **识别实体和关系**：从文档中提取实体（如人名、地名、概念等）和它们之间的关系
2. **提高多跳查询质量**：基于实体关系生成更准确的多跳查询
3. **增加查询多样性**：生成更多样化的查询类型

## 使用方法

### 基本用法

使用知识图谱生成测试集：

```bash
python generate_ragas_dataset.py \
    --kb-id recipes_kb \
    --use-testset-generator \
    --use-kg \
    --output ragas_testset_kg.json \
    --max-docs 5 \
    --max-questions-per-doc 3
```

### 参数说明

- `--use-testset-generator`: 使用 RAGAS TestsetGenerator（必需）
- `--use-kg`: 启用知识图谱（可选，但推荐用于多跳查询）
- `--kb-id`: 知识库 ID
- `--output`: 输出文件路径
- `--max-docs`: 最多处理的文档数
- `--max-questions-per-doc`: 每个文档生成的问题数

### 不使用知识图谱

如果不想使用知识图谱（使用默认的 Embeddings 方式）：

```bash
python generate_ragas_dataset.py \
    --kb-id recipes_kb \
    --use-testset-generator \
    --output ragas_testset.json \
    --max-docs 5
```

## 知识图谱构建流程

### 1. 节点创建（Node Creation）

从文档中创建节点，每个文档块成为一个节点：

```python
nodes = []
for doc in langchain_docs:
    node = Node(
        properties={"page_content": doc.page_content},
        type=NodeType.CHUNK
    )
    nodes.append(node)
```

### 2. 信息提取（Information Extraction）

使用提取器从节点中提取信息：

- **NERExtractor**: 提取命名实体（人名、地名、组织等）
- **KeyphrasesExtractor**: 提取关键词短语

```python
ner_extractor = NERExtractor()
keyphrase_extractor = KeyphrasesExtractor()
```

### 3. 关系构建（Relationship Building）

基于提取的信息建立节点之间的关系：

```python
rel_builder = JaccardSimilarityBuilder(
    property_name="entities",
    key_name="PER",  # 基于人名实体
    new_property_name="entity_jaccard_similarity"
)
```

### 4. 知识图谱应用

将提取器和关系构建器应用到知识图谱：

```python
transforms = [
    Parallel(
        ner_extractor,
        keyphrase_extractor
    ),
    rel_builder
]

await apply_transforms(kg, transforms)
```

## 知识图谱 vs 默认方式

### 默认方式（Embeddings）

- **优点**：
  - 简单快速
  - 不需要额外的处理步骤
  - 适用于大多数场景

- **缺点**：
  - 多跳查询可能不够准确
  - 依赖语义相似度，可能错过重要的实体关系

### 知识图谱方式

- **优点**：
  - 更准确地识别实体和关系
  - 支持更复杂的多跳查询
  - 提高问题生成的多样性

- **缺点**：
  - 需要额外的处理时间
  - 如果文档之间没有共享实体，可能无法建立关系
  - 需要更多的计算资源

## 示例输出

使用知识图谱生成的测试集会包含：

```json
{
  "metadata": {
    "generation_method": "ragas_testset_generator_with_kg",
    "use_knowledge_graph": true
  },
  "samples": [
    {
      "question": "在咖喱炒蟹中，洋葱要怎么处理？",
      "answer": "",
      "ground_truth": "在咖喱炒蟹的制作过程中，洋葱需要先切成洋葱碎备用...",
      "contexts": [
        "<1-hop>...洋葱...",
        "<2-hop>...洋葱切成洋葱碎，备用..."
      ]
    }
  ]
}
```

## 故障排除

### 问题 1: 未找到任何关系

**症状**：
```
⚠️  警告: 未找到任何关系，可能因为：
   1. 文档之间没有共享的实体
   2. 文档内容差异太大
   3. 提取器未能提取到相关实体
```

**解决方案**：
1. 增加文档数量（使用 `--max-docs`）
2. 确保文档之间有共同的主题或实体
3. 检查文档内容是否包含可识别的实体（如人名、地名等）

### 问题 2: 知识图谱构建失败

**症状**：
```
⚠️  构建知识图谱失败: ...
```

**解决方案**：
1. 检查 RAGAS 版本（需要 0.4.2+）
2. 确保所有依赖已安装
3. 尝试不使用知识图谱（移除 `--use-kg`）

### 问题 3: 多跳查询仍然不准确

**解决方案**：
1. 确保文档之间有明确的实体关系
2. 尝试调整提取器参数
3. 检查文档内容质量

## 技术细节

### 提取器类型

- **NERExtractor**: 基于 LLM 的命名实体识别
- **KeyphrasesExtractor**: 提取关键词短语
- **EmbeddingExtractor**: 基于 Embeddings 的提取
- **SummaryExtractor**: 生成摘要

### 关系构建器类型

- **JaccardSimilarityBuilder**: 基于 Jaccard 相似度
- **CosineSimilarityBuilder**: 基于余弦相似度
- **Custom Relationship Builder**: 自定义关系构建器

### 并行处理

使用 `Parallel` 类可以并行执行多个提取器：

```python
transforms = [
    Parallel(
        NERExtractor(),
        KeyphrasesExtractor(),
        SummaryExtractor()
    ),
    rel_builder
]
```

## 参考

- [RAGAS 官方文档 - 知识图谱](https://www.aidoczh.com/ragas/concepts/test_data_generation/rag/index.html)
- [RAGAS GitHub](https://github.com/explodinggradients/ragas)

