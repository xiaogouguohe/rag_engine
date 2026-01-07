# 文档切分的主要工作（面试准备）

## 一、整体流程概述

文档切分（Document Chunking）是 RAG 系统的核心环节，主要目的是将长文档切分成适合检索和生成的小块。我们的实现参考了 RAGFlow 和 C8 项目的最佳实践。

```
原始文档 → 解析 → 切分 → 元数据增强 → 向量化 → 存储
```

## 二、主要工作内容

### 1. 文档解析（Document Parsing）

**目的**：从不同格式的文件中提取纯文本内容

**实现方式**：
- 使用**解析器工厂模式**（ParserFactory），根据文件扩展名自动选择解析器
- 支持多种格式：TXT、Markdown（可扩展 HTML、PDF 等）
- 每个解析器继承 `BaseParser` 抽象基类，实现统一的 `parse()` 接口

**关键代码位置**：
- `document/parser_factory.py` - 解析器工厂
- `document/markdown_parser.py` - Markdown 解析器
- `document/parser.py` - TXT 解析器

### 2. 文档切分（Document Chunking）

**目的**：将长文档切分成语义完整的小块，便于检索

**两种切分策略**：

#### 策略一：Markdown 标题分割（结构感知，参考 C8）

- **适用场景**：Markdown 格式的文档（如菜谱、技术文档）
- **原理**：利用 Markdown 的标题层级（#、##、###）进行语义切分
- **优势**：
  - 保持语义完整性（每个块对应一个章节）
  - 保留标题信息作为上下文
  - 适合结构化文档

**实现**：
```python
# 使用 langchain 的 MarkdownHeaderTextSplitter
headers_to_split_on = [
    ("#", "主标题"),
    ("##", "二级标题"),
    ("###", "三级标题"),
]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(markdown_content)
```

#### 策略二：固定大小分割（通用方案）

- **适用场景**：普通文本、没有明显结构的文档
- **原理**：按字符数切分，支持重叠窗口（overlap）
- **优化**：尽量在句子边界处切分，避免截断句子

**实现**：
```python
# 固定大小 + 重叠窗口
chunker = TextChunker(
    chunk_size=500,      # 每块 500 字符
    chunk_overlap=50,    # 重叠 50 字符（保持上下文连贯性）
)
chunks = chunker.split_text(text)
```

### 3. 元数据保存（Metadata Management）⭐ **面试重点**

**目的**：保存文档的上下文信息，支持精确检索和溯源

#### 3.1 元数据的类型

**基础元数据**（每个文档块都包含）：
```python
{
    # 文档标识
    "parent_id": "文档的唯一ID（MD5哈希）",
    "chunk_id": "块的唯一ID（UUID）",
    "doc_type": "parent" | "child",  # 父文档 or 子文档
    
    # 文件信息
    "source": "文件完整路径",
    "file_name": "文件名",
    "file_type": "markdown" | "txt",
    "file_extension": ".md",
    
    # 块的位置信息
    "chunk_index": 0,  # 块在文档中的索引
    "total_chunks": 5,  # 文档总块数
    
    # 标题信息（Markdown 标题分割时）
    "主标题": "西红柿鸡蛋的做法",
    "二级标题": "必备原料",
    
    # 内容统计
    "content_length": 1234,  # 内容长度
    "line_count": 50,         # 行数
}
```

**增强元数据**（通过 MetadataEnhancer 提取，参考 C8）：
```python
{
    # 从文件路径提取
    "category": "荤菜" | "素菜" | "汤品",  # 从路径中提取分类
    "file_name": "西红柿鸡蛋",  # 文件名（不含扩展名）
    
    # 从内容提取
    "difficulty": "简单" | "中等" | "困难",  # 通过星号模式（★★）提取
}
```

#### 3.2 元数据的保存位置

**向量数据库（Milvus）中**：
- 每个向量记录包含：
  - `text`: 块的文本内容
  - `metadata`: JSON 格式的元数据字典
  - `vector`: 向量表示

**存储结构**：
```python
# 向量数据库中的一条记录
{
    "id": "chunk_uuid",
    "text": "这是文档块的内容...",
    "vector": [0.1, 0.2, ...],  # 1536 维向量
    "metadata": {
        "parent_id": "doc_hash",
        "chunk_id": "chunk_uuid",
        "file_name": "西红柿鸡蛋",
        "category": "荤菜",
        "difficulty": "简单",
        "主标题": "西红柿鸡蛋的做法",
        # ... 其他元数据
    }
}
```

#### 3.3 父子文档关系（Parent-Child Document Pattern）⭐ **面试重点**

**设计目的**：
- **检索时**：使用子文档（小块）进行精确检索
- **生成时**：使用父文档（完整文档）提供完整上下文

**实现方式**：
```python
# 1. 为每个文档生成唯一的 parent_id
parent_id = hashlib.md5(file_path.encode()).hexdigest()

# 2. 为每个块生成唯一的 chunk_id
chunk_id = str(uuid.uuid4())

# 3. 建立映射关系
parent_child_map = {
    "chunk_id_1": "parent_id",
    "chunk_id_2": "parent_id",
    # ...
}

# 4. 在元数据中保存关系
chunk_metadata = {
    "parent_id": parent_id,      # 指向父文档
    "chunk_id": chunk_id,        # 自己的ID
    "doc_type": "child",         # 标记为子文档
}
```

**使用场景**：
```python
# 检索时：找到相关的子文档块
child_chunks = vector_store.search(query_vector, top_k=5)

# 通过 parent_id 找到对应的父文档（完整内容）
parent_docs = data_module.get_parent_documents(child_chunks)
# 返回去重后的父文档列表，按相关性排序
```

### 4. 元数据增强（Metadata Enhancement）

**目的**：从文档内容和文件路径中提取结构化信息

**实现**（参考 C8）：
```python
class MetadataEnhancer:
    def enhance(self, metadata, file_path, content):
        # 1. 从路径提取分类
        if "meat_dish" in path_parts:
            metadata["category"] = "荤菜"
        
        # 2. 从内容提取难度
        if "★★" in content:
            metadata["difficulty"] = "简单"
        
        # 3. 提取文件名
        metadata["file_name"] = file_path.stem
        
        return metadata
```

**优势**：
- 支持基于元数据的过滤检索（如：只检索"简单"难度的菜谱）
- 支持分类统计和分析
- 提供更丰富的上下文信息

### 5. 向量化和存储

**流程**：
1. 将文本块通过 Embedding 模型转换为向量
2. 将向量、文本、元数据一起存储到 Milvus 向量数据库
3. 按 `kb_id`（知识库ID）组织，支持多知识库隔离

**存储代码**：
```python
# 向量化
vectors = embedding_client.embed_texts([chunk["text"] for chunk in chunks])

# 存储
vector_store.add_texts(
    kb_id="recipes_kb",
    texts=[chunk["text"] for chunk in chunks],
    vectors=vectors,
    metadatas=[chunk["metadata"] for chunk in chunks],
)
```

## 三、面试可能的问题和回答

### Q1: 为什么要保存元数据？

**A**: 
1. **检索优化**：支持基于元数据的过滤（如：只检索某个分类的文档）
2. **结果溯源**：知道检索到的内容来自哪个文件、哪个位置
3. **上下文重建**：通过 `parent_id` 可以找到完整的父文档
4. **统计分析**：支持知识库的分类统计、难度分布等

### Q2: 父子文档关系的作用是什么？

**A**:
- **检索精度**：使用小块（子文档）检索，更精确匹配用户问题
- **生成质量**：使用完整文档（父文档）生成，提供完整上下文，避免信息缺失
- **去重优化**：多个子块可能来自同一个父文档，检索后去重，避免重复生成

### Q3: Markdown 标题分割和固定大小分割的区别？

**A**:
- **Markdown 标题分割**：
  - 优点：保持语义完整性，适合结构化文档
  - 缺点：只适用于 Markdown 格式
- **固定大小分割**：
  - 优点：通用性强，适用于所有文本
  - 缺点：可能截断句子，语义不完整

**选择策略**：根据文档类型自动选择，Markdown 用标题分割，其他用固定大小。

### Q4: 元数据如何支持过滤检索？

**A**:
在检索时，可以：
1. 先进行向量相似度检索
2. 再根据元数据过滤（如：`category == "荤菜"`）
3. 或者先过滤，再检索（取决于向量数据库的能力）

**示例**：
```python
# 检索"简单"难度的菜谱
results = vector_store.search(
    query_vector,
    filter={"difficulty": "简单"},  # 元数据过滤
    top_k=5
)
```

### Q5: 重叠窗口（overlap）的作用？

**A**:
- **保持上下文连贯性**：相邻块之间有重叠，避免重要信息被截断
- **提高检索召回率**：即使切分点不理想，重叠部分也能保证信息不丢失
- **典型设置**：`chunk_size=500, chunk_overlap=50`（10% 重叠）

## 四、关键技术点总结

1. **解析器工厂模式**：统一接口，易于扩展新格式
2. **多种切分策略**：根据文档类型选择最优策略
3. **元数据体系**：完整的元数据保存和管理
4. **父子文档关系**：支持精确检索和完整生成
5. **元数据增强**：从路径和内容中提取结构化信息
6. **向量化存储**：文本、向量、元数据一体化存储

## 五、参考实现

- **RAGFlow**：工业级 RAG 系统，参考其解析器和分块策略
- **C8 项目**：菜谱 RAG 系统，参考其 Markdown 标题分割和元数据增强
- **LangChain**：使用其 `MarkdownHeaderTextSplitter` 工具类

