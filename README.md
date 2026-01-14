## RAG 引擎（学习版）

这是一个参考 RAGFlow 思想、但大幅简化的本地终端 RAG 引擎项目，主要用于：

- 学习 RAG 的关键组成模块
- 面试时展示完整但清晰的工程结构

当前阶段目标：

- ✅ 搭建 **LLM 调用层** 和 **Embedding 调用层**
- ✅ 统一的配置管理（模型名称、API Key、Base URL 等）
- ✅ **向量存储与检索**（使用 Milvus Lite，参考 cloud-edge-milk-tea-agent）
- ✅ **文档解析与分块**（支持 TXT、Markdown，使用 Python 标准库）
- ✅ **完整的 RAG 引擎**（整合所有模块，实现文档处理和问答流程）

后续阶段（可逐步实现）：

- 终端交互体验 / 简单 CLI
- 支持更多文档格式（PDF、DOCX 等）
- 高级检索策略（混合检索、重排序等）

### 目录结构（当前）

```
rag_engine/
├── config/                  # 配置模块
│   ├── __init__.py
│   └── config.py           # 统一管理 LLM、Embedding 和存储路径
├── llm/                     # LLM 模块
│   ├── __init__.py
│   └── client.py          # LLM 客户端（对齐 RAGFlow，使用 OpenAI SDK）
├── embedding/               # Embedding 模块
│   ├── __init__.py
│   └── client.py          # Embedding 客户端（对齐 RAGFlow，使用 OpenAI SDK）
├── vector_store/            # 向量存储模块
│   ├── __init__.py
│   └── store.py           # 向量存储与检索（使用 Milvus Lite）
├── document/                # 文档处理模块
│   ├── __init__.py
│   ├── parser.py          # 文档解析器（支持 TXT、Markdown）
│   └── chunker.py         # 文本分块器（固定大小 + 重叠窗口）
├── rag/                    # RAG 引擎模块
│   ├── __init__.py
│   └── engine.py          # RAG 引擎核心（整合所有模块）
├── tests/                   # 测试文件
│   ├── test_llm_embedding.py
│   ├── test_vector_store.py
│   ├── test_document.py
│   └── test_rag_engine.py
├── requirements.txt         # 项目依赖
├── .env.example            # 环境变量配置模板
├── .gitignore              # Git 忽略文件
└── README.md               # 项目说明
```

### 实现方式说明

**LLM 调用方式对比：**

1. **cloud-edge-milk-tea-agent**：
   - 使用厂商官方 SDK（如 `dashscope.Generation.call()`）
   - 每个厂商都有自己的 SDK，需要分别适配

2. **RAGFlow（工业级）**：
   - 使用 **OpenAI SDK** (`openai.OpenAI`) 作为基础客户端
   - 通过 `base_url` + `model_name` 适配不同厂商（只要兼容 OpenAI 格式）
   - 对于不兼容的厂商，使用 **LiteLLM** 作为统一接口层
   - 支持工厂模式，动态加载不同模型类

3. **rag_engine（本项目，对齐 RAGFlow）**：
   - 使用 **OpenAI SDK** 作为基础（与 RAGFlow 一致）
   - 通过 `base_url` 配置适配不同厂商
   - 支持同步和异步调用
   - 后续可扩展为 LiteLLM 支持更多厂商

**优势：**
- ✅ 标准化：统一使用 OpenAI 兼容接口，代码更简洁
- ✅ 通用性：支持所有兼容 OpenAI 格式的厂商（通义、DeepSeek、Moonshot 等）
- ✅ 可扩展：后续可以轻松集成 LiteLLM 支持更多厂商

### 使用方式（示例）

1. 安装依赖：

```bash
cd rag_engine

# 使用默认源安装（可能较慢）
pip install -r requirements.txt

# 或使用国内镜像源加速（推荐）
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

**加速提示**：如果下载速度慢，可以使用国内镜像源。详细说明请参考 `docs/installation_tips.md`

2. 配置 API Key 和模型参数（两种方式任选其一）：

**方式一：使用 .env 文件（推荐，不会提交到 git）**

```bash
# 1. 复制配置模板
cp .env.example .env

# 2. 编辑 .env 文件，填入你的 API Key 和配置
# 使用通义千问的示例：
# RAG_LLM_API_KEY=sk-your-dashscope-api-key
# RAG_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
# RAG_LLM_MODEL=qwen-plus
# RAG_EMBEDDING_MODEL=text-embedding-v3
```

**方式二：使用系统环境变量**

```bash
# 使用 OpenAI
export RAG_LLM_API_KEY="your_openai_api_key"
export RAG_LLM_BASE_URL="https://api.openai.com/v1"
export RAG_LLM_MODEL="gpt-4.1-mini"
export RAG_EMBEDDING_MODEL="text-embedding-3-small"

# 或使用通义千问
export RAG_LLM_API_KEY="your_dashscope_api_key"
export RAG_LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export RAG_LLM_MODEL="qwen-plus"
export RAG_EMBEDDING_MODEL="text-embedding-v3"
```

**配置说明：**
- `.env` 文件优先级高于系统环境变量
- `.env` 文件已被 `.gitignore` 忽略，不会提交到 git
- 参考 `.env.example` 查看所有可配置项

3. 验证功能是否正常：

**测试文档解析和分块：**

```bash
# 运行文档解析和分块测试
python tests/test_document.py
```

测试脚本会自动：
- ✅ 测试 TXT 文件解析
- ✅ 测试 Markdown 文件解析
- ✅ 测试文本分块功能
- ✅ 测试带元数据的分块
- ✅ 测试完整流程（解析 + 分块）

**测试向量存储功能：**

```bash
# 运行向量存储测试
python tests/test_vector_store.py
```

测试脚本会自动：
- ✅ 生成测试向量
- ✅ 添加到向量存储
- ✅ 测试相似度搜索
- ✅ 验证持久化功能

**测试 LLM 和 Embedding：**

**方式一：使用测试脚本（推荐）**

```bash
# 运行测试脚本
python tests/test_llm_embedding.py
```

测试脚本会自动验证：
- ✅ 配置加载（.env 文件或环境变量）
- ✅ LLM 调用
- ✅ Embedding 调用
- ✅ 异步 LLM 调用（可选）

**方式二：在 Python 中手动测试**

```python
from config import AppConfig
from llm import LLMClient
from embedding import EmbeddingClient

# 加载配置（自动从 .env 文件或环境变量读取）
config = AppConfig.load()
llm = LLMClient.from_config(config)
emb = EmbeddingClient.from_config(config)

# 测试 LLM
print("LLM 测试:")
print(llm.generate("简单介绍一下 RAG 是什么？"))

# 测试 Embedding
print("\nEmbedding 测试:")
vecs = emb.embed_texts(["RAG 是一种检索增强生成技术"])
print(f"向量维度: {len(vecs[0])}, 向量数量: {len(vecs)}")
```

### 重要说明

**✅ 无需改代码！** 由于通义千问完全兼容 OpenAI 格式，你只需要：
1. 配置 API Key（使用 `.env` 文件或环境变量）
2. 确保 `base_url` 指向通义千问的兼容接口
3. 使用正确的模型名称（如 `qwen-plus`, `text-embedding-v3`）

代码会自动使用 OpenAI SDK 调用通义千问的接口，无需任何修改！

**🔒 安全提示：**
- `.env` 文件已被 `.gitignore` 忽略，不会提交到 git
- 请勿将包含真实 API Key 的 `.env` 文件提交到代码仓库
- 使用 `.env.example` 作为模板，团队成员可以复制并填入自己的配置

### 使用示例（完整 RAG 流程）

#### 方式一：使用命令行工具（推荐）

```bash
# 1. 加载单个文档（自动触发解析、切块、向量化、存储）
python main.py ingest --kb-id my_kb --file path/to/document.md

# 2. 批量加载目录中的所有文档
python main.py ingest --kb-id my_kb --dir path/to/docs

# 3. 查询知识库
python main.py query --kb-id my_kb --question "什么是 RAG？"

# 4. 查看知识库统计信息
python main.py stats --kb-id my_kb
```

#### 方式二：使用 Python API

```python
from rag import RAGEngine

# 初始化引擎
engine = RAGEngine(kb_id="my_knowledge_base")

# 1. 处理文档（解析 → 分块 → 向量化 → 存储）
# 调用 ingest_document 会自动触发整个流程
result = engine.ingest_document("example.txt")
print(f"文档处理完成，共 {result['chunks_count']} 个块")

# 2. 问答（问题 → 向量化 → 检索 → 生成回答）
answer = engine.query("什么是 RAG？", top_k=5)
print(f"回答: {answer['answer']}")
print(f"参考了 {len(answer.get('chunks', []))} 个文档片段")

# 3. 查看统计信息
stats = engine.get_stats()
print(f"知识库中有 {stats['vector_count']} 个向量")
```

### 完整流程说明

**文档处理流程（触发方式：`engine.ingest_document(file_path)` 或 `python main.py ingest`）：**
```
文档文件 → Parser → 文本内容 → Chunker → 文本块 → Embedding → 向量 → VectorStore → 存储
```

**问答流程（触发方式：`engine.query(question)` 或 `python main.py query`）：**
```
用户问题 → Embedding → 查询向量 → VectorStore → 检索相关块 → 拼接上下文 → LLM → 生成回答
```

### 如何触发知识库的加载和切块？

**关键方法：`RAGEngine.ingest_document(file_path)`**

当你调用这个方法时，会自动触发以下流程：

1. **解析文档**：根据文件类型（TXT/Markdown）选择合适的解析器
2. **切块**：
   - 如果是 Markdown 且启用标题分割（默认启用），会按标题结构切分（参考 C8）
   - 否则按固定大小切分
3. **元数据增强**：提取文件路径信息、内容信息（如难度）等
4. **向量化**：使用 Embedding 模型将文本块转换为向量
5. **存储**：将向量和元数据存储到 Milvus Lite 向量数据库

**使用示例：**

```python
from rag import RAGEngine

# 初始化引擎
engine = RAGEngine(kb_id="my_kb")

# 触发加载和切块（一行代码完成所有流程）
result = engine.ingest_document("document.md")
# 此时文档已经被解析、切块、向量化并存储到向量数据库

# 或者使用命令行
# python main.py ingest --kb-id my_kb --file document.md
```

### 从配置文件加载知识库（推荐）⭐

**方式一：使用 JSON 配置文件（推荐）**

1. 编辑 `knowledge_bases.json` 文件：

```json
{
  "knowledge_bases": [
    {
      "kb_id": "recipes_kb",
      "source_path": "../HowToCook/dishes",
      "file_pattern": "*.md",
      "use_markdown_header_split": true,
      "description": "菜谱知识库"
    }
  ]
}
```

2. 运行加载脚本：

```bash
# 加载所有配置的知识库
python3 main.py

# 只加载指定的知识库
python3 main.py --kb-id recipes_kb
```

**方式二：使用环境变量**

在 `.env` 文件中添加：

```bash
# 格式：KB_ID:SOURCE_PATH:FILE_PATTERN
# 多个知识库用逗号分隔
RAG_KNOWLEDGE_BASES=recipes_kb:../HowToCook/dishes:*.md
```

然后运行：

```bash
python3 main.py
```

**方式三：命令行指定路径**

```bash
# 使用启动脚本加载指定目录
python3 load_recipes.py --kb-id recipes_kb --dir ../HowToCook/dishes
```

## RAG 系统评估标准

RAG 系统的评估分为三个层次：

### 1. 检索阶段评估指标

- **召回率（Recall）**：检索到的相关文档块数量 / 所有相关文档块数量（最重要）
- **精确率（Precision）**：检索到的相关文档块数量 / 检索到的所有文档块数量
- **F1 分数**：召回率和精确率的调和平均数
- **MRR（Mean Reciprocal Rank）**：平均倒数排名，衡量第一个相关文档块的平均排名
- **NDCG（Normalized Discounted Cumulative Gain）**：归一化折损累积增益，考虑文档块的相关性程度和排名位置
- **Hit Rate**：至少检索到一个相关文档块的查询比例

### 2. 生成阶段评估指标

- **忠实度（Faithfulness）**：生成的回答是否忠实于检索到的文档内容（防止幻觉）
- **相关性（Relevance）**：生成的回答是否与问题相关
- **BERTScore**：基于 BERT 嵌入的语义相似度

### 3. 端到端评估指标

- **答案准确性（Answer Accuracy）**：生成的回答是否正确的比例
- **用户满意度（User Satisfaction）**：用户对系统回答的满意程度

**详细说明**：请参考 `docs/rag_evaluation_metrics.md`

### 使用 RAGAS 生成和评估数据集（推荐）⭐

RAGAS 是专门用于评估 RAG 系统的框架，提供更专业的评估指标。

**安装 RAGAS**：
```bash
pip install ragas datasets
```

**生成评估数据集并评估**：
```bash
# 生成数据集并使用 RAGAS 评估
python generate_ragas_dataset.py --kb-id recipes_kb --output ragas_dataset.json

# 只生成数据集，不评估
python generate_ragas_dataset.py --kb-id recipes_kb --output ragas_dataset.json --no-evaluate

# 只评估已存在的数据集
python generate_ragas_dataset.py --evaluate-only ragas_dataset.json
```

**RAGAS 评估指标**：
- **Faithfulness（忠实度）**：生成的回答是否忠实于检索到的文档内容
- **Answer Relevancy（回答相关性）**：生成的回答是否与问题相关
- **Context Precision（上下文精确率）**：检索到的文档块是否相关
- **Context Recall（上下文召回率）**：是否检索到了所有相关文档块

**生成的数据集格式**：
```json
{
  "kb_id": "recipes_kb",
  "samples": [
    {
      "question": "如何做西红柿鸡蛋？",
      "contexts": ["检索到的文档块1", "检索到的文档块2"],
      "answer": "RAG 系统生成的回答",
      "ground_truth": "标准答案（文档内容）"
    }
  ]
}
```

### 自动生成评估数据集

不想手动构造测试数据？可以使用脚本自动生成：

```bash
# 从配置文件的知识库生成评估数据集
python generate_eval_dataset.py --kb-id recipes_kb --output eval_dataset.json

# 指定知识库源路径
python generate_eval_dataset.py --kb-id recipes_kb --source-path ../HowToCook/dishes --output eval_dataset.json

# 快速测试（只处理前 10 个文档）
python generate_eval_dataset.py --kb-id recipes_kb --max-docs 10 --output eval_dataset_test.json
```

脚本会自动：
1. 扫描知识库中的所有 .md 文件
2. 从文档中提取或生成问题（使用 LLM）
3. 标注相关文档块（基于文档结构）
4. 生成评估数据集（JSON 格式）

生成的数据集格式：
```json
{
  "kb_id": "recipes_kb",
  "source_path": "../HowToCook/dishes",
  "total_samples": 150,
  "samples": [
    {
      "id": "uuid",
      "question": "如何做西红柿鸡蛋？",
      "source_document": "meat_dish/西红柿鸡蛋.md",
      "parent_id": "doc_hash",
      "relevant_chunks": ["doc_hash"],
      "metadata": {
        "file_name": "西红柿鸡蛋.md",
        "file_path": "..."
      }
    }
  ]
}
```

现在你有了一个完整的、可用的 RAG 系统！

