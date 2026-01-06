## RAG 引擎（学习版）

这是一个参考 RAGFlow 思想、但大幅简化的本地终端 RAG 引擎项目，主要用于：

- 学习 RAG 的关键组成模块
- 面试时展示完整但清晰的工程结构

当前阶段目标：

- ✅ 搭建 **LLM 调用层** 和 **Embedding 调用层**
- ✅ 统一的配置管理（模型名称、API Key、Base URL 等）
- ✅ **向量存储与检索**（使用 Milvus Lite，参考 cloud-edge-milk-tea-agent）

后续阶段（可逐步实现）：

- 文档解析与分块
- 完整的 RAG 问答流程
- 终端交互体验 / 简单 CLI

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
├── tests/                   # 测试文件
│   ├── test_llm_embedding.py
│   └── test_vector_store.py
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
pip install -r requirements.txt
```

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

后续模块（文档解析、索引、检索等）可以在此基础上逐步补充。

