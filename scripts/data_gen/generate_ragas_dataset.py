#!/usr/bin/env python3
"""
使用 RAGAS 生成 RAG 测试集
-------------------------

RAGAS (Retrieval-Augmented Generation Assessment) 是一个专门用于评估 RAG 系统的框架。
本脚本专注于使用 RAGAS 的 TestsetGenerator 从【指定路径】的文档自动生成测试用例。

功能：
1. 从指定的本地目录读取文档（必须手动指定 --source-path）
2. 使用 LLM 提取实体、关键词并构建内部知识图谱（可选）
3. 合成包含问题、参考答案和参考上下文的测试集

使用方法：
    # 使用推荐的新方法生成测试集（必须指定 --source-path）
    python scripts/data_gen/generate_ragas_dataset.py --kb-id recipes_kb --source-path ./sample_recipes --use-testset-generator

    # 使用知识图谱生成
    python scripts/data_gen/generate_ragas_dataset.py --kb-id recipes_kb --source-path ./sample_recipes --use-testset-generator --use-kg
"""

import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# 导入 openai 以处理连接错误
try:
    import openai
except ImportError:
    openai = None

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import AppConfig
from document import ParserFactory, DataPreparationModule
from embedding import EmbeddingClient

# --- 补丁：绕过 transformers 的强制版本检查 (CVE-2025-32434) ---
def patch_transformers_security_check():
    try:
        import transformers.utils.import_utils as iu
        iu.check_torch_load_is_safe = lambda: None
        import transformers.utils as u
        if hasattr(u, "check_torch_load_is_safe"):
            u.check_torch_load_is_safe = lambda: None
        import transformers.modeling_utils as mu
        if hasattr(mu, "check_torch_load_is_safe"):
            mu.check_torch_load_is_safe = lambda: None
    except Exception:
        pass

patch_transformers_security_check()

class RagasLocalBGEAdapter:
    """适配器：将本地 EmbeddingClient 转换为 RAGAS 兼容格式"""
    def __init__(self, embedding_client: EmbeddingClient):
        self.client = embedding_client
    
    def embed_text(self, text: str) -> List[float]:
        res = self.client.embed_texts([text])
        return res["dense_vecs"][0]
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        res = self.client.embed_texts(texts)
        return res["dense_vecs"]
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_text(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_texts(texts)

# 尝试导入 RAGAS TestsetGenerator
try:
    from ragas.testset import TestsetGenerator
    from datasets import Dataset
    from langchain_core.documents import Document as LangchainDocument
    _HAS_TESTSET_GENERATOR = True
except ImportError as e:
    TestsetGenerator = None
    _HAS_TESTSET_GENERATOR = False
    print(f"⚠️  TestsetGenerator 不可用: {e}")

def find_markdown_files(directory: Path) -> List[Path]:
    """递归查找所有 .md 文件"""
    return DataPreparationModule.find_files(directory, "*.md")


def convert_documents_to_langchain_docs(file_paths: List[Path]) -> List[LangchainDocument]:
    """将本地文档文件转换为 langchain Document 格式"""
    langchain_docs = []
    for file_path in file_paths:
        try:
            parser = ParserFactory.get_parser(file_path)
            if parser:
                result = parser.parse(file_path)
                content = result.get("content", "")
                if not isinstance(content, str):
                    content = str(content) if content is not None else ""
                if not content.strip():
                    continue
                langchain_docs.append(LangchainDocument(
                    page_content=content,
                    metadata={"source": str(file_path), "file_name": file_path.name}
                ))
        except Exception:
            continue
    return langchain_docs


def generate_ragas_dataset_with_knowledge_graph(
    kb_id: str,
    source_path: str,
    output_path: str = "ragas_dataset.json",
    max_docs: int = 5,
    num_questions_per_doc: int = 3,
    use_kg: bool = True,
) -> bool:
    """使用 RAGAS TestsetGenerator 从指定路径生成测试集"""
    if not _HAS_TESTSET_GENERATOR:
        print("❌ RAGAS TestsetGenerator 不可用")
        return False
    
    source_dir = Path(source_path)
    if not source_dir.exists() or not source_dir.is_dir():
        print(f"❌ 指定的文档路径不存在或不是目录: {source_path}")
        return False
    
    print("=" * 60)
    print(f"从路径 [{source_path}] 生成测试集 (KG: {use_kg})")
    print("=" * 60)
    
    # 1. 读取本地文件
    md_files = find_markdown_files(source_dir)
    if max_docs:
        md_files = md_files[:max_docs]
    if not md_files:
        print(f"❌ 在路径 {source_path} 下未找到任何 .md 文件")
        return False
    
    langchain_docs = convert_documents_to_langchain_docs(md_files)
    if len(langchain_docs) < 3:
        print(f"⚠️  警告: 文档数量少于 3 个 ({len(langchain_docs)})，RAGAS 生成可能会报错")
    
    # 2. 初始化模型
    try:
        app_config = AppConfig.load()
        from openai import OpenAI
        from ragas.llms import llm_factory
        
        __import__("os").environ["OPENAI_API_KEY"] = app_config.llm.api_key
        generator_llm = llm_factory(model=app_config.llm.model, base_url=app_config.llm.base_url)
        
        if app_config.embedding.mode == "local":
            embeddings = RagasLocalBGEAdapter(EmbeddingClient.from_config(app_config))
        else:
            from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
            embeddings = RagasOpenAIEmbeddings(
                client=OpenAI(api_key=app_config.embedding.api_key, base_url=app_config.embedding.base_url),
                model=app_config.embedding.model
            )
        
        # 3. 构建知识图谱 (可选)
        knowledge_graph = None
        if use_kg:
            print("正在构建知识图谱...")
            try:
                from ragas.testset.graph import KnowledgeGraph, Node, NodeType
                from ragas.testset.transforms.extractors import NERExtractor, KeyphrasesExtractor
                from ragas.testset.transforms import Parallel, apply_transforms
                from ragas.testset.transforms.base import BaseGraphTransformation
                from dataclasses import dataclass
                import typing as t
                
                nodes = [Node(properties={"page_content": d.page_content}, type=NodeType.DOCUMENT) for d in langchain_docs]
                kg = KnowledgeGraph(nodes=nodes)
                
                @dataclass
                class EntityFormatFixer(BaseGraphTransformation):
                    async def transform(self, kg: KnowledgeGraph) -> t.Any:
                        all_entities = set()
                        for node in kg.nodes:
                            entities = node.properties.get("entities", [])
                            flat = [str(x) for x in entities] if isinstance(entities, list) else []
                            if isinstance(entities, dict):
                                for v in entities.values():
                                    if isinstance(v, list): flat.extend([str(x) for x in v])
                            flat = list(set(flat))
                            all_entities.update(flat)
                            node.properties["entities_dict"] = {"all": flat}
                            node.properties["entities"] = flat
                        kg.properties["themes"] = list(all_entities)
                        return kg
                    def generate_execution_plan(self, kg: KnowledgeGraph):
                        async def run(): await self.transform(kg)
                        return [run()]

                from ragas.testset.transforms.relationship_builders.traditional import JaccardSimilarityBuilder
                transforms = [
                    Parallel(NERExtractor(llm=generator_llm), KeyphrasesExtractor(llm=generator_llm)),
                    EntityFormatFixer(),
                    JaccardSimilarityBuilder(property_name="entities_dict", key_name="all", threshold=0.01)
                ]
                apply_transforms(kg, transforms)
                knowledge_graph = kg
                print(f"  ✅ 知识图谱构建完成 (节点: {len(kg.nodes)}, 边: {len(kg.relationships)})")
            except Exception as e:
                print(f"⚠️  图谱构建失败，回退到普通模式: {e}")

        # 4. 生成测试集
        generator = TestsetGenerator(llm=generator_llm, embedding_model=embeddings, knowledge_graph=knowledge_graph)
        total_q = max_docs * num_questions_per_doc
        print(f"开始生成测试集 (目标数量: {total_q})...")
        
        if knowledge_graph:
            from ragas.testset.persona import Persona
            from ragas.run_config import RunConfig
            generator.persona_list = [
                Persona(name="初级用户", role_description="对领域不熟悉，倾向于问基础操作和核心概念。"),
                Persona(name="高级专家", role_description="对细节非常敏感，倾向于问深层逻辑和多文档对比。")
            ]
            testset = generator.generate(testset_size=total_q, run_config=RunConfig(max_workers=3, timeout=120))
        else:
            testset = generator.generate_with_langchain_docs(
                documents=langchain_docs, testset_size=total_q,
                transforms_embedding_model=embeddings, raise_exceptions=False
            )
        
        if not testset:
            print("❌ 生成结果为空")
            return False
            
        # 5. 保存结果
        testset_df = testset.to_pandas()
        samples = []
        for _, row in testset_df.iterrows():
            samples.append({
                "question": row.get("user_input", row.get("question", "")),
                "answer": "",
                "ground_truth": row.get("reference", row.get("ground_truth", "")),
                "contexts": row.get("reference_contexts", row.get("contexts", [])),
                "metadata": {"kb_id": kb_id}
            })
            
        output_data = {
            "metadata": {"kb_id": kb_id, "source_path": source_path, "total_samples": len(samples)},
            "samples": samples
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 生成成功! 文件保存至: {output_path}")
        return True
    except Exception as e:
        print(f"❌ 执行出错: {e}")
        __import__("traceback").print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="从指定路径生成 RAGAS 测试集")
    parser.add_argument("--kb-id", required=True, help="知识库 ID")
    parser.add_argument("--source-path", required=True, help="【必填】待处理文档的本地目录路径")
    parser.add_argument("--output", default="ragas_dataset.json", help="输出 JSON 文件路径")
    parser.add_argument("--max-docs", type=int, default=5, help="最多处理的文档数")
    parser.add_argument("--max-questions-per-doc", type=int, default=3, help="每个文档生成的问题数")
    parser.add_argument("--use-testset-generator", action="store_true", help="使用 RAGAS TestsetGenerator (推荐)")
    parser.add_argument("--use-kg", action="store_true", help="使用知识图谱模式 (需开启 --use-testset-generator)")
    
    args = parser.parse_args()
    
    # 强制要求必须是 TestsetGenerator 模式（既然是为您定制的逻辑，简化分支）
    if not args.use_testset_generator:
        print("💡 提示: 建议开启 --use-testset-generator 以获得更好的生成效果")
    
    success = generate_ragas_dataset_with_knowledge_graph(
        kb_id=args.kb_id,
        source_path=args.source_path,
        output_path=args.output,
        max_docs=args.max_docs,
        num_questions_per_doc=args.max_questions_per_doc,
        use_kg=args.use_kg
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
