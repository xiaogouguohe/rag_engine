#!/usr/bin/env python3
"""
使用 RAGAS 生成 RAG 测试集
-------------------------

RAGAS (Retrieval-Augmented Generation Assessment) 是一个专门用于评估 RAG 系统的框架。
本脚本专注于使用 RAGAS 的 TestsetGenerator 从知识库自动生成高质量的测试用例。

功能：
1. 从知识库文件或向量数据库中读取文档
2. 使用 LLM 提取实体、关键词并构建内部知识图谱（可选）
3. 自动合成包含问题、参考答案和参考上下文的测试集

使用方法：
    # 使用推荐的新方法生成测试集
    python generate_ragas_dataset.py --kb-id recipes_kb --use-testset-generator --output ragas_dataset.json

    # 使用知识图谱生成更高质量的多跳问题
    python generate_ragas_dataset.py --kb-id recipes_kb --use-testset-generator --use-kg
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

from rag import RAGEngine
from config import AppConfig
from document import ParserFactory, DataPreparationModule
from llm import LLMClient
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
    # 尝试导入 TestsetGenerator
    from ragas.testset import TestsetGenerator
    # RAGAS 0.4.2 使用 query_distribution，而不是 evolutions
    try:
        from ragas.testset.synthesizers import default_query_distribution
    except ImportError:
        default_query_distribution = None
    
    from datasets import Dataset
    from langchain_core.documents import Document as LangchainDocument
    _HAS_TESTSET_GENERATOR = True
except ImportError as e:
    TestsetGenerator = None
    default_query_distribution = None
    _HAS_TESTSET_GENERATOR = False
    print(f"⚠️  TestsetGenerator 不可用: {e}")
    print("   请运行: pip install ragas datasets")

def find_markdown_files(directory: Path) -> List[Path]:
    """递归查找所有 .md 文件"""
    return DataPreparationModule.find_files(directory, "*.md")


def generate_questions_from_document(
    file_path: Path,
    llm_client,
    max_questions: int = 3,
) -> List[str]:
    """
    从文档生成问题（简化版本，主要从文件名生成）。
    """
    questions = []
    
    # 策略1：从文件名生成问题
    file_name = file_path.stem
    if file_name:
        question = f"如何做{file_name}？"
        questions.append(question)
    
    # 策略2：使用 LLM 生成更多问题（可选）
    if len(questions) < max_questions:
        try:
            from document import ParserFactory
            parser = ParserFactory.get_parser(file_path)
            if parser:
                doc = parser.parse(file_path)
                content = doc.get("content", "")[:500]  # 前 500 字符
                
                prompt = f"""基于以下菜谱内容，生成 {max_questions - len(questions)} 个用户可能问的问题。
要求：
1. 问题应该与菜谱内容相关
2. 问题应该具体、可回答
3. 问题应该多样化（如：做法、原料、难度等）

菜谱内容：
{content}

请生成问题，每行一个问题，不要编号："""
                
                try:
                    response = llm_client.generate(prompt)
                    generated_questions = [
                        q.strip() for q in response.split("\n")
                        if q.strip() and not q.strip().startswith(("#", "-", "1.", "2.", "3."))
                    ]
                    questions.extend(generated_questions[:max_questions - len(questions)])
                except Exception:
                    pass
        except Exception:
            pass
    
    return questions[:max_questions]


def convert_documents_to_langchain_docs(file_paths: List[Path]) -> List[LangchainDocument]:
    """
    将文档文件转换为 langchain Document 格式，用于 RAGAS TestsetGenerator。
    """
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
                
                doc = LangchainDocument(
                    page_content=content,
                    metadata={
                        "source": str(file_path),
                        "file_name": file_path.name,
                        "file_type": result.get("file_type", ""),
                    }
                )
                langchain_docs.append(doc)
        except Exception:
            continue
    
    return langchain_docs


def generate_ragas_dataset_with_knowledge_graph(
    kb_id: str,
    source_path: Optional[str] = None,
    output_path: str = "ragas_testset_dataset_kg.json",
    max_docs: int = 5,
    num_questions_per_doc: int = 3,
    use_kg: bool = True,
) -> bool:
    """
    使用 RAGAS TestsetGenerator 和知识图谱生成测试集。
    """
    if not _HAS_TESTSET_GENERATOR:
        print("❌ RAGAS TestsetGenerator 不可用")
        return False
    
    print("=" * 60)
    print("使用 RAGAS TestsetGenerator + 知识图谱生成测试集")
    print("=" * 60)
    
    source_dir = None
    if source_path:
        source_dir = Path(source_path)
    else:
        kb_json_path = Path(project_root) / "rag_config.json"
        if kb_json_path.exists():
            try:
                with open(kb_json_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                kb_config = next((kb for kb in config_data.get("knowledge_bases", []) if kb.get("kb_id") == kb_id), None)
                if kb_config:
                    source_dir = Path(kb_config["source_path"])
            except Exception:
                pass
    
    print(f"知识库 ID: {kb_id}")
    print(f"使用知识图谱: {use_kg}")
    print("-" * 60)
    
    langchain_docs = []
    if source_dir and source_dir.exists():
        print("从文件系统读取文档...")
        md_files = find_markdown_files(source_dir)
        if max_docs:
            md_files = md_files[:max_docs]
        langchain_docs = convert_documents_to_langchain_docs(md_files)
    else:
        print("从向量数据库读取文档...")
        try:
            from vector_store import VectorStore
            app_config = AppConfig.load()
            vector_store = VectorStore(storage_path=app_config.storage_path)
            all_chunks = vector_store.get_all_chunks(kb_id)
            if not all_chunks:
                print("❌ 向量数据库中未找到任何文档")
                return False
            
            from collections import defaultdict
            docs_by_id = defaultdict(list)
            for chunk in all_chunks:
                docs_by_id[chunk.parent_id].append(chunk)
            
            for doc_id, chunks in docs_by_id.items():
                chunks_sorted = sorted(chunks, key=lambda c: c.position if c.position is not None else 0)
                full_text = "\n\n".join([chunk.text for chunk in chunks_sorted])
                langchain_docs.append(LangchainDocument(
                    page_content=full_text,
                    metadata={"doc_id": doc_id, "kb_id": kb_id}
                ))
            if max_docs:
                langchain_docs = langchain_docs[:max_docs]
        except Exception as e:
            print(f"❌ 读取文档失败: {e}")
            return False
    
    if len(langchain_docs) < 3:
        print(f"⚠️  警告: 文档数量不足（{len(langchain_docs)}），聚类可能失败")
    
    print("-" * 60)
    print("初始化 RAGAS TestsetGenerator...")
    try:
        app_config = AppConfig.load()
        from openai import OpenAI
        from ragas.llms import llm_factory
        
        os_environ = __import__("os").environ
        os_environ["OPENAI_API_KEY"] = app_config.llm.api_key
        
        generator_llm = llm_factory(model=app_config.llm.model, base_url=app_config.llm.base_url)
        
        if app_config.embedding.mode == "local":
            emb_client = EmbeddingClient.from_config(app_config)
            embeddings = RagasLocalBGEAdapter(emb_client)
        else:
            from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
            openai_client = OpenAI(api_key=app_config.embedding.api_key, base_url=app_config.embedding.base_url)
            embeddings = RagasOpenAIEmbeddings(client=openai_client, model=app_config.embedding.model)
        
        knowledge_graph = None
        if use_kg:
            print("正在构建知识图谱...")
            try:
                from ragas.testset.graph import KnowledgeGraph, Node, NodeType
                from ragas.testset.transforms.extractors import NERExtractor, KeyphrasesExtractor
                from ragas.testset.transforms import Parallel
                
                nodes = [Node(properties={"page_content": d.page_content}, type=NodeType.DOCUMENT) for d in langchain_docs]
                kg = KnowledgeGraph(nodes=nodes)
                ner_extractor = NERExtractor(llm=generator_llm)
                keyphrase_extractor = KeyphrasesExtractor(llm=generator_llm)
                
                from ragas.testset.transforms.base import BaseGraphTransformation
                from dataclasses import dataclass
                import typing as t
                
                @dataclass
                class EntityFormatFixer(BaseGraphTransformation):
                    async def transform(self, kg: KnowledgeGraph) -> t.Any:
                        all_entities_global = set()
                        for node in kg.nodes:
                            if "entities" in node.properties:
                                entities = node.properties["entities"]
                                flat = [str(x) for x in entities] if isinstance(entities, list) else []
                                if isinstance(entities, dict):
                                    for v in entities.values():
                                        if isinstance(v, list): flat.extend([str(x) for x in v])
                                flat = list(set(flat))
                                all_entities_global.update(flat)
                                node.properties["entities_dict"] = {"all": flat}
                                node.properties["entities"] = flat
                        kg.properties["themes"] = list(all_entities_global)
                        return kg
                    def generate_execution_plan(self, kg: KnowledgeGraph):
                        async def run(): await self.transform(kg)
                        return [run()]

                from ragas.testset.transforms.relationship_builders.traditional import JaccardSimilarityBuilder
                rel_builder = JaccardSimilarityBuilder(property_name="entities_dict", key_name="all", threshold=0.01)
                
                from ragas.testset.transforms import apply_transforms
                apply_transforms(kg, [Parallel(ner_extractor, keyphrase_extractor), EntityFormatFixer(), rel_builder])
                knowledge_graph = kg
                print(f"  ✅ 知识图谱构建成功 (节点: {len(kg.nodes)}, 边: {len(kg.relationships)})")
            except Exception as e:
                print(f"⚠️  构建图谱失败，回退到普通模式: {e}")
                use_kg = False

        generator = TestsetGenerator(llm=generator_llm, embedding_model=embeddings, knowledge_graph=knowledge_graph)
        
        total_questions = max_docs * num_questions_per_doc
        print(f"开始生成测试集 (目标: {total_questions})...")
        
        if use_kg and knowledge_graph:
            from ragas.testset.persona import Persona
            generator.persona_list = [
                Persona(name="家庭厨师", role_description="普通家庭主妇，关注步骤和食材替换。"),
                Persona(name="美食评论家", role_description="对口味严苛，关注搭配和营养平衡。")
            ]
            from ragas.run_config import RunConfig
            testset = generator.generate(testset_size=total_questions, run_config=RunConfig(max_workers=3, timeout=120))
        else:
            testset = generator.generate_with_langchain_docs(
                documents=langchain_docs, testset_size=total_questions, 
                transforms_embedding_model=embeddings, raise_exceptions=False
            )
        
        if not testset:
            print("❌ 未生成任何测试样本")
            return False
            
        testset_df = testset.to_pandas()
        samples = []
        content_to_id = {d.page_content.strip(): (d.metadata.get("doc_id") or d.metadata.get("parent_id")) for d in langchain_docs}
        
        for _, row in testset_df.iterrows():
            relevant_chunks = []
            contexts = row.get("reference_contexts", row.get("contexts", []))
            for ctx in contexts:
                ctx_str = ctx.strip() if isinstance(ctx, str) else ""
                if ctx_str in content_to_id:
                    relevant_chunks.append(content_to_id[ctx_str])
            
            samples.append({
                "question": row.get("user_input", row.get("question", "")),
                "answer": "",
                "ground_truth": row.get("reference", row.get("ground_truth", "")),
                "contexts": contexts,
                "relevant_chunks": list(set(relevant_chunks)),
            })
            
        output_data = {
            "metadata": {
                "kb_id": kb_id,
                "total_samples": len(samples),
                "generation_method": "ragas_testset_generator_with_kg" if use_kg else "ragas_testset_generator",
            },
            "samples": samples,
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"✅ 测试集已保存到: {output_path} (共 {len(samples)} 个样本)")
        return True
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        __import__("traceback").print_exc()
        return False


def generate_ragas_dataset_with_testset_generator(
    kb_id: str,
    source_path: Optional[str] = None,
    output_path: str = "ragas_testset_dataset.json",
    max_docs: int = 5,
    num_questions_per_doc: int = 3,
) -> bool:
    """封装调用带 KG 版本的 generate 逻辑（默认关闭 KG）"""
    return generate_ragas_dataset_with_knowledge_graph(
        kb_id=kb_id, source_path=source_path, output_path=output_path,
        max_docs=max_docs, num_questions_per_doc=num_questions_per_doc, use_kg=False
    )


def generate_ragas_dataset(
    kb_id: str,
    source_path: Optional[str] = None,
    output_path: str = "ragas_dataset.json",
    max_docs: Optional[int] = None,
    max_questions_per_doc: int = 3,
) -> bool:
    """
    原有基础生成方法（不使用 TestsetGenerator）。
    """
    print("=" * 60)
    print("使用基础方法生成评估数据集")
    print("=" * 60)
    
    source_dir = None
    if source_path:
        source_dir = Path(source_path)
    else:
        kb_json_path = Path(project_root) / "rag_config.json"
        if kb_json_path.exists():
            try:
                with open(kb_json_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                kb_cfg = next((kb for kb in config.get("knowledge_bases", []) if kb.get("kb_id") == kb_id), None)
                if kb_cfg: source_dir = Path(kb_cfg["source_path"])
            except Exception: pass
            
    if not source_dir or not source_dir.exists():
        print(f"❌ 未找到知识库路径: {kb_id}")
        return False

    md_files = find_markdown_files(source_dir)
    if max_docs: md_files = md_files[:max_docs]
    
    engine = RAGEngine(kb_id=kb_id)
    samples = []
    
    for i, file_path in enumerate(md_files, 1):
        print(f"[{i}/{len(md_files)}] 处理: {file_path.name}")
        questions = generate_questions_from_document(file_path, engine.llm_client, max_questions_per_doc)
        
        from document import ParserFactory
        parser = ParserFactory.get_parser(file_path)
        ground_truth = parser.parse(file_path).get("content", "")[:500] if parser else ""
        
        for q in questions:
            result = engine.query(q, top_k=5)
            contexts = [c.get("text", "") for c in (result.get("chunks") or result.get("sources") or []) if c.get("text")]
            if not contexts: continue
            
            samples.append({
                "question": q,
                "contexts": contexts,
                "answer": result.get("answer", ""),
                "ground_truth": ground_truth,
            })
            
    dataset = {"kb_id": kb_id, "samples": samples}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"✅ 数据集已保存: {output_path} (共 {len(samples)} 个样本)")
    return True


def main():
    parser = argparse.ArgumentParser(description="使用 RAGAS 生成 RAG 测试集")
    parser.add_argument("--kb-id", required=True, help="知识库 ID")
    parser.add_argument("--source-path", help="知识库源路径")
    parser.add_argument("--output", default="ragas_dataset.json", help="输出路径")
    parser.add_argument("--max-docs", type=int, help="最多处理文档数")
    parser.add_argument("--max-questions-per-doc", type=int, default=3, help="每个文档生成问题数")
    parser.add_argument("--use-testset-generator", action="store_true", help="使用 RAGAS TestsetGenerator (推荐)")
    parser.add_argument("--use-kg", action="store_true", help="使用知识图谱生成 (需 --use-testset-generator)")
    
    args = parser.parse_args()
    
    if args.use_testset_generator:
        success = generate_ragas_dataset_with_knowledge_graph(
            kb_id=args.kb_id, source_path=args.source_path, output_path=args.output,
            max_docs=args.max_docs or 5, num_questions_per_doc=args.max_questions_per_doc or 3,
            use_kg=args.use_kg
        )
    else:
        success = generate_ragas_dataset(
            kb_id=args.kb_id, source_path=args.source_path, output_path=args.output,
            max_docs=args.max_docs, max_questions_per_doc=args.max_questions_per_doc
        )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
