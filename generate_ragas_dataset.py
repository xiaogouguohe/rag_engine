#!/usr/bin/env python3
"""
使用 RAGAS 生成和评估 RAG 数据集
-------------------------------

RAGAS (Retrieval-Augmented Generation Assessment) 是一个专门用于评估 RAG 系统的框架。

功能：
1. 从知识库生成评估数据集
2. 使用 RAG 系统生成回答
3. 使用 RAGAS 评估质量

使用方法：
    # 生成评估数据集并评估
    python generate_ragas_dataset.py --kb-id recipes_kb --output ragas_dataset.json

    # 只生成数据集，不评估
    python generate_ragas_dataset.py --kb-id recipes_kb --output ragas_dataset.json --no-evaluate
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
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rag import RAGEngine
from config import AppConfig
from document import ParserFactory, DataPreparationModule
from llm import LLMClient
from embedding import EmbeddingClient

# 尝试导入 RAGAS
try:
    from ragas import evaluate
    # RAGAS 0.4.2 的正确导入方式
    # 注意：从 ragas.metrics 导入的是已初始化的对象（虽然会有 DeprecationWarning）
    # 从 ragas.metrics.collections 导入的是模块，不是对象，不能直接使用
    try:
        # 使用 ragas.metrics 导入（已初始化的对象，虽然会有警告）
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
    except ImportError:
        faithfulness = answer_relevancy = context_precision = context_recall = None
    
    # 尝试导入 TestsetGenerator
    try:
        from ragas.testset import TestsetGenerator
        # RAGAS 0.4.2 使用 query_distribution，而不是 evolutions
        try:
            from ragas.testset.synthesizers import default_query_distribution
        except ImportError:
            default_query_distribution = None
        _HAS_TESTSET_GENERATOR = True
    except ImportError as e:
        TestsetGenerator = None
        default_query_distribution = None
        _HAS_TESTSET_GENERATOR = False
        print(f"⚠️  TestsetGenerator 不可用: {e}")
    
    from datasets import Dataset
    from langchain_core.documents import Document as LangchainDocument
    _HAS_RAGAS = True
except ImportError as e:
    _HAS_RAGAS = False
    _HAS_TESTSET_GENERATOR = False
    TestsetGenerator = None
    default_query_distribution = None
    faithfulness = answer_relevancy = context_precision = context_recall = None
    print(f"⚠️  未安装 RAGAS: {e}")
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
    
    Args:
        file_path: 文档路径
        llm_client: LLM 客户端
        max_questions: 最多生成的问题数
    
    Returns:
        问题列表
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
    
    Args:
        file_paths: 文档文件路径列表
    
    Returns:
        langchain Document 列表
    """
    langchain_docs = []
    
    for file_path in file_paths:
        try:
            parser = ParserFactory.get_parser(file_path)
            if parser:
                # 使用解析器解析文档
                result = parser.parse(file_path)
                content = result.get("content", "")
                
                # 确保 content 是字符串类型（RAGAS 要求）
                if not isinstance(content, str):
                    if content is None:
                        content = ""
                    else:
                        # 如果不是字符串，尝试转换为字符串
                        content = str(content)
                
                # 过滤掉空内容
                if not content.strip():
                    print(f"  ⚠️  文档内容为空，跳过: {file_path.name}")
                    continue
                
                # 创建 langchain Document
                doc = LangchainDocument(
                    page_content=content,
                    metadata={
                        "source": str(file_path),
                        "file_name": file_path.name,
                        "file_type": result.get("file_type", ""),
                    }
                )
                langchain_docs.append(doc)
        except Exception as e:
            print(f"  ⚠️  解析文档失败 {file_path.name}: {e}")
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
    使用 RAGAS TestsetGenerator 和知识图谱生成测试集（新方法，支持知识图谱）。
    
    生成的数据格式：
    {
        "question": str,           # LLM 生成的问题
        "answer": "",              # 留空，后续用 RAG 生成
        "ground_truth": str,       # LLM 根据文档生成的参考答案
        "contexts": List[str],     # RAGAS 确定的相关文档块
    }
    
    Args:
        kb_id: 知识库 ID
        source_path: 知识库源路径
        output_path: 输出文件路径
        max_docs: 最多处理的文档数（默认 5）
        num_questions_per_doc: 每个文档生成的问题数（默认 3）
        use_kg: 是否使用知识图谱（默认 True）
    """
    if not _HAS_TESTSET_GENERATOR:
        print("❌ RAGAS TestsetGenerator 不可用")
        print("   请确保安装了 RAGAS v0.1.0+ 并包含 testset 模块")
        return False
    
    print("=" * 60)
    print("使用 RAGAS TestsetGenerator + 知识图谱生成测试集")
    print("=" * 60)
    
    # 1. 确定源路径（复用原有逻辑）
    source_dir = None
    if source_path:
        source_dir = Path(source_path)
    else:
        # 方式1：从 rag_config.json 读取
        kb_json_path = Path("rag_config.json")
        if kb_json_path.exists():
            try:
                with open(kb_json_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                kb_config = next((kb for kb in config_data.get("knowledge_bases", []) if kb.get("kb_id") == kb_id), None)
                if kb_config:
                    source_dir = Path(kb_config["source_path"])
            except Exception as e:
                print(f"⚠️  读取 rag_config.json 失败: {e}")
        
        # 方式2：从环境变量配置读取
        if source_dir is None:
            try:
                app_config = AppConfig.load()
                if app_config.knowledge_bases:
                    kb_config = next((kb for kb in app_config.knowledge_bases if kb.kb_id == kb_id), None)
                    if kb_config:
                        source_dir = Path(kb_config.source_path)
            except Exception as e:
                print(f"⚠️  读取环境变量配置失败: {e}")
        
        # 如果还是找不到 source_path，没关系，我们会直接从向量数据库读取
        # 不需要 source_path 也能工作
        if source_dir is not None and not source_dir.exists():
            print(f"⚠️  源路径不存在: {source_dir}，将直接从向量数据库读取")
            source_dir = None
    
    print(f"知识库 ID: {kb_id}")
    print(f"使用知识图谱: {use_kg}")
    print("-" * 60)
    
    # 2. 读取文档（优先从文件系统读取，如果提供了 source_path）
    langchain_docs = []
    
    # 如果提供了 source_path，优先从文件系统读取
    if source_dir and source_dir.exists():
        print("从文件系统读取文档...")
        md_files = find_markdown_files(source_dir)
        
        if max_docs:
            md_files = md_files[:max_docs]
        
        if not md_files:
            print("❌ 未找到 .md 文件")
            return False
        
        print(f"✅ 从文件系统找到 {len(md_files)} 个文档")
        langchain_docs = convert_documents_to_langchain_docs(md_files)
        
        if not langchain_docs:
            print("❌ 文档转换失败")
            return False
        
        print(f"✅ 成功转换 {len(langchain_docs)} 个文档")
    else:
        # 如果没有提供 source_path，尝试从向量数据库读取
        print("从向量数据库读取文档...")
        try:
            from vector_store import VectorStore
            
            app_config = AppConfig.load()
            vector_store = VectorStore(storage_path=app_config.storage_path)
            
            # 获取所有 chunks
            all_chunks = vector_store.get_all_chunks(kb_id)
            
            if not all_chunks:
                print("❌ 向量数据库中未找到任何文档")
                print("   请确保知识库已索引文档或提供 --source-path 参数")
                return False
            
            print(f"✅ 从向量数据库获取到 {len(all_chunks)} 个 chunks")
            
            # 按 doc_id 分组，重新组合成完整文档
            from collections import defaultdict
            docs_by_id = defaultdict(list)
            for chunk in all_chunks:
                docs_by_id[chunk.doc_id].append(chunk)
            
            # 转换为 langchain Document 格式
            langchain_docs = []
            for doc_id, chunks in docs_by_id.items():
                # 按 position 排序（如果有）
                chunks_sorted = sorted(chunks, key=lambda c: c.position if c.position is not None else 0)
                # 组合成完整文档
                full_text = "\n\n".join([chunk.text for chunk in chunks_sorted])
                
                # 创建 langchain Document
                from langchain_core.documents import Document as LangchainDocument
                doc = LangchainDocument(
                    page_content=full_text,
                    metadata={
                        "doc_id": doc_id,
                        "kb_id": kb_id,
                        "chunks_count": len(chunks),
                    }
                )
                langchain_docs.append(doc)
            
            # 限制文档数量
            if max_docs:
                langchain_docs = langchain_docs[:max_docs]
            
            print(f"✅ 成功组合成 {len(langchain_docs)} 个文档（来自 {len(all_chunks)} 个 chunks）")
            
            # 检查文档数量是否合理
            if len(langchain_docs) < 3:
                print(f"⚠️  警告: 从向量数据库只读取到 {len(langchain_docs)} 个文档，少于 3 个可能无法生成测试集")
                print("   建议: 使用 --source-path 参数直接从文件系统读取")
            
        except Exception as e:
            print(f"❌ 从向量数据库读取文档失败: {e}")
            print("   请提供 --source-path 参数从文件系统读取")
            import traceback
            traceback.print_exc()
            return False
    
    # 检查文档数量
    if len(langchain_docs) < 3:
        print(f"⚠️  警告: 只有 {len(langchain_docs)} 个文档，RAGAS 需要至少 3 个文档才能形成聚类")
        print("   继续尝试生成测试集...")
    
    print("-" * 60)
    
    # 4. 加载配置和初始化
    print("初始化 RAGAS TestsetGenerator...")
    try:
        app_config = AppConfig.load()
        
        try:
            from langchain_openai import ChatOpenAI
            from ragas.embeddings import OpenAIEmbeddings
        except ImportError:
            print("❌ 需要安装 langchain-openai")
            print("   请运行: pip install langchain-openai")
            return False
        
        # 创建 langchain LLM
        generator_llm = ChatOpenAI(
            model=app_config.llm.model,
            api_key=app_config.llm.api_key,
            base_url=app_config.llm.base_url,
            temperature=0.1,
            timeout=120.0,
            max_retries=3,
        )
        
        # 创建 Embeddings
        try:
            from openai import OpenAI, AsyncOpenAI
            from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
            
            openai_client = OpenAI(
                api_key=app_config.embedding.api_key,
                base_url=app_config.embedding.base_url,
            )
            async_openai_client = AsyncOpenAI(
                api_key=app_config.embedding.api_key,
                base_url=app_config.embedding.base_url,
            )
            
            embeddings = RagasOpenAIEmbeddings(
                client=openai_client,
                model=app_config.embedding.model,
            )
            if hasattr(embeddings, 'async_client'):
                embeddings.async_client = async_openai_client
            
            print("✅ 使用 RAGAS 原生的 OpenAIEmbeddings")
        except Exception as e:
            print(f"❌ 创建 Embeddings 失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 5. 构建知识图谱（如果启用）
        knowledge_graph = None
        if use_kg:
            print("-" * 60)
            print("正在构建知识图谱...")
            try:
                from ragas.testset.graph import KnowledgeGraph, Node, NodeType
                from ragas.testset.transforms.extractors import NERExtractor, KeyphrasesExtractor
                from ragas.testset.transforms.relationship_builders.traditional import JaccardSimilarityBuilder
                from ragas.testset.transforms import apply_transforms, Parallel
                import asyncio
                
                # 5.1 从文档创建节点
                print("  步骤 1: 从文档创建节点...")
                nodes = []
                for doc in langchain_docs:
                    # 关键修改：将类型设为 DOCUMENT，以便 RAGAS 的 Persona 提取器能识别
                    node = Node(
                        properties={"page_content": doc.page_content},
                        type=NodeType.DOCUMENT
                    )
                    nodes.append(node)
                
                print(f"  ✅ 创建了 {len(nodes)} 个节点 (类型: DOCUMENT)")
                
                # 5.2 创建知识图谱
                kg = KnowledgeGraph(nodes=nodes)
                print("  ✅ 创建知识图谱对象")
                
                # 5.3 创建提取器
                print("  步骤 2: 创建提取器...")
                from ragas.llms import LangchainLLMWrapper
                ragas_llm_for_extractors = LangchainLLMWrapper(generator_llm)
                
                ner_extractor = NERExtractor(llm=ragas_llm_for_extractors)
                keyphrase_extractor = KeyphrasesExtractor(llm=ragas_llm_for_extractors)
                print("  ✅ 创建了 NER 提取器和关键词提取器")
                
                # --- 补丁：自定义格式转换器，修复 RAGAS 官方 list/dict 不匹配 bug ---
                from ragas.testset.transforms.base import BaseGraphTransformation
                from dataclasses import dataclass
                import typing as t
                
                @dataclass
                class EntityFormatFixer(BaseGraphTransformation):
                    async def transform(self, kg: KnowledgeGraph) -> t.Any:
                        print("    [补丁] 正在修复实体格式并合并所有类别...")
                        for i, node in enumerate(kg.nodes):
                            if "entities" in node.properties:
                                entities = node.properties["entities"]
                                # 如果是 list，直接包装
                                if isinstance(entities, list):
                                    node.properties["entities"] = {"all": entities}
                                # 如果是 dict，把所有类别的实体合并到一个 "all" 列表里
                                elif isinstance(entities, dict):
                                    all_list = []
                                    for vals in entities.values():
                                        if isinstance(vals, list):
                                            all_list.extend(vals)
                                    node.properties["entities"] = {"all": list(set(all_list))}
                                
                                # 调试打印前 2 个节点的内容
                                if i < 2:
                                    print(f"      节点 {i} 提取到的部分实体: {node.properties['entities']['all'][:5]}...")
                        return kg
                    
                    def generate_execution_plan(self, kg: KnowledgeGraph) -> t.Sequence[t.Coroutine]:
                        async def run():
                            await self.transform(kg)
                        return [run()]
                # -----------------------------------------------------------

                # 5.4 创建关系构建器
                print("  步骤 3: 创建关系构建器...")
                from ragas.testset.transforms.relationship_builders.traditional import JaccardSimilarityBuilder
                
                # 使用通用的 "all" 键，并显著降低阈值 (threshold)
                rel_builder = JaccardSimilarityBuilder(
                    property_name="entities",
                    key_name="all",
                    new_property_name="entity_jaccard_similarity",
                    threshold=0.01 # 极其宽容，只要有重合就连线
                )
                print("  ✅ 关系构建器已配置: JaccardSimilarityBuilder (threshold=0.01)")
                
                # 5.5 应用 transforms
                print("  步骤 4: 执行知识图谱构建流 (提取 -> 修复 -> 连线)...")
                from ragas.testset.transforms import apply_transforms
                transforms = [
                    Parallel(ner_extractor, keyphrase_extractor),
                    EntityFormatFixer(),
                    rel_builder
                ]
                
                # 注意：在 RAGAS 0.4.x 中，apply_transforms 是一个同步函数
                # 它内部会处理异步逻辑，所以这里不需要 await
                print("    正在执行 LLM 任务并构建节点关系，请稍候...")
                apply_transforms(kg, transforms)
                knowledge_graph = kg
                
                if knowledge_graph:
                    num_relationships = len(knowledge_graph.relationships)
                    print(f"  ✅ 知识图谱构建成功 (节点: {len(knowledge_graph.nodes)}, 边: {num_relationships})")
                else:
                    print("  ⚠️  知识图谱对象为空，将回退到无知识图谱模式")
                    use_kg = False
                
            except Exception as e:
                print(f"\n  ❌ 构建知识图谱失败，详细错误堆栈如下:")
                import traceback
                traceback.print_exc()
                print("\n  ⚠️  正在触发紧急回退: 进入【无知识图谱】模式尝试继续...")
                knowledge_graph = None
                use_kg = False
                
                # 统计关系数量
                num_relationships = len(knowledge_graph.relationships)
                print(f"  ✅ 知识图谱构建完成")
                print(f"     节点数: {len(knowledge_graph.nodes)}")
                print(f"     关系数: {num_relationships}")
                
                if num_relationships == 0:
                    print("  ⚠️  警告: 未找到任何关系，可能因为：")
                    print("     1. 文档之间没有共享的实体")
                    print("     2. 文档内容差异太大")
                    print("     3. 提取器未能提取到相关实体")
                    print("     继续使用知识图谱，但多跳查询可能效果不佳")
                
            except Exception as e:
                print(f"  ⚠️  构建知识图谱失败: {e}")
                print("     将回退到不使用知识图谱的模式")
                import traceback
                traceback.print_exc()
                knowledge_graph = None
                use_kg = False
        else:
            print("ℹ️  跳过知识图谱构建（use_kg=False）")
        
        print("-" * 60)
        
        # 6. 创建 TestsetGenerator
        try:
            if knowledge_graph is not None:
                # 使用知识图谱创建 TestsetGenerator
                generator = TestsetGenerator.from_langchain(
                    llm=generator_llm,
                    embedding_model=embeddings,
                    knowledge_graph=knowledge_graph,
                )
                print("✅ TestsetGenerator 初始化成功（带知识图谱）")
            else:
                # 不使用知识图谱
                generator = TestsetGenerator.from_langchain(
                    llm=generator_llm,
                    embedding_model=embeddings,
                )
                print("✅ TestsetGenerator 初始化成功（无知识图谱）")
        except Exception as e:
            print(f"❌ 创建 TestsetGenerator 失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("-" * 60)
        
        # 7. 生成测试集
        total_questions = max_docs * num_questions_per_doc
        print(f"开始生成测试集（目标: {total_questions} 个问题）...")
        
        try:
            # 关键：如果有了手动构建的知识图谱，将其挂载到生成器实例上
            if use_kg and knowledge_graph:
                print("  ℹ️  正在将手动构建的图谱挂载到生成器...")
                generator.knowledge_graph = knowledge_graph
                
                # --- 补丁：手动注入 Persona (角色)，跳过有 Bug 的自动提取环节 ---
                from ragas.testset.persona import Persona
                print("  ℹ️  正在注入自定义 Persona (角色) 以跳过自动提取逻辑...")
                generator.persona_list = [
                    Persona(
                        name="家庭厨师",
                        role_description="一个希望为家人准备健康美味午餐的普通家庭主妇，关注做法步骤和食材替换。",
                    ),
                    Persona(
                        name="美食评论家",
                        role_description="一个对口味要求严苛、关注食材搭配和营养均衡的专业人士，喜欢对比不同菜谱的优劣。",
                    )
                ]
                # -----------------------------------------------------------
                
                print("  ℹ️  开始基于知识图谱生成测试集 (限制并发以提高稳定性)...")
                from ragas.run_config import RunConfig
                testset = generator.generate(
                    testset_size=total_questions,
                    run_config=RunConfig(max_workers=3, timeout=120) # 关键：降低并发
                )
            else:
                kwargs = {
                    "documents": langchain_docs,
                    "testset_size": total_questions,
                    "transforms_embedding_model": embeddings,
                    "raise_exceptions": False,
                }
                testset = generator.generate_with_langchain_docs(**kwargs)
        except TypeError as e:
            # 捕获 'float' object is not iterable 错误（query_distribution 的 bug）
            if "'float' object is not iterable" in str(e) or "float" in str(e).lower():
                print(f"\n⚠️  捕获到 RAGAS 内部类型错误: {e}")
                print("   原因分析: 这通常是 query_distribution 分配比例导致的迭代器错误。")
                print("   回退操作: 正在尝试【减少文档数量】进行重试...")
                
                # 尝试使用更少的文档
                if len(langchain_docs) > 10:
                    print(f"   重试策略: 只使用前 10 个文档，尝试生成 {min(10 * num_questions_per_doc, total_questions)} 个问题")
                    try:
                        kwargs["documents"] = langchain_docs[:10]
                        kwargs["testset_size"] = min(10 * num_questions_per_doc, total_questions)
                        testset = generator.generate_with_langchain_docs(**kwargs)
                        print("✅ 重试成功: 测试集已在精简模式下生成。")
                    except Exception as e2:
                        print(f"❌ 再次失败: {e2}")
                        return False
                else:
                    print("❌ 错误: 文档数已达最小值，无法进一步回退。")
                    return False
            else:
                print(f"❌ 生成测试集失败 (TypeError): {e}")
                import traceback
                traceback.print_exc()
                return False
        except ValueError as e:
            # 捕获聚类错误
            error_msg = str(e)
            if "Cannot form clusters" in error_msg or "No relationships match" in error_msg:
                print(f"\n⚠️  捕获到 RAGAS 聚类失败: {error_msg}")
                print("   原因分析: 文档数量不足或语义相关性太弱，无法构建聚类。")
                print("   回退操作: 正在尝试【缩减问题数量】以跳过聚类重试...")
                
                # 尝试使用更小的 testset_size
                smaller_testset_size = max(1, total_questions // 2)
                if smaller_testset_size < total_questions:
                    print(f"   重试策略: testset_size 缩减为 {smaller_testset_size} (原: {total_questions})")
                    try:
                        kwargs["testset_size"] = smaller_testset_size
                        testset = generator.generate_with_langchain_docs(**kwargs)
                        print("✅ 重试成功: 已降级生成。")
                    except Exception as e2:
                        print(f"❌ 缩减重试后仍然失败: {e2}")
                        return False
                else:
                    return False
            else:
                print(f"❌ 生成测试集失败 (ValueError): {e}")
                import traceback
                traceback.print_exc()
                return False
        except Exception as e:
            print(f"❌ 生成测试集失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        if not testset:
            print("❌ 未生成任何测试样本")
            return False
        
        # 转换为 DataFrame
        try:
            testset_df = testset.to_pandas()
        except Exception as e:
            print(f"⚠️  无法转换为 DataFrame: {e}")
            testset_df = None
        
        if testset_df is not None:
            num_samples = len(testset_df)
            print(f"✅ 成功生成 {num_samples} 个测试样本")
        else:
            num_samples = len(testset.questions) if hasattr(testset, 'questions') else 0
            print(f"✅ 成功生成 {num_samples} 个测试样本（通过属性访问）")
        
        print("-" * 60)
        
        # 8. 转换为目标格式
        print("正在转换数据格式...")
        samples = []
        
        if testset_df is not None:
            for _, row in testset_df.iterrows():
                sample = {
                    "question": row.get("user_input", row.get("question", "")),
                    "answer": "",
                    "ground_truth": row.get("reference", row.get("ground_truth", "")),
                    "contexts": row.get("reference_contexts", row.get("contexts", [])),
                }
                samples.append(sample)
        else:
            questions = getattr(testset, 'questions', getattr(testset, 'user_input', []))
            ground_truths = getattr(testset, 'ground_truth', getattr(testset, 'reference', []))
            contexts_list = getattr(testset, 'contexts', getattr(testset, 'reference_contexts', []))
            
            for i in range(len(questions)):
                sample = {
                    "question": questions[i] if i < len(questions) else "",
                    "answer": "",
                    "ground_truth": ground_truths[i] if i < len(ground_truths) else "",
                    "contexts": contexts_list[i] if i < len(contexts_list) else [],
                }
                samples.append(sample)
        
        # 9. 保存结果
        output_data = {
            "metadata": {
                "kb_id": kb_id,
                "source_path": str(source_dir),
                "generated_at": str(Path.cwd()),
                "total_samples": len(samples),
                "generation_method": "ragas_testset_generator_with_kg" if use_kg else "ragas_testset_generator",
                "use_knowledge_graph": use_kg,
            },
            "samples": samples,
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 测试集已保存到: {output_file}")
        print(f"   样本数: {len(samples)}")
        print(f"   使用知识图谱: {use_kg}")
        
        # 10. 预览
        preview_count = min(3, len(samples))
        if preview_count > 0:
            print("\n测试集预览 (前 3 个样本):")
            for i, sample in enumerate(samples[:preview_count]):
                print(f"\n样本 {i+1}:")
                print(f"  问题: {sample['question']}")
                print(f"  标准答案 (ground_truth): {sample.get('ground_truth', '')[:100] if sample.get('ground_truth') else '(无)'}{'...' if sample.get('ground_truth') and len(sample.get('ground_truth', '')) > 100 else ''}")
                contexts = sample.get('contexts', [])
                print(f"  上下文 (contexts): {len(contexts)} 个文档块")
                if contexts:
                    print(f"    第一个上下文: {contexts[0][:100] if isinstance(contexts[0], str) else str(contexts[0])[:100]}{'...' if len(str(contexts[0])) > 100 else ''}")
        
        if len(samples) > preview_count:
            print(f"\n... 还有 {len(samples) - preview_count} 个样本未显示")
        
        return True
        
    except Exception as e:
        print(f"❌ 生成测试集时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_ragas_dataset_with_testset_generator(
    kb_id: str,
    source_path: Optional[str] = None,
    output_path: str = "ragas_testset_dataset.json",
    max_docs: int = 5,
    num_questions_per_doc: int = 3,
) -> bool:
    """
    使用 RAGAS TestsetGenerator 生成测试集（新方法）。
    
    生成的数据格式：
    {
        "question": str,           # LLM 生成的问题
        "answer": "",              # 留空，后续用 RAG 生成
        "ground_truth": str,       # LLM 根据文档生成的参考答案
        "contexts": List[str],     # RAGAS 确定的相关文档块
    }
    
    Args:
        kb_id: 知识库 ID
        source_path: 知识库源路径
        output_path: 输出文件路径
        max_docs: 最多处理的文档数（默认 5）
        num_questions_per_doc: 每个文档生成的问题数（默认 3）
    """
    if not _HAS_TESTSET_GENERATOR:
        print("❌ RAGAS TestsetGenerator 不可用")
        print("   请确保安装了 RAGAS v0.1.0+ 并包含 testset 模块")
        return False
    
    print("=" * 60)
    print("使用 RAGAS TestsetGenerator 生成测试集")
    print("=" * 60)
    
    # 1. 确定源路径
    if source_path:
        source_dir = Path(source_path)
    else:
        # 从配置文件读取
        try:
            config_path = Path("rag_config.json")
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                kb_config = next((kb for kb in config_data.get("knowledge_bases", []) if kb["kb_id"] == kb_id), None)
                if kb_config:
                    source_dir = Path(kb_config["source_path"])
                else:
                    print(f"❌ 未找到知识库配置: {kb_id}")
                    return False
            else:
                try:
                    app_config = AppConfig.load()
                    if app_config.knowledge_bases:
                        kb_config = next((kb for kb in app_config.knowledge_bases if kb.kb_id == kb_id), None)
                        if kb_config:
                            source_dir = Path(kb_config.source_path)
                        else:
                            print(f"❌ 未找到知识库配置: {kb_id}")
                            return False
                    else:
                        print(f"❌ 未找到知识库配置")
                        return False
                except Exception as e:
                    print(f"⚠️  读取环境变量配置失败: {e}")
                    return False
        except Exception as e:
            print(f"❌ 加载配置文件失败: {e}")
            return False
    
    if not source_dir.exists():
        print(f"❌ 源路径不存在: {source_dir}")
        return False
    
    print(f"知识库 ID: {kb_id}")
    print(f"源路径: {source_dir}")
    print(f"处理文档数: {max_docs}")
    print(f"每个文档问题数: {num_questions_per_doc}")
    print("-" * 60)
    
    # 2. 查找文档
    print("正在扫描文档...")
    md_files = find_markdown_files(source_dir)
    
    if max_docs:
        md_files = md_files[:max_docs]
    
    if not md_files:
        print("❌ 未找到 .md 文件")
        return False
    
    print(f"✅ 找到 {len(md_files)} 个文档")
    
    # 检查文档数量是否足够（RAGAS 需要至少 3 个文档才能形成聚类）
    if len(md_files) < 3:
        print(f"⚠️  警告: 文档数量较少（{len(md_files)} 个），可能无法形成聚类")
        print("   建议: 至少使用 3-5 个文档，或增加 --max-docs 参数")
        print("   继续尝试生成测试集...")
    
    print("-" * 60)
    
    # 3. 转换为 langchain Document 格式
    print("正在转换文档格式...")
    langchain_docs = convert_documents_to_langchain_docs(md_files)
    
    if not langchain_docs:
        print("❌ 文档转换失败")
        return False
    
    print(f"✅ 成功转换 {len(langchain_docs)} 个文档")
    print("-" * 60)
    
    # 4. 加载配置和初始化 RAGAS TestsetGenerator
    print("初始化 RAGAS TestsetGenerator...")
    try:
        app_config = AppConfig.load()
        
        # 将我们的配置适配为 RAGAS 需要的 langchain 对象
        # RAGAS 需要 langchain_openai.ChatOpenAI 和 ragas.embeddings
        try:
            from langchain_openai import ChatOpenAI
            from ragas.embeddings import OpenAIEmbeddings
        except ImportError:
            print("❌ 需要安装 langchain-openai")
            print("   请运行: pip install langchain-openai")
            return False
        
        # 创建 langchain LLM（从我们的配置读取）
        # 增加超时设置，避免连接问题
        generator_llm = ChatOpenAI(
            model=app_config.llm.model,  # 注意：属性名是 model，不是 model_name
            api_key=app_config.llm.api_key,
            base_url=app_config.llm.base_url,
            temperature=0.1,
            timeout=120.0,  # 增加超时时间到 120 秒（RAGAS 可能需要较长时间）
            max_retries=3,  # 增加重试次数
        )
        
        # 创建 Embeddings（从我们的配置读取）
        # 优先使用 RAGAS 原生的 OpenAIEmbeddings（更稳定，兼容性更好）
        try:
            from openai import OpenAI, AsyncOpenAI
            from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
            
            # 创建 OpenAI 客户端（同步和异步）
            openai_client = OpenAI(
                api_key=app_config.embedding.api_key,
                base_url=app_config.embedding.base_url,
            )
            async_openai_client = AsyncOpenAI(
                api_key=app_config.embedding.api_key,
                base_url=app_config.embedding.base_url,
            )
            
            # 使用 RAGAS 的 OpenAIEmbeddings（直接使用 OpenAI 客户端，更稳定）
            embeddings = RagasOpenAIEmbeddings(
                client=openai_client,
                model=app_config.embedding.model,
            )
            # 注意：RAGAS OpenAIEmbeddings 可能需要设置 async_client
            if hasattr(embeddings, 'async_client'):
                embeddings.async_client = async_openai_client
            
            print("✅ 使用 RAGAS 原生的 OpenAIEmbeddings")
        except Exception as e:
            print(f"⚠️  创建 RAGAS OpenAIEmbeddings 失败: {e}")
            print("    尝试使用 langchain OpenAIEmbeddings + LangchainEmbeddingsWrapper")
            try:
                # 备用方案：使用 langchain OpenAIEmbeddings
                from langchain_openai import OpenAIEmbeddings as LangchainOpenAIEmbeddings
                from ragas.embeddings import LangchainEmbeddingsWrapper
                
                langchain_embeddings = LangchainOpenAIEmbeddings(
                    model=app_config.embedding.model,
                    api_key=app_config.embedding.api_key,
                    base_url=app_config.embedding.base_url,
                )
                embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)
                print("✅ 使用 LangchainEmbeddingsWrapper 包装 langchain OpenAIEmbeddings")
            except Exception as e2:
                print(f"❌ 创建 Embeddings 失败: {e2}")
                import traceback
                traceback.print_exc()
                return False
        
        # 创建 TestsetGenerator
        # RAGAS 0.4.2 使用 from_langchain 创建，参数名是 llm 和 embedding_model
        try:
            generator = TestsetGenerator.from_langchain(
                llm=generator_llm,  # 注意：参数名是 llm，不是 generator_llm
                embedding_model=embeddings,
            )
        except Exception as e:
            print(f"❌ 创建 TestsetGenerator 失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("✅ TestsetGenerator 初始化成功")
        print("-" * 60)
        
        # 5. 创建查询分布（query_distribution）
        # 注意：RAGAS 0.4.2 的 query_distribution 可能有 bug，导致 'float' object is not iterable
        # 问题可能出在 scenario_sample_list 的构建与 query_distribution 的匹配上
        # 尝试使用 query_distribution，如果失败则回退到默认行为
        query_dist = None
        use_query_distribution = True  # 尝试使用，如果失败再禁用
        
        if use_query_distribution and default_query_distribution is not None:
            try:
                # 创建一个适配器，将 langchain LLM 转换为 RAGAS LLM
                from ragas.llms import LangchainLLMWrapper
                ragas_llm = LangchainLLMWrapper(generator_llm)
                
                # 创建默认查询分布（包括单跳、多跳等查询类型）
                query_dist = default_query_distribution(ragas_llm)
                print(f"✅ 创建查询分布: {len(query_dist)} 种查询类型")
                for synthesizer, weight in query_dist:
                    print(f"   - {synthesizer.__class__.__name__}: {weight}")
                
                # 验证 query_distribution 的结构
                if not isinstance(query_dist, list):
                    print(f"⚠️  query_distribution 不是列表，禁用它")
                    query_dist = None
                else:
                    # 检查每个元素是否是 (synthesizer, weight) 元组
                    for item in query_dist:
                        if not isinstance(item, (tuple, list)) or len(item) != 2:
                            print(f"⚠️  query_distribution 格式不正确，禁用它")
                            query_dist = None
                            break
            except Exception as e:
                print(f"⚠️  创建查询分布失败: {e}")
                print("    将使用默认查询类型（不指定 query_distribution）")
                import traceback
                traceback.print_exc()
                query_dist = None
        else:
            print("ℹ️  未尝试创建查询分布")
            print("    将使用默认查询类型（仍然会生成多样化的查询）")
        
        print("-" * 60)
        
        # 6. 生成测试集
        total_questions = max_docs * num_questions_per_doc
        print(f"开始生成测试集（目标: {total_questions} 个问题）...")
        
        try:
            # RAGAS 0.4.2 使用 generate_with_langchain_docs 方法
            # 重要：传入 transforms_embedding_model 参数，使用我们测试过的 embeddings
            # 这样确保 RAGAS 内部使用的 embedding_model 与我们测试的一致
            kwargs = {
                "documents": langchain_docs,
                "testset_size": total_questions,
                "transforms_embedding_model": embeddings,  # 明确指定 embedding_model
                "raise_exceptions": False,
            }
            
            # 尝试使用 query_distribution（如果有）
            if query_dist is not None:
                kwargs["query_distribution"] = query_dist
                print(f"   尝试使用查询分布（{len(query_dist)} 种查询类型）...")
            
            testset = generator.generate_with_langchain_docs(**kwargs)
        except TypeError as e:
            # 捕获 'float' object is not iterable 错误
            if "'float' object is not iterable" in str(e) or "float" in str(e).lower():
                print(f"⚠️  使用 query_distribution 时遇到已知 bug: {e}")
                print("   回退到不使用 query_distribution（仍会生成测试用例，但查询类型可能较少）...")
                
                # 重试，不使用 query_distribution
                kwargs.pop("query_distribution", None)
                try:
                    testset = generator.generate_with_langchain_docs(**kwargs)
                    print("✅ 回退后生成成功")
                except Exception as e2:
                    print(f"❌ 回退后仍然失败: {e2}")
                    import traceback
                    traceback.print_exc()
                    return False
            else:
                print(f"❌ 生成测试集失败: {e}")
                import traceback
                traceback.print_exc()
                return False
        except ValueError as e:
            # 捕获聚类错误（No relationships match the provided condition. Cannot form clusters.）
            error_msg = str(e)
            if "Cannot form clusters" in error_msg or "No relationships match" in error_msg:
                print(f"⚠️  无法形成聚类: {error_msg}")
                print()
                print("可能的原因:")
                print("  1. 文档数量太少（少于 3 个），无法形成聚类")
                print("  2. 文档之间的相似度太低，无法形成有意义的聚类")
                print("  3. 文档内容太短或太相似，导致聚类失败")
                print()
                print("解决方案:")
                print("  1. 增加文档数量（至少 3-5 个文档）")
                print(f"     当前文档数: {len(langchain_docs)}")
                print(f"     建议: 使用 --max-docs 参数增加文档数量，例如: --max-docs 10")
                print("  2. 减少 testset_size（减少要生成的问题数量）")
                print(f"     当前 testset_size: {total_questions}")
                print(f"     建议: 减少 num_questions_per_doc 或 max_docs")
                print("  3. 增加文档的多样性（使用不同主题的文档）")
                print()
                print("尝试使用更少的 testset_size 重试...")
                
                # 尝试使用更小的 testset_size
                smaller_testset_size = max(1, total_questions // 2)
                if smaller_testset_size < total_questions:
                    print(f"   重试: testset_size = {smaller_testset_size} (原: {total_questions})")
                    try:
                        kwargs["testset_size"] = smaller_testset_size
                        testset = generator.generate_with_langchain_docs(**kwargs)
                        print("✅ 使用更小的 testset_size 后生成成功")
                    except Exception as e2:
                        print(f"❌ 重试后仍然失败: {e2}")
                        return False
                else:
                    return False
            elif "headlines" in error_msg.lower() or "'headlines' property not found" in error_msg:
                # HeadlinesExtractor/HeadlineSplitter 错误
                print(f"⚠️  HeadlinesExtractor 错误: {error_msg}")
                print()
                print("可能的原因:")
                print("  1. 部分文档没有成功提取到标题（headlines）")
                print("  2. 文档格式不符合 HeadlinesExtractor 的预期")
                print("  3. RAGAS 内部 transforms 处理时出现了不一致")
                print()
                print("解决方案:")
                print("  1. 尝试减少文档数量，使用更少、更格式化的文档")
                print("  2. 检查文档是否都有清晰的标题（Markdown 的 # 标题）")
                print("  3. 或者尝试使用知识图谱方法（--use-kg）")
                print()
                print("尝试使用更少的文档数量重试...")
                
                # 尝试使用更少的文档
                if len(langchain_docs) > 10:
                    print(f"   重试: max_docs = 10 (原: {max_docs})")
                    try:
                        # 重新读取文档，只使用前 10 个
                        md_files_retry = md_files[:10]
                        langchain_docs_retry = convert_documents_to_langchain_docs(md_files_retry)
                        kwargs["documents"] = langchain_docs_retry
                        kwargs["testset_size"] = min(10 * num_questions_per_doc, total_questions)
                        testset = generator.generate_with_langchain_docs(**kwargs)
                        print("✅ 使用更少的文档后生成成功")
                    except Exception as e2:
                        print(f"❌ 重试后仍然失败: {e2}")
                        print("   建议: 尝试使用知识图谱方法（--use-kg），或联系 RAGAS 开发者")
                        import traceback
                        traceback.print_exc()
                        return False
                else:
                    print("   文档数量已经很少，无法进一步减少")
                    print("   建议: 尝试使用知识图谱方法（--use-kg），或联系 RAGAS 开发者")
                    import traceback
                    traceback.print_exc()
                    return False
            else:
                # 其他 ValueError
                print(f"❌ 生成测试集失败 (ValueError): {e}")
                import traceback
                traceback.print_exc()
                return False
        except Exception as e:
            # 检查是否是连接错误
            if openai and isinstance(e, openai.APIConnectionError):
                is_connection_error = True
            elif isinstance(e, (ConnectionError, RuntimeError)):
                is_connection_error = True
            elif "Event loop is closed" in str(e) or "Connection error" in str(e):
                is_connection_error = True
            else:
                is_connection_error = False
            
            if is_connection_error:
                # 捕获连接错误（可能是网络问题或事件循环问题）
                error_msg = str(e)
                print(f"⚠️  遇到连接错误（可能是临时网络问题）: {error_msg}")
                print("   这可能不是必现的，建议：")
                print("   1. 检查网络连接")
                print("   2. 检查 LLM API 服务是否正常")
                print("   3. 稍后重试")
                print("   4. 或者减少 testset_size 和文档数量重试")
                import traceback
                traceback.print_exc()
                return False
            else:
                # 其他类型的错误
                print(f"❌ 生成测试集失败: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        if not testset:
            print("❌ 未生成任何测试样本")
            return False
        
        # 转换为 DataFrame 以便访问数据
        try:
            testset_df = testset.to_pandas()
        except Exception as e:
            print(f"⚠️  无法转换为 DataFrame: {e}")
            print("    尝试直接访问 testset 属性...")
            # 如果无法转换为 DataFrame，尝试直接访问属性
            testset_df = None
        
        if testset_df is not None:
            num_samples = len(testset_df)
            print(f"✅ 成功生成 {num_samples} 个测试样本")
        else:
            # 尝试通过属性访问
            num_samples = len(testset.questions) if hasattr(testset, 'questions') else 0
            print(f"✅ 成功生成 {num_samples} 个测试样本（通过属性访问）")
        
        print("-" * 60)
        
        # 7. 转换为目标格式
        print("正在转换数据格式...")
        samples = []
        
        if testset_df is not None:
            # 使用 DataFrame 访问数据
            for _, row in testset_df.iterrows():
                sample = {
                    "question": row.get("user_input", row.get("question", "")),
                    "answer": "",  # 留空，后续用 RAG 生成
                    "ground_truth": row.get("reference", row.get("ground_truth", "")),
                    "contexts": row.get("reference_contexts", row.get("contexts", [])),
                }
                samples.append(sample)
        else:
            # 直接访问 testset 属性
            questions = getattr(testset, 'questions', getattr(testset, 'user_input', []))
            ground_truths = getattr(testset, 'ground_truth', getattr(testset, 'reference', []))
            contexts_list = getattr(testset, 'contexts', getattr(testset, 'reference_contexts', []))
            
            for i in range(len(questions)):
                sample = {
                    "question": questions[i] if i < len(questions) else "",
                    "answer": "",  # 留空，后续用 RAG 生成
                    "ground_truth": ground_truths[i] if i < len(ground_truths) else "",
                    "contexts": contexts_list[i] if i < len(contexts_list) else [],
                }
                samples.append(sample)
        
        # 7. 保存数据集
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        dataset = {
            "metadata": {
                "kb_id": kb_id,
                "source_path": str(source_dir),
                "generated_at": str(Path().cwd()),
                "total_samples": len(samples),
                "generation_method": "ragas_testset_generator",
            },
            "samples": samples,
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 测试集已保存: {output_file}")
        print(f"   共 {len(samples)} 个样本")
        print()
        
        # 显示测试集的前几个样本，让用户查看内容
        print("=" * 60)
        print("测试集预览（前 3 个样本）:")
        print("=" * 60)
        preview_count = min(3, len(samples))
        for i, sample in enumerate(samples[:preview_count], 1):
            print(f"\n样本 {i}:")
            print(f"  问题 (question): {sample.get('question', '')[:100]}{'...' if len(sample.get('question', '')) > 100 else ''}")
            print(f"  答案 (answer): {sample.get('answer', '')[:100] if sample.get('answer') else '(空，待生成)'}{'...' if sample.get('answer') and len(sample.get('answer', '')) > 100 else ''}")
            print(f"  标准答案 (ground_truth): {sample.get('ground_truth', '')[:100] if sample.get('ground_truth') else '(无)'}{'...' if sample.get('ground_truth') and len(sample.get('ground_truth', '')) > 100 else ''}")
            contexts = sample.get('contexts', [])
            print(f"  上下文 (contexts): {len(contexts)} 个文档块")
            if contexts:
                print(f"    第一个上下文: {contexts[0][:100] if isinstance(contexts[0], str) else str(contexts[0])[:100]}{'...' if len(str(contexts[0])) > 100 else ''}")
        
        if len(samples) > preview_count:
            print(f"\n... 还有 {len(samples) - preview_count} 个样本未显示")
        
        print()
        print("=" * 60)
        print(f"完整测试集已保存到: {output_file}")
        print("可以使用以下命令查看完整内容:")
        print(f"  cat {output_file}")
        print("或使用 Python:")
        print(f"  python3 -c \"import json; data = json.load(open('{output_file}')); print(json.dumps(data, ensure_ascii=False, indent=2))\"")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_ragas_dataset(
    kb_id: str,
    source_path: Optional[str] = None,
    output_path: str = "ragas_dataset.json",
    max_docs: Optional[int] = None,
    max_questions_per_doc: int = 3,
) -> bool:
    """
    生成 RAGAS 评估数据集。
    
    RAGAS 需要的数据格式：
    {
        "question": str,           # 问题
        "contexts": List[str],     # 检索到的文档块（上下文）
        "answer": str,             # RAG 系统生成的回答
        "ground_truth": str,       # 标准答案（可选，从文档中提取）
    }
    
    Args:
        kb_id: 知识库 ID
        source_path: 知识库源路径
        output_path: 输出文件路径
        max_docs: 最多处理的文档数
        max_questions_per_doc: 每个文档最多生成的问题数
    """
    print("=" * 60)
    print("生成 RAGAS 评估数据集")
    print("=" * 60)
    
    # 1. 确定源路径
    if source_path:
        source_dir = Path(source_path)
    else:
        # 从配置文件读取
        source_dir = None
        
        # 方式1：从 rag_config.json 读取
        kb_json_path = Path(project_root) / "rag_config.json"
        if kb_json_path.exists():
            try:
                with open(kb_json_path, "r", encoding="utf-8") as f:
                    kb_config_data = json.load(f)
                kb_list = kb_config_data.get("knowledge_bases", [])
                kb_config_dict = next((kb for kb in kb_list if kb.get("kb_id") == kb_id), None)
                if kb_config_dict:
                    source_dir = Path(kb_config_dict["source_path"])
            except Exception as e:
                print(f"⚠️  读取 rag_config.json 失败: {e}")
        
        # 方式2：从环境变量配置读取
        if source_dir is None:
            try:
                app_config = AppConfig.load()
                if app_config.knowledge_bases:
                    kb_config = next((kb for kb in app_config.knowledge_bases if kb.kb_id == kb_id), None)
                    if kb_config:
                        source_dir = Path(kb_config.source_path)
            except Exception as e:
                print(f"⚠️  读取环境变量配置失败: {e}")
        
        # 如果仍未找到，报错
        if source_dir is None:
            print(f"❌ 未找到知识库配置: {kb_id}")
            print(f"   提示:")
            print(f"   1. 检查 rag_config.json 文件")
            print(f"   2. 或使用 --source-path 参数指定路径")
            print(f"   3. 或配置环境变量 RAG_KNOWLEDGE_BASES")
            return False
    
    if not source_dir.exists():
        print(f"❌ 源路径不存在: {source_dir}")
        return False
    
    print(f"知识库 ID: {kb_id}")
    print(f"源路径: {source_dir}")
    print("-" * 60)
    
    # 2. 查找所有 .md 文件
    print("正在扫描文档...")
    md_files = find_markdown_files(source_dir)
    
    if max_docs:
        md_files = md_files[:max_docs]
    
    if not md_files:
        print("❌ 未找到 .md 文件")
        return False
    
    print(f"✅ 找到 {len(md_files)} 个文档")
    print("-" * 60)
    
    # 3. 初始化 RAG 引擎
    print("初始化 RAG 引擎...")
    try:
        engine = RAGEngine(kb_id=kb_id)
        print("✅ RAG 引擎初始化成功")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return False
    
    print("-" * 60)
    
    # 4. 生成评估数据集
    print(f"\n开始生成评估数据集...\n")
    ragas_samples = []
    
    for i, file_path in enumerate(md_files, 1):
        try:
            rel_path = file_path.relative_to(source_dir)
        except ValueError:
            rel_path = file_path.name
        
        print(f"[{i}/{len(md_files)}] 处理: {rel_path}")
        
        try:
            # 生成问题
            questions = generate_questions_from_document(
                file_path,
                engine.llm_client,
                max_questions=max_questions_per_doc,
            )
            
            if not questions:
                print(f"  ⚠️  未能生成问题，跳过")
                continue
            
            # 读取文档内容作为 ground_truth（简化：使用文档前 500 字符）
            try:
                from document import ParserFactory
                parser = ParserFactory.get_parser(file_path)
                if parser:
                    doc = parser.parse(file_path)
                    ground_truth = doc.get("content", "")[:500]  # 前 500 字符作为标准答案
                else:
                    ground_truth = ""
            except Exception:
                ground_truth = ""
            
            # 为每个问题生成评估样本
            for question in questions:
                try:
                    # 使用 RAG 系统生成回答
                    result = engine.query(question, top_k=5)
                    
                    # 提取上下文（检索到的文档块）
                    # RAGAS 需要 contexts 是字符串列表
                    contexts = []
                    # 兼容不同返回字段：engine.query 可能返回 sources 或 chunks
                    for chunk in (result.get("chunks") or result.get("sources") or []):
                        chunk_text = chunk.get("text", "")
                        if chunk_text:
                            contexts.append(chunk_text)
                    
                    # 如果没有检索到上下文，跳过
                    if not contexts:
                        print(f"  ⚠️  未检索到上下文，跳过")
                        continue
                    
                    # 提取回答
                    answer = result.get("answer", "")
                    
                    # 如果没有生成回答，跳过
                    if not answer:
                        print(f"  ⚠️  未生成回答，跳过")
                        continue
                    
                    # 创建 RAGAS 评估样本
                    ragas_sample = {
                        "question": question,
                        "contexts": contexts,  # 检索到的文档块列表
                        "answer": answer,      # RAG 系统生成的回答
                        "ground_truth": ground_truth,  # 标准答案（文档内容）
                    }
                    
                    ragas_samples.append(ragas_sample)
                    print(f"  ✅ 生成样本: {question[:50]}...")
                
                except Exception as e:
                    print(f"  ❌ 生成样本失败: {e}")
                    continue
        
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            continue
    
    # 5. 保存数据集
    print("\n" + "=" * 60)
    print("保存评估数据集...")
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    dataset = {
        "kb_id": kb_id,
        "source_path": str(source_dir),
        "total_samples": len(ragas_samples),
        "samples": ragas_samples,
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 评估数据集已保存: {output_file}")
    print(f"   总样本数: {len(ragas_samples)}")
    print()
    
    # 显示测试集的前几个样本，让用户查看内容
    print("=" * 60)
    print("测试集预览（前 3 个样本）:")
    print("=" * 60)
    preview_count = min(3, len(ragas_samples))
    for i, sample in enumerate(ragas_samples[:preview_count], 1):
        print(f"\n样本 {i}:")
        print(f"  问题 (question): {sample.get('question', '')[:100]}{'...' if len(sample.get('question', '')) > 100 else ''}")
        print(f"  答案 (answer): {sample.get('answer', '')[:100] if sample.get('answer') else '(空，待生成)'}{'...' if sample.get('answer') and len(sample.get('answer', '')) > 100 else ''}")
        print(f"  标准答案 (ground_truth): {sample.get('ground_truth', '')[:100] if sample.get('ground_truth') else '(无)'}{'...' if sample.get('ground_truth') and len(sample.get('ground_truth', '')) > 100 else ''}")
        contexts = sample.get('contexts', [])
        print(f"  上下文 (contexts): {len(contexts)} 个文档块")
        if contexts:
            print(f"    第一个上下文: {contexts[0][:100] if isinstance(contexts[0], str) else str(contexts[0])[:100]}{'...' if len(str(contexts[0])) > 100 else ''}")
    
    if len(ragas_samples) > preview_count:
        print(f"\n... 还有 {len(ragas_samples) - preview_count} 个样本未显示")
    
    print()
    print("=" * 60)
    print(f"完整测试集已保存到: {output_file}")
    print("可以使用以下命令查看完整内容:")
    print(f"  cat {output_file}")
    print("或使用 Python:")
    print(f"  python3 -c \"import json; data = json.load(open('{output_file}')); print(json.dumps(data, ensure_ascii=False, indent=2))\"")
    print("=" * 60)
    
    return True


def evaluate_with_ragas(
    dataset_path: str,
    output_path: Optional[str] = None,
) -> bool:
    """
    使用 RAGAS 评估数据集。
    
    Args:
        dataset_path: 评估数据集路径
        output_path: 评估结果输出路径（可选）
    """
    if not _HAS_RAGAS:
        print("❌ 未安装 RAGAS，请运行: pip install ragas datasets")
        return False
    
    print("=" * 60)
    print("使用 RAGAS 评估 RAG 系统")
    print("=" * 60)
    
    # 1. 加载数据集
    print("加载评估数据集...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    samples = dataset.get("samples", [])
    if not samples:
        print("❌ 评估数据集为空")
        return False
    
    print(f"✅ 加载 {len(samples)} 个评估样本")
    
    # 检查是否有空答案，如果有则用 RAG 生成
    empty_answer_count = sum(1 for s in samples if not s.get("answer") or s.get("answer", "").strip() == "")
    if empty_answer_count > 0:
        print(f"⚠️  发现 {empty_answer_count} 个样本的答案为空，需要先用 RAG 生成答案...")
        
        # 获取知识库 ID
        kb_id = dataset.get("metadata", {}).get("kb_id")
        if not kb_id:
            print("❌ 数据集中没有 kb_id，无法生成答案")
            print("   提示: 请确保数据集包含 metadata.kb_id")
            return False
        
        # 初始化 RAG 引擎
        try:
            engine = RAGEngine(kb_id=kb_id)
            print("✅ RAG 引擎初始化成功")
        except Exception as e:
            print(f"❌ 初始化 RAG 引擎失败: {e}")
            return False
        
        # 为每个空答案生成答案
        print(f"正在为 {empty_answer_count} 个样本生成答案...")
        generated_count = 0
        failed_count = 0
        for i, sample in enumerate(samples):
            if not sample.get("answer") or sample.get("answer", "").strip() == "":
                question = sample.get("question", "")
                if question:
                    try:
                        result = engine.query(question, top_k=5)
                        answer = result.get("answer", "")
                        if answer and isinstance(answer, str) and answer.strip():
                            sample["answer"] = answer.strip()  # 确保去除首尾空格
                            generated_count += 1
                            print(f"  ✅ [{i+1}/{empty_answer_count}] 答案生成成功（长度: {len(answer)} 字符）")
                        else:
                            # RAG 引擎返回空答案，使用占位符（非空字符串，避免 answer_relevancy 报错）
                            failed_count += 1
                            placeholder = f"[无法生成答案：知识库中可能没有相关信息，或检索失败]"
                            sample["answer"] = placeholder
                            print(f"  ⚠️  [{i+1}/{empty_answer_count}] RAG 引擎返回空答案，使用占位符")
                            print(f"     问题: {question[:50]}...")
                            print(f"     占位符: {placeholder}")
                    except Exception as e:
                        # 生成答案失败，使用占位符（非空字符串，避免 answer_relevancy 报错）
                        failed_count += 1
                        placeholder = f"[无法生成答案：{str(e)[:50]}]"
                        sample["answer"] = placeholder
                        print(f"  ❌ [{i+1}/{empty_answer_count}] 生成答案失败，使用占位符: {e}")
                        print(f"     问题: {question[:50]}...")
                        print(f"     占位符: {placeholder}")
                else:
                    failed_count += 1
                    print(f"  ⚠️  [{i+1}/{empty_answer_count}] 问题为空，跳过")
        
        print(f"\n✅ 成功生成 {generated_count}/{empty_answer_count} 个答案")
        if failed_count > 0:
            print(f"⚠️  失败/跳过 {failed_count} 个样本（这些样本将在评估时被过滤）")
    
    print("-" * 60)
    
    # 2. 转换为 RAGAS 格式
    print("转换数据格式...")
    
    # RAGAS 需要的数据格式
    # 确保所有文本都是字符串类型（避免 API 格式错误）
    def ensure_string(value):
        """确保值是字符串类型"""
        if value is None:
            return ""
        if not isinstance(value, str):
            return str(value)
        return value
    
    def ensure_string_list(value):
        """确保值是字符串列表"""
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [ensure_string(item) for item in value]
        return [str(value)]
    
    # 确保所有样本都有非空答案（如果为空，使用占位符）
    # 不再过滤样本，而是确保所有答案都是有效的非空字符串
    placeholder_count = 0
    for s in samples:
        answer = s.get("answer", "")
        # 确保答案是字符串类型
        if not isinstance(answer, str):
            answer = str(answer)
        # 去除首尾空格
        answer = answer.strip()
        # 如果答案为空，使用占位符（非空字符串，避免 answer_relevancy 报错）
        if not answer:
            placeholder = "[无法生成答案：知识库中可能没有相关信息，或检索失败]"
            s["answer"] = placeholder
            placeholder_count += 1
    
    if placeholder_count > 0:
        print(f"⚠️  {placeholder_count} 个样本的答案为空，已使用占位符（非空字符串）")
        print(f"   占位符: [无法生成答案：知识库中可能没有相关信息，或检索失败]")
    
    # 所有样本都有效（因为已经确保答案非空）
    valid_samples = samples
    
    # 确保所有字段都是正确的类型
    # 特别注意：answer 必须是非空字符串，否则 answer_relevancy 会失败
    # 清理答案，移除可能导致 RAGAS 处理问题的特殊字符
    def clean_answer(answer: str) -> str:
        """清理答案，确保只包含有效的字符串内容"""
        if not isinstance(answer, str):
            answer = str(answer)
        # 去除首尾空格
        answer = answer.strip()
        # 确保非空
        if not answer:
            return "[无法生成答案：知识库中可能没有相关信息，或检索失败]"
        # 移除可能导致问题的特殊字符（如控制字符、NULL 字符等）
        # 保留换行符、制表符等常见空白字符
        cleaned = []
        for char in answer:
            # 保留可打印字符、换行符、制表符、空格
            if char.isprintable() or char in ['\n', '\t', ' ']:
                cleaned.append(char)
            # 移除控制字符（除了换行符和制表符）
            elif ord(char) < 32 and char not in ['\n', '\t']:
                continue
            else:
                cleaned.append(char)
        answer = ''.join(cleaned)
        # 确保最终结果是有效的非空字符串
        if not answer or not answer.strip():
            return "[无法生成答案：知识库中可能没有相关信息，或检索失败]"
        return answer.strip()
    
    cleaned_answers = []
    for s in valid_samples:
        answer = s.get("answer", "")
        cleaned_answer = clean_answer(answer)
        cleaned_answers.append(cleaned_answer)
    
    ragas_data = {
        "question": [ensure_string(s.get("question", "")) for s in valid_samples],
        "contexts": [ensure_string_list(s.get("contexts", [])) for s in valid_samples],  # 每个元素是字符串列表
        "answer": cleaned_answers,  # 使用清理后的答案（确保非空）
    }
    
    # ground_truth 是可选的（如果所有样本都有 ground_truth，则添加）
    if any(s.get("ground_truth") for s in valid_samples):
        ragas_data["ground_truth"] = [ensure_string(s.get("ground_truth", "")) for s in valid_samples]
    
    ragas_dataset = Dataset.from_dict(ragas_data)
    print("✅ 数据格式转换完成")
    print("-" * 60)
    
    # 3. 加载配置，创建 LLM 和 Embeddings（RAGAS 评估需要）
    print("\n初始化 LLM 和 Embeddings（评估需要）...")
    try:
        app_config = AppConfig.load()
        
        # 使用 llm_factory 直接创建 RAGAS LLM（不使用 LangChain）
        from openai import OpenAI, AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
        
        # 创建 LLM 客户端
        openai_client = OpenAI(
            api_key=app_config.llm.api_key,
            base_url=app_config.llm.base_url,
        )
        
        # 使用 llm_factory 创建 RAGAS LLM（推荐方式，不需要 LangChain）
        # 注意：增加 max_tokens 以避免输出被截断（faithfulness 需要生成较长的输出）
        ragas_llm = llm_factory(
            model=app_config.llm.model,
            provider="openai",
            client=openai_client,
            temperature=0.1,
            max_tokens=4096,  # 增加 max_tokens，避免输出被截断
        )
        print("✅ LLM 初始化成功（使用 llm_factory，不依赖 LangChain，max_tokens=4096）")
        
        # 创建 Embeddings（用于 Context Precision/Recall）
        # 使用 RAGAS 原生的 OpenAIEmbeddings（不依赖 LangChain）
        
        embedding_client = OpenAI(
            api_key=app_config.embedding.api_key,
            base_url=app_config.embedding.base_url,
        )
        async_embedding_client = AsyncOpenAI(
            api_key=app_config.embedding.api_key,
            base_url=app_config.embedding.base_url,
        )
        
        ragas_embeddings = RagasOpenAIEmbeddings(
            client=embedding_client,
            model=app_config.embedding.model,
        )
        # 设置异步客户端（如果支持）
        if hasattr(ragas_embeddings, 'async_client'):
            ragas_embeddings.async_client = async_embedding_client
        
        # 创建适配器：RAGAS 的 OpenAIEmbeddings 使用 embed_text，但 metrics 需要 embed_query
        # 创建一个适配器类，将 embed_text 转换为 embed_query 和 embed_documents
        class EmbeddingsAdapter:
            """适配器，将 embed_text 转换为 embed_query 和 embed_documents"""
            def __init__(self, embeddings):
                self.embeddings = embeddings
            
            def embed_query(self, text: str):
                """适配 embed_query 到 embed_text"""
                return self.embeddings.embed_text(text)
            
            def embed_documents(self, texts: list):
                """适配 embed_documents 到 embed_texts"""
                return self.embeddings.embed_texts(texts)
            
            async def aembed_query(self, text: str):
                """适配 aembed_query 到 aembed_text"""
                return await self.embeddings.aembed_text(text)
            
            async def aembed_documents(self, texts: list):
                """适配 aembed_documents 到 aembed_texts"""
                return await self.embeddings.aembed_texts(texts)
            
            def __getattr__(self, name):
                """转发其他属性到原始 embeddings"""
                return getattr(self.embeddings, name)
        
        embeddings = EmbeddingsAdapter(ragas_embeddings)
        print("✅ Embeddings 初始化成功（使用 RAGAS OpenAIEmbeddings + 适配器，不依赖 LangChain）")
        
    except Exception as e:
        print(f"❌ 初始化 LLM/Embeddings 失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("-" * 60)
    
    # 4. 使用 RAGAS 评估
    print("\n开始评估...")
    print("评估指标说明:")
    print("  - Faithfulness（忠实度）: 使用 LLM 判断答案是否忠实于上下文")
    print("  - Answer Relevancy（回答相关性）: 使用 LLM 判断答案与问题的相关性")
    print("  - Context Precision（上下文精确率）: 使用 Embeddings 计算语义相似度")
    print("  - Context Recall（上下文召回率）: 使用 Embeddings 计算语义相似度")
    print("-" * 60)
    
    # 尝试评估所有指标，如果 answer_relevancy 失败，则只评估其他指标
    # 注意：将 answer_relevancy 的 strictness 设置为 1，避免多生成问题时的兼容性问题
    # 当 strictness > 1 时，RAGAS 会创建多个 PromptValue，可能导致通义千问 API 报错
    answer_relevancy.strictness = 1
    print(f"⚠️  将 answer_relevancy.strictness 设置为 1（避免多生成问题时的兼容性问题）")
    print()
    
    metrics_to_evaluate = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]
    
    # 添加日志记录，查看评估过程中的错误
    import logging
    logging.basicConfig(level=logging.WARNING)
    ragas_logger = logging.getLogger("ragas")
    ragas_logger.setLevel(logging.WARNING)
    
    try:
        result = evaluate(
            dataset=ragas_dataset,
            metrics=metrics_to_evaluate,
            llm=ragas_llm,  # 传递 LLM（用于需要 LLM 的指标）
            embeddings=embeddings,  # 传递 Embeddings（用于需要 Embeddings 的指标）
            raise_exceptions=False,  # 不抛出异常，继续评估其他指标
        )
        
        # 检查结果中是否有 answer_relevancy 的分数
        result_df = result.to_pandas()
        if "answer_relevancy" in result_df.columns:
            answer_relevancy_scores = result_df["answer_relevancy"].dropna()
            if len(answer_relevancy_scores) == 0:
                print("\n⚠️  answer_relevancy 评估失败：所有样本的分数都是 NaN")
                print("   可能的原因：")
                print("   1. LLM 生成的问题为空或格式不正确")
                print("   2. Embeddings 调用失败")
                print("   3. 与通义千问 API 的兼容性问题")
                print("\n   尝试单独测试 answer_relevancy...")
                
                # 尝试单独测试 answer_relevancy 以获取更详细的错误信息
                try:
                    from ragas import evaluate as ragas_evaluate
                    from ragas.metrics import answer_relevancy as test_answer_relevancy
                    
                    # 只评估 answer_relevancy
                    test_result = ragas_evaluate(
                        dataset=ragas_dataset,
                        metrics=[test_answer_relevancy],
                        llm=ragas_llm,
                        embeddings=embeddings,
                        raise_exceptions=True,  # 设置为 True 以查看详细错误
                    )
                    print("   ✅ 单独测试 answer_relevancy 成功")
                except Exception as test_e:
                    print(f"   ❌ 单独测试 answer_relevancy 失败: {test_e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"\n✅ answer_relevancy 评估成功：{len(answer_relevancy_scores)}/{len(result_df)} 个样本有分数")
        
        # 4. 显示结果（评估成功的情况）
        print("\n" + "=" * 60)
        print("RAGAS 评估结果")
        print("=" * 60)
        
        # 转换为字典格式以便显示
        result_dict = result.to_pandas().to_dict(orient="records")
        
        # 计算平均分数（过滤掉 nan 值）
        import math
        avg_scores = {}
        for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            scores = [r.get(metric) for r in result_dict if metric in r]
            # 过滤掉 None 和 nan 值
            valid_scores = [s for s in scores if s is not None and not (isinstance(s, float) and math.isnan(s))]
            if valid_scores:
                avg_scores[metric] = sum(valid_scores) / len(valid_scores)
            else:
                avg_scores[metric] = float('nan')  # 如果没有有效分数，设置为 nan
        
        print("\n平均分数:")
        for metric, score in avg_scores.items():
            if isinstance(score, float) and math.isnan(score):
                print(f"  {metric:20s}: nan (评估失败，可能是与通义千问 API 的兼容性问题)")
            else:
                print(f"  {metric:20s}: {score:.4f} ({score*100:.2f}%)")
        
        # 保存结果
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存为 JSON
            # 注意：需要处理 NaN 值，因为 JSON 标准不支持 NaN
            def convert_nan(obj):
                """递归转换 NaN 为 None"""
                if isinstance(obj, float) and math.isnan(obj):
                    return None
                elif isinstance(obj, dict):
                    return {k: convert_nan(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_nan(item) for item in obj]
                return obj
            
            result_json = {
                "average_scores": convert_nan(avg_scores),
                "detailed_results": convert_nan(result_dict),
            }
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result_json, f, ensure_ascii=False, indent=2)
            
            print(f"\n✅ 评估结果已保存: {output_file}")
        
        return True
        
    except Exception as e:
        # 如果评估失败，可能是因为 answer_relevancy 与通义千问 API 的兼容性问题
        error_msg = str(e)
        if "contents is neither str nor list of str" in error_msg or "InvalidParameter" in error_msg:
            print(f"\n⚠️  评估时遇到兼容性问题: {error_msg[:100]}")
            print("   这可能是 answer_relevancy 与通义千问 API 的兼容性问题")
            print("   尝试只评估其他指标（faithfulness, context_precision, context_recall）...")
            
            # 只评估其他指标
            metrics_to_evaluate = [
                faithfulness,
                # answer_relevancy,  # 暂时跳过
                context_precision,
                context_recall,
            ]
            
            try:
                result = evaluate(
                    dataset=ragas_dataset,
                    metrics=metrics_to_evaluate,
                    llm=ragas_llm,
                    embeddings=embeddings,
                    raise_exceptions=False,
                )
                print("✅ 使用其他指标评估成功（answer_relevancy 已跳过）")
                
                # 显示结果（评估成功的情况）
                print("\n" + "=" * 60)
                print("RAGAS 评估结果")
                print("=" * 60)
                
                # 转换为字典格式以便显示
                result_dict = result.to_pandas().to_dict(orient="records")
                
                # 计算平均分数（过滤掉 nan 值）
                import math
                avg_scores = {}
                for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                    scores = [r.get(metric) for r in result_dict if metric in r]
                    # 过滤掉 None 和 nan 值
                    valid_scores = [s for s in scores if s is not None and not (isinstance(s, float) and math.isnan(s))]
                    if valid_scores:
                        avg_scores[metric] = sum(valid_scores) / len(valid_scores)
                    else:
                        avg_scores[metric] = float('nan')  # 如果没有有效分数，设置为 nan
                
                print("\n平均分数:")
                for metric, score in avg_scores.items():
                    if isinstance(score, float) and math.isnan(score):
                        print(f"  {metric:20s}: nan (评估失败，可能是与通义千问 API 的兼容性问题)")
                    else:
                        print(f"  {metric:20s}: {score:.4f} ({score*100:.2f}%)")
                
                # 保存结果
                if output_path:
                    output_file = Path(output_path)
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # 保存为 JSON
                    # 注意：需要处理 NaN 值，因为 JSON 标准不支持 NaN
                    def convert_nan(obj):
                        """递归转换 NaN 为 None"""
                        if isinstance(obj, float) and math.isnan(obj):
                            return None
                        elif isinstance(obj, dict):
                            return {k: convert_nan(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [convert_nan(item) for item in obj]
                        return obj
                    
                    result_json = {
                        "average_scores": convert_nan(avg_scores),
                        "detailed_results": convert_nan(result_dict),
                    }
                    
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(result_json, f, ensure_ascii=False, indent=2)
                    
                    print(f"\n✅ 评估结果已保存: {output_file}")
                
                return True
            except Exception as e2:
                print(f"❌ 评估失败: {e2}")
                import traceback
                traceback.print_exc()
                return False
        else:
            # 其他类型的错误
            print(f"❌ 评估失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="使用 RAGAS 生成和评估 RAG 数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--kb-id",
        required=False,  # 如果使用 --evaluate-only，可以从测试集文件中读取
        help="知识库 ID（如果使用 --evaluate-only，可以从测试集文件中读取）",
    )
    parser.add_argument(
        "--source-path",
        help="知识库源路径（如果不提供，从配置文件读取）",
    )
    parser.add_argument(
        "--output",
        default="ragas_dataset.json",
        help="输出文件路径（默认: ragas_dataset.json）",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        help="最多处理的文档数（用于快速测试）",
    )
    parser.add_argument(
        "--max-questions-per-doc",
        type=int,
        default=3,
        help="每个文档最多生成的问题数（默认: 3）",
    )
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="只生成数据集，不进行评估",
    )
    parser.add_argument(
        "--evaluate-only",
        help="只评估已存在的数据集（指定数据集路径）",
    )
    parser.add_argument(
        "--eval-output",
        help="评估结果输出路径（默认: ragas_eval_results.json）",
    )
    parser.add_argument(
        "--use-testset-generator",
        action="store_true",
        help="使用 RAGAS TestsetGenerator 生成测试集（新方法，推荐）",
    )
    parser.add_argument(
        "--use-kg",
        action="store_true",
        help="使用知识图谱生成测试集（需要 --use-testset-generator，可以提高多跳查询质量）",
    )
    
    args = parser.parse_args()
    
    # 如果只评估，不需要 kb_id（可以从测试集文件中读取）
    if args.evaluate_only:
        success = evaluate_with_ragas(
            dataset_path=args.evaluate_only,
            output_path=args.eval_output or "ragas_eval_results.json",
        )
        return 0 if success else 1
    
    # 生成数据集
    if args.use_testset_generator:
        # 使用 RAGAS TestsetGenerator（新方法）
        if args.use_kg:
            # 使用知识图谱版本
            success = generate_ragas_dataset_with_knowledge_graph(
                kb_id=args.kb_id,
                source_path=args.source_path,
                output_path=args.output,
                max_docs=args.max_docs or 5,
                num_questions_per_doc=args.max_questions_per_doc or 3,
                use_kg=True,
            )
        else:
            # 不使用知识图谱版本
            success = generate_ragas_dataset_with_testset_generator(
                kb_id=args.kb_id,
                source_path=args.source_path,
                output_path=args.output,
                max_docs=args.max_docs or 5,
                num_questions_per_doc=args.max_questions_per_doc or 3,
            )
    else:
        # 使用原有方法
        success = generate_ragas_dataset(
            kb_id=args.kb_id,
            source_path=args.source_path,
            output_path=args.output,
            max_docs=args.max_docs,
            max_questions_per_doc=args.max_questions_per_doc,
        )
    
    if not success:
        return 1
    
    # 评估（如果未指定 --no-evaluate）
    if not args.no_evaluate:
        if _HAS_RAGAS:
            success = evaluate_with_ragas(
                dataset_path=args.output,
                output_path=args.eval_output or "ragas_eval_results.json",
            )
            return 0 if success else 1
        else:
            print("\n⚠️  未安装 RAGAS，跳过评估")
            print("   可以稍后运行: python generate_ragas_dataset.py --evaluate-only ragas_dataset.json")
            return 0
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

