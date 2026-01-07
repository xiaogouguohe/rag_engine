#!/usr/bin/env python3
"""
自动生成 RAG 评估数据集
---------------------

从知识库中自动生成检索阶段评估数据集。

思路：
1. 读取知识库中的所有文档
2. 从文档中提取或生成问题（使用 LLM）
3. 标注相关文档块（基于文档结构）
4. 生成评估数据集（JSON 格式）

使用方法：
    # 从配置文件的知识库生成评估数据集
    python generate_eval_dataset.py --kb-id recipes_kb --output eval_dataset.json

    # 指定知识库源路径
    python generate_eval_dataset.py --kb-id recipes_kb --source-path ../HowToCook/dishes --output eval_dataset.json

    # 快速测试（只处理前 10 个文档）
    python generate_eval_dataset.py --kb-id recipes_kb --max-docs 10 --output eval_dataset_test.json
"""

import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib
import uuid

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rag import RAGEngine
from config import AppConfig
from document import ParserFactory, MarkdownParser


def extract_questions_from_document(
    file_path: Path,
    llm_client,
    max_questions: int = 3,
) -> List[str]:
    """
    从文档中提取或生成问题。
    
    策略：
    1. 从文档标题生成问题（如："西红柿鸡蛋的做法" → "如何做西红柿鸡蛋？"）
    2. 从文档内容生成问题（使用 LLM）
    
    Args:
        file_path: 文档路径
        llm_client: LLM 客户端（用于生成问题）
        max_questions: 每个文档最多生成的问题数
    
    Returns:
        问题列表
    """
    questions = []
    
    # 策略1：从文件名生成问题
    file_name = file_path.stem
    if file_name:
        # 简单的转换：文件名 → 问题
        question = f"如何做{file_name}？"
        questions.append(question)
    
    # 策略2：从文档内容生成问题（使用 LLM）
    try:
        # 读取文档内容
        parser = ParserFactory.get_parser(file_path)
        if parser:
            doc = parser.parse(file_path)
            content = doc.get("content", "")
            
            # 提取前 500 字符作为上下文
            content_preview = content[:500]
            
            # 使用 LLM 生成问题
            prompt = f"""基于以下菜谱内容，生成 {max_questions - len(questions)} 个用户可能问的问题。
要求：
1. 问题应该与菜谱内容相关
2. 问题应该具体、可回答
3. 问题应该多样化（如：做法、原料、难度等）

菜谱内容：
{content_preview}

请生成问题，每行一个问题，不要编号："""
            
            try:
                response = llm_client.generate(prompt)
                # 解析 LLM 返回的问题（每行一个问题）
                generated_questions = [
                    q.strip() for q in response.split("\n")
                    if q.strip() and not q.strip().startswith(("#", "-", "1.", "2.", "3."))
                ]
                questions.extend(generated_questions[:max_questions - len(questions)])
            except Exception as e:
                # 如果 LLM 生成失败，使用简单策略
                pass
    except Exception as e:
        # 如果解析失败，跳过
        pass
    
    return questions[:max_questions]


def find_relevant_chunks(
    file_path: Path,
    question: str,
    engine: RAGEngine,
) -> List[str]:
    """
    找到与问题相关的文档块。
    
    策略：
    1. 如果问题来自某个文档，该文档的所有块都是相关的
    2. 可以通过向量相似度找到其他相关块（可选）
    
    Args:
        file_path: 文档路径
        question: 问题
        engine: RAG 引擎
    
    Returns:
        相关文档块的 chunk_id 列表
    """
    relevant_chunks = []
    
    # 策略1：找到该文档的所有块
    # 通过文件路径生成 parent_id
    try:
        relative_path = file_path.resolve().as_posix()
        parent_id = hashlib.md5(relative_path.encode("utf-8")).hexdigest()
    except Exception:
        parent_id = None
    
    if parent_id:
        # 从向量数据库中查找该文档的所有块
        # 注意：这里需要访问向量数据库，可能需要扩展 RAGEngine 的接口
        # 暂时使用简单策略：标记整个文档的所有块为相关
        relevant_chunks.append(f"parent_{parent_id}")
    
    return relevant_chunks


def generate_eval_dataset_from_kb(
    kb_id: str,
    source_path: Optional[str] = None,
    output_path: str = "eval_dataset.json",
    max_questions_per_doc: int = 3,
    max_docs: Optional[int] = None,
) -> bool:
    """
    从知识库生成评估数据集。
    
    Args:
        kb_id: 知识库 ID
        source_path: 知识库源路径（如果不提供，从配置文件读取）
        output_path: 输出文件路径
        max_questions_per_doc: 每个文档最多生成的问题数
        max_docs: 最多处理的文档数（用于快速测试）
    """
    print("=" * 60)
    print("生成 RAG 评估数据集")
    print("=" * 60)
    
    # 1. 确定源路径
    if source_path:
        source_dir = Path(source_path)
    else:
        # 从配置文件读取
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
                print("❌ 未找到知识库配置")
                return False
        except Exception as e:
            print(f"❌ 加载配置失败: {e}")
            return False
    
    if not source_dir.exists():
        print(f"❌ 源路径不存在: {source_dir}")
        return False
    
    print(f"知识库 ID: {kb_id}")
    print(f"源路径: {source_dir}")
    print("-" * 60)
    
    # 2. 查找所有 .md 文件
    print("正在扫描文档...")
    md_files = list(source_dir.rglob("*.md"))
    
    if max_docs:
        md_files = md_files[:max_docs]
    
    if not md_files:
        print("❌ 未找到 .md 文件")
        return False
    
    print(f"✅ 找到 {len(md_files)} 个文档")
    print("-" * 60)
    
    # 3. 初始化 RAG 引擎和 LLM 客户端
    print("初始化 RAG 引擎...")
    try:
        engine = RAGEngine(kb_id=kb_id)
        llm_client = engine.llm_client
        print("✅ RAG 引擎初始化成功")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return False
    
    print("-" * 60)
    
    # 4. 为每个文档生成问题和相关文档块
    print(f"\n开始生成评估数据集...\n")
    eval_dataset = []
    
    for i, file_path in enumerate(md_files, 1):
        print(f"[{i}/{len(md_files)}] 处理: {file_path.name}")
        
        try:
            # 生成问题
            questions = extract_questions_from_document(
                file_path,
                llm_client,
                max_questions=max_questions_per_doc,
            )
            
            if not questions:
                print(f"  ⚠️  未能生成问题，跳过")
                continue
            
            # 为每个问题创建评估样本
            for question in questions:
                # 找到相关文档块（简化策略：整个文档的所有块都是相关的）
                try:
                    relative_path = file_path.resolve().as_posix()
                    parent_id = hashlib.md5(relative_path.encode("utf-8")).hexdigest()
                except Exception:
                    parent_id = str(uuid.uuid4())
                
                # 创建评估样本
                eval_sample = {
                    "id": str(uuid.uuid4()),
                    "question": question,
                    "source_document": str(file_path.relative_to(source_dir)),
                    "parent_id": parent_id,
                    "relevant_chunks": [parent_id],  # 存储 parent_id，表示整个文档的所有块都是相关的
                    "metadata": {
                        "file_name": file_path.name,
                        "file_path": str(file_path),
                    }
                }
                
                eval_dataset.append(eval_sample)
                print(f"  ✅ 生成问题: {question[:50]}...")
        
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            continue
    
    # 5. 保存评估数据集
    print("\n" + "=" * 60)
    print("保存评估数据集...")
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    dataset = {
        "kb_id": kb_id,
        "source_path": str(source_dir),
        "total_samples": len(eval_dataset),
        "samples": eval_dataset,
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 评估数据集已保存: {output_file}")
    print(f"   总样本数: {len(eval_dataset)}")
    print(f"   处理的文档数: {len(md_files)}")
    
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="从知识库生成 RAG 评估数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--kb-id",
        required=True,
        help="知识库 ID",
    )
    parser.add_argument(
        "--source-path",
        help="知识库源路径（如果不提供，从配置文件读取）",
    )
    parser.add_argument(
        "--output",
        default="eval_dataset.json",
        help="输出文件路径（默认: eval_dataset.json）",
    )
    parser.add_argument(
        "--max-questions-per-doc",
        type=int,
        default=3,
        help="每个文档最多生成的问题数（默认: 3）",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        help="最多处理的文档数（用于快速测试，默认: 处理所有）",
    )
    
    args = parser.parse_args()
    
    success = generate_eval_dataset_from_kb(
        kb_id=args.kb_id,
        source_path=args.source_path,
        output_path=args.output,
        max_questions_per_doc=args.max_questions_per_doc,
        max_docs=args.max_docs,
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

