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

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rag import RAGEngine
from config import AppConfig

# 尝试导入 RAGAS
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from datasets import Dataset
    _HAS_RAGAS = True
except ImportError:
    _HAS_RAGAS = False
    print("⚠️  未安装 RAGAS，请运行: pip install ragas datasets")


def find_markdown_files(directory: Path) -> List[Path]:
    """递归查找所有 .md 文件"""
    md_files = []
    for file_path in directory.rglob("*.md"):
        if file_path.is_file():
            md_files.append(file_path)
    return sorted(md_files)


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
                    for chunk in result.get("chunks", []):
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
    print("-" * 60)
    
    # 2. 转换为 RAGAS 格式
    print("转换数据格式...")
    
    # RAGAS 需要的数据格式
    ragas_data = {
        "question": [s["question"] for s in samples],
        "contexts": [s["contexts"] for s in samples],  # 每个元素是字符串列表
        "answer": [s["answer"] for s in samples],
    }
    
    # ground_truth 是可选的（如果所有样本都有 ground_truth，则添加）
    if any(s.get("ground_truth") for s in samples):
        ragas_data["ground_truth"] = [s.get("ground_truth", "") for s in samples]
    
    ragas_dataset = Dataset.from_dict(ragas_data)
    print("✅ 数据格式转换完成")
    print("-" * 60)
    
    # 3. 使用 RAGAS 评估
    print("\n开始评估...")
    print("评估指标:")
    print("  - Faithfulness（忠实度）")
    print("  - Answer Relevancy（回答相关性）")
    print("  - Context Precision（上下文精确率）")
    print("  - Context Recall（上下文召回率）")
    print("-" * 60)
    
    try:
        result = evaluate(
            dataset=ragas_dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
        )
        
        # 4. 显示结果
        print("\n" + "=" * 60)
        print("RAGAS 评估结果")
        print("=" * 60)
        
        # 转换为字典格式以便显示
        result_dict = result.to_pandas().to_dict(orient="records")
        
        # 计算平均分数
        avg_scores = {}
        for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            scores = [r.get(metric, 0) for r in result_dict if metric in r]
            if scores:
                avg_scores[metric] = sum(scores) / len(scores)
        
        print("\n平均分数:")
        for metric, score in avg_scores.items():
            print(f"  {metric:20s}: {score:.4f} ({score*100:.2f}%)")
        
        # 保存结果
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存为 JSON
            result_json = {
                "average_scores": avg_scores,
                "detailed_results": result_dict,
            }
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result_json, f, ensure_ascii=False, indent=2)
            
            print(f"\n✅ 评估结果已保存: {output_file}")
        
        return True
        
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
        required=True,
        help="知识库 ID",
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
    
    args = parser.parse_args()
    
    # 如果只评估
    if args.evaluate_only:
        success = evaluate_with_ragas(
            dataset_path=args.evaluate_only,
            output_path=args.eval_output or "ragas_eval_results.json",
        )
        return 0 if success else 1
    
    # 生成数据集
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

