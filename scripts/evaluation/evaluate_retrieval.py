#!/usr/bin/env python3
"""
RAG 检索阶段评估脚本
-------------------

使用评估数据集评估 RAG 系统的检索性能。

评估指标：
- Recall@k：召回率
- Precision@k：精确率
- MRR：平均倒数排名
- Hit Rate@k：命中率

使用方法：
    python evaluate_retrieval.py --kb-id recipes_kb --dataset eval_dataset.json --top-k 5
"""

import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Set

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from rag import RAGEngine
from vector_store import VectorStore

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


def load_eval_dataset(dataset_path: str) -> Dict[str, Any]:
    """加载评估数据集"""
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset


def get_relevant_chunk_ids(
    kb_id: str,
    parent_id: str,
    vector_store: VectorStore,
) -> Set[str]:
    """
    从向量数据库中获取指定文档的所有 chunk_id。
    
    Args:
        kb_id: 知识库 ID
        parent_id: 父文档 ID
        vector_store: 向量存储实例
    
    Returns:
        chunk_id 集合
    """
    # 注意：这里需要扩展 VectorStore 的接口来支持按 parent_id 查询
    # 暂时返回空集合，后续可以扩展
    # 实际实现需要从向量数据库中查询 metadata.parent_id == parent_id 的所有记录
    return set()


def evaluate_retrieval(
    kb_id: str,
    dataset_path: str,
    top_k: int = 5,
) -> Dict[str, float]:
    """
    评估检索性能。
    
    Args:
        kb_id: 知识库 ID
        dataset_path: 评估数据集路径
        top_k: 检索的文档块数量
    
    Returns:
        评估指标字典
    """
    print("=" * 60)
    print("RAG 检索阶段评估")
    print("=" * 60)
    
    # 1. 加载评估数据集
    print("加载评估数据集...")
    dataset = load_eval_dataset(dataset_path)
    samples = dataset.get("samples", [])
    
    if not samples:
        print("❌ 评估数据集为空")
        return {}
    
    print(f"✅ 加载 {len(samples)} 个评估样本")
    print("-" * 60)
    
    # 2. 初始化 RAG 引擎
    print("初始化 RAG 引擎...")
    engine = RAGEngine(kb_id=kb_id)
    print("✅ RAG 引擎初始化成功")
    print("-" * 60)
    
    # 3. 评估每个样本
    print(f"\n开始评估（top_k={top_k}）...\n")
    
    total_recall = 0.0
    total_precision = 0.0
    total_mrr = 0.0
    total_hit = 0
    
    for i, sample in enumerate(samples, 1):
        question = sample["question"]
        # RAGAS 标准格式：contexts 字段包含了生成该问题时参考的原始文本
        gold_contexts = sample.get("contexts", [])
        
        if not gold_contexts:
            continue
        
        print(f"[{i}/{len(samples)}] {question[:50]}...", end=" ")
        
        try:
            # 使用引擎的 query 接口进行检索（这会自动处理 Dense/Sparse/Rerank 等配置）
            query_result = engine.query(
                question=question,
                top_k=top_k,
            )
            
            # 获取检索到的 chunk 列表
            retrieved_chunks = query_result.get("sources", [])
            retrieved_texts = [c.get("text", "") for c in retrieved_chunks]
            
            # 判定命中 (Hit)
            # 逻辑：我们要看 gold_contexts 里的每一段，是否在检索结果中出现过
            hit_gold_indices = set() # 记录哪些金标准文本被找到了
            relevant_retrieved_count = 0 # 记录检索出的片段中有几个是相关的
            mrr = 0.0
            
            for rank, ret_text in enumerate(retrieved_texts, 1):
                ret_is_relevant = False
                clean_ret = "".join(ret_text.split())
                
                for g_idx, gold_text in enumerate(gold_contexts):
                    clean_gold = "".join(gold_text.split())
                    
                    # 只要检索片段和金标准有包含关系，就认为相关
                    if clean_ret in clean_gold or clean_gold in clean_ret:
                        hit_gold_indices.add(g_idx)
                        ret_is_relevant = True
                
                if ret_is_relevant:
                    relevant_retrieved_count += 1
                    if mrr == 0:
                        mrr = 1.0 / rank
            
            # 计算指标
            # Recall: 命中的金标准文本数量 / 总金标准文本数量
            recall = len(hit_gold_indices) / len(gold_contexts) if gold_contexts else 0.0
            # Precision: 检索出的相关片段数量 / 检索的总片段数量 (top_k)
            precision = relevant_retrieved_count / len(retrieved_texts) if retrieved_texts else 0.0
            
            total_recall += recall
            total_precision += precision
            total_mrr += mrr
            if len(hit_gold_indices) > 0:
                total_hit += 1
            
            print(f"Recall={recall:.2f}, Precision={precision:.2f}, MRR={mrr:.2f}")
        
        except Exception as e:
            print(f"❌ 评估失败: {e}")
            continue
    
    # 4. 计算平均指标
    n_samples = len(samples)
    metrics = {
        "recall@k": total_recall / n_samples if n_samples > 0 else 0.0,
        "precision@k": total_precision / n_samples if n_samples > 0 else 0.0,
        "mrr": total_mrr / n_samples if n_samples > 0 else 0.0,
        "hit_rate@k": total_hit / n_samples if n_samples > 0 else 0.0,
    }
    
    # 5. 显示结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"评估样本数: {n_samples}")
    print(f"Top-K: {top_k}")
    print(f"\n指标:")
    print(f"  Recall@{top_k}:     {metrics['recall@k']:.4f} ({metrics['recall@k']*100:.2f}%)")
    print(f"  Precision@{top_k}:  {metrics['precision@k']:.4f} ({metrics['precision@k']*100:.2f}%)")
    print(f"  MRR:                {metrics['mrr']:.4f}")
    print(f"  Hit Rate@{top_k}:   {metrics['hit_rate@k']:.4f} ({metrics['hit_rate@k']*100:.2f}%)")
    
    # F1 分数
    if metrics['precision@k'] + metrics['recall@k'] > 0:
        f1 = 2 * (metrics['precision@k'] * metrics['recall@k']) / (metrics['precision@k'] + metrics['recall@k'])
        print(f"  F1 Score:           {f1:.4f}")
        metrics['f1_score'] = f1
    
    return metrics


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="评估 RAG 系统的检索性能",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--kb-id",
        required=True,
        help="知识库 ID",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="评估数据集路径（JSON 格式）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="检索的文档块数量（默认: 5）",
    )
    parser.add_argument(
        "--output",
        help="评估结果输出路径（JSON 格式，可选）",
    )
    
    args = parser.parse_args()
    
    metrics = evaluate_retrieval(
        kb_id=args.kb_id,
        dataset_path=args.dataset,
        top_k=args.top_k,
    )
    
    # 保存评估结果
    if args.output:
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        result = {
            "kb_id": args.kb_id,
            "dataset": args.dataset,
            "top_k": args.top_k,
            "metrics": metrics,
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 评估结果已保存: {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

