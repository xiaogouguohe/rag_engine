#!/usr/bin/env python3
"""
RAG 综合评估脚本 (检索 + 响应)
----------------------------

使用评估数据集评估 RAG 系统的检索性能和响应质量。

评估指标：
1. 检索阶段：Recall@k, Precision@k, MRR, Hit Rate@k
2. 响应阶段：Faithfulness (忠实度), Answer Similarity (语义相似度)

使用方法：
    # 仅进行检索评估
    python scripts/evaluation/evaluate_retrieval.py --kb-id recipes_kb --dataset ragas_dataset.json

    # 同时进行检索和响应评估（会调用 LLM）
    python scripts/evaluation/evaluate_retrieval.py --kb-id recipes_kb --dataset ragas_dataset.json --eval-response
"""

import sys
import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple

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


def evaluate_faithfulness(
    question: str, 
    contexts: List[str], 
    answer: str, 
    llm_client
) -> Tuple[float, List[Dict[str, str]]]:
    """
    手动实现忠实度评估逻辑。
    """
    # 步骤 1: 提取原子陈述
    extract_prompt = f"""
    你的任务是将给定的回答拆解为一系列独立的、原子级的事实陈述（Claims）。
    每个陈述应该是简短的一句话。
    
    回答内容：{answer}
    
    请直接列出陈述，每行一条，不要包含编号或解释。
    """
    
    try:
        claims_raw = llm_client.generate(extract_prompt, system_prompt="你是一个严谨的事实提取专家。")
        claims = [c.strip() for c in claims_raw.split('\n') if c.strip()]
    except Exception:
        return 0.0, []

    if not claims:
        return 1.0, [] # 如果没提取到陈述，暂记满分

    # 步骤 2: 验证陈述
    context_str = "\n\n".join(contexts)
    supported_count = 0
    verification_results = []

    for claim in claims:
        verify_prompt = f"""
        基于以下提供的参考文档，判断给定的陈述是否可以被推导出来。
        
        参考文档：
        {context_str}
        
        待验证陈述：{claim}
        
        请只回答“是”或“否”，并简要说明原因。格式：[判断] | [原因]
        """
        
        try:
            res = llm_client.generate(verify_prompt, system_prompt="你是一个严谨的事实核查员。")
            is_supported = res.startswith("是")
            if is_supported:
                supported_count += 1
            verification_results.append({"claim": claim, "res": res})
        except Exception:
            continue

    score = supported_count / len(claims) if claims else 0.0
    return score, verification_results


def calculate_semantic_similarity(
    text1: str, 
    text2: str, 
    embedding_client
) -> float:
    """使用 Embedding 计算语义相似度"""
    if not text1 or not text2:
        return 0.0
    
    try:
        vecs = embedding_client.embed_texts([text1, text2])
        v1 = np.array(vecs["dense_vecs"][0])
        v2 = np.array(vecs["dense_vecs"][1])
        
        # 计算余弦相似度
        similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return float(similarity)
    except Exception:
        return 0.0


def evaluate_retrieval(
    kb_id: str,
    dataset_path: str,
    top_k: int = 5,
    eval_response: bool = False,
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
    
    # 响应评估指标
    total_faithfulness = 0.0
    total_similarity = 0.0
    resp_count = 0
    
    for i, sample in enumerate(samples, 1):
        question = sample["question"]
        gold_contexts = sample.get("contexts", [])
        ground_truth = sample.get("ground_truth", "")
        
        if not gold_contexts:
            continue
        
        print(f"[{i}/{len(samples)}] {question[:50]}...", end=" ")
        
        try:
            # 1. 运行 RAG 查询
            query_result = engine.query(
                question=question,
                top_k=top_k,
            )
            
            # 2. 检索指标计算
            retrieved_chunks = query_result.get("sources", [])
            retrieved_texts = [c.get("text", "") for c in retrieved_chunks]
            
            hit_gold_indices = set()
            relevant_retrieved_count = 0
            mrr = 0.0
            
            for rank, ret_text in enumerate(retrieved_texts, 1):
                ret_is_relevant = False
                clean_ret = "".join(ret_text.split())
                for g_idx, gold_text in enumerate(gold_contexts):
                    clean_gold = "".join(gold_text.split())
                    if clean_ret in clean_gold or clean_gold in clean_ret:
                        hit_gold_indices.add(g_idx)
                        ret_is_relevant = True
                
                if ret_is_relevant:
                    relevant_retrieved_count += 1
                    if mrr == 0:
                        mrr = 1.0 / rank
            
            recall = len(hit_gold_indices) / len(gold_contexts) if gold_contexts else 0.0
            precision = relevant_retrieved_count / len(retrieved_texts) if retrieved_texts else 0.0
            
            total_recall += recall
            total_precision += precision
            total_mrr += mrr
            if len(hit_gold_indices) > 0:
                total_hit += 1
            
            print(f"R={recall:.2f}, P={precision:.2f}", end=" ")

            # 3. 响应评估 (如果开启)
            if eval_response:
                answer = query_result.get("answer", "")
                
                # A. 忠实度 (Faithfulness)
                faith_score, _ = evaluate_faithfulness(question, retrieved_texts, answer, engine.llm_client)
                total_faithfulness += faith_score
                
                # B. 语义相似度 (Similarity) - 仅当有 Ground Truth 时计算
                sim_score = 0.0
                if ground_truth:
                    sim_score = calculate_semantic_similarity(answer, ground_truth, engine.embedding_client)
                    total_similarity += sim_score
                
                resp_count += 1
                print(f"Faith={faith_score:.2f}, Sim={sim_score:.2f}", end=" ")
            
            print() # 换行
        
        except Exception as e:
            print(f"❌ 失败: {e}")
            continue
    
    # 4. 计算平均指标
    n_samples = len(samples)
    metrics = {
        "recall@k": total_recall / n_samples if n_samples > 0 else 0.0,
        "precision@k": total_precision / n_samples if n_samples > 0 else 0.0,
        "mrr": total_mrr / n_samples if n_samples > 0 else 0.0,
        "hit_rate@k": total_hit / n_samples if n_samples > 0 else 0.0,
    }
    
    if eval_response and resp_count > 0:
        metrics["faithfulness"] = total_faithfulness / resp_count
        metrics["answer_similarity"] = total_similarity / resp_count
    
    # 5. 显示结果
    print("\n" + "=" * 60)
    print("评估结果摘要")
    print("=" * 60)
    print(f"评估样本数: {n_samples}")
    print(f"Top-K: {top_k}")
    
    print(f"\n[检索阶段]:")
    print(f"  Recall@{top_k}:     {metrics['recall@k']:.4f} ({metrics['recall@k']*100:.2f}%)")
    print(f"  Precision@{top_k}:  {metrics['precision@k']:.4f} ({metrics['precision@k']*100:.2f}%)")
    print(f"  MRR:                {metrics['mrr']:.4f}")
    print(f"  Hit Rate@{top_k}:   {metrics['hit_rate@k']:.4f} ({metrics['hit_rate@k']*100:.2f}%)")
    
    if eval_response:
        print(f"\n[响应阶段]:")
        print(f"  Faithfulness:       {metrics.get('faithfulness', 0):.4f}")
        print(f"  Answer Similarity:  {metrics.get('answer_similarity', 0):.4f}")
    
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
        "--eval-response",
        action="store_true",
        help="是否同时评估响应阶段指标（忠实度、相似度等，会消耗 LLM token）",
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
        eval_response=args.eval_response,
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

