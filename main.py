#!/usr/bin/env python3
"""
RAG 引擎统一入口
---------------

功能：
1. load: 根据 JSON 配置文件向量化知识库（用完即退）
2. chat: 进入交互对话模式（持久化进程）
3. query: 单次问题查询
4. stats: 查看知识库统计信息
"""

import sys
import argparse
import json
from pathlib import Path
from typing import List, Optional

from rag import RAGEngine
from config import AppConfig, KnowledgeBaseConfig


def load_config_from_json(config_path: str) -> List[KnowledgeBaseConfig]:
    """从 JSON 配置文件加载知识库配置"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    kb_configs = []
    for kb in config.get("knowledge_bases", []):
        kb_configs.append(KnowledgeBaseConfig(
            kb_id=kb["kb_id"],
            source_path=kb["source_path"],
            file_pattern=kb.get("file_pattern", "*.md"),
            top_k=kb.get("top_k", 4),
            use_sparse=kb.get("use_sparse", False),
            use_multi_vector=kb.get("use_multi_vector", False),
            use_query_rewrite=kb.get("use_query_rewrite", False),
            use_markdown_header_split=kb.get("use_markdown_header_split", True),
        ))
    
    return kb_configs


def handle_load(args):
    """根据配置文件加载知识库（向量化逻辑）"""
    kb_configs = []
    config_file = Path(args.config)
    
    if config_file.exists():
        kb_configs = load_config_from_json(args.config)
        print(f"✅ 正在读取配置文件: {args.config}")
    else:
        print(f"❌ 配置文件不存在: {args.config}")
        return 1
    
    # 如果指定了具体的 kb_id，则只处理那一个
    if args.kb_id:
        kb_configs = [kb for kb in kb_configs if kb.kb_id == args.kb_id]
        if not kb_configs:
            print(f"❌ 配置文件中未找到 kb_id: {args.kb_id}")
            return 1
    
    for kb_config in kb_configs:
        print(f"\n🚀 开始处理知识库: {kb_config.kb_id}")
        engine = RAGEngine(
            kb_id=kb_config.kb_id,
            use_markdown_header_split=kb_config.use_markdown_header_split,
        )
        engine.ingest_directory(
            dir_path=kb_config.source_path,
            pattern=kb_config.file_pattern,
            verbose=True
        )
    print("\n✅ 所有向量化任务已完成。")
    return 0


def get_kb_config(kb_id: str, config_path: str = "rag_config.json") -> Optional[KnowledgeBaseConfig]:
    """获取指定 ID 的知识库配置"""
    try:
        kb_configs = load_config_from_json(config_path)
        for kb in kb_configs:
            if kb.kb_id == kb_id:
                return kb
    except Exception:
        pass
    return None


def handle_chat(args):
    """交互对话模式逻辑"""
    kb_config = get_kb_config(args.kb_id)
    
    # 确定最终使用的 top_k: 命令行指定优先，否则用配置文件，最后默认 4
    top_k = args.top_k
    if top_k == 4 and kb_config and kb_config.top_k != 4:
        top_k = kb_config.top_k

    print(f"\n💬 进入交互对话模式 (知识库: {args.kb_id}, Top-K: {top_k})")
    print("输入 'exit', 'quit' 或 'q' 退出。输入 'clear' 清屏。")
    print("-" * 50)
    
    engine = RAGEngine(kb_id=args.kb_id)
    
    # 维护对话历史
    history = []
    
    while True:
        try:
            question = input("\n👤 用户: ").strip()
            
            if not question:
                continue
            if question.lower() in ["exit", "quit", "q"]:
                print("👋 已退出对话。")
                break
            if question.lower() == "clear":
                print("\033c", end="") # 清屏
                history = [] # 清屏时也重置历史
                continue
                
            print("🤖 AI 正在思考...", end="", flush=True)
            result = engine.query(question, top_k=top_k, history=history)
            print("\r" + " " * 30 + "\r", end="") # 清除“思考中”提示
            
            print(f"🤖 AI: {result['answer']}")
            
            # 更新历史
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": result['answer']})
            if len(history) > 10:
                history = history[-10:]
            
            if args.show_sources:
                print("\n   [参考来源]")
                for i, chunk in enumerate(result.get("chunks", []), 1):
                    print(f"   ({i}) {chunk.get('text', '')[:80]}... (Score: {chunk.get('score', 0):.4f})")
                    
        except KeyboardInterrupt:
            print("\n👋 已退出对话。")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
    return 0


def handle_query(args):
    """单次查询逻辑"""
    kb_config = get_kb_config(args.kb_id)
    top_k = args.top_k
    if top_k == 4 and kb_config and kb_config.top_k != 4:
        top_k = kb_config.top_k

    engine = RAGEngine(kb_id=args.kb_id)
    result = engine.query(args.question, top_k=top_k)
    
    print(f"\n🤖 AI 回答 (Top-K: {top_k}):\n{result.get('answer', '')}")
    return 0


def handle_stats(args):
    """统计逻辑"""
    engine = RAGEngine(kb_id=args.kb_id)
    stats = engine.get_stats()
    print(f"\n📊 知识库 [{args.kb_id}] 统计信息:")
    print(f"   向量数量: {stats.get('vector_count', 0)}")
    print(f"   向量维度: {stats.get('vector_dim', '未知')}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="RAG 引擎统一入口")
    subparsers = parser.add_subparsers(dest="command")
    
    # 1. load 命令 - 仅支持通过 JSON 配置文件加载
    load_parser = subparsers.add_parser("load", help="从 JSON 配置文件加载并向量化知识库")
    load_parser.add_argument("--config", default="rag_config.json", help="配置文件路径")
    load_parser.add_argument("--kb-id", help="指定要加载的知识库 ID")
    
    # 2. chat 命令 - 交互式对话
    chat_parser = subparsers.add_parser("chat", help="进入交互式对话模式")
    chat_parser.add_argument("--kb-id", required=True, help="要对话的知识库 ID")
    chat_parser.add_argument("--top-k", type=int, default=4, help="检索数量")
    chat_parser.add_argument("--show-sources", action="store_true", help="显示参考来源")
    
    # 3. query 命令 - 单次查询
    query_parser = subparsers.add_parser("query", help="单次问题查询")
    query_parser.add_argument("--kb-id", required=True, help="知识库 ID")
    query_parser.add_argument("--question", required=True, help="问题内容")
    query_parser.add_argument("--top-k", type=int, default=4, help="检索数量")
    
    # 4. stats 命令 - 查看统计
    stats_parser = subparsers.add_parser("stats", help="查看知识库统计信息")
    stats_parser.add_argument("--kb-id", required=True, help="知识库 ID")
    
    args = parser.parse_args()
    
    if args.command == "load":
        sys.exit(handle_load(args))
    elif args.command == "chat":
        sys.exit(handle_chat(args))
    elif args.command == "query":
        sys.exit(handle_query(args))
    elif args.command == "stats":
        sys.exit(handle_stats(args))
    else:
        # 默认如果不带命令，显示帮助
        parser.print_help()


if __name__ == "__main__":
    main()
