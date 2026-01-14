#!/usr/bin/env python3
"""
RAG 引擎主入口
-------------

统一管理知识库的加载、查询和统计。
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
            use_markdown_header_split=kb.get("use_markdown_header_split", True),
        ))
    
    return kb_configs


def handle_ingest(args):
    """处理加载文档的逻辑"""
    engine = RAGEngine(
        kb_id=args.kb_id,
        use_markdown_header_split=not args.no_markdown_split,
    )
    
    if args.file:
        print(f"正在加载单个文档: {args.file}")
        result = engine.ingest_document(args.file)
        print(f"✅ 成功 - 分块数: {result['chunks_count']}")
    elif args.dir:
        result = engine.ingest_directory(
            dir_path=args.dir,
            pattern=args.pattern,
            verbose=True
        )
        print("\n" + "=" * 40)
        print("批量加载完成")
        print(f"  成功: {result['success_count']}")
        print(f"  失败: {result['fail_count']}")
        print(f"  总计: {result['total_files']}")
        print(f"  总分块数: {result['total_chunks']}")
    else:
        print("❌ 请指定 --file 或 --dir 参数")
        return 1
    return 0


def handle_load_all(args):
    """根据配置文件加载所有知识库"""
    kb_configs = []
    config_file = Path(args.config)
    
    if config_file.exists():
        kb_configs = load_config_from_json(args.config)
        print(f"✅ 从配置文件加载: {args.config}")
    else:
        app_config = AppConfig.load()
        if app_config.knowledge_bases:
            kb_configs = app_config.knowledge_bases
            print("✅ 从环境变量加载知识库配置")
    
    if not kb_configs:
        print("❌ 未找到任何知识库配置")
        return 1
    
    if args.kb_id:
        kb_configs = [kb for kb in kb_configs if kb.kb_id == args.kb_id]
    
    for kb_config in kb_configs:
        print(f"\n开始加载知识库: {kb_config.kb_id}")
        engine = RAGEngine(
            kb_id=kb_config.kb_id,
            use_markdown_header_split=kb_config.use_markdown_header_split,
        )
        engine.ingest_directory(
            dir_path=kb_config.source_path,
            pattern=kb_config.file_pattern,
            verbose=True
        )
    return 0


def handle_query(args):
    """处理查询逻辑"""
    engine = RAGEngine(kb_id=args.kb_id)
    result = engine.query(args.question, top_k=args.top_k)
    
    print("\n📚 检索到的相关内容:")
    for i, chunk in enumerate(result.get("chunks", []), 1):
        print(f"  [{i}] (Score: {chunk.get('score', 0):.4f}) {chunk.get('text', '')[:100]}...")
    
    print("\n🤖 AI 回答:")
    print(f"  {result.get('answer', '')}")
    return 0


def handle_stats(args):
    """处理统计逻辑"""
    engine = RAGEngine(kb_id=args.kb_id)
    stats = engine.get_stats()
    print(f"\n📊 知识库 [{args.kb_id}] 统计信息:")
    print(f"   向量数量: {stats.get('vector_count', 0)}")
    print(f"   向量维度: {stats.get('vector_dim', '未知')}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="RAG 引擎统一入口")
    subparsers = parser.add_subparsers(dest="command")
    
    # 1. ingest 命令 (只做向量化并退出)
    ingest_parser = subparsers.add_parser("ingest", help="执行向量化并保存到数据库")
    ingest_parser.add_argument("--kb-id", required=True, help="知识库 ID")
    ingest_parser.add_argument("--file", help="单个文件路径")
    ingest_parser.add_argument("--dir", help="文件夹路径")
    ingest_parser.add_argument("--pattern", default="*.md", help="文件匹配模式")
    ingest_parser.add_argument("--no-markdown-split", action="store_true", help="禁用 Markdown 标题分割")
    
    # 2. query 命令
    query_parser = subparsers.add_parser("query", help="查询知识库")
    query_parser.add_argument("--kb-id", required=True, help="知识库 ID")
    query_parser.add_argument("--question", required=True, help="问题内容")
    query_parser.add_argument("--top-k", type=int, default=4, help="检索数量")
    
    # 3. stats 命令
    stats_parser = subparsers.add_parser("stats", help="查看统计信息")
    stats_parser.add_argument("--kb-id", required=True, help="知识库 ID")
    
    # 默认行为：从配置加载所有知识库
    parser.add_argument("--config", default="knowledge_bases.json", help="配置文件路径")
    parser.add_argument("--kb-id", help="指定要加载的知识库 ID")
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        sys.exit(handle_ingest(args))
    elif args.command == "query":
        sys.exit(handle_query(args))
    elif args.command == "stats":
        sys.exit(handle_stats(args))
    else:
        # 如果没有子命令，默认执行批量加载
        sys.exit(handle_load_all(args))


if __name__ == "__main__":
    main()
