#!/usr/bin/env python3
"""
RAG 引擎主入口
-------------

提供命令行接口，用于：
1. 加载文档到知识库（触发解析、切块、向量化、存储）
2. 查询知识库
3. 查看知识库统计信息

使用方法：
    # 加载单个文档
    python main.py ingest --kb-id my_kb --file path/to/doc.md

    # 批量加载文档
    python main.py ingest --kb-id my_kb --dir path/to/docs

    # 查询知识库
    python main.py query --kb-id my_kb --question "你的问题"

    # 查看统计信息
    python main.py stats --kb-id my_kb
"""

import sys
import argparse
from pathlib import Path
from typing import List

from rag import RAGEngine
from config import AppConfig


def ingest_document(kb_id: str, file_path: str, **kwargs):
    """
    加载单个文档到知识库（触发解析、切块、向量化、存储）。
    
    Args:
        kb_id: 知识库 ID
        file_path: 文档文件路径
        **kwargs: 其他参数（如 use_markdown_header_split）
    """
    print(f"正在加载文档: {file_path}")
    print(f"知识库 ID: {kb_id}")
    print("-" * 60)
    
    # 初始化 RAG 引擎
    engine = RAGEngine(
        kb_id=kb_id,
        use_markdown_header_split=kwargs.get("use_markdown_header_split", True),
    )
    
    # 触发加载和切块（这里会调用 ingest_document，内部会触发解析、切块、向量化、存储）
    try:
        result = engine.ingest_document(file_path)
        
        print("✅ 文档加载成功！")
        print(f"   文档 ID: {result['doc_id']}")
        print(f"   分块数量: {result['chunks_count']}")
        print(f"   状态: {result['status']}")
        
        return True
    except Exception as e:
        print(f"❌ 文档加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def ingest_directory(kb_id: str, dir_path: str, pattern: str = "**/*", **kwargs):
    """
    批量加载目录中的文档。
    
    Args:
        kb_id: 知识库 ID
        dir_path: 目录路径
        pattern: 文件匹配模式（默认：所有文件）
        **kwargs: 其他参数
    """
    dir_path = Path(dir_path)
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"❌ 目录不存在: {dir_path}")
        return False
    
    print(f"正在批量加载文档: {dir_path}")
    print(f"知识库 ID: {kb_id}")
    print(f"匹配模式: {pattern}")
    print("-" * 60)
    
    # 查找所有文档文件
    files = list(dir_path.glob(pattern))
    
    # 过滤出支持的文件类型
    supported_extensions = {".txt", ".md", ".markdown", ".mdx"}
    files = [f for f in files if f.suffix.lower() in supported_extensions]
    
    if not files:
        print(f"❌ 未找到支持的文档文件（支持: {supported_extensions}）")
        return False
    
    print(f"找到 {len(files)} 个文档文件")
    print("-" * 60)
    
    # 初始化 RAG 引擎（只初始化一次，提高效率）
    engine = RAGEngine(
        kb_id=kb_id,
        use_markdown_header_split=kwargs.get("use_markdown_header_split", True),
    )
    
    # 逐个加载文档
    success_count = 0
    fail_count = 0
    
    for i, file_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] 处理: {file_path.name}")
        try:
            result = engine.ingest_document(file_path)
            print(f"  ✅ 成功 - 分块数: {result['chunks_count']}")
            success_count += 1
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            fail_count += 1
    
    print("\n" + "=" * 60)
    print("批量加载完成")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print(f"  总计: {len(files)}")
    
    return fail_count == 0


def query_knowledge_base(kb_id: str, question: str, top_k: int = 4):
    """
    查询知识库。
    
    Args:
        kb_id: 知识库 ID
        question: 问题
        top_k: 检索的文档块数量
    """
    print(f"问题: {question}")
    print(f"知识库 ID: {kb_id}")
    print("-" * 60)
    
    # 初始化 RAG 引擎
    engine = RAGEngine(kb_id=kb_id)
    
    try:
        # 查询
        result = engine.query(question, top_k=top_k)
        
        print("\n📚 检索到的相关文档块:")
        for i, chunk in enumerate(result.get("chunks", []), 1):
            print(f"\n  [{i}] 相似度: {chunk.get('score', 0):.4f}")
            print(f"      内容: {chunk.get('text', '')[:100]}...")
            if chunk.get("metadata"):
                print(f"      元数据: {chunk['metadata']}")
        
        print("\n🤖 AI 回答:")
        print(f"  {result.get('answer', '')}")
        
        return True
    except Exception as e:
        print(f"❌ 查询失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_stats(kb_id: str):
    """显示知识库统计信息"""
    print(f"知识库 ID: {kb_id}")
    print("-" * 60)
    
    # 初始化 RAG 引擎
    engine = RAGEngine(kb_id=kb_id)
    
    try:
        stats = engine.get_stats()
        
        print("📊 知识库统计信息:")
        print(f"   向量数量: {stats.get('vector_count', 0)}")
        print(f"   文档数量: {stats.get('document_count', 0)}")
        
        if stats.get("categories"):
            print(f"   分类分布: {stats['categories']}")
        
        if stats.get("difficulties"):
            print(f"   难度分布: {stats['difficulties']}")
        
        return True
    except Exception as e:
        print(f"❌ 获取统计信息失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="RAG 引擎命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # ingest 命令：加载文档
    ingest_parser = subparsers.add_parser("ingest", help="加载文档到知识库")
    ingest_parser.add_argument("--kb-id", required=True, help="知识库 ID")
    ingest_parser.add_argument("--file", help="单个文档文件路径")
    ingest_parser.add_argument("--dir", help="文档目录路径（批量加载）")
    ingest_parser.add_argument("--pattern", default="**/*", help="文件匹配模式（批量加载时使用）")
    ingest_parser.add_argument("--no-markdown-split", action="store_true", help="禁用 Markdown 标题分割")
    
    # query 命令：查询知识库
    query_parser = subparsers.add_parser("query", help="查询知识库")
    query_parser.add_argument("--kb-id", required=True, help="知识库 ID")
    query_parser.add_argument("--question", required=True, help="问题")
    query_parser.add_argument("--top-k", type=int, default=4, help="检索的文档块数量")
    
    # stats 命令：查看统计信息
    stats_parser = subparsers.add_parser("stats", help="查看知识库统计信息")
    stats_parser.add_argument("--kb-id", required=True, help="知识库 ID")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # 执行命令
    if args.command == "ingest":
        if args.file:
            # 加载单个文档
            success = ingest_document(
                args.kb_id,
                args.file,
                use_markdown_header_split=not args.no_markdown_split,
            )
        elif args.dir:
            # 批量加载
            success = ingest_directory(
                args.kb_id,
                args.dir,
                pattern=args.pattern,
                use_markdown_header_split=not args.no_markdown_split,
            )
        else:
            print("❌ 请指定 --file 或 --dir 参数")
            return 1
        
        return 0 if success else 1
    
    elif args.command == "query":
        success = query_knowledge_base(
            args.kb_id,
            args.question,
            top_k=args.top_k,
        )
        return 0 if success else 1
    
    elif args.command == "stats":
        success = show_stats(args.kb_id)
        return 0 if success else 1
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

