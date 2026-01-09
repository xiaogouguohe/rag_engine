#!/usr/bin/env python3
"""
RAG 引擎主入口
-------------

主要功能：从配置文件加载知识库

支持两种配置方式：
1. 使用 JSON 配置文件（knowledge_bases.json）
2. 使用环境变量（.env 文件）

使用方法：
    # 从配置文件加载所有知识库（主要功能）
    python main.py

    # 从配置文件加载指定的知识库
    python main.py --kb-id recipes_kb

    # 使用自定义配置文件
    python main.py --config custom_config.json

    # 其他辅助功能
    python main.py query --kb-id my_kb --question "你的问题"
    python main.py stats --kb-id my_kb
"""

import sys
import argparse
import json
from pathlib import Path
from typing import List, Optional

from rag import RAGEngine
from config import AppConfig, KnowledgeBaseConfig


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
    
    import time
    total_start = time.time()
    
    for i, file_path in enumerate(files, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(files)}] 处理文档: {file_path.name}")
        print(f"{'='*60}")
        
        file_start = time.time()
        try:
            result = engine.ingest_document(file_path, verbose=True)
            file_time = time.time() - file_start
            print(f"\n✅ 文档处理成功 - 分块数: {result['chunks_count']}, 总耗时: {file_time:.2f} 秒")
            success_count += 1
        except Exception as e:
            file_time = time.time() - file_start
            print(f"\n❌ 文档处理失败，耗时: {file_time:.2f} 秒")
            print(f"   错误: {e}")
            import traceback
            traceback.print_exc()
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


def load_config_from_json(config_path: str) -> List[KnowledgeBaseConfig]:
    """
    从 JSON 配置文件加载知识库配置。
    
    Args:
        config_path: JSON 配置文件路径
    
    Returns:
        知识库配置列表
    """
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


def find_markdown_files(directory: Path, pattern: str = "*.md") -> List[Path]:
    """
    递归查找目录中匹配模式的文件。
    
    Args:
        directory: 目录路径
        pattern: 文件匹配模式（默认：*.md）
    
    Returns:
        文件路径列表
    """
    files = []
    for file_path in directory.rglob(pattern):
        if file_path.is_file():
            files.append(file_path)
    return sorted(files)


def load_knowledge_base(kb_config: KnowledgeBaseConfig, verbose: bool = True):
    """
    加载单个知识库。
    
    Args:
        kb_config: 知识库配置
        verbose: 是否显示详细信息
    """
    source_path = Path(kb_config.source_path)
    
    if not source_path.exists():
        print(f"❌ 路径不存在: {source_path}")
        return False
    
    if not source_path.is_dir():
        print(f"❌ 不是目录: {source_path}")
        return False
    
    if verbose:
        print("=" * 60)
        print(f"加载知识库: {kb_config.kb_id}")
        print("=" * 60)
        print(f"源路径: {source_path}")
        print(f"文件模式: {kb_config.file_pattern}")
        print("-" * 60)
    
    # 查找文件
    if verbose:
        print("正在扫描文件...")
    files = find_markdown_files(source_path, kb_config.file_pattern)
    
    if not files:
        print(f"❌ 未找到匹配的文件（模式: {kb_config.file_pattern}）")
        return False
    
    if verbose:
        print(f"✅ 找到 {len(files)} 个文件")
        print("-" * 60)
    
    # 初始化 RAG 引擎
    if verbose:
        print("初始化 RAG 引擎...")
    engine = RAGEngine(
        kb_id=kb_config.kb_id,
        use_markdown_header_split=kb_config.use_markdown_header_split,
    )
    if verbose:
        print("✅ RAG 引擎初始化成功")
        print("-" * 60)
    
    # 批量加载文档
    if verbose:
        print(f"\n开始加载文档...\n")
    success_count = 0
    fail_count = 0
    
    for i, file_path in enumerate(files, 1):
        try:
            rel_path = file_path.relative_to(source_path)
        except ValueError:
            rel_path = file_path.name
        
        if verbose:
            print(f"[{i}/{len(files)}] {rel_path}", end=" ... ")
        
        try:
            result = engine.ingest_document(file_path)
            if verbose:
                print(f"✅ 成功 ({result['chunks_count']} 块)")
            success_count += 1
        except Exception as e:
            if verbose:
                print(f"❌ 失败: {e}")
            fail_count += 1
    
    # 显示统计信息
    if verbose:
        print("\n" + "=" * 60)
        print("加载完成")
        print("=" * 60)
        print(f"  成功: {success_count}")
        print(f"  失败: {fail_count}")
        print(f"  总计: {len(files)}")
        
        if success_count > 0:
            print("\n知识库统计信息:")
            try:
                stats = engine.get_stats()
                print(f"  向量数量: {stats.get('vector_count', 0)}")
                print(f"  文档数量: {stats.get('document_count', 0)}")
            except Exception as e:
                print(f"  获取统计信息失败: {e}")
    
    return fail_count == 0


def load_all_knowledge_bases(config_path: str = "knowledge_bases.json", kb_id: Optional[str] = None, quiet: bool = False):
    """
    从配置文件加载所有知识库（主要功能）。
    
    Args:
        config_path: JSON 配置文件路径
        kb_id: 只加载指定的知识库 ID（如果不指定，加载所有）
        quiet: 静默模式（不显示详细信息）
    """
    # 加载知识库配置
    kb_configs = []
    
    # 方式一：从 JSON 配置文件加载
    config_file = Path(config_path)
    if config_file.exists():
        try:
            kb_configs = load_config_from_json(config_path)
            if not quiet:
                print(f"✅ 从配置文件加载: {config_path}")
        except Exception as e:
            print(f"❌ 加载配置文件失败: {e}")
            return False
    else:
        # 方式二：从环境变量加载
        try:
            app_config = AppConfig.load()
            if app_config.knowledge_bases:
                kb_configs = app_config.knowledge_bases
                if not quiet:
                    print("✅ 从环境变量加载知识库配置")
            else:
                print("❌ 未找到知识库配置（请检查 knowledge_bases.json 或环境变量）")
                return False
        except Exception as e:
            print(f"❌ 加载配置失败: {e}")
            return False
    
    if not kb_configs:
        print("❌ 未找到任何知识库配置")
        return False
    
    # 过滤指定的知识库
    if kb_id:
        kb_configs = [kb for kb in kb_configs if kb.kb_id == kb_id]
        if not kb_configs:
            print(f"❌ 未找到知识库: {kb_id}")
            return False
    
    # 加载知识库
    all_success = True
    for kb_config in kb_configs:
        success = load_knowledge_base(kb_config, verbose=not quiet)
        if not success:
            all_success = False
        if not quiet and len(kb_configs) > 1:
            print()  # 空行分隔
    
    if all_success:
        if not quiet:
            print("✅ 所有知识库加载完成！")
        return True
    else:
        if not quiet:
            print("⚠️  部分知识库加载失败")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="RAG 引擎主入口（主要功能：加载知识库）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # 主功能：加载知识库（默认行为）
    parser.add_argument(
        "--config",
        default="knowledge_bases.json",
        help="JSON 配置文件路径（默认: knowledge_bases.json）",
    )
    parser.add_argument(
        "--kb-id",
        help="只加载指定的知识库 ID（如果不指定，加载所有）",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式（不显示详细信息）",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="其他命令（可选）")
    
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
    
    # 如果没有指定命令，默认执行加载知识库（主要功能）
    if not args.command:
        return 0 if load_all_knowledge_bases(
            config_path=args.config,
            kb_id=args.kb_id,
            quiet=args.quiet,
        ) else 1
    
    # 执行其他命令
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

