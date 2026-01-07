#!/usr/bin/env python3
"""
加载菜谱知识库启动脚本
--------------------

从指定目录加载所有 .md 文件到知识库。

使用方法：
    python load_recipes.py --kb-id recipes_kb --dir /path/to/dishes
    python load_recipes.py --kb-id recipes_kb --dir ../HowToCook/dishes
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rag import RAGEngine


def find_markdown_files(directory: Path) -> list[Path]:
    """
    递归查找目录中的所有 .md 文件。
    
    Args:
        directory: 目录路径
    
    Returns:
        .md 文件路径列表
    """
    md_files = []
    
    # 只查找 .md 文件
    for file_path in directory.rglob("*.md"):
        if file_path.is_file():
            md_files.append(file_path)
    
    return sorted(md_files)


def load_recipes(kb_id: str, dishes_dir: str, use_markdown_header_split: bool = True):
    """
    从指定目录加载所有 .md 文件到知识库。
    
    Args:
        kb_id: 知识库 ID
        dishes_dir: dishes 目录路径
        use_markdown_header_split: 是否使用 Markdown 标题分割
    """
    dishes_path = Path(dishes_dir)
    
    if not dishes_path.exists():
        print(f"❌ 目录不存在: {dishes_path}")
        return False
    
    if not dishes_path.is_dir():
        print(f"❌ 不是目录: {dishes_path}")
        return False
    
    print("=" * 60)
    print("加载菜谱知识库")
    print("=" * 60)
    print(f"知识库 ID: {kb_id}")
    print(f"目录路径: {dishes_path}")
    print("-" * 60)
    
    # 查找所有 .md 文件
    print("正在扫描 .md 文件...")
    md_files = find_markdown_files(dishes_path)
    
    if not md_files:
        print(f"❌ 未找到 .md 文件")
        return False
    
    print(f"✅ 找到 {len(md_files)} 个 .md 文件")
    print("-" * 60)
    
    # 初始化 RAG 引擎
    print("初始化 RAG 引擎...")
    engine = RAGEngine(
        kb_id=kb_id,
        use_markdown_header_split=use_markdown_header_split,
    )
    print("✅ RAG 引擎初始化成功")
    print("-" * 60)
    
    # 批量加载文档
    print(f"\n开始加载文档...\n")
    success_count = 0
    fail_count = 0
    
    for i, file_path in enumerate(md_files, 1):
        # 显示相对路径（更清晰）
        try:
            rel_path = file_path.relative_to(dishes_path)
        except ValueError:
            rel_path = file_path.name
        
        print(f"[{i}/{len(md_files)}] {rel_path}", end=" ... ")
        
        try:
            result = engine.ingest_document(file_path)
            print(f"✅ 成功 ({result['chunks_count']} 块)")
            success_count += 1
        except Exception as e:
            print(f"❌ 失败: {e}")
            fail_count += 1
    
    # 显示统计信息
    print("\n" + "=" * 60)
    print("加载完成")
    print("=" * 60)
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print(f"  总计: {len(md_files)}")
    
    if success_count > 0:
        print("\n知识库统计信息:")
        try:
            stats = engine.get_stats()
            print(f"  向量数量: {stats.get('vector_count', 0)}")
            print(f"  文档数量: {stats.get('document_count', 0)}")
        except Exception as e:
            print(f"  获取统计信息失败: {e}")
    
    print("\n✅ 加载完成！现在可以查询知识库了：")
    print(f"   python main.py query --kb-id {kb_id} --question \"你的问题\"")
    
    return fail_count == 0


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="加载菜谱知识库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--kb-id",
        required=True,
        help="知识库 ID（如: recipes_kb）",
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="dishes 目录路径（如: ../HowToCook/dishes）",
    )
    parser.add_argument(
        "--no-markdown-split",
        action="store_true",
        help="禁用 Markdown 标题分割（默认启用）",
    )
    
    args = parser.parse_args()
    
    success = load_recipes(
        kb_id=args.kb_id,
        dishes_dir=args.dir,
        use_markdown_header_split=not args.no_markdown_split,
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

