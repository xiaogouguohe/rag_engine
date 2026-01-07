#!/usr/bin/env python3
"""
从 GitHub 加载菜谱知识库
-----------------------

支持两种方式：
1. 从本地已 clone 的目录加载（推荐）
2. 自动 clone GitHub 仓库后加载

使用方法：
    # 方式一：从本地目录加载（需要先 clone）
    python scripts/load_github_recipes.py --kb-id recipes_kb --local-dir /path/to/HowToCook/dishes

    # 方式二：自动 clone 并加载
    python scripts/load_github_recipes.py --kb-id recipes_kb --github-url https://github.com/Anduin2017/HowToCook --subdir dishes
"""

import sys
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag import RAGEngine


def clone_github_repo(repo_url: str, target_dir: Path) -> bool:
    """
    克隆 GitHub 仓库到本地目录。
    
    Args:
        repo_url: GitHub 仓库 URL（如 https://github.com/Anduin2017/HowToCook）
        target_dir: 目标目录
    
    Returns:
        是否成功
    """
    print(f"正在克隆仓库: {repo_url}")
    print(f"目标目录: {target_dir}")
    print("-" * 60)
    
    try:
        # 如果目录已存在，先删除
        if target_dir.exists():
            print(f"目录已存在，正在删除: {target_dir}")
            shutil.rmtree(target_dir)
        
        # 执行 git clone
        result = subprocess.run(
            ["git", "clone", repo_url, str(target_dir)],
            capture_output=True,
            text=True,
        )
        
        if result.returncode == 0:
            print("✅ 仓库克隆成功")
            return True
        else:
            print(f"❌ 仓库克隆失败: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("❌ 错误: 未找到 git 命令，请先安装 Git")
        return False
    except Exception as e:
        print(f"❌ 克隆失败: {e}")
        return False


def find_markdown_files(directory: Path) -> list[Path]:
    """
    递归查找目录中的所有 Markdown 文件。
    
    Args:
        directory: 目录路径
    
    Returns:
        Markdown 文件路径列表
    """
    md_files = []
    
    # 支持的 Markdown 扩展名
    md_extensions = {".md", ".markdown", ".mdx"}
    
    for file_path in directory.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in md_extensions:
            md_files.append(file_path)
    
    return sorted(md_files)


def load_recipes_from_directory(
    kb_id: str,
    dishes_dir: Path,
    use_markdown_header_split: bool = True,
) -> bool:
    """
    从本地目录加载所有菜谱文档。
    
    Args:
        kb_id: 知识库 ID
        dishes_dir: dishes 目录路径
        use_markdown_header_split: 是否使用 Markdown 标题分割
    
    Returns:
        是否成功
    """
    if not dishes_dir.exists():
        print(f"❌ 目录不存在: {dishes_dir}")
        return False
    
    if not dishes_dir.is_dir():
        print(f"❌ 不是目录: {dishes_dir}")
        return False
    
    # 查找所有 Markdown 文件
    print(f"正在扫描目录: {dishes_dir}")
    md_files = find_markdown_files(dishes_dir)
    
    if not md_files:
        print(f"❌ 未找到 Markdown 文件")
        return False
    
    print(f"找到 {len(md_files)} 个 Markdown 文件")
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
    success_count = 0
    fail_count = 0
    
    for i, file_path in enumerate(md_files, 1):
        # 显示相对路径（更清晰）
        try:
            rel_path = file_path.relative_to(dishes_dir)
        except ValueError:
            rel_path = file_path.name
        
        print(f"\n[{i}/{len(md_files)}] 处理: {rel_path}")
        
        try:
            result = engine.ingest_document(file_path)
            print(f"  ✅ 成功 - 分块数: {result['chunks_count']}")
            success_count += 1
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            fail_count += 1
    
    # 显示统计信息
    print("\n" + "=" * 60)
    print("批量加载完成")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print(f"  总计: {len(md_files)}")
    
    if success_count > 0:
        print("\n查看知识库统计信息:")
        try:
            stats = engine.get_stats()
            print(f"  向量数量: {stats.get('vector_count', 0)}")
            print(f"  文档数量: {stats.get('document_count', 0)}")
        except Exception as e:
            print(f"  获取统计信息失败: {e}")
    
    return fail_count == 0


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="从 GitHub 加载菜谱知识库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument("--kb-id", required=True, help="知识库 ID")
    parser.add_argument(
        "--local-dir",
        help="本地 dishes 目录路径（如果已 clone 到本地）",
    )
    parser.add_argument(
        "--github-url",
        default="https://github.com/Anduin2017/HowToCook",
        help="GitHub 仓库 URL（默认: HowToCook）",
    )
    parser.add_argument(
        "--subdir",
        default="dishes",
        help="子目录名称（默认: dishes）",
    )
    parser.add_argument(
        "--clone-dir",
        help="临时 clone 目录（如果不指定，会在临时目录中 clone）",
    )
    parser.add_argument(
        "--no-markdown-split",
        action="store_true",
        help="禁用 Markdown 标题分割",
    )
    parser.add_argument(
        "--keep-clone",
        action="store_true",
        help="保留 clone 的目录（默认会自动删除）",
    )
    
    args = parser.parse_args()
    
    # 方式一：从本地目录加载
    if args.local_dir:
        dishes_dir = Path(args.local_dir)
        if not dishes_dir.exists():
            print(f"❌ 目录不存在: {dishes_dir}")
            return 1
        
        success = load_recipes_from_directory(
            args.kb_id,
            dishes_dir,
            use_markdown_header_split=not args.no_markdown_split,
        )
        return 0 if success else 1
    
    # 方式二：自动 clone 并加载
    print("=" * 60)
    print("从 GitHub 自动加载菜谱知识库")
    print("=" * 60)
    
    # 确定 clone 目录
    if args.clone_dir:
        clone_dir = Path(args.clone_dir)
    else:
        # 使用临时目录
        temp_base = Path(tempfile.gettempdir())
        clone_dir = temp_base / "HowToCook_rag_engine"
    
    # 克隆仓库
    if not clone_github_repo(args.github_url, clone_dir):
        return 1
    
    # 确定 dishes 目录
    dishes_dir = clone_dir / args.subdir
    
    if not dishes_dir.exists():
        print(f"❌ 子目录不存在: {dishes_dir}")
        if not args.keep_clone:
            print(f"正在删除临时目录: {clone_dir}")
            shutil.rmtree(clone_dir, ignore_errors=True)
        return 1
    
    # 加载文档
    try:
        success = load_recipes_from_directory(
            args.kb_id,
            dishes_dir,
            use_markdown_header_split=not args.no_markdown_split,
        )
        
        # 清理临时目录
        if not args.keep_clone and not args.clone_dir:
            print(f"\n正在删除临时目录: {clone_dir}")
            shutil.rmtree(clone_dir, ignore_errors=True)
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 清理临时目录
        if not args.keep_clone and not args.clone_dir:
            print(f"\n正在删除临时目录: {clone_dir}")
            shutil.rmtree(clone_dir, ignore_errors=True)
        
        return 1


if __name__ == "__main__":
    sys.exit(main())

