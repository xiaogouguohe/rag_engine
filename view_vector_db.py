#!/usr/bin/env python3
"""
查看向量数据库中已加载的知识
----------------------------

使用方法：
    # 查看所有知识库
    python view_vector_db.py

    # 查看指定知识库的详细信息
    python view_vector_db.py --kb-id recipes_kb

    # 查看指定知识库的文档列表
    python view_vector_db.py --kb-id recipes_kb --list-docs
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import AppConfig
from vector_store import VectorStore


def list_all_knowledge_bases():
    """列出所有知识库"""
    try:
        app_config = AppConfig.load()
        vector_store = VectorStore(storage_path=app_config.storage_path)
        
        kb_list = vector_store.list_all_knowledge_bases()
        
        if not kb_list:
            print("📭 向量数据库中没有任何知识库")
            return
        
        print("=" * 80)
        print("向量数据库中的知识库列表")
        print("=" * 80)
        print()
        
        for kb in kb_list:
            print(f"📚 知识库 ID: {kb['kb_id']}")
            print(f"   Collection 名称: {kb['collection_name']}")
            print(f"   向量数量: {kb['vector_count']}")
            if kb.get('vector_dim'):
                print(f"   向量维度: {kb['vector_dim']}")
            print()
        
        print(f"总计: {len(kb_list)} 个知识库")
        
    except Exception as e:
        print(f"❌ 列出知识库失败: {e}")
        import traceback
        traceback.print_exc()


def show_kb_details(kb_id: str, list_docs: bool = False):
    """显示知识库的详细信息"""
    try:
        app_config = AppConfig.load()
        vector_store = VectorStore(storage_path=app_config.storage_path)
        
        # 获取统计信息
        stats = vector_store.get_stats(kb_id)
        
        print("=" * 80)
        print(f"知识库详情: {kb_id}")
        print("=" * 80)
        print()
        
        if stats['vector_count'] == 0:
            print("📭 该知识库为空（没有向量数据）")
            return
        
        print(f"📚 知识库 ID: {stats['kb_id']}")
        print(f"   Collection 名称: {stats['collection_name']}")
        print(f"   向量数量: {stats['vector_count']}")
        if stats.get('vector_dim'):
            print(f"   向量维度: {stats['vector_dim']}")
        print()
        
        if list_docs:
            print("-" * 80)
            print("文档列表")
            print("-" * 80)
            print()
            
            doc_list = vector_store.get_document_list(kb_id, limit=50)
            
            if not doc_list:
                print("📭 未找到文档")
            else:
                for i, doc in enumerate(doc_list, 1):
                    print(f"{i}. 文档 ID: {doc['doc_id']}")
                    print(f"   Chunks 数量: {doc['chunks_count']}")
                    print(f"   预览: {doc['first_chunk_preview']}")
                    print()
                
                if len(doc_list) >= 50:
                    print(f"... 还有更多文档（仅显示前 50 个）")
        
    except Exception as e:
        print(f"❌ 获取知识库详情失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="查看向量数据库中已加载的知识",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--kb-id",
        help="知识库 ID（如果指定，则显示该知识库的详细信息）",
    )
    parser.add_argument(
        "--list-docs",
        action="store_true",
        help="列出知识库中的文档（需要配合 --kb-id 使用）",
    )
    
    args = parser.parse_args()
    
    if args.kb_id:
        show_kb_details(args.kb_id, args.list_docs)
    else:
        list_all_knowledge_bases()


if __name__ == "__main__":
    main()

