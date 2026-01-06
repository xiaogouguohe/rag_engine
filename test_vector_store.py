#!/usr/bin/env python3
"""
测试脚本：验证向量存储功能

使用方法：
    python test_vector_store.py
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import AppConfig
from embedding_client import EmbeddingClient
from vector_store import VectorStore


def test_vector_store():
    """测试向量存储功能"""
    print("=" * 60)
    print("向量存储功能测试")
    print("=" * 60)
    
    # 1. 加载配置
    print("\n1. 加载配置...")
    try:
        config = AppConfig.load()
        print("✅ 配置加载成功")
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False
    
    # 2. 创建 Embedding 客户端
    print("\n2. 创建 Embedding 客户端...")
    try:
        emb = EmbeddingClient.from_config(config)
        print("✅ Embedding 客户端创建成功")
    except Exception as e:
        print(f"❌ Embedding 客户端创建失败: {e}")
        return False
    
    # 3. 创建向量存储
    print("\n3. 创建向量存储...")
    try:
        store = VectorStore(storage_path=config.storage_path)
        print(f"✅ 向量存储创建成功（路径: {config.storage_path}）")
        print(f"   使用 Milvus Lite（无需 Docker）")
    except Exception as e:
        print(f"❌ 向量存储创建失败: {e}")
        print("提示: 请确保已安装 pymilvus: pip install pymilvus[milvus_lite]")
        return False
    
    # 4. 准备测试数据
    print("\n4. 准备测试数据...")
    test_kb_id = "test_kb_001"
    test_texts = [
        "RAG（检索增强生成）是一种结合信息检索和生成模型的技术",
        "它通过检索相关文档来增强大语言模型的回答准确性",
        "向量数据库用于存储文档的向量表示，支持快速相似度搜索",
        "Milvus 是一个开源的向量数据库，支持大规模向量相似度搜索",
        "Milvus Lite 是 Milvus 的轻量版本，无需 Docker，作为 Python 库直接使用",
    ]
    print(f"   测试知识库 ID: {test_kb_id}")
    print(f"   测试文本数量: {len(test_texts)}")
    
    # 5. 生成向量
    print("\n5. 生成向量...")
    try:
        vectors = emb.embed_texts(test_texts)
        print(f"✅ 向量生成成功（数量: {len(vectors)}, 维度: {len(vectors[0])}）")
    except Exception as e:
        print(f"❌ 向量生成失败: {e}")
        return False
    
    # 6. 添加到向量存储
    print("\n6. 添加向量到存储...")
    try:
        chunk_ids = store.add_texts(
            kb_id=test_kb_id,
            texts=test_texts,
            vectors=vectors,
            metadatas=[
                {"doc_id": f"doc_{i}", "position": i} for i in range(len(test_texts))
            ],
        )
        print(f"✅ 向量添加成功（chunk IDs: {len(chunk_ids)}）")
    except Exception as e:
        print(f"❌ 向量添加失败: {e}")
        return False
    
    # 7. 测试搜索
    print("\n7. 测试向量搜索...")
    test_queries = [
        "什么是 RAG？",
        "向量数据库有哪些？",
    ]
    
    for query_text in test_queries:
        print(f"\n   查询: {query_text}")
        try:
            # 生成查询向量
            query_vectors = emb.embed_texts([query_text])
            query_vector = query_vectors[0]
            
            # 搜索
            results = store.search(
                kb_id=test_kb_id,
                query_vector=query_vector,
                top_k=3,
            )
            
            print(f"   ✅ 搜索成功，找到 {len(results)} 个结果:")
            for i, (score, metadata) in enumerate(results, 1):
                print(f"      {i}. [相似度: {score:.4f}] {metadata.text[:50]}...")
                
        except Exception as e:
            print(f"   ❌ 搜索失败: {e}")
            return False
    
    # 8. 测试统计信息
    print("\n8. 测试统计信息...")
    try:
        stats = store.get_stats(test_kb_id)
        print(f"✅ 统计信息: {stats}")
    except Exception as e:
        print(f"❌ 获取统计信息失败: {e}")
    
    # 9. 测试持久化（重新加载）
    print("\n9. 测试持久化（重新创建 VectorStore 实例）...")
    try:
        # 创建新的 VectorStore 实例（模拟重启）
        store2 = VectorStore(storage_path=config.storage_path)
        results = store2.search(
            kb_id=test_kb_id,
            query_vector=emb.embed_texts(["RAG"])[0],
            top_k=1,
        )
        if results:
            print(f"✅ 持久化测试成功，可以检索到 {len(results)} 个结果")
        else:
            print("⚠️  持久化测试：未检索到结果（可能是索引未正确保存）")
    except Exception as e:
        print(f"❌ 持久化测试失败: {e}")
    
    print("\n" + "=" * 60)
    print("✅ 向量存储功能测试完成！")
    print("=" * 60)
    
    return True


def main():
    """主函数"""
    print("\n🚀 RAG 引擎 - 向量存储测试\n")
    
    success = test_vector_store()
    
    if success:
        print("\n🎉 向量存储功能正常，可以继续开发后续模块！")
        return 0
    else:
        print("\n⚠️  测试失败，请检查错误信息。")
        return 1


if __name__ == "__main__":
    sys.exit(main())

