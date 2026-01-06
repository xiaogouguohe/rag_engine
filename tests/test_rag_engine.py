#!/usr/bin/env python3
"""
测试脚本：验证完整的 RAG 流程

使用方法：
    python tests/test_rag_engine.py
"""

import sys
from pathlib import Path
import tempfile
import os

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag import RAGEngine


def test_rag_engine():
    """测试完整的 RAG 流程"""
    print("=" * 60)
    print("RAG 引擎完整流程测试")
    print("=" * 60)
    
    # 1. 初始化引擎
    print("\n1. 初始化 RAG 引擎...")
    try:
        engine = RAGEngine(
            kb_id="test_rag_kb_001",
            chunk_size=100,  # 小一点方便测试
            chunk_overlap=10,
        )
        print("✅ RAG 引擎初始化成功")
    except Exception as e:
        print(f"❌ RAG 引擎初始化失败: {e}")
        print("提示: 请确保已配置 .env 文件或环境变量")
        return False
    
    # 2. 创建测试文档
    print("\n2. 创建测试文档...")
    test_content = """RAG（检索增强生成）是一种结合信息检索和生成模型的技术。

它通过检索相关文档来增强大语言模型的回答准确性。

向量数据库用于存储文档的向量表示，支持快速相似度搜索。

FAISS 是 Facebook 开源的向量相似度搜索库。

Milvus 是一个开源的向量数据库，支持大规模向量相似度搜索。"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(test_content)
        temp_file = f.name
    
    try:
        # 3. 处理文档（解析 → 分块 → 向量化 → 存储）
        print("\n3. 处理文档（解析 → 分块 → 向量化 → 存储）...")
        result = engine.ingest_document(temp_file)
        
        print("✅ 文档处理成功！")
        print(f"   文档 ID: {result['doc_id']}")
        print(f"   分块数量: {result['chunks_count']}")
        
        # 4. 查看统计信息
        print("\n4. 查看知识库统计信息...")
        stats = engine.get_stats()
        print(f"   向量数量: {stats.get('vector_count', 0)}")
        
        # 5. 测试问答
        print("\n5. 测试问答流程...")
        test_questions = [
            "什么是 RAG？",
            "向量数据库有什么作用？",
        ]
        
        for question in test_questions:
            print(f"\n   问题: {question}")
            try:
                answer_result = engine.query(question, top_k=3)
                
                print(f"   ✅ 回答生成成功")
                print(f"   回答: {answer_result['answer'][:100]}...")
                print(f"   检索到 {len(answer_result['sources'])} 个相关片段")
                
            except Exception as e:
                print(f"   ❌ 问答失败: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        print("\n" + "=" * 60)
        print("✅ RAG 引擎完整流程测试成功！")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理临时文件
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def main():
    """主函数"""
    print("\n🚀 RAG 引擎 - 完整流程测试\n")
    
    success = test_rag_engine()
    
    if success:
        print("\n🎉 所有测试通过！RAG 引擎功能正常。")
        return 0
    else:
        print("\n⚠️  测试失败，请检查错误信息。")
        return 1


if __name__ == "__main__":
    sys.exit(main())

