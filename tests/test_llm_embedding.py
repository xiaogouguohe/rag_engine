#!/usr/bin/env python3
"""
测试脚本：验证 LLM 和 Embedding 调用是否正常工作

使用方法：
    python tests/test_llm_embedding.py
    或
    cd tests && python test_llm_embedding.py
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import AppConfig
from llm import LLMClient
from embedding import EmbeddingClient


def test_config_loading():
    """测试配置加载"""
    print("=" * 60)
    print("测试 1: 配置加载")
    print("=" * 60)
    
    try:
        config = AppConfig.load()
        print("✅ 配置加载成功！")
        print(f"   LLM Base URL: {config.llm.base_url}")
        print(f"   LLM Model: {config.llm.model}")
        print(f"   Embedding Base URL: {config.embedding.base_url}")
        print(f"   Embedding Model: {config.embedding.model}")
        return config
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        print("\n提示：")
        print("  1. 确保已创建 .env 文件（复制 .env.example 为 .env）")
        print("  2. 或在 .env 文件中设置 RAG_LLM_API_KEY")
        print("  3. 或设置系统环境变量 RAG_LLM_API_KEY")
        sys.exit(1)


def test_llm(config: AppConfig):
    """测试 LLM 调用"""
    print("\n" + "=" * 60)
    print("测试 2: LLM 调用")
    print("=" * 60)
    
    try:
        llm = LLMClient.from_config(config)
        print("✅ LLM 客户端创建成功")
        
        # 测试简单问题
        test_prompt = "用一句话解释什么是 RAG？"
        print(f"\n📝 测试问题: {test_prompt}")
        print("⏳ 正在调用 LLM...")
        
        response = llm.generate(
            prompt=test_prompt,
            system_prompt="你是一个专业的 AI 助手。",
            temperature=0.1,
        )
        
        print("✅ LLM 调用成功！")
        print(f"\n📤 LLM 回答:\n{response}\n")
        return True
        
    except Exception as e:
        print(f"❌ LLM 调用失败: {e}")
        print("\n可能的原因：")
        print("  1. API Key 无效或过期")
        print("  2. base_url 配置错误")
        print("  3. 模型名称不正确")
        print("  4. 网络连接问题")
        return False


def test_embedding(config: AppConfig):
    """测试 Embedding 调用"""
    print("\n" + "=" * 60)
    print("测试 3: Embedding 调用")
    print("=" * 60)
    
    try:
        emb = EmbeddingClient.from_config(config)
        print("✅ Embedding 客户端创建成功")
        
        # 测试文本向量化
        test_texts = [
            "RAG 是一种检索增强生成技术",
            "它结合了信息检索和生成模型",
            "可以提高大模型回答的准确性"
        ]
        print(f"\n📝 测试文本数量: {len(test_texts)}")
        print("⏳ 正在调用 Embedding...")
        
        vectors = emb.embed_texts(test_texts)
        
        print("✅ Embedding 调用成功！")
        print(f"   向量数量: {len(vectors)}")
        print(f"   向量维度: {len(vectors[0]) if vectors else 0}")
        print(f"   前 5 个维度值（示例）: {vectors[0][:5] if vectors else []}")
        return True
        
    except Exception as e:
        print(f"❌ Embedding 调用失败: {e}")
        print("\n可能的原因：")
        print("  1. API Key 无效或过期")
        print("  2. base_url 配置错误")
        print("  3. 模型名称不正确")
        print("  4. 网络连接问题")
        return False


def test_async_llm(config: AppConfig):
    """测试异步 LLM 调用（可选）"""
    print("\n" + "=" * 60)
    print("测试 4: 异步 LLM 调用（可选）")
    print("=" * 60)
    
    try:
        import asyncio
        llm = LLMClient.from_config(config)
        
        async def test():
            response = await llm.async_generate(
                prompt="用一句话说明异步调用的优势",
                temperature=0.1,
            )
            return response
        
        print("⏳ 正在异步调用 LLM...")
        response = asyncio.run(test())
        
        print("✅ 异步 LLM 调用成功！")
        print(f"\n📤 LLM 回答:\n{response}\n")
        return True
        
    except Exception as e:
        print(f"⚠️  异步调用测试失败（非必需）: {e}")
        return False


def main():
    """主测试函数"""
    print("\n" + "🚀 RAG 引擎 - LLM 和 Embedding 测试" + "\n")
    
    # 测试配置加载
    config = test_config_loading()
    
    # 测试 LLM
    llm_ok = test_llm(config)
    
    # 测试 Embedding
    emb_ok = test_embedding(config)
    
    # 测试异步 LLM（可选）
    async_ok = test_async_llm(config)
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"配置加载: {'✅ 通过' if config else '❌ 失败'}")
    print(f"LLM 调用: {'✅ 通过' if llm_ok else '❌ 失败'}")
    print(f"Embedding 调用: {'✅ 通过' if emb_ok else '❌ 失败'}")
    print(f"异步 LLM 调用: {'✅ 通过' if async_ok else '⚠️  跳过'}")
    
    if llm_ok and emb_ok:
        print("\n🎉 恭喜！所有核心功能测试通过，可以继续开发后续模块了！")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查配置和网络连接。")
        return 1


if __name__ == "__main__":
    sys.exit(main())

