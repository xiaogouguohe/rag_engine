#!/usr/bin/env python3
"""
测试使用 llm_factory 直接访问 LLM API（不使用 LangChain）
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from openai import OpenAI
from ragas.llms import llm_factory
from ragas.metrics._answer_relevance import ResponseRelevanceInput, ResponseRelevancePrompt
from config.config import AppConfig

def test_llm_factory():
    """测试使用 llm_factory"""
    print("=" * 60)
    print("测试使用 llm_factory（不使用 LangChain）")
    print("=" * 60)
    print()
    
    # 加载配置
    app_config = AppConfig.load()
    
    # 方法 1: 使用 llm_factory（推荐，不使用 LangChain）
    print("1. 使用 llm_factory 创建 LLM...")
    print("-" * 60)
    
    # 创建 OpenAI 客户端（直接使用 OpenAI SDK）
    openai_client = OpenAI(
        api_key=app_config.llm.api_key,
        base_url=app_config.llm.base_url,
    )
    
    # 使用 llm_factory 创建 RAGAS LLM
    ragas_llm = llm_factory(
        model=app_config.llm.model,
        provider="openai",
        client=openai_client,
    )
    
    print(f"   ✅ LLM 创建成功")
    print(f"   - 模型: {app_config.llm.model}")
    print(f"   - 类型: {type(ragas_llm)}")
    print(f"   - 不使用 LangChain: ✅")
    print()
    
    # 创建测试数据
    print("2. 创建测试数据...")
    test_answer = "根据文档片段，可以按照以下步骤用咖喱烹饪青蟹。"
    prompt_input = ResponseRelevanceInput(response=test_answer)
    print(f"   ✅ 测试数据创建成功")
    print()
    
    # 创建 Prompt
    print("3. 创建 Prompt...")
    prompt = ResponseRelevancePrompt()
    print(f"   ✅ Prompt 创建成功")
    print()
    
    # 测试生成
    print("4. 测试生成问题...")
    print("-" * 60)
    
    import asyncio
    async def test_generate():
        try:
            result = await prompt.generate_multiple(
                llm=ragas_llm,
                data=prompt_input,
                n=1,
            )
            print(f"\n✅ 生成成功，结果数量: {len(result)}")
            for i, r in enumerate(result, 1):
                print(f"   结果 {i}: {r.question}")
            return result
        except Exception as e:
            print(f"\n❌ 生成失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    result = asyncio.run(test_generate())
    print()
    
    print("=" * 60)
    print("测试完成")
    print("=" * 60)
    print()
    print("💡 总结:")
    print("  - ✅ 可以使用 llm_factory 直接访问 LLM API")
    print("  - ✅ 不需要 LangChain")
    print("  - ✅ 这是 RAGAS 推荐的方式（LangchainLLMWrapper 已废弃）")

if __name__ == "__main__":
    test_llm_factory()

