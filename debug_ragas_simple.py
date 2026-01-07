#!/usr/bin/env python3
"""
最简单的调试方法：启用 LangChain 的详细日志
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import logging
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 启用详细日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 启用 LangChain 和 OpenAI 的日志
logging.getLogger('langchain').setLevel(logging.DEBUG)
logging.getLogger('openai').setLevel(logging.DEBUG)
logging.getLogger('httpx').setLevel(logging.DEBUG)

from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.metrics._answer_relevance import ResponseRelevanceInput, ResponseRelevancePrompt
from config.config import AppConfig

def debug_with_logging():
    """使用日志调试"""
    print("=" * 60)
    print("调试 RAGAS 调用 LLM API（使用详细日志）")
    print("=" * 60)
    print()
    print("💡 所有 API 调用都会显示在日志中")
    print()
    
    # 加载配置
    app_config = AppConfig.load()
    
    # 创建 LangChain LLM
    print("1. 创建 LangChain LLM...")
    langchain_llm = ChatOpenAI(
        model=app_config.llm.model,
        api_key=app_config.llm.api_key,
        base_url=app_config.llm.base_url,
        temperature=0.1,
        timeout=120.0,
        max_retries=3,
    )
    print(f"   ✅ LangChain LLM 创建成功")
    print()
    
    # 创建 RAGAS LLM Wrapper
    print("2. 创建 RAGAS LLM Wrapper...")
    ragas_llm = LangchainLLMWrapper(langchain_llm)
    print(f"   ✅ RAGAS LLM Wrapper 创建成功")
    print()
    
    # 创建测试数据
    print("3. 创建测试数据...")
    test_answer = "根据文档片段，可以按照以下步骤用咖喱烹饪青蟹。"
    prompt_input = ResponseRelevanceInput(response=test_answer)
    print(f"   ✅ 测试数据创建成功")
    print()
    
    # 创建 Prompt
    print("4. 创建 Prompt...")
    prompt = ResponseRelevancePrompt()
    print(f"   ✅ Prompt 创建成功")
    print()
    
    # 测试生成
    print("5. 测试生成问题（strictness=1）...")
    print("-" * 60)
    print("📋 下面的日志会显示所有 API 调用的详细信息")
    print()
    
    import asyncio
    async def test_generate():
        try:
            result = await prompt.generate_multiple(
                llm=ragas_llm,
                data=prompt_input,
                n=1,
            )
            print(f"\n✅ 生成成功，结果数量: {len(result)}")
            return result
        except Exception as e:
            print(f"\n❌ 生成失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    result = asyncio.run(test_generate())
    print()
    
    print("=" * 60)
    print("调试完成")
    print("=" * 60)
    print()
    print("💡 查看上面的日志，可以看到：")
    print("  - HTTP 请求的 URL")
    print("  - 请求的 JSON 参数")
    print("  - messages 的内容")
    print("  - 响应内容")

if __name__ == "__main__":
    debug_with_logging()

