#!/usr/bin/env python3
"""
调试 RAGAS 调用 LLM API 时的参数（方法 2：使用 LangChain 回调）
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import json
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from ragas.llms import LangchainLLMWrapper
from ragas.metrics._answer_relevance import ResponseRelevanceInput, ResponseRelevancePrompt
from config.config import AppConfig

class APICallDebugHandler(BaseCallbackHandler):
    """回调处理器，用于记录 API 调用"""
    
    def __init__(self):
        self.api_calls = []
        self.current_call = {}
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        """LLM 开始调用时"""
        print("\n" + "=" * 60)
        print("🔍 LLM 开始调用")
        print("=" * 60)
        print(f"📋 serialized: {serialized}")
        print(f"📋 prompts 数量: {len(prompts)}")
        
        self.current_call = {
            "event": "llm_start",
            "prompts": prompts,
            "kwargs": kwargs,
        }
        
        # 检查每个 prompt
        for i, prompt in enumerate(prompts):
            print(f"\n📝 Prompt {i+1}:")
            print(f"  - 类型: {type(prompt)}")
            print(f"  - 值: {repr(str(prompt)[:200])}")
            
            if isinstance(prompt, list):
                print(f"  - 是列表，长度: {len(prompt)}")
                for j, item in enumerate(prompt):
                    print(f"    项目 {j}: 类型={type(item)}, 值={repr(str(item)[:100])}")
    
    def on_llm_end(self, response, **kwargs):
        """LLM 调用结束时"""
        print("\n" + "=" * 60)
        print("✅ LLM 调用结束")
        print("=" * 60)
        print(f"📋 response 类型: {type(response)}")
        
        self.current_call["event"] = "llm_end"
        self.current_call["response"] = str(response)[:500]  # 只保存前500字符
        self.api_calls.append(self.current_call.copy())
    
    def on_llm_error(self, error, **kwargs):
        """LLM 调用出错时"""
        print("\n" + "=" * 60)
        print("❌ LLM 调用出错")
        print("=" * 60)
        print(f"📋 error: {error}")
        print(f"📋 error 类型: {type(error)}")
        
        self.current_call["event"] = "llm_error"
        self.current_call["error"] = str(error)
        self.api_calls.append(self.current_call.copy())

def debug_with_callbacks():
    """使用回调函数调试"""
    print("=" * 60)
    print("调试 RAGAS 调用 LLM API（使用回调函数）")
    print("=" * 60)
    print()
    
    # 加载配置
    app_config = AppConfig.load()
    
    # 创建回调处理器
    debug_handler = APICallDebugHandler()
    
    # 创建 LangChain LLM
    print("1. 创建 LangChain LLM（带回调）...")
    langchain_llm = ChatOpenAI(
        model=app_config.llm.model,
        api_key=app_config.llm.api_key,
        base_url=app_config.llm.base_url,
        temperature=0.1,
        timeout=120.0,
        max_retries=3,
        callbacks=[debug_handler],
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
    
    import asyncio
    async def test_generate():
        try:
            result = await prompt.generate_multiple(
                llm=ragas_llm,
                data=prompt_input,
                n=1,
                callbacks=[debug_handler],
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
    
    # 保存调用记录（清理不可序列化的对象）
    print("6. 保存 API 调用记录...")
    output_file = Path("ragas_api_calls_callbacks.json")
    
    # 清理不可序列化的对象
    cleaned_calls = []
    for call in debug_handler.api_calls:
        cleaned_call = {}
        for key, value in call.items():
            try:
                json.dumps(value)  # 测试是否可以序列化
                cleaned_call[key] = value
            except (TypeError, ValueError):
                cleaned_call[key] = str(value)  # 转换为字符串
        cleaned_calls.append(cleaned_call)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_calls, f, ensure_ascii=False, indent=2)
    print(f"   ✅ 调用记录已保存到: {output_file}")
    print()
    
    print("=" * 60)
    print("调试完成")
    print("=" * 60)

if __name__ == "__main__":
    debug_with_callbacks()

