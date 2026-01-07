#!/usr/bin/env python3
"""
调试 RAGAS 调用 LLM 的过程，找出为什么会有 InvalidParams 错误
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.metrics._answer_relevance import ResponseRelevanceInput, ResponseRelevancePrompt
from langchain_core.prompt_values import PromptValue
from config.config import AppConfig

def debug_ragas_llm_call():
    """调试 RAGAS 调用 LLM 的过程"""
    print("=" * 60)
    print("调试 RAGAS 调用 LLM 的过程")
    print("=" * 60)
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
    print(f"   ✅ LangChain LLM 创建成功: {langchain_llm.__class__.__name__}")
    print()
    
    # 创建 RAGAS LLM Wrapper
    print("2. 创建 RAGAS LLM Wrapper...")
    ragas_llm = LangchainLLMWrapper(langchain_llm)
    print(f"   ✅ RAGAS LLM Wrapper 创建成功: {ragas_llm.__class__.__name__}")
    print()
    
    # 创建测试数据
    print("3. 创建测试数据...")
    test_answer = "根据文档片段，可以按照以下步骤用咖喱烹饪青蟹。"
    prompt_input = ResponseRelevanceInput(response=test_answer)
    print(f"   ✅ 测试数据创建成功")
    print(f"      答案: {test_answer}")
    print()
    
    # 创建 Prompt
    print("4. 创建 Prompt...")
    prompt = ResponseRelevancePrompt()
    print(f"   ✅ Prompt 创建成功: {prompt.__class__.__name__}")
    print()
    
    # 测试 to_string
    print("5. 测试 to_string()...")
    try:
        prompt_string = prompt.to_string(prompt_input)
        print(f"   ✅ to_string() 成功")
        print(f"      类型: {type(prompt_string)}")
        print(f"      长度: {len(prompt_string)}")
        print(f"      内容（前200字符）: {prompt_string[:200]}")
        print(f"      是否为 None: {prompt_string is None}")
        print(f"      是否为空: {not prompt_string}")
    except Exception as e:
        print(f"   ❌ to_string() 失败: {e}")
        import traceback
        traceback.print_exc()
        return
    print()
    
    # 测试 PromptValue
    print("6. 测试 PromptValue...")
    try:
        prompt_value = PromptValue(text=prompt_string)
        print(f"   ✅ PromptValue 创建成功")
        print(f"      text 类型: {type(prompt_value.text)}")
        print(f"      text 值: {repr(prompt_value.text[:100]) if prompt_value.text else None}")
        print(f"      text 是否为 None: {prompt_value.text is None}")
    except Exception as e:
        print(f"   ❌ PromptValue 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    print()
    
    # 测试 to_messages
    print("7. 测试 PromptValue.to_messages()...")
    try:
        messages = prompt_value.to_messages()
        print(f"   ✅ to_messages() 成功")
        print(f"      消息数量: {len(messages)}")
        for i, msg in enumerate(messages):
            print(f"      消息 {i+1}:")
            print(f"        类型: {type(msg)}")
            print(f"        content 类型: {type(msg.content)}")
            print(f"        content 值: {repr(str(msg.content)[:100]) if msg.content else None}")
            print(f"        content 是否为 None: {msg.content is None}")
            if hasattr(msg, 'content') and msg.content is not None:
                if not isinstance(msg.content, str):
                    print(f"        ⚠️  content 不是字符串类型!")
                    print(f"        content 的实际类型: {type(msg.content)}")
                    if isinstance(msg.content, list):
                        print(f"        content 是列表，长度: {len(msg.content)}")
                        for j, item in enumerate(msg.content):
                            print(f"          项目 {j}: 类型={type(item)}, 值={repr(item)[:50]}")
    except Exception as e:
        print(f"   ❌ to_messages() 失败: {e}")
        import traceback
        traceback.print_exc()
        return
    print()
    
    # 测试 agenerate_prompt（不实际调用 API）
    print("8. 检查 agenerate_prompt 的参数...")
    print("   注意: 这里不实际调用 API，只检查参数格式")
    print(f"   prompts 类型: {type([prompt_value])}")
    print(f"   prompts 长度: {len([prompt_value])}")
    print(f"   prompts[0] 类型: {type([prompt_value][0])}")
    print(f"   prompts[0].text 类型: {type([prompt_value][0].text)}")
    print(f"   prompts[0].text 值: {repr([prompt_value][0].text[:100]) if [prompt_value][0].text else None}")
    print()
    
    print("=" * 60)
    print("调试完成")
    print("=" * 60)

if __name__ == "__main__":
    debug_ragas_llm_call()

