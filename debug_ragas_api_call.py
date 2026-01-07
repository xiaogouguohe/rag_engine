#!/usr/bin/env python3
"""
调试 RAGAS 调用 LLM API 时的参数
使用 monkey patching 和回调函数来拦截和记录 API 调用
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import json
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.metrics._answer_relevance import ResponseRelevanceInput, ResponseRelevancePrompt
from config.config import AppConfig

# 存储所有 API 调用记录
api_calls = []

def debug_ragas_api_call():
    """调试 RAGAS 调用 LLM API 时的参数"""
    print("=" * 60)
    print("调试 RAGAS 调用 LLM API 时的参数")
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
    print(f"   ✅ LangChain LLM 创建成功")
    print()
    
    # 方法 1: 使用 monkey patching 拦截 agenerate_prompt
    print("2. 设置 API 调用拦截...")
    original_agenerate_prompt = langchain_llm.agenerate_prompt
    
    async def debug_agenerate_prompt(prompts, stop=None, callbacks=None, **kwargs):
        """拦截 agenerate_prompt 调用"""
        print("\n" + "=" * 60)
        print("🔍 拦截到 agenerate_prompt 调用")
        print("=" * 60)
        
        # 记录调用信息
        call_info = {
            "method": "agenerate_prompt",
            "prompts_count": len(prompts),
            "stop": stop,
            "kwargs": kwargs,
        }
        
        # 详细检查每个 prompt
        print(f"\n📋 参数信息:")
        print(f"  - prompts 数量: {len(prompts)}")
        print(f"  - stop: {stop}")
        print(f"  - kwargs: {kwargs}")
        print()
        
        # 检查每个 prompt
        for i, prompt in enumerate(prompts):
            print(f"📝 Prompt {i+1}:")
            print(f"  - 类型: {type(prompt)}")
            
            # 如果是 PromptValue，检查其属性
            if hasattr(prompt, 'text'):
                print(f"  - text 类型: {type(prompt.text)}")
                print(f"  - text 值: {repr(prompt.text[:200]) if prompt.text else None}")
                print(f"  - text 是否为 None: {prompt.text is None}")
                print(f"  - text 是否为空: {not prompt.text if prompt.text else True}")
                
                call_info[f"prompt_{i}_text"] = prompt.text
                call_info[f"prompt_{i}_text_type"] = str(type(prompt.text))
                call_info[f"prompt_{i}_text_is_none"] = prompt.text is None
            
            # 转换为消息
            try:
                messages = prompt.to_messages()
                print(f"  - to_messages() 成功，消息数量: {len(messages)}")
                
                for j, msg in enumerate(messages):
                    print(f"    消息 {j+1}:")
                    print(f"      - 类型: {type(msg)}")
                    print(f"      - content 类型: {type(msg.content)}")
                    print(f"      - content 值: {repr(str(msg.content)[:100]) if msg.content else None}")
                    print(f"      - content 是否为 None: {msg.content is None}")
                    print(f"      - content 是否为字符串: {isinstance(msg.content, str)}")
                    
                    # 检查是否有问题
                    if msg.content is None:
                        print(f"      ⚠️  content 是 None!")
                    elif not isinstance(msg.content, str):
                        print(f"      ⚠️  content 不是字符串: {type(msg.content)}")
                        if isinstance(msg.content, list):
                            print(f"      ⚠️  content 是列表，长度: {len(msg.content)}")
                            for k, item in enumerate(msg.content):
                                print(f"         项目 {k}: 类型={type(item)}, 值={repr(item)[:50]}")
                    
                    call_info[f"prompt_{i}_message_{j}_content"] = str(msg.content) if msg.content else None
                    call_info[f"prompt_{i}_message_{j}_content_type"] = str(type(msg.content))
                    call_info[f"prompt_{i}_message_{j}_content_is_none"] = msg.content is None
                    
            except Exception as e:
                print(f"  - to_messages() 失败: {e}")
                call_info[f"prompt_{i}_to_messages_error"] = str(e)
            
            print()
        
        # 保存调用记录
        api_calls.append(call_info)
        
        # 调用原始方法
        print("📤 调用原始 agenerate_prompt...")
        try:
            result = await original_agenerate_prompt(prompts, stop=stop, callbacks=callbacks, **kwargs)
            print("✅ API 调用成功")
            return result
        except Exception as e:
            print(f"❌ API 调用失败: {e}")
            print(f"   错误类型: {type(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    # 替换方法
    langchain_llm.agenerate_prompt = debug_agenerate_prompt
    print("   ✅ API 调用拦截已设置")
    print()
    
    # 创建 RAGAS LLM Wrapper
    print("3. 创建 RAGAS LLM Wrapper...")
    ragas_llm = LangchainLLMWrapper(langchain_llm)
    print(f"   ✅ RAGAS LLM Wrapper 创建成功")
    print()
    
    # 创建测试数据
    print("4. 创建测试数据...")
    test_answer = "根据文档片段，可以按照以下步骤用咖喱烹饪青蟹。"
    prompt_input = ResponseRelevanceInput(response=test_answer)
    print(f"   ✅ 测试数据创建成功")
    print(f"      答案: {test_answer}")
    print()
    
    # 创建 Prompt
    print("5. 创建 Prompt...")
    prompt = ResponseRelevancePrompt()
    print(f"   ✅ Prompt 创建成功")
    print()
    
    # 测试生成（只生成 1 个问题，避免兼容性问题）
    print("6. 测试生成问题（strictness=1）...")
    print("-" * 60)
    
    import asyncio
    async def test_generate():
        try:
            result = await prompt.generate_multiple(
                llm=ragas_llm,
                data=prompt_input,
                n=1,  # 只生成 1 个问题
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
    
    # 保存调用记录
    print("7. 保存 API 调用记录...")
    output_file = Path("ragas_api_calls_debug.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(api_calls, f, ensure_ascii=False, indent=2)
    print(f"   ✅ 调用记录已保存到: {output_file}")
    print()
    
    print("=" * 60)
    print("调试完成")
    print("=" * 60)
    print()
    print("📊 调用记录摘要:")
    print(f"  - 总调用次数: {len(api_calls)}")
    for i, call in enumerate(api_calls, 1):
        print(f"  - 调用 {i}:")
        print(f"    - prompts 数量: {call.get('prompts_count', 'N/A')}")
        for key, value in call.items():
            if key.startswith('prompt_') and 'content_is_none' in key and value:
                print(f"    - ⚠️  {key}: {value}")

if __name__ == "__main__":
    debug_ragas_api_call()

