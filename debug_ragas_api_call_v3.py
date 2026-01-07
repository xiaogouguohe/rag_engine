#!/usr/bin/env python3
"""
调试 RAGAS 调用 LLM API 时的参数（方法 3：使用 HTTP 请求拦截）
通过拦截 OpenAI 客户端的 HTTP 请求来查看实际发送的参数
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

# 存储所有 HTTP 请求
http_requests = []

def debug_ragas_api_call_v3():
    """使用 HTTP 请求拦截调试"""
    print("=" * 60)
    print("调试 RAGAS 调用 LLM API（拦截 HTTP 请求）")
    print("=" * 60)
    print()
    
    # 加载配置
    app_config = AppConfig.load()
    
    # 方法：拦截 OpenAI 客户端的 HTTP 请求
    # 通过 monkey patching httpx 或 requests 来拦截
    
    try:
        import httpx
        from openai import OpenAI
        
        # 创建原始客户端
        original_client = OpenAI(
            api_key=app_config.llm.api_key,
            base_url=app_config.llm.base_url,
        )
        
        # 保存原始的 post 方法
        original_post = httpx.AsyncClient.post
        
        async def debug_post(self, url, **kwargs):
            """拦截 HTTP POST 请求"""
            print("\n" + "=" * 60)
            print("🔍 拦截到 HTTP POST 请求")
            print("=" * 60)
            print(f"📋 URL: {url}")
            print(f"📋 kwargs keys: {list(kwargs.keys())}")
            
            # 检查 data 或 json 参数
            if 'json' in kwargs:
                print(f"\n📋 JSON 参数:")
                json_data = kwargs['json']
                print(f"  - 类型: {type(json_data)}")
                print(f"  - 内容: {json.dumps(json_data, ensure_ascii=False, indent=2)}")
                
                # 检查 messages
                if 'messages' in json_data:
                    print(f"\n📋 Messages ({len(json_data['messages'])} 个):")
                    for i, msg in enumerate(json_data['messages']):
                        print(f"  消息 {i+1}:")
                        print(f"    - role: {msg.get('role')}")
                        print(f"    - content 类型: {type(msg.get('content'))}")
                        print(f"    - content 值: {repr(msg.get('content'))}")
                        
                        # 检查 content 是否有问题
                        content = msg.get('content')
                        if content is None:
                            print(f"    ⚠️  content 是 None!")
                        elif not isinstance(content, str):
                            print(f"    ⚠️  content 不是字符串: {type(content)}")
                            if isinstance(content, list):
                                print(f"    ⚠️  content 是列表，长度: {len(content)}")
                                for j, item in enumerate(content):
                                    print(f"       项目 {j}: 类型={type(item)}, 值={repr(item)}")
                
                # 保存请求信息
                http_requests.append({
                    "url": str(url),
                    "method": "POST",
                    "json": json_data,
                })
            
            # 调用原始方法
            print("\n📤 发送 HTTP 请求...")
            try:
                response = await original_post(self, url, **kwargs)
                print("✅ HTTP 请求成功")
                return response
            except Exception as e:
                print(f"❌ HTTP 请求失败: {e}")
                raise
        
        # 替换方法（需要找到正确的位置）
        # 注意：这需要更深入的了解 httpx 的内部结构
        
    except ImportError:
        print("⚠️  无法导入 httpx，使用其他方法")
    
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
    print("💡 提示：要查看实际的 HTTP 请求，可以使用以下方法：")
    print("  1. 使用 mitmproxy 等 HTTP 代理工具")
    print("  2. 在 LangChain 中启用详细日志")
    print("  3. 使用回调函数（如上面的方法 2）")

if __name__ == "__main__":
    debug_ragas_api_call_v3()

