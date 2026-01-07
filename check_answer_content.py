#!/usr/bin/env python3
"""
检查生成的答案内容，帮助理解为什么会有参数不合法的问题
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rag import RAGEngine

def check_answer_content():
    """检查答案内容"""
    print("=" * 60)
    print("检查生成的答案内容")
    print("=" * 60)
    print()
    
    try:
        engine = RAGEngine(kb_id="recipes_kb")
        
        # 使用测试集中的问题
        test_questions = [
            "How do I cook qing xie (misspelled as '青蟹') with curry?",
            "在咖喱炒蟹这道沿海家常菜里，洋葱要怎么处理才能和咖喱块、椰浆这些材料一起做出正宗风味？",
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{'='*60}")
            print(f"样本 {i}")
            print(f"{'='*60}")
            print(f"\n问题: {question}")
            print("-" * 60)
            
            result = engine.query(question, top_k=5)
            answer = result.get("answer", "")
            
            print(f"\n答案类型: {type(answer)}")
            print(f"答案长度: {len(answer)} 字符")
            print()
            
            # 显示答案内容
            print("答案内容（完整）:")
            print("-" * 60)
            print(answer)
            print("-" * 60)
            print()
            
            # 显示答案的 repr（可以看到特殊字符）
            print("答案内容（repr，前200字符）:")
            print("-" * 60)
            print(repr(answer[:200]))
            print("-" * 60)
            print()
            
            # 检查答案的字符
            print("字符分析:")
            print("-" * 60)
            print(f"  总字符数: {len(answer)}")
            print(f"  包含换行符: {'\\n' in answer}")
            print(f"  包含制表符: {'\\t' in answer}")
            print(f"  包含空字符: {'\\x00' in answer}")
            print(f"  包含非ASCII字符: {any(ord(c) > 127 for c in answer)}")
            
            # 检查是否有控制字符
            control_chars = []
            for j, char in enumerate(answer[:500]):
                if ord(char) < 32 and char not in ['\n', '\t', '\r']:
                    control_chars.append((j, char, ord(char)))
            
            if control_chars:
                print(f"  ⚠️  发现 {len(control_chars)} 个控制字符（前10个）:")
                for pos, char, code in control_chars[:10]:
                    print(f"    位置 {pos}: {repr(char)} (ASCII {code})")
            else:
                print("  ✅ 没有发现控制字符")
            
            # 检查答案是否可以正常序列化
            print()
            print("序列化检查:")
            print("-" * 60)
            try:
                import json
                json_str = json.dumps(answer, ensure_ascii=False)
                print(f"  ✅ JSON 序列化成功（长度: {len(json_str)}）")
            except Exception as e:
                print(f"  ❌ JSON 序列化失败: {e}")
            
            # 检查答案是否可以正常编码
            try:
                utf8_bytes = answer.encode('utf-8')
                print(f"  ✅ UTF-8 编码成功（长度: {len(utf8_bytes)} 字节）")
            except Exception as e:
                print(f"  ❌ UTF-8 编码失败: {e}")
            
            print()
            print("结论:")
            print("-" * 60)
            print("答案本身是正常的字符串，包含:")
            print("  - 中文字符（Unicode）")
            print("  - 英文字符（ASCII）")
            print("  - 换行符（\\n）")
            print("  - 标点符号")
            print()
            print("问题可能在于:")
            print("  - RAGAS 的 answer_relevancy 在处理答案时，")
            print("    会从答案中提取陈述（statements）")
            print("  - 提取的陈述可能包含 None 或空列表")
            print("  - 或者，生成的问题可能包含 None 或非字符串类型")
            print("  - 这些非字符串类型的内容被传递给通义千问 API，")
            print("    导致 'contents is neither str nor list of str' 错误")
            print()
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_answer_content()

