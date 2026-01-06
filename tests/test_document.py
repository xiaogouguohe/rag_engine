#!/usr/bin/env python3
"""
测试脚本：验证文档解析和分块功能

使用方法：
    python tests/test_document.py
"""

import sys
from pathlib import Path
import tempfile
import os

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from document import DocumentParser, TextChunker


def test_txt_parser():
    """测试 TXT 文件解析"""
    print("=" * 60)
    print("测试 1: TXT 文件解析")
    print("=" * 60)
    
    # 创建临时测试文件
    test_content = """这是第一段文本。
这是第二段文本。
这是第三段文本，包含一些中文内容。

还有更多内容在这里。"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(test_content)
        temp_file = f.name
    
    try:
        parser = DocumentParser()
        result = parser.parse(temp_file)
        
        print("✅ TXT 解析成功！")
        print(f"   文件类型: {result['file_type']}")
        print(f"   文件名: {result['file_name']}")
        print(f"   内容长度: {len(result['content'])} 字符")
        print(f"   内容预览: {result['content'][:50]}...")
        
        assert result['file_type'] == 'txt'
        assert len(result['content']) > 0
        return True
        
    except Exception as e:
        print(f"❌ TXT 解析失败: {e}")
        return False
    finally:
        # 清理临时文件
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_markdown_parser():
    """测试 Markdown 文件解析"""
    print("\n" + "=" * 60)
    print("测试 2: Markdown 文件解析")
    print("=" * 60)
    
    # 创建临时测试文件
    test_content = """# 标题一

这是一段**粗体**文本和*斜体*文本。

## 标题二

- 列表项 1
- 列表项 2
- 列表项 3

这是一段包含[链接](https://example.com)的文本。

```python
# 代码块
print("Hello")
```
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write(test_content)
        temp_file = f.name
    
    try:
        parser = DocumentParser()
        result = parser.parse(temp_file)
        
        print("✅ Markdown 解析成功！")
        print(f"   文件类型: {result['file_type']}")
        print(f"   文件名: {result['file_name']}")
        print(f"   内容长度: {len(result['content'])} 字符")
        print(f"   内容预览: {result['content'][:100]}...")
        
        # 验证 Markdown 语法已被清理
        assert "#" not in result['content'] or result['content'].count("#") < test_content.count("#")
        assert "**" not in result['content']
        assert "```" not in result['content']
        
        assert result['file_type'] == 'markdown'
        assert len(result['content']) > 0
        return True
        
    except Exception as e:
        print(f"❌ Markdown 解析失败: {e}")
        return False
    finally:
        # 清理临时文件
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_text_chunker():
    """测试文本分块"""
    print("\n" + "=" * 60)
    print("测试 3: 文本分块")
    print("=" * 60)
    
    # 创建测试文本（较长）
    test_text = """这是第一段文本。包含一些内容。
这是第二段文本。也包含一些内容。
这是第三段文本。继续包含内容。
这是第四段文本。还有更多内容。
这是第五段文本。最后一段内容。"""
    
    chunker = TextChunker(
        chunk_size=50,  # 每块 50 字符
        chunk_overlap=10,  # 重叠 10 字符
    )
    
    try:
        chunks = chunker.split_text(test_text)
        
        print("✅ 文本分块成功！")
        print(f"   原始文本长度: {len(test_text)} 字符")
        print(f"   分块数量: {len(chunks)}")
        print(f"   每块大小: {chunker.chunk_size} 字符")
        print(f"   重叠大小: {chunker.chunk_overlap} 字符")
        
        for i, chunk in enumerate(chunks, 1):
            print(f"\n   块 {i}:")
            print(f"     文本: {chunk['text'][:40]}...")
            print(f"     位置: {chunk['start']}-{chunk['end']}")
            print(f"     长度: {len(chunk['text'])} 字符")
        
        assert len(chunks) > 0
        assert all(len(chunk['text']) > 0 for chunk in chunks)
        return True
        
    except Exception as e:
        print(f"❌ 文本分块失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chunker_with_metadata():
    """测试带元数据的分块"""
    print("\n" + "=" * 60)
    print("测试 4: 带元数据的分块")
    print("=" * 60)
    
    test_text = """这是第一段文本。包含一些内容。
这是第二段文本。也包含一些内容。
这是第三段文本。继续包含内容。"""
    
    metadata = {
        "doc_id": "test_doc_001",
        "file_name": "test.txt",
        "file_type": "txt",
    }
    
    chunker = TextChunker(chunk_size=40, chunk_overlap=5)
    
    try:
        chunks = chunker.chunk_document(test_text, metadata=metadata)
        
        print("✅ 带元数据的分块成功！")
        print(f"   分块数量: {len(chunks)}")
        
        for i, chunk in enumerate(chunks, 1):
            print(f"\n   块 {i}:")
            print(f"     文本: {chunk['text'][:30]}...")
            print(f"     元数据: doc_id={chunk['metadata']['doc_id']}, chunk_index={chunk['metadata']['chunk_index']}")
        
        assert len(chunks) > 0
        assert all('metadata' in chunk for chunk in chunks)
        assert all(chunk['metadata']['doc_id'] == 'test_doc_001' for chunk in chunks)
        return True
        
    except Exception as e:
        print(f"❌ 带元数据的分块失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """测试完整流程：解析 + 分块"""
    print("\n" + "=" * 60)
    print("测试 5: 完整流程（解析 + 分块）")
    print("=" * 60)
    
    # 创建测试文件
    test_content = """这是文档的第一段内容。包含一些重要信息。
这是文档的第二段内容。继续提供更多信息。
这是文档的第三段内容。最后一段内容。"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(test_content)
        temp_file = f.name
    
    try:
        # 1. 解析文档
        parser = DocumentParser()
        doc = parser.parse(temp_file)
        
        # 2. 分块
        chunker = TextChunker(chunk_size=30, chunk_overlap=5)
        chunks = chunker.chunk_document(
            doc['content'],
            metadata={
                "doc_id": "test_doc_002",
                "file_name": doc['file_name'],
                "file_type": doc['file_type'],
            }
        )
        
        print("✅ 完整流程测试成功！")
        print(f"   文档: {doc['file_name']}")
        print(f"   原始内容长度: {len(doc['content'])} 字符")
        print(f"   分块数量: {len(chunks)}")
        
        for i, chunk in enumerate(chunks, 1):
            print(f"\n   块 {i}:")
            print(f"     文本: {chunk['text']}")
            print(f"     元数据: {chunk['metadata']}")
        
        assert len(chunks) > 0
        return True
        
    except Exception as e:
        print(f"❌ 完整流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def main():
    """主测试函数"""
    print("\n🚀 RAG 引擎 - 文档解析和分块测试\n")
    
    results = []
    
    # 运行所有测试
    results.append(("TXT 解析", test_txt_parser()))
    results.append(("Markdown 解析", test_markdown_parser()))
    results.append(("文本分块", test_text_chunker()))
    results.append(("带元数据的分块", test_chunker_with_metadata()))
    results.append(("完整流程", test_integration()))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 所有测试通过！文档解析和分块功能正常。")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查错误信息。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
