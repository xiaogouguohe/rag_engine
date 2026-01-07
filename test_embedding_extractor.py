#!/usr/bin/env python3
"""
单独测试 EmbeddingExtractor 的功能
用于排查 RAGAS EmbeddingExtractor 的问题
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import AppConfig
from document import ParserFactory
from langchain_core.documents import Document as LangchainDocument

# 尝试导入 RAGAS
try:
    from ragas.testset.graph import KnowledgeGraph, Node
    from ragas.testset.transforms.extractors import EmbeddingExtractor
    from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from openai import OpenAI, AsyncOpenAI
    from langchain_openai import ChatOpenAI
    _HAS_RAGAS = True
except ImportError as e:
    print(f"❌ 导入 RAGAS 失败: {e}")
    sys.exit(1)

def test_embedding_extractor():
    """测试 EmbeddingExtractor"""
    print("=" * 60)
    print("测试 EmbeddingExtractor")
    print("=" * 60)
    
    # 1. 加载配置
    print("\n1. 加载配置...")
    try:
        app_config = AppConfig.load()
        print(f"   ✅ 配置加载成功")
        print(f"   - Embedding Model: {app_config.embedding.model}")
        print(f"   - Embedding Base URL: {app_config.embedding.base_url}")
    except Exception as e:
        print(f"   ❌ 配置加载失败: {e}")
        return False
    
    # 2. 加载一个测试文档
    print("\n2. 加载测试文档...")
    # 尝试多个可能的路径
    possible_paths = [
        Path("../HowToCook/dishes/meat_dish/红烧肉.md"),
        Path("HowToCook/dishes/meat_dish/红烧肉.md"),
        Path("../HowToCook/dishes/meat_dish").glob("*.md"),
    ]
    
    test_file = None
    for path in possible_paths:
        if isinstance(path, Path) and path.exists():
            test_file = path
            break
        elif hasattr(path, '__iter__'):
            # 如果是 glob 结果
            for p in path:
                test_file = p
                break
            if test_file:
                break
    
    if not test_file or not test_file.exists():
        print(f"   ⚠️  未找到红烧肉.md，尝试查找任意 .md 文件...")
        # 尝试查找任意 .md 文件
        for base_path in [Path("../HowToCook/dishes"), Path("HowToCook/dishes")]:
            if base_path.exists():
                md_files = list(base_path.rglob("*.md"))
                if md_files:
                    test_file = md_files[0]
                    print(f"   ✅ 找到测试文件: {test_file}")
                    break
    
    if not test_file or not test_file.exists():
        print(f"   ❌ 未找到测试文件")
        return False
    
    try:
        parser = ParserFactory.get_parser(test_file)
        if not parser:
            print(f"   ❌ 无法获取解析器")
            return False
        
        result = parser.parse(test_file)
        content = result.get("content", "")
        
        # 确保是字符串类型
        if not isinstance(content, str):
            content = str(content) if content else ""
        
        if not content.strip():
            print(f"   ❌ 文档内容为空")
            return False
        
        print(f"   ✅ 文档加载成功")
        print(f"   - 文件: {test_file.name}")
        print(f"   - 内容长度: {len(content)} 字符")
        print(f"   - 内容类型: {type(content)}")
        print(f"   - 前 200 字符: {repr(content[:200])}")
    except Exception as e:
        print(f"   ❌ 文档加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 创建 KnowledgeGraph 节点
    print("\n3. 创建 KnowledgeGraph 节点...")
    try:
        node = Node(
            properties={
                "page_content": content,
                "source": str(test_file),
            }
        )
        print(f"   ✅ 节点创建成功")
        print(f"   - page_content 类型: {type(node.properties['page_content'])}")
        print(f"   - page_content 长度: {len(node.properties['page_content'])}")
    except Exception as e:
        print(f"   ❌ 节点创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 创建 Embeddings
    print("\n4. 创建 Embeddings...")
    try:
        # 方案1：使用 RAGAS 原生的 OpenAIEmbeddings
        openai_client = OpenAI(
            api_key=app_config.embedding.api_key,
            base_url=app_config.embedding.base_url,
        )
        async_openai_client = AsyncOpenAI(
            api_key=app_config.embedding.api_key,
            base_url=app_config.embedding.base_url,
        )
        
        embeddings = RagasOpenAIEmbeddings(
            client=openai_client,
            model=app_config.embedding.model,
        )
        
        # 设置异步客户端（如果需要）
        if hasattr(embeddings, 'async_client'):
            embeddings.async_client = async_openai_client
        
        print(f"   ✅ RAGAS OpenAIEmbeddings 创建成功")
        print(f"   - Model: {app_config.embedding.model}")
        print(f"   - Base URL: {app_config.embedding.base_url}")
        
        # 测试简单的 embedding 调用
        print(f"\n   测试简单 embedding 调用...")
        test_text = "这是一个测试文本"
        try:
            # 同步调用
            test_vector = embeddings.embed_text(test_text)
            print(f"   ✅ Embedding 调用成功")
            print(f"   - 向量维度: {len(test_vector)}")
        except Exception as e:
            print(f"   ❌ Embedding 调用失败: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"   ❌ 创建 Embeddings 失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. 创建 EmbeddingExtractor
    print("\n5. 创建 EmbeddingExtractor...")
    try:
        extractor = EmbeddingExtractor(embedding_model=embeddings)
        print(f"   ✅ EmbeddingExtractor 创建成功")
    except Exception as e:
        print(f"   ❌ 创建 EmbeddingExtractor 失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 6. 测试提取
    print("\n6. 测试 EmbeddingExtractor.extract()...")
    try:
        import asyncio
        
        # 测试提取节点
        async def test_extract():
            try:
                result = await extractor.extract(node)
                print(f"   ✅ Extract 成功")
                print(f"   - 返回类型: {type(result)}")
                if isinstance(result, tuple):
                    print(f"   - 结果: property_name={result[0]}, property_value 类型={type(result[1])}")
                    if isinstance(result[1], (list, tuple)):
                        print(f"   - 向量维度: {len(result[1])}")
                else:
                    print(f"   - 结果: {result}")
                return True
            except Exception as e:
                print(f"   ❌ Extract 失败: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        success = asyncio.run(test_extract())
        if not success:
            return False
            
    except Exception as e:
        print(f"   ❌ 测试提取失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 7. 测试批量处理（模拟 RAGAS 的完整流程）
    print("\n7. 测试批量处理（模拟 RAGAS 的完整流程）...")
    try:
        # 加载多个文档
        base_path = Path("../HowToCook/dishes")
        if not base_path.exists():
            base_path = Path("HowToCook/dishes")
        
        if base_path.exists():
            md_files = list(base_path.rglob("*.md"))[:5]  # 只测试前 5 个
            print(f"   - 找到 {len(md_files)} 个文档")
            
            # 创建多个节点
            nodes = []
            for md_file in md_files:
                try:
                    parser = ParserFactory.get_parser(md_file)
                    if parser:
                        result = parser.parse(md_file)
                        content = result.get("content", "")
                        if not isinstance(content, str):
                            content = str(content) if content else ""
                        
                        if content.strip():
                            node = Node(
                                properties={
                                    "page_content": content,
                                    "source": str(md_file),
                                }
                            )
                            nodes.append(node)
                except Exception as e:
                    print(f"   ⚠️  跳过文档 {md_file.name}: {e}")
                    continue
            
            print(f"   - 创建了 {len(nodes)} 个节点")
            
            # 测试批量提取
            import asyncio
            async def batch_extract():
                success_count = 0
                fail_count = 0
                for i, test_node in enumerate(nodes, 1):
                    try:
                        result = await extractor.extract(test_node)
                        success_count += 1
                        print(f"   ✅ [{i}/{len(nodes)}] 节点提取成功")
                    except Exception as e:
                        fail_count += 1
                        print(f"   ❌ [{i}/{len(nodes)}] 节点提取失败: {e}")
                        # 打印失败节点的信息
                        page_content = test_node.properties.get("page_content", "")
                        print(f"      - 内容类型: {type(page_content)}")
                        print(f"      - 内容长度: {len(page_content) if isinstance(page_content, str) else 'N/A'}")
                        if isinstance(page_content, str) and len(page_content) > 0:
                            print(f"      - 前 100 字符: {repr(page_content[:100])}")
                        import traceback
                        traceback.print_exc()
                        break  # 只测试第一个失败的
                
                print(f"\n   - 成功: {success_count}/{len(nodes)}")
                print(f"   - 失败: {fail_count}/{len(nodes)}")
                return success_count == len(nodes)
            
            batch_success = asyncio.run(batch_extract())
            if not batch_success:
                print(f"\n   ⚠️  批量处理有失败，可能存在问题")
                return False
        else:
            print(f"   ⚠️  未找到文档目录，跳过批量测试")
            
    except Exception as e:
        print(f"   ❌ 批量处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 8. 测试 RAGAS 的完整流程（模拟 generate_with_langchain_docs）
    print("\n8. 测试 RAGAS 的完整流程（模拟 generate_with_langchain_docs）...")
    try:
        from ragas.testset import TestsetGenerator
        from langchain_openai import ChatOpenAI
        
        # 创建 LLM
        generator_llm = ChatOpenAI(
            model=app_config.llm.model,
            api_key=app_config.llm.api_key,
            base_url=app_config.llm.base_url,
            temperature=0.1,
        )
        
        # 创建 TestsetGenerator
        generator = TestsetGenerator.from_langchain(
            llm=generator_llm,
            embedding_model=embeddings,
        )
        print(f"   ✅ TestsetGenerator 创建成功")
        
        # 创建 langchain Document
        from langchain_core.documents import Document as LangchainDocument
        test_doc = LangchainDocument(
            page_content=content,
            metadata={"source": str(test_file)},
        )
        
        # 尝试生成测试集（小规模）
        print(f"\n   测试小规模生成（1 个问题）...")
        try:
            testset = generator.generate_with_langchain_docs(
                documents=[test_doc],
                testset_size=1,
                transforms_embedding_model=embeddings,  # 明确指定
                raise_exceptions=False,
            )
            print(f"   ✅ 小规模生成成功")
            if testset:
                print(f"   - 生成了测试集")
            return True
        except Exception as e:
            print(f"   ❌ 小规模生成失败: {e}")
            print(f"   这是完整流程中的错误，需要进一步排查")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"   ❌ 完整流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_embedding_extractor()
    sys.exit(0 if success else 1)

