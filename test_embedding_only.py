#!/usr/bin/env python3
"""
只测试 EmbeddingExtractor 的简化脚本
"""

import sys
import asyncio
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import AppConfig
from document import ParserFactory
from ragas.testset.graph import Node
from ragas.testset.transforms.extractors import EmbeddingExtractor
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from openai import OpenAI, AsyncOpenAI

def main():
    print("=" * 60)
    print("测试 EmbeddingExtractor")
    print("=" * 60)
    
    # 1. 加载配置
    app_config = AppConfig.load()
    
    # 2. 创建 Embeddings
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
    if hasattr(embeddings, 'async_client'):
        embeddings.async_client = async_openai_client
    
    print(f"✅ Embeddings 创建成功: {app_config.embedding.model}")
    
    # 3. 创建 EmbeddingExtractor
    extractor = EmbeddingExtractor(embedding_model=embeddings)
    print(f"✅ EmbeddingExtractor 创建成功")
    
    # 4. 加载测试文档
    base_path = Path("../HowToCook/dishes")
    if not base_path.exists():
        base_path = Path("HowToCook/dishes")
    
    if not base_path.exists():
        print(f"❌ 未找到文档目录")
        return False
    
    md_files = list(base_path.rglob("*.md"))[:3]  # 只测试前 3 个
    print(f"✅ 找到 {len(md_files)} 个测试文档")
    print()
    
    # 5. 测试每个文档
    async def test_all():
        success_count = 0
        for i, md_file in enumerate(md_files, 1):
            try:
                parser = ParserFactory.get_parser(md_file)
                if not parser:
                    continue
                
                result = parser.parse(md_file)
                content = result.get("content", "")
                if not isinstance(content, str):
                    content = str(content) if content else ""
                
                if not content.strip():
                    continue
                
                # 创建节点
                node = Node(
                    properties={
                        "page_content": content,
                        "source": str(md_file),
                    }
                )
                
                # 提取 embedding
                property_name, property_value = await extractor.extract(node)
                
                print(f"[{i}/{len(md_files)}] ✅ {md_file.name}")
                print(f"    - 内容长度: {len(content)} 字符")
                print(f"    - 提取属性: {property_name}")
                print(f"    - 向量维度: {len(property_value)}")
                success_count += 1
            except Exception as e:
                print(f"[{i}/{len(md_files)}] ❌ {md_file.name}: {e}")
        
        print()
        print("=" * 60)
        print(f"✅ 成功: {success_count}/{len(md_files)}")
        print("=" * 60)
        return success_count == len(md_files)
    
    return asyncio.run(test_all())

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

