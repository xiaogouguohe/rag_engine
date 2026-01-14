import requests
import json
import os
from pathlib import Path

# 尝试从 .env 加载 API KEY
def get_api_key():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                # 过滤注释行和空行
                clean_line = line.strip()
                if not clean_line or clean_line.startswith("#"):
                    continue
                
                if "RAG_EMBEDDING_API_KEY" in clean_line and "=" in clean_line:
                    # 先通过 = 分割，再通过 # 分割以去除行尾注释
                    value_part = clean_line.split("=")[1].strip()
                    key_value = value_part.split("#")[0].strip()
                    return key_value.strip("'").strip('"')
    return "YOUR_API_KEY_HERE"

API_KEY = get_api_key()
BASE_URL = "https://api.siliconflow.cn/v1/embeddings"

def demo_api_call():
    """
    演示如何通过底层 HTTP 请求尝试获取 BGE-M3 的多种向量。
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # 构建请求体
    payload = {
        "model": "BAAI/bge-m3",
        "input": "如何制作美味的红烧肉？",
        # 理论上支持 BGE-M3 全功能的后端会接收这些扩展参数
        "return_dense": True,
        "return_sparse": True,
        "return_colbert": True
    }

    print(f"📡 正在通过 API 获取 BGE-M3 向量...")
    print(f"🔗 URL: {BASE_URL}")
    print(f"💡 提示：公有云标准接口通常仅返回 Dense 部分。\n")

    try:
        response = requests.post(BASE_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            # 1. 解析密集向量 (Dense)
            dense = result['data'][0]['embedding']
            print(f"✅ 【密集向量 (Dense)】已获取！")
            print(f"   维度: {len(dense)}")
            print(f"   预览: {dense[:3]}...")

            # 2. 检查是否有稀疏向量 (Sparse)
            # 在标准的 OpenAI 响应中，这个字段是不存在的，需要厂商自定义返回
            sparse = result['data'][0].get('sparse_embedding') or result.get('lexical_weights')
            if sparse:
                print(f"\n✅ 【稀疏向量 (Sparse)】已获取！")
                print(f"   内容: {sparse}")
            else:
                print(f"\n❌ 【稀疏向量 (Sparse)】未获取。原因：API 节点未返回该字段。")

            # 3. 检查多向量 (ColBERT)
            colbert = result['data'][0].get('colbert_vecs')
            if colbert:
                print(f"\n✅ 【多向量 (Multi-Vector)】已获取！")
                print(f"   形状: {len(colbert)} 个 Token")
            else:
                print(f"❌ 【多向量 (Multi-Vector)】未获取。")

        else:
            print(f"❌ 请求失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")

    except Exception as e:
        print(f"❌ 发生异常: {e}")

if __name__ == "__main__":
    demo_api_call()

