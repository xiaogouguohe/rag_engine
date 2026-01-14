from pymilvus import MilvusClient
import json
import os

# 适配不同的运行路径（优先检查当前目录下的 data 目录）
db_path = "data/indices/milvus_lite.db"
if not os.path.exists(db_path):
    db_path = "rag_engine/data/indices/milvus_lite.db"

if not os.path.exists(db_path):
    print(f"❌ 找不到数据库文件，请确认路径是否正确。")
    print(f"   尝试过的路径: data/indices/milvus_lite.db 或 rag_engine/data/indices/milvus_lite.db")
    exit(1)

client = MilvusClient(uri=db_path)

collections = client.list_collections()
print(f"当前集合列表: {collections}")

for coll in collections:
    print(f"\n--- 正在检查集合: {coll} ---")
    # 获取一行数据
    res = client.query(
        collection_name=coll,
        filter="",
        limit=1,
        output_fields=["*"] # 获取所有字段
    )
    
    if res:
        row = res[0]
        print("该行包含的字段及其样例如下:")
        print("-" * 50)
        for key in row.keys():
            value = row[key]
            # 针对不同类型的字段进行美化展示
            if isinstance(value, list) and len(value) > 10:
                print(f"  🔹 {key:15}: [向量/列表] 长度: {len(value)}")
            elif isinstance(value, dict) and len(value) > 5:
                print(f"  🔹 {key:15}: [稀疏向量/字典] 包含 {len(value)} 个键值对")
            else:
                # 对 metadata 字符串做一下 JSON 格式化展示
                if key == "metadata" and isinstance(value, str):
                    try:
                        meta_json = json.loads(value)
                        print(f"  🔹 {key:15}: {json.dumps(meta_json, ensure_ascii=False)}")
                        continue
                    except: pass
                print(f"  🔹 {key:15}: {value}")
        print("-" * 50)
    else:
        print("该集合为空。")

client.close()

