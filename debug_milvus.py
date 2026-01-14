import json
from pymilvus import MilvusClient
import os

db_path = "rag_engine/data/indices/milvus_lite.db"
client = MilvusClient(uri=db_path)

print(f"Checking database at: {db_path}")
collections = client.list_collections()
print(f"Collections: {collections}")

for coll in collections:
    stats = client.describe_collection(coll)
    print(f"\nCollection: {coll}")
    print(f"Stats: {stats}")
    # Try a query to count
    res = client.query(collection_name=coll, filter="", output_fields=["count(*)"])
    print(f"Count query result: {res}")

