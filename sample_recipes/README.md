# Sample Recipes for RAGAS Test Set Generation

本目录包含 25 个菜谱文件，用于生成 RAGAS 测试集。

## 文件列表

共 **25 个菜谱文件**：

1. 可乐鸡翅.md
2. 孜然牛肉.md
3. 宫保鸡丁.md
4. 小炒肉.md
5. 尖椒炒牛肉.md
6. 桂圆红枣粥.md
7. 清炒花菜.md
8. 炒凉粉.md
9. 炒青菜.md
10. 烙饼.md
11. 牛奶燕麦.md
12. 皮蛋瘦肉粥.md
13. 皮蛋豆腐.md
14. 紫菜蛋花汤.md
15. 茶叶蛋.md
16. 葱煎豆腐.md
17. 蒜蓉西兰花.md
18. 蛋炒饭.md
19. 西红柿土豆炖牛肉.md
20. 西红柿炒鸡蛋.md
21. 西红柿豆腐汤羹.md
22. 西葫芦炒鸡蛋.md
23. 酸辣土豆丝.md
24. 鸡蛋火腿炒黄瓜.md
25. 麻婆豆腐.md

## 选择原则

### 1. 覆盖不同分类
- **肉菜（7个）**：可乐鸡翅、孜然牛肉、宫保鸡丁、小炒肉、尖椒炒牛肉、西红柿土豆炖牛肉、麻婆豆腐
- **素菜（9个）**：西红柿炒鸡蛋、西红柿豆腐汤羹、西葫芦炒鸡蛋、鸡蛋火腿炒黄瓜、皮蛋豆腐、葱煎豆腐、酸辣土豆丝、蒜蓉西兰花、清炒花菜、炒青菜
- **主食（3个）**：蛋炒饭、烙饼、炒凉粉
- **汤（3个）**：紫菜蛋花汤、皮蛋瘦肉粥、西红柿豆腐汤羹
- **早餐（3个）**：茶叶蛋、牛奶燕麦、桂圆红枣粥

### 2. 包含共同食材（便于生成多跳和对比问题）
- **西红柿相关（3个）**：西红柿炒鸡蛋、西红柿豆腐汤羹、西红柿土豆炖牛肉
- **鸡蛋相关（5个）**：西红柿炒鸡蛋、西葫芦炒鸡蛋、鸡蛋火腿炒黄瓜、茶叶蛋、蛋炒饭、紫菜蛋花汤
- **豆腐相关（4个）**：西红柿豆腐汤羹、皮蛋豆腐、麻婆豆腐、葱煎豆腐

### 3. 多样性
- 不同难度级别（从简单到复杂）
- 不同做法（炒、蒸、煮、凉拌、炖等）
- 不同风格（家常菜、地方特色等）

## 使用说明

这些菜谱文件可用于：
1. 生成 RAGAS 测试集（使用 `generate_ragas_dataset.py`）
2. 验证不同类型问题的生成（单跳、多跳、对比、总结等）
3. 测试知识图谱构建（通过共同食材建立关系）

## 预期生成效果

基于这 25 个文档，预计可以生成：
- **总问题数**：25 个文档 × 3 个问题/文档 = **75 个问题**（如果使用默认配置）
- **问题类型分布**（如果使用 query_distribution）：
  - 单跳问题：约 40%（30 个）
  - 多跳问题：约 30%（23 个）
  - 对比问题：约 20%（15 个）
  - 总结问题：约 10%（7 个）
- **预计耗时**：15-30 分钟（取决于 LLM 和向量模型的速度）

## 生成命令示例

```bash
# 使用非知识图谱方法（推荐，可控性更强）
python generate_ragas_dataset.py \
  --kb_id sample_recipes_kb \
  --source_path sample_recipes \
  --output_path sample_recipes_testset.json \
  --max_docs 25 \
  --num_questions_per_doc 3 \
  --method testset_generator

# 或使用知识图谱方法
python generate_ragas_dataset.py \
  --kb_id sample_recipes_kb \
  --source_path sample_recipes \
  --output_path sample_recipes_testset_kg.json \
  --max_docs 25 \
  --num_questions_per_doc 3 \
  --method knowledge_graph \
  --use_kg
```

