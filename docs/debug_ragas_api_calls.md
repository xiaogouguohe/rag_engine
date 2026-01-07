# 调试 RAGAS 调用 LLM API 的方法

## 方法 1: 使用详细日志（推荐）

最简单的方法是启用 LangChain 和 OpenAI 的详细日志：

```bash
python3 debug_ragas_simple.py
```

这会显示：
- HTTP 请求的 URL
- 请求的 JSON 参数（包括 messages）
- 响应内容

### 从日志中可以看到：

```json
{
  "messages": [
    {
      "content": "Generate a question for the given answer...",
      "role": "user"
    }
  ],
  "model": "qwen3-max",
  "n": 1,
  "stream": false,
  "temperature": 0.01
}
```

## 方法 2: 使用回调函数

使用 LangChain 的回调函数来拦截 API 调用：

```bash
python3 debug_ragas_api_call_v2.py
```

这会：
- 在 LLM 开始调用时记录 prompts
- 在 LLM 调用结束时记录响应
- 在出错时记录错误信息
- 保存调用记录到 JSON 文件

## 方法 3: 使用 HTTP 代理

使用 mitmproxy 等工具拦截 HTTP 请求：

```bash
# 安装 mitmproxy
pip install mitmproxy

# 启动代理
mitmproxy -p 8080

# 设置环境变量
export HTTP_PROXY=http://127.0.0.1:8080
export HTTPS_PROXY=http://127.0.0.1:8080

# 运行评估
python3 generate_ragas_dataset.py --evaluate-only ...
```

## 实际观察到的 API 请求

### 当 strictness = 1 时

```json
{
  "messages": [
    {
      "content": "Generate a question for the given answer...",
      "role": "user"
    }
  ],
  "model": "qwen3-max",
  "n": 1,
  "stream": false,
  "temperature": 0.01
}
```

**观察结果**：
- ✅ `messages[0].content` 是字符串类型（正常）
- ✅ `messages[0].role` 是 'user'（正常）
- ✅ 请求成功

### 当 strictness > 1 时（推测）

```json
{
  "messages": [
    {
      "content": "...",
      "role": "user"
    },
    {
      "content": "...",
      "role": "user"
    },
    {
      "content": "...",
      "role": "user"
    }
  ],
  "model": "qwen3-max",
  "n": 3,
  "stream": false,
  "temperature": 0.01
}
```

**可能的问题**：
- ⚠️ 如果某个 `content` 是 `None`，会导致 `InvalidParams` 错误
- ⚠️ 如果某个 `content` 不是字符串类型，也会导致错误

## 调试步骤

1. **启用详细日志**：
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   logging.getLogger('openai').setLevel(logging.DEBUG)
   ```

2. **运行评估**：
   ```bash
   python3 generate_ragas_dataset.py --evaluate-only ...
   ```

3. **查看日志**：
   - 查找 `json_data` 字段
   - 检查 `messages` 数组
   - 验证每个 `content` 的类型和值

4. **如果发现问题**：
   - 记录有问题的 `content`
   - 检查对应的 `prompt_value`
   - 查看 `to_messages()` 的转换过程

## 相关文件

- `debug_ragas_simple.py` - 使用详细日志
- `debug_ragas_api_call_v2.py` - 使用回调函数
- `debug_ragas_api_call_v3.py` - 使用 HTTP 拦截（未完成）

## 注意事项

1. **日志可能包含敏感信息**（API key 等），注意保护
2. **详细日志会产生大量输出**，建议重定向到文件
3. **回调函数可能影响性能**，仅用于调试

