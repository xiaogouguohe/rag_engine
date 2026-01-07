# 安装提示：加速 pip 下载

## 使用国内镜像源加速下载

### 方法一：临时使用镜像源（推荐）

```bash
# 使用清华镜像源
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ragas datasets

# 或使用阿里云镜像源
pip install -i https://mirrors.aliyun.com/pypi/simple/ ragas datasets

# 或使用豆瓣镜像源
pip install -i https://pypi.douban.com/simple/ ragas datasets
```

### 方法二：永久配置镜像源

**macOS/Linux**：

```bash
# 创建或编辑 pip 配置文件
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << EOF
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
[install]
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF
```

**Windows**：

```bash
# 创建或编辑 pip 配置文件
# 路径：%APPDATA%\pip\pip.ini

[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
[install]
trusted-host = pypi.tuna.tsinghua.edu.cn
```

配置后，直接运行 `pip install` 就会使用镜像源。

### 方法三：安装所有依赖时使用镜像源

```bash
# 使用镜像源安装所有依赖
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

## 常用国内镜像源

| 镜像源 | URL |
|--------|-----|
| 清华大学 | https://pypi.tuna.tsinghua.edu.cn/simple |
| 阿里云 | https://mirrors.aliyun.com/pypi/simple/ |
| 豆瓣 | https://pypi.douban.com/simple/ |
| 中科大 | https://pypi.mirrors.ustc.edu.cn/simple/ |
| 华为云 | https://mirrors.huaweicloud.com/repository/pypi/simple |

## 验证镜像源是否生效

```bash
# 查看当前使用的镜像源
pip config list

# 或查看 pip 配置
pip config get global.index-url
```

## 如果仍然很慢

1. **检查网络连接**：确保网络连接正常
2. **尝试其他镜像源**：不同镜像源的下载速度可能不同
3. **使用代理**：如果有代理，可以配置 pip 使用代理
4. **分步安装**：先安装核心依赖，再安装 RAGAS

```bash
# 先安装核心依赖
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple openai python-dotenv pymilvus langchain-core langchain-text-splitters

# 再安装 RAGAS（可能需要较长时间）
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ragas datasets
```

