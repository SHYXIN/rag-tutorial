# RAG 实战教程

从零构建一个 RAG（检索增强生成）系统。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key（二选一）
cp .env.example .env
# 编辑 .env 填入你的 OPENAI_API_KEY
# 或者设 USE_LOCAL_EMBEDDINGS=true 用本地模型（不需要 Key）

# 3. 处理文档 + 存入向量数据库
python src/01_ingest.py

# 4. 开始问答
python src/02_query.py
```

## 项目结构

```
rag-tutorial/
├── src/
│   ├── 01_ingest.py    # 文档处理 + 向量化 + 存储
│   └── 02_query.py     # 检索 + 生成回答
├── data/               # 放你的文档（PDF/TXT）
├── chroma_db/          # 向量数据库（自动生成）
├── requirements.txt
└── .env                # API Key 配置
```

## 两种运行模式

| 模式 | 需要 | 优点 |
|------|------|------|
| OpenAI | API Key | 效果好，开箱即用 |
| 本地 | 无 | 免费，数据不出本地 |

切换方式：在 `.env` 中设 `USE_LOCAL_EMBEDDINGS=true/false`
