# RAG 系统学习文档

> 从零开始搭建检索增强生成（RAG）系统的完整学习笔记

## 文档目录

### 基础篇
- [01 RAG 基础概念](./01_RAG基础概念.md) — 什么是 RAG、完整流程、核心组件
- [02 Embedding 与向量化](./02_Embedding与向量化.md) — 文本转向量、双塔架构、相似度计算
- [03 文档切块](./03_文档切块.md) — 切块策略、块大小、重叠的重要性
- [04 向量数据库](./04_向量数据库.md) — Chroma、HNSW 索引、ANN 搜索
- [05 微调 vs RAG](./05_微调vsRAG.md) — 两种方案的区别、适用场景、LoRA

### 进阶篇
- [06 多路检索](./06_多路检索.md) — Query 改写、多 query 检索、合并去重
- [07 HyDE 检索](./07_HyDE检索.md) — 用"假答案"提升检索精度
- [08 混合检索](./08_混合检索.md) — BM25 + 向量、取并集、加权排序
- [09 上下文压缩](./09_上下文压缩.md) — 分数过滤 + LLM 精选
- [10 RAG 评估](./10_RAG评估.md) — 检索相关性、答案忠实度、相关性、完整性
- [11 对话记忆](./11_对话记忆.md) — 滑动窗口、摘要压缩、实体追踪、指代消解
- [12 RAG 路由](./12_RAG路由.md) — 根据问题类型自动选择检索策略
- [13 Self-RAG](./13_Self-RAG.md) — LLM 自我反思：要不要检索、结果够不够、答案对不对

### 实战篇
- [搭建经验](./搭建经验.md) — 踩坑记录、解决方案、技术选型

## 项目结构

```
rag-tutorial/
├── data/               # 文档数据
├── models/             # 本地模型（BGE ONNX）
├── src/                # 源代码
│   ├── embedding.py           # 公共 Embedding 模块
│   ├── 01_ingest.py           # 文档导入
│   ├── 02_query.py           # 基础 RAG 问答
│   ├── 04_query_rewrite.py   # Query 改写
│   ├── 05_multi_query.py      # 多路检索
│   ├── 06_hyde.py             # HyDE 检索
│   ├── 07_rerank.py           # Rerank 精排
│   ├── 08_interactive.py      # 交互式问答
│   ├── 09_compare.py          # 三种检索对比
│   ├── 10_compress.py         # 上下文压缩
│   ├── 11_evaluate.py         # RAG 评估
│   ├── 12_memory.py           # 对话记忆
│   ├── 13_full_rag.py         # 完整 RAG 系统
│   └── 14_hybrid.py           # 混合检索
├── chroma_db/          # 向量数据库（自动生成）
└── docs/               # 本文档目录
```

## 技术栈

| 组件 | 选择 |
|------|------|
| Embedding | BAAI/bge-small-zh-v1.5 (ONNX) |
| 向量数据库 | Chroma |
| LLM | LongCat-2.0-Preview |
| 框架 | LangChain |

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
echo OPENAI_API_KEY=ak_你的key > .env
echo OPENAI_BASE_URL=https://api.longcat.chat/openai >> .env

# 3. 导入文档
python src/01_ingest.py

# 4. 问答
python src/08_interactive.py
```
