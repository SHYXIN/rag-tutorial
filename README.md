# RAG 实战教程

从零构建一个完整的 RAG（检索增强生成）系统，涵盖基础概念到进阶优化。

## 功能概览

```
基础功能
  ├── 文档导入（TXT/PDF）
  ├── 文档切块 + Embedding + 向量存储
  └── 基础 RAG 问答

进阶功能
  ├── 多路检索（Query 改写，多 query 合并）
  ├── HyDE 检索（用"假答案"提升精度）
  ├── 混合检索（BM25 关键词 + 向量语义）
  ├── 上下文压缩（分数过滤 + LLM 精选）
  ├── RAG 评估（4 个维度自动打分）
  └── 对话记忆（滑动窗口 + 摘要压缩 + 实体追踪 + 指代消解）
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env 填入你的 OPENAI_API_KEY

# 3. 导入文档
python src/01_ingest.py

# 4. 交互式问答
python src/08_interactive.py
```

## 技术栈

| 组件 | 选择 | 说明 |
|------|------|------|
| Embedding | BAAI/bge-small-zh-v1.5 (ONNX) | 中文效果好，本地运行，无需 API |
| 向量数据库 | Chroma | 轻量级，Python 原生 |
| LLM | LongCat-2.0-Preview | 兼容 OpenAPI 格式 |
| 框架 | LangChain | 组件丰富 |

## 项目结构

```
rag-tutorial/
├── data/                       # 文档数据（txt/pdf）
├── models/                     # 本地模型（BGE ONNX，手动下载）
├── chroma_db/                  # 向量数据库（自动生成）
├── docs/                       # 学习文档
│   ├── README.md               # 文档目录
│   ├── 搭建经验.md              # 踩坑记录
│   ├── 01_RAG基础概念.md
│   ├── 02_Embedding与向量化.md
│   ├── 03_文档切块.md
│   ├── 04_向量数据库.md
│   ├── 05_微调vsRAG.md
│   ├── 06_多路检索.md
│   ├── 07_HyDE检索.md
│   ├── 08_混合检索.md
│   ├── 09_上下文压缩.md
│   ├── 10_RAG评估.md
│   └── 11_对话记忆.md
├── src/
│   ├── embedding.py            # 公共 Embedding 模块（所有脚本共享）
│   ├── 01_ingest.py            # 文档导入 + 向量化 + 存储
│   ├── 02_query.py             # 基础 RAG 问答
│   ├── 03_test.py              # LongCat API 连接测试
│   ├── 04_query_rewrite.py     # Query 改写演示
│   ├── 05_multi_query.py       # 多路检索 + 对话
│   ├── 06_hyde.py              # HyDE 检索 + 对话
│   ├── 07_rerank.py            # Rerank 精排 + 对话
│   ├── 08_interactive.py       # 交互式问答（Windows 兼容）
│   ├── 09_compare.py           # 三种检索策略对比
│   ├── 10_compress.py          # 上下文压缩演示
│   ├── 11_evaluate.py          # RAG 评估系统
│   ├── 12_memory.py            # 对话记忆演示
│   ├── 13_full_rag.py          # 完整 RAG（整合所有技术）
│   └── 14_hybrid.py            # 混合检索对比
├── .env                        # API Key 配置
├── .gitignore
└── requirements.txt
```

## 使用示例

### 基础问答
```bash
python src/08_interactive.py
```

### 完整 RAG（多路检索 + 压缩 + 记忆）
```bash
python src/13_full_rag.py
```

### 评估 RAG 质量
```bash
python src/11_evaluate.py
```

### 对比不同检索策略
```bash
python src/09_compare.py
```

## 系统架构

```
用户问题
    ↓
[指代消解] 把"她"、"他"替换为具体实体
    ↓
[Query 改写] 1个问题 → 3-5个检索词
    ↓
[多路检索] BM25 + 向量，分别检索
    ↓
[合并去重] 取并集，去除重复文档
    ↓
[上下文压缩] 分数过滤(粗筛) + LLM精选(精筛)
    ↓
[构建 Prompt] 压缩上下文 + 对话记忆 + 问题
    ↓
[LLM 生成] 基于上下文生成回答
    ↓
[记忆更新] 更新对话历史 + 实体列表 + 压缩摘要
    ↓
返回答案
```

## 配置说明

编辑 `.env` 文件：

```env
# LongCat API Key
OPENAI_API_KEY=ak_你的key
OPENAI_BASE_URL=https://api.longcat.chat/openai
```

## 学习文档

详见 [docs/README.md](./docs/README.md)，包含 12 篇知识点文档：

**基础篇**
- RAG 基础概念、Embedding 与向量化、文档切块、向量数据库、微调 vs RAG

**进阶篇**
- 多路检索、HyDE 检索、混合检索、上下文压缩、RAG 评估、对话记忆

**实战篇**
- 搭建经验（踩坑记录）

## License

MIT
