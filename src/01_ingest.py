"""
步骤 1: 文档处理 + 向量化 + 存入向量数据库

流程: 加载文档 → 切块 → 向量化 → 存入 Chroma

运行: python src/01_ingest.py
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# ============================================================
# 配置区
# ============================================================
USE_LOCAL = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"

# 选择 embedding 模型
if USE_LOCAL:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embed_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5"  # 中文效果好，本地运行
    )
    print("✅ 使用本地 embedding 模型: BAAI/bge-small-zh-v1.5")
else:
    from langchain_openai import OpenAIEmbeddings
    embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
    print("✅ 使用 OpenAI embedding 模型: text-embedding-3-small")

# 选择向量数据库
from langchain_chroma import Chroma
DB_PATH = "./chroma_db"  # 向量数据库存储位置

# ============================================================
# 第 1 步: 加载文档
# ============================================================
# 方式 A: 加载 TXT 文件
# loader = TextLoader("data/sample.txt", encoding="utf-8")

# 方式 B: 加载 PDF 文件
# loader = PyPDFLoader("data/your_document.pdf")

# 方式 C: 直接用字符串（演示用）
from langchain_core.documents import Document

sample_text = """
RAG（Retrieval-Augmented Generation，检索增强生成）是一种将检索系统与生成模型结合的技术。
它的核心思想是：在让大模型回答问题之前，先从外部知识库中检索相关信息，
然后把检索结果和问题一起提供给模型，让模型基于这些资料生成回答。

RAG 系统通常包含以下组件：
1. 文档处理器：将原始文档切分成适合检索的小块
2. 嵌入模型：将文本转换为向量表示
3. 向量数据库：存储和检索向量
4. 检索器：根据用户查询找到最相关的文档
5. 生成模型：基于检索结果生成最终回答

RAG 的优势包括：知识可以实时更新、答案有据可查、不需要微调模型、
可以处理训练数据之外的最新信息。
"""

documents = [Document(page_content=sample_text, metadata={"source": "demo"})]
print(f"\n📄 加载了 {len(documents)} 个文档")

# ============================================================
# 第 2 步: 文档切块
# ============================================================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,        # 每块最多 200 字符
    chunk_overlap=30,      # 块之间重叠 30 字符
    separators=["\n\n", "\n", "。", " ", ""]  # 按优先级切分
)

chunks = splitter.split_documents(documents)
print(f"✂️  切分成 {len(chunks)} 个块")

for i, chunk in enumerate(chunks):
    print(f"\n--- 块 {i+1} (长度: {len(chunk.page_content)}) ---")
    print(chunk.page_content[:80] + "...")

# ============================================================
# 第 3 步: 向量化 + 存入数据库
# ============================================================
db = Chroma.from_documents(
    documents=chunks,
    embedding=embed_model,
    persist_directory=DB_PATH,
    collection_name="rag_demo"
)

print(f"\n💾 已存入向量数据库 (路径: {DB_PATH})")
print(f"📊 数据库中共有 {db._collection.count()} 个向量")
print("\n✅ 步骤 1 完成！接下来运行 python src/02_query.py")
