"""
步骤 2: 检索 + 生成回答（完整 RAG 流程）

流程: 用户提问 → 向量化 → 检索相关文档 → 拼接 Prompt → LLM 生成回答

运行: python src/02_query.py
"""

import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ============================================================
# 配置
# ============================================================
USE_LOCAL = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"
DB_PATH = "./chroma_db"

if USE_LOCAL:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    from langchain_community.llms import Ollama
    llm = Ollama(model="llama3.2")  # 需要本地跑 Ollama
    print("✅ 使用本地模型")
else:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    print("✅ 使用 OpenAI")

# ============================================================
# 第 1 步: 加载向量数据库
# ============================================================
db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embed_model,
    collection_name="rag_demo"
)
print(f"📊 数据库中有 {db._collection.count()} 个文档块")

# ============================================================
# 第 2 步: 创建检索器
# ============================================================
retriever = db.as_retriever(
    search_type="similarity",     # 相似度搜索
    search_kwargs={"k": 3}        # 返回最相关的 3 个块
)

# 测试检索
query = "RAG 系统有哪些组件？"
print(f"\n🔍 测试检索: '{query}'")
results = retriever.invoke(query)
for i, doc in enumerate(results):
    print(f"\n--- 检索结果 {i+1} ---")
    print(doc.page_content[:120])

# ============================================================
# 第 3 步: 构建 RAG 链
# ============================================================

# Prompt 模板——告诉 LLM 怎么使用检索到的上下文
prompt_template = """你是一个专业的技术助手。请根据以下上下文回答问题。
如果上下文中没有相关信息，请如实说"根据提供的资料无法回答"。

上下文:
{context}

问题: {question}

请用中文回答，简洁清晰。"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# 格式化检索结果
def format_docs(docs):
    return "\n\n".join(f"[文档 {i+1}] {doc.page_content}" for i, doc in enumerate(docs))

# 组装 RAG 链: 检索 → 格式化 → 填 Prompt → LLM 生成 → 解析输出
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ============================================================
# 第 4 步: 提问
# ============================================================
print("\n" + "="*50)
print("🤖 RAG 问答系统就绪！输入你的问题（输入 q 退出）")
print("="*50)

while True:
    question = input("\n❓ 你的问题: ").strip()
    if question.lower() in ("q", "quit", "退出"):
        print("👋 再见！")
        break

    print("\n⏳ 正在检索 + 生成回答...\n")

    # 展示检索到的文档（让你看到 RAG 在工作）
    retrieved = retriever.invoke(question)
    print("📋 检索到的相关文档:")
    for i, doc in enumerate(retrieved):
        print(f"  [{i+1}] {doc.page_content[:80]}...")

    # 生成回答
    answer = rag_chain.invoke(question)
    print(f"\n💡 回答:\n{answer}")
