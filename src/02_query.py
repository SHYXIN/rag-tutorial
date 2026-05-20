"""
步骤 2: 检索 + 生成回答（完整 RAG 流程）

流程: 用户提问 → 向量化 → 检索相关文档 → 拼接 Prompt → LLM 生成回答

运行: python src/02_query.py
"""

import os
import json
import ssl
import urllib.request
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from embedding import LocalBGEEmbedding

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
DB_PATH = "./chroma_db"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "bge-small-zh-v1.5", "fast-bge-small-zh-v1.5")


def call_llm(messages, max_tokens=500):
    """调 LongCat Chat API（绕过代理 SSL 问题）"""
    payload = json.dumps({
        "model": "LongCat-2.0-Preview",
        "messages": messages,
        "max_tokens": max_tokens
    }).encode('utf-8')
    req = urllib.request.Request(
        f"{OPENAI_BASE_URL}/v1/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    )
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    resp = urllib.request.urlopen(req, context=ctx, timeout=60)
    result = json.loads(resp.read().decode('utf-8'))
    return result['choices'][0]['message']['content']


# ============================================================
# 加载模型
# ============================================================
embed_model = LocalBGEEmbedding(MODEL_DIR)

llm = ChatOpenAI(
    model="LongCat-2.0-Preview",
    temperature=0,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)
print(f"[OK] 使用 LongCat LLM")

db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embed_model,
    collection_name="dinv_mingzhu"
)
print(f"[INFO] 数据库中有 {db._collection.count()} 个文档块")

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ============================================================
# RAG 链
# ============================================================
prompt_template = """你是一个专业的助手。请根据以下上下文回答问题。
如果上下文中没有相关信息，请如实说"根据提供的资料无法回答"。

上下文:
{context}

问题: {question}

请用中文回答，简洁清晰。"""

prompt = ChatPromptTemplate.from_template(prompt_template)


def format_docs(docs):
    return "\n\n".join(f"[文档 {i+1}] {doc.page_content}" for i, doc in enumerate(docs))


# 自定义 RAG 链（用 call_llm 绕过代理）
def rag_answer(question: str) -> str:
    # 检索
    docs = retriever.invoke(question)
    context = format_docs(docs)
    # 生成
    messages = [
        {"role": "system", "content": "你是一个专业的助手。请根据以下上下文回答问题。如果上下文中没有相关信息，请如实说。"},
        {"role": "user", "content": f"上下文:\n{context}\n\n问题: {question}"}
    ]
    return call_llm(messages)

# ============================================================
# 问答循环
# ============================================================
print("\n" + "=" * 50)
print("[OK] RAG 问答系统就绪！输入你的问题（输入 q 退出）")
print("=" * 50)

while True:
    try:
        question = input("\n你的问题: ").strip()
    except EOFError:
        break
    if not question or question.lower() in ("q", "quit", "退出"):
        print("再见！")
        break

    print("\n正在检索 + 生成回答...\n")

    retrieved = retriever.invoke(question)
    print("[DOC] 检索到的相关文档:")
    for i, doc in enumerate(retrieved):
        print(f"  [{i+1}] {doc.page_content[:80]}...")

    answer = rag_answer(question)
    print(f"\n[ANS] 回答:\n{answer}")
