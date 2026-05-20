"""
交互式 RAG 问答（避免 Windows 管道编码问题）

运行: python src/08_interactive.py
"""

import os
import json
import ssl
import urllib.request
from dotenv import load_dotenv
from langchain_chroma import Chroma
from embedding import LocalBGEEmbedding

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.longcat.chat/openai")
DB_PATH = "./chroma_db"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "bge-small-zh-v1.5", "fast-bge-small-zh-v1.5")


def call_llm(messages, max_tokens=500):
    payload = json.dumps({
        "model": "LongCat-2.0-Preview",
        "messages": messages,
        "max_tokens": max_tokens
    }).encode('utf-8')
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    )
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    resp = urllib.request.urlopen(req, context=ctx, timeout=60)
    result = json.loads(resp.read().decode('utf-8'))
    return result['choices'][0]['message']['content']


# 加载
print("[INFO] 加载中...")
embed_model = LocalBGEEmbedding(MODEL_DIR)
db = Chroma(persist_directory=DB_PATH, embedding_function=embed_model, collection_name="dinv_mingzhu")
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
print(f"[OK] 就绪！数据库共 {db._collection.count()} 个文档块\n")

# 预置测试问题（避免 Windows 编码问题）
test_questions = [
    "沈令月为什么要选谢初当驸马？",
    "沈令月是什么身份？",
    "谢初是什么人？",
]

print("=" * 60)
print("内置测试问题（输入编号 1-3，或输入自定义问题，q 退出）")
print("=" * 60)
for i, q in enumerate(test_questions, 1):
    print(f"  {i}. {q}")

while True:
    try:
        raw = input("\n> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not raw or raw.lower() in ("q", "quit", "退出"):
        break

    # 支持输入编号
    if raw in ("1", "2", "3"):
        question = test_questions[int(raw) - 1]
    else:
        question = raw

    print(f"\n[问题] {question}")
    print("-" * 50)

    # 检索
    docs = retriever.invoke(question)
    print("[DOC] 检索到的相关文档:")
    for i, doc in enumerate(docs):
        print(f"  [{i+1}] {doc.page_content[:100]}...")

    # 生成
    context = "\n\n".join(f"[文档 {i+1}] {d.page_content}" for i, d in enumerate(docs))
    answer = call_llm([
        {"role": "system", "content": "你是一个专业的文学分析助手。请根据以下小说片段回答问题。"},
        {"role": "user", "content": f"小说片段:\n{context}\n\n问题: {question}\n\n请用中文回答，简洁清晰。"}
    ])
    print(f"\n[回答]\n{answer}")
    print("\n" + "=" * 60)
