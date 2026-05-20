"""
步骤 9: 对比三种检索策略

普通检索 vs 多路检索 vs HyDE

运行: python src/09_compare.py
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


def call_llm(messages, max_tokens=300):
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
    return json.loads(resp.read().decode('utf-8'))['choices'][0]['message']['content']


embed_model = LocalBGEEmbedding(MODEL_DIR)
db = Chroma(persist_directory=DB_PATH, embedding_function=embed_model, collection_name="dinv_mingzhu")

question = "沈令月为什么要选谢初当驸马？"

print("=" * 60)
print(f"[问题] {question}")
print("=" * 60)

# 策略 1: 普通检索
print("\n【策略 1: 普通检索】")
print("-" * 40)
docs_basic = db.similarity_search(question, k=3)
for i, doc in enumerate(docs_basic, 1):
    print(f"  [{i}] {doc.page_content[:80]}...")

# 策略 2: 多路检索
print("\n【策略 2: 多路检索】")
print("-" * 40)
rewriter_prompt = """生成 3 个检索 query，每行一个，不要编号解释。"""
rewritten = call_llm([
    {"role": "system", "content": rewriter_prompt},
    {"role": "user", "content": f"问题：{question}"}
])
queries = [q.strip() for q in rewritten.strip().split('\n') if q.strip()]
print(f"  改写 query: {queries}")

seen = set()
multi_docs = []
for q in queries:
    for doc in db.similarity_search(q, k=3):
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            multi_docs.append(doc)
print(f"  合并去重后: {len(multi_docs)} 个文档")
for i, doc in enumerate(multi_docs, 1):
    print(f"  [{i}] {doc.page_content[:80]}...")

# 策略 3: HyDE
print("\n【策略 3: HyDE】")
print("-" * 40)
hyde_prompt = """你是文学分析专家。根据问题生成一段 100 字的"假答案"，模拟小说中的表述。只输出文字。"""
hypo = call_llm([
    {"role": "system", "content": hyde_prompt},
    {"role": "user", "content": f"问题：{question}"}
])
print(f"  假答案: {hypo[:80]}...")

hypo_emb = embed_model.embed_query(hypo)
results = db._collection.query(query_embeddings=[hypo_emb], n_results=3)
print(f"  检索结果:")
for i, (doc, dist) in enumerate(zip(results['documents'][0], results['distances'][0])):
    print(f"  [{i+1}] (距离: {dist:.3f}) {doc[:80]}...")

print("\n" + "=" * 60)
print("[总结]")
print(f"  普通检索: {len(docs_basic)} 个文档")
print(f"  多路检索: {len(multi_docs)} 个文档（覆盖面更广）")
print(f"  HyDE 检索: 3 个文档（语义最接近文档表述）")
