"""
步骤 5: 多路检索 —— Query 改写 + 多 query 分别检索 + 合并去重

核心思路：
  原始问题 → LLM 改写(3-5个) → 每个分别检索 → 合并去重 → LLM 生成

运行: python src/05_multi_query.py
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


class QueryRewriter:
    SYSTEM_PROMPT = """你是一个搜索优化专家。用户会提出一个问题，你需要：
1. 理解用户的真实意图
2. 生成 3 个不同的检索 query，从不同角度覆盖这个问题
3. 每个 query 应该是简洁的、适合向量检索的短语
4. 考虑同义词和不同的表述方式

输出格式：每行一个 query，不要编号，不要解释。"""

    def rewrite(self, query: str) -> list[str]:
        response = call_llm([
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"原始问题：{query}"}
        ])
        return [q.strip() for q in response.strip().split('\n') if q.strip()]


class MultiQueryRetriever:
    def __init__(self, db, embed_model, rewriter: QueryRewriter, k_per_query: int = 3):
        self.db = db
        self.embed_model = embed_model
        self.rewriter = rewriter
        self.k_per_query = k_per_query

    def retrieve(self, query: str) -> list:
        rewritten = self.rewriter.rewrite(query)
        print(f"[Query改写] 生成 {len(rewritten)} 个检索词:")
        for i, q in enumerate(rewritten, 1):
            print(f"  {i}. {q}")

        all_docs = []
        seen = set()
        for rq in rewritten:
            results = self.db.similarity_search(rq, k=self.k_per_query)
            for doc in results:
                key = doc.page_content[:100]
                if key not in seen:
                    seen.add(key)
                    all_docs.append(doc)

        print(f"[检索结果] 合并去重后共 {len(all_docs)} 个文档")
        return all_docs


class MultiQueryRAG:
    def __init__(self):
        print("[INFO] 加载中...")
        self.embed_model = LocalBGEEmbedding(MODEL_DIR)
        self.db = Chroma(persist_directory=DB_PATH, embedding_function=self.embed_model, collection_name="dinv_mingzhu")
        print(f"[OK] 数据库共 {self.db._collection.count()} 个文档块")
        self.rewriter = QueryRewriter()
        self.retriever = MultiQueryRetriever(self.db, self.embed_model, self.rewriter)

    def ask(self, question: str) -> str:
        print(f"\n[问题] {question}")
        print("-" * 50)

        docs = self.retriever.retrieve(question)
        if not docs:
            return "未找到相关文档。"

        context = "\n\n".join(f"[文档 {i+1}] {d.page_content}" for i, d in enumerate(docs))
        answer = call_llm([
            {"role": "system", "content": "你是一个专业的文学分析助手。请根据以下小说片段回答问题。"},
            {"role": "user", "content": f"小说片段:\n{context}\n\n问题: {question}\n\n请用中文回答，简洁清晰。"}
        ])
        return answer


if __name__ == "__main__":
    rag = MultiQueryRAG()

    test_questions = [
        "沈令月为什么要选谢初当驸马？",
        "沈令月和顾大人的关系是什么？",
    ]

    print("\n" + "=" * 60)
    print("内置测试（输入编号 1-2，或自定义问题，q 退出）")
    print("=" * 60)
    for i, q in enumerate(test_questions, 1):
        print(f"  {i}. {q}")

    while True:
        try:
            raw = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw or raw.lower() in ("q", "quit"):
            break
        if raw in ("1", "2"):
            question = test_questions[int(raw) - 1]
        else:
            question = raw

        answer = rag.ask(question)
        print(f"\n[回答]\n{answer}")
        print("\n" + "=" * 60)
