"""
步骤 6: HyDE —— 用"假答案"提升检索精度

核心思路：
  用户问题 → LLM 生成"假答案" → 用假答案检索 → 找到真实文档 → LLM 生成最终回答

运行: python src/06_hyde.py
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


class HyDERetriever:
    HYDE_PROMPT = """你是一个专业的技术/文学文档写手。用户会提出一个问题，你需要：
1. 假设你是相关领域的专家
2. 生成一段 100-200 字的"假答案"，模拟文档中可能出现的表述
3. 用专业、准确的语气
4. 只输出这段文字，不要有任何前缀或解释"""

    def __init__(self, db, embed_model):
        self.db = db
        self.embed_model = embed_model

    def _generate_hypo(self, query: str) -> str:
        return call_llm([
            {"role": "system", "content": self.HYDE_PROMPT},
            {"role": "user", "content": f"问题：{query}"}
        ])

    def retrieve(self, query: str, k: int = 3) -> list:
        hypo = self._generate_hypo(query)
        print(f"[HyDE] 假答案: {hypo[:80]}...")
        hypo_emb = self.embed_model.embed_query(hypo)
        results = self.db._collection.query(query_embeddings=[hypo_emb], n_results=k)
        docs = []
        for i, (doc, dist) in enumerate(zip(results['documents'][0], results['distances'][0])):
            docs.append({"content": doc, "distance": dist, "rank": i + 1})
        return docs


class HyDERAG:
    def __init__(self):
        print("[INFO] 加载中...")
        self.embed_model = LocalBGEEmbedding(MODEL_DIR)
        self.db = Chroma(persist_directory=DB_PATH, embedding_function=self.embed_model, collection_name="dinv_mingzhu")
        print(f"[OK] 数据库共 {self.db._collection.count()} 个文档块")
        self.retriever = HyDERetriever(self.db, self.embed_model)

    def ask(self, question: str) -> str:
        print(f"\n[问题] {question}")
        print("-" * 50)
        docs = self.retriever.retrieve(question)
        if not docs:
            return "未找到相关文档。"
        context = "\n\n".join(f"[文档 {d['rank']}] (距离: {d['distance']:.3f})\n{d['content']}" for d in docs)
        return call_llm([
            {"role": "system", "content": "你是一个专业的文学分析助手。请根据以下小说片段回答问题。"},
            {"role": "user", "content": f"小说片段:\n{context}\n\n问题: {question}\n\n请用中文回答，简洁清晰。"}
        ])


if __name__ == "__main__":
    rag = HyDERAG()
    test_questions = ["沈令月为什么要选谢初当驸马？", "沈令月是什么身份？"]
    print("\n内置测试（输入编号 1-2，或自定义问题，q 退出）")
    for i, q in enumerate(test_questions, 1):
        print(f"  {i}. {q}")
    while True:
        try:
            raw = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw or raw.lower() in ("q", "quit"):
            break
        question = test_questions[int(raw) - 1] if raw in ("1", "2") else raw
        answer = rag.ask(question)
        print(f"\n[回答]\n{answer}\n{'=' * 60}")
