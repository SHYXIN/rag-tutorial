"""
步骤 14: 混合检索 — BM25 关键词检索 + 向量语义检索

核心思路：
  BM25：关键词精确匹配 → 召回包含关键词的文档
  向量检索：语义相似度匹配 → 召回语义相关的文档
  合并：取并集 → Rerank 精排 → 返回 Top K

为什么需要混合？
  BM25 擅长：人名、术语、ID 等精确匹配
  向量擅长：同义词、近义词、语义相关
  混合：覆盖面最广，不漏掉任何可能相关的文档

运行: python src/14_hybrid.py
"""

import os
import json
import ssl
import urllib.request
import re
import math
from dotenv import load_dotenv
from langchain_chroma import Chroma
from embedding import LocalBGEEmbedding
from collections import defaultdict

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


# ============================================================
# BM25 检索器（简单实现）
# ============================================================
class BM25Retriever:
    """
    基于 TF-IDF 的 BM25 关键词检索

    原理：
    1. 对所有文档分词，构建词频统计
    2. 对查询分词，计算每个文档的 BM25 分数
    3. 返回分数最高的 Top K 个文档
    """

    def __init__(self, documents: list):
        """构建 BM25 索引"""
        self.documents = documents
        self.k1 = 1.5   # BM25 参数
        self.b = 0.75   # BM25 参数

        # 对所有文档分词
        self.doc_tokens = []
        for doc in documents:
            text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            tokens = self._tokenize(text)
            self.doc_tokens.append(tokens)

        # 计算文档频率（DF）
        self.df = defaultdict(int)
        for tokens in self.doc_tokens:
            seen = set(tokens)
            for token in seen:
                self.df[token] += 1

        # 计算平均文档长度
        self.avg_dl = sum(len(t) for t in self.doc_tokens) / len(self.doc_tokens) if self.doc_tokens else 1
        self.N = len(self.doc_tokens)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """简单中文分词（按字符 + 2-gram）"""
        text = re.sub(r'[^一-鿿\w]', ' ', text)
        chars = list(text.replace(' ', ''))
        # 2-gram
        bigrams = [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
        return chars + bigrams

    def _bm25_score(self, query_tokens: list[str], doc_idx: int) -> float:
        """计算单个文档的 BM25 分数"""
        tokens = self.doc_tokens[doc_idx]
        doc_len = len(tokens)

        if doc_len == 0:
            return 0

        # 词频
        tf = defaultdict(int)
        for t in tokens:
            tf[t] += 1

        score = 0
        for qt in query_tokens:
            if qt not in tf:
                continue
            # IDF
            idf = max(0, math.log((self.N - self.df.get(qt, 0) + 0.5) / (self.df.get(qt, 0) + 0.5) + 1))
            # TF
            tf_val = tf[qt]
            numerator = tf_val * (self.k1 + 1)
            denominator = tf_val + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl)
            score += idf * numerator / denominator

        return score

    def retrieve(self, query: str, k: int = 5) -> list:
        """BM25 检索"""
        query_tokens = self._tokenize(query)

        scores = []
        for i in range(len(self.doc_tokens)):
            score = self._bm25_score(query_tokens, i)
            if score > 0:
                scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scores[:k]:
            results.append({
                "doc": self.documents[idx],
                "score": score,
                "source": "bm25"
            })
        return results


# ============================================================
# 混合检索器
# ============================================================
class HybridRetriever:
    """
    BM25 + 向量检索，合并结果
    """

    def __init__(self, db, embed_model, documents: list):
        self.db = db
        self.embed_model = embed_model
        self.bm25 = BM25Retriever(documents)

    def retrieve(self, query: str, k: int = 5, bm25_weight: float = 0.3) -> list:
        """
        混合检索：
        1. BM25 检索 Top K
        2. 向量检索 Top K
        3. 合并去重
        4. 加权排序
        """
        # BM25 检索
        bm25_results = self.bm25.retrieve(query, k=k)
        bm25_docs = {r["doc"].page_content[:100]: r for r in bm25_results}

        # 向量检索
        vector_results = self.db.similarity_search_with_score(query, k=k)
        vector_docs = {}
        for doc, score in vector_results:
            key = doc.page_content[:100]
            vector_docs[key] = {"doc": doc, "score": score, "source": "vector"}

        # 合并
        all_keys = set(bm25_docs.keys()) | set(vector_docs.keys())
        merged = []

        for key in all_keys:
            if key in bm25_docs and key in vector_docs:
                # 两种检索都命中 → 加权合并
                combined_score = (
                    bm25_weight * bm25_docs[key]["score"] +
                    (1 - bm25_weight) * (1 - vector_docs[key]["score"])  # 向量距离转相似度
                )
                merged.append({
                    "doc": bm25_docs[key]["doc"],
                    "score": combined_score,
                    "source": "both"
                })
            elif key in bm25_docs:
                merged.append({
                    "doc": bm25_docs[key]["doc"],
                    "score": bm25_docs[key]["score"] * bm25_weight,
                    "source": "bm25"
                })
            else:
                merged.append({
                    "doc": vector_docs[key]["doc"],
                    "score": (1 - vector_docs[key]["score"]) * (1 - bm25_weight),
                    "source": "vector"
                })

        merged.sort(key=lambda x: x["score"], reverse=True)
        return merged[:k]


# ============================================================
# 测试：对比三种检索方式
# ============================================================
if __name__ == "__main__":
    print("[INFO] 加载文档...")
    all_docs = []
    # 从 Chroma 加载所有文档
    collection = db._collection
    result = collection.get(include=["documents", "metadatas"])
    from langchain_core.documents import Document
    for i, (text, meta) in enumerate(zip(result["documents"], result["metadatas"])):
        all_docs.append(Document(page_content=text, metadata=meta or {}))

    print(f"[OK] 共 {len(all_docs)} 个文档块")

    retriever = HybridRetriever(db, embed_model, all_docs)

    test_questions = [
        "沈令月的驸马是谁？",
        "谢初的封号是什么？",
        "沈令月和顾大人的关系？",
    ]

    for q in test_questions:
        print(f"\n{'=' * 60}")
        print(f"[问题] {q}")
        print("=" * 60)

        # 普通向量检索
        print("\n【向量检索】")
        vec_results = db.similarity_search(q, k=3)
        for i, doc in enumerate(vec_results, 1):
            print(f"  [{i}] {doc.page_content[:80]}...")

        # 混合检索
        print("\n【混合检索 (BM25 + 向量)】")
        hybrid_results = retriever.retrieve(q, k=3)
        for i, r in enumerate(hybrid_results, 1):
            src = r['source']
            score = r['score']
            content = r['doc'].page_content[:80]
            print(f"  [{i}] ({src}, {score:.3f}) {content}...")
