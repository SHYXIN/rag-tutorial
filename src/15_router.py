"""
步骤 15: RAG 路由（Router）— 根据问题类型自动选择检索策略

核心思路：
  用户问题 → [Router] → 判断问题类型 → 选择最优检索策略

问题类型分类：
  A) 精确查询（包含人名、术语、ID）→ 混合检索（BM25 + 向量）
  B) 语义查询（模糊、代词、开放性问题）→ 多路检索（Query 改写）
  C) 简单查询（直接的事实性问题）→ 普通检索（够用就行）

运行: python src/15_router.py
"""

import os
import json
import ssl
import urllib.request
import re
import math
from collections import defaultdict
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from embedding import LocalBGEEmbedding

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.longcat.chat/openai")
DB_PATH = "./chroma_db"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "bge-small-zh-v1.5", "fast-bge-small-zh-v1.5")


def call_llm(messages, max_tokens=200):
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
# 问题分类器（Router）
# ============================================================
class QuestionRouter:
    """
    根据问题特征，自动选择最优检索策略

    分类规则：
    - 精确查询：包含人名、术语、ID 等精确关键词 → 混合检索
    - 语义查询：模糊、代词、开放性问题 → 多路检索
    - 简单查询：直接的事实性问题 → 普通检索
    """

    # 用于 LLM 分类的 prompt
    CLASSIFY_PROMPT = """你是一个问题分类专家。请根据用户问题的特征，判断它属于哪种类型：

A) 精确查询：包含人名、术语、ID 等精确关键词，需要精确匹配
   示例："沈令月的驸马是谁？"、"谢初的封号是什么？"

B) 语义查询：模糊、包含代词、开放性问题，需要理解语义
   示例："她为什么选他？"、"这件事的后果是什么？"

C) 简单查询：直接的事实性问题，普通检索即可
   示例："沈令月是什么身份？"、"故事发生在哪个朝代？"

请只输出 A、B 或 C，不要解释。"""

    def __init__(self, use_llm=True):
        self.use_llm = use_llm

    def _rule_based_classify(self, question: str) -> str:
        """基于规则的快速分类（不需要调 LLM）"""
        # 包含代词 → 语义查询
        pronouns = ["她", "他", "它", "那个", "这个", "他们", "她们"]
        if any(p in question for p in pronouns):
            return "B"

        # 包含精确关键词模式（人名 + 属性词）→ 精确查询
        precise_patterns = [
            r'.+(?:是谁|是什么|叫什么|的封号|的职位|的年龄|的生日)',
            r'.+(?:哪一年|哪个|哪里|多少)',
        ]
        for pattern in precise_patterns:
            if re.search(pattern, question):
                return "A"

        # 短问题（< 10 字）→ 简单查询
        if len(question) < 10:
            return "C"

        # 默认：简单查询
        return "C"

    def classify(self, question: str) -> str:
        """分类问题，返回 A/B/C"""
        if self.use_llm:
            try:
                response = call_llm([
                    {"role": "system", "content": self.CLASSIFY_PROMPT},
                    {"role": "user", "content": f"问题：{question}"}
                ], max_tokens=10)
                result = response.strip().upper()
                if result in ("A", "B", "C"):
                    return result
            except Exception:
                pass
        # 降级到规则分类
        return self._rule_based_classify(question)


# ============================================================
# BM25 检索器
# ============================================================
class BM25Retriever:
    def __init__(self, documents):
        self.documents = documents
        self.k1, self.b = 1.5, 0.75
        self.doc_tokens = []
        for doc in documents:
            text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            self.doc_tokens.append(self._tokenize(text))
        self.df = defaultdict(int)
        for tokens in self.doc_tokens:
            for t in set(tokens):
                self.df[t] += 1
        self.avg_dl = sum(len(t) for t in self.doc_tokens) / len(self.doc_tokens) if self.doc_tokens else 1
        self.N = len(self.doc_tokens)

    @staticmethod
    def _tokenize(text):
        text = re.sub(r'[^一-鿿\w]', ' ', text)
        chars = list(text.replace(' ', ''))
        bigrams = [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
        return chars + bigrams

    def _score(self, q_tokens, idx):
        tokens = self.doc_tokens[idx]
        tf = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        score = 0
        for qt in q_tokens:
            if qt not in tf:
                continue
            idf = math.log((self.N - self.df.get(qt, 0) + 0.5) / (self.df.get(qt, 0) + 0.5) + 1)
            numerator = tf[qt] * (self.k1 + 1)
            denominator = tf[qt] + self.k1 * (1 - self.b + self.b * len(tokens) / self.avg_dl)
            score += idf * numerator / denominator
        return score

    def retrieve(self, query, k=5):
        q_tokens = self._tokenize(query)
        scores = [(i, self._score(q_tokens, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self.documents[i], s) for i, s in scores[:k] if s > 0]


# ============================================================
# Query 改写器
# ============================================================
class QueryRewriter:
    PROMPT = """生成 3 个检索 query，从不同角度覆盖问题。每行一个，不要解释。考虑同义词和不同表述。"""

    def rewrite(self, query):
        resp = call_llm([
            {"role": "system", "content": self.PROMPT},
            {"role": "user", "content": f"问题：{query}"}
        ], max_tokens=100)
        return [q.strip() for q in resp.strip().split('\n') if q.strip()]


# ============================================================
# 路由 RAG
# ============================================================
class RouterRAG:
    """
    根据问题类型自动选择检索策略：
    A) 精确查询 → 混合检索（BM25 + 向量）
    B) 语义查询 → 多路检索（Query 改写）
    C) 简单查询 → 普通检索
    """

    def __init__(self):
        print("[INFO] 加载文档...")
        result = db._collection.get(include=["documents", "metadatas"])
        self.all_docs = [Document(page_content=t, metadata=m or {}) for t, m in zip(result["documents"], result["metadatas"])]
        print(f"[OK] 共 {len(self.all_docs)} 个文档块")
        self.router = QuestionRouter(use_llm=False)  # 用规则分类，更快
        self.bm25 = BM25Retriever(self.all_docs)
        self.rewriter = QueryRewriter()

    def ask(self, question):
        print(f"\n[问题] {question}")
        print("-" * 50)

        # Step 1: 路由分类
        q_type = self.router.classify(question)
        type_names = {"A": "精确查询", "B": "语义查询", "C": "简单查询"}
        print(f"  [Router] 分类: {type_names.get(q_type, '未知')} ({q_type})")

        # Step 2: 根据类型选择检索策略
        if q_type == "A":
            # 精确查询 → 混合检索
            print("  [策略] 混合检索（BM25 + 向量）")
            bm25_results = self.bm25.retrieve(question, k=5)
            vec_results = db.similarity_search_with_score(question, k=5)
            # 合并去重
            seen, docs = set(), []
            for doc, score in bm25_results:
                key = doc.page_content[:100]
                if key not in seen:
                    seen.add(key)
                    docs.append(doc)
            for doc, score in vec_results:
                key = doc.page_content[:100]
                if key not in seen:
                    seen.add(key)
                    docs.append(doc)
            print(f"  [检索] 混合检索后 {len(docs)} 个文档")

        elif q_type == "B":
            # 语义查询 → 多路检索
            print("  [策略] 多路检索（Query 改写）")
            queries = self.rewriter.rewrite(question)
            print(f"  [改写] {queries}")
            seen, docs = set(), []
            for q in queries:
                for d in db.similarity_search(q, k=3):
                    key = d.page_content[:100]
                    if key not in seen:
                        seen.add(key)
                        docs.append(d)
            print(f"  [检索] 多路检索后 {len(docs)} 个文档")

        else:
            # 简单查询 → 普通检索
            print("  [策略] 普通检索")
            docs = db.similarity_search(question, k=3)
            print(f"  [检索] 普通检索后 {len(docs)} 个文档")

        # Step 3: 生成答案
        context = "\n\n".join(f"[文档 {i+1}] {d.page_content}" for i, d in enumerate(docs[:5]))
        answer = call_llm([
            {"role": "system", "content": "你是一个专业的文学分析助手。请根据以下小说片段回答问题。"},
            {"role": "user", "content": f"小说片段:\n{context}\n\n问题: {question}\n\n请用中文回答，简洁清晰。"}
        ])
        return answer


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    rag = RouterRAG()

    test_questions = [
        "沈令月的驸马是谁？",          # A: 精确查询（包含人名+属性词）
        "她为什么选他？",              # B: 语义查询（代词）
        "沈令月是什么身份？",          # C: 简单查询（短问题）
        "谢初的封号是什么？",          # A: 精确查询
        "这件事的后果是什么？",        # B: 语义查询（模糊）
    ]

    print("=" * 60)
    print("RAG 路由测试")
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
        if raw in ("1", "2", "3", "4", "5"):
            question = test_questions[int(raw) - 1]
        else:
            question = raw

        answer = rag.ask(question)
        print(f"\n[回答]\n{answer}")
        print("\n" + "=" * 60)
