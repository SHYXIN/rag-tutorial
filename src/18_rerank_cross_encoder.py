"""
步骤 18: Rerank — 交叉编码器（Cross-Encoder）精排

与 LLM Rerank 的区别：
  LLM Rerank：把问题和每个文档一起发给 LLM，让它打分（慢，成本高）
  交叉编码器：用专门的 Rerank 模型，一次性对所有候选文档打分（快，精度高）

流程：
  问题 → Embedding 检索(Top 20 粗筛) → 交叉编码器 Rerank(Top 3) → LLM 生成

运行: python src/18_rerank_cross_encoder.py
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


def call_llm(messages, max_tokens=400):
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
# 交叉编码器 Reranker
# ============================================================
class CrossEncoderReranker:
    """
    用交叉编码器对候选文档进行精排

    原理：
    1. 把问题和每个候选文档组成 (question, doc) 对
    2. 输入交叉编码器模型
    3. 模型输出相关性分数
    4. 按分数排序，返回 Top K

    与 LLM Rerank 的区别：
    - LLM Rerank：调 N 次 API，慢，成本高
    - 交叉编码器：1 次推理，快，精度高
    """

    def __init__(self, model_name="BAAI/bge-reranker-v2-m3"):
        """
        加载交叉编码器模型

        模型选择：
        - BAAI/bge-reranker-v2-m3：多语言，中文效果好，开源（推荐）
        - BAAI/bge-reranker-large：更大，更准，更慢
        """
        print(f"[INFO] 加载交叉编码器模型: {model_name}")
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
            self.use_cross_encoder = True
            print("[OK] 交叉编码器加载成功")
        except ImportError:
            print("[WARN] sentence-transformers 未安装，降级为 LLM Rerank")
            print("  安装: pip install sentence-transformers")
            self.use_cross_encoder = False

    def rerank(self, question: str, documents: list, top_k: int = 3) -> list:
        """
        对候选文档进行精排

        Args:
            question: 用户问题
            documents: 候选文档列表（LangChain Document 对象）
            top_k: 返回前 K 个

        Returns:
            排序后的文档列表，每个元素为 {"content": ..., "score": ..., "rank": ...}
        """
        if not documents:
            return []

        doc_contents = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in documents]

        if self.use_cross_encoder:
            return self._rerank_with_cross_encoder(question, doc_contents, top_k)
        else:
            return self._rerank_with_llm(question, doc_contents, top_k)

    def _rerank_with_cross_encoder(self, question: str, doc_contents: list, top_k: int) -> list:
        """用交叉编码器打分"""
        # 组成 (question, doc) 对
        pairs = [(question, doc) for doc in doc_contents]

        # 一次性推理，得到所有分数
        scores = self.model.predict(pairs)

        # 按分数排序
        scored_docs = list(zip(doc_contents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i, (content, score) in enumerate(scored_docs[:top_k]):
            results.append({
                "content": content,
                "score": float(score),
                "rank": i + 1
            })

        return results

    def _rerank_with_llm(self, question: str, doc_contents: list, top_k: int) -> list:
        """降级：用 LLM 打分"""
        scored_docs = []
        for i, doc in enumerate(doc_contents):
            prompt = f"""判断以下文档与问题的相关性，打分 0-10。

问题：{question}

文档：{doc[:300]}

只输出数字，不要解释。"""

            response = call_llm([
                {"role": "system", "content": "你是相关性评估专家。"},
                {"role": "user", "content": prompt}
            ], max_tokens=10)

            try:
                score = float(response.strip())
            except ValueError:
                score = 5.0

            scored_docs.append((doc, score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i, (content, score) in enumerate(scored_docs[:top_k]):
            results.append({
                "content": content,
                "score": score,
                "rank": i + 1
            })

        return results


# ============================================================
# 完整 RAG 链（Embedding 粗筛 + 交叉编码器 Rerank + LLM 生成）
# ============================================================
class RerankRAG:
    def __init__(self):
        print("[INFO] 加载模型...")
        self.embed_model = LocalBGEEmbedding(MODEL_DIR)
        self.db = Chroma(persist_directory=DB_PATH, embedding_function=self.embed_model, collection_name="dinv_mingzhu")
        print(f"[OK] 数据库共 {self.db._collection.count()} 个文档块")
        self.reranker = CrossEncoderReranker()

    def ask(self, question: str, top_k_retrieve: int = 20, top_k_rerank: int = 3) -> str:
        print(f"\n[问题] {question}")
        print("-" * 50)

        # Step 1: Embedding 粗筛（多捞一些给 Rerank）
        print(f"[Step 1] Embedding 粗筛 (Top {top_k_retrieve})...")
        docs = self.db.similarity_search(question, k=top_k_retrieve)
        print(f"  检索到 {len(docs)} 个候选文档")

        # Step 2: 交叉编码器 Rerank
        print(f"[Step 2] 交叉编码器 Rerank (Top {top_k_rerank})...")
        reranked = self.reranker.rerank(question, docs, top_k=top_k_rerank)
        print(f"  Rerank 后的 Top {top_k_rerank}:")
        for r in reranked:
            print(f"    [{r['rank']}] 分数: {r['score']:.3f} | {r['content'][:60]}...")

        # Step 3: 拼接上下文
        context = "\n\n".join(
            f"[文档 {r['rank']}] (相关度: {r['score']:.3f})\n{r['content']}"
            for r in reranked
        )

        # Step 4: LLM 生成
        print("[Step 3] LLM 生成...")
        answer = call_llm([
            {"role": "system", "content": "你是一个专业的文学分析助手。请根据以下小说片段回答问题，简洁清晰。"},
            {"role": "user", "content": f"小说片段:\n{context}\n\n问题: {question}"}
        ])

        return answer


# ============================================================
# 对比测试：LLM Rerank vs 交叉编码器 Rerank
# ============================================================
if __name__ == "__main__":
    rag = RerankRAG()

    test_questions = [
        "沈令月的驸马是谁？",
        "谢初是什么样的人？",
        "沈令月和顾大人的关系是什么？",
    ]

    print("=" * 60)
    print("交叉编码器 Rerank 测试")
    print("=" * 60)

    for q in test_questions:
        answer = rag.ask(q)
        print(f"\n[回答]\n{answer}")
        print("\n" + "=" * 60)

    print("\n测试完成!")
