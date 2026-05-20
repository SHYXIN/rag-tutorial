"""
步骤 7: Rerank —— 用交叉编码器精排检索结果

核心流程：
  问题 → Embedding 检索(Top 20 粗筛) → Rerank 精排(Top 3) → LLM 生成回答

Embedding vs Rerank 的区别：
  Embedding（双塔）：问题和文档分别编码 → 算余弦相似度 → 快但精度有限
  Rerank（交叉编码器）：问题和文档一起输入 → 深度交互 → 慢但更精准

类比：
  Embedding = 按分类标签找书（快，可能找错）
  Rerank = 翻几页确认内容真的相关（慢，但更准）

运行: python src/07_rerank.py
"""

import os
import json
import ssl
import urllib.request
from dotenv import load_dotenv
from langchain_chroma import Chroma
from tokenizers import Tokenizer
import onnxruntime as ort
import numpy as np

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.longcat.chat/openai")
DB_PATH = "./chroma_db"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "bge-small-zh-v1.5", "fast-bge-small-zh-v1.5")


# ============================================================
# 本地 BGE Embedding（粗筛用）
# ============================================================
class LocalBGEEmbedding:
    def __init__(self, model_dir: str):
        self.tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.session = ort.InferenceSession(
            os.path.join(model_dir, "model_optimized.onnx"),
            providers=["CPUExecutionProvider"]
        )

    def _embed(self, texts):
        encoded = self.tokenizer.encode_batch(texts)
        max_len = max(len(e.ids) for e in encoded)
        input_ids = np.zeros((len(encoded), max_len), dtype=np.int64)
        attention_mask = np.zeros((len(encoded), max_len), dtype=np.int64)
        token_type_ids = np.zeros((len(encoded), max_len), dtype=np.int64)
        for i, e in enumerate(encoded):
            input_ids[i, :len(e.ids)] = e.ids
            attention_mask[i, :len(e.attention_mask)] = e.attention_mask
            token_type_ids[i, :len(e.type_ids)] = e.type_ids
        outputs = self.session.run(None, {
            "input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids
        })
        last_hidden_state = outputs[0]
        mask_expanded = np.expand_dims(attention_mask, -1)
        sum_embeddings = np.sum(last_hidden_state * mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
        embeddings = sum_embeddings / sum_mask
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.clip(norms, a_min=1e-9, a_max=None)

    def embed_documents(self, texts):
        return self._embed(texts).tolist()

    def embed_query(self, text):
        return self._embed([text])[0].tolist()


# ============================================================
# LongCat LLM 调用
# ============================================================
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


# ============================================================
# 简单 Reranker —— 用 LLM 做重排
# ============================================================
class LLMReranker:
    """
    用 LLM 对候选文档逐条打分，选出最相关的 Top K

    原理：把问题和每个候选文档一起发给 LLM，让它判断相关性分数
    优点：不需要额外下载模型，用现有的 LLM 即可
    缺点：比专用 Rerank 模型慢一点（但精度更好）
    """

    RERANK_PROMPT = """你是一个相关性评估专家。用户提出一个问题，以及若干个候选文档片段。

请对每个文档片段与问题的相关性打分（0-10 分）：
- 10 分：直接回答了问题，信息完整
- 7-9 分：高度相关，包含问题的大部分答案
- 4-6 分：部分相关，包含一些有用信息
- 1-3 分：弱相关，只有少量关联
- 0 分：完全不相关

请按以下格式输出（每行一个）：
文档编号: 分数

不要输出解释。"""

    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    def rerank(self, query: str, documents: list[str]) -> list[dict]:
        # 构建候选文档列表
        doc_list = "\n\n".join(
            f"[文档 {i+1}]\n{doc}"
            for i, doc in enumerate(documents)
        )

        # 让 LLM 打分
        response = call_llm([
            {"role": "system", "content": self.RERANK_PROMPT},
            {"role": "user", "content": f"问题：{query}\n\n候选文档：\n{doc_list}"}
        ], max_tokens=200)

        # 解析分数
        scores = {}
        for line in response.strip().split('\n'):
            line = line.strip()
            if ':' in line:
                try:
                    doc_num = int(line.split(':')[0].replace('文档', '').strip())
                    score = float(line.split(':')[1].strip())
                    scores[doc_num - 1] = score
                except (ValueError, IndexError):
                    continue

        # 按分数排序
        scored_docs = []
        for i, doc in enumerate(documents):
            scored_docs.append({
                "content": doc,
                "score": scores.get(i, 0),
                "original_rank": i + 1
            })

        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:self.top_k]


# ============================================================
# 完整 RAG 链（Embedding 粗筛 + LLM Rerank 精排）
# ============================================================
class RerankRAG:
    def __init__(self):
        print("[INFO] 加载 embedding 模型...")
        self.embed_model = LocalBGEEmbedding(MODEL_DIR)
        self.db = Chroma(
            persist_directory=DB_PATH,
            embedding_function=self.embed_model,
            collection_name="dinv_mingzhu"
        )
        print(f"[OK] 数据库加载完成，共 {self.db._collection.count()} 个文档块")
        self.reranker = LLMReranker(top_k=2)

    def ask(self, question: str) -> str:
        print(f"\n[问题] {question}")
        print("-" * 50)

        # Step 1: Embedding 粗筛（多捞一些给 Rerank）
        print("[Step 1] Embedding 粗筛...")
        candidate_docs = self.db.similarity_search(question, k=5)
        print(f"  捞出 {len(candidate_docs)} 个候选文档")

        if not candidate_docs:
            return "未找到相关文档。"

        # Step 2: Rerank 精排
        print("[Step 2] Rerank 精排...")
        doc_contents = [doc.page_content for doc in candidate_docs]
        reranked = self.reranker.rerank(question, doc_contents)

        print(f"  Rerank 后的 Top {len(reranked)}:")
        for d in reranked:
            print(f"    原排名 #{d['original_rank']} → 分数 {d['score']:.1f}")

        # Step 3: 拼接上下文
        context = "\n\n".join(
            f"[文档 {i+1}] (相关性: {d['score']:.1f})\n{d['content']}"
            for i, d in enumerate(reranked)
        )

        # Step 4: LLM 生成
        answer = call_llm([
            {"role": "system", "content": "你是一个专业的技术助手。请根据以下上下文回答问题。"},
            {"role": "user", "content": f"上下文:\n{context}\n\n问题: {question}"}
        ])

        return answer


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    rag = RerankRAG()

    questions = [
        "RAG 系统有哪些核心组件？",
        "RAG 和微调的区别是什么？",
    ]

    for q in questions:
        answer = rag.ask(q)
        print(f"\n[回答]\n{answer}")
        print("\n" + "=" * 60)
