"""
步骤 10: 上下文压缩 — 提取关键句，减少噪音

核心思路：
  检索到的文档(长) → 句子分割 → 分数过滤(粗筛) → LLM精选(精筛) → 压缩后的上下文

两种方式：
  A) 分数过滤：快，基于相似度打分
  B) LLM 压缩：准，基于语义理解
  C) 结合使用：先分数过滤粗筛，再 LLM 精筛（推荐）

运行: python src/10_compress.py
"""

import os
import json
import ssl
import urllib.request
import re
from dotenv import load_dotenv
from langchain_chroma import Chroma
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
# 上下文压缩器
# ============================================================
class ContextCompressor:
    """
    两阶段压缩：
    阶段 1 — 分数过滤：把文档分成句子，算每个句子与问题的相似度，保留 Top N
    阶段 2 — LLM 精选：让 LLM 从候选句子中选出最相关的 2-3 句
    """

    COMPRESS_PROMPT = """你是一个信息提取专家。用户提出一个问题，以及若干候选句子。

请从中选出与问题最相关的 1-2 个句子，直接输出这些句子原文。
不要修改句子内容，不要添加解释。如果都不相关，输出"无相关信息"。"""

    def __init__(self, top_n_filter: int = 5, top_n_llm: int = 2):
        self.top_n_filter = top_n_filter  # 分数过滤后保留的句子数
        self.top_n_llm = top_n_llm        # LLM 最终选出的句子数

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        """把文本分成句子（按句号、问号、感叹号分割）"""
        sentences = re.split(r'[。！？\n]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 5]

    def _score_filter(self, sentences: list[str], question: str) -> list[str]:
        """阶段 1：分数过滤——算每个句子与问题的相似度"""
        if len(sentences) <= self.top_n_filter:
            return sentences

        q_emb = embed_model.embed_query(question)
        scores = []
        for sent in sentences:
            s_emb = embed_model.embed_query(sent)
            # 余弦相似度
            score = sum(a * b for a, b in zip(q_emb, s_emb))
            scores.append((sent, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scores[:self.top_n_filter]]

    def _llm_select(self, candidates: list[str], question: str) -> str:
        """阶段 2：LLM 精选——从候选句子中选出最相关的"""
        if not candidates:
            return ""
        if len(candidates) <= self.top_n_llm:
            return "\n".join(candidates)

        cand_text = "\n".join(f"[{i+1}] {s}" for i, s in enumerate(candidates))
        response = call_llm([
            {"role": "system", "content": self.COMPRESS_PROMPT},
            {"role": "user", "content": f"问题：{question}\n\n候选句子：\n{cand_text}"}
        ], max_tokens=150)
        return response if response != "无相关信息" else candidates[0]

    def compress(self, documents: list, question: str) -> str:
        """压缩多个文档块"""
        compressed_parts = []

        for i, doc in enumerate(documents):
            text = doc.page_content if hasattr(doc, 'page_content') else str(doc)

            # 阶段 1：分数过滤
            sentences = self.split_sentences(text)
            candidates = self._score_filter(sentences, question)

            # 阶段 2：LLM 精选
            selected = self._llm_select(candidates, question)

            if selected:
                compressed_parts.append(f"[片段 {i+1}] {selected}")

        return "\n\n".join(compressed_parts)


# ============================================================
# 对比：压缩前 vs 压缩后
# ============================================================
if __name__ == "__main__":
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    compressor = ContextCompressor(top_n_filter=5, top_n_llm=2)

    question = "沈令月为什么要选谢初当驸马？"

    print("=" * 60)
    print(f"[问题] {question}")
    print("=" * 60)

    # 检索
    docs = retriever.invoke(question)
    print(f"\n[检索到 {len(docs)} 个文档块]")

    # 压缩前：直接拼接
    raw_context = "\n\n".join(f"[文档 {i+1}] {d.page_content}" for i, d in enumerate(docs))
    print(f"\n【压缩前】上下文长度: {len(raw_context)} 字符")
    print("-" * 40)
    for i, doc in enumerate(docs):
        print(f"  [{i+1}] {doc.page_content[:100]}...")

    # 压缩后
    compressed = compressor.compress(docs, question)
    print(f"\n【压缩后】上下文长度: {len(compressed)} 字符")
    print("-" * 40)
    print(compressed)

    # 压缩比
    ratio = len(compressed) / len(raw_context) * 100
    print(f"\n[压缩比] {ratio:.0f}%（压缩掉了 {100-ratio:.0f}% 的内容）")

    # 用压缩后的上下文生成答案
    print("\n" + "=" * 60)
    print("[用压缩后的上下文生成答案]")
    print("=" * 60)
    answer = call_llm([
        {"role": "system", "content": "你是一个专业的文学分析助手。请根据以下小说片段回答问题。"},
        {"role": "user", "content": f"小说片段:\n{compressed}\n\n问题: {question}\n\n请用中文回答，简洁清晰。"}
    ])
    print(f"\n[回答]\n{answer}")
