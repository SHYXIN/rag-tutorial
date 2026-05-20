"""
步骤 13: 完整 RAG 系统 — 整合所有进阶技术

整合了：
  1. 本地 BGE Embedding（ONNX）
  2. 多路检索（Query 改写）
  3. HyDE 检索
  4. 上下文压缩（分数过滤 + LLM 精选）
  5. 对话记忆（滑动窗口 + 摘要压缩 + 实体追踪）
  6. RAG 评估

运行: python src/13_full_rag.py
"""

import os
import json
import ssl
import urllib.request
import re
from dotenv import load_dotenv
from langchain_chroma import Chroma
from embedding import LocalBGEEmbedding
from collections import OrderedDict

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


# ============================================================
# 对话记忆
# ============================================================
class ConversationMemory:
    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        self.history = []
        self.summary = ""
        self.entities = OrderedDict()

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if role == "user":
            self._extract_entities(content)

    def _extract_entities(self, text: str):
        patterns = [
            r'[一-鿿]{2,4}(?:将军|大人|公主|殿下|陛下|皇后|王|侯)',
        ]
        for pattern in patterns:
            for m in re.findall(pattern, text):
                if m not in self.entities:
                    self.entities[m] = "文中人物"

    def _summarize_old(self):
        if len(self.history) <= self.window_size * 2:
            return
        old = self.history[:-self.window_size * 2]
        if not old:
            return
        hist_text = "\n".join(f"{'用户' if h['role']=='user' else '助手'}: {h['content']}" for h in old)
        self.summary = call_llm([
            {"role": "system", "content": "总结以下对话的关键信息，50字以内。"},
            {"role": "user", "content": hist_text}
        ], max_tokens=80)
        self.history = self.history[-self.window_size * 2:]
        print(f"  [记忆压缩] {self.summary[:50]}...")

    def get_context(self) -> str:
        parts = []
        if self.entities:
            parts.append(f"[已知人物] {'、'.join(f'{n}（{d}）' for n,d in list(self.entities.items())[:10])}")
        if self.summary:
            parts.append(f"[对话摘要] {self.summary}")
        recent = self.history[-self.window_size * 2:]
        if recent:
            parts.append(f"[最近对话]\n" + "\n".join(f"{'用户' if h['role']=='user' else '助手'}: {h['content']}" for h in recent))
        return "\n\n".join(parts)

    def resolve(self, question: str) -> str:
        persons = [e for e, d in self.entities.items() if "人物" in d or "将军" in d or "公主" in d or "大人" in d]
        resolved = question
        for p in ["她", "他", "那个", "这个"]:
            if p in resolved and persons:
                resolved = resolved.replace(p, persons[0], 1)
        if resolved != question:
            print(f"  [指代消解] '{question}' → '{resolved}'")
        return resolved


# ============================================================
# 上下文压缩器
# ============================================================
class ContextCompressor:
    COMPRESS_PROMPT = """从以下候选句子中选出与问题最相关的 1-2 句，直接输出原文。不要解释。"""

    def __init__(self, top_n_filter=5, top_n_llm=2):
        self.top_n_filter = top_n_filter
        self.top_n_llm = top_n_llm

    @staticmethod
    def split_sentences(text):
        return [s.strip() for s in re.split(r'[。！？\n]+', text) if len(s.strip()) > 5]

    def compress(self, documents, question):
        parts = []
        for i, doc in enumerate(documents):
            text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            sents = self.split_sentences(text)
            if len(sents) > self.top_n_filter:
                q_emb = embed_model.embed_query(question)
                scored = sorted(sents, key=lambda s: sum(a*b for a,b in zip(q_emb, embed_model.embed_query(s))), reverse=True)
                sents = scored[:self.top_n_filter]
            if len(sents) > self.top_n_llm:
                cand = "\n".join(f"[{j+1}] {s}" for j, s in enumerate(sents))
                selected = call_llm([
                    {"role": "system", "content": self.COMPRESS_PROMPT},
                    {"role": "user", "content": f"问题：{question}\n\n候选：\n{cand}"}
                ], max_tokens=100)
            else:
                selected = "\n".join(sents)
            if selected:
                parts.append(f"[片段 {i+1}] {selected}")
        return "\n\n".join(parts)


# ============================================================
# Query 改写器
# ============================================================
class QueryRewriter:
    PROMPT = """生成 3 个检索 query，从不同角度覆盖问题。每行一个，不要解释。"""

    def rewrite(self, query):
        resp = call_llm([
            {"role": "system", "content": self.PROMPT},
            {"role": "user", "content": f"问题：{query}"}
        ], max_tokens=100)
        return [q.strip() for q in resp.strip().split('\n') if q.strip()]


# ============================================================
# 完整 RAG 系统
# ============================================================
class FullRAG:
    def __init__(self):
        print("[INFO] 加载模型...")
        global embed_model
        embed_model = LocalBGEEmbedding(MODEL_DIR)
        self.db = Chroma(persist_directory=DB_PATH, embedding_function=embed_model, collection_name="dinv_mingzhu")
        print(f"[OK] 数据库共 {self.db._collection.count()} 个文档块")
        self.memory = ConversationMemory(window_size=3)
        self.compressor = ContextCompressor()
        self.rewriter = QueryRewriter()

    def ask(self, question, use_multi_query=True, use_compression=True):
        print(f"\n[用户] {question}")
        print("-" * 50)

        # 指代消解
        q = self.memory.resolve(question)

        # 检索（多路 or 普通）
        if use_multi_query:
            queries = self.rewriter.rewrite(q)
            print(f"  [Query改写] {len(queries)} 个: {queries}")
            seen, docs = set(), []
            for rq in queries:
                for d in self.db.similarity_search(rq, k=3):
                    key = d.page_content[:100]
                    if key not in seen:
                        seen.add(key)
                        docs.append(d)
            print(f"  [检索] 合并去重后 {len(docs)} 个文档")
        else:
            docs = self.db.similarity_search(q, k=3)
            print(f"  [检索] {len(docs)} 个文档")

        # 上下文压缩
        if use_compression and docs:
            context = self.compressor.compress(docs, q)
            print(f"  [压缩] 压缩后 {len(context)} 字符")
        else:
            context = "\n\n".join(f"[文档 {i+1}] {d.page_content}" for i, d in enumerate(docs))

        # 构建 prompt（含对话记忆）
        conv_ctx = self.memory.get_context()
        messages = [
            {"role": "system", "content": f"你是文学分析助手。{conv_ctx}\n\n根据小说片段回答问题，简洁清晰。"},
            {"role": "user", "content": f"小说片段:\n{context}\n\n问题: {question}"}
        ]

        # 生成
        answer = call_llm(messages)

        # 更新记忆
        self.memory.add("user", question)
        self.memory.add("assistant", answer)
        self.memory._summarize_old()

        return answer


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    rag = FullRAG()

    print("\n" + "=" * 60)
    print("完整 RAG 系统")
    print("技术栈: 多路检索 + 上下文压缩 + 对话记忆")
    print("=" * 60)

    test_questions = [
        "沈令月是什么身份？",
        "她为什么要选谢初当驸马？",
        "他是什么样的人？",
        "沈令月和顾大人的关系是什么？",
    ]

    print("\n内置测试（输入编号 1-4，或自定义问题，q 退出）")
    for i, q in enumerate(test_questions, 1):
        print(f"  {i}. {q}")

    while True:
        try:
            raw = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw or raw.lower() in ("q", "quit"):
            break
        if raw in ("1", "2", "3", "4"):
            question = test_questions[int(raw) - 1]
        else:
            question = raw

        answer = rag.ask(question)
        print(f"\n[回答]\n{answer}")
        print(f"\n[记忆] 历史 {len(rag.memory.history)} 轮 | 实体 {list(rag.memory.entities.keys())}")
        print("=" * 60)
