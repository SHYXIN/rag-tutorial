"""
步骤 12: 对话记忆 — 多轮对话的上下文管理

核心挑战：
  1. 指代消解：用户说"她"、"那个"，LLM 需要知道指什么
  2. 上下文窗口：对话越来越长，LLM 装不下
  3. 信息压缩：选择性保留关键信息

解决方案（三种结合）：
  A) 滑动窗口：保留最近 N 轮对话
  B) 摘要压缩：把早期对话总结成摘要
  C) 实体追踪：提取关键实体，构建"人物关系图"

运行: python src/12_memory.py
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
# 对话记忆管理器
# ============================================================
class ConversationMemory:
    """
    三层记忆架构：
    1. 短期记忆（滑动窗口）：最近 N 轮对话原文
    2. 中期记忆（摘要压缩）：早期对话的摘要
    3. 长期记忆（实体追踪）：关键实体和关系
    """

    def __init__(self, window_size: int = 3):
        self.window_size = window_size      # 滑动窗口大小
        self.history = []                   # 完整对话历史 [(role, content), ...]
        self.summary = ""                   # 早期对话摘要
        self.entities = OrderedDict()       # 实体追踪 {实体名: 描述}

    def add(self, role: str, content: str):
        """添加一轮对话"""
        self.history.append({"role": role, "content": content})

        # 如果是用户消息，提取实体
        if role == "user":
            self._extract_entities(content)

    def _extract_entities(self, text: str):
        """简单实体提取：找可能的人名、地名等（中文 2-4 字）"""
        # 常见中文人名模式
        patterns = [
            r'[一-鿿]{2,4}(?:将军|大人|公主|殿下|陛下|皇后)',
            r'[一-鿿]{2,4}(?:王|侯|公|伯)',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for m in matches:
                if m not in self.entities:
                    self.entities[m] = "文中人物"

    def _summarize_old_history(self):
        """把超出窗口的早期对话压缩成摘要"""
        if len(self.history) <= self.window_size * 2:
            return  # 对话还不需要压缩

        # 取窗口之前的历史
        old_history = self.history[:-self.window_size * 2]
        if not old_history:
            return

        # 让 LLM 总结
        history_text = "\n".join(
            f"{'用户' if h['role'] == 'user' else '助手'}: {h['content']}"
            for h in old_history
        )

        self.summary = call_llm([
            {"role": "system", "content": "总结以下对话的关键信息，50字以内。"},
            {"role": "user", "content": history_text}
        ], max_tokens=100)

        # 压缩：只保留窗口内的对话
        self.history = self.history[-self.window_size * 2:]
        print(f"[记忆压缩] 早期对话已压缩为摘要: {self.summary[:50]}...")

    def get_context(self) -> str:
        """获取构建 prompt 所需的上下文"""
        parts = []

        # 长期记忆：实体关系
        if self.entities:
            entity_desc = "、".join(
                f"{name}（{desc}）" for name, desc in list(self.entities.items())[:10]
            )
            parts.append(f"[已知人物] {entity_desc}")

        # 中期记忆：摘要
        if self.summary:
            parts.append(f"[对话摘要] {self.summary}")

        # 短期记忆：最近对话
        recent = self.history[-self.window_size * 2:]
        if recent:
            recent_text = "\n".join(
                f"{'用户' if h['role'] == 'user' else '助手'}: {h['content']}"
                for h in recent
            )
            parts.append(f"[最近对话]\n{recent_text}")

        return "\n\n".join(parts)

    def resolve_references(self, question: str) -> str:
        """
        简单的指代消解：把"她"、"他"、"那个"替换为具体实体
        """
        # 如果问题很短且包含代词，尝试替换
        pronouns = {"她": None, "他": None, "那个": None, "这个": None}

        # 从实体列表中找到最可能的人名
        person_entities = [e for e, d in self.entities.items() if "人物" in d or "将军" in d or "公主" in d or "大人" in d]

        resolved = question
        for pronoun in pronouns:
            if pronoun in resolved and person_entities:
                # 用最近提到的实体替换
                resolved = resolved.replace(pronoun, person_entities[0], 1)

        if resolved != question:
            print(f"[指代消解] '{question}' → '{resolved}'")

        return resolved


# ============================================================
# 带对话记忆的 RAG
# ============================================================
class MemoryRAG:
    def __init__(self):
        print("[INFO] 加载模型...")
        self.embed_model = LocalBGEEmbedding(MODEL_DIR)
        self.db = Chroma(persist_directory=DB_PATH, embedding_function=self.embed_model, collection_name="dinv_mingzhu")
        print(f"[OK] 数据库共 {self.db._collection.count()} 个文档块")

        self.retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        self.memory = ConversationMemory(window_size=3)

    def ask(self, question: str) -> str:
        print(f"\n[用户] {question}")
        print("-" * 50)

        # Step 1: 指代消解
        resolved_question = self.memory.resolve_references(question)

        # Step 2: 检索相关文档
        docs = self.retriever.invoke(resolved_question)
        context = "\n\n".join(f"[文档 {i+1}] {d.page_content}" for i, d in enumerate(docs))

        # Step 3: 构建包含对话历史的 prompt
        conversation_context = self.memory.get_context()

        messages = [
            {"role": "system", "content": f"""你是一个专业的文学分析助手。

{conversation_context}

请根据以下小说片段回答用户的问题。如果上下文中没有相关信息，请如实说。"""},
            {"role": "user", "content": f"小说片段:\n{context}\n\n问题: {question}"}
        ]

        # Step 4: 生成答案
        answer = call_llm(messages, max_tokens=400)

        # Step 5: 更新记忆
        self.memory.add("user", question)
        self.memory.add("assistant", answer)

        # Step 6: 检查是否需要压缩
        self.memory._summarize_old_history()

        return answer


# ============================================================
# 测试：模拟多轮对话
# ============================================================
if __name__ == "__main__":
    rag = MemoryRAG()

    # 模拟多轮对话
    conversations = [
        # 第 1 轮：建立实体
        "沈令月是什么身份？",

        # 第 2 轮：指代消解（"她" = 沈令月）
        "她为什么要选谢初当驸马？",

        # 第 3 轮：指代消解（"他" = 谢初）
        "他是什么样的人？",

        # 第 4 轮：跨轮次推理
        "沈令月和顾大人的关系是什么？",

        # 第 5 轮：依赖前文
        "她最后的选择是什么？",
    ]

    print("=" * 60)
    print("多轮对话测试（模拟 5 轮对话）")
    print("=" * 60)

    for i, question in enumerate(conversations, 1):
        print(f"\n{'=' * 60}")
        print(f"[第 {i} 轮]")
        print("=" * 60)

        answer = rag.ask(question)
        print(f"\n[回答]\n{answer}")

        # 显示当前记忆状态
        print(f"\n[记忆状态]")
        print(f"  对话历史: {len(rag.memory.history)} 轮")
        print(f"  已知实体: {list(rag.memory.entities.keys())}")
        if rag.memory.summary:
            print(f"  对话摘要: {rag.memory.summary[:60]}...")
