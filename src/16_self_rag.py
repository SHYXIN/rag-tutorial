"""
步骤 16: Self-RAG — LLM 自己判断要不要检索、结果够不够

核心思路：
  问题 → [要不要检索？] → 检索 → [结果够不够？] → 生成 → [答案对不对？] → 返回

三种反思 token：
  [Retrieve]     = 需要检索
  [No Retrieve]  = 不需要检索，直接回答
  [Is Relevant]  = 检索结果相关，可以用
  [Not Relevant] = 检索结果不相关，需要重新检索
  [Is Supported] = 答案基于文档，可信
  [Not Supported]= 答案没有文档支持，可能幻觉

运行: python src/16_self_rag.py
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


# ============================================================
# Self-RAG
# ============================================================
class SelfRAG:
    """
    带自我反思的 RAG

    流程：
    1. 判断是否需要检索
    2. 如果需要，检索文档
    3. 判断检索结果是否相关
    4. 如果相关，基于文档生成答案
    5. 判断答案是否基于文档（防止幻觉）
    6. 如果不可信，重新生成
    """

    # Step 1: 判断是否需要检索
    NEED_RETRIEVE_PROMPT = """判断回答以下问题是否需要检索外部知识库。

如果需要检索（问题涉及具体事实、数据、特定领域的知识），输出 [Retrieve]
如果不需要（问题是常识、闲聊、主观意见），输出 [No Retrieve]

只输出 [Retrieve] 或 [No Retrieve]，不要解释。"""

    # Step 3: 判断检索结果是否相关
    RELEVANCE_PROMPT = """判断以下文档片段是否与问题相关。

问题：{question}

文档：{document}

如果文档包含回答问题所需的信息，输出 [Is Relevant]
如果文档与问题无关，输出 [Not Relevant]

只输出 [Is Relevant] 或 [Not Relevant]，不要解释。"""

    # Step 5: 判断答案是否基于文档
    SUPPORTED_PROMPT = """判断以下答案是否有文档支持。

文档：{context}

答案：{answer}

如果答案中的关键事实可以在文档中找到依据，输出 [Is Supported]
如果答案包含文档中没有的信息（可能是幻觉），输出 [Not Supported]

只输出 [Is Supported] 或 [Not Supported]，不要解释。"""

    def __init__(self, max_retrieve_rounds=2):
        self.max_retrieve_rounds = max_retrieve_rounds

    def _need_retrieve(self, question: str) -> bool:
        """Step 1: 判断是否需要检索"""
        response = call_llm([
            {"role": "system", "content": self.NEED_RETRIEVE_PROMPT},
            {"role": "user", "content": f"问题：{question}"}
        ], max_tokens=10)
        return "[Retrieve]" in response

    def _is_relevant(self, question: str, document: str) -> bool:
        """Step 3: 判断文档是否相关"""
        response = call_llm([
            {"role": "system", "content": self.RELEVANCE_PROMPT.format(
                question=question, document=document[:300]
            )},
            {"role": "user", "content": "判断相关性。"}
        ], max_tokens=10)
        return "[Is Relevant]" in response

    def _is_supported(self, context: str, answer: str) -> bool:
        """Step 5: 判断答案是否有文档支持"""
        response = call_llm([
            {"role": "system", "content": self.SUPPORTED_PROMPT.format(
                context=context[:500], answer=answer
            )},
            {"role": "user", "content": "判断是否有文档支持。"}
        ], max_tokens=10)
        return "[Is Supported]" in response

    def ask(self, question: str) -> str:
        print(f"\n[问题] {question}")
        print("-" * 50)

        # Step 1: 判断是否需要检索
        print("  [Step 1] 判断是否需要检索...")
        if not self._need_retrieve(question):
            print("  → [No Retrieve] 直接回答")
            answer = call_llm([
                {"role": "system", "content": "你是一个专业的助手。"},
                {"role": "user", "content": question}
            ])
            return answer
        print("  → [Retrieve] 需要检索")

        # Step 2: 检索（可能多轮）
        context = ""
        for round_num in range(self.max_retrieve_rounds):
            print(f"  [Step 2] 检索（第 {round_num + 1} 轮）...")
            docs = db.similarity_search(question, k=3)

            # Step 3: 判断相关性
            print("  [Step 3] 判断检索结果相关性...")
            relevant_docs = []
            for doc in docs:
                if self._is_relevant(question, doc.page_content):
                    relevant_docs.append(doc)
                    print(f"    [OK] 相关: {doc.page_content[:50]}...")
                else:
                    print(f"    [--] 不相关: {doc.page_content[:50]}...")

            if relevant_docs:
                context = "\n\n".join(d.page_content for d in relevant_docs)
                break
            elif round_num < self.max_retrieve_rounds - 1:
                print("  → 结果不相关，重新检索...")
            else:
                print("  → 多轮检索后仍无相关文档，使用最后一次结果")
                context = "\n\n".join(d.page_content for d in docs)

        # Step 4: 生成答案
        print("  [Step 4] 生成答案...")
        answer = call_llm([
            {"role": "system", "content": "你是一个专业的文学分析助手。请根据以下小说片段回答问题。只基于文档内容回答，不要添加文档中没有的信息。"},
            {"role": "user", "content": f"小说片段:\n{context}\n\n问题: {question}"}
        ])

        # Step 5: 检查答案是否有文档支持
        print("  [Step 5] 检查答案是否有文档支持...")
        if self._is_supported(context, answer):
            print("  → [Is Supported] 答案可信")
        else:
            print("  → [Not Supported] 答案可能有幻觉，重新生成...")
            answer = call_llm([
                {"role": "system", "content": "你是一个专业的文学分析助手。请严格根据以下小说片段回答问题。不要添加任何文档中没有的信息。如果文档中没有相关信息，请如实说明。"},
                {"role": "user", "content": f"小说片段:\n{context}\n\n问题: {question}\n\n注意：只回答文档中明确提到的内容。"}
            ])

        return answer


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    rag = SelfRAG(max_retrieve_rounds=2)

    test_questions = [
        "沈令月的驸马是谁？",          # 需要检索
        "1+1等于几？",                 # 不需要检索（常识）
        "谢初在长林盛宴中表现如何？",  # 需要检索
        "今天天气怎么样？",            # 不需要检索（LLM 不知道）
    ]

    print("=" * 60)
    print("Self-RAG 测试")
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
        if raw in ("1", "2", "3", "4"):
            question = test_questions[int(raw) - 1]
        else:
            question = raw

        answer = rag.ask(question)
        print(f"\n[回答]\n{answer}")
        print("\n" + "=" * 60)
