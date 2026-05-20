"""
步骤 4: Query 改写 —— 把模糊问题变成精准检索词

核心思路：
  原始问题 → LLM 改写 → 多个检索 query → 合并结果

运行: python src/04_query_rewrite.py
"""

import os
import json
import ssl
import urllib.request
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.longcat.chat/openai")


def call_llm(messages: list[dict], max_tokens: int = 200) -> str:
    """调 LongCat Chat API（绕过代理 SSL 问题）"""
    payload = json.dumps({
        "model": "LongCat-2.0-Preview",
        "messages": messages,
        "max_tokens": max_tokens
    }).encode('utf-8')

    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
    )

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    resp = urllib.request.urlopen(req, context=ctx, timeout=30)
    result = json.loads(resp.read().decode('utf-8'))
    return result['choices'][0]['message']['content']


# ============================================================
# Query 改写器
# ============================================================
class QueryRewriter:
    """用 LLM 把用户的模糊问题改写成多个精准检索词"""

    SYSTEM_PROMPT = """你是一个搜索优化专家。用户会提出一个问题，你需要：
1. 理解用户的真实意图
2. 生成 3 个不同的检索 query，从不同角度覆盖这个问题
3. 每个 query 应该是简洁的、适合向量检索的短语

输出格式：每行一个 query，不要编号，不要解释。"""

    def rewrite(self, query: str) -> list[str]:
        response = call_llm([
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"原始问题：{query}"}
        ])

        # 解析输出，每行一个 query
        queries = [q.strip() for q in response.strip().split('\n') if q.strip()]
        return queries


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    rewriter = QueryRewriter()

    test_queries = [
        "那个技术怎么用？",
        "RAG 和微调选哪个？",
        "怎么让检索更准？",
    ]

    print("=" * 60)
    print("Query 改写测试")
    print("=" * 60)

    for q in test_queries:
        print(f"\n[原始] {q}")
        rewritten = rewriter.rewrite(q)
        for i, rq in enumerate(rewritten, 1):
            print(f"  [改写{i}] {rq}")
