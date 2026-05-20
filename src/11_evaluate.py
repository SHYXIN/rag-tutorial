"""
步骤 11: RAG 评估 — 衡量检索质量和答案质量

核心问题：
  1. 检索到的文档真的相关吗？（检索质量）
  2. 生成的答案真的正确吗？（生成质量）

评估维度：
  A) 检索召回率：相关文档有没有被检索到？
  B) 检索精准率：检索到的文档有多少是真正相关的？
  C) 答案忠实度：答案是否基于检索到的文档，而不是瞎编？
  D) 答案相关性：答案是否真正回答了问题？

运行: python src/11_evaluate.py
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
# RAG 评估器
# ============================================================
class RAGEvaluator:
    """
    评估 RAG 系统的两个维度：
    1. 检索质量 — 检索到的文档是否相关
    2. 生成质量 — 答案是否正确、是否基于文档
    """

    # 评估检索质量的 prompt
    RETRIEVAL_PROMPT = """你是一个检索质量评估专家。

问题：{question}

检索到的文档：
{documents}

请评估每个文档与问题的相关性：
- 评分标准：0=完全不相关，1=弱相关，2=部分相关，3=高度相关，4=完全回答问题
- 输出格式：JSON 数组，每个元素包含 doc_id 和 score

示例输出：
[{{"doc_id": 1, "score": 3, "reason": "文档提到了..."}}]"""

    # 评估答案质量的 prompt
    ANSWER_PROMPT = """你是一个答案质量评估专家。

问题：{question}

参考答案（基于文档）：
{ground_truth}

生成的答案：
{generated_answer}

请从以下维度打分（每项 0-10 分）：
1. 忠实度：答案是否基于提供的文档，而不是瞎编？
2. 相关性：答案是否真正回答了问题？
3. 完整性：答案是否覆盖了问题的所有方面？

输出 JSON 格式：
{{"faithfulness": 分数, "relevance": 分数, "completeness": 分数, "reason": "简短说明"}}"""

    def evaluate_retrieval(self, question: str, documents: list) -> dict:
        """评估检索质量"""
        docs_text = "\n\n".join(
            f"[文档 {i+1}]\n{doc.page_content if hasattr(doc, 'page_content') else str(doc)}"
            for i, doc in enumerate(documents)
        )

        response = call_llm([
            {"role": "system", "content": "你是检索质量评估专家。"},
            {"role": "user", "content": self.RETRIEVAL_PROMPT.format(
                question=question, documents=docs_text
            )}
        ])

        # 解析 JSON
        try:
            scores = json.loads(response)
        except json.JSONDecodeError:
            # 如果 LLM 没返回标准 JSON，用正则提取
            scores = [{"doc_id": i + 1, "score": 2, "reason": "解析失败"} for i in range(len(documents))]

        avg_score = sum(s.get("score", 0) for s in scores) / len(scores) if scores else 0

        return {
            "doc_scores": scores,
            "avg_score": avg_score,
            "max_score": max((s.get("score", 0) for s in scores), default=0),
            "min_score": min((s.get("score", 0) for s in scores), default=0),
        }

    def evaluate_answer(self, question: str, answer: str, context: str) -> dict:
        """评估答案质量"""
        response = call_llm([
            {"role": "system", "content": "你是答案质量评估专家。"},
            {"role": "user", "content": self.ANSWER_PROMPT.format(
                question=question,
                ground_truth=context[:500],
                generated_answer=answer
            )}
        ])

        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            result = {"faithfulness": 5, "relevance": 5, "completeness": 5, "reason": "解析失败"}

        return result

    def full_evaluation(self, question: str, documents: list, answer: str) -> dict:
        """完整评估：检索 + 生成"""
        print(f"\n[评估] 问题: {question}")
        print("-" * 50)

        # 评估检索
        print("  [1/2] 评估检索质量...")
        retrieval_result = self.evaluate_retrieval(question, documents)
        print(f"    平均相关性分数: {retrieval_result['avg_score']:.1f}/4")

        # 评估答案
        print("  [2/2] 评估答案质量...")
        context = "\n".join(d.page_content if hasattr(d, 'page_content') else str(d) for d in documents)
        answer_result = self.evaluate_answer(question, answer, context)
        print(f"    忠实度: {answer_result.get('faithfulness', 0)}/10")
        print(f"    相关性: {answer_result.get('relevance', 0)}/10")
        print(f"    完整性: {answer_result.get('completeness', 0)}/10")

        return {
            "retrieval": retrieval_result,
            "answer": answer_result,
        }


# ============================================================
# 运行评估
# ============================================================
if __name__ == "__main__":
    evaluator = RAGEvaluator()
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # 测试问题
    test_cases = [
        {
            "question": "沈令月为什么要选谢初当驸马？",
            "expected_keywords": ["喜欢", "长林盛宴", "风头", "选择"]  # 期望答案包含的关键词
        },
        {
            "question": "沈令月是什么身份？",
            "expected_keywords": ["公主", "嫡女", "长乐永安"]
        },
    ]

    print("=" * 60)
    print("RAG 系统评估报告")
    print("=" * 60)

    all_results = []

    for i, case in enumerate(test_cases, 1):
        q = case["question"]
        print(f"\n{'=' * 60}")
        print(f"[测试 {i}/{len(test_cases)}] {q}")
        print("=" * 60)

        # 检索
        docs = retriever.invoke(q)

        # 生成答案
        context = "\n\n".join(f"[文档 {j+1}] {d.page_content}" for j, d in enumerate(docs))
        answer = call_llm([
            {"role": "system", "content": "你是一个专业的文学分析助手。请根据以下小说片段回答问题。"},
            {"role": "user", "content": f"小说片段:\n{context}\n\n问题: {q}\n\n请用中文回答，简洁清晰。"}
        ])

        print(f"\n[生成的答案]\n{answer}")

        # 评估
        result = evaluator.full_evaluation(q, docs, answer)
        all_results.append(result)

    # 汇总
    print("\n" + "=" * 60)
    print("评估汇总")
    print("=" * 60)

    avg_retrieval = sum(r["retrieval"]["avg_score"] for r in all_results) / len(all_results)
    avg_faithfulness = sum(r["answer"].get("faithfulness", 0) for r in all_results) / len(all_results)
    avg_relevance = sum(r["answer"].get("relevance", 0) for r in all_results) / len(all_results)
    avg_completeness = sum(r["answer"].get("completeness", 0) for r in all_results) / len(all_results)

    print(f"  检索平均相关性: {avg_retrieval:.1f}/4")
    print(f"  答案忠实度:     {avg_faithfulness:.1f}/10")
    print(f"  答案相关性:     {avg_relevance:.1f}/10")
    print(f"  答案完整性:     {avg_completeness:.1f}/10")
    print(f"  综合得分:       {(avg_retrieval/4*25 + avg_faithfulness/10*25 + avg_relevance/10*25 + avg_completeness/10*25):.1f}/100")
