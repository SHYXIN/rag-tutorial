# HyDE 检索（Hypothetical Document Embeddings）

## 问题背景

用户问题和文档表述之间存在"语义鸿沟"：

```
用户问："RAG 有什么缺点？"
文档写："RAG 的局限性包括..."
```

关键词完全不同，但语义高度相关。普通向量检索可能匹配不上。

## 核心思路

**不让 LLM 改写问题，而是让 LLM 假装自己是文档，直接生成一段"假答案"，然后用假答案去检索。**

```
用户问题 → LLM 生成"假答案" → 用假答案检索 → 找到真实文档
```

为什么有效？因为"假答案"的表述方式更接近文档中的实际内容。

## 实现

```python
class HyDERetriever:
    HYDE_PROMPT = """你是相关领域的专家。根据问题生成一段 100-200 字的"假答案"，
    模拟文档中可能出现的表述。只输出这段文字，不要有任何前缀或解释。"""

    def retrieve(self, query: str, k: int = 3) -> list:
        # 生成假答案
        hypo = call_llm([
            {"role": "system", "content": self.HYDE_PROMPT},
            {"role": "user", "content": f"问题：{query}"}
        ])
        # 用假答案检索
        hypo_emb = self.embed_model.embed_query(hypo)
        results = self.db._collection.query(
            query_embeddings=[hypo_emb], n_results=k
        )
        return results
```

## 效果对比

```
问题："RAG 和微调的区别是什么？"

普通检索：用 "RAG 微调 区别" 去匹配
HyDE 检索：用生成的假答案 "RAG（检索增强生成）和微调是两种不同的...RAG 通过外部知识库获取实时信息，微调通过训练数据调整模型权重..." 去匹配

→ 假答案的表述方式 ≈ 文档实际内容 → 检索精度更高
```

## 适用场景

- 用户问题和文档表述差异大（口语 vs 书面语）
- 英文场景效果最稳定
- 对检索精度要求高的场景

## 代价

- 每次检索要多调 1 次 LLM
- 比多路检索慢（但通常比多路检索准）
