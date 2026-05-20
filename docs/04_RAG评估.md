# RAG 评估（Evaluation）

## 为什么需要评估？

搭好 RAG 系统后，怎么知道它好不好？

- 检索到的文档真的相关吗？
- 生成的答案真的正确吗？
- 改了一个参数后，效果变好了还是变差了？

**没有评估 = 盲人摸象。**

## 四个评估维度

### 1. 检索相关性（Retrieval Relevance）

**问：检索到的文档和问题相关吗？**

```
评分标准（0-4 分）：
  4 = 完全回答问题
  3 = 高度相关，包含问题的大部分答案
  2 = 部分相关，包含一些有用信息
  1 = 弱相关，只有少量关联
  0 = 完全不相关
```

### 2. 答案忠实度（Faithfulness）

**问：答案是基于文档的，还是 LLM 瞎编的？**

```
评分标准（0-10 分）：
  10 = 完全基于文档，没有任何编造
   5 = 大部分基于文档，有少量推断
   0 = 完全瞎编，和文档无关
```

### 3. 答案相关性（Answer Relevance）

**问：答案真正回答了吗？**

```
评分标准（0-10 分）：
  10 = 完美回答了问题的所有方面
   5 = 回答了部分问题
   0 = 答非所问
```

### 4. 答案完整性（Completeness）

**问：答案覆盖全面吗？有没有遗漏？**

```
评分标准（0-10 分）：
  10 = 覆盖了问题的所有方面
   5 = 只覆盖了部分方面
   0 = 几乎没有有用信息
```

## 实现

```python
class RAGEvaluator:
    def evaluate_retrieval(self, question, documents):
        """评估检索质量"""
        docs_text = "\n\n".join(f"[文档 {i+1}]\n{d.page_content}" for i, d in enumerate(documents))
        response = call_llm([
            {"role": "system", "content": "你是检索质量评估专家。"},
            {"role": "user", "content": f"问题：{question}\n\n文档：\n{docs_text}\n\n请对每个文档的相关性打分(0-4)。"}
        ])
        return parse_scores(response)

    def evaluate_answer(self, question, answer, context):
        """评估答案质量"""
        response = call_llm([
            {"role": "system", "content": "你是答案质量评估专家。"},
            {"role": "user", "content": f"问题：{question}\n\n文档：{context[:500]}\n\n答案：{answer}\n\n请从忠实度/相关性/完整性三个维度打分(0-10)。"}
        ])
        return parse_scores(response)
```

## 评估结果示例

```
问题：沈令月的驸马是谁？

检索评估：
  文档1: 3分（提到了沈令月选驸马）
  文档2: 4分（直接提到谢初被选为驸马）
  文档3: 1分（只提到沈令月的烦恼）
  平均: 2.7/4

答案评估：
  忠实度: 8/10（答案基于文档）
  相关性: 9/10（完美回答问题）
  完整性: 7/10（缺少谢初的背景信息）

综合得分: 73.8/100
```

## 适用场景

- 对比不同检索策略的效果
- 调优 prompt 和参数
- 监控线上 RAG 系统的质量

## 注意事项

- 评估本身也有误差（LLM 打分不一定准）
- 需要足够多的测试用例（至少 20-30 个问题）
- 最好有人工评估作为基准
