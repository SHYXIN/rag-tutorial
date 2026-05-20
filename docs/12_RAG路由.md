# RAG 路由（Router）

## 问题背景

不同问题适合不同的检索策略：

```
"沈令月的驸马是谁？" → 包含精确关键词 → BM25 关键词检索最准
"她为什么选他？"     → 模糊、代词     → 多路检索（Query 改写）最准
"沈令月是什么身份？" → 简单事实       → 普通检索就够用
```

**没有路由**：所有问题都用同一种策略
- 简单问题浪费资源
- 复杂问题效果差

**有路由**：根据问题类型自动选择最优策略
- 速度 + 精度 + 成本，三者兼顾

## 核心思路

```
用户问题 → [Router] → 判断问题类型 → 选择最优检索策略

问题类型：
  A) 精确查询（包含人名、术语、ID）→ 混合检索（BM25 + 向量）
  B) 语义查询（模糊、代词、开放性问题）→ 多路检索（Query 改写）
  C) 简单查询（直接的事实性问题）→ 普通检索（够用就行）
```

## 实现

### 问题分类器

```python
class QuestionRouter:
    def _rule_based_classify(self, question: str) -> str:
        """基于规则的快速分类"""
        # 包含代词 → 语义查询
        pronouns = ["她", "他", "它", "那个", "这个"]
        if any(p in question for p in pronouns):
            return "B"

        # 包含精确关键词模式 → 精确查询
        patterns = [
            r'.+(?:是谁|是什么|叫什么|的封号|的职位)',
            r'.+(?:哪一年|哪个|哪里|多少)',
        ]
        for pattern in patterns:
            if re.search(pattern, question):
                return "A"

        # 短问题 → 简单查询
        if len(question) < 10:
            return "C"

        return "C"  # 默认
```

### 路由 RAG

```python
class RouterRAG:
    def ask(self, question):
        # Step 1: 路由分类
        q_type = self.router.classify(question)

        # Step 2: 根据类型选择检索策略
        if q_type == "A":
            docs = self.hybrid_retrieve(question)   # 混合检索
        elif q_type == "B":
            docs = self.multi_query_retrieve(question)  # 多路检索
        else:
            docs = self.basic_retrieve(question)     # 普通检索

        # Step 3: 生成答案
        return self.generate(docs, question)
```

## 效果对比

```
问题 1: "沈令月的驸马是谁？"
  → 分类: 精确查询 (A)
  → 策略: 混合检索（BM25 + 向量）
  → 检索: 9 个文档

问题 2: "她为什么选他？"
  → 分类: 语义查询 (B)
  → 策略: 多路检索（Query 改写）
  → 检索: 7 个文档

问题 3: "沈令月是什么身份？"
  → 分类: 精确查询 (A)
  → 策略: 混合检索（BM25 + 向量）
  → 检索: 10 个文档
```

## 路由规则（经验总结）

| 问题特征 | 分类 | 推荐策略 |
|----------|------|----------|
| 包含人名 + 属性词（"是谁"、"的封号"） | 精确查询 | 混合检索 |
| 包含代词（"她"、"他"、"那个"） | 语义查询 | 多路检索 |
| 短问题（< 10 字） | 简单查询 | 普通检索 |
| 开放性问题（"为什么"、"怎么办"） | 语义查询 | 多路检索 |
| 包含数字/日期 | 精确查询 | 混合检索 |

## 进阶：用 LLM 做路由

规则分类快但不够灵活，可以用 LLM 做更智能的分类：

```python
CLASSIFY_PROMPT = """请根据问题特征分类：
A) 精确查询：包含人名、术语、ID，需要精确匹配
B) 语义查询：模糊、代词、开放性问题，需要理解语义
C) 简单查询：直接的事实性问题，普通检索即可

只输出 A、B 或 C。"""

response = call_llm([
    {"role": "system", "content": CLASSIFY_PROMPT},
    {"role": "user", "content": f"问题：{question}"}
])
```

**建议**：用规则分类作为默认，LLM 分类作为可选（更准但更慢）。
