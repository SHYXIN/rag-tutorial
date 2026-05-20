# Self-RAG（自我反思的检索增强生成）

## 问题背景

普通 RAG 的问题：

1. **过度检索**：用户问"1+1=？"，也要去检索一遍
2. **检索不够**：第一次检索结果不相关，但不会再检索
3. **幻觉**：LLM 可能不基于文档，自己瞎编

## 核心思路

**让 LLM 自己判断：要不要检索？结果够不够？答案对不对？**

```
问题 → [要不要检索？] → 是 → 检索 → [结果够不够？] → 是 → 生成 → [答案对不对？] → 返回
                ↓ 否                                    ↓ 不够              ↓ 不对
           直接生成答案                          再次检索           重新生成
```

## 三种反思 Token

| Token | 含义 | 作用 |
|-------|------|------|
| `[Retrieve]` | 需要检索 | 问题涉及外部知识 |
| `[No Retrieve]` | 不需要检索 | 常识/闲聊/主观问题 |
| `[Is Relevant]` | 检索结果相关 | 文档包含答案所需信息 |
| `[Not Relevant]` | 检索结果不相关 | 文档与问题无关 |
| `[Is Supported]` | 答案有文档支持 | 答案基于文档，可信 |
| `[Not Supported]` | 答案无文档支持 | 可能有幻觉，需重新生成 |

## 实现

```python
class SelfRAG:
    def ask(self, question):
        # Step 1: 判断是否需要检索
        if not self._need_retrieve(question):
            return self._direct_answer(question)  # 直接回答

        # Step 2: 检索（可能多轮）
        for round in range(self.max_rounds):
            docs = self.retrieve(question)

            # Step 3: 判断相关性
            relevant = [d for d in docs if self._is_relevant(question, d)]
            if relevant:
                break

        # Step 4: 生成答案
        answer = self._generate(relevant, question)

        # Step 5: 检查是否有文档支持
        if not self._is_supported(relevant, answer):
            answer = self._regenerate(relevant, question)  # 重新生成

        return answer
```

## 流程详解

### Step 1: 判断是否需要检索

```
问题："1+1等于几？"
→ [No Retrieve]（常识，不需要检索）
→ 直接回答："1+1=2"

问题："沈令月的驸马是谁？"
→ [Retrieve]（需要外部知识）
→ 继续检索
```

### Step 2-3: 检索 + 相关性判断

```
检索到 3 个文档：
  文档1: "沈令月选谢初为驸马" → [Is Relevant] ✓
  文档2: "今天天气很好"       → [Not Relevant] ✗
  文档3: "谢初是昭武将军"     → [Is Relevant] ✓

如果全部不相关 → 重新检索（最多 N 轮）
```

### Step 4-5: 生成 + 幻觉检测

```
生成的答案："沈令月的驸马是谢初，他是当朝宰相"
→ 检查：文档中有"谢初"、"驸马" → [Is Supported] ✓
→ 但"当朝宰相"文档中没有 → [Not Supported] ✗
→ 重新生成，更严格地基于文档
```

## 适用场景

- 对答案质量要求极高（医疗、法律、金融）
- 知识库内容复杂，检索结果质量不稳定
- 需要防止 LLM 幻觉

## 代价

- 每次问答需要多次调用 LLM（3-5 次）
- 延迟增加（比普通 RAG 慢 2-3 倍）
- 成本增加（token 消耗更多）

## 权衡

| 方案 | 速度 | 精度 | 成本 | 适用 |
|------|------|------|------|------|
| 普通 RAG | 快 | 中 | 低 | 一般场景 |
| Self-RAG | 慢 | 高 | 高 | 高质量要求 |
| 混合 | 中 | 中高 | 中 | 推荐 |

**建议**：用 RAG 路由（Router）判断，简单问题用普通 RAG，复杂问题用 Self-RAG。
