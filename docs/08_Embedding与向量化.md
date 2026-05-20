# Embedding 与向量化

## 什么是 Embedding？

**Embedding 是把文本转换成一串数字（向量）的过程。**

```
"沈令月是公主" → [0.12, -0.34, 0.56, ..., 0.78]  (512 维)
```

这串数字的妙处在：**语义相似的文本，它们的向量在空间里距离也近。**

```
"沈令月是公主" → [0.12, -0.34, 0.56, ...]
"沈令月是嫡女" → [0.13, -0.33, 0.55, ...]  ← 距离很近
"今天天气很好" → [0.89, 0.45, -0.12, ...]  ← 距离很远
```

## 为什么需要 Embedding？

计算机不理解文字，只理解数字。把文本转向量后，就可以：
- **计算相似度** — 余弦相似度、欧氏距离
- **快速检索** — 向量数据库的 ANN 搜索
- **聚类分析** — 把相似的文档聚在一起

## 双塔架构 vs 交叉编码器

### 双塔（Embedding 模型）

```
问题 → [Encoder] → 向量 Q
文档 → [Encoder] → 向量 D
相似度 = cosine(Q, D)
```

- 问题和文档**分别**编码
- 速度快（可以预计算文档向量）
- 精度一般
- **用于：大规模初筛**

### 交叉编码器（Rerank 模型）

```
[问题 + 文档] → [Cross-Encoder] → 相关性分数
```

- 问题和文档**一起**输入
- 速度慢（每次都要算）
- 精度高
- **用于：小规模精排**

## 常见 Embedding 模型

| 模型 | 语言 | 维度 | 特点 |
|------|------|------|------|
| text-embedding-3-small | 多语言 | 1536 | OpenAI，效果好但需要 API |
| BAAI/bge-small-zh-v1.5 | 中文 | 512 | 中文效果好，可本地运行 |
| BAAI/bge-m3 | 多语言 | 1024 | 支持多语言，效果优秀 |

## 本地运行 Embedding（ONNX）

```python
import onnxruntime as ort
from tokenizers import Tokenizer
import numpy as np

class LocalBGEEmbedding:
    def __init__(self, model_dir):
        self.tokenizer = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
        self.session = ort.InferenceSession(f"{model_dir}/model_optimized.onnx")

    def embed(self, text):
        # 1. Tokenize
        encoded = self.tokenizer.encode(text)
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
        token_type_ids = np.array([encoded.type_ids], dtype=np.int64)

        # 2. ONNX 推理
        outputs = self.session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        })

        # 3. Mean pooling + L2 归一化
        last_hidden = outputs[0]
        mask = np.expand_dims(attention_mask, -1)
        embeddings = np.sum(last_hidden * mask, axis=1) / np.sum(mask, axis=1)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return (embeddings / norms).tolist()[0]
```

## 相似度计算

```python
def cosine_similarity(a, b):
    """余弦相似度：-1 到 1，越接近 1 越相似"""
    return sum(x * y for x, y in zip(a, b)) / (
        math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
    )

# 示例
vec1 = embed("沈令月是公主")
vec2 = embed("沈令月是嫡女")
vec3 = embed("今天天气很好")

print(cosine_similarity(vec1, vec2))  # 0.95（高度相似）
print(cosine_similarity(vec1, vec3))  # 0.12（几乎无关）
```
