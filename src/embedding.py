"""
公共 Embedding 模块 —— 本地 BGE (ONNX)
所有脚本共享这一个实现
"""

import os
import numpy as np
from tokenizers import Tokenizer
import onnxruntime as ort


class LocalBGEEmbedding:
    """本地 BGE Embedding，兼容 LangChain Embeddings 接口"""

    def __init__(self, model_dir: str):
        print(f"[INFO] 加载本地 BGE 模型: {model_dir}")
        self.tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.session = ort.InferenceSession(
            os.path.join(model_dir, "model_optimized.onnx"),
            providers=["CPUExecutionProvider"]
        )
        print("[OK] 模型加载完成")

    def _embed_one(self, text: str) -> list[float]:
        """单条文本转向量"""
        return self._embed_batch([text])[0]

    def _embed_batch(self, texts) -> list:
        """批量文本转向量"""
        texts = [str(t) for t in texts]
        try:
            encoded = self.tokenizer.encode_batch(texts)
        except Exception:
            encoded = [self.tokenizer.encode(t) for t in texts]
        max_len = min(max(len(e.ids) for e in encoded), 256)

        input_ids = np.zeros((len(encoded), max_len), dtype=np.int64)
        attention_mask = np.zeros((len(encoded), max_len), dtype=np.int64)
        token_type_ids = np.zeros((len(encoded), max_len), dtype=np.int64)

        for i, e in enumerate(encoded):
            ids = e.ids[:max_len]
            mask = e.attention_mask[:max_len]
            types = e.type_ids[:max_len]
            input_ids[i, :len(ids)] = ids
            attention_mask[i, :len(mask)] = mask
            token_type_ids[i, :len(types)] = types

        outputs = self.session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        })

        last_hidden_state = outputs[0]
        mask_expanded = np.expand_dims(attention_mask, -1)
        sum_embeddings = np.sum(last_hidden_state * mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
        embeddings = sum_embeddings / sum_mask
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, a_min=1e-9, a_max=None)

        return embeddings.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """LangChain 接口：批量转向量"""
        all_embeddings = []
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            all_embeddings.extend(self._embed_batch(batch))
        return all_embeddings

    def embed_query(self, text) -> list[float]:
        """LangChain 接口：单条转向量"""
        if not isinstance(text, str):
            text = str(text) if text else " "
        return self._embed_one(text)
