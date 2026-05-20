"""
步骤 1: 文档处理 + 向量化 + 存入向量数据库

流程: 加载文档 → 切块 → 向量化(分批) → 存入 Chroma

运行: python src/01_ingest.py
"""

import os
import shutil
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from tokenizers import Tokenizer
import onnxruntime as ort
import numpy as np

load_dotenv()

DB_PATH = "./chroma_db"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "bge-small-zh-v1.5", "fast-bge-small-zh-v1.5")


# ============================================================
# 本地 BGE Embedding
# ============================================================
class LocalBGEEmbedding:
    def __init__(self, model_dir: str):
        self.tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.session = ort.InferenceSession(
            os.path.join(model_dir, "model_optimized.onnx"),
            providers=["CPUExecutionProvider"]
        )

    def _embed(self, texts):
        encoded = self.tokenizer.encode_batch(texts)
        max_len = max(len(e.ids) for e in encoded)
        # 限制最大序列长度，防止内存溢出
        max_len = min(max_len, 256)
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
            "input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids
        })
        last_hidden_state = outputs[0]
        mask_expanded = np.expand_dims(attention_mask, -1)
        sum_embeddings = np.sum(last_hidden_state * mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
        embeddings = sum_embeddings / sum_mask
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.clip(norms, a_min=1e-9, a_max=None)

    def embed_documents(self, texts):
        return self._embed(texts).tolist()

    def embed_query(self, text):
        return self._embed([text])[0].tolist()


embed_model = LocalBGEEmbedding(MODEL_DIR)

# ============================================================
# 第 1 步: 加载文档
# ============================================================
DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "帝女明珠.txt")

print(f"[INFO] 加载文档: {DATA_FILE}")
with open(DATA_FILE, "r", encoding="utf-8") as f:
    raw_text = f.read()

# 去掉开头非正文内容
lines = raw_text.split("\n")
start_idx = 0
for i, line in enumerate(lines):
    if "===============" in line and i > 3:
        start_idx = i + 1
        break

clean_text = "\n".join(lines[start_idx:]).strip()
print(f"[INFO] 正文长度: {len(clean_text)} 字符")

documents = [Document(page_content=clean_text, metadata={"source": "帝女明珠.txt"})]
print(f"[DOC] 加载了 {len(documents)} 个文档")

# ============================================================
# 第 2 步: 文档切块
# ============================================================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,        # 每块 300 字符（更小的块 = 更精准的检索）
    chunk_overlap=30,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""]
)

chunks = splitter.split_documents(documents)
print(f">> 切分成 {len(chunks)} 个块")

# 显示前 3 个块
for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- 块 {i+1} (长度: {len(chunk.page_content)}) ---")
    print(chunk.page_content[:80] + "...")

# ============================================================
# 第 3 步: 向量化 + 存入数据库（分批处理，防止内存溢出）
# ============================================================
if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH)
    print(f"[INFO] 清除旧数据库")

print(f"[INFO] 开始向量化（共 {len(chunks)} 个块，每批 50 个）...")

# 先创建空数据库
db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embed_model,
    collection_name="dinv_mingzhu"
)

# 分批添加
BATCH_SIZE = 50
total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_idx in range(total_batches):
    start = batch_idx * BATCH_SIZE
    end = min(start + BATCH_SIZE, len(chunks))
    batch = chunks[start:end]

    texts = [c.page_content for c in batch]
    metadatas = [c.metadata for c in batch]
    ids = [f"chunk_{start + i}" for i in range(len(batch))]

    embeddings = embed_model.embed_documents(texts)

    db._collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    progress = (batch_idx + 1) / total_batches * 100
    print(f"  批次 {batch_idx + 1}/{total_batches} ({progress:.0f}%) - 已处理 {end}/{len(chunks)} 个块")

print(f"\n[DB] 已存入向量数据库 (路径: {DB_PATH})")
print(f"[INFO] 数据库中共有 {db._collection.count()} 个向量")
print(f"\n[OK] 步骤 1 完成！接下来可以运行:")
print(f"  python src/02_query.py     # 基础问答")
print(f"  python src/05_multi_query.py  # 多路检索")
print(f"  python src/06_hyde.py      # HyDE 检索")
print(f"  python src/07_rerank.py    # Rerank 精排")
