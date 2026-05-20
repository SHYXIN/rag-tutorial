"""
步骤 17: 生产部署 — FastAPI 封装 + 流式输出

功能：
  1. REST API 接口（同步问答）
  2. 流式输出（SSE，打字机效果）
  3. 多轮对话（带会话 ID）
  4. 健康检查

运行: python src/17_api.py
测试: curl http://localhost:8000/ask -d '{"question": "沈令月的驸马是谁？"}'
"""

import os
import json
import ssl
import urllib.request
import re
from collections import OrderedDict
from dotenv import load_dotenv
from langchain_chroma import Chroma
from embedding import LocalBGEEmbedding

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.longcat.chat/openai")
DB_PATH = "./chroma_db"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "bge-small-zh-v1.5", "fast-bge-small-zh-v1.5")


def call_llm(messages, max_tokens=400):
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


def call_llm_stream(messages, max_tokens=400):
    """流式调用 LLM，逐字返回"""
    payload = json.dumps({
        "model": "LongCat-2.0-Preview",
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True
    }).encode('utf-8')
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    )
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    resp = urllib.request.urlopen(req, context=ctx, timeout=120)
    for line in resp:
        line = line.decode('utf-8').strip()
        if line.startswith('data: '):
            data = line[6:]
            if data == '[DONE]':
                break
            try:
                chunk = json.loads(data)
                delta = chunk['choices'][0].get('delta', {})
                content = delta.get('content', '')
                if content:
                    yield content
            except json.JSONDecodeError:
                continue


# ============================================================
# 对话记忆
# ============================================================
class ConversationMemory:
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.history = []
        self.summary = ""
        self.entities = OrderedDict()

    def add(self, role, content):
        self.history.append({"role": role, "content": content})

    def get_context(self):
        parts = []
        if self.entities:
            parts.append(f"[已知人物] {'、'.join(f'{n}（{d}）' for n,d in list(self.entities.items())[:10])}")
        if self.summary:
            parts.append(f"[对话摘要] {self.summary}")
        recent = self.history[-self.window_size * 2:]
        if recent:
            parts.append(f"[最近对话]\n" + "\n".join(f"{'用户' if h['role']=='user' else '助手'}: {h['content']}" for h in recent))
        return "\n\n".join(parts)


# ============================================================
# RAG 核心
# ============================================================
class RAGCore:
    def __init__(self):
        self.embed_model = LocalBGEEmbedding(MODEL_DIR)
        self.db = Chroma(persist_directory=DB_PATH, embedding_function=self.embed_model, collection_name="dinv_mingzhu")
        self.sessions = {}  # {session_id: ConversationMemory}

    def get_memory(self, session_id: str) -> ConversationMemory:
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationMemory()
        return self.sessions[session_id]

    def ask(self, question: str, session_id: str = "default") -> str:
        memory = self.get_memory(session_id)
        docs = self.db.similarity_search(question, k=3)
        context = "\n\n".join(f"[文档 {i+1}] {d.page_content}" for i, d in enumerate(docs))
        conv_ctx = memory.get_context()
        messages = [
            {"role": "system", "content": f"你是文学分析助手。{conv_ctx}\n\n根据小说片段回答问题，简洁清晰。"},
            {"role": "user", "content": f"小说片段:\n{context}\n\n问题: {question}"}
        ]
        answer = call_llm(messages)
        memory.add("user", question)
        memory.add("assistant", answer)
        return answer

    def ask_stream(self, question: str, session_id: str = "default"):
        """流式问答，逐字返回"""
        memory = self.get_memory(session_id)
        docs = self.db.similarity_search(question, k=3)
        context = "\n\n".join(f"[文档 {i+1}] {d.page_content}" for i, d in enumerate(docs))
        conv_ctx = memory.get_context()
        messages = [
            {"role": "system", "content": f"你是文学分析助手。{conv_ctx}\n\n根据小说片段回答问题，简洁清晰。"},
            {"role": "user", "content": f"小说片段:\n{context}\n\n问题: {question}"}
        ]
        full_answer = ""
        for chunk in call_llm_stream(messages):
            full_answer += chunk
            yield chunk
        memory.add("user", question)
        memory.add("assistant", full_answer)


# ============================================================
# FastAPI 应用
# ============================================================
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    from typing import Optional

    app = FastAPI(title="RAG API", version="1.0")
    rag = RAGCore()

    class AskRequest(BaseModel):
        question: str
        session_id: Optional[str] = "default"

    class AskResponse(BaseModel):
        answer: str
        session_id: str

    @app.get("/health")
    def health():
        return {"status": "ok", "docs": rag.db._collection.count()}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest):
        """同步问答"""
        try:
            answer = rag.ask(req.question, req.session_id)
            return AskResponse(answer=answer, session_id=req.session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/ask/stream")
    def ask_stream(req: AskRequest):
        """流式问答（SSE）"""
        try:
            def generate():
                for chunk in rag.ask_stream(req.question, req.session_id):
                    yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"
            return StreamingResponse(generate(), media_type="text/event-stream")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str):
        """获取会话历史"""
        memory = rag.get_memory(session_id)
        return {
            "session_id": session_id,
            "history_length": len(memory.history),
            "entities": list(memory.entities.keys()),
            "summary": memory.summary,
        }

    @app.delete("/sessions/{session_id}")
    def clear_session(session_id: str):
        """清除会话"""
        if session_id in rag.sessions:
            del rag.sessions[session_id]
        return {"status": "ok"}

    # 挂载静态文件
    from fastapi.staticfiles import StaticFiles
    import os
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # 首页重定向到前端
    @app.get("/")
    def root():
        from fastapi.responses import FileResponse
        index_path = os.path.join(static_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"status": "ok", "docs": rag.db._collection.count(), "frontend": "not found"}

    HAS_FASTAPI = True

except ImportError:
    HAS_FASTAPI = False


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    if HAS_FASTAPI:
        import uvicorn
        print("=" * 60)
        print("RAG API 服务")
        print("=" * 60)
        print("  文档: http://localhost:8000/docs")
        print("  问答: POST /ask")
        print("  流式: POST /ask/stream")
        print("=" * 60)
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("[INFO] FastAPI 未安装，运行基础测试...")
        print("  安装: pip install fastapi uvicorn")
        rag = RAGCore()
        questions = ["沈令月的驸马是谁？", "谢初是什么样的人？"]
        for q in questions:
            print(f"\n[Q] {q}")
            answer = rag.ask(q)
            print(f"[A] {answer}")
