"""
API 测试脚本

运行: python src/test_api.py
"""

import urllib.request
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

BASE_URL = "http://localhost:8888"


def test_health():
    """测试健康检查"""
    print("[Test 1] 健康检查 GET /")
    req = urllib.request.Request(f"{BASE_URL}/")
    resp = urllib.request.urlopen(req, timeout=10)
    result = json.loads(resp.read().decode('utf-8'))
    print(f"  状态: {result['status']}")
    print(f"  文档数: {result['docs']}")
    assert result['status'] == 'ok'
    print("  [OK] 通过\n")


def test_ask():
    """测试同步问答"""
    print("[Test 2] 同步问答 POST /ask")
    data = json.dumps({"question": "沈令月的驸马是谁？"}).encode('utf-8')
    req = urllib.request.Request(
        f"{BASE_URL}/ask",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    resp = urllib.request.urlopen(req, timeout=30)
    result = json.loads(resp.read().decode('utf-8'))
    print(f"  答案: {result['answer'][:100]}...")
    print(f"  会话: {result['session_id']}")
    assert 'answer' in result
    print("  [OK] 通过\n")


def test_ask_stream():
    """测试流式问答"""
    print("[Test 3] 流式问答 POST /ask/stream")
    data = json.dumps({"question": "谢初是什么样的人？", "session_id": "test_stream"}).encode('utf-8')
    req = urllib.request.Request(
        f"{BASE_URL}/ask/stream",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    resp = urllib.request.urlopen(req, timeout=60)
    full_text = ""
    for line in resp:
        line = line.decode('utf-8').strip()
        if line.startswith('data: '):
            chunk = json.loads(line[6:])
            if chunk.get('done'):
                break
            content = chunk.get('content', '')
            full_text += content
    print(f"  答案: {full_text[:100]}...")
    assert len(full_text) > 0
    print("  [OK] 通过\n")


def test_session():
    """测试会话管理"""
    print("[Test 4] 会话管理")

    # 先问一个问题
    data = json.dumps({"question": "沈令月是什么身份？", "session_id": "test_session"}).encode('utf-8')
    req = urllib.request.Request(
        f"{BASE_URL}/ask",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    urllib.request.urlopen(req, timeout=30)

    # 获取会话历史
    req = urllib.request.Request(f"{BASE_URL}/sessions/test_session")
    resp = urllib.request.urlopen(req, timeout=10)
    result = json.loads(resp.read().decode('utf-8'))
    print(f"  历史轮数: {result['history_length']}")
    print(f"  实体: {result['entities']}")
    assert result['history_length'] > 0
    print("  [OK] 通过\n")


def test_clear_session():
    """测试清除会话"""
    print("[Test 5] 清除会话 DELETE /sessions/{id}")
    req = urllib.request.Request(
        f"{BASE_URL}/sessions/test_session",
        method="DELETE"
    )
    resp = urllib.request.urlopen(req, timeout=10)
    result = json.loads(resp.read().decode('utf-8'))
    print(f"  状态: {result['status']}")
    assert result['status'] == 'ok'
    print("  [OK] 通过\n")


def test_multi_turn():
    """测试多轮对话"""
    print("[Test 6] 多轮对话")
    session_id = "test_multi"

    questions = [
        "沈令月的驸马是谁？",
        "他是什么样的人？",
        "他们是怎么认识的？",
    ]

    for i, q in enumerate(questions, 1):
        data = json.dumps({"question": q, "session_id": session_id}).encode('utf-8')
        req = urllib.request.Request(
            f"{BASE_URL}/ask",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        resp = urllib.request.urlopen(req, timeout=30)
        result = json.loads(resp.read().decode('utf-8'))
        print(f"  轮次 {i}: {q}")
        print(f"    答案: {result['answer'][:60]}...")

    # 检查会话历史
    req = urllib.request.Request(f"{BASE_URL}/sessions/{session_id}")
    resp = urllib.request.urlopen(req, timeout=10)
    result = json.loads(resp.read().decode('utf-8'))
    print(f"  总历史轮数: {result['history_length']}")
    assert result['history_length'] >= 6  # 3 轮 × 2（用户+助手）
    print("  [OK] 通过\n")


if __name__ == "__main__":
    print("=" * 60)
    print("RAG API 测试")
    print("=" * 60)

    try:
        test_health()
        test_ask()
        test_ask_stream()
        test_session()
        test_clear_session()
        test_multi_turn()
        print("=" * 60)
        print("所有测试通过!")
        print("=" * 60)
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
