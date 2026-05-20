"""快速测试：绕过代理直连 LongCat"""
import os
import json
import ssl
import urllib.request

# 清除代理
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    os.environ.pop(k, None)

api_key = os.getenv("OPENAI_API_KEY", "ak_2k688O7UG66R1Yu6QS4zU2TR2vA6X")
base_url = "https://api.longcat.chat/openai"

# 测试 chat
payload = json.dumps({
    "model": "LongCat-2.0-Preview",
    "messages": [{"role": "user", "content": "RAG 系统有哪些组件？用中文回答"}],
    "max_tokens": 200
}).encode('utf-8')

req = urllib.request.Request(
    f"{base_url}/v1/chat/completions",
    data=payload,
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
)

# 绕过 SSL 验证
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

try:
    resp = urllib.request.urlopen(req, context=ctx, timeout=30)
    result = json.loads(resp.read().decode('utf-8'))
    print("[OK] LongCat 连接成功!")
    print(f"[ANS] {result['choices'][0]['message']['content']}")
except Exception as e:
    print(f"[ERR] {e}")
