# 不依赖项目的
import os, requests, json
from dotenv import load_dotenv

load_dotenv()  # 读取 .env（可选：也可直接把变量写在这里临时跑）

API_KEY  = os.getenv("API_KEY") or "REPLACE_ME"
BASE_URL = (os.getenv("BASE_URL") or "https://openai.qiniu.com/v1").rstrip("/")
MODEL    = os.getenv("MODEL_NAME") or "doubao-seed-1.6-flash"

url = f"{BASE_URL}/chat/completions"
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
payload = {
    "stream": False,
    "model": MODEL,
    "messages": [
        {"role":"system","content":"You are a helpful assistant."},
        {"role":"user","content":"Hello from smoke test!"}
    ],
    "max_tokens": 64,
    "temperature": 0.7
}

try:
    resp = requests.post(url, headers=headers, json=payload, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    print("✅ LLM OK. Reply:", content[:120].replace("\n"," "))
except requests.exceptions.HTTPError as e:
    print("❌ HTTPError:", e, "\nBody:", getattr(e.response, "text", ""))
except Exception as e:
    print("❌ Error:", type(e).__name__, e)
