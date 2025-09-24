# app/config/settings.py
from __future__ import annotations

# 对话/记忆
MAX_ROUNDS = 8
MAX_TOKENS_RESPONSE = 512

# 模型默认
LLM_MODEL = "gpt-4o-mini"          # 示例，占位；后续你选具体可用的
LLM_TEMPERATURE = 0.7

ASR_MODEL = "whisper-1"            # 示例
TTS_VOICE = "alloy"                 # 示例

# 触发策略
INTENT_CONF_THRESHOLD = 0.6

# 评估/埋点
ENABLE_LOGGING = True
LOG_PATH = "logs/app.jsonl"


import os
from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.getenv("MODEL_NAME", "doubao-seed-1.6-flash")
LLM_TEMPERATURE = 0.7
MAX_ROUNDS = 8
MAX_TOKENS_RESPONSE = 512

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL", "https://openai.qiniu.com/v1")
REQUEST_TIMEOUT = 15  # 秒
