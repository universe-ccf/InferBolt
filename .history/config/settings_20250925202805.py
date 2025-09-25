# app/config/settings.py
from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

# LLM
LLM_MODEL = os.getenv("MODEL_NAME", "doubao-seed-1.6-flash")
LLM_TEMPERATURE = 0.7
MAX_ROUNDS = 8
MAX_TOKENS_RESPONSE = 512

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL", "https://openai.qiniu.com/v1")
REQUEST_TIMEOUT = 15  # 秒

# 语音文本互转模型
ASR_MODEL = "whisper-1"            
TTS_VOICE = "alloy"              

# 对话/记忆
MAX_ROUNDS = 8
MAX_TOKENS_RESPONSE = 512

# 触发策略
INTENT_CONF_THRESHOLD = 0.6

# 评估/埋点
ENABLE_LOGGING = True
LOG_PATH = "logs/app.jsonl"

DEBUG = True  # 开关：是否在UI与日志中输出调试信息

