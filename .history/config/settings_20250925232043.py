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

# 思辨训练营 - 技能候选集合（可随时扩展）
SKILL_CANDIDATES = ["steelman", "x_exam", "counterfactual", "none"]

# 每个技能的简短说明（注入到提示词里，帮助模型理解边界）
SKILL_DESCRIPTIONS = {
    "steelman": "当用户想让观点/立场更有说服力、表述更扎实、更强、更完整时。",
    "x_exam": "当用户希望被挑错/交叉质询/找漏洞/被针对性提问时。",
    "counterfactual": "当用户想要在关键假设变化下推演结果（如果不这样/反过来/换前提）。",
    "none": "以上皆不符合，或只是闲聊/无法判断。"
}


