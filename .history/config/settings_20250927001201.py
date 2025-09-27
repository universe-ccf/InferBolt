# config/settings.py
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

# 语音文本互转模型
ASR_MODEL = "whisper-1"            
TTS_VOICE = "alloy"              

# 对话/记忆
MAX_ROUNDS = 8
MAX_TOKENS_RESPONSE = 512

# 选择阈值（最高分需要≥该阈值才触发技能；否则走普通对话）
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

# 超时（秒）
CONNECT_TIMEOUT = 5      # 连接建立
READ_TIMEOUT = 30        # 响应读取（生成可能较慢，适当放宽）
REQUEST_TIMEOUT = (CONNECT_TIMEOUT, READ_TIMEOUT)

# 重试
HTTP_MAX_RETRIES = 2     # 读/超时的自动重试次数
HTTP_BACKOFF_SEC = 0.5   # 指数退避初值

# === Speech configs ===
ENABLE_ASR = True           
ENABLE_TTS = True        
ASR_PROVIDER = "mock"        # 预留: "mock" | "vendor_xxx"
TTS_PROVIDER = "mock"        # 预留: "mock" | "vendor_xxx"
AUDIO_SAMPLE_RATE = 16000    # 统一采样率（Hz）


ASR_MODEL = "asr"            # ASR模型名（按官方示例）
ASR_INPUT_FORMAT = "wav"     # 我们走本地wav->base64上送。若用URL方式可切换为 "mp3" 等并走URL分支
ASR_USE_URL_UPLOAD = False   # False=base64内联上传；True=传url（见下文asr_client的两种分支）

TTS_VOICE = "qiniu_zh_female_xyqxxj"   # 默认音色：校园清新学姐
TTS_ENCODING = "mp3"         # 参照官网方式，后续可以尝试 "wav"，便于直接解码成numpy（避免mp3依赖）
TTS_SPEED = 1.0

