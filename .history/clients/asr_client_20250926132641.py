# clients/asr_client.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
import base64, io, wave
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config import settings

@dataclass
class ASRResult:
    text: str
    confidence: float
    meta: Dict[str, Any]


def _float32_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    # [-1,1] float32 -> int16 pcm wav
    a = np.clip(audio, -1.0, 1.0)
    pcm16 = (a * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()

class ASRClient:
    def __init__(self, provider: Optional[str] = None, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.base_url = (base_url or getattr(settings, "BASE_URL", "https://openai.qiniu.com/v1")).rstrip("/")
        self.api_key = api_key or getattr(settings, "API_KEY", None)
        self._url = f"{self.base_url}/voice/asr"

        # 带重试的Session
        self.session = requests.Session()
        retry = Retry(
            total=settings.HTTP_MAX_RETRIES,
            read=settings.HTTP_MAX_RETRIES,
            connect=settings.HTTP_MAX_RETRIES,
            backoff_factor=settings.HTTP_BACKOFF_SEC,
            status_forcelist=(429, 502, 503, 504),
            allowed_methods=frozenset(["POST"])
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

class ASRClient:
    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or settings.ASR_PROVIDER

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> ASRResult:
        """
        audio: shape=(n_samples,) 的 float32 [-1,1]
        sample_rate: int
        """
        if not settings.ENABLE_ASR or self.provider == "mock":
            return ASRResult(text="（ASR未启用：请用文本输入，或在 settings 开启 ENABLE_ASR 并配置真实厂商）",
                             confidence=0.0,
                             meta={"provider": "mock"})
        # === TODO: 在此处接入真实ASR厂商 ===
        # 伪代码：
        # wav_bytes = (audio * 32767).astype(np.int16).tobytes()
        # resp = requests.post(api_url, headers=..., files=..., data=...)
        # text, conf = parse(resp.json())
        # return ASRResult(text=text, confidence=conf, meta={"provider": self.provider})
        raise NotImplementedError("请实现真实ASR厂商调用")
