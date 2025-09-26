# app/clients/asr_client.py
class ASRClient:
    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
        ...

    def transcribe(self, audio_bytes: bytes, language: str | None = None) -> str:
        """音频→文本（MVP：同步返回完整文本）"""
        ...

# clients/asr_client.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np

from config import settings

@dataclass
class ASRResult:
    text: str
    confidence: float
    meta: Dict[str, Any]

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
