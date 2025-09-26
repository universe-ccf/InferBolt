# clients/tts_client.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
from math import pi, sin
from config import settings
import base64, io, wave

@dataclass
class TTSResult:
    audio: np.ndarray         # float32 mono [-1,1]
    sample_rate: int
    meta: Dict[str, Any]

class TTSClient:
    def __init__(self, provider: Optional[str] = None, voice: Optional[str] = None):
        self.provider = provider or settings.TTS_PROVIDER
        self.voice = voice or settings.TTS_VOICE

    def synthesize(self, text: str) -> TTSResult:
        if not settings.ENABLE_TTS or self.provider == "mock":
            # 本地合成 0.4 秒提示音，证明链路打通
            sr = settings.AUDIO_SAMPLE_RATE
            dur = 0.4
            t = np.linspace(0, dur, int(sr*dur), endpoint=False, dtype=np.float32)
            tone = np.sin(2*pi*440.0*t).astype(np.float32) * 0.2
            return TTSResult(audio=tone, sample_rate=sr, meta={"provider": "mock"})
        # === TODO: 在此处接入真实TTS厂商 ===
        # 伪代码：
        # resp = requests.post(api_url, json={"text": text, "voice": self.voice}, headers=...)
        # audio_bytes = base64.b64decode(resp.json()["audio"])
        # audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)/32768.0
        # return TTSResult(audio=audio, sample_rate=resp_sr, meta={"provider": self.provider})
        raise NotImplementedError("请实现真实TTS厂商调用")
