# clients/tts_client.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
from math import pi, sin
from config import settings
import base64, io, wave
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

@dataclass
class TTSResult:
    audio: np.ndarray         # float32 mono [-1,1]
    sample_rate: int
    meta: Dict[str, Any]


def _wav_bytes_to_float32(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        sr = wf.getframerate()
        nchan = wf.getnchannels()
        sampw = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if sampw != 2:
        raise ValueError("Only 16-bit PCM supported in this helper.")
    pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if nchan == 2:
        pcm = pcm.reshape(-1, 2).mean(axis=1)  # to mono
    return pcm, sr

class TTSClient:
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.base_url = (base_url or getattr(settings, "BASE_URL", "https://openai.qiniu.com/v1")).rstrip("/")
        self.api_key = api_key or getattr(settings, "API_KEY", None)
        self._url = f"{self.base_url}/voice/tts"

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

    def synthesize(self, text: str) -> TTSResult:
        if not settings.ENABLE_TTS:
            sr = settings.AUDIO_SAMPLE_RATE
            tone = np.zeros(int(sr*0.3), dtype=np.float32)
            return TTSResult(audio=tone, sample_rate=sr, meta={"provider": "qiniu", "enabled": False})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "audio": {
                "voice_type": settings.TTS_VOICE,
                "encoding": settings.TTS_ENCODING,   # 推荐 'wav'
                "speed_ratio": settings.TTS_SPEED
            },
            "request": {
                "text": text
            }
        }

        try:
            resp = self.session.post(self._url, headers=headers, json=data, timeout=settings.REQUEST_TIMEOUT)
            resp.raise_for_status()
            js = resp.json()

            # 兼容返回格式：
            # 1) 直接 base64 字段：js["audio"]["content"]
            # 2) url：js["audio"]["url"]（若是wav/mp3可回取）
            b64 = None
            audio_url = None
            # 常见层级
            if isinstance(js.get("audio"), dict):
                b64 = js["audio"].get("content")
                audio_url = js["audio"].get("url")

            wav_bytes = None
            if b64:
                wav_bytes = base64.b64decode(b64)
            elif audio_url:
                # 拉取URL（需公网网络通畅；避免mp3，故请求encoding=wav）
                r2 = self.session.get(audio_url, timeout=settings.REQUEST_TIMEOUT)
                r2.raise_for_status()
                wav_bytes = r2.content
            else:
                # 其他结构兜底再找一层 data.audio
                if "data" in js and isinstance(js["data"].get("audio"), dict):
                    b64 = js["data"]["audio"].get("content")
                    audio_url = js["data"]["audio"].get("url")
                    if b64:
                        wav_bytes = base64.b64decode(b64)
                    elif audio_url:
                        r2 = self.session.get(audio_url, timeout=settings.REQUEST_TIMEOUT)
                        r2.raise_for_status()
                        wav_bytes = r2.content

            if not wav_bytes:
                # 无法解析
                return TTSResult(audio=np.zeros(int(settings.AUDIO_SAMPLE_RATE*0.1), dtype=np.float32),
                                 sample_rate=settings.AUDIO_SAMPLE_RATE,
                                 meta={"provider": "qiniu", "error": "no_audio_in_response", "resp": str(js)[:300]})

            audio, sr = _wav_bytes_to_float32(wav_bytes)
            return TTSResult(audio=audio, sample_rate=sr, meta={"provider": "qiniu", "status": "ok"})

        except requests.exceptions.RequestException as e:
            body = getattr(e.response, "text", "") if hasattr(e, "response") else str(e)
            return TTSResult(audio=np.zeros(int(settings.AUDIO_SAMPLE_RATE*0.1), dtype=np.float32),
                             sample_rate=settings.AUDIO_SAMPLE_RATE,
                             meta={"provider": "qiniu", "error": body[:300]})
        except Exception as e:
            return TTSResult(audio=np.zeros(int(settings.AUDIO_SAMPLE_RATE*0.1), dtype=np.float32),
                             sample_rate=settings.AUDIO_SAMPLE_RATE,
                             meta={"provider": "qiniu", "error": str(e)[:300]})

