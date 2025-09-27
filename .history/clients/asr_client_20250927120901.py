# clients/asr_client.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
import base64, io, wave
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config import settings
from utils.cache import sha256_bytes, cache_get_text, cache_put_text
from utils.logging import write_log


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


    def transcribe(self, audio_np: np.ndarray, sample_rate: int, audio_url: Optional[str] = None) -> ASRResult:
        """
        优先 URL 上送（与官方契约对齐）；无URL时走 base64 内联兜底。
        返回：ASRResult(text, confidence=0.0, meta含duration_ms和upload_mode)
        """
        if not settings.ENABLE_ASR:
            return ASRResult("（ASR未启用）", 0.0, {"enabled": False})

        # === 缓存命中（可选）===
        cache_key = None
        if settings.ENABLE_SPEECH_CACHE:
            if audio_url:
                cache_key = audio_url  # 直接用url字符串哈希更好，这里简化为原串
            else:
                wav_bytes = _float32_to_wav_bytes(audio_np, sample_rate)
                cache_key = sha256_bytes(wav_bytes)
            cached = cache_get_text(settings.CACHE_ASR_DIR, cache_key)
            if cached is not None:
                return ASRResult(cached, 0.0, {"provider":"qiniu","cache":"hit"})

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        # === 请求体：URL优先 ===
        if settings.ASR_USE_URL_UPLOAD and audio_url:
            data = {"model": settings.ASR_MODEL,
                    "audio": {"format": settings.ASR_INPUT_FORMAT, "url": audio_url}}
            upload_mode = "url"
        else:
            wav_bytes = _float32_to_wav_bytes(audio_np, sample_rate)
            b64 = base64.b64encode(wav_bytes).decode("ascii")
            data = {"model": settings.ASR_MODEL,
                    "audio": {"format": "wav", "content": b64}}
            upload_mode = "inline"


        write_log(settings.LOG_PATH, {
            "event": "asr_request",
            "upload_mode": upload_mode,                      # "inline" or "url"
            "format": data["audio"].get("format"),
            "use_url": bool(settings.ASR_USE_URL_UPLOAD and audio_url),
        })

        try:
            resp = self.session.post(self._url, headers=headers, json=data, timeout=settings.REQUEST_TIMEOUT)
            resp.raise_for_status()
            js = resp.json()
            # 按文档解析
            data_node = js.get("data", {}) or {}
            result = data_node.get("result", {}) or {}
            text = result.get("text") or ""
            duration_ms = (data_node.get("audio_info", {}) or {}).get("duration")

            write_log(settings.LOG_PATH, {
                "event": "asr_response",
                "text_len": len(text or ""),
                "duration_ms": duration_ms
            })

            # 缓存落盘
            if settings.ENABLE_SPEECH_CACHE and cache_key and text:
                cache_put_text(settings.CACHE_ASR_DIR, cache_key, text)
            return ASRResult(text=text or "（空识别结果）", confidence=0.0,
                             meta={"provider":"qiniu","duration_ms":duration_ms,"upload_mode":upload_mode})
        except requests.exceptions.RequestException as e:
            body = getattr(e.response,"text","") if hasattr(e,"response") else str(e)
            write_log(settings.LOG_PATH, {
                "event": "asr_error",
                "error": (getattr(e.response, "text", "") or str(e))[:300]
            })
            return ASRResult("（ASR请求失败）", 0.0, {"provider":"qiniu","error": body[:300]})
        except Exception as e:
            write_log(settings.LOG_PATH, {
                "event": "asr_error",
                "error": str(e)[:300]
            })
            return ASRResult("（ASR解析异常）", 0.0, {"provider":"qiniu","error": str(e)[:300]})
