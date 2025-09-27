# clients/tts_client.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Any
import numpy as np
from math import pi, sin
from config import settings
import base64, io, wave
import requests, os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from utils.cache import sha256_text, cache_get_file, cache_put_file
from utils.logging import write_log


@dataclass
class TTSResult:
    audio_path: Optional[str]   # mp3 文件路径
    sample_rate: Optional[int]  # mp3 由前端解码，这里可以 None
    meta: Dict[str, Any]


class TTSClient:
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.base_url = (base_url or getattr(settings, "BASE_URL", "https://openai.qiniu.com/v1")).rstrip("/")
        self.api_key = api_key or getattr(settings, "API_KEY", None)
        self._url = f"{self.base_url}/voice/tts"
        self._list_url = f"{self.base_url}/voice/list"

        self.session = requests.Session()
        retry = Retry(total=settings.HTTP_MAX_RETRIES, read=settings.HTTP_MAX_RETRIES,
                      connect=settings.HTTP_MAX_RETRIES, backoff_factor=settings.HTTP_BACKOFF_SEC,
                      status_forcelist=(429,502,503,504), allowed_methods=frozenset(["GET","POST"]))
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter); self.session.mount("http://", adapter)

    def list_voices(self) -> List[Dict[str, Any]]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = self.session.get(self._list_url, headers=headers, timeout=settings.REQUEST_TIMEOUT)
        resp.raise_for_status()
        js = resp.json()
        # 文档返回数组形态：[{voice_name, voice_type, url, category, updatetime}, ...]
        return js if isinstance(js, list) else js.get("data", [])

    def synthesize(self, text: str, voice_type: Optional[str]=None, speed_ratio: Optional[float]=None) -> TTSResult:
        if not settings.ENABLE_TTS:
            return TTSResult(None, None, {"enabled": False})

        # 参数：角色覆盖 > 全局默认
        voice = voice_type or settings.TTS_VOICE
        speed = speed_ratio if (speed_ratio is not None) else settings.TTS_SPEED
        encoding = settings.TTS_ENCODING  # "mp3"

        # === 缓存命中（可选）===
        audio_key = None
        if settings.ENABLE_SPEECH_CACHE:
            sig = f"{text}||{voice}||{speed}||{encoding}"
            audio_key = sha256_text(sig)
            cached = cache_get_file(settings.CACHE_TTS_DIR, audio_key, encoding)
            if cached:
                return TTSResult(cached, None, {"provider":"qiniu","cache":"hit"})

        MAX_TTS_CHARS = 300  # 快速压制时延, 防止LLM文本太长，请求TTS服务器时间太长导致请求失败

        if len(text) > MAX_TTS_CHARS:
            write_log(settings.LOG_PATH, {"event":"tts_truncate", "orig_len": len(text)})
            text = text[:MAX_TTS_CHARS] + "……"

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {
            "audio": {"voice_type": voice, "encoding": encoding, "speed_ratio": float(speed)},
            "request": {"text": text}
        }

        # 埋点测试
        write_log(settings.LOG_PATH, {
            "event": "tts_request",
            "voice": voice, "speed": float(speed), "encoding": encoding
        })

        try:
            resp = self.session.post(self._url, headers=headers, json=data, timeout=settings.REQUEST_TIMEOUT)
            resp.raise_for_status()
            js = resp.json()
            b64 = js.get("data")
            if not b64:
                return TTSResult(None, None, {"provider":"qiniu","error":"no_audio_data","resp": str(js)[:300]})
            audio_bytes = base64.b64decode(b64)

            # 埋点测试
            write_log(settings.LOG_PATH, {
                "event": "tts_response",
                "bytes": len(audio_bytes)
            })

            # 落为 mp3 文件
            if settings.ENABLE_SPEECH_CACHE and audio_key:
                fpath = cache_put_file(settings.CACHE_TTS_DIR, audio_key, encoding, audio_bytes)
            else:
                # 临时文件名（不缓存时）
                os.makedirs(settings.CACHE_TTS_DIR, exist_ok=True)
                fpath = os.path.join(settings.CACHE_TTS_DIR, f"tmp_{sha256_text(b64)}.{encoding}")
                with open(fpath, "wb") as f:
                    f.write(audio_bytes)
            return TTSResult(fpath, None, {"provider":"qiniu","status":"ok","voice":voice,"speed":speed})
        except requests.exceptions.RequestException as e:
            body = getattr(e.response,"text","") if hasattr(e,"response") else str(e)

            # 埋点测试
            write_log(settings.LOG_PATH, {
                "event": "tts_error",
                "error": (getattr(e.response, "text", "") or str(e))[:300]
            })

            return TTSResult(None, None, {"provider":"qiniu","error": body[:300]})
        except Exception as e:

            # 埋点测试
            write_log(settings.LOG_PATH, {
                "event": "tts_error",
                "error": str(e)[:300]
            })

            return TTSResult(None, None, {"provider":"qiniu","error": str(e)[:300]})

