# clients/tts_client.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Any
import numpy as np
from math import pi, sin
from config import settings
import base64, io, wave
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from utils.cache import sha256_text, cache_get_file, cache_put_file

@dataclass
class TTSResult:
    audio: np.ndarray         # float32 mono [-1,1]
    sample_rate: int
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

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {
            "audio": {"voice_type": voice, "encoding": encoding, "speed_ratio": float(speed)},
            "request": {"text": text}
        }
        try:
            resp = self.session.post(self._url, headers=headers, json=data, timeout=settings.REQUEST_TIMEOUT)
            resp.raise_for_status()
            js = resp.json()
            b64 = js.get("data")
            if not b64:
                return TTSResult(None, None, {"provider":"qiniu","error":"no_audio_data","resp": str(js)[:300]})
            audio_bytes = base64.b64decode(b64)
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
            return TTSResult(None, None, {"provider":"qiniu","error": body[:300]})
        except Exception as e:
            return TTSResult(None, None, {"provider":"qiniu","error": str(e)[:300]})


# def _wav_bytes_to_float32(wav_bytes: bytes) -> tuple[np.ndarray, int]:
#     buf = io.BytesIO(wav_bytes)
#     with wave.open(buf, "rb") as wf:
#         sr = wf.getframerate()
#         nchan = wf.getnchannels()
#         sampw = wf.getsampwidth()
#         frames = wf.readframes(wf.getnframes())
#     if sampw != 2:
#         raise ValueError("Only 16-bit PCM supported in this helper.")
#     pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
#     if nchan == 2:
#         pcm = pcm.reshape(-1, 2).mean(axis=1)  # to mono
#     return pcm, sr

# class TTSClient:
#     def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
#         self.base_url = (base_url or getattr(settings, "BASE_URL", "https://openai.qiniu.com/v1")).rstrip("/")
#         self.api_key = api_key or getattr(settings, "API_KEY", None)
#         self._url = f"{self.base_url}/voice/tts"

#         self.session = requests.Session()
#         retry = Retry(
#             total=settings.HTTP_MAX_RETRIES,
#             read=settings.HTTP_MAX_RETRIES,
#             connect=settings.HTTP_MAX_RETRIES,
#             backoff_factor=settings.HTTP_BACKOFF_SEC,
#             status_forcelist=(429, 502, 503, 504),
#             allowed_methods=frozenset(["POST"])
#         )
#         adapter = HTTPAdapter(max_retries=retry)
#         self.session.mount("https://", adapter)
#         self.session.mount("http://", adapter)

#     def synthesize(self, text: str) -> TTSResult:
#         if not settings.ENABLE_TTS:
#             sr = settings.AUDIO_SAMPLE_RATE
#             tone = np.zeros(int(sr*0.3), dtype=np.float32)
#             return TTSResult(audio=tone, sample_rate=sr, meta={"provider": "qiniu", "enabled": False})

#         headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json"
#         }
#         data = {
#             "audio": {
#                 "voice_type": settings.TTS_VOICE,
#                 "encoding": settings.TTS_ENCODING,   # 推荐 'wav'
#                 "speed_ratio": settings.TTS_SPEED
#             },
#             "request": {
#                 "text": text
#             }
#         }

#         try:
#             resp = self.session.post(self._url, headers=headers, json=data, timeout=settings.REQUEST_TIMEOUT)
#             resp.raise_for_status()
#             js = resp.json()

#             # 兼容返回格式：
#             # 1) 直接 base64 字段：js["audio"]["content"]
#             # 2) url：js["audio"]["url"]（若是wav/mp3可回取）
#             b64 = None
#             audio_url = None
#             # 常见层级
#             if isinstance(js.get("audio"), dict):
#                 b64 = js["audio"].get("content")
#                 audio_url = js["audio"].get("url")

#             wav_bytes = None
#             if b64:
#                 wav_bytes = base64.b64decode(b64)
#             elif audio_url:
#                 # 拉取URL（需公网网络通畅；避免mp3，故请求encoding=wav）
#                 r2 = self.session.get(audio_url, timeout=settings.REQUEST_TIMEOUT)
#                 r2.raise_for_status()
#                 wav_bytes = r2.content
#             else:
#                 # 其他结构兜底再找一层 data.audio
#                 if "data" in js and isinstance(js["data"].get("audio"), dict):
#                     b64 = js["data"]["audio"].get("content")
#                     audio_url = js["data"]["audio"].get("url")
#                     if b64:
#                         wav_bytes = base64.b64decode(b64)
#                     elif audio_url:
#                         r2 = self.session.get(audio_url, timeout=settings.REQUEST_TIMEOUT)
#                         r2.raise_for_status()
#                         wav_bytes = r2.content

#             if not wav_bytes:
#                 # 无法解析
#                 return TTSResult(audio=np.zeros(int(settings.AUDIO_SAMPLE_RATE*0.1), dtype=np.float32),
#                                  sample_rate=settings.AUDIO_SAMPLE_RATE,
#                                  meta={"provider": "qiniu", "error": "no_audio_in_response", "resp": str(js)[:300]})

#             audio, sr = _wav_bytes_to_float32(wav_bytes)
#             return TTSResult(audio=audio, sample_rate=sr, meta={"provider": "qiniu", "status": "ok"})

#         except requests.exceptions.RequestException as e:
#             body = getattr(e.response, "text", "") if hasattr(e, "response") else str(e)
#             return TTSResult(audio=np.zeros(int(settings.AUDIO_SAMPLE_RATE*0.1), dtype=np.float32),
#                              sample_rate=settings.AUDIO_SAMPLE_RATE,
#                              meta={"provider": "qiniu", "error": body[:300]})
#         except Exception as e:
#             return TTSResult(audio=np.zeros(int(settings.AUDIO_SAMPLE_RATE*0.1), dtype=np.float32),
#                              sample_rate=settings.AUDIO_SAMPLE_RATE,
#                              meta={"provider": "qiniu", "error": str(e)[:300]})

