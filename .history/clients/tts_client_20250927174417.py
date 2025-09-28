# clients/tts_client.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
from config import settings
import base64
import requests, os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from utils.cache import sha256_text, cache_get_file, cache_put_file
from utils.logging import write_log


# ======== WAV/PCM 工具：无第三方依赖，定位静音与拼接问题 ========

def _pcm16_rms_dbfs(pcm: bytes) -> float:
    """计算 int16 PCM 的 RMS(dBFS)。空数据返回 -inf。"""
    if not pcm:
        return float("-inf")
    # 以 2 字节为一采样
    n = len(pcm) // 2
    if n == 0:
        return float("-inf")
    # 将 bytes -> int16（小端）
    import array
    a = array.array('h')
    a.frombytes(pcm[:n*2])
    # 均方根
    s = 0.0
    for v in a:
        s += (v * v)
    rms = math.sqrt(s / n)
    # dBFS（满刻度 32768）
    if rms <= 1e-3:
        return float("-inf")
    return 20.0 * math.log10(rms / 32768.0)

def _trim_silence_pcm16(pcm: bytes, sample_rate: int, thr_dbfs: float, win_ms: int, pad_ms: int) -> bytes:
    """按 RMS 窗裁剪两端静音；返回裁剪后的 PCM16。"""
    if not pcm:
        return pcm
    import array
    a = array.array('h'); a.frombytes(pcm)
    win = max(1, int(sample_rate * win_ms / 1000))
    pad = max(0, int(sample_rate * pad_ms / 1000))

    def _find_head_idx():
        acc, cnt = 0.0, 0
        best = 0
        for i in range(0, len(a), win):
            seg = a[i:i+win]
            if not seg:
                break
            s = 0.0
            for v in seg: s += v*v
            rms = math.sqrt(s / max(1, len(seg)))
            db = -999.0 if rms <= 1e-3 else 20.0*math.log10(rms/32768.0)
            if db > thr_dbfs:
                best = max(0, i - pad)
                return best
        return 0

    def _find_tail_idx():
        for i in range(len(a), 0, -win):
            seg = a[max(0, i-win):i]
            if not seg:
                break
            s = 0.0
            for v in seg: s += v*v
            rms = math.sqrt(s / max(1, len(seg)))
            db = -999.0 if rms <= 1e-3 else 20.0*math.log10(rms/32768.0)
            if db > thr_dbfs:
                return min(len(a), i + pad)
        return len(a)

    h = _find_head_idx()
    t = _find_tail_idx()
    if t <= h:
        return b""
    b = array.array('h', a[h:t])
    return b.tobytes()

def _read_wav_bytes(wav_bytes: bytes) -> tuple[int, int, int, bytes]:
    """
    解析 WAV，返回 (sr, channels, sampwidth_bytes, pcm_bytes)
    若非 PCM16/单/双声道，仍返回实际参数与原始帧方便上层处理。
    """
    bio = io.BytesIO(wav_bytes)
    with wave.open(bio, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    return sr, ch, sw, frames

def _stereo_to_mono_pcm16(pcm: bytes) -> bytes:
    """双声道 int16 -> 单声道（简单平均）。"""
    import array
    a = array.array('h'); a.frombytes(pcm)
    if len(a) % 2 != 0:
        a = a[:-1]
    out = array.array('h')
    for i in range(0, len(a), 2):
        m = int((a[i] + a[i+1]) / 2)
        out.append(m)
    return out.tobytes()

def _resample_pcm16_linear(pcm: bytes, sr_src: int, sr_tgt: int) -> bytes:
    """最简单的线性插值重采样（int16 PCM）。"""
    if sr_src == sr_tgt or not pcm:
        return pcm
    import array
    import numpy as np
    a = array.array('h'); a.frombytes(pcm)
    x = np.frombuffer(a, dtype=np.int16).astype('float32') / 32768.0
    t_old = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    t_new = np.linspace(0.0, 1.0, num=int(len(x) * (sr_tgt / sr_src)), endpoint=False)
    y = np.interp(t_new, t_old, x)
    y16 = (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16)
    return y16.tobytes()


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

