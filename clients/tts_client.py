# clients/tts_client.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
from config import settings
import base64, math
import requests, os, io, wave
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from utils.cache import sha256_text, cache_get_file, cache_put_file
from utils.logging import write_log


# ======== WAV/PCM 工具：无第三方依赖，定位静音与拼接问题 ========

def _read_wav_bytes(wav_bytes: bytes) -> tuple[int, int, int, bytes]:
    """解析 WAV，返回 (sr, channels, sampwidth_bytes, pcm_bytes)。"""
    bio = io.BytesIO(wav_bytes)
    with wave.open(bio, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    return sr, ch, sw, frames

def _pcm16_rms_dbfs(pcm: bytes) -> float:
    """计算 int16 PCM 的 RMS(dBFS)。空数据返回 -inf。"""
    if not pcm:
        return float("-inf")
    n = len(pcm) // 2
    if n == 0:
        return float("-inf")
    import array
    a = array.array('h'); a.frombytes(pcm[:n*2])
    s = 0.0
    for v in a: s += (v*v)
    rms = (s / n) ** 0.5
    if rms <= 1e-3:
        return float("-inf")
    return 20.0 * math.log10(rms / 32768.0)

def _stereo_to_mono_pcm16(pcm: bytes) -> bytes:
    """双声道 int16 -> 单声道（简单平均）。"""
    import array
    a = array.array('h'); a.frombytes(pcm)
    if len(a) % 2 != 0:
        a = a[:-1]
    out = array.array('h')
    for i in range(0, len(a), 2):
        out.append(int((a[i] + a[i+1]) / 2))
    return out.tobytes()

def _trim_silence_pcm16(pcm: bytes, sample_rate: int, thr_dbfs: float, win_ms: int, pad_ms: int) -> bytes:
    """按 RMS 窗裁剪两端静音；返回裁剪后的 PCM16。"""
    if not pcm:
        return pcm
    import array
    a = array.array('h'); a.frombytes(pcm)
    win = max(1, int(sample_rate * win_ms / 1000))
    pad = max(0, int(sample_rate * pad_ms / 1000))

    def _db(seg):
        if not seg: return -999.0
        s = 0.0
        for v in seg: s += v*v
        rms = (s / max(1, len(seg))) ** 0.5
        return -999.0 if rms <= 1e-3 else 20.0*math.log10(rms/32768.0)

    # 找头
    head = 0
    for i in range(0, len(a), win):
        if _db(a[i:i+win]) > thr_dbfs:
            head = max(0, i - pad); break
    # 找尾
    tail = len(a)
    for i in range(len(a), 0, -win):
        if _db(a[max(0, i-win):i]) > thr_dbfs:
            tail = min(len(a), i + pad); break
    if tail <= head:
        return b""
    return array.array('h', a[head:tail]).tobytes()

def _pack_wav_bytes(pcm: bytes, sample_rate: int, channels: int = 1, sampwidth: int = 2) -> bytes:
    """把 PCM16 打包成 WAV 字节。"""
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return bio.getvalue()



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

        voice = voice_type or settings.TTS_VOICE
        speed = speed_ratio if (speed_ratio is not None) else settings.TTS_SPEED
        encoding = settings.TTS_ENCODING  # 建议先设为 "wav" 便于排查/裁剪

        # === 缓存（命中则直接返回路径） ===
        audio_key = None
        if settings.ENABLE_SPEECH_CACHE:
            sig = f"{text}||{voice}||{speed}||{encoding}"
            audio_key = sha256_text(sig)
            cached = cache_get_file(settings.CACHE_TTS_DIR, audio_key, encoding)
            if cached:
                return TTSResult(cached, None, {"provider":"qiniu","cache":"hit"})

        # 防长文本导致超时
        MAX_TTS_CHARS = 300
        if len(text) > MAX_TTS_CHARS:
            write_log(settings.LOG_PATH, {"event":"tts_truncate", "orig_len": len(text)})
            text = text[:MAX_TTS_CHARS] + "……"

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {
            "audio": {"voice_type": voice, "encoding": encoding, "speed_ratio": float(speed)},
            "request": {"text": text}
        }

        write_log(settings.LOG_PATH, {"event":"tts_request","voice":voice,"speed":float(speed),"encoding":encoding})

        try:
            resp = self.session.post(self._url, headers=headers, json=data, timeout=settings.REQUEST_TIMEOUT)
            resp.raise_for_status()
            js = resp.json()
            b64 = js.get("data")
            if not b64:
                return TTSResult(None, None, {"provider":"qiniu","error":"no_audio_data","resp": str(js)[:300]})
            audio_bytes = base64.b64decode(b64)
            write_log(settings.LOG_PATH, {"event":"tts_response","bytes": len(audio_bytes)})

            audio_bytes_out = audio_bytes
            out_sr = None

            # === 只有 WAV 我们才做“静音检测/裁剪/规范化” ===
            if encoding.lower() == "wav":
                try:
                    sr, ch, sw, pcm = _read_wav_bytes(audio_bytes)
                    out_sr = sr
                    # 记录原始片的关键指标
                    rms_db = _pcm16_rms_dbfs(pcm) if sw == 2 else float("-inf")
                    write_log(settings.LOG_PATH, {
                        "event":"tts_wav_info","sr":sr,"ch":ch,"sw":sw,
                        "frames": (len(pcm)//2 if sw==2 else len(pcm)),
                        "rms_db": float(rms_db),
                    })

                    # 规范化：只处理 16-bit；其他情况不动直接落盘
                    if sw == 2:
                        # 双声道转单声道
                        if ch == 2:
                            pcm = _stereo_to_mono_pcm16(pcm)
                            ch = 1
                        # （可选）重采样到统一采样率
                        target_sr = getattr(settings, "TTS_TARGET_SR", sr)
                        if sr != target_sr:
                            # 这里为了不增加依赖，先不重采样；若需要可扩展成线性插值版
                            # 注：若你项目已装 numpy，可以引入重采样函数再启用
                            pass

                        # 分片级裁剪首尾静音
                        pcm_trim = _trim_silence_pcm16(
                            pcm, sample_rate=sr,
                            thr_dbfs=getattr(settings,"TTS_SILENCE_DBFS",-45.0),
                            win_ms=getattr(settings,"TTS_RMS_WIN_MS",30),
                            pad_ms=getattr(settings,"TTS_TRIM_PAD_MS",60),
                        )
                        if pcm_trim and len(pcm_trim) < len(pcm):
                            write_log(settings.LOG_PATH, {
                                "event":"tts_trim_applied",
                                "before_frames": len(pcm)//2,
                                "after_frames": len(pcm_trim)//2
                            })
                            pcm = pcm_trim

                        # 重新打包为 WAV 字节
                        audio_bytes_out = _pack_wav_bytes(pcm, sample_rate=sr, channels=1, sampwidth=2)
                        out_sr = sr
                    else:
                        write_log(settings.LOG_PATH, {"event":"tts_warn_non_pcm16","sw":sw})
                        # sw != 2 时，不动 audio_bytes

                except Exception as e:
                    write_log(settings.LOG_PATH, {"event":"tts_process_error","error": str(e)[:300]})
                    # 出现处理异常，就用原始 audio_bytes_out

            # === 落地为文件（缓存或临时） ===
            if settings.ENABLE_SPEECH_CACHE and audio_key:
                fpath = cache_put_file(settings.CACHE_TTS_DIR, audio_key, encoding, audio_bytes_out)
            else:
                os.makedirs(settings.CACHE_TTS_DIR, exist_ok=True)
                fpath = os.path.join(settings.CACHE_TTS_DIR, f"tmp_{sha256_text(b64)}.{encoding}")
                with open(fpath, "wb") as f:
                    f.write(audio_bytes_out)

            write_log(settings.LOG_PATH, {"event":"tts_save_done","path": fpath, "bytes": len(audio_bytes_out)})
            return TTSResult(fpath, out_sr, {"provider":"qiniu","status":"ok","voice":voice,"speed":speed})

        except requests.exceptions.RequestException as e:
            body = getattr(e.response,"text","") if hasattr(e,"response") else str(e)
            write_log(settings.LOG_PATH, {"event":"tts_error","error": body[:300]})
            return TTSResult(None, None, {"provider":"qiniu","error": body[:300]})

        except Exception as e:
            write_log(settings.LOG_PATH, {"event":"tts_error","error": str(e)[:300]})
            return TTSResult(None, None, {"provider":"qiniu","error": str(e)[:300]})

