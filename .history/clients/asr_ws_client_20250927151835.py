# clients/asr_ws_client.py
from __future__ import annotations
import asyncio, gzip, json, time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import websockets
from websockets.client import connect as ws_connect  # 显式用 client.connect

from config import settings
from utils.logging import write_log

# 与 HTTP 版一致的返回结构
@dataclass
class ASRResult:
    text: str
    confidence: float
    meta: Dict[str, Any]

# ========= 协议常量（按七牛文档） =========
PROTOCOL_VERSION = 0b0001

# Message Types
FULL_CLIENT_REQUEST   = 0b0001
AUDIO_ONLY_REQUEST    = 0b0010
FULL_SERVER_RESPONSE  = 0b1001
SERVER_ACK            = 0b1011
SERVER_ERROR_RESPONSE = 0b1111

# Message Type Specific Flags
NO_SEQUENCE       = 0b0000
POS_SEQUENCE      = 0b0001  # 包含正序序列号
NEG_SEQUENCE      = 0b0010
NEG_WITH_SEQUENCE = 0b0011

# 序列化 & 压缩
NO_SERIALIZATION   = 0b0000
JSON_SERIALIZATION = 0b0001
NO_COMPRESSION     = 0b0000
GZIP_COMPRESSION   = 0b0001

def _gen_header(message_type=FULL_CLIENT_REQUEST,
                message_type_specific_flags=NO_SEQUENCE,
                serial_method=JSON_SERIALIZATION,
                compression_type=GZIP_COMPRESSION,
                reserved_data=0x00) -> bytearray:
    """
    4字节对齐 Header：
      byte0: 高4位协议版本，低4位头部长度（以4字节为单位）
      byte1: 高4位 message_type，低4位 flags
      byte2: 高4位 序列化方式，低4位 压缩方式
      byte3: 预留
    """
    header = bytearray()
    header_size = 1  # 单位=4字节；这里固定一个基本头（4B）
    header.append((PROTOCOL_VERSION << 4) | header_size)
    header.append((message_type << 4) | message_type_specific_flags)
    header.append((serial_method << 4) | compression_type)
    header.append(reserved_data)
    return header

def _before_payload_with_seq(sequence: int) -> bytearray:
    # 紧跟 header 的 4 字节序列号（有 POS_SEQUENCE 时携带）
    b = bytearray()
    b.extend(sequence.to_bytes(4, 'big', signed=True))
    return b

def _parse_server_frame(res: bytes | bytearray | str) -> Dict[str, Any]:
    """
    解析服务端返回帧；返回 dict，主要看 payload_msg。
    - 自动解 gzip
    - 自动反序列化 JSON
    """
    if not isinstance(res, (bytes, bytearray)):
        return {'payload_msg': res}

    header_size = res[0] & 0x0F
    message_type = res[1] >> 4
    message_type_specific_flags = res[1] & 0x0F
    serialization_method = res[2] >> 4
    message_compression = res[2] & 0x0F

    payload = res[header_size * 4:]
    result: Dict[str, Any] = {}

    if message_type_specific_flags & 0x01:
        seq = int.from_bytes(payload[:4], "big", signed=True)
        result['payload_sequence'] = seq
        payload = payload[4:]

    result['is_last_package'] = bool(message_type_specific_flags & 0x02)

    if message_type == FULL_SERVER_RESPONSE:
        payload_size = int.from_bytes(payload[:4], "big", signed=True)
        payload_msg = payload[4:]
    elif message_type == SERVER_ACK:
        seq = int.from_bytes(payload[:4], "big", signed=True)
        result['seq'] = seq
        if len(payload) >= 8:
            payload_size = int.from_bytes(payload[4:8], "big", signed=False)
            payload_msg = payload[8:]
        else:
            payload_msg = b""
    elif message_type == SERVER_ERROR_RESPONSE:
        code = int.from_bytes(payload[:4], "big", signed=False)
        result['code'] = code
        payload_size = int.from_bytes(payload[4:8], "big", signed=False)
        payload_msg = payload[8:]
    else:
        payload_msg = payload

    if message_compression == GZIP_COMPRESSION:
        try:
            payload_msg = gzip.decompress(payload_msg)
        except Exception:
            pass

    if serialization_method == JSON_SERIALIZATION:
        try:
            payload_text = payload_msg.decode("utf-8")
            payload_msg = json.loads(payload_text)
        except Exception:
            try:
                payload_msg = payload_msg.decode("utf-8", errors="ignore")
            except Exception:
                pass
    else:
        try:
            payload_msg = payload_msg.decode("utf-8", errors="ignore")
        except Exception:
            pass

    result['payload_msg'] = payload_msg
    return result

def _float32_to_pcm16(audio: np.ndarray) -> bytes:
    a = np.clip(audio, -1.0, 1.0).astype(np.float32)
    pcm16 = (a * 32767.0).astype(np.int16)
    return pcm16.tobytes()

class ASRWsClient:
    """
    WebSocket 版 ASR 客户端：
      - 不需要 URL
      - 发送：配置包(JSON+gzip) + 音频分片(PCM16+gzip)
      - 接收：解析返回帧，取 result.text
    """
    def __init__(self, ws_url: Optional[str] = None, api_key: Optional[str] = None):
        self.ws_url = ws_url or settings.ASR_WS_URL
        self.api_key = api_key or getattr(settings, "API_KEY", None)

    async def _run(self, audio_np: np.ndarray, sample_rate: int,
                   seg_ms: int = 300, enable_punc: bool = True) -> Tuple[str, Dict[str, Any]]:
        
        # —— 确保单声道 float32 —— 
        if audio_np.ndim > 1:
            if audio_np.shape[1] > 1:
                audio_np = audio_np.mean(axis=1)
        # —— 强制重采样到 16kHz —— 
        def _resample_to_16k(x: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
            target = 16000
            if sr == target:
                return x.astype(np.float32), sr
            # 线性插值重采样（避免额外依赖）
            import numpy as np
            t_old = np.linspace(0, 1, num=len(x), endpoint=False)
            t_new = np.linspace(0, 1, num=int(len(x) * (target / sr)), endpoint=False)
            y = np.interp(t_new, t_old, x).astype(np.float32)
            return y, target
        
        audio_np, sample_rate = _resample_to_16k(audio_np.astype(np.float32), int(sample_rate))
        
        
        # 1) 组装配置帧
        seq = 1
        req = {
            "user": {"uid": "voicery-ws"},
            "audio": {
                "format": "pcm",        # 发送原始pcm16（gzip压缩）
                "sample_rate": int(sample_rate),
                "bits": 16,
                "channel": 1,
                "codec": "raw"
            },
            "request": {
                "model_name": "asr",
                "enable_punc": bool(enable_punc)
            }
        }
        payload_bytes = gzip.compress(json.dumps(req, ensure_ascii=False).encode("utf-8"))
        cfg_frame = bytearray(_gen_header(message_type_specific_flags=POS_SEQUENCE))
        cfg_frame.extend(_before_payload_with_seq(seq))
        cfg_frame.extend((len(payload_bytes)).to_bytes(4, "big"))
        cfg_frame.extend(payload_bytes)

        # 2) 切片音频（每 seg_ms 一片）
        pcm = _float32_to_pcm16(audio_np)
        bytes_per_sample = 2
        frames_per_seg = int(sample_rate * seg_ms / 1000)
        bytes_per_seg = frames_per_seg * bytes_per_sample
        segments: List[bytes] = [pcm[i:i+bytes_per_seg] for i in range(0, len(pcm), bytes_per_seg)] or [pcm]

        write_log(settings.LOG_PATH, {"event": "asr_ws_open", "url": self.ws_url, "segs": len(segments)})

        # 3) 连接（注意：extra_headers 用“列表[(k,v)]”形式）
        headers_list = [("Authorization", f"Bearer {self.api_key}")]
        t0 = time.time()
        try:
            async with ws_connect(self.ws_url,
                                  extra_headers=headers_list,
                                  max_size=100_000_000,
                                  open_timeout=10,
                                  ping_interval=None) as ws:

                # 3.1) 发配置帧
                await ws.send(cfg_frame)
                try:
                    res = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    parsed = _parse_server_frame(res)
                    write_log(settings.LOG_PATH, {"event": "asr_ws_cfg_ack",
                                                  "msg": str(parsed.get("payload_msg"))[:200]})
                except asyncio.TimeoutError:
                    write_log(settings.LOG_PATH, {"event": "asr_ws_cfg_timeout"})
                    return "（ASR初始化超时）", {"transport": "ws", "stage": "cfg_timeout"}

                # 3.2) 循环发送音频分片；间歇读取增量
                text_accum = ""
                for chunk in segments:
                    seq += 1
                    compressed = gzip.compress(chunk)
                    audio_frame = bytearray(_gen_header(message_type=AUDIO_ONLY_REQUEST,
                                                        message_type_specific_flags=POS_SEQUENCE))
                    audio_frame.extend(_before_payload_with_seq(seq))
                    audio_frame.extend((len(compressed)).to_bytes(4, "big"))
                    audio_frame.extend(compressed)
                    await ws.send(audio_frame)

                    # 可选：尝试读取一帧增量（不强制每片都等）
                    try:
                        res = await asyncio.wait_for(ws.recv(), timeout=0.5)
                        parsed = _parse_server_frame(res)
                        msg = parsed.get("payload_msg")
                        new_text = ""
                        if isinstance(msg, dict):
                            # 常见：{"result":{"text":"..."}}
                            new_text = (msg.get("result") or {}).get("text", "") or new_text
                        elif isinstance(msg, str):
                            new_text = msg
                        if new_text and new_text != text_accum:
                            text_accum = new_text
                            write_log(settings.LOG_PATH, {"event": "asr_ws_partial", "len": len(text_accum)})
                    except asyncio.TimeoutError:
                        pass

                # 3.3) 所有分片发完后，再等最多3秒拿最终结果
                deadline = time.time() + 3.0
                while time.time() < deadline:
                    try:
                        res = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        parsed = _parse_server_frame(res)
                        msg = parsed.get("payload_msg")
                        new_text = ""
                        if isinstance(msg, dict):
                            new_text = (msg.get("result") or {}).get("text", "") or new_text
                        elif isinstance(msg, str):
                            new_text = msg
                        if new_text and new_text != text_accum:
                            text_accum = new_text
                            write_log(settings.LOG_PATH, {"event": "asr_ws_final_partial", "len": len(text_accum)})
                        if parsed.get("is_last_package"):
                            break
                    except asyncio.TimeoutError:
                        break

                ms = int((time.time() - t0) * 1000)
                write_log(settings.LOG_PATH, {"event": "asr_ws_done", "ms": ms, "len": len(text_accum)})
                return text_accum or "", {"transport": "ws", "ms": ms}

        except Exception as e:
            write_log(settings.LOG_PATH, {"event": "asr_ws_error", "error": str(e)[:300]})
            return "（ASR请求失败）", {"transport": "ws", "error": str(e)[:300]}

    def transcribe(self, audio_np: np.ndarray, sample_rate: int, audio_url: Optional[str] = None) -> ASRResult:
        """
        与 HTTP 版对齐的同步接口（忽略 audio_url）。
        """
        # 单声道
        if audio_np.ndim > 1:
            if audio_np.shape[0] < audio_np.shape[1]:
                audio_np = audio_np.mean(axis=1)
            else:
                audio_np = audio_np[:, 0]
        text, meta = asyncio.run(self._run(audio_np.astype(np.float32), int(sample_rate)))
        return ASRResult(text=text or "", confidence=0.0, meta=meta)
