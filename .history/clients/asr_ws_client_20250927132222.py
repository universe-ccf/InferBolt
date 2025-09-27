# clients/asr_ws_client.py
from __future__ import annotations
import asyncio, gzip, json, time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import websockets

from config import settings
from utils.logging import write_log

# 与 HTTP 版保持一致的数据结构
@dataclass
class ASRResult:
    text: str
    confidence: float
    meta: Dict[str, Any]

# ====== 协议常量（按七牛文档）======
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

def generate_header(message_type=FULL_CLIENT_REQUEST,
                    message_type_specific_flags=NO_SEQUENCE,
                    serial_method=JSON_SERIALIZATION,
                    compression_type=GZIP_COMPRESSION,
                    reserved_data=0x00) -> bytearray:
    header = bytearray()
    header_size = 1  # 4字节为单位的头部长度
    header.append((PROTOCOL_VERSION << 4) | header_size)
    header.append((message_type << 4) | message_type_specific_flags)
    header.append((serial_method << 4) | compression_type)
    header.append(reserved_data)
    return header

def generate_before_payload(sequence: int) -> bytearray:
    before_payload = bytearray()
    before_payload.extend(sequence.to_bytes(4, 'big', signed=True))
    return before_payload

def parse_response(res):
    """
    解析服务端返回（兼容 bytes/str），按文档示例。返回 dict，核心看 result.payload_msg.*
    """
    if not isinstance(res, (bytes, bytearray)):
        return {'payload_msg': res}

    header_size = res[0] & 0x0F
    message_type = res[1] >> 4
    message_type_specific_flags = res[1] & 0x0F
    serialization_method = res[2] >> 4
    message_compression = res[2] & 0x0F

    payload = res[header_size * 4:]
    result = {}
    if message_type_specific_flags & 0x01:
        seq = int.from_bytes(payload[:4], "big", signed=True)
        result['payload_sequence'] = seq
        payload = payload[4:]

    result['is_last_package'] = bool(message_type_specific_flags & 0x02)

    # 各类型负载
    if message_type == FULL_SERVER_RESPONSE:
        payload_size = int.from_bytes(payload[:4], "big", signed=True)
        payload_msg = payload[4:]
    elif message_type == SERVER_ACK:
        # 可能包含 seq 与 payload_size
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

    # 解压
    if message_compression == GZIP_COMPRESSION:
        try:
            payload_msg = gzip.decompress(payload_msg)
        except Exception:
            pass

    # 反序列化
    if serialization_method == JSON_SERIALIZATION:
        try:
            payload_text = payload_msg.decode("utf-8")
            payload_msg = json.loads(payload_text)
        except Exception:
            # fallback 文本
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

def float32_to_pcm16(audio: np.ndarray) -> bytes:
    a = np.clip(audio, -1.0, 1.0).astype(np.float32)
    pcm16 = (a * 32767.0).astype(np.int16)
    return pcm16.tobytes()

class ASRWsClient:
    """
    WebSocket 版 ASR 客户端。
    - 不需要 URL
    - 发送：配置包(JSON+gzip) + 音频分片(PCM16+gzip)
    - 接收：解析返回，取 result.text
    """
    def __init__(self, ws_url: Optional[str] = None, api_key: Optional[str] = None):
        self.ws_url = ws_url or settings.ASR_WS_URL
        self.api_key = api_key or getattr(settings, "API_KEY", None)

    async def _transcribe_async(self, audio_np: np.ndarray, sample_rate: int,
                                seg_ms: int = 300, enable_punc: bool = True) -> Tuple[str, Dict[str, Any]]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        seq = 1
        text_accum = ""   # 累积文本
        t0 = time.time()
        # 构造配置包
        req = {
            "user": {"uid": "voicery-ws"},
            "audio": {
                "format": "pcm", "sample_rate": int(sample_rate),
                "bits": 16, "channel": 1, "codec": "raw"
            },
            "request": {"model_name": "asr", "enable_punc": bool(enable_punc)}
        }
        payload_bytes = gzip.compress(json.dumps(req, ensure_ascii=False).encode("utf-8"))
        full_client_request = bytearray(generate_header(message_type_specific_flags=POS_SEQUENCE))
        full_client_request.extend(generate_before_payload(sequence=seq))
        full_client_request.extend((len(payload_bytes)).to_bytes(4, "big"))
        full_client_request.extend(payload_bytes)

        # PCM 分片（把整段切成 seg_ms 发送）
        pcm = float32_to_pcm16(audio_np)
        bytes_per_sample = 2  # int16
        frames_per_seg = int(sample_rate * seg_ms / 1000)
        bytes_per_seg = frames_per_seg * bytes_per_sample
        segments = []
        for i in range(0, len(pcm), bytes_per_seg):
            segments.append(pcm[i:i+bytes_per_seg])
        if not segments:
            segments = [pcm]

        write_log(settings.LOG_PATH, {"event": "asr_ws_open", "url": self.ws_url, "segs": len(segments)})

        try:
            async with websockets.connect(self.ws_url, extra_headers=headers, max_size=100_000_000) as ws:
                # 1) 发送配置
                await ws.send(full_client_request)
                try:
                    res = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    parsed = parse_response(res)
                    write_log(settings.LOG_PATH, {"event":"asr_ws_cfg_ack", "msg": str(parsed.get("payload_msg"))[:200]})
                except asyncio.TimeoutError:
                    write_log(settings.LOG_PATH, {"event":"asr_ws_cfg_timeout"})
                    return "（ASR初始化超时）", {"transport":"ws","stage":"cfg_timeout"}

                # 2) 发送分片 + 读增量
                for chunk in segments:
                    seq += 1
                    compressed_chunk = gzip.compress(chunk)
                    audio_req = bytearray(generate_header(message_type=AUDIO_ONLY_REQUEST,
                                                          message_type_specific_flags=POS_SEQUENCE))
                    audio_req.extend(generate_before_payload(sequence=seq))
                    audio_req.extend((len(compressed_chunk)).to_bytes(4, "big"))
                    audio_req.extend(compressed_chunk)
                    await ws.send(audio_req)

                    # 尝试读一条响应（非必须每片都等，先简化为等一条）
                    try:
                        res = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        parsed = parse_response(res)
                        msg = parsed.get("payload_msg")
                        # 兼容多格式取文本
                        new_txt = ""
                        if isinstance(msg, dict):
                            # 常见路径：{"result":{"text":"..."}}
                            if msg.get("result") and msg["result"].get("text"):
                                new_txt = msg["result"]["text"]
                            elif msg.get("payload_msg") and isinstance(msg["payload_msg"], dict):
                                inner = msg["payload_msg"]
                                if inner.get("result") and inner["result"].get("text"):
                                    new_txt = inner["result"]["text"]
                        elif isinstance(msg, str):
                            new_txt = msg
                        if new_txt and new_txt != text_accum:
                            text_accum = new_txt
                            write_log(settings.LOG_PATH, {"event":"asr_ws_partial", "len":len(text_accum)})
                    except asyncio.TimeoutError:
                        # 忽略，继续发下一片
                        pass

                # 3) 发送完毕后，再等几次结果（等待最终文本）
                final_wait_deadline = time.time() + 3.0  # 最多再等3秒
                while time.time() < final_wait_deadline:
                    try:
                        res = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        parsed = parse_response(res)
                        msg = parsed.get("payload_msg")
                        new_txt = ""
                        if isinstance(msg, dict):
                            if msg.get("result") and msg["result"].get("text"):
                                new_txt = msg["result"]["text"]
                        elif isinstance(msg, str):
                            new_txt = msg
                        if new_txt and new_txt != text_accum:
                            text_accum = new_txt
                            write_log(settings.LOG_PATH, {"event":"asr_ws_final_partial", "len":len(text_accum)})
                        if parsed.get("is_last_package"):
                            break
                    except asyncio.TimeoutError:
                        break

                tcost = int((time.time()-t0)*1000)
                write_log(settings.LOG_PATH, {"event":"asr_ws_done","ms":tcost,"len":len(text_accum)})
                return text_accum or "", {"transport":"ws","ms":tcost}

        except Exception as e:
            write_log(settings.LOG_PATH, {"event":"asr_ws_error", "error": str(e)[:300]})
            return "（ASR请求失败）", {"transport":"ws","error":str(e)[:300]}

    def transcribe(self, audio_np: np.ndarray, sample_rate: int, audio_url: Optional[str]=None) -> ASRResult:
        """
        对齐 HTTP 版签名：忽略 audio_url。阻塞式封装。
        """
        if audio_np.ndim > 1:
            # 取单声道
            audio_np = audio_np[:, 0] if audio_np.shape[1] > 0 else audio_np.mean(axis=1)
        text, meta = asyncio.run(self._transcribe_async(audio_np.astype(np.float32), int(sample_rate)))
        return ASRResult(text=text or "", confidence=0.0, meta=meta)
