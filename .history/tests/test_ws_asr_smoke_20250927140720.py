# tests/test_ws_asr_smoke.py
import argparse, base64, io, wave, time, sys, os
import numpy as np
import requests
from clients.asr_ws_client import ASRWsClient
from config import settings



def read_wav_as_float32(path):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sampw = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    assert sampw == 2, "只支持16-bit PCM WAV"
    x = np.frombuffer(frames, dtype=np.int16).astype(np.float32)/32768.0
    if ch == 2:
        x = x.reshape(-1, 2).mean(axis=1)
    return x, sr

def tts_to_wav_bytes_via_http(text: str, voice: str, speed: float=1.0) -> bytes:
    """仅用于烟雾测试：直调HTTP让TTS返回WAV字节；不改项目内的mp3逻辑。"""
    url = f"{getattr(settings,'BASE_URL','https://openai.qiniu.com/v1').rstrip('/')}/voice/tts"
    headers = {"Authorization": f"Bearer {settings.API_KEY}", "Content-Type":"application/json"}
    data = {
        "audio": {"voice_type": voice, "encoding": "wav", "speed_ratio": float(speed)},
        "request": {"text": text}
    }
    r = requests.post(url, headers=headers, json=data, timeout=settings.REQUEST_TIMEOUT)
    r.raise_for_status()
    b64 = r.json().get("data")
    assert b64, f"TTS响应无data：{r.text[:200]}"
    return base64.b64decode(b64)

def wav_bytes_to_float32(wav_bytes: bytes):
    bio = io.BytesIO(wav_bytes)
    with wave.open(bio, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sampw = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    assert sampw == 2, "只支持16-bit PCM WAV"
    x = np.frombuffer(frames, dtype=np.int16).astype(np.float32)/32768.0
    if ch == 2:
        x = x.reshape(-1, 2).mean(axis=1)
    return x, sr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, default=None, help="本地WAV路径（16kHz/单声道优先）")
    parser.add_argument("--text", type=str, default="你好，我在做一次ASR WebSocket 烟雾测试。")
    parser.add_argument("--voice", type=str, default=getattr(settings, "TTS_VOICE", "qiniu_zh_male_cxkjns"))
    parser.add_argument("--speed", type=float, default=getattr(settings, "TTS_SPEED", 1.0))
    args = parser.parse_args()

    # 准备音频：优先读本地wav；否则临时用TTS合成WAV
    if args.wav:
        audio, sr = read_wav_as_float32(args.wav)
        print(f"[WAV] load ok: {args.wav}, sr={sr}, len={len(audio)}")
    else:
        print("[TTS] 生成临时WAV以做回环验证...")
        wav_bytes = tts_to_wav_bytes_via_http(args.text, args.voice, args.speed)
        audio, sr = wav_bytes_to_float32(wav_bytes)
        print(f"[TTS] ok → WAV: sr={sr}, len={len(audio)}")

    # 送入 WS-ASR
    asr = ASRWsClient()
    t0 = time.time()
    res = asr.transcribe(audio, sr)
    dt = (time.time()-t0)*1000
    print(f"[ASR-WS] done in {dt:.0f} ms")
    print("text:", res.text)
    print("meta:", res.meta)

if __name__ == "__main__":
    main()
