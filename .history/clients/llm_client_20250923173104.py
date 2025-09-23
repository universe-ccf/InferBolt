# app/clients/llm_client.py
from __future__ import annotations
from typing import List, Dict, Any
from app.core.types import Message

class LLMClient:
    def __init__(self, model: str, temperature: float = 0.7, api_key: str | None = None, base_url: str | None = None):
        ...

    def complete(self, messages: List[Message], max_tokens: int = 512) -> str:
        """通用对话补全"""
        ...

    def classify(self, text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """小型分类/意图识别（返回JSON），用作dispatcher兜底"""
        ...

# app/clients/asr_client.py
class ASRClient:
    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
        ...

    def transcribe(self, audio_bytes: bytes, language: str | None = None) -> str:
        """音频→文本（MVP：同步返回完整文本）"""
        ...

# app/clients/tts_client.py
class TTSClient:
    def __init__(self, voice: str, api_key: str | None = None, base_url: str | None = None):
        ...

    def synthesize(self, text: str, speed: float = 1.0, emotion: str | None = None) -> bytes:
        """文本→音频（返回bytes：mp3/wav）"""
        ...
