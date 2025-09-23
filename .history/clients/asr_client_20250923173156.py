# app/clients/asr_client.py
class ASRClient:
    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
        ...

    def transcribe(self, audio_bytes: bytes, language: str | None = None) -> str:
        """音频→文本（MVP：同步返回完整文本）"""
        ...