# app/clients/tts_client.py
class TTSClient:
    def __init__(self, voice: str, api_key: str | None = None, base_url: str | None = None):
        ...

    def synthesize(self, text: str, speed: float = 1.0, emotion: str | None = None) -> bytes:
        """文本→音频（返回bytes：mp3/wav）"""
        ...