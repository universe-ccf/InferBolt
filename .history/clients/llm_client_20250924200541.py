# app/clients/llm_client.py
from __future__ import annotations
import os, json, requests
from typing import List, Dict, Any
from core.types import Message
from config import settings

class LLMClient:
    def __init__(self, model: str, temperature: float = 0.7, api_key: str | None = None, base_url: str | None = None):
        ...

    def complete(self, messages: List[Message], max_tokens: int = 512) -> str:
        """通用对话补全"""
        ...

    def classify(self, text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """小型分类/意图识别（返回JSON），用作dispatcher兜底"""
        ...




