# app/clients/llm_client.py
from __future__ import annotations
import os, json, requests, re
from typing import List, Dict, Any
from core.types import Message
from config import settings

class LLMClient:
    def __init__(self, model: str | None = None, temperature: float = None,
                 api_key: str | None = None, base_url: str | None = None):
        self.model = model or settings.LLM_MODEL
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.api_key = api_key or settings.API_KEY
        self.base_url = (base_url or settings.BASE_URL).rstrip("/")
        if not self.api_key:
            raise RuntimeError("API_KEY 未配置，请在 .env 中设置 API_KEY=")
        self._chat_url = f"{self.base_url}/chat/completions"

    def _to_openai_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        # 我们内部 Message 的字段名与 OpenAI 兼容，可直接映射
        out = []
        for m in messages:
            out.append({"role": m.role, "content": m.content})
        return out

    # 通用对话补全
    def complete(self, messages: List[Message], max_tokens: int = 512) -> str:
        payload = {
            "model": self.model,
            "messages": self._to_openai_messages(messages),
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        try:
            resp = requests.post(self._chat_url, headers=headers, json=payload, timeout=settings.REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            # —— 某些厂商兼容层有细微差异，这里更稳一点：
            if isinstance(data, dict) and "choices" in data and data["choices"]:
                choice = data["choices"][0]
                # OpenAI 兼容通常是 message.content；部分实现也可能是 delta/content 或 text
                if "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"]
                if "text" in choice:
                    return choice["text"]
            # 打印帮助排查
            print("Unexpected LLM response:", resp.text[:500])
            return "（抱歉，模型响应解析失败。）"
        except requests.exceptions.RequestException as e:
            print("LLM HTTP error:", getattr(e.response, "text", str(e)))
            return "（抱歉，模型接口暂时不可用，请稍后再试。）"
        except Exception as e:
            print("LLM parse error:", type(e).__name__, e)
            return "（抱歉，模型响应异常。）"

    # 小型分类/意图识别（返回JSON），用作dispatcher兜底
    def classify(self, text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        MVP：用提示词约束输出 JSON；后续可改“函数调用/JSON模式”。
        """
        system = Message(role="system", content=(
            "你是一个严格的JSON分类器。只输出一个JSON对象，字段必须与schema一致，"
            "不要输出任何解释性文字。"
        ))
        user = Message(role="user", content=f"输入文本：{text}\n请依据schema输出JSON：{json.dumps(schema, ensure_ascii=False)}")
        raw = self.complete([system, user], max_tokens=256)

        # 粗暴从返回中截取第一个大括号块进行解析，防御非纯JSON的情况
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            candidate = raw[start:end+1] if start != -1 and end != -1 else "{}"
            data = json.loads(candidate)
        except Exception:
            data = {}
        # 兜底填充
        return {
            "intent": data.get("intent", "unknown"),
            "skill": data.get("skill"),
            "confidence": float(data.get("confidence", 0.0))
        }




