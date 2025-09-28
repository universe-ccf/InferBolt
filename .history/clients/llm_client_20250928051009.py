# clients/llm_client.py
from __future__ import annotations
import os, json, requests, re
from typing import List, Dict, Any
from core.types import Message
from config import settings
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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

        # 带重试的 Session（连超时/读超时/502/503/504 自动重试）
        self.session = requests.Session()
        retry = Retry(
            total=settings.HTTP_MAX_RETRIES,
            read=settings.HTTP_MAX_RETRIES,
            connect=settings.HTTP_MAX_RETRIES,
            backoff_factor=settings.HTTP_BACKOFF_SEC,
            status_forcelist=(429, 502, 503, 504),
            allowed_methods=frozenset(["POST"])
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _to_openai_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        # 我们内部 Message 的字段名与 OpenAI 兼容，可直接映射
        out = []
        for m in messages:
            out.append({"role": m.role, "content": m.content})
        return out

    # 通用对话补全

    def complete(self, messages, max_tokens=512, temperature=None, stream=False, timeout=None):
        """
        同时支持非流式与流式（SSE）两种返回。
        - stream=False：一次性 JSON（标准 openai 兼容）
        - stream=True ：SSE，逐行 'data: {...}'，累积 delta.content
        """

        base = (self.base_url or "").rstrip("/")
        url = f"{base}/v1/chat/completions" if not base.endswith("/v1") else f"{base}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # 兼容你的 Message 类或 dict
        payload_msgs = []
        for m in messages:
            if hasattr(m, "role") and hasattr(m, "content"):
                payload_msgs.append({"role": m.role, "content": m.content})
            elif isinstance(m, dict):
                payload_msgs.append(m)
            else:
                raise TypeError(f"Invalid message item: {m}")

        payload = {
            "model": getattr(settings, "MODEL_NAME", "doubao-seed-1.6-flash"),
            "messages": payload_msgs,
            "stream": bool(stream),
            "max_tokens": int(max_tokens) if max_tokens else None,
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)
        payload = {k: v for k, v in payload.items() if v is not None}

        req_timeout = timeout or getattr(settings, "REQUEST_TIMEOUT", 30)
        requester = getattr(self, "session", None) or requests

        try:
            if stream:
                # —— 流式：SSE —— #
                resp = requester.post(url, headers=headers, json=payload, timeout=req_timeout, stream=True)
                if resp.status_code >= 400:
                    # 流式下不能直接 resp.json()
                    txt = (resp.text or "").strip()
                    raise RuntimeError(f"LLM HTTP error (stream, status={resp.status_code}): {txt[:500]}")

                full_text = []
                for raw in resp.iter_lines(decode_unicode=True):
                    if not raw:
                        continue
                    # 常见格式： "data: {...}" 或 "data: [DONE]"
                    line = raw.strip()
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            # openai 兼容：choices[0].delta.content
                            choices = chunk.get("choices") or []
                            if choices:
                                delta = choices[0].get("delta") or {}
                                piece = delta.get("content") or ""
                                if piece:
                                    full_text.append(piece)
                        except Exception:
                            # 解析失败就忽略该行，继续积累
                            pass
                return "".join(full_text).strip()

            else:
                # —— 非流式：一次性 JSON —— #
                resp = requester.post(url, headers=headers, json=payload, timeout=req_timeout)
                # 优先尝试 JSON
                try:
                    js = resp.json()
                except requests.exceptions.JSONDecodeError:
                    txt = (resp.text or "").strip()
                    raise RuntimeError(f"LLM HTTP non-JSON (status={resp.status_code}): {txt[:500]}")

                if resp.status_code >= 400:
                    raise RuntimeError(f"LLM HTTP error (status={resp.status_code}): {str(js)[:500]}")

                choices = js.get("choices") or []
                if choices and isinstance(choices[0], dict):
                    msg = choices[0].get("message") or {}
                    return (msg.get("content") or "").strip()
                # 兜底
                return (js.get("data") or js.get("text") or str(js)).strip()

        except requests.exceptions.RequestException as e:
            body = getattr(e.response, "text", "") if hasattr(e, "response") and e.response is not None else str(e)
            raise RuntimeError(f"LLM HTTP error: {body[:500]}")

    
    
    def classify(self, text: str) -> Dict[str, Any]:
        """
        让模型输出：
        {
          "intent": "<4~10个中文动词短语>",
          "confidence": {
            "<skill_name>": float, ... , "none": float
          }
        }
        - 不再让模型直接给 skill；由我们在代码端做 argmax 选择 skill。
        - 会返回规范化后的 best_skill 与 best_score，外加 _debug。
        """
        candidates: List[str] = settings.SKILL_CANDIDATES
        # 将候选及其中文说明注入，帮助模型“对号入座”
        desc = settings.SKILL_DESCRIPTIONS

        sys_prompt = (
            "你是一个严格的JSON分类器。"
            "只输出一个JSON对象，不要任何多余文字或解释。\n\n"
            "字段说明：\n"
            "- intent：用4~10个中文动词短语，概括“用户到底想让你帮他做什么”，不要复述原文。\n"
            "- confidence：一个对象，对下列候选项逐一给出置信度，所有值相加必须等于1。\n"
            "候选项与含义如下：\n"
        )
        for k in candidates:
            cn = desc.get(k, "")
            sys_prompt += f"- {k}：{cn}\n"
        sys_prompt += (
            "\n注意：\n"
            "1) 只输出JSON，不要加任何文本。\n"
            "2) confidence 里的键必须与给定候选项完全一致（区分大小写），每个都有值。\n"
            "3) 所有置信度是0~1之间的小数，总和=1。\n"
        )

        system = Message(role="system", content=sys_prompt)
        user = Message(
            role="user",
            content=f"输入文本：{text}\n请仅按上述schema输出JSON。"
        )

        raw = self.complete([system, user], max_tokens=220)
        # raw = self.session.post(self._chat_url, 
        #                         headers=headers, 
        #                         json=payload,
        #                         timeout=(settings.CONNECT_TIMEOUT, min(settings.READ_TIMEOUT, 15)))

        # 解析：从 raw 中抽取 JSON
        parsed, candidate_json = {}, "{}"
        try:
            m = re.search(r"\{.*\}", raw, re.S)
            candidate_json = m.group(0) if m else "{}"
            parsed = json.loads(candidate_json)
        except Exception:
            parsed = {}

        # 取 intent
        intent = parsed.get("intent", "")

        # 取置信度字典，并做健壮性处理
        conf_map = parsed.get("confidence", {})
        if not isinstance(conf_map, dict):
            conf_map = {}

        # 确保所有候选项都有值（缺失补0）
        norm_map: Dict[str, float] = {}
        for k in candidates:
            try:
                v = float(conf_map.get(k, 0.0))
            except Exception:
                v = 0.0
            # 纠偏（NaN/负值）
            if v != v or v < 0:
                v = 0.0
            norm_map[k] = v

        # 归一化（有时模型和为0或不等于1）
        s = sum(norm_map.values())
        if s > 0:
            norm_map = {k: v / s for k, v in norm_map.items()}
        else:
            # 都是0的话，让 none=1 兜底
            norm_map = {k: (1.0 if k == "none" else 0.0) for k in candidates}

        # 选 argmax 作为 best
        best_skill = max(norm_map.items(), key=lambda kv: kv[1])[0]
        best_score = norm_map[best_skill]

        result = {
            "intent": intent,
            "skill": best_skill,
            "confidence": best_score,      # 最高项的分数，供阈值判断
            "confidence_map": norm_map     # 完整分布（便于debug和可解释）
        }
        if settings.DEBUG:
            result["_debug"] = {"raw": raw, "candidate": candidate_json, "parsed": parsed}
        return result
    

    



    
    

