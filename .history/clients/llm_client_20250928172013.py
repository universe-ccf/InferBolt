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

    def _ensure_openai_messages(self, messages):
        """把 List[Message] 或 List[dict] 统一转为 [{'role':'user','content':'...'}]"""
        out = []
        for m in messages:
            if isinstance(m, dict):
                role = m.get("role"); content = m.get("content")
            else:
                # dataclass Message
                role = getattr(m, "role", None)
                content = getattr(m, "content", None)
            if not role or content is None:
                # 跳过异常项，或你也可 raise
                continue
            out.append({"role": str(role), "content": str(content)})
        return out

    def _to_openai_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        # 我们内部 Message 的字段名与 OpenAI 兼容，可直接映射
        out = []
        for m in messages:
            out.append({"role": m.role, "content": m.content})
        return out

    # 通用对话补全
    def complete(self, messages: List[Dict[str, str]], max_tokens: int = 512, stream: bool = False) -> str:
        """
        OpenAI Chat Completions 兼容。
        - stream=False：普通 JSON；返回 reply 字符串
        - stream=True ：SSE（Server-Sent Events）；只拼接 delta.content；过滤控制字符避免乱码
        """
        messages = self._ensure_openai_messages(messages)
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": getattr(settings, "LLM_TEMPERATURE", 0.7),
            "max_tokens": max_tokens,
            "stream": bool(stream),
        }
        timeout = getattr(settings, "REQUEST_TIMEOUT", 30)

        if not stream:
            # 非流式
            try:
                resp = self.session.post(url, headers=headers, json=payload, timeout=timeout)
                resp.raise_for_status()
                js = resp.json()
                choice = (js.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                return (msg.get("content") or "").strip()
            except requests.exceptions.RequestException as e:
                body = getattr(e.response, "text", "") if hasattr(e, "response") else str(e)
                raise RuntimeError(f"LLM HTTP error: {body[:500]}")
            except ValueError:
                # 200 但不是 JSON，说明是 SSE 被误开（或服务端异常）
                txt = (resp.text or "").strip()
                raise RuntimeError(f"LLM HTTP non-JSON (status={resp.status_code}): {txt[:500]}")
        else:
            # 流式（SSE）
            try:
                resp = self.session.post(url, headers=headers, json=payload, timeout=timeout, stream=True)
                # 4xx/5xx 直接抛错
                if resp.status_code >= 400:
                    raise RuntimeError(f"LLM HTTP error (stream, status={resp.status_code}): {(resp.text or '')[:500]}")
                # 逐行解析 data: {...}
                def _clean_piece(s: str) -> str:
                    # 只保留可打印字符与换行，防止乱码（包含中英文）
                    return "".join(ch for ch in s if ch == "\n" or ch >= " ")
                full = []
                for raw in resp.iter_lines(decode_unicode=True):
                    if not raw:
                        continue
                    line = raw.strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        ch = (chunk.get("choices") or [{}])[0]
                        delta = ch.get("delta") or {}
                        piece = delta.get("content") or ""
                        if piece:
                            full.append(_clean_piece(piece))
                    except Exception:
                        # 跳过非 JSON 或包含推理字段的片段
                        continue
                return "".join(full).strip()
            except requests.exceptions.RequestException as e:
                body = getattr(e.response, "text", "") if hasattr(e, "response") else str(e)
                raise RuntimeError(f"LLM HTTP error (stream): {body[:500]}")
    
    def complete_chunks(self, messages, max_tokens=512):
        """
        逐片段产出文本（生成器）。调用方式：
        for piece in llm.complete_chunks(msgs): ...
        """
        messages = self._ensure_openai_messages(messages)
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": getattr(settings, "LLM_TEMPERATURE", 0.7),
            "max_tokens": max_tokens,
            "stream": True,
        }
        try:
            resp = self.session.post(url, headers=headers, json=payload,
                                    timeout=getattr(settings,"REQUEST_TIMEOUT",30),
                                    stream=True)
            resp.raise_for_status()
            def _clean(s):
                return "".join(ch for ch in s if ch == "\n" or ch >= " ")
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw: 
                    continue
                line = raw.strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                    piece = delta.get("content") or ""
                    if piece:
                        yield _clean(piece)
                except Exception:
                    continue
        except requests.exceptions.RequestException as e:
            body = getattr(e.response,"text","") if hasattr(e,"response") else str(e)
            raise RuntimeError(f"LLM HTTP error (stream): {body[:400]}")

    
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

        
        # 不走自定义 Message 了，彻底避免 JSON 序列化错误
        system = {"role": "system", "content": sys_prompt}
        user   = {"role": "user",   "content": f"输入文本：{text}\n请仅按上述schema输出JSON。"}

        raw = self.complete([system, user], max_tokens=220, stream=False)

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
