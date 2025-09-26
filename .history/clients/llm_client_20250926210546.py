# clients/llm_client.py
from __future__ import annotations
import os, json, requests, re
from typing import List, Dict, Any
from core.types import Message
from config import settings
import requests
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

        # ↑ 带重试的 Session（连超时/读超时/502/503/504 自动重试）
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
            # 第一次正常请求
            resp = self.session.post(
                self._chat_url, headers=headers, json=payload, timeout=settings.REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and "choices" in data and data["choices"]:
                choice = data["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"]
                if "text" in choice:
                    return choice["text"]

            print("Unexpected LLM response:", resp.text[:500])
            return "（抱歉，模型响应解析失败。）"

        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout) as e:
            # 降级重试：收紧 max_tokens、温度、截断历史，以更快返回
            print("Timeout, retrying with a lighter request:", type(e).__name__)
            try:
                lite_payload = dict(payload)
                lite_payload["max_tokens"] = min(256, max_tokens)
                lite_payload["temperature"] = 0.3
                resp2 = self.session.post(
                    self._chat_url, headers=headers, json=lite_payload,
                    timeout=(settings.CONNECT_TIMEOUT, settings.READ_TIMEOUT + 10)  # 第二次读超时再放宽一点
                )
                resp2.raise_for_status()
                data = resp2.json()
                if isinstance(data, dict) and "choices" in data and data["choices"]:
                    choice = data["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        return choice["message"]["content"]
                    if "text" in choice:
                        return choice["text"]
                print("Unexpected LLM response (retry):", resp2.text[:500])
                return "（抱歉，模型响应解析失败。）"
            except Exception as e2:
                print("LLM timeout retry failed:", type(e2).__name__, str(e2)[:200])
                return "（抱歉，模型接口读取超时，请稍后再试。）"

        except requests.exceptions.RequestException as e:
            # 其他HTTP错误（含 429/5xx，经 Session 已自动重试）
            body = getattr(e.response, "text", "")
            print("LLM HTTP error:", body[:300] or str(e))
            return "（抱歉，模型接口暂时不可用，请稍后再试。）"

        except Exception as e:
            print("LLM parse error:", type(e).__name__, e)
            return "（抱歉，模型响应异常。）"
    
    # def complete(self, messages: List[Message], max_tokens: int = 512) -> str:
    #     payload = {
    #         "model": self.model,
    #         "messages": self._to_openai_messages(messages),
    #         "temperature": self.temperature,
    #         "max_tokens": max_tokens,
    #         "stream": False
    #     }
    #     headers = {
    #         "Authorization": f"Bearer {self.api_key}",
    #         "Content-Type": "application/json"
    #     }
    #     try:
    #         resp = requests.post(self._chat_url, headers=headers, json=payload, timeout=settings.REQUEST_TIMEOUT)
    #         resp.raise_for_status()
    #         data = resp.json()
    #         # —— 某些厂商兼容层有细微差异，这里更稳一点：
    #         if isinstance(data, dict) and "choices" in data and data["choices"]:
    #             choice = data["choices"][0]
    #             # OpenAI 兼容通常是 message.content；部分实现也可能是 delta/content 或 text
    #             if "message" in choice and "content" in choice["message"]:
    #                 return choice["message"]["content"]
    #             if "text" in choice:
    #                 return choice["text"]
    #         # 打印帮助排查
    #         print("Unexpected LLM response:", resp.text[:500])
    #         return "（抱歉，模型响应解析失败。）"
    #     except requests.exceptions.RequestException as e:
    #         print("LLM HTTP error:", getattr(e.response, "text", str(e)))
    #         return "（抱歉，模型接口暂时不可用，请稍后再试。）"
    #     except Exception as e:
    #         print("LLM parse error:", type(e).__name__, e)
    #         return "（抱歉，模型响应异常。）"


    # def classify(self, text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    #     """
    #     让模型只输出JSON；解析失败时兜底为 unknown。
    #     schema 例：
    #     {"intent":"", "skill":"steelman|x_exam|counterfactual|none", "confidence":0.0}
    #     """
    #     """
    #     返回结构：
    #     {
    #     "intent": str,
    #     "skill": Optional[str],
    #     "confidence": float,
    #     "_debug": {
    #         "raw": str,             # LLM 原样输出
    #         "parsed": dict,         # 解析后的JSON（若成功）
    #         "candidate": str        # 从 raw 提取的大括号片段
    #     }
    #     }
    #     """  

    #     system = Message(role="system", content=(
    #         "你是一个严格的JSON分类器。"
    #         "只输出一个JSON对象，不要任何多余文字或解释。"
    #         "字段: intent(中文短语), skill(steelman|x_exam|counterfactual|none), confidence(0~1浮点)。"
    #     ))
    #     user = Message(
    #         role="user",
    #         content=f"请根据输入文本判断用户意图，并按schema输出JSON。输入：{text}\n"
    #                 f"schema={json.dumps(schema, ensure_ascii=False)}"
    #     )
    #     raw = self.complete([system, user], max_tokens=200)

    #     parsed = {}
    #     candidate = "{}"
    #     # 从返回文本中提取JSON（防脏输出）
    #     try:
    #         # 优先用贪婪匹配最后一个大括号块
    #         m = re.search(r"\{.*\}", raw, re.S)
    #         candidate = m.group(0) if m else "{}"
    #         parsed = json.loads(candidate)
    #     except Exception:
    #         parsed = {}
        
    #     intent = parsed.get("intent", "unknown")
    #     skill = parsed.get("skill", None)

    #     try:
    #         conf = float(parsed.get("confidence", 0.0))
    #     except Exception:
    #         conf = 0.0

    #     # 兜底规范化
    #     if skill not in {"steelman", "x_exam", "counterfactual"}:
    #         skill = None

    #     result = {"intent": intent, "skill": skill, "confidence": conf}
    #     if settings.DEBUG:
    #         result["_debug"] = {"raw": raw, "parsed": parsed, "candidate": candidate}
    #     return result
    
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
    

    



    
    

