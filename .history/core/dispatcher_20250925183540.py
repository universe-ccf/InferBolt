# app/core/dispatcher.py
from __future__ import annotations
from typing import Optional, Dict, Any
from .types import SkillCall, RoleConfig
from config import settings


# 思辨训练营：三技能的关键词触发
_RULES = [
    # 强化论证（Steelman）
    ({"强化", "打磨", "完善", "更强", "steelman", "最佳表述", "最强论证"}, "steelman"),

    # 交叉质询（Cross-Examination）
    ({"质询", "交叉", "挑错", "找漏洞", "反驳我", "问难", "质问"}, "x_exam"),

    # 反事实挑战（Counterfactual）
    ({"反事实", "如果不", "若相反", "假如相反", "假设变化", "换个前提"}, "counterfactual"),
]

_SCHEMA = {"intent": "", "skill": "steelman|x_exam|counterfactual|none", "confidence": 0.0}


def route(user_text: str, role: RoleConfig, llm_client=None,
          context_hint: Dict[str, Any] | None = None) -> Optional[SkillCall]:
    text = user_text.strip().lower()

    # 1) 规则优先（可解释、零成本）
    for keywords, skill_name in _RULES:
        if any(kw in text for kw in keywords):
            return SkillCall(name=skill_name, args={})

    # 2) 兜底分类（需传入 llm_client 才启用）
    if llm_client is not None:
        res = llm_client.classify(user_text, schema=_SCHEMA)
        if res.get("skill") and res.get("confidence", 0.0) >= settings.INTENT_CONF_THRESHOLD:
            return SkillCall(name=res["skill"], args={"conf": res["confidence"], "intent": res.get("intent")})

    # 3) 未命中
    return None