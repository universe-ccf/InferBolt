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

def route(user_text: str, role: RoleConfig, context_hint: Dict[str, Any] | None = None) -> Optional[SkillCall]:
    text = user_text.strip().lower()
    for keywords, skill_name in _RULES:
        if any(kw in text for kw in keywords):
            return SkillCall(name=skill_name, args={})
    # TODO：下一版可加 llm_client.classify() 做兜底
    return None