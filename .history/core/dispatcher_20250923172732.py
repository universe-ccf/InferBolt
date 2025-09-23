# app/core/dispatcher.py
from __future__ import annotations
from typing import Optional, Dict, Any
from .types import SkillCall, RoleConfig

def route(user_text: str, role: RoleConfig, context_hint: Dict[str, Any] | None = None) -> Optional[SkillCall]:
    """
    两级策略：
    1) 规则层（关键词/模式/正则）快速判断是否触发某技能；
    2) LLM 分类器兜底（输出 {intent, skill, confidence}），低置信返回None。
    返回 SkillCall 或 None（None表示走普通对话）。
    """
    ...
