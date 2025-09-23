# app/skills/socratic.py
from __future__ import annotations
from typing import Dict, Any, List
from app.core.types import SkillResult, RoleConfig, Message

def run(user_text: str, role: RoleConfig, history: List[Message]) -> SkillResult:
    """
    产出2-3个递进式问题，少给结论。
    输出固定字段：reply_text（含编号/层次），data可附带问题列表。
    """
    ...




