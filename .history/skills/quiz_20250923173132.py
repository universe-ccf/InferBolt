# app/skills/quiz.py
from __future__ import annotations
from typing import Dict, Any, List
from app.core.types import SkillResult, RoleConfig, Message

def run(user_text: str, role: RoleConfig, history: List[Message]) -> SkillResult:
    """
    单轮闭环测验：出一道题→用户答题→本轮判分与讲解。
    MVP可先实现“出题模式”，再加“判答模式”。
    """
    ...