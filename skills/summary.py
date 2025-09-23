# app/skills/summary.py
from __future__ import annotations
from typing import Dict, Any, List
from app.core.types import SkillResult, RoleConfig, Message

def run(user_text: str, role: RoleConfig, history: List[Message]) -> SkillResult:
    """输出分层要点（例如 1/1.1/1.2），data附结构化层级"""
    ...