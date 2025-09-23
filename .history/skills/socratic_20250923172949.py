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

# app/skills/summary.py
from __future__ import annotations
from typing import Dict, Any, List
from app.core.types import SkillResult, RoleConfig, Message

def run(user_text: str, role: RoleConfig, history: List[Message]) -> SkillResult:
    """输出分层要点（例如 1/1.1/1.2），data附结构化层级"""
    ...

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
