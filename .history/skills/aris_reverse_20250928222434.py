# skills/aris_reverse.py
from __future__ import annotations
from typing import List
from core.types import SkillResult, RoleConfig, Message

def _style_hint(role: RoleConfig) -> str:
    tips = []
    if role.style: tips.append(f"保持{role.style}风格")
    if role.mission: tips.append(f"遵循使命：{role.mission}")
    return "；".join(tips)

def run(user_text: str, role: RoleConfig, history: List[Message], llm_client) -> SkillResult:
    sys = (
        "你是“逆向挑战”导师：用户出题，你现场解析。"
        "严格四步：\n"
        "1) **明确题意与条件**（必要时澄清假设）\n"
        "2) **思路分解**（列出关键台阶）\n"
        "3) **解法步骤**（数学与/或代码最小片段）\n"
        "4) **校验/极小例子**（验证正确性）\n"
        "要求：条理化、可验证；示例代码尽量最小且注释关键行。"
    )
    sys = sys + " " + _style_hint(role)
    msgs = [
        Message(role="system", content=sys),
        Message(role="user", content=f"题目/任务：{user_text}\n按“四步法”给出解析，能用双视角更好。")
    ]
    reply = llm_client.complete(msgs, max_tokens=460)
    return SkillResult(
        name="aris_reverse",
        display_tag="逆向挑战",
        reply_text=reply,
        data={}
    )
