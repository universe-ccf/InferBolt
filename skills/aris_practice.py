# skills/aris_practice.py
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
        "你是“互动练习”导师：给用户一题**微型练习**并引导其作答。"
        "规则：\n"
        "• 题干简短明确（数学或编程）\n"
        "• 给出**三条递进提示**（先不直接给答案）\n"
        "• 等用户回答后，再在下一轮给详解\n"
        "语气鼓励，难度因材施教（基于用户描述）。"
    )
    sys = sys + " " + _style_hint(role)
    msgs = [
        Message(role="system", content=sys),
        Message(role="user", content=f"用户的主题/水平或目标：{user_text}\n请出1题 + 三条提示，暂不公布答案，等待用户作答。")
    ]
    reply = llm_client.complete(msgs, max_tokens=360)
    return SkillResult(
        name="aris_practice",
        display_tag="互动练习",
        reply_text=reply,
        data={}
    )
