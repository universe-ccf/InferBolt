# skills/x_exam.py
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
        "你是“交叉质询”教练。用中文，对给定命题进行三轮交叉质询，格式严格：\n"
        "第1轮：问题 → 可能暴露的薄弱点（1句）\n"
        "第2轮：问题 → 可能暴露的薄弱点（1句）\n"
        "第3轮：问题 → 可能暴露的薄弱点（1句）\n"
        "最后输出：『总结弱点：』列出2点。\n"
        "要求：问题要具体，不要泛问；每句简洁。"
    )
    sys = sys + " " + _style_hint(role)
    msgs = [
        Message(role="system", content=sys),
        Message(role="user", content=f"请针对该命题进行交叉质询：{user_text}")
    ]
    reply = llm_client.complete(msgs, max_tokens=420)
    return SkillResult(
        name="x_exam",
        display_tag="交叉质询",
        reply_text=reply,
        data={}
    )
