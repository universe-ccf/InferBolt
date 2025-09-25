# skills/counterfactual.py
from __future__ import annotations
from typing import List
from core.types import SkillResult, RoleConfig, Message

def _style_hint(role: RoleConfig) -> str:
    parts = []
    if role.style: parts.append(f"保持{role.style}风格")
    if role.mission: parts.append(f"遵循使命：{role.mission}")
    return "；".join(parts)

def run(user_text: str, role: RoleConfig, history: List[Message], llm_client) -> SkillResult:
    sys = (
        "你是“反事实挑战”教练。用中文，围绕用户的结论，找出两个关键假设，并分别给出“若相反会怎样”的推演。\n"
        "格式严格：\n"
        "关键假设A：...\n"
        "若相反，则可能：...\n"
        "关键假设B：...\n"
        "若相反，则可能：...\n"
        "最后输出『结论变化小结：』1-2句。要求：具体、可检验。"
    )
    sys = sys + " " + _style_hint(role)
    msgs = [
        Message(role="system", content=sys),
        Message(role="user", content=f"待挑战的结论/方案：{user_text}")
    ]
    reply = llm_client.complete(msgs, max_tokens=420)
    return SkillResult(
        name="counterfactual",
        display_tag="反事实挑战",
        reply_text=reply,
        data={}
    )
