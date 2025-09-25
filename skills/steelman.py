# skills/steelman.py
from __future__ import annotations
from typing import List
from core.types import SkillResult, RoleConfig, Message

def _style_hint(role: RoleConfig) -> str:
    hints = []
    if role.style: hints.append(f"保持{role.style}风格")
    if role.mission: hints.append(f"遵循使命：{role.mission}")
    if role.catchphrases: hints.append(f"可酌情用其口头禅开场：{role.catchphrases[0]}")
    return "；".join(hints)

def run(user_text: str, role: RoleConfig, history: List[Message], llm_client) -> SkillResult:
    sys = (
        "你是“强化论证（Steelman）”教练。用中文，帮用户把观点强化到“最强版本”。"
        "输出严格三段：\n"
        "【立场（最强表述）】：\n"
        "【关键论据（3-4条）】：\n"
        "【潜在反驳与预案（2-3条）】：\n"
        "要求：精准、克制、可执行。"
    )
    sys = sys + " " + _style_hint(role)
    msgs = [
        Message(role="system", content=sys),
        Message(role="user", content=f"待强化的观点/命题：{user_text}")
    ]
    reply = llm_client.complete(msgs, max_tokens=450)
    return SkillResult(
        name="steelman",
        display_tag="强化论证（Steelman）",
        reply_text=reply,
        data={}
    )



