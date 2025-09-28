# skills/luma_roleplay.py
from __future__ import annotations
from typing import List
from core.types import SkillResult, RoleConfig, Message

def _style_hint(role: RoleConfig) -> str:
    tips = []
    if role.style: tips.append(f"保持{role.style}风格")
    if role.mission: tips.append(f"遵循使命：{role.mission}")
    if getattr(role, "catchphrases", None):
        tips.append(f"可酌情加入其口头禅：{role.catchphrases[0]}")
    return "；".join(tips)

def run(user_text: str, role: RoleConfig, history: List[Message], llm_client) -> SkillResult:
    sys = (
        "你现在**扮演用户指定的亲友/伴侣/重要他人**的口吻与其对话。"
        "要求：\n"
        "1) 口吻贴合，但不得做现实承诺、不得PUA、不过度控制；\n"
        "2) 表达理解与支持，给**1个轻柔可行的建议**（可选）；\n"
        "3) 字数控制在**80~180字**；\n"
        "4) 不输出医疗/心理诊断与药物建议。"
    )
    sys = sys + " " + _style_hint(role)
    msgs = [
        Message(role="system", content=sys),
        Message(role="user", content=f"扮演请求及情境：{user_text}\n请以该角色口吻回应当前轮次。")
    ]
    reply = llm_client.complete(msgs, max_tokens=260)
    return SkillResult(
        name="luma_roleplay",
        display_tag="陪伴扮演",
        reply_text=reply,
        data={}
    )
