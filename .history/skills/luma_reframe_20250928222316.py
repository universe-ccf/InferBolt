# skills/luma_reframe.py
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
        "你是“正向重构”教练。先**共情**，再提供**3个可操作的新视角**，"
        "最后给**一个小行动**建议；避免评判、避免诊断；语言温柔、具体、可实行。"
        "输出格式严格：\n"
        "【我理解到的感觉】：(1句)\n"
        "【新的三个视角】：\n"
        "1) ...（聚焦可控点）\n"
        "2) ...（换解释框架）\n"
        "3) ...（增加证据/实验）\n"
        "【一个小行动】：(能在今天内尝试)"
    )
    sys = sys + " " + _style_hint(role)
    msgs = [
        Message(role="system", content=sys),
        Message(role="user", content=f"待重构的困扰/叙述：{user_text}")
    ]
    reply = llm_client.complete(msgs, max_tokens=420)
    return SkillResult(
        name="luma_reframe",
        display_tag="正向重构",
        reply_text=reply,
        data={}
    )
