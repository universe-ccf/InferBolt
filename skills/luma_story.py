# skills/luma_story.py
from __future__ import annotations
from typing import List
from core.types import SkillResult, RoleConfig, Message

def _style_hint(role: RoleConfig) -> str:
    tips = []
    if role.style: tips.append(f"保持{role.style}风格")
    if role.mission: tips.append(f"遵循使命：{role.mission}")
    if getattr(role, "catchphrases", None):
        tips.append(f"可酌情用其口头禅：{role.catchphrases[0]}")
    return "；".join(tips)

def run(user_text: str, role: RoleConfig, history: List[Message], llm_client) -> SkillResult:
    sys = (
        "你是“情绪化解的故事讲述者”。用一个**120~220字**的短故事，"
        "以温柔的方式承接用户的情绪与主题；通过**隐喻**带出新视角；"
        "收尾用**1-2句启发**，轻柔不可说教；不用医学/心理诊断。"
    )
    sys = sys + " " + _style_hint(role)
    msgs = [
        Message(role="system", content=sys),
        Message(role="user", content=f"请基于下列【主题/情绪】写故事：{user_text}\n输出：短故事 + 结尾1-2句启发。")
    ]
    reply = llm_client.complete(msgs, max_tokens=320)
    return SkillResult(
        name="luma_story",
        display_tag="故事生成",
        reply_text=reply,
        data={}
    )
