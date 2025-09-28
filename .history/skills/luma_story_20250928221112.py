from __future__ import annotations
from typing import List
from core.types import Message, RoleConfig, SkillResult

SYS = (
    "你是 Luma：用一个**短故事**回应用户的主题/情绪，"
    "用隐喻帮助换角度看问题；故事收尾给一条温柔的启发。"
    "避免说教；不做医学/心理诊断。"
)

def _build_messages(role: RoleConfig, history: List[Message], user_text: str) -> List[Message]:
    msgs: List[Message] = [Message(role="system", content=SYS)]
    msgs.extend(history or [])
    msgs.append(Message(role="user", content=f"主题/情绪：{user_text}\n请写一个 120~220 字的小故事，最后用1-2句点题。"))
    return msgs

def run(user_text: str, role: RoleConfig, history: List[Message], llm_client) -> SkillResult:
    reply = llm_client.complete(_build_messages(role, history, user_text), max_tokens=280, stream=False)
    return SkillResult(
        name="luma_story",
        display_tag="故事生成",
        reply_text=reply.strip()
    )
