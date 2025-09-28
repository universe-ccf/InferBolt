# core/state.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
from .types import Message

@dataclass
class SessionState:
    session_id: str
    messages: List[Message] = field(default_factory=list)  # 只存最近N轮（user/assistant）
    last_skill: Optional[str] = None

    @property
    def history(self):
        # v02/v03 中把每轮 turn 放在 self.turns 里，这里兼容返回
        return getattr(self, "turns", [])

def append_turn(state: SessionState, user_msg: Message, assistant_msg: Message, max_rounds: int = 8) -> SessionState:
    state.messages.append(user_msg)
    state.messages.append(assistant_msg)
    # 只保留最近 n 轮（user+assistant 为一轮，故 *2）
    keep = 2 * max_rounds
    if len(state.messages) > keep:
        state.messages = state.messages[-keep:]
    return state


def get_recent_messages(state, max_rounds: int):
    """
    兼容 v02/v03/v04：优先 messages，其次 turns，最终空列表。
    每轮对话约 2 条（user+assistant）。
    """
    msgs = getattr(state, "messages", None)
    if msgs is None:
        msgs = getattr(state, "turns", None)
    if msgs is None:
        return []
    try:
        return msgs[-2 * max_rounds :]
    except Exception:
        return list(msgs)[-2 * max_rounds :]

def reset_session(state: SessionState) -> SessionState:
    state.messages.clear()
    state.last_skill = None
    return state
