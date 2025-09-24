# app/core/state.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
from .types import Message

@dataclass
class SessionState:
    session_id: str
    messages: List[Message] = field(default_factory=list)  # 只存最近N轮（user/assistant）
    last_skill: Optional[str] = None

def append_turn(state: SessionState, user_msg: Message, assistant_msg: Message, max_rounds: int = 8) -> SessionState:
    state.messages.append(user_msg)
    state.messages.append(assistant_msg)
    # 只保留最近 n 轮（user+assistant 为一轮，故 *2）
    keep = 2 * max_rounds
    if len(state.messages) > keep:
        state.messages = state.messages[-keep:]
    return state

def get_recent_messages(state: SessionState, n_rounds: int = 8) -> List[Message]:
    """返回最近n轮user/assistant交替消息，用于拼Prompt"""
    ...

def reset_session(state: SessionState) -> SessionState:
    """一键重置会话（不改session_id），清空短期记忆"""
    ...
