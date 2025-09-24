# app/core/pipeline.py
from __future__ import annotations
from typing import List, Dict, Any
from .types import Message, RoleConfig, TurnResult, SkillResult
from .state import SessionState, get_recent_messages
from .dispatcher import route
from config import settings


# 根据角色配置生成system prompt（口吻、禁区、格式偏好）
def build_system_prompt(role: RoleConfig) -> str:
    parts = []
    parts.append(f"你现在扮演：{role.name}。风格：{role.style}。")
    if role.persona:
        parts.append("人设要点：" + "；".join(role.persona))
    if role.taboos:
        parts.append("避免输出：" + "；".join(role.taboos))
    if role.format_prefs:
        if role.format_prefs.get("bullets", False):
            parts.append("如可，采用分点表达。")
        if role.format_prefs.get("max_words"):
            parts.append(f"尽量不超过 {role.format_prefs['max_words']} 字。")
    parts.append("请使用中文回答。")
    return " ".join(parts)

# 把system + 历史 + 当前user 拼成LLM可用的messages
def assemble_messages(system_prompt: str, history: List[Message], user_text: str) -> List[Message]:
    msgs: List[Message] = [Message(role="system", content=system_prompt)]
    # 只保留最近 N 轮（由 get_recent_messages 控制）
    msgs.extend(history)
    msgs.append(Message(role="user", content=user_text))
    return msgs

def run_skill(skill_name: str, user_text: str, role: RoleConfig, history: List[Message]) -> SkillResult:
    """按名字调用对应技能模块，并返回统一的SkillResult"""
    ...

def respond(user_text: str, state: SessionState, role: RoleConfig, llm_client, max_rounds: int = 8) -> TurnResult:
    """
    主流程：
    1) dispatcher.route() → 命中Skill？ → run_skill()
    2) 否则 → 走普通对话：build_system_prompt + assemble_messages + llm_client.complete()
    3) 统一构造 TurnResult（先不做TTS）
    """
    ...
