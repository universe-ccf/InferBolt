# core/pipeline.py
from __future__ import annotations
from typing import List, Dict, Any
from .types import Message, RoleConfig, TurnResult, SkillResult
from .state import SessionState, get_recent_messages, append_turn
from .dispatcher import route
from config import settings
from skills import steelman as skill_steelman
from skills import x_exam as skill_x_exam
from skills import counterfactual as skill_cf


# 根据角色配置生成system prompt（口吻、禁区、格式偏好）
def build_system_prompt(role: RoleConfig) -> str:
    parts = [f"你现在扮演：{role.name}。风格：{role.style}。"]
    if role.mission:
        parts.append(f"使命：{role.mission}。")
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


def run_skill(skill_name: str, user_text: str, role: RoleConfig, history: list[Message], llm_client) -> SkillResult:
    if skill_name == "steelman":
        return skill_steelman.run(user_text, role, history, llm_client)
    if skill_name == "x_exam":
        return skill_x_exam.run(user_text, role, history, llm_client)
    if skill_name == "counterfactual":
        return skill_cf.run(user_text, role, history, llm_client)
    # 未知技能：回退普通对话
    return SkillResult(name="none", display_tag="", reply_text=user_text, data={})

def respond(user_text: str, state: SessionState, role: RoleConfig, llm_client, max_rounds: int = None) -> TurnResult:
    max_rounds = max_rounds or settings.MAX_ROUNDS
    
    # 1) 技能优先
    skill_call = route(user_text=user_text, role=role, context_hint=None)
    if skill_call is not None:
        history = get_recent_messages(state, n_rounds=max_rounds)
        sres = run_skill(skill_call.name, user_text, role, history, llm_client)
        append_turn(state, Message(role="user", content=user_text), Message(role="assistant", content=sres.reply_text), max_rounds)
        return TurnResult(reply_text=sres.reply_text, skill=sres.name, data={"display_tag": sres.display_tag}, audio_bytes=None)

    # 2) 普通对话
    system_prompt = build_system_prompt(role)   # ← 会把 mission 带进去（见下一个小补丁）
    history = get_recent_messages(state, max_rounds=max_rounds)
    messages = assemble_messages(system_prompt, history, user_text)
    reply_text = llm_client.complete(messages, max_tokens=settings.MAX_TOKENS_RESPONSE)
    append_turn(state, Message(role="user", content=user_text), Message(role="assistant", content=reply_text), max_rounds)
    return TurnResult(reply_text=reply_text, skill=None, data={}, audio_bytes=None)
    
    
    

# def respond(user_text: str, state: SessionState, role: RoleConfig, llm_client, max_rounds: int = 8) -> TurnResult:
#     """
#     主流程：
#     1) dispatcher.route() → 命中Skill？ → run_skill()
#     2) 否则 → 走普通对话：build_system_prompt + assemble_messages + llm_client.complete()
#     3) 统一构造 TurnResult（先不做TTS）
#     """
#     ...