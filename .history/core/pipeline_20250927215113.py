# core/pipeline.py
from __future__ import annotations
from typing import Generator, List, Dict, Optional, Any
from .types import Message, RoleConfig, TurnResult, SkillResult
from .state import SessionState, get_recent_messages, append_turn
from .dispatcher import route
from config import settings
from skills import steelman as skill_steelman
from skills import x_exam as skill_x_exam
from skills import counterfactual as skill_cf
from utils.logging import write_log
import time
from clients.asr_client import ASRClient
from clients.tts_client import TTSClient
from clients.asr_ws_client import ASRWsClient
from utils.textseg import split_for_tts


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
    skill_call = route(user_text=user_text, role=role, llm_client=llm_client, context_hint=None)
    
    # 未命中：可能是 "__none__"
    if skill_call and skill_call.name == "__none__":
        route_debug = skill_call.args.get("debug", {})
        # 普通对话
        system_prompt = build_system_prompt(role)
        history = get_recent_messages(state, max_rounds=max_rounds)
        messages = assemble_messages(system_prompt, history, user_text)
        reply_text = llm_client.complete(messages, max_tokens=settings.MAX_TOKENS_RESPONSE, stream=True)
        append_turn(state, Message(role="user", content=user_text), Message(role="assistant", content=reply_text), max_rounds)

        if settings.DEBUG:
            write_log(settings.LOG_PATH, {
                "event": "chat_turn",
                "path": "llm_default",
                "user_text": user_text,
                "route_debug": route_debug,
                "reply_len": len(reply_text)
            })
        return TurnResult(reply_text=reply_text, skill=None, data={"route_debug": route_debug}, audio_bytes=None)

    # 命中技能（规则或分类）
    if skill_call and skill_call.name:
        history = get_recent_messages(state, max_rounds=max_rounds)
        sres = run_skill(skill_call.name, user_text, role, history, llm_client)
        append_turn(state, Message(role="user", content=user_text), Message(role="assistant", content=sres.reply_text), max_rounds)

        if settings.DEBUG:
            write_log(settings.LOG_PATH, {
                "event": "chat_turn",
                "path": "skill",
                "skill": sres.name,
                "user_text": user_text,
                "route_debug": skill_call.args.get("debug"),
                "reply_len": len(sres.reply_text)
            })
        return TurnResult(reply_text=sres.reply_text, 
                          skill=sres.name,
                          data={"display_tag": sres.display_tag, 
                                "route_debug": skill_call.args.get("debug")},
                          audio_bytes=None)

    # 理论不会到这
    return TurnResult(reply_text="（抱歉，路由异常。）", skill=None, data={}, audio_bytes=None)


def respond_voice(audio_np, sample_rate, state: SessionState, role: RoleConfig, llm_client,
                  override_voice: Optional[str]=None, override_speed: Optional[float]=None) -> TurnResult:
    t0 = time.time()

    asr = ASRWsClient() if settings.ASR_TRANSPORT == "ws" else ASRClient()
    tts = TTSClient()

    # 1) ASR
    t_asr0 = time.time()
    asr_res = asr.transcribe(audio_np, sample_rate, audio_url=None)  # 如你有URL可传入
    t_asr1 = time.time()

    user_text = asr_res.text

    # 埋点测试
    if not user_text:
        write_log(settings.LOG_PATH, {"event": "voice_asr_empty", "asr_meta": asr_res.meta})

    # --- ASR失败守护：不再把错误文案当作用户输入推进后续链路 ---
    if not user_text.strip() or user_text.strip().startswith("（ASR请求失败"):
        write_log(settings.LOG_PATH, {"event":"voice_asr_failed_shortcircuit", 
                                      "asr_meta": asr_res.meta})
        return TurnResult(
            reply_text="语音识别未成功，请重录或改用文本输入。\n\n详情：ASR需要正确的音频/网络，请稍后重试。",
            skill=None,
            data={"route_debug":{"phase":"asr_failed","meta":asr_res.meta}},
            audio_bytes=None
        )

    # 2) LLM（复用文本流程）
    t_llm0 = time.time()
    turn_text = respond(user_text=user_text, state=state, role=role, llm_client=llm_client)
    t_llm1 = time.time()

    # 3) 角色TTS偏好（角色覆盖 > 会话覆盖 > 全局）
    tts_prefs = getattr(role, "tts", {}) or {}
    voice = override_voice or tts_prefs.get("voice_type") or settings.TTS_VOICE
    speed = override_speed if (override_speed is not None) else tts_prefs.get("speed_ratio", settings.TTS_SPEED)

    t_tts0 = time.time()
    tts_res = tts.synthesize(turn_text.reply_text, voice_type=voice, speed_ratio=speed)

    # 埋点测试
    if not getattr(tts_res, "audio_path", None):
        write_log(settings.LOG_PATH, {"event": "voice_tts_empty_path", "tts_meta": tts_res.meta})

    t_tts1 = time.time()

    total = time.time() - t0
    if settings.DEBUG:
        write_log(settings.LOG_PATH, {
            "event": "voice_turn",
            "asr_ms": int((t_asr1 - t_asr0)*1000),
            "llm_ms": int((t_llm1 - t_llm0)*1000),
            "tts_ms": int((t_tts1 - t_tts0)*1000),
            "total_ms": int(total*1000),
            "asr_meta": asr_res.meta,
            "tts_meta": tts_res.meta,
            "user_text": user_text,
            "skill": turn_text.skill,
        })

    # 注意：这里返回 audio_bytes 改为 audio_path
    return TurnResult(reply_text=turn_text.reply_text,
                      skill=turn_text.skill,
                      data={"route_debug": turn_text.data.get("route_debug"),
                            "voice_used": voice, "speed_used": speed},
                      audio_bytes=tts_res.audio_path)  # 用此字段承载路径