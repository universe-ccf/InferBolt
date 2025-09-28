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
from skills import luma_story, luma_reframe, luma_roleplay
from skills import aris_reverse, aris_practice, aris_bimap
from utils.logging import write_log
import time
from clients.asr_client import ASRClient
from clients.tts_client import TTSClient
from clients.asr_ws_client import ASRWsClient
from utils.textseg import split_for_tts
import os


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
    
    # Luma
    if skill_name == "luma_story":
        return luma_story.run(user_text, role, history, llm_client)
    if skill_name == "luma_reframe":
        return luma_reframe.run(user_text, role, history, llm_client)
    if skill_name == "luma_roleplay":
        return luma_roleplay.run(user_text, role, history, llm_client)

    # Aris
    if skill_name == "aris_reverse":
        return aris_reverse.run(user_text, role, history, llm_client)
    if skill_name == "aris_practice":
        return aris_practice.run(user_text, role, history, llm_client)
    if skill_name == "aris_bimap":
        return aris_bimap.run(user_text, role, history, llm_client)

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
        reply_text = llm_client.complete(messages, max_tokens=settings.MAX_TOKENS_RESPONSE, stream=settings.TEXT_STREAMING)
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


# 语音模式下的短回复
def respond_short(user_text: str, state: SessionState, role: RoleConfig, llm_client) -> TurnResult:
    """
    语音模式下的“短回复”：限制为 1-2 句/不超过 MAX_REPLY_CHARS_VOICE。
    复用你的 build_system_prompt / assemble_messages，只是多加一段约束。
    """
    sys_prompt = build_system_prompt(role)
    limit_note = f"【重要】请用1-2句中文回答，总字数不超过{settings.MAX_REPLY_CHARS_VOICE}字。如需展开，请最后问：要继续吗？"
    sys_prompt = sys_prompt + "\n" + limit_note

    history = getattr(state, "history", None) or getattr(state, "turns", None) or getattr(state, "messages", None) or []
    msgs = assemble_messages(sys_prompt, history, user_text)

    # 也可在 user 侧再加一句“请简洁回答”
    reply = llm_client.complete(msgs, max_tokens=256, stream=False)
    
    # 更新会话
    # 改为显式 push 到 messages：
    try:
        if not hasattr(state, "messages") or state.messages is None:
            state.messages = []
        state.messages.append(Message(role="user", content=user_text))
        state.messages.append(Message(role="assistant", content=reply))
    except Exception:
        # 容错：不因日志失败影响主流程
        pass
    return TurnResult(reply_text=reply, skill=None, data={})

# 句级：一句识别→一句短答→一句TTS→逐句产出
def voice_sentence_loop(audio_np, sample_rate, state: SessionState, role: RoleConfig, llm_client, asr_client, tts_client) -> Generator[Dict[str, Any], None, None]:
    """
    生成器：一次录音 -> ASR -> 按句切 -> 对每句做“短回复+TTS”，逐句 yield 到 UI。
    yield 字段：chatbot_messages（列表）、session_state、audio_path（每句一个文件）、status_text
    """
    t0 = time.time()
    # 一开始（收到音频）先提示
    yield {"status": "🧠 正在识别(ASR)...", "chat_add": []}

    # 1) ASR（整段识别）
    asr_t0 = time.time()
    asr_res = asr_client.transcribe(audio_np, sample_rate, audio_url=None)
    asr_t1 = time.time()
    user_text_all = (asr_res.text or "").strip()

    if not user_text_all:
        yield {
            "status": "❗未识别到有效语音，请重录或改用文本输入。",
            "audio_path": None,
            "user_text": "",
            "chat_add": [("user", "（空语音）"), ("assistant", "没听清哦，可以再试一次吗？")]
        }
        return
    
    # ASR 完成后（有 user_text）
    yield {"status": "🤖 正在思考(LLM)...", "chat_add": [("user", user_text_all)]}

    # 2) 分句
    sentences = split_for_tts(user_text_all, max_chars=settings.MAX_REPLY_CHARS_VOICE)
    yield {"status": f"🎧 已识别：{user_text_all}（分{len(sentences)}句处理）", "audio_path": None, "user_text": user_text_all, "chat_add": []}

    # 3) 逐句：短回复 -> TTS -> 逐句输出
    for idx, sent in enumerate(sentences, 1):
        step_t0 = time.time()
        # 3.1 分类（小模型兜底；你已有 classify，可选接入）
        # cls = llm_client.classify(sent, schema=...)
        # 先省略，直接短回复
        # 3.2 短回复
        turn = respond_short(user_text=sent, state=state, role=role, llm_client=llm_client)
        # LLM 得到 reply 后，马上提示
        yield {"status": "🔊 正在合成(TTS)...", "chat_add": []}
        # 3.3 TTS（单句）
        tts_res = tts_client.synthesize(turn.reply_text,
                                        voice_type=(getattr(role, "tts", {}) or {}).get("voice_type"),
                                        speed_ratio=(getattr(role, "tts", {}) or {}).get("speed_ratio"))
        audio_path = tts_res.audio_path
        
        if audio_path:
            # 统一成正斜杠，Gradio/浏览器对 Windows 路径更友好
            audio_path = os.path.normpath(audio_path).replace("\\", "/")
        # 3.4 逐句推送
        yield {
            "status": f"🗣️ 第{idx}/{len(sentences)}句：{sent}",
            "audio_path": audio_path,   # gr.Audio 可直接播
            "user_text": sent,
            "chat_add": [("assistant", turn.reply_text)]
        }
        # 3.5 间隔（让前端有时间播放）- 可由前端控制，这里不sleep

    total = int((time.time() - t0) * 1000)
    write_log(settings.LOG_PATH, {
        "event": "voice_sentence_loop_done",
        "asr_ms": int((asr_t1-asr_t0)*1000),
        "total_ms": total,
        "n_sent": len(sentences)
    })