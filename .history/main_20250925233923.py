# app/main.py
from __future__ import annotations
import os, io, uuid
import gradio as gr

# # =========================
# # 1) 导入真实模块；注释中没实现时使用“临时假实现”
# # =========================

# from core.state import SessionState, reset_session
# from core.pipeline import respond
# from core.types import RoleConfig
# from clients.llm_client import LLMClient
# from clients.asr_client import ASRClient
# from clients.tts_client import TTSClient

# try:
    # from core.state import SessionState, reset_session
    # from core.pipeline import respond
    # from core.types import RoleConfig
    # from clients.llm_client import LLMClient
    # from clients.asr_client import ASRClient
    # from clients.tts_client import TTSClient
# except Exception:
#     # ---- 临时数据结构（让页面先跑起来）----
#     class SessionState:
#         def __init__(self, session_id: str):
#             self.session_id = session_id
#             self.messages = []
#             self.last_skill = None

#     def reset_session(state: SessionState) -> SessionState:
#         state.messages = []
#         state.last_skill = None
#         return state

#     class RoleConfig:
#         def __init__(self, name, style="", persona=None, catchphrases=None, taboos=None, format_prefs=None):
#             self.name = name
#             self.style = style
#             self.persona = persona or []
#             self.catchphrases = catchphrases or []
#             self.taboos = taboos or []
#             self.format_prefs = format_prefs or {}

#     # ---- 临时客户端：不调用任何外部API，仅做回显 ----
#     class LLMClient:
#         def __init__(self, model="dummy", temperature=0.7, api_key=None, base_url=None): ...

#         def complete(self, messages, max_tokens=256) -> str:
#             # 取最后一条 user 的内容，做个“角色风格”回显
#             user_text = ""
#             for m in reversed(messages):
#                 if m["role"] == "user":
#                     user_text = m["content"]
#                     break
#             return f"（{messages[0]['content'][:10]}…风格）我听到了：{user_text}"

#     class ASRClient:
#         def __init__(self, model="dummy", api_key=None, base_url=None): ...
#         def transcribe(self, audio_bytes: bytes, language: str | None = None) -> str:
#             return "[ASR假实现]（此处应为你的音频识别文本）"

#     class TTSClient:
#         def __init__(self, voice="dummy", api_key=None, base_url=None): ...
#         def synthesize(self, text: str, speed: float = 1.0, emotion: str | None = None) -> bytes:
#             # 假实现：不生成真实音频，返回空字节流
#             return b""

#     # ---- 临时 respond：仅拼装最小 system+user，调用 LLMClient.complete ----
#     def respond(user_text: str, state: SessionState, role: RoleConfig, llm_client, max_rounds: int = 8):
#         system_prompt = f"{role.name}：保持{role.style or '角色口吻'}，避免{('、'.join(role.taboos) or '违禁内容')}"
#         # 最小 messages 结构（与我们规划的差不多）
#         messages = [
#             {"role": "system", "content": system_prompt},
#         ]
#         # 追加最近轮次（临时：直接扁平）
#         for m in state.messages[-2*max_rounds:]:
#             messages.append({"role": m["role"], "content": m["content"]})
#         messages.append({"role": "user", "content": user_text})

#         reply = llm_client.complete(messages, max_tokens=256)

#         # 更新state（临时存法）
#         state.messages.append({"role": "user", "content": user_text})
#         state.messages.append({"role": "assistant", "content": reply})

#         # 返回“与我们规划一致的 TurnResult 字段”
#         return {
#             "reply_text": reply,
#             "skill": None,
#             "data": {},
#             "audio_bytes": None
#         }

# # =========================
# # 2) 读取角色配置（MVP：用内置字典，后续替换为读取 config/roles/*.json）
# # =========================
# BUILTIN_ROLES = {
#     "Socratic Mentor": RoleConfig(
#         name="Socratic Mentor",
#         style="古希腊哲学式、少给结论、以提问引导",
#         taboos=["医疗/法律诊断", "仇恨与歧视"],
#         persona=["坚持追问", "用类比引导思考"],
#         catchphrases=["让我们更深入地想一想"]
#     ),
#     "Wizard Mentor": RoleConfig(
#         name="Wizard Mentor",
#         style="奇幻导师风格，善用类比与隐喻",
#         taboos=["现实政治站队"],
#         persona=[""],
#         catchphrases=[""]
#     ),
#     "Coach Mentor": RoleConfig(
#         name="Coach Mentor",
#         style="温和、结构化、给出具体建议",
#         taboos=["现实医疗处方与诊断", "法律建议"],
#         persona=[""],
#         catchphrases=[""]
#     )
# }

# def load_role_config(name: str) -> RoleConfig:
#     # 之后可改为：从 config/roles/{name}.json 读取
#     return BUILTIN_ROLES.get(name, BUILTIN_ROLES["Socratic Mentor"])

# # def load_role_config(name: str) -> RoleConfig:
# #     return BUILTIN_ROLES.get(name, list(BUILTIN_ROLES.values())[0])

# # =========================
# # 3) UI 回调
# # =========================
# def on_reset(session: SessionState):
#     return reset_session(session)

# # def on_user_submit_text(user_text: str, session: SessionState, role_name: str, llm: LLMClient, tts: TTSClient):
# #     role = load_role_config(role_name)
# #     turn = respond(user_text=user_text, state=session, role=role, llm_client=llm)
# #     # 合成音频（此处为假实现，返回空bytes；未来替换为真正TTS）
# #     audio_bytes = tts.synthesize(turn["reply_text"])
# #     turn["audio_bytes"] = audio_bytes

# #     # 返回给前端：聊天追加、TTS音频、会话状态（gr.State需要返回自身以更新）
# #     chat_pair = [(user_text, turn["reply_text"])]
# #     return chat_pair, audio_bytes, session

# def on_user_submit_text(user_text: str,
#                         session: SessionState,
#                         role_name: str,
#                         llm: LLMClient):
#     role = load_role_config(role_name)
#     turn = respond(user_text=user_text, state=session, role=role, llm_client=llm)
#     # 把当前这一轮追加到 Chatbot 展示（chatbot 需要 [(user, assistant)] 的列表增量）
#     chat_pair = [(user_text, turn.reply_text)]
#     return chat_pair, session

# def on_user_submit_audio(audio_filepath: str, session: SessionState, role_name: str,
#                          llm: LLMClient, asr: ASRClient, tts: TTSClient):
#     # 读取音频文件为bytes
#     with open(audio_filepath, "rb") as f:
#         audio_bytes = f.read()
#     # ASR
#     user_text = asr.transcribe(audio_bytes=audio_bytes)

#     # 调对话主流程
#     role = load_role_config(role_name)
#     turn = respond(user_text=user_text, state=session, role=role, llm_client=llm)

#     # TTS
#     audio_bytes_out = tts.synthesize(turn["reply_text"])
#     turn["audio_bytes"] = audio_bytes_out

#     chat_pair = [(f"[语音] {user_text}", turn["reply_text"])]
#     return chat_pair, audio_bytes_out, session

# # =========================
# # 4) 组装 UI
# # =========================
# def build_ui():
#     with gr.Blocks(title="AI 角色扮演 · MVP") as demo:
#         gr.Markdown("## AI 角色扮演 · 语音聊天（MVP）\n先跑通闭环，再逐步加技能与评估。")

#         # 全局状态：会话、客户端（MVP中作为全局单例）
#         session_state = gr.State(SessionState(session_id=str(uuid.uuid4())))
#         llm_client = gr.State(LLMClient())
#         asr_client = gr.State(ASRClient(model="dummy"))
#         tts_client = gr.State(TTSClient(voice="dummy"))

#         with gr.Row():
#             role_dd = gr.Dropdown(choices=list(BUILTIN_ROLES.keys()), value="Socratic Mentor", label="选择角色")
#             reset_btn = gr.Button("重置会话", variant="secondary")

#         chatbot = gr.Chatbot(label="对话区", height=350)

#         with gr.Tab("文本对话"):
#             with gr.Row():
#                 txt_in = gr.Textbox(label="输入你的话", placeholder="试试：请用苏格拉底式追问我学习目标", lines=2)
#                 send_btn = gr.Button("发送", variant="primary")
#         with gr.Tab("语音对话"):
#             with gr.Row():
#                 mic_in = gr.Audio(
#                     sources=["microphone"],
#                     type="filepath",  # 直接拿文件路径，后端再读bytes
#                     label="按下录音，放开上传"
#                 )
#                 speak_btn = gr.Button("说话并发送", variant="primary")

#         tts_audio = gr.Audio(label="AI 回复（音频）", autoplay=False)

#         # 绑定事件（Gradio多返回值需要对应组件列表顺序）
#         send_btn.click(
#             fn=on_user_submit_text,
#             inputs=[txt_in, session_state, role_dd, llm_client, tts_client],
#             outputs=[chatbot, tts_audio, session_state]
#         )
#         # .then(  # 发送后清空输入框
#         #     lambda: "", None, txt_in
#         # )

#         speak_btn.click(
#             fn=on_user_submit_audio,
#             inputs=[mic_in, session_state, role_dd, llm_client, asr_client, tts_client],
#             outputs=[chatbot, tts_audio, session_state]
#         )

#         reset_btn.click(
#             fn=on_reset,
#             inputs=[session_state],
#             outputs=[session_state]
#         ).then(  # 重置后清空UI显示
#             lambda: None, None, chatbot
#         ).then(
#             lambda: None, None, tts_audio
#         )

#     demo.launch()

# if __name__ == "__main__":
#     build_ui()





# === 引入你已写好的真实实现 ===
from clients.llm_client import LLMClient
from core.pipeline import respond
from core.state import SessionState, reset_session
from core.types import RoleConfig
from core.roles import load_all_roles
import json


SKILL_LABELS = {
    "steelman": "强化论证",
    "x_exam": "交叉质询",
    "counterfactual": "反事实挑战",
}


# === 加载角色配置===
ROLES_CACHE = load_all_roles()

def load_role_config(name: str) -> RoleConfig:
    return ROLES_CACHE.get(name, list(ROLES_CACHE.values())[0])

# === 回调：文本输入 ===
def on_user_submit_text(user_text: str,
                        session: SessionState,
                        role_name: str,
                        llm: LLMClient,
                        debug_on: bool):
    try:
        role = load_role_config(role_name)
        turn = respond(user_text=user_text, state=session, role=role, llm_client=llm)
        chat_pair = [(user_text, turn.reply_text)]
        label = SKILL_LABELS.get(turn.skill) if turn.skill else None
        skill_tag = f"🧠 已触发：`{label}`" if label else "—"

        debug_md = "—"
        if debug_on:
            rd = turn.data.get("route_debug")
            if rd:
                # 如果 classify 返回了分布，也显示
                cls = rd.get("classify")
                if isinstance(cls, dict) and "confidence_map" in cls:
                    rd_pretty = {
                        "rule_hit": rd.get("rule_hit"),
                        "rule_name": rd.get("rule_name"),
                        "best_skill": cls.get("skill"),
                        "best_confidence": cls.get("confidence"),
                        "confidence_map": cls.get("confidence_map"),
                        "_raw_len": len(str(cls.get("_debug", {}).get("raw", "")))
                    }
                    debug_md = "### 路由调试\n```json\n" + json.dumps(rd_pretty, ensure_ascii=False, indent=2) + "\n```"
                else:
                    debug_md = "### 路由调试\n```json\n" + json.dumps(rd, ensure_ascii=False, indent=2) + "\n```"

        return chat_pair, skill_tag, debug_md, session
    except Exception:
        import traceback; traceback.print_exc()
        return [(user_text, "抱歉，内部出现错误，正在修复。")], "—", "—", session


# === 回调：重置会话 ===
def on_reset(session: SessionState):
    reset_session(session)
    return session

# === 组装 UI ===
def build_ui():
    with gr.Blocks(title="AI 角色扮演 · 思辨训练营(MVP)") as demo:
        gr.Markdown("## AI 角色扮演（思辨训练营）")

        # 全局状态：会话 + LLM 客户端（持久化在 Gradio 的 State 里）
        session_state = gr.State(SessionState(session_id=str(uuid.uuid4())))
        llm_client = gr.State(LLMClient())   # 使用 .env/settings.py 配好的 API/模型

        with gr.Row():
            role_dd = gr.Dropdown(choices=list(ROLES_CACHE.keys()),
                      value=list(ROLES_CACHE.keys())[0],
                      label="选择角色")
            debug_ck = gr.Checkbox(label="调试模式", value=True)
            reset_btn = gr.Button("重置会话", variant="secondary")

        chatbot = gr.Chatbot(label="对话区", height=350)

        # 技能状态指示组件
        skill_info = gr.Markdown(value="—", label="技能状态")

        debug_panel = gr.Markdown(value="—", label="调试信息")

        with gr.Row():
            txt_in = gr.Textbox(label="输入你的话", 
                                placeholder="例：请帮我强化论证 / 做交叉质询 / 做反事实挑战", 
                                lines=2)
            send_btn = gr.Button("发送", variant="primary")

        # 事件绑定
        send_btn.click(
            fn=on_user_submit_text,
            inputs=[txt_in, session_state, role_dd, llm_client, debug_ck],
            outputs=[chatbot, skill_info, debug_panel, session_state]
        ).then(  # 发送后清空输入框
            lambda: "", None, txt_in
        )

        reset_btn.click(
            fn=on_reset,
            inputs=[session_state],
            outputs=[session_state]
        ).then(
            lambda: None, None, chatbot
        ).then(
            lambda: "—", None, skill_info  # 重置技能指示
        ).then(
            lambda: "—", None, debug_panel
        ).then(
            lambda: "", None, txt_in
        )

    demo.launch()

if __name__ == "__main__":
    build_ui()
