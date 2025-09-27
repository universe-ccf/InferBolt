# main.py
from __future__ import annotations
import os, io, uuid
import gradio as gr
from clients.llm_client import LLMClient
from core.pipeline import respond
from core.state import SessionState, reset_session
from core.types import RoleConfig
from core.roles import load_all_roles
import json
import numpy as np
from core.pipeline import respond, respond_voice
from config import settings


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

# 语音处理
def on_user_submit_audio(audio_tuple, session: SessionState, role_name: str, llm: LLMClient,
                         debug_on: bool, use_custom_voice: bool, custom_voice: str, custom_speed: float):
    try:
        if audio_tuple is None:
            return [(None, "请先录音或上传音频。")], "—", "—", None, session

        sr, audio_np = audio_tuple
        if audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32)
            maxv = max(1.0, float((abs(audio_np).max() or 1.0)))
            audio_np = audio_np / maxv

        role = load_role_config(role_name)
        ov = custom_voice if (use_custom_voice and custom_voice) else None
        ospeed = custom_speed if use_custom_voice else None

        turn = respond_voice(audio_np=audio_np, sample_rate=sr, state=session,
                             role=role, llm_client=llm, override_voice=ov, override_speed=ospeed)

        chat_pair = [("🎤(语音)", turn.reply_text)]
        label = SKILL_LABELS.get(turn.skill) if turn.skill else None
        skill_tag = f"🧠 已触发：`{label}`" if label else "—"

        debug_md = "—"
        if debug_on:
            rd = turn.data.get("route_debug")
            if rd:
                import json
                debug_md = "### 路由调试\n```json\n" + json.dumps(rd, ensure_ascii=False, indent=2) + "\n```"

        audio_path = turn.audio_bytes  # 现在承载的是文件路径
        return chat_pair, skill_tag, debug_md, audio_path, session

    except Exception:
        import traceback
        tb = traceback.format_exc()
        debug_md = "### 语音异常\n```\n" + tb[-800:] + "\n```"
        return [("🎤(语音)", "抱歉，语音处理异常。")], "—", debug_md, None, session


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

        gr.Markdown("### AI也会犯错，请检查重要信息！")

        # 技能状态指示组件
        skill_info = gr.Markdown(value="—", label="技能状态")

        debug_panel = gr.Markdown(value="—", label="调试信息")

        with gr.Tab("文本对话"):
            with gr.Row():
                txt_in = gr.Textbox(label="输入你的话", 
                                    placeholder="例：强化论证 / 交叉质询 / 反事实挑战", 
                                    lines=2)
                send_btn = gr.Button("发送", variant="primary")

        with gr.Tab("语音对话"):
            with gr.Row():
                mic = gr.Audio(sources=["microphone", "upload"], type="numpy", label="录音或上传（单声道）")
                send_v = gr.Button("发送语音", variant="primary")
                audio_out = gr.Audio(label="语音回复（TTS）", type="filepath")

            # 高级设置（会话覆盖）
            with gr.Accordion("高级设置（会话覆盖角色音色）", open=False):
                use_custom_voice = gr.Checkbox(label="使用自定义音色", value=False)
                custom_voice = gr.Textbox(label="voice_type（留空则用角色默认）", placeholder="例如：qiniu_zh_male_cxkjns")
                custom_speed = gr.Slider(0.6, 1.3, value=0.92, step=0.02, label="speed_ratio（0.6~1.3）")


        # 文本事件
        send_btn.click(
            fn=on_user_submit_text,
            inputs=[txt_in, session_state, role_dd, llm_client, debug_ck],
            outputs=[chatbot, skill_info, debug_panel, session_state]
        ).then(  # 发送后清空输入框
            lambda: "", None, txt_in
        )

        # 语音事件
        send_v.click(
            fn=on_user_submit_audio,
            inputs=[mic, session_state, role_dd, llm_client, debug_ck, use_custom_voice, custom_voice, custom_speed],
            outputs=[chatbot, skill_info, debug_panel, audio_out, session_state]
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
            lambda: None, None, audio_out
        ).then(
            lambda: "", None, txt_in
        )

    demo.launch()

if __name__ == "__main__":
    build_ui()
