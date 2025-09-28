# main.py
from __future__ import annotations
import os, io, uuid
import gradio as gr
from clients.llm_client import LLMClient
from core.state import SessionState, reset_session
from core.types import RoleConfig
from core.roles import load_all_roles
import json
import numpy as np
from core.pipeline import respond, respond_voice
from core.pipeline import voice_sentence_loop, respond_short  
from clients.asr_ws_client import ASRWsClient                          
from clients.tts_client import TTSClient                                


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


# “生成器式”的语音回调
def on_user_submit_audio_stream(audio_tuple,
                                session: SessionState,
                                role_name: str,
                                llm: LLMClient,
                                debug_on: bool,
                                use_custom_voice: bool,
                                custom_voice: str,
                                custom_speed: float):
    """
    生成器：一次录音 => 句级快速反馈。
    每次 yield 更新：Chatbot(累积)、技能标签、调试面板、Audio(单句path)、Status、Session
    """

    # 兜底：没音频
    if audio_tuple is None:
        # [chatbot, audio_out, status_badge, skill_badge, session_state]
        yield [], None, "❗未接收音频", "—", session
        return

    # Gradio type="numpy" 形态：(sr, np.ndarray[float32, -1..1])
    try:
        sr, audio_np = audio_tuple
    except Exception:
        yield [], None, "❗音频格式异常", "—", session
        return

    if getattr(audio_np, "dtype", None) is not np.float32:
        audio_np = audio_np.astype(np.float32)
        maxv = float(np.max(np.abs(audio_np)) or 1.0)
        audio_np = audio_np / max(1.0, maxv)

    # 角色 + 会话级音色覆盖
    role = load_role_config(role_name)
    if use_custom_voice:
        tts_pref = getattr(role, "tts", {}) or {}
        if custom_voice:
            tts_pref["voice_type"] = custom_voice
        if custom_speed:
            tts_pref["speed_ratio"] = float(custom_speed)
        setattr(role, "tts", tts_pref)

    # 客户端
    asr = ASRWsClient()
    tts = TTSClient()

    # UI端累积对话
    ui_msgs = []

    # 逐句生成：ASR → 切句 → 短答 → TTS → yield
    gen = voice_sentence_loop(audio_np=audio_np,
                              sample_rate=sr,
                              state=session,
                              role=role,
                              llm_client=llm,
                              asr_client=asr,
                              tts_client=tts)

    for step in gen:
        # 把 step.chat_add 合入 Chatbot
        for who, txt in step.get("chat_add", []):
            if who == "user":
                ui_msgs.append((txt, None))
            else:
                ui_msgs.append((None, txt))


        # 逐次刷新：Chatbot、Audio(单句路径)、状态徽标文本、技能徽标文本、Session
        skill_txt = step.get("skill_label", "—")  # 若还没接 classify，可先返回 “—”
        yield ui_msgs, step.get("audio_path"), step.get("status", ""), skill_txt, session


def _toggle_drawer(vis: bool):
    return gr.update(visible=not vis)
    
def _load_voices():
    try:
        tts = TTSClient()
        items = tts.list_voices()
        labels, mapping = [], {}
        for it in items:
            vt = it.get("voice_type") or ""
            name = it.get("voice_name") or vt
            cat = it.get("category", "")
            label = f"{name} ({vt})" if name else vt
            if cat:
                label = f"{label} · {cat}"
            labels.append(label); mapping[label] = vt
        default_value = labels[0] if labels else None
        # 一次更新 choices 与默认值，避免“value 不在 choices 中”的报错
        return gr.update(choices=labels, value=default_value), mapping
    except Exception:
        return gr.update(choices=[], value=None), {}

def _label_to_voice(label: str, mapping: dict):
    return mapping.get(label or "", "")

def _on_reset(session: SessionState):
    reset_session(session)
    # 依次返回：chatbot 空列表、status 文案、skill 文案、audio 停止、session
    return [], "准备就绪", "—", None, session


CSS_PATH = os.path.join(os.path.dirname(__file__), "assets", "ui.css")
CUSTOM_CSS = open(CSS_PATH, "r", encoding="utf-8").read() if os.path.exists(CSS_PATH) else ""
THEME = gr.themes.Soft(primary_hue="blue", secondary_hue="cyan")  # 中性不压字

# === 组装 UI ===
def build_ui():
    with gr.Blocks(title="Voicery · 思辨训练营", theme=THEME, css=CUSTOM_CSS) as demo:
        # 顶部：左标题 + 右上“⚙️高级设置”
        with gr.Row(elem_id="header_bar"):
            with gr.Column(scale=5):
                gr.Markdown("<div id='header_title'>🪄 Voicery </div>"
                            "<div id='header_sub'>角色扮演 · 句级快速反馈 · 科技感 UI</div>")
            with gr.Column(scale=1):
                settings_btn = gr.Button("⚙️ 高级设置", variant="secondary")

        # 全局状态：会话 + LLM 客户端（持久化在 Gradio 的 State 里）
        session_state = gr.State(SessionState(session_id=str(uuid.uuid4())))
        llm_client = gr.State(LLMClient())   # 使用 .env/settings.py 配好的 API/模型
        drawer_visible = gr.State(False)

        with gr.Row():
            role_dd = gr.Dropdown(choices=list(ROLES_CACHE.keys()),
                      value=list(ROLES_CACHE.keys())[0],
                      label="选择角色")
            reset_btn = gr.Button("重置会话", variant="secondary")

        voices_map = gr.State({})

        # 中间主体：左“聊天框（含角标）” + 右“抽屉”（默认隐藏）
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Group(elem_id="chatwrap"):
                    chatbot = gr.Chatbot(label=None, height=520, elem_id="chatbox")
                    status_badge = gr.Markdown("准备就绪", elem_classes=["badge"], elem_id="status_badge")
                    skill_badge  = gr.Markdown("—", elem_classes=["badge"], elem_id="skill_badge")
            with gr.Column(scale=2, visible=False, elem_id="drawer") as drawer:
                with gr.Group(elem_id="right_card"):
                    gr.Markdown("#### 面板")
                    # 调试开关 & 信息
                    debug_ck = gr.Checkbox(label="调试模式", value=False)
                    debug_panel = gr.Markdown(value="（调试输出显示在此）")

                with gr.Group(elem_id="right_card"):
                    gr.Markdown("#### 语音参数")
                    use_custom_voice = gr.Checkbox(label="启用自定义音色", value=True)
                    voice_label_dd   = voice_label_dd   = gr.Dropdown(
                        label="音色（从官方列表加载）",
                        choices=[], value=None, allow_custom_value=True
                    )
                    custom_voice     = gr.Textbox(label="voice_type（隐藏绑定）", visible=False)
                    custom_speed     = gr.Slider(0.7, 1.3, value=0.95, step=0.01, label="speed_ratio（0.7~1.3）")


        # 底部统一输入区
        with gr.Row(elem_id="input_row"):
            txt_in   = gr.Textbox(label=None, show_label=False,
                                  placeholder="输入文字，或点右侧 🎙️ 说话…", lines=3)
            mic      = gr.Audio(sources=["microphone"], type="numpy", label=None, show_label=False, visible=False)
            mic_btn  = gr.Button("🎙️ 语音", variant="secondary", elem_id="mic_btn")
            send_btn = gr.Button("发送", variant="primary")
            stop_btn = gr.Button("⏹ 停止播放", variant="secondary")

        # 隐形播放器：每句产出直接 autoplay
        audio_out = gr.Audio(label=None, type="filepath", autoplay=True, visible=False)

        gr.Markdown("<div style='opacity:.6'>⚠️ Voicery 可能出错, 请核验关键信息</div>")

        # 高级设置：开/合 + 拉取音色
        def _toggle_drawer_state(v: bool):
            return not v
        def _apply_drawer(v: bool):
            return gr.update(visible=v)

        settings_btn.click(
            _toggle_drawer_state, inputs=[drawer_visible], outputs=[drawer_visible]
        ).then(
            _apply_drawer, inputs=[drawer_visible], outputs=[drawer]
        ).then(
            _load_voices, inputs=None, outputs=[voice_label_dd, voices_map]
        )

        # 选择某音色 -> 写入隐藏的 custom_voice（调用链保持不变）
        voice_label_dd.change(_label_to_voice, inputs=[voice_label_dd, voices_map], outputs=[custom_voice])

        def _toggle_mic(vis: bool):
            return gr.update(visible=not vis)

        mic_btn.click(_toggle_mic, inputs=[mic], outputs=[mic])

        mic.change(
            fn=on_user_submit_audio_stream,
            inputs=[mic, session_state, role_dd, llm_client, debug_ck, use_custom_voice, custom_voice, custom_speed],
            outputs=[chatbot, audio_out, status_badge, skill_badge, session_state]   # ← 注意：输出目标变了
        )

        # 文本事件
        send_btn.click(
            fn=on_user_submit_text,
            inputs=[txt_in, session_state, role_dd, llm_client, debug_ck],
            outputs=[chatbot, skill_badge, debug_panel, session_state]   # 技能徽标=skill_badge
        ).then(lambda: "", None, txt_in)# 发送后清空输入框
        

        # 语音事件
        def _stop_play():
            return None, "⏹ 已停止播放"

        stop_btn.click(_stop_play, outputs=[audio_out, status_badge])

    demo.launch(show_api=False)   # “通过 API 使用”不显示；其它通过 CSS 已隐藏

if __name__ == "__main__":
    build_ui()
