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
