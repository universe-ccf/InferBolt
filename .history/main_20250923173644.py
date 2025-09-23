# app/main.py
from __future__ import annotations
import gradio as gr
from app.core.state import SessionState, reset_session
from app.core.pipeline import respond
from app.core.types import RoleConfig
from app.clients.llm_client import LLMClient
from app.clients.asr_client import ASRClient
from app.clients.tts_client import TTSClient
from app.config import settings

def load_role_config(name: str) -> RoleConfig:
    """从 config/roles/*.json 读取并反序列化为 RoleConfig"""
    ...

def on_reset(session: SessionState) -> SessionState:
    """UI回调：一键重置会话"""
    ...

def on_user_submit_text(user_text: str, session: SessionState, role_name: str, llm: LLMClient, tts: TTSClient):
    """文本输入流程：pipeline.respond → TTS → 返回文本与音频"""
    ...

def on_user_submit_audio(audio_bytes: bytes, session: SessionState, role_name: str, llm: LLMClient, asr: ASRClient, tts: TTSClient):
    """语音输入流程：ASR → pipeline.respond → TTS → 返回文本与音频"""
    ...

def build_ui():
    """组装 Gradio 组件（聊天区、角色选择、录音、播放、技能标识），绑定回调"""
    ...

if __name__ == "__main__":
    build_ui()
