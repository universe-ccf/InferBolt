# app/utils/textproc.py
from __future__ import annotations

def sanitize_user_text(text: str) -> str:
    """简单清洗：去控制字符、裁掉过长输入等"""
    ...

def truncate_messages_by_rounds(messages, max_rounds: int):
    """按轮数截断（MVP）；后续可换成按token截断"""
    ...
