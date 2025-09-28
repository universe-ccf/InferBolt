# utils/textseg.py
def split_for_tts(text: str, max_chars: int = 120, seps: str = "。！？!?；;，,"):
    """
    用于“句级快速反馈”的简易分句：
    - 句号/叹号/问号优先切分（达到一半长度即可切）
    - 逗号次之（达到目标长度切）
    - 超过 1.2 * max_chars 强制切
    """
    text = (text or "").strip()
    if not text:
        return []
    parts, buf = [], ""
    for ch in text:
        buf += ch
        if ch in "。！？!?；;" and len(buf) >= max_chars * 0.5:
            parts.append(buf.strip()); buf = ""
        elif ch in "，," and len(buf) >= max_chars:
            parts.append(buf.strip()); buf = ""
        elif len(buf) >= max_chars * 1.2:
            parts.append(buf.strip()); buf = ""
    if buf.strip():
        parts.append(buf.strip())
    return parts
