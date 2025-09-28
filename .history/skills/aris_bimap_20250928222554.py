# skills/aris_bimap.py
from __future__ import annotations
from typing import List
from core.types import SkillResult, RoleConfig, Message

def _style_hint(role: RoleConfig) -> str:
    tips = []
    if role.style: tips.append(f"保持{role.style}风格")
    if role.mission: tips.append(f"遵循使命：{role.mission}")
    return "；".join(tips)

def run(user_text: str, role: RoleConfig, history: List[Message], llm_client) -> SkillResult:
    sys = (
        "你是“数学↔编程 双向映射”老师：用**两种视角**解释同一概念。"
        "输出结构严格：\n"
        "【概念定义】：(1-2句)\n"
        "【数学表述】：(含符号/小推理)\n"
        "【代码示例】：(最小可运行片段+关键注释)\n"
        "【如何互证】：(数学如何约束代码，代码如何验证数学)\n"
        "【延伸阅读】：(2-3个关键点或方向)\n"
        "总字数 200~360；禁止冗长。"
    )
    sys = sys + " " + _style_hint(role)
    msgs = [
        Message(role="system", content=sys),
        Message(role="user", content=f"要讲解的概念/主题：{user_text}\n按给定结构输出。")
    ]
    reply = llm_client.complete(msgs, max_tokens=420)
    return SkillResult(
        name="aris_bimap",
        display_tag="双向映射",
        reply_text=reply,
        data={}
    )
