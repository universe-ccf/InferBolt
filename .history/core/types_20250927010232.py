# core/types.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal

RoleLiteral = Literal["system", "user", "assistant"]

@dataclass
class Message:
    role: RoleLiteral
    content: str
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SkillResult:
    name: str                      # "socratic" | "summary" | "quiz" ...
    display_tag: str               # UI上显示的技能标识，例如 "苏格拉底追问"
    reply_text: str                # 返回给用户（也会送TTS）
    data: Dict[str, Any] = field(default_factory=dict)  # 结构化补充（题目、步骤等）

@dataclass
class TurnResult:
    reply_text: str
    skill: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    audio_bytes: Optional[bytes] = None

@dataclass
class SkillCall:
    name: str                      # 技能名
    args: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RoleConfig:
    name: str
    style: str
    persona: List[str] = field(default_factory=list)
    catchphrases: List[str] = field(default_factory=list)
    taboos: List[str] = field(default_factory=list)
    format_prefs: Dict[str, Any] = field(default_factory=dict)
    mission: str = ""   # ← 新增：角色使命/场景主基调（思辨训练营）
    tts: Dict[str, Any] = field(default_factory=dict)
