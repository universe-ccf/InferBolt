# core/roles.py
from __future__ import annotations
import os, json, glob
from typing import Dict, List
from core.types import RoleConfig

_ROLES_DIR = os.path.join(os.path.dirname(__file__), "..", "config", "roles")

def list_role_files() -> List[str]:
    return sorted(glob.glob(os.path.join(_ROLES_DIR, "*.json")))

def load_role_from_file(path: str) -> RoleConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 字段缺省有默认值，直接解包
    return RoleConfig(**data)

def load_all_roles() -> Dict[str, RoleConfig]:
    out: Dict[str, RoleConfig] = {}
    for fp in list_role_files():
        role = load_role_from_file(fp)
        out[role.name] = role
    if not out:
        # 兜底：若目录为空，给一个默认角色
        out["Default"] = RoleConfig(name="Default", style="中性、克制", mission="对话助手")
    return out
