# core/dispatcher.py
from __future__ import annotations
from typing import Optional, Dict, Any
from .types import SkillCall, RoleConfig
from config import settings


_RULES = [
    # 思辨训练营：三技能的关键词触发
    # 强化论证（Steelman）
    ({"强化", "打磨", "完善", "更强", "steelman", "最佳表述", "最强论证", }, "steelman"),
    #  "扎实", "更扎实", "说服力", "立场", "论证"  
    
    # 交叉质询（Cross-Examination）
    ({"质询", "交叉", "挑错", "找漏洞", "反驳我", "问难", "质问"}, "x_exam"),

    # 反事实挑战（Counterfactual）
    ({"反事实", "如果不", "若相反", "假如相反", "假设变化", "换个前提"}, "counterfactual"),

    # Luma
    # 故事生成
    ({"故事", "讲个故事", "写个故事", "隐喻"}, "luma_story"),
    # 正向重构
    ({"开导", "正向重构", "换角度", "重构", "情绪"}, "luma_reframe"),
    # 陪伴角色扮演
    ({"扮演", "陪伴", "像我的朋友", "像我女朋友", "像我男朋友", "以XX身份说话"}, "luma_roleplay"),

    # Aris
    # 逆向挑战
    ({"逆向挑战", "你来解", "现场解析", "解题", "我出题"}, "aris_reverse"),
    # 互动练习
    ({"互动练习", "给我练习", "小测", "做题训练"}, "aris_practice"),
    # 双向映射
    ({"双向映射", "数学与编程", "代码与数学", "类比讲解"}, "aris_bimap"),
]



def route(user_text: str, role: RoleConfig, llm_client=None,
          context_hint: Dict[str, Any] | None = None) -> Optional[SkillCall]:
    text = user_text.strip().lower()
    debug: Dict[str, Any] = {"phase": "route", 
                             "rule_hit": False, 
                             "rule_name": None, 
                             "classify": None}

    # 1) 规则优先（可解释）
    for keywords, skill_name in _RULES:
        if any(kw in text for kw in keywords):
            debug["rule_hit"] = True
            debug["rule_name"] = skill_name
            return SkillCall(name=skill_name, args={"debug": debug})

    # 2) 兜底分类：拿分布，代码端 argmax
    if llm_client is not None:
        res = llm_client.classify(user_text)
        debug["classify"] = res
        best_skill = res.get("skill")
        best_score = float(res.get("confidence", 0.0))
        # 阈值判断 & 过滤 none
        if best_skill and best_skill != "none" and best_score >= settings.INTENT_CONF_THRESHOLD:
            return SkillCall(name=best_skill, args={"conf": best_score, "intent": res.get("intent"), "debug": debug})

    # 3) 未命中：带着 debug 信息返回占位
    return SkillCall(name="__none__", args={"debug": debug})
