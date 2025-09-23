# app/utils/logging.py
from __future__ import annotations
import time, json, os
from typing import Any, Dict

def timeit_ms() -> tuple[callable, callable]:
    """返回(start, stop)两个闭包，计算毫秒耗时"""
    ...

def write_log(path: str, record: Dict[str, Any]) -> None:
    """JSONL 追加一行日志"""
    ...
