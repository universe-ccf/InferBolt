# utils/logging.py
from __future__ import annotations
import time, json, os
from typing import Any, Dict

def timeit_ms() -> tuple[callable, callable]:
    """返回(start, stop)两个闭包，计算毫秒耗时"""
    ...

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def write_log(path: str, record: Dict[str, Any]) -> None:
    ensure_dir(path)
    record = {"ts_ms": int(time.time() * 1000), **record}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
