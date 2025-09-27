# utils/cache.py
from __future__ import annotations
import os, json, hashlib
from typing import Optional, Dict, Any

def _ensure_dir(d: str):
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def cache_get_text(dirpath: str, key: str) -> Optional[str]:
    _ensure_dir(dirpath)
    ftxt = os.path.join(dirpath, key + ".txt")
    if os.path.exists(ftxt):
        with open(ftxt, "r", encoding="utf-8") as f:
            return f.read()
    return None

def cache_put_text(dirpath: str, key: str, text: str) -> str:
    _ensure_dir(dirpath)
    ftxt = os.path.join(dirpath, key + ".txt")
    with open(ftxt, "w", encoding="utf-8") as f:
        f.write(text)
    return ftxt

def cache_get_file(dirpath: str, key: str, ext: str) -> Optional[str]:
    _ensure_dir(dirpath)
    fpath = os.path.join(dirpath, f"{key}.{ext}")
    return fpath if os.path.exists(fpath) else None

def cache_put_file(dirpath: str, key: str, ext: str, data: bytes) -> str:
    _ensure_dir(dirpath)
    fpath = os.path.join(dirpath, f"{key}.{ext}")
    with open(fpath, "wb") as f:
        f.write(data)
    return fpath
