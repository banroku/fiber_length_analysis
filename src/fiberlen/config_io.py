# fiberlen/config_io.py
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

T = TypeVar("T")

def save_cfg_json(cfg: T, path: str | Path) -> None:
    if not is_dataclass(cfg):
        raise TypeError("cfg must be a dataclass instance")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    d = asdict(cfg)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

def load_cfg_json(cfg_type: Type[T], path: str | Path) -> T:
    p = Path(path)
    d = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(d, dict):
        raise ValueError("config json must be an object/dict")
    return cfg_type(**d)  # キーが一致する前提
