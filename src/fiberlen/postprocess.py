# Path: src/fiberlen/postprocess.py

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from fiberlen.types import Fiber

def postprocess(
    fiber: List[Fiber],
    eliminate_length: float,
) -> List[Fiber]:

    thr = float(eliminate_length)

    filtered = [f for f in fiber if float(getattr(f, "length_px", 0.0)) >= thr]
    filtered.sort(key=lambda x: float(getattr(x, "length_px", 0.0)), reverse=True)

    return filtered
