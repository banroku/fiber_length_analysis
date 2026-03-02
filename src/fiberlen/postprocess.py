# Path: src/fiberlen/postprocess.py

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from fiberlen.types import Fiber

def postprocess(
    fiber: List[Fiber],
    eliminate_length: float,
    um_per_px: float,
) -> List[Fiber]:

    #thr_px = float(eliminate_length)
    thr_px = float(eliminate_length / um_per_px)

    filtered = [f for f in fiber if float(getattr(f, "length_px", 0.0)) >= thr_px]
    filtered.sort(key=lambda x: float(getattr(x, "length_px", 0.0)), reverse=True)

    return filtered
