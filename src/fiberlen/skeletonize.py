# src/fiberlen/skeletonize.py
from __future__ import annotations

import numpy as np


def skeletonize(img_for_skeletonized: np.ndarray) -> np.ndarray:
    """
    Skeletonize (8-neighborhood assumption in pipeline).
    Input: bool ndarray (True=foreground)
    Output: bool ndarray
    """
    img = np.asarray(img_for_skeletonized, dtype=bool)

    # Fast path: scikit-image (Cython implementation)
    try:
        from skimage.morphology import skeletonize as _sk_skeletonize
        return _sk_skeletonize(img).astype(bool)
    except Exception:
        # Fallback: keep your pure-numpy Zhang-Suen (with a correctness fix)
        return _zhang_suen_numpy(img)


def _zhang_suen_numpy(img_bool: np.ndarray) -> np.ndarray:
    img = img_bool.astype(np.uint8, copy=True)
    changed = True

    def neighbors(x: np.ndarray):
        return [
            x[:-2, 1:-1],   # P2
            x[:-2, 2:],     # P3
            x[1:-1, 2:],    # P4
            x[2:, 2:],      # P5
            x[2:, 1:-1],    # P6
            x[2:, :-2],     # P7
            x[1:-1, :-2],   # P8
            x[:-2, :-2],    # P9
        ]

    while changed:
        changed = False

        # --- step 1 ---
        x = img  # view
        nbs = neighbors(x)

        B = nbs[0].copy()
        for k in range(1, 8):
            B += nbs[k]
        A = np.zeros_like(B, dtype=np.uint8)
        for i in range(8):
            A += ((nbs[i] == 0) & (nbs[(i + 1) % 8] == 1)).astype(np.uint8)

        c = x[1:-1, 1:-1]
        m1 = (c == 1)
        m2 = (B >= 2) & (B <= 6)
        m3 = (A == 1)
        m4 = (nbs[0] * nbs[2] * nbs[4] == 0)
        m5 = (nbs[2] * nbs[4] * nbs[6] == 0)

        remove = m1 & m2 & m3 & m4 & m5
        if np.any(remove):
            img[1:-1, 1:-1][remove] = 0
            changed = True

        # --- step 2 ---
        x = img
        nbs = neighbors(x)

        B = nbs[0].copy()
        for k in range(1, 8):
            B += nbs[k]
        A = np.zeros_like(B, dtype=np.uint8)
        for i in range(8):
            A += ((nbs[i] == 0) & (nbs[(i + 1) % 8] == 1)).astype(np.uint8)

        c = x[1:-1, 1:-1]
        m1 = (c == 1)  # FIX: recompute for step2
        m2 = (B >= 2) & (B <= 6)
        m3 = (A == 1)
        m4 = (nbs[0] * nbs[2] * nbs[6] == 0)
        m5 = (nbs[0] * nbs[4] * nbs[6] == 0)

        remove = m1 & m2 & m3 & m4 & m5
        if np.any(remove):
            img[1:-1, 1:-1][remove] = 0
            changed = True

    return img.astype(bool)
