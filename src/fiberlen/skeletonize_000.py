# Path: src/fiberlen/skeletonize.py

from __future__ import annotations
import numpy as np


def skeletonize(img_for_skeletonized: np.ndarray) -> np.ndarray:
    """
    Zhang-Suen thinning によるスケルトナイズ

    入力
    ----
    img_for_skeletonized : bool ndarray
        True が前景（繊維）

    出力
    ----
    img_skeletonized : bool ndarray

    Notes
    -----
    ・8近傍前提
    ・外部ライブラリ不使用
    ・トポロジ保存型細線化
    """

    img = np.asarray(img_for_skeletonized, dtype=np.uint8)
    changed = True

    # 近傍順序 (P2〜P9)
    def neighbors(x):
        return [
            x[:-2,1:-1],   # P2
            x[:-2,2:],     # P3
            x[1:-1,2:],    # P4
            x[2:,2:],      # P5
            x[2:,1:-1],    # P6
            x[2:,:-2],     # P7
            x[1:-1,:-2],   # P8
            x[:-2,:-2]     # P9
        ]

    while changed:
        changed = False

        # --- step 1 ---
        x = img.copy()
        nbs = neighbors(x)

        B = sum(nbs)
        A = sum((nbs[i] == 0) & (nbs[(i+1)%8] == 1) for i in range(8))

        m1 = (x[1:-1,1:-1] == 1)
        m2 = (B >= 2) & (B <= 6)
        m3 = (A == 1)
        m4 = (nbs[0] * nbs[2] * nbs[4] == 0)
        m5 = (nbs[2] * nbs[4] * nbs[6] == 0)

        remove = m1 & m2 & m3 & m4 & m5
        img[1:-1,1:-1][remove] = 0

        if np.any(remove):
            changed = True

        # --- step 2 ---
        x = img.copy()
        nbs = neighbors(x)

        B = sum(nbs)
        A = sum((nbs[i] == 0) & (nbs[(i+1)%8] == 1) for i in range(8))

        m4 = (nbs[0] * nbs[2] * nbs[6] == 0)
        m5 = (nbs[0] * nbs[4] * nbs[6] == 0)

        remove = m1 & m2 & m3 & m4 & m5
        img[1:-1,1:-1][remove] = 0

        if np.any(remove):
            changed = True

    return img.astype(bool)
