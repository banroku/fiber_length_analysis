# Path: src/fiberlen/noize_elimination.py

from __future__ import annotations
import numpy as np
from collections import deque
from typing import List, Tuple, Set


Pixel = Tuple[int, int]


def noize_elimination(img_binarized: np.ndarray, eliminate_length: int) -> np.ndarray:
    """
    ノイズ除去（連結成分の外接矩形サイズで除去）

    仕様（あなたの文章どおり）
    ----------------------
    「縦、横の長さが eliminate_length より小さいノイズを除去」
    ここでの縦・横は連結成分の外接矩形の height/width を指す。
    両方が eliminate_length 未満の成分だけを消す。

    前提
    ----
    img_binarized は bool または {0,1} の 2値画像
    True が繊維（前景）

    連結性
    ------
    skeletonize/graphは8固定ですが、ここも8連結で実装します（分断を避ける）。

    Parameters
    ----------
    img_binarized : ndarray
    eliminate_length : int

    Returns
    -------
    img_for_skeletonized : bool ndarray
    """
    bw = np.asarray(img_binarized)
    if bw.dtype != np.bool_:
        bw = bw != 0

    elim = int(eliminate_length)
    if elim <= 0:
        return bw.copy()

    h, w = bw.shape
    visited = np.zeros((h, w), dtype=np.bool_)
    out = bw.copy()

    nbrs = [(-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)]

    for r0 in range(h):
        for c0 in range(w):
            if not bw[r0, c0] or visited[r0, c0]:
                continue

            # BFSで連結成分を取得
            q: deque[Pixel] = deque()
            q.append((r0, c0))
            visited[r0, c0] = True

            comp: List[Pixel] = []
            rmin = rmax = r0
            cmin = cmax = c0

            while q:
                r, c = q.popleft()
                comp.append((r, c))

                if r < rmin: rmin = r
                if r > rmax: rmax = r
                if c < cmin: cmin = c
                if c > cmax: cmax = c

                for dr, dc in nbrs:
                    rr = r + dr
                    cc = c + dc
                    if rr < 0 or rr >= h or cc < 0 or cc >= w:
                        continue
                    if visited[rr, cc]:
                        continue
                    if not bw[rr, cc]:
                        continue
                    visited[rr, cc] = True
                    q.append((rr, cc))

            height = (rmax - rmin + 1)
            width = (cmax - cmin + 1)

            # 両方が eliminate_length 未満なら除去
            if height < elim and width < elim:
                for (r, c) in comp:
                    out[r, c] = False

    return out
