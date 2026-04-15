from __future__ import annotations

import numpy as np
from skimage.measure import label, regionprops


def noize_elimination(img_binarized: np.ndarray, eliminate_length: int) -> np.ndarray:
    """
    ノイズ除去（skimage, 外接矩形の w/h ベース）

    方針：
      連結成分ごとに外接矩形の幅 w、高さ h を求め、
      w と h の両方が eliminate_length 未満の成分を除去する。

    判定条件：
      keep if (w >= eliminate_length) or (h >= eliminate_length)
      remove if (w < eliminate_length) and (h < eliminate_length)

    連結性：
      8連結

    Parameters
    ----------
    img_binarized : bool or {0,1} ndarray
        True が前景（繊維）
    eliminate_length : int
        除去しきい値 [px]

    Returns
    -------
    img_for_skeletonized : bool ndarray
    """
    bw = np.asarray(img_binarized, dtype=bool)

    elim = int(eliminate_length)
    if elim <= 0:
        return bw.copy()

    # 8連結
    labels = label(bw, connectivity=2)

    cleaned = np.zeros_like(bw, dtype=bool)

    for region in regionprops(labels):
        min_row, min_col, max_row, max_col = region.bbox
        h = max_row - min_row
        w = max_col - min_col

        if (w >= elim) or (h >= elim):
            cleaned[labels == region.label] = True

    return cleaned