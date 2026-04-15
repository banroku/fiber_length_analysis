from __future__ import annotations

import numpy as np
import cv2


def noize_elimination(img_binarized: np.ndarray, eliminate_length: int) -> np.ndarray:
    """
    ノイズ除去（OpenCV, 外接矩形の w/h ベース）

    方針：
      連結成分ごとに外接矩形 (x, y, w, h) を求め、
      w と h の両方が eliminate_length 未満の成分を除去する。

    判定条件：
      keep if (w >= eliminate_length) or (h >= eliminate_length)
      remove if (w < eliminate_length) and (h < eliminate_length)

    連結性：
      8連結

    Parameters
    ----------
    img_binarized : bool or {0,1} ndarray
        True/1 が前景（繊維）
    eliminate_length : int
        除去しきい値 [px]

    Returns
    -------
    img_for_skeletonized : bool ndarray
    """
    bw = np.asarray(img_binarized)

    if bw.dtype != np.uint8:
        bw_u8 = (bw != 0).astype(np.uint8)
    else:
        bw_u8 = (bw != 0).astype(np.uint8)

    elim = int(eliminate_length)
    if elim <= 0:
        return (bw_u8 != 0)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw_u8, connectivity=8)

    cleaned = np.zeros_like(bw_u8, dtype=np.uint8)

    # label 0 は背景なので 1 から開始
    for label in range(1, num_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]

        if (w >= elim) or (h >= elim):
            cleaned[labels == label] = 1

    return cleaned.astype(np.bool_, copy=False)