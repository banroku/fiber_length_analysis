# Path: src/fiberlen/binarize.py

from __future__ import annotations
import numpy as np


def binarize(img_preprocessed: np.ndarray, threshold: float) -> np.ndarray:
    """
    二値化（背景が黒、繊維が明るい前提）

    Parameters
    ----------
    img_preprocessed : float ndarray [0,1]
        前処理後画像（背景が黒側）
    threshold : float
        0.0-1.0 の閾値。これより大きい画素を繊維(=1)とする。

    Returns
    -------
    img_binarized : bool ndarray
        True が繊維（前景=1）
    """
    x = np.asarray(img_preprocessed, dtype=np.float32)
    t = float(threshold)
    return (x > t)
