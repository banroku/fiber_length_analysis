# Path: src/fiberlen/subtract_background.py

from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter


def subtract_background(
    img_raw: np.ndarray,
    blur_sigma_px: float,
) -> np.ndarray:
    """
    背景均一化（masked blur方式）

    前提
    ----
    img_raw は float 0-1, 背景が黒であること
    （input_img 済み）

    手順
    ----
    1) 画像平均を基準値 ref とする
    2) ref より明るい領域を「背景候補」とみなす
    3) 背景候補以外（=繊維）は ref に置換
    4) Gaussian blur で局所平均背景生成
    5) 元画像から差分を取って正規化

    Parameters
    ----------
    img_raw : float ndarray [0,1]
    blur_sigma_px : float
        局所平均スケール（繊維幅の10倍程度想定）

    Returns
    -------
    img_preprocessed : float ndarray [0,1]
    """

    img = np.asarray(img_raw, dtype=np.float32)

    # ----- 基準輝度 -----
    #ref = float(np.mean(img))
    ref = gaussian_filter(img_raw, sigma=float(blur_sigma_px))

    # ----- 背景マスク -----
    background_mask = img > ref

    # ----- マスク置換 -----
    masked = img.copy()
    masked[~background_mask] = ref[~background_mask]

    # ----- 局所平均 -----
    background = gaussian_filter(masked, sigma=float(blur_sigma_px))

    # ----- 差分 -----
    corrected = img - background

    # ----- 正規化 -----
    mn = float(np.min(corrected))
    mx = float(np.max(corrected))

    if mx > mn:
        corrected = (corrected - mn) / (mx - mn)
    else:
        corrected[:] = 0.0

    return corrected.astype(np.float32, copy=False)
