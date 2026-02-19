# Path: src/fiberlen/calc_Otsu_threshold.py

from __future__ import annotations
import numpy as np


def calc_Otsu_threshold(img_preprocessed: np.ndarray) -> float:
    """
    大津の二値化閾値を計算して返す（GUIでの推奨値表示用）

    前提
    ----
    img_preprocessed は float 0-1 を想定（subtract_backgroundの出力）

    注意
    ----
    これは「推奨閾値」を返すだけで、パイプラインで二値化を実行しません。

    Returns
    -------
    threshold_Otsu : float
        0.0-1.0 の閾値
    """
    x = np.asarray(img_preprocessed, dtype=np.float32)

    # NaN/inf除外
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.5

    # 0-1 を仮定（念のためクリップ）
    x = np.clip(x, 0.0, 1.0)

    # ヒストグラム（0..255）
    nbins = 256
    hist = np.bincount((x * (nbins - 1)).astype(np.int32), minlength=nbins).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.5

    # 確率
    p = hist / total
    omega = np.cumsum(p)                          # class0の累積確率
    mu = np.cumsum(p * np.arange(nbins))          # class0の累積平均（濃度）
    mu_t = mu[-1]                                 # 全体平均

    # クラス間分散 sigma_b^2
    # 分母0回避のため安全に
    denom = omega * (1.0 - omega)
    sigma_b2 = np.zeros_like(denom)
    valid = denom > 1e-12
    sigma_b2[valid] = ((mu_t * omega[valid] - mu[valid]) ** 2) / denom[valid]

    k = int(np.argmax(sigma_b2))                  # 最良bin
    thr = k / float(nbins - 1)                    # 0..1へ戻す

    return float(thr)
