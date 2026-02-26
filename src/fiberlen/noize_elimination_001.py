# Path: src/fiberlen/noize_elimination.py

from __future__ import annotations

# import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import remove_small_objects


def noize_elimination(img_binarized: np.ndarray, eliminate_length: int) -> np.ndarray:
    """
    ノイズ除去（skimage.remove_small_objects）

    方針：
      連結成分の「面積（ピクセル数）」が小さいものを除去する。

    eliminate_length の解釈：
      GUI/設定で「長さ(px)」として扱いやすいように、
      min_size = eliminate_length**2（ピクセル数）を採用する。
      例：eliminate_length=10 -> 面積 100 未満を除去

      eliminate_length を「ピクセル数そのもの」にしたいなら、
      min_size の行を min_size = int(eliminate_length) に変更すればよい。

    連結性：
      8連結固定（2Dの connectivity=2）

    Parameters
    ----------
    img_binarized : bool or {0,1} ndarray
        True が前景（繊維）
    eliminate_length : int
        ノイズ除去の強さ（長さっぽく指定）

    Returns
    -------
    img_for_skeletonized : bool ndarray
    """
#    bw = np.asarray(img_binarized)
#    if bw.dtype != np.bool_:
#        bw = bw != 0
    bw = img_binarized

    elim = int(eliminate_length)
    if elim <= 0:
        return bw.copy()

    min_size = elim * elim  # ←必要ならここを変更（面積ピクセル数）

    # remove_small_objects は bool を返す
    cleaned = remove_small_objects(bw, max_size=min_size, connectivity=2)

#     fig, ax = plt.subplots(1, 2, figsize=(10, 4))
#     ax[0].imshow(bw, cmap="gray", vmin=0, vmax=1)
#     ax[0].set_title("Input (binary)")
#     ax[0].axis("off")
# 
#     ax[1].imshow(cleaned, cmap="gray", vmin=0, vmax=1)
#     ax[1].set_title(f"After noize_elimination (L={eliminate_length})")
#     ax[1].axis("off")
# 
#     plt.tight_layout()
#     plt.show()


    return cleaned.astype(np.bool_, copy=False)
