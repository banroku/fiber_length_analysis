# Path: script/inspect_noize_elimination.py
# 使い方: python script/inspect_noize_elimination.py
#
# 入力画像はデフォルトで data/input/noize_test_binary.png を見に行きます。
# そこに置かない場合は、このファイルの INPUT_PATH を書き換えてください。

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio

from fiberlen.noize_elimination import noize_elimination


INPUT_PATH = os.path.join("data", "input", "noize_test_binary.png")


def _read_binary_image(path: str) -> np.ndarray:
    """
    画像を読み込み、2値(bool)にする。
    白(>0)=True, 黒(0)=False の前提。
    """
    img = iio.imread(path)
    if img.ndim == 3:
        # RGB等なら輝度に落とす（単純平均で十分）
        img = img[..., :3].mean(axis=2)
    bw = (img > 0)
    return bw.astype(bool, copy=False)


def main() -> None:
    print(f"[inspect_noize_elimination] Using: {INPUT_PATH}")
    bw = _read_binary_image(INPUT_PATH)

    print("[inspect_noize_elimination] input dtype:", bw.dtype)
    print("[inspect_noize_elimination] input true pixels:", int(bw.sum()), "fraction:", float(bw.mean()))

    eliminate_length = 10  # ここを変えて挙動を見る（例: 5, 10, 20, 40）
    cleaned = noize_elimination(bw, eliminate_length=eliminate_length)

    cleaned = np.asarray(cleaned).astype(bool, copy=False)
    print("[inspect_noize_elimination] eliminate_length:", int(eliminate_length))
    print("[inspect_noize_elimination] output dtype:", cleaned.dtype)
    print("[inspect_noize_elimination] output true pixels:", int(cleaned.sum()), "fraction:", float(cleaned.mean()))
    print("[inspect_noize_elimination] removed pixels:", int(bw.sum() - cleaned.sum()))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(bw, cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("Input (binary)")
    ax[0].axis("off")

    ax[1].imshow(cleaned, cmap="gray", vmin=0, vmax=1)
    ax[1].set_title(f"After noize_elimination (L={eliminate_length})")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()

    print("[inspect_noize_elimination] done")


if __name__ == "__main__":
    main()
