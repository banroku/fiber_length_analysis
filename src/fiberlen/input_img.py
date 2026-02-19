# Path: src/fiberlen/io.py
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import imageio.v3 as iio

PathLike = Union[str, Path]


def input_img(img_path: PathLike, background_is_dark: bool) -> np.ndarray:
    """
    画像を読み込み、float32の0.0-1.0へ正規化して返す。
    以後の処理は「背景が黒（低輝度）」前提なので、
    background_is_dark=False（背景が黒でないタイプ）ならネガポジ変換して背景を黒側に揃える。
    """
    arr = iio.imread(str(Path(img_path)))
    gray = _to_grayscale(arr)
    img01 = _normalize01(gray)

    if not background_is_dark:
        img01 = 1.0 - img01

    return img01.astype(np.float32, copy=False)


def read_skeleton_tif(img_path: PathLike, foreground: str = "auto") -> np.ndarray:
    """
    skeleton用のtifを読み込み、bool配列で返す。

    foreground:
      - "auto": True側が少ない方を前景（スケルトン）とみなす
      - "white": 白(高輝度)が前景
      - "black": 黒(低輝度)が前景
    """
    arr = iio.imread(str(Path(img_path)))
    if arr.ndim == 3:
        arr = _to_grayscale(arr)

    if arr.dtype == np.bool_:
        sk = arr.copy()
    else:
        a = np.asarray(arr)
        if np.issubdtype(a.dtype, np.integer):
            info = np.iinfo(a.dtype)
            thr = info.max // 2
            sk = a > thr
        else:
            a01 = _normalize01(a)
            sk = a01 > 0.5

    fg = foreground.lower()
    if fg not in ("auto", "white", "black"):
        raise ValueError("foreground must be 'auto', 'white', or 'black'")

    if fg == "white":
        return sk.astype(bool, copy=False)
    if fg == "black":
        return (~sk).astype(bool, copy=False)

    # auto: Trueの方が多いなら反転して「細い線」をTrueに寄せる
    return (sk if np.count_nonzero(sk) <= (sk.size // 2) else ~sk).astype(bool, copy=False)


def _to_grayscale(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr)

    if x.ndim == 2:
        return x

    if x.ndim == 3 and x.shape[2] in (3, 4):
        rgb = x[..., :3].astype(np.float32, copy=False)
        return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

    raise ValueError(f"Unsupported image shape: {x.shape}")


def _normalize01(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x)

    if a.dtype == np.bool_:
        return a.astype(np.float32)

    if np.issubdtype(a.dtype, np.integer):
        info = np.iinfo(a.dtype)
        return (a.astype(np.float32) / float(info.max)).astype(np.float32, copy=False)

    # float or other: min-max normalize
    a2 = a.astype(np.float32, copy=False)
    mn = float(np.nanmin(a2))
    mx = float(np.nanmax(a2))
    if mx <= mn:
        return np.zeros_like(a2, dtype=np.float32)
    return (a2 - mn) / (mx - mn)
