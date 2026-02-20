# generate_inspect_image.py
# -*- coding: utf-8 -*-
"""
黒地(0)の画像に、配置・角度ランダムで白色(255)の直線(繊維)を描画してTIFFで保存します。

配置:
  workingdirectly/script/ に本スクリプトを置く想定
出力:
  workingdirectly/data/input_inspect/ に保存

ファイル名:
  test_ww_aaaa_nnnn.tiff
    ww   : fiber_width (0埋め, 2桁)
    aaaa : fiber_aspect_ratio (0埋め, 4桁)
    nnnn : fiber_number (0埋め, 4桁)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class Params:
    img_size: tuple[int, int] = (400, 400)  # (W, H)
    fiber_width: int = 2
    fiber_aspect_ratio: int = 30
    fiber_number: int = 50
    seed: int | None = None


def format_filename(fiber_width: int, aspect_ratio: int, fiber_number: int) -> str:
    # ww:2桁, aaaa:4桁, nnnn:4桁（足りない分は0埋め）
    return f"test_w{fiber_width:02d}_a{aspect_ratio:04d}_n{fiber_number:04d}.tiff"


def generate_inspection_image(params: Params) -> Image.Image:
    w, h = params.img_size
    img = Image.new("L", (w, h), 0)  # 黒地
    draw = ImageDraw.Draw(img)

    rng = np.random.default_rng(params.seed)

    # aspect_ratio = length / width -> length = width * aspect_ratio
    length_px = max(1.0, float(params.fiber_width) * float(params.fiber_aspect_ratio))
    half = length_px / 2.0

    for _ in range(params.fiber_number):
        cx = rng.uniform(0.0, w - 1.0)
        cy = rng.uniform(0.0, h - 1.0)
        theta = rng.uniform(0.0, 2.0 * math.pi)

        dx = half * math.cos(theta)
        dy = half * math.sin(theta)

        x0 = cx - dx
        y0 = cy - dy
        x1 = cx + dx
        y1 = cy + dy

        # 画像外にはみ出しても PIL がクリップして描画する
        draw.line((x0, y0, x1, y1), fill=255, width=params.fiber_width)

    return img


def main() -> None:
    # workingdirectly/script/ に置かれる想定なので、親が workingdirectly
    script_dir = Path(__file__).resolve().parent
    working_dir = script_dir.parent

    out_dir = working_dir / "data" / "input_inspect"
    out_dir.mkdir(parents=True, exist_ok=True)

    params = Params()
    img = generate_inspection_image(params)

    out_path = out_dir / format_filename(
        params.fiber_width, params.fiber_aspect_ratio, params.fiber_number
    )
    img.save(out_path, format="TIFF")

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
