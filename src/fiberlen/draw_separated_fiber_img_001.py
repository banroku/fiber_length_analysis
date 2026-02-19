# src/fiberlen/draw_separated_fiber_img.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict

import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio

from fiberlen.types import CompressedGraph, NodeKind


_OUT_DIR: Optional[Path] = None
_TAG: str = "result"


def configure_draw_output(out_dir: str | Path, tag: str) -> None:
    global _OUT_DIR, _TAG
    _OUT_DIR = Path(out_dir)
    _TAG = str(tag)


def draw_separated_fiber_img(graph_paired: CompressedGraph, img_skeletonized: np.ndarray) -> None:
    """
    仕様
    ・segments を seg_id 昇順で走査して色を変える
    ・junction で pairing がある場合は色を変えずに pairing 先へ飛ぶ（同色付与）
    ・出力サイズ/解像度は img_skeletonized と完全一致
    ・segmentのpixel座標に沿ってRGB配列へ直接描画し、TIFFで保存する

    出力: {out_dir}/{tag}__paired_segments.tif
    """
    out_dir = _OUT_DIR if _OUT_DIR is not None else Path("data/output")
    out_dir.mkdir(parents=True, exist_ok=True)

    h, w = img_skeletonized.shape[:2]

    # seg_id -> color_index
    color_map: Dict[int, int] = {}
    color_index = 0

    def follow_pair_chain(node: int, incoming_seg: int, cidx: int) -> None:
        n = int(node)
        inc = int(incoming_seg)
        while graph_paired.nodes[n].kind == NodeKind.JUNCTION:
            nxt = graph_paired.get_paired_next(n, inc)
            if nxt is None:
                return
            nxt = int(nxt)
            if nxt in color_map:
                return
            color_map[nxt] = cidx
            n = graph_paired.other_node(nxt, n)
            inc = nxt

    # seg_id 昇順で色付け開始
    for sid in sorted(graph_paired.segments.keys()):
        sid = int(sid)
        if sid in color_map:
            continue

        color_index += 1
        color_map[sid] = color_index

        seg = graph_paired.segments[sid]
        follow_pair_chain(seg.start_node, sid, color_index)
        follow_pair_chain(seg.end_node, sid, color_index)

    # ラベル画像（各ピクセルがどの color_index か）を作ってから、LUTでRGB化する
    label = np.zeros((h, w), dtype=np.int32)

    for sid in sorted(graph_paired.segments.keys()):
        sid = int(sid)
        seg = graph_paired.segments[sid]
        if not seg.pixels:
            continue

        rr = np.fromiter((p[0] for p in seg.pixels), dtype=np.int32)
        cc = np.fromiter((p[1] for p in seg.pixels), dtype=np.int32)

        cidx = color_map.get(sid, 1)
        label[rr, cc] = cidx

    # color_index -> RGB (uint8) のLUT
    cmap = plt.get_cmap("tab20")
    lut = (cmap((np.arange(color_index + 1) % cmap.N))[:, :3] * 255).astype(np.uint8)
    lut[0] = 0  # 背景は黒

    rgb = lut[label]  # (H, W, 3) uint8

    out_path = out_dir / f"{_TAG}__paired_segments.tif"
    iio.imwrite(out_path, rgb)
