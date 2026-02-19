from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict

import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio

from fiberlen.types import (
    CompressedGraph,
    NodeKind,
    SegmentKind,  # ★追加
)


_OUT_DIR: Optional[Path] = None
_TAG: str = "result"


def configure_draw_output(out_dir: str | Path, tag: str) -> None:
    global _OUT_DIR, _TAG
    _OUT_DIR = Path(out_dir)
    _TAG = str(tag)


def draw_separated_fiber_img(graph_paired: CompressedGraph, img_skeletonized: np.ndarray) -> None:

    out_dir = _OUT_DIR if _OUT_DIR is not None else Path("data/output")
    out_dir.mkdir(parents=True, exist_ok=True)

    h, w = img_skeletonized.shape[:2]

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

    # ----------------------------------------
    # 色付け（通常セグメントのみ）
    # ----------------------------------------
    for sid in sorted(graph_paired.segments.keys()):
        seg = graph_paired.segments[sid]

        if seg.kind != SegmentKind.SEGMENT:
            continue

        if sid in color_map:
            continue

        color_index += 1
        color_map[sid] = color_index

        follow_pair_chain(seg.start_node, sid, color_index)
        follow_pair_chain(seg.end_node, sid, color_index)

    # ----------------------------------------
    # 描画ラベル作成（通常セグメント）
    # ----------------------------------------
    label = np.zeros((h, w), dtype=np.int32)

    for sid, seg in graph_paired.segments.items():
        if seg.kind != SegmentKind.SEGMENT:
            continue

        if not seg.pixels:
            continue

        rr = np.fromiter((p[0] for p in seg.pixels), dtype=np.int32)
        cc = np.fromiter((p[1] for p in seg.pixels), dtype=np.int32)

        cidx = color_map.get(sid, 1)
        label[rr, cc] = cidx

    # ----------------------------------------
    # LUT生成
    # ----------------------------------------
    cmap = plt.get_cmap("tab20")
    lut = (cmap((np.arange(color_index + 1) % cmap.N))[:, :3] * 255).astype(np.uint8)
    lut[0] = 0

    rgb = lut[label]

    # ----------------------------------------
    # ★ blob要素を白で上書き
    # ----------------------------------------

    WHITE = np.array([255, 255, 255], dtype=np.uint8)

    # セグメントblob
    for seg in graph_paired.segments.values():
        if seg.kind != SegmentKind.JUNCTION_ELEMENT:
            continue
        for r, c in seg.pixels:
            rgb[r, c] = WHITE

    # ノードblob
    for node in graph_paired.nodes.values():
        if node.kind != NodeKind.JUNCTION_ELEMENT:
            continue
        r, c = node.coord
        rgb[r, c] = WHITE

    # ----------------------------------------
    # 保存
    # ----------------------------------------
    out_path = out_dir / f"{_TAG}__paired_segments.tif"
    iio.imwrite(out_path, rgb)

