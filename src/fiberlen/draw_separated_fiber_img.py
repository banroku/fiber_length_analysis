from __future__ import annotations

from typing import Set

import numpy as np
import matplotlib.pyplot as plt

from fiberlen.types import CompressedGraph, NodeKind, SegmentKind


def draw_separated_fiber_img(graph_paired: CompressedGraph, img_skeletonized: np.ndarray) -> np.ndarray:
    """
    img_skeletonized (0/1) を土台に、pair_map に従って連結したセグメント列を同色で上書き描画する。
    touches_border=True の chain は白で描画する（白で辿って visited にも入れる）。
    戻り値は uint8 の RGB 画像 (H, W, 3)。
    """
    h, w = img_skeletonized.shape[:2]

    base = (img_skeletonized.astype(np.uint8) * 255)
    out = np.repeat(base[:, :, None], 3, axis=2)

    cmap = plt.get_cmap("tab20")
    WHITE = np.array([255, 255, 255], dtype=np.uint8)

    visited: Set[int] = set()
    color_i = 0

    def draw_segment(seg_id: int, rgb: np.ndarray) -> None:
        seg = graph_paired.segments[seg_id]
        if not seg.pixels:
            return
        rr = np.fromiter((p[0] for p in seg.pixels), dtype=np.int32)
        cc = np.fromiter((p[1] for p in seg.pixels), dtype=np.int32)
        out[rr, cc] = rgb

    def walk_from(node: int, incoming_seg: int, rgb: np.ndarray) -> None:
        n = int(node)
        inc = int(incoming_seg)
        while graph_paired.nodes[n].kind == NodeKind.JUNCTION:
            nxt = graph_paired.get_paired_next(n, inc)
            if nxt is None:
                return
            nxt = int(nxt)

            nxt_seg = graph_paired.segments[nxt]
            if nxt_seg.kind != SegmentKind.SEGMENT:
                return
            if nxt in visited:
                return

            visited.add(nxt)
            draw_segment(nxt, rgb)

            n = graph_paired.other_node(nxt, n)
            inc = nxt

    for sid in sorted(graph_paired.segments.keys()):
        seg = graph_paired.segments[sid]
        if seg.kind != SegmentKind.SEGMENT:
            continue
        if sid in visited:
            continue

        if seg.touches_border:
            rgb = WHITE
        else:
            rgb = (np.array(cmap(color_i % cmap.N)[:3]) * 255).astype(np.uint8)
            color_i += 1

        visited.add(sid)
        draw_segment(sid, rgb)

        walk_from(seg.start_node, sid, rgb)
        walk_from(seg.end_node, sid, rgb)

#     # blob要素を白で上書き（従来仕様）
#     for seg in graph_paired.segments.values():
#         if seg.kind != SegmentKind.JUNCTION_ELEMENT:
#             continue
#         for r, c in seg.pixels:
#             out[r, c] = WHITE
# 
#     for node in graph_paired.nodes.values():
#         if node.kind != NodeKind.JUNCTION_ELEMENT:
#             continue
#         r, c = node.coord
#         out[r, c] = WHITE

    return out
