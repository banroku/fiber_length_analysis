from __future__ import annotations

import math
import numpy as np

from fiberlen.types import CompressedGraph, Node, NodeKind, Segment, SegmentKind


def kink_cut(
    graph: CompressedGraph,
    threshold_of_nonlinear: float,
    cut_max: int,
    cut_angle: float,
) -> CompressedGraph:
    """
    曲率ベースの分割 / 削除

    方針
    ----
    1. まず L/D で足切り
    2. 通過したセグメントだけ、各点 i について
       (i-3) -> i と i -> (i+3) の折れ角を計算
    3. 強いピークだけ拾う
    4. 強いピークが
       - 0個 : 何もしない
       - 1?cut_max個 : その点で分割
       - cut_max個より多い : セグメント削除
    """
    g = graph

    thr = float(threshold_of_nonlinear)
    cm = int(cut_max)
    ca = float(cut_angle)

    if cm < 0:
        return g

    next_node_id = 1 if not g.nodes else max(g.nodes.keys()) + 1
    next_seg_id = 1 if not g.segments else max(g.segments.keys()) + 1

    seg_ids = list(g.segments.keys())

    for sid in seg_ids:
        seg = g.segments.get(sid)
        if seg is None:
            continue

        if seg.kind != SegmentKind.SEGMENT:
            continue

        pix = seg.pixels
        if len(pix) < 7:
            continue

        # 端点距離 D
        r0, c0 = pix[0]
        r1, c1 = pix[-1]
        D = math.hypot(float(r1 - r0), float(c1 - c0))
        if D < 5.0:
            continue

        # セグメント長 L
        if seg.length_px is not None and seg.length_px > 0.0:
            L = float(seg.length_px)
        else:
            L = 0.0
            for (ra, ca0), (rb, cb) in zip(pix[:-1], pix[1:]):
                L += math.hypot(float(rb - ra), float(cb - ca0))

        if D <= 0.0:
            continue

        # L/D 足切り
        if (L / D) <= thr:
            continue

        # 各点の折れ角を計算（±3）
        angles = [0.0] * len(pix)
        for i in range(3, len(pix) - 3):
            p0 = pix[i - 3]
            p1 = pix[i]
            p2 = pix[i + 3]

            v1 = np.array([float(p1[0] - p0[0]), float(p1[1] - p0[1])], dtype=np.float64)
            v2 = np.array([float(p2[0] - p1[0]), float(p2[1] - p1[1])], dtype=np.float64)

            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 == 0.0 or n2 == 0.0:
                angles[i] = 0.0
                continue

            u1 = v1 / n1
            u2 = v2 / n2
            d = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
            angles[i] = float(math.degrees(math.acos(d)))

        # 強い局所ピークを拾う
        peaks = []
        for i in range(4, len(pix) - 4):
            a = angles[i]
            if a < ca:
                continue
            if a >= angles[i - 1] and a > angles[i + 1]:
                peaks.append(i)

        # 近すぎるピークはまとめる
        merged_peaks = []
        for i in peaks:
            if not merged_peaks:
                merged_peaks.append(i)
            elif i - merged_peaks[-1] <= 3:
                if angles[i] > angles[merged_peaks[-1]]:
                    merged_peaks[-1] = i
            else:
                merged_peaks.append(i)

        # ピークなし
        if len(merged_peaks) == 0:
            continue

        # ピーク多すぎ → 削除
        if len(merged_peaks) > cm:
            if seg.start_node in g.adjacency:
                g.adjacency[seg.start_node].discard(sid)
            if seg.end_node in g.adjacency:
                g.adjacency[seg.end_node].discard(sid)
            del g.segments[sid]
            continue

        # 少数ピーク → 分割
        cut_points = [i for i in merged_peaks if 0 < i < len(pix) - 1]
        if not cut_points:
            continue

        old_seg = g.segments.get(sid)
        if old_seg is None:
            continue

        if old_seg.start_node in g.adjacency:
            g.adjacency[old_seg.start_node].discard(sid)
        if old_seg.end_node in g.adjacency:
            g.adjacency[old_seg.end_node].discard(sid)
        del g.segments[sid]

        cut_node_ids = []
        for i in cut_points:
            rr, cc = pix[i]
            nid = next_node_id
            next_node_id += 1

            new_node = Node(
                node_id=nid,
                coord=(int(rr), int(cc)),
                kind=NodeKind.JUNCTION,
                degree=2,
            )
            g.add_node(new_node)
            cut_node_ids.append(nid)

        chain_nodes = [old_seg.start_node] + cut_node_ids + [old_seg.end_node]
        chain_idx = [0] + cut_points + [len(pix) - 1]

        for k in range(len(chain_nodes) - 1):
            n_start = chain_nodes[k]
            n_end = chain_nodes[k + 1]
            i0 = chain_idx[k]
            i1 = chain_idx[k + 1]

            if i1 <= i0:
                continue

            sub_pixels = [(int(r), int(c)) for (r, c) in pix[i0:i1 + 1]]
            if len(sub_pixels) < 2:
                continue

            sub_len = 0.0
            for (ra, ca0), (rb, cb) in zip(sub_pixels[:-1], sub_pixels[1:]):
                sub_len += math.hypot(float(rb - ra), float(cb - ca0))

            new_seg = Segment(
                seg_id=next_seg_id,
                start_node=int(n_start),
                end_node=int(n_end),
                pixels=sub_pixels,
                length_px=sub_len,
                touches_border=bool(old_seg.touches_border),
                is_pruned=bool(old_seg.is_pruned),
                kind=SegmentKind.SEGMENT,
            )
            g.add_segment(new_seg)
            next_seg_id += 1

    return g
