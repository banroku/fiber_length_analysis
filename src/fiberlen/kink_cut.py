# Path: src/fiberlen/kink_cut.py

from __future__ import annotations

from dataclasses import replace
from typing import List, Tuple, Dict

import math
import numpy as np

from fiberlen.types import CompressedGraph, Node, NodeKind, Segment, SegmentKind, Pixel


def kink_cut(
    graph: CompressedGraph,
    threshold_of_nonlinear: float,
    blob_px: int,
    cut_max: int,
    cut_angle: float,
) -> CompressedGraph:
    """
    曲率が大きいセグメントの分離または廃棄処理

    仕様（あなたの設計どおり）
    ------------------------
    ・端部ノード間距離 D が 5px 以上のセグメントのみ対象
    ・セグメント長 L と D の比 L/D が threshold_of_nonlinear を超えるものだけ対象
    ・端から blob_px ごとにサンプリング点を取って角度（折れ曲がり）を評価
    ・折れ曲がり角度が cut_angle(°) を超える「キンク点」が cut_max 以内なら分割
    ・キンク点が cut_max より多ければ、そのセグメントは異常として丸ごと削除

    注意
    ----
    ・graph は「seg を含む」前提で、graph.segments を更新します。
    ・ノードdegreeは最終的に整合する必要がありますが、ここでは簡単化のため
      追加したノードは degree=2 で仮置きし、後段（convert_to_graph相当の検証/再計算）
      で厳密化してもよい設計にしてあります。
      （ただし、後段 pairing では degree を厳密に使う場合があるので、
       あなたの types.py の中で degree の再計算メソッドを持たせるのが安全です。）

    Returns
    -------
    graph_kink_cut : CompressedGraph
    """

    g = graph  # in-place 更新方針

    thr = float(threshold_of_nonlinear)
    blob = int(blob_px)
    cm = int(cut_max)
    ca = float(cut_angle)

    if blob <= 0 or cm < 0:
        return g

    # 新規ID採番
    next_node_id = _next_node_id(g)
    next_seg_id = _next_seg_id(g)

    # 対象セグメント一覧（走査中に辞書を書き換えない）
    seg_ids = list(g.segments.keys())

    for sid in seg_ids:
        seg = g.segments.get(sid)
        if seg is None:
            continue

        if getattr(seg, "kind", SegmentKind.SEGMENT) != SegmentKind.SEGMENT:
            continue

        pix = seg.pixels
        if len(pix) < 3:
            continue

        # 端点距離 D
        (r0, c0) = pix[0]
        (r1, c1) = pix[-1]
        D = math.hypot(float(r1 - r0), float(c1 - c0))

        if D < 5.0:
            continue

        L = float(getattr(seg, "length_px", 0.0))
        if L <= 0.0:
            L = _polyline_length_px(pix)

        if D <= 0.0:
            continue

        if (L / D) <= thr:
            continue

        # ---- blob_px ごとにサンプル点 index を取る（端点含む） ----
        sample_idx = _sample_indices_by_step(pix, step_px=float(blob))
        if len(sample_idx) < 3:
            continue  # 角度が評価できない

        # ---- 各サンプル点の折れ曲がり角（degree）を計算 ----
        # ---- 各サンプル点の折れ曲がり角（degree）を計算 ----
        # k=1..len-2 のみ（端点は除外）
        angs: List[float] = [0.0] * len(sample_idx)
        for k in range(1, len(sample_idx) - 1):
            i_prev = sample_idx[k - 1]
            i_cur = sample_idx[k]
            i_next = sample_idx[k + 1]
            angs[k] = _turn_angle_deg(pix[i_prev], pix[i_cur], pix[i_next])

        # ---- 極大（局所最大）だけ候補にする ----
        # 直線=0°, kinkほど大きい。cut_angle以上の局所最大のみ採用。
        candidates: List[int] = []
        for k in range(2, len(sample_idx) - 2):
            a = angs[k]
            if a < ca:
                continue
            # 局所最大（plateauは先頭側を採る）
            if a >= angs[k - 1] and a > angs[k + 1]:
                candidates.append(k)

        if not candidates:
            continue

        # ---- 近接候補の抑制（NMS） ----
        # 隣接（k差=1）または同一近傍に複数出た場合、角度が最大のものだけ残す
        candidates.sort(key=lambda k: angs[k], reverse=True)

        selected: List[int] = []
        suppressed: Set[int] = set()

        for k in candidates:
            if k in suppressed:
                continue
            selected.append(k)
            # 近いものを潰す：ここでは ±2 の範囲（必要なら修正）
            suppressed.add(k - 2)
            suppressed.add(k)
            suppressed.add(k + 2)

        kink_indices_in_samples = sorted(selected)

        if len(kink_indices_in_samples) == 0:
            continue


#         # ---- 各サンプル点の折れ曲がり角（degree）を計算 ----
#         kink_indices_in_samples: List[int] = []
#         for k in range(1, len(sample_idx) - 1):
#             i_prev = sample_idx[k - 1]
#             i_cur = sample_idx[k]
#             i_next = sample_idx[k + 1]
# 
#             ang = _turn_angle_deg(pix[i_prev], pix[i_cur], pix[i_next])
#             if ang >= ca:
#                 kink_indices_in_samples.append(k)
# 
#         if len(kink_indices_in_samples) == 0:
#             continue
# 
#         # キンクが多すぎる => セグメント丸ごと削除
#         if len(kink_indices_in_samples) > cm:
#             del g.segments[sid]
#             continue

        # ---- 分割する：キンク点ごとにノード追加し、セグメントを置換 ----
        cut_points_idx = [sample_idx[k] for k in kink_indices_in_samples]  # pixel配列内index
        cut_points_idx = sorted(set(cut_points_idx))

        # 端点と重なる切断点を除外（安全）
        cut_points_idx = [i for i in cut_points_idx if 0 < i < (len(pix) - 1)]
        if not cut_points_idx:
            continue

        # 元セグメント削除
        del g.segments[sid]

        # 新規ノードを作成（仮にdegree=2、kindはjunction扱いしない）
        cut_node_ids: List[int] = []
        for i in cut_points_idx:
            nid = next_node_id
            next_node_id += 1
            rr, cc = pix[i]
            g.nodes[nid] = Node(
                node_id=nid,
                coord=(int(rr), int(cc)),
                kind=NodeKind.JUNCTION,  # ここは「分割のための人工ノード」。endpointではないためjunctionで統一
                degree=2,
            )
            cut_node_ids.append(nid)

        # 分割後セグメントを作成
        # start_node -> cut1 -> cut2 -> ... -> end_node
        chain_nodes = [seg.start_node] + cut_node_ids + [seg.end_node]
        chain_indices = [0] + cut_points_idx + [len(pix) - 1]

        for a in range(len(chain_nodes) - 1):
            n_start = int(chain_nodes[a])
            n_end = int(chain_nodes[a + 1])
            i0 = int(chain_indices[a])
            i1 = int(chain_indices[a + 1])

            if i1 <= i0:
                continue

            sub_pixels = pix[i0:i1 + 1]
            new_seg = Segment(
                seg_id=next_seg_id,
                start_node=n_start,
                end_node=n_end,
                pixels=[(int(r), int(c)) for (r, c) in sub_pixels],
                kind=SegmentKind.SEGMENT,
            )
            new_seg.length_px = _polyline_length_px(new_seg.pixels)
            new_seg.touches_border = bool(getattr(seg, "touches_border", False))
            g.segments[next_seg_id] = new_seg
            next_seg_id += 1

    return g


# ----------------------- helpers -----------------------


def _next_node_id(g: CompressedGraph) -> int:
    if not g.nodes:
        return 1
    return int(max(g.nodes.keys())) + 1


def _next_seg_id(g: CompressedGraph) -> int:
    if not g.segments:
        return 1
    return int(max(g.segments.keys())) + 1


def _polyline_length_px(pixels: List[Pixel]) -> float:
    if len(pixels) < 2:
        return 0.0
    s = 0.0
    for (r0, c0), (r1, c1) in zip(pixels[:-1], pixels[1:]):
        dr = abs(int(r1) - int(r0))
        dc = abs(int(c1) - int(c0))
        if dr == 1 and dc == 1:
            s += 1.4142135623730951
        else:
            s += 1.0
    return float(s)


def _turn_angle_deg(p0: Pixel, p1: Pixel, p2: Pixel) -> float:
    """
    p0->p1 と p1->p2 のなす「折れ曲がり角」を返す（度）。
    0° は一直線（向きが同じ）、180° はUターン。

    ここでは「鋭い角」が大きい値になるようにしている。
    （直線に近いほど 0 に近い）
    """
    v1 = np.array([float(p1[0] - p0[0]), float(p1[1] - p0[1])], dtype=np.float64)
    v2 = np.array([float(p2[0] - p1[0]), float(p2[1] - p1[1])], dtype=np.float64)

    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0

    u1 = v1 / n1
    u2 = v2 / n2

    d = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
    ang = math.degrees(math.acos(d))
    return float(ang)


def _sample_indices_by_step(pixels: List[Pixel], step_px: float) -> List[int]:
    """
    polyline上を「累積距離」で step_px ごとにサンプルする。
    返すのは pixels 配列の index のリスト（端点含む）。
    """
    n = len(pixels)
    if n == 0:
        return []
    if n == 1:
        return [0]

    out = [0]
    acc = 0.0
    target = step_px

    for i in range(1, n):
        (r0, c0) = pixels[i - 1]
        (r1, c1) = pixels[i]
        dr = abs(int(r1) - int(r0))
        dc = abs(int(c1) - int(c0))
        d = 1.4142135623730951 if (dr == 1 and dc == 1) else 1.0

        acc += d
        if acc >= target:
            out.append(i)
            # 次のターゲットへ
            while acc >= target:
                target += step_px

    if out[-1] != (n - 1):
        out.append(n - 1)

    # 重複除去（安全）
    out2: List[int] = []
    for i in out:
        if not out2 or out2[-1] != i:
            out2.append(i)
    return out2
