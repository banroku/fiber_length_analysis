# Path: src/fiberlen/convert_to_graph.py

from __future__ import annotations

from typing import Dict, List, Tuple, Set

import numpy as np

from fiberlen.types import CompressedGraph, Node, NodeKind, Segment, Pixel


# 8近傍（固定）
_N8: List[Pixel] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def convert_to_graph(img_skeletonized: np.ndarray, border_margin_px: int) -> CompressedGraph:
    """
    スケルトン画像（bool）から compressed graph を生成する。

    仕様（あなたの設計どおり）
    ------------------------
    ・connectivity は 8 固定（4は実装しない）
    ・ノード：degree!=2 のスケルトン画素（端点=1、分岐>=3）
    ・セグメント：ノード間を結ぶ、degree==2 の鎖（polyline）
    ・セグメント長さ（length_px）を計算して Segment に格納
    ・border_margin_px 以内に触れているセグメントは touches_border=True

    Parameters
    ----------
    img_skeletonized : bool ndarray (H,W)
        True がスケルトン（前景）
    border_margin_px : int

    Returns
    -------
    graph : CompressedGraph
    """
    sk = np.asarray(img_skeletonized)
    if sk.dtype != np.bool_:
        sk = sk != 0

    h, w = sk.shape
    bm = int(border_margin_px)
    if bm < 0:
        bm = 0

    # ---- 1) 各スケルトン画素のdegree(8近傍)を計算 ----
    deg = np.zeros((h, w), dtype=np.uint8)

    # 端の扱いを簡単にするため、周囲1ピクセルパディング
    pad = np.pad(sk.astype(np.uint8), 1, mode="constant", constant_values=0)
    for dr, dc in _N8:
        deg += pad[1 + dr:1 + dr + h, 1 + dc:1 + dc + w].astype(np.uint8)

    deg = deg * sk.astype(np.uint8)  # 背景は0に

    # ---- 2) ノード候補：degree != 2 （かつ sk=True） ----
    node_mask = sk & (deg != 2)

    # degree=0 の孤立点があった場合も node として扱う（スケルトンが不完全でも落ちないため）
    # kind: endpoint(<=1), junction(>=3) で分類（degree=0/1 -> endpoint）
    coords = list(zip(*np.nonzero(node_mask)))  # [(r,c),...]

    g = CompressedGraph()

    coord_to_nid: Dict[Pixel, int] = {}
    nid = 1
    for (r, c) in coords:
        d = int(deg[r, c])
        kind = NodeKind.ENDPOINT if d <= 1 else NodeKind.JUNCTION
        node = Node(node_id=nid, coord=(int(r), int(c)), kind=kind, degree=d)
        g.add_node(node)
        coord_to_nid[(int(r), int(c))] = nid
        nid += 1

    # ノードがゼロなら空グラフを返す（後段で落ちないように）
    if not coord_to_nid:
        return g

    # ---- 3) セグメント抽出：ノードから伸びる枝を辿って次ノードまで ----
    # 重複作成を避けるため「無向エッジ」を訪問管理する
    # edge_key = ((r1,c1),(r2,c2)) をソートして保存
    visited_edges: Set[Tuple[Pixel, Pixel]] = set()

    seg_id = 1

    # 近傍の取得
    def neighbors_of(rc: Pixel) -> List[Pixel]:
        r, c = rc
        out: List[Pixel] = []
        for dr, dc in _N8:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and sk[rr, cc]:
                out.append((rr, cc))
        return out

    def edge_key(a: Pixel, b: Pixel) -> Tuple[Pixel, Pixel]:
        return (a, b) if a <= b else (b, a)

    def touches_border_pixels(pixels: List[Pixel]) -> bool:
        if bm <= 0:
            return False
        r_lo = bm
        c_lo = bm
        r_hi = h - 1 - bm
        c_hi = w - 1 - bm
        for r, c in pixels:
            if r < r_lo or c < c_lo or r > r_hi or c > c_hi:
                return True
        return False

    def segment_length_px(pixels: List[Pixel]) -> float:
        # 8近傍の移動長：縦横=1、斜め=sqrt(2)
        if len(pixels) < 2:
            return 0.0
        s = 0.0
        for (r0, c0), (r1, c1) in zip(pixels[:-1], pixels[1:]):
            dr = abs(r1 - r0)
            dc = abs(c1 - c0)
            if dr == 1 and dc == 1:
                s += 1.4142135623730951
            else:
                s += 1.0
        return float(s)

    # 各ノードから探索
    for (r, c), start_nid in coord_to_nid.items():
        start_px: Pixel = (r, c)
        for nb in neighbors_of(start_px):
            ek = edge_key(start_px, nb)
            if ek in visited_edges:
                continue

            # パス開始
            path: List[Pixel] = [start_px, nb]
            visited_edges.add(ek)

            prev = start_px
            cur = nb

            # cur がノードに到達するまで degree==2 の鎖を辿る
            # 途中で分岐(deg!=2)が出たらそこで止めてノード扱い（node_maskがTrueのはず）
            while True:
                if cur in coord_to_nid and cur != start_px:
                    # 次ノードに到達
                    break

                nbs = neighbors_of(cur)
                # prev を除いた「次の候補」
                cand = [p for p in nbs if p != prev]

                if len(cand) == 0:
                    # 行き止まり（本来はdeg=1でノードになってる想定だが、安全のためここで終了）
                    break

                if len(cand) >= 2:
                    # 分岐点に突入している（node_maskがTrueであるべき）
                    # ここを終端にする（ノード化はしていないので、終端ノードが見つからない可能性がある）
                    # ただし、このケースは「分岐ピクセルが node_mask に含まれる」なら while 冒頭で止まる。
                    # ここに来た場合は保険：打ち切り。
                    break

                nxt = cand[0]
                ek2 = edge_key(cur, nxt)
                if ek2 in visited_edges:
                    # 既に他の探索で消費済み。ここで終了
                    break
                visited_edges.add(ek2)

                path.append(nxt)
                prev, cur = cur, nxt

                # ループ保険（非常に稀）
                if len(path) > h * w:
                    break

            # end node の決定
            end_px = cur
            if end_px not in coord_to_nid:
                # 端点ノード化されていない等のケースでは segment を捨てる（グラフ一貫性優先）
                continue

            end_nid = coord_to_nid[end_px]

            seg = Segment(
                seg_id=seg_id,
                start_node=int(start_nid),
                end_node=int(end_nid),
                pixels=[(int(rr), int(cc)) for (rr, cc) in path],
            )
            seg.length_px = segment_length_px(seg.pixels)
            seg.touches_border = touches_border_pixels(seg.pixels)

            g.add_segment(seg)
            seg_id += 1

    return g
