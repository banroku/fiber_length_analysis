# src/fiberlen/merge_nodes.py

from __future__ import annotations

from collections import deque
from typing import Dict, List, Set, Tuple, Iterable

import numpy as np

from fiberlen.types import CompressedGraph, Node, NodeKind, SegmentKind, Pixel


# 8-neighborhood
_N8: List[Pixel] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def merge_nodes(graph: CompressedGraph, merge_short_seg_px: float) -> CompressedGraph:
    """
    近接ノードの縮約 (blob化)

    ・ノードが8近傍で直接隣接している場合は同一blobにまとめる
    ・merge_short_seg_px 以下の length_px を持つセグメントで接続されるノードも同一blobにまとめる
    ・blobの代表点は、blobを構成するpixel集合の幾何中心に最も近いpixelとし、新規ノードをNodeKind.JUNCTIONで生成する
    ・blobに吸収された既存ノードは graph に残しつつ NodeKind.JUNCTION_ELEMENT に変更する（統計対象外）
    ・blob内部の短セグメントは graph に残しつつ SegmentKind.JUNCTION_ELEMENT に変更する（統計対象外）
    ・blob外へ接続しているセグメントは、blob側端点ノードIDを代表ノードへ付け替える
    ・代表ノード座標とセグメント端点pixelが8近傍で繋がらない場合に備え、
      blob内pixel集合上の最短経路（8近傍）をセグメントpixelsに付加し、length_pxを更新する
      （measure_length が segment.length_px の総和で測る前提への対策）
    """
    thr = float(merge_short_seg_px)

    # 対象とするノード（blob要素は対象外）
    active_nodes: Dict[int, Node] = {
        int(nid): n for nid, n in graph.nodes.items()
        if n.kind != NodeKind.JUNCTION_ELEMENT
    }
    if not active_nodes:
        return graph

    # ノード座標 -> node_id
    coord_to_nid: Dict[Pixel, int] = {tuple(map(int, n.coord)): int(nid) for nid, n in active_nodes.items()}

    # ノード間隣接グラフを構築
    adj: Dict[int, Set[int]] = {nid: set() for nid in active_nodes.keys()}

    # (1) 8近傍で直接隣接しているノードを接続
    for (r, c), nid in coord_to_nid.items():
        for dr, dc in _N8:
            nb = (r + dr, c + dc)
            nbid = coord_to_nid.get(nb)
            if nbid is not None and nbid != nid:
                adj[nid].add(nbid)
                adj[nbid].add(nid)

    # (2) 短セグメントで接続されているノードを接続
    # ここでは「統計対象のセグメント (SEGMENT)」のみを参照する
    for seg in graph.segments.values():
        if seg.kind != SegmentKind.SEGMENT:
            continue
        if seg.length_px is None:
            continue
        if float(seg.length_px) > thr:
            continue
        a = int(seg.start_node)
        b = int(seg.end_node)
        if a == b:
            continue
        if a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)

    # 連結成分（サイズ>=2）を抽出
    comps: List[List[int]] = []
    seen: Set[int] = set()
    for nid in adj.keys():
        if nid in seen:
            continue
        q = deque([nid])
        seen.add(nid)
        comp: List[int] = []
        while q:
            x = q.popleft()
            comp.append(x)
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    q.append(y)
        if len(comp) >= 2:
            comps.append(comp)

    if not comps:
        return graph

    # 新規node_id発行用
    max_nid = int(max(graph.nodes.keys())) if graph.nodes else 0

    def _step_len(p0: Pixel, p1: Pixel) -> float:
        dr = abs(int(p1[0]) - int(p0[0]))
        dc = abs(int(p1[1]) - int(p0[1]))
        return 1.4142135623730951 if (dr == 1 and dc == 1) else 1.0

    def _path_len(pxs: List[Pixel]) -> float:
        if len(pxs) < 2:
            return 0.0
        s = 0.0
        for p0, p1 in zip(pxs[:-1], pxs[1:]):
            s += _step_len(p0, p1)
        return float(s)

    def _bfs_path(blob_set: Set[Pixel], src: Pixel, dst: Pixel) -> List[Pixel]:
        # blob_set 上で src -> dst の8近傍経路（含む）を返す。失敗時は [src]。
        if src == dst:
            return [src]
        q = deque([src])
        prev: Dict[Pixel, Pixel] = {}
        visited: Set[Pixel] = {src}
        while q:
            cur = q.popleft()
            r, c = cur
            for dr, dc in _N8:
                nb = (r + dr, c + dc)
                if nb not in blob_set or nb in visited:
                    continue
                visited.add(nb)
                prev[nb] = cur
                if nb == dst:
                    q.clear()
                    break
                q.append(nb)

        if dst not in prev:
            return [src]

        path = [dst]
        cur = dst
        while cur != src:
            cur = prev[cur]
            path.append(cur)
        path.reverse()
        return path

    # compごとに縮約
    for comp in comps:
        comp_set = set(comp)

        # blobを構成するpixel集合
        # 最低限、(a) comp内ノード座標、(b) comp内ノード同士を結ぶ短セグメントpixels を含める
        blob_pixels: Set[Pixel] = set(tuple(map(int, graph.nodes[nid].coord)) for nid in comp)

        for seg in graph.segments.values():
            if seg.kind != SegmentKind.SEGMENT:
                continue
            if seg.length_px is None or float(seg.length_px) > thr:
                continue
            a = int(seg.start_node)
            b = int(seg.end_node)
            if a in comp_set and b in comp_set:
                for p in seg.pixels:
                    blob_pixels.add((int(p[0]), int(p[1])))

        # 代表座標: 幾何中心に最も近いpixel
        pts = np.array(list(blob_pixels), dtype=np.float32)
        center = pts.mean(axis=0)
        d2 = (pts[:, 0] - center[0]) ** 2 + (pts[:, 1] - center[1]) ** 2
        rep_coord: Pixel = tuple(map(int, pts[int(np.argmin(d2))]))

        # 代表ノードを新規作成（kind=JUNCTION）
        max_nid += 1
        rep_nid = max_nid
        rep_node = Node(node_id=rep_nid, coord=rep_coord, kind=NodeKind.JUNCTION, degree=0)
        graph.add_node(rep_node)

        # comp内ノードは kind=JUNCTION_ELEMENT にして残す
        for nid in comp:
            n = graph.nodes[nid]
            n.kind = NodeKind.JUNCTION_ELEMENT

        # セグメントの処理
        # (i) comp内短セグメントは kind=JUNCTION_ELEMENT にして残す
        # (ii) comp外接続セグメントは comp側端点を rep_nid に付け替え、必要ならpixelsを補完しlength更新
        for seg in graph.segments.values():
            if seg.kind != SegmentKind.SEGMENT:
                continue

            a0 = int(seg.start_node)
            b0 = int(seg.end_node)

            a_in = a0 in comp_set
            b_in = b0 in comp_set

            if a_in and b_in:
                # blob内部セグメント。短いものはblob要素に落とす。長いものがあるならそのままでも良いが、
                # 本仕様では「短セグメントを挟んだノードもまとめる」前提なので、内部に残るSEGMENTは基本想定しない。
                if seg.length_px is not None and float(seg.length_px) <= thr:
                    seg.kind = SegmentKind.JUNCTION_ELEMENT
                continue

            if not (a_in or b_in):
                continue

            # 外部セグメント：blob側端点を代表ノードへ
            pxs: List[Pixel] = [(int(p[0]), int(p[1])) for p in seg.pixels]

            if a_in:
                # a側を代表へ
                end_px = pxs[0] if pxs else tuple(map(int, graph.nodes[a0].coord))
                if end_px in blob_pixels and rep_coord in blob_pixels:
                    path_end_to_rep = _bfs_path(blob_pixels, end_px, rep_coord)  # end -> rep
                    if len(path_end_to_rep) >= 2:
                        rep_to_end = list(reversed(path_end_to_rep))  # rep -> end
                        # rep_to_end[-1] == end_px を重複させない
                        pxs = rep_to_end[:-1] + pxs
                seg.start_node = rep_nid
            else:
                # b側を代表へ
                end_px = pxs[-1] if pxs else tuple(map(int, graph.nodes[b0].coord))
                if end_px in blob_pixels and rep_coord in blob_pixels:
                    path_end_to_rep = _bfs_path(blob_pixels, end_px, rep_coord)  # end -> rep
                    if len(path_end_to_rep) >= 2:
                        # end_px を重複させないで末尾へ
                        pxs = pxs + path_end_to_rep[1:]
                seg.end_node = rep_nid

            seg.pixels = pxs
            seg.length_px = _path_len(pxs)

    # 最後に degree を SEGMENT だけで再計算して、NodeKind.ENDPOINT/JUNCTION を整える
    # （JUNCTION_ELEMENTは統計対象外なのでdegree/kind整形は不要）
    deg: Dict[int, int] = {int(nid): 0 for nid, n in graph.nodes.items() if n.kind != NodeKind.JUNCTION_ELEMENT}

    for seg in graph.segments.values():
        if seg.kind != SegmentKind.SEGMENT:
            continue
        a = int(seg.start_node)
        b = int(seg.end_node)
        if a in deg:
            deg[a] += 1
        if b in deg:
            deg[b] += 1

    for nid, d in deg.items():
        n = graph.nodes[nid]
        n.degree = int(d)
        n.kind = NodeKind.ENDPOINT if int(d) <= 1 else NodeKind.JUNCTION

    return graph
