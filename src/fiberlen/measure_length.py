# src/fiberlen/measure_length.py

from __future__ import annotations

from typing import Dict, List, Set, Tuple, Optional

import numpy as np

from fiberlen.types import CompressedGraph, Fiber
from fiberlen.types import NodeKind, SegmentKind  # ★追加


def measure_length(graph: CompressedGraph, top_cut: int) -> List[Fiber]:
    """
    セグメント長さ測定（= 繊維トレースして Fiber を作る）

    修正点
    ------
    ・NodeKind.JUNCTION_ELEMENT は統計対象外として無視
    ・SegmentKind.SEGMENT のみトレース対象（JUNCTION_ELEMENT等は無視）
    ・incident map も SEGMENT のみで構築
    """
    pair_map: Dict[Tuple[int, int], int] = getattr(graph, "pair_map", {}) or {}

    incident: Dict[int, List[int]] = _build_incident_map(graph)

    used_segments: Set[int] = set()
    fibers: List[Fiber] = []
    fid = 1

    # 端点（degree==1）かつ blob要素でないノードだけ
    endpoint_nodes = [
        nid for nid, n in graph.nodes.items()
        if getattr(n, "kind", None) != NodeKind.JUNCTION_ELEMENT
        and int(getattr(n, "degree", 0)) == 1
    ]

    for start_node in endpoint_nodes:
        for start_seg in incident.get(start_node, []):
            if start_seg in used_segments:
                continue
            f = _trace_one_fiber(graph, incident, pair_map, start_node, start_seg)
            if f is None:
                continue
            for sid in f.seg_ids:
                used_segments.add(sid)
            f.fiber_id = fid
            fid += 1
            fibers.append(f)

    # 閉ループ等（端点が無い成分）：SEGMENT のみを起点に拾う
    for sid in sorted(graph.segments.keys()):
        seg = graph.segments[sid]
        if getattr(seg, "kind", SegmentKind.SEGMENT) != SegmentKind.SEGMENT:
            continue
        if sid in used_segments:
            continue
        f = _trace_one_fiber(graph, incident, pair_map, seg.start_node, sid)
        if f is None:
            continue
        for s2 in f.seg_ids:
            used_segments.add(s2)
        f.fiber_id = fid
        fid += 1
        fibers.append(f)

    # length計算、ソート、top_cut
    for f in fibers:
        f.length_px = float(
            sum(
                graph.segments[s].length_px
                for s in f.seg_ids
                if (s in graph.segments)
                and getattr(graph.segments[s], "kind", SegmentKind.SEGMENT) == SegmentKind.SEGMENT
                and getattr(graph.segments[s], "length_px", None) is not None
            )
            - top_cut * 2  # 繊維両末端が長めに計測される分を削る
        )

    fibers.sort(key=lambda x: x.length_px, reverse=True)

#    tc = int(top_cut)  # 誤った実装。挙動が問題なければ削除する
#    if tc > 0:
#        fibers = fibers[:tc]

    return fibers


# ------------------------ helpers ------------------------


def _build_incident_map(g: CompressedGraph) -> Dict[int, List[int]]:
    incident: Dict[int, List[int]] = {nid: [] for nid in g.nodes.keys()}
    for sid, seg in g.segments.items():
        if getattr(seg, "kind", SegmentKind.SEGMENT) != SegmentKind.SEGMENT:
            continue
        incident.setdefault(seg.start_node, []).append(sid)
        incident.setdefault(seg.end_node, []).append(sid)
    return incident


def _other_node(seg, node_id: int) -> int:
    if seg.start_node == node_id:
        return seg.end_node
    return seg.start_node


def _trace_one_fiber(
    g: CompressedGraph,
    incident: Dict[int, List[int]],
    pair_map: Dict[Tuple[int, int], int],
    start_node: int,
    start_seg: int,
) -> Optional[Fiber]:
    if start_seg not in g.segments:
        return None

    # ★開始セグメントがSEGMENT以外なら無視
    if getattr(g.segments[start_seg], "kind", SegmentKind.SEGMENT) != SegmentKind.SEGMENT:
        return None

    seg_ids: List[int] = []
    visited_local: Set[int] = set()

    node_id = int(start_node)
    sid = int(start_seg)

    while True:
        if sid in visited_local:
            break
        if sid not in g.segments:
            break

        seg = g.segments[sid]

        # ★途中でSEGMENT以外に入ったら終了
        if getattr(seg, "kind", SegmentKind.SEGMENT) != SegmentKind.SEGMENT:
            break

        visited_local.add(sid)
        seg_ids.append(sid)

        next_node = _other_node(seg, node_id)

        if next_node not in g.nodes:
            break

        # ★blob要素ノードに入ったら終了（統計対象外）
        if getattr(g.nodes[next_node], "kind", None) == NodeKind.JUNCTION_ELEMENT:
            break

        deg = int(getattr(g.nodes[next_node], "degree", 0))

        if deg <= 1:
            break

        if deg >= 3:
            nxt = pair_map.get((next_node, sid), None)
            if nxt is None:
                break
            node_id = next_node
            sid = int(nxt)
            continue

        cands = [x for x in incident.get(next_node, []) if x != sid]
        if not cands:
            break
        node_id = next_node
        sid = int(cands[0])

    if not seg_ids:
        return None

    return Fiber(fiber_id=-1, seg_ids=seg_ids, length_px=0.0)
