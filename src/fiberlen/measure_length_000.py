# Path: src/fiberlen/measure_length.py

from __future__ import annotations

from typing import Dict, List, Set, Tuple, Optional

import numpy as np

from fiberlen.types import CompressedGraph, Fiber, Pixel


def measure_length(graph: CompressedGraph, top_cut: int) -> List[Fiber]:
    """
    セグメント長さ測定（= 繊維トレースして Fiber を作る）

    仕様（あなたの設計どおり）
    ------------------------
    ・入力は pairing 後の graph（graph.segments を含み、必要なら graph.pair_map を含む）
    ・内部単位は pixel
    ・返す Fiber は長い順にソート
    ・top_cut > 0 の場合は上位 top_cut 本だけ返す（0以下なら全件）

    重要な前提
    ----------
    ・pairing は「graph.pair_map[(junction_id, incoming_seg_id)] = outgoing_seg_id」
      を作ってあるものとする。
    ・pair_map が無い/該当しない junction では、その junction で繊維は途切れる。

    Returns
    -------
    fibers : List[Fiber]
    """

    # pair_map は無いこともある（その場合は junction で途切れるだけ）
    pair_map: Dict[Tuple[int, int], int] = getattr(graph, "pair_map", {}) or {}

    # incident map：node_id -> seg_ids
    incident: Dict[int, List[int]] = _build_incident_map(graph)

    used_segments: Set[int] = set()
    fibers: List[Fiber] = []
    fid = 1

    # 端点（degree==1）のノードから開始してトレース
    endpoint_nodes = [nid for nid, n in graph.nodes.items() if int(getattr(n, "degree", 0)) == 1]

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

    # 次に、閉ループなど（端点が無い成分）を拾う
    for sid in sorted(graph.segments.keys()):
        if sid in used_segments:
            continue
        # このセグメントから適当に始める（start_node 側から）
        seg = graph.segments[sid]
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
        f.length_px = float(sum(graph.segments[s].length_px for s in f.seg_ids if s in graph.segments))

    fibers.sort(key=lambda x: x.length_px, reverse=True)

    tc = int(top_cut)
    if tc > 0:
        fibers = fibers[:tc]

    return fibers


# ------------------------ helpers ------------------------


def _build_incident_map(g: CompressedGraph) -> Dict[int, List[int]]:
    incident: Dict[int, List[int]] = {nid: [] for nid in g.nodes.keys()}
    for sid, seg in g.segments.items():
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
    """
    1本分をトレースして Fiber を返す。
    start_node から start_seg に入るところから開始する。

    トレース規則
    -----------
    ・現在セグメント sid を node_id から出て other_node に到達
    ・到達先が junction の場合：pair_map[(junction, sid)] があればその seg に乗り換えて継続
      なければ終了（その junction で途切れる）
    ・到達先が endpoint の場合：終了
    ・安全のため、同一セグメントの再訪を防ぐ（ループ対策）
    """
    if start_seg not in g.segments:
        return None

    seg_ids: List[int] = []
    visited_local: Set[int] = set()

    node_id = int(start_node)
    sid = int(start_seg)

    while True:
        if sid in visited_local:
            # ループに入ったので終了
            break
        if sid not in g.segments:
            break

        visited_local.add(sid)
        seg_ids.append(sid)

        seg = g.segments[sid]
        next_node = _other_node(seg, node_id)

        # 次ノードが存在しない（不整合）なら終了
        if next_node not in g.nodes:
            break

        deg = int(getattr(g.nodes[next_node], "degree", 0))

        if deg <= 1:
            # endpoint（または孤立）で終了
            break

        if deg >= 3:
            # junction：pair_map に従って進む
            nxt = pair_map.get((next_node, sid), None)
            if nxt is None:
                break
            # 交差点で “入ってきたセグメント sid” に対する “出るセグメント nxt”
            node_id = next_node
            sid = int(nxt)
            continue

        # deg == 2：鎖は “もう一方のセグメント” へ自動で継続
        # （pairing不要）
        cands = [x for x in incident.get(next_node, []) if x != sid]
        if not cands:
            break
        node_id = next_node
        sid = int(cands[0])

    if not seg_ids:
        return None

    # Fiber は types.py 側に合わせる（最低限 fiber_id/seg_ids/length_px を想定）
    f = Fiber(fiber_id=-1, seg_ids=seg_ids, length_px=0.0)
    return f
