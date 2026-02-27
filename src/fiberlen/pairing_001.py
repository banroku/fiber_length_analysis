# src/fiberlen/pairing.py

from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import math
import numpy as np

from fiberlen.types import CompressedGraph, NodeKind, Segment, SegmentKind, Pixel


def pairing(
    graph_kink_cut: CompressedGraph,
    pairing_angle_max: float,
    pairing_length_for_calc_angle: int,
) -> CompressedGraph:
    """
    交点（3本以上がつながるノード）に対して、直進関係にあるセグメント同士をペアリングする。
    （JUNCTION_ELEMENT / SegmentKind.JUNCTION_ELEMENT は対象外）
    """
    g = graph_kink_cut

    angle_max = float(pairing_angle_max)
    probe = int(pairing_length_for_calc_angle)
    if probe <= 0:
        probe = 1

    # ノード -> 接続セグメント一覧（SEGMENTのみ）
    incident: Dict[int, List[int]] = _build_incident_map(g)

    pair_map: Dict[Tuple[int, int], int] = {}

    for nid, node in g.nodes.items():
        # ★JUNCTIONのみ対象（JUNCTION_ELEMENTは除外）
        if node.kind != NodeKind.JUNCTION:
            continue

        # ★degree>=3 の junction のみ
        if int(getattr(node, "degree", 0)) < 3:
            continue

        seg_ids = incident.get(nid, [])
        if len(seg_ids) < 3:
            continue

        node_rc = node.coord

        # 各セグメントの outward unit vector を計算（SEGMENTのみ）
        vecs: Dict[int, np.ndarray] = {}
        for sid in seg_ids:
            seg = g.segments.get(sid)
            if seg is None:
                continue
            if seg.kind != SegmentKind.SEGMENT:
                continue
            v = _outward_unit_vector(seg, node_rc, probe)
            if v is None:
                continue
            vecs[sid] = v

        if len(vecs) < 2:
            continue

        # 候補ペアを列挙（foldが小さいほど直進）
        candidates: List[Tuple[float, int, int]] = []
        ids = sorted(vecs.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a = ids[i]
                b = ids[j]
                fold = _fold_angle_deg(vecs[a], vecs[b])
                if fold <= angle_max:
                    candidates.append((fold, a, b))

        if not candidates:
            continue

        # 直進性が高い順に貪欲確定
        candidates.sort(key=lambda x: x[0])
        used: set[int] = set()
        for fold, a, b in candidates:
            if a in used or b in used:
                continue
            used.add(a)
            used.add(b)
            pair_map[(nid, a)] = b
            pair_map[(nid, b)] = a

    setattr(g, "pair_map", pair_map)
    return g


# --------------------- helpers ---------------------


def _build_incident_map(g: CompressedGraph) -> Dict[int, List[int]]:
    incident: Dict[int, List[int]] = {nid: [] for nid in g.nodes.keys()}
    for sid, seg in g.segments.items():
        # ★SEGMENTのみ登録（blob要素を混ぜない）
        if seg.kind != SegmentKind.SEGMENT:
            continue
        incident.setdefault(seg.start_node, []).append(sid)
        incident.setdefault(seg.end_node, []).append(sid)
    return incident


def _outward_unit_vector(seg: Segment, node_rc: Pixel, probe_len_px: int) -> Optional[np.ndarray]:
    pix = seg.pixels
    if len(pix) < 2:
        return None

    nr, nc = int(node_rc[0]), int(node_rc[1])

    d0 = (pix[0][0] - nr) ** 2 + (pix[0][1] - nc) ** 2
    d1 = (pix[-1][0] - nr) ** 2 + (pix[-1][1] - nc) ** 2

    if d0 <= d1:
        start_idx = 0
        step = +1
    else:
        start_idx = len(pix) - 1
        step = -1

    r0, c0 = pix[start_idx]

    acc = 0.0
    i = start_idx
    while True:
        j = i + step
        if j < 0 or j >= len(pix):
            break
        r1, c1 = pix[j]
        dr = abs(int(r1) - int(r0))
        dc = abs(int(c1) - int(c0))
        acc += 1.4142135623730951 if (dr == 1 and dc == 1) else 1.0

        r0, c0 = r1, c1
        i = j

        if acc >= float(probe_len_px):
            break

    rs, cs = pix[start_idx]
    rt, ct = pix[i]
    v = np.array([float(rt - rs), float(ct - cs)], dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return None
    return v / n


def _fold_angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    uu = u / max(float(np.linalg.norm(u)), 1e-12)
    vv = v / max(float(np.linalg.norm(v)), 1e-12)

    d = float(np.clip(np.dot(uu, -vv), -1.0, 1.0))
    return float(math.degrees(math.acos(d)))
