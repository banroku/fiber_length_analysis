# src/fiberlen/pairing.py

from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Set

import math
import numpy as np

from fiberlen.types import CompressedGraph, NodeKind, Segment, SegmentKind, Pixel


def pairing(
    graph_kink_cut: CompressedGraph,
    pairing_angle_max: float,
    pairing_length_for_calc_angle: int,
) -> CompressedGraph:
    """
    既存仕様：
      交点（3本以上がつながるノード）に対して、直進関係にあるセグメント同士をペアリングする。
      （JUNCTION_ELEMENT / SegmentKind.JUNCTION_ELEMENT は対象外）

    追加仕様（今回の要望）：
      枝落とし後に「2セグメントがつながったノード」が発生するため、
      そのノードに接続する2本のセグメントが直進（角度条件）なら “接合（merge）” して
      1本のセグメントに連結する処理を追加する。
      判定は pairing_angle_max をそのまま使用する。

    注意：
      Streamlitの後段（measure_length等）で graph.adjacency を使う前提に合わせて、
      接合時は segments / adjacency / nodes を整合する。
    """
    g = graph_kink_cut

    angle_max = float(pairing_angle_max)
    probe = int(pairing_length_for_calc_angle)
    if probe <= 0:
        probe = 1

    # ---- (A) 追加：degree==2 ノードの「接合」処理 --------------------------------
    _merge_straight_degree2_nodes(g, angle_max=angle_max, probe_len_px=probe)

    # ---- (B) 既存：junction(deg>=3) の pairing -----------------------------------
    # ノード -> 接続セグメント一覧（SEGMENTのみ）
    incident: Dict[int, List[int]] = _build_incident_map(g)

    pair_map: Dict[Tuple[int, int], int] = {}

    for nid, node in g.nodes.items():
        # ★JUNCTIONのみ対象（JUNCTION_ELEMENTは除外）
        if node.kind != NodeKind.JUNCTION:
            continue

        seg_ids = incident.get(nid, [])
        # ★3本以上の junction のみ（node.degreeはpixel graph由来なので使わない）
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


# --------------------- merge (new) ---------------------


def _merge_straight_degree2_nodes(g: CompressedGraph, angle_max: float, probe_len_px: int) -> None:
    """
    degree==2（SEGMENTが2本だけ接続）ノードについて、
    2本が直進（fold<=angle_max）なら、セグメントを1本に接合する。

    変更点は g に in-place。
    """
    if probe_len_px <= 0:
        probe_len_px = 1

    # 連結で新たに degree2 が生まれる可能性があるので、収束するまで回す
    while True:
        incident = _build_incident_map(g)
        merged_any = False

        # ループ中に dict が変わるので候補 nid を固定
        cand_nids = list(g.nodes.keys())

        for nid in cand_nids:
            node = g.nodes.get(nid)
            if node is None:
                continue
            if node.kind == NodeKind.JUNCTION_ELEMENT:
                continue

            seg_ids = incident.get(nid, [])
            if len(seg_ids) != 2:
                continue

            s1_id, s2_id = seg_ids[0], seg_ids[1]
            s1 = g.segments.get(s1_id)
            s2 = g.segments.get(s2_id)
            if s1 is None or s2 is None:
                continue
            if s1.kind != SegmentKind.SEGMENT or s2.kind != SegmentKind.SEGMENT:
                continue

            # outward vectors at the degree-2 node
            v1 = _outward_unit_vector(s1, node.coord, probe_len_px)
            v2 = _outward_unit_vector(s2, node.coord, probe_len_px)
            if v1 is None or v2 is None:
                continue

            fold = _fold_angle_deg(v1, v2)
            if fold > float(angle_max):
                continue

            # ---- 接合する ----
            other1 = _other_node_of_segment(s1, nid)
            other2 = _other_node_of_segment(s2, nid)
            if other1 is None or other2 is None:
                continue
            if other1 == other2:
                # ループ（同一ノードへ戻る）などはここでは扱わない（安全側）
                continue

            pix1 = _pixels_other_to_node(s1, nid)  # other1 -> nid
            pix2 = _pixels_node_to_other(s2, nid)  # nid -> other2
            if not pix1 or not pix2:
                continue

            # node pixel が重複するので pix2 の先頭を落として連結
            merged_pixels = pix1 + pix2[1:]
            if len(merged_pixels) < 2:
                continue

            new_seg_id = _next_seg_id(g)
            new_seg = Segment(
                seg_id=new_seg_id,
                start_node=int(other1),
                end_node=int(other2),
                pixels=merged_pixels,
                length_px=_polyline_length_px(merged_pixels),
                touches_border=bool(getattr(s1, "touches_border", False) or getattr(s2, "touches_border", False)),
                is_pruned=False,
                kind=SegmentKind.SEGMENT,
            )

            # ① old segments を削除
            g.segments.pop(s1_id, None)
            g.segments.pop(s2_id, None)

            # ② adjacency を修正（削除→追加）
            for n in (nid, other1, other2):
                g.adjacency.setdefault(int(n), set())

            g.adjacency[int(nid)].discard(int(s1_id))
            g.adjacency[int(nid)].discard(int(s2_id))

            g.adjacency[int(other1)].discard(int(s1_id))
            g.adjacency[int(other1)].discard(int(s2_id))

            g.adjacency[int(other2)].discard(int(s1_id))
            g.adjacency[int(other2)].discard(int(s2_id))

            g.segments[int(new_seg_id)] = new_seg
            g.adjacency[int(other1)].add(int(new_seg_id))
            g.adjacency[int(other2)].add(int(new_seg_id))

            # nid は孤立するはずなので削除（残すと後段でノイズになりやすい）
            if len(g.adjacency.get(int(nid), set())) == 0:
                g.adjacency.pop(int(nid), None)
                g.nodes.pop(int(nid), None)

            # kind / degree の再計算（最小限）
            _refresh_node_kinds_from_adjacency(g, node_ids={int(other1), int(other2)})

            merged_any = True
            break  # incident を作り直す必要があるので break

        if not merged_any:
            break


def _refresh_node_kinds_from_adjacency(g: CompressedGraph, node_ids: Set[int]) -> None:
    for nid in node_ids:
        node = g.nodes.get(nid)
        if node is None:
            continue
        if node.kind == NodeKind.JUNCTION_ELEMENT:
            continue
        deg = len(g.adjacency.get(nid, set()))
        node.degree = int(deg)
        node.kind = NodeKind.ENDPOINT if deg <= 1 else NodeKind.JUNCTION


def _next_seg_id(g: CompressedGraph) -> int:
    if not g.segments:
        return 1
    return int(max(int(k) for k in g.segments.keys()) + 1)


def _other_node_of_segment(seg: Segment, node_id: int) -> Optional[int]:
    if int(seg.start_node) == int(node_id):
        return int(seg.end_node)
    if int(seg.end_node) == int(node_id):
        return int(seg.start_node)
    return None


def _pixels_other_to_node(seg: Segment, node_id: int) -> List[Pixel]:
    """
    seg.pixels を「(other end) -> node_id」向きにして返す（両端含む）
    """
    if int(seg.end_node) == int(node_id):
        # start -> end (=node) なのでそのまま
        return list(seg.pixels)
    if int(seg.start_node) == int(node_id):
        # node -> end なので逆向き
        return list(reversed(seg.pixels))
    return []


def _pixels_node_to_other(seg: Segment, node_id: int) -> List[Pixel]:
    """
    seg.pixels を「node_id -> (other end)」向きにして返す（両端含む）
    """
    if int(seg.start_node) == int(node_id):
        return list(seg.pixels)
    if int(seg.end_node) == int(node_id):
        return list(reversed(seg.pixels))
    return []


def _polyline_length_px(pixels: List[Pixel]) -> float:
    """
    8近傍の重み（縦横=1, 斜め=sqrt(2)）で polyline 長さを計算
    """
    if len(pixels) < 2:
        return 0.0
    acc = 0.0
    for (r0, c0), (r1, c1) in zip(pixels[:-1], pixels[1:]):
        dr = abs(int(r1) - int(r0))
        dc = abs(int(c1) - int(c0))
        acc += 1.4142135623730951 if (dr == 1 and dc == 1) else 1.0
    return float(acc)


# --------------------- helpers (existing) ---------------------


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
