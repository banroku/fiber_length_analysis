# Path: src/fiberlen/pairing.py

from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import math
import numpy as np

from fiberlen.types import CompressedGraph, NodeKind, Segment, Pixel


def pairing(
    graph_kink_cut: CompressedGraph,
    pairing_angle_max: float,
    pairing_length_for_calc_angle: int,
) -> CompressedGraph:
    """
    交点（3本以上がつながるノード）に対して、直進関係にあるセグメント同士をペアリングする。

    仕様（あなたの設計どおり）
    ------------------------
    ・対象は degree>=3 のノード（junction）
    ・各セグメントの方向は「ノードから pairing_length_for_calc_angle px 離れた点」へのベクトルで評価
    ・2セグメント間の「折れ曲がり角度」が pairing_angle_max(°) 以下なら接続（ペア）とみなす
    ・結果は graph に格納する（graph.pair_map を追加/更新）
      pair_map[(node_id, seg_id)] = paired_seg_id

    注意
    ----
    ・この処理は「セグメント自体を結合」しません。トレース時に junction を跨いで進むための
      “通過ルール” を与えるだけです（graphの一貫性を保つため）。
    """

    g = graph_kink_cut

    angle_max = float(pairing_angle_max)
    probe = int(pairing_length_for_calc_angle)
    if probe <= 0:
        probe = 1

    # ノード -> 接続セグメント一覧
    incident: Dict[int, List[int]] = _build_incident_map(g)

    pair_map: Dict[Tuple[int, int], int] = {}

    for nid, node in g.nodes.items():
        # junction判定：kind か degree で見る（どちらかでOK）
        if getattr(node, "kind", None) is not None:
            if node.kind != NodeKind.JUNCTION:
                continue
        else:
            if int(getattr(node, "degree", 0)) < 3:
                continue

        seg_ids = incident.get(nid, [])
        if len(seg_ids) < 3:
            continue

        node_rc = node.coord

        # 各セグメントの outward unit vector を計算
        vecs: Dict[int, np.ndarray] = {}
        for sid in seg_ids:
            seg = g.segments.get(sid)
            if seg is None:
                continue
            v = _outward_unit_vector(seg, node_rc, probe)
            if v is None:
                continue
            vecs[sid] = v

        if len(vecs) < 2:
            continue

        # 候補ペアを列挙（折れ曲がり角度が小さいほど「直進に近い」）
        candidates: List[Tuple[float, int, int]] = []
        ids = sorted(vecs.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a = ids[i]
                b = ids[j]
                fold = _fold_angle_deg(vecs[a], vecs[b])  # 0が直進、90がT、180がUターン
                if fold <= angle_max:
                    candidates.append((fold, a, b))

        if not candidates:
            continue

        # 直進性が高い順（foldが小さい順）に貪欲にペアを確定
        candidates.sort(key=lambda x: x[0])
        used: set[int] = set()
        for fold, a, b in candidates:
            if a in used or b in used:
                continue
            used.add(a)
            used.add(b)
            pair_map[(nid, a)] = b
            pair_map[(nid, b)] = a

    # graph に格納（既存があれば上書き）
    setattr(g, "pair_map", pair_map)
    return g


# --------------------- helpers ---------------------


def _build_incident_map(g: CompressedGraph) -> Dict[int, List[int]]:
    incident: Dict[int, List[int]] = {nid: [] for nid in g.nodes.keys()}
    for sid, seg in g.segments.items():
        incident.setdefault(seg.start_node, []).append(sid)
        incident.setdefault(seg.end_node, []).append(sid)
    return incident


def _outward_unit_vector(seg: Segment, node_rc: Pixel, probe_len_px: int) -> Optional[np.ndarray]:
    """
    ノードからセグメント外側へ向かう単位ベクトルを返す。
    probe_len_px 分だけ polyline 上を進んだ点を使う。
    """
    pix = seg.pixels
    if len(pix) < 2:
        return None

    nr, nc = int(node_rc[0]), int(node_rc[1])

    # ノードが pixels の端にいる想定だが、ズレても耐えるように「端点に近い側」を採用する
    d0 = (pix[0][0] - nr) ** 2 + (pix[0][1] - nc) ** 2
    d1 = (pix[-1][0] - nr) ** 2 + (pix[-1][1] - nc) ** 2

    if d0 <= d1:
        # node は pix[0] 側
        start_idx = 0
        step = +1
    else:
        # node は pix[-1] 側
        start_idx = len(pix) - 1
        step = -1

    # 開始点（ノード座標が端点と一致しない場合でも、端点を「ノード位置」として扱う）
    r0, c0 = pix[start_idx]

    # probe_len_px だけ進んだ点を探す
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

        # 更新
        r0, c0 = r1, c1
        i = j

        if acc >= float(probe_len_px):
            break

    # node端点（pix[start_idx]）→ probe点（pix[i]）のベクトル
    rs, cs = pix[start_idx]
    rt, ct = pix[i]
    v = np.array([float(rt - rs), float(ct - cs)], dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return None
    return v / n


def _fold_angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    """
    折れ曲がり角度（度）を返す。

    定義：
    ノードで「直進」する2本は、ノードから外向きのベクトルが互いに反対向き（180°）になる。
    そこで fold_angle = angle(u, -v) とする。

    直進: 0°
    T字: 90° 付近
    Uターン: 180°（同方向）
    """
    uu = u / max(float(np.linalg.norm(u)), 1e-12)
    vv = v / max(float(np.linalg.norm(v)), 1e-12)

    d = float(np.clip(np.dot(uu, -vv), -1.0, 1.0))
    return float(math.degrees(math.acos(d)))
