# src/fiberlen/trim_graph.py
from __future__ import annotations

import copy
from typing import Set

from fiberlen.types import CompressedGraph, NodeKind


def trim_graph(graph: CompressedGraph, trim_length: float) -> CompressedGraph:
    """
    グラフ化後の「短すぎる枝（片側だけ末端）」を除去する。

    定義（要件）
    ・対象は「セグメントのうち、片方だけが末端（endpoint）で、もう片方は末端ではない」もの
    ・さらに length_px < trim_length のものだけ除去

    実施手順（二段階）
    ① graph_trimmed.segments から対象セグメントを直接削除
    ② 削除したセグメントが接続していたノードの接続情報（graph_trimmed.adjacency）を修正

    注意
    ・元の graph は改変しない（deep copyして処理）
    ・pair_map はこの段階では未作成の前提のため触らない
    """
    graph_trimmed: CompressedGraph = copy.deepcopy(graph)
    trim_length = float(trim_length)

    if trim_length <= 0:
        return graph_trimmed

    # endpoint 判定：Node.kind を優先し、補助的に adjacency の次数を使う
    # （Node.degree は元のpixel graph由来で、必ずしも adjacency と一致しない可能性があるため）
    def is_endpoint(node_id: int) -> bool:
        node = graph_trimmed.nodes.get(node_id)
        if node is None:
            return False
        if node.kind == NodeKind.ENDPOINT:
            return True
        return len(graph_trimmed.adjacency.get(node_id, set())) <= 1

    # 削除対象セグメントIDを確定（探索中に構造を壊さない）
    to_remove: Set[int] = set()

    for seg_id, seg in list(graph_trimmed.segments.items()):
        # セグメント長が無いものは対象外（安全側）
        if seg.length_px is None:
            continue
        if float(seg.length_px) >= trim_length:
            continue

        a = int(seg.start_node)
        b = int(seg.end_node)

        a_end = is_endpoint(a)
        b_end = is_endpoint(b)

        # 「片側だけ末端」
        if (a_end and not b_end) or (b_end and not a_end):
            to_remove.add(int(seg_id))

    if not to_remove:
        return graph_trimmed

    # ① セグメントを直接削除
    for seg_id in to_remove:
        graph_trimmed.segments.pop(seg_id, None)

    # ② adjacency（node_id -> set(seg_id)）を修正
    for node_id, incident in list(graph_trimmed.adjacency.items()):
        if not incident:
            continue
        new_incident = set(incident) - to_remove
        if new_incident != incident:
            graph_trimmed.adjacency[node_id] = new_incident

    # 参考：後段が adjacency を使うなら、degree/kind を合わせておくと安全
    for node_id, node in graph_trimmed.nodes.items():
        deg = len(graph_trimmed.adjacency.get(node_id, set()))
        node.degree = int(deg)
        # JUNCTION_ELEMENT を勝手に上書きしない
        if node.kind != NodeKind.JUNCTION_ELEMENT:
            node.kind = NodeKind.ENDPOINT if deg <= 1 else NodeKind.JUNCTION

    return graph_trimmed
