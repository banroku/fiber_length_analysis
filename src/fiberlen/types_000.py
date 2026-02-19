# src/fiberlen/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional, Set, Any


Pixel = Tuple[int, int]          # (row, col)
Vector2 = Tuple[float, float]    # (dy, dx)


class NodeKind(str, Enum):
    ENDPOINT = "endpoint"
    JUNCTION = "junction"


@dataclass(frozen=False)
class Node:
    """
    Compressed graph node.

    coord: representative pixel coordinate (row, col)
    kind: endpoint / junction
    degree: degree in the original pixel graph (8-neighborhood)
    """
    node_id: int
    coord: Pixel
    kind: NodeKind
    degree: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": int(self.node_id),
            "coord": [int(self.coord[0]), int(self.coord[1])],
            "kind": str(self.kind.value),
            "degree": int(self.degree),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Node":
        return Node(
            node_id=int(d["node_id"]),
            coord=(int(d["coord"][0]), int(d["coord"][1])),
            kind=NodeKind(str(d["kind"])),
            degree=int(d["degree"]),
        )


@dataclass
class Segment:
    """
    Segment is an ordered polyline between two Nodes following skeleton pixels.

    pixels: ordered (row, col) coordinates from start_node -> end_node, inclusive
    length_px: computed (8-neighborhood weighted length), stored here
    touches_border: whether this segment is within border_margin_px of the image edge
    is_pruned: optional flag (if later you want to keep-but-ignore segments)
    """
    seg_id: int
    start_node: int
    end_node: int
    pixels: List[Pixel]

    length_px: Optional[float] = None
    touches_border: bool = False
    is_pruned: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seg_id": int(self.seg_id),
            "start_node": int(self.start_node),
            "end_node": int(self.end_node),
            "pixels": [[int(r), int(c)] for (r, c) in self.pixels],
            "length_px": None if self.length_px is None else float(self.length_px),
            "touches_border": bool(self.touches_border),
            "is_pruned": bool(self.is_pruned),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Segment":
        return Segment(
            seg_id=int(d["seg_id"]),
            start_node=int(d["start_node"]),
            end_node=int(d["end_node"]),
            pixels=[(int(rc[0]), int(rc[1])) for rc in d["pixels"]],
            length_px=None if d.get("length_px") is None else float(d["length_px"]),
            touches_border=bool(d.get("touches_border", False)),
            is_pruned=bool(d.get("is_pruned", False)),
        )


@dataclass
class CompressedGraph:
    """
    Sparse (compressed) skeleton graph.

    nodes: node_id -> Node
    segments: seg_id -> Segment
    adjacency: node_id -> set(seg_id)

    pair_map:
      pairing() writes decisions into this dictionary.
      Key:   (junction_node_id, incoming_seg_id)
      Value: outgoing_seg_id

      This keeps pairing information INSIDE graph (your requirement),
      so later steps (measure_length / overlays) only need "graph".
    """
    nodes: Dict[int, Node] = field(default_factory=dict)
    segments: Dict[int, Segment] = field(default_factory=dict)
    adjacency: Dict[int, Set[int]] = field(default_factory=dict)

    # IMPORTANT: pairing results live here (inside graph)
    pair_map: Dict[Tuple[int, int], int] = field(default_factory=dict)

    # ---- basic graph ops -------------------------------------------------
    def add_node(self, node: Node) -> None:
        self.nodes[node.node_id] = node
        self.adjacency.setdefault(node.node_id, set())

    def add_segment(self, seg: Segment) -> None:
        self.segments[seg.seg_id] = seg
        self.adjacency.setdefault(seg.start_node, set()).add(seg.seg_id)
        self.adjacency.setdefault(seg.end_node, set()).add(seg.seg_id)

    def incident_segments(self, node_id: int) -> List[int]:
        return sorted(self.adjacency.get(node_id, set()))

    def other_node(self, seg_id: int, node_id: int) -> int:
        seg = self.segments[seg_id]
        if seg.start_node == node_id:
            return seg.end_node
        if seg.end_node == node_id:
            return seg.start_node
        raise ValueError(f"Segment {seg_id} not incident to node {node_id}.")

    # ---- pairing helpers -------------------------------------------------
    def set_pair(self, junction_node_id: int, seg_a: int, seg_b: int) -> None:
        """
        Register a symmetric pairing at one junction:
          incoming seg_a -> outgoing seg_b
          incoming seg_b -> outgoing seg_a
        """
        self.pair_map[(int(junction_node_id), int(seg_a))] = int(seg_b)
        self.pair_map[(int(junction_node_id), int(seg_b))] = int(seg_a)

    def get_paired_next(self, junction_node_id: int, incoming_seg_id: int) -> Optional[int]:
        return self.pair_map.get((int(junction_node_id), int(incoming_seg_id)))

    # ---- serialization ---------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        # pair_map keys are tuples -> store as list entries for JSON friendliness
        pair_list = [
            {"junction": int(j), "incoming": int(s_in), "outgoing": int(s_out)}
            for (j, s_in), s_out in self.pair_map.items()
        ]
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "segments": [s.to_dict() for s in self.segments.values()],
            "adjacency": {str(k): sorted(int(x) for x in v) for k, v in self.adjacency.items()},
            "pair_map": pair_list,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CompressedGraph":
        g = CompressedGraph()
        for nd in d.get("nodes", []):
            g.add_node(Node.from_dict(nd))
        for sd in d.get("segments", []):
            g.add_segment(Segment.from_dict(sd))

        # Rebuild adjacency if provided (optional)
        adj = d.get("adjacency")
        if isinstance(adj, dict):
            g.adjacency = {int(k): set(int(x) for x in v) for k, v in adj.items()}

        # Pair map
        g.pair_map = {}
        for item in d.get("pair_map", []):
            j = int(item["junction"])
            s_in = int(item["incoming"])
            s_out = int(item["outgoing"])
            g.pair_map[(j, s_in)] = s_out
        return g

    # ---- convenience alias (matches your wording "node_to_segments") -----
    @property
    def node_to_segments(self) -> Dict[int, Set[int]]:
        return self.adjacency


@dataclass
class FiberPath:
    """
    One reconstructed fiber after resolving junction pairings.

    seg_ids: ordered list of segments
    length_px: measured from segment lengths / pixel polylines
    touches_border: True if any segment touches border (as per Segment.touches_border)
    """
    fiber_id: int
    seg_ids: List[int]

    length_px: Optional[float] = None
    touches_border: bool = False

    # Optional, for overlay/debug (can be omitted to save memory)
    pixels: Optional[List[Pixel]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fiber_id": int(self.fiber_id),
            "seg_ids": [int(x) for x in self.seg_ids],
            "length_px": None if self.length_px is None else float(self.length_px),
            "touches_border": bool(self.touches_border),
            "pixels": None if self.pixels is None else [[int(r), int(c)] for (r, c) in self.pixels],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "FiberPath":
        px = d.get("pixels")
        return FiberPath(
            fiber_id=int(d["fiber_id"]),
            seg_ids=[int(x) for x in d["seg_ids"]],
            length_px=None if d.get("length_px") is None else float(d["length_px"]),
            touches_border=bool(d.get("touches_border", False)),
            pixels=None if px is None else [(int(rc[0]), int(rc[1])) for rc in px],
        )


@dataclass(frozen=True)
class Scale:
    """
    Pixel-to-micrometer conversion.
    Internal processing uses pixels; UI/CSV uses this for μm conversion.
    """
    um_per_pix: float

    def to_dict(self) -> Dict[str, Any]:
        return {"um_per_pix": float(self.um_per_pix)}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Scale":
        return Scale(um_per_pix=float(d["um_per_pix"]))


@dataclass
class Fiber:
    """
    1本の繊維（グラフ上で連結したセグメント列）を表す。

    設計上の前提
    ----------
    ・内部単位は pixel
    ・measure_length() は最低限 (fiber_id, seg_ids, length_px) を使う
    """
    fiber_id: int
    seg_ids: List[int]
    length_px: float
