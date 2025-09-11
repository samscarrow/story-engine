from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class GraphNode:
    id: str
    type: str  # 'beat' | 'scene'
    data: Dict[str, Any]


@dataclass
class GraphEdge:
    src: str
    dst: str
    type: str  # 'cause' | 'raises_stakes' | 'value_shift' | 'payoff' | 'chronology'
    weight: float = 1.0
    data: Dict[str, Any] = field(default_factory=dict)


class NarrativeGraph:
    """Minimal narrative graph to map beats and scenes with causal/tension/value edges."""

    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []

    def add_beat(self, beat: Dict[str, Any], node_id: Optional[str] = None) -> str:
        nid = node_id or str(uuid.uuid4())
        self.nodes[nid] = GraphNode(id=nid, type="beat", data=beat)
        return nid

    def add_scene(self, plan: Dict[str, Any], node_id: Optional[str] = None) -> str:
        nid = node_id or str(uuid.uuid4())
        self.nodes[nid] = GraphNode(id=nid, type="scene", data=plan)
        return nid

    def link(
        self,
        src_id: str,
        dst_id: str,
        edge_type: str,
        weight: float = 1.0,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        if src_id not in self.nodes or dst_id not in self.nodes:
            raise ValueError("Invalid node id for edge")
        self.edges.append(
            GraphEdge(
                src=src_id, dst=dst_id, type=edge_type, weight=weight, data=data or {}
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [
                {"id": n.id, "type": n.type, "data": n.data}
                for n in self.nodes.values()
            ],
            "edges": [asdict(e) for e in self.edges],
        }

    # Persistence helpers using core.storage
    def persist(self, storage, workflow_name: str = "narrative_graph") -> None:
        payload = self.to_dict()
        storage.store_output(workflow_name, payload)
