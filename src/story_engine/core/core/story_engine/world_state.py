from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


@dataclass
class WorldState:
    """Lightweight world/continuity registry for planning + checks.

    - facts: canonical booleans/enums (e.g., "Passover": true)
    - relationships: nested dict: {"A->B": {trust: -1..1, affection: -1..1, respect: -1..1}}
    - timeline: list of events: {time: ISO/date/label, desc: str}
    - availability: {"character_id": {locations: [..], status: "present|away|unknown"}}
    - locations: {name: {status, notes}}
    - props: {name: {location, status}}
    """

    facts: Dict[str, Any] = field(default_factory=dict)
    relationships: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    availability: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    locations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    props: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "WorldState":
        return WorldState(
            facts=d.get("facts", {}),
            relationships=d.get("relationships", {}),
            timeline=d.get("timeline", []),
            availability=d.get("availability", {}),
            locations=d.get("locations", {}),
            props=d.get("props", {}),
        )

    def merge(self, other: "WorldState") -> "WorldState":
        base = self.to_dict()
        od = other.to_dict()
        for k in base.keys():
            if isinstance(base[k], dict):
                base[k].update(od.get(k, {}))
            elif isinstance(base[k], list):
                base[k] = base[k] + [x for x in od.get(k, [])]
            else:
                base[k] = od.get(k, base[k])
        return WorldState.from_dict(base)

    # Convenience updates
    def set_fact(self, key: str, value: Any):
        self.facts[key] = value

    def set_relationship(self, src: str, dst: str, **kwargs):
        key = f"{src}->{dst}"
        rel = self.relationships.get(key, {})
        rel.update(kwargs)
        self.relationships[key] = rel
