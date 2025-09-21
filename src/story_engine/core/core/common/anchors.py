from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


def load_anchors(path: str | Path = "decisions.yaml") -> Dict[str, Any]:
    p = Path(path)
    if not p.exists() or yaml is None:
        return {}
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return data.get("anchors") or {}


def compute_decisions_id(anchors: Dict[str, Any]) -> str:
    try:
        s = json.dumps(anchors or {}, sort_keys=True, ensure_ascii=False)
    except Exception:
        s = str(anchors)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


class AnchorsValidator:
    def __init__(self, anchors: Dict[str, Any]):
        self.anchors = anchors or {}
        self.forbidden: List[str] = list(
            (self.anchors.get("lexicon_policy") or {}).get("forbidden") or []
        )

    def validate_text(self, text: str) -> List[str]:
        if not text or not isinstance(text, str):
            return []
        t = text.lower()
        hits = []
        for w in self.forbidden:
            w2 = str(w).strip().lower()
            if not w2:
                continue
            if w2 in t:
                hits.append(w)
        return hits
