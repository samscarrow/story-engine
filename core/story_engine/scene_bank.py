"""
Scene Bank utility for storing and retrieving pre-authored scenes.
Parses screenplay-like text into discrete scenes and provides lookup/search.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any


_SCENE_HEADER_RE = re.compile(r"^(COLD OPEN:?|ACT\s+\w+:|INT\.|EXT\.|INT/EXT\.)", re.IGNORECASE)


def _strip_md(line: str) -> str:
    """Lightweight Markdown cleanup for parsing screenplay text.

    - Remove code fences
    - Strip heading markers (#, ##, ###)
    - Remove common emphasis markers (** __ * _ `)
    - Keep scene headings like INT./EXT. intact
    """
    s = line.rstrip("\n\r")
    # Drop code fence lines entirely
    if s.strip().startswith("```"):
        return ""
    # Strip heading markers at line start
    s = re.sub(r"^\s*#{1,6}\s*", "", s)
    # Remove emphasis/backtick markers (leave inner text)
    s = s.replace("**", "").replace("__", "")
    s = s.replace("`", "")
    # Trim leftover single * or _ pairs conservatively
    s = s.replace("*", "").replace("_", "")
    return s.strip()


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9\s-]", "", text).strip().lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s[:80]


@dataclass
class SceneEntry:
    id: str
    title: str
    header: str
    act: Optional[str]
    body: str
    tags: List[str]
    source: str

    def summary(self, max_chars: int = 240) -> str:
        text = self.body.strip().replace("\n", " ")
        return (text[: max_chars - 3] + "...") if len(text) > max_chars else text


class SceneBank:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.scenes: Dict[str, SceneEntry] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.scenes = {s["id"]: SceneEntry(**s) for s in data.get("scenes", [])}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"scenes": [asdict(s) for s in self.scenes.values()]}, f, indent=2, ensure_ascii=False)

    def add(self, scene: SceneEntry) -> None:
        self.scenes[scene.id] = scene

    def get(self, scene_id_or_slug: str) -> Optional[SceneEntry]:
        key = scene_id_or_slug
        if key in self.scenes:
            return self.scenes[key]
        # fallback by slug match on title
        slug = _slugify(scene_id_or_slug)
        for s in self.scenes.values():
            if _slugify(s.title) == slug:
                return s
        return None

    def search(self, query: str) -> List[SceneEntry]:
        q = query.strip().lower()
        out = []
        for s in self.scenes.values():
            if q in s.title.lower() or q in s.body.lower():
                out.append(s)
        return out

    def list(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": s.id,
                "title": s.title,
                "act": s.act,
                "tags": s.tags,
                "summary": s.summary(),
            }
            for s in sorted(self.scenes.values(), key=lambda x: x.id)
        ]


def parse_screenplay_to_scenes(text: str, source: str) -> List[SceneEntry]:
    # Preprocess markdown to make headings consistent
    raw_lines = text.splitlines()
    lines = [_strip_md(l) for l in raw_lines]
    scenes: List[SceneEntry] = []
    current_title: Optional[str] = None
    current_header: Optional[str] = None
    current_act: Optional[str] = None
    buf: List[str] = []

    def _flush():
        nonlocal scenes, current_title, current_header, current_act, buf
        if current_header and buf:
            title = current_title or current_header
            scene_id = _slugify(f"{current_act or 'act'}-{title}")
            body = "\n".join(buf).strip()
            tags: List[str] = []
            if current_header.upper().startswith("INT") or current_header.upper().startswith("EXT"):
                tags.append("screenplay-scene")
            if current_act:
                tags.append(current_act.lower())
            scenes.append(SceneEntry(
                id=scene_id,
                title=title.strip(),
                header=(current_header or "").strip(),
                act=current_act,
                body=body,
                tags=tags,
                source=source,
            ))
        buf = []

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            buf.append("")
            continue
        if line.strip("=").strip() == "":
            # separator line of ====
            _flush()
            continue
        # Normalize COLD OPEN variants
        if line.upper().startswith("COLD OPEN") and not line.upper().startswith("COLD OPEN:"):
            line = "COLD OPEN: " + line[len("COLD OPEN"):].strip()

        if _SCENE_HEADER_RE.match(line):
            # Handle act lines separately to carry act context
            if line.upper().startswith("ACT "):
                # flush any previous scene buffer
                _flush()
                current_act = line.strip()
                current_title = None
                current_header = line.strip()
                # Not a scene body itself; continue
                continue
            # New scene header
            _flush()
            current_header = line.strip()
            # Use prior non-empty line as title if it looks like a labeled section
            current_title = None
            buf.append(line)
        else:
            # capture potential titled sections like COLD OPEN: ...
            if line.upper().startswith("COLD OPEN:"):
                _flush()
                current_title = line.strip()
                current_header = line.strip()
                buf.append(line)
            else:
                buf.append(line)

    _flush()
    return scenes
