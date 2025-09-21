from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .interfaces import ArtifactRepo, JobRepo, StoryRepo
from ..storage.database import SQLiteConnection, ensure_message_tables


class InMemoryArtifactRepo(ArtifactRepo):
    def __init__(self) -> None:
        self._store: Dict[tuple[str, str], Dict[str, Any]] = {}

    def save(self, kind: str, key: str, payload: Dict[str, Any]) -> None:
        self._store[(kind, key)] = dict(payload)

    def load(self, kind: str, key: str) -> Optional[Dict[str, Any]]:
        return self._store.get((kind, key))


class FileArtifactRepo(ArtifactRepo):
    """Stores artifacts as JSON files under a root directory.

    Useful for local development and CI where a DB may not be available.
    """

    def __init__(self, root: str | os.PathLike[str] = "artifacts") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, kind: str, key: str) -> Path:
        safe_kind = kind.replace("/", "_")
        safe_key = key.replace("/", "_")
        return self.root / safe_kind / f"{safe_key}.json"

    def save(self, kind: str, key: str, payload: Dict[str, Any]) -> None:
        p = self._path(kind, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load(self, kind: str, key: str) -> Optional[Dict[str, Any]]:
        p = self._path(kind, key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None


class InMemoryJobRepo(JobRepo):
    def __init__(self) -> None:
        self._seen: set[str] = set()

    def was_processed(self, idempotency_key: str) -> bool:
        return idempotency_key in self._seen

    def mark_processed(self, idempotency_key: str) -> None:
        self._seen.add(idempotency_key)


class InMemoryStoryRepo(StoryRepo):
    def __init__(self) -> None:
        self._states: Dict[str, Dict[str, Any]] = {}

    def save_story_state(self, story_id: str, state: Dict[str, Any]) -> None:
        self._states[story_id] = dict(state)

    def load_story_state(self, story_id: str) -> Optional[Dict[str, Any]]:
        return self._states.get(story_id)


@dataclass
class InMemoryRepos:
    artifacts: ArtifactRepo = InMemoryArtifactRepo()
    jobs: JobRepo = InMemoryJobRepo()
    stories: StoryRepo = InMemoryStoryRepo()


class FileJobRepo(InMemoryJobRepo):
    """Very simple file-backed idempotency store for local runs.

    Not concurrency-safe; acceptable for single-process development.
    """

    def __init__(self, path: str | os.PathLike[str] = "artifacts/idempotency.json") -> None:
        super().__init__()
        self._path = Path(path)
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self._seen = set(map(str, data or []))
            except Exception:
                self._seen = set()

    def mark_processed(self, idempotency_key: str) -> None:
        super().mark_processed(idempotency_key)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._path.write_text(json.dumps(sorted(list(self._seen))), encoding="utf-8")
        except Exception:
            pass


@dataclass
class LocalRepos:
    """Local-first repositories: filesystem artifacts + file-backed idempotency."""

    artifacts: ArtifactRepo = FileArtifactRepo()
    jobs: JobRepo = FileJobRepo()
    stories: StoryRepo = InMemoryStoryRepo()


class SQLiteJobRepo(JobRepo):
    """Job idempotency backed by the processed_messages table in SQLite."""

    def __init__(self, db_path: str = "engine.db") -> None:
        self._db = SQLiteConnection(db_name=db_path)
        self._db.connect()
        ensure_message_tables(self._db)

    def was_processed(self, idempotency_key: str) -> bool:
        conn = getattr(self._db, "conn", None)
        if conn is None:
            return False
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT 1 FROM processed_messages WHERE idempotency_key = ?",
                (idempotency_key,),
            )
            row = cur.fetchone()
            cur.close()
            return bool(row)
        except Exception:
            return False

    def mark_processed(self, idempotency_key: str) -> None:
        from ..storage.database import mark_processed as _mark

        try:
            _mark(self._db, idempotency_key)
        except Exception:
            pass


@dataclass
class SQLiteRepos:
    """SQLite-backed repos for local development when DB connectivity is desired."""

    artifacts: ArtifactRepo = FileArtifactRepo()
    jobs: JobRepo = SQLiteJobRepo()
    stories: StoryRepo = InMemoryStoryRepo()
