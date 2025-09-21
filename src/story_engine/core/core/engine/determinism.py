from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol


class SupportsGenerate(Protocol):
    async def generate(self, prompt: str, **kwargs: Any) -> Any: ...


@dataclass
class ReplayConfig:
    mode: str = "auto"  # record|replay|auto
    dir: str = "artifacts/replays"


class RecordReplayAI:
    """Wraps an AI client to record requests/responses for deterministic tests.

    Keys are derived from a hash of the prompt and selected parameters to avoid
    leaking PII directly. In replay mode, the recorded response object is
    reconstructed with minimal shape (text and metadata), sufficient for most
    callers in this project.
    """

    def __init__(
        self, inner: SupportsGenerate, config: Optional[ReplayConfig] = None
    ) -> None:
        self.inner = inner
        self.cfg = config or ReplayConfig()
        self._dir = Path(self.cfg.dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _key(self, prompt: str, kwargs: Dict[str, Any]) -> str:
        # Only hash stable fields to improve hit rate
        stable = {
            "prompt": prompt,
            "temperature": kwargs.get("temperature"),
            "max_tokens": kwargs.get("max_tokens"),
            "system": kwargs.get("system"),
            "model": kwargs.get("model"),
        }
        raw = json.dumps(stable, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.json"

    async def generate(self, prompt: str, **kwargs: Any) -> Any:
        mode = (self.cfg.mode or "auto").lower()
        key = self._key(prompt, kwargs)
        path = self._path(key)

        if mode in {"replay", "auto"} and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))

            # Minimal response shape expected by pipelines
            class _Replay:
                def __init__(self, d: Dict[str, Any]):
                    self.text = d.get("text", "")
                    self.metadata = d.get("metadata", {})
                    self.timestamp = d.get("timestamp", "")

            return _Replay(data)

        # Fall back to live call
        resp = await self.inner.generate(prompt=prompt, **kwargs)
        # Extract minimal record
        record = {
            "text": getattr(resp, "text", ""),
            "metadata": getattr(resp, "metadata", {}) or {},
            "timestamp": getattr(resp, "timestamp", ""),
        }
        if mode in {"record", "auto"}:
            try:
                path.write_text(
                    json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except Exception:
                pass
        return resp
