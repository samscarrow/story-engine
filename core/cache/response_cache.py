"""
In-memory response cache for LLM generations.
Keyed by a stable hash of provider + prompt + parameters.
Intended for short-lived caching within a process.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


def _stable_hash(data: Dict[str, Any]) -> str:
    encoded = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(encoded).hexdigest()


@dataclass
class CacheEntry:
    value: Any
    created_at: float


class ResponseCache:
    """Simple TTL cache for LLM responses."""

    def __init__(self, ttl_seconds: int = 3600, max_items: int = 2000):
        self.ttl = ttl_seconds
        self.max_items = max_items
        self._store: Dict[str, CacheEntry] = {}

    def make_key(
        self,
        provider: str,
        prompt: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> str:
        payload = {
            "provider": provider,
            "prompt": prompt,
            "params": params or {},
        }
        return _stable_hash(payload)

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if not entry:
            return None
        if time.time() - entry.created_at > self.ttl:
            # Expire
            self._store.pop(key, None)
            return None
        return entry.value

    def set(self, key: str, value: Any) -> None:
        # Evict oldest if exceeding max_items
        if len(self._store) >= self.max_items:
            oldest_key = min(self._store.items(), key=lambda kv: kv[1].created_at)[0]
            self._store.pop(oldest_key, None)
        self._store[key] = CacheEntry(value=value, created_at=time.time())

    def clear(self) -> None:
        self._store.clear()

