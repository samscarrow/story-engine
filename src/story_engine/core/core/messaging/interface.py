"""
Lightweight messaging interfaces and an in-memory adapter.

This module defines a simple Message envelope and publisher/consumer
interfaces. The in-memory bus is synchronous and intended for tests
and local development. It avoids external dependencies.
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

# Optional contract enforcement
try:
    from story_engine.core.core.contracts.topics import VALIDATORS, dlq_topic
except Exception:  # pragma: no cover - fallback if contracts not available yet
    VALIDATORS = {}
    def dlq_topic(topic: str) -> str:  # type: ignore
        return f"dlq.{topic}"


@dataclass
class Message:
    type: str
    payload: Dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    created_at: float = field(default_factory=lambda: time.time())
    retry_count: int = 0
    headers: Dict[str, Any] = field(default_factory=dict)


Handler = Callable[[Message, "Publisher"], None]


class Publisher:
    def publish(self, topic: str, message: Message) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class Consumer:
    def subscribe(self, topic: str, handler: Handler, *, prefetch: int = 1) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class InMemoryBus(Publisher, Consumer):
    """Synchronous in-memory message bus.

    - publish immediately invokes registered handlers for the topic.
    - subscribe registers a single handler per topic (last wins) for simplicity.
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, Handler] = {}
        self._seen: Dict[str, int] = defaultdict(int)

    def publish(self, topic: str, message: Message) -> None:
        # Normalize message type to match the topic and mark mismatch
        if message.type != topic:
            # preserve original for diagnostics then overwrite to enforce consistency
            message.headers["type_mismatch"] = True
            message.headers["original_type"] = message.type
            message.type = topic

        # Validate against contract keyed by topic
        validator = VALIDATORS.get(topic)
        if validator is not None:
            try:
                validator(message.payload)
            except Exception as e:
                # Route to DLQ for this topic if a subscriber exists; otherwise raise
                dlq = dlq_topic(topic)
                dlq_handler = self._handlers.get(dlq)
                if dlq_handler is not None:
                    dlq_msg = Message(
                        type=dlq,
                        payload={
                            "original": message.payload,
                            "error": str(e),
                            "original_type": message.headers.get("original_type"),
                        },
                        correlation_id=message.correlation_id,
                        causation_id=message.id,
                        headers={"validation_error": True},
                    )
                    self._seen[dlq] += 1
                    dlq_handler(dlq_msg, self)
                    return
                raise

        self._seen[topic] += 1
        handler = self._handlers.get(topic)
        if handler is not None:
            handler(message, self)

    def subscribe(self, topic: str, handler: Handler, *, prefetch: int = 1) -> None:
        # Wrap handler with a validator if the topic has a known contract
        def _wrapped(msg: Message, pub: "Publisher") -> None:
            # Enforce message.type consistency for consumers too
            if msg.type != topic:
                # Treat as a type mismatch error and route to DLQ
                dlq = dlq_topic(topic)
                dlq_handler = self._handlers.get(dlq)
                if dlq_handler is not None:
                    dlq_msg = Message(
                        type=dlq,
                        payload={
                            "original": msg.payload,
                            "error": "type_mismatch",
                            "original_type": msg.type,
                        },
                        correlation_id=msg.correlation_id,
                        causation_id=msg.id,
                        headers={"validation_error": True, "type_mismatch": True},
                    )
                    self._seen[dlq] += 1
                    dlq_handler(dlq_msg, self)
                    return
                # If no DLQ, surface the error
                raise ValueError(f"Message type '{msg.type}' does not match subscribed topic '{topic}'")

            validator = VALIDATORS.get(topic)
            if validator is not None:
                try:
                    validator(msg.payload)
                except Exception as e:
                    dlq = dlq_topic(topic)
                    dlq_handler = self._handlers.get(dlq)
                    if dlq_handler is not None:
                        dlq_msg = Message(
                            type=dlq,
                            payload={
                                "original": msg.payload,
                                "error": str(e),
                                "original_type": msg.headers.get("original_type"),
                            },
                            correlation_id=msg.correlation_id,
                            causation_id=msg.id,
                            headers={"validation_error": True},
                        )
                        self._seen[dlq] += 1
                        dlq_handler(dlq_msg, self)
                        return
                    # If no DLQ, surface the error
                    raise
            handler(msg, pub)

        self._handlers[topic] = _wrapped

    # Utility for tests/metrics
    def published_count(self, topic: str) -> int:
        return self._seen.get(topic, 0)
