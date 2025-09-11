from __future__ import annotations

import logging
from typing import Protocol

from .interface import Message


class _Subscribable(Protocol):
    def subscribe(self, topic: str, handler, *, prefetch: int = 1) -> None: ...


def register_dlq_logger(
    bus: _Subscribable, request_topic: str, logger_name: str
) -> None:
    """Subscribe a simple DLQ handler that logs validation errors.

    - bus: object with .subscribe(topic, handler)
    - request_topic: e.g., "plot.request"
    - logger_name: logging channel to use
    """

    dlq_topic = f"dlq.{request_topic}"

    def _handle_dlq(msg: Message, _pub) -> None:
        logging.getLogger(logger_name).error(
            "DLQ received",
            extra={
                "topic": msg.type,
                "error": msg.payload.get("error"),
                "original_type": msg.payload.get("original_type"),
            },
        )

    try:
        bus.subscribe(dlq_topic, _handle_dlq)
    except Exception:
        # In-memory bus may not be ready; ignore optional wiring errors
        pass
