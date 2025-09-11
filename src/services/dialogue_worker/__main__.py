from __future__ import annotations

import logging
import time

from story_engine.core.core.common.config import load_config
from story_engine.core.core.common.logging import configure_json_logging

try:
    from story_engine.core.core.messaging.interface import (
        InMemoryBus,
        Message,
        Consumer,
        Publisher,
    )
except Exception:  # fallback to legacy alias path
    from story_engine.core.messaging.interface import (  # type: ignore
        InMemoryBus,
        Message,
        Consumer,
        Publisher,
    )
from story_engine.core.core.messaging.rabbitmq import RabbitMQBus
from story_engine.core.core.contracts.dialogue import DialogueRequest

try:
    from story_engine.core.core.contracts.topics import DIALOGUE_REQUEST, DIALOGUE_DONE
except Exception:  # fallback string constants
    DIALOGUE_REQUEST = "dialogue.request"  # type: ignore
    DIALOGUE_DONE = "dialogue.done"  # type: ignore
from story_engine.core.core.messaging.helpers import register_dlq_logger


def _select_bus(cfg) -> Publisher | Consumer:
    m = cfg.get("messaging", {})
    mtype = (m.get("type") or "inmemory").lower()
    if mtype == "rabbitmq":
        return RabbitMQBus(
            url=m.get("url", "amqp://localhost"),
            queue_prefix=m.get("queue_prefix", "story"),
            prefetch=int(m.get("prefetch", 1) or 1),
            max_retries=int(m.get("max_retries", 3) or 3),
        )
    return InMemoryBus()


def _handle_dialogue_request(msg: Message, pub: Publisher) -> None:
    log = logging.getLogger("dialogue_worker")
    req = DialogueRequest.validate(msg.payload)
    text = f"[{req.character_id}] says: placeholder dialogue."
    done = Message(
        type=DIALOGUE_DONE,
        payload={
            "job_id": req.job_id,
            "scene_id": req.scene_id,
            "text": text,
        },
        correlation_id=msg.correlation_id or req.job_id,
        causation_id=msg.id,
    )
    log.info("dialogue.done", extra={"job_id": req.job_id, "message_id": done.id})
    pub.publish(DIALOGUE_DONE, done)


def main() -> None:
    configure_json_logging()
    cfg = load_config()
    bus = _select_bus(cfg)

    register_dlq_logger(bus, DIALOGUE_REQUEST, "dialogue_worker")

    bus.subscribe(DIALOGUE_REQUEST, _handle_dialogue_request)
    logging.getLogger("dialogue_worker").info("worker ready")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:  # pragma: no cover
        pass


if __name__ == "__main__":  # pragma: no cover
    main()
