from __future__ import annotations

import logging
import time
import uuid

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
from story_engine.core.core.contracts.scene import SceneRequest
from story_engine.core.core.messaging.helpers import register_dlq_logger

# Use explicit string constants to avoid import fragility in CI environments
SCENE_REQUEST = "scene.request"
SCENE_DONE = "scene.done"


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


def _handle_scene_request(msg: Message, pub: Publisher) -> None:
    log = logging.getLogger("scene_worker")
    req = SceneRequest.validate(msg.payload)
    scene_id = str(uuid.uuid4())
    # Build a short placeholder description without exceeding line length
    beat = req.beat_name or "unknown"
    prompt_snippet = (req.prompt or "")[:40]
    scene_description = (
        f"Scene for beat '{beat}': placeholder description "
        f"based on prompt '{prompt_snippet}'."
    )
    done = Message(
        type=SCENE_DONE,
        payload={
            "job_id": req.job_id,
            "scene_id": scene_id,
            "scene_description": scene_description,
        },
        correlation_id=msg.correlation_id or req.job_id,
        causation_id=msg.id,
    )
    log.info("scene.done", extra={"job_id": req.job_id, "message_id": done.id})
    pub.publish(SCENE_DONE, done)


def main() -> None:
    configure_json_logging()
    cfg = load_config()
    bus = _select_bus(cfg)

    # DLQ subscriber for visibility
    register_dlq_logger(bus, SCENE_REQUEST, "scene_worker")

    bus.subscribe(SCENE_REQUEST, _handle_scene_request)
    logging.getLogger("scene_worker").info("worker ready")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:  # pragma: no cover
        pass


if __name__ == "__main__":  # pragma: no cover
    main()
