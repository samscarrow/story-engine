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
except Exception:
    from story_engine.core.messaging.interface import (  # type: ignore
        InMemoryBus,
        Message,
        Consumer,
        Publisher,
    )
from story_engine.core.core.messaging.rabbitmq import RabbitMQBus
from story_engine.core.core.contracts.evaluation import EvaluationRequest

try:
    from story_engine.core.core.contracts.topics import (
        EVALUATION_REQUEST,
        EVALUATION_DONE,
    )
except Exception:
    EVALUATION_REQUEST = "evaluation.request"  # type: ignore
    EVALUATION_DONE = "evaluation.done"  # type: ignore
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


def _handle_evaluation_request(msg: Message, pub: Publisher) -> None:
    log = logging.getLogger("evaluation_worker")
    req = EvaluationRequest.validate(msg.payload)
    evaluation_text = "Narrative Coherence: 7/10 - Placeholder evaluation."
    done = Message(
        type=EVALUATION_DONE,
        payload={
            "job_id": req.job_id,
            "evaluation_text": evaluation_text,
        },
        correlation_id=msg.correlation_id or req.job_id,
        causation_id=msg.id,
    )
    log.info("evaluation.done", extra={"job_id": req.job_id, "message_id": done.id})
    pub.publish(EVALUATION_DONE, done)


def main() -> None:
    configure_json_logging()
    cfg = load_config()
    bus = _select_bus(cfg)

    register_dlq_logger(bus, EVALUATION_REQUEST, "evaluation_worker")

    bus.subscribe(EVALUATION_REQUEST, _handle_evaluation_request)
    logging.getLogger("evaluation_worker").info("worker ready")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:  # pragma: no cover
        pass


if __name__ == "__main__":  # pragma: no cover
    main()
