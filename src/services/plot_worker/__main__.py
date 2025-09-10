from __future__ import annotations

import logging
import time
import uuid

from story_engine.core.core.common.config import load_config
from story_engine.core.core.common.logging import configure_json_logging
from story_engine.core.core.messaging.interface import InMemoryBus, Message, Consumer, Publisher
from story_engine.core.core.messaging.rabbitmq import RabbitMQBus
from story_engine.core.core.contracts.plot import PlotRequest
from story_engine.core.core.contracts.topics import PLOT_REQUEST
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


def _handle_plot_request(msg: Message, pub: Publisher) -> None:
    log = logging.getLogger("plot_worker")
    req = PlotRequest.validate(msg.payload)
    # Placeholder outline generation. Real logic would call orchestrators/LLMs.
    outline_id = str(uuid.uuid4())
    outline_ref = f"outline:{outline_id}"
    done = Message(
        type="plot.done",
        payload={
            "job_id": req.job_id,
            "outline_id": outline_id,
            "outline_ref": outline_ref,
        },
        correlation_id=msg.correlation_id or req.job_id,
        causation_id=msg.id,
    )
    log.info("plot.done", extra={"job_id": req.job_id, "message_id": done.id})
    pub.publish("plot.done", done)


def main() -> None:
    configure_json_logging()
    cfg = load_config()
    bus = _select_bus(cfg)
    # Subscribe to DLQ for visibility
    register_dlq_logger(bus, PLOT_REQUEST, "plot_worker")

    # Enforce contract by validating on consume inside handler
    bus.subscribe("plot.request", _handle_plot_request)
    # Idle loop for demonstration; in prod a real MQ consumer would block
    logging.getLogger("plot_worker").info("worker ready")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:  # pragma: no cover
        pass


if __name__ == "__main__":  # pragma: no cover
    main()
