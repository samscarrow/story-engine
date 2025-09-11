from __future__ import annotations

import argparse
import logging
import time
import uuid

from story_engine.core.core.common.config import load_config
from story_engine.core.core.common.logging import configure_json_logging
from story_engine.core.core.messaging.interface import InMemoryBus, Message, Publisher
from story_engine.core.core.contracts.topics import (
    PLOT_REQUEST,
    SCENE_REQUEST,
    DIALOGUE_REQUEST,
    EVALUATION_REQUEST,
)
from story_engine.core.core.messaging.rabbitmq import RabbitMQBus
from story_engine.core.core.contracts.plot import PlotRequest
from story_engine.core.core.messaging.helpers import register_dlq_logger


def _select_bus(cfg) -> Publisher:
    mcfg = cfg.get("messaging", {})
    mtype = (mcfg.get("type") or "inmemory").lower()
    if mtype == "rabbitmq":
        return RabbitMQBus(
            url=mcfg.get("url", "amqp://localhost"),
            queue_prefix=mcfg.get("queue_prefix", "story"),
            prefetch=int(mcfg.get("prefetch", 1) or 1),
            max_retries=int(mcfg.get("max_retries", 3) or 3),
        )
    return InMemoryBus()


def main() -> None:
    parser = argparse.ArgumentParser(description="Story Initializer Service")
    parser.add_argument("prompt", help="Seed prompt for the story")
    parser.add_argument(
        "--job-id", default=None, help="Job identifier (defaults to UUID)"
    )
    parser.add_argument(
        "--send-scene", action="store_true", help="Also publish a demo scene.request"
    )
    parser.add_argument(
        "--send-dialogue",
        action="store_true",
        help="Also publish a demo dialogue.request",
    )
    parser.add_argument(
        "--send-evaluation",
        action="store_true",
        help="Also publish a demo evaluation.request",
    )
    args = parser.parse_args()

    configure_json_logging()
    log = logging.getLogger("story_initializer")
    cfg = load_config()
    bus = _select_bus(cfg)

    job_id = args.job_id or str(uuid.uuid4())

    # Subscribe DLQ loggers for any request types we might publish
    register_dlq_logger(bus, PLOT_REQUEST, "story_initializer")
    register_dlq_logger(bus, SCENE_REQUEST, "story_initializer")
    register_dlq_logger(bus, DIALOGUE_REQUEST, "story_initializer")
    register_dlq_logger(bus, EVALUATION_REQUEST, "story_initializer")
    payload = {"job_id": job_id, "prompt": args.prompt, "constraints": {}}
    # Validate to ensure publisher side conforms to contract too
    PlotRequest.validate(payload)
    msg = Message(type="plot.request", payload=payload, correlation_id=job_id)
    log.info("publishing plot.request", extra={"job_id": job_id})
    bus.publish("plot.request", msg)

    # Optional demo scene request
    if args.send_scene:
        from story_engine.core.core.contracts.scene import SceneRequest

        scene_payload = {
            "job_id": job_id,
            "outline_id": None,
            "beat_name": "Setup",
            "prompt": f"Create a scene for: {args.prompt}",
            "characters": [{"id": "c1", "name": "Alice", "role": "protagonist"}],
            "constraints": {},
        }
        SceneRequest.validate(scene_payload)
        bus.publish(
            "scene.request",
            Message(type="scene.request", payload=scene_payload, correlation_id=job_id),
        )

    # Optional demo dialogue request
    if args.send_dialogue:
        from story_engine.core.core.contracts.dialogue import DialogueRequest

        dialogue_payload = {
            "job_id": job_id,
            "scene_id": None,
            "character_id": "c1",
            "opening_line": "Hello",
            "context": {"goal": "greet"},
        }
        DialogueRequest.validate(dialogue_payload)
        bus.publish(
            "dialogue.request",
            Message(
                type="dialogue.request", payload=dialogue_payload, correlation_id=job_id
            ),
        )

    # Optional demo evaluation request
    if args.send_evaluation:
        from story_engine.core.core.contracts.evaluation import EvaluationRequest

        evaluation_payload = {
            "job_id": job_id,
            "content": "Some content to evaluate",
            "criteria": ["coherence", "pacing"],
            "options": {},
        }
        EvaluationRequest.validate(evaluation_payload)
        bus.publish(
            "evaluation.request",
            Message(
                type="evaluation.request",
                payload=evaluation_payload,
                correlation_id=job_id,
            ),
        )
    # For demo CLI, just wait briefly
    time.sleep(0.1)


if __name__ == "__main__":  # pragma: no cover
    main()
