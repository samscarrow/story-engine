"""
RabbitMQ adapter using pika (blocking).

This adapter provides a minimal bridge that implements the same
`Publisher` and `Consumer` interface as the in-memory bus.

Notes

- Requires `pika` to be installed when `messaging.type = rabbitmq`.
- Declares queues for topics and corresponding DLQs.
- Applies basic QoS prefetch.
- Converts between RabbitMQ deliveries and the Message dataclass.

This is intentionally conservative and synchronous for initial E2E.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict
from typing import Dict, Optional

from .interface import Consumer, Handler, Message, Publisher
try:
    from story_engine.core.core.contracts.topics import VALIDATORS, dlq_topic
except Exception:  # pragma: no cover
    VALIDATORS = {}
    def dlq_topic(topic: str) -> str:  # type: ignore
        return f"dlq.{topic}"

try:  # Optional dependency
    import pika  # type: ignore
except Exception:  # pragma: no cover
    pika = None  # type: ignore


log = logging.getLogger(__name__)


class RabbitMQBus(Publisher, Consumer):
    def __init__(
        self,
        *,
        url: str,
        queue_prefix: str = "story",
        prefetch: int = 1,
        max_retries: int = 3,
    ) -> None:
        if pika is None:  # pragma: no cover
            raise RuntimeError(
                "RabbitMQ adapter requires 'pika'. Install extra and set messaging.type=rabbitmq."
            )
        self.url = url
        self.queue_prefix = queue_prefix
        self.prefetch = prefetch
        self.max_retries = max_retries
        self._connection: Optional["pika.BlockingConnection"] = None
        self._channel: Optional["pika.adapters.blocking_connection.BlockingChannel"] = None
        self._consumer_thread: Optional[threading.Thread] = None
        self._handlers: Dict[str, Handler] = {}

    # Connection management
    def _connect(self) -> None:
        params = pika.URLParameters(self.url)
        self._connection = pika.BlockingConnection(params)
        ch = self._connection.channel()
        ch.basic_qos(prefetch_count=self.prefetch)
        self._channel = ch

    def _ensure(self) -> None:
        if self._connection is None or self._channel is None or self._connection.is_closed:
            self._connect()

    # Topology helpers
    def _queue_name(self, topic: str) -> str:
        return f"{self.queue_prefix}.{topic}"

    def _dlq_name(self, topic: str) -> str:
        return f"{self.queue_prefix}.dlq.{topic}"

    def _declare(self, topic: str) -> None:
        ch = self._channel
        assert ch is not None
        qname = self._queue_name(topic)
        dlq = self._dlq_name(topic)
        ch.queue_declare(queue=dlq, durable=True)
        ch.queue_declare(
            queue=qname,
            durable=True,
            arguments={
                "x-dead-letter-exchange": "",
                "x-dead-letter-routing-key": dlq,
            },
        )

    # Interface: Publisher
    def publish(self, topic: str, message: Message) -> None:
        self._ensure()
        ch = self._channel
        assert ch is not None
        self._declare(topic)

        # Enforce type/topic consistency
        if message.type != topic:
            message.headers["type_mismatch"] = True
            message.headers["original_type"] = message.type
            message.type = topic

        # Validate pre-publish; on failure publish to DLQ with error payload
        validator = VALIDATORS.get(topic)
        if validator is not None:
            try:
                validator(message.payload)
            except Exception as e:
                # Ensure DLQ exists and publish enriched diagnostic message
                dlq = dlq_topic(topic)
                ch.queue_declare(queue=self._dlq_name(topic), durable=True)
                dlq_body = json.dumps(
                    asdict(
                        Message(
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
                    )
                ).encode("utf-8")
                ch.basic_publish(
                    exchange="",
                    routing_key=self._queue_name(dlq),
                    body=dlq_body,
                    properties=pika.BasicProperties(
                        content_type="application/json",
                        delivery_mode=2,
                        correlation_id=message.correlation_id,
                    ),
                )
                return

        body = json.dumps(asdict(message)).encode("utf-8")
        ch.basic_publish(
            exchange="",
            routing_key=self._queue_name(topic),
            body=body,
            properties=pika.BasicProperties(
                content_type="application/json",
                delivery_mode=2,  # persistent
                correlation_id=message.correlation_id,
                headers=message.headers,
            ),
        )

    # Interface: Consumer
    def subscribe(self, topic: str, handler: Handler, *, prefetch: int | None = None) -> None:
        self._handlers[topic] = handler
        # Start or restart consumer thread with the latest handlers
        if self._consumer_thread is None or not self._consumer_thread.is_alive():
            self._consumer_thread = threading.Thread(target=self._consume_loop, daemon=True)
            self._consumer_thread.start()

    # Consumer loop
    def _consume_loop(self) -> None:  # pragma: no cover - requires broker
        while True:
            try:
                self._ensure()
                ch = self._channel
                assert ch is not None

                # Declare queues for all registered topics
                for topic in list(self._handlers.keys()):
                    self._declare(topic)

                def _make_callback(topic: str):
                    def _on_message(ch, method, properties, body):
                        data = json.loads(body.decode("utf-8"))
                        msg = Message(
                            type=topic,
                            payload=data.get("payload") or {},
                            id=data.get("id") or None,  # default in dataclass if None
                            correlation_id=properties.correlation_id or data.get("correlation_id"),
                            causation_id=data.get("causation_id"),
                            created_at=data.get("created_at") or time.time(),
                            retry_count=(data.get("retry_count") or 0),
                            headers=(properties.headers or data.get("headers") or {}),
                        )
                        # Validate before processing
                        validator = VALIDATORS.get(topic)
                        if validator is not None:
                            try:
                                validator(msg.payload)
                            except Exception as ve:
                                # Publish enriched DLQ diagnostics and ACK original
                                dlq = dlq_topic(topic)
                                ch.queue_declare(queue=self._dlq_name(topic), durable=True)
                                dlq_body = json.dumps(
                                    asdict(
                                        Message(
                                            type=dlq,
                                            payload={
                                                "original": msg.payload,
                                                "error": str(ve),
                                                "original_type": msg.headers.get("original_type"),
                                            },
                                            correlation_id=msg.correlation_id,
                                            causation_id=msg.id,
                                            headers={"validation_error": True},
                                        )
                                    )
                                ).encode("utf-8")
                                ch.basic_publish(
                                    exchange="",
                                    routing_key=self._queue_name(dlq),
                                    body=dlq_body,
                                    properties=pika.BasicProperties(
                                        content_type="application/json",
                                        delivery_mode=2,
                                        correlation_id=msg.correlation_id,
                                    ),
                                )
                                ch.basic_ack(delivery_tag=method.delivery_tag)
                                return

                        # Retry processing with exponential backoff for non-validation errors
                        attempts = 0
                        while True:
                            try:
                                self._handlers[topic](msg, self)
                                ch.basic_ack(delivery_tag=method.delivery_tag)
                                break
                            except Exception as pe:
                                attempts += 1
                                if attempts > self.max_retries:
                                    log.exception(
                                        "handler error for topic %s after %s attempts: %s",
                                        topic,
                                        attempts - 1,
                                        pe,
                                    )
                                    # Publish enriched DLQ diagnostics and ACK original
                                    dlq = dlq_topic(topic)
                                    ch.queue_declare(queue=self._dlq_name(topic), durable=True)
                                    dlq_body = json.dumps(
                                        asdict(
                                            Message(
                                                type=dlq,
                                                payload={
                                                    "original": msg.payload,
                                                    "error": str(pe),
                                                    "attempts": attempts - 1,
                                                },
                                                correlation_id=msg.correlation_id,
                                                causation_id=msg.id,
                                                headers={"retries_exhausted": True},
                                            )
                                        )
                                    ).encode("utf-8")
                                    ch.basic_publish(
                                        exchange="",
                                        routing_key=self._queue_name(dlq),
                                        body=dlq_body,
                                        properties=pika.BasicProperties(
                                            content_type="application/json",
                                            delivery_mode=2,
                                            correlation_id=msg.correlation_id,
                                        ),
                                    )
                                    ch.basic_ack(delivery_tag=method.delivery_tag)
                                    break
                                # Backoff before retrying
                                sleep_for = min(2 ** (attempts - 1), 10)
                                time.sleep(sleep_for)


                    return _on_message

                # Bind consumers
                for topic in list(self._handlers.keys()):
                    qname = self._queue_name(topic)
                    ch.basic_consume(queue=qname, on_message_callback=_make_callback(topic))

                ch.start_consuming()
            except Exception as e:
                log.warning("RabbitMQ connection error: %s. Reconnecting shortly...", e)
                time.sleep(2.0)
                # reconnect loop continues
